#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS T_orig vs T_opt LLM A/B 추론 성능 및 경로 충실도 벤치마크 실행기 (공정성 보정 + 통계 검정).

특징:
1. 백엔드 자동 감지: 로컬 OpenAI 호환 API (--base_url) 또는 로컬 HuggingFace 모델(Qwen2.5-32B-Instruct)
2. 중단 재개(Resume): data/benchmark/ab_raw_results.jsonl에 실시간 append 저장
3. 엄격한 A/B 통제:
   - temperature=0.0, 동일 시스템 프롬프트(단락 규칙 및 표준 1차 축 명칭 안내 공통 적용)
   - 트리 컨텍스트(T_orig vs T_opt)만 교체
4. 임상 안전성(Critical Under-triage, Over-triage), 층화 분석(Hotspot vs General),
   통계 검정(McNemar's test, Wilcoxon signed-rank test) 자동 산출
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from scipy.stats import binomtest, chi2, wilcoxon
from tqdm.auto import tqdm


EXP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXP_ROOT.parents[1]

SYSTEM_PROMPT = (
    "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의사입니다. "
    "주어진 주호소의 [결정 트리 규칙]만을 엄격히 따라 환자의 상태에 해당하는 최종 KTAS 등급(1~5)을 판단하십시오.\n"
    "[공통 결정 규칙]: 예외 게이트가 없으면 매칭되는 모든 조건 중 가장 위급한 등급(최소 KTAS 숫자, 1이 가장 위급)을 선택하십시오. "
    "이미 더 위급한 등급이 확정되면 하위 등급 규칙은 건너뛰어도 되며, 확인한 1차 고려사항 축은 아래 표준 축 이름"
    "(1차_활력징후, 통증_척도, 출혈_양상, 고위험_사고기전)으로 모두 기록하십시오.\n"
    "임의의 의학 지식으로 규칙을 지어내지 말고 아래 JSON 스키마로만 정확히 응답하십시오(마크다운 코드블록 외 다른 설명 금지):\n"
    "{\n"
    '  "reasoning_steps": ["확인한 단계별 사고 과정..."],\n'
    '  "checked_exception_gates": ["확인한 예외 게이트명 (없으면 빈 배열)"],\n'
    '  "checked_first_order_axes": ["확인한 1차 축 표준 이름 (예: 1차_활력징후, 통증_척도, 출혈_양상, 고위험_사고기전)"],\n'
    '  "matched_rule_text": "최종 등급 결정에 직접 매칭된 규칙 원문",\n'
    '  "final_ktas_level": 1\n'
    "}"
)

# T_orig의 세부 항목 표현도 공정하게 표준 축으로 인식하도록 별칭 확장
AXIS_ALIASES = {
    "1차_활력징후": (
        "1차_활력징후", "1차 활력징후", "활력징후", "1차활력징후",
        "호흡곤란", "쇼크", "혈역학", "무의식", "의식변화", "의식수준", "체온", "패혈증", "sirs", "열 (", "열(",
    ),
    "통증_척도": ("통증_척도", "통증 척도", "통증", "중심성", "말초성", "8-10", "4-7"),
    "출혈_양상": ("출혈_양상", "출혈성 질환", "출혈성질환", "출혈"),
    "고위험_사고기전": ("고위험_사고기전", "고위험성 사고기전", "사고기전"),
}


def load_orig_complaint_map(raw_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """원본 Ground Truth 주호소 맵을 로드한다."""
    orig_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in sorted(raw_dir.glob("ktas_*.json")):
        if p.name == "ktas_part1_common.json":
            continue
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ctx = "PED" if "Pediatric" in str(data.get("target_population", "")) else "ADULT"
        for _, cat_val in data.get("categories", {}).items():
            for code, comp in cat_val.get("complaints", {}).items():
                nacrs = str(comp.get("nacrs_code", code))
                orig_map[(ctx, nacrs, comp.get("name", ""))] = comp
    return orig_map


def serialize_t_orig_context(orig_comp: dict[str, Any]) -> str:
    """T_orig 단일 주호소의 평면 규칙 컨텍스트를 직렬화한다."""
    lines = [
        f"[주호소]: {orig_comp.get('name')} (NACRS {orig_comp.get('nacrs_code')})",
        f"[평가 우선순위]: {orig_comp.get('evaluation_priority', 'standard')}",
    ]
    if orig_comp.get("priority_exception_rule"):
        lines.append(f"[예외 규칙]: {orig_comp['priority_exception_rule']}")
    first_app = orig_comp.get("first_order_modifiers_applicable", [])
    if first_app:
        lines.append("[적용 가능 1차 고려사항]: " + " | ".join(first_app))
    lines.append("[증상별 2차 고려사항 (순차 평가 리스트)]:")
    for idx, m in enumerate(orig_comp.get("second_order_modifiers", []), start=1):
        lines.append(f"  {idx}. (KTAS {m['ktas_level']}단계) {m['condition']}")
    return "\n".join(lines)


def serialize_t_opt_context(opt_comp: dict[str, Any]) -> str:
    """T_opt 단일 주호소의 계층·변수 그룹화 컨텍스트를 직렬화한다."""
    lines = [
        f"[주호소]: {opt_comp.get('name')} (NACRS {opt_comp.get('nacrs_code')})",
        f"[평가 우선순위]: {opt_comp.get('evaluation_priority', 'standard')}",
    ]
    ex_gates = opt_comp.get("exception_gates", [])
    if ex_gates:
        lines.append("[1. 예외 게이트 (최우선 평가)]:")
        for g in ex_gates:
            lines.append(f"  - (KTAS {g['ktas_level']}단계) {g['condition']}")

    first_pass = opt_comp.get("first_order_single_pass", {})
    if first_pass:
        lines.append("[2. 1차 공통 축 Single-Pass]:")
        for axis, rules in first_pass.items():
            items = " / ".join(f"{r['ktas_level']}단계:{r['condition']}" for r in rules)
            lines.append(f"  - [{axis}] {items}")

    sec_groups = opt_comp.get("second_order_variable_groups", {})
    if sec_groups:
        lines.append("[3. 2차 변수 그룹]:")
        for axis, rules in sec_groups.items():
            items = " / ".join(f"{r['ktas_level']}단계:{r['condition']}" for r in rules)
            lines.append(f"  - [{axis}] {items}")
    return "\n".join(lines)


def build_user_prompt(case: dict[str, Any], tree_context: str) -> str:
    return (
        f"### 결정 트리 규칙\n{tree_context}\n\n"
        f"### 환자 평가 케이스\n{case['patient_presentation']}\n\n"
        "위 결정 트리 규칙에 따라 JSON 형식으로만 판정 결과를 출력하십시오."
    )


def parse_llm_json(raw_text: str) -> dict[str, Any]:
    """모델 출력에서 JSON을 안전하게 파싱한다."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    lvl_m = re.search(r'"final_ktas_level"\s*:\s*([1-5])', cleaned)
    if not lvl_m:
        lvl_m = re.search(r"(?:KTAS|최종\s*등급)[^\d]{0,10}([1-5])", cleaned)
    return {
        "reasoning_steps": [cleaned[:200]],
        "checked_exception_gates": [],
        "checked_first_order_axes": [],
        "matched_rule_text": "",
        "final_ktas_level": int(lvl_m.group(1)) if lvl_m else 0,
        "_parse_fallback": True,
    }


class InferenceBackend:
    """OpenAI 호환 API 및 로컬 HuggingFace Transformers 통합 추론 래퍼."""

    def __init__(self, backend: str, model_arg: str, base_url: str | None, max_new_tokens: int = 384):
        self.backend = backend
        self.model_arg = model_arg
        self.base_url = base_url.rstrip("/") if base_url else None
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None
        self.model_label = model_arg
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "auto":
            if self.base_url:
                self.backend = "openai"
            else:
                for url in ("http://localhost:8000/v1", "http://localhost:11434/v1"):
                    try:
                        with urllib.request.urlopen(f"{url}/models", timeout=1.5) as resp:
                            if resp.status == 200:
                                data = json.loads(resp.read().decode("utf-8"))
                                if data.get("data"):
                                    self.base_url = url
                                    if not self.model_arg:
                                        self.model_arg = data["data"][0]["id"]
                                    self.backend = "openai"
                                    break
                    except Exception:
                        continue
                if self.backend == "auto":
                    self.backend = "hf"

        if self.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = self.model_arg or str(REPO_ROOT / "models" / "Qwen2.5-32B-Instruct")
            self.model_label = Path(model_path).name
            print(f"[*] HuggingFace 로컬 모델 로드 중: {model_path} (bfloat16, sdpa, cuda:0)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="cuda:0",
            )
            self.model.eval()
            print("[+] HuggingFace 모델 준비 완료")
        else:
            local_tok_dir = REPO_ROOT / "models" / "Qwen2.5-32B-Instruct"
            if local_tok_dir.exists():
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_tok_dir), local_files_only=True, trust_remote_code=True)
            self.model_label = f"API:{self.model_arg}@{self.base_url}"
            print(f"[+] OpenAI 호환 API 백엔드 연결: {self.model_label}")

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return max(1, len(text) // 3)

    def generate(self, user_prompt: str) -> tuple[str, int, int]:
        if self.backend == "hf":
            import torch

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
            in_tokens = int(inputs["input_ids"].shape[1])

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = outputs[0][in_tokens:]
            out_tokens = int(len(gen_ids))
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            return gen_text, in_tokens, out_tokens

        payload = json.dumps(
            {
                "model": self.model_arg,
                "temperature": 0.0,
                "max_tokens": self.max_new_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_toks = usage.get("prompt_tokens") or self.count_tokens(SYSTEM_PROMPT + user_prompt)
        out_toks = usage.get("completion_tokens") or self.count_tokens(content)
        return content, int(in_toks), int(out_toks)


def _macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    f1s: list[float] = []
    for cls in range(1, 6):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append(2 * tp / denom)
    return (sum(f1s) / len(f1s) * 100.0) if f1s else 0.0


def _axis_covered(req_axis: str, checked_axes: list[str], reasoning_steps: list[str]) -> bool:
    aliases = AXIS_ALIASES.get(req_axis, (req_axis,))
    haystack = (" ".join(str(x) for x in checked_axes) + " " + " ".join(str(x) for x in reasoning_steps)).lower()
    return any(al.lower() in haystack for al in aliases)


def _rule_cited_correctly(case: dict[str, Any], pred_rule: str, pred_steps: list[str]) -> bool:
    target_rule = ""
    if case.get("matched_exception_gate"):
        target_rule = case["matched_exception_gate"]["condition"]
    elif case.get("matched_second_order_rule"):
        target_rule = case["matched_second_order_rule"]["condition"]
    else:
        return True

    norm_t = re.sub(r"\s+", "", target_rule)
    norm_p = re.sub(r"\s+", "", pred_rule or "")
    if not norm_p:
        norm_p = re.sub(r"\s+", "", " ".join(str(x) for x in pred_steps))
    if norm_t in norm_p or SequenceMatcher(None, norm_t, norm_p).ratio() >= 0.50:
        return True
    t_tokens = [w for w in re.split(r"[\s,/()]+", target_rule) if len(w) >= 2]
    if t_tokens:
        hit = sum(1 for w in t_tokens if w in (pred_rule + " " + " ".join(str(x) for x in pred_steps)))
        return (hit / len(t_tokens)) >= 0.50
    return False


def mcnemar_pvalue(b_vec: list[bool], c_vec: list[bool]) -> float | None:
    """쌍체 이진 지표에 대한 McNemar 검정 p-value를 반환한다."""
    b = sum(1 for x, y in zip(b_vec, c_vec) if x and not y)
    c = sum(1 for x, y in zip(b_vec, c_vec) if not x and y)
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    if n_disc < 25:
        return float(binomtest(min(b, c), n_disc, 0.5).pvalue)
    stat = (abs(b - c) - 1.0) ** 2 / n_disc
    return float(chi2.sf(stat, df=1))


def wilcoxon_pvalue(vals_a: list[float], vals_b: list[float]) -> float | None:
    """쌍체 연속 지표에 대한 Wilcoxon signed-rank test p-value를 반환한다."""
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    if not any(abs(d) > 1e-12 for d in diffs):
        return 1.0
    try:
        res = wilcoxon(vals_a, vals_b, zero_method="wilcox")
        val = float(res.pvalue)
        return None if math.isnan(val) else val
    except Exception:
        return None


def compute_metrics(records: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """단일 변인(T_orig 또는 T_opt) 레코드 리스트에 대해 임상 안전성/충실도/토큰 지표를 계산한다."""
    n = len(records)
    if n == 0:
        return {}
    rec_by_id = {r["case_id"]: r for r in records}
    ordered_ids = sorted(rec_by_id.keys())

    y_true = [cases_by_id[cid]["ground_truth_level"] for cid in ordered_ids]
    y_pred = [int(rec_by_id[cid]["parsed"].get("final_ktas_level", 0)) for cid in ordered_ids]

    acc_vec = [t == p for t, p in zip(y_true, y_pred)]
    acc = sum(acc_vec) / n * 100.0
    f1 = _macro_f1(y_true, y_pred)

    # 임상 오류 방향 분리
    parse_fail = sum(1 for p in y_pred if p == 0) / n * 100.0
    gen_under = sum(1 for t, p in zip(y_true, y_pred) if p > t and p != 0) / n * 100.0
    over_triage = sum(1 for t, p in zip(y_true, y_pred) if 1 <= p < t) / n * 100.0

    # 치명적 과소평가율: 실제 중증(1~2등급) 환자 중 3~5등급으로 낮춘 비율
    crit_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in (1, 2)]
    crit_under_vec = [p in (3, 4, 5) for _, p in crit_pairs]
    crit_under = (sum(crit_under_vec) / len(crit_pairs) * 100.0) if crit_pairs else 0.0

    # 필수 1차 축 누락률 (전체 축 커버리지 vs 핵심 1차 활력징후/트리거 축 커버리지)
    req_ids = [cid for cid in ordered_ids if cases_by_id[cid].get("required_first_order_checks")]
    miss_all_vec: list[bool] = []
    miss_crit_vec: list[bool] = []
    for cid in req_ids:
        c = cases_by_id[cid]
        r = rec_by_id[cid]
        checked = r["parsed"].get("checked_first_order_axes", [])
        steps = r["parsed"].get("reasoning_steps", [])
        all_ok = all(_axis_covered(ax, checked, steps) for ax in c["required_first_order_checks"])
        miss_all_vec.append(not all_ok)

        # 핵심 1차 축: 1차_활력징후가 필수 목록에 있으면 반드시 확인해야 하며, first_order_axis 결정 케이스는 해당 트리거 포함
        crit_axes = [ax for ax in c["required_first_order_checks"] if ax == "1차_활력징후"]
        if not crit_axes:
            crit_axes = c["required_first_order_checks"][:1]
        crit_ok = all(_axis_covered(ax, checked, steps) for ax in crit_axes)
        miss_crit_vec.append(not crit_ok)

    missing_first_all = (sum(miss_all_vec) / len(req_ids) * 100.0) if req_ids else 0.0
    missing_first_crit = (sum(miss_crit_vec) / len(req_ids) * 100.0) if req_ids else 0.0

    # 근거 규칙 일치율
    rule_ids = [
        cid for cid in ordered_ids
        if cases_by_id[cid].get("matched_second_order_rule") or cases_by_id[cid].get("matched_exception_gate")
    ]
    cite_vec: list[bool] = []
    for cid in rule_ids:
        c = cases_by_id[cid]
        r = rec_by_id[cid]
        p_lvl = int(r["parsed"].get("final_ktas_level", 0))
        ok = (p_lvl == c["ground_truth_level"]) and _rule_cited_correctly(
            c, r["parsed"].get("matched_rule_text", ""), r["parsed"].get("reasoning_steps", [])
        )
        cite_vec.append(ok)
    rule_cite_acc = (sum(cite_vec) / len(rule_ids) * 100.0) if rule_ids else 0.0

    in_toks = [float(rec_by_id[cid]["input_tokens"]) for cid in ordered_ids]
    out_toks = [float(rec_by_id[cid]["output_tokens"]) for cid in ordered_ids]
    tot_toks = [i + o for i, o in zip(in_toks, out_toks)]

    return {
        "n": n,
        "n_critical": len(crit_pairs),
        "acc": acc,
        "macro_f1": f1,
        "crit_under_triage": crit_under,
        "gen_under_triage": gen_under,
        "over_triage": over_triage,
        "parse_fail": parse_fail,
        "missing_first_all": missing_first_all,
        "missing_first_crit": missing_first_crit,
        "rule_cite_acc": rule_cite_acc,
        "avg_in": sum(in_toks) / n,
        "avg_out": sum(out_toks) / n,
        "avg_tot": sum(tot_toks) / n,
        "_vecs": {
            "ordered_ids": ordered_ids,
            "acc": acc_vec,
            "crit_under": crit_under_vec,
            "miss_all": miss_all_vec,
            "miss_crit": miss_crit_vec,
            "cite": cite_vec,
            "in_toks": in_toks,
            "out_toks": out_toks,
            "tot_toks": tot_toks,
        },
    }


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "N/A"
    if p < 0.001:
        return f"`{p:.2e}` (*)"
    if p < 0.05:
        return f"`{p:.4f}` (*)"
    return f"`{p:.4f}`"


def build_ab_report(
    raw_results: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    model_label: str,
) -> str:
    """전체 / Hotspot Top 20 / 일반·DNR 층화 비교 및 통계 검정 p-value 리포트를 생성한다."""
    orig_all = [r for r in raw_results if r["tree_variant"] == "T_orig"]
    opt_all = [r for r in raw_results if r["tree_variant"] == "T_opt"]

    orig_hs = [r for r in orig_all if cases_by_id[r["case_id"]]["cohort"] == "hotspot_top20"]
    opt_hs = [r for r in opt_all if cases_by_id[r["case_id"]]["cohort"] == "hotspot_top20"]

    orig_gen = [r for r in orig_all if cases_by_id[r["case_id"]]["cohort"] == "general"]
    opt_gen = [r for r in opt_all if cases_by_id[r["case_id"]]["cohort"] == "general"]

    def format_section(title: str, r_orig: list[dict[str, Any]], r_opt: list[dict[str, Any]]) -> list[str]:
        m_orig = compute_metrics(r_orig, cases_by_id)
        m_opt = compute_metrics(r_opt, cases_by_id)
        if not m_orig or not m_opt:
            return [f"## {title}", "- 데이터 없음", ""]

        vo = m_orig["_vecs"]
        vp = m_opt["_vecs"]
        opt_map = {r["case_id"]: int(r["parsed"].get("final_ktas_level", 0)) for r in r_opt}
        orig_map = {r["case_id"]: int(r["parsed"].get("final_ktas_level", 0)) for r in r_orig}
        common_ids = sorted(set(opt_map) & set(orig_map))
        agree = (
            sum(1 for cid in common_ids if orig_map[cid] == opt_map[cid]) / len(common_ids) * 100.0
            if common_ids else 0.0
        )

        p_acc = mcnemar_pvalue(vo["acc"], vp["acc"]) if vo["ordered_ids"] == vp["ordered_ids"] else None
        p_crit_u = mcnemar_pvalue(vo["crit_under"], vp["crit_under"]) if len(vo["crit_under"]) == len(vp["crit_under"]) else None
        p_miss_all = mcnemar_pvalue(vo["miss_all"], vp["miss_all"]) if len(vo["miss_all"]) == len(vp["miss_all"]) else None
        p_miss_crit = mcnemar_pvalue(vo["miss_crit"], vp["miss_crit"]) if len(vo["miss_crit"]) == len(vp["miss_crit"]) else None
        p_cite = mcnemar_pvalue(vo["cite"], vp["cite"]) if len(vo["cite"]) == len(vp["cite"]) else None
        p_in = wilcoxon_pvalue(vo["in_toks"], vp["in_toks"]) if vo["ordered_ids"] == vp["ordered_ids"] else None
        p_out = wilcoxon_pvalue(vo["out_toks"], vp["out_toks"]) if vo["ordered_ids"] == vp["ordered_ids"] else None
        p_tot = wilcoxon_pvalue(vo["tot_toks"], vp["tot_toks"]) if vo["ordered_ids"] == vp["ordered_ids"] else None

        tok_diff_pct = (m_opt["avg_tot"] - m_orig["avg_tot"]) / m_orig["avg_tot"] * 100.0 if m_orig["avg_tot"] else 0.0
        out_diff_pct = (m_opt["avg_out"] - m_orig["avg_out"]) / m_orig["avg_out"] * 100.0 if m_orig["avg_out"] else 0.0

        return [
            f"## {title} (`N={m_orig['n']}`건, 중증 1~2등급 `{m_orig['n_critical']}`건, A/B 최종 판정 일치율: `{agree:.2f}%`)",
            "",
            "| 평가 지표 | `T_orig` (원본) | `T_opt` (최적화) | 변화폭 (Delta) | 통계 검정 ($p$-value) |",
            "| :--- | ---: | ---: | :--- | :---: |",
            f"| **[전제] 최종 등급 정확도 (Accuracy)** | {m_orig['acc']:.2f}% | {m_opt['acc']:.2f}% | {m_opt['acc'] - m_orig['acc']:+.2f}%p | {_fmt_p(p_acc)} |",
            f"| **[전제] Macro-F1** | {m_orig['macro_f1']:.2f}% | {m_opt['macro_f1']:.2f}% | {m_opt['macro_f1'] - m_orig['macro_f1']:+.2f}%p | - |",
            f"| **[안전] 치명적 과소평가율 (1~2➔3~5등급, ↓)** | {m_orig['crit_under_triage']:.2f}% | {m_opt['crit_under_triage']:.2f}% | {m_opt['crit_under_triage'] - m_orig['crit_under_triage']:+.2f}%p | {_fmt_p(p_crit_u)} |",
            f"| **[안전] 일반 과소평가율 (`pred > gt`, ↓)** | {m_orig['gen_under_triage']:.2f}% | {m_opt['gen_under_triage']:.2f}% | {m_opt['gen_under_triage'] - m_orig['gen_under_triage']:+.2f}%p | - |",
            f"| **[안전] 과대평가율 (`pred < gt`, ↓)** | {m_orig['over_triage']:.2f}% | {m_opt['over_triage']:.2f}% | {m_opt['over_triage'] - m_orig['over_triage']:+.2f}%p | - |",
            f"| **[안전] JSON 파싱 실패율 (↓)** | {m_orig['parse_fail']:.2f}% | {m_opt['parse_fail']:.2f}% | {m_opt['parse_fail'] - m_orig['parse_fail']:+.2f}%p | - |",
            f"| **[지표 1] 전체 1차 축 누락률 (↓)** | {m_orig['missing_first_all']:.2f}% | {m_opt['missing_first_all']:.2f}% | {m_opt['missing_first_all'] - m_orig['missing_first_all']:+.2f}%p | {_fmt_p(p_miss_all)} |",
            f"| **[지표 1] 핵심 1차 활력징후 누락률 (↓)** | {m_orig['missing_first_crit']:.2f}% | {m_opt['missing_first_crit']:.2f}% | {m_opt['missing_first_crit'] - m_orig['missing_first_crit']:+.2f}%p | {_fmt_p(p_miss_crit)} |",
            f"| **[지표 1] 근거 규칙 일치율 (Rule Citation, ↑)** | {m_orig['rule_cite_acc']:.2f}% | {m_opt['rule_cite_acc']:.2f}% | {m_opt['rule_cite_acc'] - m_orig['rule_cite_acc']:+.2f}%p | {_fmt_p(p_cite)} |",
            f"| **[지표 2] 평균 입력 토큰 수** | {m_orig['avg_in']:.1f} | {m_opt['avg_in']:.1f} | {m_opt['avg_in'] - m_orig['avg_in']:+.1f} | {_fmt_p(p_in)} |",
            f"| **[지표 2] 평균 출력(추론) 토큰 수 (↓)** | {m_orig['avg_out']:.1f} | {m_opt['avg_out']:.1f} | {m_opt['avg_out'] - m_orig['avg_out']:+.1f} ({out_diff_pct:+.1f}%) | {_fmt_p(p_out)} |",
            f"| **[지표 2] 환자당 총 토큰 수 (↓)** | {m_orig['avg_tot']:.1f} | {m_opt['avg_tot']:.1f} | **{m_opt['avg_tot'] - m_orig['avg_tot']:+.1f} ({tok_diff_pct:+.1f}%)** | {_fmt_p(p_tot)} |",
            "",
        ]

    lines = [
        "# KTAS `T_orig` vs `T_opt` LLM A/B 벤치마크 평가 결과 리포트",
        "",
        f"- **평가 모델**: `{model_label}` (`temperature=0.0`, 결정론적 생성)",
        f"- **통계 검정**: 이진 지표 `McNemar's test`, 토큰 수 `Wilcoxon signed-rank test` (`*`: $p < 0.05$)",
        f"- **완료된 추론 기록**: `T_orig` {len(orig_all)}건 / `T_opt` {len(opt_all)}건",
        "",
        "---",
        "",
        *format_section("1. 전체 평가 벤치마크 비교", orig_all, opt_all),
        "---",
        "",
        *format_section("2. 병목 Hotspot Top 20 주호소 군 비교", orig_hs, opt_hs),
        "---",
        "",
        *format_section("3. 일반 / DNR 예외 주호소 군 비교", orig_gen, opt_gen),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="KTAS T_orig vs T_opt LLM A/B 벤치마크 실행기")
    parser.add_argument("--backend", choices=["auto", "hf", "openai"], default="auto")
    parser.add_argument("--base_url", default=None, help="OpenAI 호환 API 주소")
    parser.add_argument("--model", default="", help="모델 ID 또는 로컬 HuggingFace 모델 경로")
    parser.add_argument("--cases", default=str(EXP_ROOT / "data" / "benchmark" / "ktas_eval_cases.json"))
    parser.add_argument("--raw-dir", default=str(EXP_ROOT / "data" / "raw"))
    parser.add_argument("--opt-json", default=str(EXP_ROOT / "data" / "optimized" / "ktas_optimized_tree.json"))
    parser.add_argument("--out-jsonl", default=str(EXP_ROOT / "data" / "benchmark" / "ab_raw_results.jsonl"))
    parser.add_argument("--out-report", default=str(EXP_ROOT / "reports" / "ab_benchmark_results.md"))
    parser.add_argument("--limit", type=int, default=0, help="상위 N건만 실행 (0이면 전체)")
    parser.add_argument("--report-only", action="store_true", help="추론 없이 기존 jsonl로 리포트만 재생성")
    args = parser.parse_args()

    with Path(args.cases).open("r", encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)
    if args.limit > 0:
        cases = cases[: args.limit]
    cases_by_id = {c["case_id"]: c for c in cases}

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    existing_records: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str]] = set()
    if out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_records.append(rec)
                    done_keys.add((rec["case_id"], rec["tree_variant"]))

    model_label = args.model or "Qwen2.5-32B-Instruct"

    if not args.report_only:
        orig_map = load_orig_complaint_map(Path(args.raw_dir))
        with Path(args.opt_json).open("r", encoding="utf-8") as f:
            opt_list = json.load(f)["complaints"]
        opt_map = {(c["context"], c["nacrs_code"], c["name"]): c for c in opt_list}

        tasks: list[tuple[dict[str, Any], str, str]] = []
        for c in cases:
            key = (c["context"], c["nacrs_code"], c["complaint_name"])
            if (c["case_id"], "T_orig") not in done_keys:
                tasks.append((c, "T_orig", serialize_t_orig_context(orig_map[key])))
            if (c["case_id"], "T_opt") not in done_keys:
                tasks.append((c, "T_opt", serialize_t_opt_context(opt_map[key])))

        print(f"[*] 기존 완료: {len(done_keys)}건 | 남은 추론 작업: {len(tasks)}건")
        if tasks:
            engine = InferenceBackend(args.backend, args.model, args.base_url)
            model_label = engine.model_label
            with out_jsonl.open("a", encoding="utf-8") as f:
                for c, variant, tree_ctx in tqdm(tasks, desc="LLM A/B Benchmark"):
                    t0 = time.perf_counter()
                    prompt = build_user_prompt(c, tree_ctx)
                    raw_out, in_toks, out_toks = engine.generate(prompt)
                    elapsed = time.perf_counter() - t0
                    parsed = parse_llm_json(raw_out)

                    rec = {
                        "case_id": c["case_id"],
                        "tree_variant": variant,
                        "ground_truth_level": c["ground_truth_level"],
                        "input_tokens": in_toks,
                        "output_tokens": out_toks,
                        "elapsed_sec": round(elapsed, 3),
                        "parsed": parsed,
                        "raw_output": raw_out,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    existing_records.append(rec)

    filtered_records = [r for r in existing_records if r["case_id"] in cases_by_id]
    report_md = build_ab_report(filtered_records, cases_by_id, model_label)
    Path(args.out_report).write_text(report_md, encoding="utf-8")
    print(f"[+] A/B 벤치마크 리포트 저장 완료: {args.out_report}")


if __name__ == "__main__":
    main()
