#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS T_orig vs T_opt LLM A/B 추론 성능 및 경로 충실도 벤치마크 실행기.

특징:
1. 백엔드 자동 감지: 로컬 OpenAI 호환 API (--base_url) 또는 로컬 HuggingFace 모델(Qwen2.5-32B-Instruct) 지원
2. 중단 재개(Resume): data/benchmark/ab_raw_results.jsonl에 실시간 append 저장
3. 엄격한 A/B 통제: temperature=0.0, 동일 시스템 프롬프트/출력 스키마 유지, 트리 컨텍스트만 교체
4. 2대 핵심 지표 + 전제 조건 자동 채점 및 reports/ab_benchmark_results.md 생성
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


EXP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXP_ROOT.parents[1]

SYSTEM_PROMPT = (
    "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의사입니다. "
    "주어진 주호소의 [결정 트리 규칙]만을 엄격히 따라 환자의 상태에 해당하는 최종 KTAS 등급(1~5)을 판단하십시오.\n"
    "임의의 의학 지식으로 규칙을 건너뛰지 말고, 반드시 트리의 예외 게이트, 주호소에 명시된 모든 1차 공통 축(이상 유무 포함), 2차 세부 규칙을 순서대로 확인한 뒤 "
    "아래 JSON 스키마로만 정확히 응답하십시오(마크다운 코드블록 외 다른 설명 금지):\n"
    '{\n'
    '  "reasoning_steps": ["확인한 단계별 사고 과정..."],\n'
    '  "checked_exception_gates": ["확인한 예외 게이트명 (없으면 빈 배열)"],\n'
    '  "checked_first_order_axes": ["해당 주호소에서 확인한 모든 1차 축 이름 (예: 1차_활력징후, 통증_척도, 출혈_양상, 고위험_사고기전 등 누락 없이 기재)"],\n'
    '  "matched_rule_text": "최종 등급 결정에 직접 매칭된 규칙 원문",\n'
    '  "final_ktas_level": 1\n'
    '}'
)

AXIS_ALIASES = {
    "1차_활력징후": ("1차_활력징후", "1차 활력징후", "활력징후", "1차활력징후"),
    "통증_척도": ("통증_척도", "통증 척도", "통증", "중심성", "말초성"),
    "출혈_양상": ("출혈_양상", "출혈성 질환", "출혈"),
    "고위험_사고기전": ("고위험_사고기전", "사고기전"),
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
        "[결정 규칙]: 예외 게이트가 없으면 1차 공통 축과 2차 변수 그룹에서 매칭되는 모든 조건 중 가장 위급한 등급(최소 KTAS 숫자, 1이 가장 위급)을 선택하고, 이미 더 위급한 등급이 확정되면 하위 등급 규칙은 건너뛰십시오.",
    ]
    ex_gates = opt_comp.get("exception_gates", [])
    if ex_gates:
        lines.append("[1. 예외 게이트 (최우선 평가)]:")
        for g in ex_gates:
            lines.append(f"  - (KTAS {g['ktas_level']}단계) {g['condition']}")

    first_pass = opt_comp.get("first_order_single_pass", {})
    if first_pass:
        lines.append("[2. 1차 공통 축 Single-Pass (축별 1회 확인, 최고 중증도 우선)]:")
        for axis, rules in first_pass.items():
            items = " / ".join(f"{r['ktas_level']}단계:{r['condition']}" for r in rules)
            lines.append(f"  - [{axis}] {items}")

    sec_groups = opt_comp.get("second_order_variable_groups", {})
    if sec_groups:
        lines.append("[3. 2차 변수 그룹 (상위 등급 확보 시 하위 그룹 스킵 가능)]:")
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
    # 정규식 폴백
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
                # 로컬 OpenAI 호환 서버 자동 탐지
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
            # OpenAI 호환 API 사용 시에도 정확한 토큰 측정을 위해 로컬 토크나이저(있으면) 로드
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

        # OpenAI 호환 REST 호출 (외부 openai SDK 미설치 환경에서도 동작)
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
    haystack = " ".join(str(x) for x in checked_axes) + " " + " ".join(str(x) for x in reasoning_steps)
    return any(al in haystack for al in aliases)


def _rule_cited_correctly(case: dict[str, Any], pred_rule: str, pred_steps: list[str]) -> bool:
    target_rule = ""
    if case.get("matched_exception_gate"):
        target_rule = case["matched_exception_gate"]["condition"]
    elif case.get("matched_second_order_rule"):
        target_rule = case["matched_second_order_rule"]["condition"]
    else:
        return True  # 1차 축 결정 케이스는 1차 축 누락 여부로 판정

    norm_t = re.sub(r"\s+", "", target_rule)
    norm_p = re.sub(r"\s+", "", pred_rule or "")
    if not norm_p:
        norm_p = re.sub(r"\s+", "", " ".join(str(x) for x in pred_steps))
    if norm_t in norm_p or SequenceMatcher(None, norm_t, norm_p).ratio() >= 0.50:
        return True
    # 핵심 토큰 교집합 확인
    t_tokens = [w for w in re.split(r"[\s,/()]+", target_rule) if len(w) >= 2]
    if t_tokens:
        hit = sum(1 for w in t_tokens if w in (pred_rule + " " + " ".join(str(x) for x in pred_steps)))
        return (hit / len(t_tokens)) >= 0.50
    return False


def compute_metrics(records: list[dict[str, Any]], cases_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """단일 변인(T_orig 또는 T_opt) 레코드 리스트에 대해 핵심 지표를 계산한다."""
    n = len(records)
    if n == 0:
        return {}
    y_true = [cases_by_id[r["case_id"]]["ground_truth_level"] for r in records]
    y_pred = [int(r["parsed"].get("final_ktas_level", 0)) for r in records]

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n * 100.0
    f1 = _macro_f1(y_true, y_pred)
    under = sum(1 for t, p in zip(y_true, y_pred) if p > t or p == 0) / n * 100.0

    # 필수 1차 축 누락률
    req_cases = [r for r in records if cases_by_id[r["case_id"]].get("required_first_order_checks")]
    missing_first = 0
    for r in req_cases:
        c = cases_by_id[r["case_id"]]
        checked = r["parsed"].get("checked_first_order_axes", [])
        steps = r["parsed"].get("reasoning_steps", [])
        if not all(_axis_covered(ax, checked, steps) for ax in c["required_first_order_checks"]):
            missing_first += 1
    missing_first_rate = (missing_first / len(req_cases) * 100.0) if req_cases else 0.0

    # 근거 규칙 일치율 (2차/예외 결정 케이스 기준)
    rule_cases = [
        r for r in records
        if cases_by_id[r["case_id"]].get("matched_second_order_rule") or cases_by_id[r["case_id"]].get("matched_exception_gate")
    ]
    cited_ok = 0
    for r in rule_cases:
        c = cases_by_id[r["case_id"]]
        p_lvl = int(r["parsed"].get("final_ktas_level", 0))
        if p_lvl == c["ground_truth_level"] and _rule_cited_correctly(
            c, r["parsed"].get("matched_rule_text", ""), r["parsed"].get("reasoning_steps", [])
        ):
            cited_ok += 1
    rule_cite_acc = (cited_ok / len(rule_cases) * 100.0) if rule_cases else 0.0

    avg_in = sum(r["input_tokens"] for r in records) / n
    avg_out = sum(r["output_tokens"] for r in records) / n
    avg_tot = avg_in + avg_out

    return {
        "n": n,
        "acc": acc,
        "macro_f1": f1,
        "under_triage": under,
        "missing_first": missing_first_rate,
        "rule_cite_acc": rule_cite_acc,
        "avg_in": avg_in,
        "avg_out": avg_out,
        "avg_tot": avg_tot,
    }


def build_ab_report(
    raw_results: list[dict[str, Any]],
    cases_by_id: dict[str, dict[str, Any]],
    model_label: str,
) -> str:
    """전체 및 Hotspot Top 20 비교 표를 포함한 ab_benchmark_results.md를 생성한다."""
    orig_all = [r for r in raw_results if r["tree_variant"] == "T_orig"]
    opt_all = [r for r in raw_results if r["tree_variant"] == "T_opt"]

    orig_hs = [r for r in orig_all if cases_by_id[r["case_id"]]["cohort"] == "hotspot_top20"]
    opt_hs = [r for r in opt_all if cases_by_id[r["case_id"]]["cohort"] == "hotspot_top20"]

    def format_section(title: str, r_orig: list[dict[str, Any]], r_opt: list[dict[str, Any]]) -> list[str]:
        m_orig = compute_metrics(r_orig, cases_by_id)
        m_opt = compute_metrics(r_opt, cases_by_id)
        if not m_orig or not m_opt:
            return [f"## {title}", "- 데이터 부족"]

        opt_map = {r["case_id"]: int(r["parsed"].get("final_ktas_level", 0)) for r in r_opt}
        orig_map = {r["case_id"]: int(r["parsed"].get("final_ktas_level", 0)) for r in r_orig}
        common_ids = sorted(set(opt_map) & set(orig_map))
        agree = (
            sum(1 for cid in common_ids if orig_map[cid] == opt_map[cid]) / len(common_ids) * 100.0
            if common_ids else 0.0
        )
        tok_red = (m_orig["avg_tot"] - m_opt["avg_tot"]) / m_orig["avg_tot"] * 100.0 if m_orig["avg_tot"] else 0.0
        out_red = (m_orig["avg_out"] - m_opt["avg_out"]) / m_orig["avg_out"] * 100.0 if m_orig["avg_out"] else 0.0

        return [
            f"## {title} (`N={m_orig['n']}`건, A/B 최종 판정 일치율: `{agree:.2f}%`)",
            "",
            "| 평가 지표 | `T_orig` (원본 평면 트리) | `T_opt` (최적화 계층 트리) | 개선폭 (Delta) |",
            "| :--- | ---: | ---: | :--- |",
            f"| **[전제] 최종 등급 정확도 (Accuracy)** | {m_orig['acc']:.2f}% | {m_opt['acc']:.2f}% | {m_opt['acc'] - m_orig['acc']:+.2f}%p |",
            f"| **[전제] Macro-F1** | {m_orig['macro_f1']:.2f}% | {m_opt['macro_f1']:.2f}% | {m_opt['macro_f1'] - m_orig['macro_f1']:+.2f}%p |",
            f"| **[전제] 과소평가율 (Under-triage, ↓)** | {m_orig['under_triage']:.2f}% | {m_opt['under_triage']:.2f}% | {m_opt['under_triage'] - m_orig['under_triage']:+.2f}%p |",
            f"| **[지표 1] 필수 1차 축 누락률 (↓)** | {m_orig['missing_first']:.2f}% | {m_opt['missing_first']:.2f}% | {m_opt['missing_first'] - m_orig['missing_first']:+.2f}%p |",
            f"| **[지표 1] 근거 규칙 일치율 (Rule Citation, ↑)** | {m_orig['rule_cite_acc']:.2f}% | {m_opt['rule_cite_acc']:.2f}% | {m_opt['rule_cite_acc'] - m_orig['rule_cite_acc']:+.2f}%p |",
            f"| **[지표 2] 평균 입력 토큰 수** | {m_orig['avg_in']:.1f} | {m_opt['avg_in']:.1f} | {m_opt['avg_in'] - m_orig['avg_in']:+.1f} |",
            f"| **[지표 2] 평균 출력(추론) 토큰 수 (↓)** | {m_orig['avg_out']:.1f} | {m_opt['avg_out']:.1f} | **{m_opt['avg_out'] - m_orig['avg_out']:+.1f} ({-out_red:+.1f}%)** |",
            f"| **[지표 2] 환자당 총 토큰 수 (↓)** | {m_orig['avg_tot']:.1f} | {m_opt['avg_tot']:.1f} | **{m_opt['avg_tot'] - m_orig['avg_tot']:+.1f} ({-tok_red:+.1f}%)** |",
            "",
        ]

    lines = [
        "# KTAS `T_orig` vs `T_opt` LLM A/B 벤치마크 평가 결과 리포트",
        "",
        f"- **평가 모델**: `{model_label}` (`temperature=0.0`, 결정론적 생성)",
        f"- **완료된 추론 기록**: `T_orig` {len(orig_all)}건 / `T_opt` {len(opt_all)}건",
        "",
        "---",
        "",
        *format_section("1. 전체 평가 벤치마크 비교", orig_all, opt_all),
        "---",
        "",
        *format_section("2. 병목 Hotspot Top 20 주호소 단독 비교", orig_hs, opt_hs),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="KTAS T_orig vs T_opt LLM A/B 벤치마크 실행기")
    parser.add_argument("--backend", choices=["auto", "hf", "openai"], default="auto")
    parser.add_argument("--base_url", default=None, help="OpenAI 호환 API 주소 (예: http://localhost:8000/v1)")
    parser.add_argument("--model", default="", help="모델 ID 또는 로컬 HuggingFace 모델 경로")
    parser.add_argument("--cases", default=str(EXP_ROOT / "data" / "benchmark" / "ktas_eval_cases.json"))
    parser.add_argument("--raw-dir", default=str(EXP_ROOT / "data" / "raw"))
    parser.add_argument("--opt-json", default=str(EXP_ROOT / "data" / "optimized" / "ktas_optimized_tree.json"))
    parser.add_argument("--out-jsonl", default=str(EXP_ROOT / "data" / "benchmark" / "ab_raw_results.jsonl"))
    parser.add_argument("--out-report", default=str(EXP_ROOT / "reports" / "ab_benchmark_results.md"))
    parser.add_argument("--limit", type=int, default=0, help="상위 N건만 실행 (0이면 전체 300건)")
    parser.add_argument("--report-only", action="store_true", help="추론 없이 기존 ab_raw_results.jsonl로 리포트만 재생성")
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

        # 실행할 작업 목록 구성
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

    # 현재 케이스셋에 해당하는 레코드만 필터링해 채점
    filtered_records = [r for r in existing_records if r["case_id"] in cases_by_id]
    report_md = build_ab_report(filtered_records, cases_by_id, model_label)
    Path(args.out_report).write_text(report_md, encoding="utf-8")
    print(f"[+] A/B 벤치마크 리포트 저장 완료: {args.out_report}")


if __name__ == "__main__":
    main()

