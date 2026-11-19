#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS 최적화 트리(T_opt) 기반 300건 평가용 벤치마크 데이터셋 생성기.

구성:
- Hotspot Top 20 주호소에서 200건 (주호소당 10건)
- 일반 주호소에서 100건 (001/002 DNR 예외 게이트 케이스 12건 + 일반 주호소 88건)
- 각 케이스에 대해 결정론적 파이썬 기준 엔진(T_orig == T_opt)을 실행하여
  ground_truth_level, required_first_order_checks, matched_second_order_rule을 자동 라벨링.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP_ROOT / "src" / "analyzer"))
sys.path.insert(0, str(EXP_ROOT / "src" / "optimizer"))

from detect_inefficiencies import analyze_hotspots  # noqa: E402
from verify_equivalence import eval_t_opt, eval_t_orig, extract_rule_atoms  # noqa: E402


def _build_patient_presentation(
    opt_comp: dict[str, Any],
    active_conds: list[str],
    case_type: str,
    rng: random.Random,
) -> str:
    """활성 조건에 기반한 구조화·자연어 혼합 임상 시나리오 텍스트를 생성한다."""
    is_ped = opt_comp["context"] == "PED"
    age_str = f"만 {rng.randint(2, 13)}세 소아" if is_ped else f"만 {rng.randint(22, 76)}세 성인"
    if "임신" in opt_comp["name"] or "출산" in opt_comp["name"]:
        age_str = f"만 {rng.randint(25, 38)}세 가임기 여성"

    findings = ", ".join(active_conds)
    if case_type == "exception_gate":
        return (
            f"[{age_str}] 주호소: {opt_comp['name']} (NACRS {opt_comp['nacrs_code']}). "
            f"내원 시 확인된 임상 상태: {findings}. "
            "의식/호흡/순환 평가와 함께 연명의료결정(DNR) 여부를 우선 확인함."
        )
    if case_type == "first_order_axis":
        return (
            f"[{age_str}] 주호소: {opt_comp['name']} (NACRS {opt_comp['nacrs_code']}). "
            f"초기 활력징후 및 1차 고려사항 평가 결과: {findings}. "
            "해당 주호소의 기타 2차 세부 증상은 특이 소견 없음."
        )
    return (
        f"[{age_str}] 주호소: {opt_comp['name']} (NACRS {opt_comp['nacrs_code']}). "
        f"1차 활력징후는 특이 위험 징후 없으나, 문진 및 진찰상 다음 2차 세부 소견이 확인됨: {findings}."
    )


def _trace_opt_decision(
    opt_comp: dict[str, Any],
    assignment: dict[str, bool],
) -> tuple[int, list[str], dict[str, Any] | None, dict[str, Any] | None]:
    """T_opt 경로를 추적해 (최종 등급, 필수 1차 확인 축, 매칭 2차 규칙, 매칭 예외 게이트)를 반환한다."""
    req_first_axes = list(opt_comp.get("first_order_single_pass", {}).keys())

    for eg in opt_comp.get("exception_gates", []):
        if assignment.get(eg["condition"], False):
            return eg["ktas_level"], req_first_axes, None, eg

    first_by_lvl: dict[int, list[tuple[str, str]]] = {}
    for axis, rules in opt_comp.get("first_order_single_pass", {}).items():
        for r in rules:
            first_by_lvl.setdefault(r["ktas_level"], []).append((axis, r["condition"]))

    sec_groups = list(opt_comp.get("second_order_variable_groups", {}).items())
    for cur_lvl in range(1, 6):
        for axis, cond in first_by_lvl.get(cur_lvl, []):
            if assignment.get(cond, False):
                return cur_lvl, req_first_axes, None, None
        for axis, rules in sec_groups:
            if min((r["ktas_level"] for r in rules), default=99) > cur_lvl:
                continue
            for r in rules:
                if r["ktas_level"] != cur_lvl:
                    continue
                op, atoms = extract_rule_atoms(r["condition"])
                matched = any(assignment.get(a, False) for a in atoms) if op == "OR" else assignment.get(atoms[0], False)
                if matched:
                    return (
                        cur_lvl,
                        req_first_axes,
                        {
                            "variable_axis": axis,
                            "ktas_level": cur_lvl,
                            "condition": r["condition"],
                        },
                        None,
                    )
    raise ValueError(f"No matching rule for {opt_comp['name']}")


def _sample_case_for_complaint(
    opt_comp: dict[str, Any],
    orig_comp: dict[str, Any],
    preferred_type: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    """단일 주호소에서 지정된 유형(1차/2차/예외)의 유효 평가 케이스 1건을 생성 및 교차검증한다."""
    first_pool = [
        (axis, r["ktas_level"], r["condition"])
        for axis, rules in opt_comp.get("first_order_single_pass", {}).items()
        for r in rules
    ]
    sec_pool = [
        (axis, r["ktas_level"], r["condition"])
        for axis, rules in opt_comp.get("second_order_variable_groups", {}).items()
        for r in rules
    ]
    exc_pool = opt_comp.get("exception_gates", [])

    chosen_type = preferred_type
    active_conds: list[str] = []

    if chosen_type == "exception_gate" and exc_pool:
        # 50% DNR=True(5등급), 50% DNR=False + 심정지(1등급)
        if rng.random() < 0.5 and sec_pool:
            _, _, sec_c = rng.choice(sec_pool)
            active_conds = [sec_c, exc_pool[0]["condition"]]
        else:
            if sec_pool:
                active_conds = [rng.choice(sec_pool)[2]]
            else:
                return None
    elif chosen_type == "first_order_axis" and first_pool:
        _, _, c = rng.choice(first_pool)
        active_conds = [c]
        # 가끔 하위 등급 2차 조건을 함께 켜서 상위 1차 축 우선순위 테스트
        if sec_pool and rng.random() < 0.35:
            active_conds.append(rng.choice(sec_pool)[2])
    elif sec_pool:
        chosen_type = "second_order_variable"
        _, lvl, c = rng.choice(sec_pool)
        op, atoms = extract_rule_atoms(c)
        active_conds = [rng.choice(atoms) if op == "OR" else atoms[0]]
        # 하위 등급(더 큰 숫자) 노이즈 조건을 동반해 단락 평가 검증
        lower_candidates = [sc for _, sl, sc in sec_pool if sl > lvl]
        if lower_candidates and rng.random() < 0.30:
            op2, atoms2 = extract_rule_atoms(rng.choice(lower_candidates))
            active_conds.append(rng.choice(atoms2) if op2 == "OR" else atoms2[0])
    elif first_pool:
        chosen_type = "first_order_axis"
        active_conds = [rng.choice(first_pool)[2]]
    else:
        return None

    assignment = {c: True for c in active_conds}
    orig_lvl, orig_steps = eval_t_orig(orig_comp, assignment)
    opt_lvl, opt_steps = eval_t_opt(opt_comp, assignment)
    if orig_lvl is None or orig_lvl != opt_lvl:
        raise AssertionError(f"Equivalence mismatch on {opt_comp['name']}: orig={orig_lvl}, opt={opt_lvl}")

    gt_lvl, req_first, matched_sec, matched_exc = _trace_opt_decision(opt_comp, assignment)
    assert gt_lvl == opt_lvl

    # 실제 결정이 1차로 났는지 2차로 났는지 최종 분류 정리
    if matched_exc is not None:
        final_type = "exception_gate"
    elif matched_sec is not None:
        final_type = "second_order_variable"
    else:
        final_type = "first_order_axis"

    return {
        "context": opt_comp["context"],
        "category": opt_comp["category"],
        "nacrs_code": opt_comp["nacrs_code"],
        "complaint_name": opt_comp["name"],
        "case_type": final_type,
        "patient_presentation": _build_patient_presentation(opt_comp, active_conds, final_type, rng),
        "active_conditions": active_conds,
        "ground_truth_level": gt_lvl,
        "required_first_order_checks": req_first,
        "matched_second_order_rule": matched_sec,
        "matched_exception_gate": matched_exc,
        "eval_steps_orig": orig_steps,
        "eval_steps_opt": opt_steps,
    }


def generate_dataset(raw_dir: Path, opt_json: Path, seed: int = 2026) -> list[dict[str, Any]]:
    """Hotspot 200건 + 일반 100건 = 총 300건의 평가 벤치마크를 생성한다."""
    rng = random.Random(seed)

    # 원본 주호소 맵
    orig_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for json_path in sorted(raw_dir.glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ctx = "PED" if "Pediatric" in str(data.get("target_population", "")) else "ADULT"
        for _, cat_val in data.get("categories", {}).items():
            for code, comp in cat_val.get("complaints", {}).items():
                nacrs = str(comp.get("nacrs_code", code))
                orig_map[(ctx, nacrs, comp.get("name", ""))] = comp

    with opt_json.open("r", encoding="utf-8") as f:
        opt_complaints: list[dict[str, Any]] = json.load(f)["complaints"]

    opt_by_key = {(c["context"], c["nacrs_code"], c["name"]): c for c in opt_complaints}

    # Top 20 Hotspot 키 추출
    hotspot_results, _ = analyze_hotspots(raw_dir)
    top20_keys = [
        ("PED" if r["context"] == "소아" else "ADULT", r["code"], r["name"])
        for r in hotspot_results[:20]
    ]
    top20_set = set(top20_keys)

    cases: list[dict[str, Any]] = []

    # 1) Hotspot Top 20에서 200건 (주호소당 10건: 1차 3건 + 2차 7건 비율 목표)
    hs_idx = 1
    for key in top20_keys:
        opt_c = opt_by_key[key]
        orig_c = orig_map[key]
        for slot in range(10):
            pref = "first_order_axis" if slot < 3 else "second_order_variable"
            sample = _sample_case_for_complaint(opt_c, orig_c, pref, rng)
            assert sample is not None
            sample["case_id"] = f"EVAL_HS_{hs_idx:03d}"
            sample["cohort"] = "hotspot_top20"
            cases.append(sample)
            hs_idx += 1

    # 2) 일반 주호소에서 100건 (예외 게이트 001/002 12건 + 일반 주호소 88건)
    exc_complaints = [c for c in opt_complaints if c.get("exception_gates")]
    gen_pool = [
        c for c in opt_complaints
        if (c["context"], c["nacrs_code"], c["name"]) not in top20_set
        and not c.get("exception_gates")
        and (c.get("first_order_single_pass") or c.get("second_order_variable_groups"))
    ]

    gen_idx = 1
    # 2a) 예외 게이트 12건 (4개 심정지 주호소 × 3건씩)
    for opt_c in exc_complaints:
        orig_c = orig_map[(opt_c["context"], opt_c["nacrs_code"], opt_c["name"])]
        for _ in range(3):
            sample = _sample_case_for_complaint(opt_c, orig_c, "exception_gate", rng)
            assert sample is not None
            sample["case_id"] = f"EVAL_GEN_{gen_idx:03d}"
            sample["cohort"] = "general"
            cases.append(sample)
            gen_idx += 1

    # 2b) 일반 주호소 88건
    rng.shuffle(gen_pool)
    pool_iter = 0
    while gen_idx <= 100:
        opt_c = gen_pool[pool_iter % len(gen_pool)]
        orig_c = orig_map[(opt_c["context"], opt_c["nacrs_code"], opt_c["name"])]
        pref = "first_order_axis" if (gen_idx % 3 == 0) else "second_order_variable"
        sample = _sample_case_for_complaint(opt_c, orig_c, pref, rng)
        if sample is not None:
            sample["case_id"] = f"EVAL_GEN_{gen_idx:03d}"
            sample["cohort"] = "general"
            cases.append(sample)
            gen_idx += 1
        pool_iter += 1

    return cases


def build_summary_report(cases: list[dict[str, Any]]) -> str:
    """benchmark_dataset_summary.md 리포트를 생성한다."""
    total = len(cases)
    cohort_cnt = Counter(c["cohort"] for c in cases)
    type_cnt = Counter(c["case_type"] for c in cases)
    level_cnt = Counter(c["ground_truth_level"] for c in cases)
    ctx_cnt = Counter(c["context"] for c in cases)

    avg_orig_steps = sum(c["eval_steps_orig"] for c in cases) / total
    avg_opt_steps = sum(c["eval_steps_opt"] for c in cases) / total

    lines = [
        "# KTAS 최적화 평가 벤치마크 데이터셋 요약 리포트",
        "",
        f"- **총 생성 평가 케이스 수**: **{total}건** (`data/benchmark/ktas_eval_cases.json`)",
        f"- **코호트 구성**: Hotspot Top 20 주호소 `{cohort_cnt['hotspot_top20']}건` / 일반 주호소 `{cohort_cnt['general']}건`",
        f"- **연령 컨텍스트 분포**: 성인(ADULT) `{ctx_cnt['ADULT']}건` / 소아(PED) `{ctx_cnt['PED']}건`",
        f"- **기준 엔진 실행 평균 스텝 수 (`T_orig` ➔ `T_opt`)**: `{avg_orig_steps:.2f}` ➔ **`{avg_opt_steps:.2f}`**",
        "",
        "## 1. 케이스 유형별 분포",
        "| 유형 (`case_type`) | 건수 | 비율 (%) | 설명 |",
        "| :--- | ---: | ---: | :--- |",
        f"| `second_order_variable` | {type_cnt['second_order_variable']} | {type_cnt['second_order_variable']/total*100:.1f}% | 2차 세부 변수(혈압/진통/출혈 등)로 최종 결정 |",
        f"| `first_order_axis` | {type_cnt['first_order_axis']} | {type_cnt['first_order_axis']/total*100:.1f}% | 1차 공통 축(활력징후/통증 등)으로 최종 결정 |",
        f"| `exception_gate` | {type_cnt['exception_gate']} | {type_cnt['exception_gate']/total*100:.1f}% | 심정지(`001`/`002`) DNR 예외 게이트 분기 |",
        "",
        "## 2. KTAS 1~5등급별 정답(`ground_truth_level`) 분포",
        "| KTAS 등급 | 건수 | 비율 (%) |",
        "| :---: | ---: | ---: |",
    ]
    for lvl in range(1, 6):
        cnt = level_cnt[lvl]
        lines.append(f"| {lvl}등급 | {cnt} | {cnt/total*100:.1f}% |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="KTAS Phase 3 평가 벤치마크 데이터셋 생성기")
    parser.add_argument("--raw-dir", default=str(EXP_ROOT / "data" / "raw"))
    parser.add_argument("--opt-json", default=str(EXP_ROOT / "data" / "optimized" / "ktas_optimized_tree.json"))
    parser.add_argument("--out-cases", default=str(EXP_ROOT / "data" / "benchmark" / "ktas_eval_cases.json"))
    parser.add_argument("--out-report", default=str(EXP_ROOT / "reports" / "benchmark_dataset_summary.md"))
    args = parser.parse_args()

    cases = generate_dataset(Path(args.raw_dir), Path(args.opt_json))
    out_cases = Path(args.out_cases)
    out_cases.parent.mkdir(parents=True, exist_ok=True)
    with out_cases.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    report = build_summary_report(cases)
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report, encoding="utf-8")

    print(report)
    print(f"[+] 벤치마크 데이터셋 저장 완료: {out_cases}")


if __name__ == "__main__":
    main()

