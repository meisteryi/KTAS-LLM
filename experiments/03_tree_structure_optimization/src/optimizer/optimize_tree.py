#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS 결정 트리 최적화 변환기 (T_orig -> T_opt).

설계 원칙:
1. 예외 게이트 보존: evaluation_priority == "exception_first" (심정지 DNR 등) 최상단 분리.
2. 1차 공통 축 Single-Pass 통합: 2차 고려사항에 흩어진 순수 1차 활력징후/공통 참조를
   단일 1차 평가 블록으로 통합하여 중복 제거.
3. 2차 조건 변수별 그룹화(Variable Factoring): 반복 등장하는 임상 변수 축 단위로 묶고,
   각 그룹 내부는 상위 등급(1->5) 순서를 유지.
4. 최종 등급 산출: 예외 게이트 우선 적용 후, 1차 Single-Pass 결과와 2차 변수 그룹 결과 중
   최고 중증도(최소 KTAS 숫자) 선택.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyzer"))
from detect_inefficiencies import extract_axes  # noqa: E402


OR_SPLIT_RE = re.compile(r"\s+(?:또는|혹은)\s+")
PURE_FIRST_ORDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("1차_활력징후", re.compile(r"^(?:소아\s*)?1차\s*활력징후\s*([1-5])\s*단계(?:\s*조건)?(?:\s*만족)?$")),
    ("통증_척도", re.compile(r"^(?:급성|만성)?\s*(?:중심성|말초성)?(?:[/·](?:중심성|말초성))?\s*(?:중증|중등도|경증)?\s*통증\s*([1-5])\s*단계(?:\s*만족)?$")),
    ("통증_척도", re.compile(r"^(?:급성|만성)\s*(?:중심성|말초성)?\s*(?:중증|중등도|경증)?\s*통증\s*\([^)]+\)$")),
    ("출혈_양상", re.compile(r"^출혈성\s*질환\s*(?:[1-5]\s*단계|\([^)]+\))(?:\s*만족)?$")),
    ("고위험_사고기전", re.compile(r"^고위험(?:성)?\s*사고기전$")),
]


def classify_pure_first_order(atom: str) -> str | None:
    """단일 원자 조건이 순수 1차 공통 축 참조인지 판별하고 축 이름을 반환한다."""
    s = atom.strip().rstrip(",")
    for axis, pat in PURE_FIRST_ORDER_PATTERNS:
        if pat.match(s):
            return axis
    return None


def split_or_items(cond: str) -> tuple[str, list[str]]:
    """OR 및 쉼표 나열된 순수 1차 참조를 일관되게 분해한다."""
    has_and_connector = any(k in cond for k in ("이면서", "이고", "동반", "+"))
    if has_and_connector:
        return "ATOM", [cond.strip()]

    raw_or_parts = [p.strip().rstrip(",") for p in OR_SPLIT_RE.split(cond) if p.strip().rstrip(",")]
    expanded_parts: list[str] = []
    for part in raw_or_parts:
        if "," in part:
            comma_parts = [cp.strip().rstrip(",") for cp in part.split(",") if cp.strip().rstrip(",")]
            if comma_parts and all(classify_pure_first_order(cp) is not None for cp in comma_parts):
                expanded_parts.extend(comma_parts)
                continue
        expanded_parts.append(part)

    if len(expanded_parts) > 1:
        return "OR", expanded_parts
    if len(expanded_parts) == 1 and classify_pure_first_order(expanded_parts[0]) is not None:
        return "ATOM", [expanded_parts[0]]
    return "ATOM", [cond.strip()]


def optimize_complaint(
    comp: dict[str, Any],
    code: str,
    cat_name: str,
    is_ped: bool,
    source_file: str,
) -> dict[str, Any]:
    """단일 주호소를 T_opt 구조(예외 게이트 + 1차 Single-Pass + 2차 변수 그룹)로 변환한다."""
    mods = comp.get("second_order_modifiers", [])
    eval_priority = comp.get("evaluation_priority", "standard")
    first_applicable = comp.get("first_order_modifiers_applicable", [])

    exception_gates: list[dict[str, Any]] = []
    first_order_single_pass: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_first_atoms: set[tuple[str, int, str]] = set()
    second_order_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, mod in enumerate(mods):
        lvl = mod.get("ktas_level")
        cond = str(mod.get("condition", "")).strip()
        if not isinstance(lvl, int) or not cond:
            continue

        # 1) 예외 게이트 (DNR 등 exception_first 주호소의 5단계 예외)
        if eval_priority == "exception_first" and lvl == 5 and any(
            k in cond for k in ("DNR", "심폐소생술의 대상이 아닌")
        ):
            exception_gates.append({"ktas_level": lvl, "condition": cond, "orig_index": idx})
            continue

        # 2) OR/쉼표 나열로 결합된 순수 1차 참조 분리 (단일 shared helper 사용)
        op, parts = split_or_items(cond)
        remaining_parts: list[str] = []
        for part in parts:
            axis = classify_pure_first_order(part)
            if axis is not None:
                key = (axis, lvl, part)
                if key not in seen_first_atoms:
                    seen_first_atoms.add(key)
                    first_order_single_pass[axis].append(
                        {"ktas_level": lvl, "condition": part, "orig_index": idx}
                    )
            else:
                remaining_parts.append(part)

        if not remaining_parts:
            continue
        cond_for_second = " 또는 ".join(remaining_parts) if op == "OR" else remaining_parts[0]

        # 3) 2차 조건 변수별 그룹화
        axes = extract_axes(cond_for_second)
        primary_axis = axes[0]
        second_order_groups[primary_axis].append(
            {"ktas_level": lvl, "condition": cond_for_second, "orig_index": idx}
        )

    # 각 그룹 내부는 상위 등급(1 -> 5) 순서 유지
    for axis in first_order_single_pass:
        first_order_single_pass[axis].sort(key=lambda x: (x["ktas_level"], x["orig_index"]))
    for axis in second_order_groups:
        second_order_groups[axis].sort(key=lambda x: (x["ktas_level"], x["orig_index"]))

    # 변수 그룹 순서도 최상위 도달 가능 등급(min ktas_level) 순으로 정렬해 조기 종료 극대화
    ordered_second_groups = dict(
        sorted(
            second_order_groups.items(),
            key=lambda kv: (min(r["ktas_level"] for r in kv[1]), -len(kv[1]), kv[0]),
        )
    )

    return {
        "source_file": source_file,
        "context": "PED" if is_ped else "ADULT",
        "category": cat_name,
        "nacrs_code": str(comp.get("nacrs_code", code)),
        "name": comp.get("name", code),
        "evaluation_priority": eval_priority,
        "decision_rule": "exception_first_else_min(first_order_single_pass, second_order_variable_groups)",
        "exception_gates": exception_gates,
        "first_order_modifiers_applicable": first_applicable,
        "first_order_single_pass": dict(first_order_single_pass),
        "second_order_variable_groups": ordered_second_groups,
    }


def optimize_all(raw_dir: Path) -> dict[str, Any]:
    """전체 Ground Truth JSON 파일들을 읽어 단일 T_opt 트리 구조로 변환한다."""
    with (raw_dir / "ktas_part1_common.json").open("r", encoding="utf-8") as f:
        common = json.load(f)

    optimized_complaints: list[dict[str, Any]] = []
    for json_path in sorted(raw_dir.glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        is_ped = "Pediatric" in str(data.get("target_population", ""))
        for cat_id, cat_val in data.get("categories", {}).items():
            cat_name = cat_val.get("category_name", cat_id)
            for code, comp in cat_val.get("complaints", {}).items():
                optimized_complaints.append(
                    optimize_complaint(comp, code, cat_name, is_ped, json_path.name)
                )

    return {
        "version": "KTAS_2021_ver1.6_Optimized_T_opt",
        "common_framework": common,
        "complaints": optimized_complaints,
    }


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="KTAS 트리 최적화 변환기 (T_orig -> T_opt)")
    parser.add_argument("--raw-dir", default=str(exp_root / "data" / "raw"))
    parser.add_argument("--output", "-o", default=str(exp_root / "data" / "optimized" / "ktas_optimized_tree.json"))
    args = parser.parse_args()

    t_opt = optimize_all(Path(args.raw_dir))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(t_opt, f, ensure_ascii=False, indent=2)

    print(f"[+] T_opt 생성 완료: {len(t_opt['complaints'])}개 주호소 -> {out_path}")


if __name__ == "__main__":
    main()
