#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS Ground Truth 기반 보수적 원자 명제 정규화 및 중복 분석 스크립트.

원칙:
1. 무손실 구조적 정규화: AND/OR 복합 조건을 최소 명제로 분해하되 수식어(연령, 발병, 부위 등) 손실 금지.
2. 자동 동일 ID 매핑은 ① 공백/구두점 정규화 후 완전 일치, ② 동일 생체징후 지표 + 동일 부등호 + 동일 수치(단위 환산 포함)만 허용.
3. 유사 표현(수치 동일하나 서술 상이, 고도 유사 문자열)은 절대 자동 병합하지 않고 merge_candidates.md로 분리 출력.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


# 복합 조건 분해 패턴 (수식어를 자르지 않도록 명시적 접속사/구분자 기준)
OR_SPLIT_RE = re.compile(r"\s+(?:또는|혹은)\s+")
AND_SPLIT_RE = re.compile(r"\s*(?:,\s*(?=1차|소아\s*1차|중심성|말초성|출혈성|고위험|열\b|급성|만성)|;\s*|\s+이면서\s+|\s+이고\s+)")

# 생체징후/수치 정규화 패턴: 지표명 + 부등호 + 수치 + 단위
METRIC_ALIASES = {
    "수축기 혈압": "SBP",
    "수축기혈압": "SBP",
    "sbp": "SBP",
    "이완기 혈압": "DBP",
    "이완기혈압": "DBP",
    "dbp": "DBP",
    "체온": "BT",
    "t": "BT",
    "산소포화도": "SPO2",
    "산소 포화도": "SPO2",
    "spo2": "SPO2",
    "혈당": "GLU",
    "심박수": "HR",
    "호흡수": "RR",
    "gcs": "GCS",
    "pefr": "PEFR",
    "fev1": "FEV1",
    "임신주수": "GA_WEEKS",
}

METRIC_EXPR_RE = re.compile(
    r"^(?P<metric>[A-Za-z가-힣0-9\s]+?)\s*(?P<op>>=|<=|>|<|=|≥|≤)\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>mmHg|mg/dL|mg/dl|°C|℃|%|주|개월|세|시간|분|회/분)?$",
    re.IGNORECASE,
)

OP_NORM = {"≥": ">=", "≤": "<="}
UNIT_NORM = {"mg/dl": "mg/dL", "℃": "°C"}

FIRST_REF_RE = re.compile(r"(?:소아\s*)?1차\s*활력징후\s*([1-5])\s*단계")


def normalize_whitespace_punct(text: str) -> str:
    """공백·괄호·단위 표기를 보수적으로 정규화한다 (의미 수식어 변경 없음)."""
    s = text.strip()
    s = s.replace("℃", "°C").replace("≥", ">=").replace("≤", "<=")
    s = re.sub(r"mg\s*/\s*dl", "mg/dL", s, flags=re.IGNORECASE)
    s = re.sub(r"mmhg", "mmHg", s, flags=re.IGNORECASE)
    s = re.sub(r"spo2", "SpO2", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


def canonical_atom_key(atom: str, context: str) -> tuple[str, str]:
    """엄격 규칙에 따라 원자 명제의 정규화 키와 매칭 방식(exact_text / exact_metric)을 반환한다."""
    norm = normalize_whitespace_punct(atom)
    m = METRIC_EXPR_RE.match(norm)
    if m:
        raw_metric = m.group("metric").strip().lower()
        metric = METRIC_ALIASES.get(raw_metric)
        if metric:
            op = OP_NORM.get(m.group("op"), m.group("op"))
            val = float(m.group("val"))
            val_str = str(int(val)) if val.is_integer() else str(val)
            raw_unit = (m.group("unit") or "").strip()
            unit = UNIT_NORM.get(raw_unit.lower(), raw_unit)
            return f"METRIC|{context}|{metric}|{op}|{val_str}|{unit}", "exact_metric"
    return f"TEXT|{context}|{norm}", "exact_text"


def split_compound_condition(cond: str) -> tuple[str, list[str]]:
    """복합 조건을 OR/AND 최상위 경계에서 무손실 분해한다."""
    norm = normalize_whitespace_punct(cond)
    or_parts = [p.strip() for p in OR_SPLIT_RE.split(norm) if p.strip()]
    if len(or_parts) > 1:
        atoms: list[str] = []
        for part in or_parts:
            sub = [s.strip() for s in AND_SPLIT_RE.split(part) if s.strip()]
            atoms.extend(sub)
        return "OR", atoms

    and_parts = [p.strip() for p in AND_SPLIT_RE.split(norm) if p.strip()]
    if len(and_parts) > 1:
        return "AND", and_parts
    return "ATOM", [norm]


def map_first_order_reference(
    atom: str,
    is_ped: bool,
    first_applicable: list[str],
    common_data: dict[str, Any],
) -> list[str]:
    """1차 활력징후 X단계 참조를 실제 first_order_modifiers_applicable 및 ktas_part1_common.json 축 ID로 연결한다."""
    m = FIRST_REF_RE.search(atom)
    if not m:
        return []
    lvl = int(m.group(1))
    prefix = "pediatric" if is_ped else "adult"
    common_group = common_data.get(f"{prefix}_first_order_modifiers", {})

    linked_ids: list[str] = []
    # 1) 해당 주호소의 first_order_modifiers_applicable 중 동일 등급 항목 연결
    for item in first_applicable:
        if item.endswith(f"({lvl})"):
            linked_ids.append(f"{prefix}:applicable:{item}")

    # 2) ktas_part1_common.json의 실제 축(consciousness, hemodynamics, respiratory, temperature 등) 연결
    for axis_name, axis_val in common_group.items():
        if isinstance(axis_val, list):
            for entry in axis_val:
                if isinstance(entry, dict) and entry.get("ktas_level") == lvl:
                    linked_ids.append(f"{prefix}:{axis_name}:L{lvl}")
                    break
    return linked_ids


def check_first_second_overlap(
    atom: str,
    ktas_level: int | None,
    first_applicable: list[str],
) -> str | None:
    """2차 고려사항의 원자 명제가 이미 해당 주호소의 1차 적용 목록에 포함되어 있는지 확인한다."""
    if not first_applicable or ktas_level is None:
        return None
    norm_atom = normalize_whitespace_punct(atom)
    if FIRST_REF_RE.search(norm_atom):
        return None  # 명시적 cross_reference 자체는 참조 링크로 분류
    for fa in first_applicable:
        # "중증 호흡곤란 (1)" -> cond="중증 호흡곤란", lvl=1
        fa_m = re.match(r"^(.*)\((\d+)\)$", fa.strip())
        if not fa_m:
            continue
        fa_cond = normalize_whitespace_punct(fa_m.group(1))
        fa_lvl = int(fa_m.group(2))
        if fa_lvl == ktas_level and (fa_cond == norm_atom or (len(fa_cond) >= 4 and fa_cond in norm_atom)):
            return fa
    return None


def find_merge_candidates(
    pred_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """자동 병합하지 않은 서로 다른 표현 중 수치 공유 또는 높은 유사도를 가진 검토 후보군을 추출한다."""
    candidates: list[dict[str, Any]] = []
    items = [
        v for v in pred_records.values()
        if not FIRST_REF_RE.search(v["canonical_text"]) and len(v["canonical_text"]) >= 4
    ]
    num_re = re.compile(r"\d+(?:\.\d+)?")

    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(items)):
        a = items[i]
        a_txt = a["canonical_text"]
        a_nums = set(num_re.findall(a_txt))
        for j in range(i + 1, len(items)):
            b = items[j]
            if a["context"] != b["context"]:
                continue  # 성인/소아 컨텍스트가 다르면 병합 후보에서도 제외
            b_txt = b["canonical_text"]
            if a_txt == b_txt:
                continue

            sim = SequenceMatcher(None, a_txt.replace(" ", ""), b_txt.replace(" ", "")).ratio()
            b_nums = set(num_re.findall(b_txt))
            shared_nums = a_nums & b_nums

            reason = None
            if sim >= 0.76:
                reason = f"문자열 고도 유사 (유사도 {sim:.2f}) — 수식어/표현 차이 검증 필요"
            elif shared_nums and sim >= 0.50:
                reason = f"동일 수치({', '.join(sorted(shared_nums))}) 공유 및 부분 유사 (유사도 {sim:.2f})"

            if reason:
                pair_key = tuple(sorted((a["pred_id"], b["pred_id"])))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append(
                    {
                        "pred_a": a,
                        "pred_b": b,
                        "sim": sim,
                        "reason": reason,
                    }
                )

    candidates.sort(key=lambda x: x["sim"], reverse=True)
    return candidates[:80]


def run_analysis(raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """전체 Ground Truth JSON을 정규화하고 유형 A/B 중복 및 병합 후보를 수집한다."""
    with (raw_dir / "ktas_part1_common.json").open("r", encoding="utf-8") as f:
        common_data = json.load(f)

    pred_records: dict[str, dict[str, Any]] = {}
    key_to_pid: dict[str, str] = {}
    pid_counter = 0

    total_raw_conditions = 0
    total_atomic_occurrences = 0
    metric_merged_count = 0
    ref_linked_count = 0
    type_b_overlaps: list[dict[str, Any]] = []
    normalized_trees: list[dict[str, Any]] = []

    for json_path in sorted(raw_dir.glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        is_ped = "Pediatric" in str(data.get("target_population", ""))
        context = "PED" if is_ped else "ADULT"

        for cat_id, cat_val in data.get("categories", {}).items():
            cat_name = cat_val.get("category_name", cat_id)
            for code, comp in cat_val.get("complaints", {}).items():
                comp_name = comp.get("name", code)
                first_app = comp.get("first_order_modifiers_applicable", [])
                second_mods = comp.get("second_order_modifiers", [])

                comp_norm_rules = []
                for mod in second_mods:
                    total_raw_conditions += 1
                    lvl = mod.get("ktas_level")
                    cond = str(mod.get("condition", "")).strip()
                    logic_op, atoms = split_compound_condition(cond)

                    atom_ids = []
                    for atom in atoms:
                        total_atomic_occurrences += 1
                        ckey, match_mode = canonical_atom_key(atom, context)
                        if ckey not in key_to_pid:
                            pid_counter += 1
                            pid = f"P_{context}_{pid_counter:04d}"
                            key_to_pid[ckey] = pid
                            pred_records[pid] = {
                                "pred_id": pid,
                                "context": context,
                                "canonical_key": ckey,
                                "canonical_text": normalize_whitespace_punct(atom),
                                "match_mode": match_mode,
                                "count": 0,
                                "occurrences": [],
                                "first_order_links": [],
                            }
                        else:
                            if match_mode == "exact_metric":
                                metric_merged_count += 1

                        pid = key_to_pid[ckey]
                        rec = pred_records[pid]
                        rec["count"] += 1
                        occ_label = f"{comp_name}({code}) [KTAS {lvl}]"
                        if occ_label not in rec["occurrences"]:
                            rec["occurrences"].append(occ_label)

                        # 1차 참조 연결
                        links = map_first_order_reference(atom, is_ped, first_app, common_data)
                        if links:
                            ref_linked_count += 1
                            for lk in links:
                                if lk not in rec["first_order_links"]:
                                    rec["first_order_links"].append(lk)

                        # 유형 B: 1차-2차 중복 검사
                        dup_fa = check_first_second_overlap(atom, lvl, first_app)
                        if dup_fa:
                            type_b_overlaps.append(
                                {
                                    "file": json_path.name,
                                    "context": context,
                                    "complaint": f"{cat_name} / {comp_name} ({code})",
                                    "ktas_level": lvl,
                                    "first_order_item": dup_fa,
                                    "second_order_atom": atom,
                                    "full_condition": cond,
                                }
                            )

                        atom_ids.append(pid)

                    comp_norm_rules.append(
                        {
                            "ktas_level": lvl,
                            "logic_op": logic_op,
                            "atom_ids": atom_ids,
                            "raw_condition": cond,
                        }
                    )

                normalized_trees.append(
                    {
                        "file": json_path.name,
                        "context": context,
                        "category": cat_name,
                        "nacrs_code": str(comp.get("nacrs_code", code)),
                        "name": comp_name,
                        "evaluation_priority": comp.get("evaluation_priority", "standard"),
                        "first_order_modifiers_applicable": first_app,
                        "rules": comp_norm_rules,
                    }
                )

    candidates = find_merge_candidates(pred_records)
    summary = {
        "total_raw_conditions": total_raw_conditions,
        "total_atomic_occurrences": total_atomic_occurrences,
        "unique_predicates": len(pred_records),
        "exact_duplicate_atoms": total_atomic_occurrences - len(pred_records),
        "safe_compression_rate": (
            (total_atomic_occurrences - len(pred_records)) / total_atomic_occurrences * 100.0
            if total_atomic_occurrences
            else 0.0
        ),
        "metric_merged_count": metric_merged_count,
        "ref_linked_count": ref_linked_count,
        "type_b_count": len(type_b_overlaps),
    }
    artifacts = {
        "summary": summary,
        "predicates": pred_records,
        "trees": normalized_trees,
    }
    return summary, type_b_overlaps, {"candidates": candidates, "artifacts": artifacts}


def build_overlap_report(summary: dict[str, Any], pred_records: dict[str, dict[str, Any]], type_b: list[dict[str, Any]]) -> str:
    """overlap_and_normalization_report.md 본문을 생성한다."""
    adult_multi = sorted(
        [r for r in pred_records.values() if r["context"] == "ADULT" and r["count"] >= 2],
        key=lambda x: x["count"],
        reverse=True,
    )
    ped_multi = sorted(
        [r for r in pred_records.values() if r["context"] == "PED" and r["count"] >= 2],
        key=lambda x: x["count"],
        reverse=True,
    )
    lines = [
        "# KTAS 원자 명제 보수적 정규화 및 완전 중복 분석 리포트",
        "",
        "## 1. 무손실 정규화 및 안전 압축률 요약",
        f"- **원본 2차 조건식 수**: {summary['total_raw_conditions']}개",
        f"- **무손실 분해 후 전체 원자 명제 출현 수**: {summary['total_atomic_occurrences']}개",
        f"- **고유 원자 명제 ID 수 (`Strict Match`, 성인/소아 컨텍스트 완전 분리)**: **{summary['unique_predicates']}개**",
        f"- **완전 일치 중복 명제 수**: {summary['exact_duplicate_atoms']}개 (단위/부등호 정규화 병합 {summary['metric_merged_count']}건 포함)",
        f"- **완전 일치 중복만 제거 시 안전 압축률**: **{summary['safe_compression_rate']:.2f}%**",
        f"- **1차 공통 축과 연결된 참조 명제 출현 수**: {summary['ref_linked_count']}건",
        f"- **유형 B (동일 주호소 내 1차-2차 간 중복) 검출 건수**: **{summary['type_b_count']}건**",
        "",
        "---",
        "",
        "## 2. 유형 A: 동일 연령 컨텍스트 내 주호소 간 100% 완전 중복 조건",
        "> 성인(`ADULT`)과 소아(`PED`)는 별도 ID 네임스페이스로 분리되어 있으며, 아래 표는 각각 **성인 주호소끼리**, **소아 주호소끼리** 겹치는 조건만 집계한 결과입니다.",
        "",
        "### 2.1 성인 (`ADULT`) 주호소 간 중복 Top 15",
        "| 순위 | 명제 ID | 빈도 | 정규화 명제 원문 | 대표 등장 성인 주호소 (최대 3개) |",
        "| :---: | :--- | ---: | :--- | :--- |",
    ]
    for idx, r in enumerate(adult_multi[:15], start=1):
        occs = ", ".join(r["occurrences"][:3])
        lines.append(f"| {idx} | `{r['pred_id']}` | {r['count']} | {r['canonical_text']} | {occs} |")

    lines.extend([
        "",
        "### 2.2 소아 (`PED`) 주호소 간 중복 Top 15",
        "| 순위 | 명제 ID | 빈도 | 정규화 명제 원문 | 대표 등장 소아 주호소 (최대 3개) |",
        "| :---: | :--- | ---: | :--- | :--- |",
    ])
    for idx, r in enumerate(ped_multi[:15], start=1):
        occs = ", ".join(r["occurrences"][:3])
        lines.append(f"| {idx} | `{r['pred_id']}` | {r['count']} | {r['canonical_text']} | {occs} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. 유형 B: 1차-2차 고려사항 간 중복 조건 목록",
        f"> 해당 주호소의 `first_order_modifiers_applicable`에 이미 포함된 조건이 `second_order_modifiers`에도 중복 기재된 사례 ({len(type_b)}건)",
        "",
        "| # | 파일 | 주호소 | 등급 | 1차 적용 목록 내 항목 | 2차 고려사항 내 중복 명제 |",
        "| :---: | :--- | :--- | :---: | :--- | :--- |",
    ])
    for idx, tb in enumerate(type_b[:50], start=1):
        lines.append(
            f"| {idx} | `{tb['file']}` | {tb['complaint']} | {tb['ktas_level']} | {tb['first_order_item']} | {tb['second_order_atom']} |"
        )
    if not type_b:
        lines.append("| - | - | - | - | - | 중복 없음 |")

    return "\n".join(lines) + "\n"


def build_candidates_report(candidates: list[dict[str, Any]]) -> str:
    """사람(연구자)이 직접 검토할 유사 표현 후보군 리포트(merge_candidates.md)를 생성한다."""
    lines = [
        "# 유사 표현 병합 검토 후보군 (자동 병합 금지 목록)",
        "",
        "> **주의**: 아래 표현들은 보수적 자동 병합 규칙(문자열 완전 일치 / 동일 지표+부등호+수치)을 통과하지 못해 **서로 다른 Predicate ID로 분리 유지**된 항목입니다. 의학적 의미 동일성 여부를 직접 검토한 뒤에만 병합을 승인하십시오.",
        "",
        "| 후보 ID | 표현 A (등장 주호소 및 등급) | 표현 B (등장 주호소 및 등급) | 차이점 요약 |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for idx, c in enumerate(candidates, start=1):
        pa = c["pred_a"]
        pb = c["pred_b"]
        a_occ = "; ".join(pa["occurrences"][:2])
        b_occ = "; ".join(pb["occurrences"][:2])
        lines.append(
            f"| `CAND_{idx:03d}` | `{pa['pred_id']}`: **{pa['canonical_text']}**<br>*(등장: {a_occ})* | "
            f"`{pb['pred_id']}`: **{pb['canonical_text']}**<br>*(등장: {b_occ})* | {c['reason']} |"
        )
    if not candidates:
        lines.append("| - | 후보 없음 | - | - |")
    return "\n".join(lines) + "\n"


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="KTAS 보수적 원자 명제 정규화 및 중복 분석")
    parser.add_argument("--raw-dir", default=str(exp_root / "data" / "raw"))
    parser.add_argument("--out-report", default=str(exp_root / "reports" / "overlap_and_normalization_report.md"))
    parser.add_argument("--out-candidates", default=str(exp_root / "reports" / "merge_candidates.md"))
    parser.add_argument("--out-json", default=str(exp_root / "data" / "normalized_predicates.json"))
    args = parser.parse_args()

    summary, type_b, extra = run_analysis(Path(args.raw_dir))
    artifacts = extra["artifacts"]
    candidates = extra["candidates"]

    Path(args.out_report).write_text(
        build_overlap_report(summary, artifacts["predicates"], type_b), encoding="utf-8"
    )
    Path(args.out_candidates).write_text(build_candidates_report(candidates), encoding="utf-8")
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)

    print(f"[+] 정규화 완료: 원본 조건 {summary['total_raw_conditions']}개 -> 원자 출현 {summary['total_atomic_occurrences']}개 -> 고유 명제 {summary['unique_predicates']}개")
    print(f"[+] 안전 압축률: {summary['safe_compression_rate']:.2f}% | 유형 B(1차-2차 중복): {summary['type_b_count']}건 | 유사 후보: {len(candidates)}쌍")
    print(f"[+] 저장: {args.out_report}, {args.out_candidates}, {args.out_json}")


if __name__ == "__main__":
    main()

