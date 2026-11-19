#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS 결정 트리 정밀 병목 및 무결성 분석 스크립트.

1. ktas_part1_common.json 기반 cross_reference 실체 및 컨텍스트(성인/소아/통증) 매핑 분석
2. 탐색 깊이 병목(Outlier) 주호소(최대 깊이 Top 10, 1~2등급 심층 도달 케이스) 및 원인 분석
3. 데이터 무결성 전수 점검 (누락/빈 second_order_modifiers, 역전된 등급 순서 등)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inspect_ktas import REF_CONDITION_RE, _safe_int


LEVEL_MENTION_RE = re.compile(r"([1-5])\s*단계")
PAIN_MENTION_RE = re.compile(r"(?:통증|중심성|말초성)")
PED_HINT_RE = re.compile(r"소아")
FIRST_VITAL_RE = re.compile(r"1차\s*활력징후")
DNR_RE = re.compile(r"DNR|심폐소생술의 대상이 아닌")


def parse_common_framework(common_path: Path) -> dict[str, Any]:
    """ktas_part1_common.json에서 성인/소아 1차 고려사항 및 통증 매트릭스를 파싱한다."""
    with common_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    adult = data.get("adult_first_order_modifiers", {})
    ped = data.get("pediatric_first_order_modifiers", {})
    pain = data.get("pain_decision_matrix", {})

    def summarize_group(group: dict[str, Any]) -> dict[str, dict[int, list[str]]]:
        summary: dict[str, dict[int, list[str]]] = {}
        for axis, items in group.items():
            if not isinstance(items, list) or not any(isinstance(i, dict) for i in items):
                continue
            by_lvl: dict[int, list[str]] = defaultdict(list)
            for item in items:
                if not isinstance(item, dict):
                    continue
                lvl = _safe_int(item.get("ktas_level"))
                label = item.get("classification") or item.get("age_range") or "기준"
                crit = item.get("criteria", "")
                extra = []
                for k in ("gcs_range", "avpu", "spo2", "pefr", "age_range"):
                    if k in item:
                        extra.append(f"{k}={item[k]}")
                desc = f"{label}" + (f" [{', '.join(extra)}]" if extra else "") + (f": {crit}" if crit else "")
                if lvl is not None:
                    by_lvl[lvl].append(desc)
            summary[axis] = dict(by_lvl)
        return summary

    return {
        "adult_age": adult.get("target_age", ">= 15세"),
        "ped_age": ped.get("target_age", "< 15세"),
        "adult_axes": summarize_group(adult),
        "ped_axes": summarize_group(ped),
        "adult_psych_keys": list(adult.get("psychiatric_emergency_criteria", {}).keys()),
        "ped_critical_look": ped.get("critical_look", []),
        "pain_matrix": pain,
    }


def resolve_cross_ref_target(cond: str, target_pop: str, common: dict[str, Any]) -> tuple[str, str]:
    """cross_reference 문자열이 실제로 매핑되는 컨텍스트와 1차 기준 축을 해석한다."""
    is_ped = ("Pediatric" in target_pop) or bool(PED_HINT_RE.search(cond))
    ctx = f"소아 ({common['ped_age']})" if is_ped else f"성인 ({common['adult_age']})"

    if DNR_RE.search(cond):
        return ctx, "소생술 제외(DNR) 특수 조건"

    levels = [int(m) for m in LEVEL_MENTION_RE.findall(cond)]
    has_vital = bool(FIRST_VITAL_RE.search(cond))
    has_pain = bool(PAIN_MENTION_RE.search(cond))

    axes = common["ped_axes"] if is_ped else common["adult_axes"]
    matched_axes = []
    if has_vital:
        for ax_name, by_lvl in axes.items():
            if any(lvl in by_lvl for lvl in levels):
                matched_axes.append(ax_name)
    if has_pain:
        matched_axes.append("pain_decision_matrix")

    if not matched_axes:
        return ctx, "기타 1차/공통 조건"
    return ctx, " + ".join(matched_axes)


def analyze_all(raw_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """전체 KTAS JSON 파일을 스캔하여 참조, 병목, 무결성 데이터를 수집한다."""
    common_path = raw_dir / "ktas_part1_common.json"
    common = parse_common_framework(common_path)

    complaints_records: list[dict[str, Any]] = []
    integrity_issues: list[dict[str, Any]] = []
    cross_ref_stats: Counter = Counter()
    cross_ref_mapping_examples: dict[tuple[str, str], list[str]] = defaultdict(list)

    for json_path in sorted(raw_dir.glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        target_pop = str(data.get("target_population", "Unknown"))
        categories = data.get("categories", {})
        if not isinstance(categories, dict) or not categories:
            integrity_issues.append(
                {
                    "file": json_path.name,
                    "location": "root",
                    "issue_type": "missing_categories",
                    "detail": "categories 필드가 비어 있거나 존재하지 않음",
                }
            )
            continue

        for cat_id, cat_val in categories.items():
            if not isinstance(cat_val, dict):
                continue
            cat_name = str(cat_val.get("category_name", cat_id))
            complaints = cat_val.get("complaints", {})
            if not isinstance(complaints, dict) or not complaints:
                integrity_issues.append(
                    {
                        "file": json_path.name,
                        "location": f"{cat_name} ({cat_id})",
                        "issue_type": "empty_complaints",
                        "detail": "complaints 딕셔너리가 비어 있음",
                    }
                )
                continue

            for comp_code, comp_val in complaints.items():
                if not isinstance(comp_val, dict):
                    continue
                comp_name = str(comp_val.get("name", comp_code))
                first_mods = comp_val.get("first_order_modifiers_applicable", [])
                second_mods = comp_val.get("second_order_modifiers")

                if not isinstance(second_mods, list) or len(second_mods) == 0:
                    integrity_issues.append(
                        {
                            "file": json_path.name,
                            "location": f"{cat_name} / {comp_name} ({comp_code})",
                            "issue_type": "missing_or_empty_second_order_modifiers",
                            "detail": "second_order_modifiers가 없거나 빈 리스트임",
                        }
                    )
                    continue

                base_depth = 2 + (1 if first_mods else 0)
                parsed_mods: list[dict[str, Any]] = []
                prev_lvl: int | None = None
                non_monotonic_jumps: list[str] = []

                for idx, mod in enumerate(second_mods, start=1):
                    if not isinstance(mod, dict):
                        integrity_issues.append(
                            {
                                "file": json_path.name,
                                "location": f"{cat_name} / {comp_name} ({comp_code}) [#{idx}]",
                                "issue_type": "invalid_modifier_type",
                                "detail": f"modifier 항목이 dict가 아님: {type(mod).__name__}",
                            }
                        )
                        continue
                    cond = str(mod.get("condition", mod.get("criteria", ""))).strip()
                    lvl = _safe_int(mod.get("ktas_level"))
                    seq_depth = base_depth + idx

                    if lvl is None:
                        integrity_issues.append(
                            {
                                "file": json_path.name,
                                "location": f"{cat_name} / {comp_name} ({comp_code}) [#{idx}]",
                                "issue_type": "missing_or_invalid_ktas_level",
                                "detail": f"ktas_level 누락/오류 (raw={mod.get('ktas_level')}, cond='{cond}')",
                            }
                        )
                    if not cond:
                        integrity_issues.append(
                            {
                                "file": json_path.name,
                                "location": f"{cat_name} / {comp_name} ({comp_code}) [#{idx}]",
                                "issue_type": "empty_condition",
                                "detail": f"condition 문자열이 비어 있음 (ktas_level={lvl})",
                            }
                        )
                    if lvl is not None and prev_lvl is not None and lvl < prev_lvl:
                        non_monotonic_jumps.append(f"#{idx - 1}(KTAS {prev_lvl}) -> #{idx}(KTAS {lvl})")
                    if lvl is not None:
                        prev_lvl = lvl

                    if cond and REF_CONDITION_RE.search(cond):
                        ctx, target_axes = resolve_cross_ref_target(cond, target_pop, common)
                        cross_ref_stats[(ctx, target_axes)] += 1
                        if len(cross_ref_mapping_examples[(ctx, target_axes)]) < 3 and cond not in cross_ref_mapping_examples[(ctx, target_axes)]:
                            cross_ref_mapping_examples[(ctx, target_axes)].append(cond)

                    parsed_mods.append(
                        {
                            "idx": idx,
                            "seq_depth": seq_depth,
                            "ktas_level": lvl,
                            "condition": cond,
                        }
                    )

                if non_monotonic_jumps:
                    integrity_issues.append(
                        {
                            "file": json_path.name,
                            "location": f"{cat_name} / {comp_name} ({comp_code})",
                            "issue_type": "non_monotonic_level_order",
                            "detail": "낮은 우선순위(큰 숫자) 뒤에 더 위급한 등급이 배치됨: " + ", ".join(non_monotonic_jumps),
                        }
                    )

                max_seq = max((m["seq_depth"] for m in parsed_mods), default=0)
                lvl1_depths = [m["seq_depth"] for m in parsed_mods if m["ktas_level"] == 1]
                lvl2_depths = [m["seq_depth"] for m in parsed_mods if m["ktas_level"] == 2]

                complaints_records.append(
                    {
                        "file": json_path.name,
                        "target_population": target_pop,
                        "category": cat_name,
                        "code": comp_code,
                        "name": comp_name,
                        "has_first_mods": bool(first_mods),
                        "num_modifiers": len(parsed_mods),
                        "max_seq_depth": max_seq,
                        "max_lvl1_depth": max(lvl1_depths) if lvl1_depths else None,
                        "max_lvl2_depth": max(lvl2_depths) if lvl2_depths else None,
                        "modifiers": parsed_mods,
                        "non_monotonic": non_monotonic_jumps,
                    }
                )

    ref_info = {
        "common": common,
        "stats": cross_ref_stats,
        "examples": cross_ref_mapping_examples,
    }
    return ref_info, complaints_records, integrity_issues


def build_bottleneck_report(
    ref_info: dict[str, Any],
    complaints: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> str:
    """3대 정밀 분석 결과를 마크다운 보고서로 작성한다."""
    common = ref_info["common"]
    stats: Counter = ref_info["stats"]
    examples: dict[tuple[str, str], list[str]] = ref_info["examples"]

    lines: list[str] = [
        "# KTAS 결정 트리 정밀 병목 및 무결성 분석 리포트",
        "",
        "> **핵심 전제**: `\"1차 활력징후 1단계 만족\"` 등 동일/유사 문자열이라도 성인(`>= 15세`)과 소아(`< 15세`, 세부 월령 구간 포함)의 실제 임상 임계치(SpO2, 체온, 활력징후 등)가 다르므로, 문자열 일치 기반 단순 DAG 병합은 논리적 오류를 유발한다.",
        "",
        "---",
        "",
        "## 1. 공통 참조(`cross_reference`) 실체 및 컨텍스트 매핑 분석",
        "",
        "### 1.1 `ktas_part1_common.json` 실제 분기 구조",
        f"- **성인 컨텍스트 (`{common['adult_age']}`) 1차 고려사항 축 ({len(common['adult_axes'])}개 + 정신건강 3축)**:",
    ]
    for ax, by_lvl in common["adult_axes"].items():
        lvl_str = ", ".join(f"{k}단계({len(v)}개 기준)" for k, v in sorted(by_lvl.items()))
        lines.append(f"  - `{ax}`: {lvl_str}")
    lines.append(f"  - `psychiatric_emergency_criteria`: {', '.join(common['adult_psych_keys'])}")

    lines.append(f"- **소아 컨텍스트 (`{common['ped_age']}`) 1차 고려사항 축 ({len(common['ped_axes'])}개 + Critical Look 3축)**:")
    lines.append(f"  - `critical_look`: {', '.join(common['ped_critical_look'])}")
    for ax, by_lvl in common["ped_axes"].items():
        lvl_str = ", ".join(f"{k}단계({len(v)}개 기준)" for k, v in sorted(by_lvl.items()))
        lines.append(f"  - `{ax}`: {lvl_str}")
    lines.append("  - **주의(스키마 비대칭)**: `pediatric_first_order_modifiers`에는 성인용 `respiratory`, `bleeding_disorder`, `high_risk_mechanism` 블록이 별도로 존재하지 않으며, `temperature`는 `< 3개월`, `3-18개월`, `> 18개월` 3개 연령 구간으로 세분화됨.")
    lines.append("- **공통 통증 매트릭스 (`pain_decision_matrix`)**: `central_pain`(급성 2~4 / 만성 3~5), `peripheral_pain`(급성 3~5 / 만성 4~5)으로 2계층 매핑됨.")

    lines.extend([
        "",
        "### 1.2 주호소 내 `cross_reference` 조건의 실제 매핑 분포",
        "| 컨텍스트 (연령) | 매핑되는 1차/공통 기준 축 | 출현 건수 | 대표 조건식 예시 |",
        "| :--- | :--- | ---: | :--- |",
    ])
    for (ctx, ax), cnt in stats.most_common():
        ex = " / ".join(examples[(ctx, ax)][:2])
        lines.append(f"| {ctx} | `{ax}` | {cnt} | {ex} |")

    lines.extend([
        "",
        "**구조적 결론**: `cross_reference` 노드를 병합하거나 확장할 때는 반드시 `(컨텍스트[성인/소아/세부월령], 대상 축, 등급)` 튜플로 스코프(Scope)를 한정해 인라인 확장 또는 컨텍스트 분리형 서브그래프로 취급해야 논리적 동치성이 보존된다.",
        "",
        "---",
        "",
        "## 2. 탐색 깊이 병목(Outlier) 주호소 및 원인 분석",
        "",
        "### 2.1 순차 평가 깊이 최장 상위 10개 주호소",
        "| 순위 | 파일 | 계통 / 주호소 (코드) | `1차적용필드` | 조건 개수 | 최대 순차 깊이 | 등급 배치 패턴 |",
        "| :---: | :--- | :--- | :---: | ---: | ---: | :--- |",
    ])

    top10 = sorted(complaints, key=lambda r: (r["max_seq_depth"], r["num_modifiers"]), reverse=True)[:10]
    for rank, rec in enumerate(top10, start=1):
        lvls = "->".join(str(m["ktas_level"]) for m in rec["modifiers"])
        lines.append(
            f"| {rank} | `{rec['file']}` | {rec['category']} / **{rec['name']}** (`{rec['code']}`) | "
            f"{'O' if rec['has_first_mods'] else 'X'} | {rec['num_modifiers']} | **{rec['max_seq_depth']}** | `{lvls}` |"
        )

    lines.append("")
    lines.append("#### 최장 깊이 Top 3 내부 조건 나열 상세")
    for rec in top10[:3]:
        lines.append(f"- **`{rec['file']}` > {rec['category']} / {rec['name']} (`{rec['code']}`, 최대 깊이 {rec['max_seq_depth']})**")
        for m in rec["modifiers"]:
            lines.append(f"  - `depth={m['seq_depth']}` | **KTAS {m['ktas_level']}** | {m['condition']}")

    lvl1_outliers = sorted(
        [r for r in complaints if r["max_lvl1_depth"] is not None and r["max_lvl1_depth"] >= 5],
        key=lambda r: r["max_lvl1_depth"],
        reverse=True,
    )
    lvl2_outliers = sorted(
        [r for r in complaints if r["max_lvl2_depth"] is not None and r["max_lvl2_depth"] >= 10],
        key=lambda r: r["max_lvl2_depth"],
        reverse=True,
    )

    lines.extend([
        "",
        "### 2.2 위급 등급(1~2등급) 심층 도달 Outlier 분석",
        f"- **KTAS 1등급 도달 깊이 >= 5인 주호소 ({len(lvl1_outliers)}건, 최대 {max((r['max_lvl1_depth'] for r in lvl1_outliers), default=0)})**:",
    ])
    for rec in lvl1_outliers:
        l1_items = [f"#{m['idx']}(depth={m['seq_depth']}): {m['condition']}" for m in rec["modifiers"] if m["ktas_level"] == 1]
        lines.append(f"  - `{rec['file']}` > {rec['category']} / **{rec['name']}** (`{rec['code']}`): " + "; ".join(l1_items))

    lines.append(f"- **KTAS 2등급 도달 깊이 >= 10인 주호소 ({len(lvl2_outliers)}건, 최대 {max((r['max_lvl2_depth'] for r in lvl2_outliers), default=0)})**:")
    for rec in lvl2_outliers:
        l2_depths = [f"#{m['idx']}(d={m['seq_depth']})" for m in rec["modifiers"] if m["ktas_level"] == 2]
        lines.append(
            f"  - `{rec['file']}` > {rec['category']} / **{rec['name']}** (`{rec['code']}`): "
            f"전체 {rec['num_modifiers']}개 중 2등급 출현 위치 [{', '.join(l2_depths)}]"
        )

    lines.extend([
        "",
        "### 2.3 심층 도달 원인 진단",
        "1. **단일 등급 내 다수 세부 조건의 평면 선형 나열(Flat Linear List)**: `second_order_modifiers`가 계층 분기(예: 공통 전제 조건 선별 후 세부 분기) 없이 10~21개의 조건을 일렬로 나열하고 있어, 동일한 1등급 또는 2등급 내에서도 5~13번째 슬롯까지 순차 스캔해야 도달하는 병목이 발생함.",
        "2. **다중 등급 간 비단조/혼재 또는 2등급 다발**: 일부 주호소(예: 호흡곤란, 두통, 복통 등)는 2등급 특이 증상 조건이 8~12개씩 연속 나열되어, 후순위 2등급 조건의 순차 깊이가 10~16까지 증가함.",
        "3. **단순 순서 역전 여부**: 아래 무결성 점검에서 하위 등급 뒤에 상위 등급이 배치된 비단조 케이스가 있는지 별도로 검증함.",
        "",
        "---",
        "",
        "## 3. 데이터 무결성 전수 점검 결과",
        f"- **검출된 전체 무결성 이슈**: **총 {len(issues)}건**",
        "",
    ])

    issue_counts = Counter(i["issue_type"] for i in issues)
    if issues:
        lines.extend([
            "| 이슈 유형 | 건수 |",
            "| :--- | ---: |",
        ])
        for itype, cnt in issue_counts.most_common():
            lines.append(f"| `{itype}` | {cnt} |")
        lines.extend([
            "",
            "### 전체 이슈 상세 리스트",
            "| # | 파일 | 위치 (계통 / 주호소) | 이슈 유형 | 상세 내용 |",
            "| :---: | :--- | :--- | :--- | :--- |",
        ])
        for idx, iss in enumerate(issues, start=1):
            lines.append(f"| {idx} | `{iss['file']}` | {iss['location']} | `{iss['issue_type']}` | {iss['detail']} |")
    else:
        lines.append("- 검출된 구조적 무결성 결함이 없습니다.")

    return "\n".join(lines) + "\n"


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    default_raw = exp_root / "data" / "raw"
    default_out = exp_root / "reports" / "bottleneck_analysis.md"

    parser = argparse.ArgumentParser(description="KTAS 결정 트리 정밀 병목 및 무결성 분석")
    parser.add_argument("--raw-dir", default=str(default_raw), help="원본 KTAS JSON 폴더")
    parser.add_argument("--output", "-o", default=str(default_out), help="리포트 출력 경로")
    args = parser.parse_args()

    ref_info, complaints, issues = analyze_all(Path(args.raw_dir))
    report_md = build_bottleneck_report(ref_info, complaints, issues)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    print(report_md)
    print(f"[+] 병목 분석 리포트 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
