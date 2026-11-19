#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS 결정 트리 내 동일 변수 반복 질문 및 평면 선형 나열 병목 탐지 스크립트.

목표:
- Z3/BDD 변환 없이, LLM 추론 토큰 절감과 탐색 깊이 단축에 직결되는
  1) 주호소별 동일 임상 변수/평가 축 반복 등장 횟수 (Variable Repetition Count)
  2) 동일 등급 내 평면 나열 길이 (Same-Level Flat Run) 및 2단 카테고리 묶음 시 깊이 절감폭
  3) 최적화 우선순위 Top 20 주호소 및 구체적 리팩토링 포인트
를 정량 산출하여 reports/inefficiency_hotspots.md로 저장한다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# 임상 변수 / 평가 축 키워드 사전 (순서가 구체적 임상 축 우선)
CLINICAL_AXES: list[tuple[str, tuple[str, ...]]] = [
    ("1차_활력징후", ("1차 활력징후", "활력징후")),
    ("혈압(SBP/DBP)", ("sbp", "dbp", "수축기", "이완기", "혈압", "저혈압", "고혈압")),
    ("심박수/맥박", ("심박수", "빈맥", "서맥", "맥박", "부정맥", "심정지")),
    ("호흡/SpO2/PEFR", ("호흡", "spo2", "산소포화도", "pefr", "fev1", "협착음", "천명음", "청색증", "기도")),
    ("체온/발열", ("체온", "발열", "열(", "열,", "저체온", "°c", "℃")),
    ("의식/신경학적", ("의식", "gcs", "avpu", "경련", "발작", "마비", "신경학적", "두통", "시력", "실신", "혼수", "지남력", "뇌졸중")),
    ("통증_척도", ("통증", "중심성", "말초성", "8-10", "4-7", "<4", "< 4")),
    ("출혈_양상", ("출혈", "혈변", "토혈", "혈뇨", "객혈", "코피", "출혈성 질환")),
    ("혈당(Glucose)", ("혈당", "mg/dl", "저혈당", "고혈당", "dka")),
    ("임신/분만/진통", ("임신", "자궁수축", "진통", "태동", "양수", "탯줄", "태아", "분만", "출산", "주수")),
    ("외상/손상", ("외상", "둔상", "관통상", "골절", "탈구", "사고기전", "열상", "화상", "봉합", "체표면적")),
    ("시간/경과기간", ("시간", "분", "일 이상", "일 이내", "주", "개월", "만성", "급성", "경과", "간격")),
    ("탈수/소화기", ("탈수", "구토", "설사", "소변", "수유", "황달")),
    ("부종/발적/감염", ("부종", "발적", "부기", "발진", "감염", "분비물", "농양")),
    ("정신/중독/학대", ("자살", "불안", "초조", "환각", "망상", "금단", "중독", "오용", "과다복용", "학대")),
]


def extract_axes(text: str) -> list[str]:
    """조건 문자열에서 해당하는 임상 변수/평가 축 목록을 추출한다."""
    low = text.lower()
    matched = [axis for axis, kws in CLINICAL_AXES if any(kw in low for kw in kws)]
    return matched if matched else ["기타_특이증상"]


def simulate_two_tier_depth(mods: list[dict[str, Any]]) -> tuple[int, int, int, dict[int, dict[str, Any]]]:
    """동일 등급 내 2단 임상 축 묶음 적용 시 최악 탐색 깊이 절감 효과를 계산한다.

    같은 KTAS 등급 내 조건들은 논리적으로 OR 관계이므로,
    임상 축(카테고리) 단위로 2단 그룹화해도 판정 동치성이 100% 유지된다.
    """
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for m in mods:
        lvl = m.get("ktas_level")
        if isinstance(lvl, int):
            by_level[lvl].append(m)

    flat_total = len(mods)
    grouped_total = 0
    max_flat_run = 0
    level_details: dict[int, dict[str, Any]] = {}

    for lvl in sorted(by_level.keys()):
        items = by_level[lvl]
        n_items = len(items)
        if n_items > max_flat_run:
            max_flat_run = n_items

        # 각 조건의 대표 임상 축(가장 먼저 매칭되는 축) 기준으로 2단 그룹화
        groups: dict[str, list[str]] = defaultdict(list)
        for it in items:
            axes = extract_axes(str(it.get("condition", "")))
            primary_axis = axes[0]
            groups[primary_axis].append(str(it.get("condition", "")))

        n_groups = len(groups)
        max_group_size = max(len(v) for v in groups.values())
        # 2단 묶음이 실제로 깊이를 줄이는 경우(그룹 수 >= 2이고 평면 길이 >= 4)에만 적용
        if n_items >= 4 and n_groups >= 2 and (n_groups + max_group_size) < n_items:
            lvl_grouped_cost = n_groups + max_group_size
            applied = True
        else:
            lvl_grouped_cost = n_items
            applied = False

        grouped_total += lvl_grouped_cost
        level_details[lvl] = {
            "flat_count": n_items,
            "grouped_cost": lvl_grouped_cost,
            "num_groups": n_groups,
            "max_group_size": max_group_size,
            "applied": applied,
            "groups": {k: len(v) for k, v in groups.items()},
        }

    return flat_total, grouped_total, max_flat_run, level_details


def analyze_hotspots(raw_dir: Path) -> tuple[list[dict[str, Any]], Counter]:
    """전체 주호소를 스캔해 변수 반복 횟수 및 2단 묶음 깊이 절감 잠재력을 계산한다."""
    results: list[dict[str, Any]] = []
    global_repeated_axes: Counter = Counter()

    for json_path in sorted(raw_dir.glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        is_ped = "Pediatric" in str(data.get("target_population", ""))
        ctx = "소아" if is_ped else "성인"

        for cat_id, cat_val in data.get("categories", {}).items():
            cat_name = cat_val.get("category_name", cat_id)
            for code, comp in cat_val.get("complaints", {}).items():
                mods = comp.get("second_order_modifiers", [])
                if not mods:
                    continue

                axis_total_counts: Counter = Counter()
                axis_levels: dict[str, set[int]] = defaultdict(set)

                for m in mods:
                    lvl = m.get("ktas_level")
                    cond = str(m.get("condition", ""))
                    for ax in extract_axes(cond):
                        if ax == "기타_특이증상":
                            continue
                        axis_total_counts[ax] += 1
                        if isinstance(lvl, int):
                            axis_levels[ax].add(lvl)

                # 중복 등장 횟수 (한 주호소 내에서 2회 이상 등장한 변수의 초과 횟수 합)
                repeat_excess = sum(cnt - 1 for cnt in axis_total_counts.values() if cnt >= 2)
                cross_level_axes = {
                    ax: sorted(lvls) for ax, lvls in axis_levels.items() if len(lvls) >= 2
                }
                repeated_axes_detail = {
                    ax: cnt for ax, cnt in axis_total_counts.most_common() if cnt >= 2
                }
                for ax, cnt in repeated_axes_detail.items():
                    global_repeated_axes[ax] += cnt - 1

                flat_d, grouped_d, max_run, lvl_details = simulate_two_tier_depth(mods)
                depth_saved = flat_d - grouped_d

                # 우선순위 점수: 변수 반복 초과 * 2 + 깊이 절감 * 3 + 교차등급 반복 축 * 2
                priority_score = repeat_excess * 2 + depth_saved * 3 + len(cross_level_axes) * 2

                results.append(
                    {
                        "file": json_path.name,
                        "context": ctx,
                        "category": cat_name,
                        "code": str(comp.get("nacrs_code", code)),
                        "name": comp.get("name", code),
                        "num_mods": len(mods),
                        "repeat_excess": repeat_excess,
                        "cross_level_axes": cross_level_axes,
                        "repeated_axes_detail": repeated_axes_detail,
                        "max_flat_run": max_run,
                        "flat_depth": flat_d,
                        "grouped_depth": grouped_d,
                        "depth_saved": depth_saved,
                        "priority_score": priority_score,
                        "level_details": lvl_details,
                        "mods": mods,
                    }
                )

    results.sort(key=lambda r: (r["priority_score"], r["depth_saved"], r["repeat_excess"]), reverse=True)
    return results, global_repeated_axes


def build_report(results: list[dict[str, Any]], global_axes: Counter) -> str:
    """inefficiency_hotspots.md 마크다운 리포트를 생성한다."""
    total_complaints = len(results)
    total_flat_steps = sum(r["flat_depth"] for r in results)
    total_grouped_steps = sum(r["grouped_depth"] for r in results)
    total_saved = total_flat_steps - total_grouped_steps
    improvable_cnt = sum(1 for r in results if r["depth_saved"] > 0)

    lines = [
        "# KTAS 결정 트리 변수 반복 및 평면 나열 비효율 탐지 리포트",
        "",
        "> **분석 초점**: 복잡한 Z3/BDD 변환 없이, LLM 추론 토큰을 절감하고 필수 경로 누락을 막기 위한 **'동일 변수 반복 질문(Variable Repetition)'** 및 **'동일 등급 내 평면 나열(Same-Level Flat Run)'** 집중 분석.",
        "",
        "## 1. 전체 요약 통계",
        f"- **분석된 주호소 수**: {total_complaints}개 (2차 고려사항 보유 기준)",
        f"- **전체 최악 탐색 깊이 합계 (Before ➔ After 2단 묶음)**: {total_flat_steps} ➔ **{total_grouped_steps}** (**-{total_saved}단계, {total_saved/total_flat_steps*100:.1f}% 단축**)",
        f"- **2단 임상 카테고리 묶음으로 즉시 깊이 단축이 가능한 주호소**: **{improvable_cnt}개**",
        "",
        "### 주호소 내부에서 가장 자주 반복 질문되는 임상 변수 Top 10",
        "| 순위 | 임상 변수 / 평가 축 | 주호소 내 초과 반복 총합 |",
        "| :---: | :--- | ---: |",
    ]
    for idx, (ax, cnt) in enumerate(global_axes.most_common(10), start=1):
        lines.append(f"| {idx} | `{ax}` | {cnt}회 |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. 최적화 우선순위 Top 20 주호소 랭킹",
        "> `우선순위 점수 = (변수 반복 횟수 × 2) + (2단 묶음 깊이 절감 × 3) + (다중 등급 교차 반복 축 수 × 2)`",
        "",
        "| 순위 | 구분 | 계통 / 주호소 (코드) | 전체 조건 | 변수 반복 | 최장 동일등급 나열 | 깊이 절감 (Before➔After) | 점수 | 리팩토링 포인트 (묶어야 할 변수/축) |",
        "| :---: | :---: | :--- | ---: | ---: | ---: | :---: | ---: | :--- |",
    ])

    for rank, r in enumerate(results[:20], start=1):
        top_axes = ", ".join(f"{ax}({cnt}회)" for ax, cnt in list(r["repeated_axes_detail"].items())[:3])
        cross_str = ", ".join(f"{ax}{lvls}" for ax, lvls in list(r["cross_level_axes"].items())[:2])
        refactor_note = top_axes + (f" / 교차등급: {cross_str}" if cross_str else "")
        lines.append(
            f"| {rank} | {r['context']} | {r['category']} / **{r['name']}** (`{r['code']}`) | "
            f"{r['num_mods']} | {r['repeat_excess']} | {r['max_flat_run']} | "
            f"**{r['flat_depth']} ➔ {r['grouped_depth']}** (-{r['depth_saved']}) | {r['priority_score']} | {refactor_note} |"
        )

    # Top 1 상세 Before/After 구조 예시
    if results:
        top1 = results[0]
        lines.extend([
            "",
            "---",
            "",
            f"## 3. 최우선 Hotspot (#1: `{top1['context']} {top1['name']} ({top1['code']})`) 2단 계층화 예시",
            f"- **현재 상태**: 총 {top1['num_mods']}개 조건이 평면 일렬로 나열됨 (최악 깊이 `{top1['flat_depth']}`, 최장 동일 등급 나열 `{top1['max_flat_run']}`개)",
            f"- **개선 후**: 각 등급 내 조건을 임상 카테고리(예: `1차_활력징후`, `출혈_양상`, `혈압`, `임신/분만`, `의식/신경학적`)로 2단 묶음 ➔ **최악 깊이 `{top1['grouped_depth']}` (-{top1['depth_saved']}단계 절감)**",
            "",
            "| KTAS 등급 | 기존 평면 나열 수 | 2단 묶음 후 최악 깊이 | 묶인 임상 카테고리 구성 (카테고리: 조건 수) |",
            "| :---: | ---: | ---: | :--- |",
        ])
        for lvl, det in sorted(top1["level_details"].items()):
            grp_str = ", ".join(f"`{g}`({c})" for g, c in det["groups"].items())
            lines.append(f"| {lvl}등급 | {det['flat_count']} | **{det['grouped_cost']}** | {grp_str} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="KTAS 변수 반복 및 평면 나열 비효율 탐지")
    parser.add_argument("--raw-dir", default=str(exp_root / "data" / "raw"))
    parser.add_argument("--output", "-o", default=str(exp_root / "reports" / "inefficiency_hotspots.md"))
    args = parser.parse_args()

    results, global_axes = analyze_hotspots(Path(args.raw_dir))
    report_md = build_report(results, global_axes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    print(f"[+] 분석 완료: {len(results)}개 주호소 스캔")
    print(f"[+] Top 1 Hotspot: {results[0]['context']} {results[0]['name']}({results[0]['code']}) "
          f"| 깊이 {results[0]['flat_depth']} -> {results[0]['grouped_depth']} (-{results[0]['depth_saved']})")
    print(f"[+] 리포트 저장: {out_path}")


if __name__ == "__main__":
    main()

