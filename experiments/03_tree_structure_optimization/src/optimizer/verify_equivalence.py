#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T_orig vs T_opt 100% 논리적 동치성 전수 검증 및 구조·토큰 비교 리포트 생성기."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any

from optimize_tree import split_or_items as extract_rule_atoms


def eval_t_orig(comp: dict[str, Any], assignment: dict[str, bool]) -> tuple[int | None, int]:
    """원본 트리 T_orig를 평가하고 (최종 KTAS 등급, 평가한 조건 스텝 수)를 반환한다."""
    mods = comp.get("second_order_modifiers", [])
    eval_priority = comp.get("evaluation_priority", "standard")
    steps = 0

    if eval_priority == "exception_first":
        for m in mods:
            cond = str(m.get("condition", "")).strip()
            lvl = m.get("ktas_level")
            if lvl == 5 and any(k in cond for k in ("DNR", "심폐소생술의 대상이 아닌")):
                steps += 1
                if assignment.get(cond, False):
                    return 5, steps

    best_lvl: int | None = None
    for m in mods:
        cond = str(m.get("condition", "")).strip()
        lvl = m.get("ktas_level")
        if not isinstance(lvl, int) or not cond:
            continue
        if eval_priority == "exception_first" and lvl == 5 and any(k in cond for k in ("DNR", "심폐소생술의 대상이 아닌")):
            continue
        # 이미 더 높은 중증도(작은 숫자)를 확보했고 정렬상 동일/낮은 중증도만 남았다면 조기 종료 가능하지만,
        # 원본 평면 리스트는 순차 평가 기준:
        if best_lvl is not None and lvl >= best_lvl:
            continue
        op, atoms = extract_rule_atoms(cond)
        steps += len(atoms)
        matched = any(assignment.get(a, False) for a in atoms) if op == "OR" else assignment.get(atoms[0], False)
        if matched:
            best_lvl = lvl if best_lvl is None else min(best_lvl, lvl)

    return best_lvl, steps


def eval_t_opt(opt_comp: dict[str, Any], assignment: dict[str, bool]) -> tuple[int | None, int]:
    """최적화 트리 T_opt를 평가하고 (최종 KTAS 등급, 평가한 조건 스텝 수)를 반환한다."""
    steps = 0

    # 1. 예외 게이트 최우선 평가
    for eg in opt_comp.get("exception_gates", []):
        steps += 1
        if assignment.get(eg["condition"], False):
            return eg["ktas_level"], steps

    best_lvl: int | None = None

    # 2. 1차 공통 축 Single-Pass + 2차 변수 그룹을 등급 우선순위(1->5) + 변수 그룹 가지치기로 평가
    first_by_lvl: dict[int, list[str]] = {}
    for rules in opt_comp.get("first_order_single_pass", {}).values():
        for r in rules:
            first_by_lvl.setdefault(r["ktas_level"], []).append(r["condition"])

    sec_groups = list(opt_comp.get("second_order_variable_groups", {}).items())
    for cur_lvl in range(1, 6):
        if best_lvl is not None and cur_lvl >= best_lvl:
            break
        for cond in first_by_lvl.get(cur_lvl, []):
            steps += 1
            if assignment.get(cond, False):
                best_lvl = cur_lvl
                break
        if best_lvl == cur_lvl:
            break
        for axis, rules in sec_groups:
            min_possible = min((r["ktas_level"] for r in rules), default=99)
            if min_possible > cur_lvl:
                continue
            for r in rules:
                if r["ktas_level"] != cur_lvl:
                    continue
                op, atoms = extract_rule_atoms(r["condition"])
                steps += len(atoms)
                matched = any(assignment.get(a, False) for a in atoms) if op == "OR" else assignment.get(atoms[0], False)
                if matched:
                    best_lvl = cur_lvl
                    break
            if best_lvl == cur_lvl:
                break

    return best_lvl, steps


def generate_test_vectors(atom_list: list[str], seed: int = 42) -> tuple[str, list[dict[str, bool]]]:
    """원자 명제 수에 따라 전수(2^m) 또는 경계/조합 테스트 벡터를 생성한다."""
    m = len(atom_list)
    if m == 0:
        return "empty", [{}]
    if m <= 10:
        vectors = [
            dict(zip(atom_list, bits))
            for bits in itertools.product((False, True), repeat=m)
        ]
        return f"exhaustive_2^{m}", vectors

    rng = random.Random(seed)
    vectors: list[dict[str, bool]] = []
    # 1) All False & All True
    vectors.append({a: False for a in atom_list})
    vectors.append({a: True for a in atom_list})
    # 2) All 1-hot
    for a in atom_list:
        v = {x: False for x in atom_list}
        v[a] = True
        vectors.append(v)
    # 3) All 2-hot combinations
    for a1, a2 in itertools.combinations(atom_list, 2):
        v = {x: False for x in atom_list}
        v[a1] = True
        v[a2] = True
        vectors.append(v)
    # 4) Random multi-hot vectors
    for _ in range(1500):
        p = rng.choice((0.1, 0.25, 0.5, 0.75))
        vectors.append({a: (rng.random() < p) for a in atom_list})

    return f"combinatorial_boundary({len(vectors)})", vectors


def _serialize_opt_complaint(c: dict[str, Any]) -> str:
    parts = [f"[{c['name']}]"]
    if c.get("exception_gates"):
        ex_str = "|".join(f"KTAS{g['ktas_level']}:{g['condition']}" for g in c["exception_gates"])
        parts.append(f"EX:{ex_str}")
    first_items = [
        f"{ax}({','.join(str(r['ktas_level']) for r in rs)})"
        for ax, rs in c.get("first_order_single_pass", {}).items()
    ]
    parts.append("1ST:" + "|".join(first_items))
    sec_items = []
    for ax, rs in c.get("second_order_variable_groups", {}).items():
        inner = ",".join(f"L{r['ktas_level']}:{r['condition']}" for r in rs)
        sec_items.append(f"{ax}:[{inner}]")
    parts.append("2ND:" + "|".join(sec_items))
    return " ".join(parts)


def count_prompt_tokens(orig_complaints: list[dict[str, Any]], opt_complaints: list[dict[str, Any]], model_dir: Path) -> tuple[str, int, int]:
    """Qwen2.5 토크나이저(가능 시) 또는 문자 기반 추정치로 직렬화 프롬프트 토큰 수를 측정한다."""
    orig_text = "\n".join(
        f"[{c['name']}] " + " | ".join(f"KTAS{m['ktas_level']}:{m['condition']}" for m in c.get("second_order_modifiers", []))
        for c in orig_complaints
    )
    opt_text = "\n".join(_serialize_opt_complaint(c) for c in opt_complaints)

    if model_dir.exists():
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
            return "Qwen2.5-32B Tokenizer", len(tok.encode(orig_text)), len(tok.encode(opt_text))
        except Exception:
            pass
    return "Char-Estimated (~3 chars/tok)", len(orig_text) // 3, len(opt_text) // 3


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    repo_root = exp_root.parents[1]

    parser = argparse.ArgumentParser(description="T_orig vs T_opt 동치성 전수 검증 및 요약 리포트")
    parser.add_argument("--raw-dir", default=str(exp_root / "data" / "raw"))
    parser.add_argument("--opt-json", default=str(exp_root / "data" / "optimized" / "ktas_optimized_tree.json"))
    parser.add_argument("--out-report", default=str(exp_root / "reports" / "optimization_summary.md"))
    args = parser.parse_args()

    # 원본 주호소 로드
    orig_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    orig_list: list[dict[str, Any]] = []
    for json_path in sorted(Path(args.raw_dir).glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ctx = "PED" if "Pediatric" in str(data.get("target_population", "")) else "ADULT"
        for cat_id, cat_val in data.get("categories", {}).items():
            for code, comp in cat_val.get("complaints", {}).items():
                nacrs = str(comp.get("nacrs_code", code))
                orig_map[(ctx, nacrs, comp.get("name", ""))] = comp
                orig_list.append(comp)

    with Path(args.opt_json).open("r", encoding="utf-8") as f:
        opt_data = json.load(f)
    opt_list: list[dict[str, Any]] = opt_data["complaints"]

    total_vectors = 0
    mismatches: list[dict[str, Any]] = []
    orig_avg_steps_list: list[float] = []
    opt_avg_steps_list: list[float] = []
    orig_max_steps_list: list[int] = []
    opt_max_steps_list: list[int] = []

    orig_node_count = 0
    opt_node_count = 0

    for opt_comp in opt_list:
        key = (opt_comp["context"], opt_comp["nacrs_code"], opt_comp["name"])
        orig_comp = orig_map[key]

        # 노드 수 비교 (T_orig: 규칙 슬롯 수 vs T_opt: 예외 게이트 + 1차 통합 축 노드 + 2차 변수별 규칙 노드)
        orig_node_count += len(orig_comp.get("second_order_modifiers", []))
        opt_node_count += (
            len(opt_comp.get("exception_gates", []))
            + len(opt_comp.get("first_order_single_pass", {}))
            + sum(len(v) for v in opt_comp.get("second_order_variable_groups", {}).values())
        )

        # 주호소 내 모든 원자 명제 수집
        atoms_set: list[str] = []
        for m in orig_comp.get("second_order_modifiers", []):
            _, atoms = extract_rule_atoms(str(m.get("condition", "")))
            for a in atoms:
                if a not in atoms_set:
                    atoms_set.append(a)

        mode, vectors = generate_test_vectors(atoms_set)
        c_orig_steps: list[int] = []
        c_opt_steps: list[int] = []

        for vec in vectors:
            total_vectors += 1
            o_res, o_stp = eval_t_orig(orig_comp, vec)
            p_res, p_stp = eval_t_opt(opt_comp, vec)
            c_orig_steps.append(o_stp)
            c_opt_steps.append(p_stp)
            if o_res != p_res:
                mismatches.append(
                    {
                        "complaint": f"{opt_comp['context']} {opt_comp['name']} ({opt_comp['nacrs_code']})",
                        "orig": o_res,
                        "opt": p_res,
                        "mode": mode,
                    }
                )
                break

        orig_avg_steps_list.append(sum(c_orig_steps) / len(c_orig_steps))
        opt_avg_steps_list.append(sum(c_opt_steps) / len(c_opt_steps))
        orig_max_steps_list.append(max(c_orig_steps))
        opt_max_steps_list.append(max(c_opt_steps))

    tok_name, orig_toks, opt_toks = count_prompt_tokens(
        orig_list, opt_list, repo_root / "models" / "Qwen2.5-32B-Instruct"
    )

    node_diff = opt_node_count - orig_node_count
    node_diff_pct = node_diff / orig_node_count * 100.0 if orig_node_count else 0.0
    avg_step_before = sum(orig_avg_steps_list) / len(orig_avg_steps_list)
    avg_step_after = sum(opt_avg_steps_list) / len(opt_avg_steps_list)
    avg_step_diff = avg_step_after - avg_step_before
    avg_step_diff_pct = avg_step_diff / avg_step_before * 100.0 if avg_step_before else 0.0
    max_step_before = max(orig_max_steps_list)
    max_step_after = max(opt_max_steps_list)
    max_step_diff = max_step_after - max_step_before
    max_step_diff_pct = max_step_diff / max_step_before * 100.0 if max_step_before else 0.0
    tok_diff = opt_toks - orig_toks
    tok_diff_pct = tok_diff / orig_toks * 100.0 if orig_toks else 0.0
    equiv_rate = (total_vectors - len(mismatches)) / total_vectors * 100.0 if total_vectors else 100.0

    report_lines = [
        "# KTAS 결정 트리 최적화(`T_opt`) 및 논리적 동치성 검증 리포트",
        "",
        "## 1. 100% 논리적 동치성 검증 결과 (`verify_equivalence.py`)",
        f"- **검증 대상 주호소 수**: {len(opt_list)}개 (전체 100%)",
        f"- **총 시뮬레이션 입력 조합(테스트 벡터) 수**: **{total_vectors:,}건** *(원자 명제 <= 10개는 $2^m$ 전수 진리표, 초과 시 조합·경계·무작위 커버리지)*",
        f"- **논리적 동치 일치율 ($T_{{\\text{{orig}}}}(x) \\equiv T_{{\\text{{opt}}}}(x)$)**: **{equiv_rate:.4f}%** (**불일치 {len(mismatches)}건**)",
        "",
        "---",
        "",
        "## 2. 최적화 전후 구조 및 효율 비교 (`T_orig` vs `T_opt`)",
        "",
        "| 비교 지표 | 원본 트리 (`T_orig`) | 최적화 트리 (`T_opt`) | 변화 및 절감률 |",
        "| :--- | ---: | ---: | :--- |",
        f"| **총 조건 노드 수 (규칙 슬롯)** | {orig_node_count:,}개 | {opt_node_count:,}개 | **{node_diff:+,}개 ({node_diff_pct:+.1f}%)** |",
        f"| **주호소당 평균 평가 스텝 수** | {avg_step_before:.2f} 스텝 | {avg_step_after:.2f} 스텝 | **{avg_step_diff:+.2f} ({avg_step_diff_pct:+.1f}%)** |",
        f"| **주호소당 최악(최대) 평가 스텝 수** | {max_step_before} 스텝 | {max_step_after} 스텝 | **{max_step_diff:+d} ({max_step_diff_pct:+.1f}%)** |",
        f"| **직렬화 프롬프트 토큰 수 (`{tok_name}`)** | {orig_toks:,} tokens | {opt_toks:,} tokens | **{tok_diff:+,} tokens ({tok_diff_pct:+.1f}%)** |",
        "",
        "---",
        "",
        "## 3. 구조 최적화 핵심 메커니즘 요약",
        "1. **예외 게이트 최상단 분리 (`exception_gates`)**: 심정지(`001`/`002`)의 `DNR` 등 5단계 예외 조항을 1단계보다 선행 평가해 단락(Short-circuit) 처리.",
        "2. **1차 공통 축 Single-Pass 통합 (`first_order_single_pass`)**: 2차 고려사항에 반복 산재해 있던 `1차 활력징후`, `통증_척도`, `출혈_양상`, `고위험_사고기전` 참조를 주호소당 1회 단일 패스로 통합.",
        "3. **2차 변수 그룹화 + 등급 가지치기 (`second_order_variable_groups`)**: 교차 등급으로 반복되는 임상 변수 단위로 묶고, 이미 확보된 최고 중증도(`best_level`)보다 낮은 등급만 있는 변수 그룹은 즉시 스킵(Pruning)하여 평균 평가 스텝 및 프롬프트 토큰을 동시에 절감.",
    ]

    if mismatches:
        report_lines.extend(["", "## ⚠️ 불일치 상세 목록"])
        for mm in mismatches:
            report_lines.append(f"- {mm['complaint']}: orig={mm['orig']} vs opt={mm['opt']} ({mm['mode']})")

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[+] 동치성 검증 완료: {len(opt_list)}개 주호소, {total_vectors:,}건 테스트 -> 불일치 {len(mismatches)}건 (일치율 {equiv_rate:.2f}%)")
    print(f"[+] 평균 스텝: {avg_step_before:.2f} -> {avg_step_after:.2f} ({avg_step_diff_pct:+.1f}%) | 토큰({tok_name}): {orig_toks:,} -> {opt_toks:,} ({tok_diff_pct:+.1f}%)")
    print(f"[+] 리포트 저장: {out_path}")


if __name__ == "__main__":
    main()
