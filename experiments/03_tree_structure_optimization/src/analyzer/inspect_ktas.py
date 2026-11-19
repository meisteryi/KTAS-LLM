#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KTAS 결정 트리 JSON 구조 정밀 진단 및 베이스라인 측정 스크립트.

지원 스키마:
1. KTAS 모듈형 중첩 딕셔너리 (categories -> complaints -> modifiers)
2. 명시적 노드-엣지 그래프 ({"nodes": ..., "edges": ...})
3. 재귀적 결정 트리 (condition / children / yes-no / ktas_level)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RANGE_NUMERIC_RE = re.compile(
    r"(?:>=|<=|!=|==|>|<|\d+\s*[-~]\s*\d+|\d+(?:\.\d+)?\s*(?:%|세|시간|분|일|회|점|mmHg|bpm|℃|도))"
)
REF_CONDITION_RE = re.compile(
    r"(?:1차\s*활력징후|1차\s*고려|공통\s*평가|조건\s*만족|DNR|선행\s*조건)"
)
BOOLEAN_HINT_RE = re.compile(
    r"^(?:예|아니오|true|false|yes|no|유|무|있음|없음|해당|비해당)$",
    re.IGNORECASE,
)


@dataclass
class LeafPath:
    """트리의 루트부터 리프(KTAS 등급)까지의 단일 결정 경로."""

    source_file: str
    category: str
    complaint: str
    ktas_level: int | None
    tree_depth: int
    sequential_depth: int
    conditions: list[str] = field(default_factory=list)


@dataclass
class InspectionResult:
    """단일 실행에서 수집된 전체 구조 진단 지표."""

    input_paths: list[str]
    schema_types: Counter = field(default_factory=Counter)
    total_nodes: int = 0
    internal_nodes: int = 0
    leaf_paths: list[LeafPath] = field(default_factory=list)
    condition_strings: list[str] = field(default_factory=list)
    condition_types: Counter = field(default_factory=Counter)
    subtree_signatures: list[str] = field(default_factory=list)
    complaint_modifier_blocks: list[tuple[str, str, str]] = field(default_factory=list)
    unparsed_sections: list[str] = field(default_factory=list)


def classify_condition(text: str) -> str:
    """조건식 문자열을 Boolean/단순범주, 수치/구간, 1차참조, 비정형 자연어로 분류한다."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if BOOLEAN_HINT_RE.match(stripped):
        return "boolean"
    if REF_CONDITION_RE.search(stripped):
        return "cross_reference"
    if RANGE_NUMERIC_RE.search(stripped):
        return "numeric_range"
    if len(stripped) <= 12 and not any(sep in stripped for sep in (",", "또는", "및", "경우", "상태")):
        return "categorical_short"
    return "unstructured_text"


def hash_structure(obj: Any) -> str:
    """구조적 동치 비교를 위해 정규화된 JSON 문자열의 SHA-256 해시를 반환한다."""
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def detect_schema(data: Any) -> str:
    """JSON 루트 구조를 판별한다."""
    if isinstance(data, dict):
        if "nodes" in data and "edges" in data:
            return "node_edge_graph"
        if "categories" in data:
            return "ktas_nested_module"
        if any(k in data for k in ("adult_first_order_modifiers", "pediatric_first_order_modifiers", "level_definitions")):
            return "ktas_common_framework"
        if any(k in data for k in ("condition", "children", "yes", "no", "branches", "rules")):
            return "recursive_decision_tree"
    if isinstance(data, list):
        return "rule_list"
    return "generic_nested_dict"


def parse_ktas_nested_module(data: dict[str, Any], source_name: str, res: InspectionResult) -> None:
    """KTAS categories -> complaints -> modifiers 계층 구조를 파싱한다."""
    res.total_nodes += 1  # module root
    res.internal_nodes += 1

    categories = data.get("categories", {})
    for cat_key, cat_val in categories.items():
        if not isinstance(cat_val, dict):
            continue
        cat_name = str(cat_val.get("category_name", cat_key))
        res.total_nodes += 1
        res.internal_nodes += 1

        complaints = cat_val.get("complaints", {})
        for comp_key, comp_val in complaints.items():
            if not isinstance(comp_val, dict):
                continue
            comp_name = str(comp_val.get("name", comp_key))
            res.total_nodes += 1
            res.internal_nodes += 1

            first_mods = comp_val.get("first_order_modifiers_applicable", [])
            second_mods = comp_val.get("second_order_modifiers", [])

            if isinstance(second_mods, list) and second_mods:
                sig = hash_structure(second_mods)
                res.subtree_signatures.append(sig)
                res.complaint_modifier_blocks.append((cat_name, comp_name, sig))

                for idx, mod in enumerate(second_mods, start=1):
                    if not isinstance(mod, dict):
                        continue
                    cond = str(mod.get("condition", mod.get("criteria", ""))).strip()
                    level = _safe_int(mod.get("ktas_level"))
                    res.total_nodes += 1
                    if cond:
                        res.condition_strings.append(cond)
                        res.condition_types[classify_condition(cond)] += 1
                    # 구조적 깊이: module(1)->category(2)->complaint(3)->modifier(4)
                    # 순차 평가 깊이: category(1)->complaint(2)->1차조건(optional)+idx
                    seq_depth = 2 + (1 if first_mods else 0) + idx
                    res.leaf_paths.append(
                        LeafPath(
                            source_file=source_name,
                            category=cat_name,
                            complaint=comp_name,
                            ktas_level=level,
                            tree_depth=4,
                            sequential_depth=seq_depth,
                            conditions=[cat_name, comp_name, cond] if cond else [cat_name, comp_name],
                        )
                    )
            else:
                res.unparsed_sections.append(f"{source_name}:{cat_name}/{comp_name}(no second_order_modifiers)")


def parse_ktas_common_framework(data: dict[str, Any], source_name: str, res: InspectionResult) -> None:
    """KTAS Part 1 공통 1차 고려사항 프레임워크를 파싱한다."""
    res.total_nodes += 1
    res.internal_nodes += 1

    for mod_group_key, mod_group_val in data.items():
        if not isinstance(mod_group_val, dict) or mod_group_key == "level_definitions":
            continue
        for axis_key, axis_items in mod_group_val.items():
            if not isinstance(axis_items, list):
                continue
            res.total_nodes += 1
            res.internal_nodes += 1
            sig = hash_structure(axis_items)
            res.subtree_signatures.append(sig)
            for idx, item in enumerate(axis_items, start=1):
                if not isinstance(item, dict):
                    continue
                cond = str(item.get("criteria", item.get("classification", ""))).strip()
                level = _safe_int(item.get("ktas_level"))
                res.total_nodes += 1
                if cond:
                    res.condition_strings.append(cond)
                    res.condition_types[classify_condition(cond)] += 1
                res.leaf_paths.append(
                    LeafPath(
                        source_file=source_name,
                        category=mod_group_key,
                        complaint=str(axis_key),
                        ktas_level=level,
                        tree_depth=3,
                        sequential_depth=1 + idx,
                        conditions=[mod_group_key, str(axis_key), cond],
                    )
                )


def parse_node_edge_graph(data: dict[str, Any], source_name: str, res: InspectionResult) -> None:
    """명시적 nodes/edges 그래프 스키마를 파싱한다."""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_map: dict[str, dict[str, Any]] = {}
    children_map: dict[str, list[str]] = defaultdict(list)
    indegree: Counter = Counter()

    if isinstance(nodes, dict):
        for nid, nval in nodes.items():
            node_map[str(nid)] = nval if isinstance(nval, dict) else {"label": nval}
    elif isinstance(nodes, list):
        for n in nodes:
            if isinstance(n, dict) and "id" in n:
                node_map[str(n["id"])] = n

    for e in edges:
        if isinstance(e, dict):
            src = str(e.get("source", e.get("from", "")))
            dst = str(e.get("target", e.get("to", "")))
            if src and dst:
                children_map[src].append(dst)
                indegree[dst] += 1

    res.total_nodes += len(node_map)
    roots = [nid for nid in node_map if indegree[nid] == 0] or (list(node_map.keys())[:1] if node_map else [])

    visited_sig_cache: dict[str, str] = {}

    def dfs(nid: str, depth: int, conds: list[str], seen: set[str]) -> str:
        if nid in seen:
            return "cycle"
        node = node_map.get(nid, {})
        cond = str(node.get("condition", node.get("label", ""))).strip()
        level = _safe_int(node.get("ktas_level", node.get("level")))
        kids = children_map.get(nid, [])

        next_conds = conds + ([cond] if cond else [])
        if not kids:
            res.leaf_paths.append(
                LeafPath(
                    source_file=source_name,
                    category="graph",
                    complaint=nid,
                    ktas_level=level,
                    tree_depth=depth,
                    sequential_depth=depth,
                    conditions=next_conds,
                )
            )
            sig = hash_structure({"level": level, "cond": cond})
            res.subtree_signatures.append(sig)
            return sig

        res.internal_nodes += 1
        if cond:
            res.condition_strings.append(cond)
            res.condition_types[classify_condition(cond)] += 1

        child_sigs = [dfs(k, depth + 1, next_conds, seen | {nid}) for k in kids]
        sig = visited_sig_cache.get(nid) or hash_structure({"cond": cond, "children": child_sigs})
        visited_sig_cache[nid] = sig
        res.subtree_signatures.append(sig)
        return sig

    for r in roots:
        dfs(r, 1, [], set())


def parse_recursive_tree(node: Any, source_name: str, res: InspectionResult, depth: int = 1, conds: list[str] | None = None) -> str:
    """재귀적 결정 트리(children / yes-no / branches / list)를 파싱한다."""
    conds = conds or []
    if isinstance(node, list):
        sigs = [parse_recursive_tree(item, source_name, res, depth, conds) for item in node]
        return hash_structure(sigs)

    if not isinstance(node, dict):
        return hash_structure(str(node))

    res.total_nodes += 1
    cond = str(node.get("condition", node.get("question", node.get("rule", "")))).strip()
    level = _safe_int(node.get("ktas_level", node.get("level")))

    child_items: list[Any] = []
    for key in ("children", "branches", "rules"):
        val = node.get(key)
        if isinstance(val, list):
            child_items.extend(val)
        elif isinstance(val, dict):
            child_items.extend(val.values())
    for key in ("yes", "no", "true", "false"):
        if key in node and node[key] is not None:
            child_items.append(node[key])

    next_conds = conds + ([cond] if cond else [])
    if not child_items:
        res.leaf_paths.append(
            LeafPath(
                source_file=source_name,
                category="recursive_tree",
                complaint=next_conds[0] if next_conds else "root",
                ktas_level=level,
                tree_depth=depth,
                sequential_depth=depth,
                conditions=next_conds,
            )
        )
        sig = hash_structure({"level": level, "cond": cond})
        res.subtree_signatures.append(sig)
        return sig

    res.internal_nodes += 1
    if cond:
        res.condition_strings.append(cond)
        res.condition_types[classify_condition(cond)] += 1

    child_sigs = [parse_recursive_tree(c, source_name, res, depth + 1, next_conds) for c in child_items]
    sig = hash_structure({"cond": cond, "children": child_sigs})
    res.subtree_signatures.append(sig)
    return sig


def _safe_int(val: Any) -> int | None:
    """KTAS 등급 값을 정수로 안전하게 변환한다."""
    if val is None:
        return None
    try:
        iv = int(val)
        return iv if 1 <= iv <= 5 else None
    except (TypeError, ValueError):
        return None


def inspect_files(paths: list[Path]) -> InspectionResult:
    """지정된 JSON 파일 목록을 순회하며 구조 진단 지표를 수집한다."""
    res = InspectionResult(input_paths=[str(p) for p in paths])
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        schema = detect_schema(data)
        res.schema_types[schema] += 1

        if schema == "ktas_nested_module":
            parse_ktas_nested_module(data, p.name, res)
        elif schema == "ktas_common_framework":
            parse_ktas_common_framework(data, p.name, res)
        elif schema == "node_edge_graph":
            parse_node_edge_graph(data, p.name, res)
        else:
            parse_recursive_tree(data, p.name, res)
    return res


def build_markdown_report(res: InspectionResult) -> str:
    """진단 결과를 학술/엔지니어링 관점의 마크다운 리포트로 생성한다."""
    total_leaves = len(res.leaf_paths)
    total_conds = len(res.condition_strings)
    unique_conds = len(set(res.condition_strings))
    dup_cond_ratio = ((total_conds - unique_conds) / total_conds * 100.0) if total_conds else 0.0

    total_subtrees = len(res.subtree_signatures)
    unique_subtrees = len(set(res.subtree_signatures))
    dup_subtree_ratio = ((total_subtrees - unique_subtrees) / total_subtrees * 100.0) if total_subtrees else 0.0

    seq_depths = [lp.sequential_depth for lp in res.leaf_paths]
    tree_depths = [lp.tree_depth for lp in res.leaf_paths]

    avg_seq = sum(seq_depths) / len(seq_depths) if seq_depths else 0.0
    max_seq = max(seq_depths) if seq_depths else 0
    avg_tree = sum(tree_depths) / len(tree_depths) if tree_depths else 0.0
    max_tree = max(tree_depths) if tree_depths else 0

    depths_by_level: dict[int, list[int]] = defaultdict(list)
    for lp in res.leaf_paths:
        if lp.ktas_level is not None:
            depths_by_level[lp.ktas_level].append(lp.sequential_depth)

    level_avg = {lvl: (sum(ds) / len(ds)) for lvl, ds in depths_by_level.items() if ds}
    inversion_notes = []
    for high_lvl in (1, 2):
        for low_lvl in (4, 5):
            if high_lvl in level_avg and low_lvl in level_avg and level_avg[high_lvl] > level_avg[low_lvl]:
                inversion_notes.append(
                    f"KTAS {high_lvl}등급 평균 깊이({level_avg[high_lvl]:.2f}) > "
                    f"KTAS {low_lvl}등급 평균 깊이({level_avg[low_lvl]:.2f})"
                )

    cond_counter = Counter(res.condition_strings)
    top_dup_conds = [(c, cnt) for c, cnt in cond_counter.most_common(10) if cnt > 1]

    unstructured_cnt = res.condition_types.get("unstructured_text", 0)
    unstructured_ratio = (unstructured_cnt / total_conds * 100.0) if total_conds else 0.0

    lines = [
        "# KTAS 원본 결정 트리 구조 진단 리포트 (Baseline Report)",
        "",
        f"- **분석 대상 파일 ({len(res.input_paths)}개)**:",
        *[f"  - `{p}`" for p in res.input_paths],
        "",
        "## 1. 스키마 및 조건식 형태 점검",
        f"- **감지된 스키마 유형**: `{dict(res.schema_types)}`",
        f"- **전체 조건식 수**: {total_conds}개 (고유 조건식: {unique_conds}개)",
        "",
        "| 조건식 유형 | 건수 | 비율 (%) | 비고 |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for ctype, cnt in res.condition_types.most_common():
        pct = (cnt / total_conds * 100.0) if total_conds else 0.0
        lines.append(f"| `{ctype}` | {cnt} | {pct:.1f}% | - |")

    lines.extend([
        "",
        "## 2. 트리 규모 및 중복도 분석",
        f"- **전체 노드 수**: {res.total_nodes}개 (내부 분기 노드: {res.internal_nodes}개, 리프 경로: {total_leaves}개)",
        f"- **조건 문자열 중복률**: **{dup_cond_ratio:.2f}%** ({total_conds}건 중 {total_conds - unique_conds}건 중복)",
        f"- **서브트리(조건 블록) 구조 중복률**: **{dup_subtree_ratio:.2f}%** (고유 서브트리 {unique_subtrees} / 전체 {total_subtrees})",
        "",
        "### 상위 반복 조건식 (Top Repeated Conditions)",
        "| 순위 | 반복 횟수 | 조건식 내용 |",
        "| :---: | ---: | :--- |",
    ])
    for idx, (cond, cnt) in enumerate(top_dup_conds, start=1):
        lines.append(f"| {idx} | {cnt} | {cond} |")
    if not top_dup_conds:
        lines.append("| - | 0 | 중복 조건 없음 |")

    lines.extend([
        "",
        "## 3. 탐색 깊이 베이스라인 (Depth Baseline)",
        f"- **구조적 트리 깊이(Tree Depth)**: 평균 `{avg_tree:.2f}` / 최대 `{max_tree}`",
        f"- **순차 평가 깊이(Sequential Rule Depth)**: 평균 `{avg_seq:.2f}` / 최대 `{max_seq}`",
        "",
        "| KTAS 등급 | 리프 경로 수 | 평균 순차 도달 깊이 | 최소 깊이 | 최대 깊이 |",
        "| :---: | ---: | ---: | ---: | ---: |",
    ])
    for lvl in range(1, 6):
        ds = depths_by_level.get(lvl, [])
        if ds:
            lines.append(f"| {lvl}등급 | {len(ds)} | {sum(ds)/len(ds):.2f} | {min(ds)} | {max(ds)} |")
        else:
            lines.append(f"| {lvl}등급 | 0 | - | - | - |")

    lines.extend([
        "",
        f"- **등급 간 탐색 깊이 역전 여부**: "
        + ("; ".join(inversion_notes) if inversion_notes else "뚜렷한 역전 구간 미관측 (단, 단일 주호소 내부 분기 순서 점검 필요)"),
        "",
        "## 4. 잠재적 최적화 포인트 및 비판적 요약",
        f"1. **공통 서브트리 병합(DAG/BDD) 적합성**: 조건식 중복률 `{dup_cond_ratio:.1f}%`, 서브트리 중복률 `{dup_subtree_ratio:.1f}%`로, "
        + ("다수의 주호소에서 1차 활력징후/공통 조건이 반복 출현하므로 **공통 서브트리 병합(DAG/BDD 변환)**이 메모리 및 노드 수 압축에 매우 효과적입니다." if dup_cond_ratio >= 15.0 else "중복률이 낮아 서브트리 병합보다는 분기 순서/인덱싱 최적화가 우선입니다."),
        f"2. **비정형 자연어 정규화 난이도**: 전체 조건식 중 비정형 텍스트(`unstructured_text`) 비율이 **`{unstructured_ratio:.1f}%`**입니다. "
        + ("SMT/BDD 기반 100% 논리적 동치성 검증을 수행하려면 먼저 비정형 텍스트 조건을 원자 명제(Atomic Predicate ID)로 정규화하는 매핑 테이블 구축이 필수적입니다." if unstructured_ratio > 20.0 else "정형/수치 조건 비중이 높아 SMT 기반 동치성 검증기 구축이 용이합니다."),
        "3. **선후행 의존성 및 Don't-Care(`*`) 보존 주의**: KTAS는 `계통(대분류) -> 주호소 -> 1차/2차 수정인자`의 선후행 의존성을 가지므로, 대분류 확정 전 하위 조건을 끌어올리는 전역 BDD 재정렬은 `Don't-Care(*)` 경로 침해를 유발할 수 있습니다. 따라서 계층 경계 내 국소 재정렬 + 공통 서브트리 DAG 병합 전략을 권장합니다.",
    ])

    if res.unparsed_sections:
        lines.extend(["", "### 미파싱/특이 섹션 참고", *[f"- `{s}`" for s in res.unparsed_sections[:20]]])

    return "\n".join(lines) + "\n"


def resolve_input_paths(raw_inputs: list[str]) -> list[Path]:
    """파일 또는 디렉토리 입력에서 JSON 경로 리스트를 확장한다."""
    resolved: list[Path] = []
    for item in raw_inputs:
        p = Path(item)
        if p.is_dir():
            resolved.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            resolved.append(p)
        else:
            matches = sorted(Path().glob(item))
            resolved.extend(m for m in matches if m.is_file())
    # 순서 유지 중복 제거
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in resolved:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)
    return unique_paths


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    default_input = str(exp_root / "data" / "raw")
    default_output = str(exp_root / "reports" / "baseline_report.md")

    parser = argparse.ArgumentParser(description="KTAS 결정 트리 JSON 구조 진단 도구")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=[default_input],
        help=f"분석할 KTAS JSON 파일 또는 디렉토리 경로 (기본값: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help=f"진단 리포트 출력 경로 (기본값: {default_output})",
    )
    args = parser.parse_args()

    paths = resolve_input_paths(args.input)
    if not paths:
        raise FileNotFoundError(
            f"분석할 JSON 파일을 찾을 수 없습니다: {args.input}. "
            f"JSON 파일을 {default_input}에 넣거나 --input <경로>로 지정하세요."
        )

    res = inspect_files(paths)
    report_md = build_markdown_report(res)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    print(report_md)
    print(f"[+] 리포트 저장 완료: {out_path}")


if __name__ == "__main__":
    main()

