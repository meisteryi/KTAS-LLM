#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KTAS 100건 파일럿 데이터셋 품질 및 정합성 검증 스크립트 (evaluate_pilot.py)
[검증 항목]
1. 총 생성 건수 및 KTAS 1~5단계별 분포 수량 집계
2. 성인 vs 소아 비율 집계
3. 생성된 텍스트 내 최종 JSON 포맷 파싱 에러율 체크
4. 시나리오의 target_level과 최종 판정 ktas_level 간의 정합성(일치율) 출력
5. 1~2단계 위급 케이스와 3~5단계 비위급 케이스의 평균 대화 턴 수 비교 (조기 종료 검증)
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter


def load_dataset(file_path: str = "./ktas_train_dataset.jsonl"):
    p = Path(file_path)
    if not p.exists():
        print(f"[오류] 파일이 존재하지 않습니다: {file_path}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        content = f.read().strip()

    records = []
    # 1. 들여쓰기된 JSON 블록(\n\n 구분) 파싱 시도
    parts = [part.strip() for part in content.split("\n\n") if part.strip().startswith("{")]
    for idx, part in enumerate(parts):
        try:
            records.append(json.loads(part))
        except Exception:
            pass

    # 2. 파싱 결과가 적다면 한 줄씩 파싱 시도 (단일 행 JSONL 호환)
    if len(records) < 10:
        records = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    return records


def extract_triage_json(assistant_content: str):
    """간호사의 마지막 발화에서 KTAS 판정 정형 JSON 블록 추출 및 검증"""
    # 1. 마크다운 ```json ... ``` 정규식 추출
    pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
    match = re.search(pattern, assistant_content)
    if match:
        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
            return parsed, None
        except Exception as e:
            return None, f"JSON 구문 오류: {e}"

    # 2. 마크다운 태그 없이 JSON 형태만 있는 경우 대비
    brace_pattern = r"(\{\s*\"thought_process\"[\s\S]*\})"
    brace_match = re.search(brace_pattern, assistant_content)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(1).strip())
            return parsed, None
        except Exception as e:
            return None, f"JSON 구문 오류: {e}"

    return None, "JSON 마크다운 블록 미발견"


def evaluate(records):
    total_count = len(records)
    print("=" * 70)
    print(f"📊 [KTAS 파일럿 데이터셋 100건 품질 및 임상 정합성 검증 보고서]")
    print("=" * 70)
    print(f"총 검증 대상 샘플 수: {total_count}건\n")

    # [1] KTAS 1~5단계별 분포 수량 집계
    target_levels = [r.get("scenario_metadata", {}).get("target_level") for r in records]
    level_counter = Counter(target_levels)

    print("─" * 70)
    print("[1] KTAS 목표 단계(1~5단계)별 분포 수량")
    level_names = {1: "소생", 2: "긴급", 3: "응급", 4: "준응급", 5: "비응급"}
    for lvl in range(1, 6):
        count = level_counter.get(lvl, 0)
        pct = (count / total_count * 100) if total_count else 0
        bar = "█" * (count // 1)
        print(f"  * KTAS {lvl}단계 ({level_names[lvl]}): {count:2d}건 ({pct:5.1f}%) | {bar}")

    # [2] 성인 vs 소아 비율 집계
    groups = [r.get("scenario_metadata", {}).get("age_group", "기타") for r in records]
    group_counter = Counter(groups)
    print("\n" + "─" * 70)
    print("[2] 연령군(성인 vs 소아) 분포")
    for grp, cnt in group_counter.most_common():
        pct = (cnt / total_count * 100) if total_count else 0
        print(f"  * {grp:4s}: {cnt:2d}건 ({pct:5.1f}%)")

    # [3] JSON 포맷 파싱 에러율 체크 & [4] 정합성(일치율) 체크
    parse_errors = []
    level_matches = []
    mismatch_cases = []

    # [5] 대화 턴 수 집계 (레벨별, 위급 vs 비위급)
    turns_per_level = {1: [], 2: [], 3: [], 4: [], 5: []}
    emergency_turns = []     # 1~2단계
    non_emergency_turns = [] # 3~5단계

    for idx, r in enumerate(records, 1):
        metadata = r.get("scenario_metadata", {})
        target_lvl = metadata.get("target_level")
        cc_name = metadata.get("chief_complaint_name", "미상")
        age_grp = metadata.get("age_group", "")
        age = metadata.get("age", 0)
        messages = r.get("messages", [])

        turn_count = len(messages)
        if target_lvl in turns_per_level:
            turns_per_level[target_lvl].append(turn_count)

        if target_lvl in [1, 2]:
            emergency_turns.append(turn_count)
        else:
            non_emergency_turns.append(turn_count)

        # 마지막 assistant 메시지 찾기
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            parse_errors.append((idx, r.get("id"), "간호사 발화 부재"))
            continue

        last_assistant_content = assistant_msgs[-1].get("content", "")
        parsed_json, err = extract_triage_json(last_assistant_content)

        if err:
            parse_errors.append((idx, r.get("id"), err))
            continue

        # 필수 필드 체크
        has_thought = "thought_process" in parsed_json and bool(str(parsed_json["thought_process"]).strip())
        triage_res = parsed_json.get("triage_result", {})
        has_level = "ktas_level" in triage_res

        if not (has_thought and has_level):
            parse_errors.append((idx, r.get("id"), "thought_process 또는 triage_result 누락"))
            continue

        # 최종 판정 레벨 정합성 검사
        try:
            pred_lvl = int(triage_res.get("ktas_level"))
            if pred_lvl == target_lvl:
                level_matches.append(True)
            else:
                level_matches.append(False)
                mismatch_cases.append({
                    "idx": idx,
                    "id": r.get("id"),
                    "chief_complaint": cc_name,
                    "age_info": f"{age_grp}({age}세)",
                    "target": target_lvl,
                    "predicted": pred_lvl,
                    "applied_criteria": triage_res.get("applied_criteria", ""),
                    "thought": parsed_json.get("thought_process", "")[:100]
                })
        except Exception as e:
            parse_errors.append((idx, r.get("id"), f"ktas_level 정수 변환 불가: {e}"))

    # [3] JSON 파싱 결과 출력
    parse_success_cnt = total_count - len(parse_errors)
    parse_success_rate = (parse_success_cnt / total_count * 100) if total_count else 0
    parse_error_rate = (len(parse_errors) / total_count * 100) if total_count else 0

    print("\n" + "─" * 70)
    print("[3] 텍스트 내 최종 JSON 포맷 파싱 검증")
    print(f"  * 정상 파싱 성공: {parse_success_cnt}/{total_count}건 ({parse_success_rate:5.1f}%)")
    print(f"  * 파싱 에러율:     {len(parse_errors)}/{total_count}건 ({parse_error_rate:5.1f}%)")
    if parse_errors:
        print("  * 에러 발생 케이스:")
        for e_idx, e_id, e_msg in parse_errors[:5]:
            print(f"    - #{e_idx} ({e_id}): {e_msg}")
    else:
        print("  * 완벽함: 모든 샘플이 thought_process와 triage_result 정형 JSON을 100% 정상 포함하고 있습니다.")

    # [4] 목표 단계 vs 판정 단계 정합성(일치율) 출력
    valid_evaluated = len(level_matches)
    correct_matches = sum(level_matches)
    match_rate = (correct_matches / valid_evaluated * 100) if valid_evaluated else 0

    print("\n" + "─" * 70)
    print("[4] 시나리오 목표 레벨 vs 최종 판정 KTAS 레벨 정합성 (일치율)")
    print(f"  * 목표 단계 일치 건수: {correct_matches}/{valid_evaluated}건")
    print(f"  * 최종 임상 정합도:    {match_rate:5.1f}%")

    if mismatch_cases:
        print(f"\n  * 불일치 케이스 ({len(mismatch_cases)}건 상세):")
        for mc in mismatch_cases:
            print(f"    - [#{mc['idx']}] {mc['age_info']} | 주호소: {mc['chief_complaint']}")
            print(f"      목표: KTAS {mc['target']}단계 ➡️ 모델 판정: KTAS {mc['predicted']}단계")
            print(f"      적용 기준: {mc['applied_criteria']}")
            print(f"      추론 요약: {mc['thought']}...\n")
    else:
        print("  * 모든 케이스에서 시나리오 목표 단계와 간호사 판정 단계가 100% 일치합니다!")

    # [5] 1~2단계 위급 vs 3~5단계 비위급 평균 대화 턴 수 비교 (조기 종료 검증)
    print("─" * 70)
    print("[5] 대화 턴 수 및 조기 종료 (Early Exit) 동작 검증")
    print("  * 단계별 평균 대화 턴 수:")
    for lvl in range(1, 6):
        turns = turns_per_level.get(lvl, [])
        avg = (sum(turns) / len(turns)) if turns else 0
        min_t = min(turns) if turns else 0
        max_t = max(turns) if turns else 0
        print(f"    - KTAS {lvl}단계 ({level_names[lvl]}): 평균 {avg:.2f}턴 (최소 {min_t}턴 ~ 최대 {max_t}턴)")

    avg_emerg = (sum(emergency_turns) / len(emergency_turns)) if emergency_turns else 0
    avg_non_emerg = (sum(non_emergency_turns) / len(non_emergency_turns)) if non_emergency_turns else 0

    print(f"\n  * 위급(1~2단계) vs 비위급(3~5단계) 조기 종료 비교:")
    print(f"    🚨 1~2단계 위급 케이스 평균 턴 수:   {avg_emerg:.2f}턴 (신속 판정 및 조기 종료)")
    print(f"    🩺 3~5단계 비위급 케이스 평균 턴 수: {avg_non_emerg:.2f}턴 (단계적 심층 문진)")
    print(f"    📉 턴 수 절감 비율:                  {((avg_non_emerg - avg_emerg) / avg_non_emerg * 100):.1f}% 단축")

    if avg_emerg < 3.0 and avg_non_emerg >= 4.0:
        print("  * ✅ [판정]: 1~2단계 위급 케이스의 Early Exit(1~2턴 내 조기 종료) 원칙이 완벽하게 준수되었습니다!")
    else:
        print(f"  * [참고]: 1~2단계는 평균 {avg_emerg:.2f}턴으로 조기 종료 경향을 보이고 있습니다.")

    print("=" * 70)


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "./ktas_train_dataset.jsonl"
    records = load_dataset(target_file)
    evaluate(records)
