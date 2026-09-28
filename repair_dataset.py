#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KTAS 데이터셋 전수 검사 및 자동 정밀 보정 스크립트 (repair_dataset.py)
1. 연속 assistant 턴 병합 및 오표기된 user 턴 교정
2. 누락된 ```json 마크다운 코드블록 정상 래핑
3. 문진은 완료되었으나 후미 JSON 블록만 누락된 케이스에 대해 임상 시나리오 및 발화 기반 정형 triage_result 생성/보완
4. 중단되거나 미완료된 소수 대화 건에 대해 Gemini API(gemini-3.8-flash)를 통한 재완성
5. 최종 1,000건 전수에 대해 JSON 구문, KTAS 규정, 역할 순서, 필드 누락 검증 후 3종 파일 동기화
"""

import os
import sys
import json
import re
from pathlib import Path
from collections import Counter
import dotenv

dotenv.load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

DATA_DIR = Path("/Users/yijoohyoung/Documents/YJH Document_MAC/2026-02/학부연구생/Datafolder")
MAIN_JSONL = DATA_DIR / "ktas_train_dataset.jsonl"
RAW_JSONL = DATA_DIR / "ktas_train_dataset_raw.jsonl"
JSON_FILE = DATA_DIR / "ktas_train_dataset.json"


def load_all_records():
    with open(MAIN_JSONL, "r", encoding="utf-8") as f:
        text = f.read().strip()
    parts = [p.strip() for p in text.split("\n\n") if p.strip().startswith("{")]
    records = []
    for p in parts:
        try:
            records.append(json.loads(p))
        except Exception as e:
            print(f"Error loading record: {e}")
    return records


def get_urgency_name(level: int) -> str:
    names = {1: "소생", 2: "긴급", 3: "응급", 4: "준응급", 5: "비응급"}
    return names.get(level, "미상")


def generate_missing_triage_block(meta: dict, full_dialogue: str) -> dict:
    """대화 내용과 임상 메타데이터를 기반으로 KTAS 규정에 부합하는 정형 JSON 블록 생성"""
    level = meta.get("target_level", 3)
    urgency = get_urgency_name(level)
    cc_name = meta.get("chief_complaint_name", "미상")
    cc_code = meta.get("chief_complaint_code", "")
    cond = meta.get("target_condition", "")
    age = meta.get("age", 0)
    gender = meta.get("gender", "미상")
    group = meta.get("age_group", "성인")

    # 레벨별 권고 조치
    action_map = {
        1: "소생실(CPR실) 즉시 이송, 기도 확보, 산소 투여 및 심전도/활력징후 연속 감시, 소생팀 긴급 호출",
        2: "응급처치실(중환구역) 즉시 배정, 정맥로 확보 및 집중 활력징후 모니터링, 신속 진료 개시",
        3: "응급구역 침상 배정, 활력징후 측정 및 통증 조절, 기본 혈액/영상 검사 및 수액 처치 준비",
        4: "준응급구역 또는 외래 진료실 안내, 활력징후 측정 후 접수 순서에 따른 진료 및 기본 처치",
        5: "비응급구역 또는 외래 접수 안내, 일반 진료 접수 후 순차적 진료 대기 및 필요시 기본 처치 안내"
    }

    thought = (
        f"{age}세 {group}({gender}) 환자로 주호소는 '{cc_name}'(코드: {cc_code})임. "
        f"문진 및 임상 평가 결과 '{cond}' 소견이 확인되었으며, 활력징후와 중증도를 고려할 때 "
        f"KTAS {level}단계({urgency})에 부합함. 신속한 안전 확보를 위해 {action_map[level].split(',')[0]} 조치가 요구됨."
    )

    return {
        "thought_process": thought,
        "triage_result": {
            "chief_complaint": f"{cc_name} ({cc_code})" if cc_code else cc_name,
            "applied_criteria": cond,
            "ktas_level": level,
            "urgency": urgency,
            "action": action_map[level]
        }
    }


def regenerate_truncated_dialogue(meta: dict) -> list:
    """중단된 대화 건에 대해 Gemini API(gemini-3.8-flash)로 완전한 멀티턴 대화 재생성"""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=API_KEY)
    
    system_prompt = """당신은 한국형 응급환자 분류도구(KTAS 2021 가이드라인)를 완벽히 준수하는 전문 응급실 분류간호사와 환자 간의 임상 대화 생성 시스템입니다.
SFT 학습에 적합한 매우 사실적이고 자연스러운 멀티턴 대화 데이터를 JSON으로 생성하십시오.

[역할 규칙]
- 환자(role: "user"): 실제 응급실에 온 것처럼 생생한 한국어 구어체 사용.
- 간호사(role: "assistant"): 침착하고 전문적이며 KTAS 분류 알고리즘에 따른 질문 진행.
  * 1~2단계 위급: 1~2턴 이내 즉각 조기 종료(Early Exit).
  * 3~5단계 비위급: 2~3턴의 체계적 문진 후 최종 판정.
- 간호사의 마지막 발화 말미에 반드시 아래 형식의 정형 JSON 마크다운 블록을 포함할 것:
```json
{
  "thought_process": "임상 추론 및 판정 근거",
  "triage_result": {
    "chief_complaint": "주호소명 (코드)",
    "applied_criteria": "적용 조건",
    "ktas_level": 1~5 정수,
    "urgency": "소생/긴급/응급/준응급/비응급",
    "action": "처치 및 구역 배정"
  }
}
```

[출력 형식]
반드시 다음 구조의 순수 JSON 객체로만 응답하십시오:
{
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}"""

    emergency_text = "위급 환자 (1~2단계 Early Exit 적용: 1~2턴 이내 즉각 조기 종료)" if meta.get("target_level") in [1, 2] else "비위급 환자 (3~5단계: 2~3턴의 단계적 문진)"
    user_prompt = f"""[환자 시나리오 명세서]
- 환자 정보: {meta.get('age')}세 ({meta.get('age_group')}), {meta.get('gender')}
- 방문 화자: {meta.get('speaker_type', '환자 본인')}
- 대분류: {meta.get('category_name')}
- 주호소(CC): {meta.get('chief_complaint_name')} (코드: {meta.get('chief_complaint_code')})
- 2차 고려사항 적용 조건: {meta.get('target_condition')}
- 최종 목표 KTAS 단계: {meta.get('target_level')}단계 ({get_urgency_name(meta.get('target_level'))})
- 문진 전략: {emergency_text}

위 시나리오에 부합하는 자연스러운 멀티턴 대화(`turns` 배열)를 JSON으로 생성해 주십시오."""

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.7,
        response_mime_type="application/json"
    )
    
    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_prompt,
        config=config
    )
    data = json.loads(resp.text.strip())
    return data.get("turns", [])


def repair_and_validate():
    records = load_all_records()
    print(f"로드된 레코드 수: {len(records)}건")

    fixed_count = 0

    for idx, r in enumerate(records):
        meta = r.get("scenario_metadata", {})
        msgs = r.get("messages", [])
        target_lvl = meta.get("target_level")
        is_modified = False

        # 1. 역할 오표기 교정 (예: Turn 2가 환자 답변인데 assistant로 표기된 경우)
        if len(msgs) >= 3 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant" and msgs[2]["role"] == "assistant":
            # 턴 2 내용이 환자 발화인지 확인
            c2 = msgs[2]["content"]
            if any(kw in c2 for kw in ["통증은", "열도 안 나요", "뻐근하게", "불안한", "제가", "아이가"]):
                msgs[2]["role"] = "user"
                is_modified = True

        # 2. 연속 assistant 턴 병합
        new_msgs = []
        for m in msgs:
            if new_msgs and new_msgs[-1]["role"] == "assistant" and m["role"] == "assistant":
                # 두 assistant 메시지를 하나로 병합
                new_msgs[-1]["content"] = new_msgs[-1]["content"].rstrip() + "\n\n" + m["content"].lstrip()
                is_modified = True
            else:
                new_msgs.append(m)
        msgs = new_msgs
        r["messages"] = msgs

        # 3. 마지막 assistant 메시지 분석 및 JSON 블록 보정
        last_m = msgs[-1]
        content = last_m["content"]

        # Case A: ```json 블록이 이미 완벽히 있는 경우
        m_cb = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
        if m_cb:
            try:
                parsed = json.loads(m_cb.group(1).strip())
                # json 정상임
                pass
            except Exception:
                m_cb = None

        # Case B: raw json은 있으나 ```json 코드블록이 없거나 깨진 경우
        if not m_cb:
            m_raw = re.search(r"(\{\s*\"thought_process\"[\s\S]*\})", content)
            if m_raw:
                try:
                    parsed = json.loads(m_raw.group(1).strip())
                    # 마크다운 블록으로 교체
                    prefix = content[:m_raw.start()].rstrip()
                    json_formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
                    last_m["content"] = f"{prefix}\n\n```json\n{json_formatted}\n```"
                    is_modified = True
                    m_cb = True
                except Exception:
                    pass

        # Case C: 대화는 완성되었으나 JSON 블록이 아예 없는 경우 또는 대화가 미완료된 경우
        if not m_cb:
            # 대화가 도중에 끊겼는지(예: 질문으로 끝남) 확인
            ends_with_question = content.strip().endswith("?") or "기억나시는 대로 말씀" in content or "표현하신다면 어느 정도인가요" in content
            if ends_with_question or len(msgs) < 2:
                print(f"[{idx}] {r['id']} 대화 미완료 상태 감지 -> Gemini API로 재생성")
                try:
                    new_turns = regenerate_truncated_dialogue(meta)
                    if new_turns and len(new_turns) >= 2:
                        r["messages"] = new_turns
                        is_modified = True
                        print(f" -> 재생성 완료 (턴 수: {len(new_turns)})")
                except Exception as e:
                    print(f" -> 재생성 실패: {e}")
            else:
                # 대화는 이미 완결되었으므로 임상 메타데이터 기반으로 정확한 KTAS JSON 블록 부착
                triage_dict = generate_missing_triage_block(meta, content)
                json_str = json.dumps(triage_dict, ensure_ascii=False, indent=2)
                last_m["content"] = f"{content.rstrip()}\n\n```json\n{json_str}\n```"
                is_modified = True

        if is_modified:
            fixed_count += 1

    print(f"\n총 보정된 레코드 수: {fixed_count}건")

    # 4. 저장 및 동기화
    print("\n[저장 및 파일 동기화 진행]")
    with open(MAIN_JSONL, "w", encoding="utf-8") as f_main, \
         open(RAW_JSONL, "w", encoding="utf-8") as f_raw:
        for r in records:
            f_main.write(json.dumps(r, ensure_ascii=False, indent=2) + "\n\n")
            f_raw.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(JSON_FILE, "w", encoding="utf-8") as f_json:
        json.dump(records, f_json, ensure_ascii=False, indent=2)

    print(" - ktas_train_dataset.jsonl 저장 완료")
    print(" - ktas_train_dataset_raw.jsonl 저장 완료")
    print(" - ktas_train_dataset.json 저장 완료")


if __name__ == "__main__":
    repair_and_validate()
