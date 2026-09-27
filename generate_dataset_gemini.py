#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KTAS (한국형 응급환자 분류도구) 기반 멀티턴 대화형 SFT 학습 데이터셋 생성 파이프라인
- KTAS 2021 가이드라인 구조화 JSON (7개 파일) 통합 로드 및 계층적 인덱싱
- 환자 시나리오 샘플러: 성인/소아, 주호소, 1차/2차 고려사항, 위급/비응급 균형 샘플링
- Google GenAI 모델 기반 멀티턴 대화 생성 (1~2단계 Early Exit, 정형 JSON 출력)
- 표준 SFT JSONL 형식으로 실시간 flush 누적 저장
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# .env 파일이 존재하는 경우 환경 변수 자동 로드
load_dotenv()


def check_environment():
    """파이썬 환경 및 Gemini API 패키지/키 점검"""
    print("=" * 60)
    print("[1] 파이썬 및 라이브러리 환경 점검")
    print(f" - Python 버전: {sys.version.split()[0]}")

    has_google_genai = False
    try:
        from google import genai
        has_google_genai = True
        print(" - google-genai SDK: 사용 가능 (최신 공식 SDK)")
    except ImportError:
        print(" - google-genai SDK: 미설치")

    has_legacy_genai = False
    if not has_google_genai:
        try:
            import google.generativeai as legacy_genai
            has_legacy_genai = True
            print(" - google-generativeai SDK: 사용 가능 (레거시)")
        except ImportError:
            pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(" - GEMINI_API_KEY: [미설정] 환경변수 또는 .env 파일에 등록되지 않았습니다.")
        return False, None, has_google_genai
    else:
        masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f" - GEMINI_API_KEY: [설정됨] ({masked_key})")
        return True, api_key, has_google_genai


class KTASDatabase:
    """KTAS 가이드라인 데이터 통합 관리 및 계층적 색인 클래스"""
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.common = {}
        self.categories = {"adult": {}, "pediatric": {}}
        self.level_pools = {"adult": {1: [], 2: [], 3: [], 4: [], 5: []},
                            "pediatric": {1: [], 2: [], 3: [], 4: [], 5: []}}
        self.all_pools = {1: [], 2: [], 3: [], 4: [], 5: []}
        self.load_all()

    def load_all(self):
        """7개 JSON 파일 통합 로드 및 레벨별 색인 구성"""
        # 1. Common Framework
        common_path = self.data_dir / "ktas_part1_common.json"
        if common_path.exists():
            with open(common_path, "r", encoding="utf-8") as f:
                self.common = json.load(f)

        # 2. Adult Modules
        adult_files = [
            "ktas_part2_adult_core.json",
            "ktas_part3_adult_somatic.json",
            "ktas_part4_adult_sensory_general.json"
        ]
        for fname in adult_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.categories["adult"].update(data.get("categories", {}))

        # 3. Pediatric Modules
        ped_files = [
            "ktas_p5a_ped_core.json",
            "ktas_p5b_ped_somatic.json",
            "ktas_p5c_ped_sensory_general.json"
        ]
        for fname in ped_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.categories["pediatric"].update(data.get("categories", {}))

        # 4. 레벨별 검색 풀 인덱싱
        for group in ["adult", "pediatric"]:
            for cat_id, cat_info in self.categories[group].items():
                cat_name = cat_info.get("category_name", cat_id)
                complaints = cat_info.get("complaints", {})
                for code, comp in complaints.items():
                    comp_name = comp.get("name", code)
                    nacrs_code = comp.get("nacrs_code", code)
                    modifiers = comp.get("second_order_modifiers", [])
                    for mod in modifiers:
                        level = mod.get("ktas_level")
                        if level in [1, 2, 3, 4, 5]:
                            item = {
                                "group": group,
                                "category_id": cat_id,
                                "category_name": cat_name,
                                "complaint_code": nacrs_code,
                                "complaint_name": comp_name,
                                "level": level,
                                "condition": mod.get("condition", ""),
                                "all_modifiers": modifiers
                            }
                            self.level_pools[group][level].append(item)
                            self.all_pools[level].append(item)

        print(f" - KTAS 규칙 로드 완료 (성인 대분류: {len(self.categories['adult'])}개, 소아 대분류: {len(self.categories['pediatric'])}개)")
        print(f" - 레벨별 2차 고려사항 색인: 1단계({len(self.all_pools[1])}), 2단계({len(self.all_pools[2])}), 3단계({len(self.all_pools[3])}), 4단계({len(self.all_pools[4])}), 5단계({len(self.all_pools[5])})")

    def get_first_order_hints(self, group: str, level: int):
        """레벨 및 연령대에 따른 1차 고려사항(생체징후, 의식, 호흡, 통증 등) 임상 힌트 생성"""
        hints = []
        if level == 1:
            hints = [
                "의식: 무의식 (GCS 3-8, 통증 자극 무반응 또는 비정상 굴곡/신전 반응)",
                "호흡: 심한 호흡부전 (청색증, 얕고 불규칙한 빈호흡, SpO2 < 85%)",
                "혈역학: 중증 혈역학적 불안정 (맥박 촉지 불가 또는 미약한 빈맥, 심정지/패혈성 쇼크 의심)",
                "통증: 의식 소실로 평가 불가 또는 생애 최악의 급격한 파열성 극심 통증"
            ]
        elif level == 2:
            hints = [
                "의식: 의식수준의 변화 (GCS 9-13, 시간/장소/사람 지남력 상실, 언어 혼미)",
                "호흡: 중증 호흡곤란 (호흡보조근 사용, 숨이 차서 한두 마디도 잇기 어려움, SpO2 85-89%)",
                "혈역학: 혈역학적 불안정 (현저한 저혈압 SBP < 90, 기립성 어지럼, 심한 빈맥 HR > 130)",
                "통증: 급성 중심성 중증 통증 (NRS 8~10점, 식은땀, 극심한 고통 호소)",
                "체온: 고열(>=38.5도)과 전신 쇠약 및 면역저하 상태"
            ]
        elif level == 3:
            hints = [
                "의식: 명료 (GCS 14-15, 질문에 적절히 응답)",
                "호흡: 중등도 호흡곤란 (짧은 문장 대화 가능, SpO2 90-94%)",
                "혈역학: 경계성 활력징후 (경미한 빈맥 또는 혈압 상승 유지)",
                "통증: 급성 중심성 중등도 통증 (NRS 4~7점) 또는 조절되지 않는 만성 중증 통증 (NRS 8~10점)",
                "체온: 38.0도 전후 발열, 전신 상태 비교적 보존됨"
            ]
        elif level == 4:
            hints = [
                "의식: 완전 명료 (GCS 15)",
                "호흡: 안정적 (SpO2 >= 95%, 분당 호흡수 정상 16~20회)",
                "혈역학: 정상 혈압 및 정상 맥박",
                "통증: 급성 말초성 또는 경증 통증 (NRS 1~3점), 또는 만성 중등도 통증 (NRS 4~7점)",
                "상태: 1~2시간 내 치료 권장, 보행 및 대화 원활"
            ]
        else: # level 5
            hints = [
                "의식: 완전 명료 (GCS 15)",
                "활력징후: 혈압, 맥박, 호흡, 체온 등 모든 생체징후 정상",
                "통증: 통증 없음 또는 경미한 만성 통증 (NRS 0~2점)",
                "상태: 만성적이거나 경미한 증상, 투약 처방 및 단순 처치 요구"
            ]
        return random.choice(hints)


class ScenarioSampler:
    """균형잡힌 임상 환자 시나리오 샘플러"""
    def __init__(self, ktas_db: KTASDatabase):
        self.db = ktas_db
        self.urgency_names = {
            1: "소생",
            2: "긴급",
            3: "응급",
            4: "준응급",
            5: "비응급"
        }

    def sample(self, target_level: int = None, force_group: str = None) -> dict:
        """
        위급(1-2) vs 비응급(3-5) 균형을 고려한 시나리오 추출
        """
        # 1. 성인/소아 결정 (기본: 성인 70%, 소아 30%)
        if force_group in ["adult", "pediatric"]:
            group = force_group
        else:
            group = random.choices(["adult", "pediatric"], weights=[0.7, 0.3])[0]

        is_adult = (group == "adult")
        age = random.randint(15, 82) if is_adult else random.randint(1, 14)
        gender = random.choice(["남성", "여성"])

        # 2. KTAS 타겟 레벨 결정
        # 1~2단계(위급 45%)와 3~5단계(비응급 55%) 골고루 분포
        if target_level is None or target_level not in [1, 2, 3, 4, 5]:
            target_level = random.choices([1, 2, 3, 4, 5], weights=[0.20, 0.25, 0.25, 0.15, 0.15])[0]

        candidate_pool = self.db.level_pools[group][target_level]
        if not candidate_pool:
            candidate_pool = self.db.all_pools[target_level]

        selected_item = random.choice(candidate_pool)
        first_order_hint = self.db.get_first_order_hints(group, target_level)
        is_emergency = target_level in [1, 2]

        # 화자 설정
        if not is_adult:
            speaker_type = "소아 환자 보호자(부모)" if age <= 10 else random.choice(["소아 환자 본인", "소아 보호자"])
        else:
            if target_level == 1:
                speaker_type = "환자 곁의 보호자 또는 119 구급대원(환자 의식 저하)"
            else:
                speaker_type = "환자 본인" if random.random() < 0.85 else "동행 보호자"

        return {
            "age": age,
            "age_group": "성인" if is_adult else "소아",
            "gender": gender,
            "speaker_type": speaker_type,
            "category_name": selected_item["category_name"],
            "chief_complaint_code": selected_item["complaint_code"],
            "chief_complaint_name": selected_item["complaint_name"],
            "target_condition": selected_item["condition"],
            "target_level": target_level,
            "urgency": self.urgency_names[target_level],
            "is_emergency": is_emergency,
            "first_order_hint": first_order_hint
        }


SYSTEM_PROMPT = """당신은 한국형 응급환자 분류도구(KTAS 2021 가이드라인)를 완벽히 준수하는 전문 응급실 분류간호사와 환자 간의 임상 대화 생성 시스템입니다.
제공되는 [환자 시나리오 명세서]를 기반으로, SFT(Supervised Fine-Tuning) 학습에 적합한 매우 사실적이고 자연스러운 멀티턴 대화 데이터를 생성하십시오.

[작성 및 역할 규칙]
1. 환자(또는 보호자, role: "user"):
   - 절대 책에 나오는 정형화된 의학 용어나 진단명을 말하지 마십시오.
   - 실제 응급실에 온 것처럼 생생한 한국어 구어체를 사용하십시오. (예: "가슴이 쥐어짜듯 뻐근하고 숨이 턱턱 막혀요", "배가 뒤틀리고 식은땀이 줄줄 나요", "어제부터 열이 안 떨어지고 애가 축 늘어져요")
   - 심한 통증이나 위급 상황에서는 말을 더듬거나 호흡 곤란을 호소하는 생생한 묘사를 포함하십시오.

2. 분류간호사(role: "assistant"):
   - 매우 침착하고 전문적이며 신속한 어조로 응대합니다.
   - KTAS 분류 알고리즘에 따라 필요한 핵심 질문(발병 시점, 통증 양상 및 부위, 통증 점수 NRS, 호흡 곤란 여부, 기저 질환 등)을 정확하게 질문합니다.
   - [★조기 종료(Early Exit) 규칙]:
     * KTAS 1~2단계 위급 케이스의 경우: 환자의 첫마디나 첫 번째 답변에서 극심한 생체징후 이상(호흡부전, 무의식, 흉통, NRS 8-10점 급성 통증 등)이 확인되면 불필요하게 3~4턴 이상 길게 묻지 말고, **단 1~2턴 만에 즉시 중증도를 판정하고 응급 처치실로 안내하며 대화를 조기 종료**하십시오.
     * KTAS 3~5단계 비응급/준응급 케이스의 경우: 2~4턴의 체계적인 문진(동반 증상, 통증 강도, 과거력 등)을 단계별로 수행한 후 판정하십시오.

3. 간호사의 마지막 발화(최종 턴) 필수 포맷:
   - 간호사의 마지막 답변은 환자(또는 보호자)에게 배정된 중증도와 다음 진행 절차를 안내한 뒤, **발화 말미에 반드시 아래 형식의 정형 JSON 마크다운 블록**을 포함해야 합니다.

```json
{
  "thought_process": "환자의 연령, 주호소 및 관찰된 1차/2차 고려사항에 대한 의학적 추론과 KTAS 판정 근거 상세 기술",
  "triage_result": {
    "chief_complaint": "주호소명 (코드)",
    "applied_criteria": "적용된 1차/2차 고려사항 조건",
    "ktas_level": 1~5 (정수),
    "urgency": "소생/긴급/응급/준응급/비응급",
    "action": "의료진 처치 권고사항 및 진료구역 배정 안내"
  }
}
```

[출력 형식]
반드시 다음 구조의 순수 JSON 객체로만 응답하십시오 (추가 서술이나 바깥 텍스트 일체 금지):
{
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
"""


def build_user_prompt(scenario: dict) -> str:
    """시나리오에 따른 동적 생성 프롬프트 빌드"""
    emergency_text = "위급 환자 (1~2단계 Early Exit 적용: 1~2턴 이내 즉각 조기 종료 및 긴급 진료구역 안내)" if scenario['is_emergency'] else "비위급 환자 (3~5단계: 2~4턴의 단계적이고 체계적인 문진 진행)"
    
    return f"""[환자 시나리오 명세서]
- 환자 정보: {scenario['age']}세 ({scenario['age_group']}), {scenario['gender']}
- 방문 화자: {scenario['speaker_type']}
- 대분류: {scenario['category_name']}
- 주호소(CC): {scenario['chief_complaint_name']} (코드: {scenario['chief_complaint_code']})
- 2차 고려사항 적용 조건: {scenario['target_condition']}
- 1차 고려사항/생체징후 임상 힌트: {scenario['first_order_hint']}
- 최종 목표 KTAS 단계: {scenario['target_level']}단계 ({scenario['urgency']})
- 문진 전략: {emergency_text}

위 시나리오 명세서에 부합하는 자연스러운 한국어 멀티턴 대화(`turns` 배열)를 JSON 형식으로 생성해 주십시오."""


class DatasetGenerator:
    """Gemini API를 이용한 대화 데이터셋 생성기 (다중 모델 폴백 풀 지원)"""
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        self.api_key = api_key
        # 쿼터 분산을 위한 모델 풀 구성
        self.model_pool = ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemini-flash-latest"]
        if model_name in self.model_pool:
            self.model_pool.remove(model_name)
            self.model_pool.insert(0, model_name)
        else:
            self.model_pool.insert(0, model_name)
            
        self.current_model_idx = 0
        self.use_new_sdk = False

        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.api_key)
            self.genai_types = types
            self.use_new_sdk = True
            print(f" - [초기화] 최신 google-genai SDK 클라이언트를 사용합니다. (기본 모델: {self.current_model})")
        except ImportError:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self.legacy_genai = legacy_genai
            print(f" - [초기화] 레거시 google-generativeai SDK 클라이언트를 사용합니다. (기본 모델: {self.current_model})")

    @property
    def current_model(self):
        return self.model_pool[self.current_model_idx % len(self.model_pool)]

    def switch_to_next_model(self):
        self.current_model_idx += 1
        new_model = self.current_model
        print(f"    -> [모델 전환] 다음 가용 모델로 자동 전환합니다: {new_model}")
        return new_model

    def generate_single(self, scenario: dict, max_retries: int = 5) -> list:
        """단일 시나리오에 대해 대화 턴 목록 생성 및 검증 (재시도 및 모델 폴백 포함)"""
        prompt = build_user_prompt(scenario)

        for attempt in range(1, max_retries + 1):
            active_model = self.current_model
            try:
                if self.use_new_sdk:
                    config = self.genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.7,
                        response_mime_type="application/json"
                    )
                    response = self.client.models.generate_content(
                        model=active_model,
                        contents=prompt,
                        config=config
                    )
                    text = response.text.strip()
                else:
                    legacy_m = self.legacy_genai.GenerativeModel(
                        model_name=active_model,
                        system_instruction=SYSTEM_PROMPT,
                        generation_config={
                            "temperature": 0.7,
                            "response_mime_type": "application/json"
                        }
                    )
                    response = legacy_m.generate_content(prompt)
                    text = response.text.strip()

                # JSON 파싱
                try:
                    data = json.loads(text)
                    turns = data.get("turns", [])
                except Exception:
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                        data = json.loads(text)
                        turns = data.get("turns", [])
                    else:
                        raise ValueError(f"JSON 파싱 실패: {text[:200]}")

                if not turns or len(turns) < 2:
                    raise ValueError(f"유효하지 않은 대화 턴 수: {len(turns)}")

                return turns

            except Exception as e:
                err_msg = str(e)
                # 429/503/네트워크 일시 단절 처리
                is_transient = any(k in err_msg for k in ["503", "429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "nodename nor servname", "Connection", "timeout", "Temporary"])
                if attempt < max_retries and is_transient:
                    import re, time
                    # 429 Quota인 경우 다른 모델로 즉시 교체 시도
                    if "429" in err_msg or "ResourceExhausted" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        self.switch_to_next_model()
                        time.sleep(1.0)
                        continue

                    delay_match = re.search(r"retry\s+in\s+([\d\.]+)\s*s", err_msg, re.IGNORECASE) or re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", err_msg)
                    if delay_match:
                        wait_time = int(float(delay_match.group(1))) + 2
                    elif "nodename nor servname" in err_msg or "Connection" in err_msg:
                        wait_time = 10  # 네트워크 일시 단절 대기
                        print(f"    [인터넷 연결 일시 단절 감지, 네트워크 재연결 대기 ({wait_time}초)...]")
                    else:
                        wait_time = attempt * 10
                    print(f"    [일시적 부하/네트워크 이슈, {wait_time}초 대기 후 자동 재시도 ({attempt}/{max_retries})]")
                    time.sleep(wait_time)
                else:
                    raise e


def main():
    parser = argparse.ArgumentParser(description="KTAS 멀티턴 대화형 SFT 학습 데이터셋 생성기")
    parser.add_argument("--num_samples", type=int, default=100, help="생성할 샘플 개수 (기본값: 100)")
    parser.add_argument("--output", type=str, default="./ktas_train_dataset.jsonl", help="저장할 JSONL 파일 경로")
    parser.add_argument("--target_level", type=int, default=None, choices=[1, 2, 3, 4, 5], help="특정 KTAS 레벨만 강제 생성 (1~5)")
    parser.add_argument("--group", type=str, default=None, choices=["adult", "pediatric"], help="성인/소아 강제 지정")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="사용할 Gemini 모델명 (기본: gemini-3-flash-preview)")
    parser.add_argument("--sleep", type=float, default=1.0, help="호출 간 딜레이 초 (기본: 1.0초)")
    parser.add_argument("--no_pretty", action="store_true", help="한 줄 포맷으로 저장 (기본값은 가독성을 위한 들여쓰기 적용)")
    parser.add_argument("--topup_to", type=int, default=None, help="기존 파일의 데이터를 읽어 목표 총 건수까지 등급별 부족분을 자동 계산해 채움")
    parser.add_argument("--dry_run", action="store_true", help="API 호출 없이 샘플러만 테스트")
    args = parser.parse_args()

    # 1. 환경 및 API 키 검사
    has_key, api_key, has_google_genai = check_environment()

    # 2. KTAS 데이터베이스 로드
    print("\n[2] KTAS 데이터베이스 로드 및 인덱싱")
    db = KTASDatabase(data_dir="./data")
    sampler = ScenarioSampler(db)

    # dry-run 모드 검사
    if args.dry_run:
        print("\n[Dry Run 모드] 샘플러 시나리오 5건을 추출합니다:")
        for i in range(5):
            sc = sampler.sample(target_level=args.target_level, force_group=args.group)
            print(f"[{i+1}] {sc['age_group']}({sc['age']}세/{sc['gender']}) - {sc['chief_complaint_name']} | 조건: {sc['target_condition']} -> KTAS {sc['target_level']}단계({sc['urgency']})")
        print("\nDry Run 완료.")
        return

    if not has_key:
        print("\n[오류] GEMINI_API_KEY가 설정되지 않았습니다.")
        print("API 키를 등록한 후 다시 실행해 주십시오:")
        print("  방법 1: export GEMINI_API_KEY='your_api_key_here'")
        print("  방법 2: ./.env 파일에 GEMINI_API_KEY=your_api_key_here 작성")
        sys.exit(1)

    # 3. 데이터셋 생성기 초기화
    print(f"\n[3] Gemini 모델 초기화 (모델: {args.model})")
    generator = DatasetGenerator(api_key=api_key, model_name=args.model)

    output_path = Path(args.output)
    print(f" - 출력 파일: {output_path.resolve()}")

    # 기존 파일 분석
    existing_records = []
    from collections import Counter
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            parts = [p.strip() for p in text.split("\n\n") if p.strip().startswith("{")]
            for p in parts:
                try:
                    existing_records.append(json.loads(p))
                except Exception:
                    pass

    existing_counts = Counter([r.get("scenario_metadata", {}).get("target_level") for r in existing_records])
    existing_total = len(existing_records)
    print(f" - 기존 누적 데이터: 총 {existing_total}건 (1단계:{existing_counts[1]}, 2단계:{existing_counts[2]}, 3단계:{existing_counts[3]}, 4단계:{existing_counts[4]}, 5단계:{existing_counts[5]})")

    # 레벨별 분배 계획 수립
    if args.topup_to is not None:
        target_total = args.topup_to
        target_per_level = target_total // 5
        planned_levels = []
        for lvl in range(1, 6):
            needed = max(0, target_per_level - existing_counts[lvl])
            planned_levels.extend([lvl] * needed)
        # 부족분 합계가 남으면 고르게 분배
        remaining_needed = max(0, target_total - (existing_total + len(planned_levels)))
        for i in range(remaining_needed):
            planned_levels.append((i % 5) + 1)
        random.shuffle(planned_levels)
        print(f" - [Top-up 모드] 목표 {target_total}건을 채우기 위해 부족분 {len(planned_levels)}건을 추가 생성합니다.")
    elif args.target_level is not None:
        planned_levels = [args.target_level] * args.num_samples
        print(f" - 목표 레벨: {args.target_level}단계 고정 ({args.num_samples}건)")
    else:
        base_count = args.num_samples // 5
        remainder = args.num_samples % 5
        planned_levels = []
        for lvl in range(1, 6):
            planned_levels.extend([lvl] * base_count)
        for i in range(remainder):
            planned_levels.append((i % 5) + 1)
        random.shuffle(planned_levels)

    planned_counts = Counter(planned_levels)
    print(f" - 이번 생성 계획 ({len(planned_levels)}건): 1단계({planned_counts[1]}건), 2단계({planned_counts[2]}건), 3단계({planned_counts[3]}건), 4단계({planned_counts[4]}건), 5단계({planned_counts[5]}건)")
    print(f" - 호출 간 간격: {args.sleep}초 | 실시간 디스크 flush 활성화\n")

    success_count = 0
    total_target = args.num_samples
    pending_levels = list(planned_levels)

    with open(output_path, "a", encoding="utf-8") as f:
        attempt_idx = 0
        while pending_levels:
            lvl = pending_levels[0]
            attempt_idx += 1
            scenario = sampler.sample(target_level=lvl, force_group=args.group)
            print(f"[{success_count + 1}/{total_target}] KTAS {scenario['target_level']}단계 ({scenario['urgency']}) | {scenario['age_group']} ({scenario['age']}세/{scenario['gender']}) | {scenario['chief_complaint_name']} 생성 시도...")

            try:
                turns = generator.generate_single(scenario)
                timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{success_count+1:04d}"
                record = {
                    "id": f"ktas_{timestamp_id}",
                    "scenario_metadata": scenario,
                    "messages": turns
                }
                
                # 들여쓰기 포맷팅 (가독성 기본 적용)
                indent_val = None if args.no_pretty else 2
                separator = "\n" if args.no_pretty else "\n\n"
                f.write(json.dumps(record, ensure_ascii=False, indent=indent_val) + separator)
                f.flush()
                os.fsync(f.fileno())  # 즉시 디스크 반영
                
                pending_levels.pop(0)  # 성공 시 큐에서 제거
                success_count += 1
                print(f" -> 성공 (총 {len(turns)}턴 대화 저장 완료 | 전체 진행: {success_count}/{total_target})")
                
                import time
                time.sleep(args.sleep)
            except Exception as e:
                print(f" -> [실패 후 10초 대기 재시도]: {e}")
                import time
                time.sleep(10)

    print(f"\n" + "=" * 60)
    print(f"생성 완료: 총 {success_count}/{total_target}건 정상 기록됨")
    print(f"저장 위치: {output_path.resolve()}")

    # raw jsonl 및 json 배열 파일 자동 동기화
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()
        all_parts = [p.strip() for p in full_text.split("\n\n") if p.strip().startswith("{")]
        all_recs = []
        for p in all_parts:
            try:
                all_recs.append(json.loads(p))
            except Exception:
                pass

        with open(output_path.parent / "ktas_train_dataset_raw.jsonl", "w", encoding="utf-8") as f:
            for r in all_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(output_path.parent / "ktas_train_dataset.json", "w", encoding="utf-8") as f:
            json.dump(all_recs, f, ensure_ascii=False, indent=2)

        print(f" - [동기화 완료] 총 누적 {len(all_recs)}건 (ktas_train_dataset_raw.jsonl & ktas_train_dataset.json 갱신 완료)")
    except Exception as e:
        print(f" - [동기화 오류]: {e}")


if __name__ == "__main__":
    main()
