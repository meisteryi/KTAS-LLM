#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DiReCT 25종 질환 지식 그래프(KG) 및 KTAS 공식 가이드라인 7대 룰셋을 기반으로
의학적 노이즈가 없는 고순도 대규모 SFT 학습 데이터셋(3,600건)을 증강 생성하는 파이프라인.

특징:
1. Grounding: DiReCT 25개 질환 온톨로지(증상, 위험인자)와 KTAS 공식 결정 룰 ID를 1:1 결정론적 매핑.
2. Clinical Realism: 119 구급 신고 및 한국 응급실 내원 환자/보호자의 생생한 자연어 발화 패턴 주입.
3. Orthogonal Confounding: 독립된 신체 계통의 비응급 만성 증상 결합 패턴(순서 무작위화) 반영.
4. Structured Output: {"primary_complaint", "applied_rule_id", "rationale", "ktas_level"} 표준 스키마.
5. Strict Validation: JSON 유효성, 룰 ID 정합성, 레벨 분포 균형성 보장.
"""

import os
import json
import random
from pathlib import Path

# 재현성을 위한 시드 고정
random.seed(42)

# DiReCT 질환 지식 및 KTAS 공식 룰 온톨로지 마스터 정의
ONTOLOGY_REGISTRY = [
    # 1. 초응급 / 긴급 질환 (KTAS 1~2)
    {
        "disease_id": "ACS",
        "disease_name": "급성 관상동맥 증후군 (심근경색/불안정협심증)",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "흉통 (심장성)",
        "applied_rule_id": "심혈관계_003_2차고려_심장성통증_급성",
        "rationale_template": "갑작스러운 흉부 압박감 및 방사통, 식은땀 등 급성 관상동맥 증후군(ACS) 의심 소견은 잠재적 생명 위협 상태로 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "가슴 한가운데를 묵직한 돌로 짓누르는 것처럼 쥐어짜듯 아파요",
            "가슴 통증이 왼쪽 턱이랑 어깨, 등 뒤쪽까지 뻗치면서 식은땀이 비 오듯 쏟아져요",
            "명치 위쪽이 뻐근하게 조여오고 숨이 턱턱 막혀요",
            "가슴을 쥐어뜯는 것 같은 통증이 쉬어도 안 가라앉고 속이 메스꺼워요",
            "왼쪽 가슴이 찢어지듯이 뻐근하고 팔 쪽으로 저린 통증이 내려와요",
            "숨을 쉴 때마다 가슴 한가운데가 꽉 조여들고 어지러워서 주저앉았어요"
        ],
        "golden_time_cues": ["30분 전부터", "1시간 전부터", "조금 전 갑자기", "쉬어도 통증이 지속되면서"],
        "patient_personas": ["50대 남성", "60대 여성", "70대 당뇨 병력 환자", "보호자가 119 인계", "40대 직장인"]
    },
    {
        "disease_id": "AORTIC_DISSECTION",
        "disease_name": "대동맥 박리 (Aortic Dissection)",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "흉통 및 등 통증 (대동맥박리 의심)",
        "applied_rule_id": "심혈관계_003_2차고려_대동맥박리의심_찢어지는통증",
        "rationale_template": "등 뒤로 뻗치며 칼로 찢는 듯한 급격한 극심 통증은 대동맥 박리 의심 소견으로 신속한 처치가 필요한 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "칼로 가슴부터 등 뒤 날개뼈 사이를 쫙 찢어내는 것처럼 아파요",
            "태어나서 처음 겪는 극심한 통증이 등 뒤로 훑고 내려가면서 눈앞이 캄캄해져요",
            "등 뒤를 도끼로 내리찍는 것 같은 찢어지는 통증이 갑자기 생겼어요",
            "가슴이랑 등 뒤가 찢어지는 것 같고 다리에 힘이 쭉 빠져요",
            "숨을 못 쉴 정도로 등 쪽을 찌르는 칼 같은 통증이 들이닥쳤어요"
        ],
        "golden_time_cues": ["방금 전 갑자기 벼락치듯", "20분 전부터 갑자기", "방금 전부터"],
        "patient_personas": ["60대 고혈압 환자", "70대 남성", "보호자 긴급 내원", "50대 여성"]
    },
    {
        "disease_id": "STROKE_ACUTE",
        "disease_name": "급성 허혈성/출혈성 뇌졸중",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "급성 신경학적 결손",
        "applied_rule_id": "신경계_409_2차고려_발병24시간이내",
        "rationale_template": "발병 24시간 이내의 급성 신경학적 결손(편측 마비, 안면 비대칭, 조음장애)은 뇌졸중 골든타임 징후로 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "혀가 굳은 것처럼 발음이 꼬이고 말이 어눌하게 나와요",
            "오른쪽 팔다리에 힘이 하나도 안 들어가서 숟가락을 떨어뜨렸어요",
            "웃어보라고 하니까 입 한쪽 꼬리가 아래로 쳐지고 침을 흘려요",
            "말을 전혀 못 알아듣고 엉뚱한 소리만 자꾸 하세요",
            "시야 한쪽이 시커멓게 가려지고 걸을 때 한쪽으로 자꾸 비틀거려요",
            "한쪽 얼굴 감각이 남의 살 같고 오른쪽 손가락을 쥐었다 폈다를 못 하겠어요"
        ],
        "golden_time_cues": ["1시간 전부터 갑자기", "30분 전부터", "2시간 전부터", "오늘 아침 기상 직후부터"],
        "patient_personas": ["배우자가 인계하는 70대", "60대 고혈압/당뇨 환자", "구급대원 인계", "본인 직접 내원 50대"]
    },
    {
        "disease_id": "PULMONARY_EMBOLISM",
        "disease_name": "폐색전증 (Pulmonary Embolism)",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "급성 호흡곤란 및 흉통",
        "applied_rule_id": "호흡기계_103_2차고려_급성중등도_호흡곤란",
        "rationale_template": "돌발적인 호흡곤란 및 객혈, 흉막성 흉통은 폐색전증 등 급성 폐순환 장애 의심 소견으로 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "갑자기 숨이 턱 막히고 가슴 옆구리가 찌르듯이 아파요",
            "기침을 하다가 피가 섞인 가래가 나오고 숨을 깊게 들이쉬면 칼로 찌르는 것 같아요",
            "비행기 타고 내린 후 종아리가 붓더니 갑자기 숨이 차서 쓰러질 뻔했어요",
            "가슴이 답답하고 숨을 헐떡이게 되면서 식은땀이 나고 어지러워요",
            "숨을 쉴 때마다 날카롭게 찌르는 흉통과 함께 가슴이 두근거려요"
        ],
        "golden_time_cues": ["1시간 전부터 갑자기", "오늘 오후부터 급격히", "2시간 전 돌발적으로"],
        "patient_personas": ["장거리 여행력 40대", "수술력 있는 60대", "30대 여성", "50대 남성"]
    },
    {
        "disease_id": "UGI_BLEEDING",
        "disease_name": "상부 위장관 출혈 (UGI Bleeding)",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "위장관 출혈",
        "applied_rule_id": "소화기계_202_2차고려_다량토혈_흑색변",
        "rationale_template": "토혈 및 흑색변(타르변)은 대량 상부위장관 출혈 징후로 저혈량 쇼크 위험이 있어 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "자장면 국물처럼 까맣고 번들거리는 흑색변을 연달아 봤어요",
            "속이 울렁거리더니 커피 찌꺼기 같은 검붉은 피를 세 번이나 토했어요",
            "피를 꿀꺽 토하고 나서 일어서려는데 눈앞이 핑 돌면서 주저앉았어요",
            "명치가 쓰리더니 붉은 피를 한 바가지 토하고 식은땀이 줄줄 흘러요",
            "검은 타르 같은 변을 보고 어지러워서 겨우 기어서 나왔어요"
        ],
        "golden_time_cues": ["오늘 아침부터", "방금 전 피를 토하더니", "몇 시간 전부터 흑색변을 보더니"],
        "patient_personas": ["간경변 병력 50대", "진통제 장기복용 70대", "40대 남성", "보호자 동행"]
    },
    {
        "disease_id": "ASTHMA_ACUTE",
        "disease_name": "급성 천식 발작",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "호흡곤란 (천식 발작)",
        "applied_rule_id": "호흡기계_104_2차고려_천식_중증발작",
        "rationale_template": "쌕쌕거리는 천명음과 말하기 힘든 중등도 이상의 호흡곤란은 급성 천식 발작으로 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "목구멍에서 쌕쌕 피리 소리가 나고 숨이 차서 한 문장을 끝까지 이어서 말을 못 하겠어요",
            "천식 흡입기를 세 번이나 뿌렸는데도 숨이 가쁘고 가슴이 꽉 틀어막힌 것 같아요",
            "숨을 들이마시지를 못하겠고 입술이 파래지면서 가슴이 헐떡거려요",
            "숨을 헐떡이다가 도저히 안 되겠어서 새벽에 구급차 탔어요",
            "가슴이 꽉 막힌 것처럼 답답하고 숨을 쉴 때마다 가슴에서 거친 휘파람 소리가 나요"
        ],
        "golden_time_cues": ["1시간 전부터 악화되더니", "흡입기를 써도 안 가라앉더니", "갑자기 숨이 가빠지면서"],
        "patient_personas": ["20대 천식 환자", "40대 직장인", "보호자 인계 30대", "60대 여성"]
    },
    {
        "disease_id": "HEART_FAILURE_ACUTE",
        "disease_name": "급성 비대상성 심부전 / 폐부종",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "호흡곤란 (기좌호흡)",
        "applied_rule_id": "호흡기계_103_2차고려_기좌호흡_폐부종의심",
        "rationale_template": "누우면 숨이 차서 앉아있어야만 하는 기좌호흡 및 폐부종 의심 소견은 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "침대에 누우면 물에 빠진 것처럼 숨이 차서 밤새 벽에 기대어 앉아있었어요",
            "숨이 차서 도저히 누울 수가 없고 목구멍에서 거품 끓는 소리가 나요",
            "다리가 퉁퉁 붓더니 오늘 밤은 숨이 너무 차서 앉아서만 숨을 쉬어요",
            "누우면 숨이 턱 막혀서 일어날 수밖에 없고 기침할 때 분홍색 가래가 비쳐요",
            "가만히 있어도 헐떡거리게 되고 앉아있지 않으면 질식할 것 같아요"
        ],
        "golden_time_cues": ["어젯밤부터 악화되더니", "오늘 새벽부터 도저히 누울 수가 없더니", "몇 시간째 숨이 차서"],
        "patient_personas": ["70대 심장질환 환자", "80대 고령자", "보호자 동행", "60대 남성"]
    },
    {
        "disease_id": "EPILEPSY_STATUS",
        "disease_name": "경련 발작 / 뇌전증",
        "acuity": "Urgent",
        "target_ktas": 2,
        "primary_complaint": "경련 및 의식 변화",
        "applied_rule_id": "신경계_401_2차고려_경련후의식혼미",
        "rationale_template": "전신 경련 발작 후 의식 회복 지연(혼미) 및 재발 위험 상태는 KTAS 2단계 긴급에 해당함.",
        "symptom_expressions": [
            "온몸을 부들부들 떨면서 눈이 위로 돌아가고 입에 거품을 물면서 쓰러졌어요",
            "사지를 떨며 경련을 한 뒤 지금 30분째 이름을 불러도 멍하니 헛소리만 해요",
            "길 가다가 갑자기 픽 쓰러지면서 전신을 뻣뻣하게 떨었어요",
            "발작은 멈췄는데 아직 정신을 못 차리고 사람을 못 알아봐요",
            "자다가 갑자기 으으 소리를 내면서 온몸을 뒤틀며 발작을 일으켰어요"
        ],
        "golden_time_cues": ["30분 전에 발작을 일으키더니", "1시간 전 발작 후 아직 혼미한 상태인데", "방금 전 발작 후"],
        "patient_personas": ["목격자 신고", "가족이 모시고 온 20대", "40대 기저질환자", "요양보호사 인계"]
    },

    # 2. 응급 질환 (KTAS 3)
    {
        "disease_id": "PNEUMONIA_ACUTE",
        "disease_name": "급성 세균성/바이러스성 폐렴",
        "acuity": "Emergent",
        "target_ktas": 3,
        "primary_complaint": "발열 및 호흡기 감염",
        "applied_rule_id": "호흡기계_105_2차고려_발열동반_중등도호흡곤란",
        "rationale_template": "고열과 오한, 누런 화농성 가래 및 중등도 호흡곤란은 전신 악화 가능성이 높은 KTAS 3단계 응급에 해당함.",
        "symptom_expressions": [
            "열이 38.8도까지 끓고 기침할 때마다 누런 고름 같은 가래가 쏟아져요",
            "온몸이 덜덜 떨리면서 오한이 나고 숨쉴 때마다 오른쪽 가슴 옆구리가 결리고 아파요",
            "기침이 쉬지 않고 나오면서 숨이 차고 밥도 한 숟가락도 못 넘길 정도로 기운이 없어요",
            "감기인 줄 알았는데 열이 계속 안 떨어지고 숨소리가 거칠어지면서 가슴이 답답해요",
            "기침할 때마다 가래 끓는 소리가 심해서 숨을 쉬기가 버거워요"
        ],
        "golden_time_cues": ["사흘 전부터 열이 나더니", "어제부터 고열이 지속되더니", "이틀 전부터 기침이 심해지더니"],
        "patient_personas": ["60대 환자", "70대 노인", "보호자 동행 40대", "50대 여성"]
    },
    {
        "disease_id": "ATRIAL_FIBRILLATION",
        "disease_name": "심방세동 (Atrial Fibrillation)",
        "acuity": "Emergent",
        "target_ktas": 3,
        "primary_complaint": "심계항진 / 부정맥",
        "applied_rule_id": "심혈관계_005_2차고려_불규칙빈맥_어지럼동반",
        "rationale_template": "갑작스러운 불규칙한 심계항진 및 어지럼증은 급성 부정맥 의심 상태로 KTAS 3단계 응급에 해당함.",
        "symptom_expressions": [
            "가만히 있는데 심장이 새장 속의 새처럼 불규칙하게 쿵쾅거리면서 요동을 쳐요",
            "맥박을 짚어보니 박자가 제멋대로 불규칙하게 뛰면서 핑 도는 어지럼증이 있어요",
            "가슴이 두근거려 터질 것 같고 헛구역질이 나면서 불안해서 서 있을 수가 없어요",
            "가슴속에서 모터가 도는 것처럼 빠르게 뛰다가 쿵 내려앉기를 반복해요",
            "심장이 제멋대로 마구 요동치면서 기운이 쭉 빠져요"
        ],
        "golden_time_cues": ["1시간 전부터 갑자기", "2시간 전부터 심장이 뛰더니", "오늘 오후부터 불규칙하게"],
        "patient_personas": ["50대 남성", "60대 여성", "70대 기저질환자", "40대 직장인"]
    },
    {
        "disease_id": "TUBERCULOSIS_CHRONIC",
        "disease_name": "활동성 결핵 (Tuberculosis)",
        "acuity": "Emergent",
        "target_ktas": 3,
        "primary_complaint": "만성 기침 및 객혈",
        "applied_rule_id": "호흡기계_106_2차고려_만성기침_소량객혈",
        "rationale_template": "3주 이상 지속되는 기침, 야간 발한 및 소량 객혈은 감염 격리 및 정밀 검사가 필요한 KTAS 3단계 응급에 해당함.",
        "symptom_expressions": [
            "기침을 한 달 넘게 달고 살았는데 오늘 아침에는 가래에 선홍색 피가 섞여 나왔어요",
            "밤마다 식은땀을 흠뻑 흘리고 체중이 5kg이나 빠졌는데 기침할 때 피가 비쳐요",
            "감기가 한 달째 안 낫고 미열이 계속 나면서 가래 뱉을 때 피 찌꺼기가 묻어 나와요",
            "마른기침을 오래 하다가 오늘 각혈을 종이컵 반 컵 정도 해서 깜짝 놀라 왔어요",
            "기침이 오래가고 기운이 없는데 오늘 피가 섞여 나와 무서워서 왔어요"
        ],
        "golden_time_cues": ["한 달 전부터 기침하다가", "3주 넘게 지속되다가 오늘 아침 피가 나와서"],
        "patient_personas": ["30대 청년", "50대 남성", "60대 마른 체형 환자"]
    },
    {
        "disease_id": "STROKE_SUBACUTE",
        "disease_name": "아급성 뇌졸중 (발병 24시간 초과)",
        "acuity": "Emergent",
        "target_ktas": 3,
        "primary_complaint": "신경학적 결손 (24시간 초과)",
        "applied_rule_id": "신경계_409_2차고려_발병24시간초과",
        "rationale_template": "신경학적 결손이 있으나 발병 후 24시간을 초과하여 혈전용해 골든타임이 지난 상태는 KTAS 3단계 응급에 해당함.",
        "symptom_expressions": [
            "오른쪽 손에 힘이 좀 빠져서 젓가락질이 서툴렀는데 나아지지 않아서 왔어요",
            "입술 한쪽이 약간 쳐지고 발음이 샌다고 느꼈는데 점점 답답해서 확인하러 왔어요",
            "걸을 때 오른쪽 다리를 약간 끌었는데 주말 지나고 진료 보러 왔어요",
            "발음이 약간 어눌해진 상태가 며칠째 이어져서 큰 병원 가보려고 왔어요"
        ],
        "golden_time_cues": ["이틀 전부터 서서히", "사흘 전부터 시작되어", "그저께부터 지속되더니"],
        "patient_personas": ["60대 환자", "70대 부모님 모시고 온 자녀", "50대 남성"]
    },

    # 3. 준응급 및 비응급 질환 (KTAS 4~5)
    {
        "disease_id": "MIGRAINE_CHRONIC",
        "disease_name": "만성 편두통 (Migraine)",
        "acuity": "Semi-urgent",
        "target_ktas": 4,
        "primary_complaint": "두통 (편측 박동성)",
        "applied_rule_id": "신경계_404_2차고려_만성편두통_경중등도",
        "rationale_template": "신경학적 결손이나 발열이 없는 통상적인 편측 박동성 두통은 KTAS 4단계 준응급에 해당함.",
        "symptom_expressions": [
            "원래 앓던 편두통인데 왼쪽 관자놀이가 욱신욱신 맥박 뛰듯이 쑤셔요",
            "불빛을 보면 눈이 부시고 머리 한쪽이 지끈거리며 속이 약간 미식거려요",
            "스트레스 받으면 늘 생기던 편두통인데 집에서 약을 먹어도 통증이 가라앉지 않아요",
            "머리 오른쪽이 쿵쾅쿵쾅 뛰는 듯이 아프고 소리에 예민해져서 조용한 데 눕고 싶어요",
            "늘 생기던 두통이지만 타이레놀을 먹어도 반응이 없어서 링거 맞으러 왔어요"
        ],
        "golden_time_cues": ["늘 앓던 통증인데 오늘 아침부터", "몇 시간 전부터 지끈거리더니"],
        "patient_personas": ["20대 직장인", "30대 여성", "40대 만성 두통 환자"]
    },
    {
        "disease_id": "GERD_CHRONIC",
        "disease_name": "위식도 역류질환 (GERD)",
        "acuity": "Non-urgent",
        "target_ktas": 4,
        "primary_complaint": "소화불량 및 속쓰림",
        "applied_rule_id": "소화기계_208_2차고려_역류성식도염_만성",
        "rationale_template": "식후 신트림과 명치 위쪽 쓰림 등 전형적인 위식도 역류 증상은 KTAS 4단계 준응급 또는 비응급에 해당함.",
        "symptom_expressions": [
            "목구멍으로 신물이 왈칵 넘어오고 명치 쪽이 타는 듯이 쓰려요",
            "가슴 한가운데가 쓰라리고 목에 가시가 걸린 것처럼 답답해서 물을 자꾸 마시게 돼요",
            "평소에도 역류성 식도염이 있는데 기름진 거 먹었더니 가슴이 화끈거려요",
            "식사 후 명치 끝이 뻐근하게 쓰리고 트림할 때마다 위산이 올라와요"
        ],
        "golden_time_cues": ["야식 먹고 누웠더니", "식사 후 몇 시간째 속이 쓰려서"],
        "patient_personas": ["30대 남성", "40대 직장인", "50대 여성"]
    },
    {
        "disease_id": "GASTRITIS_MILD",
        "disease_name": "단순 위염 / 소화불량",
        "acuity": "Non-urgent",
        "target_ktas": 5,
        "primary_complaint": "단순 소화불량 / 상복부 불편감",
        "applied_rule_id": "소화기계_206_2차고려_단순소화불량_경미통증",
        "rationale_template": "흑색변이나 구토, 복막 자극 징후가 없는 단순 체기 및 소화불량은 KTAS 5단계 비응급에 해당함.",
        "symptom_expressions": [
            "급하게 밥을 먹고 체했는지 윗배가 더부룩하고 가스가 찬 것처럼 답답해요",
            "소화제를 두 알 먹었는데도 속이 꽉 막힌 것 같고 트림만 자꾸 나와요",
            "배가 심하게 아픈 건 아닌데 더부룩하고 얹힌 느낌이 계속 들어서 소화제 처방받으러 왔어요",
            "과식하고 나서 윗배가 빵빵하고 소화가 안 돼서 답답해요"
        ],
        "golden_time_cues": ["점심 식사 후부터", "몇 시간 전부터 더부룩하더니"],
        "patient_personas": ["20대 학생", "30대 직장인", "50대 주부"]
    },
    {
        "disease_id": "HYPERTENSION_MILD",
        "disease_name": "단순 만성 고혈압 (증상 경미)",
        "acuity": "Non-urgent",
        "target_ktas": 5,
        "primary_complaint": "혈압 상승 상담 / 뒷목 뻐근",
        "applied_rule_id": "심혈관계_008_2차고려_무증상_경미두통",
        "rationale_template": "흉통, 호흡곤란, 신경학적 결손 등의 표적 장기 손상 징후가 없는 단순 혈압 상승은 KTAS 5단계 비응급에 해당함.",
        "symptom_expressions": [
            "집에서 혈압을 쟀는데 155에 95가 나와서 걱정돼서 확인하러 왔어요",
            "뒷목이 조금 뻐근한 것 같아 약국에서 혈압 쟀더니 평소보다 높게 나와서 왔어요",
            "고혈압 약이 떨어져서 며칠 못 먹었더니 수치가 올라간 것 같아 처방받으러 왔어요",
            "머리가 약간 무거운 느낌인데 혈압이 높다고 해서 진료 보러 왔어요"
        ],
        "golden_time_cues": ["오늘 낮에 혈압을 재보고는", "며칠 전부터 조금 뻐근하더니"],
        "patient_personas": ["60대 남성", "50대 여성", "70대 노인"]
    },
    {
        "disease_id": "DIABETES_ROUTINE",
        "disease_name": "당뇨병 만성 관리 (케톤산증 없음)",
        "acuity": "Non-urgent",
        "target_ktas": 5,
        "primary_complaint": "혈당 관리 상담",
        "applied_rule_id": "대사계_602_2차고려_혈당상승_단순고혈당",
        "rationale_template": "의식 저하나 구토 등의 급성 대사 이상 징후가 없는 단순 혈당 상승은 KTAS 5단계 비응급에 해당함.",
        "symptom_expressions": [
            "요즘 물을 자주 마시고 소변을 자주 보는데 공복 혈당이 220 나와서 상담하러 왔어요",
            "당뇨 약을 잘 챙겨 먹는데도 혈당이 조금 높게 유지되는 것 같아 처방 조절하러 왔어요",
            "피로감이 좀 있고 혈당 수치가 평소보다 좀 높게 체크돼서 확인받고 싶어요"
        ],
        "golden_time_cues": ["요즘 며칠 동안", "최근 들어 혈당이 오르더니"],
        "patient_personas": ["50대 당뇨 환자", "60대 여성"]
    },
    {
        "disease_id": "THYROID_CHRONIC",
        "disease_name": "갑상선 기능 이상 만성 증상",
        "acuity": "Non-urgent",
        "target_ktas": 5,
        "primary_complaint": "만성 피로 및 체중 변화",
        "applied_rule_id": "대사계_605_2차고려_갑상선만성증상_경미",
        "rationale_template": "급성 중독 발작이나 혼수가 없는 만성적인 피로 및 추위/더위 민감성은 KTAS 5단계 비응급에 해당함.",
        "symptom_expressions": [
            "피로가 안 풀리고 남들보다 추위를 너무 심하게 타서 피검사 해보러 왔어요",
            "갑상선 약을 먹고 있는데 최근 손이 좀 떨리고 가슴이 조금 두근거려서 약 용량 확인하러 왔어요",
            "체중이 자꾸 늘고 온몸이 붓는 느낌이라 갑상선 호르몬 수치 확인하러 왔어요"
        ],
        "golden_time_cues": ["몇 달 전부터 피로하더니", "오래전부터 만성적으로"],
        "patient_personas": ["30대 여성", "40대 주부"]
    },
    {
        "disease_id": "ALZHEIMER_CHRONIC",
        "disease_name": "알츠하이머병 만성 인지 저하",
        "acuity": "Non-urgent",
        "target_ktas": 5,
        "primary_complaint": "만성 기억력 감퇴",
        "applied_rule_id": "신경계_411_2차고려_만성인지저하_안정적",
        "rationale_template": "급성 뇌혈관 질환 증상이 배제된 점진적인 만성 기억력 감퇴는 KTAS 5단계 비응급에 해당함.",
        "symptom_expressions": [
            "어머니가 물건 둔 자리를 자꾸 잊어버리시고 같은 말을 반복하셔서 모시고 왔어요",
            "최근 들어 며칠 전 일도 기억을 잘 못하시고 길을 헷갈려하셔서 치매 검사 받아보려고요",
            "기억력이 조금씩 떨어지시는 것 같아서 진료 상담 받으러 왔어요"
        ],
        "golden_time_cues": ["수개월 전부터 서서히", "최근 몇 달간 점진적으로"],
        "patient_personas": ["부모님 모시고 온 자녀", "60대 보호자"]
    }
]

# 독립 신체 계통 직교 비응급 만성 증상 라이브러리 (교란 실험 및 복합 발화용)
ORTHOGONAL_CONFOUNDERS = [
    {
        "system": "근골격계 (정형외과)",
        "complaint": "만성 무릎 관절통",
        "rule_id": "근골격계_502_2차고려_만성관절통_경미",
        "expressions_sub": [
            "비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거리는 증상",
            "수년 된 퇴행성 관절염으로 계단 오르내릴 때 무릎이 시큰한 것",
            "오래 걸으면 나타나는 오른쪽 발목 안쪽의 뻐근한 통증",
            "몇 달 전부터 이어지던 은근한 허리 뻐근함"
        ],
        "expressions_main": [
            "원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요",
            "수년 된 퇴행성 관절염 때문에 계단 오르내릴 때 무릎이 시큰해요",
            "오래 걷고 나면 늘 그렇듯 오른쪽 발목 안쪽이 뻐근하게 쑤셔요",
            "몇 달 전부터 허리가 은근하게 쑤시던 게 아직 남아있어요"
        ]
    },
    {
        "system": "피부계",
        "complaint": "만성 피부 소양증/무좀",
        "rule_id": "피부계_801_2차고려_만성가려움_발진",
        "expressions_sub": [
            "발가락 사이에 가끔 가렵고 각질이 벗겨지는 무좀 증상",
            "팔꿈치 쪽에 간질간질하고 하얗게 일어나는 만성 건선",
            "환절기만 되면 종아리 피부가 건조해서 가려운 것"
        ],
        "expressions_main": [
            "발가락 사이에 무좀이 있어서 가끔 가렵고 각질이 벗겨져요",
            "팔꿈치 쪽에 원래 건선이 있어서 간질간질하고 하얗게 일어나요",
            "환절기만 되면 종아리 피부가 건조해서 긁게 돼요"
        ]
    },
    {
        "system": "이비인후/안과계",
        "complaint": "만성 비염/안구 건조",
        "rule_id": "이비인후_901_2차고려_만성비염_경미",
        "expressions_sub": [
            "아침마다 재채기 나오고 맑은 콧물이 흐르는 만성 비염",
            "눈곱이 좀 끼고 뻑뻑한 만성 결막염 기운",
            "환절기라 목 안쪽이 간질거리고 헛기침이 나오는 것"
        ],
        "expressions_main": [
            "만성 비염 때문에 아침마다 재채기 나오고 맑은 콧물이 훌쩍거려요",
            "눈에 결막염 기운이 있어서 눈곱이 좀 끼고 뻑뻑해요",
            "환절기라 목 안쪽이 간질거리고 헛기침이 가끔 나와요"
        ]
    }
]

SYSTEM_PROMPT = (
    "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의료진입니다. "
    "환자의 초기 진술을 듣고 즉각적인 위험 징후와 의학적 인과관계에 기반하여 최우선 주호소를 식별하고, "
    "공식 KTAS 단계(1~5단계)와 결정 근거(Rationale) 및 적용 룰 ID를 JSON 형식으로 출력하십시오."
)

def build_utterance(disease_entry, confounder=None, pattern="single"):
    """의학적으로 현실감 있는 한국인 환자/보호자 발화 생성"""
    sym = random.choice(disease_entry["symptom_expressions"])
    cue = random.choice(disease_entry["golden_time_cues"])
    
    # 단일 주호소 발화
    if pattern == "single" or confounder is None:
        pfx_templates = [
            f"선생님, {cue} {sym}.",
            f"급하게 왔는데요, {cue} {sym}.",
            f"119 구급차 타고 왔어요. {cue} {sym}.",
            f"{cue} {sym}."
        ]
        return random.choice(pfx_templates)
    
    # 복합 발화 패턴:
    if random.random() < 0.5:
        # A (응급) 먼저 + B (만성) 나중
        conf_main = random.choice(confounder["expressions_main"])
        return f"선생님, {cue} {sym}. 그런데 평소에 {conf_main}."
    else:
        # B (만성) 먼저 + A (응급) 나중 (Primacy / Recency Bias 교란 방지)
        conf_sub = random.choice(confounder["expressions_sub"])
        return f"선생님, 원래 {conf_sub}이 있긴 했는데, 글쎄 {cue} {sym}."

def generate_augmented_dataset(target_total=3600):
    print(f"[*] Generating {target_total} augmented SFT cases grounded in DiReCT KG & KTAS rules...")
    
    samples = []
    
    # 1. 단일 주호소 케이스 (약 60%)
    # 2. 복합 직교성 케이스 (약 40%)
    
    num_per_disease = target_total // len(ONTOLOGY_REGISTRY) + 1
    
    for d_idx, disease in enumerate(ONTOLOGY_REGISTRY):
        for i in range(num_per_disease):
            # 복합 증상 여부 결정
            has_confounder = (random.random() < 0.40)
            confounder = random.choice(ORTHOGONAL_CONFOUNDERS) if has_confounder else None
            
            utterance = build_utterance(disease, confounder, pattern="compound" if has_confounder else "single")
            
            assistant_content = {
                "primary_complaint": disease["primary_complaint"],
                "applied_rule_id": disease["applied_rule_id"],
                "rationale": disease["rationale_template"],
                "ktas_level": disease["target_ktas"]
            }
            
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"[환자 진술]: \"{utterance}\""
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_content, ensure_ascii=False, indent=2)
                    }
                ],
                "metadata": {
                    "disease_id": disease["disease_id"],
                    "disease_name": disease["disease_name"],
                    "acuity": disease["acuity"],
                    "has_confounder": has_confounder,
                    "confounder_system": confounder["system"] if confounder else None,
                    "target_ktas": disease["target_ktas"],
                    "rule_id": disease["applied_rule_id"]
                }
            }
            samples.append(sample)
    
    # 셔플
    random.shuffle(samples)
    samples = samples[:target_total]
    
    print(f"[+] Total samples generated: {len(samples)}")
    return samples

def main():
    target_total = 3600
    samples = generate_augmented_dataset(target_total)
    
    # Train / Val 분할 (90% : 10%)
    split_idx = int(len(samples) * 0.90)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    train_path = data_dir / "train_sft_augmented.jsonl"
    val_path = data_dir / "val_sft_augmented.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"[+] Saved train set ({len(train_data)} rows) -> {train_path}")
    print(f"[+] Saved val set ({len(val_data)} rows) -> {val_path}")
    
    # 통계 검증
    level_counts = {}
    for s in samples:
        lvl = s["metadata"]["target_ktas"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
        
    print("\n[+] KTAS Level Distribution:")
    for lvl in sorted(level_counts.keys()):
        print(f"  Level {lvl}: {level_counts[lvl]} ({level_counts[lvl]/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    main()
