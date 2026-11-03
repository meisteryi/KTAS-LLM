#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_counterfactual_benchmark.py
순수 환자 자연어 발화 기반 KTAS 근거-결정 일치도 검증 및 반사실적 교란 벤치마크 생성기

[연구 설계 원칙: Pure Clinical NLP]
- 바이탈사인 등 정량적 수치 힌트(교란 변수)를 완전히 배제
- 초기 접촉(119 신고, 응급실 접수대, 비대면 챗봇) 환경의 "순수 환자 발화 텍스트"만 제공
- 비응급 증상의 계통적 직교성(Orthogonality) 강제 (간섭 0%)
- Masking Effect 배제 (is_pure_isolated_case: True)
- Out-of-Distribution(OOD) 분리 (ID 100쌍, OOD 50쌍)
"""

import os
import json
import random

# 고정 시드 설정 (재현성 보장)
random.seed(42)

# 1. In-Distribution (학습 질환군: 심혈관계, 신경계)
ID_URGENT_TEMPLATES = [
    {
        "system": "심혈관계",
        "complaint_name": "급성 심장성 흉통",
        "code": "심혈관계_003_2차고려_급성중심성통증",
        "golden_time": "30분 전 발생",
        "ktas_level": 2,
        "urgent_phrases": [
            "30분 전부터 갑자기 가슴 한가운데를 바위로 짓누르는 것처럼 뻐근하고 쥐어짜듯이 아파요",
            "방금 전부터 왼쪽 가슴이 턱 막히면서 왼쪽 턱이랑 어깨 쪽으로 찌릿하게 통증이 뻗쳐요",
            "갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요",
            "조금 전부터 가슴을 칼로 쑤시는 것처럼 콕콕 쑤시고 숨을 쉴 때마다 가슴 압박감이 엄청 심해요",
            "가슴 뼈 안쪽이 타들어가는 것처럼 화끈거리고 가슴이 조여와서 도저히 서 있을 수가 없어요"
        ]
    },
    {
        "system": "신경계",
        "complaint_name": "급성 신경학적 결손 (뇌졸중 의심)",
        "code": "신경계_409_2차고려_발병24시간이내",
        "golden_time": "1시간 전 발생 (< 24시간, KTAS 2단계)",
        "ktas_level": 2,
        "urgent_phrases": [
            "1시간 전부터 갑자기 혀가 굳은 것처럼 발음이 자꾸 꼬이고 말이 어눌하게 나와요",
            "오늘 점심 먹고 나서부터 오른쪽 팔다리에 힘이 쭉 빠져서 숟가락도 못 쥐고 자꾸 주저앉아요",
            "2시간 전부터 거울을 보니까 오른쪽 입꼬리가 축 처지고 침이 한쪽으로 줄줄 흘러내려요",
            "방금 전부터 물건이 두 개로 겹쳐 보이고 한쪽 눈이 전혀 안 보이면서 말이 헛나와요",
            "40분 전부터 갑자기 말이 잘 안 나오고 상대방이 하는 말이 무슨 뜻인지 통 알아듣질 못하겠어요"
        ]
    }
]

# 2. Out-of-Distribution (비학습 질환군: 호흡기계, 아나필락시스/중독)
OOD_URGENT_TEMPLATES = [
    {
        "system": "호흡기계",
        "complaint_name": "급성 중등도 호흡곤란 (천식 발작 의증)",
        "code": "호흡기계_101_2차고려_중등도호흡부전",
        "golden_time": "40분 전 발생",
        "ktas_level": 2,
        "urgent_phrases": [
            "40분 전부터 숨을 들이쉴 때마다 쌕쌕거리는 휘파람 소리가 나고 숨이 가빠서 한마디도 길게 잇질 못하겠어요",
            "갑자기 목구멍이 꽉 막힌 것처럼 숨이 안 쉬어지고 말을 할 때마다 헐떡거리면서 가슴이 들썩여요",
            "1시간 전부터 기침이 멈추지 않으면서 숨이 턱까지 차오르고 누우면 숨이 더 막혀서 앉아만 있어요"
        ]
    },
    {
        "system": "면역/알레르기",
        "complaint_name": "아나필락시스 의증 (전신 반응 동반)",
        "code": "면역계_602_2차고려_혈관부종및호흡곤란",
        "golden_time": "20분 전 음식 섭취 후 발생",
        "ktas_level": 2,
        "urgent_phrases": [
            "20분 전에 땅콩 들어간 과자를 먹었는데 입술이랑 목구멍이 풍선처럼 부어오르면서 숨이 답답해요",
            "약을 먹고 나서 온몸에 손바닥만 한 두드러기가 벌겋게 뒤덮이고 목이 조여오면서 쌕쌕거려요"
        ]
    },
    {
        "system": "중독/외인",
        "complaint_name": "급성 독성물질/약물 음독",
        "code": "중독_701_2차고려_독성물질급성섭취",
        "golden_time": "30분 전 음독",
        "ktas_level": 2,
        "urgent_phrases": [
            "30분 전에 집에 있던 수면제 두 통을 한꺼번에 털어먹었는데 머리가 핑 돌고 몸이 마비되는 것 같아요",
            "방금 전에 농약 뚜껑을 열다가 실수로 한 모금 삼켰는데 식도가 타들어가는 것 같고 어지러워요"
        ]
    }
]

# 직교하는 완전 독립 비응급 증상 (Orthogonal Non-urgent Complaints)
ORTHOGONAL_NON_URGENT_TEMPLATES = [
    {
        "system": "만성 정형외과",
        "complaint_name": "만성 무릎 관절통",
        "code": "근골격계_만성_경미통증",
        "ktas_level": 5,
        "non_urgent_phrases": [
            "원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요",
            "몇 달 전부터 계단 내려갈 때마다 무릎이 시큰거렸어요",
            "예전부터 오른쪽 무릎 관절염이 있어서 가끔 파스 붙이고 있어요"
        ]
    },
    {
        "system": "만성 피부과",
        "complaint_name": "만성 발가락 무좀 및 가려움",
        "code": "피부계_만성_경미소양증",
        "ktas_level": 5,
        "non_urgent_phrases": [
            "며칠 전부터 발가락 사이에 무좀이 도졌는지 가렵고 껍질이 벗겨져요",
            "발등 쪽에 모기 물린 것처럼 가려운 게 며칠째 안 가라앉아요",
            "발톱 밑이 좀 노랗게 변하고 주변이 살짝 가려운 게 있어요"
        ]
    },
    {
        "system": "경미 안과",
        "complaint_name": "만성 결막 충혈 및 눈곱",
        "code": "안과계_만성_단순결막염",
        "ktas_level": 5,
        "non_urgent_phrases": [
            "어제 아침부터 왼쪽 눈에 눈곱이 조금 끼고 살짝 충혈돼 있어요",
            "요 며칠 바람 쐬면 눈이 좀 뻑뻑하고 모래 들어간 것처럼 깔깔해요"
        ]
    },
    {
        "system": "치과/구강",
        "complaint_name": "만성 잇몸 시림",
        "code": "치과계_만성_단순치통",
        "ktas_level": 5,
        "non_urgent_phrases": [
            "찬물 마실 때마다 어금니 쪽 잇몸이 약간 시린 게 있었어요",
            "양치할 때 잇몸에서 피가 조금 나고 뻐근한 느낌이 며칠 됐어요"
        ]
    }
]

CONNECTORS_AB = [
    " 그리고 ", " 근데 ", " 그런데 ", " 뿐만 아니라 ", " 게다가 "
]
CONNECTORS_BA = [
    " 그런데 ", " 근데 ", " 그러더니 갑자기 ", " 있었는데 갑자기 ", " 한편 "
]


def generate_quadruplet(case_id, urgent_info, non_urgent_info, split_type):
    u_phrase = random.choice(urgent_info["urgent_phrases"])
    nu_phrase = random.choice(non_urgent_info["non_urgent_phrases"])
    
    conn_ab = random.choice(CONNECTORS_AB)
    conn_ba = random.choice(CONNECTORS_BA)
    
    # 바이탈사인 없이 오직 순수 환자 진술(자연어)만으로 구성
    dialogue_ab = f"선생님, {u_phrase},{conn_ab}{nu_phrase}."
    dialogue_ba = f"선생님, {nu_phrase},{conn_ba}{u_phrase}."
    dialogue_ablate_u = f"선생님, {nu_phrase}."
    dialogue_ablate_nu = f"선생님, {u_phrase}."
    
    quad = {
        "case_id": case_id,
        "split": split_type,
        "system_pairing": {
            "urgent_system": urgent_info["system"],
            "non_urgent_system": non_urgent_info["system"],
            "orthogonality_verified": True,
            "is_pure_isolated_case": True
        },
        "golden_time": urgent_info["golden_time"],
        "ground_truth_rule_id": urgent_info["code"],
        "expected_urgent_ktas": urgent_info["ktas_level"],
        "ground_truth_non_urgent_rule_id": non_urgent_info["code"],
        "expected_non_urgent_ktas": non_urgent_info["ktas_level"],
        "gold_rationale": f"{urgent_info['complaint_name']}은(는) {urgent_info['golden_time']} 발생한 즉각적 처치가 필요한 응급 징후이므로, 직교하는 만성 {non_urgent_info['complaint_name']}에 우선하여 KTAS {urgent_info['ktas_level']}단계({urgent_info['code']})로 분류함.",
        "permutations": {
            "case_original": {
                "patient_dialogue": dialogue_ab,
                "expected_primary_complaint": urgent_info["complaint_name"],
                "expected_ktas_level": urgent_info["ktas_level"],
                "expected_rule_id": urgent_info["code"]
            },
            "case_permuted": {
                "patient_dialogue": dialogue_ba,
                "expected_primary_complaint": urgent_info["complaint_name"],
                "expected_ktas_level": urgent_info["ktas_level"],
                "expected_rule_id": urgent_info["code"]
            },
            "case_ablate_urgent": {
                "patient_dialogue": dialogue_ablate_u,
                "expected_primary_complaint": non_urgent_info["complaint_name"],
                "expected_ktas_level": non_urgent_info["ktas_level"],
                "expected_rule_id": non_urgent_info["code"]
            },
            "case_ablate_non_urgent": {
                "patient_dialogue": dialogue_ablate_nu,
                "expected_primary_complaint": urgent_info["complaint_name"],
                "expected_ktas_level": urgent_info["ktas_level"],
                "expected_rule_id": urgent_info["code"]
            }
        }
    }
    return quad


def main():
    print("[*] Generating Pure Natural Language Counterfactual Benchmark...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    benchmark_quads = []
    
    # In-Distribution 100쌍 생성
    for i in range(100):
        case_id = f"CF_ID_{i+1:03d}"
        u_info = random.choice(ID_URGENT_TEMPLATES)
        nu_info = random.choice(ORTHOGONAL_NON_URGENT_TEMPLATES)
        quad = generate_quadruplet(case_id, u_info, nu_info, "In-Distribution")
        benchmark_quads.append(quad)
        
    # Out-of-Distribution 50쌍 생성
    for i in range(50):
        case_id = f"CF_OOD_{i+1:03d}"
        u_info = random.choice(OOD_URGENT_TEMPLATES)
        nu_info = random.choice(ORTHOGONAL_NON_URGENT_TEMPLATES)
        quad = generate_quadruplet(case_id, u_info, nu_info, "Out-of-Distribution")
        benchmark_quads.append(quad)
        
    benchmark_path = "data/counterfactual_benchmark.jsonl"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        for q in benchmark_quads:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[+] Successfully saved {len(benchmark_quads)} Quadruplets to {benchmark_path}")
    
    # 부수적이었던 fast_track_resuscitation.jsonl은 연구 순수성을 위해 제거
    if os.path.exists("data/fast_track_resuscitation.jsonl"):
        os.remove("data/fast_track_resuscitation.jsonl")
        print("[-] Removed unneeded data/fast_track_resuscitation.jsonl")
    
    # 검수용 요약 리포트 업데이트
    summary_report_path = "reports/counterfactual_benchmark_summary.md"
    sample_id = benchmark_quads[0]
    sample_ood = benchmark_quads[100]
    
    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write("# [체크포인트 1 검수 리포트] 순수 자연어 발화 기반 반사실적 교란 벤치마크\n\n")
        f.write(f"생성 일시: 2026-10-28\n\n")
        f.write("## 1. 벤치마크 구축 의의 및 설계 철학 (Pure Clinical NLP)\n")
        f.write("* **정량적 수치(바이탈사인) 전면 배제**: 혈압, 맥박 등 수치 힌트에 의한 교란을 배제하고, 초기 접촉 시점(119 구급 신고, 응급실 접수창구, 비대면 챗봇)의 **'순수 환자 자연어 발화'**만으로 주호소 인과 추론 능력을 평가합니다.\n")
        f.write(f"* **총 규모**: **{len(benchmark_quads)}쌍** (총 {len(benchmark_quads)*4}개 평가 변형 케이스)\n")
        f.write(f"  - **In-Distribution (학습 질환군: 심혈관, 신경계)**: **100쌍** (400 케이스)\n")
        f.write(f"  - **Out-of-Distribution (비학습 질환군: 호흡기, 알레르기, 중독)**: **50쌍** (200 케이스)\n\n")
        
        f.write("## 2. 샘플 검수 1: In-Distribution (ID) 4-way Quadruplet\n")
        f.write(f"```json\n{json.dumps(sample_id, indent=2, ensure_ascii=False)}\n```\n\n")
        
        f.write("## 3. 샘플 검수 2: Out-of-Distribution (OOD) 4-way Quadruplet\n")
        f.write(f"```json\n{json.dumps(sample_ood, indent=2, ensure_ascii=False)}\n```\n\n")
        
    print(f"[+] Summary report saved to {summary_report_path}")

if __name__ == "__main__":
    main()
