# [체크포인트 1 검수 리포트] 순수 자연어 발화 기반 반사실적 교란 벤치마크

생성 일시: 2026-10-28

## 1. 벤치마크 구축 의의 및 설계 철학 (Pure Clinical NLP)
* **정량적 수치(바이탈사인) 전면 배제**: 혈압, 맥박 등 수치 힌트에 의한 교란을 배제하고, 초기 접촉 시점(119 구급 신고, 응급실 접수창구, 비대면 챗봇)의 **'순수 환자 자연어 발화'**만으로 주호소 인과 추론 능력을 평가합니다.
* **총 규모**: **150쌍** (총 600개 평가 변형 케이스)
  - **In-Distribution (학습 질환군: 심혈관, 신경계)**: **100쌍** (400 케이스)
  - **Out-of-Distribution (비학습 질환군: 호흡기, 알레르기, 중독)**: **50쌍** (200 케이스)

## 2. 샘플 검수 1: In-Distribution (ID) 4-way Quadruplet
```json
{
  "case_id": "CF_ID_001",
  "split": "In-Distribution",
  "system_pairing": {
    "urgent_system": "심혈관계",
    "non_urgent_system": "만성 정형외과",
    "orthogonality_verified": true,
    "is_pure_isolated_case": true
  },
  "golden_time": "30분 전 발생",
  "ground_truth_rule_id": "심혈관계_003_2차고려_급성중심성통증",
  "expected_urgent_ktas": 2,
  "ground_truth_non_urgent_rule_id": "근골격계_만성_경미통증",
  "expected_non_urgent_ktas": 5,
  "gold_rationale": "급성 심장성 흉통은(는) 30분 전 발생 발생한 즉각적 처치가 필요한 응급 징후이므로, 직교하는 만성 만성 무릎 관절통에 우선하여 KTAS 2단계(심혈관계_003_2차고려_급성중심성통증)로 분류함.",
  "permutations": {
    "case_original": {
      "patient_dialogue": "선생님, 갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요, 근데 원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요.",
      "expected_primary_complaint": "급성 심장성 흉통",
      "expected_ktas_level": 2,
      "expected_rule_id": "심혈관계_003_2차고려_급성중심성통증"
    },
    "case_permuted": {
      "patient_dialogue": "선생님, 원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요, 근데 갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요.",
      "expected_primary_complaint": "급성 심장성 흉통",
      "expected_ktas_level": 2,
      "expected_rule_id": "심혈관계_003_2차고려_급성중심성통증"
    },
    "case_ablate_urgent": {
      "patient_dialogue": "선생님, 원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요.",
      "expected_primary_complaint": "만성 무릎 관절통",
      "expected_ktas_level": 5,
      "expected_rule_id": "근골격계_만성_경미통증"
    },
    "case_ablate_non_urgent": {
      "patient_dialogue": "선생님, 갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요.",
      "expected_primary_complaint": "급성 심장성 흉통",
      "expected_ktas_level": 2,
      "expected_rule_id": "심혈관계_003_2차고려_급성중심성통증"
    }
  }
}
```

## 3. 샘플 검수 2: Out-of-Distribution (OOD) 4-way Quadruplet
```json
{
  "case_id": "CF_OOD_001",
  "split": "Out-of-Distribution",
  "system_pairing": {
    "urgent_system": "호흡기계",
    "non_urgent_system": "치과/구강",
    "orthogonality_verified": true,
    "is_pure_isolated_case": true
  },
  "golden_time": "40분 전 발생",
  "ground_truth_rule_id": "호흡기계_101_2차고려_중등도호흡부전",
  "expected_urgent_ktas": 2,
  "ground_truth_non_urgent_rule_id": "치과계_만성_단순치통",
  "expected_non_urgent_ktas": 5,
  "gold_rationale": "급성 중등도 호흡곤란 (천식 발작 의증)은(는) 40분 전 발생 발생한 즉각적 처치가 필요한 응급 징후이므로, 직교하는 만성 만성 잇몸 시림에 우선하여 KTAS 2단계(호흡기계_101_2차고려_중등도호흡부전)로 분류함.",
  "permutations": {
    "case_original": {
      "patient_dialogue": "선생님, 1시간 전부터 기침이 멈추지 않으면서 숨이 턱까지 차오르고 누우면 숨이 더 막혀서 앉아만 있어요, 게다가 찬물 마실 때마다 어금니 쪽 잇몸이 약간 시린 게 있었어요.",
      "expected_primary_complaint": "급성 중등도 호흡곤란 (천식 발작 의증)",
      "expected_ktas_level": 2,
      "expected_rule_id": "호흡기계_101_2차고려_중등도호흡부전"
    },
    "case_permuted": {
      "patient_dialogue": "선생님, 찬물 마실 때마다 어금니 쪽 잇몸이 약간 시린 게 있었어요, 근데 1시간 전부터 기침이 멈추지 않으면서 숨이 턱까지 차오르고 누우면 숨이 더 막혀서 앉아만 있어요.",
      "expected_primary_complaint": "급성 중등도 호흡곤란 (천식 발작 의증)",
      "expected_ktas_level": 2,
      "expected_rule_id": "호흡기계_101_2차고려_중등도호흡부전"
    },
    "case_ablate_urgent": {
      "patient_dialogue": "선생님, 찬물 마실 때마다 어금니 쪽 잇몸이 약간 시린 게 있었어요.",
      "expected_primary_complaint": "만성 잇몸 시림",
      "expected_ktas_level": 5,
      "expected_rule_id": "치과계_만성_단순치통"
    },
    "case_ablate_non_urgent": {
      "patient_dialogue": "선생님, 1시간 전부터 기침이 멈추지 않으면서 숨이 턱까지 차오르고 누우면 숨이 더 막혀서 앉아만 있어요.",
      "expected_primary_complaint": "급성 중등도 호흡곤란 (천식 발작 의증)",
      "expected_ktas_level": 2,
      "expected_rule_id": "호흡기계_101_2차고려_중등도호흡부전"
    }
  }
}
```

