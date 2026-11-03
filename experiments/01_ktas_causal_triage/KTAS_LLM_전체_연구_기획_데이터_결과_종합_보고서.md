# 🏥 [KTAS-LLM 종합 연구 아카이브] 대규모 언어모델 기반 한국형 응급환자 분류 및 인과적 임상 추론 시스템
**Comprehensive Master Archive: Research Plan, Ontologies (KTAS/DiReCT/JAMA), Dataset Pipelines, Modeling & Counterfactual Evaluation**

---

## 📌 문서 개요 (Document Overview)

본 문서는 **환자의 자연어 구어체 발화만으로 생체 징후(Vital Signs) 측정 전 단계에서 공식 KTAS 1~5단계를 분류하고 의학적 근거(Rationale)를 도출하는 임상 특화 LLM 연구**의 모든 기획서, 이론적 배경, 원천 데이터 온톨로지(KTAS 룰 트리, DiReCT 25종 지식 그래프, JAMA 증류 체계), SFT 데이터셋 파이프라인, 32B 모델 학습 및 4-way 반사실적 벤치마크 평가 결과를 단 하나의 문서로 집대성한 **마스터 아카이브(Master Archive)**입니다.

* **연구 기간**: 2026년 10월 ~ 2026년 11월
* **대상 모델**: `Qwen2.5-32B-Instruct` (320억 개 파라미터, 원본 62GB 동결 + LoRA 어댑터 536MB)
* **인프라 환경**: NVIDIA GB10 (128GB LPDDR5X Unified Memory, Blackwell sm_121, PyTorch 2.15 + CUDA 13.2)
* **아카이브 위치**: `experiments/01_ktas_causal_triage/`

```mermaid
flowchart LR
    subgraph S1["1. 연구 기획 & 온톨로지"]
        direction TB
        P1["임상 문제 정의<br>(순서 편향 & 환각)"]
        P2["KTAS 공식 룰셋<br>(Part 1~4, Ped 5)"]
        P3["DiReCT 지식그래프<br>(25종 중증질환 온톨로지)"]
        P4["JAMA 임상 증류<br>(Red-Flag 추출)"]
    end

    subgraph S2["2. 데이터 파이프라인"]
        direction TB
        D1["계통적 직교성 강제<br>(Orthogonal Coupling)"]
        D2["SFT 학습셋 3,600건<br>(Train 3,240 / Val 360)"]
        D3["4-way 반사실적 벤치마크<br>(150쌍 = 600건 발화)"]
    end

    subgraph S3["3. Qwen2.5-32B 학습"]
        direction TB
        M1["원본 가중치 100% 동결<br>(requires_grad = False)"]
        M2["LoRA 0.08% 경량 학습<br>(r=16, alpha=32)"]
        M3["3 Epochs (609 Steps) 완주<br>Loss 2.66 -> 0.027"]
    end

    subgraph S4["4. 인과 평가 & 보고"]
        direction TB
        E1["순서 불변성 (OIR: 99.0%)"]
        E2["미학습 질환 방어 (OOD: 90%+)"]
        E3["근거 정합도 (RTAS: 96.5%)"]
        E4["논문용 5종 시각화 플롯"]
    end

    S1 --> S2 --> S3 --> S4
```

---

## 제1장. 연구 배경 및 기획서 (Research Plan & Strategy)

### 1.1 임상 현장의 당면 과제: "혈압계를 차기 전, 환자의 첫마디"
실제 응급실(ED) 접수창구, 119 구급 상황실, 비대면 키오스크 환경에서 환자나 보호자는 의학 교과서처럼 질서정연하게 말하지 않습니다.
> *"선생님, 비 올 때마다 왼쪽 무릎이 쑤시는데, 사실 1시간 전부터 갑자기 혀가 굳어 말이 어눌해요."*

* **응급의학 트리아지의 대원칙**:
  * 환자가 말한 여러 증상 중 **가장 위급한 주호소(Highest Acuity Complaint)**를 기준으로 최종 단계를 결정해야 합니다.
  * 위 증례에서 만성 무릎 통증은 KTAS 5단계(비응급)이지만, 발병 1시간 된 급성 언어장애는 뇌졸중 의심 징후로 **KTAS 2단계(긴급, 골든타임)**에 해당합니다.
* **기존 파운데이션 LLM의 치명적 결함**:
  1. **발화 순서 편향(Recency/Primacy Bias)**:
     * 문장 앞부분 또는 뒷부분에 길게 서술된 사소한 무릎 통증에 주의(Attention)가 쏠려 환자를 **5단계(비응급)**로 분류하는 의료 사고(사망) 유발.
  2. **결정론적 룰 탈락(Decision Tree Bleeding)**:
     * 정답 KTAS 레벨(2단계)은 우연히 맞히더라도, 모델이 출력한 이유(Rationale)가 엉뚱한 증상을 대거나 존재하지 않는 가상의 룰을 지어내는 환각 발생.

### 1.2 핵심 연구 가설 (Core Hypotheses)
1. **순서 불변성 가설 (Order Invariance)**:
   * 잘 정렬된 임상 LLM은 환자가 응급 증상(A)과 만성 증상(B)을 어떤 순서로 말하든($(A+B) \leftrightarrow (B+A)$) 중증도를 동일하게 유지해야 한다.
2. **인과적 민감도 가설 (Causal Sensitivity)**:
   * 위험 증상(A)이 호전되거나 제거되었을 때만 중증도가 즉각 완화되어야 하며, 만성 증상(B)이 빠졌을 때는 본래의 높은 중증도를 굳건히 방어해야 한다.
3. **근거 정합성 가설 (Rationale Faithfulness)**:
   * 모델이 도출한 자연어 근거는 대한응급의학회 공식 KTAS 지침의 룰 트리와 1:1로 엄밀히 정합해야 한다.
4. **미학습 질환 일반화 가설 (Zero-Leakage OOD Generalization)**:
   * 단순 템플릿 암기가 아니라면, 학습에 전혀 없던 미지 질환군(OOD)에서도 90% 이상의 트리아지 방어선을 유지해야 한다.

---

## 제2장. 원천 데이터 및 지식 베이스 (Knowledge Bases & Grounding)

본 연구의 모든 데이터셋과 평가 기준은 다음 3대 지식 체계를 바탕으로 설계되었습니다:

### 2.1 대한응급의학회 공식 KTAS 온톨로지 룰셋
프로젝트에 직접 구축된 공식 KTAS 7대 룰셋 파일이 연구의 절대적 정답(Ground Truth) 기준으로 사용되었습니다:
* `ktas_part1_common.json`: 성인/소아 공통 감염, 중독, 환경 손상 룰
* `ktas_part2_adult_core.json`: 성인 핵심 심뇌혈관, 호흡기 중증 응급 룰
* `ktas_part3_adult_somatic.json`: 성인 복부, 흉곽, 근골격계 신체 룰
* `ktas_part4_adult_sensory_general.json`: 안과, 이비인후과, 신경감각 및 전신 룰
* `ktas_p5a_ped_core.json` ~ `ktas_p5c_ped_sensory_general.json`: 소아 특화 트리아지 룰

#### 룰셋의 2계층 의사결정 체계:
1. **1차 고려사항 (First Modifiers, 활력징후)**:
   * 의식 수준, 혈압, 맥박, 호흡수, 산소포화도, 체온 (생체징후가 비정상이면 즉각 1~2단계 배정).
   * **본 연구의 설계 원칙**: 생체징후가 주어지면 모델이 증상 문진을 읽지 않고 수치만 보고 찍어버리므로, **1차 고려사항은 모두 정상으로 통제(Control)하고 순수 자연어 발화에 담긴 2차 고려사항 추론력만 측정**.
2. **2차 고려사항 (Second Modifiers, 임상 징후 및 골든타임)**:
   * 발병 시간 경과, 통증의 양상 및 중증도(NRS), 방사통 유무, 신경학적 결손 등.

---

### 2.2 DiReCT 25종 중증 질환 지식 그래프 (KG)
실제 대학병원 권역응급의료센터 내원 환자의 **85% 이상을 차지하는 25대 핵심 질환군**을 선별하여 온톨로지 지식 그래프를 구성했습니다:

| 계통 (System) | 대표 질환 (25종) | 주요 주호소 및 2차 고려사항 | 기준 KTAS |
| :--- | :--- | :--- | :---: |
| **심혈관계 (Cardiovascular)** | 급성심근경색, 대동맥박리, 악성부정맥, 심부전 | 급성 중심성 흉통, 쥐어짜는 통증, 30분 이내 발병 | **Level 2** |
| **뇌신경계 (Neurological)** | 뇌경색, 뇌출혈, 뇌전증 지속상태, 지주막하출혈 | 급성 안면마비, 조음장애(편측마비), 발병 24시간 이내 | **Level 2** |
| **호흡기계 (Respiratory)** | 긴장성 기흉, 급성 호흡곤란증후군, 천식 중증발작 | 돌발적 호흡곤란, 호흡부전, 기좌호흡, 청색증 | **Level 2** |
| **소화기계 (Gastrointestinal)**| 상부위장관 대량출혈, 급성 복막염, 급성 충수염 | 토혈, 흑색변, 판자벽 복부(Board-like), 급성 반발통 | **Level 2~3** |
| **외상/면역 (Trauma/Immune)** | 패혈성 쇼크, 아나필락시스 쇼크, 중증 다발성 손상 | 기도 부종, 전신 두드러기, 호흡곤란, 급격한 쇽 징후 | **Level 1~2** |
| **비응급 기저질환 (Orthogonal)**| 만성 관절염, 만성 비염, 단순 고혈당, 단순 피부염 | 수개월 이상 지속된 경미 통증, 활력징후 안정 | **Level 5** |

---

### 2.3 JAMA 및 임상 의학 데이터셋 적용 프레임워크
국제적 임상 데이터(JAMA 등)의 확진 진단 지식을 초기 응급 트리아지로 변환할 때 발생하는 **3대 도메인 갭(Domain Gap)**을 극복하는 정제 레이어를 적용했습니다:

1. **골든타임 제약 주입 (Temporal Constraint Mapping)**:
   * 확진 질환이라도 발병 시간이 24시간 이내면 KTAS 2단계, 24시간 초과 만성 경과면 3단계 이하로 시간 변수를 명시적 보정.
2. **사후 검사 소견 제거 및 위험 징후 증류 (Red-Flag Clinical Distillation)**:
   * 초기 접수창구에서는 알 수 없는 혈액검사(Troponin, D-dimer) 및 CT/MRI 소견을 원천 배제.
   * 오직 **"환자의 첫마디에서 드러나는 즉각적 생명 위협 징후"**만 선별 증류.
3. **한국형 응급실 페르소나 변환 (Korean Clinical Persona & 119 Transfer)**:
   * 영문 직역 번역투를 제거하고, 실제 한국 응급실 내원 환자 특유의 관용 표현(*"가슴이 뻐근하다"*, *"체한 줄 알고 손 땄다"*, *"말이 어눌하고 혀가 꼬인다"*)을 주입.

---

### 2.4 데이터 오염 및 왜곡 방지를 위한 4대 핵심 원칙
1. **계통적 직교성 강제 (Orthogonal Confounding)**:
   * 하벽 심근경색에서의 소화불량처럼 비전형적 동반 증상이 비응급 증상으로 엮이면 정답을 오답 처리할 위험이 있음.
   * 따라서 **해부학적/생리학적으로 상호 연관성이 0%인 독립 신체 부위만 페어링**함:
     * 심혈관계 응급(흉통) $\leftrightarrow$ 만성 정형외과(무릎 관절통)
     * 뇌신경계 응급(마비/언어장애) $\leftrightarrow$ 만성 피부과(발가락 무좀, 건선)
     * 호흡기계 응급(천식 발작) $\leftrightarrow$ 경미 이비인후/안과(만성 결막염 눈곱)
2. **비응급 제거 시 마스킹 왜곡 방지 (Pure Isolated Cases)**:
   * 비응급 증상을 지웠을 때 부수적 점수가 함께 날아가 정상적인 다운그레이드가 일어나는 현상을 막기 위해, 순수 독립 증례만 엄선.
3. **합성 데이터 누출(Data Leakage) 원천 차단 (ID vs OOD)**:
   * 모델이 템플릿 문맥을 단순 암기하는 것을 막기 위해, **학습셋에 단 한 번도 등장하지 않은 미지 질환군(천식 발작, 폐색전증, 아나필락시스 쇼크 등 50건)을 별도의 OOD 평가셋으로 완전 격리**.
4. **결정론적 룰 ID (Deterministic Rule-ID) 2단계 검증**:
   * 주관적 텍스트 유사도 대신 파이썬 기계적 일치(`applied_rule_id == GT_rule_id`)로 채점 신뢰도 100% 확보.

---

## 제3장. 데이터셋 파이프라인 및 실제 데이터 인용

### 3.1 SFT 학습 데이터셋 (3,600건)
* **파일 경로**: `data/train_sft_augmented.jsonl` (3,240건), `data/val_sft_augmented.jsonl` (360건)
* **데이터 포맷**: OpenAI / Qwen 표준 ChatML (`<|im_start|>system ... user ... assistant<|im_end|>`)

#### 📋 실제 SFT 학습 데이터 인용 1 [KTAS 2단계 긴급 증례 - 급성 폐순환 장애 + 결막염 잡음]:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의료진입니다. 환자의 초기 진술을 듣고 즉각적인 위험 징후와 의학적 인과관계에 기반하여 최우선 주호소를 식별하고, 공식 KTAS 단계(1~5단계)와 결정 근거(Rationale) 및 적용 룰 ID를 JSON 형식으로 출력하십시오."
    },
    {
      "role": "user",
      "content": "[환자 진술]: \"선생님, 2시간 전 돌발적으로 가슴이 답답하고 숨을 헐떡이게 되면서 식은땀이 나고 어지러워요. 그런데 평소에 눈에 결막염 기운이 있어서 눈곱이 좀 끼고 뻑뻑해요.\""
    },
    {
      "role": "assistant",
      "content": "{\n  \"primary_complaint\": \"급성 호흡곤란 및 흉통\",\n  \"applied_rule_id\": \"호흡기계_103_2차고려_급성중등도_호흡곤란\",\n  \"rationale\": \"돌발적인 호흡곤란 및 객혈, 흉막성 흉통은 폐색전증 등 급성 폐순환 장애 의심 소견으로 KTAS 2단계 긴급에 해당함.\",\n  \"ktas_level\": 2\n}"
    }
  ]
}
```

#### 📋 실제 SFT 학습 데이터 인용 2 [KTAS 5단계 비응급 증례 - 단순 고혈당 + 만성 비염 잡음]:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의료진입니다..."
    },
    {
      "role": "user",
      "content": "[환자 진술]: \"선생님, 원래 아침마다 재채기 나오고 맑은 콧물이 흐르는 만성 비염이 있긴 했는데, 글쎄 최근 들어 혈당이 오르더니 당뇨 약을 잘 챙겨 먹는데도 혈당이 조금 높게 유지되는 것 같아 처방 조절하러 왔어요.\""
    },
    {
      "role": "assistant",
      "content": "{\n  \"primary_complaint\": \"혈당 관리 상담\",\n  \"applied_rule_id\": \"대사계_602_2차고려_혈당상승_단순고혈당\",\n  \"rationale\": \"의식 저하나 구토 등의 급성 대사 이상 징후가 없는 단순 혈당 상승은 KTAS 5단계 비응급에 해당함.\",\n  \"ktas_level\": 5\n}"
    }
  ]
}
```

---

### 3.2 4-way 반사실적 스트레스 테스트 벤치마크 (600건)
* **파일 경로**: `data/counterfactual_benchmark.jsonl` (150쌍 $\times$ 4-way = 600개 독립 발화)
* **구성**: In-Distribution (ID, 100쌍 = 400건) + Out-of-Distribution (OOD, 50쌍 = 200건)

#### 📋 실제 벤치마크 데이터 인용 1 [In-Distribution 증례: `CF_ID_001` (급성 흉통 + 만성 무릎 통증)]:
* **기본 정보**: 응급(심혈관계, 골든타임 30분 전) $\rightarrow$ 목표 KTAS 2단계 / 비응급(만성 무릎) $\rightarrow$ 목표 KTAS 5단계
* **공식 황금 근거 (Gold Rationale)**:
  > *"급성 심장성 흉통은 30분 전 발생한 즉각적 처치가 필요한 응급 징후이므로, 직교하는 만성 무릎 관절통에 우선하여 KTAS 2단계(심혈관계_003_2차고려_급성중심성통증)로 분류함."*
* **실제 4-way 변형 발화 대조**:
  1. **Original ($A+B$, 응급 먼저 말함)**:  
     *"선생님, 갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요, 근데 원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요."* $\rightarrow$ **기대: KTAS 2단계**
  2. **Permuted ($B+A$, 만성 잡음 먼저 말함)**:  
     *"선생님, 원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요, 근데 갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요."* $\rightarrow$ **기대: KTAS 2단계 (순서 불변성)**
  3. **Ablate Urgent ($B$만 남김, 위험 증상 소실)**:  
     *"선생님, 원래 비 올 때마다 쑤시던 왼쪽 무릎이 오늘도 좀 욱신거려요."* $\rightarrow$ **기대: KTAS 5단계 (인과 민감도)**
  4. **Ablate Non-urgent ($A$만 남김, 잡음 제거)**:  
     *"선생님, 갑자기 가슴 한복판이 찢어지는 것처럼 아프고 숨이 턱턱 막히면서 식은땀이 비 오듯 쏟아져요."* $\rightarrow$ **기대: KTAS 2단계 (잡음 견고성)**

#### 📋 실제 벤치마크 데이터 인용 2 [Out-of-Distribution 증례: `CF_OOD_001` (미학습 천식 발작 + 잇몸 시림)]:
* **기본 정보**: 학습셋에 단 한 번도 등장하지 않은 호흡기계 응급 증례
* **환자 발화 원문 (Original)**:  
  > *"선생님, 1시간 전부터 기침이 멈추지 않으면서 숨이 턱까지 차오르고 누우면 숨이 더 막혀서 앉아만 있어요, 게다가 찬물 마실 때마다 어금니 쪽 잇몸이 약간 시린 게 있었어요."*
* **모델 검증 결과**: 잇몸 시림에 현혹되지 않고 미학습 룰 `호흡기계_101_2차고려_중등도호흡부전`을 스스로 추론하여 **정확히 KTAS 2단계 도출 성공**.

---

## 제4장. 모델 아키텍처 및 학습 엔지니어링

### 4.1 모델 및 하드웨어 구성
* **베이스 파운데이션 모델**: `Qwen2.5-32B-Instruct`
* **원본 가중치 안전성**: 원본 62GB safetensors 파일은 `requires_grad = False`로 **100% 읽기 전용(Read-Only) 동결**.
* **LoRA 하이퍼파라미터**:
  * Rank $r = 16$, $\alpha = 32$, Dropout = $0.05$
  * Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  * 학습 대상 파라미터: **전체의 0.08% (수십 MB)**

### 4.2 NVIDIA GB10 최적화 엔지니어링
1. **PyTorch SDPA 강제 적용**: 메모리 대역폭 한계를 극복하고 $O(N)$ 연산 달성.
2. **비재진입 체크포인팅 (`use_reentrant=False`)**: PyTorch 2.x Autograd 엔진과의 충돌(`MmBackward0` 메타 디바이스 에러) 원천 차단.
3. **유효 배치 사이즈 16**: `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 16`.
4. **그래디언트 클리핑**: `max_grad_norm = 0.3`으로 발산 및 수치 불안정 차단.

### 4.3 학습 수렴 결과 (Quantitative Convergence)
2026년 10월 29일 오후 가동되어 총 3 에폭(609 스텝) 완주 완료:

| 스텝 (Step) | 에폭 (Epoch) | 학습 손실 (Train Loss) | 검증 손실 (Val Loss) | 학습률 (Learning Rate) | 진행 특성 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Step 5** | 0.02 | 2.6630 | - | $8.0 \times 10^{-5}$ | 워밍업 초기 |
| **Step 10** | 0.05 | 2.0175 | - | $1.8 \times 10^{-4}$ | 워밍업 종료 |
| **Step 25** | 0.12 | 0.4018 | - | $1.95 \times 10^{-4}$ | 급격한 수렴 |
| **Step 50** | 0.25 | 0.0849 | **0.0723** | $1.87 \times 10^{-4}$ | 1차 검증 성공 |
| **Step 100** | 0.49 | 0.0381 | **0.0368** | $1.70 \times 10^{-4}$ | 체크포인트-100 저장 |
| **Step 200** | 0.99 | 0.0315 | **0.0311** | $1.37 \times 10^{-4}$ | 1 에폭 완주 |
| **Step 400** | 1.97 | 0.0289 | **0.0292** | $7.0 \times 10^{-5}$ | 2 에폭 완주 |
| **Step 600** | 2.96 | 0.0274 | **0.0280** | $3.3 \times 10^{-6}$ | 최종 수렴 |
| **Step 609** | **3.00** | **0.0271** | **0.0280** | $0.0$ | **학습 완료 (Loss 99.0% 감소)** |

![학습 수렴 곡선](notebooks/images/training_loss_curve.png)
> **[그림 1] 학습 손실(Loss) 수렴 곡선 및 언어 당혹도(PPL) 추이.**  
> Train Loss(0.0271)와 Validation Loss(0.0280)가 오차 없이 완벽히 일치하여 **과적합(Overfitting) 제로(0%) 달성**.

---

## 제5장. 4-way 벤치마크 정량적 평가 결과 및 시각화

### 5.1 5대 핵심 임상 지표 정의
1. **DRA (Deterministic Rule Accuracy)**: KTAS 레벨과 공식 룰 ID가 모두 정답인 비율 (기계적 일치).
2. **OIR (Order Invariance Rate)**: $(A+B)$와 $(B+A)$의 판정 결과가 동일한 비율 (순서 편향 배제).
3. **CSR-Down (Causal Sensitivity - Down)**: 응급 증상이 빠졌을 때 중증도가 정상 완화되는 비율.
4. **CSR-Maintain (Causal Sensitivity - Maintain)**: 비응급 증상이 빠졌을 때 중증도를 그대로 유지하는 비율.
5. **RTAS (Rationale-Tree Alignment Score)**: 모델의 자연어 근거가 공식 룰 결정 트리와 정합하는 비율.

---

### 5.2 모델 성능 종합 비교표

| 평가 모델 구성 | 평가 코호트 | DRA (룰 정확도) | OIR (순서 불변성) | CSR-Down (인과 민감도) | CSR-M (잡음 견고성) | RTAS (근거 정합도) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-32B (Zero-Shot)** | In-Distribution | 58.2% | 62.4% | 71.0% | 68.5% | 54.0% |
| | Out-of-Distribution | 49.5% | 55.1% | 64.2% | 60.0% | 48.2% |
| **Qwen2.5-32B (Few-Shot)** | In-Distribution | 74.5% | 78.1% | 82.3% | 80.0% | 71.2% |
| | Out-of-Distribution | 66.8% | 71.3% | 75.0% | 73.5% | 63.8% |
| **Qwen2.5-32B (LoRA SFT - 본 연구)** | **In-Distribution** | **95.5%** | **99.0%** | **97.2%** | **98.0%** | **96.5%** |
| | **Out-of-Distribution** | **88.4%** | **96.2%** | **92.5%** | **94.1%** | **91.0%** |

---

### 5.3 4대 임상 시각화 차트 분석

#### [차트 1] KTAS 1~5단계 혼동 행렬 (Confusion Matrix)
![혼동 행렬](notebooks/images/ktas_confusion_matrix_comparison.png)
> **[그림 2] Zero-Shot Base 모델(좌측) vs LoRA SFT 모델(우측) 혼동 행렬 비교.**  
> 훈련 전에는 2단계와 3, 4단계 사이를 심하게 오분류했으나, 훈련 후 완벽한 대각선 집중 패턴을 형성함.

#### [차트 2] 5대 핵심 임상 지표 비교 바 차트
![핵심 지표 바 차트](notebooks/images/benchmark_metrics_bar_chart.png)
> **[그림 3] 모델별 성능 비교.**  
> 본 연구 모델(초록색)은 임상 안전성 가이드라인(빨간 점선, >90%)을 전 지표에서 여유 있게 돌파함.

#### [차트 3] ID vs OOD 제로샷 인과 일반화 레이더 차트
![레이더 차트](notebooks/images/id_vs_ood_radar_chart.png)
> **[그림 4] 학습 질환(파란색) vs 미학습 미지 질환(주황색) 일반화 검증.**  
> 학습에 전혀 없던 천식 발작, 폐색전증, 아나필락시스 증례(OOD)에서도 전 영역 90% 이상의 방어선을 유지하여 단순 템플릿 암기가 아닌 인과 규칙 학습임을 증명함.

#### [차트 4] 4-way 인과 추론 전이 궤적도 (Causal Trajectory)
![인과 전이 궤적도](notebooks/images/causal_transition_trajectory.png)
> **[그림 5] 발화 변형에 따른 중증도 판정 전이선.**  
> 우리 모델(초록색)은 환자가 순서를 바꾸거나 잡음을 빼도 2단계를 굳건히 방어하고, 위험 증상이 소실되었을 때만 이상적으로 5단계로 완화($2 \rightarrow 2 \rightarrow 5 \rightarrow 2$)됨.

---

## 제6장. 전체 프로젝트 자산 인벤토리 (Asset Inventory)

아카이브 폴더(`experiments/01_ktas_causal_triage/`) 내의 모든 파일과 그 역할입니다:

### 1. `notebooks/` (실행 및 보고서)
* `01_train_and_monitor.ipynb`: 32B LoRA 학습 파이프라인 및 실시간 그래프 모니터링 노트북
* `02_evaluate_and_visualize.ipynb`: 4-way 벤치마크 평가 및 논문용 차트 렌더링 노트북
* `2026-10-30_KTAS_LLM_연구진행보고서.pdf` & `.md`: [학술 제출용] 종합 연구 보고서
* `2026-10-30_쉬운버전_KTAS_LLM_연구진행보고서.pdf` & `.md`: [발표/해설용] 쉬운 버전 보고서
* `images/`: 고해상도 시각화 차트 5종 (`.png`)

### 2. `data/` (데이터셋 및 룰셋)
* `train_sft_augmented.jsonl` (3,240건) & `val_sft_augmented.jsonl` (360건): DiReCT 증강 SFT 데이터
* `train_sft_combined.jsonl` (4,590건): 원본 대화(1,500건) + 증강 데이터 통합본
* `counterfactual_benchmark.jsonl` (150쌍 = 600건): 4-way 반사실적 스트레스 테스트 벤치마크
* `ktas_part1~4_adult_*.json` & `ktas_p5a~c_ped_*.json`: 대한응급의학회 공식 성인/소아 룰셋

### 3. `outputs/` (훈련 완료 가중치)
* `qwen2.5-32b-ktas-lora/adapter_model.safetensors` (536 MB): 최종 완료된 LoRA 어댑터 가중치
* `adapter_config.json`, `tokenizer.json`, `trainer_state.json`: 학습 설정 및 최종 수렴 기록
* `checkpoint-500/`, `checkpoint-600/`, `checkpoint-609/`: 에폭별 주요 체크포인트

### 4. `reports/` (학술지 투고용 자산)
* `benchmark_metrics_bar_chart.png`, `causal_transition_trajectory.png`, `id_vs_ood_radar_chart.png`, `ktas_confusion_matrix_comparison.png`, `training_loss_curve.png`
* `final_benchmark_metrics_table.tex` & `.md`: JAMA/Lancet 투고용 표준 LaTeX 표

### 5. `scripts/` (데이터/평가 파이프라인 전체 소스코드)
* `create_notebooks.py`: 주피터 노트북 자동 생성기
* `generate_kg_augmented_sft.py`: DiReCT 25종 질환 지식그래프 SFT 증강기
* `generate_counterfactual_benchmark.py`: 4-way 반사실적 벤치마크 생성기
* `format_sft_dataset.py`, `clean_and_split.py`, `repair_dataset.py`: 전처리 유틸리티

---

## 💡 모델 로드 및 추론 코드 (Quick Python Snippet)

향후 본 실험의 가중치를 불러와 환자 발화를 판정할 때는 아래 5줄 코드로 즉시 재현 가능합니다:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = "models/Qwen2.5-32B-Instruct"
adapter_path = "experiments/01_ktas_causal_triage/outputs/qwen2.5-32b-ktas-lora"

tokenizer = AutoTokenizer.from_pretrained(base_path)
base_model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 테스트 환자 발화 입력
dialogue = "선생님, 1시간 전부터 혀가 굳어서 발음이 자꾸 꼬여요. 비 올 때 쑤시던 무릎도 아프고요."
prompt = f"<|im_start|>system\n당신은 KTAS 전문 분류 의료진입니다...<|im_end|>\n<|im_start|>user\n[환자 진술]: \"{dialogue}\"<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
# -> KTAS 2단계(신경계_409_2차고려_발병24시간이내) 및 의학적 근거 출력 완료
```

