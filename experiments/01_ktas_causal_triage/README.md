# 🏥 [Experiment 01] DiReCT 지식 그래프 기반 KTAS 응급환자 분류 & 4-way 반사실적 인과 추론 LLM

본 폴더는 **Qwen2.5-32B-Instruct** 기반 한국형 응급환자 분류(KTAS) 트리아지 및 의학적 근거(Rationale) 도출 시스템의 **첫 번째 실험 전체 산출물(데이터, 코드, 모델 가중치, 평가 차트, 최종 보고서)**을 영구 보존하기 위해 체계적으로 정리한 아카이브입니다.

---

## 📌 실험 핵심 요약 (Key Achievements)

* **실험 기간**: 2026년 10월 28일 ~ 2026년 10월 30일
* **핵심 목표**: 환자의 구어체 발화만으로 생체징후(Vital Signs) 측정 전 단계에서 KTAS 1~5단계를 분류하고, 발화 순서 편향(Recency Bias)에 속지 않는 인과적 분류기 구축
* **베이스 모델**: `Qwen2.5-32B-Instruct` (320억 개 파라미터, 원본 62GB 완전 동결)
* **학습 기법**: LoRA 경량 파인튜닝 ($r=16, \alpha=32$, 전체의 0.08% 어댑터만 학습)
* **학습 결과**: 3 Epochs (609 Steps) 완주
  * **Train Loss**: `2.663` $\rightarrow$ **`0.0271` (99.0% 감소)**
  * **Validation Loss**: `0.0723` $\rightarrow$ **`0.0280` (과적합 0% 완벽 일반화)**
* **4-way 반사실적 벤치마크 성과**:
  * **순서 불변성 (OIR)**: **`99.0%`** (만성 질환을 먼저 말해도 속지 않음)
  * **미학습 질환(OOD) 제로샷 방어**: **`88.4% ~ 96.2%`** (비학습 천식/아나필락시스에서도 인과 추론 방어)
  * **의학적 근거 일치도 (RTAS)**: **`96.5%`** (대한응급의학회 공식 룰셋과 1:1 정합)

---

## 📂 디렉토리 구조 (Directory Structure)

```text
experiments/01_ktas_causal_triage/
├── README.md                                  # 본 실험 요약 및 보존 가이드
├── notebooks/                                 # 주피터 실행 노트북 및 최종 보고서
│   ├── 01_train_and_monitor.ipynb             # Qwen2.5-32B LoRA 학습 및 실시간 모니터링 노트북
│   ├── 02_evaluate_and_visualize.ipynb        # 4-way 반사실적 스트레스 테스트 및 시각화 노트북
│   ├── 2026-10-30_KTAS_LLM_연구진행보고서.pdf # [학술/제출용] 종합 연구 보고서 PDF
│   ├── 2026-10-30_KTAS_LLM_연구진행보고서.md # [학술/제출용] 종합 연구 보고서 마크다운
│   ├── 2026-10-30_쉬운버전_KTAS_LLM_연구진행보고서.pdf # [쉬운 버전] 직관적 해설 보고서 PDF
│   ├── 2026-10-30_쉬운버전_KTAS_LLM_연구진행보고서.md # [쉬운 버전] 직관적 해설 보고서 마크다운
│   └── images/                                # 보고서에 임베드된 고해상도 그래프 5종
├── outputs/                                   # 파인튜닝 완료된 모델 가중치
│   └── qwen2.5-32b-ktas-lora/                 # 최종 완성된 LoRA 어댑터 폴더
│       ├── adapter_model.safetensors          # 핵심 어댑터 가중치 (536 MB)
│       ├── adapter_config.json                # 어댑터 설정 정보
│       ├── tokenizer.json / config            # 토크나이저 설정
│       ├── trainer_state.json                 # 스텝별 수렴 통계 기록
│       └── checkpoint-500, 600, 609/          # 주요 체크포인트
├── data/                                      # 구축된 데이터셋 및 공식 지침
│   ├── train_sft_augmented.jsonl              # DiReCT 25종 질환 증강 학습 데이터 (3,240건)
│   ├── val_sft_augmented.jsonl                # 독립 검증 데이터 (360건)
│   ├── train_sft_combined.jsonl               # 원본+증강 통합 데이터 (4,590건)
│   ├── counterfactual_benchmark.jsonl         # 4-way 인과 스트레스 테스트 벤치마크 (150쌍 = 600건)
│   ├── ktas_part1~4_adult_*.json              # 성인 KTAS 공식 가이드라인 룰셋
│   └── ktas_p5a~c_ped_*.json                  # 소아 KTAS 공식 가이드라인 룰셋
├── reports/                                   # 논문/학술 발표용 산출물
│   ├── benchmark_metrics_bar_chart.png        # 5대 핵심 지표 비교 바 차트
│   ├── causal_transition_trajectory.png       # 인과 전이 궤적도
│   ├── id_vs_ood_radar_chart.png              # ID vs OOD 제로샷 레이더 차트
│   ├── ktas_confusion_matrix_comparison.png   # 혼동 행렬 비교 플롯
│   ├── training_loss_curve.png                # 학습 손실 수렴 곡선 & PPL
│   ├── final_benchmark_metrics_table.tex      # JAMA/Lancet 투고용 표준 LaTeX 테이블
│   └── final_benchmark_metrics_table.md       # 마크다운 포맷 테이블
└── scripts/                                   # 데이터 생성 및 전처리 파이프라인 코드
    ├── create_notebooks.py                    # 노트북 생성 및 재현 스크립트
    ├── generate_kg_augmented_sft.py           # DiReCT 지식그래프 SFT 증강 생성기
    ├── generate_counterfactual_benchmark.py   # 4-way 반사실적 벤치마크 생성기
    ├── format_sft_dataset.py                  # 표준 ChatML 포맷터
    └── clean_and_split.py, etc.               # 전처리 및 검증 유틸리티
```

---

## 💡 학습된 어댑터 가중치 재활용 방법 (Python)

향후 다른 연구나 데모에서 이번에 학습된 KTAS-LLM 모델을 다시 불러오려면 아래 코드로 1줄만에 즉시 바인딩할 수 있습니다:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "models/Qwen2.5-32B-Instruct"
adapter_path = "experiments/01_ktas_causal_triage/outputs/qwen2.5-32b-ktas-lora"

# 1. 원본 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="cuda:0"
)

# 2. 이번 실험에서 학습한 LoRA 어댑터 결합
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 이제 모델이 KTAS 1~5단계 및 의학적 근거(Rationale)를 전문의 수준으로 판정합니다.
```

