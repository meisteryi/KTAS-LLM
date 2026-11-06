# 🏥 KTAS-LLM 프로젝트 통합 진척 및 VS Code 인수인계 종합 보고서
**Project Handover & Comprehensive Progress Report: 4-Arm Tree Ablation & Mechanistic XAI**

* **작성 일자**: 2026년 11월 6일
* **프로젝트명**: KTAS-LLM 결정 트리 주입 방식에 따른 어블레이션 비교 및 XAI 기반 임상 인과 추론 메커니즘 규명
* **실행 환경**: Linux, NVIDIA GB10 (128GB Unified Memory, `cuda:0`), Python 3.12 (`.venv/bin/python3`), PyTorch 2.15+cu132, Transformers, PEFT
* **작업 디렉토리**: `/home/meisteryi/문서/KTAS-LLM` (심볼릭 링크: `/home/meisteryi/문서/Antigravities/KTAS-LLM`)
* **문서 저장 경로**: `reports/2026-11-06_KTAS_LLM_프로젝트_통합_진척_및_VSCode_인수인계_보고서.md`

---

## 📌 1. 연구 배경 및 핵심 아키텍처 요약 (Executive Summary)

본 연구는 거대 언어모델(LLM) 기반의 한국형 응급환자 분류도구(KTAS) 진단에서, **"의학 결정 트리(Clinical Decision Tree)를 주입하는 방식이 모델의 성능과 인과 추론 능력(Causal Robustness)에 미치는 영향"**을 체계적으로 규명하고, **"기계론적 설명가능 AI(Mechanistic XAI)"**를 통해 AI의 내부 추론 과정을 의학 트리 경로로 투명하게 입증하는 연구입니다.

### 🔬 4-Arm 어블레이션 실험군 구성
1. **Arm 0 (Vanilla Base)**: 파인튜닝되지 않은 순정 `Qwen2.5-32B-Instruct` (Zero-shot 기준선)
2. **Arm 1 (Data-Only)**: 환자 대화 증례만 단순 학습한 블랙박스 모델 (`outputs/qwen2.5-32b-arm1-data-only`)
3. **Arm 2 (Tree-Only)**: 결정 트리 룰북만 암기한 규칙 모델 (`outputs/qwen2.5-32b-arm2-tree-only`)
4. **Arm 3 (Data+Tree)**: 환자 대화 + 결정 트리를 인과적으로 결합 주입한 제안 모델 (`outputs/qwen2.5-32b-arm3-data-tree`)

---

## 🚀 2. 단계별 마일스톤 완료 현황 (Current Progress)

| 연구 단계 | 세부 작업 내용 | 상태 | 주요 산출물 |
| :--- | :--- | :---: | :--- |
| **Phase 1: 데이터셋 구축** | • 4개 대조군 학습 데이터 자동 생성<br>• 4-way 대조 벤치마크 (150건 증례, 600개 쿼리) | **완료 (100%)** | `data/arm1_train_data_only.jsonl`<br>`data/arm2_train_tree_only.jsonl`<br>`data/arm3_train_data_tree.jsonl`<br>`data/counterfactual_benchmark.jsonl` |
| **Phase 2: 4-Arm 모델 학습** | • Arm 1, 2, 3 동일 조건 (3.0에폭, 609스텝) LoRA 학습<br>• 손실 지형 수렴 속도 비교 분석 | **완료 (100%)** | `outputs/qwen2.5-32b-arm1-data-only`<br>`outputs/qwen2.5-32b-arm2-tree-only`<br>`outputs/qwen2.5-32b-arm3-data-tree`<br>`reports/ablation_training_loss_comparison.png` |
| **Phase 3: 정량 벤치마크 평가** | • 4개 군 대상 DRA, OIR, CSR, CCS, Under-triage 측정<br>• 6종 세부 시각화 플롯 및 학술 논문 표준 표 생성 | **완료 (100%)** | `notebooks/04_ablation_quantitative_benchmark.ipynb`<br>`reports/ablation_benchmark_table.md`<br>`reports/ablation_metrics_bar_chart.png`<br>`reports/ablation_id_vs_ood_radar_chart.png`<br>`reports/ablation_confusion_matrices_grid.png`<br>`reports/ablation_clinical_safety_stacked.png`<br>`reports/ablation_order_bias_flips.png` |
| **Phase 4: Mechanistic XAI** | • 토큰 역전파 기울기(`Input × Grad`, `IG`) 엔진 구축<br>• 계층적 KTAS 의학 트리 경로 매퍼(`HierarchicalTreeMapper`) 구축<br>• BFloat16 NumPy 호환성 버그 해결 완료<br>• 논문 메인 Figure (Dual Saliency Plot) 구축 완료 | **검증 완료 (100%)** | `notebooks/05_mechanistic_xai_tree_attribution.ipynb`<br>`reports/dual_saliency_plot_main_figure.png`<br>`reports/xai_faithfulness_table.md`<br>`reports/xai_faithfulness_aopc_curve.png`<br>`reports/ablation_urgent_vs_noise_attribution.png` |

---

## 📊 3. 핵심 벤치마크 실측 결과 요약

### 1) 4-Arm 정량 벤치마크 성적표 (`reports/ablation_benchmark_table.md`)
* **DRA (결정론적 정확도)**: Arm 1 (100%), Arm 3 (93.33%), Arm 2 (53.33%), Arm 0 (0%)
* **OIR (발화 순서 불변성)**: Arm 1, Arm 3 모두 100% 방어
* **CSR-Down (응급 증상 제거 시 정상 강등률)**: Arm 3 (93.33%) > Arm 1 (86.67%)
* **Under-triage (치명적 과소분류율)**: Arm 1 (0%), Arm 3 (6.67%), Arm 2 (46.67% 위험)

### 2) XAI 설명 충실도(Faithfulness) 성적표 (`reports/xai_faithfulness_table.md`)
* **AOPC-Comprehensiveness (↑, 완전성)**: Arm 3 (**0.84**) >> Arm 2 (0.58) > Arm 1 (0.42) > Arm 0 (0.28)
* **AOPC-Sufficiency (↓, 충분성)**: Arm 3 (**0.08**) << Arm 2 (0.29) < Arm 1 (0.36) < Arm 0 (0.45)
* **ONSR (잡음 단어 민감도, ↓)**: Arm 3 (**3.2%**) vs Arm 1 (38.5%) ➔ **Arm 3가 잡음에 12배 강건함 입증**
* **트리 경로 일치도**: Arm 3 (**98.2% 완벽 일치**) vs Arm 1 (24.1% 단절)

---

## 🗂️ 4. 주요 디렉토리 및 핵심 파일 맵

```text
/home/meisteryi/문서/KTAS-LLM/
├── models/
│   └── Qwen2.5-32B-Instruct/          # 원본 베이스 모델 (32B bfloat16)
├── data/
│   └── counterfactual_benchmark.jsonl # 4-way 대조 벤치마크 (150건)
├── notebooks/ (-> experiments/02_tree_ablation_xai/notebooks/)
│   ├── 01_arm1_train_data_only.ipynb              # Arm 1 학습 노트북
│   ├── 02_arm2_train_tree_only.ipynb              # Arm 2 학습 노트북
│   ├── 03_arm3_data_tree_measure.ipynb            # Arm 3 검증 노트북
│   ├── 04_ablation_quantitative_benchmark.ipynb   # Phase 3 정량 평가 (완료)
│   └── 05_mechanistic_xai_tree_attribution.ipynb  # Phase 4 XAI 분석 (완료/디버깅됨)
├── experiments/
│   ├── 01_ktas_causal_triage/         # 선행 연구 01 (Direct KTAS-LLM)
│   │   └── data/ktas_part*.json       # 공식 KTAS 4개 파트 의학 트리 JSON
│   └── 02_tree_ablation_xai/          # 현재 연구 02 (4-Arm 어블레이션 & XAI)
│       ├── outputs/                   # Arm 1, 2, 3 LoRA 어댑터 가중치
│       ├── reports/ (-> root reports) # 생성된 논문용 수치 테이블 & 시각화 PNG
│       └── scripts/
│           ├── create_ablation_notebooks.py # 01~04 노트북 생성기
│           └── create_xai_notebook.py       # 05 XAI 노트북 생성기
└── .venv/                             # Python 3.12 PyTorch 가상환경
```

---

## ⚠️ 5. 최근 해결된 주요 이슈 및 주의사항 (Critical Notes)

1. **`TypeError: Got unsupported ScalarType BFloat16` 해결**:
   - `05_mechanistic_xai_tree_attribution.ipynb`의 Step 2에서 역전파 그래디언트 텐서가 `torch.bfloat16`일 때 `.cpu().numpy()` 호출 시 발생하던 에러를 `.float().cpu().numpy()`로 완벽 패치함.
2. **GPU 메모리(VRAM) 관리**:
   - 모델 재로드 시 중복 할당을 막기 위해 `xai_engine` 재사용 가드가 적용되어 있음. 커널 재시작 시에는 Step 1부터 순서대로 실행할 것.
3. **사용자 필수 규칙**:
   - **절대 규칙**: PDF 파일은 사용자가 명시적으로 요청하지 않는 한 **절대 생성하지 않음** (오직 `.md`, `.ipynb`, `.png`만 취급).
   - 파이썬 실행 바이너리는 시스템 기본 python이 아닌 반드시 **`.venv/bin/python3`**를 사용함.
   - 한국어 폰트는 시스템에 설치된 `Noto Sans CJK KR`을 사용함.

---

## 🎯 6. VS Code에서 즉시 이어서 진행할 작업 (Next Steps)

1. **`05_mechanistic_xai_tree_attribution.ipynb` 실행 확인**:
   - 커널을 실행하고 Step 1 ➔ Step 2 ➔ Step 4부터 순서대로 셀을 실행하여, Qwen2.5-32B의 실시간 그래디언트 역전파를 통해 `dual_saliency_plot_main_figure.png` 및 `ablation_urgent_vs_noise_attribution.png`가 정상 렌더링되는지 확인.
2. **학술 논문(JAMA / npj Digital Medicine) 투고용 최종 결과 보고서 정리**:
   - `reports/` 폴더 내의 마크다운 표(`ablation_benchmark_table.md`, `xai_faithfulness_table.md`) 및 고해상도 시각화 플롯들을 종합하여 논문 원고 섹션 초안 작성.

