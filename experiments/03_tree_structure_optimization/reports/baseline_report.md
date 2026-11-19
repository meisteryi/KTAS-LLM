# KTAS 원본 결정 트리 구조 진단 리포트 (Baseline Report)

- **분석 대상 파일 (7개)**:
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_p5a_ped_core.json`
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_p5b_ped_somatic.json`
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_p5c_ped_sensory_general.json`
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_part1_common.json`
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_part2_adult_core.json`
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_part3_adult_somatic.json`
  - `/home/meisteryi/문서/KTAS-LLM/experiments/03_tree_structure_optimization/data/raw/ktas_part4_adult_sensory_general.json`

## 1. 스키마 및 조건식 형태 점검
- **감지된 스키마 유형**: `{'ktas_nested_module': 6, 'ktas_common_framework': 1}`
- **전체 조건식 수**: 1582개 (고유 조건식: 730개)

| 조건식 유형 | 건수 | 비율 (%) | 비고 |
| :--- | ---: | ---: | :--- |
| `cross_reference` | 715 | 45.2% | - |
| `unstructured_text` | 467 | 29.5% | - |
| `numeric_range` | 211 | 13.3% | - |
| `categorical_short` | 189 | 11.9% | - |

## 2. 트리 규모 및 중복도 분석
- **전체 노드 수**: 1958개 (내부 분기 노드: 376개, 리프 경로: 1582개)
- **조건 문자열 중복률**: **53.86%** (1582건 중 852건 중복)
- **서브트리(조건 블록) 구조 중복률**: **3.29%** (고유 서브트리 323 / 전체 334)

### 상위 반복 조건식 (Top Repeated Conditions)
| 순위 | 반복 횟수 | 조건식 내용 |
| :---: | ---: | :--- |
| 1 | 72 | 소아 1차 활력징후 1단계 만족 |
| 2 | 69 | 1차 활력징후 1단계 만족 |
| 3 | 40 | 소아 1차 활력징후 2단계 만족 |
| 4 | 36 | 1차 활력징후 2단계 만족 |
| 5 | 19 | 소아 1차 활력징후 3단계 만족 |
| 6 | 18 | 1차 활력징후 2단계 또는 급성 중심성 중증 통증 (8-10) |
| 7 | 16 | 1차 활력징후 3단계 또는 급성 말초성 중증 통증 (8-10) |
| 8 | 16 | 1차 활력징후 4단계 또는 말초성 중등도 통증 (4-7) |
| 9 | 15 | 소아 1차 활력징후 2단계 또는 급성 중증 통증 (8-10) |
| 10 | 15 | 급성 말초성 중증 통증 (8-10) |

## 3. 탐색 깊이 베이스라인 (Depth Baseline)
- **구조적 트리 깊이(Tree Depth)**: 평균 `3.98` / 최대 `4`
- **순차 평가 깊이(Sequential Rule Depth)**: 평균 `5.42` / 최대 `23`

| KTAS 등급 | 리프 경로 수 | 평균 순차 도달 깊이 | 최소 깊이 | 최대 깊이 |
| :---: | ---: | ---: | ---: | ---: |
| 1등급 | 208 | 3.17 | 2 | 8 |
| 2등급 | 484 | 4.58 | 2 | 16 |
| 3등급 | 415 | 6.01 | 3 | 22 |
| 4등급 | 307 | 6.67 | 3 | 23 |
| 5등급 | 168 | 6.88 | 3 | 13 |

- **등급 간 탐색 깊이 역전 여부**: 뚜렷한 역전 구간 미관측 (단, 단일 주호소 내부 분기 순서 점검 필요)

## 4. 잠재적 최적화 포인트 및 비판적 요약
1. **공통 서브트리 병합(DAG/BDD) 적합성**: 조건식 중복률 `53.9%`, 서브트리 중복률 `3.3%`로, 다수의 주호소에서 1차 활력징후/공통 조건이 반복 출현하므로 **공통 서브트리 병합(DAG/BDD 변환)**이 메모리 및 노드 수 압축에 매우 효과적입니다.
2. **비정형 자연어 정규화 난이도**: 전체 조건식 중 비정형 텍스트(`unstructured_text`) 비율이 **`29.5%`**입니다. SMT/BDD 기반 100% 논리적 동치성 검증을 수행하려면 먼저 비정형 텍스트 조건을 원자 명제(Atomic Predicate ID)로 정규화하는 매핑 테이블 구축이 필수적입니다.
3. **선후행 의존성 및 Don't-Care(`*`) 보존 주의**: KTAS는 `계통(대분류) -> 주호소 -> 1차/2차 수정인자`의 선후행 의존성을 가지므로, 대분류 확정 전 하위 조건을 끌어올리는 전역 BDD 재정렬은 `Don't-Care(*)` 경로 침해를 유발할 수 있습니다. 따라서 계층 경계 내 국소 재정렬 + 공통 서브트리 DAG 병합 전략을 권장합니다.

### 미파싱/특이 섹션 참고
- `ktas_p5a_ped_core.json:심혈관계/고혈압(no second_order_modifiers)`
