#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3: 비정형 응급실 간호사 초진 기록(Noisy Stress Test) 40건 생성기.

ktas_eval_cases.json에서 중증(1~2등급) 20건, 경증(3~5등급) 20건을 균형 샘플링한 뒤,
규칙 키워드 직접 노출 없이 구어체 표현, 불필요한 배경 문장(Distractor), 간접 임상 묘사를 결합해
data/benchmark/ktas_noisy_cases.json으로 저장한다. (정답 등급 및 추적 라벨 100% 유지)
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[2]

DISTRACTORS = [
    "보호자(배우자)와 함께 도보로 내원하였으며 평소 고혈압 약을 아침에 복용했다고 함.",
    "내원 3일 전부터 기운이 없고 입맛이 떨어졌다고 진술하며, 약물 알레르기는 부인함.",
    "대기실에서 물을 한 모금 마신 뒤 접수하였으며 최근 감기 기운이 있었다고 언급함.",
    "과거 5년 전 타 병원에서 단순 건강검진을 받은 적 있고 최근 해외여행력은 없음.",
    "119 구급차 대신 자가용을 이용해 응급실로 내원하였으며 보호자가 접수를 진행함.",
]

COLLOQUIAL_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:소아\s*)?1차\s*활력징후\s*1\s*단계(?:\s*조건)?(?:\s*만족)?"),
     "의식이 거의 없고 맥박이 매우 약하며 호흡이 가빠져 청색증이 관찰되는 위급한 활력징후 소견"),
    (re.compile(r"(?:소아\s*)?1차\s*활력징후\s*2\s*단계(?:\s*조건)?(?:\s*만족)?"),
     "맥박이 빠르고 혈압이 평소보다 떨어져 있으며 식은땀과 함께 숨이 차고 약간 처지는 양상"),
    (re.compile(r"(?:소아\s*)?1차\s*활력징후\s*3\s*단계(?:\s*조건)?(?:\s*만족)?"),
     "활력징후가 정상 상·하한 경계에 걸려 있거나 미열과 경미한 숨참이 동반되나 의식은 명료함"),
    (re.compile(r"(?:소아\s*)?1차\s*활력징후\s*4\s*단계(?:\s*조건)?(?:\s*만족)?"),
     "미열만 살짝 있으나 전반적으로 활력징후가 안정적이고 대화에 무리가 없음"),
    (re.compile(r"급성\s*(?:중심성|말초성)?\s*중증\s*통증\s*\(8-10\)"),
     "오늘 갑자기 시작된 참기 힘들 정도로 극심한 통증(통증 점수 9점 수준)"),
    (re.compile(r"급성\s*(?:중심성|말초성)?\s*중등도\s*통증\s*\(4-7\)"),
     "최근 발생하여 꽤 뻐근하고 불편한 중등도 통증(통증 점수 5~6점 수준)"),
    (re.compile(r"(?:급성|만성)\s*(?:중심성|말초성)?\s*경증\s*통증\s*\(<4\)"),
     "살짝 욱신거리는 정도의 참을 만한 경미한 통증(통증 점수 2~3점 수준)"),
    (re.compile(r"출혈성\s*질환\s*\(?(?:2단계|위급)\)?"),
     "평소 항응고제(와파린)를 복용 중인 환자로 지혈이 되지 않는 다량의 출혈 소견"),
    (re.compile(r"출혈성\s*질환\s*\(?(?:3단계|경도/중등도)\)?"),
     "혈소판 저하 기저력이 있으며 소량의 지속적인 출혈이 관찰됨"),
    (re.compile(r"고위험(?:성)?\s*사고기전"),
     "고속 주행 중 차량이 전복되면서 안전벨트 미착용 상태로 충격받은 사고 경력"),
    (re.compile(r"SBP\s*>\s*160\s*mmHg\s*또는\s*DBP\s*>\s*100\s*mmHg"),
     "혈압 측정상 수축기 172 mmHg / 이완기 106 mmHg로 매우 높게 측정됨"),
    (re.compile(r"SBP\s*>\s*140\s*mmHg\s*또는\s*DBP\s*>\s*90\s*mmHg"),
     "혈압 측정상 수축기 148 mmHg / 이완기 94 mmHg로 상승 소견"),
    (re.compile(r"진통\s*\(자궁수축\s*<=\s*5분\s*간격\)"),
     "배가 3~4분마다 규칙적으로 뭉치며 진통이 빠르게 진행되는 양상"),
    (re.compile(r"진통\s*\(자궁수축\s*>\s*5분\s*간격\)"),
     "배가 약 10~15분 간격으로 간헐적으로 뭉치는 초기 진통 양상"),
    (re.compile(r"심폐소생술의\s*대상이\s*아닌\s*경우\s*\(DNR\s*등\)"),
     "가족 및 사전연명의료의향서상 심폐소생술을 시행하지 않기로(DNR) 명확히 동의된 상태"),
]


def paraphrase_condition(cond: str) -> str:
    """정형 조건 문자열을 구어체/간접 임상 묘사로 변환한다."""
    out = cond
    for pat, repl in COLLOQUIAL_REPLACEMENTS:
        out = pat.sub(repl, out)
    # 괄호 안의 기계적 메타 표기 완화
    out = out.replace("(", "[").replace(")", "]")
    return out


def build_noisy_presentation(case: dict[str, Any], rng: random.Random) -> str:
    """간호사 초진 기록 스타일의 비정형 노이즈 환자 진술을 생성한다."""
    is_ped = case["context"] == "PED"
    age_phrase = f"{rng.randint(2, 12)}세 환아" if is_ped else f"{rng.randint(24, 75)}세 환자"
    if "임신" in case["complaint_name"] or "출산" in case["complaint_name"]:
        age_phrase = f"{rng.randint(26, 37)}세 여성 환자"

    paraphrased = [paraphrase_condition(c) for c in case["active_conditions"]]
    clinical_desc = "; ".join(paraphrased)
    distractor = rng.choice(DISTRACTORS)

    return (
        f"[응급실 초진 간호기록] {age_phrase}가 '{case['complaint_name']}' 문제로 내원함. "
        f"{distractor} "
        f"초기 사정 및 문진에서 확인된 핵심 소견으로는 {clinical_desc} 등이 보고됨. "
        "추가적인 특이 증상 호소는 없음."
    )


def generate_noisy_dataset(eval_cases_path: Path, n_total: int = 40, seed: int = 2026) -> list[dict[str, Any]]:
    """중증(1~2등급) 20건 + 경증(3~5등급) 20건 총 40건의 노이즈 테스트셋을 생성한다."""
    rng = random.Random(seed)
    with eval_cases_path.open("r", encoding="utf-8") as f:
        all_cases: list[dict[str, Any]] = json.load(f)

    severe = [c for c in all_cases if c["ground_truth_level"] in (1, 2)]
    mild = [c for c in all_cases if c["ground_truth_level"] in (3, 4, 5)]

    n_each = n_total // 2
    rng.shuffle(severe)
    rng.shuffle(mild)
    sampled = severe[:n_each] + mild[: (n_total - n_each)]
    rng.shuffle(sampled)

    noisy_cases: list[dict[str, Any]] = []
    for idx, orig in enumerate(sampled, start=1):
        c = dict(orig)
        c["orig_case_id"] = orig["case_id"]
        c["case_id"] = f"NOISY_{idx:03d}"
        c["original_presentation"] = orig["patient_presentation"]
        c["patient_presentation"] = build_noisy_presentation(orig, rng)
        noisy_cases.append(c)

    return noisy_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="KTAS 비정형 노이즈 스트레스 테스트셋 40건 생성")
    parser.add_argument("--eval-cases", default=str(EXP_ROOT / "data" / "benchmark" / "ktas_eval_cases.json"))
    parser.add_argument("--output", "-o", default=str(EXP_ROOT / "data" / "benchmark" / "ktas_noisy_cases.json"))
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    noisy = generate_noisy_dataset(Path(args.eval_cases), n_total=args.n)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(noisy, f, ensure_ascii=False, indent=2)

    sev_cnt = sum(1 for c in noisy if c["ground_truth_level"] in (1, 2))
    mild_cnt = len(noisy) - sev_cnt
    print(f"[+] 비정형 노이즈 케이스 {len(noisy)}건 생성 완료 (중증 1~2등급: {sev_cnt}건 / 경증 3~5등급: {mild_cnt}건) -> {out_path}")


if __name__ == "__main__":
    main()

