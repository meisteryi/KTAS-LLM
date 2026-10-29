#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
format_sft_dataset.py
기존 KTAS 대화 데이터(1,500건)를 한 줄 CoT 근거 및 Deterministic Rule-ID가 포함된 표준 SFT 포맷으로 변환하는 스크립트
"""

import os
import json
import re

def extract_rule_and_rationale(level, complaint_name, scenario_meta=None):
    # KTAS 레벨 및 주호소에 기반한 표준 룰 코드 및 한 줄 CoT 근거 매핑
    rule_map = {
        1: {
            "rule_id": "1차고려_소생_즉각적생명위협",
            "rationale": f"{complaint_name} 증상은 즉각적인 기도 확보 및 심폐 소생술이 필요한 최고 위급 상태임."
        },
        2: {
            "rule_id": f"2차고려_{complaint_name}_중증응급",
            "rationale": f"{complaint_name} 증상은 활력징후 및 장기 손상 위험이 있는 긴급 상태로 다른 만성 질환에 우선하여 2단계 적용."
        },
        3: {
            "rule_id": f"2차고려_{complaint_name}_응급",
            "rationale": f"{complaint_name} 증상은 급성 진행 가능성이 있어 수시간 내 진료와 검사가 필요한 3단계 응급 상태임."
        },
        4: {
            "rule_id": f"2차고려_{complaint_name}_준응급",
            "rationale": f"{complaint_name} 증상은 생체징후가 안정적이나 통증 조절 및 단순 처치가 필요한 4단계 준응급 상태임."
        },
        5: {
            "rule_id": f"2차고려_{complaint_name}_비응급",
            "rationale": f"{complaint_name} 증상은 만성적이거나 경미하여 일반 외래 진료 순서에 따라 대기 가능한 5단계 비응급 상태임."
        }
    }
    return rule_map.get(level, rule_map[3])


def format_split(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[-] Input file not found: {input_file}")
        return 0

    converted_count = 0
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            ktas_level = data.get("ktas_level", 3)
            messages = data.get("messages", [])
            
            # 주호소 추정 (메타데이터 또는 환자 첫 발화 기반)
            complaint_name = "급성 질환 의증"
            if "scenario_metadata" in data:
                complaint_name = data["scenario_metadata"].get("chief_complaint_name", complaint_name)
            elif len(messages) > 0 and messages[0]["role"] == "user":
                content = messages[0]["content"]
                if "가슴" in content or "흉통" in content:
                    complaint_name = "흉통"
                elif "머리" in content or "어지" in content or "말이" in content:
                    complaint_name = "신경학적 증상"
                elif "숨" in content or "호흡" in content:
                    complaint_name = "호흡곤란"
                elif "배" in content or "복통" in content:
                    complaint_name = "복통"

            meta = extract_rule_and_rationale(ktas_level, complaint_name)
            
            # 마지막 assistant 응답을 표준 구조화된 JSON으로 변환
            if messages and messages[-1]["role"] == "assistant":
                original_resp = messages[-1]["content"]
                
                # 순수 텍스트 응답 추출 (기존 json 코드블록 제거)
                cleaned_resp = re.sub(r'```json.*?```', '', original_resp, flags=re.DOTALL).strip()
                if not cleaned_resp:
                    cleaned_resp = f"환자분, 현재 호소하시는 {complaint_name} 증상은 KTAS {ktas_level}단계로 분류되었습니다. 안내에 따라 신속히 진료를 준비해 드리겠습니다."

                structured_output = {
                    "primary_complaint": complaint_name,
                    "applied_rule_id": meta["rule_id"],
                    "rationale": meta["rationale"],
                    "ktas_level": ktas_level,
                    "response": cleaned_resp
                }
                
                messages[-1]["content"] = json.dumps(structured_output, ensure_ascii=False, indent=2)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            converted_count += 1

    return converted_count


def main():
    print("[*] Formatting SFT training & validation datasets...")
    train_in = "/home/meisteryi/문서/KTAS-LLM/data/train.jsonl"
    train_out = "/home/meisteryi/문서/KTAS-LLM/data/train_sft.jsonl"
    val_in = "/home/meisteryi/문서/KTAS-LLM/data/val.jsonl"
    val_out = "/home/meisteryi/문서/KTAS-LLM/data/val_sft.jsonl"

    c_train = format_split(train_in, train_out)
    c_val = format_split(val_in, val_out)
    print(f"[+] Formatted {c_train} train cases to {train_out}")
    print(f"[+] Formatted {c_val} val cases to {val_out}")

if __name__ == "__main__":
    main()

