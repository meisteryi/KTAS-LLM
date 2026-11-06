#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_ablation_datasets.py
4-Arm 어블레이션 실험을 위한 3종 데이터셋(Arm 1, Arm 2, Arm 3) 자동 생성 스크립트:
- Arm 1 (Data-Only): 환자 발화 -> {"ktas_level": N}
- Arm 2 (Tree-Only): 임상 규칙 시나리오 -> {"applied_rule_id": ..., "ktas_level": N}
- Arm 3 (Data+Tree): 환자 발화 -> {"primary_complaint": ..., "applied_rule_id": ..., "rationale": ..., "ktas_level": N}
"""

import json
from pathlib import Path

# 경로 정의
SOURCE_DIR = Path("experiments/01_ktas_causal_triage/data")
TARGET_DIR = Path("experiments/02_tree_ablation_xai/data")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPTS = {
    "arm1": "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의료진입니다. 환자의 초기 진술을 듣고 즉각적인 위험 징후와 의학적 인과관계에 기반하여 공식 KTAS 단계(1~5단계)를 JSON 형식으로 출력하십시오.",
    "arm2": "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의료진입니다. 제시된 의학적 임상 규칙 및 룰 트리를 바탕으로 적용 룰 ID와 공식 KTAS 단계를 JSON 형식으로 출력하십시오.",
    "arm3": "당신은 한국형 응급환자 분류도구(KTAS) 전문 분류 의료진입니다. 환자의 초기 진술을 듣고 즉각적인 위험 징후와 의학적 인과관계에 기반하여 최우선 주호소를 식별하고, 공식 KTAS 단계(1~5단계)와 결정 근거(Rationale) 및 적용 룰 ID를 JSON 형식으로 출력하십시오."
}

def convert_dataset(split_name):
    src_file = SOURCE_DIR / f"{split_name}_sft_augmented.jsonl"
    assert src_file.exists(), f"원본 파일을 찾을 수 없습니다: {src_file}"
    
    arm1_records = []
    arm2_records = []
    arm3_records = []
    
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            user_msg = sample["messages"][1]["content"]
            assistant_content = sample["messages"][2]["content"]
            assistant_data = json.loads(assistant_content)
            
            # 1. Arm 1 (Data-Only: 숫자만 출력)
            arm1_sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["arm1"]},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": json.dumps({"ktas_level": assistant_data["ktas_level"]}, ensure_ascii=False, indent=2)}
                ]
            }
            arm1_records.append(arm1_sample)
            
            # 2. Arm 2 (Tree-Only: 규칙 문장 -> 룰 ID + 단계)
            tree_scenario = f"[임상 규칙 시나리오]: \"{assistant_data['primary_complaint']} 증상이 관찰되며, {assistant_data['rationale']}\""
            arm2_sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["arm2"]},
                    {"role": "user", "content": tree_scenario},
                    {"role": "assistant", "content": json.dumps({
                        "applied_rule_id": assistant_data["applied_rule_id"],
                        "ktas_level": assistant_data["ktas_level"]
                    }, ensure_ascii=False, indent=2)}
                ]
            }
            arm2_records.append(arm2_sample)
            
            # 3. Arm 3 (Data + Tree: 제안군, 전체 풀 포맷)
            arm3_sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["arm3"]},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            arm3_records.append(arm3_sample)
            
    # 저장
    files_to_save = [
        (f"arm1_{split_name}_data_only.jsonl", arm1_records),
        (f"arm2_{split_name}_tree_only.jsonl", arm2_records),
        (f"arm3_{split_name}_data_tree.jsonl", arm3_records),
    ]
    
    for filename, records in files_to_save:
        out_path = TARGET_DIR / filename
        with open(out_path, "w", encoding="utf-8") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[+] {out_path.name} 저장 완료 ({len(records):,} 건)")

def main():
    print("=" * 65)
    print("[Phase 1] 4-Arm 어블레이션 실험용 데이터셋 생성 시작...")
    print(f" - 원천 데이터: {SOURCE_DIR}")
    print(f" - 대상 폴더  : {TARGET_DIR}")
    print("=" * 65)
    
    convert_dataset("train")
    convert_dataset("val")
    
    # 4-way 벤치마크 파일도 링크 복사
    src_bench = SOURCE_DIR / "counterfactual_benchmark.jsonl"
    dst_bench = TARGET_DIR / "counterfactual_benchmark.jsonl"
    if src_bench.exists():
        with open(src_bench, "r", encoding="utf-8") as f_in, open(dst_bench, "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
        print(f"[+] 4-way 반사실적 벤치마크 복사 완료: {dst_bench.name}")
        
    print("=" * 65)
    print("[+] 모든 어블레이션 데이터셋 생성이 완벽하게 완료되었습니다!")

if __name__ == "__main__":
    main()

