#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""원본 KTAS PDF(2021_KTAS_guideline_ver1.6.pdf) 기준 JSON Ground Truth 보정 스크립트.

수행 작업:
1. data/raw/ 원본 파일을 data/backup/에 무손실 백업
2. ktas_part1_common.json의 pediatric_first_order_modifiers 누락 4개 축 복원 (p.197, 200, 201, 202-203)
3. PDF 대조로 확인된 소아 등급 불일치(052, 153, 262, 356, 705) 및 조건 누락(854) 보정,
   소아 NACRS 코드 정렬(601~610) 및 누락 주호소 611(소아의 선천적 문제) 추가
4. PDF 각 주호소 페이지의 '활력징후/그 밖의 1차 고려사항' 패널을 파싱하여
   전체 주호소에 first_order_modifiers_applicable 필드 복원
5. 심정지(001, 002) 등 DNR 예외 게이트 주호소에 evaluation_priority: "exception_first" 명시
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pymupdf

from verify_process_equivalence import extract_pdf_complaints


PEDIATRIC_CODE_REMAP = {
    "266": "601",  # 신생아 수유곤란
    "267": "602",  # 신생아 황달
    "867": "603",  # 영유아의 달랠 수 없는 울음
    "659": "604",  # 천명음 - 다른 증상 호소 없음
    "559": "605",  # 소아 보행 장애 / 보행시 통증
    "559_ped": "605",  # 소아 보행 장애 / 보행시 통증
    "660": "606",  # 영아의 무호흡 발작
    "406_floppy": "609",  # 축 늘어진 소아 (Floppy child)
    "658": "610",  # 협착음
}

PEDIATRIC_611_DATA = {
    "name": "소아의 선천적 문제",
    "nacrs_code": "611",
    "second_order_modifiers": [
        {"ktas_level": 1, "condition": "소아 1차 활력징후 1단계 만족"},
        {"ktas_level": 2, "condition": "소아 1차 활력징후 2단계 만족"},
        {"ktas_level": 2, "condition": "대사성 이상, 경련, 구토 / 설사, 1형 당뇨병 또는 부신 기능부전증"},
        {"ktas_level": 3, "condition": "소아 1차 활력징후 3단계, 출혈성 질환 3단계 만족"},
        {"ktas_level": 3, "condition": "치료가 필요한지 보호자가 확인하고 싶은 경우"},
        {"ktas_level": 4, "condition": "잠재적인 문제가 있는 선천적 질환을 가진 안정적인 상태의 소아"},
    ],
}


def backup_raw_files(raw_dir: Path, backup_dir: Path) -> None:
    """최초 실행 시 data/raw/ 원본 JSON을 data/backup/에 보존한다."""
    if backup_dir.exists() and any(backup_dir.glob("ktas_*.json")):
        print(f"[*] 기존 백업 유지: {backup_dir}")
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(raw_dir.glob("ktas_*.json")):
        shutil.copy2(p, backup_dir / p.name)
    print(f"[+] 원본 백업 완료: {backup_dir}")


def patch_pediatric_common(common_path: Path) -> None:
    """ktas_part1_common.json에 PDF p.197, 199, 200, 201, 202-203 기반 소아 1차 기준을 복원한다."""
    with common_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ped = data["pediatric_first_order_modifiers"]
    ped["respiratory"] = [
        {
            "ktas_level": 1,
            "classification": "중증 호흡곤란",
            "spo2": "< 90%",
            "criteria": "과도한 호흡 노력, 청색증, 혼수/착란, 보호자 인지 불가, 통증 반응 감소, 한 단어로만 말하거나 발화 불가, 빈맥/서맥, 무호흡/불규칙 호흡, 심한 흉부함몰, 코 벌렁임, 그렁거림, 호흡음 감소/소실, 상부기도 폐쇄, 기도 유지 불가, 약한 근육 긴장도",
        },
        {
            "ktas_level": 2,
            "classification": "중등도 호흡곤란",
            "spo2": "< 92%",
            "criteria": "증가된 호흡 노력, 안절부절못함, 불안, 공격성, 빈호흡/과호흡, 경도 호흡 보조근 사용, 흉부 함몰, 코 벌렁임, 구문/끊어진 문장 발화, 협착음 있으나 기도 유지, 길어진 호기 시간",
        },
        {
            "ktas_level": 3,
            "classification": "경증 호흡곤란",
            "spo2": "92-94%",
            "criteria": "호흡곤란, 빈호흡, 운동 시 숨참, 명백한 호흡 노력 증가 없음, 완전한 문장 발화 가능, 명백한 기도 폐쇄 없는 협착음, 잦은 기침 (단, SpO2 92-94%이더라도 무증상 시 4~5단계)",
        },
    ]
    ped["bleeding_disorder"] = [
        {
            "ktas_level": 2,
            "classification": "출혈성 질환 - 위급한 출혈",
            "criteria": "선천성/후천성 지혈 이상 또는 항혈소판/항응고제 복용 환자의 생명 혹은 사지 소실 위험 출혈 (두부/두개내, 경부, 흉부, 복부, 골반, 척추, 대량 질 출혈, 관절 혈관절증, 엉덩허리근/엉덩이, 사지 근육 구획 증후군)",
        },
        {
            "ktas_level": 3,
            "classification": "출혈성 질환 - 중등도나 경도의 출혈",
            "criteria": "지혈 이상 환자의 코피, 입/잇몸 출혈, 월경과다, 찰과상과 단순 열상, 골절 또는 탈구, 건/신경손상 동반 심부 열상, 멈추지 않는 출혈",
        },
    ]
    ped["high_risk_mechanism"] = [
        {
            "ktas_level": 2,
            "classification": "고위험성 사고기전 (소아)",
            "criteria": "일반적 외상(자동차 사고 차량 이탈/전복/구조>20분/함입/동승자 사망/미착용 >40km/h 또는 착용 >60km/h, 오토바이 >30km/h, 보행자/자전거 >10km/h, 추락 >6m 또는 5계단, 두부/경부/몸통/근위부 관통상), 두부 외상(차량 이탈, 차유리 충돌, 보행자 충돌, 추락 >1m 또는 5계단, 둔기 폭행), 경부 외상(고속 차량/오토바이 사고, 추락 >1m 또는 5계단, 머리 종축 외력)",
        }
    ]
    ped["psychiatric_emergency_criteria"] = {
        "suicide": {
            "level_2": [
                "자살시도 (죽으려는 의도에 따른 노골적/암시적 행동)",
                "뚜렷한 자살의도 (자살로 이어질 수 있는 주관적 생각이나 바람)",
                "도주의 가능성 또는 안전에 대한 위험 (폭력적 위협, 조절되지 않는 분노/초조/편집증/환각, 집중 관찰 필요)",
            ],
            "level_3": ["자살관념 (구체적인 계획이나 즉각적 위험 없는 죽음에 대한 생각)"],
        },
        "agitation_anxiety": {
            "level_2": ["중증 불안 / 초조 (빈맥과 초조함 동반, 극도의 흥분상태로 의료진 협조 요청에도 진정되지 않음)"],
            "level_3": ["중등도 불안 / 초조 (명백한 불안/걱정, 빈맥/떨림 없음, 지시를 일관성 있게 수행하지 못함)"],
            "level_4": ["경증 불안 / 초조 (가벼운 불안, 안심시키면 나아지고 협조적임)"],
        },
        "bizarre_behavior": {
            "level_1": ["조절되지 않는 행동 (기이하며 지남력 없고 이성/언어로 조절 불가, 환자나 타인을 신체적 위험에 빠뜨림)"],
            "level_3": ["조절되는 행동 (기이하고 비이성적이나 합리적 설득으로 통제 가능, 가족/친구 동반)"],
            "level_4": ["해롭지 않은 행동 (장기간 지속, 급격한 변화 없고 위협이 되지 않음)"],
        },
    }
    ped["pain_assessment_tools"] = ["FLACC (<4세 또는 의사소통 불가)", "FACES (4~7세)", "VAS / 숫자척도 (>=8세)"]

    with common_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[+] ktas_part1_common.json 소아 1차 고려사항 4개 축 및 통증 도구 복원 완료")


def fix_complaint_files(raw_dir: Path, pdf_recs: dict[tuple[bool, str], dict[str, Any]]) -> dict[str, int]:
    """개별 JSON 파일의 등급 오류, 코드 불일치, 누락 주호소, 1차 적용 필드, 예외 플래그를 보정한다."""
    stats = {
        "level_fixes": 0,
        "text_fixes": 0,
        "code_remaps": 0,
        "added_611": 0,
        "first_order_populated": 0,
        "exception_flags": 0,
    }

    for json_path in sorted(raw_dir.glob("ktas_*.json")):
        if json_path.name == "ktas_part1_common.json":
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        is_ped = "Pediatric" in str(data.get("target_population", ""))
        categories = data.get("categories", {})

        for cat_id, cat_val in categories.items():
            complaints = cat_val.get("complaints", {})

            # 1) 소아 NACRS 코드 정렬 (266->601 등)
            if is_ped:
                new_complaints: dict[str, Any] = {}
                for old_code, comp in complaints.items():
                    if old_code in PEDIATRIC_CODE_REMAP:
                        new_code = PEDIATRIC_CODE_REMAP[old_code]
                        comp["nacrs_code"] = new_code
                        new_complaints[new_code] = comp
                        stats["code_remaps"] += 1
                    else:
                        new_complaints[old_code] = comp
                complaints = new_complaints
                cat_val["complaints"] = complaints

                # 611 소아의 선천적 문제 추가
                if json_path.name == "ktas_p5c_ped_sensory_general.json" and cat_id == "general" and "611" not in complaints:
                    complaints["611"] = dict(PEDIATRIC_611_DATA)
                    stats["added_611"] += 1

            # 2) 주호소별 내용 보정
            for code, comp in complaints.items():
                nacrs = str(comp.get("nacrs_code", code))
                mods = comp.get("second_order_modifiers", [])

                # 2a) 확인된 소아 등급 오류 및 조건 누락 보정
                if is_ped:
                    if nacrs == "052":  # 귀의 이물질 (PDF p.234: 2, 3, 4)
                        for m in mods:
                            c = m.get("condition", "")
                            if "중증 통증" in c and m.get("ktas_level") == 3:
                                m["ktas_level"] = 2
                                stats["level_fixes"] += 1
                            elif "중등도 통증" in c and m.get("ktas_level") == 4:
                                m["ktas_level"] = 3
                                stats["level_fixes"] += 1
                            elif "경증 통증" in c and m.get("ktas_level") == 5:
                                m["ktas_level"] = 4
                                stats["level_fixes"] += 1
                    elif nacrs == "153":  # 코의 이물질 (PDF p.248: 4)
                        for m in mods:
                            if "경증 통증" in m.get("condition", "") and m.get("ktas_level") == 5:
                                m["ktas_level"] = 4
                                stats["level_fixes"] += 1
                    elif nacrs == "262":  # 딸꾹질 (PDF p.269: 4)
                        for m in mods:
                            if "만성, 힘들어 보이지 않음" in m.get("condition", "") and m.get("ktas_level") == 5:
                                m["ktas_level"] = 4
                                stats["level_fixes"] += 1
                    elif nacrs == "356":  # 사회문제 (PDF p.290: 2)
                        for m in mods:
                            if "학대" in m.get("condition", "") and m.get("ktas_level") == 3:
                                m["ktas_level"] = 2
                                stats["level_fixes"] += 1
                    elif nacrs == "705":  # 화상 (PDF p.348: 2, 3)
                        for m in mods:
                            c = m.get("condition", "")
                            if "5-25%" in c and m.get("ktas_level") == 3:
                                m["ktas_level"] = 2
                                stats["level_fixes"] += 1
                            elif "<5%" in c and "<10%" in c and m.get("ktas_level") == 4:
                                m["ktas_level"] = 3
                                stats["level_fixes"] += 1
                    elif nacrs == "854":  # 소아 저혈당 (PDF p.373: 1세 이상에서)
                        for m in mods:
                            c = m.get("condition", "")
                            if "54-70" in c and "무증상" in c and "1세 이상" not in c:
                                m["condition"] = "1세 이상에서 " + c
                                stats["text_fixes"] += 1

                # 2b) PDF 페이지의 활력징후/그 밖의 1차 고려사항을 first_order_modifiers_applicable로 복원
                prec = pdf_recs.get((is_ped, nacrs))
                if prec:
                    first_items = []
                    for lvl, txt in prec["sections"].get("first_vital", []) + prec["sections"].get("first_other", []):
                        entry = f"{txt} ({lvl})"
                        if entry not in first_items:
                            first_items.append(entry)
                    if first_items:
                        comp["first_order_modifiers_applicable"] = first_items
                        stats["first_order_populated"] += 1

                # 2c) DNR 등 예외 우선 평가 플래그 추가 (상위 1단계 뒤에 5단계 예외가 오는 경우)
                has_dnr_exception = any(
                    m.get("ktas_level") == 5 and any(k in m.get("condition", "") for k in ("DNR", "심폐소생술의 대상이 아닌"))
                    for m in mods
                )
                has_lvl1 = any(m.get("ktas_level") == 1 for m in mods)
                if has_dnr_exception and has_lvl1:
                    comp["evaluation_priority"] = "exception_first"
                    comp["priority_exception_rule"] = "DNR 등 소생술 제외 대상 확인 시 1단계보다 우선하여 5단계로 분류"
                    stats["exception_flags"] += 1

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    exp_root = Path(__file__).resolve().parents[2]
    repo_root = exp_root.parents[1]
    raw_dir = exp_root / "data" / "raw"
    backup_dir = exp_root / "data" / "backup"
    pdf_path = repo_root / "2021_KTAS_guideline_ver1.6.pdf"

    backup_raw_files(raw_dir, backup_dir)
    doc = pymupdf.open(pdf_path)
    pdf_recs = extract_pdf_complaints(doc)

    patch_pediatric_common(raw_dir / "ktas_part1_common.json")
    stats = fix_complaint_files(raw_dir, pdf_recs)
    print(f"[+] 주호소 보정 통계: {stats}")


if __name__ == "__main__":
    main()
