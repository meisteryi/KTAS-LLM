#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter

def check_dataset(file_path="ktas_train_dataset.jsonl"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    parts = [p.strip() for p in text.split("\n\n") if p.strip().startswith("{")]
    records = []
    for p in parts:
        try:
            records.append(json.loads(p))
        except Exception as e:
            print("Outer JSON parse error:", e)

    print(f"Total records in file: {len(records)}")

    issues = []
    
    for idx, r in enumerate(records):
        case_id = r.get("id", f"case_{idx}")
        meta = r.get("scenario_metadata", {})
        target_lvl = meta.get("target_level")
        msgs = r.get("messages", [])

        if not msgs:
            issues.append({"idx": idx, "id": case_id, "type": "EMPTY_MESSAGES", "meta": meta, "detail": None})
            continue

        # Check role alternating
        for m_i, m in enumerate(msgs):
            expected_role = "user" if m_i % 2 == 0 else "assistant"
            if m.get("role") != expected_role:
                issues.append({"idx": idx, "id": case_id, "type": f"ROLE_MISMATCH_{m_i}", "meta": meta, "detail": f"got {m.get('role')} expected {expected_role}"})

        last_m = msgs[-1]
        if last_m.get("role") != "assistant":
            issues.append({"idx": idx, "id": case_id, "type": "LAST_ROLE_NOT_ASSISTANT", "meta": meta, "detail": last_m.get("role")})
            continue

        content = last_m.get("content", "")

        # Try to find triage JSON block
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        triage_obj = None
        if m:
            try:
                triage_obj = json.loads(m.group(1).strip())
            except Exception as e:
                issues.append({"idx": idx, "id": case_id, "type": "JSON_PARSE_ERROR_IN_BLOCK", "meta": meta, "detail": str(e), "content": content})
        else:
            brace_m = re.search(r"(\{\s*\"thought_process\"[\s\S]*\})", content)
            if brace_m:
                try:
                    triage_obj = json.loads(brace_m.group(1).strip())
                    issues.append({"idx": idx, "id": case_id, "type": "MISSING_MARKDOWN_CODEBLOCK", "meta": meta, "detail": "Raw JSON without codeblock", "parsed": triage_obj, "content": content})
                except Exception as e:
                    issues.append({"idx": idx, "id": case_id, "type": "INCOMPLETE_JSON", "meta": meta, "detail": str(e), "content": content})
            else:
                issues.append({"idx": idx, "id": case_id, "type": "NO_TRIAGE_JSON", "meta": meta, "detail": "No JSON found in assistant message", "content": content})

        if triage_obj:
            thought = triage_obj.get("thought_process")
            res = triage_obj.get("triage_result", {})
            if not thought:
                issues.append({"idx": idx, "id": case_id, "type": "MISSING_THOUGHT_PROCESS", "meta": meta, "detail": None})
            if not res:
                issues.append({"idx": idx, "id": case_id, "type": "MISSING_TRIAGE_RESULT", "meta": meta, "detail": None})
            else:
                actual_lvl = res.get("ktas_level")
                if actual_lvl != target_lvl:
                    issues.append({"idx": idx, "id": case_id, "type": "LEVEL_MISMATCH", "meta": meta, "detail": f"target={target_lvl}, actual={actual_lvl}"})
                for req_key in ["chief_complaint", "applied_criteria", "urgency", "action"]:
                    if not res.get(req_key):
                        issues.append({"idx": idx, "id": case_id, "type": f"MISSING_{req_key.upper()}", "meta": meta, "detail": None})

    print(f"Total issue occurrences: {len(issues)}")
    type_counts = Counter(i["type"] for i in issues)
    for t, c in type_counts.most_common():
        print(f"  * {t}: {c}건")

    return records, issues

if __name__ == "__main__":
    records, issues = check_dataset()
    with open("audit_issues.json", "w", encoding="utf-8") as out:
        json.dump(issues, out, ensure_ascii=False, indent=2)
