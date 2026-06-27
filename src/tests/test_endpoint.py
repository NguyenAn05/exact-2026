#!/usr/bin/env python3
"""Test the EXACT 2026 prediction endpoint with datatest.jsonl."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ENDPOINT = "https://velbertrack--exact2026-optionb-qwen25-predict-api.modal.run/predict"
# Let's locate the datatest file from the other workspace directory we downloaded it to, or copy it over.
DATA_FILE = Path("/Users/newnol/workspace/03-competition/mlops-exact-pipeline/exact2026_modal_ready/tests/datatest.jsonl")
TIMEOUT = 120.0  # Modal cold start can take 60s+


def load_records(path: Path, limit: int | None = None) -> list[dict]:
    """Load JSON records from the file, handling potentially raw list or concatenated JSON objects."""
    if not path.exists():
        print(f"Error: {path} does not exist. Please place datatest.jsonl in tests/", file=sys.stderr)
        sys.exit(1)
        
    content = path.read_text("utf-8").strip()

    # If it's a valid JSON array or object
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data[:limit] if limit else data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # If it is formatted as [ {..}, {..} ] but maybe has tailing commas or wraps
    if not content.startswith("["):
        try:
            data = json.loads(f"[{content}]")
            if isinstance(data, list):
                return data[:limit] if limit else data
        except json.JSONDecodeError:
            pass

    # Fallback to manual parser
    records = []
    decoder = json.JSONDecoder()
    pos = 0
    content_len = len(content)
    while pos < content_len:
        while pos < content_len and content[pos] in " \t\n\r,[]":
            pos += 1
        if pos >= content_len:
            break
        try:
            obj, next_pos = decoder.raw_decode(content, pos)
            if isinstance(obj, dict) and "request_payload" in obj:
                records.append(obj)
            pos = next_pos
        except json.JSONDecodeError:
            pos += 1

    return records[:limit] if limit else records


def main():
    records = load_records(DATA_FILE)
    print(f"Loaded {len(records)} test records.")

    results = {"passed": 0, "failed": 0, "errors": [], "details": []}

    for idx, record in enumerate(records, 1):
        payload = record["request_payload"]
        expected = record["expected"]
        qid = payload.get("query_id", f"record_{idx}")

        print(f"\n[{idx}/{len(records)}] {qid} ... ", end="", flush=True)

        start = time.time()

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(ENDPOINT, json=payload)
                elapsed = time.time() - start
        except httpx.TimeoutException:
            elapsed = time.time() - start
            print(f"TIMEOUT ({elapsed:.0f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Timeout after {elapsed:.0f}s"})
            results["details"].append({
                "query_id": qid,
                "status": "timeout",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
            })
            continue
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.0f}s): {e}")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": str(e)})
            continue

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
            results["details"].append({
                "query_id": qid,
                "status": f"http_{resp.status_code}",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
            })
            continue

        try:
            data = resp.json()
        except Exception as e:
            print(f"INVALID JSON ({elapsed:.1f}s): {e}")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Invalid JSON: {e}"})
            continue

        # Check answer
        if isinstance(data, list) and len(data) > 0:
            actual = data[0]
        else:
            actual = data

        if not isinstance(actual, dict):
            print(f"INVALID RESPONSE SHAPE ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Expected dict, got {type(actual)}"})
            continue

        actual_answer = str(actual.get("answer", "")).strip()
        expected_answer = str(expected.get("answer", "")).strip()
        aliases = [str(a).strip() for a in expected.get("aliases", [])]

        is_correct = False
        if actual_answer == expected_answer:
            is_correct = True
        elif actual_answer in aliases:
            is_correct = True
        elif expected_answer in actual_answer or any(a.lower() in actual_answer.lower() for a in aliases):
            is_correct = True
            print(f"MATCH (fuzzy) ({elapsed:.1f}s) expected={expected_answer}, got={actual_answer}", end=" | ")
        else:
            print(f"MISMATCH ({elapsed:.1f}s) expected={expected_answer}, got={actual_answer}", end=" | ")

        # Check unit  
        actual_unit = str(actual.get("unit", "")).strip()
        expected_unit = str(expected.get("unit", "")).strip()
        unit_ok = actual_unit == expected_unit or (not expected_unit and not actual_unit)

        # Check premises_used
        actual_premises = actual.get("premises_used", [])
        expected_premises = expected.get("premises_used", [])

        if is_correct:
            results["passed"] += 1
            status = "PASS"
        else:
            results["failed"] += 1
            status = "FAIL"

        results["details"].append({
            "query_id": qid,
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "expected_unit": expected_unit,
            "actual_unit": actual_unit,
            "unit_ok": unit_ok,
            "expected_premises": expected_premises,
            "actual_premises": actual_premises,
        })

        print(f"[{status}]")

    # Summary
    total = len(records)
    passed = results["passed"]
    failed = results["failed"]
    pct = (passed / total * 100) if total else 0
    avg_time = sum(d["elapsed_s"] for d in results["details"]) / len(results["details"]) if results["details"] else 0

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed ({pct:.1f}%)")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Avg time per request: {avg_time:.1f}s")
    print(f"  Total time: {sum(d['elapsed_s'] for d in results['details']):.1f}s")

    if results["errors"]:
        print(f"\n  Errors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"    - {e['query_id']}: {e['error']}")

    # Print details of failed tests
    failed_details = [d for d in results["details"] if d["status"] != "PASS"]
    if failed_details:
        print(f"\n  Failed details:")
        for d in failed_details:
            print(f"    - {d['query_id']}: expected={d['expected_answer']}, got={d['actual_answer']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
