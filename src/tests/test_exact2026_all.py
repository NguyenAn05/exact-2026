#!/usr/bin/env python3
"""Test EXACT2026 all-in-one dataset against the prediction endpoint."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ENDPOINT = os.environ.get(
    "PREDICT_URL",
    "https://main-newnol--exact2026-optionb-qwen25-predict-api.modal.run/predict",
)
TIMEOUT = 120.0

RESULTS_DIR = Path(__file__).resolve().parent.parent / "test_results"
DATA_FILE = Path(__file__).resolve().parent / "EXACT2026_test_all.json"


def load_records(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text("utf-8"))


def _normalize_choice(value: str) -> str:
    return value.strip().lower()


def run_test_suite(records: list[dict[str, Any]], suite_name: str) -> dict[str, Any]:
    print(f"\n=== Starting Test Suite: {suite_name} ({len(records)} items) ===")
    results = {"passed": 0, "failed": 0, "errors": [], "details": []}

    for idx, record in enumerate(records, 1):
        payload = {
            "query_id": record["query_id"],
            "type": record["type"],
            "query": record["query"],
            "premises": record.get("premises", []),
            "options": record.get("options", []),
        }
        expected = record
        qid = record["query_id"]
        qtype = record["type"]

        print(f"[{idx}/{len(records)}] {qid} ({qtype}) ... ", end="", flush=True)
        start = time.time()

        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                resp = client.post(ENDPOINT, json=payload)
                elapsed = time.time() - start
        except httpx.TimeoutException:
            elapsed = time.time() - start
            print(f"TIMEOUT ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Timeout after {elapsed:.1f}s"})
            results["details"].append({
                "query_id": qid, "type": qtype, "status": "timeout", "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"], "actual_answer": "TIMEOUT",
            })
            continue
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR: {e} ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": str(e)})
            continue

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
            results["details"].append({
                "query_id": qid, "type": qtype, "status": f"http_{resp.status_code}",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"], "actual_answer": f"HTTP {resp.status_code}",
            })
            continue

        try:
            data = resp.json()
        except Exception as e:
            print(f"INVALID JSON ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Invalid JSON: {e}"})
            continue

        actual = data[0] if isinstance(data, list) and data else data

        if not isinstance(actual, dict):
            print(f"INVALID SHAPE ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Expected dict, got {type(actual)}"})
            continue

        actual_answer = str(actual.get("answer", "")).strip()
        expected_answer = str(expected.get("answer", "")).strip()

        is_correct = False
        if actual_answer == expected_answer:
            is_correct = True
        else:
            try:
                if float(actual_answer) == float(expected_answer):
                    is_correct = True
            except ValueError:
                pass

        if not is_correct and qtype == "type1" and payload.get("options"):
            if _normalize_choice(actual_answer) == _normalize_choice(expected_answer):
                is_correct = True

        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            results["passed"] += 1
            print(f"PASS ({elapsed:.1f}s)")
        else:
            results["failed"] += 1
            print(f"FAIL ({elapsed:.1f}s) expected={expected_answer}, got={actual_answer}")

        results["details"].append({
            "query_id": qid,
            "type": qtype,
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "expected_unit": str(expected.get("unit", "")),
            "actual_unit": str(actual.get("unit", "")),
            "expected_premises": expected.get("premises_used", []),
            "actual_premises": actual.get("premises_used", []),
            "explanation": actual.get("explanation", ""),
        })

    return results


def print_summary(results: dict[str, Any], suite_name: str) -> None:
    total = results["passed"] + results["failed"]
    pct = (results["passed"] / total * 100) if total else 0
    avg_time = sum(d["elapsed_s"] for d in results["details"]) / len(results["details"]) if results["details"] else 0

    lines = [
        "=" * 60,
        f"EXACT 2026 TEST SUITE: {suite_name}",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Passed: {results['passed']}/{total} ({pct:.1f}%)",
        f"Failed: {results['failed']}",
        f"Avg Request Latency: {avg_time:.1f}s",
        f"Total Time: {sum(d['elapsed_s'] for d in results['details']):.1f}s",
    ]

    type1_details = [d for d in results["details"] if d["type"] == "type1"]
    type2_details = [d for d in results["details"] if d["type"] == "type2"]
    if type1_details:
        t1_passed = sum(1 for d in type1_details if d["status"] == "PASS")
        lines.append(f"Type1: {t1_passed}/{len(type1_details)} ({t1_passed/len(type1_details)*100:.1f}%)")
    if type2_details:
        t2_passed = sum(1 for d in type2_details if d["status"] == "PASS")
        lines.append(f"Type2: {t2_passed}/{len(type2_details)} ({t2_passed/len(type2_details)*100:.1f}%)")

    failures = [d for d in results["details"] if d["status"] != "PASS"]
    if failures:
        lines.append(f"\nFailed Details ({len(failures)}):")
        for f in failures:
            lines.append(f"  - {f['query_id']} ({f['type']}): expected='{f['expected_answer']}', got='{f['actual_answer']}'")

    print("\n" + "\n".join(lines))
    return lines


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(DATA_FILE)
    print(f"Loaded {len(records)} test records from {DATA_FILE}")

    results = run_test_suite(records, "EXACT2026_test_all")

    timestamp = int(time.time())
    out_json = RESULTS_DIR / f"EXACT2026_test_all_{timestamp}.json"
    out_txt = RESULTS_DIR / f"EXACT2026_test_all_{timestamp}.txt"

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = print_summary(results, "EXACT2026_test_all")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved results to:\n- {out_json}\n- {out_txt}")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
