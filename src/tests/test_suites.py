#!/usr/bin/env python3
"""Runner to test the EXACT 2026 endpoint with both datatest1.json and datatest2.jsonl, saving results to a dedicated directory."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

ENDPOINT = os.environ.get(
    "PREDICT_URL",
    "https://main-newnol--exact2026-optionb-qwen25-predict-api.modal.run/predict",
)
TIMEOUT = 120.0

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "test_results"
DATATEST1_PATH = ROOT / "tests" / "datatest1.json"
DATATEST2_PATH = ROOT / "tests" / "datatest2.jsonl"


def _normalize_choice(value: str) -> str:
    return value.strip().lower()


def load_datatest1(path: Path) -> list[dict]:
    """Load records from datatest1.json (keys: logic, physics)."""
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        return []
    try:
        data = json.loads(path.read_text("utf-8"))
        records = []
        if isinstance(data, dict):
            for k in ["logic", "physics"]:
                if k in data and isinstance(data[k], list):
                    records.extend(data[k])
        return records
    except Exception as e:
        print(f"Error parsing datatest1: {e}", file=sys.stderr)
        return []


def load_datatest2(path: Path) -> list[dict]:
    """Load records from datatest2.jsonl."""
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        return []
        
    content = path.read_text("utf-8").strip()

    # Try normal array load
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Wrap raw list
    if not content.startswith("["):
        try:
            data = json.loads(f"[{content}]")
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Fallback decoder
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

    return records


def run_test_suite(records: list[dict], suite_name: str) -> dict[str, Any]:
    print(f"\n=== Starting Test Suite: {suite_name} ({len(records)} items) ===")
    results = {"passed": 0, "failed": 0, "errors": [], "details": []}

    for idx, record in enumerate(records, 1):
        payload = record["request_payload"]
        expected = record["expected"]
        qid = payload.get("query_id", f"record_{idx}")
        qtype = payload.get("type", "unknown")

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
                "query_id": qid,
                "type": qtype,
                "status": "timeout",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": "TIMEOUT",
            })
            continue
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR: {e} ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": str(e)})
            results["details"].append({
                "query_id": qid,
                "type": qtype,
                "status": "error",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": f"ERROR: {e}",
            })
            continue

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"})
            results["details"].append({
                "query_id": qid,
                "type": qtype,
                "status": f"http_{resp.status_code}",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": f"HTTP {resp.status_code}",
            })
            continue

        try:
            data = resp.json()
        except Exception as e:
            print(f"INVALID JSON ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Invalid JSON: {e}"})
            results["details"].append({
                "query_id": qid,
                "type": qtype,
                "status": "invalid_json",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": "INVALID JSON",
            })
            continue

        if isinstance(data, list) and len(data) > 0:
            actual = data[0]
        else:
            actual = data

        if not isinstance(actual, dict):
            print(f"INVALID SHAPE ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Expected dict, got {type(actual)}"})
            continue

        actual_answer = str(actual.get("answer", "")).strip()
        expected_answer = str(expected.get("answer", "")).strip()
        aliases = [str(a).strip() for a in expected.get("aliases", [])]

        is_correct = False
        # Direct Match
        if actual_answer == expected_answer:
            is_correct = True
        elif actual_answer in aliases:
            is_correct = True
        # Numeric equivalence check for safety (e.g. 50 vs 50.0)
        else:
            try:
                if float(actual_answer) == float(expected_answer):
                    is_correct = True
            except ValueError:
                pass

        # Case insensitive options match for logic
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
            "explanation": actual.get("explanation", ""),
            "premises_used": actual.get("premises_used", []),
        })

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading test data...")
    suite1_records = load_datatest1(DATATEST1_PATH)
    suite2_records = load_datatest2(DATATEST2_PATH)

    results_all = {}

    if suite1_records:
        r1 = run_test_suite(suite1_records, "datatest1")
        results_all["datatest1"] = r1
    if suite2_records:
        r2 = run_test_suite(suite2_records, "datatest2")
        results_all["datatest2"] = r2

    # Save summary files
    timestamp = int(time.time())

    # 1. Save SEPARATE files per suite
    for suite, res in results_all.items():
        suite_json = RESULTS_DIR / f"{suite}_{timestamp}.json"
        suite_txt = RESULTS_DIR / f"{suite}_{timestamp}.txt"

        with open(suite_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        total = res["passed"] + res["failed"]
        pct = (res["passed"] / total * 100) if total else 0
        avg_time = sum(d["elapsed_s"] for d in res["details"]) / len(res["details"]) if res["details"] else 0

        lines = []
        lines.append("="*60)
        lines.append(f"EXACT 2026 TEST SUITE: {suite}")
        lines.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("="*60)
        lines.append(f"Passed: {res['passed']}/{total} ({pct:.1f}%)")
        lines.append(f"Failed: {res['failed']}")
        lines.append(f"Avg Request Latency: {avg_time:.1f}s")
        lines.append(f"Total Time: {sum(d['elapsed_s'] for d in res['details']):.1f}s")

        failures = [d for d in res["details"] if d["status"] != "PASS"]
        if failures:
            lines.append(f"\nFailed Details ({len(failures)}):")
            for f in failures:
                lines.append(f"  - {f['query_id']} ({f['type']}): expected='{f['expected_answer']}', got='{f['actual_answer']}'")

        lines.append(f"\nAll Details ({total} items):")
        for d in res["details"]:
            status = d["status"]
            lines.append(f"  [{status}] {d['query_id']} ({d['type']}): {d['expected_answer']} -> {d['actual_answer']} ({d['elapsed_s']}s)")

        suite_txt.write_text("\n".join(lines), "utf-8")
        print(f"\nSaved: {suite_json}")
        print(f"Saved: {suite_txt}")

    # 2. Save a combined comparison summary
    combined_path = RESULTS_DIR / f"combined_{timestamp}.txt"
    combined = []
    combined.append("="*60)
    combined.append(f"COMBINED COMPARISON - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    combined.append("="*60)

    for suite, res in results_all.items():
        total = res["passed"] + res["failed"]
        pct = (res["passed"] / total * 100) if total else 0
        combined.append(f"\n{suite}: {res['passed']}/{total} ({pct:.1f}%)")

    combined.append("\n" + "-"*60)
    combined.append("Failed items comparison:")
    # Collect failures per suite
    for suite, res in results_all.items():
        failures = [d for d in res["details"] if d["status"] != "PASS"]
        for f in failures:
            combined.append(f"  [{suite}] {f['query_id']} ({f['type']}): expected='{f['expected_answer']}', got='{f['actual_answer']}'")

    combined_path.write_text("\n".join(combined), "utf-8")
    print(f"Saved: {combined_path}")

    # 3. Print combined summary to console
    print("\n" + "\n".join(combined))


if __name__ == "__main__":
    main()
