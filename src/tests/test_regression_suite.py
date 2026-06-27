#!/usr/bin/env python3
"""Quick regression runner for the patched failure cases only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request

ENDPOINT = os.environ.get(
    "PREDICT_URL",
    "https://main-newnol--exact2026-optionb-qwen25-predict-api.modal.run/predict",
)
TIMEOUT = 120.0

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "test_results"
REGRESSION_PATH = ROOT / "tests" / "datatest_regression.json"


def _normalize_choice(value: str) -> str:
    return value.strip().lower()


def load_regression_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text("utf-8"))
    records: list[dict[str, Any]] = []
    for key in ("logic", "physics"):
        items = data.get(key, [])
        if isinstance(items, list):
            records.extend(items)
    return records


def run_test_suite(records: list[dict[str, Any]], suite_name: str) -> dict[str, Any]:
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
            req = request.Request(
                ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=TIMEOUT) as resp:
                elapsed = time.time() - start
                status_code = resp.status
                body = resp.read().decode("utf-8")
        except Exception as exc:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": str(exc)})
            results["details"].append({
                "query_id": qid,
                "type": qtype,
                "status": "error",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": f"ERROR: {exc}",
            })
            continue

        if status_code != 200:
            print(f"HTTP {status_code} ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"HTTP {status_code}: {body[:200]}"})
            results["details"].append({
                "query_id": qid,
                "type": qtype,
                "status": f"http_{status_code}",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": f"HTTP {status_code}",
            })
            continue

        try:
            data = json.loads(body)
        except Exception as exc:
            print(f"INVALID JSON ({elapsed:.1f}s)")
            results["failed"] += 1
            results["errors"].append({"query_id": qid, "error": f"Invalid JSON: {exc}"})
            results["details"].append({
                "query_id": qid,
                "type": qtype,
                "status": "invalid_json",
                "elapsed_s": round(elapsed, 1),
                "expected_answer": expected["answer"],
                "actual_answer": "INVALID JSON",
            })
            continue

        actual = data[0] if isinstance(data, list) and data else data
        actual_answer = str(actual.get("answer", "")).strip()
        expected_answer = str(expected.get("answer", "")).strip()
        aliases = [str(a).strip() for a in expected.get("aliases", [])]

        is_correct = False
        if actual_answer == expected_answer or actual_answer in aliases:
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
        results["passed" if is_correct else "failed"] += 1
        print(f"{status} ({elapsed:.1f}s)" if is_correct else f"{status} ({elapsed:.1f}s) expected={expected_answer}, got={actual_answer}")

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


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_regression_records(REGRESSION_PATH)
    results = run_test_suite(records, "datatest_regression")

    timestamp = int(time.time())
    out_json = RESULTS_DIR / f"datatest_regression_{timestamp}.json"
    out_txt = RESULTS_DIR / f"datatest_regression_{timestamp}.txt"

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    total = results["passed"] + results["failed"]
    pct = (results["passed"] / total * 100) if total else 0
    avg_time = sum(d["elapsed_s"] for d in results["details"]) / len(results["details"]) if results["details"] else 0

    lines = [
        "=" * 60,
        "EXACT 2026 REGRESSION SUITE",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"Passed: {results['passed']}/{total} ({pct:.1f}%)",
        f"Failed: {results['failed']}",
        f"Avg Request Latency: {avg_time:.1f}s",
    ]
    if results["failed"]:
        lines.append("")
        lines.append("Failed Details:")
        for d in results["details"]:
            if d["status"] != "PASS":
                lines.append(f"  - {d['query_id']} ({d['type']}): expected='{d['expected_answer']}', got='{d['actual_answer']}'")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved regression results to:\n- {out_json}\n- {out_txt}")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
