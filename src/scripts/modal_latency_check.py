#!/usr/bin/env python3
"""Latency check for public EXACT endpoints."""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from urllib.request import Request, urlopen

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

PREDICT_URL = os.environ.get("PREDICT_URL")
VLLM_MODELS_URL = os.environ.get("VLLM_MODELS_URL")

TYPE1 = {
    "query_id": "T1_LATENCY",
    "type": "type1",
    "query": "Is Student A eligible for graduation?",
    "premises": ["A student with at least 120 credits is eligible for graduation.", "Student A has completed 118 credits."],
    "options": ["Yes", "No", "Uncertain"],
}
TYPE2 = {
    "query_id": "T2_LATENCY",
    "type": "type2",
    "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
    "premises": [],
    "options": [],
}


def post_json(url: str, obj: dict) -> tuple[float, object]:
    body = json.dumps(obj).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    return time.perf_counter() - t0, data


def get_json(url: str) -> object:
    with urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def run_case(name: str, payload: dict, n: int = 3):
    times = []
    last = None
    for i in range(n):
        dt, data = post_json(PREDICT_URL, payload)  # type: ignore[arg-type]
        times.append(dt)
        last = data
        print(f"{name} run {i+1}/{n}: {dt:.2f}s")
    print(f"{name} avg={statistics.mean(times):.2f}s max={max(times):.2f}s")
    print(json.dumps(last, indent=2, ensure_ascii=False))
    if max(times) >= 60:
        print(f"FAIL: {name} exceeded 60s timeout")
        return False
    return True


def main() -> int:
    if not PREDICT_URL or not VLLM_MODELS_URL:
        print("Set PREDICT_URL and VLLM_MODELS_URL first.")
        return 2
    models = get_json(VLLM_MODELS_URL)
    print("/v1/models:")
    print(json.dumps(models, indent=2, ensure_ascii=False))
    text = json.dumps(models)
    for required in ["type1-logic", "type2-physics"]:
        if required not in text:
            print(f"FAIL: {required} not visible in /v1/models")
            return 3
    ok1 = run_case("TYPE1", TYPE1)
    ok2 = run_case("TYPE2", TYPE2)
    return 0 if ok1 and ok2 else 4


if __name__ == "__main__":
    raise SystemExit(main())
