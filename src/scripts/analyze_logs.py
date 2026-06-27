#!/usr/bin/env python3
"""Analyze downloaded log file from Modal.

Usage:
  python scripts/analyze_logs.py logs.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        path = Path("logs.jsonl")
    else:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"File not found: {path}")
        print("Download first:  uv run modal volume get exact2026-optionb-logs requests.jsonl > logs.jsonl")
        sys.exit(1)

    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"Total entries: {len(lines)}")
    print(f"Time range:    {_ts(lines[0])} → {_ts(lines[-1])}")
    print()

    events = Counter(l.get("event", "?") for l in lines)
    print("Events:")
    for ev, count in events.most_common():
        print(f"  {ev}: {count}")
    print()

    # Predict summaries
    predicts = [l for l in lines if l.get("event") == "predict"]
    if predicts:
        latencies = [l.get("latency_s", 0) for l in predicts]
        print(f"Predict ({len(predicts)}):")
        print(f"  Latency: min={min(latencies):.2f}s  max={max(latencies):.2f}s  avg={sum(latencies)/len(latencies):.2f}s")
        types = Counter(l.get("type", "?") for l in predicts)
        print(f"  Types: {dict(types)}")
        # HW samples if available
        sample = predicts[0]
        hw = sample.get("hw_before") or {}
        if hw:
            print(f"  HW (first request):")
            if "gpu" in hw:
                g = hw["gpu"]
                print(f"    GPU: {g.get('name','?')} util={g.get('gpu_util_pct')}% mem={g.get('mem_used_mib')}/{g.get('mem_total_mib')} MiB temp={g.get('temp_c')}°C")
            if "ram" in hw:
                r = hw["ram"]
                print(f"    RAM: {r.get('used_pct','?')}% used ({r.get('used_kb','?')} kB / {r.get('total_kb','?')} kB)")
            if "cpu" in hw:
                c = hw["cpu"]
                print(f"    CPU: user={c.get('user_pct')}% sys={c.get('system_pct')}% idle={c.get('idle_pct')}%")
            if "vllm_procs" in hw:
                print(f"    vLLM processes: {hw['vllm_procs']}")

    # LLM call summaries
    llm_calls = [l for l in lines if l.get("event") == "llm_call"]
    if llm_calls:
        latencies = [l.get("latency_s", 0) for l in llm_calls]
        print(f"\nLLM calls ({len(llm_calls)}):")
        print(f"  Latency: min={min(latencies):.2f}s  max={max(latencies):.2f}s  avg={sum(latencies)/len(latencies):.2f}s")
        pipes = Counter(l.get("pipeline", "?") for l in llm_calls)
        print(f"  Pipelines: {dict(pipes)}")
        # Show prompt lengths
        sys_lens = [len(l.get("system_prompt", "")) for l in llm_calls]
        user_lens = [len(l.get("user_prompt", "")) for l in llm_calls]
        resp_lens = [len(l.get("raw_response", "")) for l in llm_calls]
        print(f"  System prompt length: avg={sum(sys_lens)/len(sys_lens):.0f}")
        print(f"  User prompt length:   avg={sum(user_lens)/len(user_lens):.0f}")
        print(f"  Response length:      avg={sum(resp_lens)/len(resp_lens):.0f}")

    # Premise audit summaries
    audits = [l for l in lines if l.get("event") == "premise_audit"]
    if audits:
        print(f"\nPremise auditor corrections ({len(audits)}):")
        for a in audits:
            print(f"  {a.get('query_id','?')}: model={a.get('model_gave')} → auditor={a.get('auditor_fixed')}")


def _ts(entry: dict) -> str:
    t = entry.get("_timestamp", 0)
    import datetime
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
