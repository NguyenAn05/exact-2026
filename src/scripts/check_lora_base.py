#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    print("Missing dependency: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(2) from exc


def read_base(repo_or_path: str) -> str:
    p = Path(repo_or_path)
    if p.exists():
        cfg_path = p / "adapter_config.json"
    else:
        cfg_path = Path(hf_hub_download(repo_id=repo_or_path, filename="adapter_config.json"))
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return data.get("base_model_name_or_path", "<missing>")


if len(sys.argv) < 2:
    print("Usage: python scripts/check_lora_base.py <lora_repo_or_path> [<lora_repo_or_path> ...]", file=sys.stderr)
    raise SystemExit(2)

bases = {}
for repo in sys.argv[1:]:
    try:
        base = read_base(repo)
    except Exception as exc:
        print(f"{repo}: ERROR: {exc}")
        continue
    bases[repo] = base
    print(f"{repo}: base_model_name_or_path = {base}")

unique = set(bases.values())
if len(unique) > 1:
    print("\nWARNING: LoRA adapters do not share the same base model. Do NOT serve them in one shared vLLM base.", file=sys.stderr)
    raise SystemExit(1)
print("\nOK: all checked LoRA adapters declare the same base model.")
