#!/usr/bin/env python3
"""Preflight check for EXACT Option B.

Verifies that both LoRA adapters declare the same base model as BASE_MODEL in modal_exact2026.py.
This script uses Hugging Face Hub metadata and should be run before modal deploy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def load_modal_constants():
    spec = importlib.util.spec_from_file_location("modal_exact2026", ROOT / "modal_exact2026.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load modal_exact2026.py")
    module = importlib.util.module_from_spec(spec)
    # Avoid requiring Modal import just to read constants if modal is not installed.
    # Fallback to text parsing if import fails.
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module.BASE_MODEL, module.TYPE1_LORA_REPO, module.TYPE2_LORA_REPO
    except Exception:
        text = (ROOT / "modal_exact2026.py").read_text()
        vals = {}
        for key in ["BASE_MODEL", "TYPE1_LORA_REPO", "TYPE2_LORA_REPO"]:
            for line in text.splitlines():
                if line.startswith(key):
                    vals[key] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        return vals["BASE_MODEL"], vals["TYPE1_LORA_REPO"], vals["TYPE2_LORA_REPO"]


def norm(s: str) -> str:
    return s.strip().lower().replace("_", "-").rstrip("/")


def adapter_base(repo_id: str) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json", token=token)
    data = json.loads(Path(path).read_text())
    base = data.get("base_model_name_or_path") or data.get("base_model")
    if not base:
        raise RuntimeError(f"{repo_id}: adapter_config.json does not contain base_model_name_or_path")
    return str(base)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that Type1/Type2 LoRA adapters match the Qwen2.5 base.")
    parser.add_argument("--allow-mismatch", action="store_true", help="Print warnings but exit 0.")
    args = parser.parse_args()

    expected_base, type1_repo, type2_repo = load_modal_constants()
    repos = [("type1-logic", type1_repo), ("type2-physics", type2_repo)]

    print(f"Expected shared base: {expected_base}")
    ok = True
    for name, repo in repos:
        try:
            base = adapter_base(repo)
        except Exception as exc:
            print(f"ERROR: {name} ({repo}) could not be checked: {exc}")
            ok = False
            continue
        status = "OK" if norm(base) == norm(expected_base) else "MISMATCH"
        print(f"{name}: repo={repo}")
        print(f"  declared base_model_name_or_path={base} [{status}]")
        if status != "OK":
            ok = False

    if ok:
        print("PASS: both LoRA adapters match the shared base. You can deploy Option B.")
        return 0

    print("FAIL: at least one LoRA adapter does not match BASE_MODEL.")
    print("Do not deploy this Option B build until the mismatching LoRA is retrained/rebuilt on Qwen/Qwen2.5-7B-Instruct.")
    return 0 if args.allow_mismatch else 2


if __name__ == "__main__":
    raise SystemExit(main())
