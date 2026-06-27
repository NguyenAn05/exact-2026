#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Create log volume (Modal will auto-create if missing) ==="
~/.local/bin/uvx modal volume create exact2026-optionb-logs 2>/dev/null || true

echo ""
echo "=== 2. Preflight: Checking LoRA adapter bases ==="
python3 scripts/preflight_check_loras.py

echo ""
echo "=== 3. Deploying to Modal ==="
~/.local/bin/uvx modal deploy modal_exact2026.py
~/.local/bin/uvx modal app rollover exact2026-optionb-qwen25

echo ""
echo "=== 4. Get URLs ==="
echo "Run this to print endpoint URLs:"
echo ""
echo "  ~/.local/bin/uvx modal run modal_exact2026.py"
echo ""
echo "Save them to .env as:"
echo "  PREDICT_URL=https://...modal.run/predict"
echo "  VLLM_MODELS_URL=https://...modal.run/v1/models"
echo ""
echo "Then verify:"
echo "  bash scripts/modal_test_public.sh"
echo ""
echo "To download logs:"
echo "  ~/.local/bin/uvx modal volume ls exact2026-optionb-logs"
echo "  ~/.local/bin/uvx modal volume get exact2026-optionb-logs requests.jsonl"
