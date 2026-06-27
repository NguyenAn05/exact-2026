#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TYPE1_LORA_NAME="${TYPE1_LORA_NAME:-type1-logic}"
TYPE1_LORA_REPO="${TYPE1_LORA_REPO:-NguyenAn05/qwen2.5-type1-grpo-lora}"
TYPE2_LORA_NAME="${TYPE2_LORA_NAME:-type2-physics}"
TYPE2_LORA_REPO="${TYPE2_LORA_REPO:-not-a-real-ai-guy/qwen2.5-type2-option-b-modes-lora}"
PORT="${PORT:-8000}"

cat <<EOF
Starting vLLM with multi-LoRA:
  base:         $BASE_MODEL
  type1 LoRA:   $TYPE1_LORA_NAME=$TYPE1_LORA_REPO
  type2 LoRA:   $TYPE2_LORA_NAME=$TYPE2_LORA_REPO

IMPORTANT: both LoRA adapters must be trained from the same base model.
Check with: python scripts/check_lora_base.py $TYPE1_LORA_REPO $TYPE2_LORA_REPO
EOF

vllm serve "$BASE_MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$BASE_MODEL" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-loras 2 \
  --lora-modules \
    "$TYPE1_LORA_NAME=$TYPE1_LORA_REPO" \
    "$TYPE2_LORA_NAME=$TYPE2_LORA_REPO"
