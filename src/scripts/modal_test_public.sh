#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

: "${PREDICT_URL:?Set PREDICT_URL in .env or export it}"
: "${VLLM_MODELS_URL:?Set VLLM_MODELS_URL in .env or export it}"

echo "== /v1/models =="
MODELS_JSON=$(curl -sS "$VLLM_MODELS_URL")
echo "$MODELS_JSON" | python -m json.tool

echo "$MODELS_JSON" | grep -q 'type1-logic' || { echo 'ERROR: type1-logic not in /v1/models'; exit 2; }
echo "$MODELS_JSON" | grep -q 'type2-physics' || { echo 'ERROR: type2-physics not in /v1/models'; exit 2; }

echo "== Type 1 =="
time curl -sS -X POST "$PREDICT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "T1_TEST",
    "type": "type1",
    "query": "Is Student A eligible for graduation?",
    "premises": ["A student with >= 120 credits is eligible.", "Student A has 118 credits."],
    "options": ["Yes", "No", "Uncertain"]
  }' | python -m json.tool

echo "== Type 2 =="
time curl -sS -X POST "$PREDICT_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "T2_TEST",
    "type": "type2",
    "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
    "premises": [],
    "options": []
  }' | python -m json.tool
