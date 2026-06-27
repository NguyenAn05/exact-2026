#!/usr/bin/env bash
# Comprehensive test for EXACT 2026 Option B — sends all question types,
# prints responses, and optionally fetches recent logs from Modal.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && set -a && source "$SCRIPT_DIR/.env" && set +a

: "${PREDICT_URL:?Set PREDICT_URL in .env or export it}"
: "${VLLM_MODELS_URL:?Set VLLM_MODELS_URL in .env or export it}"

PASS=0
FAIL=0

green()  { printf "\033[32m%s\033[0m\n" "$1"; }
red()    { printf "\033[31m%s\033[0m\n" "$1"; }
bold()   { printf "\033[1m%s\033[0m\n" "$1"; }
yellow() { printf "\033[33m%s\033[0m\n" "$1"; }

request() {
  local label="$1" payload="$2"
  echo ""
  bold "──────────────────────────────────────────────"
  bold "  $label"
  bold "──────────────────────────────────────────────"
  echo "Request:"
  echo "$payload" | python -m json.tool 2>/dev/null || echo "$payload"
  echo ""
  local start; start=$(date +%s%N)
  RESP=$(curl -sS --max-time 120 -X POST "$PREDICT_URL" \
    -H "Content-Type: application/json" -d "$payload")
  local end; end=$(date +%s%N)
  local ms=$(( (end - start) / 1000000 ))
  echo "Response (${ms}ms):"
  echo "$RESP" | python -m json.tool 2>/dev/null || echo "$RESP"
  # Basic health check
  if echo "$RESP" | python -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    green "  ✓ valid JSON"
    ((PASS++))
  else
    red "  ✗ invalid JSON"
    ((FAIL++))
  fi
}

bold "================================================"
bold "  EXACT 2026 — Full Integration Test Suite"
bold "  Server: $PREDICT_URL"
bold "================================================"

# ----- Type 1: Yes/No -----
request "Type1 — Yes/No (Asha active contributor?)" '{
  "query_id": "type1_yesno",
  "type": "type1",
  "query": "Is Asha listed as an active contributor?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": ["Yes", "No", "Uncertain"]
}'

# ----- Type 1: Uncertain -----
request "Type1 — Uncertain (budget approval?)" '{
  "query_id": "type1_uncertain",
  "type": "type1",
  "query": "Does Asha have budget approval?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": ["Yes", "No", "Uncertain"]
}'

# ----- Type 1: Number -----
request "Type1 — Number (enrolled participants?)" '{
  "query_id": "type1_number",
  "type": "type1",
  "query": "How many enrolled participants does Study Alpha have?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": []
}'

# ----- Type 1: Text (free-form) -----
request "Type1 — Text (who may join?)" '{
  "query_id": "type1_text",
  "type": "type1",
  "query": "Which researcher may join Study Alpha?",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": []
}'

# ----- Type 1: Multiple Choice -----
request "Type1 — Multiple Choice" '{
  "query_id": "type1_mc",
  "type": "type1",
  "query": "Based on the premises, which option is logically supported?\nA. Asha may join Study Alpha\nB. Asha cannot handle participant data\nC. Asha has budget approval\nD. Study Alpha has 20 enrolled participants",
  "premises": [
    "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
    "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
    "Every researcher who may join Study Alpha is listed as an active contributor.",
    "Asha completed ethics training.",
    "Asha has lab access.",
    "Asha has supervisor approval.",
    "Study Alpha has 12 enrolled participants.",
    "No premise states whether Asha has budget approval."
  ],
  "options": ["A", "B", "C", "D"]
}'

# ----- Type 1: Custom logic puzzle -----
request "Type1 — Custom (access grant)" '{
  "query_id": "type1_custom",
  "type": "type1",
  "query": "Is Bob granted access to the server room?",
  "premises": [
    "If an employee has a keycard and completed security training, then that employee is granted access to the server room.",
    "Every employee who is granted access to the server room must sign the logbook.",
    "If an employee is on probation, then that employee has not completed security training.",
    "Bob is an employee.",
    "Bob has a keycard.",
    "Bob is on probation."
  ],
  "options": ["Yes", "No", "Uncertain"]
}'

# ----- Type 2: Parallel resistors -----
request "Type2 — Parallel resistors" '{
  "query_id": "type2_parallel",
  "type": "type2",
  "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are connected in parallel across a 12V battery. Find the total current flowing from the battery.",
  "premises": [],
  "options": []
}'

# ----- Type 2: RC circuit -----
request "Type2 — RC time constant" '{
  "query_id": "type2_rc",
  "type": "type2",
  "query": "A 10 uF capacitor is charged through a 2 kohm resistor. What is the RC time constant in seconds?",
  "premises": [],
  "options": []
}'

# ----- Summary -----
echo ""
bold "================================================"
bold "  Results: $PASS passed, $FAIL failed"
bold "================================================"

if [ "$FAIL" -gt 0 ]; then
  red "  Some tests failed!"
  exit 1
else
  green "  All tests passed!"
fi

echo ""
yellow "To download logs from Modal:"
echo "  uv run modal volume ls exact2026-optionb-logs"
echo "  uv run modal volume get exact2026-optionb-logs requests.jsonl > logs.jsonl"
echo ""
yellow "To download & analyze logs:"
echo "  uv run modal volume get exact2026-optionb-logs requests.jsonl > logs.jsonl"
echo "  python scripts/analyze_logs.py logs.jsonl"
