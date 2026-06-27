#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:8080/predict}"

curl -sS -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -d '{
    "query_id": "T2_0001",
    "type": "type2",
    "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
    "premises": [],
    "options": []
  }' | python -m json.tool
