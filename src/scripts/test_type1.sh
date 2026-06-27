#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://localhost:8080/predict}"

curl -sS -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -d '{
    "query_id": "T1_0001",
    "type": "type1",
    "query": "Is Student A eligible for graduation?",
    "premises": ["A student with >= 120 credits is eligible.", "Student A has 118 credits."],
    "options": ["Yes", "No", "Uncertain"]
  }' | python -m json.tool
