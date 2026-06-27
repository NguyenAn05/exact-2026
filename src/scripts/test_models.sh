#!/usr/bin/env bash
set -euo pipefail
curl -s http://localhost:8000/v1/models | python -m json.tool
