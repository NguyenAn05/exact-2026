#!/usr/bin/env bash
set -euo pipefail

TEAM_NAME="${1:-my_team}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist_submission"
PKG_DIR="$OUT_DIR/$TEAM_NAME"

rm -rf "$OUT_DIR"
mkdir -p "$PKG_DIR"

# 1) solution.pdf: replace this placeholder with your final one-page PDF.
if [[ -f "$ROOT_DIR/submission/solution.pdf" ]]; then
  cp "$ROOT_DIR/submission/solution.pdf" "$PKG_DIR/solution.pdf"
else
  echo "ERROR: missing submission/solution.pdf" >&2
  echo "Create your one-page PDF first." >&2
  exit 1
fi

# 2) source_code.zip
(
  cd "$ROOT_DIR"
  zip -r "$PKG_DIR/source_code.zip" \
    app requirements.txt requirements-modal-local.txt Dockerfile docker-compose.yml Caddyfile.example Makefile \
    modal_exact2026.py MODAL_DEPLOY.md scripts tests README.md .env.example pytest.ini \
    -x '*/__pycache__/*' '*.pyc' '.venv/*' 'dist_submission/*'
)

# 3) urls.txt
if [[ -f "$ROOT_DIR/submission/urls.txt" ]]; then
  cp "$ROOT_DIR/submission/urls.txt" "$PKG_DIR/urls.txt"
else
  echo "ERROR: missing submission/urls.txt" >&2
  exit 1
fi

# 4) notation_mapping.csv
if [[ -f "$ROOT_DIR/submission/notation_mapping.csv" ]]; then
  cp "$ROOT_DIR/submission/notation_mapping.csv" "$PKG_DIR/notation_mapping.csv"
else
  echo "ERROR: missing submission/notation_mapping.csv" >&2
  exit 1
fi

(
  cd "$OUT_DIR"
  zip -r "$TEAM_NAME.zip" "$TEAM_NAME"
)

echo "Created: $OUT_DIR/$TEAM_NAME.zip"
