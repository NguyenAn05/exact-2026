# EXACT 2026 Option B on Modal

This repo serves one EXACT 2026 submission endpoint on Modal:

- `POST /predict`
- `GET /v1/models`

Architecture:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- Type 1 adapter: `NguyenAn05/qwen2.5-type1-grpo-lora`
- Type 2 adapter: `not-a-real-ai-guy/qwen2.5-type2-option-b-modes-lora`
- one shared vLLM server
- no model swapping during a request

`/predict` only selects the already-loaded adapter by model id:

- `type1-logic`
- `type2-physics`

## Current main result

Latest validated `main` result on `tests/datatest (1).jsonl`:

- total: `41/50`
- Type 1: `16/25`
- Type 2: `25/25`

## Repo layout

- `modal_exact2026.py`: Modal deploy entrypoint
- `app/`: API, graph pipeline, LLM client, normalization
- `scripts/fix_vllm_metrics.py`: robust vLLM/prometheus patch used at image build time
- `tests/`: regression and endpoint tests

## Deploy

Minimal flow:

```bash
pip install -r requirements-modal-local.txt
~/.local/bin/uvx modal setup
~/.local/bin/uvx modal secret create huggingface-secret HF_TOKEN=hf_xxx
python3 scripts/preflight_check_loras.py
~/.local/bin/uvx modal deploy modal_exact2026.py
~/.local/bin/uvx modal app rollover exact2026-optionb-qwen25
```

There is also a helper:

```bash
bash scripts/deploy.sh
```

## Verify live deployment

```bash
export PREDICT_URL="https://<workspace>--exact2026-optionb-qwen25-predict-api.modal.run/predict"
export VLLM_MODELS_URL="https://<workspace>--exact2026-optionb-qwen25-vllm-server.modal.run/v1/models"

bash scripts/modal_test_public.sh
python3 scripts/modal_latency_check.py
```

Expected `/v1/models` entries:

```txt
Qwen/Qwen2.5-7B-Instruct
type1-logic
type2-physics
```

## Local shape test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
MOCK_MODE=true pytest -q
```

## Notes

- `modal deploy` is not enough on its own. Run `modal app rollover` after deploy.
- `scripts/fix_vllm_metrics.py` is required because some vLLM/prometheus combinations break `/v1/models` and `/v1/chat/completions` with `_IncludedRouter` route inspection errors.
- Type 2 is currently stable on the visible test set. Remaining work is concentrated in Type 1.
