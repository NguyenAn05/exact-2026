# Modal Deploy Guide

## 1. Local setup

```bash
pip install -r requirements-modal-local.txt
~/.local/bin/uvx modal setup
```

## 2. Hugging Face secret

```bash
~/.local/bin/uvx modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

If the secret already exists, update it in the Modal dashboard or recreate it.

## 3. Verify adapter compatibility

```bash
python3 scripts/preflight_check_loras.py
```

Do not deploy if this fails. Both adapters must match:

- `Qwen/Qwen2.5-7B-Instruct`

## 4. Deploy and switch traffic

```bash
~/.local/bin/uvx modal deploy modal_exact2026.py
~/.local/bin/uvx modal app rollover exact2026-optionb-qwen25
```

## 5. Verify

```bash
export PREDICT_URL="https://...modal.run/predict"
export VLLM_MODELS_URL="https://...modal.run/v1/models"

bash scripts/modal_test_public.sh
python3 scripts/modal_latency_check.py
```

Expected `/v1/models` entries:

```txt
Qwen/Qwen2.5-7B-Instruct
type1-logic
type2-physics
```

## 6. Keep warm during grading

Before the grading slot:

```bash
python3 scripts/modal_keep_warm_on.py
```

After the slot:

```bash
python3 scripts/modal_keep_warm_off.py
```

## 7. If `/v1/models` returns 500

Check whether the image was built with `scripts/fix_vllm_metrics.py`.

Known failure signature:

- `'_IncludedRouter' object has no attribute 'path'`

That means the old prometheus/vLLM route bug is back.
