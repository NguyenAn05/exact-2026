# AGENT.md

Project state for the next coding pass.

## Main branch

- branch: `main`
- latest pushed commit: `907f68b`
- deployment entrypoint: `modal_exact2026.py`
- Modal app: `exact2026-optionb-qwen25`

## Current architecture

### Shared serving plan

- base model: `Qwen/Qwen2.5-7B-Instruct`
- Type 1 LoRA id: `type1-logic`
- Type 2 LoRA id: `type2-physics`
- one shared vLLM server
- one FastAPI `/predict` wrapper

### Type 1

Pipeline on `main` is intentionally LLM-first:

1. `llm_type1`
2. `sandbox`
3. `check_type1`
4. retry if needed
5. `formatter`

The symbolic Type 1 experiments were tested and not merged into `main`.

Relevant files:

- `app/pipelines/graph/node_type1.py`
- `app/pipelines/graph/node_check_type1.py`
- `app/pipelines/graph/node_formatter.py`
- `app/pipelines/graph/__init__.py`

### Type 2

Pipeline on `main` is hybrid:

1. deterministic fast path if covered
2. code-generation LLM path otherwise
3. sandbox execution
4. direct LLM fallback if code path fails

Relevant files:

- `app/pipelines/graph/node_type2.py`
- `app/utils/normalize.py`
- `app/utils/sandbox.py`

## Important deploy behavior

### vLLM metrics patch

The repo uses `scripts/fix_vllm_metrics.py` during image build.

Reason:

- some vLLM/prometheus setups break `/v1/models` and `/v1/chat/completions`
- error signature:
  - `'_IncludedRouter' object has no attribute 'path'`

Do not remove this patch unless the target vLLM version is revalidated.

### Rollover

After deploy, run:

```bash
~/.local/bin/uvx modal app rollover exact2026-optionb-qwen25
```

Without rollover, the public endpoint may continue serving an older revision.

## Current validated result on main

Dataset:

- `tests/datatest (1).jsonl`

Latest validated `main` result:

- total: `41/50`
- Type 1: `16/25`
- Type 2: `25/25`
- avg latency: `18.3s`

Remaining Type 1 failures on `main`:

- `T1_0031`
- `T1_0025`
- `T1_0027`
- `T1_0033`
- `T1_0013`
- `T1_0042`
- `T1_0024`
- `T1_0032`
- `T1_0048`

## Experiments already run

### Symbolic Type 1 primary route

Branch:

- `experiment/type1-symbolic-type2-format`

Outcome:

- not merged
- Type 2 became perfect on the visible set
- Type 1 dropped too hard

### Trust model-generated premises + 30s Type 1 retry budget

Branch:

- `experiment/type1-trust-premises-30s`

Outcome on visible set:

- total: `42/50`
- Type 1: `17/25`
- Type 2: `25/25`

Reason not merged:

- `T1_0032` timed out badly in one run
- latency rose to about `24s`
- better score, weaker stability

## Recommended next work

Priority:

1. Fix the 9 remaining Type 1 failures on `main`
2. Improve Type 1 stability before trying more symbolic routing
3. Keep Type 2 unchanged unless a regression appears

Avoid:

- reintroducing the old `scripts/patch_vllm_metrics.py` approach
- merging symbolic Type 1 as the default route without new evidence
