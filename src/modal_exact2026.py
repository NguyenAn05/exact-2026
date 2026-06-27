"""Modal deployment for EXACT 2026 — Option B.

Architecture:
  - ONE vLLM server, kept warm during the grading slot.
  - ONE shared base model: Qwen/Qwen2.5-7B-Instruct.
  - TWO LoRA adapters loaded at server startup:
      * type1-logic
      * type2-physics
  - NO model load/unload/swap during a query. /predict only selects the already-loaded
    LoRA adapter via the OpenAI-compatible `model` field.

Deploy:
  pip install -r requirements-modal-local.txt
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_xxx
  python scripts/preflight_check_loras.py
  modal deploy modal_exact2026.py
  modal run modal_exact2026.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "exact2026-optionb-qwen25"

# ---- Model plan: Option B -------------------------------------------------
# IMPORTANT: both adapters must have adapter_config.json -> base_model_name_or_path
# compatible with BASE_MODEL. Run scripts/preflight_check_loras.py before deploying.
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

TYPE1_LORA_NAME = "type1-logic"
TYPE1_LORA_REPO = "NguyenAn05/qwen2.5-type1-grpo-lora"

TYPE2_LORA_NAME = "type2-physics"
# Keep this repo if you have rebuilt/retrained it on Qwen/Qwen2.5-7B-Instruct.
# If adapter_config.json still declares a Qwen3 base, do not deploy until fixed.
TYPE2_LORA_REPO = "not-a-real-ai-guy/qwen2.5-type2-option-b-modes-lora"

# L40S is usually enough for Qwen2.5-7B + 2 LoRAs. For safety/latency use A100-80GB/H100/H200.
GPU_TYPE = "L40S"
N_GPU = 1
VLLM_PORT = 8000
MINUTES = 20

# Keep 0 during normal testing to avoid surprise GPU cost.
# Run scripts/modal_keep_warm_on.py before the grading slot to set min_containers=1.
MIN_CONTAINERS = 0
TARGET_INPUTS = 1  # EXACT evaluation is sequential; avoid queueing/latency
MAX_INPUTS = 2  # allow /v1/models or a tiny burst without overloading GPU

LOCAL_ROOT = Path(__file__).parent

modal_app = modal.App(APP_NAME)
app = modal_app  # modal deploy expects top-level `app`

hf_cache_vol = modal.Volume.from_name("exact2026-optionb-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("exact2026-optionb-vllm-cache", create_if_missing=True)
log_vol = modal.Volume.from_name("exact2026-optionb-logs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        # If this exact pin fails in your Modal workspace, change to "vllm" and redeploy.
        "vllm==0.21.0",
        "huggingface_hub[hf_xet]>=0.36.0",
        "sympy==1.13.3",
        "z3-solver==4.13.3.0",
    )
    .add_local_file(
        LOCAL_ROOT / "scripts" / "fix_vllm_metrics.py",
        remote_path="/root/fix_vllm_metrics.py",
        copy=True,
    )
    .run_commands("python /root/fix_vllm_metrics.py")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]==0.115.6",
        "uvicorn[standard]==0.34.0",
        "httpx==0.28.1",
        "pydantic==2.10.4",
        "pydantic-settings==2.7.1",
        "python-dotenv==1.0.1",
        "sympy==1.13.3",
        "z3-solver==4.13.3.0",
        "langchain-core",
        "langgraph",
    )
    .add_local_dir(LOCAL_ROOT / "app", remote_path="/root/exact/app")
)


@modal_app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    timeout=15 * MINUTES,
    startup_timeout=15 * MINUTES,
    scaledown_window=3600,  # max allowed by Modal
    min_containers=MIN_CONTAINERS,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret],
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def vllm_server():
    """Serve Qwen2.5 base + both LoRAs from startup; no mid-query swapping."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--served-model-name",
        BASE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--tensor-parallel-size",
        str(N_GPU),
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "2",
        "--lora-modules",
        f"{TYPE1_LORA_NAME}={TYPE1_LORA_REPO}",
        f"{TYPE2_LORA_NAME}={TYPE2_LORA_REPO}",
        "--uvicorn-log-level",
        "info",
    ]

    print("Starting vLLM Option B:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


@modal_app.function(
    image=api_image,
    timeout=150,
    scaledown_window=3600,  # max allowed by Modal
    min_containers=0,
    max_containers=2,
    volumes={"/root/exact/logs": log_vol},
)
@modal.asgi_app()
def predict_api():
    """Serve the EXACT /predict wrapper."""
    import os
    import sys

    sys.path.insert(0, "/root/exact")

    vllm_url = os.environ.get("PUBLIC_VLLM_URL") or vllm_server.get_web_url()
    if not vllm_url:
        raise RuntimeError("Could not resolve vLLM web URL from Modal")

    # One shared vLLM server; /predict only changes the model id to select loaded LoRA.
    os.environ["VLLM_BASE_URL"] = vllm_url.rstrip("/") + "/v1"
    os.environ["TYPE1_VLLM_BASE_URL"] = vllm_url.rstrip("/") + "/v1"
    os.environ["TYPE2_VLLM_BASE_URL"] = vllm_url.rstrip("/") + "/v1"
    os.environ["TYPE1_MODEL_NAME"] = TYPE1_LORA_NAME
    os.environ["TYPE2_MODEL_NAME"] = TYPE2_LORA_NAME
    os.environ["VLLM_API_KEY"] = "EMPTY"
    os.environ["MOCK_MODE"] = "false"
    os.environ["TYPE2_FALLBACK_TO_LLM"] = "true"
    os.environ["REQUEST_TIMEOUT_SECONDS"] = "50"
    os.environ["LLM_TEMPERATURE"] = "0"
    os.environ["LLM_MAX_TOKENS"] = "768"
    os.environ["TYPE1_LLM_MAX_TOKENS"] = "1536"
    os.environ["TYPE2_LLM_MAX_TOKENS"] = "768"
    os.environ["LOG_FILE_PATH"] = "/root/exact/logs/requests.jsonl"

    from app.main import app as fastapi_app

    return fastapi_app


@modal_app.local_entrypoint()
def show_urls():
    """Print URLs after deploy/run. Use these in submission/urls.txt."""
    print("Prediction URL:", predict_api.get_web_url() + "/predict")
    print("vLLM models URL:", vllm_server.get_web_url() + "/v1/models")
    print("vLLM chat URL:", vllm_server.get_web_url() + "/v1/chat/completions")
