"""Turn off Modal warm containers after grading.

Run this after your EXACT slot:
  python scripts/modal_keep_warm_off.py
"""
import modal

APP_NAME = "exact2026-optionb-qwen25"

vllm = modal.Function.from_name(APP_NAME, "vllm_server")
api = modal.Function.from_name(APP_NAME, "predict_api")

vllm.update_autoscaler(min_containers=0, max_containers=1, scaledown_window=15 * 60)
api.update_autoscaler(min_containers=0, max_containers=2, scaledown_window=15 * 60)

print("Warm mode OFF: min_containers=0")
