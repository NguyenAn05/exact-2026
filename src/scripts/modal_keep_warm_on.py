"""Keep Modal endpoints warm during your EXACT grading slot.

Run this 10-20 minutes before your slot:
  python scripts/modal_keep_warm_on.py

Warning: min_containers=1 for vLLM keeps a GPU container running and bills continuously.
"""
import modal

APP_NAME = "exact2026-optionb-qwen25"

vllm = modal.Function.from_name(APP_NAME, "vllm_server")
api = modal.Function.from_name(APP_NAME, "predict_api")

vllm.update_autoscaler(min_containers=1, max_containers=1, scaledown_window=3600)
api.update_autoscaler(min_containers=1, max_containers=2, scaledown_window=3600)

print("Warm mode ON: vLLM min_containers=1, API min_containers=1")
print("vLLM models URL:", vllm.get_web_url() + "/v1/models")
print("Prediction URL:", api.get_web_url() + "/predict")
