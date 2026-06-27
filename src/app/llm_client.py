from __future__ import annotations

import time

import httpx

from app.config import Settings
from app.logging_utils import log_entry


class VLLMClient:
    """Small async client for vLLM's OpenAI-compatible /v1/chat/completions API."""

    def __init__(self, settings: Settings, *, base_url: str | None = None, model_name: str | None = None):
        self.settings = settings
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model_name = model_name or ""
        self.headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}

    async def get_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        if self.settings.model_name:
            return self.settings.model_name
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/models", headers=self.headers)
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                raise RuntimeError("/v1/models returned no models")
            return data[0]["id"]

    async def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        query_id: str = "",
        pipeline: str = "",
        max_tokens: int | None = None,
    ) -> str:
        model = await self.get_model_name()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.llm_max_tokens,
            # Some vLLM/chat templates support this; if your model rejects it, remove it.
            "response_format": {"type": "json_object"},
        }
        t0 = time.time()
        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        raw = data["choices"][0]["message"]["content"]
        latency = time.time() - t0

        log_entry({
            "event": "llm_call",
            "query_id": query_id,
            "pipeline": pipeline,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw,
            "latency_s": round(latency, 3),
        })
        return raw
