from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the EXACT 2026 API wrapper.

    The server can talk to one shared vLLM server or to separate vLLM servers.
    For your current setup, use one vLLM server:
      - base: Qwen/Qwen2.5-7B-Instruct
      - LoRA adapter model id: type1-logic
      - Type 2 LoRA adapter model id: type2-physics
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8080, alias="APP_PORT")

    # Backward-compatible defaults. If TYPE1_/TYPE2_ values are empty, these are used.
    vllm_base_url: str = Field(default="http://localhost:8000/v1", alias="VLLM_BASE_URL")
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")
    model_name: str = Field(default="", alias="MODEL_NAME")

    # Type-specific routing. This lets Type 1 use the LoRA adapter and Type 2 use base/fallback.
    type1_vllm_base_url: str = Field(default="", alias="TYPE1_VLLM_BASE_URL")
    type1_model_name: str = Field(default="type1-logic", alias="TYPE1_MODEL_NAME")
    type2_vllm_base_url: str = Field(default="", alias="TYPE2_VLLM_BASE_URL")
    type2_model_name: str = Field(default="type2-physics", alias="TYPE2_MODEL_NAME")
    type2_fallback_to_llm: bool = Field(default=True, alias="TYPE2_FALLBACK_TO_LLM")

    # Keep this below the competition 60s timeout. If the model stalls, return a valid fallback.
    request_timeout_seconds: float = Field(default=50.0, alias="REQUEST_TIMEOUT_SECONDS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=768, alias="LLM_MAX_TOKENS")
    type1_llm_max_tokens: int = Field(default=1536, alias="TYPE1_LLM_MAX_TOKENS")
    type2_llm_max_tokens: int = Field(default=768, alias="TYPE2_LLM_MAX_TOKENS")

    # Useful for local API-format testing without a GPU/vLLM server.
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")

    def resolved_type1_base_url(self) -> str:
        return (self.type1_vllm_base_url or self.vllm_base_url).rstrip("/")

    def resolved_type2_base_url(self) -> str:
        return (self.type2_vllm_base_url or self.vllm_base_url).rstrip("/")

    def resolved_type1_model_name(self) -> str:
        return self.type1_model_name or self.model_name

    def resolved_type2_model_name(self) -> str:
        return self.type2_model_name or self.model_name


@lru_cache
def get_settings() -> Settings:
    return Settings()
