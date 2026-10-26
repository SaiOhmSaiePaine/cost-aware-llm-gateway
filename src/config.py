"""Centralised, validated runtime configuration loaded from environment / `.env`.

Every URL, secret, threshold, timeout and price lives here so no module
contains hardcoded magic numbers.
"""
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway settings. Field names map 1:1 to upper-case env vars."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---
    app_name: str = "Cost-Aware LLM Gateway"
    log_level: str = "INFO"

    # --- Redis semantic cache ---
    redis_url: str = "redis://localhost:6379/0"
    redis_socket_timeout_seconds: float = Field(default=0.5, gt=0)
    cache_index_name: str = "idx:llm_cache"
    cache_key_prefix: str = "cache:"
    cache_ttl_seconds: int = Field(default=86_400, gt=0)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # --- Embeddings (ONNX via FastEmbed) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, gt=0)

    # --- Complexity routing ---
    complexity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- Local model (Ollama / vLLM, OpenAI-compatible API) ---
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "llama3.2:1b"
    local_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- Cloud model (Google Gemini) ---
    gemini_api_key: SecretStr | None = None
    cloud_model: str = "gemini-3.5-flash-lite"
    cloud_timeout_seconds: float = Field(default=20.0, gt=0)

    # --- Circuit breaker ---
    breaker_failure_threshold: int = Field(default=3, ge=1)
    breaker_recovery_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- Cost ledger (USD per request; rough estimates for savings tracking) ---
    cloud_cost_per_request_usd: float = Field(default=0.005, ge=0.0)
    local_cost_per_request_usd: float = Field(default=0.0, ge=0.0)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
