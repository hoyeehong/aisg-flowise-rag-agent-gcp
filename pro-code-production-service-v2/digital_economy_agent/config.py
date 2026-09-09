"""
Runtime configuration.

Everything here is environment-driven with explicit defaults, so a deployment is
described by its environment rather than by code edits. Secrets are read from the
environment (populated from Secret Manager / Vault at container start) and are never
written to logs or the /metrics surface.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .gateway import ModelSpec

# Groq's OpenAI-compatible endpoint. The v1 agents ran openai/gpt-oss-20b here, so the
# default chain keeps behavioural continuity with the prototype being replaced.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODEL_CHAIN = "openai/gpt-oss-20b,llama-3.3-70b-versatile"

# Anchor the .env lookup to the service root, not the process working directory, so
# `uvicorn` started from the repo root behaves the same as from this directory. A .env
# in the CWD is read second and therefore wins, which keeps ad-hoc overrides working.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Service settings, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=(_SERVICE_ROOT / ".env", ".env"),
        extra="ignore",
        frozen=True,
    )

    environment: str = "development"
    log_level: str = "INFO"

    # --- model gateway ---------------------------------------------------------
    # Accepts the unprefixed GROQ_API_KEY as well as AGENT_GROQ_API_KEY. Third-party
    # credentials conventionally use the provider's own variable name, so forcing the
    # service prefix onto them is a needless source of "why is it not picking up my
    # key" confusion.
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("AGENT_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    groq_base_url: str = GROQ_BASE_URL
    model_chain: str = Field(
        default=DEFAULT_MODEL_CHAIN,
        description="Comma-separated Groq model ids, highest preference first.",
    )
    usd_per_million_input: float = 0.10
    usd_per_million_output: float = 0.50
    request_timeout_seconds: float = 60.0

    # --- agent behaviour -------------------------------------------------------
    default_top_k: int = Field(default=5, ge=1, le=50)
    max_revisions: int = Field(default=3, ge=0, le=20)

    # --- retrieval -------------------------------------------------------------
    # Phase 1 ships the in-memory retriever. Phase 2 introduces pgvector behind the
    # same Retriever protocol and this flag selects between them.
    use_in_memory_retriever: bool = True

    @field_validator("model_chain")
    @classmethod
    def _chain_not_empty(cls, value: str) -> str:
        if not [m.strip() for m in value.split(",") if m.strip()]:
            raise ValueError("model_chain must list at least one model")
        return value

    @property
    def models(self) -> list[ModelSpec]:
        """The gateway's fallback chain, in preference order."""
        return [
            ModelSpec(
                provider="groq",
                model=name.strip(),
                usd_per_million_input=self.usd_per_million_input,
                usd_per_million_output=self.usd_per_million_output,
            )
            for name in self.model_chain.split(",")
            if name.strip()
        ]

    @property
    def has_model_credentials(self) -> bool:
        """Whether a real provider call could succeed. Drives /readyz."""
        return bool(self.groq_api_key.get_secret_value())


def get_settings() -> Settings:
    """Build settings. Not cached, so tests can vary the environment per case."""
    return Settings()
