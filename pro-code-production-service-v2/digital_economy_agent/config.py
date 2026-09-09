"""
Runtime configuration.

Everything here is environment-driven with explicit defaults, so a deployment is
described by its environment rather than by code edits. Secrets are read from the
environment (populated from Secret Manager / Vault at container start) and are never
written to logs or the /metrics surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .gateway import ModelSpec

if TYPE_CHECKING:
    from .retrieval import HybridConfig

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
    service_name: str = "digital-economy-agent"

    # --- observability ---------------------------------------------------------
    # Empty disables tracing. Deliberately not best-effort: an exporter pointing at a
    # collector that is not there retries in the background and adds latency to every
    # request, turning a missing telemetry sidecar into a user-visible problem.
    otlp_endpoint: str = ""
    trace_to_console: bool = False

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
    # Selects between the Phase 1 in-memory retriever and the Phase 2 pgvector hybrid
    # retriever, both behind the same Retriever protocol. Defaults to in-memory so the
    # service still starts with no database, which keeps the test suite and a plain
    # `uvicorn` run free of infrastructure.
    use_in_memory_retriever: bool = True

    # Postgres, shared by the vector store and the LangGraph checkpointer.
    postgres_dsn: str = ""
    embedding_dimensions: int = Field(default=768, ge=64, le=3072)
    embedding_model: str = "gemini-embedding-001"
    # Embeddings come from Google; the chat models come from Groq. Two providers, two
    # keys -- both accepted unprefixed for the same reason as the Groq key.
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("AGENT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    tenant_id: str = "default"

    # --- tenancy and least privilege -------------------------------------------
    # The service queries as a non-superuser role so the row-level-security policy on
    # `chunks` actually applies: superusers ignore policies, and FORCE ROW LEVEL
    # SECURITY only subjects the table owner. DDL needs ownership, so bootstrap uses a
    # separate administrative DSN. Empty falls back to `postgres_dsn`, which keeps a
    # single-DSN local setup working -- at the cost of RLS being inert there, which
    # `/readyz` now reports rather than leaving it to be discovered.
    postgres_admin_dsn: str = ""
    # When set, bootstrap creates and grants this role. Left empty in production, where
    # the role and its password are managed by Terraform.
    postgres_app_role: str = ""
    postgres_app_password: SecretStr = SecretStr("")
    # Opting in to the deterministic embedder is a configuration decision, not a
    # degradation: it is how retrieval is evaluated in CI without an API key. Left
    # false, a missing embedding credential correctly reports the service as degraded,
    # so an operator who *meant* to have embeddings still gets a signal.
    allow_hashing_embedder: bool = False

    @property
    def admin_dsn(self) -> str:
        """DSN for DDL and role management; the app DSN when none is configured."""
        return self.postgres_admin_dsn or self.postgres_dsn

    # --- hybrid search ---------------------------------------------------------
    retrieval_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    retrieval_vector_weight: float = Field(default=1.0, ge=0.0)
    retrieval_lexical_weight: float = Field(default=1.0, ge=0.0)
    retrieval_mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    retrieval_enable_mmr: bool = True
    # 0 disables LLM query rewriting, which costs an extra model call per retrieval.
    retrieval_rewrite_variants: int = Field(default=0, ge=0, le=5)

    # --- durability ------------------------------------------------------------
    # With Postgres configured, paused human-review runs survive a restart. Without it
    # the graph falls back to an in-process saver and a restart loses them.
    use_postgres_checkpointer: bool = True

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
        """Whether a real chat completion could succeed. Drives /readyz."""
        return bool(self.groq_api_key.get_secret_value())

    @property
    def has_embedding_credentials(self) -> bool:
        """Whether real embeddings are available."""
        return bool(self.gemini_api_key.get_secret_value())

    @property
    def embeddings_satisfied(self) -> bool:
        """
        Whether the embedding configuration is coherent, which is what readiness needs.

        Satisfied by either a real credential or an explicit opt-in to the deterministic
        embedder. Asserting the credential unconditionally would mark a deliberately
        quota-free deployment as degraded.
        """
        return self.has_embedding_credentials or self.allow_hashing_embedder

    @property
    def postgres_configured(self) -> bool:
        return bool(self.postgres_dsn.strip())

    @property
    def hybrid_config(self) -> HybridConfig:
        from .retrieval import HybridConfig

        return HybridConfig(
            candidate_multiplier=self.retrieval_candidate_multiplier,
            vector_weight=self.retrieval_vector_weight,
            lexical_weight=self.retrieval_lexical_weight,
            mmr_lambda=self.retrieval_mmr_lambda,
            enable_mmr=self.retrieval_enable_mmr,
            rewrite_variants=self.retrieval_rewrite_variants,
        )


def get_settings() -> Settings:
    """Build settings. Not cached, so tests can vary the environment per case."""
    return Settings()
