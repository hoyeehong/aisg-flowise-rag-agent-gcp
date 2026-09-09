# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Build stage: resolve and install dependencies into a self-contained venv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer first: it changes far less often than application code, so it stays
# cached across source edits. --frozen fails the build if uv.lock is stale, which keeps
# the image reproducible rather than silently resolving new versions.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY digital_economy_agent ./digital_economy_agent
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Runtime stage: no uv, no build tooling, no package manager caches.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Non-root, no login shell, no home directory contents to write to.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# Drop the base image's build tooling. The application runs entirely from /app/.venv
# and needs neither pip nor setuptools at runtime, while setuptools' vendored copies of
# jaraco.context and wheel are a recurring source of HIGH CVEs (CVE-2026-23949,
# CVE-2026-24049). Removing them is a smaller attack surface *and* fewer findings to
# triage on every scan.
RUN python -m pip uninstall -y pip setuptools wheel 2>/dev/null || true; \
    rm -rf /usr/local/lib/python3.11/site-packages/pip \
           /usr/local/lib/python3.11/site-packages/setuptools \
           /usr/local/lib/python3.11/site-packages/wheel \
           /usr/local/lib/python3.11/site-packages/pkg_resources \
           /usr/local/lib/python3.11/site-packages/*.dist-info \
           /usr/local/bin/pip* /usr/local/bin/wheel

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_ENVIRONMENT=production

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/digital_economy_agent /app/digital_economy_agent

USER app
EXPOSE 8080

# Liveness only. Readiness (/readyz) is checked by the orchestrator, which must be able
# to pull an instance out of rotation for a missing credential without restarting it.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"

# Single worker per container: scale with replicas, not in-process workers, so the
# orchestrator's autoscaler and the cost meter both see one unit of work per pod.
CMD ["uvicorn", "--factory", "digital_economy_agent.api.app:create_app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
