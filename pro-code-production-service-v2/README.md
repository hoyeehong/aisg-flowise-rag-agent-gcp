# v2 — Pro-Code Production Service

**Status:** Phase 1 delivered — the service runs, is typed, and is covered by 34 tests.
Phases 2–5 are still design-stage; the naming conventions below were fixed first so
Phase 1 landed in the right shape rather than being reorganised later.

v2 re-platforms the [v1 Flowise prototype](../low-code-rapid-prototype-v1/README.md) as a
code-owned service: a typed API over a LangGraph agent, a RAG pipeline declared in code rather
than UI state, and an evaluation harness that calls the running system. v1 stays alive as the
low-code demo surface — the guiding principle is *pro-code services first, low-code surfaces only
where appropriate*.

---

## 1. Naming the tier itself

v1 is named `[approach]-[maturity]-v[n]`. Keeping that construction makes the progression readable
at a glance in the repo root, which is where a reviewer forms their first impression.

| Candidate | Reads as | Assessment |
| :--- | :--- | :--- |
| **`pro-code-production-service-v2`** | Exact parallel: `low-code`→`pro-code`, `rapid-prototype`→`production-service` | **Recommended.** The contrast with v1 is legible without opening either folder, and "service" is the word the target JD leads with (*"you'll ship services, not just notebooks"*). |
| `production-agent-platform-v2` | Emphasises platform breadth | "Platform" overclaims for a single service, and drops the pro-code/low-code contrast that makes v1 meaningful. |
| `code-first-production-service-v2` | Same idea, different idiom | Accurate but longer; "code-first" is a weaker antonym of "low-code" than "pro-code". |
| `v2-agentic-service` | Version-led sort order | Sorts predictably but says nothing about what changed between tiers. |

This folder currently uses the recommended name. To switch:

```bash
git mv pro-code-production-service-v2 <new-name>
```

> [!IMPORTANT]
> **A hyphenated folder is not an importable Python package.** `pro-code-production-service-v2`
> cannot be `import`ed, and neither can any `-v2` suffix. Keep the tier folder hyphenated for
> readability, and put a single underscore-named package *inside* it as the import root
> (`digital_economy_agent/`). This is why the layout below separates the two.

---

## 2. Target layout and naming conventions

`✓` exists as of Phase 1; the rest is the agreed shape for later phases.

```
pro-code-production-service-v2/
├── digital_economy_agent/          ✓ import root — underscores, singular domain noun
│   ├── api/                        ✓ FastAPI routes, schemas, service layer
│   ├── agents/                     ✓ LangGraph graph, nodes, state
│   ├── tools/                      ✓ typed tool adapters (Retriever protocol + in-memory impl)
│   ├── gateway/                    ✓ model routing, retries, fallback chain, cost meter
│   ├── prompts/                    ✓ versioned prompt files + pinned loader
│   ├── config.py                   ✓ environment-driven settings
│   ├── retrieval/                  — chunking, embeddings, hybrid search, re-ranking (P2)
│   ├── ingestion/                  — Prefect flows, PII redaction, governance (P2)
│   └── observability/              ✓ package placeholder; OTel + Prometheus land in P4
├── tests/
│   ├── unit/                       ✓ graph transitions, gateway taxonomy
│   ├── integration/                ✓ real routes + real graph, scripted providers
│   └── load/                       — k6 / Locust scenarios (P4)
├── evals/                          — golden sets + harness against the live API (P3)
├── infra/terraform/                — one module per resource group (P4)
├── charts/digital-economy-agent/   — Helm chart, matches the K8s object name (P4)
├── Dockerfile                      ✓ multi-stage, non-root, healthcheck
└── pyproject.toml                  ✓
```

### Rules worth stating once

| Scope | Convention | Example |
| :--- | :--- | :--- |
| Python package / modules | `snake_case`, singular domain nouns | `digital_economy_agent/retrieval/hybrid_search.py` |
| Python classes | `PascalCase`, no `Manager`/`Helper`/`Util` suffixes | `HybridRetriever`, `ModelGateway` |
| Tool adapters | `<verb>_<noun>.py`, one adapter per module | `tools/search_documents.py`, `tools/query_sql.py` |
| LangGraph nodes | verb-first, matching the node's single responsibility | `research`, `write_draft`, `await_review`, `revise` |
| Folders (non-package) | `kebab-case` | `charts/digital-economy-agent/` |
| Container image | `<registry>/<service>:<git-sha>` — **digest-pinned in deploys, never `latest`** | `asia-southeast1-docker.pkg.dev/<proj>/agents/digital-economy-agent:a1b2c3d` |
| Cloud Run / K8s service | `kebab-case`, matches the Helm chart name | `digital-economy-agent` |
| Env vars | `SCREAMING_SNAKE`, prefixed by owning subsystem | `GATEWAY_PRIMARY_MODEL`, `RETRIEVAL_TOP_K`, `PGVECTOR_DSN` |
| Secret Manager IDs | `<service>-<purpose>` | `digital-economy-agent-gemini-key` |
| Prompt files | `<node>.v<n>.md`, immutable once released | `prompts/write_draft.v3.md` |
| Git branches | `<type>/<phase>-<slug>` | `feat/phase1-fastapi-langgraph` |
| Tags | `v2.<phase>.<patch>` | `v2.1.0` |

**Version prompts in filenames, not in git history alone.** Treating prompts as code means a diff
must be reviewable and a rollback must be a one-line change. `write_draft.v3.md` makes both true
and lets an eval report cite the exact prompt version that produced a score.

---

## 3. What each subsystem owns

| Subsystem | Responsibility | Closes v1 limitation |
| :--- | :--- | :--- |
| `api/` | Typed HTTP surface: `/chat` (SSE), `/ingest`, `/evals`, `/healthz`, `/metrics` | No API existed |
| `agents/` | The research → write → review → revise graph, using `interrupt_before` for the HITL gate | HITL logic was UI state |
| `tools/` | Pydantic-contracted adapters; v1 shipped `agentTools: []` | No tools at all |
| `gateway/` | Provider routing with retries and a fallback chain; per-request token and cost metering | Model config inline in graph JSON |
| `retrieval/` | Chunking, embedding, pgvector + BM25 hybrid search, re-ranker, query rewriting — **all in code** | Limitation 1: config unreproducible |
| `ingestion/` | Idempotent Prefect flow over `../data/imda_report.pdf`, with Presidio PII redaction | No pipeline existed |
| `evals/harness/` | Calls the live API; separates retrieval metrics (recall@k, nDCG, MRR) from generation metrics | Limitations 2–4: fixture scoring, floors, dead gate |
| `observability/` | OTel spans end-to-end, Prometheus metrics, cost per request/model | Observability was a UI toggle |
| `infra/`, `charts/` | Terraform + Helm; dedicated service account, per-secret IAM, digest-pinned images | Least-privilege regressions in v1 deploy |

---

## 4. Golden set naming

The eval harness is the part of v2 most likely to be read closely, so its datasets need to be
self-describing:

```
evals/golden/
├── policy-analysis.v1.jsonl        # the 5 v1 cases, migrated as the regression baseline
├── policy-analysis.v2.jsonl        # expanded to ~40 cases
├── adversarial-jailbreak.v1.jsonl  # prompt-injection and role-override attempts
└── out-of-scope.v1.jsonl           # queries the agent must refuse rather than answer
```

One case per line, each carrying `case_id`, `query`, `expected_sources`, `reference_answer` and
`tags`. Datasets are append-only within a version; a changed expectation means a new `.v<n>` file,
so a score is always attributable to an exact dataset revision.

---

## 5. What Phase 1 delivered

### Running it

```bash
cd pro-code-production-service-v2
uv sync --extra dev

# Serve (works without credentials; /readyz reports degraded until a key is set)
AGENT_GROQ_API_KEY=... uv run uvicorn --factory \
    digital_economy_agent.api.app:create_app --port 8080

# The full CI gate, locally
uv run ruff check . && uv run ruff format --check . \
    && uv run mypy digital_economy_agent/ && uv run pytest -q
```

### HTTP surface

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/v1/reports` | Start a run. Returns **202** with a draft awaiting review — the run pauses for a human rather than completing. |
| `POST` | `/v1/reports/stream` | Same, streaming node progress as SSE. Research and drafting take tens of seconds; the client sees progress instead of silence. |
| `POST` | `/v1/reports/{id}/review` | Approve, or request a revision with feedback. |
| `GET` | `/v1/reports/{id}` | Current state of a run. |
| `GET` | `/healthz` | Liveness. Checks no dependency, deliberately. |
| `GET` | `/readyz` | Readiness. Reports each check, so a missing credential drains traffic without a restart loop. |
| `GET` | `/metrics` | Gateway counters and a per-model cost meter. |

### What changed versus v1

| Concern | v1 (Flowise) | v2 Phase 1 |
| :--- | :--- | :--- |
| HITL gate | UI session state | `interrupt()` against a checkpointer — a paused run is durable state keyed by `thread_id`, resumable across separate HTTP requests |
| Revision loop | Unbounded loop node | `max_revisions` ceiling; exhausting it halts with `halted_reason` set and still returns the latest draft |
| Tools | `agentTools: []` — none | `Retriever` Protocol with Pydantic-validated request/response; empty retrieval is reported, not hidden |
| Model config | Inline in graph JSON | Gateway with a fallback chain, the three-way retry taxonomy, sticky selection and per-request cost metering |
| Prompts | Text in graph JSON | Versioned files (`write_draft.v1.md`) with a pinned loader; every report records the version that produced it |
| Tests | None | 34 tests (24 unit, 10 integration), no network required |
| Types | n/a | `mypy --strict` clean across 19 modules |

### Deliberate limitations

* **`InMemorySaver`, not Postgres.** Paused runs live in process memory, so a restart
  loses them. Phase 2 swaps in the Postgres checkpointer once Cloud SQL exists — the
  graph code does not change, only the `checkpointer` argument.
* **In-memory retriever, not vector search.** `InMemoryRetriever` scores token overlap
  over a small fixed corpus. It exists so the graph, the API and the tests run with no
  infrastructure, and so Phase 2 has a behavioural baseline. `describe()` says so at
  runtime.
* **The container image is unverified.** No Docker daemon was available in the
  environment where this was written. The Dockerfile is reviewed but unbuilt; CI builds
  it, runs it, and curls `/healthz` on every push, so the first CI run is the real test.
* **`/metrics` returns JSON, not Prometheus exposition format.** Phase 4 replaces it.

## 5. Phase order

Phases are unchanged from the review; the naming above is what each one lands into.

| Phase | Deliverable | Lands in |
| :---: | :--- | :--- |
| 0 ✓ | Truth pass — remove score floors, fix the dead gate, reconcile Groq/Gemini, correct deploy IAM | `../low-code-rapid-prototype-v1/` |
| 1 ✓ | FastAPI + LangGraph service, Dockerfile, pytest, CI | `api/`, `agents/`, `gateway/`, `tools/`, `tests/` |
| 2 | pgvector hybrid retrieval + Prefect ingestion with PII redaction | `retrieval/`, `ingestion/` |
| 3 | Eval harness against the live API, wired as a required PR check | `evals/` |
| 4 | Terraform, Helm, OTel + Prometheus, Trivy/SBOM/signing | `infra/`, `charts/`, `observability/` |
| 5 | *(optional)* Kafka/Pub-Sub consumer, tenant isolation via Postgres RLS | `ingestion/`, `api/` |

Phase 0 is deliberately scoped to v1: the published scores should be honest before any new
architecture is built on top of them.
