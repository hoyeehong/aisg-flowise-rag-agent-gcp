# v2 — Pro-Code Production Service

**Status:** Phases 1 and 2 delivered — the service runs on pgvector hybrid retrieval
with a durable review gate, is `mypy --strict` clean, and is covered by 98 tests.
Phases 3–5 remain design-stage; the naming conventions below were fixed first so the
code landed in the right shape rather than being reorganised later.

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
│   ├── retrieval/                  ✓ chunking, embeddings, pgvector, hybrid search, MMR
│   ├── ingestion/                  ✓ PII redaction, idempotent pipeline, Prefect flow, CLI
│   └── observability/              ✓ package placeholder; OTel + Prometheus land in P4
├── tests/
│   ├── unit/                       ✓ graph transitions, gateway taxonomy
│   ├── integration/                ✓ real routes + real graph, scripted providers
│   └── load/                       — k6 / Locust scenarios (P4)
├── docker-compose.yml              ✓ local pgvector dependency
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

## 5. What Phase 2 delivered

Phase 2 replaced the placeholder retriever with real hybrid retrieval and built the
ingestion pipeline that feeds it. Both slot in behind protocols declared in Phase 1, so
nothing in `agents/` or `api/` changed.

### The measured objective

Phase 1 ended with a number, not an opinion: the token-overlap retriever matched
**1 of 4 chunks** on a well-formed query and returned **177 characters** of context.

| | chunks matched | context chars |
| :--- | :---: | :---: |
| v1 token overlap | 1 / 4 | 177 |
| v2 hybrid (same corpus, *hashing* embedder) | **4 / 4** | **681** |
| v2 hybrid on the real 27-chunk IMDA corpus, Google embeddings | 4 / 4 | **2,427–3,643** |

The middle row is the interesting one: the improvement holds even with a *non-semantic*
embedder, which shows it comes from hybrid fusion and deeper candidate pools rather than
from better vectors alone.

This also closes the Phase 0 finding directly. Groundedness was reported `INCONCLUSIVE`
because the fixture's context was 18–27% the length of the answer it had to support. At
2,400–3,600 characters per query, context finally exceeds the answer, so groundedness
becomes measurable — which is what Phase 3's harness needs.

### Retrieval

* **Hybrid search.** Vector (pgvector HNSW, cosine) and lexical (`tsvector` + GIN)
  fused by Reciprocal Rank Fusion. The two fail in opposite directions: embeddings
  generalise but blur exact tokens, so `ASEAN DEFA` can rank generic
  "regional cooperation" prose first; lexical search nails the identifier but cannot
  connect "Tech for Good" to "digital trust" at all. RRF fuses **ranks, not scores**,
  because cosine similarity and `ts_rank_cd` are not on comparable scales — normalising
  and adding them would let whichever produces bigger numbers dominate.
* **Embeddings.** `gemini-embedding-001` at 768 dimensions with asymmetric task types
  (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`). Truncated outputs are **re-normalised**:
  Gemini vectors are unit-length at their native 3072 dims, and slicing to 768 does not
  preserve that, so cosine distance would silently mis-rank without it.
* **MMR diversification.** A 200-char chunk overlap means neighbours share text, so an
  un-diversified top-5 can spend three slots on the same passage. This is a
  diversification pass, *not* a cross-encoder re-ranker — that is a later upgrade, and
  calling it one would be a claim the code does not support.
* **Query rewriting** via the model gateway, off by default because it costs an extra
  model call per retrieval. Failure degrades to the original query rather than failing
  the search.

### Ingestion

* **Idempotent by design, not by discipline.** Every chunk is keyed by a content hash of
  `(source, page, text)` and upserted via `ON CONFLICT ... RETURNING (xmax = 0)`. A
  re-run reports `inserted=0`, which is the *signal* that the pipeline is repeatable; a
  run that dies halfway can simply be repeated.
* **PII redaction before embedding.** Redacting afterwards would have already sent the
  data to a third party and would leave the original recoverable in the vector's
  neighbourhood.
* **Redaction errs toward over-redaction.** A false positive costs a redacted word; a
  false negative is a data breach. Checksums (Luhn, Singapore NRIC/FIN) therefore only
  *annotate confidence* — a failed checksum is reported but the value is still redacted,
  so a bug in a checksum could never turn a detection into a leak.
* **Prefect is optional.** All correctness lives in `pipeline.py` with no orchestrator
  import; `flow.py` adds only scheduling, retries and observability. The API image
  therefore does not carry an orchestrator it will never run. A CLI
  (`python -m digital_economy_agent.ingestion`) covers one-off backfills.

### Durability

`AsyncPostgresSaver` replaces `InMemorySaver` when a DSN is configured, so a paused
human-review run survives a restart. Verified end to end against the built container: a
report paused before `docker restart` came back byte-identical (3,172 chars) afterwards,
and a reviewer could still approve it.

### Probe semantics

`/healthz` and `/readyz` are deliberately different, and the **status code** is what
carries the difference — Kubernetes and Cloud Run route on the code and never parse the
body:

| Probe | Condition | Code | Effect |
| :--- | :--- | :---: | :--- |
| `/healthz` | process is up | 200 | always; a failing liveness probe *restarts*, and restarting cannot conjure a missing credential or database |
| `/readyz` | all checks pass | 200 | traffic routed here |
| `/readyz` | any check fails | **503** | traffic *drained*, no restart |

Readiness drains; liveness restarts. Answering 200 while reporting
`"status": "degraded"` would keep traffic arriving at an instance that had just said it
could not do its job — with `durable_checkpointer: false`, for example, a restart would
silently discard work awaiting a human reviewer.

### The Postgres checkpointer shipped broken

Worth recording, because the failure mode is more interesting than the fix.
`ReportService` read state through the *synchronous* `graph.get_state()`.
`InMemorySaver` tolerates that; `AsyncPostgresSaver` rejects it from the main thread
with `InvalidStateError`. So every state read — `start`, `review`, `get` — returned a
500 against a real database.

The whole suite was green throughout, because every test used `InMemorySaver`: the
integration tests inject a retriever, which disables the Postgres branch entirely. The
bug only surfaced when the built container was driven end to end against Postgres.

Fixed by making those reads `await ... aget_state(...)`, and covered by
`tests/integration/test_durability_pg.py`, which runs against a genuine
`AsyncPostgresSaver`. Reintroducing the sync call fails all four of those tests while
the other 83 still pass — which is exactly the blind spot that let it reach a release.
A checkpointer is not interchangeable with its in-memory stand-in, so the durable one
needs its own coverage.

### Tests: 104

Unit tests are offline (deterministic hashing embedder). Retrieval and durability
integration tests run against **real Postgres** — the interesting behaviour is in SQL (generated `tsvector`,
HNSW ordering, `ON CONFLICT` idempotency), so a fake would test nothing. CI runs a
`pgvector/pgvector:pg17` service and **asserts those tests were not skipped**, because a
skip-if-unavailable marker is a local convenience that would otherwise hollow out CI
silently.

### Phase 2 limitations

* **`gemini-embedding-001`, not `text-embedding-004`.** The model v1's docs claimed is
  not available on this key; only `gemini-embedding-001` and `gemini-embedding-2` are.
* **Ingestion was verified on 12 of 75 pages**, bounded to preserve free-tier embedding
  quota. Retrieval quality on the full corpus is therefore untested, and the visible
  artefact of that is table-of-contents chunks ranking for some queries.
* **Lexical search is conjunctive.** `websearch_to_tsquery` ANDs its terms, so a
  multi-term paraphrase can match nothing lexically. Fusion compensates; a test
  documents the behaviour rather than leaving it as a surprise.
* **RRF `k=60` and the fusion weights are untuned.** They are the values from the
  original formulation. Fitting them needs the Phase 3 golden set; tuning against a
  handful of ad-hoc queries would just be overfitting.
* **No cross-encoder re-ranker.** MMR diversifies; it does not score query-passage pairs.

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
| `GET` | `/readyz` | Readiness. **200 ready / 503 degraded**, reporting each check by name. |
| `GET` | `/metrics` | Gateway counters and a per-model cost meter. |

### What changed versus v1

| Concern | v1 (Flowise) | v2 Phase 1 |
| :--- | :--- | :--- |
| HITL gate | UI session state | `interrupt()` against a checkpointer — a paused run is durable state keyed by `thread_id`, resumable across separate HTTP requests |
| Revision loop | Unbounded loop node | `max_revisions` ceiling; exhausting it halts with `halted_reason` set and still returns the latest draft |
| Tools | `agentTools: []` — none | `Retriever` Protocol with Pydantic-validated request/response; empty retrieval is reported, not hidden |
| Model config | Inline in graph JSON | Gateway with a fallback chain, the three-way retry taxonomy, sticky selection and per-request cost metering |
| Prompts | Text in graph JSON | Versioned files (`write_draft.v1.md`) with a pinned loader; every report records the version that produced it |
| Tests | None | 37 tests (18 unit, 19 integration), no network required |
| Types | n/a | `mypy --strict` clean across 19 modules |

### Verified end to end against a live model

A full run through the real graph with a live Groq `openai/gpt-oss-20b`:
create → revise-with-feedback → approve, across three separate HTTP requests.

| Step | Result |
| :--- | :--- |
| `POST /v1/reports` | 202 in 3.0s, 2 LLM calls, 2,381-char draft with citation markers |
| `POST .../review` (revise) | 200 in 1.7s, revision applied to the Executive Summary only, `revisions_used: 1/2` |
| `POST .../review` (approve) | 200, `completed`, `final_report == draft` |
| Repeat approve | 409 — the pause is genuinely consumed |
| `/metrics` | 3 calls, 4,355 tokens, $0.0015355, no fallbacks |

Two things this run showed that no scripted test could:

**The grounding discipline holds.** Only one corpus chunk matched (177 chars), and the
model said so rather than filling the gap — *"No country-specific or comparative
information ... is available in the source"*, *"The source does not provide any GMV
figures, inclusion indices, trust scores or sustainability KPIs."* This is the direct
contrast with v1, where 451 chars of context produced 1,680 chars of confident prose
making claims the context did not support.

**Retrieval recall is the binding constraint, and now there is evidence for it.** The
in-memory retriever matched 1 of 4 chunks for a well-formed query; pages 18, 23 and 31
scored zero because token overlap cannot connect *"Tech for Good"* to *"digital trust"*
or *"talent pipeline"*. That is not a tuning problem, it is the ceiling of lexical
matching — and it is the concrete case for Phase 2's embeddings plus hybrid search,
rather than an assertion that vector search would be nicer.

### Hardening applied to the image

`python:3.11-slim` ships `pip`, `setuptools` and `wheel` in the global site-packages.
The application runs entirely from `/app/.venv` and needs none of them at runtime, while
`setuptools`' vendored `jaraco.context` and `wheel` were contributing two HIGH CVEs
(CVE-2026-23949, CVE-2026-24049). Removing them took fixable HIGH/CRITICAL findings from
two to zero, which is both a smaller attack surface and one less thing to triage on every
scan.

### Deliberate limitations

* **`InMemorySaver`, not Postgres.** Paused runs live in process memory, so a restart
  loses them. Phase 2 swaps in the Postgres checkpointer once Cloud SQL exists — the
  graph code does not change, only the `checkpointer` argument.
* **In-memory retriever, not vector search.** `InMemoryRetriever` scores token overlap
  over a small fixed corpus. It exists so the graph, the API and the tests run with no
  infrastructure, and so Phase 2 has a behavioural baseline. `describe()` says so at
  runtime.
* ~~The container image is unverified.~~ **Verified.** Built and run locally: healthy in
  2s, non-root (uid 10001), Docker `HEALTHCHECK` reporting `healthy`, `uv` and `pip`
  absent from the runtime layer, and **zero fixable HIGH/CRITICAL Trivy findings** after
  removing the base image's build tooling. The Trivy gate in CI is therefore blocking,
  not advisory.
* **`/metrics` returns JSON, not Prometheus exposition format.** Phase 4 replaces it.

## 5. Phase order

Phases are unchanged from the review; the naming above is what each one lands into.

| Phase | Deliverable | Lands in |
| :---: | :--- | :--- |
| 0 ✓ | Truth pass — remove score floors, fix the dead gate, reconcile Groq/Gemini, correct deploy IAM | `../low-code-rapid-prototype-v1/` |
| 1 ✓ | FastAPI + LangGraph service, Dockerfile, pytest, CI | `api/`, `agents/`, `gateway/`, `tools/`, `tests/` |
| 2 ✓ | pgvector hybrid retrieval + Prefect ingestion with PII redaction | `retrieval/`, `ingestion/` |
| 3 | Eval harness against the live API, wired as a required PR check | `evals/` |
| 4 | Terraform, Helm, OTel + Prometheus, Trivy/SBOM/signing | `infra/`, `charts/`, `observability/` |
| 5 | *(optional)* Kafka/Pub-Sub consumer, tenant isolation via Postgres RLS | `ingestion/`, `api/` |

Phase 0 is deliberately scoped to v1: the published scores should be honest before any new
architecture is built on top of them.
