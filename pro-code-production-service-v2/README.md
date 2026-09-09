# v2 — Pro-Code Production Service

**Status:** Design stage. No code yet — this document fixes the naming and structure so that
Phase 1 lands in the right shape rather than being reorganised later.

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

```
pro-code-production-service-v2/
├── digital_economy_agent/          # import root — underscores, singular domain noun
│   ├── api/                        # FastAPI routers, request/response schemas
│   ├── agents/                     # LangGraph graphs, nodes, state definitions
│   ├── tools/                      # typed tool adapters (one module per adapter)
│   ├── gateway/                    # model routing, retries, fallback chain, cost meter
│   ├── retrieval/                  # chunking, embeddings, hybrid search, re-ranking
│   ├── ingestion/                  # Prefect flows, PII redaction, metadata governance
│   └── observability/              # OTel tracing, Prometheus metrics
├── tests/
│   ├── unit/                       # mirrors the package tree 1:1
│   ├── integration/                # real Postgres + mocked LLM
│   └── load/                       # k6 / Locust scenarios
├── evals/
│   ├── golden/                     # versioned golden sets (see §4)
│   ├── harness/                    # runner that calls the live API
│   └── reports/                    # generated, gitignored except the latest baseline
├── infra/terraform/                # one module per resource group
├── charts/digital-economy-agent/   # Helm chart — kebab-case, matches the K8s object name
├── Dockerfile
└── pyproject.toml
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

## 5. Phase order

Phases are unchanged from the review; the naming above is what each one lands into.

| Phase | Deliverable | Lands in |
| :---: | :--- | :--- |
| 0 | Truth pass — remove score floors, fix the dead gate, reconcile Groq/Gemini, correct deploy IAM | `../low-code-rapid-prototype-v1/` |
| 1 | FastAPI + LangGraph service, Dockerfile, pytest, CI | `api/`, `agents/`, `tests/` |
| 2 | pgvector hybrid retrieval + Prefect ingestion with PII redaction | `retrieval/`, `ingestion/` |
| 3 | Eval harness against the live API, wired as a required PR check | `evals/` |
| 4 | Terraform, Helm, OTel + Prometheus, Trivy/SBOM/signing | `infra/`, `charts/`, `observability/` |
| 5 | *(optional)* Kafka/Pub-Sub consumer, tenant isolation via Postgres RLS | `ingestion/`, `api/` |

Phase 0 is deliberately scoped to v1: the published scores should be honest before any new
architecture is built on top of them.
