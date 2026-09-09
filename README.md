# Autonomous Multi-Agent RAG System & Evaluation Suite
### Digital Economy Policy Research with Human-in-the-Loop (HITL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/Python->=3.10-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed_by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![Orchestration: Flowise](https://img.shields.io/badge/Flowise-Agentflow_v2-black)](https://flowiseai.com/)
[![Observability: Arize AI](https://img.shields.io/badge/Observability-Arize_AI-0A58CA)](https://arize.com/)
[![LLM: Groq](https://img.shields.io/badge/LLM-Groq_gpt--oss--20b-F55036)](https://groq.com/)
[![Eval Judge: Gemini](https://img.shields.io/badge/Eval_Judge-Google_Gemini-4285F4)](https://aistudio.google.com/)
[![Vector: text--embedding--004](https://img.shields.io/badge/Embedding-text--embedding--004-34A853)](https://ai.google.dev/)
[![Vector Database: Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone_Serverless-040404?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Cloud: GCP Cloud Run](https://img.shields.io/badge/Deploy-GCP_Cloud_Run-orange)](https://cloud.google.com/run)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Interactive_Chatbot-brightgreen?logo=google-cloud&logoColor=white)](https://aisg-ladp-capstone5-303326639199.asia-southeast1.run.app/chatbot/26cb92c3-305d-4ed3-a3de-11522faa362b)

An end-to-end, production-oriented GenAI application demonstrating a **specialized multi-agent architecture** with **Document Store RAG**, **Human-in-the-Loop (HITL) iterative governance**, **live Arize AI observability & tracing**, **RAG Triad automated evaluation**, and **automated zero-secrets deployment to Google Cloud Run**.

Designed around the dense 50+ page policy report published by the **Infocomm Media Development Authority (IMDA)** and the **Tech for Good Institute**: *"From Tech for Growth to Tech for Good: Shaping the Next Phase of Southeast Asia’s Growth through the Digital Economy"*.

> [!NOTE]
> **Upstream AISG Capstone Contribution**: This repository serves as the standalone, open-source companion and full deployment/evaluation suite for the author's capstone project merged into the official AI Singapore repository: [**AISG-AIAP/LADP-Essentials (`yeehong_ho`)**](https://github.com/AISG-AIAP/LADP-Essentials/tree/main/LADPE_Project_Phase/contributions_from_learners/yeehong_ho).

---

## 🚀 Live Demo

Experience the autonomous research agent in action without installing anything:

👉 **[Launch Interactive Flowise Chatbot](https://aisg-ladp-capstone5-303326639199.asia-southeast1.run.app/chatbot/26cb92c3-305d-4ed3-a3de-11522faa362b)**

> [!TIP]
> **Sample Query to Try:**
> *"Write a brief report on the shift from 'Tech for Growth' to 'Tech for Good' in Southeast Asia."*
> 
> *Note: If the chatbot takes ~15–20 seconds on the initial request, it is Cloud Run cold-starting from 0 instances (cost-optimized autoscaling).*

---

## 🏛️ System Architecture

<p align="center">
  <img src="images/rag_retrieval_pipeline.png" alt="Multi-Agent RAG System Architecture with Arize AI Observability" width="100%"/>
</p>

<details>
<summary><b>View Mermaid Diagram Source</b></summary>

```mermaid
graph TD
    classDef startNode fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef agentNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef hitlNode fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef toolNode fill:#ede7f6,stroke:#512da8,stroke-width:2px;
    classDef arizeNode fill:#e8eaf6,stroke:#3949ab,stroke-width:2px;

    User([User Prompt / Topic]) --> Start[Start Node]:::startNode
    Start --> ResearchAgent[Research Specialist Agent<br/>Groq openai/gpt-oss-20b]:::agentNode
    
    subgraph RAG Retrieval Pipeline
        Store[(Pinecone Serverless Vector DB<br/>text-embedding-004 · 768-dim)]:::toolNode
        Tool[RAG Retrieval Tool]:::toolNode
        Store <--> Tool
    end
    
    ResearchAgent <--> Tool
    ResearchAgent -->|Structured Findings| WriterAgent[Policy Writer Agent<br/>Groq openai/gpt-oss-20b]:::agentNode
    
    WriterAgent --> DraftReport[Publication-Grade Mini-Report]
    DraftReport --> HITL{Human Review Gate<br/>HITL Governance}:::hitlNode
    
    HITL -->|Approved| Approved[Direct Reply / Final Report]:::startNode
    HITL -->|Request Revision| Loop[Loop Node<br/>Feedback Memory]:::hitlNode
    Loop -->|Iterative Critique| WriterAgent

    subgraph Observability Layer (Configured in Deployed Flowise)
        Arize[Arize AI Observability Platform<br/>OpenTelemetry / OpenInference]:::arizeNode
        SpanTool[Span: Retrieval Similarity & Latency]:::arizeNode
        SpanLLM[Span: Reasoning, Tokens & Cost]:::arizeNode
        SpanHITL[Span: HITL Pause & Loop Iterations]:::arizeNode
        
        SpanTool --- Arize
        SpanLLM --- Arize
        SpanHITL --- Arize
    end

    Tool -.->|Trace Span| SpanTool
    ResearchAgent -.->|Trace Span| SpanLLM
    WriterAgent -.->|Trace Span| SpanLLM
    HITL -.->|Trace Span| SpanHITL
```
</details>

> [!NOTE]
> **Model roles.** The agents run on **Groq `openai/gpt-oss-20b`** (temperature 0.3), as
> declared in [`flowise_scenario_5_workflow.json`](low-code-rapid-prototype-v1/flowise_scenario_5_workflow.json).
> **Google Gemini is used only as the evaluation judge**, never as the system under test —
> keeping the judge on a different model family avoids the self-preference bias a model
> shows when grading its own output. Embedding configuration (`text-embedding-004`) lives
> in Flowise Document Store server state and is not verifiable from this repository; see
> [v1 limitations](low-code-rapid-prototype-v1/README.md#known-limitations-of-this-tier).

### Key Architectural Highlights
* **Specialized Separation of Concerns:**
  * **Research Agent:** High-recall factual retrieval across country benchmarks (Singapore, Malaysia, Indonesia, Thailand, Philippines, Vietnam) and the four structural pillars (Infrastructure, Talent, Trust/Cybersecurity, Governance).
  * **Writer Agent:** Executive policy synthesis structured strictly into: *Executive Summary*, *Key Findings & Regional Analysis*, *Strategic Enablers & Recommendations*, and *Future Outlook*.
* **Enterprise Human-in-the-Loop (HITL) Gate:** Prevents unchecked autonomous generation by providing an interactive review barrier where human domain experts can approve or request targeted revisions.
* **Full-Stack Arize AI Observability:** Configured directly within the deployed Flowise interface on Google Cloud Run to stream OpenInference and OpenTelemetry traces into **Arize AI**, monitoring latency, token economics, retrieval chunk quality, and feedback iteration counts in production.
* **Pinecone Serverless Cloud Vector Database:** Ingests and stores the 768-dimensional embeddings generated by `text-embedding-004` (cosine similarity, namespace `imda-sea-report`). Offloading vectors to Pinecone Serverless decouples vector storage from compute, so the index survives Cloud Run scale-to-zero cycles independently of container state. Query latency and durability are not benchmarked in this repository; treat them as Pinecone's published characteristics rather than measurements of this deployment.
* **Asymmetric Semantic Embeddings:** Uses Google's `text-embedding-004` with asymmetric indexing (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) for higher retrieval precision at 768 dimensions with 50% lower memory footprint than traditional 1536-dim vectors.

---

## 🧭 Project Tiers

This repository is organised into two deliberately separate tiers:

| Tier | Folder | Status | Purpose |
| :--- | :--- | :--- | :--- |
| **v1 — Low-code rapid prototype** | [`low-code-rapid-prototype-v1/`](low-code-rapid-prototype-v1/) | Shipped, live | Flowise Agentflow v2 graph that validated the multi-agent + HITL approach in days. Frozen for feature work; retained as the demo surface and behavioural baseline. |
| **v2 — Pro-code production service** | [`pro-code-production-service-v2/`](pro-code-production-service-v2/) | Phase 1 shipped | FastAPI service over a LangGraph agent, with the HITL gate as a durable `interrupt()`, a model gateway (fallback chain + cost meter), versioned prompts, 34 tests and `mypy --strict`. Phases 2–5 in design. |

The v1 folder documents its own [known limitations](low-code-rapid-prototype-v1/README.md#known-limitations-of-this-tier);
each is carried forward as a v2 requirement rather than patched in place.

---

## 📂 Repository Structure

```
├── low-code-rapid-prototype-v1/        # v1 — Flowise low-code prototype (shipped, live)
│   ├── flowise_scenario_5_workflow.json  # Sanitized Flowise Agentflow v2 workflow export
│   ├── prompts/
│   │   ├── research_agent_prompt.txt   # System directives for factual extraction & country breakouts
│   │   └── writer_agent_prompt.txt     # Policy writer persona, report schema & revision rules
│   ├── evals/
│   │   ├── run_evaluations.py          # RAG Triad evaluation runner (fixture-scored — see v2 roadmap)
│   │   ├── evaluation_dataset.json     # Benchmark dataset with queries, context & golden references
│   │   ├── evaluation_dataset.csv      # Tabular version of benchmark scenarios
│   │   ├── evaluation_report.md        # Generated benchmark report and scorecard
│   │   └── evaluations_notebook.ipynb  # Interactive Jupyter analysis notebook
│   └── deploy/
│       ├── deploy_gcp.sh               # Cloud Run deployment script for the Flowise container
│       ├── env.example                 # Sanitized deployment configuration template
│       ├── flowise.html                # Lightweight embeddable chat web interface
│       └── README_DEPLOY_GCP.md        # GCP infrastructure deployment guide
├── pro-code-production-service-v2/     # v2 — pro-code service (Phase 1 shipped)
│   ├── digital_economy_agent/          # FastAPI + LangGraph service, gateway, tools, prompts
│   ├── tests/                          # 34 unit + integration tests, no network required
│   ├── Dockerfile                      # Multi-stage, non-root, healthcheck
│   ├── pyproject.toml                  # Service dependencies (separate from the v1 tooling)
│   └── README.md                       # Architecture, naming conventions & phased roadmap
├── .github/workflows/
│   └── v2-service-ci.yml               # Lint, mypy --strict, pytest, image build + smoke + Trivy
├── data/
│   └── imda_report.pdf                 # Source corpus shared by v1 and v2
├── images/
│   └── rag_retrieval_pipeline.png      # High-resolution system architecture diagram
├── main.py                             # Central project CLI entrypoint
├── pyproject.toml                      # Project metadata and dependencies (PEP 518/621)
├── uv.lock                             # Deterministic lockfile managed by uv
├── LICENSE                             # MIT License
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart Guide

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package and environment management.

### 1. Prerequisites
* Python `>= 3.10`
* [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)
* [Flowise](https://flowiseai.com/) (Local via `npx flowise start` or Docker container)

### 2. Environment Setup
Clone the repository and synchronize dependencies:
```bash
git clone https://github.com/hoyeehong/aisg-flowise-rag-agent-gcp.git
cd aisg-flowise-rag-agent-gcp

# Install dependencies into an isolated virtual environment
uv sync
```

---

## 🧪 Automated RAG Triad Evaluation Suite

The evaluation suite implements the **RAG Triad** framework (TruLens / RAGAS standard) to rigorously score hallucination resistance, retrieval relevance, and structural completeness.

### Metric Definitions

| Metric | Target | What it measures |
| :--- | :---: | :--- |
| Context Relevance | $\ge 0.80$ | Whether retrieved chunks contain facts relevant to the research query. |
| Groundedness / Faithfulness | $\ge 0.85$ | Whether the Writer Agent's claims are supported by retrieved context (hallucination check). |
| Answer Relevance | $\ge 0.80$ | Whether the mini-report addresses the query and follows the 4-part structure. |
| RAG Triad Composite | $\ge 0.80$ | Arithmetic mean of the three triad dimensions. |
| Token F1 vs Reference | $\ge 0.70$ | Lexical token overlap against curated reference answers. |
| Structure Completeness | $\ge 80\%$ | Presence of Executive Summary, Key Findings, Strategic Enablers, Conclusion. |

### Verified baseline (Gemini judge)

Scored by `gemini-3-flash-preview` as an independent judge against the 5-case fixture.
Run `2026-09-09`, dataset SHA-256 `50690196a05a...`. Context relevance, answer relevance
and structure were identical across four runs; groundedness was not (see the variance note
below), so it is quoted as a range.

| Metric | Score | Target | Status |
| :--- | :---: | :---: | :---: |
| Context Relevance | **1.000** | $\ge 0.80$ | `PASS` |
| Answer Relevance | **1.000** | $\ge 0.80$ | `PASS` |
| Structure Completeness | **100%** | $\ge 80\%$ | `PASS` |
| Groundedness / Faithfulness | 0.44–0.50 | $\ge 0.85$ | `INCONCLUSIVE` |
| Token F1 vs Reference | 0.990 | $\ge 0.70$ | `INCONCLUSIVE` |
| **Overall** | — | — | **`INCONCLUSIVE`** |

**Retrieval and instruction-following are strong.** Context relevance and answer relevance
both score 1.000 across all five cases: the retrieved chunks are on-topic and every report
follows the required 4-part structure.

**Two metrics are inconclusive by construction, and the harness now says so rather than
scoring them:**

* **Groundedness (0.44–0.50 across four runs).** The fixture's `retrieved_context` is 18–27% the character
  length of its `generated_response` (261–451 chars of context against 1,446–1,680 chars
  of report). No answer could be grounded in that little context, so the low score
  measures a truncated fixture, not hallucination by the agent. Groundedness is therefore
  excluded from the verdict. Note how tightly the score tracks the defect: TC-05 has the
  smallest context ratio (0.20) *and* the lowest groundedness (0.20).
* **Token F1 (0.990).** `generated_response` is a near-verbatim paraphrase of
  `reference_answer`, so this measures the dataset's internal consistency.

Both are dataset defects that a fixture-based harness cannot fix. Capturing the real
retrieved chunk set requires calling the live system — Phase 3 of the
[v2 roadmap](pro-code-production-service-v2/README.md).

> [!NOTE]
> **Superseded scores.** Earlier revisions of this README quoted a passing scorecard
> (composite `0.887`, groundedness `0.880`). Those came from a deterministic fallback
> whose score floors sat *above* the pass thresholds, so they could not fail, and the
> pass/fail gate was a no-op ternary returning `PASS` on both branches. Both defects are
> fixed; the scores are withdrawn rather than restated.

> [!TIP]
> **Judge variance is real, within *and* across models.** Four consecutive runs of
> `gemini-3-flash-preview` at `temperature 0.0` returned groundedness `0.500`, `0.480`,
> `0.460`, `0.440` — a 0.06 spread on the metric that decides the verdict. Switching model
> widened it further: `gemini-3.6-flash` returned `0.380`–`0.460` on the same fixture.
> Cross-model spread exceeds within-model spread, which is why a blended run is reported
> `INCONCLUSIVE` and why a published baseline must pin one model. A single run is a sample,
> not a constant; running *n* trials and reporting mean with spread is scoped into Phase 3.

### Judge Model Gateway (Routing, Retries & Fallback)

Free-tier Gemini enforces per-model quotas, so a single 429 used to abort an entire
evaluation run. The judge now routes through a fallback chain with a retry policy, which
is the same shape as the model gateway v2 needs for the system itself.

**Default chain** (newest first; every name verified present via the ListModels API):

```
gemini-3.8-flash → gemini-3.7-flash → gemini-3.6-flash → gemini-3.5-flash → gemini-3-flash-preview
```

Discover what your own key can call, and see which entries are in the chain:

```bash
uv run python main.py eval --list-judge-models
```

Override the chain, or pin one model, with a comma-separated list (or `JUDGE_MODEL_CHAIN`):

```bash
uv run python main.py eval --judge gemini --judge-models "gemini-3.8-flash,gemini-3.5-flash"
uv run python main.py eval --judge gemini --judge-models "gemini-3.6-flash"   # pinned
```

**Failure classification drives the response** — the three cases are not interchangeable:

| Condition | Response | Why |
| :--- | :--- | :--- |
| `429` quota | Advance to the next model **immediately**, no sleep | Quota resets in minutes, not seconds. Sleeping stalls the run while a sibling model is very likely available. |
| `5xx` / timeout | Backoff-retry the *same* model (3 attempts, exponential + jitter) | "High demand" genuinely clears in seconds; jitter avoids synchronised retries. |
| `404` / `400` / `403` | **Retire** the model for the rest of the run | A wrong name or missing access will never succeed; retrying it per case wastes a call every time. |
| Entire chain rate-limited | One bounded cooldown (≤ 90s), then retry the chain once | With nothing left to fall back to, waiting out a per-minute limit beats aborting. A longer reported reset implies a *daily* quota, so it fails fast instead. |

Model selection is **sticky**: once a model answers, it serves every subsequent case.
Re-probing from the top per case would be slower and would let judge identity oscillate
mid-run.

> [!IMPORTANT]
> **A blended run is never `PASSED`.** If the chain advances mid-run, different cases were
> graded by different models, so the aggregate is a mix of judges rather than one
> measurement. The runner records the judge model per case, sets `judge_consistent: false`,
> and downgrades the verdict to `INCONCLUSIVE`. For a publishable baseline, pin a single
> model with `--judge-models`.
>
> Moving aliases such as `gemini-flash-latest` are deliberately **excluded** from the
> default chain: an alias can silently change which model produced a score, which defeats
> the purpose of recording judge identity in provenance at all.

### Running Evaluations

Run the automated evaluation runner:
```bash
uv run python low-code-rapid-prototype-v1/evals/run_evaluations.py
```

To run with **Google Gemini as the LLM-as-a-Judge** (the only mode producing publishable scores):
```bash
export GEMINI_API_KEY="your-google-gemini-api-key"
uv run python main.py eval --judge gemini
```

Without a judge key the runner **refuses to run** rather than substituting a fallback
score. For an offline plumbing check, lexical proxies are available and are always
reported `UNVERIFIED`:
```bash
uv run python main.py eval --judge heuristic
```

Exit codes: `0` all gates passed, `1` a gate failed, `2` run could not be verified — so
the runner can gate CI directly.

To explore interactively in Jupyter:
```bash
uv run jupyter lab low-code-rapid-prototype-v1/evals/evaluations_notebook.ipynb
```

---

## 🔭 Observability, Tracing & Production Engineering (Arize AI)

Monitoring complex agentic workflows in production requires granular visibility into node transitions, vector retrieval quality, and LLM reasoning steps.

### Arize AI Integration via Deployed Flowise Interface
In the production deployment on **Google Cloud Run**, observability is enabled natively via the Flowise UI (**Configuration / Settings $\rightarrow$ Analytics $\rightarrow$ Arize AI / OpenInference**):

* **OpenTelemetry & OpenInference Telemetry:** Every execution event emits standardized traces into the Arize platform:
  * **Span 1 (`startAgentflow`):** Prompt ingestion, session metadata, and user query timestamp.
  * **Span 2 (`agentAgentflow` / RAG Tool):** Embedding conversion with `text-embedding-004`, top-K retrieved chunk similarity scores, retrieved context payload, and retrieval latency.
  * **Span 3 (`llmAgentflow`):** Writer Agent reasoning step, prompt/completion token consumption, temperature parameters, and execution latency.
  * **Span 4 (`humanInputAgentflow` & `loopAgentflow`):** Human-in-the-Loop approval/revision events, user critique payload, and multi-turn iteration counters.

### Production Observability Capabilities with Arize:
1. **RAG Retrieval Quality & Semantic Drift:** Continuously inspects whether retrieved context chunks remain tightly aligned with policy queries over time.
2. **Multi-Agent Cost & Token Tracking:** Monitors token consumption broken down by agent role (Research vs Writer) across successive HITL feedback loops.
3. **Continuous Groundedness & Hallucination Guardrails:** Correlates production trace inputs against generated answers to identify hallucinated citations or ungrounded claims in real time.

---

## ☁️ Cloud Deployment (Google Cloud Run)

The [`low-code-rapid-prototype-v1/deploy/`](low-code-rapid-prototype-v1/deploy/) directory provides a production deployment setup for Google Cloud Platform (`asia-southeast1`):

* **Secrets via Secret Manager:** Model API keys *and the Flowise admin password* are injected at startup from **Google Secret Manager**. (Before Phase 0 the admin password was passed via `--set-env-vars`, which is readable through `gcloud run services describe` and Cloud Audit Logs.)
* **Least-privilege IAM:** The service runs as a dedicated service account with per-secret `secretAccessor` bindings and bucket-scoped `objectAdmin` — not the shared default compute SA, and with no project-level grants.
* **Persistent Storage:** Cloud Run integrates a **Google Cloud Storage (GCS) FUSE** mount (`gs://flowise-data-<PROJECT_ID>`) to persist Flowise SQLite databases, sessions, and document stores across restarts.
* **Cost Efficiency:** Automated scale-to-zero (`min-instances: 0`).
* **Single-writer constraint:** `max-instances` defaults to **1**. Flowise persists to SQLite on a GCS FUSE mount, and FUSE does not provide the POSIX advisory locking SQLite requires — concurrent writers risk database corruption. Raising the ceiling requires migrating to Postgres (Cloud SQL) first.

### Deploy in One Command:
```bash
# Optional: copy configuration template
cp low-code-rapid-prototype-v1/deploy/env.example low-code-rapid-prototype-v1/deploy/.env

# Execute deployment
bash low-code-rapid-prototype-v1/deploy/deploy_gcp.sh
```

For complete step-by-step deployment and operational management instructions, see [`low-code-rapid-prototype-v1/deploy/README_DEPLOY_GCP.md`](low-code-rapid-prototype-v1/deploy/README_DEPLOY_GCP.md).

---

## ⚙️ How to Import into Flowise

1. Launch Flowise locally (`npx flowise start`) or on Cloud Run.
2. Under **Credentials**, add your:
   * **Google Generative AI API** key (from [Google AI Studio](https://aistudio.google.com/))
   * **Pinecone API** key (from [Pinecone Console](https://app.pinecone.io/))
3. Under **Document Stores**, create `imda_sea_digital_economy_report`:
   * **Loader:** PDF File Loader $\rightarrow$ upload `data/imda_report.pdf`
   * **Text Splitter:** Recursive Character Text Splitter (`Chunk: 1000`, `Overlap: 200`)
   * **Embeddings:** Google GenerativeAI Embeddings (`text-embedding-004`, 768 dimensions)
   * **Vector Store:** **Pinecone** (Index: `ladp-capstone`, Dimension: `768`, Metric: `cosine`, Namespace: `imda-sea-report`)
   * Click **Save & Upsert Chunk**
4. Under **Agentflows**, click **Add New** $\rightarrow$ **Settings** $\rightarrow$ **Load / Import Chatflow** $\rightarrow$ Select [`low-code-rapid-prototype-v1/flowise_scenario_5_workflow.json`](low-code-rapid-prototype-v1/flowise_scenario_5_workflow.json).
5. Click **Save** and test in the chat canvas.

---

## 📜 Acknowledgements & References
* **AI Singapore (AISG):** Developed for the *LLM Application Developer Programme (Essentials)* (LADP-E) Capstone Project. Official learner submission merged upstream in [**AISG-AIAP/LADP-Essentials (`yeehong_ho`)**](https://github.com/AISG-AIAP/LADP-Essentials/tree/main/LADPE_Project_Phase/contributions_from_learners/yeehong_ho).
* **IMDA & Tech for Good Institute:** Source policy report: *"From Tech for Growth to Tech for Good: Shaping the Next Phase of Southeast Asia’s Growth through the Digital Economy"*.
* **Flowise AI:** Low-code/no-code visual framework for LangChain and LangGraph agentic workflows.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
