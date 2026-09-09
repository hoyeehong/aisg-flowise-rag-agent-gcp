# Autonomous Multi-Agent RAG System & Evaluation Suite
### Digital Economy Policy Research with Human-in-the-Loop (HITL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: >=3.10](https://img.shields.io/badge/Python->=3.10-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed_by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![Orchestration: Flowise](https://img.shields.io/badge/Flowise-Agentflow_v2-black)](https://flowiseai.com/)
[![Observability: Arize AI](https://img.shields.io/badge/Observability-Arize_AI-0A58CA)](https://arize.com/)
[![LLM: Google Gemini](https://img.shields.io/badge/Google_Gemini-Flash_Preview-4285F4)](https://aistudio.google.com/)
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
    Start --> ResearchAgent[Research Specialist Agent<br/>Gemini 3 Flash Preview]:::agentNode
    
    subgraph RAG Retrieval Pipeline
        Store[(Pinecone Serverless Vector DB<br/>text-embedding-004 · 768-dim)]:::toolNode
        Tool[RAG Retrieval Tool]:::toolNode
        Store <--> Tool
    end
    
    ResearchAgent <--> Tool
    ResearchAgent -->|Structured Findings| WriterAgent[Policy Writer Agent<br/>Gemini 3 Flash Preview]:::agentNode
    
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

### Key Architectural Highlights
* **Specialized Separation of Concerns:**
  * **Research Agent:** High-recall factual retrieval across country benchmarks (Singapore, Malaysia, Indonesia, Thailand, Philippines, Vietnam) and the four structural pillars (Infrastructure, Talent, Trust/Cybersecurity, Governance).
  * **Writer Agent:** Executive policy synthesis structured strictly into: *Executive Summary*, *Key Findings & Regional Analysis*, *Strategic Enablers & Recommendations*, and *Future Outlook*.
* **Enterprise Human-in-the-Loop (HITL) Gate:** Prevents unchecked autonomous generation by providing an interactive review barrier where human domain experts can approve or request targeted revisions.
* **Full-Stack Arize AI Observability:** Configured directly within the deployed Flowise interface on Google Cloud Run to stream OpenInference and OpenTelemetry traces into **Arize AI**, monitoring latency, token economics, retrieval chunk quality, and feedback iteration counts in production.
* **Pinecone Serverless Cloud Vector Database:** Ingests and stores the 768-dimensional embeddings generated by `text-embedding-004` (cosine similarity, namespace `imda-sea-report`). Offloading vectors to Pinecone Serverless decouples storage from compute, guaranteeing zero data loss, sub-50ms vector query latency, and seamless state persistence across Cloud Run container scale-to-zero cycles.
* **Asymmetric Semantic Embeddings:** Uses Google's `text-embedding-004` with asymmetric indexing (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) for higher retrieval precision at 768 dimensions with 50% lower memory footprint than traditional 1536-dim vectors.

---

## 🧭 Project Tiers

This repository is organised into two deliberately separate tiers:

| Tier | Folder | Status | Purpose |
| :--- | :--- | :--- | :--- |
| **v1 — Low-code rapid prototype** | [`low-code-rapid-prototype-v1/`](low-code-rapid-prototype-v1/) | Shipped, live | Flowise Agentflow v2 graph that validated the multi-agent + HITL approach in days. Frozen for feature work; retained as the demo surface and behavioural baseline. |
| **v2 — Pro-code production service** | [`pro-code-production-service-v2/`](pro-code-production-service-v2/) | Design stage | Re-platform as a code-owned service: typed API over LangGraph, RAG declared in code, evals that call the running system. |

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
├── pro-code-production-service-v2/     # v2 — pro-code service re-platform (design stage)
│   └── README.md                       # Target architecture, naming conventions & phased roadmap
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

### Evaluation Metrics & Rubric
| Metric | Score | Target | Status | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Relevance** | **0.908** | $\ge 0.80$ | `PASS` | Evaluates if retrieved text chunks from `text-embedding-004` contain strictly relevant facts. |
| **Groundedness / Faithfulness** | **0.880** | $\ge 0.85$ | `PASS` | Evaluates whether claims made by the Writer Agent are grounded in retrieved context (Hallucination check). |
| **Answer Relevance** | **0.871** | $\ge 0.80$ | `PASS` | Evaluates whether the generated mini-report directly addresses the user query and intent. |
| **RAG Triad Composite** | **0.887** | $\ge 0.80$ | **`PASSED`** | Harmonic composite score across all three triad dimensions. |
| **Token F1 vs Reference** | **0.990** | $\ge 0.70$ | `PASS` | Lexical & token overlap against calibrated expert reference reports. |
| **Structure Completeness** | **100%** | $\ge 80\%$ | `PASS` | Validates presence of Executive Summary, Key Findings, Strategic Enablers, and Conclusion. |

### Running Evaluations

Run the automated evaluation runner:
```bash
uv run python low-code-rapid-prototype-v1/evals/run_evaluations.py
```

To run with **Google Gemini as an active LLM-as-a-Judge**:
```bash
export GEMINI_API_KEY="your-google-gemini-api-key"
uv run python low-code-rapid-prototype-v1/evals/run_evaluations.py --gemini-key $GEMINI_API_KEY
```

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

* **Zero Plaintext Secrets:** Model API keys and admin credentials are injected at container startup via **Google Secret Manager**.
* **Persistent Storage:** Cloud Run integrates a **Google Cloud Storage (GCS) FUSE** mount (`gs://flowise-data-<PROJECT_ID>`) to persist Flowise SQLite databases, sessions, and document stores across restarts.
* **Cost Efficiency:** Automated scale-to-zero (`min-instances: 0`, `max-instances: 5`).

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
