# Digital Economy Research & Report Agent (Agentic Workflow with Human-in-the-Loop)

**Learner:** Ho Yee Hong

**Scenario:** 5 — *From "Tech for Growth" to "Tech for Good": Shaping the Next Phase of Southeast Asia's Growth through the Digital Economy* (IMDA / Tech for Good Institute)

**Build Type:** Multi-Agent Agentic Workflow (Flowise Agentflow v2) with Document Store RAG & Human-in-the-Loop (HITL)  
**LLM Engine:** `Google Gemini 3 Flash Preview` (`chatGoogleGenerativeAI`)  
**Embedding Engine:** `Google text-embedding-004` (768-dim, Asymmetric Retrieval)  
**Workflow File:** `yeehong_ho_scenario_5.json`

---

## 1. Executive Summary & Scenario Rationale

### Why Scenario 5?
In Southeast Asia's rapidly maturing digital landscape, national governments and enterprises are pivoting from purely measuring Gross Merchandise Value (GMV) / digital transaction volume ("Tech for Growth") toward building sustainable, equitable, and trustworthy digital ecosystems ("Tech for Good"). 

The **IMDA / Tech for Good Institute Special Report** presents dense policy frameworks, cross-border analyses across the **SEA-6 economies** (Singapore, Malaysia, Indonesia, Thailand, the Philippines, and Vietnam), and four foundational enablers (*Infrastructure, Talent, Trust/Cybersecurity, and Governance*). 

Analyzing and synthesizing this 50+ page policy paper into actionable executive mini-reports requires more than simple naive Q&A:
1. **Specialized Division of Labor:** Separation between factual, high-recall research retrieval (`Research Agent`) and coherent, structured policy synthesis (`Writer Agent`).
2. **Enterprise Human-in-the-Loop (HITL) Governance:** Real-world think tanks and public sector agencies cannot rely on unsupervised one-shot generation. Analysts require an interactive review gate to approve, adjust emphasis, or request iterative revisions.
3. **Multi-Turn Adaptive Feedback Loop:** Incorporating human critique dynamically through iterative looping until the report meets executive standards.

---

## 2. Architecture & Multi-Agent Pipeline

```
                                  +-------------------------------------------------+
                                  |             Flowise Agentflow Canvas            |
                                  +-------------------------------------------------+

                                                      +------------------------+
                                                      | Google text-embed-004  |
                                                      | (768-dim, Asymmetric)  |
                                                      +-----------+------------+
                                                                  |
                                                                  v
[User Query] ---> [Start Node] ---> [Research Agent] ---> [Writer Agent] ---> [Human Review Gate (HITL)]
                                    (Gemini 3 Flash Preview)    (Gemini 3 Flash Preview)           |
                                                                               +-------+-------+
                                                                               |               |
                                                             [Approved] (Output 0)    [Revise] (Output 1)
                                                                       |                       |
                                                                       v                       v
                                                             [Final Outcome]        [Loop to Writer Agent]
                                                             (Published Report)     (Max 5 iterations)
```

### Pipeline Component Breakdown:
* **Start Node (`startAgentflow`):** Accepts the analyst's research topic or prompt.
* **Research Agent (`agentAgentflow`):**
  * Powered by **`Google Gemini 3 Flash Preview`** (temperature: `0.1`) equipped with a Flowise **Document Store RAG Tool**.
  * Connected to Google's **`text-embedding-004`** index using asymmetric retrieval to query facts across SEA-6 country data, regional benchmarks, and the 4 structural enablers with source citations.
* **Writer Agent (`llmAgentflow`):**
  * Powered by **`Google Gemini 3 Flash Preview`** (temperature: `0.3`, memory: `allMessages`).
  * Ingests both original user prompt and raw retrieved findings to synthesize a publication-grade brief structured into: **Executive Summary**, **Key Findings & Regional Analysis**, **Strategic Enablers & Policy Recommendations**, and **Conclusion**.
* **Human Review Gate (`humanInputAgentflow`):**
  * Pauses execution and presents the drafted report to the analyst with interactive actions:
    * **Approve:** Passes directly to `DirectReply` for final sign-off.
    * **Request Revision:** Captures user critique (e.g., *"Focus more on Singapore's green data center roadmap"*).
* **Loop Node (`loopAgentflow`):**
  * Routes human revision feedback back into the Writer Agent (up to 5 iterations) ensuring iterative refinement without manual copy-pasting.
* **Final Outcome Node (`directReplyAgentflow`):**
  * Delivers the approved, verified policy brief.

---

## 3. Key Design Decisions

### A. Google Embedding & Vector Store Strategy (Module 2 Concepts)
* **Embedding Model:** Google **`text-embedding-004`** (768 dimensions, cosine similarity).
  * **Asymmetric Task Types:** Indexes document chunks using `RETRIEVAL_DOCUMENT` and transforms queries using `RETRIEVAL_QUERY`. This asymmetric mapping yields superior semantic alignment compared to symmetric embeddings.
  * **Efficiency:** 768 dimensions provide higher retrieval accuracy on MTEB (~66.3) while using **50% less RAM/storage** than standard 1536-dim vectors.
* **Vector Store Options:**
  * **Production Cloud (Pinecone Serverless):** Index configured with `768` dimensions, `cosine` metric, and namespace `imda-sea-report` for isolated, persistent cloud vector search.
  * **Local In-Memory:** Built-in Flowise in-memory store for rapid local development.
* **Text Splitter:** `RecursiveCharacterTextSplitter` configured with:
  * **Chunk Size:** `1,000 characters` (~200–250 tokens). Keeps section headings together with their analytical context.
  * **Chunk Overlap:** `200 characters` (20% overlap). Maintains semantic continuity across chunk boundaries.
* **Top-K Retrieval:** Set to `Top-K = 5` to gather broad cross-country context across all SEA-6 nations simultaneously.

### B. Google Gemini 3 Flash Preview LLM Selection (Module 1 & 3 Concepts)
* **Sub-Second Latency & 1M Context Window:** `gemini-3-flash-preview` executes retrieval reasoning in under 500ms and eliminates "lost-in-the-middle" degradation across dense policy contexts.
* **Multilingual SEA Competence:** Native tokenization for Southeast Asian terminology (e.g. MSME formalization in Indonesia, MyDIGITAL in Malaysia, ASEAN DEFA).
* **Role Calibration:**
  * **Research Agent Prompt:** Enforces strict zero-hallucination grounding with country-specific breakouts (`[Singapore]`, `[Indonesia]`, etc.).
  * **Writer Agent Prompt:** Enforces strict markdown report taxonomy and mandates incorporating human review feedback.

---

## 4. Challenges Faced & Resolutions

| Challenge Encountered | Root Cause | Engineering Resolution |
| :--- | :--- | :--- |
| **Cross-Country Fact Blending** | LLM tended to conflate Malaysia's infrastructure goals with Indonesia's digital talent initiatives when asked general queries. | Added structured extraction instructions to the Research Agent prompt, mandating itemized country breakouts (e.g. `[Singapore]`, `[Indonesia]`, `[Vietnam]`). |
| **Over-Summarization in Writer Node** | Writer agent initially generated generic high-level summaries without retaining granular statistics. | Added explicit instruction in the Writer prompt: *"Retain all quantitative metrics, dates, and initiative names from the Research Agent's output."* |
| **State Retention during HITL Loop** | Early iterations lost user feedback context across loops. | Enabled `allMessages` conversation memory on the LLM nodes and mapped the loop handle back to the Writer node input state. |

---

## 5. Automated Evaluation Suite & Benchmarks (Module 4.2 Aligned)

A fully automated, reproducible evaluation suite is implemented in the [`evals/`](evals/) directory, adhering to the **RAG Triad** (TruLens / RAGAS) and **LLM-as-a-Judge** methodology taught in **AISG Module 4.2**:

### A. RAG Triad Scorecard (Google Stack Benchmark Results)

| Metric | Score | Target | Status | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Context Relevance** | **0.908** | $\ge 0.80$ | `PASS` | Evaluates if retrieved chunks from the IMDA PDF (via `text-embedding-004`) are strictly relevant to the research query. |
| **Groundedness / Faithfulness** | **0.880** | $\ge 0.85$ | `PASS` | Evaluates whether claims made by the Writer Agent (`gemini-3-flash-preview`) are fully grounded in retrieved facts (hallucination check). |
| **Answer Relevance** | **0.871** | $\ge 0.80$ | `PASS` | Evaluates whether the generated mini-report answers the prompt and adheres to the 4-part structure. |
| **RAG Triad Composite** | **0.887** | $\ge 0.80$ | **`PASSED`** | Harmonic composite across the RAG Triad. |
| **Token F1 vs Reference** | **0.990** | $\ge 0.70$ | `PASS` | Lexical and semantic overlap against calibrated expert reference reports. |
| **Structure Completeness** | **100%** | $\ge 80\%$ | `PASS` | Verifies presence of Executive Summary, Key Findings, Strategic Enablers, and Conclusion. |

### B. Evaluation Assets Included
* [`evals/run_evaluations.py`](evals/run_evaluations.py): Automated Python evaluation runner supporting Google Gemini LLM-as-a-Judge and statistical rubrics.
* [`evals/evaluation_dataset.json`](evals/evaluation_dataset.json) & [`evals/evaluation_dataset.csv`](evals/evaluation_dataset.csv): 5 benchmark test queries with ground-truth contexts, reference answers, and model responses.
* [`evals/evaluation_report.md`](evals/evaluation_report.md): Automatically generated markdown benchmark scorecard.
* [`evals/evaluations_notebook.ipynb`](evals/evaluations_notebook.ipynb): Interactive Jupyter evaluation notebook.

To run the evaluations locally:
```bash
python evals/run_evaluations.py
```

---

### C. Sample Conversation Runs & Verification

#### Query 1: *"Write a brief report on the shift from 'Tech for Growth' to 'Tech for Good' in Southeast Asia."*
* **Research Agent Output:** Identified foundational themes: transition from volume metrics (e-commerce GMV, user adoption) to digital inclusion, sustainability, trustworthy AI, and SME resilience across ASEAN.
* **Writer Agent Draft:** Formulated 4-part executive report detailing why the initial wave of digital growth created unintended digital divides, and how "Tech for Good" establishes sustainable long-term economic resilience.
* **HITL Action:** *Approved by Analyst.*

#### Query 2: *"Summarise the SEA-6 economies' ambitions and objectives for the digital economy."*
* **Research Agent Output:** Extracted country-specific goals:
  * **Singapore:** Global digital innovation hub, AI governance leadership, green data centers.
  * **Indonesia:** Digital inclusion, MSME digital onboarding, rural connectivity.
  * **Malaysia:** MyDIGITAL blueprint, digital investment acceleration.
  * **Thailand & Vietnam:** National 4.0 strategy, digital talent development, semiconductor/manufacturing digitalization.
  * **Philippines:** E-governance adoption, digital payments scaling.
* **Writer Agent Draft:** Produced comparative matrix and executive narrative highlighting divergence in digital maturity and convergence under ASEAN DEFA.
* **HITL Action:** *Approved by Analyst.*

#### Query 3: *"What are the key enablers for sustainable digital development identified in the report?"*
* **Research Agent Output:** Extracted the 4 core pillars: (1) Resilient Digital Infrastructure, (2) Digital Talent & Future-Ready Skills, (3) Digital Trust & Cybersecurity (Responsible AI & Cross-Border Data), (4) Regulatory Cohesion (ASEAN DEFA).
* **Writer Agent Draft:** Generated structured briefing with actionable policy levers per pillar.
* **HITL Action:** *Revision Requested: "Please expand on the Digital Trust and Responsible AI pillar."*
* **Refined Output:** Writer Agent re-generated Section 3 with expanded focus on Model AI Governance Framework and cross-border data alignment across ASEAN.

---

## 6. How to Run & Verify in Flowise (Google Option A + Pinecone / In-Memory)

1. Open **Flowise** (`npx flowise start` or local Docker at `http://localhost:3000`).
2. **Create Credentials in Flowise:**
   * **Google API:** Go to **Credentials** $\rightarrow$ **Add Credential** $\rightarrow$ **Google Generative AI API** (Enter API Key from [Google AI Studio](https://aistudio.google.com/)).
   * **Pinecone API (Optional for Cloud Vector Store):** Go to **Credentials** $\rightarrow$ **Add Credential** $\rightarrow$ **Pinecone API** (Enter API Key from [Pinecone Console](https://app.pinecone.io/)).
3. **Configure Document Store:**
   * Go to **Document Stores** $\rightarrow$ **Add New** $\rightarrow$ Name: `imda_sea_digital_economy_report`.
   * **1. Document Loader:** Select **PDF File Loader** $\rightarrow$ Upload [`imda_report.pdf`](imda_report.pdf).
   * **2. Text Splitter:** Select **Recursive Character Text Splitter** (`Chunk Size: 1000`, `Chunk Overlap: 200`).
   * **3. Embeddings:** Select **Google GenerativeAI Embeddings** (`text-embedding-004`) $\rightarrow$ Connect your Google API Key.
   * **4. Vector Store:**
     * *Option A (Pinecone Cloud):* Select **Pinecone** $\rightarrow$ Connect Pinecone API Key credential $\rightarrow$ Index: `ladp-capstone` $\rightarrow$ Namespace: `imda-sea-report`.
     * *Option B (Local Development):* Select **In-Memory** (or **Memory Vector Store**).
   * **5. Record Manager:** Select **SQLite Record Manager** (or **Default / None**).
   * Click **Save & Upsert Chunk** (wait for the green success notification).
4. **Import the Agentflow Workflow:**
   * Go to **Agentflows** $\rightarrow$ Click **Add New** $\rightarrow$ **Settings (Gear Icon)** $\rightarrow$ **Load / Import Chatflow**.
   * Select [`scenario_5_imda_digital_economy_agentflow3.json`](scenario_5_imda_digital_economy_agentflow3.json).
   * Confirm the `Research Agent` and `Writer Agent` nodes are connected to your **Google Generative AI** credential.
   * Click **Save** and open the chat window to test.

---

## 7. Cloud Deployment to Google Cloud Run (Module 4.3 Aligned)

For production deployment to Google Cloud Platform (Singapore region `asia-southeast1`) with **Cloud Storage (GCS FUSE)** persistence and **Google Secret Manager**, an automated deployment script and guide are available:

* **Deployment Script:** [`deploy/deploy_gcp.sh`](deploy/deploy_gcp.sh)
* **Configuration Template:** [`deploy/env.example`](deploy/env.example)
* **Full Deployment Guide:** [`deploy/README_DEPLOY_GCP.md`](deploy/README_DEPLOY_GCP.md)

### Quick Deploy Command:
```bash
./deploy/deploy_gcp.sh
```
