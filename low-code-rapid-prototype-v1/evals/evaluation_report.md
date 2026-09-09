# Scenario 5 -- RAG Triad Evaluation Report

> [!IMPORTANT]
> **Evaluation mode: `fixture`.** This harness scores a pre-recorded dataset. The
> `retrieved_context` and `generated_response` fields come from
> `evaluation_dataset.json`; the deployed Flowise agent is **not** invoked. A regression in
> the live system would not change these scores. Replacing this with a harness that calls
> the running service is Phase 3 of the
> [v2 roadmap](../../pro-code-production-service-v2/README.md).

**Project:** Scenario 5 -- Digital Economy Research & Report Agent (Multi-Agent + HITL)
**System under test:** `Groq openai/gpt-oss-20b (Flowise Agentflow v2, temperature 0.3)`
**Judge:** `Google Gemini (chain: gemini-3.8-flash -> gemini-3.7-flash -> gemini-3.6-flash -> gemini-3.5-flash -> gemini-3-flash-preview)` -- a different model family from the system under test, to avoid self-preference bias
**Judge models served:** `gemini-3.7-flash`, `gemini-3.8-flash` — **BLENDED, chain advanced mid-run**
**Evaluation framework:** RAG Triad (TruLens / RAGAS methodology) & reference comparison
**Run (UTC):** `2026-09-09T03:58:14.317721+00:00`
**Dataset SHA-256:** `50690196a05ac3c3...`
**Runner SHA-256:** `a94923a3bbc1f66d...`
**Overall result:** **INCONCLUSIVE**

---

## 1. Metric Scorecard

| Metric | Score | Target | Status | Description |
| :--- | :---: | :---: | :---: | :--- |
| Context Relevance | 0.980 | >= 0.80 | `PASS` | Are the retrieved chunks relevant to the research query? |
| Groundedness / Faithfulness | 0.390 | >= 0.85 | `INCONCLUSIVE` | Are the Writer Agent's claims supported by retrieved context? (hallucination check) |
| Answer Relevance | 0.960 | >= 0.80 | `PASS` | Does the report answer the prompt and follow the 4-part structure? |
| RAG Triad Composite | 0.777 | >= 0.80 | `INCONCLUSIVE` | Arithmetic mean of the three triad dimensions. |
| Token F1 vs Reference | 0.990 | >= 0.70 | `INCONCLUSIVE` | Lexical token overlap against curated reference answers. |
| Structure Completeness | 1.00 | >= 0.80 | `PASS` | Presence of Executive Summary, Key Findings, Enablers, Conclusion. |

### Caveats

* JUDGE CHANGED MID-RUN. Cases were graded by 2 different models (gemini-3.7-flash, gemini-3.8-flash) because the fallback chain advanced on quota exhaustion. Different models score differently, so the aggregate is a blend of judges, not a single measurement. Re-run when quota allows, or pin one model with --judge-models <name>, before quoting these numbers.
* Token F1 >= 0.95 on TC-01, TC-02, TC-03, TC-04, TC-05: the fixture's generated_response is a near-copy of its reference_answer, so this metric measures the dataset, not the system.
* GROUNDEDNESS IS INCONCLUSIVE for TC-01, TC-02, TC-03, TC-04, TC-05. The recorded retrieved_context is shorter than 50% of the recorded generated_response (context/response char ratio: TC-01 0.27, TC-02 0.27, TC-03 0.20, TC-04 0.18, TC-05 0.20), so it cannot support the answer regardless of how the live system behaved. These scores measure the fixture's truncated context, NOT hallucination by the agent. Capturing the full retrieved chunk set requires a harness that calls the live system (Phase 3).

---

## 2. Per-Query Breakdown

| Test ID | Query | Ctx Rel. | Grnd. | Ans Rel. | Triad | F1 | Judge | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| TC-01 | Write a brief report on the shift from "... | 1.0 | 0.55 | 1.0 | 0.85 | 0.952 | `gemini-3.7-flash` | `INCONCLUSIVE` |
| TC-02 | Summarise the SEA-6 economies' ambitions... | 1.0 | 0.5 | 0.95 | 0.817 | 1.0 | `gemini-3.8-flash` | `INCONCLUSIVE` |
| TC-03 | What are the key enablers for sustainabl... | 1.0 | 0.35 | 0.95 | 0.767 | 1.0 | `gemini-3.8-flash` | `INCONCLUSIVE` |
| TC-04 | What are the report's main recommendatio... | 1.0 | 0.35 | 0.95 | 0.767 | 1.0 | `gemini-3.7-flash` | `INCONCLUSIVE` |
| TC-05 | How does the report address digital trus... | 0.9 | 0.2 | 0.95 | 0.683 | 1.0 | `gemini-3.8-flash` | `INCONCLUSIVE` |

---

## 3. Per-Case Detail

### [TC-01] Write a brief report on the shift from "Tech for Growth" to "Tech for Good" in Southeast Asia.

* **Triad:** context `1.0` | groundedness `0.55` | answer `1.0`
* **Token F1 vs reference:** `0.952`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The retrieved context is directly relevant to the query. The generated response follows the user prompt exceptionally well with a structured mini-report; however, it extrapolates significantly beyond the brief context snippet, introducing ungrounded specifics such as ASEAN DEFA, data center energy certifications, digital capability vouchers, and urban-rural divides.

---

### [TC-02] Summarise the SEA-6 economies' ambitions and objectives for the digital economy.

* **Triad:** context `1.0` | groundedness `0.5` | answer `0.95`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The retrieved context is perfectly relevant to the query. The generated response follows a clear 4-part report structure and thoroughly answers the query, but it introduces a significant amount of external knowledge and ungrounded specifics (e.g., green compute, corridors in Johor and Cyberjaya, ASEAN DEFA, specific industries for Thailand) not present in the retrieved context.

---

### [TC-03] What are the key enablers for sustainable digital development identified in the report?

* **Triad:** context `1.0` | groundedness `0.35` | answer `0.95`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The retrieved context directly answers the query with the four core pillars. The generated response follows a comprehensive 4-part report format and addresses the query directly. However, groundedness is low because the response introduces extensive specific details, institutions (IMDA / Tech for Good Institute), regional scoping (SEA-6), and policy recommendations that are not present in the provided context.

---

### [TC-04] What are the report's main recommendations for advancing the region's digital economy?

* **Triad:** context `1.0` | groundedness `0.35` | answer `0.95`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The retrieved context is perfectly relevant to the query. However, the generated response hallucinates extensive background details, organizations (IMDA/Tech for Good Institute), statistics, and specific policy names not found in the retrieved context. The response thoroughly answers the user's question and is well-structured.

---

### [TC-05] How does the report address digital trust, cybersecurity, and responsible AI governance?

* **Triad:** context `0.9` | groundedness `0.2` | answer `0.95`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The retrieved context is directly relevant to the query topic. However, the generated response is severely ungrounded, fabricating detailed external policies (Model AI Governance Framework, AI Verify, ASEAN data flow mechanisms, anti-scam hubs) that are not present in the provided snippet. The response nonetheless directly answers the query and effectively adopts the 4-part report format.

---

## 4. Reproducing This Run

```bash
# From the repository root, with a real judge (publishable scores)
export GEMINI_API_KEY="your-google-api-key"
uv run python main.py eval --judge gemini

# Lexical proxies only -- always reported UNVERIFIED, exit code 2
uv run python main.py eval --judge heuristic
```

Exit codes: `0` all gates passed, `1` a gate failed, `2` run unverified.
