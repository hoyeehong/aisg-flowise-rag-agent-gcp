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
**Judge:** `Google Gemini (gemini-3-flash-preview)` -- a different model family from the system under test, to avoid self-preference bias
**Evaluation framework:** RAG Triad (TruLens / RAGAS methodology) & reference comparison
**Run (UTC):** `2026-09-09T03:26:39.059763+00:00`
**Dataset SHA-256:** `50690196a05ac3c3...`
**Runner SHA-256:** `33099022908d6c8b...`
**Overall result:** **INCONCLUSIVE**

---

## 1. Metric Scorecard

| Metric | Score | Target | Status | Description |
| :--- | :---: | :---: | :---: | :--- |
| Context Relevance | 1.000 | >= 0.80 | `PASS` | Are the retrieved chunks relevant to the research query? |
| Groundedness / Faithfulness | 0.440 | >= 0.85 | `INCONCLUSIVE` | Are the Writer Agent's claims supported by retrieved context? (hallucination check) |
| Answer Relevance | 1.000 | >= 0.80 | `PASS` | Does the report answer the prompt and follow the 4-part structure? |
| RAG Triad Composite | 0.813 | >= 0.80 | `PASS` | Arithmetic mean of the three triad dimensions. |
| Token F1 vs Reference | 0.990 | >= 0.70 | `INCONCLUSIVE` | Lexical token overlap against curated reference answers. |
| Structure Completeness | 1.00 | >= 0.80 | `PASS` | Presence of Executive Summary, Key Findings, Enablers, Conclusion. |

### Caveats

* Token F1 >= 0.95 on TC-01, TC-02, TC-03, TC-04, TC-05: the fixture's generated_response is a near-copy of its reference_answer, so this metric measures the dataset, not the system.
* GROUNDEDNESS IS INCONCLUSIVE for TC-01, TC-02, TC-03, TC-04, TC-05. The recorded retrieved_context is shorter than 50% of the recorded generated_response (context/response char ratio: TC-01 0.27, TC-02 0.27, TC-03 0.20, TC-04 0.18, TC-05 0.20), so it cannot support the answer regardless of how the live system behaved. These scores measure the fixture's truncated context, NOT hallucination by the agent. Capturing the full retrieved chunk set requires a harness that calls the live system (Phase 3).

---

## 2. Per-Query Breakdown

| Test ID | Query | Ctx Rel. | Grnd. | Ans Rel. | Triad | F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TC-01 | Write a brief report on the shift from "Tech ... | 1.0 | 0.6 | 1.0 | 0.867 | 0.952 | `INCONCLUSIVE` |
| TC-02 | Summarise the SEA-6 economies' ambitions and ... | 1.0 | 0.5 | 1.0 | 0.833 | 1.0 | `INCONCLUSIVE` |
| TC-03 | What are the key enablers for sustainable dig... | 1.0 | 0.4 | 1.0 | 0.8 | 1.0 | `INCONCLUSIVE` |
| TC-04 | What are the report's main recommendations fo... | 1.0 | 0.5 | 1.0 | 0.833 | 1.0 | `INCONCLUSIVE` |
| TC-05 | How does the report address digital trust, cy... | 1.0 | 0.2 | 1.0 | 0.733 | 1.0 | `INCONCLUSIVE` |

---

## 3. Per-Case Detail

### [TC-01] Write a brief report on the shift from "Tech for Growth" to "Tech for Good" in Southeast Asia.

* **Triad:** context `1.0` | groundedness `0.6` | answer `1.0`
* **Token F1 vs reference:** `0.952`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The context is highly relevant as it provides the core themes for the report. The response is perfectly structured and answers the query directly. However, the groundedness score is lower because the response introduces significant external information not found in the retrieved context, such as specific policy recommendations (vouchers, AI literacy), mentions of 'ASEAN DEFA', and the 'urban vs rural' paradox.

---

### [TC-02] Summarise the SEA-6 economies' ambitions and objectives for the digital economy.

* **Triad:** context `1.0` | groundedness `0.5` | answer `1.0`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The context was highly relevant to the query. The response followed the requested 4-part report structure perfectly and answered the prompt comprehensively. However, the groundedness score is lower because the response included significant external information not found in the retrieved context, such as 'green compute', specific locations in Malaysia (Johor/Cyberjaya), specific industries in Thailand (automotive/electronics), and the ASEAN DEFA framework.

---

### [TC-03] What are the key enablers for sustainable digital development identified in the report?

* **Triad:** context `1.0` | groundedness `0.4` | answer `1.0`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness, rag_triad_composite
* **Judge reasoning:** The context is highly relevant as it explicitly lists the four pillars requested. The response perfectly follows the requested 4-part report structure and answers the query. However, the groundedness score is low because the response includes many specific details (e.g., fiber, subsea cables, 5G, anti-scam measures, clean energy partnerships, and AI ethics sandboxes) that are not present in the provided context snippet, constituting hallucinations relative to the source text.

---

### [TC-04] What are the report's main recommendations for advancing the region's digital economy?

* **Triad:** context `1.0` | groundedness `0.5` | answer `1.0`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness
* **Judge reasoning:** The context is highly relevant as it provides the specific recommendations requested. The response follows the 4-part report structure perfectly and answers the query. However, the groundedness score is reduced because the response includes significant external information not found in the retrieved context, such as the specific report authors (IMDA/Tech for Good Institute), the $1 trillion valuation, specific MSME statistics, and references to Singapore's AI Verify.

---

### [TC-05] How does the report address digital trust, cybersecurity, and responsible AI governance?

* **Triad:** context `1.0` | groundedness `0.2` | answer `1.0`
* **Token F1 vs reference:** `1.0`
* **Structure completeness:** `100%`
* **Status:** `INCONCLUSIVE` -- failed gates: groundedness, rag_triad_composite
* **Judge reasoning:** The retrieved context is highly relevant to the query as it directly mentions the three pillars requested. The generated response perfectly follows the requested 4-part report structure and answers the query comprehensively. However, the groundedness is very low because the response introduces a significant amount of external information not found in the retrieved context, such as specific details on phishing, 'AI Verify', 'Model AI Governance Framework', and 'ASEAN Cross-Border Data Flows Mechanism'.

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
