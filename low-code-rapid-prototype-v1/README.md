# v1 — Low-Code Rapid Prototype (Flowise Agentflow v2)

**Status:** Shipped and live. Frozen for feature work; maintained as the stakeholder-facing
demo surface and as the behavioural baseline that v2 must match or beat.

This tier exists to answer one question fast: *does a multi-agent research → write → human-review
loop produce useful policy analysis over the IMDA SEA digital economy report?* It answered yes in
days rather than weeks, which is exactly what a low-code surface is for.

## What's here

| Path | Role |
| :--- | :--- |
| `flowise_scenario_5_workflow.json` | The 6-node Agentflow graph: Start → Research Agent → Writer Agent → HITL Gate → Loop → Direct Reply |
| `prompts/` | System prompts for the Research and Writer agents |
| `evals/` | RAG Triad scoring suite and the 5-case benchmark dataset |
| `deploy/` | Cloud Run deployment of the upstream `flowiseai/flowise` container |

## Running it

```bash
# Evaluations (from the repository root)
uv run python main.py eval

# Deploy the Flowise container to Cloud Run
bash low-code-rapid-prototype-v1/deploy/deploy_gcp.sh
```

Import instructions for the workflow and Document Store are in the
[root README](../README.md).

## Known limitations of this tier

These are properties of the low-code approach, not defects to patch in place. Each one is a
requirement carried into [v2](../pro-code-production-service-v2/README.md).

1. **Runtime config is not version controlled.** The graph references its knowledge base by an
   opaque Document Store UUID. Chunk size, overlap, embedding model, index name and top-K all live
   in Flowise server state, so the retrieval half of the RAG pipeline cannot be reproduced from
   this repository.
2. **The evaluation suite scores fixtures, not the system.** `retrieved_context` and
   `generated_response` are pre-baked in `evals/evaluation_dataset.json`; nothing calls the live
   prediction API. A regression in the agent would produce identical scores.
3. **The heuristic judge cannot fail.** The deterministic fallback in `evals/run_evaluations.py`
   applies score floors (0.85 context relevance, 0.88 groundedness) that sit *above* the pass
   thresholds. Published scores from that path are not measurements.
4. **The pass/fail gate is a no-op.** Both branches of the status ternary return `PASS`, so the
   Token-F1 threshold never gates anything.
5. **Model drift between docs and artifact.** This graph is configured for Groq
   `openai/gpt-oss-20b`; the root README and eval reports describe Google Gemini throughout.
6. **No tests, no CI, no IaC.** There is nothing to unit test, because there is no code path
   between the user and the model that this repository owns.
7. **Single-writer persistence.** Flowise state is SQLite on a GCS FUSE mount with
   `max-instances=5`. FUSE does not provide the POSIX locking SQLite needs; concurrent writers
   risk corruption.

> [!NOTE]
> Items 3, 4 and 5 are documentation-accuracy issues rather than architectural ones, and are
> scheduled as the first commit of the v2 effort (Phase 0 — Truth pass). Until then, treat the
> scores in `evals/evaluation_report.md` as unverified.
