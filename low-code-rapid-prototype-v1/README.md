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

Phase 0 (the "truth pass") corrected the items marked **[fixed]** below. The rest are
properties of the low-code approach rather than defects to patch in place, and each is
carried into [v2](../pro-code-production-service-v2/README.md) as a requirement.

### Fixed in Phase 0

3. ~~**The heuristic judge cannot fail.**~~ **[fixed]** Score floors (0.85 context
   relevance, 0.88 groundedness) sat *above* the pass thresholds, so those metrics passed
   by construction. The floors are gone; a judge failure is now a hard error (exit 2)
   rather than a substituted score, and heuristic mode is reported `UNVERIFIED` and can
   never pass.
4. ~~**The pass/fail gate is a no-op.**~~ **[fixed]** Both branches of the status ternary
   returned `PASS`. Gates are now genuinely conjunctive, with exit codes `0` pass /
   `1` gate failed / `2` unverified, so the runner can gate CI.
5. ~~**Model drift between docs and artifact.**~~ **[fixed]** The agents run on **Groq
   `openai/gpt-oss-20b`**; docs previously described Gemini throughout. Gemini is now
   documented in its actual role — the *evaluation judge*, deliberately a different model
   family from the system under test to avoid self-preference bias. The deploy script
   also provisioned only a Gemini key and now provisions Groq as well.
7. ~~**Single-writer persistence.**~~ **[fixed]** `MAX_INSTANCES` now defaults to `1`,
   because GCS FUSE lacks the POSIX advisory locking SQLite requires. Also in Phase 0:
   the admin password moved from `--set-env-vars` (readable via
   `gcloud run services describe` and Cloud Audit Logs) into Secret Manager, the service
   now runs as a dedicated least-privilege service account with per-secret bindings
   instead of the shared default compute SA, the project-level `secretAccessor` grant is
   removed, and the image is digest-pinned where resolvable.

### Carried into v2

1. **Runtime config is not version controlled.** The graph references its knowledge base
   by an opaque Document Store UUID. Chunk size, overlap, embedding model, index name and
   top-K all live in Flowise server state, so the retrieval half of the RAG pipeline
   cannot be reproduced from this repository. → v2 Phase 2.
2. **The evaluation suite scores fixtures, not the system.** `retrieved_context` and
   `generated_response` are pre-recorded; nothing calls the live prediction API. A
   regression in the deployed agent would produce identical scores. Every report is now
   stamped `evaluation_mode: fixture` so this is explicit. → v2 Phase 3.
6. **No tests, no CI, no IaC.** There is nothing to unit test, because no code path
   between the user and the model is owned by this repository. → v2 Phases 1 and 4.
8. **The fixture cannot support a groundedness measurement.** *(found during Phase 0)*
   Recorded `retrieved_context` is 18–27% the length of `generated_response` (261–451
   chars against 1,446–1,680), so no answer could be grounded in it. The Gemini judge
   scores groundedness ≈0.48, tracking the truncation almost 1:1 — TC-05 has both the
   smallest context ratio (0.20) and the lowest groundedness (0.20). The runner now
   detects this and reports `INCONCLUSIVE`, excluding the metric from the verdict rather
   than blaming the agent. Fixing it requires capturing the real retrieved chunk set from
   a live call. → v2 Phase 3.
9. **Judge variance is unmeasured.** *(found during Phase 0)* Two consecutive runs at
   `temperature 0.0` returned groundedness `0.500` and `0.480`. A single judge run is a
   sample, not a constant; the harness should run *n* trials and report spread.
   → v2 Phase 3.

## Current baseline

Context relevance `1.000` and answer relevance `1.000` (both `PASS`); groundedness and
Token F1 `INCONCLUSIVE` for the reasons above; overall `INCONCLUSIVE`. See
[`evals/evaluation_report.md`](evals/evaluation_report.md) and the
[root README](../README.md) for the full scorecard and provenance.

```bash
# Discover which judge models your key can call
uv run python main.py eval --list-judge-models

# Publishable scores (requires a judge key). Pin one model: a run whose judge
# changed mid-chain is reported INCONCLUSIVE, since the aggregate blends judges.
export GEMINI_API_KEY="..."
uv run python main.py eval --judge gemini --judge-models "gemini-3.6-flash"

# Default 5-model fallback chain (survives per-model quota exhaustion)
uv run python main.py eval --judge gemini

# Lexical proxies only -- always UNVERIFIED, exit 2
uv run python main.py eval --judge heuristic
```
