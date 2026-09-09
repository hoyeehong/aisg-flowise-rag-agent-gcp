# Evaluation harness

Scores the **running service over HTTP**, not a fixture. That is the whole point: v1's
suite scored pre-recorded responses, so a regression in the deployed system produced
identical numbers. Anything reachable over HTTP — local `uvicorn`, a container, a Cloud
Run URL — is evaluable by the same harness.

## Running it

```bash
# Retrieval only: no chat model is invoked. With the deterministic embedder this needs
# no API key and no quota, and it is what CI gates on.
uv run python -m evals.harness --target http://localhost:8000 --retrieval-only --gate

# Full evaluation with a pinned LLM judge (spends provider quota).
export GEMINI_API_KEY=...
uv run python -m evals.harness --target http://localhost:8000 --judge-model gemini-3.6-flash

# Record a new baseline. Never implicit -- a self-updating baseline always agrees with
# the latest run and can therefore never detect a regression.
uv run python -m evals.harness --target ... --retrieval-only \
    --baseline evals/baselines/retrieval-ci.json --write-baseline
```

Exit codes: `0` passed, `1` a gate finding (below a floor, or a regression beyond
tolerance), `2` the run is unusable (dataset error, service not ready, judge failure,
incomparable baseline, or too few cases completed).

## Golden sets

| File | Cases | Scores | Ground truth |
| :--- | :---: | :--- | :--- |
| `retrieval-anchored.v1.jsonl` | 10 | retrieval + generation | Pages whose extracted text contains a distinctive literal term. A page containing `DEFA` *is* relevant to a query about DEFA — verifiable by grep, and not circular with the retriever under test. |
| `policy-analysis.v1.jsonl` | 5 | **generation only** | None. v1's `ground_truth_context` is a synthesised summary, not a verbatim excerpt: it matched no single page above 38%, so it cannot establish retrieval ground truth. Fabricating `expected_pages` would make recall a measurement of guesses. |
| `adversarial-jailbreak.v1.jsonl` | 5 | behaviour | Prompt injection, role override, false premises, fabrication bait. Each carries `must_not_contain` guards. |
| `out-of-scope.v1.jsonl` | 4 | behaviour | Answerable-sounding questions the corpus cannot support. Every absence was verified: `responsible ai`, `european union`, `latin america`, `corporate tax` and `quantum computing` appear on **0** of 75 pages. |

`OOS-01` is worth noting: v1's research prompt instructs the agent to check the
"Responsible AI" enabler, and that phrase appears nowhere in the report. A grounded
system must report the gap rather than invent coverage.

Datasets are append-only within a version. A changed expectation means a new `.v<n>`
file, so an old score stays reproducible.

## What the metrics mean

**Retrieval and generation are scored separately.** In v1 a retrieval regression and a
generation regression were indistinguishable in one aggregate, so neither was
actionable.

Three preconditions decide whether a metric is reported at all, rather than being
scored as zero. Scoring a broken setup produces a number that looks like a system
failure:

1. **No verifiable ground truth** → retrieval not scored for that case.
2. **Ground truth not in the index** → retrieval not scored. Three of the ten anchored
   cases hit this on the first run, because their pages had not been ingested.
   `GET /v1/corpus` supplies the coverage that makes this checkable.
3. **Context shorter than half the answer** → groundedness flagged unmeasurable, the
   Phase 0 lesson encoded.

**`recall_normalised` is the metric the gate uses.** Raw `recall@k` is bounded by
`k/|expected|`: with 19 relevant pages and `k=5` it cannot exceed 0.26 however good the
retriever is. One case scored a raw 0.385 that was in fact a perfect 1.000 — it
retrieved every page it could. The normalised value answers "of what was retrievable at
this `k`, how much did we retrieve", and is comparable across cases.

**The judge runs several trials and reports spread.** Phase 0 measured groundedness
varying 0.500 / 0.480 / 0.460 / 0.440 across identical runs of one model at
temperature 0, and 0.380–0.460 across models. One trial is a sample. The judge model is
pinned rather than chained, because a chain that advances mid-run blends two graders
into one aggregate.

## The gate

| Rule | Behaviour |
| :--- | :--- |
| Completion below 90% | **Unusable** (exit 2), not failed. An aggregate over a fraction of the set is not comparable to a baseline, and a rate-limited provider must not be reported as a quality result. |
| Forbidden content present | Hard failure. Zero tolerance: it means content was invented or instructions leaked. |
| Below an absolute floor | Failure. Catches "never good enough". |
| Worse than baseline beyond tolerance | Failure. Catches "got worse", which floors alone miss — a drop from 0.95 to 0.81 passes a 0.80 floor. |
| Dataset fingerprint or judge model differs from the baseline | **Unusable**. Otherwise a dataset change is reported as a system regression. |

Retrieval tolerances are tight (0.02) because retrieval is deterministic given a fixed
corpus and embedder. Generation tolerances are 0.08 to cover the measured judge spread —
a zero-tolerance gate on a judge-scored metric fails on noise, and a gate that cries
wolf gets switched off.

## Current baseline

`evals/baselines/retrieval-ci.json`, recorded in `--retrieval-only` mode against the
full 75-page corpus with the deterministic embedder:

| Metric | Value |
| :--- | :---: |
| recall@k (raw) | 0.384 |
| mean recall ceiling | 0.713 |
| **recall normalised** | **0.580** |
| precision@k | 0.420 |
| MRR | 0.800 |
| nDCG@k | 0.632 |

Two consecutive runs are byte-identical, which is why this can be a required check
rather than an advisory one.

## Bugs this harness found on its first live run

Both were invisible to 104 passing tests, and both were in code shipped in Phase 2.

**The lexical half of hybrid retrieval never fired.** `websearch_to_tsquery` ANDs bare
terms, so passing a whole question required one chunk to contain every word including
"how", "does" and "report". Every question-form query returned zero lexical hits, making
"hybrid" search vector search with extra steps. Phase 2 documented the conjunctive
behaviour and even tested it — without connecting that it made the retriever inert.
Queries are now reduced to OR-joined content terms.

**MMR's `lambda` was inert.** RRF produces scores around `1/(60+rank)` ≈ 0.016 while
redundancy is a 0–1 Jaccard overlap, so `lambda * relevance` was swamped by
`(1-lambda) * redundancy` and every lambda behaved like 0.0 — pure diversity. This is
the same scale mismatch RRF itself avoids by fusing ranks rather than scores,
reintroduced between the two stages. Fused scores are normalised before the trade-off.

Together these moved one case's recall from 0.000 to 1.000. Disabling the lexical half
now measurably costs 0.10 normalised recall, 0.10 MRR and 0.11 nDCG — before the fix it
cost nothing, because it contributed nothing.

## Known limits

* **A full-corpus baseline with real models does not exist yet.** Groq's free tier
  allows 8,000 tokens per minute against roughly 2,500 per report, so a 24-case live run
  exceeds the budget. The client retries with backoff and the gate reports an incomplete
  run as unusable, but a real-model baseline needs a paid tier or a slow nightly job.
* **Generation metrics are therefore ungated.** CI gates retrieval and the harness's own
  correctness; the judge path is exercised manually.
* **`policy-analysis.v1.jsonl` contributes no retrieval signal**, by design — see above.
* **Refusal detection is a phrase heuristic.** It is deliberately broad, since for a
  refusal check, missing a genuine refusal is the costlier error.
