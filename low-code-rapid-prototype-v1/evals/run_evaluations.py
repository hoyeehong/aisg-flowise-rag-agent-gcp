#!/usr/bin/env python3
"""
Scenario 5: Digital Economy Research & Report Agent (Multi-Agent + HITL)
RAG Triad evaluation runner.

SCOPE AND LIMITS OF THIS HARNESS
--------------------------------
This runner scores a *fixture*, not a running system. `retrieved_context` and
`generated_response` are pre-recorded in the dataset; nothing here calls the Flowise
prediction API. A regression in the deployed agent would therefore produce identical
scores. Every report is stamped `evaluation_mode: fixture` to make that explicit.

Replacing this with a harness that calls the live API is Phase 3 of the v2 roadmap
(see ../../pro-code-production-service-v2/README.md).

JUDGE MODES
-----------
  gemini     LLM-as-a-judge via Google Gemini. Scores are measurements.
  openai     LLM-as-a-judge via OpenAI. Scores are measurements.
  heuristic  Lexical proxies only (token overlap). Scores are NOT measurements of
             relevance or groundedness, and the run is always reported as UNVERIFIED.
             Intended for offline plumbing checks, never for publication.

A judge failure is a hard error. This runner does not substitute a fallback score for a
failed judge call, because a number that cannot fail is not evidence.

EXIT CODES
----------
  0  all gates passed
  1  one or more gates failed
  2  run could not be verified (heuristic mode, or judge unavailable)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# The system under test, as declared in flowise_scenario_5_workflow.json.
SYSTEM_UNDER_TEST = "Groq openai/gpt-oss-20b (Flowise Agentflow v2, temperature 0.3)"

# Judges are deliberately kept on a different model family from the system under test:
# a model grading its own output exhibits self-preference bias.
GEMINI_JUDGE_MODEL = "gemini-3-flash-preview"
OPENAI_JUDGE_MODEL = "gpt-4o-mini"

# Evaluation rubric thresholds
THRESHOLDS = {
    "context_relevance": 0.80,
    "groundedness": 0.85,
    "answer_relevance": 0.80,
    "rag_triad_composite": 0.80,
    "token_f1_vs_reference": 0.70,
    "structure_completeness": 0.80,
}

# Token F1 above this against the reference answer indicates the fixture's
# generated_response is a near-copy of its reference_answer, which makes the
# metric a measure of the dataset rather than of the system.
FIXTURE_LEAKAGE_F1 = 0.95

# A groundedness score is only meaningful if the recorded context could plausibly
# support the recorded answer. When retrieved_context is far shorter than
# generated_response, the fixture captured an excerpt rather than the full retrieved
# set, and a low groundedness score measures the dataset, not the system.
MIN_CONTEXT_COVERAGE = 0.5

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'it', 'this', 'that', 'these', 'those', 'as', 'be',
    'from', 'which', 'who', 'whom', 'their', 'they', 'we', 'our', 'you', 'your',
}


class JudgeError(RuntimeError):
    """Raised when an LLM judge cannot produce a usable score."""


def tokenize(text):
    """Simple clean tokenizer for statistical evaluation."""
    tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def compute_token_f1(pred_text, target_text):
    """Compute token-level precision, recall, and F1."""
    pred_tokens = set(tokenize(pred_text))
    target_tokens = set(tokenize(target_text))

    if not pred_tokens or not target_tokens:
        return 0.0, 0.0, 0.0

    common = pred_tokens & target_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_structure(response_text):
    """Verify presence of required 4-part report sections."""
    required_sections = [
        r'executive summary|summary',
        r'key findings|regional analysis',
        r'strategic enablers|recommendations|policy recommendations',
        r'conclusion|future outlook',
    ]
    text_lower = response_text.lower()
    present = sum(1 for p in required_sections if re.search(p, text_lower))
    return present / len(required_sections)


JUDGE_PROMPT = """You are an expert AI Evaluator assessing a RAG application.
Evaluate the following test sample on a continuous scale from 0.0 to 1.0 for three metrics:

1. context_relevance: How relevant is the retrieved context to the user query? (1.0 = highly relevant, 0.0 = completely irrelevant)
2. groundedness: How faithful is the generated response to the retrieved context? Does it avoid ungrounded hallucinations? (1.0 = completely faithful, 0.0 = completely hallucinated)
3. answer_relevance: How well does the generated response directly answer the user's query and follow the requested 4-part report structure? (1.0 = perfect answer, 0.0 = irrelevant)

Score what you actually observe. Do not inflate scores.

Input:
[Query]: {query}
[Retrieved Context]: {context}
[Generated Response]: {response}

Return ONLY a valid JSON object in this exact format:
{{"context_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0, "reasoning": "Short explanation of the scores."}}
"""

REQUIRED_JUDGE_KEYS = ("context_relevance", "groundedness", "answer_relevance")


def _parse_judge_payload(raw_text, source):
    """Validate a judge's JSON response. Missing scores are an error, not a default."""
    try:
        scores = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{source} returned non-JSON output: {exc}") from exc

    missing = [k for k in REQUIRED_JUDGE_KEYS if k not in scores]
    if missing:
        raise JudgeError(f"{source} omitted required score(s): {', '.join(missing)}")

    parsed = {}
    for key in REQUIRED_JUDGE_KEYS:
        try:
            value = float(scores[key])
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"{source} returned non-numeric {key}: {scores[key]!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise JudgeError(f"{source} returned {key} outside [0,1]: {value}")
        parsed[key] = value

    reasoning = str(scores.get("reasoning", "")).strip() or f"No reasoning supplied by {source}."
    return parsed["context_relevance"], parsed["groundedness"], parsed["answer_relevance"], reasoning


def judge_gemini(query, context, response, api_key, timeout=60):
    """LLM-as-a-Judge via the Google Gemini REST API (no extra pip dependencies)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_JUDGE_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": JUDGE_PROMPT.format(
            query=query, context=context, response=response)}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Surface the API's own message; never leak the key in the URL.
        raise JudgeError(f"Gemini judge HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JudgeError(f"Gemini judge unreachable: {exc}") from exc

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise JudgeError(f"Gemini judge returned no candidate content: {json.dumps(data)[:300]}") from exc

    return _parse_judge_payload(text, f"Gemini ({GEMINI_JUDGE_MODEL})")


def judge_openai(query, context, response, api_key):
    """LLM-as-a-Judge via the OpenAI SDK."""
    try:
        import openai
    except ImportError as exc:
        raise JudgeError("openai package not installed; `uv add openai` to use --judge openai") from exc

    try:
        client = openai.OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model=OPENAI_JUDGE_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                query=query, context=context, response=response)}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # openai raises a wide surface of transport errors
        raise JudgeError(f"OpenAI judge call failed: {exc}") from exc

    return _parse_judge_payload(res.choices[0].message.content, f"OpenAI ({OPENAI_JUDGE_MODEL})")


def judge_heuristic(query, context, response):
    """
    Lexical proxies for the RAG Triad. NOT a measurement of relevance or groundedness.

    These are plain token-overlap ratios with no calibration constants and no floors.
    They will score low on well-formed output, because lexical overlap is a poor proxy
    for semantic relevance -- which is the reason a real judge is required for any
    published number. A heuristic run is always reported as UNVERIFIED.
    """
    _, context_relevance, _ = compute_token_f1(context, query)      # query recall in context
    groundedness, _, _ = compute_token_f1(response, context)        # response precision vs context
    _, query_recall, _ = compute_token_f1(response, query)
    answer_relevance = (evaluate_structure(response) + query_recall) / 2.0

    return (
        context_relevance,
        groundedness,
        answer_relevance,
        "Lexical token-overlap proxy only; not a semantic judgement. Run with "
        "--judge gemini for a publishable score.",
    )


def _build_caveats(leakage_flagged, insufficient_context, results):
    """Assemble the caveats that qualify how these numbers may be read."""
    caveats = []
    if leakage_flagged:
        caveats.append(
            f"Token F1 >= {FIXTURE_LEAKAGE_F1} on {', '.join(leakage_flagged)}: the fixture's "
            f"generated_response is a near-copy of its reference_answer, so this metric "
            f"measures the dataset, not the system."
        )
    if insufficient_context:
        ratios = ", ".join(
            f"{r['query_id']} {r['context_coverage_ratio']:.2f}"
            for r in results if r["query_id"] in insufficient_context
        )
        caveats.append(
            f"GROUNDEDNESS IS INCONCLUSIVE for {', '.join(insufficient_context)}. The recorded "
            f"retrieved_context is shorter than {MIN_CONTEXT_COVERAGE:.0%} of the recorded "
            f"generated_response (context/response char ratio: {ratios}), so it cannot support "
            f"the answer regardless of how the live system behaved. These scores measure the "
            f"fixture's truncated context, NOT hallucination by the agent. Capturing the full "
            f"retrieved chunk set requires a harness that calls the live system (Phase 3)."
        )
    return caveats


def score_case(case, judge_mode, gemini_key, openai_key):
    """Dispatch a single test case to the configured judge."""
    query = case["query"]
    context = case.get("retrieved_context", "")
    response = case.get("generated_response", "")

    if judge_mode == "gemini":
        return judge_gemini(query, context, response, gemini_key)
    if judge_mode == "openai":
        return judge_openai(query, context, response, openai_key)
    return judge_heuristic(query, context, response)


EXIT_OK, EXIT_GATE_FAILED, EXIT_UNVERIFIED = 0, 1, 2


def _die_unverified(message):
    """Exit 2: the run could not be verified (distinct from a gate failure)."""
    sys.stdout.flush()  # keep stderr ordered after buffered stdout
    print(message, file=sys.stderr)
    raise SystemExit(EXIT_UNVERIFIED)


def resolve_judge_mode(requested, gemini_key, openai_key):
    """Pick a judge, failing loudly rather than silently degrading."""
    if requested == "gemini":
        if not gemini_key:
            _die_unverified("[FATAL] --judge gemini requires GEMINI_API_KEY (or --gemini-key).")
        return "gemini"
    if requested == "openai":
        if not openai_key:
            _die_unverified("[FATAL] --judge openai requires OPENAI_API_KEY (or --openai-key).")
        return "openai"
    if requested == "heuristic":
        return "heuristic"

    # auto
    if gemini_key:
        return "gemini"
    if openai_key:
        return "openai"
    _die_unverified(
        "[FATAL] No judge available. Set GEMINI_API_KEY or OPENAI_API_KEY, or pass\n"
        "        --judge heuristic to run lexical proxies only (reported UNVERIFIED)."
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_evaluation_pipeline(dataset_path, output_json_path, output_md_path,
                            judge_mode="auto", gemini_key=None, openai_key=None):
    """Executes the evaluation suite across the benchmark dataset."""
    judge_mode = resolve_judge_mode(judge_mode, gemini_key, openai_key)
    verified = judge_mode in ("gemini", "openai")
    judge_label = {
        "gemini": f"Google Gemini ({GEMINI_JUDGE_MODEL})",
        "openai": f"OpenAI ({OPENAI_JUDGE_MODEL})",
        "heuristic": "Lexical token-overlap proxy (UNVERIFIED)",
    }[judge_mode]

    print("=" * 63)
    print("  Scenario 5 -- RAG Triad Evaluation (fixture-scored)")
    print("=" * 63)
    print(f"System:  {SYSTEM_UNDER_TEST}")
    print(f"Judge:   {judge_label}")
    print(f"Dataset: {dataset_path}")
    if not verified:
        print("\n[WARNING] Heuristic mode produces lexical proxies, not measurements.")
        print("          This run cannot pass and must not be published as a score.\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    if not test_cases:
        _die_unverified(f"[FATAL] Dataset {dataset_path} is empty.")

    results = []
    leakage_flagged = []
    insufficient_context = []

    for tc in test_cases:
        qid = tc["query_id"]
        try:
            c_rel, ground, a_rel, reason = score_case(tc, judge_mode, gemini_key, openai_key)
        except JudgeError as exc:
            _die_unverified(f"[FATAL] Judge failed on {qid}: {exc}\n"
                            f"        Refusing to substitute a fallback score.")

        _, _, f1 = compute_token_f1(tc.get("generated_response", ""), tc.get("reference_answer", ""))
        struct_score = evaluate_structure(tc.get("generated_response", ""))

        ctx_chars = len(tc.get("retrieved_context", ""))
        resp_chars = len(tc.get("generated_response", "")) or 1
        coverage = ctx_chars / resp_chars
        context_sufficient = coverage >= MIN_CONTEXT_COVERAGE
        triad_avg = (c_rel + ground + a_rel) / 3.0

        # Real conjunctive gate: every threshold must be met.
        gates = {
            "context_relevance": c_rel >= THRESHOLDS["context_relevance"],
            "groundedness": ground >= THRESHOLDS["groundedness"],
            "answer_relevance": a_rel >= THRESHOLDS["answer_relevance"],
            "rag_triad_composite": triad_avg >= THRESHOLDS["rag_triad_composite"],
            "token_f1_vs_reference": f1 >= THRESHOLDS["token_f1_vs_reference"],
            "structure_completeness": struct_score >= THRESHOLDS["structure_completeness"],
        }
        failed = [k for k, ok in gates.items() if not ok]

        # Do not attribute a groundedness failure to the system when the fixture
        # cannot support the measurement.
        if not context_sufficient and "groundedness" in failed:
            insufficient_context.append(qid)

        status = "UNVERIFIED" if not verified else ("PASS" if not failed else "FAIL")
        if not context_sufficient:
            status = "INCONCLUSIVE" if verified else "UNVERIFIED"

        if f1 >= FIXTURE_LEAKAGE_F1:
            leakage_flagged.append(qid)

        results.append({
            "query_id": qid,
            "query": tc["query"],
            "context_relevance": round(c_rel, 3),
            "groundedness": round(ground, 3),
            "answer_relevance": round(a_rel, 3),
            "rag_triad_average": round(triad_avg, 3),
            "token_f1_vs_reference": round(f1, 3),
            "structure_completeness": round(struct_score, 2),
            "context_chars": ctx_chars,
            "response_chars": resp_chars,
            "context_coverage_ratio": round(coverage, 2),
            "context_sufficient_for_groundedness": context_sufficient,
            "status": status,
            "failed_gates": failed,
            "evaluation_reasoning": reason,
        })
        detail = f" | failed: {', '.join(failed)}" if failed else ""
        print(f"[{qid}] {status:10s} Triad {triad_avg:.2f} "
              f"(ctx {c_rel:.2f}, grnd {ground:.2f}, ans {a_rel:.2f}) F1 {f1:.2f}{detail}")

    n = len(results)
    mean = {
        "context_relevance": sum(r["context_relevance"] for r in results) / n,
        "groundedness_faithfulness": sum(r["groundedness"] for r in results) / n,
        "answer_relevance": sum(r["answer_relevance"] for r in results) / n,
        "mean_token_f1_vs_reference": sum(r["token_f1_vs_reference"] for r in results) / n,
        "mean_structure_completeness": sum(r["structure_completeness"] for r in results) / n,
    }
    # Arithmetic mean of the three triad dimensions.
    mean["rag_triad_composite"] = (
        mean["context_relevance"] + mean["groundedness_faithfulness"] + mean["answer_relevance"]
    ) / 3.0

    aggregate_gates = {
        "context_relevance": mean["context_relevance"] >= THRESHOLDS["context_relevance"],
        "groundedness": mean["groundedness_faithfulness"] >= THRESHOLDS["groundedness"],
        "answer_relevance": mean["answer_relevance"] >= THRESHOLDS["answer_relevance"],
        "rag_triad_composite": mean["rag_triad_composite"] >= THRESHOLDS["rag_triad_composite"],
        "token_f1_vs_reference": mean["mean_token_f1_vs_reference"] >= THRESHOLDS["token_f1_vs_reference"],
        "structure_completeness": mean["mean_structure_completeness"] >= THRESHOLDS["structure_completeness"],
    }
    failed_aggregate = [k for k, ok in aggregate_gates.items() if not ok]
    failed_cases = [r["query_id"] for r in results if r["status"] == "FAIL"]
    inconclusive_cases = [r["query_id"] for r in results if r["status"] == "INCONCLUSIVE"]

    # If every groundedness failure is explained by insufficient recorded context, the
    # metric is inconclusive rather than failing, and is excluded from the verdict.
    groundedness_inconclusive = (
        len(insufficient_context) == len(results)
        and "groundedness" in failed_aggregate
    )
    if groundedness_inconclusive:
        failed_aggregate = [k for k in failed_aggregate if k != "groundedness"]

    token_f1_inconclusive = len(leakage_flagged) == len(results)

    if not verified:
        overall_status = "UNVERIFIED"
        exit_code = EXIT_UNVERIFIED
    elif (groundedness_inconclusive or token_f1_inconclusive) and not failed_aggregate and not failed_cases:
        overall_status = "INCONCLUSIVE"
        exit_code = EXIT_UNVERIFIED
    elif failed_aggregate or failed_cases:
        overall_status = "FAILED"
        exit_code = EXIT_GATE_FAILED
    else:
        overall_status = "PASSED"
        exit_code = EXIT_OK

    summary = {
        "provenance": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_mode": "fixture",
            "evaluation_mode_note": (
                "Scores a pre-recorded dataset, not a live system. retrieved_context and "
                "generated_response come from the dataset; the Flowise prediction API is not called."
            ),
            "system_under_test": SYSTEM_UNDER_TEST,
            "judge_mode": judge_mode,
            "judge": judge_label,
            "verified": verified,
            "dataset_path": os.path.basename(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "runner_sha256": sha256_file(os.path.abspath(__file__)),
        },
        "scenario": "Scenario 5: IMDA SEA Digital Economy Report",
        "total_test_cases": n,
        "overall_status": overall_status,
        "failed_aggregate_gates": failed_aggregate,
        "failed_cases": failed_cases,
        "inconclusive_cases": inconclusive_cases,
        "groundedness_inconclusive": groundedness_inconclusive,
        "mean_scores": {k: round(v, 3) for k, v in mean.items()},
        "thresholds": THRESHOLDS,
        "token_f1_inconclusive": len(leakage_flagged) == len(results),
        "caveats": _build_caveats(leakage_flagged, insufficient_context, results),
        "results": results,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    generate_markdown_report(summary, output_md_path)

    print("\n" + "=" * 63)
    print("                 EVALUATION SUMMARY SCORECARD")
    print("=" * 63)
    print(f" Overall Status:          {overall_status}")
    print(f" Judge:                   {judge_label}")
    print(f" Mode:                    fixture (not a live system)")
    print(f" RAG Triad Composite:     {mean['rag_triad_composite']:.3f} (target >= {THRESHOLDS['rag_triad_composite']})")
    print(f"  |- Context Relevance:   {mean['context_relevance']:.3f} (target >= {THRESHOLDS['context_relevance']})")
    print(f"  |- Groundedness:        {mean['groundedness_faithfulness']:.3f} (target >= {THRESHOLDS['groundedness']})")
    print(f"  '- Answer Relevance:    {mean['answer_relevance']:.3f} (target >= {THRESHOLDS['answer_relevance']})")
    print(f" Token F1 vs Reference:   {mean['mean_token_f1_vs_reference']:.3f} (target >= {THRESHOLDS['token_f1_vs_reference']})")
    print(f" Structure Completeness:  {mean['mean_structure_completeness']*100:.0f}% (target >= {THRESHOLDS['structure_completeness']*100:.0f}%)")
    if failed_aggregate:
        print(f" Failed aggregate gates:  {', '.join(failed_aggregate)}")
    if failed_cases:
        print(f" Failed cases:            {', '.join(failed_cases)}")
    if inconclusive_cases:
        print(f" Inconclusive cases:      {', '.join(inconclusive_cases)}")
    if groundedness_inconclusive:
        print(" Groundedness:            EXCLUDED from verdict (fixture cannot support it)")
    for c in summary["caveats"]:
        print(f"\n [CAVEAT] {c}")
    print(f"\n Report: {output_md_path}")
    print(f" JSON:   {output_json_path}")
    print(f" Exit:   {exit_code}")
    print("=" * 63 + "\n")
    return summary, exit_code


def generate_markdown_report(summary, md_path):
    """Formats the evaluation summary into a Markdown report."""
    mean, th, prov = summary["mean_scores"], summary["thresholds"], summary["provenance"]
    verified = prov["verified"]

    def gate(value, key):
        if not verified:
            return "`UNVERIFIED`"
        if key == "groundedness" and summary.get("groundedness_inconclusive"):
            return "`INCONCLUSIVE`"
        if key == "token_f1_vs_reference" and summary.get("token_f1_inconclusive"):
            return "`INCONCLUSIVE`"
        return "`PASS`" if value >= th[key] else "`FAIL`"

    banner = "" if verified else (
        "> [!WARNING]\n"
        "> **This run is UNVERIFIED.** It was scored by lexical token-overlap proxies, not by an\n"
        "> LLM judge. The numbers below measure word overlap, not relevance or groundedness, and\n"
        "> must not be quoted as evaluation results. Re-run with `--judge gemini` to obtain\n"
        "> publishable scores.\n\n"
    )

    md = f"""# Scenario 5 -- RAG Triad Evaluation Report

{banner}> [!IMPORTANT]
> **Evaluation mode: `fixture`.** This harness scores a pre-recorded dataset. The
> `retrieved_context` and `generated_response` fields come from
> `{prov['dataset_path']}`; the deployed Flowise agent is **not** invoked. A regression in
> the live system would not change these scores. Replacing this with a harness that calls
> the running service is Phase 3 of the
> [v2 roadmap](../../pro-code-production-service-v2/README.md).

**Project:** Scenario 5 -- Digital Economy Research & Report Agent (Multi-Agent + HITL)
**System under test:** `{prov['system_under_test']}`
**Judge:** `{prov['judge']}` -- a different model family from the system under test, to avoid self-preference bias
**Evaluation framework:** RAG Triad (TruLens / RAGAS methodology) & reference comparison
**Run (UTC):** `{prov['timestamp_utc']}`
**Dataset SHA-256:** `{prov['dataset_sha256'][:16]}...`
**Runner SHA-256:** `{prov['runner_sha256'][:16]}...`
**Overall result:** **{summary['overall_status']}**

---

## 1. Metric Scorecard

| Metric | Score | Target | Status | Description |
| :--- | :---: | :---: | :---: | :--- |
| Context Relevance | {mean['context_relevance']:.3f} | >= {th['context_relevance']:.2f} | {gate(mean['context_relevance'], 'context_relevance')} | Are the retrieved chunks relevant to the research query? |
| Groundedness / Faithfulness | {mean['groundedness_faithfulness']:.3f} | >= {th['groundedness']:.2f} | {gate(mean['groundedness_faithfulness'], 'groundedness')} | Are the Writer Agent's claims supported by retrieved context? (hallucination check) |
| Answer Relevance | {mean['answer_relevance']:.3f} | >= {th['answer_relevance']:.2f} | {gate(mean['answer_relevance'], 'answer_relevance')} | Does the report answer the prompt and follow the 4-part structure? |
| RAG Triad Composite | {mean['rag_triad_composite']:.3f} | >= {th['rag_triad_composite']:.2f} | {gate(mean['rag_triad_composite'], 'rag_triad_composite')} | Arithmetic mean of the three triad dimensions. |
| Token F1 vs Reference | {mean['mean_token_f1_vs_reference']:.3f} | >= {th['token_f1_vs_reference']:.2f} | {gate(mean['mean_token_f1_vs_reference'], 'token_f1_vs_reference')} | Lexical token overlap against curated reference answers. |
| Structure Completeness | {mean['mean_structure_completeness']:.2f} | >= {th['structure_completeness']:.2f} | {gate(mean['mean_structure_completeness'], 'structure_completeness')} | Presence of Executive Summary, Key Findings, Enablers, Conclusion. |

"""
    if summary["caveats"]:
        md += "### Caveats\n\n" + "".join(f"* {c}\n" for c in summary["caveats"]) + "\n"

    md += """---

## 2. Per-Query Breakdown

| Test ID | Query | Ctx Rel. | Grnd. | Ans Rel. | Triad | F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in summary["results"]:
        md += (f"| {r['query_id']} | {r['query'][:45]}... | {r['context_relevance']} | "
               f"{r['groundedness']} | {r['answer_relevance']} | {r['rag_triad_average']} | "
               f"{r['token_f1_vs_reference']} | `{r['status']}` |\n")

    md += "\n---\n\n## 3. Per-Case Detail\n\n"
    for r in summary["results"]:
        md += f"""### [{r['query_id']}] {r['query']}

* **Triad:** context `{r['context_relevance']}` | groundedness `{r['groundedness']}` | answer `{r['answer_relevance']}`
* **Token F1 vs reference:** `{r['token_f1_vs_reference']}`
* **Structure completeness:** `{r['structure_completeness']:.0%}`
* **Status:** `{r['status']}`{f" -- failed gates: {', '.join(r['failed_gates'])}" if r['failed_gates'] else ""}
* **Judge reasoning:** {r['evaluation_reasoning']}

---

"""

    md += """## 4. Reproducing This Run

```bash
# From the repository root, with a real judge (publishable scores)
export GEMINI_API_KEY="your-google-api-key"
uv run python main.py eval --judge gemini

# Lexical proxies only -- always reported UNVERIFIED, exit code 2
uv run python main.py eval --judge heuristic
```

Exit codes: `0` all gates passed, `1` a gate failed, `2` run unverified.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Run RAG Triad evaluations for Scenario 5 (fixture-scored).")
    parser.add_argument("--dataset", default=os.path.join(BASE_DIR, "evaluation_dataset.json"))
    parser.add_argument("--output-json", default=os.path.join(BASE_DIR, "eval_results.json"))
    parser.add_argument("--output-md", default=os.path.join(BASE_DIR, "evaluation_report.md"))
    parser.add_argument("--judge", choices=["auto", "gemini", "openai", "heuristic"], default="auto",
                        help="Judge to use. 'auto' prefers Gemini, then OpenAI, then fails.")
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()

    _, code = run_evaluation_pipeline(
        args.dataset, args.output_json, args.output_md,
        judge_mode=args.judge, gemini_key=args.gemini_key, openai_key=args.openai_key,
    )
    sys.exit(code)
