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
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# The system under test, as declared in flowise_scenario_5_workflow.json.
SYSTEM_UNDER_TEST = "Groq openai/gpt-oss-20b (Flowise Agentflow v2, temperature 0.3)"

# Judges are deliberately kept on a different model family from the system under test:
# a model grading its own output exhibits self-preference bias.
#
# JUDGE FALLBACK CHAIN
# Free-tier Gemini enforces per-model quotas, so a 429 on one model does not imply a
# 429 on the next -- falling back across model versions recovers a run that would
# otherwise abort. Ordered newest-first; each entry was verified present via the
# ListModels API (run with --list-judge-models to re-check against your own key).
#
# Deliberately EXCLUDED: `gemini-flash-latest` and other moving aliases. An alias can
# silently change which model produced a score, which destroys attribution -- the whole
# point of recording judge identity in provenance.
DEFAULT_GEMINI_JUDGE_CHAIN = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
)
OPENAI_JUDGE_MODEL = "gpt-4o-mini"

# Retry policy for transient judge failures (429 / 5xx / timeout).
JUDGE_MAX_ATTEMPTS_PER_MODEL = 3
JUDGE_BACKOFF_BASE_SECONDS = 2.0
# If the API asks us to wait longer than this, stop waiting and try the next model
# instead. A long retryDelay signals a daily quota rather than a burst rate limit.
JUDGE_MAX_WAIT_SECONDS = 45.0

# Last resort: when EVERY model in the chain is rate-limited, there is nothing left to
# fall back to, and free-tier limits are often per-minute (the API reports a delay of
# ~30-60s). Waiting once is then far better than aborting the run. Bounded so a genuine
# daily-quota exhaustion still fails fast rather than hanging.
JUDGE_COOLDOWN_BUDGET_SECONDS = 90.0

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


class ModelUnavailable(JudgeError):
    """This model will not work for the rest of the run (404 / 400 / 403)."""


class ModelRateLimited(JudgeError):
    """This model is out of quota (429).

    Kept distinct from ModelTransient because the response differs: a quota reset is
    minutes-to-hours away, so sleeping on it stalls the run. The right move is to
    advance to the next model in the chain immediately.
    """


class ModelTransient(JudgeError):
    """A transient server fault (5xx / timeout) that typically clears in seconds."""


# Retained as the umbrella type for callers that do not care which kind it was.
ModelExhausted = (ModelRateLimited, ModelTransient)


def _error_message(body, limit=160):
    """Pull the human-readable message out of a Google API error envelope."""
    try:
        msg = json.loads(body).get("error", {}).get("message", "")
    except (json.JSONDecodeError, AttributeError):
        msg = ""
    msg = " ".join((msg or body).split())
    return msg[:limit] + ("..." if len(msg) > limit else "")


def _classify_http_error(exc, body):
    """
    Map an HTTP failure onto a retry decision.

    ModelUnavailable -> wrong model or no access; retire it for the run.
    ModelRateLimited -> out of quota; advance to the next model now, do not sleep.
    ModelTransient   -> server fault; short backoff on the same model is worthwhile.
    """
    detail = f"HTTP {exc.code}: {_error_message(body)}"
    if exc.code in (400, 401, 403, 404):
        return ModelUnavailable(detail)
    if exc.code == 429:
        return ModelRateLimited(detail)
    return ModelTransient(detail)


def _retry_delay_from_body(body):
    """
    Extract Google's suggested retry delay, if present.

    The API returns it as a google.rpc.RetryInfo detail, e.g. {"retryDelay": "36s"}.
    """
    try:
        details = json.loads(body).get("error", {}).get("details", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    for d in details:
        if "RetryInfo" in str(d.get("@type", "")):
            raw = str(d.get("retryDelay", "")).rstrip("s")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _gemini_generate(model, prompt, api_key, timeout=60):
    """One generateContent call. Raises ModelUnavailable / ModelExhausted."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Read the body once; it carries both the message and any RetryInfo.
        body = exc.read().decode("utf-8", "replace")
        err = _classify_http_error(exc, body)
        err.retry_after = _retry_delay_from_body(body)
        raise err from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ModelTransient(f"unreachable: {exc}") from exc

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        # A blocked or empty candidate is a property of this request, not the model.
        raise ModelTransient(
            f"no candidate content: {json.dumps(data)[:200]}") from exc


class GeminiJudgeGateway:
    """
    Routes judge calls across a chain of Gemini models with retries and fallback.

    Selection is *sticky*: once a model answers, it keeps serving every subsequent
    case. Re-probing the chain per case would be slower and, worse, would let the
    judge identity oscillate mid-run -- which makes the aggregate score a blend of
    different judges rather than a measurement.
    """

    def __init__(self, chain, api_key, verbose=True):
        self.chain = list(chain)
        self.api_key = api_key
        self.verbose = verbose
        self.dead = {}            # model -> reason it was retired (permanent)
        self.rate_limited = {}    # model -> suggested retry delay, if any
        self.calls_by_model = {}  # model -> successful call count
        self.cooldowns_used = 0
        self._active = None

    @property
    def models_used(self):
        return [m for m, n in self.calls_by_model.items() if n > 0]

    def _candidates(self):
        """Active model first, then the rest of the chain, skipping retired ones."""
        ordered = ([self._active] if self._active else []) + \
                  [m for m in self.chain if m != self._active]
        return [m for m in ordered if m not in self.dead]

    def _try_model(self, model, prompt):
        """
        Attempt one model. Returns text, or raises the failure that ended the attempt.

        A 429 returns immediately without sleeping: quota does not come back in
        seconds, so the next model in the chain is a far better use of the time.
        Only 5xx/timeout earns a backoff retry against the same model.
        """
        last = None
        for attempt in range(1, JUDGE_MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                return _gemini_generate(model, prompt, self.api_key)
            except (ModelUnavailable, ModelRateLimited) as exc:
                if isinstance(exc, ModelRateLimited):
                    self.rate_limited[model] = getattr(exc, "retry_after", None)
                raise
            except ModelTransient as exc:
                last = exc
                if attempt == JUDGE_MAX_ATTEMPTS_PER_MODEL:
                    break
                suggested = getattr(exc, "retry_after", None)
                if suggested and suggested > JUDGE_MAX_WAIT_SECONDS:
                    break
                wait = suggested or min(
                    JUDGE_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                    JUDGE_MAX_WAIT_SECONDS)
                wait += random.uniform(0, 0.5 * wait)  # jitter: avoid lockstep retries
                if self.verbose:
                    print(f"    {model}: {exc} -> retry {attempt}/"
                          f"{JUDGE_MAX_ATTEMPTS_PER_MODEL} in {wait:.1f}s")
                time.sleep(wait)
        raise last

    def score(self, query, context, response):
        """Score one case, walking the chain until a model answers."""
        prompt = JUDGE_PROMPT.format(query=query, context=context, response=response)
        candidates = self._candidates()
        if not candidates:
            raise JudgeError(
                "every model in the judge chain is retired: "
                + "; ".join(f"{m} ({r})" for m, r in self.dead.items()))

        try:
            return self._attempt_chain(prompt, candidates)
        except JudgeError as first_failure:
            # Everything failed. If the blocker was rate limiting rather than outage,
            # one bounded cooldown is worth trying before giving up on the run.
            waits = [d for d in (self.rate_limited.get(m) for m in candidates) if d]
            if not waits:
                raise
            wait = min(waits)
            if wait > JUDGE_COOLDOWN_BUDGET_SECONDS:
                raise JudgeError(
                    f"{first_failure} -- soonest quota reset is ~{wait:.0f}s, beyond the "
                    f"{JUDGE_COOLDOWN_BUDGET_SECONDS:.0f}s cooldown budget. This looks "
                    f"like a daily quota; re-run later or use --judge-models to select a "
                    f"model with remaining quota.") from first_failure
            if self.cooldowns_used >= 1:
                raise JudgeError(
                    f"{first_failure} -- already spent a cooldown this run; not "
                    f"waiting again.") from first_failure

            self.cooldowns_used += 1
            wait += 2.0  # small buffer past the reported reset
            if self.verbose:
                print(f"    entire chain rate-limited; cooling down {wait:.0f}s "
                      f"(one-time) then retrying")
            time.sleep(wait)
            self.rate_limited.clear()
            return self._attempt_chain(prompt, self._candidates())

    def _attempt_chain(self, prompt, candidates):
        """Walk the candidate list once, returning the first successful score."""
        errors = []
        for model in candidates:
            try:
                text = self._try_model(model, prompt)
            except ModelUnavailable as exc:
                self.dead[model] = str(exc)
                errors.append(f"{model}: {exc}")
                if self.verbose:
                    print(f"    {model}: unavailable, retiring for this run -- {exc}")
                continue
            except ModelRateLimited as exc:
                errors.append(f"{model}: {exc}")
                if self.verbose:
                    delay = self.rate_limited.get(model)
                    hint = f" (resets in ~{delay:.0f}s)" if delay else ""
                    print(f"    {model}: quota exhausted{hint}, advancing chain")
                continue
            except ModelTransient as exc:
                errors.append(f"{model}: {exc}")
                if self.verbose:
                    print(f"    {model}: transient fault persisted, advancing chain")
                continue

            if model != self._active and self.verbose and self._active is not None:
                print(f"    judge switched: {self._active} -> {model}")
            self._active = model
            self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
            c, g, a, reason = _parse_judge_payload(text, f"Gemini ({model})")
            return c, g, a, reason, model

        raise JudgeError("all judge models failed for this case -- " + " | ".join(errors))


def list_gemini_models(api_key):
    """Print the models this key can actually call, for configuring the chain."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}&pageSize=200"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"[FATAL] ListModels HTTP {exc.code}: "
              f"{exc.read().decode('utf-8', 'replace')[:300]}", file=sys.stderr)
        return EXIT_UNVERIFIED

    names = sorted(
        m["name"].removeprefix("models/") for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", []))
    print(f"{len(names)} models support generateContent with this key:\n")
    for n in names:
        marker = "  <- in default chain" if n in DEFAULT_GEMINI_JUDGE_CHAIN else ""
        print(f"  {n}{marker}")
    missing = [m for m in DEFAULT_GEMINI_JUDGE_CHAIN if m not in names]
    if missing:
        print(f"\n[WARN] Default chain references models this key cannot call: "
              f"{', '.join(missing)}")
    return EXIT_OK


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


def _build_caveats(leakage_flagged, insufficient_context, results, judges_used=()):
    """Assemble the caveats that qualify how these numbers may be read."""
    caveats = []
    if len(judges_used) > 1:
        caveats.append(
            f"JUDGE CHANGED MID-RUN. Cases were graded by {len(judges_used)} different "
            f"models ({', '.join(judges_used)}) because the fallback chain advanced on "
            f"quota exhaustion. Different models score differently, so the aggregate is "
            f"a blend of judges, not a single measurement. Re-run when quota allows, or "
            f"pin one model with --judge-models <name>, before quoting these numbers."
        )
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


def score_case(case, judge_mode, gateway, openai_key):
    """
    Dispatch a single test case to the configured judge.

    Returns (context_relevance, groundedness, answer_relevance, reasoning, judge_model).
    """
    query = case["query"]
    context = case.get("retrieved_context", "")
    response = case.get("generated_response", "")

    if judge_mode == "gemini":
        return gateway.score(query, context, response)
    if judge_mode == "openai":
        return judge_openai(query, context, response, openai_key) + (OPENAI_JUDGE_MODEL,)
    return judge_heuristic(query, context, response) + ("lexical-proxy",)


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
                            judge_mode="auto", gemini_key=None, openai_key=None,
                            judge_chain=None):
    """Executes the evaluation suite across the benchmark dataset."""
    judge_mode = resolve_judge_mode(judge_mode, gemini_key, openai_key)
    verified = judge_mode in ("gemini", "openai")
    chain = list(judge_chain or DEFAULT_GEMINI_JUDGE_CHAIN)

    gateway = None
    if judge_mode == "gemini":
        gateway = GeminiJudgeGateway(chain, gemini_key)
        judge_label = f"Google Gemini (chain: {' -> '.join(chain)})"
    elif judge_mode == "openai":
        judge_label = f"OpenAI ({OPENAI_JUDGE_MODEL})"
    else:
        judge_label = "Lexical token-overlap proxy (UNVERIFIED)"

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
            c_rel, ground, a_rel, reason, judge_model = score_case(
                tc, judge_mode, gateway, openai_key)
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
            # Same inheritance at case level: drop the composite from the reasons.
            failed = [k for k in failed if k != "rag_triad_composite"]

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
            "judge_model": judge_model,
            "status": status,
            "failed_gates": failed,
            "evaluation_reasoning": reason,
        })
        detail = f" | failed: {', '.join(failed)}" if failed else ""
        print(f"[{qid}] {status:12s} Triad {triad_avg:.2f} "
              f"(ctx {c_rel:.2f}, grnd {ground:.2f}, ans {a_rel:.2f}) "
              f"F1 {f1:.2f} | judge {judge_model}{detail}")

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
        # The composite is the arithmetic mean of the three triad dimensions, so it
        # inherits groundedness's invalidity. Excluding groundedness from the verdict
        # while letting the composite fail on it would be incoherent.
        failed_aggregate = [k for k in failed_aggregate
                            if k not in ("groundedness", "rag_triad_composite")]

    token_f1_inconclusive = len(leakage_flagged) == len(results)

    # If the chain fell back mid-run, different cases were graded by different models,
    # so the aggregate is a blend of judges rather than one measurement. Record it and
    # refuse to call the run PASSED on that basis.
    judges_used = sorted({r["judge_model"] for r in results})
    judge_consistent = len(judges_used) == 1
    judge_fallback_occurred = verified and not judge_consistent

    if not verified:
        overall_status = "UNVERIFIED"
        exit_code = EXIT_UNVERIFIED
    elif judge_fallback_occurred:
        # Scores exist, but they are not attributable to a single judge.
        overall_status = "INCONCLUSIVE"
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
            "judge_chain": chain if judge_mode == "gemini" else None,
            "judge_models_used": judges_used,
            "judge_consistent": judge_consistent,
            "judge_calls_by_model": (gateway.calls_by_model if gateway else None),
            "judge_models_retired": (gateway.dead if gateway else None),
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
        "caveats": _build_caveats(leakage_flagged, insufficient_context, results, judges_used),
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
        print(" RAG Triad Composite:     EXCLUDED (contains groundedness)")
    if verified:
        served = ", ".join(f"{m} x{n}" for m, n in (gateway.calls_by_model.items()
                                                    if gateway else [(judges_used[0], n)]))
        print(f" Judge models served:     {served}")
        if gateway and gateway.dead:
            print(f" Judge models retired:    "
                  f"{', '.join(f'{m} ({r[:40]})' for m, r in gateway.dead.items())}")
        if not judge_consistent:
            print(" Judge consistency:       BLENDED -- chain advanced mid-run")
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
        if key == "rag_triad_composite" and summary.get("groundedness_inconclusive"):
            # Contains groundedness; inherits its invalidity.
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
**Judge models served:** {', '.join(f"`{m}`" for m in prov['judge_models_used'])}{'' if prov['judge_consistent'] else ' — **BLENDED, chain advanced mid-run**'}
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

| Test ID | Query | Ctx Rel. | Grnd. | Ans Rel. | Triad | F1 | Judge | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
"""
    for r in summary["results"]:
        md += (f"| {r['query_id']} | {r['query'][:40]}... | {r['context_relevance']} | "
               f"{r['groundedness']} | {r['answer_relevance']} | {r['rag_triad_average']} | "
               f"{r['token_f1_vs_reference']} | `{r['judge_model']}` | `{r['status']}` |\n")

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
    parser.add_argument("--judge-models", default=os.getenv("JUDGE_MODEL_CHAIN"),
                        help="Comma-separated Gemini fallback chain, newest first. "
                             "Pass a single name to pin one judge for reproducible scores. "
                             f"Default: {','.join(DEFAULT_GEMINI_JUDGE_CHAIN)}")
    parser.add_argument("--list-judge-models", action="store_true",
                        help="List models this API key can call, then exit.")
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()

    if args.list_judge_models:
        if not args.gemini_key:
            print("[FATAL] --list-judge-models requires GEMINI_API_KEY (or --gemini-key).",
                  file=sys.stderr)
            sys.exit(EXIT_UNVERIFIED)
        sys.exit(list_gemini_models(args.gemini_key))

    chain = None
    if args.judge_models:
        chain = [m.strip() for m in args.judge_models.split(",") if m.strip()]
        if not chain:
            print("[FATAL] --judge-models was empty after parsing.", file=sys.stderr)
            sys.exit(EXIT_UNVERIFIED)

    _, code = run_evaluation_pipeline(
        args.dataset, args.output_json, args.output_md,
        judge_mode=args.judge, gemini_key=args.gemini_key, openai_key=args.openai_key,
        judge_chain=chain,
    )
    sys.exit(code)
