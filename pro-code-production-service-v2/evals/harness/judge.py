"""
LLM-as-a-judge for generation quality.

Two constraints inherited from Phase 0, where the v1 judge was audited:

* **A judge failure is a hard error, never a substituted score.** v1's fallback applied
  score floors that sat above the pass thresholds, so its metrics could not fail.
* **Multiple trials, reported with spread.** Four runs of one model at temperature 0
  returned groundedness 0.500 / 0.480 / 0.460 / 0.440, and a different model returned
  0.380-0.460 on the same input. One trial is a sample.

The judge model is pinned by the caller rather than chained. A chain that advances
mid-run blends two graders into one aggregate, which Phase 0 reported as INCONCLUSIVE.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .metrics import mean_and_stdev
from .types import JudgeScores

DEFAULT_JUDGE_MODEL = "gemini-3.6-flash"
DEFAULT_TRIALS = 3

PROMPT = """You are evaluating a retrieval-augmented policy analysis system.

Score three metrics on a continuous 0.0-1.0 scale. Score what you observe; do not \
inflate.

1. context_relevance: does the retrieved context address the query?
2. groundedness: is every claim in the response supported by the retrieved context? \
A response that correctly states the context cannot answer the query is fully \
grounded and scores high.
3. answer_relevance: does the response address the query? A correct, explicit refusal \
for a query the context cannot support is a relevant answer and scores high.

[QUERY]
{query}

[RETRIEVED CONTEXT]
{context}

[RESPONSE]
{answer}

Return ONLY JSON:
{{"context_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0, \
"reasoning": "one sentence"}}
"""

_KEYS = ("context_relevance", "groundedness", "answer_relevance")


class JudgeError(RuntimeError):
    """The judge could not produce a usable score. Always fatal to the run."""


class GeminiJudge:
    """Single pinned Gemini model, invoked ``trials`` times per case."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        trials: int = DEFAULT_TRIALS,
        timeout: float = 90.0,
    ) -> None:
        if trials < 1:
            raise ValueError("trials must be at least 1")
        self.model = model
        self.trials = trials
        self._api_key = api_key
        self._timeout = timeout

    def _call(self, prompt: str) -> dict[str, float | str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self._api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            # Temperature 0 does not make the judge deterministic -- Phase 0 measured
            # the residual spread -- which is precisely why trials are averaged.
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", "replace")[:200]
            raise JudgeError(f"{self.model}: HTTP {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JudgeError(f"{self.model}: unreachable: {exc}") from exc

        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise JudgeError(f"{self.model}: no candidate content") from exc

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeError(f"{self.model}: non-JSON output: {text[:160]}") from exc

        missing = [k for k in _KEYS if k not in parsed]
        if missing:
            # Absent scores are an error, not a default. A default is a number that
            # cannot fail, which is the v1 defect this harness exists to avoid.
            raise JudgeError(f"{self.model}: omitted {', '.join(missing)}")

        scored: dict[str, float | str] = {}
        for key in _KEYS:
            try:
                value = float(parsed[key])
            except (TypeError, ValueError) as exc:
                raise JudgeError(f"{self.model}: non-numeric {key}: {parsed[key]!r}") from exc
            if not 0.0 <= value <= 1.0:
                raise JudgeError(f"{self.model}: {key} outside [0,1]: {value}")
            scored[key] = value
        scored["reasoning"] = str(parsed.get("reasoning", ""))[:400]
        return scored

    def score(self, query: str, context: str, answer: str) -> JudgeScores:
        prompt = PROMPT.format(
            query=query, context=context or "(no context retrieved)", answer=answer
        )
        runs = [self._call(prompt) for _ in range(self.trials)]

        collected = {key: [float(r[key]) for r in runs] for key in _KEYS}
        means_stdevs = {key: mean_and_stdev(values) for key, values in collected.items()}
        return JudgeScores(
            context_relevance_mean=round(means_stdevs["context_relevance"][0], 4),
            groundedness_mean=round(means_stdevs["groundedness"][0], 4),
            answer_relevance_mean=round(means_stdevs["answer_relevance"][0], 4),
            context_relevance_stdev=round(means_stdevs["context_relevance"][1], 4),
            groundedness_stdev=round(means_stdevs["groundedness"][1], 4),
            answer_relevance_stdev=round(means_stdevs["answer_relevance"][1], 4),
            trials=self.trials,
            judge_model=self.model,
            reasoning=str(runs[-1].get("reasoning", ""))[:400],
        )
