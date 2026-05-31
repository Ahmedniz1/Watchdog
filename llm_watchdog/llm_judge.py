"""V2 — LLM-judge verifiers (the opt-in, paid evaluation layer).

The V1 verifiers in ``verifiers.py`` and ``semantic.py`` cover most regression
checks for free and offline. Some checks, though, genuinely need a model to read
the output: "is this empathetic?", "does this contradict the policy doc?". Those
are what this module provides.

Three verifiers, all registered like any other condition so they drop straight
into the existing schema and (later) the YAML loader:

- ``llm_judge``    : pass/fail against one plain-English criterion.
- ``llm_factual``  : factual consistency of the output against a reference doc.
- ``llm_rubric``   : multiple criteria scored individually, then aggregated.

Design rules baked in (spec section 5):

* **Judge model is always explicit.** ``judge_model`` is required with no
  default. Using the model under test as its own judge creates self-serving
  bias, so the developer is forced to choose the judge consciously. Document the
  recommendation to use a small, cheap model (``gpt-4o-mini``, ``claude-haiku``).

* **Calls are lazy and injectable.** Like the runner, the network call goes
  through a ``completion_fn`` parameter (default: the lazy LiteLLM wrapper). Every
  test in this module runs offline by injecting a fake judge.

* **A judge that misbehaves fails the condition, it does not crash the run.** If
  the judge returns unparseable text, the condition fails with the raw text in
  ``detail`` rather than raising — a regression suite must stay robust.
"""

from __future__ import annotations

import json
from typing import List, Optional

from llm_watchdog.core import Condition, ConditionResult
from llm_watchdog.runner import CompletionFn, _litellm_complete
from llm_watchdog.verifiers import _parse_json, register

#: Judges should be reproducible across runs, same reasoning as the runner.
DEFAULT_JUDGE_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _LLMCondition(Condition):
    """Common machinery for judge-backed conditions: call a model, parse a verdict.

    Subclasses build the judge prompt and turn the parsed verdict into a
    :class:`ConditionResult`. Verdicts are requested as strict JSON so they can
    be parsed deterministically; a non-JSON or malformed reply fails the
    condition (never raises) so one flaky judge call can't abort a whole suite.
    """

    def __init__(
        self,
        judge_model: str,
        completion_fn: Optional[CompletionFn] = None,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    ):
        if not judge_model or not isinstance(judge_model, str):
            raise ValueError(
                "judge_model is required (e.g. 'gpt-4o-mini'). Pick a model "
                "different from the one under test to avoid self-serving bias."
            )
        self.judge_model = judge_model
        self._completion_fn = completion_fn
        self.temperature = temperature

    def _ask_judge(self, prompt: str) -> str:
        """Send ``prompt`` to the judge model and return its raw text reply."""
        fn = self._completion_fn or _litellm_complete
        return fn(prompt, model=self.judge_model, temperature=self.temperature)

    @staticmethod
    def _parse_verdict(raw: str) -> dict:
        """Parse the judge's JSON reply (a surrounding code fence is tolerated).

        Raises ``ValueError`` on anything that isn't a JSON object so callers
        can convert that into a clean condition failure.
        """
        data = _parse_json(raw)  # strips ```fences``` then json.loads
        if not isinstance(data, dict):
            raise ValueError("judge reply was not a JSON object")
        return data

    @staticmethod
    def _coerce_score(verdict: dict) -> Optional[float]:
        """Pull a 0..1 ``score`` out of a verdict, clamping; ``None`` if absent."""
        if "score" not in verdict or verdict["score"] is None:
            return None
        try:
            return max(0.0, min(1.0, float(verdict["score"])))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_pass(verdict: dict) -> Optional[bool]:
        """Pull a boolean ``pass`` (or ``passed``) out of a verdict; ``None`` if absent."""
        for key in ("pass", "passed"):
            if key in verdict:
                return bool(verdict[key])
        return None


# ---------------------------------------------------------------------------
# llm_judge
# ---------------------------------------------------------------------------


@register("llm_judge")
class LLMJudge(_LLMCondition):
    """Pass if the judge model decides the output meets ``criteria``.

    The judge returns a 0..1 confidence ``score``; the condition passes when that
    score is at least ``threshold``. The judge's one-line reason is carried into
    ``detail`` (and surfaces in the HTML report later, spec 5.4 step 7).
    """

    _INSTRUCTIONS = (
        "You are an impartial evaluator. Decide whether the OUTPUT satisfies the "
        "CRITERION. Respond with ONLY a JSON object of the form "
        '{{"pass": true/false, "score": 0.0-1.0, "reason": "<one sentence>"}}. '
        "score is your confidence (1.0 = fully satisfies, 0.0 = clearly fails).\n\n"
        "CRITERION:\n{criteria}\n\nOUTPUT:\n{output}"
    )

    def __init__(
        self,
        criteria: str,
        judge_model: str,
        threshold: float = 0.8,
        completion_fn: Optional[CompletionFn] = None,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    ):
        super().__init__(judge_model, completion_fn, temperature)
        if not criteria:
            raise ValueError("llm_judge needs a non-empty 'criteria' string")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.criteria = criteria
        self.threshold = threshold
        self.description = f"llm_judge: {criteria!r} (>= {threshold:.2f})"

    def evaluate(self, output: str) -> ConditionResult:
        prompt = self._INSTRUCTIONS.format(criteria=self.criteria, output=output)
        try:
            raw = self._ask_judge(prompt)
            verdict = self._parse_verdict(raw)
        except (json.JSONDecodeError, ValueError) as e:
            return self._result(False, score=0.0, detail=f"could not parse judge reply: {e}")

        score = self._coerce_score(verdict)
        passed_flag = self._coerce_pass(verdict)
        if score is None:
            # No numeric score: fall back to the boolean, treating it as 1.0/0.0.
            if passed_flag is None:
                return self._result(False, score=0.0, detail="judge reply had no score or pass field")
            score = 1.0 if passed_flag else 0.0

        passed = score >= self.threshold
        reason = str(verdict.get("reason", "")).strip()
        detail = f"score {score:.2f} (threshold {self.threshold:.2f})"
        if reason:
            detail += f" - {reason}"
        return self._result(passed, score=score, detail=detail)


# ---------------------------------------------------------------------------
# llm_factual
# ---------------------------------------------------------------------------


@register("llm_factual")
class LLMFactual(_LLMCondition):
    """Pass if the output is factually consistent with ``reference_doc``.

    Built for RAG pipelines, where the main risk is hallucination: claims in the
    output that the source document does not support. The judge scores how well
    the output is grounded in the reference; ``threshold`` sets the bar.
    """

    _INSTRUCTIONS = (
        "You are a fact-checker. Decide whether every claim in the OUTPUT is "
        "supported by the REFERENCE document. Unsupported or contradicted claims "
        "should lower the score. Respond with ONLY a JSON object of the form "
        '{{"pass": true/false, "score": 0.0-1.0, "reason": "<one sentence>"}}. '
        "score is the fraction of the output that is grounded in the reference "
        "(1.0 = fully supported, 0.0 = contradicted or fabricated).\n\n"
        "REFERENCE:\n{reference_doc}\n\nOUTPUT:\n{output}"
    )

    def __init__(
        self,
        reference_doc: str,
        judge_model: str,
        threshold: float = 0.8,
        completion_fn: Optional[CompletionFn] = None,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    ):
        super().__init__(judge_model, completion_fn, temperature)
        if not reference_doc:
            raise ValueError("llm_factual needs a non-empty 'reference_doc'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.reference_doc = reference_doc
        self.threshold = threshold
        self.description = f"llm_factual: consistent with reference (>= {threshold:.2f})"

    def evaluate(self, output: str) -> ConditionResult:
        prompt = self._INSTRUCTIONS.format(reference_doc=self.reference_doc, output=output)
        try:
            raw = self._ask_judge(prompt)
            verdict = self._parse_verdict(raw)
        except (json.JSONDecodeError, ValueError) as e:
            return self._result(False, score=0.0, detail=f"could not parse judge reply: {e}")

        score = self._coerce_score(verdict)
        passed_flag = self._coerce_pass(verdict)
        if score is None:
            if passed_flag is None:
                return self._result(False, score=0.0, detail="judge reply had no score or pass field")
            score = 1.0 if passed_flag else 0.0

        passed = score >= self.threshold
        reason = str(verdict.get("reason", "")).strip()
        detail = f"grounding {score:.2f} (threshold {self.threshold:.2f})"
        if reason:
            detail += f" - {reason}"
        return self._result(passed, score=score, detail=detail)


# ---------------------------------------------------------------------------
# llm_rubric
# ---------------------------------------------------------------------------


@register("llm_rubric")
class LLMRubric(_LLMCondition):
    """Score the output against several criteria at once and aggregate.

    The judge evaluates each rubric point as pass/fail in a single call (cheaper
    than one call per point). The condition ``score`` is the fraction of points
    that passed; it passes overall when that fraction is at least
    ``pass_threshold``. Per-point reasons are summarized in ``detail``.
    """

    _INSTRUCTIONS = (
        "You are an impartial evaluator. For EACH criterion below, decide whether "
        "the OUTPUT satisfies it. Respond with ONLY a JSON object of the form "
        '{{"points": [{{"criterion": "<text>", "pass": true/false, '
        '"reason": "<one sentence>"}}, ...]}} with exactly one entry per '
        "criterion, in order.\n\nCRITERIA:\n{rubric}\n\nOUTPUT:\n{output}"
    )

    def __init__(
        self,
        rubric: List[str],
        judge_model: str,
        pass_threshold: float = 0.7,
        completion_fn: Optional[CompletionFn] = None,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    ):
        super().__init__(judge_model, completion_fn, temperature)
        if isinstance(rubric, str):
            rubric = [rubric]
        if not rubric:
            raise ValueError("llm_rubric needs at least one criterion")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0 and 1")
        self.rubric = list(rubric)
        self.pass_threshold = pass_threshold
        self.description = f"llm_rubric: {len(self.rubric)} criteria (>= {pass_threshold:.0%})"

    def _format_rubric(self) -> str:
        return "\n".join(f"{i}. {point}" for i, point in enumerate(self.rubric, 1))

    def evaluate(self, output: str) -> ConditionResult:
        prompt = self._INSTRUCTIONS.format(rubric=self._format_rubric(), output=output)
        try:
            raw = self._ask_judge(prompt)
            verdict = self._parse_verdict(raw)
        except (json.JSONDecodeError, ValueError) as e:
            return self._result(False, score=0.0, detail=f"could not parse judge reply: {e}")

        points = verdict.get("points")
        if not isinstance(points, list) or not points:
            return self._result(False, score=0.0, detail="judge reply had no 'points' list")

        passed_points = sum(1 for p in points if isinstance(p, dict) and bool(p.get("pass")))
        # Score against the rubric we asked about, so a judge that drops or
        # invents points can't quietly inflate the result.
        total = len(self.rubric)
        score = passed_points / total
        passed = score >= self.pass_threshold

        failed = [
            str(p.get("criterion", "?"))
            for p in points
            if isinstance(p, dict) and not bool(p.get("pass"))
        ]
        detail = f"{passed_points}/{total} rubric points passed"
        if failed:
            detail += "; failed: " + "; ".join(failed)
        return self._result(passed, score=score, detail=detail)


# ---------------------------------------------------------------------------
# Cost transparency (spec 5.3)
# ---------------------------------------------------------------------------

# Rough per-call USD estimates for common judge models. These are deliberately
# approximate order-of-magnitude figures for a short judge prompt + reply; the
# point is to warn the developer of scale before a run, not to bill them. Tune
# or extend as needed. Unknown models fall back to ``_DEFAULT_CALL_COST``.
_PER_CALL_COST = {
    "gpt-4o-mini": 0.0003,
    "gpt-4o": 0.005,
    "gpt-4.1-mini": 0.0003,
    "claude-3-5-haiku": 0.0005,
    "claude-3-5-sonnet": 0.006,
}
_DEFAULT_CALL_COST = 0.002


def is_judge_condition(condition: Condition) -> bool:
    """True if ``condition`` is one of the paid LLM-judge verifiers."""
    return isinstance(condition, _LLMCondition)


def estimate_run_cost(suite) -> dict:
    """Estimate the cost of the judge calls a :class:`~llm_watchdog.core.Suite` run will make.

    Returns ``{"calls": int, "usd": float, "by_model": {model: calls}}``. Each
    judge condition is one model call (``llm_rubric`` batches all its points into
    a single call). The figure is intentionally rough — see ``_PER_CALL_COST`` —
    and is meant to be shown to the developer before any paid call executes::

        LLM judge conditions detected.
        Estimated cost: ~$0.004 for this run (12 judge calls x gpt-4o-mini)
        Continue? [y/N]

    The interactive confirm/``--yes`` flow itself belongs to the CLI feature;
    this function is the pure estimator that backs it.
    """
    by_model: dict = {}
    usd = 0.0
    for case in suite.cases:
        model = suite.effective_model(case)
        for condition in case.conditions:
            if not is_judge_condition(condition):
                continue
            judge_model = getattr(condition, "judge_model", None) or model or "?"
            by_model[judge_model] = by_model.get(judge_model, 0) + 1
            usd += _PER_CALL_COST.get(judge_model, _DEFAULT_CALL_COST)
    return {"calls": sum(by_model.values()), "usd": round(usd, 4), "by_model": by_model}
