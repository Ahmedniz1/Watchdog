"""The runner — the one place LLM Tripwire actually calls a model.

Everything else in the package is offline and deterministic. This module is the
seam where a :class:`~llm_tripwire.core.TestCase` stops being pure data and
becomes a real network call: render the prompt, send it to the model under test
via LiteLLM, hand the raw text back to the offline scoring layer
(``TestCase.evaluate_output``). That clean split is deliberate — the scoring is
fully unit-tested without any API, and only this thin layer needs mocking.

Why LiteLLM (spec Decision 3): abstracting OpenAI / Anthropic / Gemini / local
models by hand is a maintenance black hole. LiteLLM gives one ``completion()``
call across every provider, keyed by a model string like ``"gpt-4o-mini"`` or
``"claude-3-5-haiku"``. It is an optional dependency, imported lazily (mirroring
how ``semantic.py`` defers ``sentence-transformers``) so that importing the core
library stays fast and dependency-free::

    pip install llm-tripwire[llm]

Testability: the actual network call goes through a single ``completion_fn``
parameter. Production code leaves it ``None`` and the LiteLLM wrapper is used;
tests pass a fake callable and exercise the whole runner with no API key, no
cost, and no flakiness.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

from llm_tripwire.core import CaseResult, ConditionResult, Suite, SuiteResult, TestCase

#: Signature of the pluggable completion function. It receives the rendered
#: prompt plus the resolved model string and returns the model's text output.
CompletionFn = Callable[..., str]

#: Regression testing wants reproducibility, so we default to greedy decoding.
#: A non-deterministic temperature would make a passing suite flake to failing
#: for reasons that have nothing to do with a prompt change. Callers can still
#: override per run when they specifically want to sample.
DEFAULT_TEMPERATURE = 0.0

#: With ``runs=N`` sampling, a condition counts as passed when it passed in at
#: least this fraction of the samples. The default is a simple majority, which
#: smooths out one-off non-deterministic blips (the whole point of sampling)
#: while still failing a condition that breaks more often than not. Set it to
#: ``1.0`` to require every sample to pass, or lower it to be more tolerant.
DEFAULT_MIN_PASS_RATE = 0.5


class RunnerError(RuntimeError):
    """Raised when a model run cannot be completed (missing model, API error)."""


def _litellm_complete(
    prompt: str,
    *,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> str:
    """Default completion function: a thin lazy wrapper over ``litellm.completion``.

    Imported lazily so the heavy/optional ``litellm`` dependency is only required
    when an actual run happens, not on ``import llm_tripwire``. Raises a clear,
    actionable :class:`ImportError` if the extra isn't installed.
    """
    try:
        import litellm  # noqa: WPS433 (deliberately lazy)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Running cases against a model requires the 'llm' extra. "
            "Install it with:  pip install llm-tripwire[llm]"
        ) from e

    messages = [{"role": "user", "content": prompt}]
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    # LiteLLM normalizes every provider to the OpenAI response shape.
    content = response.choices[0].message.content
    return content or ""


def run_case(
    case: TestCase,
    model: Optional[str] = None,
    completion_fn: Optional[CompletionFn] = None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    runs: int = 1,
    min_pass_rate: float = DEFAULT_MIN_PASS_RATE,
    **kwargs,
) -> CaseResult:
    """Call the model for one ``case`` and score the output. Makes ``runs`` LLM calls.

    Model resolution order: explicit ``model`` argument > ``case.model``. If
    neither is set, a :class:`RunnerError` is raised rather than guessing — the
    spec keeps the model as explicit config so a retired model string is a
    one-line fix, never a hidden default.

    **Sampling (``runs``).** With ``runs=1`` (the default) this behaves exactly
    as before: one call, one :class:`CaseResult`. With ``runs > 1`` the model is
    called ``runs`` times and the per-condition outcomes are aggregated — each
    condition's ``score`` becomes the mean across samples and its ``pass_rate``
    the fraction of samples it passed. A condition is then considered passed
    when ``pass_rate >= min_pass_rate``. Sampling only varies the output if the
    model is non-deterministic, so ``runs > 1`` with ``temperature == 0`` emits
    a warning.

    Any failure from the model call is wrapped in :class:`RunnerError` with the
    case name for context. A missing ``litellm`` dependency is left as the
    original :class:`ImportError` so its install hint isn't buried.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be between 0 and 1")

    resolved = model or case.model
    if not resolved:
        raise RunnerError(
            f"no model set for case {case.name!r}: set TestCase.model, "
            "Suite.model, or pass model= to run()"
        )

    if runs > 1 and temperature == 0:
        warnings.warn(
            f"run_case(runs={runs}) with temperature=0 will usually produce "
            "identical samples; set temperature > 0 to actually sample variation.",
            stacklevel=2,
        )

    fn = completion_fn or _litellm_complete
    prompt = case.render_prompt()

    per_run: List[CaseResult] = []
    for _ in range(runs):
        try:
            output = fn(prompt, model=resolved, temperature=temperature, max_tokens=max_tokens, **kwargs)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001 - provider errors are deliberately broad
            raise RunnerError(f"model call failed for case {case.name!r}: {e}") from e
        per_run.append(case.evaluate_output(output, model=resolved))

    if runs == 1:
        return per_run[0]
    return _aggregate_runs(per_run, min_pass_rate)


def _aggregate_runs(results: List[CaseResult], min_pass_rate: float) -> CaseResult:
    """Collapse ``N`` single-run :class:`CaseResult`s into one sampled result.

    Conditions are aligned by position (every run evaluates the same ordered
    condition list). For each condition we report the mean score and the pass
    rate across runs; ``passed`` is decided by ``min_pass_rate``. The first
    sample's output text is kept for the report — the others differ by design
    and the aggregate is what matters for pass/fail.
    """
    first = results[0]
    n = len(results)
    aggregated: List[ConditionResult] = []
    for i in range(len(first.condition_results)):
        column = [r.condition_results[i] for r in results]
        mean_score = sum(c.score for c in column) / n
        pass_count = sum(1 for c in column if c.passed)
        pass_rate = pass_count / n
        aggregated.append(
            ConditionResult(
                condition_type=column[0].condition_type,
                description=column[0].description,
                passed=pass_rate >= min_pass_rate,
                score=mean_score,
                detail=f"passed {pass_count}/{n} runs, mean score {mean_score:.2f}",
                runs=n,
                pass_rate=pass_rate,
            )
        )
    return CaseResult(
        case_name=first.case_name,
        output=first.output,
        condition_results=aggregated,
        model=first.model,
    )


def run_suite(
    suite: Suite,
    model: Optional[str] = None,
    completion_fn: Optional[CompletionFn] = None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    runs: int = 1,
    min_pass_rate: float = DEFAULT_MIN_PASS_RATE,
    **kwargs,
) -> SuiteResult:
    """Run every case in ``suite`` and assemble a :class:`SuiteResult`.

    Each case's model is resolved as: explicit ``model`` argument > case override
    > suite default (via :meth:`Suite.effective_model`). Cases run sequentially.
    ``runs`` / ``min_pass_rate`` are forwarded to :func:`run_case`, so each case
    makes ``runs`` calls and its conditions are aggregated by pass rate.
    """
    case_results = []
    for case in suite.cases:
        resolved = model or suite.effective_model(case)
        case_results.append(
            run_case(
                case,
                model=resolved,
                completion_fn=completion_fn,
                temperature=temperature,
                max_tokens=max_tokens,
                runs=runs,
                min_pass_rate=min_pass_rate,
                **kwargs,
            )
        )
    return SuiteResult(suite_name=suite.name, case_results=case_results, model=suite.model)
