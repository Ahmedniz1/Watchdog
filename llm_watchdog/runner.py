"""The runner — the one place LLM Watchdog actually calls a model.

Everything else in the package is offline and deterministic. This module is the
seam where a :class:`~llm_watchdog.core.TestCase` stops being pure data and
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

    pip install llm-watchdog[llm]

Testability: the actual network call goes through a single ``completion_fn``
parameter. Production code leaves it ``None`` and the LiteLLM wrapper is used;
tests pass a fake callable and exercise the whole runner with no API key, no
cost, and no flakiness.
"""

from __future__ import annotations

from typing import Callable, Optional

from llm_watchdog.core import CaseResult, Suite, SuiteResult, TestCase

#: Signature of the pluggable completion function. It receives the rendered
#: prompt plus the resolved model string and returns the model's text output.
CompletionFn = Callable[..., str]

#: Regression testing wants reproducibility, so we default to greedy decoding.
#: A non-deterministic temperature would make a passing suite flake to failing
#: for reasons that have nothing to do with a prompt change. Callers can still
#: override per run when they specifically want to sample.
DEFAULT_TEMPERATURE = 0.0


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
    when an actual run happens, not on ``import llm_watchdog``. Raises a clear,
    actionable :class:`ImportError` if the extra isn't installed.
    """
    try:
        import litellm  # noqa: WPS433 (deliberately lazy)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Running cases against a model requires the 'llm' extra. "
            "Install it with:  pip install llm-watchdog[llm]"
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
    **kwargs,
) -> CaseResult:
    """Call the model for one ``case`` and score the output. Makes one LLM call.

    Model resolution order: explicit ``model`` argument > ``case.model``. If
    neither is set, a :class:`RunnerError` is raised rather than guessing — the
    spec keeps the model as explicit config so a retired model string is a
    one-line fix, never a hidden default.

    Any failure from the model call is wrapped in :class:`RunnerError` with the
    case name for context. A missing ``litellm`` dependency is left as the
    original :class:`ImportError` so its install hint isn't buried.
    """
    resolved = model or case.model
    if not resolved:
        raise RunnerError(
            f"no model set for case {case.name!r}: set TestCase.model, "
            "Suite.model, or pass model= to run()"
        )

    fn = completion_fn or _litellm_complete
    prompt = case.render_prompt()
    try:
        output = fn(prompt, model=resolved, temperature=temperature, max_tokens=max_tokens, **kwargs)
    except ImportError:
        raise
    except Exception as e:  # noqa: BLE001 - provider errors are deliberately broad
        raise RunnerError(f"model call failed for case {case.name!r}: {e}") from e

    return case.evaluate_output(output, model=resolved)


def run_suite(
    suite: Suite,
    model: Optional[str] = None,
    completion_fn: Optional[CompletionFn] = None,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> SuiteResult:
    """Run every case in ``suite`` and assemble a :class:`SuiteResult`.

    Each case's model is resolved as: explicit ``model`` argument > case override
    > suite default (via :meth:`Suite.effective_model`). Cases run sequentially;
    one call per case.
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
                **kwargs,
            )
        )
    return SuiteResult(suite_name=suite.name, case_results=case_results, model=suite.model)
