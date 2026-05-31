"""Tests for the LiteLLM runner (Feature 4).

The runner is the only part of the package that makes a network call, so the
whole point of these tests is to exercise it *without* one. The network call
goes through an injectable ``completion_fn``; here we pass fakes that record
their arguments and return canned text. No litellm, no API key, no cost.
"""

import importlib.util

import pytest

from llm_watchdog import RunnerError, TestCase, Suite, run_case, run_suite
from llm_watchdog.core import Condition

HAS_LITELLM = importlib.util.find_spec("litellm") is not None


class FakeContains(Condition):
    """Minimal offline condition reused from the core tests."""

    type = "fake_contains"

    def __init__(self, value: str):
        self.value = value
        self.description = f"contains '{value}'"

    def evaluate(self, output: str):
        return self._result(self.value.lower() in output.lower())


def make_fn(output: str, recorder: dict = None):
    """Build a fake completion_fn that returns ``output`` and records its call."""

    def _fn(prompt, *, model, temperature=0.0, max_tokens=None, **kwargs):
        if recorder is not None:
            recorder.update(prompt=prompt, model=model, temperature=temperature,
                            max_tokens=max_tokens, kwargs=kwargs)
        return output

    return _fn


# --- run_case ----------------------------------------------------------------


def test_run_case_renders_prompt_and_scores():
    rec = {}
    case = TestCase(
        name="refund",
        prompt="Answer: {input}",
        input="how do I get a refund?",
        model="gpt-4o-mini",
        conditions=[FakeContains("refund"), FakeContains("sorry")],
    )
    res = case.run(completion_fn=make_fn("we will process your refund soon", rec))

    # The rendered prompt (with {input} substituted) is what reaches the model.
    assert rec["prompt"] == "Answer: how do I get a refund?"
    assert rec["model"] == "gpt-4o-mini"
    assert rec["temperature"] == 0.0  # deterministic default
    # Scoring uses the offline condition layer: 1 of 2 conditions pass.
    assert res.passed_count == 1
    assert res.total == 2
    assert res.model == "gpt-4o-mini"


def test_explicit_model_overrides_case_model():
    rec = {}
    case = TestCase(name="c", prompt="{input}", model="case-model")
    run_case(case, model="override-model", completion_fn=make_fn("x", rec))
    assert rec["model"] == "override-model"


def test_missing_model_raises_runner_error():
    case = TestCase(name="c", prompt="hi")  # no model anywhere
    with pytest.raises(RunnerError) as ei:
        run_case(case, completion_fn=make_fn("x"))
    assert "no model" in str(ei.value)


def test_provider_error_wrapped_in_runner_error():
    def boom(prompt, *, model, **kwargs):
        raise RuntimeError("429 rate limited")

    case = TestCase(name="c", prompt="hi", model="gpt-4o-mini")
    with pytest.raises(RunnerError) as ei:
        run_case(case, completion_fn=boom)
    assert "c" in str(ei.value)
    assert "429" in str(ei.value)


def test_extra_kwargs_forwarded_to_completion_fn():
    rec = {}
    case = TestCase(name="c", prompt="hi", model="m")
    run_case(case, completion_fn=make_fn("x", rec), max_tokens=64, top_p=0.9)
    assert rec["max_tokens"] == 64
    assert rec["kwargs"]["top_p"] == 0.9


# --- run_suite ---------------------------------------------------------------


def test_run_suite_resolves_model_per_case():
    seen = []

    def fn(prompt, *, model, **kwargs):
        seen.append(model)
        return "ok"

    c1 = TestCase(name="a", prompt="{input}", conditions=[FakeContains("ok")])
    c2 = TestCase(name="b", prompt="{input}", model="case-model", conditions=[FakeContains("ok")])
    suite = Suite(name="s", cases=[c1, c2], model="suite-model")

    res = suite.run(completion_fn=fn)
    # c1 inherits the suite default; c2 keeps its own override.
    assert seen == ["suite-model", "case-model"]
    assert res.suite_name == "s"
    assert res.passed is True
    assert res.total_conditions == 2


# --- default (litellm) path --------------------------------------------------


@pytest.mark.skipif(HAS_LITELLM, reason="litellm installed; the missing-dep path can't be exercised")
def test_default_completion_requires_litellm():
    """With no completion_fn and litellm absent, a helpful ImportError surfaces."""
    from llm_watchdog.runner import _litellm_complete

    with pytest.raises(ImportError) as ei:
        _litellm_complete("hi", model="gpt-4o-mini")
    assert "pip install" in str(ei.value)
