"""Tests for the core data model (Feature 1).

Real verifiers arrive in the next feature; here we use a tiny fake condition
to exercise the schema, scoring, and templating in isolation. Everything runs
without any LLM API call.
"""

from llm_tripwire import TestCase, Suite, CaseResult, SuiteResult
from llm_tripwire.core import Condition


class FakeContains(Condition):
    """Minimal stand-in for the real `contains` verifier (built next feature)."""

    type = "fake_contains"

    def __init__(self, value: str):
        self.value = value
        self.description = f"contains '{value}'"

    def evaluate(self, output: str):
        passed = self.value.lower() in output.lower()
        detail = "" if passed else f"'{self.value}' not found in output"
        return self._result(passed, detail=detail)


# --- ConditionResult / _result hook -----------------------------------------


def test_result_helper_defaults_score_from_pass():
    c = FakeContains("refund")
    r = c.evaluate("we will process your refund")
    assert r.passed is True
    assert r.score == 1.0
    assert r.condition_type == "fake_contains"

    r2 = c.evaluate("no relevant content here")
    assert r2.passed is False
    assert r2.score == 0.0
    assert "not found" in r2.detail


# --- CaseResult scoring ------------------------------------------------------


def test_case_all_pass():
    case = TestCase(
        name="refund - basic",
        prompt="Answer: {input}",
        input="how do I get a refund?",
        conditions=[FakeContains("refund"), FakeContains("process")],
    )
    res = case.evaluate_output("we will process your refund in 5 days")
    assert isinstance(res, CaseResult)
    assert res.passed is True
    assert res.passed_count == 2
    assert res.total == 2
    assert res.score == 1.0
    assert res.failure_summary() == ""


def test_case_partial_failure():
    case = TestCase(
        name="refund - partial",
        prompt="Answer: {input}",
        conditions=[FakeContains("refund"), FakeContains("apology")],
    )
    res = case.evaluate_output("we will process your refund")
    assert res.passed is False
    assert res.passed_count == 1
    assert res.total == 2
    assert res.score == 0.5
    fs = res.failure_summary()
    assert "apology" in fs
    assert "refund" not in fs.split("\n", 1)[1]  # passing condition not listed


def test_empty_conditions_passes_vacuously():
    case = TestCase(name="empty", prompt="hi", conditions=[])
    res = case.evaluate_output("anything")
    assert res.passed is True
    assert res.score == 1.0
    assert res.total == 0


# --- Prompt templating (the brace-safety fix) --------------------------------


def test_render_prompt_substitutes_input():
    case = TestCase(name="t", prompt="Answer: {input}", input="hello")
    assert case.render_prompt() == "Answer: hello"


def test_render_prompt_preserves_literal_braces():
    # A prompt containing a JSON example must survive substitution intact.
    case = TestCase(
        name="t",
        prompt='Reply as JSON like {"status": "ok"} for: {input}',
        input="ping",
    )
    rendered = case.render_prompt()
    assert rendered == 'Reply as JSON like {"status": "ok"} for: ping'


# --- Suite aggregation -------------------------------------------------------


def test_suite_score_is_conditions_passed_over_total():
    c1 = TestCase(name="a", prompt="{input}", conditions=[FakeContains("x"), FakeContains("y")])
    c2 = TestCase(name="b", prompt="{input}", conditions=[FakeContains("z")])
    suite = Suite(name="s", cases=[c1, c2], model="some-model")

    res = suite.evaluate_outputs({"a": "just x", "b": "z here"})
    assert isinstance(res, SuiteResult)
    # case a: x passes, y fails (1/2). case b: z passes (1/1). total 2/3.
    assert res.passed_conditions == 2
    assert res.total_conditions == 3
    assert round(res.score, 3) == round(2 / 3, 3)
    assert res.passed is False


def test_suite_effective_model_prefers_case_override():
    case = TestCase(name="a", prompt="{input}", model="case-model")
    suite = Suite(name="s", cases=[case], model="suite-model")
    assert suite.effective_model(case) == "case-model"

    case2 = TestCase(name="b", prompt="{input}")
    suite2 = Suite(name="s", cases=[case2], model="suite-model")
    assert suite2.effective_model(case2) == "suite-model"


# --- Serialization seam (used later by baseline storage) ---------------------


def test_to_dict_omits_raw_output():
    case = TestCase(name="a", prompt="{input}", conditions=[FakeContains("x")])
    res = case.evaluate_output("x is here")
    d = res.to_dict()
    assert "output" not in d  # Design Decision 1: never persist captured text
    assert d["score"] == 1.0
    assert d["conditions"][0]["condition_type"] == "fake_contains"