"""Tests for the V2 LLM-judge verifiers (Feature: V2 judge layer).

These verifiers call a judge model, so — exactly like the runner tests — the
model call is injected as a fake ``completion_fn`` that returns canned JSON.
Nothing here touches litellm or the network.
"""

import json

import pytest

from llm_tripwire import (
    LLMJudge,
    LLMFactual,
    LLMRubric,
    Suite,
    TestCase,
    available_types,
    build_condition,
    estimate_run_cost,
)


def fixed(reply: str):
    """A fake judge that always returns ``reply`` (ignores the prompt)."""

    def _fn(prompt, *, model, temperature=0.0, **kwargs):
        return reply

    return _fn


def json_reply(**fields):
    return fixed(json.dumps(fields))


# --- registration & config validation ---------------------------------------


def test_judge_types_registered():
    for t in ("llm_judge", "llm_factual", "llm_rubric"):
        assert t in available_types()


def test_judge_model_is_required():
    with pytest.raises(ValueError):
        LLMJudge(criteria="be nice", judge_model="")
    with pytest.raises(ValueError):
        LLMFactual(reference_doc="doc", judge_model=None)
    with pytest.raises(ValueError):
        LLMRubric(rubric=["x"], judge_model="")


# --- llm_judge ---------------------------------------------------------------


def test_llm_judge_passes_when_score_meets_threshold():
    cond = LLMJudge(
        criteria="The response is empathetic",
        judge_model="gpt-4o-mini",
        threshold=0.8,
        completion_fn=json_reply(**{"pass": True, "score": 0.9, "reason": "warm tone"}),
    )
    r = cond.evaluate("I'm so sorry to hear that, let me help.")
    assert r.passed is True
    assert r.score == 0.9
    assert "warm tone" in r.detail


def test_llm_judge_fails_below_threshold():
    cond = LLMJudge(
        criteria="empathetic",
        judge_model="gpt-4o-mini",
        threshold=0.8,
        completion_fn=json_reply(score=0.5, reason="curt"),
    )
    r = cond.evaluate("Deal with it.")
    assert r.passed is False
    assert r.score == 0.5


def test_llm_judge_handles_code_fenced_json():
    reply = '```json\n{"pass": true, "score": 1.0, "reason": "ok"}\n```'
    cond = LLMJudge(criteria="x", judge_model="m", completion_fn=fixed(reply))
    r = cond.evaluate("anything")
    assert r.passed is True
    assert r.score == 1.0


def test_llm_judge_falls_back_to_pass_flag_when_no_score():
    cond = LLMJudge(criteria="x", judge_model="m", completion_fn=json_reply(**{"pass": True}))
    r = cond.evaluate("out")
    assert r.passed is True
    assert r.score == 1.0


def test_llm_judge_malformed_reply_fails_gracefully():
    cond = LLMJudge(criteria="x", judge_model="m", completion_fn=fixed("not json at all"))
    r = cond.evaluate("out")
    assert r.passed is False
    assert r.score == 0.0
    assert "could not parse" in r.detail


# --- llm_factual -------------------------------------------------------------


def test_llm_factual_grounded_output_passes():
    cond = LLMFactual(
        reference_doc="Refunds are accepted within 30 days.",
        judge_model="gpt-4o-mini",
        completion_fn=json_reply(score=0.95, reason="supported"),
    )
    r = cond.evaluate("You can return items within 30 days for a refund.")
    assert r.passed is True
    assert "grounding" in r.detail


def test_llm_factual_hallucination_fails():
    cond = LLMFactual(
        reference_doc="Refunds within 30 days.",
        judge_model="gpt-4o-mini",
        completion_fn=json_reply(score=0.2, reason="claims 90 days, unsupported"),
    )
    r = cond.evaluate("You have 90 days to return anything, no questions asked.")
    assert r.passed is False


# --- llm_rubric --------------------------------------------------------------


def test_llm_rubric_aggregates_fraction():
    reply = json_reply(points=[
        {"criterion": "acknowledges frustration", "pass": True},
        {"criterion": "gives next step", "pass": True},
        {"criterion": "no jargon", "pass": False, "reason": "used 'synergy'"},
    ])
    cond = LLMRubric(
        rubric=["acknowledges frustration", "gives next step", "no jargon"],
        judge_model="gpt-4o-mini",
        pass_threshold=0.7,
        completion_fn=reply,
    )
    r = cond.evaluate("out")
    # 2/3 = 0.667 < 0.7 -> fails, and the failing criterion is named.
    assert r.score == pytest.approx(2 / 3)
    assert r.passed is False
    assert "no jargon" in r.detail


def test_llm_rubric_passes_at_threshold():
    reply = json_reply(points=[{"pass": True}, {"pass": True}, {"pass": True}])
    cond = LLMRubric(rubric=["a", "b", "c"], judge_model="m", pass_threshold=0.7, completion_fn=reply)
    r = cond.evaluate("out")
    assert r.score == 1.0
    assert r.passed is True


def test_llm_rubric_missing_points_fails_gracefully():
    cond = LLMRubric(rubric=["a"], judge_model="m", completion_fn=fixed('{"nope": 1}'))
    r = cond.evaluate("out")
    assert r.passed is False
    assert "points" in r.detail


# --- build_condition integration (the YAML seam) -----------------------------


def test_build_condition_constructs_judge_from_spec():
    cond = build_condition({
        "type": "llm_judge",
        "criteria": "is polite",
        "judge_model": "gpt-4o-mini",
        "threshold": 0.6,
    })
    assert isinstance(cond, LLMJudge)
    assert cond.threshold == 0.6


# --- cost estimation ---------------------------------------------------------


def test_estimate_run_cost_counts_only_judge_calls():
    from tests.test_runner import FakeContains  # offline, non-judge condition

    judge = LLMJudge(criteria="x", judge_model="gpt-4o-mini", completion_fn=fixed("{}"))
    case_a = TestCase(name="a", prompt="{input}", conditions=[judge, FakeContains("x")])
    case_b = TestCase(name="b", prompt="{input}", conditions=[
        LLMRubric(rubric=["p"], judge_model="gpt-4o-mini", completion_fn=fixed("{}")),
    ])
    suite = Suite(name="s", cases=[case_a, case_b], model="gpt-4o")

    est = estimate_run_cost(suite)
    # 2 judge calls total (the FakeContains is free/offline and not counted).
    assert est["calls"] == 2
    assert est["by_model"]["gpt-4o-mini"] == 2
    assert est["usd"] > 0
