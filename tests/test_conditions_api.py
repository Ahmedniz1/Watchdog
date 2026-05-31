"""Tests for the friendly `llm_watchdog.conditions` constructor API.

These guard the promise the README makes: ``from llm_watchdog.conditions import
contains`` works, the lowercase names build the right verifier objects, and the
quickstart example actually runs. All offline.
"""

from llm_watchdog import TestCase
from llm_watchdog.conditions import (
    bullet_count,
    contains,
    ends_with,
    json_has_keys,
    llm_factual,
    llm_judge,
    llm_rubric,
    not_contains,
    regex_match,
    semantic_similarity,
    starts_with,
    valid_json,
    word_count,
)
from llm_watchdog.core import Condition
from llm_watchdog.llm_judge import LLMJudge
from llm_watchdog.verifiers import Contains, WordCount


def test_constructors_build_condition_instances():
    cond = contains("refund")
    assert isinstance(cond, Condition)
    assert isinstance(cond, Contains)  # the alias really is the class
    assert cond.evaluate("we will refund you").passed is True


def test_same_vocabulary_as_classes():
    # The function alias takes the same kwargs as the underlying class — there's
    # only one set of names to learn (no parallel `max=` vs `max_words=`).
    cond = word_count(max_words=150)
    assert isinstance(cond, WordCount)
    assert cond.max_words == 150


def test_all_names_importable_and_callable():
    built = [
        contains("x"),
        not_contains("y"),
        starts_with("Dear"),
        ends_with("."),
        word_count(min_words=1),
        bullet_count(count=2),
        regex_match(r"\d+"),
        valid_json(),
        json_has_keys(keys=["a"]),
        semantic_similarity(reference="ref", threshold=0.5),
        llm_judge(criteria="is polite", judge_model="gpt-4o-mini"),
        llm_factual(reference_doc="doc", judge_model="gpt-4o-mini"),
        llm_rubric(rubric=["a", "b"], judge_model="gpt-4o-mini"),
    ]
    assert all(isinstance(c, Condition) for c in built)
    assert isinstance(llm_judge(criteria="x", judge_model="m"), LLMJudge)


def test_readme_quickstart_runs():
    case = TestCase(
        name="refund query",
        prompt="You are a helpful support agent. Answer: {input}",
        input="How do I get a refund?",
        conditions=[
            contains("refund"),
            not_contains("cannot help"),
            word_count(max_words=150),
        ],
    )
    result = case.evaluate_output(
        "Sure - we can process your refund within 5-7 business days."
    )
    assert result.passed is True
    assert result.summary() == "refund query: 3/3 conditions passed (PASS)"
