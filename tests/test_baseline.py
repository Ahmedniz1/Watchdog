"""Tests for baseline storage and regression diffing (Feature: baseline + diff).

All offline: we build SuiteResults from a fake condition and a known set of
outputs, save a baseline to a tmp dir, then diff later runs against it. No model
and no network.
"""

import json

import pytest

from llm_watchdog import (
    Suite,
    TestCase,
    BaselineError,
    compare,
    load_baseline,
    save_baseline,
)
from llm_watchdog.baseline import baseline_path
from llm_watchdog.core import Condition


class FakeContains(Condition):
    type = "fake_contains"

    def __init__(self, value: str):
        self.value = value
        self.description = f"contains '{value}'"

    def evaluate(self, output: str):
        return self._result(self.value.lower() in output.lower())


def make_suite():
    return Suite(
        name="support bot",  # space on purpose: exercises filename sanitizing
        cases=[
            TestCase(name="refund - basic", prompt="{input}",
                     conditions=[FakeContains("refund"), FakeContains("days")]),
            TestCase(name="refund - angry", prompt="{input}",
                     conditions=[FakeContains("refund"), FakeContains("sorry")]),
        ],
    )


# --- save / load round trip --------------------------------------------------


def test_save_writes_sanitized_filename(tmp_path):
    suite = make_suite()
    result = suite.evaluate_outputs({
        "refund - basic": "refund in 5 days",
        "refund - angry": "sorry, your refund is coming",
    })
    path = save_baseline(result, directory=str(tmp_path))
    assert path == baseline_path("support bot", str(tmp_path))
    assert path.name == "support_bot.json"  # space -> underscore
    assert path.exists()


def test_saved_baseline_omits_raw_output(tmp_path):
    suite = make_suite()
    result = suite.evaluate_outputs({"refund - basic": "refund in 5 days"})
    path = save_baseline(result, directory=str(tmp_path))
    raw = path.read_text(encoding="utf-8")
    assert "5 days" not in raw  # Design Decision 1: never persist captured text
    data = json.loads(raw)
    assert data["version"] == 1
    assert data["suite_name"] == "support bot"
    assert "saved_at" in data


def test_load_missing_returns_none(tmp_path):
    assert load_baseline("never saved", directory=str(tmp_path)) is None


def test_load_rejects_wrong_schema_version(tmp_path):
    p = baseline_path("s", str(tmp_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 999, "result": {}}), encoding="utf-8")
    with pytest.raises(BaselineError):
        load_baseline("s", directory=str(tmp_path))


# --- diff semantics ----------------------------------------------------------


def test_no_change_is_not_a_regression(tmp_path):
    suite = make_suite()
    outputs = {"refund - basic": "refund in 5 days", "refund - angry": "sorry refund"}
    save_baseline(suite.evaluate_outputs(outputs), directory=str(tmp_path))

    diff = compare(suite.evaluate_outputs(outputs), load_baseline("support bot", str(tmp_path)))
    assert diff.delta == 0.0
    assert diff.regressed_conditions == 0
    assert diff.within_threshold(0) is True
    assert "No regressions." in diff.summary()


def test_regression_detected_and_counted(tmp_path):
    suite = make_suite()
    # Baseline: everything passes (4/4 conditions).
    good = {"refund - basic": "refund in 5 days", "refund - angry": "sorry refund here"}
    save_baseline(suite.evaluate_outputs(good), directory=str(tmp_path))

    # Now the angry case loses both conditions (no 'refund', no 'sorry').
    worse = {"refund - basic": "refund in 5 days", "refund - angry": "deal with it"}
    diff = compare(suite.evaluate_outputs(worse), load_baseline("support bot", str(tmp_path)))

    assert diff.now_passed == 2 and diff.base_passed == 4
    assert diff.regressed_conditions == 2
    assert round(diff.delta * 100) == -50
    # Default threshold (0) fails on any drop; a 50-point allowance passes.
    assert diff.within_threshold(0) is False
    assert diff.within_threshold(60) is True
    summ = diff.summary()
    assert "REGRESSION" in summ
    assert "refund - angry" in summ


def test_new_case_is_not_a_regression(tmp_path):
    suite = make_suite()
    save_baseline(
        suite.evaluate_outputs({"refund - basic": "refund days", "refund - angry": "sorry refund"}),
        directory=str(tmp_path),
    )
    # Add a brand-new case the baseline never saw.
    suite.cases.append(
        TestCase(name="brand new", prompt="{input}", conditions=[FakeContains("hi")])
    )
    diff = compare(
        suite.evaluate_outputs({
            "refund - basic": "refund days",
            "refund - angry": "sorry refund",
            "brand new": "hi there",
        }),
        load_baseline("support bot", str(tmp_path)),
    )
    new = next(cd for cd in diff.case_diffs if cd.case_name == "brand new")
    assert new.is_new is True
    assert new.regressed is False
    assert diff.regressed_conditions == 0


def test_removed_case_listed(tmp_path):
    suite = make_suite()
    save_baseline(
        suite.evaluate_outputs({"refund - basic": "refund days", "refund - angry": "sorry refund"}),
        directory=str(tmp_path),
    )
    suite.cases.pop()  # drop "refund - angry"
    diff = compare(
        suite.evaluate_outputs({"refund - basic": "refund days"}),
        load_baseline("support bot", str(tmp_path)),
    )
    assert "refund - angry" in diff.removed_cases


# --- convenience methods on SuiteResult --------------------------------------


def test_suiteresult_convenience_methods(tmp_path):
    suite = make_suite()
    outputs = {"refund - basic": "refund days", "refund - angry": "sorry refund"}
    result = suite.evaluate_outputs(outputs)

    assert result.diff_against_baseline(directory=str(tmp_path)) is None  # none yet
    result.save_baseline(directory=str(tmp_path))
    diff = result.diff_against_baseline(directory=str(tmp_path))
    assert diff is not None
    assert diff.delta == 0.0
