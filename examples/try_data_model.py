"""
Try the LLM Watchdog verifier library (Features 1 + 2) by hand.

This uses the REAL built-in verifiers now (the earlier throwaway stand-ins are
gone). It walks through:

  1. Evaluating a single output against one case.
  2. The same case with a bad output (shows failures + failure_summary).
  3. A whole suite, including JSON and bullet checks.
  4. JSON serialization (the baseline-storage seam; raw output omitted).
  5. Building conditions from config dicts (the YAML-loader seam).
  6. A mini regression demo: good vs. degraded outputs on the same suite.

Run from anywhere:   python examples/try_data_model.py
No API key, no network, no LLM call.
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from llm_watchdog import TestCase, Suite, build_conditions, available_types  # noqa: E402


def show_case(res):
    print(res.summary())
    for r in res.condition_results:
        mark = "PASS" if r.passed else "FAIL"
        extra = f"  ->  {r.detail}" if r.detail else ""
        print(f"     [{mark}] {r.description}{extra}")
    print()


def hr(title=""):
    print("\n" + "=" * 68)
    if title:
        print(title); print("=" * 68)


# 1. ONE CASE, ONE OUTPUT -----------------------------------------------------
hr("1. ONE CASE, ONE OUTPUT")
refund_case = TestCase(
    name="refund query - basic",
    prompt="You are a support agent. Answer: {input}",
    input="How do I get a refund?",
    model="example-model",
    conditions=build_conditions([
        {"type": "contains", "value": "refund"},
        {"type": "not_contains", "value": "cannot help"},
        {"type": "word_count", "max_words": 150},
    ]),
)
result = refund_case.evaluate_output(
    "Sure! We can process your refund within 5 to 7 business days."
)
show_case(result)
print(f"case.passed = {result.passed}   case.score = {result.score}")

# 2. SAME CASE, BAD OUTPUT ----------------------------------------------------
hr("2. SAME CASE, A BAD OUTPUT")
bad = refund_case.evaluate_output("Sorry, I cannot help with that request.")
show_case(bad)
print("failure_summary():"); print(bad.failure_summary())

# 3. A SUITE ------------------------------------------------------------------
hr("3. A SUITE OF CASES")
json_case = TestCase(
    name="structured output",
    prompt='Reply ONLY as JSON like {"status": "ok"} for: {input}',
    input="ping",
    conditions=build_conditions([
        {"type": "valid_json"},
        {"type": "json_has_keys", "keys": ["status"]},
    ]),
)
list_case = TestCase(
    name="three tips",
    prompt="Give exactly three bullet-point tips for: {input}",
    input="saving money",
    conditions=build_conditions([{"type": "bullet_count", "count": 3}]),
)
suite = Suite(name="support_bot", model="example-model",
              cases=[refund_case, json_case, list_case])

good_outputs = {
    "refund query - basic": "We can process your refund within 5 to 7 business days.",
    "structured output": '```json\n{"status": "ok"}\n```',   # fenced JSON still passes
    "three tips": "- Track spending\n- Cancel unused subscriptions\n- Cook at home",
}
suite_result = suite.evaluate_outputs(good_outputs)
for c in suite_result.case_results:
    show_case(c)
print(suite_result.summary())

# 4. SERIALIZATION ------------------------------------------------------------
hr("4. SERIALIZED RESULT (raw output is NOT included)")
print(json.dumps(suite_result.case_results[0].to_dict(), indent=2))

# 5. CONFIG-DRIVEN CONSTRUCTION ----------------------------------------------
hr("5. BUILDING CONDITIONS FROM CONFIG (the YAML-loader seam)")
print("Registered condition types:")
print("  " + ", ".join(available_types()))

# 6. REGRESSION PREVIEW -------------------------------------------------------
hr("6. REGRESSION PREVIEW: good vs. degraded outputs")
degraded_outputs = {
    "refund query - basic": "We cannot help you here. " + ("blah " * 200),
    "structured output": "Sure! Here is your answer: status ok",   # not JSON
    "three tips": "- only one tip",
}
before = suite.evaluate_outputs(good_outputs)
after = suite.evaluate_outputs(degraded_outputs)
print(f"Baseline run : {before.passed_conditions}/{before.total_conditions} ({round(before.score*100)}%)")
print(f"Current run  : {after.passed_conditions}/{after.total_conditions} ({round(after.score*100)}%)")
print(f"Delta        : {round((after.score-before.score)*100):+d}%")
if after.score < before.score:
    print("\n>>> Conditions regressed. This is what the tool will flag in CI.")
hr()