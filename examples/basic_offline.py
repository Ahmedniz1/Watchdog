"""Offline example — no API key, no extra dependencies.

Run it with:  python examples/basic_offline.py

This scores output you *already have* against a set of conditions. It's the
fastest way to see how cases, conditions, and scoring fit together, and it's
exactly what runs in the test suite (no model is called).
"""

from llm_watchdog import TestCase
from llm_watchdog.conditions import contains, not_contains, valid_json, word_count


def main() -> None:
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

    good = "Sure! We can process your refund within 5-7 business days."
    bad = "Sorry, I cannot help with that."

    for label, output in [("good output", good), ("bad output", bad)]:
        print(f"\n=== {label} ===")
        result = case.evaluate_output(output)
        print(result.summary())
        if not result.passed:
            print(result.failure_summary())

    # Conditions are also data: a `valid_json` check on a JSON reply.
    json_case = TestCase(
        name="structured reply",
        prompt="Reply as JSON: {input}",
        conditions=[valid_json()],
    )
    print("\n=== json check ===")
    print(json_case.evaluate_output('```json\n{"status": "ok"}\n```').summary())


if __name__ == "__main__":
    main()
