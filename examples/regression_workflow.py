"""Regression workflow — save a baseline, then catch a regression.

Run it with:  python examples/regression_workflow.py

Offline (no model, no key): it scores canned outputs so you can see the whole
save-baseline / diff loop. In a real project you'd call `suite.run()` instead of
`evaluate_outputs(...)`, save the baseline once when the output is good, commit
`.tripwire/` to your repo, and then diff on every prompt change.
"""

import tempfile

from llm_tripwire import Suite, TestCase
from llm_tripwire.conditions import contains


def build_suite() -> Suite:
    return Suite(
        name="support_bot",
        cases=[
            TestCase(name="refund - basic", prompt="{input}",
                     conditions=[contains("refund"), contains("days"), contains("help")]),
            TestCase(name="refund - angry", prompt="{input}",
                     conditions=[contains("refund"), contains("sorry"), contains("calm")]),
        ],
    )


def main() -> None:
    suite = build_suite()
    store = tempfile.mkdtemp()  # a real project commits ".tripwire/" instead

    # 1) Bless a known-good run as the baseline.
    good = {
        "refund - basic": "We can process your refund in 5 days, happy to help.",
        "refund - angry": "I'm sorry — we'll refund you and help you stay calm.",
    }
    suite.evaluate_outputs(good).save_baseline(directory=store)
    print("Saved baseline.\n")

    # 2) A prompt change later degrades the angry-user reply.
    after_change = dict(good, **{"refund - angry": "We'll refund you in a few days."})
    diff = suite.evaluate_outputs(after_change).diff_against_baseline(directory=store)

    print(diff.summary())
    threshold = 5  # tolerate up to a 5-point drop in CI
    ok = diff.within_threshold(threshold)
    print(f"\nWithin {threshold}% threshold: {ok}")
    if not ok:
        print("CI would fail here - review before shipping.")


if __name__ == "__main__":
    main()
