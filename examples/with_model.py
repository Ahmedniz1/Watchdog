"""Live example — calls a real model.

Requires the `llm` extra and a provider API key:

    pip install "llm-tripwire[llm]"
    export OPENAI_API_KEY=sk-...        # or any provider LiteLLM supports
    python examples/with_model.py

It runs a small suite against the model, samples each case a few times so a
single non-deterministic reply doesn't decide the result, and prints a summary.
"""

from llm_tripwire import Suite, TestCase
from llm_tripwire.conditions import contains, semantic_similarity, word_count

MODEL = "gpt-4o-mini"  # any LiteLLM model string

suite = Suite(
    name="support_bot",
    model=MODEL,
    cases=[
        TestCase(
            name="refund - basic",
            prompt="You are a helpful support agent. Answer concisely: {input}",
            input="How do I get a refund?",
            conditions=[
                contains("refund"),
                word_count(max_words=150),
                semantic_similarity(
                    reference="Refunds are processed within 5 to 7 business days.",
                    threshold=0.6,
                ),
            ],
        ),
        TestCase(
            name="refund - angry user",
            prompt="You are a helpful support agent. Answer calmly: {input}",
            input="I want my money back RIGHT NOW this is ridiculous",
            conditions=[
                contains("refund"),
                word_count(max_words=200),
            ],
        ),
    ],
)


def main() -> None:
    # runs=3 with a non-zero temperature: each condition reports a pass rate
    # across the samples instead of a single coin-flip.
    report = suite.run(runs=3, temperature=0.7)
    print(report.summary())
    print(f"\nSuite passed: {report.passed}")


if __name__ == "__main__":
    main()
