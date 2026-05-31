# LLM Watchdog

**Catch prompt regressions before they ship — locally, in CI, without sending your data anywhere.**

You change a prompt, swap a model, or nudge the temperature, eyeball a couple of
outputs, and ship. A week later users complain the tone is off or the JSON broke.
LLM Watchdog gives you a way to *define what "good" looks like* for an LLM
feature, run those checks against real model output, and fail your CI when fewer
checks pass than before.

- 🪶 **Lightweight** — the core has **zero dependencies** and runs in milliseconds.
- 🔒 **Local-first** — no data leaves your machine; embeddings run locally.
- 🧪 **Pytest-native** — it's just Python; drop it into the test setup you already have.
- 🤖 **Provider-agnostic** — any model [LiteLLM](https://github.com/BerriAI/litellm) supports (OpenAI, Anthropic, Gemini, local…).
- 💸 **Free by default** — heuristic + local-embedding checks cost nothing; LLM-as-judge is strictly opt-in.

---

## Install

```bash
pip install llm-watchdog                 # core: heuristic checks, scoring, pytest use
pip install "llm-watchdog[semantic]"     # + local semantic_similarity (sentence-transformers)
pip install "llm-watchdog[llm]"          # + run prompts / LLM-judge checks (litellm)
pip install "llm-watchdog[all]"          # everything
```

The core install pulls in nothing extra. You only download the heavy bits if you
actually use semantic similarity or call a model.

---

## Quickstart

Define a case, say what its output must satisfy, and check it. This first example
needs **no API key and no extra dependencies** — you score output you already have:

```python
from llm_watchdog import TestCase
from llm_watchdog.conditions import contains, not_contains, word_count

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

result = case.evaluate_output("Sure — we can process your refund within 5–7 business days.")
print(result.summary())          # refund query: 3/3 conditions passed (PASS)
assert result.passed, result.failure_summary()
```

To actually call the model under test, install the `[llm]` extra, set the
provider's API key, give the case a `model`, and use `.run()`:

```python
from llm_watchdog import TestCase
from llm_watchdog.conditions import contains, semantic_similarity

case = TestCase(
    name="refund query",
    prompt="You are a helpful support agent. Answer: {input}",
    input="How do I get a refund?",
    model="gpt-4o-mini",         # any LiteLLM model string
    conditions=[
        contains("refund"),
        semantic_similarity(reference="processed within 5-7 business days", threshold=0.7),
    ],
)

result = case.run()              # makes the model call, then scores
print(result.summary())
```

### In Pytest

It's just a test — no plugin, no custom runner:

```python
def test_refund_query():
    result = case.run()
    assert result.passed, result.failure_summary()
```

---

## How conditions work

A **condition** checks one property of the output. All of these are deterministic,
free, and run locally:

| Condition | Example | Checks |
|---|---|---|
| `contains` | `contains("refund")` | output contains the string (case-insensitive) |
| `not_contains` | `not_contains("cannot help")` | output does **not** contain the string |
| `starts_with` / `ends_with` | `starts_with("Dear")` | output begins / ends with the string |
| `word_count` | `word_count(min_words=20, max_words=150)` | word count within bounds |
| `bullet_count` | `bullet_count(count=3)` | exact or ranged number of bullet points |
| `regex_match` | `regex_match(r"\d{4}-\d{2}-\d{2}")` | output matches the pattern |
| `valid_json` | `valid_json()` | output parses as JSON (tolerates ```code fences```) |
| `json_has_keys` | `json_has_keys(keys=["status", "id"])` | JSON object has all the keys |
| `semantic_similarity` | `semantic_similarity(reference="…", threshold=0.75)` | local-embedding cosine similarity ≥ threshold |

Each condition yields a pass/fail **and** a 0–1 score. A case's score is
`conditions passed / total`; a suite's is `total conditions passed / total`.

### Grouping cases into a suite

```python
from llm_watchdog import Suite

suite = Suite(name="support_bot", model="gpt-4o-mini", cases=[case, ...])
report = suite.run()
print(report.summary())
assert report.passed
```

---

## Handling non-determinism with `runs=N`

Models don't return the same text twice, so a single call gives you a coin-flip
pass/fail. Sample several times and aggregate instead:

```python
result = case.run(runs=5, temperature=0.7)

cond = result.condition_results[0]
print(cond.pass_rate)   # e.g. 0.8  → passed 4 of 5 samples
print(cond.score)       # mean score across the 5 samples
```

A condition counts as passed when its **pass rate ≥ `min_pass_rate`** (default
`0.5`, a simple majority), so one unlucky sample won't fail your suite. Raise
`min_pass_rate` to `1.0` to demand every sample pass. (Sampling only helps if the
model can vary, so `runs > 1` with `temperature=0` warns you.)

---

## LLM-as-judge (opt-in)

When a check genuinely needs a model to read the output — "is this empathetic?",
"does this contradict the policy doc?" — use a judge. These cost money and add
latency, so they are never on by default. The judge model is **required and must
be set explicitly** (use a different, cheaper model than the one under test to
avoid self-serving bias):

```python
from llm_watchdog.conditions import llm_judge, llm_factual, llm_rubric

llm_judge(
    criteria="The response is empathetic and makes no specific financial promises",
    judge_model="gpt-4o-mini",
    threshold=0.8,
)

llm_factual(
    reference_doc="Our refund policy: returns accepted within 30 days…",
    judge_model="gpt-4o-mini",
)

llm_rubric(
    rubric=[
        "Acknowledges the user's frustration",
        "Provides a clear next step",
        "Avoids corporate jargon",
    ],
    judge_model="gpt-4o-mini",
    pass_threshold=0.7,
)
```

Estimate the cost of a run before paying for it:

```python
from llm_watchdog import estimate_run_cost
print(estimate_run_cost(suite))   # {'calls': 12, 'usd': 0.0036, 'by_model': {'gpt-4o-mini': 12}}
```

---

## Config-driven suites (no code)

Every condition maps to a plain dict, so suites can come from config:

```python
from llm_watchdog import TestCase, build_conditions

case = TestCase(
    name="refund query",
    prompt="You are a helpful support agent. Answer: {input}",
    input="How do I get a refund?",
    conditions=build_conditions([
        {"type": "contains", "value": "refund"},
        {"type": "word_count", "max_words": 150},
        {"type": "semantic_similarity", "reference": "processed within 5-7 days", "threshold": 0.75},
    ]),
)
```

This is the seam a YAML loader and CLI build on (on the roadmap below).

---

## Why not …?

- **DeepEval** — leans on LLM-as-judge for most checks: slower and costs money on every run. Watchdog makes heuristic + local-embedding checks the free default and judges strictly opt-in.
- **LangSmith / Arize** — hosted observability platforms; great at scale, heavy to adopt, and your data leaves your machine. Watchdog is a local library you `pip install`.
- **Pytest with hand-written asserts** — fine until you want semantic checks, scoring, sampling across non-deterministic runs, and regression baselines. Watchdog gives you those without a new framework.

---

## Roadmap

Shipped: core model · heuristic + semantic verifiers · LiteLLM runner ·
`runs=N` sampling · LLM-judge layer · packaging.

Next: baseline storage + regression diff (`--save-baseline` / `--diff`) ·
terminal & self-contained HTML reports · YAML loader · `watchdog` CLI ·
GitHub Actions template.

Deliberately out of scope: a web UI/dashboard, prompt storage/versioning,
production traffic monitoring, and non-text output evaluation.

The full design rationale lives in [docs/SPEC.md](docs/SPEC.md).

---

## Contributing

Run the tests (they're fully offline — no API key needed):

```bash
pip install -e ".[dev]"
pytest
```

Adding a verifier is a small, self-contained change: subclass `Condition`,
decorate it with `@register("your_type")`, set `self.description` and implement
`evaluate()`, then expose it in `llm_watchdog/conditions.py`. See the existing
verifiers in `llm_watchdog/verifiers.py` for the pattern.

## License

MIT — see [LICENSE](LICENSE).
