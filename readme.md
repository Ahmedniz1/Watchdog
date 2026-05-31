# LLM Watchdog — Project Specification

**Version:** 1.0  
**Status:** Pre-build  
**Type:** Open-source Python package  

---

## 1. The Problem

Anyone shipping an LLM feature changes their prompt regularly. They tweak wording, adjust the system prompt, swap models, or change temperature. Every one of those changes can silently degrade output quality in ways that are subtle — wrong tone, broken format, missed instructions, semantic drift.

The current practice is: change the prompt, eyeball a few outputs, think "looks fine," ship. Then users complain a week later.

There is no lightweight, local, CI-friendly way to catch prompt regressions before they ship. Existing tools either:

- Require heavy infrastructure setup (LangSmith, Arize)
- Use LLM-as-judge for everything, which costs money and is slow (DeepEval)
- Are built for traditional ML, not LLM output evaluation (Evidently, WhyLabs)
- Force custom DSLs that don't fit into existing test workflows

**LLM Watchdog is a Python-native, local-first regression testing library for LLM outputs.** It gives developers a structured way to define what "correct" looks like, run verifiers against real outputs, and catch regressions before they ship — without sending data to a third party, without paying per evaluation, and without learning a new framework.

---

## 2. Core Design Decisions

These decisions were made deliberately and should not be revisited lightly.

**Decision 1 — Condition scores, not raw output diffs**  
LLM outputs are non-deterministic. Diffing raw text across runs causes alert fatigue and is meaningless. The baseline is always a set of *condition pass rates*, never a captured LLM output. Regression means "fewer conditions passing now than before" — not "the text changed."

**Decision 2 — Python API first, YAML as convenience**  
Developers live in Python and Pytest. A rigid YAML-only DSL creates friction. The primary interface is a Python API that drops natively into Pytest. YAML is a secondary layer for teams that want config-driven test suites without writing code.

**Decision 3 — LiteLLM for provider abstraction**  
Abstracting OpenAI, Anthropic, Gemini, and local models from scratch is a maintenance black hole. LiteLLM handles this. It is a declared dependency, not a detail to hide. This lets development focus entirely on the verifier logic and reporting — which is the actual value of the tool.

**Decision 4 — Local-first, no data leaves the machine**  
Training data, prompts, and outputs are often sensitive. No data is sent to any external service for evaluation. Embedding models run locally via `sentence-transformers`. This is a feature, not a limitation, and should be prominent in documentation.

**Decision 5 — LLM judge is opt-in, explicitly scoped to V2**  
Heuristic and embedding-based verifiers cover ~70% of real use cases for free and instantly. LLM-based verification is powerful but costs money and adds latency. It is designed as an explicit extension, not the default path.

---

## 3. Who This Is For

**Primary audience:** Solo developers and small teams (2–5 people) shipping LLM-powered features — chatbots, RAG pipelines, AI writing tools, classification systems — who are changing prompts regularly and have no safety net.

**Secondary audience:** ML engineers at mid-size companies who want lightweight prompt regression testing in CI without onboarding a full observability platform.

**Not for:** Large enterprises with dedicated MLOps teams (they have internal tooling). Pure research (no production deployment concern). Non-text LLM outputs (images, audio — out of scope).

---

## 4. V1 — Core Package (No LLM Calls)

### 4.1 What V1 Delivers

A pip-installable Python package that lets a developer:

1. Define test cases with conditions in Python or YAML
2. Run those cases against any LLM via LiteLLM
3. Get a pass/fail result per condition, per test case
4. Store results and compare against the previous run (regression delta)
5. Get a terminal summary and a local HTML report
6. Drop it into GitHub Actions with a one-line config

### 4.2 Python API

The primary interface. Works natively inside a standard Pytest file.

```python
from llm_watchdog import TestCase, Suite
from llm_watchdog.conditions import contains, not_contains, word_count, semantic_similarity, regex_match, valid_json

# Define a test case
case = TestCase(
    name="refund query - basic",
    prompt="You are a helpful support agent. Answer the following: {input}",
    input="How do I get a refund?",
    model="gpt-4o-mini",  # any LiteLLM-supported model string
    conditions=[
        contains("refund"),
        not_contains("cannot help"),
        word_count(max=150),
        semantic_similarity(
            reference="We can process your refund within 5 to 7 business days",
            threshold=0.75
        ),
    ]
)

# Run a single case
result = case.run()
print(result.summary())

# Or group into a suite and run all
suite = Suite(name="support_bot", cases=[case, ...])
report = suite.run()
report.save("./watchdog_report.html")
assert report.passed, f"Regression detected: {report.delta_summary()}"
```

Inside a Pytest file, this is just a regular test. No custom runner, no special plugin needed.

```python
# test_prompts.py
def test_support_refund_query():
    result = case.run()
    assert result.passed, result.failure_summary()
```

### 4.3 YAML Interface

For teams that prefer config-driven test suites. The YAML is a thin layer that maps directly to the Python API — no functionality exists in YAML that doesn't exist in Python.

```yaml
# watchdog/support_suite.yaml
suite: support_bot
model: gpt-4o-mini
cases:
  - name: refund query - basic
    prompt: "You are a helpful support agent. Answer: {input}"
    input: "How do I get a refund?"
    conditions:
      - type: contains
        value: "refund"
      - type: not_contains
        value: "cannot help"
      - type: word_count
        max: 150
      - type: semantic_similarity
        reference: "We can process your refund within 5 to 7 business days"
        threshold: 0.75

  - name: refund query - edge case angry user
    prompt: "You are a helpful support agent. Answer: {input}"
    input: "I want my money back RIGHT NOW this is ridiculous"
    conditions:
      - type: contains
        value: "refund"
      - type: not_contains
        value: "calm down"
      - type: word_count
        max: 200
```

Run from CLI:

```bash
watchdog run --suite watchdog/support_suite.yaml
watchdog run --suite watchdog/support_suite.yaml --save-baseline
watchdog run --suite watchdog/support_suite.yaml --diff
```

### 4.4 Verifier Types (V1)

All verifiers are deterministic, free, and run locally. No API calls.

| Verifier | Config | What it checks |
|---|---|---|
| `contains` | `value: str` | Output contains this string (case-insensitive) |
| `not_contains` | `value: str` | Output does not contain this string |
| `word_count` | `min: int, max: int` | Output word count within bounds |
| `regex_match` | `pattern: str` | Output matches regex pattern |
| `starts_with` | `value: str` | Output begins with this string |
| `ends_with` | `value: str` | Output ends with this string |
| `valid_json` | — | Output is parseable as valid JSON |
| `json_has_keys` | `keys: list` | Output is valid JSON containing these keys |
| `bullet_count` | `expected: int` | Output has exactly N bullet points |
| `semantic_similarity` | `reference: str, threshold: float` | Cosine similarity between output and reference via local embeddings ≥ threshold |

The `semantic_similarity` verifier uses `sentence-transformers` with a small local model (`all-MiniLM-L6-v2`, ~80MB). It downloads once on first use and runs entirely locally thereafter.

### 4.5 Scoring and Regression

**Per test case:** Each condition is pass (1) or fail (0). Case score = conditions passed / total conditions.

**Per suite:** Suite score = total conditions passed / total conditions across all cases.

**Baseline:** When you run `--save-baseline`, results are stored in a local `.watchdog/` directory as JSON with a timestamp and suite name. This is committed to the repo.

**Regression delta:** On subsequent runs, current suite score is compared against the stored baseline. The diff output shows:

```
Suite: support_bot
────────────────────────────────────────────
✓  refund query - basic         4/4   (was 4/4)
✗  refund query - angry user    2/4   (was 4/4)  ← REGRESSION
✓  tone check                   3/3   (was 3/3)
────────────────────────────────────────────
Score:   9/11  (81%)
Baseline: 11/11 (100%)
Delta:   -19%

2 conditions regressed. Review before shipping.
```

A configurable `--threshold` flag (default: 0%) controls how much regression is tolerable before the CLI exits with a non-zero code (failing CI).

```bash
watchdog run --suite suite.yaml --diff --threshold 10
# passes if score dropped less than 10% from baseline
```

### 4.6 HTML Report

Generated locally at a path you specify. Contains:

- Suite summary with score and delta
- Per-case breakdown with each condition result
- Side-by-side view of actual output vs. reference (for semantic conditions)
- Timestamp and model used
- Exportable as JSON for downstream processing

No external assets. Single self-contained HTML file.

### 4.7 GitHub Actions Integration

A template developers copy into their repo. Documented in the README with a one-copy-paste setup.

```yaml
# .github/workflows/prompt-regression.yml
name: Prompt Regression Tests

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'watchdog/**'

jobs:
  watchdog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install llm-watchdog
      - run: watchdog run --suite watchdog/suite.yaml --diff --threshold 5
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The workflow only triggers on PRs that touch prompt or test files — not on every commit. This avoids unnecessary API costs.

### 4.8 V1 Build Order

Build in this sequence. Each step is independently testable before moving to the next.

1. **Test case schema** — Python dataclasses for `TestCase`, `Condition`, `ConditionResult`, `CaseResult`, `SuiteResult`
2. **Verifier library** — implement each verifier type, fully unit tested with no LLM dependency
3. **Semantic similarity verifier** — integrate `sentence-transformers`, test locally
4. **LiteLLM integration** — the runner that takes a `TestCase`, calls the model, returns raw output
5. **Scoring logic** — pass/fail per condition, aggregate per case and suite
6. **Baseline storage** — read/write `.watchdog/` JSON files, diff calculator
7. **Terminal reporter** — formatted CLI output with color (use `rich`)
8. **HTML report generator** — self-contained HTML, no external dependencies
9. **YAML parser** — maps YAML schema to Python `TestCase` objects
10. **CLI** — `watchdog run`, `watchdog diff`, `watchdog baseline` commands (use `typer`)
11. **GitHub Actions template + README**
12. **PyPI packaging** — `pyproject.toml`, publish to PyPI

### 4.9 V1 Dependencies

```toml
[dependencies]
litellm = ">=1.0"
sentence-transformers = ">=2.0"
pyyaml = ">=6.0"
typer = ">=0.9"
rich = ">=13.0"
jinja2 = ">=3.0"      # HTML report templating
```

No other dependencies. `sentence-transformers` is the heaviest but unavoidable for local semantic checks.

---

## 5. V2 — LLM Judge Extension

V2 adds LLM-based verifiers as an explicit opt-in layer. Nothing in V1 changes. V2 is purely additive.

### 5.1 New Verifier Types

**`llm_judge`** — The most general. Provide a plain-English criterion, the judge model evaluates whether the output meets it. Returns pass/fail with a brief explanation.

```python
from llm_watchdog.conditions import llm_judge

llm_judge(
    criteria="The response is empathetic and does not make any specific financial promises",
    judge_model="gpt-4o-mini",   # separate from the model under test
    threshold=0.8                 # judge returns a 0-1 confidence score
)
```

**`llm_factual`** — Checks factual consistency of the output against a provided reference document. Useful for RAG pipelines where hallucination is the main risk.

```python
from llm_watchdog.conditions import llm_factual

llm_factual(
    reference_doc="Our refund policy states: returns accepted within 30 days...",
    judge_model="gpt-4o-mini"
)
```

**`llm_rubric`** — Multi-point rubric scoring. Define several criteria, get a score per criterion and an aggregate.

```python
from llm_watchdog.conditions import llm_rubric

llm_rubric(
    rubric=[
        "Response acknowledges the user's frustration",
        "Response provides a clear next step",
        "Response does not use corporate jargon",
    ],
    judge_model="gpt-4o-mini",
    pass_threshold=0.7   # at least 70% of rubric points must pass
)
```

### 5.2 Judge Model Design

The judge model is always configured separately from the model under test. Using the same model as both subject and judge creates self-serving bias. This is enforced by making `judge_model` a required parameter — there is no default — forcing the developer to make a conscious choice.

Recommended defaults to document: use a smaller, cheaper model as judge (`gpt-4o-mini`, `claude-haiku`) unless the task genuinely requires frontier capability. This keeps costs low and avoids the worst of same-model bias.

### 5.3 Cost Transparency

Every suite run with LLM judge conditions prints a cost estimate before executing:

```
LLM judge conditions detected.
Estimated cost: ~$0.004 for this run (12 judge calls × gpt-4o-mini)
Continue? [y/N]
```

This is skipped with a `--yes` flag for CI use.

### 5.4 V2 Build Order

1. Judge model caller (thin wrapper over LiteLLM with structured output parsing)
2. `llm_judge` verifier
3. `llm_factual` verifier
4. `llm_rubric` verifier
5. Cost estimator
6. YAML support for V2 condition types
7. HTML report updated to show judge explanations inline

---

## 6. Project Structure

```
llm-watchdog/
├── llm_watchdog/
│   ├── __init__.py
│   ├── core.py              # TestCase, Suite, result dataclasses
│   ├── runner.py            # LiteLLM integration, executes prompts
│   ├── conditions/
│   │   ├── __init__.py
│   │   ├── heuristic.py     # contains, word_count, regex, format checks
│   │   ├── semantic.py      # semantic_similarity via sentence-transformers
│   │   └── llm_judge.py     # V2: llm_judge, llm_factual, llm_rubric
│   ├── scoring.py           # pass/fail aggregation, delta calculation
│   ├── baseline.py          # read/write .watchdog/ JSON baseline files
│   ├── reporter/
│   │   ├── terminal.py      # rich-based CLI output
│   │   ├── html.py          # HTML report generator
│   │   └── templates/
│   │       └── report.html.j2
│   ├── yaml_parser.py       # YAML → Python TestCase objects
│   └── cli.py               # typer CLI: run, diff, baseline
├── tests/
│   ├── test_conditions.py
│   ├── test_scoring.py
│   ├── test_baseline.py
│   └── test_yaml_parser.py
├── examples/
│   ├── basic_python.py
│   ├── pytest_integration.py
│   └── suite.yaml
├── .github/
│   └── workflows/
│       └── prompt-regression.yml   # template for users to copy
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

---

## 7. README Structure (What the Repo Looks Like to a Visitor)

The README is as important as the code for an open-source portfolio project. Structure:

1. **One-line description** — what it does, who it's for
2. **The problem** — 3 sentences, no jargon
3. **Install** — `pip install llm-watchdog`
4. **Quickstart** — working code example, copy-pasteable, under 20 lines
5. **How conditions work** — table of all verifier types
6. **Regression detection** — show the terminal output screenshot
7. **GitHub Actions** — the copy-paste workflow block
8. **V2: LLM Judge** — clearly marked as opt-in extension
9. **Why not X?** — honest comparison vs DeepEval, LangSmith, Pytest alone
10. **Contributing** — how to add a new verifier type

---

## 8. What This Demonstrates (Portfolio Value)

Each component of this project maps to a real engineering skill:

| Component | Skill demonstrated |
|---|---|
| Verifier library | Clean Python API design, separation of concerns |
| Semantic similarity | Applied ML, embeddings, practical NLP |
| Baseline + delta | Software engineering thinking, regression testing concepts |
| LiteLLM integration | Third-party dependency management, provider-agnostic design |
| CLI with Typer | Developer tooling, UX thinking |
| HTML reporter | Jinja2 templating, end-to-end feature delivery |
| GitHub Actions template | DevOps, CI/CD integration |
| Pytest compatibility | Understanding of existing developer workflows |
| V2 LLM judge | LLM evaluation concepts, prompt engineering, cost awareness |
| PyPI packaging | Open-source distribution, project maintenance |

---

## 9. Out of Scope (Explicitly)

These are not on the roadmap and should not be built:

- Web UI or dashboard
- Prompt storage or versioning
- Production traffic monitoring
- Multi-user or team collaboration features
- Non-text output evaluation (images, audio, embeddings)
- Automatic prompt improvement suggestions
- Support for non-Python languages

Each of these would either duplicate existing tools or expand scope beyond what one person can ship to a quality bar worth showing.

---

## 10. Success Criteria

V1 is done when:
- `pip install llm-watchdog` works
- The quickstart example in the README runs end-to-end in under 5 minutes
- All verifier types have unit tests that pass without any LLM API call
- A GitHub Actions run catches a real regression on a prompt change
- The HTML report renders correctly and is self-contained

V2 is done when:
- `llm_judge` condition type works with at least OpenAI and Anthropic judge models
- Cost estimate is shown before any judge calls execute
- All V1 tests still pass unchanged
