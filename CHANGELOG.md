# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-31

First packaged release: `pip install llm-watchdog` now works, with a
dependency-free core and optional `[semantic]` / `[llm]` extras.

### Added

- **Core model** — `TestCase`, `Suite`, and the `ConditionResult` /
  `CaseResult` / `SuiteResult` scoring objects. Fully offline and unit-tested.
- **Heuristic verifiers** — `contains`, `not_contains`, `starts_with`,
  `ends_with`, `word_count`, `bullet_count`, `regex_match`, `valid_json`,
  `json_has_keys`, with a type registry and a YAML-friendly `build_condition`.
- **`semantic_similarity`** — local embedding similarity via
  `sentence-transformers` (optional `[semantic]` extra, imported lazily).
- **Runner** — LiteLLM integration (`run_case` / `run_suite`, plus
  `TestCase.run()` / `Suite.run()`) behind the optional `[llm]` extra.
- **`runs=N` sampling** — call the model several times and aggregate per
  condition (mean score + pass rate) so non-determinism doesn't cause flaky
  pass/fail.
- **V2 LLM judges** — `llm_judge`, `llm_factual`, `llm_rubric`, with a required
  (no-default) `judge_model` and an `estimate_run_cost` helper.
- **Friendly API** — `llm_watchdog.conditions` exposes every verifier as a
  lowercase, function-style constructor.

[Unreleased]: https://github.com/Ahmedniz1/Watchdog/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ahmedniz1/Watchdog/releases/tag/v0.1.0
