"""LLM Watchdog — local-first regression testing for LLM outputs."""

from llm_watchdog.core import (
    Condition,
    ConditionResult,
    CaseResult,
    SuiteResult,
    TestCase,
    Suite,
)
from llm_watchdog.verifiers import (
    build_condition,
    build_conditions,
    available_types,
    ConditionConfigError,
)
from llm_watchdog.runner import (
    run_case,
    run_suite,
    RunnerError,
)
from llm_watchdog.llm_judge import (
    LLMJudge,
    LLMFactual,
    LLMRubric,
    estimate_run_cost,
    is_judge_condition,
)
from llm_watchdog.baseline import (
    save_baseline,
    load_baseline,
    compare,
    Baseline,
    SuiteDiff,
    CaseDiff,
    BaselineError,
)

# Importing this module registers the optional semantic verifier's *type*
# without importing torch (the heavy import is deferred to evaluation time).
from llm_watchdog import semantic  # noqa: F401

# Likewise, importing the runner and llm_judge modules above registers the
# LLM-judge verifier *types* without importing litellm — that network
# dependency is deferred until a run actually happens.

__version__ = "0.1.0"

__all__ = [
    "Condition",
    "ConditionResult",
    "CaseResult",
    "SuiteResult",
    "TestCase",
    "Suite",
    "build_condition",
    "build_conditions",
    "available_types",
    "ConditionConfigError",
    "run_case",
    "run_suite",
    "RunnerError",
    "LLMJudge",
    "LLMFactual",
    "LLMRubric",
    "estimate_run_cost",
    "is_judge_condition",
    "save_baseline",
    "load_baseline",
    "compare",
    "Baseline",
    "SuiteDiff",
    "CaseDiff",
    "BaselineError",
]