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

# Importing this module registers the optional semantic verifier's *type*
# without importing torch (the heavy import is deferred to evaluation time).
from llm_watchdog import semantic  # noqa: F401

__version__ = "0.0.1"

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
]