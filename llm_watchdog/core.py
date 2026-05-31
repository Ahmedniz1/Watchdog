"""Core data model for LLM Watchdog.

Defines the building blocks used across the whole package:

- ``Condition``      : abstract base every verifier implements.
- ``ConditionResult``: outcome of evaluating one condition against one output.
- ``CaseResult``     : outcome of evaluating all conditions for one test case.
- ``SuiteResult``    : aggregate outcome across every case in a suite.
- ``TestCase``       : a single prompt + input + the conditions to check.
- ``Suite``          : a named collection of test cases.

This module performs **no** LLM calls. Everything here can be unit tested by
feeding in plain strings, which is what lets every verifier test run without
hitting an API (Success Criteria, spec section 10).

Getting the model output is the runner's job (a later feature). The contract
that connects the two is ``TestCase.evaluate_output(output)`` -> ``CaseResult``:
the runner fetches the output, this layer scores it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class Condition(ABC):
    """Abstract base for every verifier (``contains``, ``word_count``, ...).

    Subclasses (built in the verifier-library feature) must:
      * set the class attribute ``type`` to a short identifier, e.g. ``"contains"``;
      * set ``self.description`` to a human-readable string in ``__init__``;
      * implement :meth:`evaluate`.

    The :meth:`_result` helper keeps subclasses tiny — they decide pass/fail
    (and optionally a continuous score) and the base fills in the metadata.
    """

    #: Short machine identifier for the condition, e.g. "contains".
    type: str = "condition"

    #: Human-readable description, set by each subclass in ``__init__``.
    description: str = ""

    @abstractmethod
    def evaluate(self, output: str) -> "ConditionResult":
        """Check ``output`` against this condition and return a result."""
        raise NotImplementedError

    def _result(
        self,
        passed: bool,
        score: Optional[float] = None,
        detail: str = "",
    ) -> "ConditionResult":
        """Build a :class:`ConditionResult` tagged with this condition's metadata.

        ``score`` defaults to ``1.0`` / ``0.0`` for plain pass/fail checks. Checks
        that are naturally continuous (semantic similarity, LLM judge) pass an
        explicit 0..1 value. Carrying a float here — rather than only a bool —
        is the hook that lets a future ``runs=N`` sampling layer average scores
        across repeated runs instead of comparing single coin-flip outcomes.
        """
        if score is None:
            score = 1.0 if passed else 0.0
        return ConditionResult(
            condition_type=self.type,
            description=self.description,
            passed=passed,
            score=score,
            detail=detail,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} {self.description!r}>"


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    """Outcome of evaluating a single condition against a single output.

    ``runs`` and ``pass_rate`` describe sampling. For a normal single-run
    evaluation they stay at their defaults (``runs=1``, ``pass_rate=None``) and
    nothing changes. When the runner samples a case ``N`` times (``runs=N``),
    these results are aggregated: ``score`` becomes the mean across runs and
    ``pass_rate`` the fraction of runs in which the condition passed — the
    stable signal that the data model's float-valued ``score`` was designed to
    enable (see :meth:`Condition._result`).
    """

    condition_type: str
    description: str
    passed: bool
    score: float = 0.0  # 0.0..1.0
    detail: str = ""
    runs: int = 1  # number of samples aggregated into this result
    pass_rate: Optional[float] = None  # fraction of runs passed; None when runs == 1

    def to_dict(self) -> Dict:
        return {
            "condition_type": self.condition_type,
            "description": self.description,
            "passed": self.passed,
            "score": self.score,
            "detail": self.detail,
            "runs": self.runs,
            "pass_rate": self.pass_rate,
        }


@dataclass
class CaseResult:
    """Outcome of evaluating every condition for one test case.

    Per spec section 4.5: case score = conditions passed / total conditions.
    """

    case_name: str
    output: str
    condition_results: List[ConditionResult] = field(default_factory=list)
    model: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.condition_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.condition_results if r.passed)

    @property
    def score(self) -> float:
        """Fraction of conditions that passed. Empty case scores 1.0."""
        if self.total == 0:
            return 1.0
        return self.passed_count / self.total

    @property
    def passed(self) -> bool:
        """True only if every condition passed (empty case passes vacuously)."""
        return all(r.passed for r in self.condition_results)

    def summary(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"{self.case_name}: {self.passed_count}/{self.total} conditions passed ({flag})"

    def failure_summary(self) -> str:
        """Multi-line description of just the failing conditions. Empty if all pass."""
        failures = [r for r in self.condition_results if not r.passed]
        if not failures:
            return ""
        lines = [f"{self.case_name}: {len(failures)} condition(s) failed"]
        for r in failures:
            detail = f" - {r.detail}" if r.detail else ""
            lines.append(f"  - {r.description}{detail}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "case_name": self.case_name,
            "model": self.model,
            "passed": self.passed,
            "score": self.score,
            "passed_count": self.passed_count,
            "total": self.total,
            "conditions": [r.to_dict() for r in self.condition_results],
            # NOTE: raw `output` is deliberately NOT persisted to the baseline
            # (Design Decision 1: baselines store condition pass rates, never
            # captured LLM text). It lives on the in-memory object for the
            # HTML report only.
        }


@dataclass
class SuiteResult:
    """Aggregate outcome across every case in a suite.

    Per spec section 4.5: suite score = total conditions passed / total
    conditions across all cases (not the mean of per-case scores).
    """

    suite_name: str
    case_results: List[CaseResult] = field(default_factory=list)
    model: Optional[str] = None

    @property
    def total_conditions(self) -> int:
        return sum(c.total for c in self.case_results)

    @property
    def passed_conditions(self) -> int:
        return sum(c.passed_count for c in self.case_results)

    @property
    def score(self) -> float:
        if self.total_conditions == 0:
            return 1.0
        return self.passed_conditions / self.total_conditions

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.case_results)

    def summary(self) -> str:
        lines = [f"Suite: {self.suite_name}"]
        for c in self.case_results:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.case_name}  {c.passed_count}/{c.total}")
        pct = round(self.score * 100)
        lines.append(f"Score: {self.passed_conditions}/{self.total_conditions} ({pct}%)")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "suite_name": self.suite_name,
            "model": self.model,
            "score": self.score,
            "passed": self.passed,
            "passed_conditions": self.passed_conditions,
            "total_conditions": self.total_conditions,
            "cases": [c.to_dict() for c in self.case_results],
        }

    # `save()`, `delta_summary()` and baseline comparison arrive with the
    # baseline-storage and reporter features. `to_dict()` is the seam they
    # will serialize through.


# ---------------------------------------------------------------------------
# Test case + suite definitions
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A single prompt + input + the conditions its output must satisfy.

    ``model`` is optional here; if unset, a :class:`Suite` default or the
    runner supplies it. Keeping the model as plain data (never hard-coded)
    means a retired or renamed model string is a one-line config change.
    """

    # The class is named `TestCase` (the spec's public API), which collides with
    # pytest's "Test*" collection heuristic and unittest.TestCase. This flag tells
    # pytest it is not a test class, avoiding spurious collection warnings.
    __test__ = False

    name: str
    prompt: str
    input: str = ""
    model: Optional[str] = None
    conditions: List[Condition] = field(default_factory=list)

    def render_prompt(self) -> str:
        """Substitute ``{input}`` into the prompt template.

        Uses literal replacement of the single ``{input}`` token rather than
        ``str.format``. ``str.format`` would choke on (or silently mangle) any
        other braces in the prompt — JSON examples, code snippets, f-string-like
        text — which real prompts frequently contain.
        """
        return self.prompt.replace("{input}", self.input)

    def evaluate_output(self, output: str, model: Optional[str] = None) -> CaseResult:
        """Score a model output against this case's conditions. No LLM call.

        This is the pure, fully testable core: pass any string and get a
        :class:`CaseResult`. The runner will call this after fetching ``output``
        from the model under test.
        """
        results = [c.evaluate(output) for c in self.conditions]
        return CaseResult(
            case_name=self.name,
            output=output,
            condition_results=results,
            model=model or self.model,
        )

    def run(self, model: Optional[str] = None, **kwargs) -> CaseResult:
        """Call the model under test and score the output. Makes one LLM call.

        Convenience wrapper around :func:`llm_watchdog.runner.run_case` so the
        quickstart ``case.run()`` works. ``runner`` is imported lazily here:
        it pulls in the optional ``litellm`` dependency, and keeping that off
        the core import path is what lets the offline layer stay dependency-free.
        """
        from llm_watchdog.runner import run_case  # lazy: keeps litellm optional

        return run_case(self, model=model, **kwargs)


@dataclass
class Suite:
    """A named collection of test cases.

    ``model`` is a suite-wide default applied to any case that doesn't set its
    own. Running the suite (calling the model) is the runner's job; this layer
    only assembles a :class:`SuiteResult` from per-case outputs.
    """

    name: str
    cases: List[TestCase] = field(default_factory=list)
    model: Optional[str] = None

    def effective_model(self, case: TestCase) -> Optional[str]:
        """Resolve which model a case should use: case override, else suite default."""
        return case.model or self.model

    def evaluate_outputs(self, outputs: Dict[str, str]) -> SuiteResult:
        """Score a mapping of ``{case_name: output}`` into a :class:`SuiteResult`.

        Lets the suite be tested end-to-end with canned outputs, no API needed.
        Cases missing from ``outputs`` are skipped (the runner guarantees one
        output per case in real use).
        """
        case_results: List[CaseResult] = []
        for case in self.cases:
            if case.name not in outputs:
                continue
            case_results.append(
                case.evaluate_output(outputs[case.name], model=self.effective_model(case))
            )
        return SuiteResult(
            suite_name=self.name,
            case_results=case_results,
            model=self.model,
        )

    def run(self, model: Optional[str] = None, **kwargs) -> SuiteResult:
        """Run every case against its model and score the outputs. One call per case.

        Convenience wrapper around :func:`llm_watchdog.runner.run_suite` so the
        quickstart ``suite.run()`` works. Imported lazily for the same reason as
        :meth:`TestCase.run` — to keep the optional ``litellm`` dependency off
        the core import path.
        """
        from llm_watchdog.runner import run_suite  # lazy: keeps litellm optional

        return run_suite(self, model=model, **kwargs)
