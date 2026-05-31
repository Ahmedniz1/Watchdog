"""Baseline storage and regression diffing — the heart of "did it get worse?".

A single run tells you how many conditions pass *today*. Regression testing
needs a *yesterday* to compare against. That comparison is this module's job.

The model follows spec Decision 1: a baseline is a set of **condition pass
rates**, never captured LLM text. We persist a :class:`SuiteResult` (whose
``to_dict`` already omits raw output) under ``.tripwire/<suite>.json``, commit
it to the repo, and on later runs compare the current result against it. A
"regression" is *fewer conditions passing than before* — not "the text changed",
which for non-deterministic models would be meaningless noise.

Typical flow::

    result = suite.run()
    result.save_baseline()                  # once, when output is known-good

    # ...later, after a prompt change...
    result = suite.run()
    diff = result.diff_against_baseline()
    print(diff.summary())
    assert diff.within_threshold(5), "regression > 5% from baseline"

Everything here is pure stdlib (``json`` + ``pathlib``) so it stays in the
dependency-free core.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from llm_tripwire.core import SuiteResult

#: Where baselines live by default. Committed to the repo so the comparison
#: point travels with the code that produced it.
DEFAULT_DIR = ".tripwire"

#: Bumped if the on-disk JSON layout ever changes, so old files fail loudly
#: instead of being silently misread.
SCHEMA_VERSION = 1


class BaselineError(RuntimeError):
    """Raised when a baseline file is missing, malformed, or the wrong version."""


def _safe_filename(suite_name: str) -> str:
    """Turn a suite name into a safe, readable file stem (``support bot`` -> ``support_bot``)."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", suite_name).strip("_")
    return stem or "suite"


def baseline_path(suite_name: str, directory: str = DEFAULT_DIR) -> Path:
    """Path of the baseline file for ``suite_name`` (it need not exist)."""
    return Path(directory) / f"{_safe_filename(suite_name)}.json"


def save_baseline(result: SuiteResult, directory: str = DEFAULT_DIR) -> Path:
    """Write ``result`` as the baseline for its suite and return the file path.

    Overwrites any existing baseline for the same suite — the baseline is always
    "the last run you blessed as good", not a history. A UTC timestamp is stored
    alongside the result for display.
    """
    path = baseline_path(result.suite_name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "version": SCHEMA_VERSION,
        "suite_name": result.suite_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "result": result.to_dict(),
    }
    path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return path


@dataclass
class Baseline:
    """A loaded baseline: when it was saved and the stored result snapshot."""

    suite_name: str
    saved_at: str
    result: Mapping  # the SuiteResult.to_dict() snapshot

    @property
    def passed_conditions(self) -> int:
        return int(self.result.get("passed_conditions", 0))

    @property
    def total_conditions(self) -> int:
        return int(self.result.get("total_conditions", 0))

    @property
    def score(self) -> float:
        return float(self.result.get("score", 0.0))

    def case(self, name: str) -> Optional[Mapping]:
        """Stored snapshot for one case by name, or ``None`` if it wasn't present."""
        for c in self.result.get("cases", []):
            if c.get("case_name") == name:
                return c
        return None


def load_baseline(suite_name: str, directory: str = DEFAULT_DIR) -> Optional[Baseline]:
    """Load the baseline for ``suite_name``, or ``None`` if none has been saved.

    Raises :class:`BaselineError` if the file exists but is corrupt or was
    written by a newer, incompatible schema.
    """
    path = baseline_path(suite_name, directory)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise BaselineError(f"could not read baseline {path}: {e}") from e
    if data.get("version") != SCHEMA_VERSION:
        raise BaselineError(
            f"baseline {path} has schema version {data.get('version')!r}, "
            f"expected {SCHEMA_VERSION}. Re-save it with save_baseline()."
        )
    return Baseline(
        suite_name=data.get("suite_name", suite_name),
        saved_at=data.get("saved_at", ""),
        result=data.get("result", {}),
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class CaseDiff:
    """How one case's passing-condition count compares to the baseline."""

    case_name: str
    now_passed: int
    now_total: int
    was_passed: Optional[int]  # None => case is new (absent from baseline)
    was_total: Optional[int]

    @property
    def is_new(self) -> bool:
        return self.was_passed is None

    @property
    def regressed(self) -> bool:
        """Fewer conditions pass now than in the baseline."""
        return self.was_passed is not None and self.now_passed < self.was_passed

    @property
    def improved(self) -> bool:
        return self.was_passed is not None and self.now_passed > self.was_passed


@dataclass
class SuiteDiff:
    """Current run vs. baseline, with a pass/fail gate and a printable summary."""

    suite_name: str
    now_passed: int
    now_total: int
    base_passed: int
    base_total: int
    case_diffs: List[CaseDiff] = field(default_factory=list)
    removed_cases: List[str] = field(default_factory=list)  # in baseline, gone now
    saved_at: str = ""

    @property
    def now_score(self) -> float:
        return 1.0 if self.now_total == 0 else self.now_passed / self.now_total

    @property
    def base_score(self) -> float:
        return 1.0 if self.base_total == 0 else self.base_passed / self.base_total

    @property
    def delta(self) -> float:
        """Change in suite score as a fraction (negative means regression)."""
        return self.now_score - self.base_score

    @property
    def regressed_conditions(self) -> int:
        """Total conditions that stopped passing, summed across cases."""
        return sum(
            max(0, cd.was_passed - cd.now_passed)
            for cd in self.case_diffs
            if cd.was_passed is not None
        )

    @property
    def regressed_cases(self) -> List[CaseDiff]:
        return [cd for cd in self.case_diffs if cd.regressed]

    def within_threshold(self, threshold_pct: float = 0.0) -> bool:
        """True if the score dropped by no more than ``threshold_pct`` points.

        ``threshold_pct`` is in percentage points of the suite score, matching
        the spec's ``--threshold`` flag: ``0`` (default) fails on any drop;
        ``5`` tolerates up to a 5-point drop before failing CI.
        """
        drop_points = (self.base_score - self.now_score) * 100
        return drop_points <= threshold_pct

    def summary(self) -> str:
        """ASCII, terminal-safe table of per-case results vs. baseline."""
        lines = [f"Suite: {self.suite_name}"]
        lines.append("-" * 52)
        name_w = max((len(cd.case_name) for cd in self.case_diffs), default=10)
        name_w = min(max(name_w, 10), 40)
        for cd in self.case_diffs:
            mark = "PASS" if cd.now_passed == cd.now_total else "FAIL"
            now = f"{cd.now_passed}/{cd.now_total}"
            if cd.is_new:
                was = "(new)"
            else:
                was = f"(was {cd.was_passed}/{cd.was_total})"
            flag = "  <-- REGRESSION" if cd.regressed else ""
            lines.append(f"  [{mark:4}] {cd.case_name:<{name_w}}  {now:>7}  {was}{flag}")
        for name in self.removed_cases:
            lines.append(f"  [gone] {name:<{name_w}}  {'--':>7}  (was in baseline)")
        lines.append("-" * 52)
        lines.append(f"Score:    {self.now_passed}/{self.now_total}  ({round(self.now_score * 100)}%)")
        lines.append(f"Baseline: {self.base_passed}/{self.base_total}  ({round(self.base_score * 100)}%)")
        lines.append(f"Delta:    {self.delta * 100:+.0f}%")
        n = self.regressed_conditions
        if n:
            lines.append(f"\n{n} condition(s) regressed. Review before shipping.")
        else:
            lines.append("\nNo regressions.")
        return "\n".join(lines)


def compare(current: SuiteResult, baseline: Baseline) -> SuiteDiff:
    """Diff a fresh :class:`SuiteResult` against a loaded :class:`Baseline`.

    Cases are matched by name. A case absent from the baseline is marked new
    (never counts as a regression); a case in the baseline but gone now is
    listed separately. Conditions within a case are compared by count — the
    spec's "fewer conditions passing" definition — which stays meaningful even
    when conditions are added or removed.
    """
    case_diffs: List[CaseDiff] = []
    current_names = set()
    for c in current.case_results:
        current_names.add(c.case_name)
        base_case = baseline.case(c.case_name)
        case_diffs.append(
            CaseDiff(
                case_name=c.case_name,
                now_passed=c.passed_count,
                now_total=c.total,
                was_passed=None if base_case is None else int(base_case.get("passed_count", 0)),
                was_total=None if base_case is None else int(base_case.get("total", 0)),
            )
        )
    removed = [
        c.get("case_name", "?")
        for c in baseline.result.get("cases", [])
        if c.get("case_name") not in current_names
    ]
    return SuiteDiff(
        suite_name=current.suite_name,
        now_passed=current.passed_conditions,
        now_total=current.total_conditions,
        base_passed=baseline.passed_conditions,
        base_total=baseline.total_conditions,
        case_diffs=case_diffs,
        removed_cases=removed,
        saved_at=baseline.saved_at,
    )
