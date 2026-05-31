"""Deterministic verifiers — the built-in condition library.

Every verifier here is pure and offline: it takes a string and returns a
:class:`~llm_watchdog.core.ConditionResult`. No network, no API key, no
randomness. That is what lets the whole test suite run in CI for free and
makes results reproducible (spec sections 4.9 and 10).

The continuous/embedding-based ``semantic_similarity`` verifier lives in
``llm_watchdog.semantic`` instead, because it pulls in heavy optional
dependencies (sentence-transformers / torch). Keeping it out of this module
means importing the core library stays fast and dependency-free.

Registry
--------
Each verifier registers under a short ``type`` string. The YAML loader
(a later feature) turns ``{"type": "contains", "value": "refund"}`` into the
right object via :func:`build_condition`. Construction errors are raised as
:class:`ConditionConfigError` so config mistakes surface clearly.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Type

from llm_watchdog.core import Condition, ConditionResult


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[Condition]] = {}


class ConditionConfigError(ValueError):
    """Raised when a condition cannot be built from its config dict."""


def register(type_name: str) -> Callable[[Type[Condition]], Type[Condition]]:
    """Class decorator that records a verifier under ``type_name``."""

    def _decorator(cls: Type[Condition]) -> Type[Condition]:
        if type_name in _REGISTRY:
            raise ValueError(f"condition type {type_name!r} already registered")
        cls.type = type_name
        _REGISTRY[type_name] = cls
        return cls

    return _decorator


def available_types() -> List[str]:
    """Sorted list of registered condition type names."""
    return sorted(_REGISTRY)


def build_condition(spec: Dict) -> Condition:
    """Build a single condition from a config dict like ``{"type": ..., ...}``.

    The ``type`` key selects the class; all other keys are passed as keyword
    arguments to its constructor. This is the seam the YAML loader uses.
    """
    if not isinstance(spec, dict):
        raise ConditionConfigError(f"condition spec must be a mapping, got {type(spec).__name__}")
    if "type" not in spec:
        raise ConditionConfigError(f"condition spec missing 'type' key: {spec!r}")

    type_name = spec["type"]
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise ConditionConfigError(
            f"unknown condition type {type_name!r}. "
            f"Available: {', '.join(available_types())}"
        )

    kwargs = {k: v for k, v in spec.items() if k != "type"}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConditionConfigError(f"bad arguments for condition {type_name!r}: {e}") from e


def build_conditions(specs: List[Dict]) -> List[Condition]:
    """Build a list of conditions from a list of config dicts."""
    return [build_condition(s) for s in specs]


# ---------------------------------------------------------------------------
# Substring / affix verifiers
# ---------------------------------------------------------------------------


@register("contains")
class Contains(Condition):
    """Pass if ``value`` appears in the output. Case-insensitive by default."""

    def __init__(self, value: str, case_sensitive: bool = False):
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        self.value = value
        self.case_sensitive = case_sensitive
        cs = " (case-sensitive)" if case_sensitive else ""
        self.description = f"contains {value!r}{cs}"

    def evaluate(self, output: str) -> ConditionResult:
        hay, needle = _maybe_lower(output, self.value, self.case_sensitive)
        passed = needle in hay
        return self._result(passed, detail="" if passed else f"{self.value!r} not found in output")


@register("not_contains")
class NotContains(Condition):
    """Pass if ``value`` does NOT appear in the output. Case-insensitive by default."""

    def __init__(self, value: str, case_sensitive: bool = False):
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        self.value = value
        self.case_sensitive = case_sensitive
        cs = " (case-sensitive)" if case_sensitive else ""
        self.description = f"does not contain {value!r}{cs}"

    def evaluate(self, output: str) -> ConditionResult:
        hay, needle = _maybe_lower(output, self.value, self.case_sensitive)
        present = needle in hay
        return self._result(
            not present,
            detail=f"forbidden phrase {self.value!r} present" if present else "",
        )


@register("starts_with")
class StartsWith(Condition):
    """Pass if the output (after stripping leading whitespace) starts with ``value``."""

    def __init__(self, value: str, case_sensitive: bool = False):
        self.value = value
        self.case_sensitive = case_sensitive
        self.description = f"starts with {value!r}"

    def evaluate(self, output: str) -> ConditionResult:
        text, prefix = _maybe_lower(output.lstrip(), self.value, self.case_sensitive)
        passed = text.startswith(prefix)
        return self._result(passed, detail="" if passed else f"does not start with {self.value!r}")


@register("ends_with")
class EndsWith(Condition):
    """Pass if the output (after stripping trailing whitespace) ends with ``value``."""

    def __init__(self, value: str, case_sensitive: bool = False):
        self.value = value
        self.case_sensitive = case_sensitive
        self.description = f"ends with {value!r}"

    def evaluate(self, output: str) -> ConditionResult:
        text, suffix = _maybe_lower(output.rstrip(), self.value, self.case_sensitive)
        passed = text.endswith(suffix)
        return self._result(passed, detail="" if passed else f"does not end with {self.value!r}")


# ---------------------------------------------------------------------------
# Counting verifiers
# ---------------------------------------------------------------------------


@register("word_count")
class WordCount(Condition):
    """Pass if the output's word count is within ``[min_words, max_words]``.

    Either bound may be omitted. Words are whitespace-delimited tokens.
    """

    def __init__(self, min_words: Optional[int] = None, max_words: Optional[int] = None):
        if min_words is None and max_words is None:
            raise ValueError("word_count needs at least one of min_words / max_words")
        if min_words is not None and max_words is not None and min_words > max_words:
            raise ValueError("min_words cannot exceed max_words")
        self.min_words = min_words
        self.max_words = max_words
        bits = []
        if min_words is not None:
            bits.append(f">= {min_words}")
        if max_words is not None:
            bits.append(f"<= {max_words}")
        self.description = "word count " + " and ".join(bits)

    def evaluate(self, output: str) -> ConditionResult:
        n = len(output.split())
        ok_min = self.min_words is None or n >= self.min_words
        ok_max = self.max_words is None or n <= self.max_words
        passed = ok_min and ok_max
        return self._result(passed, detail=f"{n} words")


@register("bullet_count")
class BulletCount(Condition):
    """Count Markdown-style bullet lines and check the total.

    A bullet line is one whose first non-whitespace character is ``-``, ``*``,
    ``+`` or ``\u2022`` followed by a space, OR an ordered-list marker like
    ``1.`` / ``2)``. This definition is explicit (review fix) so results are
    predictable; nested/indented bullets still count.

    Provide ``count`` for an exact match, or ``min_count`` / ``max_count`` for
    a range.
    """

    _BULLET_RE = re.compile(r"^\s*(?:[-*+\u2022]\s+|\d+[.)]\s+)")

    def __init__(
        self,
        count: Optional[int] = None,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
    ):
        if count is None and min_count is None and max_count is None:
            raise ValueError("bullet_count needs count, min_count, or max_count")
        if count is not None and (min_count is not None or max_count is not None):
            raise ValueError("use either count, or min_count/max_count, not both")
        self.count = count
        self.min_count = min_count
        self.max_count = max_count
        if count is not None:
            self.description = f"exactly {count} bullet(s)"
        else:
            bits = []
            if min_count is not None:
                bits.append(f">= {min_count}")
            if max_count is not None:
                bits.append(f"<= {max_count}")
            self.description = "bullets " + " and ".join(bits)

    def _count_bullets(self, output: str) -> int:
        return sum(1 for line in output.splitlines() if self._BULLET_RE.match(line))

    def evaluate(self, output: str) -> ConditionResult:
        n = self._count_bullets(output)
        if self.count is not None:
            passed = n == self.count
        else:
            ok_min = self.min_count is None or n >= self.min_count
            ok_max = self.max_count is None or n <= self.max_count
            passed = ok_min and ok_max
        return self._result(passed, detail=f"{n} bullet(s) found")


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------


@register("regex_match")
class RegexMatch(Condition):
    """Pass if ``pattern`` is found in the output.

    Uses :func:`re.search` (match anywhere) by default; set ``full_match=True``
    to require the pattern to match the entire string. ``ignore_case`` maps to
    ``re.IGNORECASE``.
    """

    def __init__(self, pattern: str, ignore_case: bool = False, full_match: bool = False):
        self.pattern = pattern
        self.ignore_case = ignore_case
        self.full_match = full_match
        flags = re.IGNORECASE if ignore_case else 0
        try:
            self._compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"invalid regex {pattern!r}: {e}") from e
        kind = "fully matches" if full_match else "matches"
        self.description = f"{kind} /{pattern}/"

    def evaluate(self, output: str) -> ConditionResult:
        if self.full_match:
            hit = self._compiled.fullmatch(output)
        else:
            hit = self._compiled.search(output)
        passed = hit is not None
        return self._result(passed, detail="" if passed else f"no match for /{self.pattern}/")


# ---------------------------------------------------------------------------
# JSON verifiers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence if present.

    LLMs very often wrap JSON in ```json ... ``` or ``` ... ```. Stripping the
    fence (review fix) means a correct payload isn't failed on formatting. If
    there's no fence, the text is returned unchanged.
    """
    s = text.strip()
    if not s.startswith("```"):
        return text
    lines = s.splitlines()
    # drop opening fence line (``` or ```json)
    lines = lines[1:]
    # drop closing fence line if present
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _parse_json(output: str):
    """Parse output as JSON after stripping any code fence. Raises ValueError."""
    candidate = _strip_code_fences(output)
    return json.loads(candidate)


@register("valid_json")
class ValidJson(Condition):
    """Pass if the output parses as JSON (a surrounding code fence is tolerated)."""

    def __init__(self):
        self.description = "output is valid JSON"

    def evaluate(self, output: str) -> ConditionResult:
        try:
            _parse_json(output)
            return self._result(True)
        except (json.JSONDecodeError, ValueError) as e:
            return self._result(False, detail=f"not valid JSON: {e}")


@register("json_has_keys")
class JsonHasKeys(Condition):
    """Pass if the output is a JSON object containing all of ``keys``.

    Top-level keys only. A surrounding code fence is tolerated. ``score`` is
    the fraction of required keys present, so partial credit is visible even
    though ``passed`` requires all of them.
    """

    def __init__(self, keys: List[str]):
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            raise ValueError("json_has_keys needs at least one key")
        self.keys = list(keys)
        self.description = f"JSON has keys {self.keys}"

    def evaluate(self, output: str) -> ConditionResult:
        try:
            data = _parse_json(output)
        except (json.JSONDecodeError, ValueError) as e:
            return self._result(False, score=0.0, detail=f"not valid JSON: {e}")
        if not isinstance(data, dict):
            return self._result(False, score=0.0, detail="JSON is not an object")

        present = [k for k in self.keys if k in data]
        missing = [k for k in self.keys if k not in data]
        score = len(present) / len(self.keys)
        passed = not missing
        detail = "" if passed else f"missing keys: {missing}"
        return self._result(passed, score=score, detail=detail)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _maybe_lower(text: str, needle: str, case_sensitive: bool):
    """Return (text, needle), lower-cased together unless case_sensitive."""
    if case_sensitive:
        return text, needle
    return text.lower(), needle.lower()