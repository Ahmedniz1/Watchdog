"""The friendly, function-style way to build conditions.

This is the import most people will use::

    from llm_tripwire.conditions import contains, word_count, semantic_similarity

    conditions = [
        contains("refund"),
        word_count(max_words=150),
        semantic_similarity(reference="processed within 5-7 days", threshold=0.75),
    ]

Each name here is the corresponding verifier class exposed under a lowercase,
verb-like alias, so ``contains("refund")`` simply constructs a
:class:`~llm_tripwire.verifiers.Contains`. Two consequences worth knowing:

* The keyword arguments are exactly the class's arguments — the *same*
  vocabulary the YAML loader and ``build_condition`` use, so there's only ever
  one set of names to learn (``word_count(max_words=150)``, not a second alias).
* The objects are normal :class:`~llm_tripwire.core.Condition` instances, so
  ``isinstance``, ``.evaluate()`` and the registry all behave as expected.

The heavier verifiers keep their lazy-import behaviour: ``semantic_similarity``
only pulls in ``sentence-transformers`` when evaluated, and the ``llm_*`` judges
only pull in ``litellm`` when run. Importing this module does neither.
"""

from __future__ import annotations

from llm_tripwire.llm_judge import LLMFactual as llm_factual
from llm_tripwire.llm_judge import LLMJudge as llm_judge
from llm_tripwire.llm_judge import LLMRubric as llm_rubric
from llm_tripwire.semantic import SemanticSimilarity as semantic_similarity
from llm_tripwire.verifiers import BulletCount as bullet_count
from llm_tripwire.verifiers import Contains as contains
from llm_tripwire.verifiers import EndsWith as ends_with
from llm_tripwire.verifiers import JsonHasKeys as json_has_keys
from llm_tripwire.verifiers import NotContains as not_contains
from llm_tripwire.verifiers import RegexMatch as regex_match
from llm_tripwire.verifiers import StartsWith as starts_with
from llm_tripwire.verifiers import ValidJson as valid_json
from llm_tripwire.verifiers import WordCount as word_count

__all__ = [
    # heuristic
    "contains",
    "not_contains",
    "starts_with",
    "ends_with",
    "word_count",
    "bullet_count",
    "regex_match",
    "valid_json",
    "json_has_keys",
    # semantic (optional extra)
    "semantic_similarity",
    # LLM judges (optional extra)
    "llm_judge",
    "llm_factual",
    "llm_rubric",
]
