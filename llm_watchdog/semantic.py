"""Optional semantic-similarity verifier.

This is the one verifier that is *not* dependency-free: it needs
``sentence-transformers`` (which pulls in torch/transformers). To keep the
core install lightweight and deterministic, that dependency is an optional
extra::

    pip install llm-watchdog[semantic]

and it is imported lazily — only when a ``semantic_similarity`` condition is
actually evaluated. Importing this module does NOT import torch.

Calibration caveat (review note): cosine-similarity thresholds are model- and
domain-specific. A value like 0.75 that's reasonable for one embedding model
may be too strict or too loose for another, and long outputs compared against
short references score lower than intuition suggests. Tune thresholds against
your own data rather than trusting a default.
"""

from __future__ import annotations

from typing import Optional

from llm_watchdog.core import Condition, ConditionResult
from llm_watchdog.verifiers import register

# Default model: small, fast, widely used. ~80MB, downloaded once and cached.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Process-wide cache of loaded embedding models, keyed by model name, so we pay
# the load cost once per model rather than per condition.
_MODEL_CACHE = {}


def _get_model(model_name: str):
    """Lazily import sentence-transformers and load (and cache) a model.

    Raises a clear, actionable error if the optional dependency is missing.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433 (lazy)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "semantic_similarity requires the 'semantic' extra. "
            "Install it with:  pip install llm-watchdog[semantic]"
        ) from e
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


@register("semantic_similarity")
class SemanticSimilarity(Condition):
    """Pass if the output is semantically similar enough to ``reference``.

    Cosine similarity of sentence embeddings is compared against ``threshold``
    (0..1). The continuous similarity is reported as the result ``score`` so a
    sampling/aggregation layer can average it across repeated runs rather than
    collapsing to pass/fail on a single sample.
    """

    def __init__(
        self,
        reference: str,
        threshold: float = 0.75,
        model: str = DEFAULT_MODEL,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.reference = reference
        self.threshold = threshold
        self.model_name = model
        self.description = f"semantically similar to reference (>= {threshold:.2f})"

    def _similarity(self, output: str) -> float:
        model = _get_model(self.model_name)
        # util.cos_sim returns a 1x1 tensor; pull out the scalar.
        from sentence_transformers import util  # lazy

        emb = model.encode([self.reference, output])
        sim = util.cos_sim(emb[0], emb[1])
        return float(sim.item())

    def evaluate(self, output: str) -> ConditionResult:
        sim = self._similarity(output)
        # Clamp to [0,1]; cosine sim can dip slightly negative for unrelated text.
        score = max(0.0, min(1.0, sim))
        passed = sim >= self.threshold
        detail = f"similarity {sim:.3f} (threshold {self.threshold:.2f})"
        return self._result(passed, score=score, detail=detail)