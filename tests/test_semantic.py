"""Tests for the optional semantic_similarity verifier (Feature 2).

The heavy dependency (sentence-transformers) is optional. These tests:
  * always verify the *type* is registered and the import is lazy;
  * run a real similarity check only if the dependency is installed,
    otherwise skip — so core CI stays fast and dependency-free.
"""

import importlib.util

import pytest

from llm_tripwire import available_types
from llm_tripwire.semantic import SemanticSimilarity

HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
needs_st = pytest.mark.skipif(not HAS_ST, reason="sentence-transformers not installed")


def test_semantic_type_registered_without_torch():
    # Registered for the YAML loader even when torch isn't installed.
    assert "semantic_similarity" in available_types()


def test_threshold_validation():
    with pytest.raises(ValueError):
        SemanticSimilarity(reference="x", threshold=1.5)


def test_missing_dependency_message(monkeypatch):
    """When the extra isn't installed, evaluation raises a helpful ImportError."""
    if HAS_ST:
        pytest.skip("dependency is installed; the missing-dep path can't be exercised")
    cond = SemanticSimilarity(reference="hello world")
    with pytest.raises(ImportError) as ei:
        cond.evaluate("hi there")
    assert "pip install" in str(ei.value)


@needs_st
def test_similar_text_passes():
    cond = SemanticSimilarity(reference="How do I reset my password?", threshold=0.5)
    r = cond.evaluate("What's the process to change my password?")
    assert r.passed
    assert 0.0 <= r.score <= 1.0


@needs_st
def test_unrelated_text_fails():
    cond = SemanticSimilarity(reference="How do I reset my password?", threshold=0.6)
    r = cond.evaluate("The weather in Patagonia is cold this time of year.")
    assert r.passed is False