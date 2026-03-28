"""
Critic Model  (Section 6.2 + 14)
──────────────────────────────────
Scores rewrite candidates on semantic fidelity, originality, and
human-likeness. Selects the best of 6 style-diverse candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from humanizer.utils import argmax

logger = logging.getLogger(__name__)

# Lazy-loaded models
_sbert = None


def _get_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sbert


@dataclass
class CriticScore:
    """Critic evaluation of a single candidate."""

    semantic_fidelity: float    # SBERT cosine similarity with input
    originality: float          # 1 - n-gram overlap
    human_likeness: float       # Placeholder — filled by perplexity controller in Phase 3
    overall: float              # Weighted combination

    def __repr__(self) -> str:
        return (
            f"CriticScore(semantic={self.semantic_fidelity:.3f}, "
            f"orig={self.originality:.3f}, human={self.human_likeness:.3f}, "
            f"overall={self.overall:.3f})"
        )


def _compute_ngram_novelty(source: str, candidate: str, n: int = 3) -> float:
    """1 - n-gram overlap between source and candidate."""
    def ngrams(text: str) -> set[tuple[str, ...]]:
        tokens = text.lower().split()
        return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    src_ng = ngrams(source)
    cand_ng = ngrams(candidate)
    if not cand_ng:
        return 0.0
    overlap = len(src_ng & cand_ng) / len(cand_ng)
    return 1.0 - overlap


def score_candidate(
    source: str,
    candidate: str,
    *,
    weights: Optional[dict[str, float]] = None,
) -> CriticScore:
    """
    Score a single rewrite candidate (Section 6.2).

    Weights (default from architecture doc):
      semantic_fidelity: 0.40
      originality:       0.35
      human_likeness:    0.25
    """
    w = weights or {
        "semantic_fidelity": 0.40,
        "originality": 0.35,
        "human_likeness": 0.25,
    }

    # Semantic fidelity (SBERT cosine sim)
    model = _get_sbert()
    embs = model.encode([source, candidate], convert_to_numpy=True)
    semantic = float(
        np.dot(embs[0], embs[1])
        / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
    )

    # Originality (n-gram novelty)
    originality = _compute_ngram_novelty(source, candidate)

    # Human-likeness placeholder (Phase 3 adds perplexity-based scoring)
    human_likeness = 0.5

    overall = (
        w["semantic_fidelity"] * semantic
        + w["originality"] * originality
        + w["human_likeness"] * human_likeness
    )

    return CriticScore(
        semantic_fidelity=semantic,
        originality=originality,
        human_likeness=human_likeness,
        overall=overall,
    )


def score_candidates(
    source: str,
    candidates: list[str],
    **kwargs,
) -> list[CriticScore]:
    """Score multiple candidates and return all scores."""
    return [score_candidate(source, c, **kwargs) for c in candidates]


def select_best(
    source: str,
    candidates: list[str],
    **kwargs,
) -> tuple[str, CriticScore, int]:
    """
    Score all candidates and return the best one.
    Returns (best_text, best_score, best_index).
    """
    scores = score_candidates(source, candidates, **kwargs)
    best_idx = argmax([s.overall for s in scores])
    logger.info(
        "Critic selected candidate %d/%d (overall=%.3f)",
        best_idx + 1, len(candidates), scores[best_idx].overall,
    )
    return candidates[best_idx], scores[best_idx], best_idx
