"""
Memory Drift Simulation  [Upgrade 5]  (Section 10)
────────────────────────────────────────────────────
Simulates human memory imperfections during multi-pass rewriting.
Humans rewriting from memory produce slight paraphrase inconsistencies.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded SBERT for similarity checking
_sbert = None


def _get_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sbert


def _cosine_sim(a: str, b: str) -> float:
    """Quick cosine similarity between two texts."""
    import numpy as np
    model = _get_sbert()
    embs = model.encode([a, b], convert_to_numpy=True)
    return float(
        np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
    )


# ── Proposition-level drift (Pass 2) ──────────────────────────────────────

def drift_paraphrase(proposition: str, *, min_similarity: float = 0.78) -> str:
    """
    Generate a lightly drifted paraphrase of a single proposition.

    In production this calls the inference layer for a quick rephrase.
    Phase 2 uses a rule-based approach as a placeholder until the full
    model pipeline is wired up.
    """
    # Simple rule-based drift strategies
    strategies = [
        _swap_clause_order,
        _add_hedge,
        _simplify_wording,
        _change_connector,
    ]
    drifted = random.choice(strategies)(proposition)

    # Similarity guard — reject if drifted too far
    sim = _cosine_sim(proposition, drifted)
    if sim < min_similarity:
        logger.debug("Drift rejected (sim=%.3f < %.3f), keeping original", sim, min_similarity)
        return proposition
    return drifted


def apply_memory_drift(
    propositions: list[str],
    *,
    drift_rate: float = 0.05,
    min_similarity: float = 0.78,
) -> list[str]:
    """
    Apply memory drift to a list of propositions (Section 10).

    At *drift_rate* probability, each proposition is replaced with a
    lightly drifted paraphrase that maintains semantic similarity ≥ *min_similarity*.
    """
    result = []
    drifted_count = 0
    for prop in propositions:
        if random.random() < drift_rate:
            drifted = drift_paraphrase(prop, min_similarity=min_similarity)
            result.append(drifted)
            if drifted != prop:
                drifted_count += 1
        else:
            result.append(prop)
    if drifted_count:
        logger.info("Memory drift: %d/%d propositions drifted", drifted_count, len(propositions))
    return result


def apply_reference_drift(
    text: str,
    *,
    drift_rate: float = 0.08,
) -> str:
    """
    Apply back-reference drift (Pass 4, Section 10).

    8% of sentences have references rephrased with different surface form
    while preserving entity names exactly.
    """
    sentences = _split_sentences(text)
    result = []
    for sent in sentences:
        if random.random() < drift_rate:
            result.append(_rephrase_surface(sent))
        else:
            result.append(sent)
    return " ".join(result)


def compress_context_summary(text: str, *, max_tokens: int = 60) -> str:
    """
    Compress cross-chunk context summary to simulate partial recall.
    v2.0 reduces from 80 → 60 tokens (Section 10, long-form).
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text
    # Keep first and last portions for context continuity
    keep_start = max_tokens * 2 // 3
    keep_end = max_tokens - keep_start
    compressed = words[:keep_start] + ["..."] + words[-keep_end:]
    return " ".join(compressed)


# ── Internal helpers ───────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _swap_clause_order(text: str) -> str:
    """Swap clauses around a comma if present."""
    if "," in text:
        parts = text.split(",", 1)
        if len(parts) == 2 and len(parts[1].strip()) > 10:
            return f"{parts[1].strip()}, {parts[0].strip().lower()}"
    return text


def _add_hedge(text: str) -> str:
    """Add a hedging phrase."""
    hedges = [
        "In some sense, ", "Generally speaking, ", "It could be said that ",
        "To a degree, ", "Broadly, ",
    ]
    return random.choice(hedges) + text[0].lower() + text[1:]


def _simplify_wording(text: str) -> str:
    """Replace some formal words with simpler alternatives."""
    replacements = {
        "utilize": "use", "implement": "apply", "demonstrate": "show",
        "significant": "notable", "consequently": "so", "therefore": "thus",
        "approximately": "about", "facilitate": "help", "obtain": "get",
    }
    result = text
    for formal, simple in replacements.items():
        if formal in result.lower():
            result = result.replace(formal, simple).replace(formal.capitalize(), simple.capitalize())
            break  # One substitution per drift
    return result


def _change_connector(text: str) -> str:
    """Swap discourse connectors."""
    swaps = {
        "However": "That said", "Moreover": "Also", "Furthermore": "In addition",
        "Therefore": "As a result", "Nevertheless": "Still", "Although": "Even though",
        "In contrast": "On the other hand", "Similarly": "Likewise",
    }
    for original, replacement in swaps.items():
        if original in text:
            return text.replace(original, replacement, 1)
    return text


def _rephrase_surface(sentence: str) -> str:
    """Lightly rephrase a sentence's surface form (for back-reference drift)."""
    # Simple strategy: swap active/passive voice markers
    transforms = [
        _simplify_wording,
        _change_connector,
        lambda s: s.replace(" is ", " remains ").replace(" are ", " remain ") if random.random() < 0.5 else s,
    ]
    return random.choice(transforms)(sentence)
