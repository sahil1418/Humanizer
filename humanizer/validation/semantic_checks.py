"""
Semantic Validation Checks  (Section 11)
──────────────────────────────────────────
SBERT cosine similarity · spaCy NER entity preservation · DeBERTa NLI ·
Readability (Flesch-Kincaid) · Lexical novelty (n-gram overlap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import textstat

logger = logging.getLogger(__name__)

# Lazy-loaded heavy models
_sbert_model = None
_nli_model = None
_spacy_nlp = None


def _get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loaded SBERT model")
    return _sbert_model


def _get_nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        logger.info("Loaded NLI model")
    return _nli_model


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found — run: python -m spacy download en_core_web_sm")
            _spacy_nlp = None
    return _spacy_nlp


# ── Result dataclass ───────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    """Aggregated validation scores for a single (input, output) pair."""

    semantic_similarity: float = 0.0
    entity_preserved: bool = True
    missing_entities: list[str] = None
    nli_label: str = "entailment"        # entailment | contradiction | neutral
    nli_entailment_score: float = 1.0
    lexical_novelty: float = 0.0
    readability_input: float = 0.0
    readability_output: float = 0.0
    readability_delta: float = 0.0

    def __post_init__(self):
        if self.missing_entities is None:
            self.missing_entities = []

    @property
    def passed(self) -> bool:
        """True if all safety-critical checks pass (ignores adaptive thresholds)."""
        return (
            self.entity_preserved
            and self.nli_label != "contradiction"
        )


# ── Individual checks ─────────────────────────────────────────────────────
def cosine_similarity(text_a: str, text_b: str) -> float:
    """SBERT cosine similarity between two texts."""
    model = _get_sbert()
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
    )
    return float(cos_sim)


def check_entity_preservation(input_text: str, output_text: str) -> tuple[bool, list[str]]:
    """
    Verify that all named entities from the input appear in the output.
    Returns (all_preserved, list_of_missing_entities).
    """
    nlp = _get_spacy()
    if nlp is None:
        return True, []  # Graceful fallback

    input_ents = {ent.text.lower() for ent in nlp(input_text).ents}
    output_ents = {ent.text.lower() for ent in nlp(output_text).ents}

    missing = input_ents - output_ents
    return len(missing) == 0, list(missing)


def check_nli_entailment(premise: str, hypothesis: str) -> tuple[str, float]:
    """
    Run NLI (DeBERTa) to check factual consistency.
    Returns (label, entailment_probability).
    Label is one of: 'contradiction', 'entailment', 'neutral'.
    """
    model = _get_nli()
    # CrossEncoder returns scores for [contradiction, entailment, neutral]
    scores = model.predict([(premise, hypothesis)])[0]
    labels = ["contradiction", "entailment", "neutral"]
    idx = int(np.argmax(scores))
    return labels[idx], float(scores[1])  # entailment score


def compute_lexical_novelty(input_text: str, output_text: str, n: int = 3) -> float:
    """
    1 − n-gram overlap between input and output.
    Higher = more novel (more surface-level originality).
    """
    def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        tokens = text.lower().split()
        return {tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}

    input_ng = _ngrams(input_text, n)
    output_ng = _ngrams(output_text, n)

    if not output_ng:
        return 0.0

    overlap = len(input_ng & output_ng) / len(output_ng)
    return 1.0 - overlap


def compute_readability(text: str) -> float:
    """Flesch-Kincaid grade level."""
    return textstat.flesch_kincaid_grade(text)


# ── Orchestrator ───────────────────────────────────────────────────────────
def validate_semantic(input_text: str, output_text: str) -> ValidationResult:
    """Run all semantic validation checks and return aggregated result."""
    # Semantic similarity
    sim = cosine_similarity(input_text, output_text)

    # Entity preservation
    ent_ok, missing = check_entity_preservation(input_text, output_text)

    # NLI
    nli_label, nli_score = check_nli_entailment(input_text, output_text)

    # Lexical novelty
    novelty = compute_lexical_novelty(input_text, output_text)

    # Readability
    read_in = compute_readability(input_text)
    read_out = compute_readability(output_text)

    return ValidationResult(
        semantic_similarity=sim,
        entity_preserved=ent_ok,
        missing_entities=missing,
        nli_label=nli_label,
        nli_entailment_score=nli_score,
        lexical_novelty=novelty,
        readability_input=read_in,
        readability_output=read_out,
        readability_delta=abs(read_out - read_in),
    )
