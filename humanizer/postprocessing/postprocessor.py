"""
Post-Processing Layer  (Sections 2 + 20.3)
───────────────────────────────────────────
PII re-injection · Confidence scoring · Final cleanup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from humanizer.preprocessing.sanitizer import PIIMap
from humanizer.validation.semantic_checks import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Weighted confidence score (Section 20.3)."""

    semantic_fidelity: float = 0.0
    factual_confidence: float = 0.0
    originality: float = 0.0
    human_likeness: float = 0.0  # Placeholder — populated in Phase 3 with perplexity
    overall: float = 0.0


def compute_confidence(validation: ValidationResult) -> ConfidenceScore:
    """
    Compute weighted confidence from validation scores.

    Weights (architecture doc):
      semantic fidelity  0.35
      factual confidence 0.30
      originality        0.20
      human likeness     0.15  (perplexity — added in Phase 3)
    """
    semantic = validation.semantic_similarity
    factual = validation.nli_entailment_score
    originality = validation.lexical_novelty
    human_likeness = 0.5  # Neutral default until perplexity controller is added

    overall = (
        0.35 * semantic
        + 0.30 * factual
        + 0.20 * originality
        + 0.15 * human_likeness
    )

    return ConfidenceScore(
        semantic_fidelity=round(semantic, 4),
        factual_confidence=round(factual, 4),
        originality=round(originality, 4),
        human_likeness=round(human_likeness, 4),
        overall=round(overall, 4),
    )


def restore_pii(text: str, pii_map: PIIMap) -> str:
    """Replace PII placeholders with original values."""
    return pii_map.restore(text)


def postprocess(
    rewritten_text: str,
    *,
    pii_map: PIIMap | None = None,
    validation: ValidationResult | None = None,
) -> dict:
    """
    Full post-processing pipeline:
      1. PII re-injection
      2. Confidence scoring
      3. Return structured result
    """
    # 1. PII restoration
    final_text = rewritten_text
    if pii_map:
        final_text = restore_pii(final_text, pii_map)

    # 2. Confidence
    confidence = None
    if validation:
        confidence = compute_confidence(validation)

    return {
        "text": final_text,
        "confidence": confidence,
    }
