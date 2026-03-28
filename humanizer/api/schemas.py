"""
Pydantic schemas for API request / response models.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────
class StyleVector(BaseModel):
    """Optional per-request style overrides."""

    formality: float = Field(0.55, ge=0.0, le=1.0)
    complexity: float = Field(0.45, ge=0.0, le=1.0)
    density: float = Field(0.50, ge=0.0, le=1.0)
    hedging: float = Field(0.30, ge=0.0, le=1.0)
    nominalization: float = Field(0.45, ge=0.0, le=1.0)
    sentence_length: float = Field(0.50, ge=0.0, le=1.0)


class RewriteRequest(BaseModel):
    """POST /rewrite request body."""

    text: str = Field(..., min_length=1, max_length=50000, description="Input text to rewrite")
    quality_tier: Literal["standard", "premium"] = Field("standard")
    style_profile: Optional[str] = Field(
        None,
        description="Named style profile: formal_academic, semi_formal, conversational, "
                    "mixed_tone, compressed, expanded"
    )
    style_vector: Optional[StyleVector] = Field(
        None,
        description="Custom style vector — overrides style_profile if provided"
    )
    pipeline_mode: Literal["full", "lite", "deterministic"] = Field("full")


# ── Response ───────────────────────────────────────────────────────────────
class ConfidenceScoreResponse(BaseModel):
    semantic_fidelity: float
    factual_confidence: float
    originality: float
    human_likeness: float
    overall: float


class ValidationScoreResponse(BaseModel):
    semantic_similarity: float
    entity_preserved: bool
    missing_entities: list[str]
    nli_label: str
    nli_entailment_score: float
    lexical_novelty: float
    readability_delta: float


class RewriteResponse(BaseModel):
    """POST /rewrite response body."""

    rewritten_text: str
    model_used: str
    token_count: int
    confidence: Optional[ConfidenceScoreResponse] = None
    validation: Optional[ValidationScoreResponse] = None
    language: str = "en"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
