"""
API Routes  (Section 16 — FastAPI Application Layer)
─────────────────────────────────────────────────────
Core /rewrite endpoint + health check.
Phase 2: Multi-pass pipeline integration with multi-style sampling.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from humanizer import __version__
from humanizer.api.schemas import (
    ConfidenceScoreResponse,
    HealthResponse,
    RewriteRequest,
    RewriteResponse,
    ValidationScoreResponse,
)
from humanizer.pipeline.multi_pass import run_document_pipeline, run_pipeline
from humanizer.postprocessing.postprocessor import compute_confidence, restore_pii
from humanizer.preprocessing.sanitizer import (
    AbuseDetectedError,
    InputTooLongError,
    UnsupportedLanguageError,
    sanitize_input,
)
from humanizer.router.model_router import route_request
from humanizer.style.style_vector import StyleVector as StyleVectorModel
from humanizer.utils import count_tokens
from humanizer.validation.safety import run_safety_checks
from humanizer.validation.semantic_checks import validate_semantic

logger = logging.getLogger(__name__)

router = APIRouter()

# Threshold for triggering long-form document pipeline
_LONG_FORM_THRESHOLD = 500  # tokens


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version=__version__)


@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite(req: RewriteRequest):
    """
    Main rewriting endpoint.

    Pipeline (Phase 2 — Multi-Pass):
      1. Input sanitisation (PII, lang, abuse)
      2. Model routing (token count × tier)
      3. Multi-pass pipeline with multi-style sampling
      4. Safety checks (toxicity, injection)
      5. Semantic validation (SBERT, NER, NLI, readability)
      6. Post-processing (PII restore, confidence)
    """
    t0 = time.perf_counter()

    # ── 1. Sanitise Input ──────────────────────────────────────────────
    try:
        sanitized = await sanitize_input(req.text)
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except AbuseDetectedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InputTooLongError as e:
        raise HTTPException(status_code=413, detail=str(e))

    # ── 2. Route to Model ──────────────────────────────────────────────
    model_name = route_request(sanitized.token_count)

    # ── 3. Multi-Pass Pipeline ─────────────────────────────────────────
    # Build style vector from request
    style_vec = None
    if req.style_vector:
        style_vec = StyleVectorModel(
            formality=req.style_vector.formality,
            complexity=req.style_vector.complexity,
            density=req.style_vector.density,
            hedging=req.style_vector.hedging,
            nominalization=req.style_vector.nominalization,
            sentence_length=req.style_vector.sentence_length,
        )

    try:
        # Use document pipeline for long texts
        if sanitized.token_count > _LONG_FORM_THRESHOLD:
            pipeline_result = await run_document_pipeline(
                sanitized.text,
                style_profile=req.style_profile,
                style_vector=style_vec,
                model=model_name,
                enable_multi_style=False,  # Single-style for long-form (latency)
            )
        else:
            pipeline_result = await run_pipeline(
                sanitized.text,
                style_profile=req.style_profile,
                style_vector=style_vec,
                model=model_name,
                pipeline_mode=req.pipeline_mode,
                enable_multi_style=(req.pipeline_mode == "full"),
            )
        rewritten = pipeline_result.text
    except Exception as e:
        logger.exception("Pipeline failed on model %s", model_name)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # ── 4. Safety Checks ──────────────────────────────────────────────
    safety = run_safety_checks(rewritten)
    if not safety.is_safe:
        logger.warning("Safety check failed: %s — regenerating", safety.message)
        # Retry with lite pipeline
        try:
            pipeline_result = await run_pipeline(
                sanitized.text,
                style_profile=req.style_profile,
                model=model_name,
                pipeline_mode="lite",
                enable_multi_style=False,
            )
            rewritten = pipeline_result.text
        except Exception:
            pass
        safety = run_safety_checks(rewritten)
        if not safety.is_safe:
            raise HTTPException(
                status_code=422,
                detail=f"Output failed safety checks after retry: {safety.message}",
            )

    # ── 5. Semantic Validation ─────────────────────────────────────────
    validation = validate_semantic(sanitized.text, rewritten)

    # ── 6. Post-Processing ─────────────────────────────────────────────
    final_text = restore_pii(rewritten, sanitized.pii_map)
    confidence = compute_confidence(validation)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Rewrite complete — model=%s profile=%s tokens=%d time=%.2fs confidence=%.3f",
        model_name, pipeline_result.style_profile_used,
        sanitized.token_count, elapsed, confidence.overall,
    )

    return RewriteResponse(
        rewritten_text=final_text,
        model_used=model_name,
        token_count=count_tokens(final_text),
        language=sanitized.language,
        confidence=ConfidenceScoreResponse(
            semantic_fidelity=confidence.semantic_fidelity,
            factual_confidence=confidence.factual_confidence,
            originality=confidence.originality,
            human_likeness=confidence.human_likeness,
            overall=confidence.overall,
        ),
        validation=ValidationScoreResponse(
            semantic_similarity=validation.semantic_similarity,
            entity_preserved=validation.entity_preserved,
            missing_entities=validation.missing_entities,
            nli_label=validation.nli_label,
            nli_entailment_score=validation.nli_entailment_score,
            lexical_novelty=validation.lexical_novelty,
            readability_delta=validation.readability_delta,
        ),
    )
