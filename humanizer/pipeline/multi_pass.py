"""
Multi-Pass Generation Pipeline  (Section 4 + 6 + 12)
──────────────────────────────────────────────────────
4-pass rewriting with multi-style sampling and memory drift.

Pipeline flow:
  Pass 1: Abstractive Summarisation — collapse to semantic propositions
  Pass 2: Proposition Expansion + Memory Drift [U5]
  Pass 3: Syntactic Diversification
  Pass 4: Coherence & Register Polish + Memory Drift [U5]

Multi-Style Sampling [U3]:
  Generate one candidate per style profile (6 total), then critic selects best.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from humanizer.config import GenerationConfig, PipelineConfig
from humanizer.critic.critic_model import CriticScore, select_best
from humanizer.inference.inference_layer import generate_text
from humanizer.pipeline.memory_drift import (
    apply_memory_drift,
    apply_reference_drift,
    compress_context_summary,
)
from humanizer.pipeline.pipeline_modes import (
    PipelineModeConfig,
    get_mode,
    set_deterministic_seed,
    should_fallback_to_lite,
)
from humanizer.preprocessing.chunker import TextChunk, coherence_stitch, semantic_chunk
from humanizer.style.style_profiles import get_all_profiles, get_profile
from humanizer.style.style_vector import StyleVector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the full multi-pass pipeline."""

    text: str
    style_profile_used: str
    critic_score: Optional[CriticScore] = None
    passes_completed: int = 4
    candidates_generated: int = 1
    elapsed_ms: float = 0.0
    pipeline_mode: str = "full"


# ── Individual Passes ──────────────────────────────────────────────────────

async def _pass1_summarize(text: str, model: str = "t5-base") -> str:
    """
    Pass 1: Abstractive Summarisation (Section 4.1).
    Collapse to semantic propositions — remove all surface form.
    """
    result = await generate_text(
        model, text,
        prefix="summarize: ",
        max_new_tokens=256,
    )
    logger.debug("Pass 1 (summarize): %d → %d chars", len(text), len(result))
    return result


async def _pass2_expand(
    propositions: str,
    style: StyleVector,
    model: str = "t5-base",
    *,
    drift_rate: float = 0.05,
) -> str:
    """
    Pass 2: Proposition Expansion + Memory Drift [U5] (Section 4.1).
    Rebuild from atomic facts — never from original wording.
    """
    # Apply memory drift at 5% of propositions
    sentences = [s.strip() for s in propositions.split(".") if s.strip()]
    drifted = apply_memory_drift(sentences, drift_rate=drift_rate)
    drifted_text = ". ".join(drifted) + "."

    # Expand with style conditioning
    prompt = f"expand and elaborate: {drifted_text}"
    result = await generate_text(model, prompt, prefix="", max_new_tokens=512)
    logger.debug("Pass 2 (expand+drift): %d → %d chars", len(propositions), len(result))
    return result


async def _pass3_diversify(text: str, style: StyleVector, model: str = "t5-base") -> str:
    """
    Pass 3: Syntactic Diversification (Section 4.1).
    Active↔passive, nominalization, clause reordering, lexical swap.
    """
    prefix = "paraphrase: "
    if style.nominalization > 0.6:
        prefix = "paraphrase with formal structure: "
    elif style.formality < 0.3:
        prefix = "rephrase casually: "

    result = await generate_text(model, text, prefix=prefix, max_new_tokens=512)
    logger.debug("Pass 3 (diversify): %d → %d chars", len(text), len(result))
    return result


async def _pass4_polish(
    text: str,
    style: StyleVector,
    model: str = "t5-base",
    *,
    drift_rate: float = 0.08,
) -> str:
    """
    Pass 4: Coherence & Register Polish + Memory Drift [U5] (Section 4.1).
    Add discourse connectives, align tone, ensure global consistency.
    """
    # Apply back-reference drift at 8%
    drifted = apply_reference_drift(text, drift_rate=drift_rate)

    prompt = f"improve coherence and polish: {drifted}"
    result = await generate_text(model, prompt, prefix="", max_new_tokens=512)
    logger.debug("Pass 4 (polish+drift): %d → %d chars", len(text), len(result))
    return result


# ── Single-Style Pipeline ─────────────────────────────────────────────────

async def _run_single_pipeline(
    text: str,
    style: StyleVector,
    model: str = "t5-base",
    *,
    config: Optional[PipelineConfig] = None,
) -> str:
    """Run the 4-pass pipeline for a single style profile."""
    cfg = config or PipelineConfig()

    # Pass 1: Collapse to propositions
    summarized = await _pass1_summarize(text, model)

    # Pass 2: Expand with drift
    expanded = await _pass2_expand(
        summarized, style, model,
        drift_rate=cfg.drift_rate_pass2,
    )

    # Pass 3: Syntactic diversification
    diversified = await _pass3_diversify(expanded, style, model)

    # Pass 4: Polish with drift
    polished = await _pass4_polish(
        diversified, style, model,
        drift_rate=cfg.drift_rate_pass4,
    )

    return polished


# ── Multi-Style Sampling Pipeline ─────────────────────────────────────────

async def run_pipeline(
    text: str,
    *,
    style_profile: Optional[str] = None,
    style_vector: Optional[StyleVector] = None,
    model: str = "t5-base",
    config: Optional[PipelineConfig] = None,
    pipeline_mode: str = "full",
    enable_multi_style: bool = True,
) -> PipelineResult:
    """
    Full multi-pass rewriting pipeline with multi-style sampling (Section 4 + 6).

    In 'full' mode with multi_style=True:
      - Generate one candidate per style profile (6 total)
      - Critic selects the best candidate

    In 'lite' mode or with multi_style=False:
      - Run single pipeline with the specified style

    Returns a PipelineResult with the best rewrite, score, and metadata.
    """
    t0 = time.perf_counter()
    cfg = config or PipelineConfig()
    mode = get_mode(pipeline_mode)

    if mode.deterministic:
        set_deterministic_seed()

    # ── Single-style path ──────────────────────────────────────────────
    if not enable_multi_style or pipeline_mode == "lite":
        sv = style_vector or (get_profile(style_profile) if style_profile else StyleVector())
        profile_name = style_profile or "custom"

        result_text = await _run_single_pipeline(text, sv, model, config=cfg)
        elapsed = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            text=result_text,
            style_profile_used=profile_name,
            passes_completed=4,
            candidates_generated=1,
            elapsed_ms=elapsed,
            pipeline_mode=pipeline_mode,
        )

    # ── Multi-style sampling path (U3) ─────────────────────────────────
    profiles = get_all_profiles()

    # If a specific style_profile is requested, still generate all 6 but
    # bias toward the requested one
    if style_profile and style_profile in profiles:
        logger.info("Multi-style sampling with bias toward '%s'", style_profile)

    # Generate all 6 candidates (could be parallelised with asyncio.gather
    # once models are loaded on separate workers — for now sequential)
    candidates: list[str] = []
    profile_names: list[str] = []

    for name, sv in profiles.items():
        try:
            candidate = await _run_single_pipeline(text, sv, model, config=cfg)
            candidates.append(candidate)
            profile_names.append(name)
        except Exception as e:
            logger.warning("Pipeline failed for profile '%s': %s", name, e)
            continue

    if not candidates:
        raise RuntimeError("All style profile pipelines failed — no candidates generated")

    # ── Critic selects best ────────────────────────────────────────────
    best_text, best_score, best_idx = select_best(text, candidates)
    best_profile = profile_names[best_idx]

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Multi-style pipeline: %d candidates, best='%s' (%.3f) in %.0fms",
        len(candidates), best_profile, best_score.overall, elapsed,
    )

    return PipelineResult(
        text=best_text,
        style_profile_used=best_profile,
        critic_score=best_score,
        passes_completed=4,
        candidates_generated=len(candidates),
        elapsed_ms=elapsed,
        pipeline_mode=pipeline_mode,
    )


# ── Long-Form Document Pipeline ───────────────────────────────────────────

async def run_document_pipeline(
    text: str,
    *,
    style_profile: Optional[str] = None,
    style_vector: Optional[StyleVector] = None,
    model: str = "t5-base",
    config: Optional[PipelineConfig] = None,
    enable_multi_style: bool = False,
) -> PipelineResult:
    """
    Long-form document rewriting pipeline (Section 12).

    1. Semantic chunking
    2. Per-chunk rewrite with cross-chunk context
    3. Coherence stitching pass
    """
    t0 = time.perf_counter()
    cfg = config or PipelineConfig()
    sv = style_vector or (get_profile(style_profile) if style_profile else StyleVector())

    # 1. Chunk the document
    chunks = semantic_chunk(text)

    if len(chunks) == 1:
        # Short document — use regular pipeline
        return await run_pipeline(
            text, style_profile=style_profile, style_vector=style_vector,
            model=model, config=config, enable_multi_style=enable_multi_style,
        )

    # 2. Per-chunk rewrite with cross-chunk context
    rewritten_chunks: list[str] = []
    prev_summary = ""

    for chunk in chunks:
        # Prepend context summary from previous chunk
        context_text = chunk.text
        if prev_summary:
            context_text = f"[Previous context: {prev_summary}]\n\n{chunk.text}"

        result = await _run_single_pipeline(context_text, sv, model, config=cfg)
        rewritten_chunks.append(result)

        # Compress context for next chunk (v2.0: 60 tokens → partial recall)
        prev_summary = compress_context_summary(
            result, max_tokens=cfg.cross_chunk_summary_tokens
        )

    # 3. Coherence stitching
    stitched = coherence_stitch(rewritten_chunks)
    elapsed = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        text=stitched,
        style_profile_used=style_profile or "mixed_tone",
        passes_completed=4,
        candidates_generated=len(chunks),
        elapsed_ms=elapsed,
        pipeline_mode="full",
    )
