"""
Multi-Pass Generation Pipeline  (Section 4 + 6 + 12)
──────────────────────────────────────────────────────
4-pass rewriting with style-aware prompts and memory drift.

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

from humanizer.config import DEFAULT_MODEL, GenerationConfig, PipelineConfig
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


# ── Style-to-prefix mapping ───────────────────────────────────────────────

def _style_to_prefix_hint(style: StyleVector) -> str:
    """
    Map a StyleVector into a natural-language style instruction fragment.
    Injected into Pass 2 and Pass 3 prompts so the model adjusts register.
    """
    hints = []

    # Formality axis
    if style.formality > 0.75:
        hints.append("Use formal, academic language with precise terminology.")
    elif style.formality > 0.5:
        hints.append("Use professional, clear language.")
    elif style.formality > 0.3:
        hints.append("Use a balanced, accessible tone.")
    else:
        hints.append("Use casual, conversational language as if talking to a friend.")

    # Complexity axis
    if style.complexity > 0.7:
        hints.append("Use sophisticated vocabulary and complex sentence structures.")
    elif style.complexity < 0.3:
        hints.append("Use simple, everyday words.")

    # Density axis
    if style.density > 0.7:
        hints.append("Be concise and dense — remove filler words.")
    elif style.density < 0.3:
        hints.append("Expand and elaborate with examples and explanations.")

    # Hedging axis
    if style.hedging > 0.5:
        hints.append("Include hedging phrases like 'it appears', 'suggests', 'may'.")
    elif style.hedging < 0.2:
        hints.append("Be direct and assertive.")

    # Sentence length axis
    if style.sentence_length > 0.7:
        hints.append("Use long, compound-complex sentences.")
    elif style.sentence_length < 0.3:
        hints.append("Use short, punchy sentences.")

    return " ".join(hints)


# ── Language guard ─────────────────────────────────────────────────────────

def _is_likely_english(text: str) -> bool:
    """Quick heuristic: check if text is mostly ASCII/English."""
    if not text or len(text) < 10:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) > 0.85


# ── Individual Passes ──────────────────────────────────────────────────────

async def _pass1_summarize(text: str, model: str = DEFAULT_MODEL) -> str:
    """
    Pass 1: Abstractive Summarisation (Section 4.1).
    Extract the core meaning — strip away the original surface form.
    Uses extractive key-point decomposition to force the model to decompose
    rather than copy.
    """
    result = await generate_text(
        model, text,
        prefix=(
            "Read the following text carefully, then list its 3-5 main claims "
            "as short bullet points using completely different words. "
            "Do not copy any phrases from the original text. "
            "Write in English:\n\n"
        ),
        max_new_tokens=256,
    )
    if not _is_likely_english(result):
        logger.warning("Pass 1 produced non-English output, falling back")
        return text
    logger.debug("Pass 1 (summarize): %d → %d chars", len(text), len(result))
    return result


async def _pass2_expand(
    propositions: str,
    style: StyleVector,
    model: str = DEFAULT_MODEL,
    *,
    drift_rate: float = 0.05,
) -> str:
    """
    Pass 2: Proposition Expansion + Memory Drift [U5] (Section 4.1).
    Rebuild from key ideas into a full paragraph using new wording.
    Chain-of-thought prompting forces fresh vocabulary.
    """
    # Apply memory drift at 5% of propositions
    sentences = [s.strip() for s in propositions.split(".") if s.strip()]
    drifted = apply_memory_drift(sentences, drift_rate=drift_rate)
    drifted_text = ". ".join(drifted) + "."

    # Style-aware expansion instruction
    style_hint = _style_to_prefix_hint(style)

    result = await generate_text(
        model, drifted_text,
        prefix=(
            "You are a university student rewriting an essay from memory. "
            "Using only these notes, write a detailed paragraph entirely in "
            "your own words. Do not reuse any phrases from the notes. "
            f"{style_hint} "
            "Write in English:\n\n"
        ),
        max_new_tokens=512,
    )
    if not _is_likely_english(result):
        logger.warning("Pass 2 produced non-English output, falling back")
        return propositions
    logger.debug("Pass 2 (expand+drift): %d → %d chars", len(propositions), len(result))
    return result


async def _pass3_diversify(text: str, style: StyleVector, model: str = DEFAULT_MODEL) -> str:
    """
    Pass 3: Syntactic Diversification (Section 4.1).
    Restructure sentences — change voice, clause order, word choice.
    Uses contrastive instruction to ensure no 3-word overlap.
    """
    style_hint = _style_to_prefix_hint(style)

    prefix = (
        "Rewrite the following text so that no 3-word phrase from the "
        "original appears in your version. Change sentence order, voice "
        "(active/passive), and vocabulary completely. "
        f"{style_hint} "
        "Write in English:\n\n"
    )

    # Override for extreme style points
    if style.formality > 0.8 and style.nominalization > 0.6:
        prefix = (
            "Rewrite this text in a formal academic style using completely "
            "different sentence structures. Use nominalization, passive voice, "
            "and technical vocabulary. No phrase from the original should appear "
            "in your version. Write in English:\n\n"
        )
    elif style.formality < 0.3:
        prefix = (
            "Explain this in your own words as if telling a friend about it "
            "over coffee. Use casual language, contractions, and short sentences. "
            "Don't copy any phrases from the original. Write in English:\n\n"
        )
    elif style.density > 0.7:
        prefix = (
            "Condense this text into fewer sentences while keeping all key "
            "information. Use completely different wording — no phrase from "
            "the original should appear. Write in English:\n\n"
        )
    elif style.density < 0.3:
        prefix = (
            "Expand and elaborate on this text with additional explanation "
            "and examples. Rewrite every sentence using new vocabulary. "
            "Write in English:\n\n"
        )

    result = await generate_text(model, text, prefix=prefix, max_new_tokens=512)
    if not _is_likely_english(result):
        logger.warning("Pass 3 produced non-English output, falling back")
        return text
    logger.debug("Pass 3 (diversify): %d → %d chars", len(text), len(result))
    return result


async def _pass4_polish(
    text: str,
    style: StyleVector,
    model: str = DEFAULT_MODEL,
    *,
    drift_rate: float = 0.08,
) -> str:
    """
    Pass 4: Coherence & Register Polish + Memory Drift [U5] (Section 4.1).
    Final pass — improve flow, fix awkward phrasing, ensure consistency.
    """
    # Apply back-reference drift at 8%
    drifted = apply_reference_drift(text, drift_rate=drift_rate)

    result = await generate_text(
        model, drifted,
        prefix=(
            "Improve the coherence, grammar, and flow of the following text. "
            "Fix any awkward phrasing and ensure smooth transitions between ideas. "
            "Keep the same meaning but make it read naturally. "
            "Write in English:\n\n"
        ),
        max_new_tokens=512,
    )
    if not _is_likely_english(result):
        logger.warning("Pass 4 produced non-English output, falling back")
        return text
    logger.debug("Pass 4 (polish+drift): %d → %d chars", len(text), len(result))
    return result


# ── Single-Style Pipeline ─────────────────────────────────────────────────

async def _run_single_pipeline(
    text: str,
    style: StyleVector,
    model: str = DEFAULT_MODEL,
    *,
    config: Optional[PipelineConfig] = None,
) -> str:
    """Run the 4-pass pipeline for a single style profile."""
    cfg = config or PipelineConfig()

    # Pass 1: Collapse to propositions
    summarized = await _pass1_summarize(text, model)
    logger.info("Pass 1 done: %d → %d chars", len(text), len(summarized))

    # Pass 2: Expand with drift
    expanded = await _pass2_expand(
        summarized, style, model,
        drift_rate=cfg.drift_rate_pass2,
    )
    logger.info("Pass 2 done: %d → %d chars", len(summarized), len(expanded))

    # Pass 3: Syntactic diversification
    diversified = await _pass3_diversify(expanded, style, model)
    logger.info("Pass 3 done: %d → %d chars", len(expanded), len(diversified))

    # Pass 4: Polish with drift
    polished = await _pass4_polish(
        diversified, style, model,
        drift_rate=cfg.drift_rate_pass4,
    )
    logger.info("Pass 4 done: %d → %d chars", len(diversified), len(polished))

    return polished


# ── Multi-Style Sampling Pipeline ─────────────────────────────────────────

async def run_pipeline(
    text: str,
    *,
    style_profile: Optional[str] = None,
    style_vector: Optional[StyleVector] = None,
    model: str = DEFAULT_MODEL,
    config: Optional[PipelineConfig] = None,
    pipeline_mode: str = "full",
    enable_multi_style: bool = True,
) -> PipelineResult:
    """
    Full multi-pass rewriting pipeline with multi-style sampling (Section 4 + 6).

    ALWAYS uses the 4-pass pipeline (summarize → expand → diversify → polish).

    In 'full' mode with multi_style=True:
      - Generate one candidate per style profile (6 total)
      - Critic selects the best candidate

    In 'lite' mode or with multi_style=False:
      - Run single 4-pass pipeline with the specified style

    Returns a PipelineResult with the best rewrite, score, and metadata.
    """
    t0 = time.perf_counter()
    cfg = config or PipelineConfig()
    mode = get_mode(pipeline_mode)

    if mode.deterministic:
        set_deterministic_seed()

    # ── Single-style 4-pass path ──────────────────────────────────────
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

    if style_profile and style_profile in profiles:
        logger.info("Multi-style sampling with bias toward '%s'", style_profile)

    # Generate all 6 candidates via 4-pass pipeline
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
    model: str = DEFAULT_MODEL,
    config: Optional[PipelineConfig] = None,
    enable_multi_style: bool = False,
) -> PipelineResult:
    """
    Long-form document rewriting pipeline (Section 12).

    1. Semantic chunking
    2. Per-chunk 4-pass rewrite with cross-chunk context
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
