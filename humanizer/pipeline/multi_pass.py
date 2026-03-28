"""
Multi-Pass Generation Pipeline  (Section 4 + 6 + 12)
──────────────────────────────────────────────────────
4-pass rewriting with sentence-level processing.

Key insight: FLAN-T5-XL handles SHORT inputs (1-2 sentences) with SIMPLE
prefixes far better than long paragraphs with complex instructions.

Pipeline flow:
  Pass 1: Sentence-level paraphrase (short prefix per sentence)
  Pass 2: Structural reorganisation — merge/split/reorder sentences
  Pass 3: Style-aware vocabulary pass (sentence-level)
  Pass 4: Coherence & register polish (full paragraph)

Multi-Style Sampling [U3]:
  Generate one candidate per style profile (6 total), then critic selects best.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
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


# ── Sentence utilities ─────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def _is_likely_english(text: str) -> bool:
    """Quick heuristic: check if text is mostly ASCII/English."""
    if not text or len(text) < 5:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) > 0.85


# ── Style-to-prefix mapping ───────────────────────────────────────────────

def _style_prefix_for_pass3(style: StyleVector) -> str:
    """
    Return a SHORT T5-compatible prefix for Pass 3 based on style.
    FLAN-T5 works best with brief, direct prefixes — not paragraphs.
    """
    if style.formality > 0.75 and style.nominalization > 0.6:
        return "Rewrite this formally and academically: "
    elif style.formality > 0.75:
        return "Rewrite this in formal language: "
    elif style.formality < 0.3:
        return "Rewrite this casually as if talking to a friend: "
    elif style.density > 0.7:
        return "Rewrite this more concisely: "
    elif style.density < 0.3:
        return "Rewrite this with more detail and explanation: "
    elif style.complexity > 0.7:
        return "Rewrite this using sophisticated vocabulary: "
    elif style.complexity < 0.3:
        return "Rewrite this using simple everyday words: "
    elif style.sentence_length > 0.7:
        return "Rewrite this using longer, more complex sentences: "
    elif style.sentence_length < 0.3:
        return "Rewrite this using short, punchy sentences: "
    else:
        return "Paraphrase this: "


# ── Individual Passes ──────────────────────────────────────────────────────

async def _pass1_sentence_paraphrase(
    text: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Pass 1: Sentence-level paraphrase.

    Split into individual sentences and paraphrase each one separately.
    FLAN-T5-XL produces much better rewrites on short inputs with simple
    prefixes like "paraphrase:" than on full paragraphs.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text

    rewritten = []
    for sent in sentences:
        # Use multiple short prefix variants to add diversity
        prefix = random.choice([
            "paraphrase: ",
            "Rephrase this sentence: ",
            "Rewrite in different words: ",
        ])
        result = await generate_text(
            model, sent,
            prefix=prefix,
            max_new_tokens=max(len(sent.split()) * 3, 60),
        )
        # Guard: reject if output is non-English, empty, or too short
        if not _is_likely_english(result) or len(result.split()) < 3:
            rewritten.append(sent)  # keep original sentence
        else:
            rewritten.append(result.strip())

    result_text = " ".join(rewritten)
    logger.debug("Pass 1 (sentence paraphrase): %d → %d chars, %d sentences",
                 len(text), len(result_text), len(sentences))
    return result_text


async def _pass2_restructure(
    text: str,
    style: StyleVector,
    model: str = DEFAULT_MODEL,
    *,
    drift_rate: float = 0.05,
) -> str:
    """
    Pass 2: Structural reorganisation + memory drift.

    - Reorder sentences (move 1-2 sentences to different positions)
    - Merge short consecutive sentences into compound ones
    - Apply memory drift to simulate human recall imperfections
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        return text

    # Apply memory drift (5% of sentences get light paraphrasing)
    sentences = apply_memory_drift(sentences, drift_rate=drift_rate)

    # Structural operations:
    result_sentences = list(sentences)

    # 1. Sentence reordering — swap 1-2 adjacent pairs
    #    (humans don't recall information in exact original order)
    n_swaps = max(1, len(result_sentences) // 4)
    for _ in range(n_swaps):
        if len(result_sentences) >= 2:
            idx = random.randint(0, len(result_sentences) - 2)
            result_sentences[idx], result_sentences[idx + 1] = (
                result_sentences[idx + 1], result_sentences[idx]
            )

    # 2. Merge short consecutive sentences (if both < 12 words)
    merged = []
    i = 0
    while i < len(result_sentences):
        if (i + 1 < len(result_sentences)
                and len(result_sentences[i].split()) < 12
                and len(result_sentences[i + 1].split()) < 12
                and random.random() < 0.4):
            # Merge with a connector
            connectors = [", and ", ", while ", "; moreover, ", " — in fact, ", ", which means "]
            connector = random.choice(connectors)
            s1 = result_sentences[i].rstrip(".")
            s2 = result_sentences[i + 1]
            s2 = s2[0].lower() + s2[1:] if s2 else s2
            merged.append(f"{s1}{connector}{s2}")
            i += 2
        else:
            merged.append(result_sentences[i])
            i += 1

    result_text = " ".join(merged)
    logger.debug("Pass 2 (restructure+drift): %d → %d sentences",
                 len(sentences), len(merged))
    return result_text


async def _pass3_style_rewrite(
    text: str,
    style: StyleVector,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Pass 3: Style-aware sentence-level rewrite.

    Each sentence gets rewritten with a style-specific SHORT prefix.
    This pass changes vocabulary and register to match the target style.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text

    prefix = _style_prefix_for_pass3(style)

    rewritten = []
    for sent in sentences:
        result = await generate_text(
            model, sent,
            prefix=prefix,
            max_new_tokens=max(len(sent.split()) * 3, 60),
        )
        if not _is_likely_english(result) or len(result.split()) < 3:
            rewritten.append(sent)
        else:
            rewritten.append(result.strip())

    result_text = " ".join(rewritten)
    logger.debug("Pass 3 (style rewrite): %d → %d chars, style=%s",
                 len(text), len(result_text), prefix[:30])
    return result_text


async def _pass4_polish(
    text: str,
    style: StyleVector,
    model: str = DEFAULT_MODEL,
    *,
    drift_rate: float = 0.08,
) -> str:
    """
    Pass 4: Coherence & Register Polish + back-reference drift.

    Full-paragraph pass to smooth transitions and fix awkward phrasing
    from the sentence-level operations in Passes 1-3.

    Uses SHORT prefix — FLAN-T5-XL responds better to brief instructions.
    """
    # Apply back-reference drift at 8%
    drifted = apply_reference_drift(text, drift_rate=drift_rate)

    result = await generate_text(
        model, drifted,
        prefix="Improve the grammar and flow of this paragraph: ",
        max_new_tokens=max(len(drifted.split()) * 2, 200),
    )

    # Guard: reject if output lost too much content
    if not _is_likely_english(result):
        logger.warning("Pass 4 produced non-English output, falling back")
        return text

    result_words = len(result.split())
    input_words = len(text.split())

    if result_words < input_words * 0.5:
        # Model truncated — output is less than 50% of input length
        logger.warning("Pass 4 truncated (%d → %d words), falling back",
                       input_words, result_words)
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
    input_word_count = len(text.split())

    # Pass 1: Sentence-level paraphrase
    paraphrased = await _pass1_sentence_paraphrase(text, model)
    logger.info("Pass 1 done: %d → %d words", input_word_count, len(paraphrased.split()))

    # Pass 2: Structural reorganisation + drift
    restructured = await _pass2_restructure(
        paraphrased, style, model,
        drift_rate=cfg.drift_rate_pass2,
    )
    logger.info("Pass 2 done: %d → %d words",
                len(paraphrased.split()), len(restructured.split()))

    # Pass 3: Style-aware sentence rewrite
    styled = await _pass3_style_rewrite(restructured, style, model)
    logger.info("Pass 3 done: %d → %d words",
                len(restructured.split()), len(styled.split()))

    # Pass 4: Coherence polish + drift
    polished = await _pass4_polish(
        styled, style, model,
        drift_rate=cfg.drift_rate_pass4,
    )
    logger.info("Pass 4 done: %d → %d words",
                len(styled.split()), len(polished.split()))

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

    ALWAYS uses the 4-pass pipeline (paraphrase → restructure → style → polish).

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
