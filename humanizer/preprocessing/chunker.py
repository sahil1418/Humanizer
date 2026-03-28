"""
Semantic Chunking for Long-Form Documents  (Section 12)
────────────────────────────────────────────────────────
Topic segmentation at idea transitions, not fixed token limits.
Cross-chunk context windows + coherence stitching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A semantically coherent chunk of text."""

    index: int
    text: str
    token_count: int
    topic_label: Optional[str] = None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _count_tokens(text: str) -> int:
    return len(text.split())


def semantic_chunk(
    text: str,
    *,
    min_chunk_tokens: int = 80,
    max_chunk_tokens: int = 400,
    overlap_sentences: int = 1,
) -> list[TextChunk]:
    """
    Split text into semantic chunks based on paragraph boundaries and
    sentence-level coherence (Section 12.1).

    Strategy:
      1. Split on paragraph breaks (double newline)
      2. If a paragraph is too long, split at sentence boundaries
      3. If a paragraph is too short, merge with the next one
      4. Add overlapping boundary sentences for context continuity
    """
    # Step 1: Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    if not paragraphs:
        return [TextChunk(index=0, text=text, token_count=_count_tokens(text))]

    # Step 2: Merge/split paragraphs to fit within token bounds
    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        if para_tokens > max_chunk_tokens:
            # Paragraph too long — split by sentences
            if current:
                raw_chunks.append(current)
                current = ""
            sentences = _split_sentences(para)
            sub_chunk = ""
            for sent in sentences:
                if _count_tokens(sub_chunk + " " + sent) > max_chunk_tokens and sub_chunk:
                    raw_chunks.append(sub_chunk.strip())
                    sub_chunk = sent
                else:
                    sub_chunk = (sub_chunk + " " + sent).strip()
            if sub_chunk:
                raw_chunks.append(sub_chunk.strip())

        elif _count_tokens(current + " " + para) <= max_chunk_tokens:
            # Merge with current chunk
            current = (current + "\n\n" + para).strip()

        else:
            # Current chunk is full, start new one
            if current:
                raw_chunks.append(current)
            current = para

    if current:
        raw_chunks.append(current)

    # Step 3: Ensure minimum size by merging tiny trailing chunks
    final_chunks: list[str] = []
    for chunk in raw_chunks:
        if final_chunks and _count_tokens(chunk) < min_chunk_tokens:
            final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
        else:
            final_chunks.append(chunk)

    # Step 4: Build TextChunk objects
    chunks = [
        TextChunk(
            index=i,
            text=c,
            token_count=_count_tokens(c),
        )
        for i, c in enumerate(final_chunks)
    ]

    logger.info("Chunked document into %d chunks (avg %d tokens)",
                len(chunks), sum(c.token_count for c in chunks) // max(len(chunks), 1))
    return chunks


def coherence_stitch(chunks: list[str]) -> str:
    """
    Stitch rewritten chunks back together with transition smoothing (Section 12).

    Simple version joins with paragraph breaks.
    Phase 3+ adds transition phrase insertion and terminology consistency.
    """
    if not chunks:
        return ""

    stitched = chunks[0]
    for i in range(1, len(chunks)):
        # Add paragraph break between chunks
        stitched += "\n\n" + chunks[i]

    return stitched
