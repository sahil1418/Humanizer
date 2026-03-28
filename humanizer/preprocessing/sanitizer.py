"""
Input Sanitisation Pipeline  (Section 19)
─────────────────────────────────────────
Language detection · PII masking · Abuse / risk classification · Token counting.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from langdetect import detect, LangDetectException

from humanizer.config import SUPPORTED_LANGUAGES
from humanizer.utils import count_tokens

logger = logging.getLogger(__name__)

# ── PII patterns (lightweight regex layer — production would add spaCy NER) ──
_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("EMAIL", "[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", "[PHONE]", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")),
    ("SSN", "[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", "[CARD]", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("IP_ADDR", "[IP]", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
]

# ── Abuse / prompt-injection keywords ──────────────────────────────────────
_ABUSE_KEYWORDS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "system prompt",
    "you are now",
    "act as if",
    "pretend you are",
    "jailbreak",
]


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class PIIMap:
    """Stores the mapping from placeholder → original PII value for re-injection."""

    entries: dict[str, str] = field(default_factory=dict)
    _counter: int = field(default=0, repr=False)

    def add(self, tag: str, original: str) -> str:
        """Register a PII match and return its unique placeholder."""
        self._counter += 1
        placeholder = f"<{tag}_{self._counter}>"
        self.entries[placeholder] = original
        return placeholder

    def restore(self, text: str) -> str:
        """Replace all placeholders with their original values."""
        for placeholder, original in self.entries.items():
            text = text.replace(placeholder, original)
        return text


@dataclass
class SanitizedInput:
    """Result of running the sanitisation pipeline."""

    text: str
    original_text: str
    pii_map: PIIMap
    token_count: int
    language: str
    risk_score: float = 0.0


# ── Exceptions ─────────────────────────────────────────────────────────────
class UnsupportedLanguageError(ValueError):
    """Raised when the input language is not in SUPPORTED_LANGUAGES."""


class AbuseDetectedError(ValueError):
    """Raised when the input is classified as abusive or a prompt injection."""


class InputTooLongError(ValueError):
    """Raised when input exceeds the maximum allowed token count."""


# ── Public API ─────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Detect language code (ISO 639-1).  Falls back to 'en' on failure."""
    try:
        return detect(text)
    except LangDetectException:
        logger.warning("Language detection failed — defaulting to 'en'")
        return "en"


def mask_pii(text: str) -> tuple[str, PIIMap]:
    """Replace PII patterns with tagged placeholders. Returns (masked_text, pii_map)."""
    pii_map = PIIMap()
    masked = text
    for tag, _label, pattern in _PII_PATTERNS:
        for match in pattern.finditer(masked):
            original = match.group()
            placeholder = pii_map.add(tag, original)
            masked = masked.replace(original, placeholder, 1)
    return masked, pii_map


def classify_input_risk(text: str) -> float:
    """
    Simple keyword-based risk scorer (0.0 = safe, 1.0 = definite abuse).
    Production system should use a trained classifier.
    """
    text_lower = text.lower()
    hits = sum(1 for kw in _ABUSE_KEYWORDS if kw in text_lower)
    # Normalise — 2+ keyword matches ⇒ high risk
    return min(hits / 2.0, 1.0)


async def sanitize_input(
    raw_text: str,
    *,
    max_tokens: int = 8192,
) -> SanitizedInput:
    """
    Full input sanitisation pipeline (Section 19.1):
      1. Language detection
      2. PII masking
      3. Abuse / prompt-injection check
      4. Token counting
    """
    # 1. Language
    lang = detect_language(raw_text)
    if lang not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"Language '{lang}' is not supported. Supported: {SUPPORTED_LANGUAGES}"
        )

    # 2. PII
    masked, pii_map = mask_pii(raw_text)

    # 3. Abuse
    risk = classify_input_risk(masked)
    if risk > 0.8:
        raise AbuseDetectedError("Input classified as abusive or prompt injection attempt.")

    # 4. Token count
    token_count = count_tokens(masked)
    if token_count > max_tokens:
        raise InputTooLongError(
            f"Input has {token_count} tokens — exceeds maximum of {max_tokens}."
        )

    return SanitizedInput(
        text=masked,
        original_text=raw_text,
        pii_map=pii_map,
        token_count=token_count,
        language=lang,
        risk_score=risk,
    )
