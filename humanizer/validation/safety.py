"""
Safety Guardrails  (Section 19.2)
──────────────────────────────────
Toxicity filtering · Prompt injection detection · Output length bounds.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy-loaded toxicity model
_detoxify_model = None


def _get_detoxify():
    global _detoxify_model
    if _detoxify_model is None:
        from detoxify import Detoxify
        _detoxify_model = Detoxify("original")
        logger.info("Loaded Detoxify model")
    return _detoxify_model


# ── Prompt injection patterns ─────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(your\s+)?instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|<\|system\|>|<\|user\|>", re.IGNORECASE),
]


@dataclass
class SafetyResult:
    """Result of safety checks on a text."""

    is_safe: bool = True
    toxicity_score: float = 0.0
    toxicity_categories: dict[str, float] = None
    prompt_injection_detected: bool = False
    length_ok: bool = True
    message: str = ""

    def __post_init__(self):
        if self.toxicity_categories is None:
            self.toxicity_categories = {}


def check_toxicity(text: str, threshold: float = 0.10) -> tuple[bool, float, dict[str, float]]:
    """
    Run Detoxify toxicity check.
    Returns (is_safe, max_score, all_category_scores).
    """
    model = _get_detoxify()
    results = model.predict(text)
    # results is dict like {'toxicity': 0.01, 'severe_toxicity': 0.0, ...}
    max_score = max(results.values())
    is_safe = max_score < threshold
    return is_safe, max_score, results


def detect_prompt_injection(text: str) -> bool:
    """Return True if any prompt-injection pattern is found."""
    return any(pat.search(text) for pat in _INJECTION_PATTERNS)


def check_output_length(
    text: str,
    *,
    min_tokens: int = 5,
    max_tokens: int = 8192,
) -> bool:
    """Verify that output is within acceptable token bounds."""
    token_count = len(text.split())
    return min_tokens <= token_count <= max_tokens


def run_safety_checks(
    text: str,
    *,
    toxicity_threshold: float = 0.10,
    max_tokens: int = 8192,
) -> SafetyResult:
    """
    Full safety guardrail pipeline (Section 19.2):
      1. Toxicity filter
      2. Prompt injection detection
      3. Output length bounds
    """
    # 1. Toxicity
    tox_safe, tox_score, tox_cats = check_toxicity(text, toxicity_threshold)

    # 2. Prompt injection
    injection = detect_prompt_injection(text)

    # 3. Length
    length_ok = check_output_length(text, max_tokens=max_tokens)

    is_safe = tox_safe and not injection and length_ok

    messages = []
    if not tox_safe:
        messages.append(f"Toxicity score {tox_score:.3f} exceeds threshold {toxicity_threshold}")
    if injection:
        messages.append("Prompt injection pattern detected")
    if not length_ok:
        messages.append("Output length out of bounds")

    return SafetyResult(
        is_safe=is_safe,
        toxicity_score=tox_score,
        toxicity_categories=tox_cats,
        prompt_injection_detected=injection,
        length_ok=length_ok,
        message="; ".join(messages),
    )
