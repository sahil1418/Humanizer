"""
Model Router  (Section 16)
──────────────────────────
Selects the inference model based on token count + quality tier.
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

QualityTier = Literal["standard", "premium"]


def route_request(input_tokens: int, quality_tier: QualityTier = "standard") -> str:
    """
    Determine which model to use for the given input length and tier.

    Routing logic (from architecture doc Section 16.1):
        <200  tokens + standard → t5-base         (fast lane)
        200–1K tokens + standard → flan-t5-xl
        200–1K tokens + premium  → bart-large
        ≥1K  tokens  or premium  → llama-3-8b-ft
        >4K  tokens              → mixtral-8x7b-q4
    """
    if input_tokens > 4000:
        model = "mixtral-8x7b-q4"
    elif input_tokens >= 1000 or quality_tier == "premium":
        model = "llama-3-8b-ft"
    elif input_tokens >= 200 and quality_tier == "premium":
        model = "bart-large"
    elif input_tokens >= 200:
        model = "flan-t5-xl"
    else:
        model = "t5-base" if quality_tier == "standard" else "bart-large"

    logger.info("Routed %d tokens / %s tier → %s", input_tokens, quality_tier, model)
    return model
