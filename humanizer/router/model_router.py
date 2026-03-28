"""
Model Router  (Simplified — Personal Use)
──────────────────────────────────────────
Single-model architecture — always routes to the default model.
"""

from __future__ import annotations

import logging

from humanizer.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def route_request(input_tokens: int = 0, quality_tier: str = "standard") -> str:
    """
    Always route to the default model (personal use — single model).

    Parameters are kept for API compatibility but ignored.
    """
    model = DEFAULT_MODEL
    logger.info("Routed %d tokens → %s (single-model mode)", input_tokens, model)
    return model
