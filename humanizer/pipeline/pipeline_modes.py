"""
Pipeline Mode Selection  (Section 22.5)
────────────────────────────────────────
Full / Lite / Deterministic pipeline modes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

PipelineModeName = Literal["full", "lite", "deterministic"]


@dataclass(frozen=True)
class PipelineModeConfig:
    """Configuration for a pipeline execution mode."""

    name: PipelineModeName
    noise_enabled: bool
    meta_rewriter_enabled: bool
    deterministic: bool
    description: str


# ── Mode definitions (from architecture doc Section 22.5) ──────────────────
PIPELINE_MODES: dict[PipelineModeName, PipelineModeConfig] = {
    "full": PipelineModeConfig(
        name="full",
        noise_enabled=True,
        meta_rewriter_enabled=True,
        deterministic=False,
        description="Full pipeline — all layers active including noise and meta-rewriter",
    ),
    "lite": PipelineModeConfig(
        name="lite",
        noise_enabled=False,
        meta_rewriter_enabled=False,
        deterministic=False,
        description="Lite pipeline — Passes 1–4 only, no noise injection or meta-rewriter. "
                    "Used for edge cases, latency spikes, or retry ≥ 2.",
    ),
    "deterministic": PipelineModeConfig(
        name="deterministic",
        noise_enabled=True,
        meta_rewriter_enabled=True,
        deterministic=True,
        description="Deterministic mode — all stochastic parameters use fixed seeds. "
                    "For debugging and regression testing only.",
    ),
}


def get_mode(name: PipelineModeName) -> PipelineModeConfig:
    """Return the pipeline mode configuration."""
    return PIPELINE_MODES[name]


def should_fallback_to_lite(
    *,
    retry_count: int = 0,
    latency_ms: float = 0,
    p95_threshold_ms: float = 5000,
    health_ok: bool = True,
) -> bool:
    """
    Determine if the pipeline should fall back to lite mode.
    Triggers: latency > P95, retry ≥ 2, or health-check failure.
    """
    if retry_count >= 2:
        return True
    if latency_ms > p95_threshold_ms:
        return True
    if not health_ok:
        return True
    return False


def set_deterministic_seed(seed: int = 42) -> None:
    """Set all random seeds for deterministic mode."""
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
