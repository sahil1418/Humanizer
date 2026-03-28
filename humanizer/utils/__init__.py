"""Utility helpers used across the Humanizer pipeline."""

from __future__ import annotations

import random
from typing import Sequence


def weighted_sample(items: Sequence[str], weights: Sequence[float]) -> str:
    """Return a single item sampled according to *weights*."""
    return random.choices(items, weights=weights, k=1)[0]


def rand_float(lo: float, hi: float) -> float:
    """Uniform random float in [lo, hi]."""
    return random.uniform(lo, hi)


def rand_int(lo: int, hi: int) -> int:
    """Uniform random integer in [lo, hi]."""
    return random.randint(lo, hi)


def jitter_rate(base_rate: float, jitter_factor: float = 0.30) -> float:
    """Apply ±jitter_factor to *base_rate* (e.g. 0.10 ± 30% → 0.07–0.13)."""
    lo = base_rate * (1.0 - jitter_factor)
    hi = base_rate * (1.0 + jitter_factor)
    return random.uniform(lo, hi)


def count_tokens(text: str) -> int:
    """Rough whitespace-based token count (fast, no model dependency)."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def argmax(values: Sequence[float]) -> int:
    """Return index of the maximum value."""
    return max(range(len(values)), key=lambda i: values[i])
