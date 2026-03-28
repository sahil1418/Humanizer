"""
Style Vector Schema  (Section 5)
──────────────────────────────────
Multi-dimensional continuous style representation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class StyleVector:
    """
    6-axis continuous style vector (Section 5.1).

    Each axis is a float in [0.0, 1.0].
    """

    formality: float = 0.55       # 0.0 = casual, 1.0 = formal
    complexity: float = 0.45      # 0.0 = simple, 1.0 = complex vocab
    density: float = 0.50         # 0.0 = expansive, 1.0 = dense/concise
    hedging: float = 0.30         # 0.0 = assertive, 1.0 = heavily hedged
    nominalization: float = 0.45  # 0.0 = verb-heavy, 1.0 = noun-heavy
    sentence_length: float = 0.50 # 0.0 = short, 1.0 = long/complex

    def to_list(self) -> list[float]:
        """Return as a flat list [f, c, d, h, n, sl] for tensor conversion."""
        return [
            self.formality, self.complexity, self.density,
            self.hedging, self.nominalization, self.sentence_length,
        ]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleVector":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def apply_tone_drift(self, drift_amount: float = 0.15) -> "StyleVector":
        """
        Simulate tone drift (U5) by shifting each axis by ±drift_amount.
        Returns a new StyleVector.
        """
        import random

        def _drift(val: float) -> float:
            shifted = val + random.uniform(-drift_amount, drift_amount)
            return max(0.0, min(1.0, shifted))

        return StyleVector(
            formality=_drift(self.formality),
            complexity=_drift(self.complexity),
            density=_drift(self.density),
            hedging=_drift(self.hedging),
            nominalization=_drift(self.nominalization),
            sentence_length=_drift(self.sentence_length),
        )

    @property
    def style_dim(self) -> int:
        return 6
