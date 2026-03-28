"""
Style Prefix Encoder  (Section 5.2)
────────────────────────────────────
Projects the 6-dim style vector into soft prefix embeddings
prepended to the decoder context.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StylePrefixEncoder(nn.Module):
    """
    Projects a continuous style vector into soft prefix embeddings (Section 5.2).

    The prefix is prepended to the decoder hidden states, allowing gradients
    to flow through the style representation during fine-tuning.

    Args:
        style_dim:  Number of style axes (default 6).
        prefix_len: Number of prefix tokens to generate (default 20).
        hidden:     Hidden dimension matching the transformer model (default 768).
    """

    def __init__(
        self,
        style_dim: int = 6,
        prefix_len: int = 20,
        hidden: int = 768,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.prefix_len = prefix_len
        self.hidden = hidden

        self.proj = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, prefix_len * hidden),
        )

    def forward(self, style_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_vector: (batch, style_dim) tensor of style values in [0, 1].

        Returns:
            (batch, prefix_len, hidden) soft prefix embeddings.
        """
        # style_vector shape: (batch, 6)
        projected = self.proj(style_vector)  # (batch, prefix_len * hidden)
        return projected.view(-1, self.prefix_len, self.hidden)

    @classmethod
    def from_style_vector_list(
        cls,
        style_values: list[float],
        *,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """
        Convenience: create prefix embeddings from a raw style list.
        Useful for inference without constructing tensors manually.
        """
        encoder = cls(**kwargs).to(device).eval()
        vec = torch.tensor([style_values], dtype=torch.float32, device=device)
        with torch.no_grad():
            return encoder(vec)
