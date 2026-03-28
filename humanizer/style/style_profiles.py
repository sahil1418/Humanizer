"""
Multi-Style Sampling — 6 Named Profiles  [Upgrade 3]  (Section 6)
──────────────────────────────────────────────────────────────────
Each profile overrides the base style vector to maximise diversity
across candidates before critic ranking.
"""

from __future__ import annotations

import logging
from typing import Optional

from humanizer.config import load_yaml_config
from humanizer.style.style_vector import StyleVector

logger = logging.getLogger(__name__)

# ── Built-in profiles (Section 6.1) ───────────────────────────────────────
_BUILTIN_PROFILES: dict[str, StyleVector] = {
    "formal_academic": StyleVector(
        formality=0.90, complexity=0.80, density=0.75,
        hedging=0.60, nominalization=0.80, sentence_length=0.80,
    ),
    "semi_formal": StyleVector(
        formality=0.65, complexity=0.55, density=0.55,
        hedging=0.35, nominalization=0.55, sentence_length=0.55,
    ),
    "conversational": StyleVector(
        formality=0.20, complexity=0.25, density=0.35,
        hedging=0.15, nominalization=0.25, sentence_length=0.30,
    ),
    "mixed_tone": StyleVector(
        formality=0.55, complexity=0.45, density=0.50,
        hedging=0.30, nominalization=0.45, sentence_length=0.50,
    ),
    "compressed": StyleVector(
        formality=0.60, complexity=0.50, density=0.90,
        hedging=0.20, nominalization=0.50, sentence_length=0.25,
    ),
    "expanded": StyleVector(
        formality=0.55, complexity=0.40, density=0.20,
        hedging=0.40, nominalization=0.40, sentence_length=0.80,
    ),
}

# Profile descriptions for logging / UI
PROFILE_DESCRIPTIONS: dict[str, str] = {
    "formal_academic": "Research, reports",
    "semi_formal": "Business, professional",
    "conversational": "Blogs, casual writing",
    "mixed_tone": "General purpose",
    "compressed": "Executive summaries",
    "expanded": "Explanatory content",
}


def get_profile(name: str) -> StyleVector:
    """Return the StyleVector for a named profile."""
    if name in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[name]

    # Try loading from YAML config
    yaml_profiles = load_yaml_config("style_profiles.yaml").get("profiles", {})
    if name in yaml_profiles:
        return StyleVector.from_dict(yaml_profiles[name])

    raise ValueError(
        f"Unknown style profile '{name}'. "
        f"Available: {list(_BUILTIN_PROFILES.keys())}"
    )


def get_all_profiles() -> dict[str, StyleVector]:
    """Return all 6 built-in style profiles."""
    return dict(_BUILTIN_PROFILES)


def get_profile_names() -> list[str]:
    """Return list of all profile names."""
    return list(_BUILTIN_PROFILES.keys())


def detect_best_profile(text: str) -> str:
    """
    Heuristic profile suggestion based on input text characteristics.
    Simple rules — production version would use a classifier.
    """
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    avg_sent_len = len(words) / max(text.count(".") + text.count("!") + text.count("?"), 1)

    # Simple heuristic
    if avg_word_len > 6 and avg_sent_len > 20:
        return "formal_academic"
    elif avg_word_len > 5 and avg_sent_len > 15:
        return "semi_formal"
    elif avg_word_len < 4.5 and avg_sent_len < 12:
        return "conversational"
    elif avg_sent_len < 10:
        return "compressed"
    elif avg_sent_len > 25:
        return "expanded"
    else:
        return "mixed_tone"
