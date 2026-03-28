"""
Global configuration for the Humanizer v2.0 system.
All thresholds, model paths, and defaults centralised here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / ".model_cache")))

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "pt", "it"}

# ---------------------------------------------------------------------------
# Model catalogue — maps logical name → HuggingFace model ID
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, str] = {
    "t5-base": "google/t5-v1_1-base",
    "t5-large": "google/t5-v1_1-large",
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "bart-large": "facebook/bart-large",
    "llama-3-8b-ft": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mixtral-8x7b-q4": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # Auxiliary
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "nli-deberta": "cross-encoder/nli-deberta-v3-base",
    "spacy-en": "en_core_web_sm",
}

# Maps meta-rewriter primary → secondary model (cross-family fingerprint break)
META_REWRITER_MAP: dict[str, str] = {
    "t5-base": "llama-3-8b-ft",
    "flan-t5-xl": "llama-3-8b-ft",
    "bart-large": "mixtral-8x7b-q4",
    "llama-3-8b-ft": "mixtral-8x7b-q4",
    "mixtral-8x7b-q4": "llama-3-8b-ft",
}


# ---------------------------------------------------------------------------
# Validation thresholds (v2.0 — adaptive + safety-locked)
# ---------------------------------------------------------------------------
@dataclass
class ValidationConfig:
    """Immutable defaults — per-request adaptive thresholds are generated at runtime."""

    # Adaptive ranges (randomised per-request)
    semantic_min_range: tuple[float, float] = (0.75, 0.90)
    novelty_min_range: tuple[float, float] = (0.55, 0.72)
    readability_delta_range: tuple[int, int] = (7, 14)

    # Safety-locked (never randomised)
    entity_preservation: str = "strict"
    factual_consistency: str = "not_contradiction"
    toxicity_max: float = 0.10

    # Perplexity & burstiness (U2)
    perplexity_min: float = 35.0
    perplexity_max: float = 85.0
    burstiness_min: float = 0.35

    # Revision
    max_revision_attempts: int = 3


# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    quality_tier: Literal["standard", "premium"] = "standard"
    pipeline_mode: Literal["full", "lite", "deterministic"] = "full"
    max_input_tokens: int = 8192
    default_style_profile: str = "mixed_tone"

    # Human noise defaults (U1)
    connector_rate: float = 0.10
    length_jitter: float = 0.15
    redundancy_prob: float = 0.06
    grammar_noise: float = 0.04
    tone_drift_interval: int = 250

    # Memory drift (U5)
    drift_rate_pass2: float = 0.05
    drift_rate_pass4: float = 0.08
    drift_min_similarity: float = 0.78

    # Cross-chunk context (long-form)
    cross_chunk_summary_tokens: int = 60

    # Validation
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
@dataclass
class GenerationConfig:
    """Decoding parameters shared across style profiles."""

    num_beams: int = 6
    num_beam_groups: int = 3
    diversity_penalty: float = 0.80
    temperature: float = 0.75
    top_p: float = 0.90
    repetition_penalty: float = 1.30
    no_repeat_ngram_size: int = 4
    max_new_tokens: int = 512

    # Meta-rewriter uses higher temp
    meta_temperature: float = 0.85
    meta_top_p: float = 0.92
    meta_repetition_penalty: float = 1.20


# ---------------------------------------------------------------------------
# Latency targets (P99)
# ---------------------------------------------------------------------------
LATENCY_TARGETS_MS: dict[str, int] = {
    "t5-base": 200,
    "flan-t5-xl": 1200,
    "bart-large": 1200,
    "llama-3-8b-ft": 3500,
    "mixtral-8x7b-q4": 12000,
}


# ---------------------------------------------------------------------------
# Convenience loader for YAML config overrides
# ---------------------------------------------------------------------------
def load_yaml_config(filename: str) -> dict:
    """Load a YAML config from the configs/ directory."""
    path = CONFIG_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
