"""Tests for Phase 2 — Multi-pass pipeline, style, memory drift, chunker, critic."""

import pytest

from humanizer.style.style_vector import StyleVector
from humanizer.style.style_profiles import (
    get_profile,
    get_all_profiles,
    get_profile_names,
    detect_best_profile,
)
from humanizer.pipeline.memory_drift import (
    apply_memory_drift,
    apply_reference_drift,
    compress_context_summary,
)
from humanizer.pipeline.pipeline_modes import (
    get_mode,
    should_fallback_to_lite,
    PIPELINE_MODES,
)
from humanizer.preprocessing.chunker import semantic_chunk, coherence_stitch


# ── Style Vector ───────────────────────────────────────────────────────────
class TestStyleVector:
    def test_default_values(self):
        sv = StyleVector()
        assert sv.formality == 0.55
        assert sv.style_dim == 6

    def test_to_list(self):
        sv = StyleVector()
        lst = sv.to_list()
        assert len(lst) == 6
        assert all(isinstance(v, float) for v in lst)

    def test_from_dict(self):
        d = {"formality": 0.9, "complexity": 0.8, "density": 0.7}
        sv = StyleVector.from_dict(d)
        assert sv.formality == 0.9
        assert sv.complexity == 0.8
        # Non-specified values get defaults
        assert sv.hedging == 0.30

    def test_to_dict(self):
        sv = StyleVector(formality=0.9)
        d = sv.to_dict()
        assert d["formality"] == 0.9
        assert "complexity" in d

    def test_tone_drift_stays_in_bounds(self):
        sv = StyleVector(formality=0.0, complexity=1.0)
        for _ in range(100):
            drifted = sv.apply_tone_drift(0.15)
            assert 0.0 <= drifted.formality <= 1.0
            assert 0.0 <= drifted.complexity <= 1.0

    def test_tone_drift_changes_values(self):
        sv = StyleVector(formality=0.5)
        drifts = [sv.apply_tone_drift(0.15).formality for _ in range(20)]
        # Not all identical (would be astronomically unlikely)
        assert len(set(drifts)) > 1


# ── Style Profiles ─────────────────────────────────────────────────────────
class TestStyleProfiles:
    def test_all_profiles_available(self):
        profiles = get_all_profiles()
        assert len(profiles) == 6
        assert "formal_academic" in profiles
        assert "conversational" in profiles

    def test_profile_names(self):
        names = get_profile_names()
        assert len(names) == 6
        assert "mixed_tone" in names

    def test_get_profile(self):
        sv = get_profile("formal_academic")
        assert sv.formality == 0.90
        assert sv.complexity == 0.80

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown style profile"):
            get_profile("nonexistent")

    def test_detect_profile_returns_valid(self):
        text = "This is a simple casual blog post about a fun topic."
        profile = detect_best_profile(text)
        assert profile in get_profile_names()

    def test_detect_formal(self):
        text = (
            "The investigation demonstrated that the utilization of "
            "sophisticated methodologies significantly contributed to the "
            "establishment of comprehensive analytical frameworks within "
            "the organizational infrastructure of contemporary institutions."
        )
        profile = detect_best_profile(text)
        assert profile in ("formal_academic", "semi_formal", "expanded")


# ── Memory Drift ───────────────────────────────────────────────────────────
class TestMemoryDrift:
    def test_drift_preserves_count(self):
        props = ["First point.", "Second point.", "Third point.", "Fourth point."]
        result = apply_memory_drift(props, drift_rate=0.5)
        assert len(result) == len(props)

    def test_zero_drift_rate(self):
        props = ["Exact text.", "Another sentence."]
        result = apply_memory_drift(props, drift_rate=0.0)
        assert result == props

    def test_reference_drift_returns_string(self):
        text = "The model works well. However, it needs tuning. Furthermore, validation is key."
        result = apply_reference_drift(text, drift_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compress_context_summary(self):
        text = " ".join([f"word{i}" for i in range(100)])
        compressed = compress_context_summary(text, max_tokens=60)
        assert len(compressed.split()) <= 62  # 60 + "..."

    def test_compress_short_text_unchanged(self):
        text = "Short text here."
        assert compress_context_summary(text, max_tokens=60) == text


# ── Pipeline Modes ─────────────────────────────────────────────────────────
class TestPipelineModes:
    def test_full_mode(self):
        mode = get_mode("full")
        assert mode.noise_enabled is True
        assert mode.meta_rewriter_enabled is True
        assert mode.deterministic is False

    def test_lite_mode(self):
        mode = get_mode("lite")
        assert mode.noise_enabled is False
        assert mode.meta_rewriter_enabled is False

    def test_deterministic_mode(self):
        mode = get_mode("deterministic")
        assert mode.deterministic is True

    def test_fallback_on_retry(self):
        assert should_fallback_to_lite(retry_count=2) is True
        assert should_fallback_to_lite(retry_count=0) is False

    def test_fallback_on_latency(self):
        assert should_fallback_to_lite(latency_ms=6000, p95_threshold_ms=5000) is True

    def test_fallback_on_health(self):
        assert should_fallback_to_lite(health_ok=False) is True


# ── Semantic Chunker ───────────────────────────────────────────────────────
class TestChunker:
    def test_short_text_single_chunk(self):
        text = "This is a short text. It has just a few sentences."
        chunks = semantic_chunk(text)
        assert len(chunks) == 1

    def test_paragraph_based_chunking(self):
        paragraphs = [
            "This is the first paragraph. " * 20,
            "This is the second paragraph. " * 20,
            "This is the third paragraph. " * 20,
        ]
        text = "\n\n".join(paragraphs)
        chunks = semantic_chunk(text, max_chunk_tokens=200)
        assert len(chunks) >= 2

    def test_chunk_token_counts(self):
        text = "Sentence one. " * 100 + "\n\n" + "Sentence two. " * 100
        chunks = semantic_chunk(text, max_chunk_tokens=150)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_coherence_stitch(self):
        parts = ["Part one text.", "Part two text.", "Part three text."]
        result = coherence_stitch(parts)
        assert "Part one" in result
        assert "Part two" in result
        assert "\n\n" in result

    def test_empty_stitch(self):
        assert coherence_stitch([]) == ""
