"""Tests for utility helpers."""

from humanizer.utils import argmax, count_tokens, jitter_rate, rand_float, rand_int, weighted_sample


class TestHelpers:
    def test_count_tokens(self):
        assert count_tokens("hello world foo bar") == 4

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_argmax(self):
        assert argmax([0.1, 0.9, 0.5]) == 1

    def test_rand_float_in_range(self):
        for _ in range(100):
            val = rand_float(0.5, 0.9)
            assert 0.5 <= val <= 0.9

    def test_rand_int_in_range(self):
        for _ in range(100):
            val = rand_int(7, 14)
            assert 7 <= val <= 14

    def test_jitter_rate_bounds(self):
        for _ in range(200):
            val = jitter_rate(0.10, 0.30)
            assert 0.07 - 0.001 <= val <= 0.13 + 0.001

    def test_weighted_sample_returns_valid(self):
        items = ["a", "b", "c"]
        weights = [0.5, 0.3, 0.2]
        result = weighted_sample(items, weights)
        assert result in items
