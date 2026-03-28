"""
Humanizer v2.0 — Pipeline Validation Script (v2)
──────────────────────────────────────────────────
Run REAL inference with actual models to verify:
  1. Outputs are structurally different from inputs
  2. Different runs produce different outputs
  3. Style profiles generate noticeably different text
  4. Multi-pass pipeline produces genuine rewrites
  5. Memory drift and stochastic params create variation
  6. Semantic validation scores are within target ranges

Usage (on Lightning AI):
    python scripts/validate_pipeline.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanizer.config import MODEL_REGISTRY


# ── Test inputs ────────────────────────────────────────────────────────────
TEST_INPUTS = [
    {
        "name": "Academic paragraph",
        "text": (
            "Artificial intelligence has fundamentally transformed the landscape of modern "
            "technology. Machine learning algorithms, particularly deep neural networks, have "
            "demonstrated remarkable capabilities in pattern recognition, natural language "
            "processing, and autonomous decision making. These advances have led to widespread "
            "adoption across industries, from healthcare diagnostics to financial trading systems. "
            "However, concerns about bias, transparency, and accountability remain significant "
            "challenges that researchers and policymakers must address."
        ),
    },
    {
        "name": "Casual blog post",
        "text": (
            "So I tried this new coffee shop downtown and honestly it was amazing. The barista "
            "was super friendly and they had this oat milk latte that just blew my mind. I know "
            "everyone says they have the best coffee but seriously this place is different. They "
            "roast their own beans and you can actually taste the difference. Would definitely "
            "recommend checking it out if you're in the area."
        ),
    },
    {
        "name": "Technical documentation",
        "text": (
            "The system implements a three-tier architecture consisting of a presentation layer, "
            "business logic layer, and data access layer. The presentation layer handles HTTP "
            "requests through RESTful API endpoints. The business logic layer processes requests "
            "using a pipeline pattern with configurable middleware components. The data access "
            "layer abstracts database operations through a repository pattern, supporting both "
            "PostgreSQL and MongoDB backends. Connection pooling is managed by the framework "
            "to optimize resource utilization."
        ),
    },
]


def print_header(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")


def print_comparison(label: str, original: str, rewritten: str) -> None:
    """Print side-by-side comparison with word overlap stats."""
    orig_words = set(original.lower().split())
    rewrite_words = set(rewritten.lower().split())
    overlap = len(orig_words & rewrite_words) / max(len(orig_words), 1)

    # N-gram overlap (more informative)
    def ngrams(text, n=3):
        tokens = text.lower().split()
        return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    ng_orig = ngrams(original)
    ng_rewrite = ngrams(rewritten)
    ng_overlap = len(ng_orig & ng_rewrite) / max(len(ng_orig), 1) if ng_orig else 0

    print(f"  [{label}]")
    print(f"  ORIGINAL  ({len(original.split())} words):")
    print(f"    {original[:200]}{'...' if len(original) > 200 else ''}")
    print(f"  REWRITTEN ({len(rewritten.split())} words):")
    print(f"    {rewritten[:200]}{'...' if len(rewritten) > 200 else ''}")
    print(f"  Word overlap:   {overlap:.1%}")
    print(f"  3-gram overlap: {ng_overlap:.1%}")
    print()


async def test_1_four_pass_rewrite():
    """Test: Does the 4-pass pipeline produce structurally different outputs?"""
    print_header("TEST 1: 4-Pass Pipeline Rewrite — All Input Types")

    from humanizer.pipeline.multi_pass import _run_single_pipeline
    from humanizer.style.style_vector import StyleVector

    sv = StyleVector()  # default mixed_tone

    for sample in TEST_INPUTS:
        t0 = time.perf_counter()
        result = await _run_single_pipeline(
            sample["text"], sv, model="flan-t5-xl",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print_comparison(f"{sample['name']} ({elapsed:.0f}ms)", sample["text"], result)

    print("  → All inputs should show <50% word overlap and <30% 3-gram overlap")


async def test_2_run_variation():
    """Test: Do different runs produce different outputs?"""
    print_header("TEST 2: Run Variation — 3 runs on same input (4-pass)")

    from humanizer.pipeline.multi_pass import _run_single_pipeline
    from humanizer.style.style_vector import StyleVector

    sv = StyleVector()
    sample = TEST_INPUTS[0]
    outputs = []
    for i in range(3):
        result = await _run_single_pipeline(
            sample["text"], sv, model="flan-t5-xl",
        )
        outputs.append(result)
        print(f"  Run {i+1}: {result[:150]}...")

    # Check diversity
    unique = len(set(outputs))
    print(f"\n  → Unique outputs: {unique}/3")
    if unique == 1:
        print("  ⚠️  All identical — stochastic params may not be effective")
    else:
        print("  ✓  Outputs vary between runs")


async def test_3_style_profiles():
    """Test: Do style profiles produce noticeably different text via 4-pass?"""
    print_header("TEST 3: Style Profiles — Same input, 6 styles via 4-pass")

    from humanizer.pipeline.multi_pass import _run_single_pipeline
    from humanizer.style.style_profiles import get_all_profiles

    sample = TEST_INPUTS[0]
    profiles = get_all_profiles()

    results = {}
    for name, sv in profiles.items():
        t0 = time.perf_counter()
        result = await _run_single_pipeline(
            sample["text"], sv, model="flan-t5-xl",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        results[name] = result
        print(f"  [{name}] (f={sv.formality}, d={sv.density}) ({elapsed:.0f}ms):")
        print(f"    {result[:200]}...")
        print()

    # Check diversity across profiles
    unique = len(set(results.values()))
    print(f"  → Unique outputs: {unique}/{len(profiles)} profiles")
    if unique <= 2:
        print("  ⚠️  Profiles produce similar output — style conditioning needs work")
    else:
        print("  ✓  Profiles produce varied output")


async def test_4_per_pass_trace():
    """Test: Show what each pass produces for the academic paragraph."""
    print_header("TEST 4: Per-Pass Trace — What each pass does")

    from humanizer.inference.inference_layer import generate_text
    from humanizer.style.style_vector import StyleVector
    from humanizer.pipeline.multi_pass import (
        _pass1_summarize, _pass2_expand, _pass3_diversify, _pass4_polish,
    )

    sample = TEST_INPUTS[0]
    sv = StyleVector()

    print(f"  ORIGINAL ({len(sample['text'].split())} words):")
    print(f"    {sample['text'][:200]}...")
    print()

    # Pass 1
    t0 = time.perf_counter()
    p1 = await _pass1_summarize(sample["text"], "flan-t5-xl")
    print(f"  PASS 1 — Summarize ({(time.perf_counter()-t0)*1000:.0f}ms, {len(p1.split())} words):")
    print(f"    {p1[:300]}{'...' if len(p1) > 300 else ''}")
    print()

    # Pass 2
    t0 = time.perf_counter()
    p2 = await _pass2_expand(p1, sv, "flan-t5-xl")
    print(f"  PASS 2 — Expand ({(time.perf_counter()-t0)*1000:.0f}ms, {len(p2.split())} words):")
    print(f"    {p2[:300]}{'...' if len(p2) > 300 else ''}")
    print()

    # Pass 3
    t0 = time.perf_counter()
    p3 = await _pass3_diversify(p2, sv, "flan-t5-xl")
    print(f"  PASS 3 — Diversify ({(time.perf_counter()-t0)*1000:.0f}ms, {len(p3.split())} words):")
    print(f"    {p3[:300]}{'...' if len(p3) > 300 else ''}")
    print()

    # Pass 4
    t0 = time.perf_counter()
    p4 = await _pass4_polish(p3, sv, "flan-t5-xl")
    print(f"  PASS 4 — Polish ({(time.perf_counter()-t0)*1000:.0f}ms, {len(p4.split())} words):")
    print(f"    {p4[:300]}{'...' if len(p4) > 300 else ''}")
    print()

    # Final comparison
    print_comparison("End-to-end (4 passes)", sample["text"], p4)


async def test_5_encoder_penalty_effect():
    """Test: Does encoder_repetition_penalty reduce copy-through?"""
    print_header("TEST 5: Encoder Repetition Penalty — Copy-Through Test")

    from humanizer.inference.inference_layer import generate_text

    sample = TEST_INPUTS[0]["text"]  # Academic (hardest case)
    prefix = "Rewrite the following text using completely different words: "

    # Test with penalty disabled vs enabled
    for penalty, label in [(1.0, "No penalty"), (1.5, "Penalty=1.5"), (2.0, "Penalty=2.0")]:
        result = await generate_text(
            "flan-t5-xl", sample,
            prefix=prefix,
            encoder_repetition_penalty=penalty,
        )
        orig_words = set(sample.lower().split())
        rewrite_words = set(result.lower().split())
        overlap = len(orig_words & rewrite_words) / max(len(orig_words), 1)
        print(f"  [{label}] Word overlap: {overlap:.1%}")
        print(f"    {result[:180]}...")
        print()

    print("  → Lower overlap = penalty is working")


async def test_6_full_validation():
    """Test: Semantic validation on 4-pass output."""
    print_header("TEST 6: Semantic Validation on 4-Pass Output")

    from humanizer.pipeline.multi_pass import _run_single_pipeline
    from humanizer.style.style_vector import StyleVector
    from humanizer.validation.semantic_checks import validate_semantic

    sample = TEST_INPUTS[0]
    sv = StyleVector()

    rewritten = await _run_single_pipeline(
        sample["text"], sv, model="flan-t5-xl",
    )

    validation = validate_semantic(sample["text"], rewritten)

    sim_status = "✓" if 0.75 <= validation.semantic_similarity <= 0.90 else "⚠️"
    nov_status = "✓" if 0.55 <= validation.lexical_novelty <= 0.72 else "⚠️"
    nli_status = "✓" if validation.nli_label == "entailment" else "⚠️"
    ent_status = "✓" if validation.entity_preserved else "⚠️"
    read_status = "✓" if validation.readability_delta < 14 else "⚠️"

    print(f"  {sim_status} Semantic Similarity:  {validation.semantic_similarity:.3f}  (want: 0.75–0.90)")
    print(f"  {nov_status} Lexical Novelty:     {validation.lexical_novelty:.3f}  (want: 0.55–0.72)")
    print(f"  {nli_status} NLI Label:           {validation.nli_label}  (want: entailment)")
    print(f"    NLI Score:           {validation.nli_entailment_score:.3f}")
    print(f"  {ent_status} Entity Preserved:    {validation.entity_preserved}")
    print(f"  {read_status} Readability Delta:   {validation.readability_delta:.1f}  (want: < 14)")
    print(f"    Overall Passed:      {validation.passed}")
    print()
    print(f"  ORIGINAL:  {sample['text'][:150]}...")
    print(f"  REWRITTEN: {rewritten[:150]}...")


async def main():
    print_header("HUMANIZER v2.0 — PIPELINE VALIDATION (v2)")
    print("  Running 6 tests with FLAN-T5-XL.")
    print("  Changes from v1:")
    print("    • All tests now use the 4-pass pipeline")
    print("    • encoder_repetition_penalty=1.5 (anti-copy)")
    print("    • Tuned temp=1.2, rep_penalty=2.0, no_repeat_ngram=4")
    print("    • Chain-of-thought prompts for each pass")
    print("    • Style-aware prefix injection in Pass 2 & 3")
    print()

    # Ensure model registry has flan-t5-xl
    MODEL_REGISTRY["flan-t5-xl"] = "google/flan-t5-xl"

    t_start = time.perf_counter()

    await test_1_four_pass_rewrite()
    await test_2_run_variation()
    await test_3_style_profiles()
    await test_4_per_pass_trace()
    await test_5_encoder_penalty_effect()
    await test_6_full_validation()

    total = time.perf_counter() - t_start
    print_header(f"VALIDATION COMPLETE — Total time: {total:.1f}s")
    print("  Review the outputs above and confirm:")
    print("    1. 4-pass rewrites show <50% word overlap for all text types")
    print("    2. Different runs produce varied outputs")
    print("    3. Style profiles create noticeably different text")
    print("    4. Each pass contributes meaningful transformation")
    print("    5. encoder_repetition_penalty reduces copy-through")
    print("    6. Validation scores are within expected ranges")
    print()
    print("  Target metrics:")
    print("    • Word overlap:           <50% (was 98-100%)")
    print("    • 3-gram overlap:         <30%")
    print("    • Semantic similarity:    0.75-0.90")
    print("    • Lexical novelty:        0.55-0.72 (was 0.108)")
    print("    • NLI:                    entailment")


if __name__ == "__main__":
    asyncio.run(main())
