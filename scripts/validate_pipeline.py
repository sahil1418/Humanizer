"""
Humanizer v2.0 — Pipeline Validation Script
─────────────────────────────────────────────
Run REAL inference with actual models to verify:
  1. Outputs are structurally different from inputs
  2. Different runs produce different outputs
  3. Style profiles generate noticeably different text
  4. Multi-pass pipeline produces genuine rewrites
  5. Memory drift and stochastic params create variation

Usage (on Lightning AI):
    python scripts/validate_pipeline.py
"""

import asyncio
import json
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

    print(f"  [{label}]")
    print(f"  ORIGINAL  ({len(original.split())} words):")
    print(f"    {original[:200]}{'...' if len(original) > 200 else ''}")
    print(f"  REWRITTEN ({len(rewritten.split())} words):")
    print(f"    {rewritten[:200]}{'...' if len(rewritten) > 200 else ''}")
    print(f"  Word overlap: {overlap:.1%}")
    print()


async def test_1_basic_rewrite():
    """Test: Do outputs differ structurally from inputs?"""
    print_header("TEST 1: Basic Rewrite — Is output structurally different?")

    from humanizer.inference.inference_layer import generate_text

    for sample in TEST_INPUTS:
        t0 = time.perf_counter()
        result = await generate_text(
            "flan-t5-base",
            sample["text"],
            prefix="paraphrase: ",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print_comparison(f"{sample['name']} ({elapsed:.0f}ms)", sample["text"], result)

    print("  → Check: Are rewrites genuinely different, or just synonym swaps?")


async def test_2_run_variation():
    """Test: Do different runs produce different outputs?"""
    print_header("TEST 2: Run Variation — 3 runs on same input")

    from humanizer.inference.inference_layer import generate_text

    sample = TEST_INPUTS[0]
    outputs = []
    for i in range(3):
        result = await generate_text(
            "flan-t5-base",
            sample["text"],
            prefix="paraphrase: ",
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
    """Test: Do style profiles produce noticeably different text?"""
    print_header("TEST 3: Style Profiles — Same input, different styles")

    from humanizer.inference.inference_layer import generate_text
    from humanizer.style.style_profiles import get_all_profiles

    sample = TEST_INPUTS[0]
    profiles = get_all_profiles()

    results = {}
    for name, sv in profiles.items():
        # Craft style-aware prefix based on profile
        if sv.formality > 0.7:
            prefix = "Rewrite this formally and academically: "
        elif sv.formality < 0.3:
            prefix = "Rewrite this casually and conversationally: "
        elif sv.density > 0.7:
            prefix = "Rewrite this concisely: "
        elif sv.density < 0.3:
            prefix = "Rewrite this with more detail and explanation: "
        else:
            prefix = "Paraphrase this: "

        result = await generate_text("flan-t5-base", sample["text"], prefix=prefix)
        results[name] = result
        print(f"  [{name}] (f={sv.formality}, d={sv.density}):")
        print(f"    {result[:200]}...")
        print()

    # Check diversity across profiles
    unique = len(set(results.values()))
    print(f"  → Unique outputs: {unique}/{len(profiles)} profiles")
    if unique <= 2:
        print("  ⚠️  Profiles produce similar output — style conditioning needs work")
    else:
        print("  ✓  Profiles produce varied output")


async def test_4_multi_pass():
    """Test: Does the multi-pass pipeline produce genuine rewrites?"""
    print_header("TEST 4: Multi-Pass Pipeline — 4-pass single style")

    from humanizer.pipeline.multi_pass import _run_single_pipeline
    from humanizer.style.style_vector import StyleVector

    sample = TEST_INPUTS[0]
    sv = StyleVector()

    t0 = time.perf_counter()
    result = await _run_single_pipeline(
        sample["text"],
        sv,
        model="flan-t5-base",
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print_comparison(f"4-pass pipeline ({elapsed:.0f}ms)", sample["text"], result)
    print("  → Check: Is this genuinely reconstructed, or surface-level?")


async def test_5_prompt_strategies():
    """Test: Which prompt prefix works best for FLAN-T5?"""
    print_header("TEST 5: Prompt Strategy Comparison")

    from humanizer.inference.inference_layer import generate_text

    sample = TEST_INPUTS[0]["text"]
    prefixes = {
        "paraphrase": "paraphrase: ",
        "rewrite": "Rewrite the following text in different words: ",
        "restructure": "Restructure and rewrite this paragraph with different sentence structure: ",
        "human-like": "Rewrite this to sound like a human wrote it from memory: ",
        "simplify": "Explain this in your own words: ",
    }

    for name, prefix in prefixes.items():
        result = await generate_text("flan-t5-base", sample, prefix=prefix)
        print(f"  [{name}]")
        print(f"    {result[:200]}...")
        print()

    print("  → Which prefix produces the most natural, structurally different output?")


async def test_6_full_validation():
    """Test: Semantic validation on real outputs."""
    print_header("TEST 6: Semantic Validation on Real Output")

    from humanizer.inference.inference_layer import generate_text
    from humanizer.validation.semantic_checks import validate_semantic

    sample = TEST_INPUTS[0]
    rewritten = await generate_text(
        "flan-t5-base",
        sample["text"],
        prefix="Rewrite this paragraph in completely different words and structure: ",
    )

    validation = validate_semantic(sample["text"], rewritten)

    print(f"  Semantic Similarity:  {validation.semantic_similarity:.3f}  (want: 0.75–0.90)")
    print(f"  Lexical Novelty:     {validation.lexical_novelty:.3f}  (want: 0.55–0.72)")
    print(f"  NLI Label:           {validation.nli_label}  (want: entailment)")
    print(f"  NLI Score:           {validation.nli_entailment_score:.3f}")
    print(f"  Entity Preserved:    {validation.entity_preserved}")
    print(f"  Readability Delta:   {validation.readability_delta:.1f}  (want: < 14)")
    print(f"  Overall Passed:      {validation.passed}")
    print()
    print(f"  ORIGINAL:  {sample['text'][:150]}...")
    print(f"  REWRITTEN: {rewritten[:150]}...")


async def main():
    print_header("HUMANIZER v2.0 — PIPELINE VALIDATION")
    print("  Running 6 real-inference tests on FLAN-T5-base.")
    print("  This will download the model (~990MB) on first run.\n")

    # Update model registry to use flan-t5-base (smaller, instruction-tuned)
    MODEL_REGISTRY["flan-t5-base"] = "google/flan-t5-base"

    t_start = time.perf_counter()

    await test_1_basic_rewrite()
    await test_2_run_variation()
    await test_3_style_profiles()
    await test_4_multi_pass()
    await test_5_prompt_strategies()
    await test_6_full_validation()

    total = time.perf_counter() - t_start
    print_header(f"VALIDATION COMPLETE — Total time: {total:.1f}s")
    print("  Review the outputs above and confirm:")
    print("    1. Outputs are structurally different from inputs")
    print("    2. Different runs produce varied outputs")
    print("    3. Style profiles create noticeably different text")
    print("    4. Multi-pass pipeline produces genuine rewrites")
    print("    5. Validation scores are within expected ranges")
    print()
    print("  If outputs are shallow/synonym-only, we'll need to:")
    print("    - Switch to FLAN-T5-large or T5-large")
    print("    - Adjust prompts (Test 5 shows which works best)")
    print("    - Add chain-of-thought prompting for deeper restructuring")


if __name__ == "__main__":
    asyncio.run(main())
