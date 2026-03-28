"""
Humanizer v2.0 — Pipeline Validation Script (v3)
──────────────────────────────────────────────────
Sentence-level rewriting pipeline validation.

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


def compute_metrics(original: str, rewritten: str) -> dict:
    """Compute overlap metrics."""
    orig_words = set(original.lower().split())
    rewrite_words = set(rewritten.lower().split())
    word_overlap = len(orig_words & rewrite_words) / max(len(orig_words), 1)

    def ngrams(text, n=3):
        tokens = text.lower().split()
        return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    ng_orig = ngrams(original)
    ng_rewrite = ngrams(rewritten)
    ng_overlap = len(ng_orig & ng_rewrite) / max(len(ng_orig), 1) if ng_orig else 0

    return {
        "word_overlap": word_overlap,
        "ng_overlap": ng_overlap,
        "orig_words": len(original.split()),
        "rewrite_words": len(rewritten.split()),
        "length_ratio": len(rewritten.split()) / max(len(original.split()), 1),
    }


def print_comparison(label: str, original: str, rewritten: str) -> None:
    """Print side-by-side comparison with metrics."""
    m = compute_metrics(original, rewritten)
    len_status = "✓" if m["length_ratio"] > 0.6 else "⚠️"
    word_status = "✓" if m["word_overlap"] < 0.5 else "⚠️"
    ng_status = "✓" if m["ng_overlap"] < 0.3 else "⚠️"

    print(f"  [{label}]")
    print(f"  ORIGINAL  ({m['orig_words']} words):")
    print(f"    {original[:200]}{'...' if len(original) > 200 else ''}")
    print(f"  REWRITTEN ({m['rewrite_words']} words):")
    print(f"    {rewritten[:200]}{'...' if len(rewritten) > 200 else ''}")
    print(f"  {word_status} Word overlap:   {m['word_overlap']:.1%}  (want <50%)")
    print(f"  {ng_status} 3-gram overlap: {m['ng_overlap']:.1%}  (want <30%)")
    print(f"  {len_status} Length ratio:   {m['length_ratio']:.1%}  (want >60%)")
    print()


async def test_1_four_pass_rewrite():
    """Test: Does the 4-pass pipeline produce structurally different outputs?"""
    print_header("TEST 1: 4-Pass Pipeline — All Input Types")

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


async def test_2_run_variation():
    """Test: Do different runs produce different outputs?"""
    print_header("TEST 2: Run Variation — 3 runs on same input")

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
        print(f"  Run {i+1} ({len(result.split())}w): {result[:150]}...")

    unique = len(set(outputs))
    print(f"\n  → Unique outputs: {unique}/3")
    status = "✓" if unique >= 2 else "⚠️"
    print(f"  {status}  {'Outputs vary' if unique >= 2 else 'All identical'}")


async def test_3_style_profiles():
    """Test: Do style profiles produce noticeably different text?"""
    print_header("TEST 3: Style Profiles — Same input, 6 styles")

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
        m = compute_metrics(sample["text"], result)
        print(f"  [{name}] ({elapsed:.0f}ms, {len(result.split())}w, "
              f"word={m['word_overlap']:.0%}, 3g={m['ng_overlap']:.0%}):")
        print(f"    {result[:180]}...")
        print()

    unique = len(set(results.values()))
    status = "✓" if unique >= 4 else "⚠️"
    print(f"  → Unique outputs: {unique}/{len(profiles)} profiles")
    print(f"  {status}  Profiles {'produce varied output' if unique >= 4 else 'need more differentiation'}")


async def test_4_per_pass_trace():
    """Test: Show what each pass produces."""
    print_header("TEST 4: Per-Pass Trace — What each pass does")

    from humanizer.style.style_vector import StyleVector
    from humanizer.pipeline.multi_pass import (
        _pass1_sentence_paraphrase, _pass2_restructure,
        _pass3_style_rewrite, _pass4_polish,
    )

    sample = TEST_INPUTS[0]
    sv = StyleVector()

    print(f"  ORIGINAL ({len(sample['text'].split())} words):")
    print(f"    {sample['text'][:250]}...")
    print()

    # Pass 1
    t0 = time.perf_counter()
    p1 = await _pass1_sentence_paraphrase(sample["text"], "flan-t5-xl")
    m1 = compute_metrics(sample["text"], p1)
    print(f"  PASS 1 — Sentence Paraphrase ({(time.perf_counter()-t0)*1000:.0f}ms, "
          f"{len(p1.split())}w, word={m1['word_overlap']:.0%}):")
    print(f"    {p1[:300]}{'...' if len(p1) > 300 else ''}")
    print()

    # Pass 2
    t0 = time.perf_counter()
    p2 = await _pass2_restructure(p1, sv, "flan-t5-xl")
    m2 = compute_metrics(sample["text"], p2)
    print(f"  PASS 2 — Restructure ({(time.perf_counter()-t0)*1000:.0f}ms, "
          f"{len(p2.split())}w, word={m2['word_overlap']:.0%}):")
    print(f"    {p2[:300]}{'...' if len(p2) > 300 else ''}")
    print()

    # Pass 3
    t0 = time.perf_counter()
    p3 = await _pass3_style_rewrite(p2, sv, "flan-t5-xl")
    m3 = compute_metrics(sample["text"], p3)
    print(f"  PASS 3 — Style Rewrite ({(time.perf_counter()-t0)*1000:.0f}ms, "
          f"{len(p3.split())}w, word={m3['word_overlap']:.0%}):")
    print(f"    {p3[:300]}{'...' if len(p3) > 300 else ''}")
    print()

    # Pass 4
    t0 = time.perf_counter()
    p4 = await _pass4_polish(p3, sv, "flan-t5-xl")
    m4 = compute_metrics(sample["text"], p4)
    print(f"  PASS 4 — Polish ({(time.perf_counter()-t0)*1000:.0f}ms, "
          f"{len(p4.split())}w, word={m4['word_overlap']:.0%}):")
    print(f"    {p4[:300]}{'...' if len(p4) > 300 else ''}")
    print()

    print(f"  ── End-to-end comparison ──")
    print_comparison("4-pass total", sample["text"], p4)


async def test_5_sentence_vs_paragraph():
    """Test: Compare sentence-level vs paragraph-level rewriting."""
    print_header("TEST 5: Sentence-Level vs Paragraph-Level Rewrite")

    from humanizer.inference.inference_layer import generate_text

    sample = TEST_INPUTS[0]["text"]

    # Paragraph-level (old approach — known to fail)
    print("  [Paragraph-level] — full text sent as one block:")
    para_result = await generate_text(
        "flan-t5-xl", sample,
        prefix="paraphrase: ",
    )
    m_para = compute_metrics(sample, para_result)
    print(f"    ({len(para_result.split())}w, word={m_para['word_overlap']:.0%}, "
          f"3g={m_para['ng_overlap']:.0%})")
    print(f"    {para_result[:180]}...")
    print()

    # Sentence-level (new approach)
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sample) if s.strip()]
    sent_results = []
    for sent in sentences:
        r = await generate_text("flan-t5-xl", sent, prefix="paraphrase: ")
        sent_results.append(r.strip())
    sent_combined = " ".join(sent_results)

    m_sent = compute_metrics(sample, sent_combined)
    print(f"  [Sentence-level] — each sentence paraphrased individually:")
    print(f"    ({len(sent_combined.split())}w, word={m_sent['word_overlap']:.0%}, "
          f"3g={m_sent['ng_overlap']:.0%})")
    print(f"    {sent_combined[:180]}...")
    print()

    print(f"  → Paragraph word overlap: {m_para['word_overlap']:.0%}")
    print(f"  → Sentence  word overlap: {m_sent['word_overlap']:.0%}")
    if m_sent['word_overlap'] < m_para['word_overlap']:
        print("  ✓ Sentence-level produces more diverse output")
    else:
        print("  ⚠️ Paragraph-level was better (unexpected)")


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

    m = compute_metrics(sample["text"], rewritten)

    sim_status = "✓" if 0.70 <= validation.semantic_similarity <= 0.95 else "⚠️"
    nov_status = "✓" if validation.lexical_novelty >= 0.40 else "⚠️"
    nli_status = "✓" if validation.nli_label == "entailment" else "⚠️"
    ent_status = "✓" if validation.entity_preserved else "⚠️"
    read_status = "✓" if validation.readability_delta < 14 else "⚠️"
    len_status = "✓" if m["length_ratio"] > 0.6 else "⚠️"

    print(f"  {sim_status} Semantic Similarity:  {validation.semantic_similarity:.3f}  (want: 0.70–0.95)")
    print(f"  {nov_status} Lexical Novelty:     {validation.lexical_novelty:.3f}  (want: ≥0.40)")
    print(f"  {nli_status} NLI Label:           {validation.nli_label}  (want: entailment)")
    print(f"    NLI Score:           {validation.nli_entailment_score:.3f}")
    print(f"  {ent_status} Entity Preserved:    {validation.entity_preserved}")
    print(f"  {read_status} Readability Delta:   {validation.readability_delta:.1f}  (want: < 14)")
    print(f"  {len_status} Length Ratio:        {m['length_ratio']:.0%}  (want: >60%)")
    print(f"    Overall Passed:      {validation.passed}")
    print()
    print(f"  ORIGINAL:  {sample['text'][:150]}...")
    print(f"  REWRITTEN: {rewritten[:150]}...")


async def main():
    print_header("HUMANIZER v2.0 — PIPELINE VALIDATION (v3)")
    print("  Running 6 tests with FLAN-T5-XL (L4 GPU).")
    print("  Key changes in v3:")
    print("    • Sentence-level paraphrase in Pass 1 & 3 (short inputs work)")
    print("    • Rule-based restructuring in Pass 2 (reorder + merge)")
    print("    • Truncation guard in Pass 4 (reject <50% length)")
    print("    • Short T5-native prefixes ('paraphrase:', 'Rephrase:')")
    print("    • Test 5: direct comparison sentence-level vs paragraph-level")
    print()

    MODEL_REGISTRY["flan-t5-xl"] = "google/flan-t5-xl"

    t_start = time.perf_counter()

    await test_1_four_pass_rewrite()
    await test_2_run_variation()
    await test_3_style_profiles()
    await test_4_per_pass_trace()
    await test_5_sentence_vs_paragraph()
    await test_6_full_validation()

    total = time.perf_counter() - t_start
    print_header(f"VALIDATION COMPLETE — Total time: {total:.1f}s")
    print("  Target metrics:")
    print("    • Word overlap:        <50%")
    print("    • 3-gram overlap:      <30%")
    print("    • Length ratio:        >60% (no severe truncation)")
    print("    • Semantic similarity: 0.70-0.95")
    print("    • Lexical novelty:     ≥0.40")
    print("    • NLI:                 entailment")


if __name__ == "__main__":
    asyncio.run(main())
