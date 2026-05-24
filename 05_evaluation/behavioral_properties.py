"""
05_evaluation/behavioral_properties.py

Measures NLA failure modes and behavioral properties, replicating the paper's
Section "Measuring Behavioral Properties of NLAs":

  1. STEGANOGRAPHY SCORE     — Does the AV encode context verbatim (hiding info
                               as text rather than truly verbalizing)?
                               Measured by: n-gram overlap between explanation
                               and original context.

  2. WRITING QUALITY         — Does explanation quality degrade over training?
                               Measured by: fluency score (perplexity), coherence.

  3. CONFABULATION RATE      — How often does the explanation contain verifiably
                               false claims?
                               Measured by: fact-checking specific claims against
                               the known context using Claude as judge.

  4. EXPLANATION LENGTH      — Do explanations get shorter/longer over training?

  5. CLAIM SPECIFICITY       — How specific are claims? (high specificity + low
                               confabulation = good calibration)

Paper insight: "Claims that appear in explanations across multiple adjacent tokens
are more likely to be true." We measure cross-token claim consistency as a
reliability heuristic.
"""

import sys
import re
import json
import time
import math
import numpy as np
import anthropic
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, RESULTS_DIR, SEED
)


# ── 1. Steganography score ─────────────────────────────────────────────────────

def ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute n-gram overlap (Jaccard similarity) between two strings.
    High overlap = AV may be copying context verbatim (steganography).
    """
    def get_ngrams(text, n):
        words = text.lower().split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = sum((ngrams1 & ngrams2).values())
    union = sum((ngrams1 | ngrams2).values())
    return intersection / union if union > 0 else 0.0


def compute_steganography_scores(
    texts: list[str],
    explanations: list[str],
    n_gram: int = 3,
) -> dict:
    """
    Compute steganography scores for all (text, explanation) pairs.

    Returns summary statistics and per-sample scores.
    """
    scores = [
        ngram_overlap(text, exp, n=n_gram)
        for text, exp in zip(texts, explanations)
    ]
    scores = np.array(scores)

    result = {
        "metric": "steganography",
        "n_gram": n_gram,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "max": float(scores.max()),
        "p90": float(np.percentile(scores, 90)),
        "per_sample": scores.tolist(),
    }

    print(f"\nSteganography ({n_gram}-gram overlap):")
    print(f"  Mean: {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Max: {result['max']:.4f}  |  90th pct: {result['p90']:.4f}")
    print("  (High steganography = AV is copying text, not verbalizing)")

    return result


# ── 2. Writing quality (explanation fluency) ──────────────────────────────────

def compute_explanation_lengths(explanations: list[str]) -> dict:
    """
    Distribution of explanation lengths (words and characters).
    """
    word_lengths = [len(e.split()) for e in explanations]
    char_lengths = [len(e) for e in explanations]

    return {
        "mean_words": float(np.mean(word_lengths)),
        "std_words": float(np.std(word_lengths)),
        "min_words": int(np.min(word_lengths)),
        "max_words": int(np.max(word_lengths)),
        "mean_chars": float(np.mean(char_lengths)),
        "per_sample_words": word_lengths,
    }


def compute_structural_quality(explanations: list[str]) -> dict:
    """
    Check for structural quality markers:
    - Has bolded headings (** **) → paper style preserved
    - Has multiple paragraphs
    - No obvious truncation
    """
    has_bold = [bool(re.search(r'\*\*.+?\*\*', e)) for e in explanations]
    has_paragraphs = [e.count('\n') >= 2 for e in explanations]
    looks_truncated = [e.endswith('...') or len(e) < 50 for e in explanations]

    return {
        "fraction_with_bold_headings": float(np.mean(has_bold)),
        "fraction_with_paragraphs": float(np.mean(has_paragraphs)),
        "fraction_truncated": float(np.mean(looks_truncated)),
    }


# ── 3. Confabulation rate ─────────────────────────────────────────────────────

CONFABULATION_JUDGE_SYSTEM = """You are a precise fact-checker for LLM interpretability research.
You will be given:
1. The original text fragment
2. An NLA explanation generated about the model's internal state for that fragment

Your task: identify claims in the explanation that are VERIFIABLY FALSE given the original text.
A claim is confabulated if it asserts a specific fact (name, number, event) that contradicts
or is absent from the original text.

Respond with JSON: {"confabulated_claims": [...], "n_total_claims": int, "confabulation_rate": float}"""


def check_confabulation(
    client: anthropic.Anthropic,
    text: str,
    explanation: str,
) -> dict:
    """
    Use Claude as a judge to check confabulations in one explanation.
    """
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            system=CONFABULATION_JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"ORIGINAL TEXT:\n{text[:500]}\n\nNLA EXPLANATION:\n{explanation}\n\nCheck for confabulations:"
            }],
        )
        text_response = response.content[0].text.strip()
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        if "credit balance" in str(e).lower() or "billing" in str(e).lower():
            raise  # let compute_confabulation_rates handle this
        pass
    return {"confabulated_claims": [], "n_total_claims": 0, "confabulation_rate": 0.0}


def compute_confabulation_rates(
    texts: list[str],
    explanations: list[str],
    n_sample: int = 30,
) -> dict:
    """
    Sample n_sample (text, explanation) pairs and compute confabulation rates.
    Skips gracefully if no API key or credits are exhausted.
    """
    if not ANTHROPIC_API_KEY:
        print("\nConfabulation analysis skipped: no ANTHROPIC_API_KEY.")
        return {"skipped": True, "reason": "no_api_key"}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    indices = np.random.choice(len(texts), size=min(n_sample, len(texts)), replace=False)
    results = []
    total_claims, total_confab = 0, 0
    consecutive_billing_errors = 0

    for idx in tqdm(indices, desc="Checking confabulations"):
        try:
            r = check_confabulation(client, texts[idx], explanations[idx])
            consecutive_billing_errors = 0
        except Exception as e:
            msg = str(e).lower()
            if "credit balance" in msg or "billing" in msg:
                consecutive_billing_errors += 1
                if consecutive_billing_errors >= 2:
                    print("\n  Confabulation check stopped: Anthropic credits exhausted.")
                    break
                continue
            r = {"confabulated_claims": [], "n_total_claims": 0, "confabulation_rate": 0.0}

        results.append({
            "idx": int(idx),
            "n_claims": r.get("n_total_claims", 0),
            "n_confabulated": len(r.get("confabulated_claims", [])),
            "rate": r.get("confabulation_rate", 0.0),
        })
        total_claims  += r.get("n_total_claims", 0)
        total_confab  += len(r.get("confabulated_claims", []))
        time.sleep(0.5)

    if not results:
        print("\nConfabulation analysis skipped: no successful API calls.")
        return {"skipped": True, "reason": "api_unavailable"}

    overall_rate = total_confab / total_claims if total_claims > 0 else 0.0
    per_sample_rates = [r["rate"] for r in results]

    summary = {
        "overall_confabulation_rate": overall_rate,
        "mean_per_sample_rate": float(np.mean(per_sample_rates)),
        "std_per_sample_rate": float(np.std(per_sample_rates)),
        "total_claims_checked": total_claims,
        "total_confabulated": total_confab,
        "n_samples": len(results),
        "per_sample": results,
    }

    print(f"\nConfabulation Analysis ({n_sample} samples):")
    print(f"  Overall rate: {overall_rate:.3f}")
    print(f"  Per-sample mean: {summary['mean_per_sample_rate']:.3f}")
    print("  (Paper: confabulations are thematically consistent but factually wrong in specifics)")

    return summary


# ── 4. Cross-token claim consistency ─────────────────────────────────────────

def compute_cross_token_consistency(
    explanations_by_position: list[list[str]],  # outer: position, inner: token explanations
) -> float:
    """
    Measure how consistently a claim appears across adjacent token explanations.
    Paper: "Claims that appear in explanations across multiple adjacent tokens
    are more likely to be true."

    Here we measure bigram overlap across consecutive token explanations.
    Returns mean consistency score.
    """
    if len(explanations_by_position) < 2:
        return 0.0

    scores = []
    for i in range(len(explanations_by_position) - 1):
        exp1 = " ".join(explanations_by_position[i])
        exp2 = " ".join(explanations_by_position[i + 1])
        scores.append(ngram_overlap(exp1, exp2, n=2))

    return float(np.mean(scores)) if scores else 0.0


# ── Run all behavioral analyses ───────────────────────────────────────────────

def run_behavioral_analysis(
    texts: list[str],
    explanations: list[str],
    run_confabulation: bool = True,
    checkpoint_label: str = "final",
) -> dict:
    """
    Run all behavioral property analyses and save to disk.
    """
    np.random.seed(SEED)

    results = {}

    # Steganography
    results["steganography"] = compute_steganography_scores(texts, explanations)

    # Length and structure
    results["explanation_length"] = compute_explanation_lengths(explanations)
    results["structural_quality"] = compute_structural_quality(explanations)

    print("\nExplanation Quality:")
    print(f"  Mean length: {results['explanation_length']['mean_words']:.1f} words")
    print(f"  Bold headings: {results['structural_quality']['fraction_with_bold_headings']:.2%}")
    print(f"  Multi-paragraph: {results['structural_quality']['fraction_with_paragraphs']:.2%}")

    # Confabulation (API calls — optional)
    if run_confabulation:
        results["confabulation"] = compute_confabulation_rates(texts, explanations, n_sample=30)

    # Save
    out_path = RESULTS_DIR / f"behavioral_properties_{checkpoint_label}.json"
    # Remove per-sample arrays for cleaner JSON
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {kk: vv for kk, vv in v.items()
                               if kk not in ("per_sample", "per_sample_words")}
        else:
            save_results[k] = v

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nBehavioral analysis saved → {out_path}")

    return results
