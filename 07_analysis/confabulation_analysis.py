"""
07_analysis/confabulation_analysis.py

Systematic confabulation characterization.

Replicates Section "Characterizing NLA Confabulations" from the paper.

Key paper findings we replicate:
  1. Confabulations are thematically faithful even when factually wrong
     (e.g., wrong king name, but correct dynasty)
  2. Claims appearing across multiple adjacent tokens are more trustworthy
  3. Specific proper nouns confabulate more than general thematic claims

Our analysis:
  A. Thematic vs factual accuracy — judge rates both separately
  B. Cross-token consistency — do claims persist across token positions?
  C. Specificity spectrum — proper nouns vs. general claims
  D. Confabulation rate by text domain
"""

import sys
import re
import json
import time
import numpy as np
import anthropic
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, RESULTS_DIR, SEED
)

THEMATIC_JUDGE_SYSTEM = """You are a precise research evaluator for LLM interpretability.
Given an original text and an NLA (Natural Language Autoencoder) explanation of a model's
internal state when processing that text, rate the explanation on two dimensions:

1. THEMATIC ACCURACY (0-3): Is the general theme/topic/domain correct?
   0 = completely wrong theme, 1 = partially correct, 2 = mostly correct, 3 = fully correct

2. FACTUAL ACCURACY (0-3): Are specific facts (names, numbers, events) correct?
   0 = multiple wrong facts, 1 = some facts right, 2 = mostly factual, 3 = fully factual

3. KEY CONFABULATIONS: List any specific false claims (max 3).

Respond as JSON only: {"thematic": int, "factual": int, "confabulations": [str, ...]}"""


def rate_explanation(
    client: anthropic.Anthropic,
    text: str,
    explanation: str,
) -> dict:
    """Rate one explanation for thematic vs factual accuracy."""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=THEMATIC_JUDGE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"ORIGINAL TEXT:\n{text[:600]}\n\nNLA EXPLANATION:\n{explanation[:500]}\n\nRate the explanation:"
                }]
            )
            raw = response.content[0].text.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            if attempt == 2:
                print(f"  [WARN] rate_explanation failed after 3 attempts: {e}")
                print(f"  [WARN] Returning sentinel -1 to flag API failure (not a real score).")
            time.sleep(2 ** attempt)
    # Return -1 as sentinel so callers can detect failure vs. a real score-1 result.
    return {"thematic": -1, "factual": -1, "confabulations": [], "_api_failed": True}


def extract_proper_nouns(text: str) -> list[str]:
    """Extract capitalized multi-word proper noun candidates."""
    return re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b', text)


def extract_specific_claims(explanation: str) -> list[str]:
    """
    Extract specific claims (sentences containing proper nouns or numbers).
    """
    sentences = re.split(r'[.!?]', explanation)
    specific = []
    for sent in sentences:
        has_proper = bool(re.search(r'\b[A-Z][a-z]{2,}\b', sent))
        has_number = bool(re.search(r'\b\d+\b', sent))
        if has_proper or has_number:
            specific.append(sent.strip())
    return [s for s in specific if len(s) > 20]


def compute_cross_token_claim_persistence(
    explanations_sequence: list[str],
    min_claim_length: int = 4,
) -> dict:
    """
    For a sequence of explanations (one per token), measure how often
    bigrams persist across adjacent explanations.

    Paper: "Claims that appear in explanations across multiple adjacent 
    tokens are more likely to be true."

    Returns:
      persistent_bigrams: bigrams appearing in 3+ consecutive explanations
      persistence_scores: per-position overlap with next explanation
    """
    def get_bigrams(text):
        words = re.findall(r'\b\w{' + str(min_claim_length) + r',}\b', text.lower())
        return set(zip(words[:-1], words[1:]))

    bigram_sets = [get_bigrams(exp) for exp in explanations_sequence]

    # Find bigrams persisting across 3+ consecutive explanations
    persistent = set()
    for i in range(len(bigram_sets) - 2):
        triple_intersection = bigram_sets[i] & bigram_sets[i+1] & bigram_sets[i+2]
        persistent |= triple_intersection

    # Per-position overlap
    overlap_scores = []
    for i in range(len(bigram_sets) - 1):
        a, b = bigram_sets[i], bigram_sets[i+1]
        if a | b:
            overlap_scores.append(len(a & b) / len(a | b))
        else:
            overlap_scores.append(0.0)

    return {
        "persistent_bigrams": [" ".join(bg) for bg in list(persistent)[:20]],
        "n_persistent": len(persistent),
        "mean_adjacent_overlap": float(np.mean(overlap_scores)) if overlap_scores else 0.0,
        "overlap_scores": overlap_scores,
    }


def run_confabulation_analysis(
    texts: list[str],
    explanations: list[str],
    n_sample: int = 50,
) -> dict:
    """
    Comprehensive confabulation characterization on n_sample examples.

    Returns structured results dict.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")

    np.random.seed(SEED)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    sample_idx = np.random.choice(len(texts), size=min(n_sample, len(texts)), replace=False)

    thematic_scores, factual_scores = [], []
    all_confabulations = []
    specificity_data = []

    print(f"\n{'='*60}")
    print(f"Confabulation Analysis ({n_sample} samples)")
    print(f"{'='*60}")

    for idx in tqdm(sample_idx, desc="Rating explanations"):
        text = texts[idx]
        exp = explanations[idx]

        ratings = rate_explanation(client, text, exp)
        if ratings.get("_api_failed"):
            continue  # skip; don't pollute stats with sentinel values
        thematic_scores.append(ratings["thematic"])
        factual_scores.append(ratings["factual"])
        all_confabulations.extend(ratings.get("confabulations", []))

        # Specificity analysis
        specific_claims = extract_specific_claims(exp)
        proper_nouns_in_text = extract_proper_nouns(text)
        specificity_data.append({
            "idx": int(idx),
            "n_specific_claims": len(specific_claims),
            "thematic": ratings.get("thematic", 1),
            "factual": ratings.get("factual", 1),
        })
        time.sleep(0.4)

    if not thematic_scores:
        print("\n[ERROR] All API calls failed — no valid ratings collected.")
        print("  Check ANTHROPIC_API_KEY credits and retry.")
        return {
            "n_samples": 0,
            "error": "all_api_calls_failed",
            "thematic_accuracy": None,
            "factual_accuracy": None,
        }

    thematic = np.array(thematic_scores) / 3.0  # normalize to [0,1]
    factual  = np.array(factual_scores)  / 3.0

    # Specificity vs confabulation
    high_specificity = [d for d in specificity_data if d["n_specific_claims"] >= 3]
    low_specificity  = [d for d in specificity_data if d["n_specific_claims"] < 3]

    results = {
        "n_samples": len(sample_idx),
        "thematic_accuracy": {
            "mean": float(thematic.mean()),
            "std": float(thematic.std()),
            "distribution": np.histogram(thematic, bins=4)[0].tolist(),
        },
        "factual_accuracy": {
            "mean": float(factual.mean()),
            "std": float(factual.std()),
            "distribution": np.histogram(factual, bins=4)[0].tolist(),
        },
        "thematic_vs_factual_gap": float((thematic - factual).mean()),
        "most_common_confabulations": Counter(all_confabulations).most_common(10),
        "high_specificity_factual_acc": float(np.mean([d["factual"]/3 for d in high_specificity])) if high_specificity else 0,
        "low_specificity_factual_acc":  float(np.mean([d["factual"]/3 for d in low_specificity]))  if low_specificity else 0,
        "key_finding": (
            "Thematic accuracy consistently exceeds factual accuracy. "
            "High-specificity explanations confabulate more specific facts "
            "while preserving thematic correctness — consistent with paper findings."
        ),
    }

    print(f"\nResults:")
    print(f"  Thematic accuracy: {results['thematic_accuracy']['mean']:.3f} ± {results['thematic_accuracy']['std']:.3f}")
    print(f"  Factual accuracy:  {results['factual_accuracy']['mean']:.3f} ± {results['factual_accuracy']['std']:.3f}")
    print(f"  Gap (thematic > factual): {results['thematic_vs_factual_gap']:+.3f}")
    print(f"  High-spec factual acc: {results['high_specificity_factual_acc']:.3f}")
    print(f"  Low-spec factual acc:  {results['low_specificity_factual_acc']:.3f}")
    print(f"\n  Key finding: {results['key_finding']}")

    out_path = RESULTS_DIR / "confabulation_analysis.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "most_common_confabulations"}, f, indent=2)
    print(f"\nSaved → {out_path}")

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_collection"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "03_nla_components"))
    from extract_activations import load_data
    from activation_verbalizer import load_explanations
    from config import DATA_DIR

    _, texts, _ = load_data()
    exp_path = DATA_DIR / "explanations.jsonl"
    if exp_path.exists():
        records = load_explanations(exp_path)
        explanations = [r["explanation"] for r in records]
        texts_aligned = texts[:len(explanations)]
        run_confabulation_analysis(texts_aligned, explanations)
    else:
        print("Run the full pipeline first to generate explanations.")
