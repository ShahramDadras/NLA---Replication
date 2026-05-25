"""
06_case_studies/language_switching.py

Replicates the "Language Switching" case study from Section 3.2 of the paper.

Original finding: When Opus 4.6 switches to responding in a foreign language,
NLA explanations show persistent internal representations of that language
well before the model starts producing foreign-language tokens.

Our adaptation on GPT-2:
  - Create multilingual prompts where GPT-2 is likely to switch register
    or generate foreign-language tokens
  - For each token position, verbalize the activation
  - Track frequency of target-language mentions in explanations
  - Compare: explanations in "language-switching" context vs neutral context

Additionally, we test with deliberately multilingual contexts:
  - Contexts seeded with French/Spanish/German words
  - Track whether NLA explanations reflect the embedded language

Paper figure: smooth Gaussian-weighted line plot of language mentions
over token positions across multiple transcripts. We replicate this.
"""

import sys
import re
import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TARGET_MODEL_NAME, TARGET_LAYER, RESULTS_DIR,
    AI_PROVIDER, SEED
)

# ── Language test cases ────────────────────────────────────────────────────────

LANGUAGE_CONTEXTS = {
    "french": {
        "prompts": [
            "The French ambassador said 'Bonjour' to the crowd. He continued speaking and",
            "In Paris, the café au lait was served by a garçon who",
            "She learned to say 'merci beaucoup' and 'au revoir' before her trip, and when she arrived",
        ],
        "keywords": ["french", "france", "paris", "francais", "français"],
        "language_tokens": ["bonjour", "merci", "oui", "non", "parle", "vous"],
    },
    "spanish": {
        "prompts": [
            "El presidente announced the new policy to the crowd. The people responded with",
            "In Madrid, she ordered 'café con leche' and the waiter said 'gracias'. She",
            "The Spanish word 'mañana' means tomorrow, but culturally it implies",
        ],
        "keywords": ["spanish", "spain", "español", "madrid", "mexico"],
        "language_tokens": ["hola", "gracias", "señor", "por", "favor", "como"],
    },
    "german": {
        "prompts": [
            "The Berlin conference opened with 'Guten Morgen' from the chancellor. He then",
            "In German-speaking countries, 'Kindergarten' and 'Zeitgeist' are everyday words that",
            "She studied German and could say 'Ich verstehe' when the professor",
        ],
        "keywords": ["german", "germany", "deutsch", "berlin", "german"],
        "language_tokens": ["ich", "das", "ist", "ein", "und", "nicht"],
    },
    "neutral": {
        "prompts": [
            "The scientist published a paper about quantum computing and the results showed that",
            "In the library, the student found a book about ancient history and",
            "The weather forecast predicted rain for the entire week, so the farmers",
        ],
        "keywords": [],
        "language_tokens": [],
    }
}


def extract_all_token_activations(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    layer: int = TARGET_LAYER,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """
    Extract activations for ALL token positions in a text.
    Returns:
      activations : (seq_len, d_model)
      tokens      : list of token strings
    """
    storage = {}
    def hook(module, input, output):
        storage["hidden"] = output[0].detach().cpu()

    handle = model.transformer.h[layer].register_forward_hook(hook)
    enc = tokenizer(text, return_tensors="pt", max_length=100, truncation=True).to(device)

    with torch.no_grad():
        model(**enc)
    handle.remove()

    h = storage["hidden"]
    # Defensive: some transformers versions give (batch, seq, d), others (seq, d)
    if h.dim() == 3:
        activations = h[0].numpy().astype(np.float32)   # (seq_len, d_model)
    else:
        activations = h.numpy().astype(np.float32)       # already (seq_len, d_model)

    # Normalize each token's activation
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    activations = activations / np.maximum(norms, 1e-8)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
    return activations, tokens


def verbalize_activation_batch(
    activations: np.ndarray,
    tokens: list[str],
    positions: list[int],
    layer: int = TARGET_LAYER,
    provider: str = AI_PROVIDER,
) -> list[str]:
    """
    Verbalize activations at specified positions using the configured AV provider.
    Returns list of explanation strings (one per position).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "03_nla_components"))
    from activation_verbalizer import ActivationVerbalizer
    av = ActivationVerbalizer(layer=layer, provider=provider)

    explanations = []
    for pos in tqdm(positions, desc="  Verbalizing tokens"):
        exp = av.verbalize(activations[pos], tokens[:pos + 1])
        explanations.append(exp)
    return explanations


def count_language_mentions(explanation: str, keywords: list[str]) -> int:
    """Count how many times language keywords appear in an explanation."""
    exp_lower = explanation.lower()
    return sum(1 for kw in keywords if kw in exp_lower)


def gaussian_smooth(values: np.ndarray, window_fraction: float = 0.05) -> np.ndarray:
    """
    Apply Gaussian-weighted smoothing.
    Paper: "smoothed with a Gaussian-weighted average over windows with 
    length equal to 5% of the transcript."
    """
    n = len(values)
    window = max(3, int(window_fraction * n))
    sigma = window / 3.0

    smoothed = np.zeros(n)
    for i in range(n):
        weights = np.exp(-0.5 * ((np.arange(n) - i) / sigma) ** 2)
        smoothed[i] = np.average(values, weights=weights)
    return smoothed


def analyze_language_switching(
    language: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    device: str = "cpu",
    provider: str = AI_PROVIDER,
) -> dict:
    """
    Analyze language representation in NLA explanations for one language.
    Returns time series of target-language mention rates.
    """
    config = LANGUAGE_CONTEXTS[language]
    target_keywords = config["keywords"]

    all_mention_series = []

    for prompt_idx, prompt in enumerate(config["prompts"]):
        print(f"\n  Prompt {prompt_idx+1}: '{prompt[:60]}...'")

        activations, tokens = extract_all_token_activations(
            model, tokenizer, prompt, device=device
        )
        n_tokens = len(tokens)

        # Sample positions (every 2nd token for efficiency)
        positions = list(range(0, n_tokens, 2))[:20]

        explanations = verbalize_activation_batch(activations, tokens, positions, provider=provider)

        # Count target language mentions
        mention_counts = np.zeros(n_tokens)
        for pos, exp in zip(positions, explanations):
            if target_keywords:
                mention_counts[pos] = count_language_mentions(exp, target_keywords)

        # Smooth
        smoothed = gaussian_smooth(mention_counts)
        all_mention_series.append(smoothed)

    return {
        "language": language,
        "prompts": config["prompts"],
        "mention_series": [s.tolist() for s in all_mention_series],
        "mean_mentions": float(np.mean([s.mean() for s in all_mention_series])),
    }


def run_language_switching_analysis(device: str = "cpu", provider: str = AI_PROVIDER) -> dict:
    """
    Full language switching case study.
    Tests three language-seeded contexts + neutral baseline.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\n{'='*60}")
    print("Case Study: Language Switching Detection")
    print(f"{'='*60}")

    model, tokenizer = load_gpt2_lm(device)

    results = {}
    for language in ["french", "spanish", "neutral"]:
        print(f"\nAnalyzing: {language.upper()}")
        results[language] = analyze_language_switching(language, model, tokenizer, device, provider=provider)

    # Save
    out_path = RESULTS_DIR / "case_study_language_switching.json"
    save_results = {
        lang: {k: v for k, v in r.items() if k != "mention_series"}
        for lang, r in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print("\n✓ Language switching analysis complete.")
    return results


def load_gpt2_lm(device: str = "cpu"):
    tokenizer = GPT2Tokenizer.from_pretrained(TARGET_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(TARGET_MODEL_NAME)
    model.eval()
    model.to(device)
    return model, tokenizer


if __name__ == "__main__":
    results = run_language_switching_analysis()
