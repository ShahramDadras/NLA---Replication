"""
06_case_studies/planning_in_poetry.py

Replicates the "Planning in Poetry" case study from Section 3.1 of the paper.

Original finding (Lindsey et al., Opus 4.6): At the end of the couplet's
first line ("grab it"), the model already represents the intended end-rhyme
for the second line ("rabbit").

Our adaptation on GPT-2:
  - Use the same prompt: "A rhyming couplet: He saw a carrot and had to grab it,"
  - Run NLA verbalization on the final token of the first line
  - Check if "rabbit" (or rhyme-class: habit/rabbit/cabinet) appears in explanation
  - Demonstrate NLA-based steering: edit "rabbit" → "mouse" in explanation,
    reconstruct a steering vector, apply it to GPT-2, observe changed completions

Results expected: GPT-2 (at layer 7) should show some planning signal,
but less robustly than Opus 4.6. We document the difference honestly.
"""

import sys
import re
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TARGET_MODEL_NAME, TARGET_LAYER, D_MODEL,
    RESULTS_DIR, FIGURES_DIR, ANTHROPIC_API_KEY, SEED
)

POETRY_PROMPT = "A rhyming couplet: He saw a carrot and had to grab it,"
RHYME_WORDS = {"rabbit", "habit", "cabinet", "inhabit", "sabbath"}
MOUSE_WORDS = {"mouse", "house", "louse", "grouse", "blouse"}

STEERING_ALPHA_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
N_COMPLETIONS = 20   # completions per steering strength


def load_gpt2_lm(device: str = "cpu"):
    """Load GPT-2 with LM head for completion."""
    tokenizer = GPT2Tokenizer.from_pretrained(TARGET_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(TARGET_MODEL_NAME)
    model.eval()
    model.to(device)
    return model, tokenizer


def extract_activation_at_token(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    text: str,
    token_position: int,  # -1 = last token
    layer: int = TARGET_LAYER,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """
    Extract residual stream activation at a specific token position.
    """
    storage = {}
    def hook(module, input, output):
        storage["hidden"] = output[0].detach().cpu()

    handle = model.transformer.h[layer].register_forward_hook(hook)
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**enc)
    handle.remove()

    h = storage["hidden"]
    # Defensive: some transformers versions give (batch, seq, d), others (seq, d)
    if h.dim() == 3:
        hidden = h[0, token_position, :].numpy().astype(np.float32)
    else:
        hidden = h[token_position, :].numpy().astype(np.float32)
    # Normalize to unit norm
    norm = np.linalg.norm(hidden)
    if norm > 1e-8:
        hidden = hidden / norm

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
    return hidden, tokens


def verbalize_poetry_activation(
    activation: np.ndarray,
    tokens: list[str],
    layer: int = TARGET_LAYER,
    provider: str = "local",
) -> str:
    """
    Verbalize the "grab it" token activation using the configured AV provider.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "03_nla_components"))
    from activation_verbalizer import ActivationVerbalizer
    av = ActivationVerbalizer(layer=layer, provider=provider)
    return av.verbalize(activation, tokens)


def check_rhyme_in_explanation(explanation: str, rhyme_class: set) -> bool:
    """Check if any word from the rhyme class appears in the explanation."""
    exp_lower = explanation.lower()
    return any(word in exp_lower for word in rhyme_class)


def edit_explanation(explanation: str, original_words: dict) -> str:
    """
    Edit an explanation by substituting rabbit-rhyme words with mouse-rhyme words.
    Paper: "rabbit"→"mouse," "habit"→"house," "carrots"→"cheese"
    """
    edited = explanation
    for old, new in original_words.items():
        edited = re.sub(r'\b' + old + r'\b', new, edited, flags=re.IGNORECASE)
    return edited


def reconstruct_activation_from_explanation(
    ar_wrapper,
    explanation: str,
) -> np.ndarray:
    """Reconstruct an activation from a text explanation using the AR."""
    return ar_wrapper.reconstruct_single(explanation)


def compute_steering_vector(
    ar_wrapper,
    explanation_original: str,
    explanation_edited: str,
) -> np.ndarray:
    """
    Compute NLA-based steering vector:
      Δ = AR(z_edit) - AR(z_orig)
    Per paper equation.
    """
    h_orig = ar_wrapper.reconstruct_single(explanation_original)
    h_edit = ar_wrapper.reconstruct_single(explanation_edited)
    delta = h_edit - h_orig
    return delta


def apply_steering(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    delta: np.ndarray,
    alpha: float,
    layer: int = TARGET_LAYER,
    device: str = "cpu",
    n_completions: int = N_COMPLETIONS,
    max_new_tokens: int = 20,
) -> list[str]:
    """
    Apply NLA steering vector at the final token of `prompt` during generation.

    h → h + α * ||h|| * (Δ / ||Δ||)

    Returns list of completion strings.
    """
    delta_norm = delta / (np.linalg.norm(delta) + 1e-8)
    delta_tensor = torch.tensor(delta_norm, dtype=torch.float32).to(device)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]

    completions = []

    for _ in range(n_completions):
        # Hook: add steering vector at the final token of the prompt
        def steering_hook(module, input, output):
            h = output[0]  # (1, seq, d_model)
            h_last = h[0, seq_len - 1, :]  # final token
            h_norm = torch.norm(h_last)
            h[0, seq_len - 1, :] = h_last + alpha * h_norm * delta_tensor
            return (h,) + output[1:]

        handle = model.transformer.h[layer].register_forward_hook(steering_hook)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        handle.remove()

        generated = tokenizer.decode(out[0][seq_len:], skip_special_tokens=True)
        completions.append(generated.strip())

    return completions


def run_poetry_case_study(ar_wrapper=None, device: str = "cpu", provider: str = "local") -> dict:
    """
    Full poetry planning case study.
    Returns results dict with verbalization, rhyme check, and steering results.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\n{'='*60}")
    print("Case Study: Planning in Poetry")
    print(f"{'='*60}")
    print(f"Prompt: '{POETRY_PROMPT}'")

    model, tokenizer = load_gpt2_lm(device)

    # Extract activation at the final token ("grab it,")
    activation, tokens = extract_activation_at_token(
        model, tokenizer, POETRY_PROMPT, token_position=-1, device=device
    )
    print(f"\nExtracted activation at token: '{tokens[-1]}'")
    print(f"  Shape: {activation.shape}, Norm: {np.linalg.norm(activation):.4f}")

    # Verbalize
    print("\nVerbalizing activation...")
    explanation = verbalize_poetry_activation(activation, tokens, provider=provider)
    print(f"\nNLA Explanation (excerpt):\n{explanation[:400]}...")

    # Check for rhyme planning signal
    rabbit_found = check_rhyme_in_explanation(explanation, RHYME_WORDS)
    print(f"\nRabbit-rhyme words detected in explanation: {rabbit_found}")

    # Steering experiment (if AR is available)
    steering_results = {}
    if ar_wrapper is not None:
        print("\nRunning NLA steering experiment...")
        substitutions = {
            "rabbit": "mouse", "habit": "house", "carrots": "cheese",
            "carrot": "cheese", "Rabbit": "Mouse", "Habit": "House"
        }
        explanation_edited = edit_explanation(explanation, substitutions)
        print(f"\nEdited explanation (rabbit→mouse): {explanation_edited[:300]}...")

        delta = compute_steering_vector(ar_wrapper, explanation, explanation_edited)
        print(f"  Steering vector norm: {np.linalg.norm(delta):.4f}")

        # Baseline completions (no steering)
        baseline = apply_steering(
            model, tokenizer, POETRY_PROMPT, delta, alpha=0.0, device=device, n_completions=10
        )
        print(f"\nBaseline completions (α=0.0):")
        for c in baseline[:5]:
            print(f"    → '{c}'")

        # Steered completions at different alphas
        for alpha in [1.0, 3.0, 5.0]:
            steered = apply_steering(
                model, tokenizer, POETRY_PROMPT, delta,
                alpha=alpha, device=device, n_completions=10
            )
            rabbit_count = sum(1 for c in steered if any(w in c.lower() for w in RHYME_WORDS))
            mouse_count  = sum(1 for c in steered if any(w in c.lower() for w in MOUSE_WORDS))

            print(f"\nSteered completions (α={alpha}):")
            for c in steered[:3]:
                print(f"    → '{c}'")
            print(f"    Rabbit-class: {rabbit_count}/10 | Mouse-class: {mouse_count}/10")

            steering_results[f"alpha_{alpha}"] = {
                "completions": steered,
                "rabbit_count": rabbit_count,
                "mouse_count": mouse_count,
            }
    else:
        print("  (Skipping steering: no AR provided)")

    results = {
        "prompt": POETRY_PROMPT,
        "final_token": tokens[-1],
        "explanation": explanation,
        "rabbit_rhyme_detected": rabbit_found,
        "steering": steering_results,
    }

    out_path = RESULTS_DIR / "case_study_poetry.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "steering"}, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_poetry_case_study()
    print("\n✓ Poetry case study complete.")
