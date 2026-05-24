"""
05_evaluation/compute_fve.py

Fraction of Variance Explained (FVE) computation — the primary metric.

FVE = 1 - Var(h - h_hat) / Var(h)

where:
  h     = original activation
  h_hat = reconstruction from NLA (AV→text→AR)

FVE=0 → predicting the mean (baseline)
FVE=1 → perfect reconstruction
Paper reports 0.6–0.8 on Claude models (frontier scale, RL-trained).
We expect 0.2–0.5 given: (a) API proxy AV, (b) small GPT-2 target,
(c) limited training steps.

This module provides:
  - compute_fve()              single scalar FVE
  - compute_fve_per_dim()      per-dimension FVE (diagnostic)
  - compute_fve_vs_baseline()  compare NLA FVE vs mean-prediction baseline
  - layerwise_fve()            FVE across multiple layers
  - fve_distribution()         FVE distribution over samples
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, D_MODEL


# ── Core metric ───────────────────────────────────────────────────────────────

def compute_fve(
    originals: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """
    Compute scalar FVE.

    Parameters
    ----------
    originals    : (N, d_model) original activations
    reconstructed: (N, d_model) reconstructed activations

    Returns
    -------
    fve : float in (-inf, 1.0]; starts near 0 at random init.

    Both inputs are L2-normalized before comparison because stored activations
    have unit norm (NORMALIZE_ACTIVATIONS=True). Without this, a freely-scaled
    h_hat causes Var(residual) >> Var(h) and FVE starts deeply negative.
    """
    orig  = originals   / (np.linalg.norm(originals,    axis=1, keepdims=True) + 1e-8)
    recon = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)
    residual_var = np.var(orig - recon, axis=0).sum()
    original_var = np.var(orig, axis=0).sum()
    if original_var < 1e-10:
        return 0.0
    return float(1.0 - residual_var / original_var)


def compute_fve_per_dim(
    originals: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Per-dimension FVE. Returns array of shape (d_model,).
    Both inputs are L2-normalized (same reason as compute_fve).
    """
    orig  = originals    / (np.linalg.norm(originals,    axis=1, keepdims=True) + 1e-8)
    recon = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)
    residual_var = np.var(orig - recon, axis=0)
    original_var = np.var(orig, axis=0)
    return np.where(
        original_var > 1e-10,
        1.0 - residual_var / np.maximum(original_var, 1e-10),
        0.0,
    )


def compute_per_sample_fve(
    originals: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    FVE per individual sample (not the standard definition, but diagnostic).
    Uses per-sample MSE normalized by global variance.
    Returns (N,) array.
    """
    global_var = np.var(originals, axis=0).sum()
    per_sample_mse = np.mean((originals - reconstructed) ** 2, axis=1) * originals.shape[1]
    normalized = 1.0 - per_sample_mse / max(global_var, 1e-10)
    return normalized


# ── Baselines ─────────────────────────────────────────────────────────────────

def mean_prediction_baseline(activations: np.ndarray) -> np.ndarray:
    """Baseline: predict the mean activation for every sample. FVE=0 by definition."""
    mean = activations.mean(axis=0, keepdims=True)
    return np.repeat(mean, len(activations), axis=0)


def random_baseline_fve(activations: np.ndarray, n_trials: int = 10) -> float:
    """
    Estimate FVE for random reconstruction (shuffled activations).
    Should be well below 0.
    """
    fves = []
    for _ in range(n_trials):
        shuffled = activations[np.random.permutation(len(activations))]
        fves.append(compute_fve(activations, shuffled))
    return float(np.mean(fves))


# ── Full evaluation suite ─────────────────────────────────────────────────────

def evaluate_reconstruction(
    originals: np.ndarray,
    reconstructed: np.ndarray,
    label: str = "NLA",
) -> dict:
    """
    Comprehensive reconstruction evaluation.

    Returns dict with:
      fve, mse, cosine_similarity, per_dim_fve_stats,
      baseline_fve (mean prediction), random_baseline_fve
    """
    # FVE
    fve = compute_fve(originals, reconstructed)

    # Per-dim FVE
    per_dim_fve = compute_fve_per_dim(originals, reconstructed)

    # MSE
    mse = float(np.mean((originals - reconstructed) ** 2))

    # Cosine similarity (per sample, then averaged)
    norms_orig = np.linalg.norm(originals, axis=1, keepdims=True) + 1e-8
    norms_recon = np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8
    cos_sim = float(np.mean(
        np.sum((originals / norms_orig) * (reconstructed / norms_recon), axis=1)
    ))

    # Baselines
    mean_pred = mean_prediction_baseline(originals)
    baseline_fve = compute_fve(originals, mean_pred)  # should be 0.0

    per_sample_fve = compute_per_sample_fve(originals, reconstructed)

    result = {
        "label": label,
        "n_samples": len(originals),
        "fve": fve,
        "mse": mse,
        "cosine_similarity": cos_sim,
        "baseline_fve": baseline_fve,
        "per_dim_fve_mean": float(per_dim_fve.mean()),
        "per_dim_fve_std": float(per_dim_fve.std()),
        "per_dim_fve_positive_fraction": float((per_dim_fve > 0).mean()),
        "per_sample_fve_mean": float(per_sample_fve.mean()),
        "per_sample_fve_std": float(per_sample_fve.std()),
        "per_dim_fve": per_dim_fve.tolist(),
        "per_sample_fve": per_sample_fve.tolist(),
    }

    print(f"\n{'='*50}")
    print(f"Reconstruction Evaluation: {label}")
    print(f"{'='*50}")
    print(f"  FVE:                    {fve:.4f}")
    print(f"  MSE:                    {mse:.6f}")
    print(f"  Cosine similarity:      {cos_sim:.4f}")
    print(f"  Baseline FVE (mean):    {baseline_fve:.4f}")
    print(f"  Per-dim FVE (mean±std): {per_dim_fve.mean():.4f} ± {per_dim_fve.std():.4f}")
    print(f"  % dims with FVE>0:      {100*(per_dim_fve>0).mean():.1f}%")

    return result


def layerwise_fve_analysis(
    model,
    tokenizer,
    texts: list[str],
    av,
    ar_wrapper,
    layers: list[int],
    device: str = "cpu",
) -> dict:
    """
    Compute FVE at multiple layers to understand where NLA is most effective.
    Replicates paper's layer sensitivity analysis.

    Returns dict: layer_idx → fve
    """
    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_collection"))
    from extract_activations import extract_final_token_activation

    results = {}
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}:")
        activations, token_lists = [], []
        for text in texts[:50]:  # subset for speed
            act, tokens = extract_final_token_activation(
                model, tokenizer, text, layer_idx, device
            )
            if act is not None:
                activations.append(act)
                token_lists.append(tokens)

        if not activations:
            continue

        acts = np.stack(activations)
        explanations = av.verbalize_batch(acts, token_lists)
        h_hats = ar_wrapper.reconstruct(explanations)
        fve = compute_fve(acts, h_hats)
        results[layer_idx] = fve
        print(f"  → FVE: {fve:.4f}")

    return results


def save_evaluation_results(results: dict, filename: str = "eval_results.json") -> None:
    # Convert per_dim_fve from list back for saving (already list in results)
    path = RESULTS_DIR / filename
    # Remove large arrays for JSON (keep summary stats)
    save_results = {k: v for k, v in results.items() if k not in ("per_dim_fve", "per_sample_fve")}
    with open(path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved evaluation results → {path}")
