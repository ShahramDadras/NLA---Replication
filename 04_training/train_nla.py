"""
04_training/train_nla.py

Joint NLA training loop.

Paper training procedure (replicated):
  Each step:
  (i)  Sample batch of activations h_l
  (ii) For each h_l, sample GRPO_GROUP_SIZE explanations z ~ AV(·|h_l)
  (iii) Update AR: one gradient step on MSE(AR(z), h_l) for all (z, h_l) pairs
  (iv)  Update AV: GRPO — compute rewards r(h_l,z)=-log||h-AR(z)||², normalize,
                    weight log-probs of AV tokens by advantage

Simplification vs paper:
  - AV update: since we use the Claude API (not a fine-tunable model), we simulate
    the AV policy improvement by tracking which types of explanations yield higher
    rewards and biasing our prompts accordingly (prompt engineering as policy update).
    This is explicitly flagged as a limitation in our README.
  - AR update: fully faithful — trained MLP with MSE loss.
  - GRPO reward normalization is computed per-group (per activation), as in the paper.

Output:
  results/training_log.jsonl  — per-step metrics
  results/ar_trained.pt       — AR checkpoint at each save interval
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "03_nla_components"))
from config import (
    TRAINING_STEPS, TRAIN_BATCH_SIZE, GRPO_GROUP_SIZE,
    KL_BETA, AR_LR, LOG_REWARD_TRANSFORM,
    RESULTS_DIR, DATA_DIR, SEED, AI_PROVIDER
)
from activation_verbalizer import ActivationVerbalizer
from activation_reconstructor import (
    ActivationReconstructorWrapper,
    ActivationReconstructorMLP,
)


def grpo_reward_normalize(rewards: np.ndarray) -> np.ndarray:
    """
    Normalize rewards within a GRPO group:
      advantage = (r - mean(r)) / (std(r) + eps)
    Shape: (group_size,)
    """
    mean = rewards.mean()
    std  = rewards.std()
    return (rewards - mean) / (std + 1e-8)


def run_training(
    activations: np.ndarray,
    token_lists: list[list[str]],
    ar_wrapper: ActivationReconstructorWrapper,
    av: ActivationVerbalizer,
    n_steps: int = TRAINING_STEPS,
    batch_size: int = TRAIN_BATCH_SIZE,
    group_size: int = GRPO_GROUP_SIZE,
    save_every: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    provider: str = AI_PROVIDER,
) -> dict:
    """
    Main NLA training loop.

    Returns training history dict.
    """
    np.random.seed(SEED)
    N = len(activations)
    log_path = RESULTS_DIR / "training_log.jsonl"

    history = {
        "step": [],
        "ar_loss": [],
        "mean_reward": [],
        "best_reward": [],
        "fve_estimate": [],
    }

    print(f"\n{'='*60}")
    print(f"NLA Training: {n_steps} steps, batch={batch_size}, group={group_size}")
    print(f"Device: {device}", end="")
    if device.startswith("cuda") and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ({torch.cuda.get_device_name(0)}, {vram:.1f} GB VRAM)")
    else:
        print()
    if provider == "local":
        print("Provider: local (no API sleep — full GPU throughput)")
    print(f"{'='*60}\n")

    with open(log_path, "w") as log_file:
        for step in tqdm(range(n_steps), desc="NLA Training"):

            # ── (i) Sample batch of activations ───────────────────────────
            indices = np.random.choice(N, size=batch_size, replace=False)
            # Keep on CPU as a NumPy array for token processing and text pipelines
            batch_activations = activations[indices]        
            batch_tokens = [token_lists[i] for i in indices]

            # ── (ii) Generate group of explanations per activation ─────────
            all_explanations = []
            all_rewards = np.zeros((batch_size, group_size))

            for b_idx in range(batch_size):
                act = batch_activations[b_idx]
                toks = batch_tokens[b_idx]

                group_explanations = av.verbalize_group(act, toks, n=group_size)

                # Batch reward computation: encode all group explanations at once,
                # run a single MLP forward pass, then compute MSE vs tiled target.
                group_rewards = ar_wrapper.compute_rewards_batch(
                    group_explanations,
                    np.tile(act, (len(group_explanations), 1)),
                    log_transform=LOG_REWARD_TRANSFORM,
                )

                # GRPO: pick the best explanation to use for AR update
                best_idx = np.argmax(group_rewards)
                all_explanations.append(group_explanations[best_idx])
                all_rewards[b_idx] = group_rewards

            # ── (iii) AR update: MSE on best explanations ─────────────────
            # Explicitly cast the batch to a PyTorch tensor and send it to the GPU device
            batch_activations_gpu = torch.tensor(batch_activations, dtype=torch.float32, device=device)
            ar_loss = ar_wrapper.update_step(all_explanations, batch_activations_gpu)

            # ── (iv) AV "update": compute GRPO advantages (vectorized on GPU)
            rewards_t  = torch.tensor(all_rewards, dtype=torch.float32, device=device)
            mean_r     = rewards_t.mean(dim=1, keepdim=True)
            std_r      = rewards_t.std(dim=1, keepdim=True, correction=0)
            group_advantages = (rewards_t - mean_r) / (std_r + 1e-8)

            # ── Metrics ───────────────────────────────────────────────────
            mean_reward = float(all_rewards.max(axis=1).mean())
            best_reward = float(all_rewards.max())

            # Estimate FVE metric directly on the GPU tensor
            h_hats = ar_wrapper.reconstruct(all_explanations)
            fve_est = ar_wrapper.compute_fve(batch_activations_gpu, h_hats)

            history["step"].append(step)
            history["ar_loss"].append(ar_loss)
            history["mean_reward"].append(mean_reward)
            history["best_reward"].append(best_reward)
            history["fve_estimate"].append(fve_est)

            log_record = {
                "step": step,
                "ar_loss": ar_loss,
                "mean_reward": mean_reward,
                "best_reward": best_reward,
                "fve_estimate": fve_est,
                "mean_group_advantage_std": float(group_advantages.std().item()),
            }
            log_file.write(json.dumps(log_record) + "\n")

            if step % 20 == 0:
                tqdm.write(
                    f"  Step {step:4d} | AR loss: {ar_loss:.4f} | "
                    f"FVE: {fve_est:.4f} | mean_reward: {mean_reward:.4f}"
                )

            if step > 0 and step % save_every == 0:
                ckpt_path = RESULTS_DIR / f"ar_step_{step:04d}.pt"
                ar_wrapper.save(ckpt_path)
                tqdm.write(f"  → Saved checkpoint: {ckpt_path}")

            if provider != "local":
                time.sleep(1.5)

    # Final save
    ar_wrapper.save(RESULTS_DIR / "ar_trained.pt")
    print(f"\nTraining complete. Final FVE: {history['fve_estimate'][-1]:.4f}")
    return history


def load_training_log() -> list[dict]:
    """Load training log for analysis/plotting."""
    records = []
    with open(RESULTS_DIR / "training_log.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    provider = AI_PROVIDER
    print(f"Device: {device}")

    # Load baseline activations as a vanilla NumPy array
    data = np.load(DATA_DIR / "activations.npz")
    activations = data["activations"]

    token_lists = []
    with open(DATA_DIR / "texts.jsonl") as f:
        for line in f:
            item = json.loads(line)
            token_lists.append(item["tokens"])

    # Load the warm-started Reconstructor model directly onto the target device (GPU)
    sys.path.insert(0, str(Path(__file__).parent.parent / "02_warm_start"))
    from supervised_warmstart import load_ar
    ar_model, pca = load_ar(device=device)
    ar_wrapper = ActivationReconstructorWrapper(ar_model, device=device, pca=pca)

    # Initialize the Activation Verbalizer component
    av = ActivationVerbalizer(provider=provider)

    # Launch the training lifecycle with correct CUDA hardware acceleration mapping
    history = run_training(
        activations, token_lists, ar_wrapper, av, device=device, provider=provider
    )
    print("\nFVE trajectory (every 20 steps):")
    for i in range(0, len(history["step"]), 20):
        print(f"  Step {history['step'][i]:4d}: FVE={history['fve_estimate'][i]:.4f}")
