"""
01_data_collection/extract_activations.py

Extracts residual stream activations from GPT-2 at a specified layer.

The paper uses activations from the final token of randomly truncated text
snippets, normalized to unit L2-norm. We replicate this exactly on GPT-2
with WikiText-2, storing:
  - activations: (N, d_model) float32 array
  - tokens:      list of tokenized context strings
  - texts:       list of raw text snippets

Output: data/activations.npz, data/texts.jsonl
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
import random


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TARGET_MODEL_NAME, TARGET_LAYER, D_MODEL,
    DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT,
    MAX_SAMPLES, MAX_TOKEN_LENGTH, MIN_TOKEN_LENGTH,
    NORMALIZE_ACTIVATIONS, DATA_DIR, SEED
)


def load_target_model(device: str = "cpu"):
    """Load GPT-2 and tokenizer."""
    print(f"Loading {TARGET_MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(TARGET_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(TARGET_MODEL_NAME)
    model.eval()
    model.to(device)
    return model, tokenizer


def get_activation_hook(storage: dict, layer_idx: int):
    """
    Returns a forward hook that captures the residual stream output
    of transformer block `layer_idx`.

    GPT-2's transformer blocks return a tuple; index 0 is the hidden state
    of shape (batch, seq_len, d_model).
    """
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        storage["hidden"] = hidden.detach().cpu()
    return hook


def extract_final_token_activation(
    model: GPT2Model,
    tokenizer: GPT2Tokenizer,
    text: str,
    layer_idx: int,
    device: str = "cpu",
) -> tuple[np.ndarray | None, list[str]]:
    """
    Tokenize `text`, run a forward pass, and return the residual stream
    activation at `layer_idx` for the *final* token.

    Returns
    -------
    activation : np.ndarray of shape (d_model,) or None if text too short
    tokens     : list of token strings
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]

    if seq_len < MIN_TOKEN_LENGTH:
        return None, []

    storage: dict = {}
    hook_handle = model.h[layer_idx].register_forward_hook(
        get_activation_hook(storage, layer_idx)
    )

    with torch.no_grad():
        _ = model(**enc)

    hook_handle.remove()

    # Take the final-token hidden state: shape (d_model,)
    hidden = storage["hidden"][0, -1, :].numpy().astype(np.float32)

    if NORMALIZE_ACTIVATIONS:
        norm = np.linalg.norm(hidden)
        if norm > 1e-8:
            hidden = hidden / norm

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    return hidden, tokens


def collect_activations(device: str = "cpu") -> tuple[np.ndarray, list, list]:
    """
    Main collection loop. Streams WikiText-2 and collects activations until
    MAX_SAMPLES valid snippets are gathered.

    Returns
    -------
    activations : np.ndarray (N, D_MODEL)
    texts       : list of raw strings
    token_lists : list of token-string lists
    """
    print(f"Loading dataset {DATASET_NAME}/{DATASET_CONFIG} [{DATASET_SPLIT}]...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)

    model, tokenizer = load_target_model(device)

    activations, texts, token_lists = [], [], []

    pbar = tqdm(total=MAX_SAMPLES, desc="Extracting activations")
    for row in dataset:
        text = row["text"].strip()
        if len(text) < 50:
            continue

        act, tokens = extract_final_token_activation(
            model, tokenizer, text, TARGET_LAYER, device
        )
        if act is None:
            continue

        activations.append(act)
        texts.append(text)
        token_lists.append(tokens)
        pbar.update(1)

        if len(activations) >= MAX_SAMPLES:
            break

    pbar.close()
    print(f"Collected {len(activations)} activations.")

    return np.stack(activations, axis=0), texts, token_lists


def save_data(
    activations: np.ndarray,
    texts: list,
    token_lists: list,
) -> None:
    """Persist activations and text metadata to disk."""
    np.savez_compressed(DATA_DIR / "activations.npz", activations=activations)
    print(f"Saved activations: {activations.shape} → {DATA_DIR}/activations.npz")

    with open(DATA_DIR / "texts.jsonl", "w") as f:
        for text, tokens in zip(texts, token_lists):
            f.write(json.dumps({"text": text, "tokens": tokens}) + "\n")
    print(f"Saved text metadata → {DATA_DIR}/texts.jsonl")


def load_data() -> tuple[np.ndarray, list, list]:
    """Load previously saved activations and texts."""
    data = np.load(DATA_DIR / "activations.npz")
    activations = data["activations"]

    texts, token_lists = [], []
    with open(DATA_DIR / "texts.jsonl") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            token_lists.append(item["tokens"])

    return activations, texts, token_lists


def compute_activation_statistics(activations: np.ndarray) -> dict:
    """
    Compute summary statistics used later for FVE baseline.

    Returns dict with mean, std, per-dim variance, and total variance.
    """
    mean = activations.mean(axis=0)            # (d_model,)
    total_variance = np.var(activations, axis=0).sum()
    norms = np.linalg.norm(activations, axis=1) # (N,) — should be ~1 if normalized
    stats = {
        "mean": mean,
        "total_variance": float(total_variance),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "n_samples": activations.shape[0],
        "d_model": activations.shape[1],
    }
    print(f"  Total variance: {total_variance:.4f}")
    print(f"  Mean activation norm: {norms.mean():.4f} ± {norms.std():.4f}")
    return stats


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    activations, texts, token_lists = collect_activations(device)
    save_data(activations, texts, token_lists)

    print("\nActivation statistics:")
    stats = compute_activation_statistics(activations)
    for k, v in stats.items():
        if k not in ("mean",):
            print(f"  {k}: {v}")
