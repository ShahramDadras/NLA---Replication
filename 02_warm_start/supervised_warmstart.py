"""
02_warm_start/supervised_warmstart.py

Warm-starts the Activation Reconstructor (AR) using supervised training on
(summary, activation) pairs generated in generate_summaries.py.

Architecture (with PCA):
  summary → SentenceTransformer (frozen) → 384-dim embedding
          → MLP (384 → 512 → 512 → PCA_COMPONENTS) → PCA coords
          → PCA inverse_transform → reconstructed activation (768-dim)

Why PCA? Training a 384→768 regression is hard: 768 outputs, all noisy.
PCA first projects activations to their top-20 principal components (~90% of
variance), so the MLP only predicts 20 numbers. Inverse transform recovers
the 768-dim vector. This raises warm-start FVE from ~0.05 to ~0.3.

Output:
  results/ar_warmstart.pt   — MLP state dict + training history
  results/ar_warmstart_pca.pkl — fitted sklearn PCA (needed at inference)
"""

import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    D_MODEL, SENTENCE_ENCODER, AR_HIDDEN_DIM, AR_NUM_LAYERS,
    WARMSTART_EPOCHS, WARMSTART_LR, RESULTS_DIR, DATA_DIR, SEED,
    PCA_COMPONENTS,
)


# ── AR Model ───────────────────────────────────────────────────────────────────

class ActivationReconstructor(nn.Module):
    """
    Maps a sentence embedding to PCA coordinates of an activation vector.
    The caller applies pca.inverse_transform() to recover the full 768-dim vector.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = AR_HIDDEN_DIM,
        output_dim: int = PCA_COMPONENTS,
        n_layers: int = AR_NUM_LAYERS,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(embeddings)


# ── Dataset ────────────────────────────────────────────────────────────────────

class NLAWarmstartDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, pca_targets: np.ndarray):
        self.embeddings  = torch.tensor(embeddings,   dtype=torch.float32)
        self.pca_targets = torch.tensor(pca_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.pca_targets[idx]


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_summaries(summaries: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"Loading sentence encoder: {SENTENCE_ENCODER}")
    encoder = SentenceTransformer(SENTENCE_ENCODER)
    embeddings = encoder.encode(
        summaries,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ── FVE in original 768-dim space ─────────────────────────────────────────────

def compute_fve_full(
    pca: PCA,
    pca_preds: np.ndarray,
    orig_activations: np.ndarray,
) -> float:
    """
    Inverse-transform MLP predictions back to 768-dim, then compute FVE
    against the original (unit-norm) activations.
    Normalizes both arrays to unit L2 norm to avoid scale mismatch.
    """
    h_hat = pca.inverse_transform(pca_preds).astype(np.float32)
    h     = orig_activations.astype(np.float32)
    h     = h     / (np.linalg.norm(h,     axis=1, keepdims=True) + 1e-8)
    h_hat = h_hat / (np.linalg.norm(h_hat, axis=1, keepdims=True) + 1e-8)
    res_var  = np.var(h - h_hat, axis=0).sum()
    orig_var = np.var(h, axis=0).sum()
    return float(1.0 - res_var / orig_var) if orig_var > 1e-10 else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train_ar_warmstart(
    embeddings: np.ndarray,
    activations: np.ndarray,
    n_epochs: int = WARMSTART_EPOCHS,
    lr: float = WARMSTART_LR,
    device: str = "cpu",
    n_pca: int = PCA_COMPONENTS,
) -> tuple:
    """
    1. Fit PCA on activations → project to n_pca dimensions.
    2. Train MLP: embedding → PCA coords (MSE loss).
    3. Report FVE in original 768-dim space (inverse transform).

    Returns (model, pca, history).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Fit PCA ──────────────────────────────────────────────────────────
    pca = PCA(n_components=n_pca, random_state=SEED)
    pca.fit(activations)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA fitted: {n_pca} components capture {explained:.1%} of activation variance")

    activations_pca = pca.transform(activations).astype(np.float32)  # (N, n_pca)

    # ── Dataset split ─────────────────────────────────────────────────────
    dataset = NLAWarmstartDataset(embeddings, activations_pca)
    n_val   = max(10, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

    # Keep original activations aligned to val indices for FVE computation
    val_indices    = val_set.indices
    val_acts_full  = activations[val_indices]

    # ── Model ─────────────────────────────────────────────────────────────
    input_dim = embeddings.shape[1]
    model     = ActivationReconstructor(input_dim=input_dim, output_dim=n_pca).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_fve": []}

    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0.0
        for emb_b, pca_b in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            emb_b = emb_b.to(device)
            pca_b = pca_b.to(device)
            pred  = model(emb_b)
            loss  = criterion(pred, pca_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(emb_b)

        train_loss = total_loss / n_train

        # Validation — report FVE in original 768-dim space
        model.eval()
        val_pca_preds = []
        with torch.no_grad():
            for emb_b, _ in val_loader:
                val_pca_preds.append(model(emb_b.to(device)).cpu().numpy())
        val_pca_preds = np.concatenate(val_pca_preds)

        val_loss = float(np.mean((val_pca_preds - activations_pca[val_indices]) ** 2))
        val_fve  = compute_fve_full(pca, val_pca_preds, val_acts_full)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_fve"].append(val_fve)

        print(
            f"  Epoch {epoch+1}: train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_FVE={val_fve:.4f}"
        )
        scheduler.step()

    print(f"\nWarm-start complete. Final val FVE: {history['val_fve'][-1]:.4f}")
    print("  (Paper ~0.3–0.4; PCA bottleneck raises this vs raw 768-dim regression)")

    return model, pca, history


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_ar(model: ActivationReconstructor, pca: PCA, history: dict) -> None:
    pt_path  = RESULTS_DIR / "ar_warmstart.pt"
    pca_path = RESULTS_DIR / "ar_warmstart_pca.pkl"
    torch.save({"state_dict": model.state_dict(), "history": history}, pt_path)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved AR warm-start → {pt_path}")
    print(f"Saved PCA model     → {pca_path}")


def load_ar(device: str = "cpu") -> tuple:
    """
    Returns (ActivationReconstructor, PCA).
    Raises FileNotFoundError with a clear message if the checkpoint was saved
    with the old output_dim=768 architecture (pre-PCA refactor) so the caller
    knows to delete it and retrain.
    """
    pt_path  = RESULTS_DIR / "ar_warmstart.pt"
    pca_path = RESULTS_DIR / "ar_warmstart_pca.pkl"

    ckpt  = torch.load(pt_path, map_location=device)
    model = ActivationReconstructor()
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as e:
        if "size mismatch" in str(e):
            pt_path.unlink(missing_ok=True)
            pca_path.unlink(missing_ok=True)
            raise FileNotFoundError(
                "Warm-start checkpoint was built with the old output_dim=768 "
                "architecture. It has been deleted automatically — re-run to "
                "train with the new PCA-20 architecture."
            ) from e
        raise
    model.to(device).eval()

    pca = None
    if pca_path.exists():
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

    return model, pca


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(SEED)

    act_data    = np.load(DATA_DIR / "activations.npz")
    activations = act_data["activations"]

    summaries_raw = []
    with open(DATA_DIR / "summaries.jsonl") as f:
        for line in f:
            summaries_raw.append(json.loads(line))

    summaries_raw.sort(key=lambda x: x["idx"])
    indices              = [r["idx"]     for r in summaries_raw]
    summaries            = [r["summary"] for r in summaries_raw]
    activations_aligned  = activations[indices]

    print(f"Training AR warm-start on {len(summaries)} pairs.")

    embeddings = encode_summaries(summaries)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, pca, history = train_ar_warmstart(
        embeddings, activations_aligned, device=device
    )
    save_ar(model, pca, history)

    print("\nFVE over epochs:", [f"{v:.4f}" for v in history["val_fve"]])
