"""
03_nla_components/activation_reconstructor.py

Activation Reconstructor (AR): maps a natural-language explanation back to
an activation vector.

Paper: "The AR is an LLM with the same architecture as M, but truncated to
its first l layers. To reconstruct an activation from an explanation z, we
wrap z in a fixed prompt, pass it through the model, then apply a learned
affine map to the layer-l activations at the final token."

Our implementation:
  - SentenceTransformer encodes the explanation (frozen).
  - Learned MLP head maps embedding → activation (trained, unfrozen).
  - Supports incremental training (RL loop AR update step).
  - Also includes an affine-map variant (linear probe) for ablation.
"""

import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    D_MODEL, SENTENCE_ENCODER, AR_HIDDEN_DIM, AR_NUM_LAYERS,
    AR_LR, RESULTS_DIR, SEED, PCA_COMPONENTS,
)


class ActivationReconstructorMLP(nn.Module):
    """
    Full MLP reconstructor.
    SentenceTransformer (frozen) → MLP (trained) → PCA coords → (inverse PCA) → h_hat.

    output_dim defaults to PCA_COMPONENTS (20), not D_MODEL (768).
    The ActivationReconstructorWrapper applies pca.inverse_transform() afterwards.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = AR_HIDDEN_DIM,
        output_dim: int = PCA_COMPONENTS,
        n_layers: int = AR_NUM_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

        # Initialize last layer near zero for stable warm-start
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(embeddings)


class ActivationReconstructorAffine(nn.Module):
    """
    Ablation: linear (affine) reconstructor.
    This is the weakest baseline — approximates the paper's affine map.
    """

    def __init__(self, input_dim: int = 384, output_dim: int = D_MODEL):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


class ActivationReconstructorWrapper:
    """
    High-level wrapper.

    The MLP predicts PCA coordinates (PCA_COMPONENTS-dim).
    All public methods (reconstruct, update_step, compute_reward) work in
    original 768-dim space — PCA transform/inverse-transform is internal.

    Pass pca= (a fitted sklearn PCA) from the warm-start checkpoint, or call
    fit_pca(activations) before training.
    """

    def __init__(
        self,
        model: nn.Module,
        encoder_name: str = SENTENCE_ENCODER,
        device: str = "cpu",
        lr: float = AR_LR,
        pca=None,
    ):
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.model   = model.to(device)
        self.device  = device
        self.pca     = pca
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self._step_count = 0

    # ── PCA helpers ───────────────────────────────────────────────────────────

    def fit_pca(self, activations: np.ndarray, n_components: int = PCA_COMPONENTS) -> None:
        """Fit PCA on training activations. Must be called before update_step."""
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=n_components, random_state=SEED)
        self.pca.fit(activations)
        print(
            f"PCA fitted: {n_components} components capture "
            f"{self.pca.explained_variance_ratio_.sum():.1%} of activation variance"
        )

    def _to_pca(self, activations: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return activations
        return self.pca.transform(activations).astype(np.float32)

    def _from_pca(self, coords: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return coords
        return self.pca.inverse_transform(coords).astype(np.float32)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts using the frozen sentence transformer."""
        return self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def reconstruct(self, texts: list[str]) -> np.ndarray:
        """
        End-to-end: texts → sentence embeddings → MLP → PCA coords → 768-dim.
        Returns numpy array of shape (N, d_model).
        """
        embeddings = self.encode(texts)
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pca_pred = self.model(emb_tensor).cpu().numpy()
        return self._from_pca(pca_pred)

    def reconstruct_single(self, text: str) -> np.ndarray:
        """Reconstruct a single explanation → activation."""
        return self.reconstruct([text])[0]

    def update_step(
        self,
        explanations: list[str],
        target_activations,  # np.ndarray or torch.Tensor (CPU or GPU)
    ) -> float:
        """
        One AR gradient step: MSE(AR(z), PCA(h_l)).
        Targets are projected to PCA space before computing loss.
        Returns the MSE loss in PCA space.
        """
        embeddings = self.encode(explanations)
        # sklearn PCA requires numpy; accept GPU tensors from the training loop
        act_np = (
            target_activations.cpu().numpy()
            if isinstance(target_activations, torch.Tensor)
            else target_activations
        )
        pca_targets = self._to_pca(act_np)

        emb_tensor = torch.tensor(embeddings,  dtype=torch.float32).to(self.device)
        act_tensor = torch.tensor(pca_targets, dtype=torch.float32).to(self.device)

        self.model.train()
        pred = self.model(emb_tensor)
        loss = nn.functional.mse_loss(pred, act_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._step_count += 1

        return float(loss.item())

    def compute_reward(
        self,
        explanation: str,
        target_activation: np.ndarray,
        log_transform: bool = True,
    ) -> float:
        """
        Compute the RL reward for an explanation:
          r = -log||h - AR(z)||² (with log transform, per paper)
          r = -||h - AR(z)||²    (without)
        Used in the AV's GRPO update.
        """
        h_hat = self.reconstruct_single(explanation)
        sq_error = float(np.sum((target_activation - h_hat) ** 2))
        if log_transform:
            return -float(np.log(max(sq_error, 1e-8)))
        return -sq_error

    def compute_rewards_batch(
        self,
        explanations: list[str],
        target_activations: np.ndarray,
        log_transform: bool = True,
    ) -> np.ndarray:
        """
        Batch reward computation.
        Returns (N,) array of rewards.
        """
        h_hats = self.reconstruct(explanations)
        sq_errors = np.sum((target_activations - h_hats) ** 2, axis=1)
        if log_transform:
            rewards = -np.log(np.maximum(sq_errors, 1e-8))
        else:
            rewards = -sq_errors
        return rewards

    @staticmethod
    def compute_fve(
        originals,       # np.ndarray or torch.Tensor (CPU or GPU)
        reconstructed,   # np.ndarray or torch.Tensor
    ) -> float:
        """
        FVE = 1 - Var(residual) / Var(original)

        Both arrays are L2-normalized before comparison because activations are
        stored as unit-norm vectors (NORMALIZE_ACTIVATIONS=True). Without this,
        scale mismatch between h (||h||=1) and a freely-scaled h_hat makes
        Var(residual) >> Var(original) and FVE starts deeply negative.

        Accepts GPU tensors for zero-copy computation on the same device.
        """
        if isinstance(originals, torch.Tensor):
            orig = originals / (originals.norm(dim=1, keepdim=True) + 1e-8)
            recon_t = (
                torch.tensor(reconstructed, dtype=torch.float32, device=originals.device)
                if isinstance(reconstructed, np.ndarray)
                else reconstructed.to(originals.device)
            )
            recon = recon_t / (recon_t.norm(dim=1, keepdim=True) + 1e-8)
            residual_var = float((orig - recon).var(dim=0).sum())
            original_var = float(orig.var(dim=0).sum())
        else:
            orig = originals / (np.linalg.norm(originals, axis=1, keepdims=True) + 1e-8)
            recon = reconstructed / (np.linalg.norm(reconstructed, axis=1, keepdims=True) + 1e-8)
            residual_var = float(np.var(orig - recon, axis=0).sum())
            original_var = float(np.var(orig, axis=0).sum())
        if original_var < 1e-10:
            return 0.0
        return float(1.0 - residual_var / original_var)

    def evaluate_fve(
        self,
        explanations: list[str],
        target_activations: np.ndarray,
    ) -> dict:
        """
        Full FVE evaluation: reconstruct all, compute FVE and per-sample MSE.
        Returns a dict with 'fve', 'mse', 'per_sample_mse'.
        """
        h_hats = self.reconstruct(explanations)
        fve = self.compute_fve(target_activations, h_hats)
        per_sample_mse = np.mean((target_activations - h_hats) ** 2, axis=1)
        return {
            "fve": fve,
            "mse": float(per_sample_mse.mean()),
            "per_sample_mse": per_sample_mse.tolist(),
            "n_samples": len(explanations),
        }

    def save(self, path: Path) -> None:
        torch.save({"state_dict": self.model.state_dict()}, path)
        if self.pca is not None:
            pca_path = Path(path).parent / (Path(path).stem + "_pca.pkl")
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        pca_path = Path(path).parent / (Path(path).stem + "_pca.pkl")
        if pca_path.exists():
            with open(pca_path, "rb") as f:
                self.pca = pickle.load(f)
