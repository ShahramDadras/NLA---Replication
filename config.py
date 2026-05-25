"""
config.py — Central configuration for the NLA replication.

Design choices documented here:
- Target model: GPT-2 small (117M params). Small enough for free-tier GPU (T4/P100),
  well-studied, publicly available. d_model=768, 12 layers.
- Target layer: layer 8 (index 7). Middle-to-late, per paper recommendation.
- AV: Gemini API (gemini-2.5-flash) as proxy verbalizer.
  Justification: we cannot fine-tune GPT-2 to inject raw activation embeddings
  without a full training run; the API approach lets us study the pipeline
  faithfully while staying within compute constraints.
- AR: Sentence-transformer embedding of AV output → MLP → activation vector.
  Trained with MSE on (description, activation) pairs.
- Dataset: WikiText-2 (freely available, pretraining-like, standard NLP benchmark).
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

for _d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    _d.mkdir(exist_ok=True)

# ── Target model ───────────────────────────────────────────────────────────────
TARGET_MODEL_NAME = "gpt2"          # HuggingFace model id
TARGET_LAYER = 7                    # 0-indexed; layer 8 of 12 (middle-to-late)
D_MODEL = 768                       # residual stream dimension for GPT-2

# ── Dataset ────────────────────────────────────────────────────────────────────
# The paper uses Anthropic-internal data, which is inaccessible. WikiText-2 is
# the natural proxy: it is GPT-2's standard pretraining benchmark, so activations
# are in-distribution and the texts are long enough to produce rich final-token
# representations. The official code for open models similarly uses standard
# web-text corpora (The Pile / OpenWebText).
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_SPLIT = "train"
MAX_SAMPLES = 500                   # number of text snippets to process
MAX_TOKEN_LENGTH = 128              # truncate inputs for speed
MIN_TOKEN_LENGTH = 20               # skip very short snippets

# ── Warm-start SFT ─────────────────────────────────────────────────────────────
WARMSTART_SAMPLES = 300             # (h_l, summary) pairs for SFT
WARMSTART_EPOCHS = 3
WARMSTART_LR = 1e-4

# ── NLA training (RL loop) ─────────────────────────────────────────────────────
TRAINING_STEPS = 200                # RL iterations (each step = one batch)
TRAIN_BATCH_SIZE = 8
GRPO_GROUP_SIZE = 4                 # number of candidate descriptions per activation
KL_BETA = 0.01                      # KL penalty coefficient
AR_LR = 1e-4
AV_LR = 5e-6
LOG_REWARD_TRANSFORM = True         # use r = -log||h - AR(z)||² per paper

# ── Activation normalization ───────────────────────────────────────────────────
NORMALIZE_ACTIVATIONS = True        # unit L2-norm per paper

# ── AR model ──────────────────────────────────────────────────────────────────
# Reconstructor: sentence-transformer encoder + MLP head
SENTENCE_ENCODER = "all-MiniLM-L6-v2"   # 384-dim sentence embeddings
AR_HIDDEN_DIM = 512
AR_NUM_LAYERS = 3

# PCA dimensionality reduction applied to activations before AR training.
# The MLP predicts PCA coordinates (20-dim) instead of raw activations (768-dim).
# Rationale: 768→20 makes the regression 38× easier; 20 PCA components typically
# capture >90% of activation variance. The inverse transform recovers 768-dim
# for FVE computation and reward calculation.
PCA_COMPONENTS = 20

# ── API Keys ──────────────────────────────────────────────────────────────────
_keys_dir = ROOT / "API-Keys"

def _read_key(filename: str, env_var: str = "") -> str:
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    p = _keys_dir / filename
    return p.read_text().strip() if p.exists() else ""

ANTHROPIC_API_KEY = _read_key("Anth-API-key.txt",    "ANTHROPIC_API_KEY")
GEMINI_API_KEY    = _read_key("Gemini-API-key.txt",   "GEMINI_API_KEY")
DEEPSEEK_API_KEY  = _read_key("DeepSeek-API-key.txt", "DEEPSEEK_API_KEY")
OPENAI_API_KEY    = _read_key("ChatGPT-Api-key.txt",  "OPENAI_API_KEY")

# ── AI Provider ───────────────────────────────────────────────────────────────
# Which provider to use for the verbalizer (AV) and warm-start summaries.
# Overridden by --ai CLI arg. Choices: "anth" | "gem" | "deep" | "gpt" | "local"
AI_PROVIDER = "gem"

# Model names per provider
PROVIDER_MODELS = {
    # ── External API providers ─────────────────────────────────────────────────
    # These produce high-quality verbalizations but require credits/quota.
    "anth":  "claude-sonnet-4-6",      # best quality; needs Anthropic credits
    "gem":   "gemini-2.5-flash",  # free tier but strict RPM quota
    "deep":  "deepseek-chat",          # cheapest paid option (~$0.001/call)
    "gpt":   "gpt-4o-mini",            # OpenAI; moderate cost
    # ── Local provider (no API, no cost) ──────────────────────────────────────
    # Uses GPT-2's own LM head to verbalize its activations locally.
    # Quality is lower than frontier LLMs (GPT-2 cannot introspect itself
    # the way a larger model can), but the full pipeline runs with zero API
    # calls, making it ideal for Colab free tier / offline debugging.
    # FVE gap vs paper is larger (~0.05-0.15 vs 0.60-0.80) because the
    # verbalizer has no external world knowledge to describe what it sees.
    "local": "gpt2",
}

CLAUDE_MODEL = PROVIDER_MODELS["anth"]  # backward-compat alias
AV_MAX_TOKENS = 300                 # max tokens for verbalizer output
AV_TEMPERATURE = 1.0                # sample temperature (T=1 per paper)
# Set True to skip real API calls and use placeholder text — for pipeline debugging only.
MOCK_API = False

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_SAMPLES = 100                  # samples for FVE and prediction tasks
EVAL_BATCH_SIZE = 16
PREDICTION_TASK_SAMPLES = 50       # per prediction task

# ── Plotting ──────────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE = (10, 6)
COLOR_PALETTE = {
    "primary":   "#1f77b4",
    "secondary": "#ff7f0e",
    "accent":    "#2ca02c",
    "neutral":   "#7f7f7f",
    "highlight": "#d62728",
    "purple":    "#9467bd",
}
FONT_SIZE = 12
TITLE_SIZE = 14

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
