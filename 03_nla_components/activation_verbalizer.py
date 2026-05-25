"""
03_nla_components/activation_verbalizer.py

Activation Verbalizer (AV): maps an activation vector + context → natural language explanation.

Paper: "The AV is an LLM ... given a fixed prompt containing both instructions
to verbalize the contents of an activation and a special token for the activation itself."

Providers (select with --ai flag):
  anth  — Claude API (best quality; needs Anthropic credits)
          Ideal proxy for the paper's methodology. Cannot fine-tune GPT-2 to
          inject raw activations without a full training run, so the API is the
          most faithful approximation within compute constraints.

  gem   — Gemini API (free tier; strict RPM quota)
          Good quality but rate-limited aggressively on the free plan.

  deep  — DeepSeek API (~$0.001/call; OpenAI-compatible)
          Best cost/quality tradeoff when a paid key is available.

  gpt   — OpenAI GPT-4o-mini (moderate cost)

  local — GPT-2 LM head, fully offline, zero API cost.
          Prompts GPT-2 with a structured summary of its own activation
          (top-N dimension indices + values) and lets it generate a
          natural-language continuation. Quality is lower than frontier
          LLMs — GPT-2 has no external world knowledge to describe what
          its own internals mean — but the full NLA pipeline runs without
          any network calls, making it ideal for Colab free tier or
          offline debugging. Expected FVE ~0.05–0.15 vs 0.60–0.80 in paper.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY,
    AI_PROVIDER, PROVIDER_MODELS, AV_MAX_TOKENS,
    AV_TEMPERATURE, GRPO_GROUP_SIZE, MOCK_API
)

VERBALIZER_SYSTEM = """You are an activation verbalizer for a language model interpretability system.
You receive:
1. A context: the tokens that the language model processed
2. An activation vector summary from an intermediate layer of the model

Your task: generate a structured natural-language explanation of what semantic, syntactic,
and contextual information is likely encoded in this activation. Use 2-4 short paragraphs
with **bolded topic headings**. Be specific. Focus on:
- What topic/domain/concept the model is tracking
- What linguistic or grammatical features are salient
- What the model likely predicts will come next
- Any notable patterns in the high/low activation dimensions

Keep your explanation under 200 words. Do not hallucinate facts not implied by the context."""

VERBALIZER_USER_TEMPLATE = """**Context tokens**: {tokens}

**Activation statistics** (layer {layer}, final token):
- Shape: {d_model} dimensions
- First 32 dims (scaled): {first_32}
- Activation norm: {norm:.4f}
- Top positive dims (indices): {top_pos_dims}
- Top negative dims (indices): {top_neg_dims}
- Mean: {mean:.4f}, Std: {std:.4f}

Generate a structured explanation of what is encoded in this activation."""


def format_activation_for_prompt(
    activation: np.ndarray,
    tokens: list[str],
    layer: int = 7,
    n_display: int = 32,
) -> str:
    """
    Format an activation vector into a human-readable prompt string.
    We show the first n_display dimensions and top/bottom indices.
    """
    d = len(activation)
    first_32 = [f"{v:.3f}" for v in activation[:n_display]]
    top_pos = np.argsort(activation)[-8:][::-1].tolist()
    top_neg = np.argsort(activation)[:8].tolist()

    return VERBALIZER_USER_TEMPLATE.format(
        tokens=" ".join(tokens[-30:]),  # last 30 tokens for context
        layer=layer,
        d_model=d,
        first_32=", ".join(first_32),
        norm=float(np.linalg.norm(activation)),
        top_pos_dims=str(top_pos),
        top_neg_dims=str(top_neg),
        mean=float(activation.mean()),
        std=float(activation.std()),
    )


class ActivationVerbalizer:
    """
    Wraps the Claude API to act as the Activation Verbalizer.

    Methods
    -------
    verbalize(activation, tokens) → str
        Generate one explanation for a single activation.

    verbalize_group(activation, tokens, n) → list[str]
        Generate n candidate explanations (for GRPO group sampling).

    verbalize_batch(activations, token_lists) → list[str]
        Verbalize a batch of activations.
    """

    def __init__(self, layer: int = 7, provider: str = AI_PROVIDER):
        self.layer = layer
        self.provider = provider
        self.model = PROVIDER_MODELS[provider]
        self._call_count = 0

        if MOCK_API:
            self.client = None
            return

        if provider == "local":
            # ── Local GPT-2 LM head (no API) ──────────────────────────────────
            # Load the same GPT-2 used as target model but with the LM head so
            # we can generate text. The model is kept frozen — we only need
            # inference, not fine-tuning, for verbalization.
            # Why not fine-tune? Fine-tuning GPT-2 to accept raw activation
            # embeddings (as the paper does) requires modifying the embedding
            # layer and a full training run — impractical here. Instead we
            # prompt it with a human-readable activation summary and let it
            # continue the text, which is a zero-cost approximation.
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self._local_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._local_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self._local_tokenizer.pad_token = self._local_tokenizer.eos_token
            self._local_lm = GPT2LMHeadModel.from_pretrained("gpt2").to(self._local_device)
            self._local_lm.eval()
            self.client = None  # no HTTP client needed

        elif provider == "anth":
            import anthropic
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set.")
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        elif provider == "gem":
            from google import genai
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set.")
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        elif provider == "deep":
            from openai import OpenAI
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        elif provider == "gpt":
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    def _call_api(self, user_content: str, temperature: float = AV_TEMPERATURE) -> str:
        """Single API call with exponential backoff, provider-aware."""
        if MOCK_API:
            return f"[MOCK AV] provider={self.provider} layer={self.layer}"

        # ── Local provider: GPT-2 LM head, no HTTP call ────────────────────────
        # We extract the top-8 most active dimensions from the activation vector
        # and encode them as a readable string inside the prompt. GPT-2 then
        # continues the prompt as a text completion.
        # Limitation: GPT-2 generates plausible-sounding continuations but has
        # no ground-truth knowledge of what each dimension means, so the
        # descriptions are weaker than a frontier LLM's. The FVE is lower as a
        # result, but the pipeline structure is identical to the API variants.
        if self.provider == "local":
            import torch
            # Parse context tokens out of the structured prompt string
            ctx_line = next((l for l in user_content.split("\n") if "Context tokens" in l), "")
            ctx = ctx_line.split("**Context tokens**:")[-1].strip()[:120]

            # Build a compact activation fingerprint from the raw activation
            # (user_content already contains dim stats; re-parse the top dims line)
            top_line = next((l for l in user_content.split("\n") if "Top positive" in l), "")
            local_prompt = (
                f'Context: "{ctx}"\n'
                f"Activation: [{top_line.split(':', 1)[-1].strip()}]\n"
                "In one sentence, this activation encodes:"
            )
            enc = self._local_tokenizer(
                local_prompt, return_tensors="pt",
                truncation=True, max_length=200,
            ).to(self._local_device)
            with torch.no_grad():
                out = self._local_lm.generate(
                    **enc,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._local_tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )
            new_tokens = out[0][enc["input_ids"].shape[1]:]
            return self._local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for attempt in range(5):
            try:
                if self.provider == "anth":
                    import anthropic
                    r = self.client.messages.create(
                        model=self.model, max_tokens=AV_MAX_TOKENS,
                        temperature=temperature, system=VERBALIZER_SYSTEM,
                        messages=[{"role": "user", "content": user_content}],
                    )
                    self._call_count += 1
                    return r.content[0].text.strip()

                elif self.provider == "gem":
                    from google.genai import types, errors as genai_errors
                    try:
                        r = self.client.models.generate_content(
                            model=self.model, contents=user_content,
                            config=types.GenerateContentConfig(
                                system_instruction=VERBALIZER_SYSTEM,
                                max_output_tokens=AV_MAX_TOKENS,
                                temperature=temperature,
                            ),
                        )
                        self._call_count += 1
                        return r.text.strip()
                    except genai_errors.ClientError as e:
                        if e.code == 429:
                            import re
                            m = re.search(r'retry in (\d+)', str(e))
                            wait = int(m.group(1)) + 2 if m else 30
                            print(f"  Gemini quota hit, waiting {wait}s...")
                            time.sleep(wait)
                            continue
                        raise

                else:  # deep or gpt

                    r = self.client.chat.completions.create(
                        model=self.model, max_tokens=AV_MAX_TOKENS,
                        temperature=temperature,
                        messages=[
                            {"role": "system", "content": VERBALIZER_SYSTEM},
                            {"role": "user",   "content": user_content},
                        ],
                    )
                    self._call_count += 1
                    return r.choices[0].message.content.strip()

            except Exception as e:
                print(f"  API error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        return "[AV FAILED]"

    def verbalize(
        self,
        activation: np.ndarray,
        tokens: list[str],
    ) -> str:
        """Generate a single explanation for one activation."""
        prompt = format_activation_for_prompt(activation, tokens, self.layer)
        return self._call_api(prompt)

    def verbalize_group(
        self,
        activation: np.ndarray,
        tokens: list[str],
        n: int = GRPO_GROUP_SIZE,
    ) -> list[str]:
        """
        Sample n candidate explanations (temperature=1) for GRPO.
        Each is an independent sample from the verbalizer distribution.
        """
        prompt = format_activation_for_prompt(activation, tokens, self.layer)
        return [self._call_api(prompt, temperature=1.0) for _ in range(n)]

    def verbalize_batch(
        self,
        activations: np.ndarray,
        token_lists: list[list[str]],
        delay: float = 0.2,
    ) -> list[str]:
        """
        Verbalize a batch of activations sequentially.
        Returns list of explanation strings.
        """
        explanations = []
        for act, tokens in tqdm(
            zip(activations, token_lists),
            total=len(activations),
            desc="Verbalizing activations",
        ):
            exp = self.verbalize(act, tokens)
            explanations.append(exp)
            time.sleep(delay)
        return explanations

    def verbalize_and_save(
        self,
        activations: np.ndarray,
        token_lists: list[list[str]],
        out_path: Path,
        resume: bool = True,
    ) -> list[dict]:
        """
        Verbalize all activations and save to JSONL.
        Supports resuming from partial runs.
        """
        done = {}
        records = []
        if resume and out_path.exists():
            with open(out_path) as f:
                for line in f:
                    item = json.loads(line)
                    done[item["idx"]] = item
                    records.append(item)
            print(f"  Resuming: {len(done)} already verbalized.")

        with open(out_path, "a") as f:
            for i, (act, tokens) in enumerate(
                tqdm(zip(activations, token_lists), total=len(activations))
            ):
                if i in done:
                    continue
                explanation = self.verbalize(act, tokens)
                record = {
                    "idx": i,
                    "tokens": tokens[-20:],
                    "explanation": explanation,
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)
                time.sleep(0.3)

        return sorted(records, key=lambda x: x["idx"])


def load_explanations(path: Path) -> list[dict]:
    """Load saved verbalizations from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return sorted(records, key=lambda x: x["idx"])
