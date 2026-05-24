"""
02_warm_start/generate_summaries.py

Generates natural-language summaries for each text snippet using a configurable
AI provider (Anthropic, Gemini, DeepSeek, or OpenAI).
These (text → summary) pairs are used to warm-start the AR before RL training.

Output: data/summaries.jsonl with fields {idx, text, summary}
"""

import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY,
    AI_PROVIDER, PROVIDER_MODELS, DATA_DIR,
    WARMSTART_SAMPLES, AV_MAX_TOKENS, MOCK_API
)

SYSTEM_PROMPT = """You are a precise linguistic analyzer describing the content and structure of text fragments.
Given a text fragment, produce a compact structured summary with bolded topic headings (2-3 short paragraphs).
Focus on:
1. The semantic content and topics present
2. The linguistic register, style, and domain
3. What information is encoded at the final token — what has the text established so far?

Be concise (under 120 words). Do not quote the text verbatim."""

VERBALIZATION_PROMPT = """Text fragment (analyze the semantic content up to and including the final token):

{text}

Produce a structured summary describing what is encoded at the final token of this fragment."""


def _make_client(provider: str):
    """Return a (client, model) pair for the chosen provider."""
    model = PROVIDER_MODELS[provider]
    if provider == "local":
        # ── Local GPT-2 LM head (no API) ──────────────────────────────────────
        # For warm-start summaries we prompt GPT-2 to continue a short
        # instruction ("Summarize this text in one sentence:"). Quality is
        # lower than frontier LLMs but zero cost and fully offline.
        # The warm-start checkpoint will have weaker initial FVE, but the RL
        # training loop (step 4) can partially compensate.
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        lm  = GPT2LMHeadModel.from_pretrained("gpt2")
        lm.eval()
        return {"lm": lm, "tok": tok, "device": "cuda" if torch.cuda.is_available() else "cpu"}, model
    elif provider == "anth":
        import anthropic
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), model
    elif provider == "gem":
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY), model
    elif provider == "deep":
        from openai import OpenAI
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"), model
    elif provider == "gpt":
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY), model
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use anth | gem | deep | gpt | local")


def _call_api(client, model: str, provider: str, text: str) -> str:
    """Single API call, provider-specific."""
    prompt = VERBALIZATION_PROMPT.format(text=text[:800])

    if provider == "local":
        # ── Local GPT-2 LM head: prompt-based text continuation ───────────────
        # We prefix a short instruction and let GPT-2 continue it as a
        # summarization task. Quality is much lower than a frontier LLM —
        # GPT-2 has no instruction-following training — but produces a
        # plausible-sounding one-sentence continuation at zero cost.
        import torch
        lm  = client["lm"]
        tok = client["tok"]
        device = client["device"]
        local_prompt = (
            f"Summarize the following text in one sentence:\n"
            f"{text[:300]}\n"
            "Summary:"
        )
        enc = tok(
            local_prompt, return_tensors="pt",
            truncation=True, max_length=256,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        lm = lm.to(device)
        with torch.no_grad():
            out = lm.generate(
                **enc,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=1.3,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    elif provider == "anth":
        import anthropic
        try:
            r = client.messages.create(
                model=model, max_tokens=AV_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text.strip()
        except anthropic.RateLimitError:
            time.sleep(10)
            return _call_api(client, model, provider, text)
        except Exception as e:
            if "credit balance" in str(e).lower() or "billing" in str(e).lower():
                raise RuntimeError(
                    "Anthropic billing error: add credits at "
                    "https://console.anthropic.com/settings/billing"
                ) from e
            raise

    elif provider == "gem":
        from google.genai import types, errors as genai_errors
        import re
        for _attempt in range(6):
            try:
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=AV_MAX_TOKENS,
                    ),
                )
                return r.text.strip()
            except genai_errors.ClientError as e:
                if e.code == 429:
                    m = re.search(r'retry in (\d+)', str(e))
                    wait = int(m.group(1)) + 2 if m else 30
                    print(f"  Gemini quota, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return ""

    else:  # deep or gpt — both use OpenAI-compatible SDK
        r = client.chat.completions.create(
            model=model,
            max_tokens=AV_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return r.choices[0].message.content.strip()


def generate_summary(client, model: str, provider: str, text: str) -> str:
    """Generate one summary, returning '' on non-fatal errors."""
    if MOCK_API:
        words = text.split()[:8]
        return f"[MOCK] Text about: {' '.join(words)}... (debug placeholder)"
    try:
        return _call_api(client, model, provider, text)
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  API error ({provider}): {e}")
        return ""


def generate_all_summaries(
    texts: list[str],
    provider: str = AI_PROVIDER,
    n_samples: int = WARMSTART_SAMPLES,
    resume: bool = True,
) -> list[dict]:
    """
    Generate summaries for the first `n_samples` text snippets.
    Supports resuming from a partial run.

    Returns list of dicts: {idx, text, summary}
    """
    _key_map = {
        "anth": ANTHROPIC_API_KEY,
        "gem":  GEMINI_API_KEY,
        "deep": DEEPSEEK_API_KEY,
        "gpt":  OPENAI_API_KEY,
    }
    if not MOCK_API and not _key_map.get(provider):
        raise ValueError(f"No API key found for provider '{provider}'.")

    client, model = _make_client(provider)
    print(f"Provider: {provider}  |  Model: {model}")
    out_path = DATA_DIR / "summaries.jsonl"

    # Resume from checkpoint (ignore empty files from aborted runs)
    done_indices: set[int] = set()
    records: list[dict] = []
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            for line in f:
                item = json.loads(line)
                done_indices.add(item["idx"])
                records.append(item)
        print(f"Resuming: {len(done_indices)} summaries already done.")

    texts_to_process = [
        (i, t) for i, t in enumerate(texts[:n_samples])
        if i not in done_indices
    ]

    for idx, text in tqdm(texts_to_process, desc="Generating summaries"):
        summary = generate_summary(client, model, provider, text)
        if not summary:
            continue
        record = {"idx": idx, "text": text, "summary": summary}
        with open(out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        records.append(record)
        time.sleep(0.3)  # gentle rate limiting

    if not records:
        raise RuntimeError("No summaries were generated. Check API errors above.")

    print(f"Total summaries: {len(records)}")
    return sorted(records, key=lambda x: x["idx"])


def load_summaries() -> list[dict]:
    """Load previously generated summaries."""
    path = DATA_DIR / "summaries.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate_summaries.py first.")
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "01_data_collection"))
    from extract_activations import load_data

    _, texts, _ = load_data()
    print(f"Loaded {len(texts)} text snippets.")
    records = generate_all_summaries(texts)
    print(f"Generated {len(records)} summaries.")
    print("\nSample summary:")
    print(f"  TEXT: {records[0]['text'][:120]}...")
    print(f"  SUMMARY: {records[0]['summary'][:300]}...")
