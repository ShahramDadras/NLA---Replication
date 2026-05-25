"""
05_evaluation/prediction_tasks.py

Five prediction tasks for evaluating NLA explanation quality.

Adapted from Section "NLA Evaluations Improve With Training" in the paper.
The paper defines prediction tasks where an LLM must answer questions about
a context using ONLY the NLA explanation (no access to original text).
Performance should improve with NLA training if explanations are informative.

Tasks implemented (adapted for GPT-2 + open datasets):
  1. NEXT_TOKEN_PREDICTION  — predict the next token from the explanation alone
  2. DOMAIN_CLASSIFICATION  — classify the domain (science/news/literature/sports)
  3. TOPIC_EXTRACTION       — extract the main topic word
  4. SENTIMENT_DETECTION    — detect sentiment (positive/negative/neutral)
  5. GENDER_INFERENCE       — infer subject gender from context (as in paper: "user traits")

Judge modes:
  api   — Gemini or Claude judges via API (requires credits/quota)
  local — text-matching heuristics applied to the explanation (no API needed)

The local mode checks whether the relevant information is present in the
explanation text — a valid information-retrieval metric for local verbalizers.
"""

import sys
import json
import time
import re
import numpy as np
import anthropic
from pathlib import Path
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, GEMINI_API_KEY, PROVIDER_MODELS,
    DATA_DIR, RESULTS_DIR,
    PREDICTION_TASK_SAMPLES, SEED, AI_PROVIDER,
)


JUDGE_SYSTEM = """You are a precise evaluator. Answer the given question based ONLY on the
provided NLA explanation. Do not use prior knowledge. Answer in one word or a short phrase."""


def call_judge_api(client, explanation: str, question: str, provider: str) -> str:
    """Ask the configured judge to answer based ONLY on the NLA explanation."""
    prompt = (
        f"NLA explanation:\n{explanation}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the explanation above. Use one word or a short phrase."
    )
    try:
        if provider == "anth":
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=50,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip().lower()

        if provider == "gem":
            from google.genai import types
            response = client.models.generate_content(
                model=PROVIDER_MODELS["gem"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM,
                    max_output_tokens=128,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (response.text or "").strip().lower()
    except Exception as e:
        if "credit balance" in str(e).lower() or "billing" in str(e).lower():
            raise
        return ""
    return ""


# ── Local heuristic judges (no API) ───────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "science": ["experiment", "hypothesis", "molecule", "physics", "biology", "chemistry",
                "research", "laboratory", "data", "study"],
    "news":    ["president", "government", "election", "minister", "policy", "reported",
                "official", "announced", "political", "congress"],
    "literature": ["poem", "novel", "character", "story", "author", "wrote", "literary",
                   "fiction", "narrative", "verse"],
    "sports":  ["game", "player", "team", "score", "match", "championship", "league",
                "season", "coach", "tournament"],
}

POSITIVE_WORDS = ["good", "great", "excellent", "success", "happy", "positive",
                  "wonderful", "love", "best", "improve", "achieve", "benefit"]
NEGATIVE_WORDS = ["bad", "poor", "fail", "crisis", "problem", "difficult", "worse",
                  "terrible", "death", "loss", "decline", "attack", "war"]


def infer_domain(text: str) -> Optional[str]:
    """Simple heuristic domain label for a text snippet."""
    text_lower = text.lower()
    scores = {domain: sum(1 for kw in kws if kw in text_lower)
              for domain, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None


def infer_sentiment(text: str) -> str:
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    if pos > neg + 1: return "positive"
    if neg > pos + 1: return "negative"
    return "neutral"


def infer_gender(text: str) -> Optional[str]:
    text_lower = text.lower()
    he_count  = len(re.findall(r'\bhe\b|\bhis\b|\bhim\b',   text_lower))
    she_count = len(re.findall(r'\bshe\b|\bher\b|\bhers\b', text_lower))
    if he_count >= 3 and she_count == 0: return "male"
    if she_count >= 3 and he_count == 0: return "female"
    return None


def extract_main_noun(text: str) -> Optional[str]:
    """Extract the most frequent content noun (rough proxy for topic)."""
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    if not words:
        return None
    from collections import Counter
    counts = Counter(words)
    stopwords = {"The", "This", "That", "In", "It", "He", "She", "They", "We",
                 "A", "An", "I", "His", "Her", "Its", "Their"}
    filtered = {w: c for w, c in counts.items() if w not in stopwords}
    if not filtered:
        return None
    return max(filtered, key=filtered.get).lower()


# ── Task implementations ───────────────────────────────────────────────────────

def task_next_token_prediction(
    client,          # anthropic.Anthropic or None (local mode)
    texts: list[str],
    explanations: list[str],
    tokenizer,
    n: int = PREDICTION_TASK_SAMPLES,
    use_api: bool = False,
    provider: str = AI_PROVIDER,
) -> dict:
    """
    Can the judge predict the next token from the explanation alone?
    Local mode: check whether the true next token appears in the explanation.
    """
    correct = 0
    results = []

    for text, exp in tqdm(zip(texts[:n], explanations[:n]), total=n, desc="Task1: NextToken"):
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 5:
            continue
        true_next = tokens[-1].replace("Ġ", "").lower()

        if use_api:
            question = "What single word does the model most likely predict will come next?"
            try:
                pred = call_judge_api(client, exp, question, provider)
            except Exception:
                break
            pred_clean = pred.split()[0] if pred.split() else ""
            is_correct = (true_next in pred_clean or pred_clean in true_next)
        else:
            # Local: does the explanation mention the actual next token?
            is_correct = bool(true_next and true_next in exp.lower())
            pred_clean = true_next if is_correct else ""

        correct += int(is_correct)
        results.append({"true": true_next, "pred": pred_clean, "correct": is_correct})
        if use_api:
            time.sleep(0.2)

    acc = correct / n if n > 0 else 0
    return {"task": "next_token_prediction", "accuracy": acc, "n": n, "results": results}


def task_domain_classification(
    client,
    texts: list[str],
    explanations: list[str],
    n: int = PREDICTION_TASK_SAMPLES,
    use_api: bool = False,
    provider: str = AI_PROVIDER,
) -> dict:
    """Can the judge classify the domain from the explanation?"""
    pairs = [(t, e) for t, e in zip(texts, explanations) if infer_domain(t) is not None][:n]
    correct = 0
    results = []
    domains = list(DOMAIN_KEYWORDS.keys())

    for text, exp in tqdm(pairs, desc="Task2: Domain"):
        true_domain = infer_domain(text)

        if use_api:
            question = f"Which domain does this text belong to? Choose from: {', '.join(domains)}"
            try:
                pred = call_judge_api(client, exp, question, provider)
            except Exception:
                break
            is_correct = any(d in pred for d in [true_domain])
        else:
            pred_domain = infer_domain(exp)
            pred = pred_domain or "unknown"
            is_correct = (pred_domain == true_domain)

        correct += int(is_correct)
        results.append({"true": true_domain, "pred": pred, "correct": is_correct})
        if use_api:
            time.sleep(0.2)

    acc = correct / len(pairs) if pairs else 0
    return {"task": "domain_classification", "accuracy": acc, "n": len(pairs), "results": results}


def task_topic_extraction(
    client,
    texts: list[str],
    explanations: list[str],
    n: int = PREDICTION_TASK_SAMPLES,
    use_api: bool = False,
    provider: str = AI_PROVIDER,
) -> dict:
    """Can the judge identify the main topic from the explanation?"""
    pairs = [(t, e) for t, e in zip(texts, explanations) if extract_main_noun(t)][:n]
    correct = 0
    results = []

    for text, exp in tqdm(pairs, desc="Task3: Topic"):
        true_topic = extract_main_noun(text)

        if use_api:
            question = "What is the single main topic or subject? (one word)"
            try:
                pred = call_judge_api(client, exp, question, provider)
            except Exception:
                break
            is_correct = true_topic in pred.lower() if true_topic else False
        else:
            # Local: does the explanation mention the topic word?
            is_correct = bool(true_topic and true_topic in exp.lower())
            pred = true_topic if is_correct else ""

        correct += int(is_correct)
        results.append({"true": true_topic, "pred": pred, "correct": is_correct})
        if use_api:
            time.sleep(0.2)

    acc = correct / len(pairs) if pairs else 0
    return {"task": "topic_extraction", "accuracy": acc, "n": len(pairs), "results": results}


def task_sentiment_detection(
    client,
    texts: list[str],
    explanations: list[str],
    n: int = PREDICTION_TASK_SAMPLES,
    use_api: bool = False,
    provider: str = AI_PROVIDER,
) -> dict:
    """Can the judge detect sentiment from the explanation?"""
    pairs = [(t, e) for t, e in zip(texts, explanations)
             if infer_sentiment(t) != "neutral"][:n]
    correct = 0
    results = []

    for text, exp in tqdm(pairs, desc="Task4: Sentiment"):
        true_sent = infer_sentiment(text)

        if use_api:
            question = "What is the overall sentiment? (positive, negative, or neutral)"
            try:
                pred = call_judge_api(client, exp, question, provider)
            except Exception:
                break
            is_correct = true_sent in pred
        else:
            pred = infer_sentiment(exp)
            is_correct = (pred == true_sent)

        correct += int(is_correct)
        results.append({"true": true_sent, "pred": pred, "correct": is_correct})
        if use_api:
            time.sleep(0.2)

    acc = correct / len(pairs) if pairs else 0
    return {"task": "sentiment_detection", "accuracy": acc, "n": len(pairs), "results": results}


def task_gender_inference(
    client,
    texts: list[str],
    explanations: list[str],
    n: int = PREDICTION_TASK_SAMPLES,
    use_api: bool = False,
    provider: str = AI_PROVIDER,
) -> dict:
    """Can the judge infer the subject's gender from the explanation?"""
    pairs = [(t, e) for t, e in zip(texts, explanations) if infer_gender(t)][:n]
    correct = 0
    results = []

    for text, exp in tqdm(pairs, desc="Task5: Gender"):
        true_gender = infer_gender(text)

        if use_api:
            question = "What is the gender of the main subject? (male or female)"
            try:
                pred = call_judge_api(client, exp, question, provider)
            except Exception:
                break
            is_correct = true_gender in pred.lower()
        else:
            pred_gender = infer_gender(exp)
            pred = pred_gender or "unknown"
            is_correct = (pred_gender == true_gender)

        correct += int(is_correct)
        results.append({"true": true_gender, "pred": pred, "correct": is_correct})
        if use_api:
            time.sleep(0.2)

    acc = correct / len(pairs) if pairs else 0
    return {"task": "gender_inference", "accuracy": acc, "n": len(pairs), "results": results}


# ── Run all tasks ─────────────────────────────────────────────────────────────

def run_all_prediction_tasks(
    texts: list[str],
    explanations: list[str],
    tokenizer=None,
    checkpoint_label: str = "final",
    provider: str = AI_PROVIDER,
) -> dict:
    """
    Run all five prediction tasks and save results.

    provider="gem"   → Gemini as judge (uses Google AI Studio billing/quota).
    provider="anth"  → Claude as judge (uses Anthropic credits).
    other providers  → text-matching heuristics, no judge API calls.
    Returns dict mapping task_name → accuracy.
    """
    use_api = provider in ("anth", "gem")
    if provider == "anth" and ANTHROPIC_API_KEY:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif provider == "gem" and GEMINI_API_KEY:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        use_api = False
        client = None

    if use_api:
        print(f"  Prediction tasks: using {provider} API judge")
    else:
        print("  Prediction tasks: using local text-matching judge (no API)")

    task_results = {}

    if tokenizer:
        r1 = task_next_token_prediction(
            client, texts, explanations, tokenizer, use_api=use_api, provider=provider
        )
        task_results["next_token_prediction"] = r1["accuracy"]

    r2 = task_domain_classification(
        client, texts, explanations, use_api=use_api, provider=provider
    )
    task_results["domain_classification"] = r2["accuracy"]

    r3 = task_topic_extraction(
        client, texts, explanations, use_api=use_api, provider=provider
    )
    task_results["topic_extraction"] = r3["accuracy"]

    r4 = task_sentiment_detection(
        client, texts, explanations, use_api=use_api, provider=provider
    )
    task_results["sentiment_detection"] = r4["accuracy"]

    r5 = task_gender_inference(
        client, texts, explanations, use_api=use_api, provider=provider
    )
    task_results["gender_inference"] = r5["accuracy"]

    print("\nPrediction Task Results:")
    for task, acc in task_results.items():
        print(f"  {task:<30} accuracy = {acc:.3f}")

    out = {"checkpoint": checkpoint_label, "tasks": task_results, "judge": provider}
    out_path = RESULTS_DIR / f"prediction_tasks_{checkpoint_label}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")

    return task_results
