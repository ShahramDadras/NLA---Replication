"""
main.py — End-to-end NLA replication pipeline.

Run as:
  python main.py [--step STEP] [--device cuda|cpu]

Steps:
  1   Extract activations from GPT-2
  2   Generate warm-start summaries (calls Claude API)
  3   Train AR warm-start (supervised)
  4   Run NLA training (RL loop)
  5   Evaluate: FVE, prediction tasks, behavioral properties
  6   Run case studies (poetry, language switching)
  7   Run confabulation analysis
  8   Generate all figures
  all Run all steps
"""

import argparse
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, SEED,
    TARGET_LAYER, D_MODEL, EVAL_SAMPLES, AI_PROVIDER,
)

np.random.seed(SEED)
torch.manual_seed(SEED)


def step1_extract_activations(device: str) -> tuple:
    """Step 1: Extract GPT-2 residual stream activations."""
    print("\n" + "="*60)
    print("STEP 1: Extracting Activations")
    print("="*60)
    sys.path.insert(0, str(Path(__file__).parent / "01_data_collection"))
    from extract_activations import (
        collect_activations, save_data, load_data,
        compute_activation_statistics
    )

    act_path = DATA_DIR / "activations.npz"
    if act_path.exists():
        print("Found existing activations. Loading...")
        activations, texts, token_lists = load_data()
    else:
        activations, texts, token_lists = collect_activations(device)
        save_data(activations, texts, token_lists)

    print(f"\nActivation statistics:")
    compute_activation_statistics(activations)
    return activations, texts, token_lists


def step2_generate_summaries(texts: list, provider: str = AI_PROVIDER) -> list:
    """Step 2: Generate warm-start summaries via selected AI provider."""
    print("\n" + "="*60)
    print(f"STEP 2: Generating Warm-Start Summaries  [{provider}]")
    print("="*60)
    sys.path.insert(0, str(Path(__file__).parent / "02_warm_start"))
    from generate_summaries import generate_all_summaries, load_summaries

    summ_path = DATA_DIR / "summaries.jsonl"
    if summ_path.exists() and summ_path.stat().st_size > 0:
        print("Found existing summaries. Loading...")
        return load_summaries()

    return generate_all_summaries(texts, provider=provider)


def step3_warmstart_ar(summaries: list, activations: np.ndarray, device: str):
    """Step 3: Train AR with supervised warm-start."""
    print("\n" + "="*60)
    print("STEP 3: Supervised Warm-Start (AR)")
    print("="*60)
    sys.path.insert(0, str(Path(__file__).parent / "02_warm_start"))
    sys.path.insert(0, str(Path(__file__).parent / "03_nla_components"))
    from supervised_warmstart import (
        encode_summaries, train_ar_warmstart, save_ar, load_ar,
        ActivationReconstructor
    )
    from activation_reconstructor import ActivationReconstructorWrapper

    ckpt_path = RESULTS_DIR / "ar_warmstart.pt"

    # Align summaries with activations
    indices = [r["idx"] for r in summaries]
    texts_for_enc = [r["summary"] for r in summaries]
    acts_aligned = activations[indices]

    if ckpt_path.exists():
        print("Found existing warm-start checkpoint. Loading...")
        ar_model, pca = load_ar(device=device)
    else:
        embeddings = encode_summaries(texts_for_enc)
        ar_model, pca, history = train_ar_warmstart(embeddings, acts_aligned, device=device)
        save_ar(ar_model, pca, history)

    ar_wrapper = ActivationReconstructorWrapper(ar_model, device=device, pca=pca)

    # Quick FVE check
    texts_for_enc_sample = texts_for_enc[:50]
    acts_sample = acts_aligned[:50]
    h_hats = ar_wrapper.reconstruct(texts_for_enc_sample)
    sys.path.insert(0, str(Path(__file__).parent / "05_evaluation"))
    from compute_fve import compute_fve
    fve = compute_fve(acts_sample, h_hats)
    print(f"\nWarm-start FVE (on summaries): {fve:.4f}")
    print("  (Expected ~0.3-0.4 per paper; gap due to API proxy AV)")

    return ar_wrapper


def step4_train_nla(activations: np.ndarray, token_lists: list,
                    ar_wrapper, device: str, provider: str = AI_PROVIDER) -> dict:
    """Step 4: Joint NLA training (RL loop)."""
    print("\n" + "="*60)
    print("STEP 4: NLA Training (RL loop)")
    print("="*60)
    sys.path.insert(0, str(Path(__file__).parent / "04_training"))
    sys.path.insert(0, str(Path(__file__).parent / "03_nla_components"))
    from train_nla import run_training, load_training_log
    from activation_verbalizer import ActivationVerbalizer

    trained_path = RESULTS_DIR / "ar_trained.pt"
    log_path = RESULTS_DIR / "training_log.jsonl"

    if trained_path.exists() and log_path.exists():
        print("Found existing training checkpoint. Loading...")
        ar_wrapper.load(trained_path)
        return {"loaded": True}

    av = ActivationVerbalizer(layer=TARGET_LAYER, provider=provider)
    history = run_training(
        activations, token_lists, ar_wrapper, av, device=device, provider=provider
    )
    return history


def step5_evaluate(activations: np.ndarray, texts: list,
                   token_lists: list, ar_wrapper, device: str,
                   provider: str = AI_PROVIDER) -> dict:
    """Step 5: Full evaluation suite."""
    print("\n" + "="*60)
    print("STEP 5: Evaluation")
    print("="*60)

    sys.path.insert(0, str(Path(__file__).parent / "03_nla_components"))
    sys.path.insert(0, str(Path(__file__).parent / "05_evaluation"))
    from activation_verbalizer import ActivationVerbalizer, load_explanations
    from compute_fve import evaluate_reconstruction, save_evaluation_results
    from behavioral_properties import run_behavioral_analysis

    # Verbalize a sample
    exp_path = DATA_DIR / "explanations.jsonl"
    av = ActivationVerbalizer(layer=TARGET_LAYER, provider=provider)

    sample_acts = activations[:EVAL_SAMPLES]
    sample_tokens = token_lists[:EVAL_SAMPLES]
    sample_texts = texts[:EVAL_SAMPLES]

    if exp_path.exists():
        records = load_explanations(exp_path)
        explanations = [r["explanation"] for r in records[:EVAL_SAMPLES]]
    else:
        explanations = av.verbalize_batch(sample_acts, sample_tokens)
        # Save
        import json
        with open(exp_path, "w") as f:
            for i, (exp, toks) in enumerate(zip(explanations, sample_tokens)):
                f.write(json.dumps({"idx": i, "tokens": toks[-10:], "explanation": exp}) + "\n")

    # FVE evaluation
    h_hats = ar_wrapper.reconstruct(explanations)
    eval_results = evaluate_reconstruction(sample_acts, h_hats, label="NLA (this work)")
    eval_results["per_dim_fve"] = eval_results.get("per_dim_fve", [])
    save_evaluation_results(eval_results)

    # Behavioral properties
    run_behavioral_analysis(
        sample_texts, explanations,
        run_confabulation=(provider != "local"),
        checkpoint_label="final",
    )

    # Prediction tasks
    from prediction_tasks import run_all_prediction_tasks
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    task_results = run_all_prediction_tasks(
        sample_texts, explanations, tokenizer,
        checkpoint_label="final", provider=provider,
    )

    return {
        "fve": eval_results["fve"],
        "task_results": task_results,
        "explanations": explanations,
    }


def step6_case_studies(device: str, provider: str = AI_PROVIDER) -> None:
    """Step 6: Run qualitative case studies."""
    print("\n" + "="*60)
    print("STEP 6: Case Studies")
    print("="*60)

    # Poetry case study
    print("\n--- Planning in Poetry ---")
    sys.path.insert(0, str(Path(__file__).parent / "06_case_studies"))
    from planning_in_poetry import run_poetry_case_study
    run_poetry_case_study(device=device, provider=provider)

    # Language switching (optional — only runs if the module exists)
    try:
        print("\n--- Language Switching ---")
        from language_switching import run_language_switching_analysis
        run_language_switching_analysis(device=device, provider=provider)
    except ImportError:
        print("  (Skipping: language_switching.py not found)")


def step7_confabulation_analysis(texts: list, explanations: list) -> None:
    """Step 7: Systematic confabulation analysis."""
    print("\n" + "="*60)
    print("STEP 7: Confabulation Analysis")
    print("="*60)
    sys.path.insert(0, str(Path(__file__).parent / "07_analysis"))
    from confabulation_analysis import run_confabulation_analysis
    run_confabulation_analysis(texts, explanations, n_sample=40)


def step8_generate_figures() -> None:
    """Step 8: Generate all README figures."""
    print("\n" + "="*60)
    print("STEP 8: Generating Figures")
    print("="*60)
    sys.path.insert(0, str(Path(__file__).parent / "08_plots"))
    from generate_all_plots import generate_all_figures
    generate_all_figures()


def run_all(device: str = "cpu", provider: str = AI_PROVIDER) -> None:
    """Run the complete pipeline end to end."""
    start = time.time()

    activations, texts, token_lists = step1_extract_activations(device)
    summaries = step2_generate_summaries(texts, provider=provider)
    ar_wrapper = step3_warmstart_ar(summaries, activations, device)
    step4_train_nla(activations, token_lists, ar_wrapper, device, provider=provider)

    eval_out = step5_evaluate(activations, texts, token_lists, ar_wrapper, device, provider=provider)

    step6_case_studies(device, provider=provider)
    if eval_out.get("explanations"):
        step7_confabulation_analysis(texts[:EVAL_SAMPLES], eval_out["explanations"])
    step8_generate_figures()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes.")
    print(f"Results in:  {RESULTS_DIR}")
    print(f"Figures in:  {FIGURES_DIR}")
    print(f"Final FVE:   {eval_out.get('fve', 'N/A'):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLA Replication Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["1","2","3","4","5","6","7","8","all"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ai", type=str, default=AI_PROVIDER,
                        choices=["anth", "gem", "deep", "gpt", "local"],
                        help="AI provider for verbalizer: anth=Claude, gem=Gemini, "
                             "deep=DeepSeek, gpt=OpenAI, local=GPT-2 (no API)")
    args = parser.parse_args()

    print(f"Device: {args.device}  |  AI provider: {args.ai}")

    if args.step == "all":
        run_all(args.device, provider=args.ai)
    elif args.step == "1":
        step1_extract_activations(args.device)
    elif args.step == "8":
        step8_generate_figures()
    elif args.step == "2":
        activations, texts, token_lists = step1_extract_activations(args.device)
        step2_generate_summaries(texts, provider=args.ai)
    elif args.step == "3":
        activations, texts, token_lists = step1_extract_activations(args.device)
        summaries = step2_generate_summaries(texts, provider=args.ai)
        step3_warmstart_ar(summaries, activations, args.device)
    elif args.step == "4":
        activations, texts, token_lists = step1_extract_activations(args.device)
        summaries = step2_generate_summaries(texts, provider=args.ai)
        ar_wrapper = step3_warmstart_ar(summaries, activations, args.device)
        step4_train_nla(activations, token_lists, ar_wrapper, args.device, provider=args.ai)
    elif args.step == "5":
        activations, texts, token_lists = step1_extract_activations(args.device)
        summaries = step2_generate_summaries(texts, provider=args.ai)
        ar_wrapper = step3_warmstart_ar(summaries, activations, args.device)
        step4_train_nla(activations, token_lists, ar_wrapper, args.device, provider=args.ai)
        step5_evaluate(activations, texts, token_lists, ar_wrapper, args.device, provider=args.ai)
    elif args.step == "6":
        step6_case_studies(args.device, provider=args.ai)
    elif args.step == "7":
        activations, texts, token_lists = step1_extract_activations(args.device)
        summaries = step2_generate_summaries(texts, provider=args.ai)
        ar_wrapper = step3_warmstart_ar(summaries, activations, args.device)
        step4_train_nla(activations, token_lists, ar_wrapper, args.device, provider=args.ai)
        eval_out = step5_evaluate(activations, texts, token_lists, ar_wrapper, args.device, provider=args.ai)
        step7_confabulation_analysis(texts[:EVAL_SAMPLES], eval_out["explanations"])
