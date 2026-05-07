"""
evaluate.py — MME-style evaluation pipeline.

Pipeline overview
-----------------
1. Build a tiny Qwen2VL model with random weights (``model.build_random_model``).
2. Build a dummy processor (``model.DummyProcessor``) that encodes image+text
   into the tensors Qwen2VL expects.
3. Run inference on the toy MME dataset (``dataset.make_toy_dataset``).
4. Extract Yes/No predictions from last-token logits.
5. Compute MME scores (``score.compute_mme_score``).
6. Optionally compare against a VLMEval-compatible baseline.

VLMEval comparison baseline
----------------------------
``compare_with_vlmeval_baseline()`` generates a *synthetic* VLMEval reference
by simulating a model that answers randomly (50 % yes, 50 % no, seeded for
reproducibility).  In a real evaluation environment one would replace this with
results loaded from a VLMEval run (e.g. a JSON produced by ``run.py``).

The function ``vlmeval_format_predictions()`` serialises our predictions in the
TSV format that VLMEval writes to disk, making it easy to diff against an
actual VLMEval output file.

Usage
-----
    python -m vlm_eval.evaluate          # run evaluation and print results
    python -m vlm_eval.evaluate --seed 1 # different random seed
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import torch

from vlm_eval.dataset import MMESample, make_toy_dataset
from vlm_eval.model import (
    YES_TOKEN_ID,
    NO_TOKEN_ID,
    build_random_model,
    DummyProcessor,
)
from vlm_eval.score import (
    PredictionRecord,
    MMEResult,
    compute_mme_score,
    extract_yes_no_from_logits,
    extract_yes_no_from_text,
)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    seed: int = 42,
    verbose: bool = True,
) -> tuple[MMEResult, List[PredictionRecord]]:
    """Run the full MME evaluation pipeline with a random-weight model.

    Steps
    -----
    1. Build model (random weights, no download required).
    2. Build processor (local construction, no download required).
    3. Iterate over toy MME dataset.
    4. For each sample, run a single forward pass and extract yes/no from
       last-token logits.
    5. Accumulate ``PredictionRecord`` objects and compute MME score.

    Args:
        seed:    Random seed used for model initialisation.
        verbose: If True, print per-sample predictions.

    Returns:
        (MMEResult, list of PredictionRecord)
    """
    if verbose:
        print("Building random Qwen2VL model (no pre-trained weights)...")
    model = build_random_model(seed=seed)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

    processor = DummyProcessor()

    dataset: List[MMESample] = make_toy_dataset(seed=seed)
    if verbose:
        print(f"Dataset: {len(dataset)} samples across"
              f" {len(set(s.task for s in dataset))} tasks")
        print()

    records: List[PredictionRecord] = []
    t0 = time.time()

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            inputs = processor(image=sample.image, text=sample.question)

            # Forward pass – only need logits at the last position
            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
                attention_mask=inputs["attention_mask"],
            )

            last_logits = outputs.logits[0, -1]  # [vocab_size]
            prediction = extract_yes_no_from_logits(
                last_logits, YES_TOKEN_ID, NO_TOKEN_ID
            )

            record = PredictionRecord(
                task=sample.task,
                question=sample.question,
                ground_truth=sample.answer,
                prediction=prediction,
            )
            records.append(record)

            if verbose:
                ok = "✓" if prediction == sample.answer else "✗"
                print(
                    f"  [{i+1:2d}/{len(dataset)}] task={sample.task:<10}"
                    f"  gt={sample.answer}  pred={prediction}  {ok}"
                )

    elapsed = time.time() - t0
    if verbose:
        print(f"\nInference time: {elapsed:.2f}s  "
              f"({elapsed/len(dataset)*1000:.0f}ms/sample)")
        print()

    result = compute_mme_score(records)
    return result, records


# ---------------------------------------------------------------------------
# VLMEval comparison helpers
# ---------------------------------------------------------------------------


def _simulate_random_vlmeval(
    dataset: List[MMESample],
    seed: int = 0,
) -> List[PredictionRecord]:
    """Simulate a random-answer VLMEval baseline for comparison.

    In a real setup this would load predictions from a VLMEval result JSON.
    Here we generate 50/50 yes/no answers seeded for reproducibility.

    VLMEval format note: VLMEval stores raw model text in its TSV output and
    then post-processes with extract_answer().  We use extract_yes_no_from_text
    to show the same interface.
    """
    rng = random.Random(seed)
    records = []
    for sample in dataset:
        raw_text = rng.choice(["yes.", "no."])
        prediction = extract_yes_no_from_text(raw_text)
        records.append(
            PredictionRecord(
                task=sample.task,
                question=sample.question,
                ground_truth=sample.answer,
                prediction=prediction,
            )
        )
    return records


def compare_with_vlmeval_baseline(
    our_records: List[PredictionRecord],
    dataset: List[MMESample],
    seed: int = 0,
) -> None:
    """Print a side-by-side comparison of our results vs. a VLMEval baseline.

    Args:
        our_records: Predictions from our evaluation pipeline.
        dataset:     The MME dataset used to generate the baseline.
        seed:        Seed for the synthetic random baseline.
    """
    baseline_records = _simulate_random_vlmeval(dataset, seed=seed)
    baseline_result = compute_mme_score(baseline_records)
    our_result = compute_mme_score(our_records)

    print("=" * 60)
    print("VLMEval-style Comparison")
    print("=" * 60)
    print(f"{'Task':<12}  {'Ours score':>12}  {'Random baseline':>16}")
    print("-" * 60)
    all_tasks = sorted(
        set(our_result.task_results) | set(baseline_result.task_results)
    )
    for task in all_tasks:
        ours = our_result.task_results.get(task)
        base = baseline_result.task_results.get(task)
        our_s = f"{ours.score:.1f}" if ours else "N/A"
        base_s = f"{base.score:.1f}" if base else "N/A"
        print(f"  {task:<10}  {our_s:>12}  {base_s:>16}")
    print("-" * 60)
    print(f"  {'TOTAL':<10}  {our_result.total_score:>12.1f}"
          f"  {baseline_result.total_score:>16.1f}")
    print()
    print("Note: random-weight model expected near-random performance (~100/800).")
    print("Baseline simulates VLMEval's random-answer reference; real VLMEval")
    print("results would be loaded from a JSON file produced by run.py.")


def vlmeval_format_predictions(records: List[PredictionRecord]) -> str:
    """Serialise predictions in VLMEval's TSV output format.

    VLMEval writes evaluation results to files like::

        question_id \\t prediction \\t ground_truth \\n

    This function produces a compatible string for diff / import into
    VLMEval's analysis scripts.

    Args:
        records: Prediction records from our pipeline.

    Returns:
        TSV string with header line.
    """
    lines = ["question_id\ttask\tprediction\tground_truth\tcorrect"]
    for i, r in enumerate(records):
        correct = "1" if r.prediction == r.ground_truth else "0"
        lines.append(
            f"{i}\t{r.task}\t{r.prediction}\t{r.ground_truth}\t{correct}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MME-style evaluation with a random-weight Qwen2VL model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-sample output"
    )
    parser.add_argument(
        "--save-tsv", metavar="PATH",
        help="Save predictions in VLMEval TSV format to PATH"
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Show side-by-side comparison with a simulated VLMEval baseline"
    )
    args = parser.parse_args()

    result, records = run_evaluation(seed=args.seed, verbose=not args.quiet)

    print(result)
    print()

    if args.compare_baseline:
        dataset = make_toy_dataset(seed=args.seed)
        compare_with_vlmeval_baseline(records, dataset, seed=0)

    if args.save_tsv:
        tsv_path = Path(args.save_tsv)
        tsv_path.write_text(vlmeval_format_predictions(records))
        print(f"Predictions saved to {tsv_path}")


if __name__ == "__main__":
    main()
