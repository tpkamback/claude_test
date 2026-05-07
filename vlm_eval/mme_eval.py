"""
mme_eval.py — MME-style evaluation for VLMs (self-implemented)

Reference implementation:
  VLMEvalKit MME / image_yorn.py
  https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_yorn.py
  https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/yorn.py

MME Score formula (per task):
  - For each question: correct = 1 if model answer == ground truth else 0
  - accuracy     = mean(correct) * 100          [normal acc]
  - accuracy_plus = mean(all_correct_per_image) * 100  [strict per-image pair]
  - task_score   = accuracy + accuracy_plus
  (In our toy dataset each "image" has exactly one question, so accuracy_plus == accuracy)

Yes/No extraction (YOrN_Extraction — VLMEval-equivalent):
  1. Lowercase the response
  2. Strip punctuation
  3. Split on whitespace
  4. If "yes" in words and "no" not in words → "Yes"
  5. If "no"  in words and "yes" not in words → "No"
  6. Otherwise → "Unknown"

Prompt format (VLMEval-compatible):
  Messages list: [{"type": "image", "value": <PIL.Image>},
                  {"type": "text",  "value": "<question>\nAnswer the question using a single word or phrase."}]
"""

from __future__ import annotations

import re
import string
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLImageProcessor

from dataset import MMESample, make_toy_dataset


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_mini_qwen2vl() -> Qwen2VLForConditionalGeneration:
    """
    Build a tiny Qwen2.5-VL-style model with random weights.

    Architecture mirrors Qwen2-VL-2B-Instruct but with drastically reduced
    dimensions so it fits in CPU memory and runs quickly for pipeline testing.

    Config choices:
      hidden_size=256, num_hidden_layers=2, num_attention_heads=4  → head_dim=64
      mrope_section=[8, 12, 12]  (sums to 32 = head_dim//2, required by M-RoPE)
    """
    from transformers.models.qwen2_vl.configuration_qwen2_vl import (
        Qwen2VLTextConfig,
        Qwen2VLVisionConfig,
    )
    from transformers import Qwen2VLConfig

    head_dim = 256 // 4  # 64
    # mrope_section must sum to head_dim // 2 = 32
    # splits the rotary dim into [temporal, height, width] components
    mrope_section = [8, 12, 12]
    assert sum(mrope_section) == head_dim // 2, "mrope_section must sum to head_dim//2"

    text_cfg = Qwen2VLTextConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        vocab_size=152064,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000.0,
            "mrope_section": mrope_section,
        },
    )
    vision_cfg = Qwen2VLVisionConfig(
        depth=2,
        embed_dim=128,
        hidden_size=256,
        num_heads=4,
        mlp_ratio=2,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
    )
    config = Qwen2VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    model = Qwen2VLForConditionalGeneration(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Yes/No extraction — VLMEval-equivalent implementation
# ---------------------------------------------------------------------------

def _process_punctuation(text: str) -> str:
    """Remove punctuation (mirrors VLMEval's process_punctuation)."""
    return text.translate(str.maketrans("", "", string.punctuation))


def extract_yesno(response: str) -> str:
    """
    VLMEval-equivalent YOrN_Extraction logic.

    Reference:
      vlmeval/dataset/utils/yorn.py :: YOrN_Extraction()

    Returns "Yes", "No", or "Unknown".
    """
    s = response.lower()
    words = _process_punctuation(s).split()
    has_yes = "yes" in words
    has_no = "no" in words
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return "Unknown"


# ---------------------------------------------------------------------------
# Prompt building — VLMEval-compatible
# ---------------------------------------------------------------------------

_PROMPT_SUFFIX = "Answer the question using a single word or phrase."

# Special token IDs for Qwen2-VL (same across all size variants)
_VISION_START_ID = 151652
_VISION_END_ID = 151653
_IMAGE_PAD_ID = 151655
_EOS_ID = 151645

# Number of image tokens produced by the vision encoder for a 224x224 image
# with patch_size=14, spatial_merge_size=2:
#   patches = (224//14) * (224//14) = 16 * 16 = 256
#   after merge: 256 / (2*2) = 64
_NUM_IMAGE_TOKENS = 64


def build_prompt_text(question: str) -> str:
    """
    Construct the text portion of the VLMEval-style prompt.

    VLMEval appends the instruction suffix to the question for MME.
    The image placeholder is handled separately via token IDs.
    """
    return f"{question}\n{_PROMPT_SUFFIX}"


def build_model_inputs(
    image: Image.Image,
    question: str,
    image_processor: Qwen2VLImageProcessor,
) -> dict:
    """
    Build model inputs (input_ids, attention_mask, pixel_values, image_grid_thw)
    without using the full Qwen2VLProcessor (which requires a pretrained tokenizer).

    Token layout:
      [vision_start] [image_pad * N] [vision_end] [text_tokens...]

    Since we don't have a real tokenizer, text tokens are represented as a
    small dummy sequence (question hash → token IDs in valid range).
    The model uses random weights so actual text tokens don't affect evaluation
    quality — only the pipeline correctness matters.
    """
    # --- Image features ---
    img_out = image_processor(images=[image])
    pixel_values = img_out["pixel_values"]          # (num_patches, C*patch*patch)
    image_grid_thw = img_out["image_grid_thw"]      # (1, 3) = [T, H, W]

    # --- Build token sequence ---
    # Vision segment
    vision_tokens = (
        [_VISION_START_ID]
        + [_IMAGE_PAD_ID] * _NUM_IMAGE_TOKENS
        + [_VISION_END_ID]
    )

    # Dummy text tokens: encode question as reproducible token IDs
    # Use ord() values clamped to valid vocab range as a stand-in for real tokens
    text_str = build_prompt_text(question)
    text_tokens = [
        (ord(c) % 50000) + 100  # map to [100, 50099], well within vocab
        for c in text_str[:32]  # truncate to keep inputs short
    ]

    input_ids = torch.tensor(
        [vision_tokens + text_tokens], dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: Qwen2VLForConditionalGeneration,
    image_processor: Qwen2VLImageProcessor,
    dataset: List[MMESample],
    max_new_tokens: int = 5,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Run MME-style evaluation over the dataset.

    For each sample:
      1. Build prompt (VLMEval-compatible format)
      2. Run model.generate()
      3. Extract Yes/No from raw output tokens
      4. Compare with ground-truth answer

    Returns dict mapping task → {"score": float, "total": int,
                                  "correct": int, "unknown": int}

    Score formula mirrors VLMEval MME_rating:
      accuracy      = correct / total * 100
      accuracy_plus = (strict per-image-pair correct) / total * 100
      task_score    = accuracy + accuracy_plus
    (With one question per image, accuracy_plus == accuracy here.)
    """
    device = next(model.parameters()).device

    # Per-task accumulators
    # task → list of (predicted: str, ground_truth: str)
    results: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for idx, sample in enumerate(dataset):
        inputs = build_model_inputs(sample.image, sample.question, image_processor)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=_EOS_ID,
            )

        # Decode: take only the newly generated tokens
        new_token_ids = output_ids[0, inputs["input_ids"].shape[1]:].tolist()

        # Convert token IDs back to a pseudo-string for yes/no extraction.
        # With random weights the output is arbitrary integers, so we map
        # each token ID to its string representation for extraction testing.
        # We also try common token-ID → word mappings for Qwen2 vocabulary.
        raw_text = _decode_tokens_heuristic(new_token_ids)

        predicted = extract_yesno(raw_text)
        ground_truth = sample.answer.capitalize()  # "yes"→"Yes", "no"→"No"

        results[sample.task].append((predicted, ground_truth))

        if verbose:
            print(
                f"[{idx+1:02d}/{len(dataset)}] task={sample.task:10s} "
                f"gt={ground_truth:3s}  pred={predicted:7s}  "
                f"raw_tokens={new_token_ids}"
            )

    # Compute scores
    scores: Dict[str, dict] = {}
    for task, preds in results.items():
        correct = sum(1 for p, g in preds if p == g)
        unknown = sum(1 for p, _ in preds if p == "Unknown")
        total = len(preds)
        accuracy = correct / total * 100 if total > 0 else 0.0

        # MME task_score = accuracy + accuracy_plus
        # (accuracy_plus == accuracy when each image has exactly 1 question)
        task_score = accuracy + accuracy  # = 2 * accuracy (matches VLMEval formula)

        scores[task] = {
            "score": task_score,
            "accuracy": accuracy,
            "correct": correct,
            "unknown": unknown,
            "total": total,
        }

    return scores


# ---------------------------------------------------------------------------
# Token decoding heuristic
# ---------------------------------------------------------------------------

# Qwen2 vocabulary approximate mappings for common tokens
# (from inspection of the tokenizer config — not exhaustive)
_TOKEN_WORD_MAP: Dict[int, str] = {
    9978:   "Yes",
    902:    "No",
    13:     "\n",
    151645: "<|endoftext|>",
    151643: "<|im_start|>",
    151644: "<|im_end|>",
}

# Known "Yes" / "No" token IDs in Qwen2 tokenizer (approximate)
_YES_TOKEN_IDS = {9978, 14335}   # "Yes", "yes"
_NO_TOKEN_IDS  = {902, 2152}     # "No", "no"


def _decode_tokens_heuristic(token_ids: List[int]) -> str:
    """
    Heuristically decode generated token IDs to a string for yes/no extraction.

    Priority:
      1. If any token is in the known Yes/No ID sets, return that word directly.
      2. Otherwise, look up _TOKEN_WORD_MAP for known tokens.
      3. Fall back to joining the token IDs as strings (always gives "Unknown").
    """
    for tid in token_ids:
        if tid in _YES_TOKEN_IDS:
            return "yes"
        if tid in _NO_TOKEN_IDS:
            return "no"

    words = [_TOKEN_WORD_MAP.get(tid, str(tid)) for tid in token_ids]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Score display
# ---------------------------------------------------------------------------

def print_results(scores: Dict[str, dict]) -> None:
    """Print a formatted score table matching VLMEval output style."""
    print("\n" + "=" * 60)
    print("MME-style Evaluation Results (random-weight Qwen2.5-VL)")
    print("=" * 60)
    print(f"{'Task':<14} {'Score':>8} {'Accuracy':>10} {'Correct':>8} {'Unknown':>8} {'Total':>6}")
    print("-" * 60)

    total_score = 0.0
    for task, stats in sorted(scores.items()):
        print(
            f"{task:<14} {stats['score']:>8.1f} {stats['accuracy']:>9.1f}%"
            f" {stats['correct']:>8d} {stats['unknown']:>8d} {stats['total']:>6d}"
        )
        total_score += stats["score"]

    print("-" * 60)
    print(f"{'Perception total':<14} {total_score:>8.1f}")
    print("=" * 60)
    print()
    print("Note: Score = accuracy + accuracy_plus (VLMEval MME_rating formula).")
    print("      With random weights, expected accuracy ≈ 50% → score ≈ 100/task.")
    print("      'Unknown' predictions score 0 (neither Yes nor No extracted).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building toy MME dataset...")
    dataset = make_toy_dataset(seed=42)
    print(f"  {len(dataset)} samples across {len(set(s.task for s in dataset))} tasks")

    print("\nBuilding mini Qwen2.5-VL model (random weights)...")
    t0 = time.time()
    model = build_mini_qwen2vl()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters  ({time.time()-t0:.1f}s)")

    print("\nBuilding image processor...")
    image_processor = Qwen2VLImageProcessor()
    print("  Qwen2VLImageProcessor ready")

    print("\nRunning evaluation...")
    print("-" * 60)
    t0 = time.time()
    scores = evaluate(model, image_processor, dataset, verbose=True)
    elapsed = time.time() - t0
    print(f"\nEvaluation complete in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s/sample)")

    print_results(scores)


if __name__ == "__main__":
    main()
