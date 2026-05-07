"""
test_pipeline.py — Self-contained test suite for the MME evaluation pipeline.

Tests cover:
1. Dataset generation (shape, balance, task variety)
2. Yes/No extraction from text and logits
3. MME scoring (ACC, ACC+, score formula)
4. Model instantiation (random weights, forward pass)
5. Processor (image encoding, token layout, mm_token_type_ids)
6. End-to-end evaluation loop (runs to completion, produces valid scores)
7. VLMEval format serialisation

Run with:
    python -m pytest vlm_eval/test_pipeline.py -v
or:
    python vlm_eval/test_pipeline.py
"""

from __future__ import annotations

import sys
import traceback
from typing import List

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def _ok(name: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  PASS  {name}")


def _fail(name: str, reason: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  {name}: {reason}")


def _assert(cond: bool, name: str, msg: str = "") -> None:
    if cond:
        _ok(name)
    else:
        _fail(name, msg or "assertion failed")


# ---------------------------------------------------------------------------
# 1. Dataset tests
# ---------------------------------------------------------------------------

def test_dataset() -> None:
    print("\n[1] Dataset")
    from vlm_eval.dataset import make_toy_dataset, MMESample

    ds = make_toy_dataset(seed=42)

    _assert(len(ds) == 40, "total_samples=40", f"got {len(ds)}")

    tasks = {s.task for s in ds}
    _assert(
        tasks == {"existence", "color", "count", "position"},
        "all_four_tasks_present",
        f"got {tasks}",
    )

    # Each task: 5 yes + 5 no
    from collections import Counter
    for task in ("existence", "color", "count", "position"):
        task_samples = [s for s in ds if s.task == task]
        counts = Counter(s.answer for s in task_samples)
        _assert(
            counts["yes"] == 5 and counts["no"] == 5,
            f"balanced_{task}",
            f"got {dict(counts)}",
        )

    # Images are PIL RGB 224x224
    for s in ds[:2]:
        _assert(
            s.image.size == (224, 224) and s.image.mode == "RGB",
            f"image_shape_{s.task}",
            f"got size={s.image.size} mode={s.image.mode}",
        )


# ---------------------------------------------------------------------------
# 2. Yes/No extraction tests
# ---------------------------------------------------------------------------

def test_extraction() -> None:
    print("\n[2] Yes/No extraction")
    from vlm_eval.score import extract_yes_no_from_text, extract_yes_no_from_logits

    cases_text = [
        ("yes", "yes"),
        ("Yes.", "yes"),
        ("YES, I see it.", "yes"),
        ("no", "no"),
        ("No.", "no"),
        ("No, I don't think so.", "no"),
        ("I think yes", "yes"),
        ("The answer is no.", "no"),
        ("maybe", "unknown"),
        ("", "unknown"),
    ]
    for text, expected in cases_text:
        got = extract_yes_no_from_text(text)
        _assert(got == expected, f"text_extract_{repr(text)}", f"got={got}")

    # Logit-based extraction
    from vlm_eval.model import YES_TOKEN_ID, NO_TOKEN_ID
    vocab_size = 256

    logits_yes_higher = torch.zeros(vocab_size)
    logits_yes_higher[YES_TOKEN_ID] = 1.0
    logits_yes_higher[NO_TOKEN_ID] = -1.0
    _assert(
        extract_yes_no_from_logits(logits_yes_higher, YES_TOKEN_ID, NO_TOKEN_ID) == "yes",
        "logit_extract_yes",
    )

    logits_no_higher = torch.zeros(vocab_size)
    logits_no_higher[YES_TOKEN_ID] = -1.0
    logits_no_higher[NO_TOKEN_ID] = 1.0
    _assert(
        extract_yes_no_from_logits(logits_no_higher, YES_TOKEN_ID, NO_TOKEN_ID) == "no",
        "logit_extract_no",
    )

    # Tie → "yes" (yes >= no)
    logits_tie = torch.zeros(vocab_size)
    _assert(
        extract_yes_no_from_logits(logits_tie, YES_TOKEN_ID, NO_TOKEN_ID) == "yes",
        "logit_extract_tie_yields_yes",
    )


# ---------------------------------------------------------------------------
# 3. Scoring tests
# ---------------------------------------------------------------------------

def test_scoring() -> None:
    print("\n[3] MME Scoring")
    from vlm_eval.score import (
        PredictionRecord,
        compute_task_score,
        compute_mme_score,
    )

    def make_record(task, gt, pred):
        return PredictionRecord(task=task, question="Q?", ground_truth=gt, prediction=pred)

    # Perfect answers: ACC=1.0, ACC+=1.0, score=200
    perfect = [
        make_record("existence", "yes", "yes"),
        make_record("existence", "no", "no"),
        make_record("existence", "yes", "yes"),
        make_record("existence", "no", "no"),
    ]
    tr = compute_task_score(perfect)
    _assert(tr.acc == 1.0, "perfect_acc_1.0", f"got {tr.acc}")
    _assert(tr.acc_plus == 1.0, "perfect_acc_plus_1.0", f"got {tr.acc_plus}")
    _assert(abs(tr.score - 200.0) < 1e-6, "perfect_score_200", f"got {tr.score}")

    # All wrong: ACC=0.0, ACC+=0.0, score=0
    wrong = [
        make_record("color", "yes", "no"),
        make_record("color", "no", "yes"),
        make_record("color", "yes", "no"),
        make_record("color", "no", "yes"),
    ]
    tr2 = compute_task_score(wrong)
    _assert(tr2.acc == 0.0, "wrong_acc_0.0", f"got {tr2.acc}")
    _assert(tr2.acc_plus == 0.0, "wrong_acc_plus_0.0", f"got {tr2.acc_plus}")
    _assert(abs(tr2.score - 0.0) < 1e-6, "wrong_score_0", f"got {tr2.score}")

    # Half correct: first pair both correct, second pair both wrong
    # ACC = 2/4 = 0.5, ACC+ = 1/2 = 0.5, score = 100
    mixed = [
        make_record("count", "yes", "yes"),  # pair1 yes ✓
        make_record("count", "no", "no"),    # pair1 no  ✓
        make_record("count", "yes", "no"),   # pair2 yes ✗
        make_record("count", "no", "yes"),   # pair2 no  ✗
    ]
    tr3 = compute_task_score(mixed)
    _assert(abs(tr3.acc - 0.5) < 1e-6, "mixed_acc_0.5", f"got {tr3.acc}")
    _assert(abs(tr3.acc_plus - 0.5) < 1e-6, "mixed_acc_plus_0.5", f"got {tr3.acc_plus}")
    _assert(abs(tr3.score - 100.0) < 1e-6, "mixed_score_100", f"got {tr3.score}")

    # Multi-task total
    all_records = perfect + wrong + mixed
    mme = compute_mme_score(all_records)
    expected_total = 200.0 + 0.0 + 100.0
    _assert(
        abs(mme.total_score - expected_total) < 1e-6,
        "multi_task_total",
        f"expected {expected_total}, got {mme.total_score}",
    )
    _assert(len(mme.task_results) == 3, "three_tasks_in_result",
            f"got {len(mme.task_results)}")


# ---------------------------------------------------------------------------
# 4. Model instantiation tests
# ---------------------------------------------------------------------------

def test_model() -> None:
    print("\n[4] Model instantiation")
    from vlm_eval.model import build_random_model, VOCAB_SIZE

    model = build_random_model(seed=0)
    _assert(model is not None, "model_not_none")

    n_params = sum(p.numel() for p in model.parameters())
    _assert(n_params > 0, "model_has_params", f"got {n_params}")
    print(f"    Parameters: {n_params:,}")

    # Text-only forward pass
    input_ids = torch.tensor([[2, 4, 5, 3]])
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids)
            _assert(
                out.logits.shape == (1, 4, VOCAB_SIZE),
                "text_only_logits_shape",
                f"got {out.logits.shape}",
            )
        except Exception as e:
            _fail("text_only_forward", str(e))

    # Determinism: same seed → same logits
    model2 = build_random_model(seed=0)
    with torch.no_grad():
        out2 = model2(input_ids=input_ids)
    _assert(
        torch.allclose(out.logits, out2.logits),
        "same_seed_deterministic",
    )


# ---------------------------------------------------------------------------
# 5. Processor tests
# ---------------------------------------------------------------------------

def test_processor() -> None:
    print("\n[5] Processor")
    from vlm_eval.model import (
        DummyProcessor,
        IMAGE_TOKEN_ID,
        VISION_START_ID,
        VISION_END_ID,
    )

    proc = DummyProcessor()
    img = Image.new("RGB", (224, 224), (100, 150, 200))
    text = "Is there a red circle? Please answer yes or no."

    inputs = proc(image=img, text=text)

    required_keys = {
        "input_ids", "pixel_values", "image_grid_thw",
        "mm_token_type_ids", "attention_mask"
    }
    _assert(
        required_keys.issubset(inputs.keys()),
        "all_required_keys_present",
        f"missing {required_keys - inputs.keys()}",
    )

    # Batch dim = 1
    for key in ("input_ids", "mm_token_type_ids", "attention_mask"):
        _assert(
            inputs[key].shape[0] == 1,
            f"{key}_batch_dim_1",
            f"got {inputs[key].shape}",
        )

    # Check VISION_START at position 0 of input_ids
    _assert(
        inputs["input_ids"][0, 0].item() == VISION_START_ID,
        "vision_start_at_pos0",
        f"got {inputs['input_ids'][0, 0].item()}",
    )

    # All image token positions have mm_token_type_ids == 1
    ids_flat = inputs["input_ids"][0].tolist()
    types_flat = inputs["mm_token_type_ids"][0].tolist()
    for pos, (tok_id, t_id) in enumerate(zip(ids_flat, types_flat)):
        if tok_id == IMAGE_TOKEN_ID:
            _assert(
                t_id == 1,
                f"img_token_type_is_1_at_pos{pos}",
                f"got {t_id}",
            )
            break  # just check first occurrence

    # pixel_values shape: [n_patches, patch_dim]
    pv = inputs["pixel_values"]
    _assert(len(pv.shape) == 2, "pixel_values_2d", f"got shape {pv.shape}")

    # image_grid_thw shape: [1, 3]
    thw = inputs["image_grid_thw"]
    _assert(thw.shape == (1, 3), "image_grid_thw_shape", f"got {thw.shape}")


# ---------------------------------------------------------------------------
# 6. End-to-end evaluation
# ---------------------------------------------------------------------------

def test_end_to_end() -> None:
    print("\n[6] End-to-end evaluation")
    from vlm_eval.evaluate import run_evaluation
    from vlm_eval.score import MMEResult

    try:
        result, records = run_evaluation(seed=42, verbose=False)
    except Exception as e:
        traceback.print_exc()
        _fail("run_evaluation_no_crash", str(e))
        return

    _assert(isinstance(result, MMEResult), "result_is_MMEResult")
    _assert(len(records) == 40, "forty_records", f"got {len(records)}")

    # All predictions are yes or no
    invalid = [r for r in records if r.prediction not in ("yes", "no")]
    _assert(len(invalid) == 0, "all_predictions_yes_or_no",
            f"{len(invalid)} invalid: {invalid[:3]}")

    # Score is in [0, 800] (4 tasks × 200 max each)
    _assert(
        0.0 <= result.total_score <= 800.0,
        "total_score_in_range",
        f"got {result.total_score}",
    )

    # All four tasks present in results
    _assert(
        {"existence", "color", "count", "position"}.issubset(result.task_results),
        "all_tasks_in_result",
        f"got {set(result.task_results.keys())}",
    )

    # Per-task scores are in [0, 200]
    for task, tr in result.task_results.items():
        _assert(
            0.0 <= tr.score <= 200.0,
            f"task_{task}_score_in_range",
            f"got {tr.score}",
        )

    print(f"\n  {result}")


# ---------------------------------------------------------------------------
# 7. VLMEval format
# ---------------------------------------------------------------------------

def test_vlmeval_format() -> None:
    print("\n[7] VLMEval format")
    from vlm_eval.score import PredictionRecord
    from vlm_eval.evaluate import vlmeval_format_predictions

    records = [
        PredictionRecord(task="existence", question="Q?", ground_truth="yes", prediction="yes"),
        PredictionRecord(task="existence", question="Q?", ground_truth="no",  prediction="no"),
        PredictionRecord(task="color",     question="Q?", ground_truth="yes", prediction="no"),
    ]
    tsv = vlmeval_format_predictions(records)
    lines = tsv.splitlines()

    _assert(lines[0] == "question_id\ttask\tprediction\tground_truth\tcorrect",
            "tsv_header_correct", f"got {lines[0]!r}")
    _assert(len(lines) == 4, "tsv_row_count", f"got {len(lines)}")

    # Row 0: correct=1
    parts0 = lines[1].split("\t")
    _assert(parts0[-1] == "1", "correct_flag_1_for_match", f"got {parts0[-1]}")

    # Row 2: correct=0
    parts2 = lines[3].split("\t")
    _assert(parts2[-1] == "0", "correct_flag_0_for_mismatch", f"got {parts2[-1]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 50)
    print("vlm_eval test suite")
    print("=" * 50)

    test_dataset()
    test_extraction()
    test_scoring()
    test_model()
    test_processor()
    test_end_to_end()
    test_vlmeval_format()

    print()
    print("=" * 50)
    print(f"Results: {_PASS} passed, {_FAIL} failed")
    print("=" * 50)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
