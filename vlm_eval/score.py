"""
score.py — MME scoring utilities.

MME (MultiModal Evaluation) benchmark reference
------------------------------------------------
Paper: "MME: A Comprehensive Evaluation Benchmark for Multimodal Large
Language Models" (Fu et al., 2023).  Code: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation

Scoring rules (identical to the official MME evaluation script):
- Each task has N question pairs.  A *pair* is two questions about the same
  image: one with ground-truth "yes" and one with ground-truth "no".
- **Accuracy** (ACC): fraction of individual questions answered correctly.
- **Accuracy+** (ACC+): fraction of *pairs* where *both* questions are
  answered correctly (stricter metric).
- **Task score** = ACC × 100 + ACC+ × 100 (max = 200 per task).
- **Total MME score** = sum of task scores across all tasks.

VLMEval alignment
-----------------
VLMEval (https://github.com/open-compass/VLMEvalKit) computes the same metrics
inside ``vlmeval/dataset/mme.py``.  The key function there is
``MME.evaluate()``, which collects per-question predictions, extracts
"yes"/"no" from raw model output, and then applies the same pair-based scoring
we implement here.

Yes/No extraction
-----------------
VLMEval uses a simple rule: strip whitespace/punctuation and check whether the
output starts with "yes" or "no" (case-insensitive).  We replicate this in
``extract_yes_no_from_logits`` and ``extract_yes_no_from_text``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    """A single model prediction for one MME question."""

    task: str
    question: str
    ground_truth: str   # "yes" or "no"
    prediction: str     # "yes", "no", or "unknown"


@dataclass
class TaskResult:
    """Per-task MME metrics."""

    task: str
    num_questions: int
    num_pairs: int
    acc: float          # individual accuracy  [0, 1]
    acc_plus: float     # pair accuracy        [0, 1]
    score: float        # task MME score  = (acc + acc_plus) * 100


@dataclass
class MMEResult:
    """Aggregate MME evaluation result."""

    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    total_score: float = 0.0

    def __str__(self) -> str:
        lines = ["MME Evaluation Results", "=" * 40]
        for task, tr in sorted(self.task_results.items()):
            lines.append(
                f"  {task:<12}  ACC={tr.acc:.3f}  ACC+={tr.acc_plus:.3f}"
                f"  score={tr.score:.1f}"
            )
        lines.append("-" * 40)
        lines.append(f"  Total MME score: {self.total_score:.1f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Yes/No extraction helpers
# ---------------------------------------------------------------------------

def extract_yes_no_from_text(text: str) -> str:
    """Extract 'yes' or 'no' from a model's free-form text response.

    Replicates VLMEval's ``extract_answer`` heuristic:
    1. Lowercase and strip punctuation/whitespace.
    2. If the cleaned string starts with "yes" → "yes".
    3. If it starts with "no" → "no".
    4. Otherwise search for the first occurrence of yes/no in the string.
    5. If neither found → "unknown".

    Args:
        text: Raw model output string.

    Returns:
        One of "yes", "no", or "unknown".
    """
    cleaned = text.lower().strip(" \t\n.,!?\"'")
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    # Fallback: find first occurrence
    match = re.search(r"\b(yes|no)\b", cleaned)
    if match:
        return match.group(1)
    return "unknown"


def extract_yes_no_from_logits(
    logits_last_token,  # torch.Tensor shape [vocab_size]
    yes_token_id: int,
    no_token_id: int,
) -> str:
    """Extract yes/no prediction from last-token logits.

    Compares the raw logit (pre-softmax) at YES and NO token positions.
    The token with the higher logit is chosen.  This matches the behaviour of
    greedy decoding used in VLMEval's generate() call.

    Args:
        logits_last_token: Float tensor of shape [vocab_size].
        yes_token_id: Vocabulary index for "yes".
        no_token_id:  Vocabulary index for "no".

    Returns:
        "yes" or "no".
    """
    yes_logit = logits_last_token[yes_token_id].item()
    no_logit = logits_last_token[no_token_id].item()
    return "yes" if yes_logit >= no_logit else "no"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _pair_predictions(
    records: List[PredictionRecord],
) -> List[Tuple[PredictionRecord, PredictionRecord]]:
    """Group prediction records into (yes_question, no_question) pairs.

    MME pairs are formed by taking consecutive questions: record[2i] (answer=yes)
    and record[2i+1] (answer=no) belong to the same pair.  This matches the
    ordering produced by ``dataset.make_toy_dataset()``.

    If the number of records is odd the last record is silently discarded.
    """
    pairs = []
    for i in range(0, len(records) - 1, 2):
        a, b = records[i], records[i + 1]
        # Ensure one is yes and one is no
        if a.ground_truth == "yes" and b.ground_truth == "no":
            pairs.append((a, b))
        elif a.ground_truth == "no" and b.ground_truth == "yes":
            pairs.append((b, a))
        # If both same polarity (shouldn't happen in balanced dataset), skip
    return pairs


def compute_task_score(records: List[PredictionRecord]) -> TaskResult:
    """Compute MME ACC, ACC+, and task score for one task.

    Args:
        records: All prediction records for a single task.

    Returns:
        TaskResult with computed metrics.
    """
    assert records, "records list must not be empty"
    task = records[0].task

    # Individual accuracy
    correct_individual = sum(
        1 for r in records if r.prediction == r.ground_truth
    )
    acc = correct_individual / len(records)

    # Pair accuracy (both questions in a pair must be correct)
    pairs = _pair_predictions(records)
    if pairs:
        correct_pairs = sum(
            1
            for yes_rec, no_rec in pairs
            if yes_rec.prediction == "yes" and no_rec.prediction == "no"
        )
        acc_plus = correct_pairs / len(pairs)
    else:
        acc_plus = 0.0

    score = (acc + acc_plus) * 100.0

    return TaskResult(
        task=task,
        num_questions=len(records),
        num_pairs=len(pairs),
        acc=acc,
        acc_plus=acc_plus,
        score=score,
    )


def compute_mme_score(records: List[PredictionRecord]) -> MMEResult:
    """Compute full MME score from a list of prediction records.

    Groups records by task, computes per-task metrics, and sums for the total.

    Args:
        records: All prediction records across all tasks.

    Returns:
        MMEResult containing per-task TaskResult objects and total score.
    """
    # Group by task
    by_task: Dict[str, List[PredictionRecord]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(r)

    result = MMEResult()
    for task, task_records in by_task.items():
        tr = compute_task_score(task_records)
        result.task_results[task] = tr
        result.total_score += tr.score

    return result
