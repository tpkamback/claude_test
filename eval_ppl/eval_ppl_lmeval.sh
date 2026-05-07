#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_ppl_lmeval.sh
# Run PPL evaluation via lm-evaluation-harness (lm-eval) as a reference.
#
# Usage:
#   bash eval_ppl_lmeval.sh <model_id> [num_gpus]
#
# Examples:
#   bash eval_ppl_lmeval.sh Qwen/Qwen2.5-0.5B 1
#   bash eval_ppl_lmeval.sh Qwen/Qwen2.5-3B   2
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
NUM_GPUS="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: .venv not found. Run setup first:"
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "=== lm-eval reference run ==="
echo "Model   : $MODEL"
echo "GPUs    : $NUM_GPUS"
echo ""

# lm-eval task name for WikiText-2 perplexity is "wikitext"
"$VENV/bin/lm_eval" \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=auto" \
    --tasks wikitext \
    --device "cuda" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "./results/lmeval_${MODEL//\//_}" \
    2>&1 | tee "./results/lmeval_${MODEL//\//_}.log"
