#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh
# Evaluate PPL for all target models and summarize results.
#
# Usage:
#   bash run_all.sh [num_gpus]
#   bash run_all.sh 2          # use 2 GPUs (tensor parallel via device_map="auto")
# ---------------------------------------------------------------------------

set -euo pipefail

NUM_GPUS="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

PYTHON="$VENV/bin/python"
MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-3B"
)

SUMMARY="$RESULTS_DIR/summary.tsv"
echo -e "model\tdataset\tppl\ttime_sec\tmethod" > "$SUMMARY"

for MODEL in "${MODELS[@]}"; do
    SAFE="${MODEL//\//_}"

    # --- custom eval_ppl.py ---
    echo ">>> [eval_ppl.py] $MODEL"
    LOG="$RESULTS_DIR/custom_${SAFE}.log"
    "$PYTHON" "$SCRIPT_DIR/eval_ppl.py" \
        --model "$MODEL" \
        --num_gpus "$NUM_GPUS" \
        --stride 512 \
        2>&1 | tee "$LOG"

    PPL=$(grep "PPL" "$LOG" | awk '{print $NF}')
    TIME=$(grep "Time" "$LOG" | awk '{print $NF}' | tr -d 's')
    echo -e "${MODEL}\twikitext-2-test\t${PPL}\t${TIME}\tcustom" >> "$SUMMARY"

    # --- lm-eval reference ---
    echo ">>> [lm-eval] $MODEL"
    bash "$SCRIPT_DIR/eval_ppl_lmeval.sh" "$MODEL" "$NUM_GPUS"
done

echo ""
echo "=== Summary ==="
column -t -s $'\t' "$SUMMARY"
