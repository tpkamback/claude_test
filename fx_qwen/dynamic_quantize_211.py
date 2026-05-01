"""
dynamic_quantize_211.py — Dynamic Activation + Weight Quantization via torchao

Weight-only（fx_quantize.py）との違い:
  Weight-only  → 重みだけ int8、activation は float32 のまま
  Dynamic      → 重みも activation も int8
                 activation は推論時にトークンごとにリアルタイムで量子化
                 → calibration 不要（Static と異なる）

torchao での位置づけ:
  Int8WeightOnlyConfig                  : weight のみ
  Int8DynamicActivationInt8WeightConfig : weight + activation (dynamic) ← これ
  Int8StaticActivationInt8WeightConfig  : weight + activation (static, calibration 必要)

Dynamic vs Static:
  Dynamic: activation の scale を推論時に毎回計算（per-token）
            → 可変長シーケンスに対応
            → Static より精度良い場合が多い（実データ統計を使うため）
  Static : calibration データで事前に scale を決定
            → seq_len 固定が必要
            → 推論時の scale 計算コスト不要

Usage:
    python dynamic_quantize_211.py
    python dynamic_quantize_211.py --model ../eval_ppl/models/qwen-tiny --seq_len 16
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../eval_ppl/models/qwen-tiny")
    parser.add_argument("--seq_len", type=int, default=16)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load
# ─────────────────────────────────────────────────────────────────────────────

def load(model_id: str):
    print(f"[1/3] Loading: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Dynamic Quantization
# ─────────────────────────────────────────────────────────────────────────────

def apply_dynamic_quant(model: torch.nn.Module) -> torch.nn.Module:
    """
    activation + weight を両方 int8 で量子化する。

    Dynamic Quantization の仕組み:
      - 推論時、Linear に入力 activation が来るたびに
        その最小値・最大値から scale を計算（per-token）
      - weight は事前に int8 に変換済み（ここで quantize_() 実行時）
      - calibration データ不要 → static_quantize_211.py より手軽

    Int8DynamicActivationInt8WeightConfig のデフォルト granularity:
      - weight: PerRow（行ごと、= out_features ごとに 1 つの scale）
      - activation: PerToken（トークンごとに 1 つの scale）
    """
    print("[2/3] Applying Int8 Dynamic Activation + Weight quantization ...")
    # version=2: PlainLayout 非推奨警告を回避（Int8WeightOnlyConfig と同じ対処）
    quantize_(model, Int8DynamicActivationInt8WeightConfig(version=2))

    # GPT-2 系アーキテクチャは Conv1D を使うため nn.Linear は lm_head のみ。
    # Qwen2.5 などは全 projection が nn.Linear なので count が多くなる。
    n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"      {n_linear} nn.Linear layers quantized (Conv1D 系は対象外)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Inference
# ─────────────────────────────────────────────────────────────────────────────

def verify(model: torch.nn.Module, seq_len: int):
    """
    Dynamic Quantization は可変長シーケンスに対応しているため
    seq_len を変えて複数回推論できる（Static と異なる）。
    """
    print("[3/3] Running inference ...")
    for slen in [seq_len, seq_len * 2]:
        dummy = torch.zeros(1, slen, dtype=torch.long)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy, use_cache=False)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"      seq_len={slen:3d}  Logits: {out.logits.shape}  Time: {elapsed:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model = load(args.model)
    apply_dynamic_quant(model)
    verify(model, args.seq_len)
    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
