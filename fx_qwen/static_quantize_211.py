"""
static_quantize_211.py — Static Quantization on torch 2.11 via torchao

Pipeline:
  1. observer を Linear 層に埋め込む（LinearActivationWeightObservedTensor）
  2. calibration データを流して activation の min/max を観測
  3. weight を float に戻す（observer tensor を除去）
  4. 観測した act_scale を使って quantize_() (Int8StaticActivationInt8WeightConfig)
  5. 推論確認

torch 2.11 では prepare_pt2e / convert_pt2e が torchao に移管されており
PT2E スタイルの static quant は torchao の observer API で代替する。

Usage:
    python static_quantize_211.py
    python static_quantize_211.py --model ../eval_ppl/models/qwen-tiny --seq_len 16
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM
from torchao.quantization import (
    quantize_,
    Int8StaticActivationInt8WeightConfig,
    AffineQuantizedMinMaxObserver,
    MappingType,
    PerRow,
)
from torchao.quantization.linear_activation_weight_observed_tensor import (
    LinearActivationWeightObservedTensor,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../eval_ppl/models/qwen-tiny")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length (must be fixed for static quant)")
    parser.add_argument("--n_calib", type=int, default=4,
                        help="Number of calibration samples")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: observer を Linear 層に埋め込む
# ─────────────────────────────────────────────────────────────────────────────

def insert_observers(model: torch.nn.Module) -> dict:
    """
    Linear 層の weight を LinearActivationWeightObservedTensor でラップする。
    forward 時に activation の統計（min/max）を observer が自動収集する。

    PerRow granularity:
      activation shape (batch, seq, hidden) に対して
      各 token（行）ごとに scale を持つ → scale shape: (1, seq, 1)
      静的量子化では seq_len が固定である前提。

    torch 2.11 での注意:
      prepare_pt2e は torchao に移管されて未安定なため、
      この observer ラッパー方式で代替する。
    """
    observers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            act_obs = AffineQuantizedMinMaxObserver(
                MappingType.SYMMETRIC, torch.int8,
                granularity=PerRow(), keepdim=True,
            )
            w_obs = AffineQuantizedMinMaxObserver(
                MappingType.SYMMETRIC, torch.int8,
                granularity=PerRow(), keepdim=True,
            )
            mod.weight = torch.nn.Parameter(
                LinearActivationWeightObservedTensor.from_float(mod.weight, act_obs, w_obs)
            )
            observers[name] = act_obs
    return observers


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(model: torch.nn.Module, seq_len: int, n_samples: int):
    """
    ランダム入力を流して observer に activation 統計を収集させる。

    Static Quantization では seq_len が固定である必要がある。
    （act_scale の shape が seq_len に依存するため）
    Dynamic Quantization ではこのステップは不要。
    """
    with torch.no_grad():
        for _ in range(n_samples):
            model(torch.randint(0, 100, (1, seq_len)), use_cache=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: observer を除去して weight を float に戻す
# ─────────────────────────────────────────────────────────────────────────────

def remove_observers(model: torch.nn.Module):
    """
    calibration 後、LinearActivationWeightObservedTensor を通常の float Tensor に戻す。
    quantize_() は float weight を受け取る必要があるため必須。

    除去しないと quantize_() 内で aten.view が
    LinearActivationWeightObservedTensor に対して呼ばれ NotImplementedError になる。
    """
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if isinstance(mod.weight, LinearActivationWeightObservedTensor):
                mod.weight = torch.nn.Parameter(
                    mod.weight.original_weight_tensor.detach().float()
                )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Static Quantization 適用
# ─────────────────────────────────────────────────────────────────────────────

def apply_static_quant(model: torch.nn.Module, observers: dict):
    """
    observer から収集した act_scale を Int8StaticActivationInt8WeightConfig に渡す。

    act_scale の shape は (1, seq_len, 1):
      - PerRow granularity では各 token ごとに 1 つの scale
      - ndim は activation tensor (3D) と一致する必要がある
      - (1, 1, 1) などに reshape すると block_size assertion で失敗する

    weight は quantize_() 内で自動的に観測・量子化される。
    """
    first_name = next(iter(observers))
    act_scale, act_zp = observers[first_name].calculate_qparams()
    print(f"      act_scale shape: {act_scale.shape}, value range: "
          f"[{act_scale.min().item():.4f}, {act_scale.max().item():.4f}]")

    quantize_(model, Int8StaticActivationInt8WeightConfig(
        act_quant_scale=act_scale,
        act_quant_zero_point=act_zp.to(torch.int8),
        granularity=PerRow(),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"[1/4] Loading: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params")

    print("[2/4] Inserting observers ...")
    observers = insert_observers(model)
    print(f"      {len(observers)} Linear layers instrumented")

    print(f"[3/4] Calibrating ({args.n_calib} samples, seq_len={args.seq_len}) ...")
    calibrate(model, args.seq_len, args.n_calib)
    remove_observers(model)
    print("      done")

    print("[4/4] Applying static quantization ...")
    apply_static_quant(model, observers)
    print("      done")

    # 推論確認
    dummy = torch.zeros(1, args.seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy, use_cache=False)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  Logits : {out.logits.shape}")
    print(f"  Time   : {elapsed:.1f} ms")
    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
