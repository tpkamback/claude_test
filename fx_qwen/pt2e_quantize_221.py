"""
pt2e_quantize_221.py — Full PT2E quantization pipeline on torch 2.2.1

torch 2.2.1 での正しいパイプライン:
  capture_pre_autograd_graph  (← torch.export.export ではなくこちらを使う)
  → prepare_pt2e + XNNPACKQuantizer (is_dynamic=True)
  → convert_pt2e
  → 推論確認

torch 2.2.1 で発生したエラーと対処は docs/fx_quantization_errors.md を参照。

Usage:
    python pt2e_quantize_221.py
    python pt2e_quantize_221.py --model ../eval_ppl/models/qwen-tiny
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

# ── torch 2.2.1 固有のパッチ ──────────────────────────────────────────────────
# torch.compiler.is_compiling() が 2.2.1 に未実装。
# transformers 4.44 が内部で呼ぶため export 時に InternalTorchDynamoError が発生。
# False を返す stub で補完する。
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# ── PT2E API (torch 2.2.1) ────────────────────────────────────────────────────
# torch 2.2.1 での推奨キャプチャ方法。
# torch.export.export では aten.linear が aten.addmm に分解されてしまい
# XNNPACKQuantizer の "linear" パターンがマッチしない。
# capture_pre_autograd_graph は aten.linear.default を保持したまま capture する。
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../eval_ppl/models/qwen-tiny")
    parser.add_argument("--seq_len", type=int, default=16)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load
# ─────────────────────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    """
    KV キャッシュを無効化し logits のみ返すラッパー。
    DynamicCache が pytree 未登録のため export 時にエラーになる問題を回避。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False).logits


def load(model_id: str, seq_len: int):
    print(f"[1/4] Loading: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params")
    wrapped = NoCacheWrapper(model)
    example = (torch.zeros(1, seq_len, dtype=torch.long),)
    return wrapped, example


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Capture (capture_pre_autograd_graph)
# ─────────────────────────────────────────────────────────────────────────────

def capture(model, example):
    """
    torch 2.2.x の PT2E 推奨キャプチャ方法。

    torch.export.export との違い:
      torch.export.export        → aten.linear.default が aten.addmm に分解される
                                   → XNNPACKQuantizer の linear パターンがマッチしない
      capture_pre_autograd_graph → aten.linear.default を保持したまま capture
                                   → XNNPACKQuantizer が正しく動作する

    torch 2.3 以降では torch.export.export + decomp テーブル指定が推奨になる。
    """
    print("[2/4] capture_pre_autograd_graph ...")
    gm = capture_pre_autograd_graph(model, example)
    ops = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    linear_ops = [o for o in sorted(ops) if any(k in o for k in ["mm", "linear", "addmm"])]
    print(f"      linear-related ops: {linear_ops}")
    print(f"      total nodes: {len(list(gm.graph.nodes))}")
    return gm


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: prepare_pt2e (Dynamic Quantization)
# ─────────────────────────────────────────────────────────────────────────────

def prepare(gm):
    """
    is_dynamic=True を使う理由:
      set_global(static config) では activation observer が全ノードに挿入される。
      そのうち embedding 入力など Long 型テンソルを持つノードにも挿入されてしまい、
      キャリブレーション時に HistogramObserver が histc を Long tensor に対して
      呼び出してエラーになる:
        RuntimeError: torch.histogram: input tensor and hist tensor
                      should have the same dtype, got long int and float

      is_dynamic=True (Dynamic Quantization) は weight のみ量子化し
      activation observer を挿入しないため、この問題を回避できる。
    """
    print("[3/4] prepare_pt2e (dynamic quantization) ...")
    cfg = get_symmetric_quantization_config(is_dynamic=True)
    quantizer = XNNPACKQuantizer().set_global(cfg)
    prepared = prepare_pt2e(gm, quantizer)
    obs = sum(1 for n in prepared.graph.nodes if "activation_post_process" in n.name)
    print(f"      observers inserted: {obs}")
    return prepared


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: convert_pt2e + 推論確認
# ─────────────────────────────────────────────────────────────────────────────

def convert_and_verify(prepared, seq_len: int):
    """
    Dynamic Quantization では calibration 不要。
    prepare 直後に convert_pt2e を呼べる。
    """
    print("[4/4] convert_pt2e + inference ...")
    quantized = convert_pt2e(prepared)
    q_nodes = sum(
        1 for n in quantized.graph.nodes
        if "quantize" in n.name or "dequantize" in n.name
    )
    print(f"      quantize/dequantize nodes: {q_nodes}")

    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = quantized(dummy)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"      Logits: {out.shape}  |  Inference: {elapsed:.1f} ms")
    return quantized


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    wrapped, example = load(args.model, args.seq_len)

    try:
        gm = capture(wrapped, example)
    except Exception as e:
        print(f"[ERROR] capture: {type(e).__name__}: {e}"); return

    try:
        prepared = prepare(gm)
    except Exception as e:
        print(f"[ERROR] prepare_pt2e: {type(e).__name__}: {e}"); return

    try:
        convert_and_verify(prepared, args.seq_len)
    except Exception as e:
        print(f"[ERROR] convert_pt2e: {type(e).__name__}: {e}"); return

    print("\nAll steps succeeded on torch", torch.__version__)


if __name__ == "__main__":
    main()
