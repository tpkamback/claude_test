"""
pt2e_full_quantize_211.py — Full Static Quantization via PT2E Pipeline

Automates observer insertion using torch.export.export + Custom Quantizer:
  1. NoCacheWrapper + torch.export.export
  2. AllOpsQuantizer annotates all target aten ops
  3. prepare_pt2e -> calibrate -> convert_pt2e

Usage:
    python pt2e_full_quantize_211.py
    python pt2e_full_quantize_211.py --model ../eval_ppl/models/Qwen2.5-0.5B-random --seq_len 16
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.fx
from transformers import AutoModelForCausalLM

import torchao.quantization.pt2e as torchao_pt2e
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)

# Use torchao's MinMaxObserver (recognized as activation_post_process by convert_pt2e)
MinMaxObserver = torchao_pt2e.MinMaxObserver


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: NoCacheWrapper
# ─────────────────────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids, use_cache=False).logits


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Custom Quantizer
# ─────────────────────────────────────────────────────────────────────────────

TARGET_OPS = {
    torch.ops.aten.linear.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten._softmax.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.silu_.default,
    # Note: rms_norm is decomposed in torch.export (no aten.rms_norm.default)
    # scaled_dot_product_attention covers attention matmul+softmax
    torch.ops.aten.scaled_dot_product_attention.default,
}

# Per-tensor symmetric int8 quantization spec
_INT8_QSPEC = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_symmetric,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MinMaxObserver,
)


_FLOAT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _is_float_node(node: torch.fx.Node) -> bool:
    """Return True if the node produces a floating-point tensor."""
    val = node.meta.get("val", None)
    if val is None:
        return True  # conservative: annotate if unknown
    if hasattr(val, "dtype"):
        return val.dtype in _FLOAT_DTYPES
    return False


class AllOpsQuantizer(Quantizer):
    """Custom Quantizer that annotates all TARGET_OPS nodes in the graph."""

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in TARGET_OPS:
                continue
            # Annotate only float tensor input args with quantization spec
            input_qspec_map = {
                arg: _INT8_QSPEC
                for arg in node.args
                if isinstance(arg, torch.fx.Node) and _is_float_node(arg)
            }
            if not input_qspec_map:
                continue
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../eval_ppl/models/Qwen2.5-0.5B-random")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--n_calib", type=int, default=4)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seq_len = args.seq_len

    # ── [1/5] Load ──────────────────────────────────────────────────────────
    print(f"[1/5] Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params")

    # ── [2/5] Export FX graph ────────────────────────────────────────────────
    print("[2/5] Exporting FX graph ...")
    wrapped = NoCacheWrapper(model)
    example = (torch.zeros(1, seq_len, dtype=torch.long),)

    exported = torch.export.export(wrapped, example)
    gm = exported.module()

    total_nodes = len(list(gm.graph.nodes))
    print(f"      Graph nodes: {total_nodes}")

    # Count target ops before prepare
    op_counts_before = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in TARGET_OPS:
            name = str(node.target).split(".")[-2] if "." in str(node.target) else str(node.target)
            # Use a friendly name
            tgt = node.target
            if tgt == torch.ops.aten.linear.default:
                op_counts_before["linear"] = op_counts_before.get("linear", 0) + 1
            elif tgt == torch.ops.aten.mm.default:
                op_counts_before["mm"] = op_counts_before.get("mm", 0) + 1
            elif tgt == torch.ops.aten.bmm.default:
                op_counts_before["bmm"] = op_counts_before.get("bmm", 0) + 1
            elif tgt in (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default):
                op_counts_before["softmax"] = op_counts_before.get("softmax", 0) + 1
            elif tgt in (torch.ops.aten.silu.default, torch.ops.aten.silu_.default):
                op_counts_before["silu"] = op_counts_before.get("silu", 0) + 1
            elif tgt == torch.ops.aten.scaled_dot_product_attention.default:
                op_counts_before["scaled_dot_product_attention"] = op_counts_before.get("scaled_dot_product_attention", 0) + 1
            else:
                op_counts_before[name] = op_counts_before.get(name, 0) + 1

    # ── [3/5] prepare_pt2e ───────────────────────────────────────────────────
    print("[3/5] prepare_pt2e (AllOpsQuantizer) ...")
    quantizer = AllOpsQuantizer()
    prepared = prepare_pt2e(gm, quantizer)

    # Count observer nodes inserted (use torchao's observer base classes)
    observer_count = 0
    for node in prepared.graph.nodes:
        if node.op == "call_module":
            mod = prepared.get_submodule(node.target)
            if isinstance(mod, (torchao_pt2e.ObserverBase,
                                torchao_pt2e.FakeQuantizeBase,
                                torchao_pt2e.AffineQuantizedObserverBase)):
                observer_count += 1

    print(f"      Observer nodes inserted: {observer_count}")
    # rms_norm is not present as a single op; it is decomposed to pow/mean/rsqrt/mul
    # sdpa (scaled_dot_product_attention) covers attention matmul + softmax
    print(f"      Target ops found: "
          f"linear={op_counts_before.get('linear', 0)}, "
          f"mm={op_counts_before.get('mm', 0)}, "
          f"bmm={op_counts_before.get('bmm', 0)}, "
          f"softmax={op_counts_before.get('softmax', 0)}, "
          f"silu={op_counts_before.get('silu', 0)}, "
          f"rms_norm=0, "
          f"sdpa={op_counts_before.get('scaled_dot_product_attention', 0)}")

    # ── [4/5] Calibrate ──────────────────────────────────────────────────────
    print(f"[4/5] Calibrating ({args.n_calib} samples) ...")
    with torch.no_grad():
        for _ in range(args.n_calib):
            prepared(torch.randint(0, 100, (1, seq_len)))
    print("      done")

    # ── [5/5] convert_pt2e ──────────────────────────────────────────────────
    print("[5/5] convert_pt2e ...")
    quantized = convert_pt2e(prepared)

    # Count quantize/dequantize nodes
    qdq_count = 0
    for node in quantized.graph.nodes:
        if node.op == "call_function":
            name = str(node.target)
            if "quantize" in name or "dequantize" in name:
                qdq_count += 1

    print(f"      quantize/dequantize nodes: {qdq_count}")

    # ── Inference check ──────────────────────────────────────────────────────
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = quantized(dummy)
    elapsed = (time.perf_counter() - t0) * 1000

    print()
    print(f"  Logits : {logits.shape}")
    print(f"  Time   : {elapsed:.1f} ms")
    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
