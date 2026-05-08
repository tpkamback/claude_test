"""
fx_quantize.py — FX-based quantization of Qwen2.5 (or local toy model) via torchao.

Pipeline:
  1. Load model (use_cache=False wrapper to avoid DynamicCache issue)
  2. torch.export.export → FX graph
  3. torchao quantize_ (INT8 weight-only, eager mode)
  4. Verify inference

Usage:
    python fx_quantize.py --model ../eval_ppl/models/qwen-tiny
    python fx_quantize.py --model Qwen/Qwen2.5-0.5B   # requires HF access
    python fx_quantize.py --quant int4
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../eval_ppl/models/qwen-tiny",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--quant", type=str, default="int8",
                        choices=["int8", "int4"],
                        help="Quantization type (default: int8)")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Dummy sequence length for export/inference test")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────────
# Step 1: Load model
# ─────────────────────────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    """
    Wraps a CausalLM to disable KV cache and return only logits.

    Required because torch.export.export cannot handle DynamicCache
    (an unregistered pytree type returned by transformers models by default).
    Passing use_cache=False avoids the cache output entirely.

    Error solved:
        RuntimeError: Found <class 'transformers.cache_utils.DynamicCache'>
        in output, which is not a known type.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False).logits


def load_model(model_id: str):
    print(f"[1/4] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      Parameters: {params:.2f}M")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────────
# Step 2: Export FX graph via torch.export
# ─────────────────────────────────────────────────────────────────────────────────

def export_model(model: torch.nn.Module, seq_len: int) -> torch.export.ExportedProgram:
    """
    Export the model to an FX graph using torch.export.export (PT2 / dynamo).

    Why not torch.fx.symbolic_trace (even with concrete_args)?
        symbolic_trace uses Python bytecode patching and fails when the model's
        forward() has too many parameters:
            ValueError: code: co_varnames is too small
        Using NoCacheWrapper reduces forward() args to just input_ids, but
        concrete_args={"input_ids": example} still fails in transformers 5.x:
            TraceError: symbolically traced variables cannot be used as inputs to control flow
        The root cause is that concrete_args only concretizes the named argument
        itself; tensors derived inside the model (position_ids, batch_size, etc.)
        remain Symbolic Proxies. transformers 5.x masking_utils.py calls
        `if batch_size != position_ids.shape[0]:`, triggering Proxy.__bool__().
        torch.export uses torch.compile / dynamo tracing, which handles control
        flow symbolically and is robust to this pattern.

    Why not torch.ao.quantization.quantize_fx.prepare_fx?
        torch.ao.quantization is deprecated in PyTorch >= 2.10:
            DeprecationWarning: torch.ao.quantization is deprecated ...
            please migrate to torchao pt2e quantization API instead
    """
    print("[2/4] Exporting FX graph via torch.export.export ...")
    wrapped = NoCacheWrapper(model)
    example = (torch.zeros(1, seq_len, dtype=torch.long),)
    exported = torch.export.export(wrapped, example)
    nodes = len(list(exported.graph.nodes))
    print(f"      Graph nodes: {nodes}")
    return exported


# ─────────────────────────────────────────────────────────────────────────────────
# Step 3: Quantize with torchao
# ─────────────────────────────────────────────────────────────────────────────────

def quantize_model(model: torch.nn.Module, quant_type: str) -> torch.nn.Module:
    """
    Apply post-training weight-only quantization using torchao.

    torchao.quantize_() modifies Linear layers in-place, replacing
    the float32 weight tensor with an AffineQuantizedTensor.

    API note: Int8WeightOnlyConfig(version=2) avoids the deprecation warning
    about v1 PlainLayout/AffineQuantizedTensor:
        UserWarning: Config Deprecation: version 1 of Int8WeightOnlyConfig
        is deprecated ... please use version 2

    INT8 weight-only: weights stored as int8, dequantized to fp32 at runtime.
    INT4 weight-only: weights stored as int4 (2x compression vs int8).
    """
    print(f"[3/4] Applying {quant_type.upper()} weight-only quantization ...")
    if quant_type == "int8":
        config = Int8WeightOnlyConfig(version=2)
    else:
        config = Int4WeightOnlyConfig()

    quantize_(model, config)
    print("      quantize_() complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────────
# Step 4: Verify inference
# ─────────────────────────────────────────────────────────────────────────────────

def verify_inference(model: torch.nn.Module, seq_len: int):
    print("[4/4] Running inference on quantized model ...")
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy, use_cache=False)
    elapsed = time.perf_counter() - t0
    print(f"      Logits shape: {out.logits.shape}")
    print(f"      Inference time: {elapsed*1000:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)

    # FX export (for graph inspection / PT2E pipeline)
    try:
        exported = export_model(model, args.seq_len)
        print(f"      Export OK — {len(list(exported.graph.nodes))} nodes")
    except Exception as e:
        print(f"      Export failed: {e}")

    # Quantize (eager mode, in-place)
    quantize_model(model, args.quant)

    # Verify
    verify_inference(model, args.seq_len)
    print("\nDone.")


if __name__ == "__main__":
    main()
