"""
Qwen2-tiny: PT2E with use_cache=True and dynamic seq_len.

Step 1: use_cache=True (KV cache enabled)
Step 2: dynamic seq_len via torch.export Dim
Step 3: both combined
"""

import torch
import traceback
import warnings
warnings.filterwarnings("ignore")

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.quantizer import (
    Quantizer, QuantizationAnnotation, QuantizationSpec,
)
from torchao.quantization.pt2e.observer import MinMaxObserver
from transformers import AutoModelForCausalLM

MODEL_PATH = "/home/user/claude_test/eval_ppl/models/qwen2-tiny"
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class AllOpsQuantizer(Quantizer):
    TARGET_OPS = {
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten._softmax.default,
        torch.ops.aten.layer_norm.default,
    }

    def _is_float(self, node):
        val = node.meta.get("val")
        return val is not None and getattr(val, "dtype", None) in _FLOAT_DTYPES

    def annotate(self, model):
        act_spec = QuantizationSpec(
            dtype=torch.int8, quant_min=-128, quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            observer_or_fake_quant_ctr=MinMaxObserver,
        )
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in self.TARGET_OPS:
                continue
            if not self._is_float(node):
                continue
            input_map = {
                arg: act_spec for arg in node.args
                if isinstance(arg, torch.fx.Node) and self._is_float(arg)
            }
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_map, output_qspec=act_spec,
            )
        return model

    def validate(self, model): pass


def run_pipeline(gm, example_inputs, label):
    print(f"\n  [prepare_pt2e]", flush=True)
    try:
        prepared = prepare_pt2e(gm, AllOpsQuantizer())
    except Exception as e:
        print(f"  FAIL prepare: {type(e).__name__}: {e}")
        return False

    print(f"  [calibrate]", flush=True)
    try:
        with torch.no_grad():
            prepared(*example_inputs)
    except Exception as e:
        print(f"  FAIL calibrate: {type(e).__name__}: {e}")
        return False

    print(f"  [convert_pt2e]", flush=True)
    try:
        quantized = convert_pt2e(prepared)
    except Exception as e:
        print(f"  FAIL convert: {type(e).__name__}: {e}")
        return False

    print(f"  [inference]", flush=True)
    try:
        with torch.no_grad():
            out = quantized(*example_inputs)
        print(f"  OK — output shape: {out[0].shape if isinstance(out, tuple) else out.shape}")
        return True
    except Exception as e:
        print(f"  FAIL inference: {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: use_cache=True (KV cache enabled)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: use_cache=True, fixed seq_len=16")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()
ids = torch.ones(1, 16, dtype=torch.long)

print("\n  [export] use_cache=True ...", flush=True)
try:
    ep = torch.export.export(model, (ids,), strict=False)
    gm = ep.module()
    print(f"  export OK — nodes: {len(list(gm.graph.nodes))}")
    run_pipeline(gm, (ids,), "use_cache=True")
except Exception as e:
    print(f"  FAIL export: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: use_cache=False, dynamic seq_len
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: use_cache=False, dynamic seq_len")
print("=" * 60)

class NoCacheWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids): return self.m(input_ids, use_cache=False).logits

model2 = NoCacheWrapper(AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval())
ids16 = torch.ones(1, 16, dtype=torch.long)

print("\n  [export] dynamic seq_len via Dim ...", flush=True)
try:
    seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
    dynamic_shapes = {"input_ids": {1: seq_dim}}
    ep = torch.export.export(model2, (ids16,), dynamic_shapes=dynamic_shapes, strict=False)
    gm = ep.module()
    print(f"  export OK — nodes: {len(list(gm.graph.nodes))}")

    # test with different seq lens
    for seq_len in [8, 16, 32, 64]:
        ids_test = torch.ones(1, seq_len, dtype=torch.long)
        with torch.no_grad():
            out = gm(ids_test)
        print(f"    dynamic forward seq_len={seq_len}: {out.shape}")

    run_pipeline(gm, (ids16,), "dynamic seq_len")
except Exception as e:
    print(f"  FAIL export: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: use_cache=True + dynamic seq_len (最難関)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: use_cache=True + dynamic seq_len (hardest)")
print("=" * 60)

model3 = AutoModelForCausalLM.from_pretrained(MODEL_PATH).eval()
ids16 = torch.ones(1, 16, dtype=torch.long)

print("\n  [export] use_cache=True + dynamic seq_len ...", flush=True)
try:
    seq_dim = torch.export.Dim("seq_len", min=1, max=2048)
    dynamic_shapes = {"input_ids": {1: seq_dim}}
    ep = torch.export.export(model3, (ids16,), dynamic_shapes=dynamic_shapes, strict=False)
    gm = ep.module()
    print(f"  export OK — nodes: {len(list(gm.graph.nodes))}")
    run_pipeline(gm, (ids16,), "use_cache=True + dynamic")
except Exception as e:
    print(f"  FAIL export: {type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()
