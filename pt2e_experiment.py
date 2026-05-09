"""
PT2E quantization experiment across multiple model architectures.
torch 2.11: API moved to torchao.quantization.pt2e
"""

import torch
import traceback
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.quantizer import (
    Quantizer,
    QuantizationAnnotation,
    QuantizationSpec,
)
from torchao.quantization.pt2e.observer import MinMaxObserver
from torchao.testing.pt2e._xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


@dataclass
class Result:
    model_name: str
    export_ok: bool = False
    prepare_ok: bool = False
    convert_ok: bool = False
    inference_ok: bool = False
    note: str = ""


def get_xnnpack_quantizer():
    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config())
    return q


_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Linear-only quantizer for LLMs: avoids inserting observers on Long/int indices
class LinearOnlyQuantizer(Quantizer):
    """Quantize only aten.linear.default (and addmm/mm) with per-tensor symmetric int8."""

    TARGET_OPS = {
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
    }

    def _is_float(self, node: torch.fx.Node) -> bool:
        val = node.meta.get("val")
        if val is None:
            return False
        return getattr(val, "dtype", None) in _FLOAT_DTYPES

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        act_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            observer_or_fake_quant_ctr=MinMaxObserver,
        )
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in self.TARGET_OPS:
                continue
            if not self._is_float(node):
                continue
            input_map = {
                arg: act_spec
                for arg in node.args
                if isinstance(arg, torch.fx.Node) and self._is_float(arg)
            }
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_map,
                output_qspec=act_spec,
            )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass


def run_pt2e(model_name: str, model, example_inputs: tuple,
             quantizer=None) -> Result:
    result = Result(model_name)
    model.eval()
    if quantizer is None:
        quantizer = get_xnnpack_quantizer()

    # Step 1: export
    try:
        ep = torch.export.export(model, example_inputs, strict=False)
        gm = ep.module()
        result.export_ok = True
    except Exception as e:
        result.note = f"export: {type(e).__name__}: {str(e)[:100]}"
        return result

    # Step 2: prepare_pt2e
    try:
        prepared = prepare_pt2e(gm, quantizer)
        result.prepare_ok = True
    except Exception as e:
        result.note = f"prepare: {type(e).__name__}: {str(e)[:100]}"
        return result

    # Step 3: calibration
    try:
        with torch.no_grad():
            prepared(*example_inputs)
    except Exception as e:
        result.note = f"calibrate: {type(e).__name__}: {str(e)[:100]}"
        return result

    # Step 4: convert_pt2e
    try:
        quantized = convert_pt2e(prepared)
        result.convert_ok = True
    except Exception as e:
        result.note = f"convert: {type(e).__name__}: {str(e)[:100]}"
        return result

    # Step 5: inference
    try:
        with torch.no_grad():
            quantized(*example_inputs)
        result.inference_ok = True
        result.note = "ALL OK"
    except Exception as e:
        result.note = f"inference: {type(e).__name__}: {str(e)[:100]}"

    return result


# ── Vision CNN ────────────────────────────────────────────────────────────────

def test_resnet50():
    import torchvision.models as m
    return run_pt2e("ResNet50", m.resnet50(weights=None),
                    (torch.randn(1, 3, 224, 224),))


def test_mobilenet_v2():
    import torchvision.models as m
    return run_pt2e("MobileNetV2", m.mobilenet_v2(weights=None),
                    (torch.randn(1, 3, 224, 224),))


def test_efficientnet_b0():
    import torchvision.models as m
    return run_pt2e("EfficientNet-B0", m.efficientnet_b0(weights=None),
                    (torch.randn(1, 3, 224, 224),))


# ── Vision Transformer ────────────────────────────────────────────────────────

def test_vit_b16():
    import torchvision.models as m
    return run_pt2e("ViT-B/16", m.vit_b_16(weights=None),
                    (torch.randn(1, 3, 224, 224),))


def test_swin_t():
    import torchvision.models as m
    return run_pt2e("Swin-T", m.swin_t(weights=None),
                    (torch.randn(1, 3, 224, 224),))


def test_convnext_tiny():
    import timm
    return run_pt2e("ConvNeXt-Tiny", timm.create_model("convnext_tiny", pretrained=False),
                    (torch.randn(1, 3, 224, 224),))


# ── NLP Encoder ───────────────────────────────────────────────────────────────

def test_bert():
    from transformers import BertModel, BertConfig
    config = BertConfig(num_hidden_layers=2, num_attention_heads=4,
                        hidden_size=256, intermediate_size=512)
    model = BertModel(config)
    ids = torch.ones(1, 32, dtype=torch.long)
    mask = torch.ones(1, 32, dtype=torch.long)
    return run_pt2e("BERT-tiny (fixed len)", model, (ids, mask),
                    quantizer=LinearOnlyQuantizer())


# ── LLM / Decoder ─────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    """Disable KV cache so export doesn't choke on DynamicCache."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids, use_cache=False).logits


def test_gpt2_tiny():
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config(n_layer=2, n_head=4, n_embd=256)
    model = NoCacheWrapper(GPT2LMHeadModel(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("GPT-2-tiny (2L)", model, (ids,),
                    quantizer=LinearOnlyQuantizer())


def test_qwen2_local():
    import os
    path = "/home/user/claude_test/eval_ppl/models/qwen2-tiny"
    if not os.path.exists(path):
        r = Result("Qwen2-tiny (local)")
        r.note = "model not found"
        return r
    from transformers import AutoModelForCausalLM
    model = NoCacheWrapper(AutoModelForCausalLM.from_pretrained(path))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Qwen2-tiny (local)", model, (ids,),
                    quantizer=LinearOnlyQuantizer())


def test_qwen1_local():
    import os
    path = "/home/user/claude_test/eval_ppl/models/qwen-tiny"
    if not os.path.exists(path):
        r = Result("Qwen1-tiny (local)")
        r.note = "model not found"
        return r
    from transformers import AutoModelForCausalLM
    model = NoCacheWrapper(
        AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    )
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Qwen1-tiny (local)", model, (ids,),
                    quantizer=LinearOnlyQuantizer())


def test_gpt2_local():
    import os
    path = "/home/user/claude_test/eval_ppl/models/gpt2-tiny"
    if not os.path.exists(path):
        r = Result("GPT2-tiny (local)")
        r.note = "model not found"
        return r
    from transformers import AutoModelForCausalLM
    model = NoCacheWrapper(AutoModelForCausalLM.from_pretrained(path))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("GPT2-tiny (local)", model, (ids,),
                    quantizer=LinearOnlyQuantizer())


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    test_resnet50,
    test_mobilenet_v2,
    test_efficientnet_b0,
    test_vit_b16,
    test_swin_t,
    test_convnext_tiny,
    test_bert,
    test_gpt2_tiny,
    test_qwen2_local,
    test_qwen1_local,
    test_gpt2_local,
]


def print_results(results):
    def mark(b): return "OK  " if b else "FAIL"
    print("\n" + "=" * 90)
    print(f"torch {torch.__version__}  |  PT2E Quantization Results")
    print("=" * 90)
    print(f"{'Model':<28} {'Export':^6} {'Prepare':^8} {'Convert':^8} {'Infer':^6}  Note")
    print("-" * 90)
    for r in results:
        print(f"{r.model_name:<28} {mark(r.export_ok):^6} {mark(r.prepare_ok):^8} "
              f"{mark(r.convert_ok):^8} {mark(r.inference_ok):^6}  {r.note[:42]}")
    print("=" * 90)


if __name__ == "__main__":
    results = []
    for fn in TESTS:
        name = fn.__name__.replace("test_", "")
        print(f"\n>>> {name}", flush=True)
        try:
            r = fn()
        except Exception as e:
            r = Result(name, note=f"crash: {type(e).__name__}: {str(e)[:80]}")
            traceback.print_exc()
        results.append(r)
        print(f"    export={r.export_ok} prepare={r.prepare_ok} "
              f"convert={r.convert_ok} infer={r.inference_ok}  {r.note[:60]}")

    print_results(results)
