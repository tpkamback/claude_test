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


class _BaseFloatQuantizer(Quantizer):
    """Base: annotate target ops whose output is float; skip non-float inputs."""

    TARGET_OPS: set = set()

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
            # Only float tensor args get observers — Long indices are skipped
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


class LinearOnlyQuantizer(_BaseFloatQuantizer):
    """Quantize only linear/matmul ops (safe baseline for any model)."""
    TARGET_OPS = {
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
    }


class AllOpsQuantizer(_BaseFloatQuantizer):
    """Quantize all float compute ops (linear, bmm, activation, norm, etc.).

    scaled_dot_product_attention is excluded: its optional `scale` kwarg
    triggers an assertion inside prepare_pt2e (only clone/zeros_like/gelu
    kwargs are allowed).
    """
    TARGET_OPS = {
        torch.ops.aten.linear.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        # torch.ops.aten.scaled_dot_product_attention.default,  # scale kwarg → crash
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten._softmax.default,
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.tanh.default,
    }


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
                    quantizer=AllOpsQuantizer())


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
                    quantizer=AllOpsQuantizer())


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
                    quantizer=AllOpsQuantizer())


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
                    quantizer=AllOpsQuantizer())


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
                    quantizer=AllOpsQuantizer())


def test_opt_tiny():
    from transformers import OPTForCausalLM, OPTConfig
    config = OPTConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, ffn_dim=512, word_embed_proj_dim=256,
    )
    model = NoCacheWrapper(OPTForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("OPT-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


def test_bloom_tiny():
    from transformers import BloomForCausalLM, BloomConfig
    config = BloomConfig(
        n_layer=2, n_head=4, hidden_size=256,
    )
    model = NoCacheWrapper(BloomForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("BLOOM-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


def test_llama_tiny():
    from transformers import LlamaForCausalLM, LlamaConfig
    config = LlamaConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, intermediate_size=512,
        num_key_value_heads=4,
    )
    model = NoCacheWrapper(LlamaForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("LLaMA-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


def test_mistral_tiny():
    from transformers import MistralForCausalLM, MistralConfig
    config = MistralConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, intermediate_size=512,
        num_key_value_heads=4, sliding_window=16,
    )
    model = NoCacheWrapper(MistralForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Mistral-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


def test_falcon_tiny():
    from transformers import FalconForCausalLM, FalconConfig
    config = FalconConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256,
    )
    model = NoCacheWrapper(FalconForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Falcon-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


def test_phi2_tiny():
    from transformers import PhiForCausalLM, PhiConfig
    config = PhiConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, intermediate_size=512,
    )
    model = NoCacheWrapper(PhiForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Phi-2-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


def test_t5_tiny():
    from transformers import T5ForConditionalGeneration, T5Config
    config = T5Config(
        num_layers=2, num_heads=4, d_model=256, d_ff=512, d_kv=64,
        num_decoder_layers=2,
    )
    model = T5ForConditionalGeneration(config).eval()
    enc_ids = torch.ones(1, 16, dtype=torch.long)
    dec_ids = torch.ones(1, 8, dtype=torch.long)
    # T5 needs encoder + decoder input
    class T5Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, decoder_input_ids):
            return self.m(input_ids=input_ids,
                          decoder_input_ids=decoder_input_ids,
                          use_cache=False).logits
    wrapped = T5Wrapper(model)
    return run_pt2e("T5-tiny (enc-dec)", wrapped, (enc_ids, dec_ids),
                    quantizer=AllOpsQuantizer())


def test_roberta_tiny():
    from transformers import RobertaModel, RobertaConfig
    config = RobertaConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, intermediate_size=512,
    )
    model = RobertaModel(config)
    ids = torch.ones(1, 32, dtype=torch.long)
    mask = torch.ones(1, 32, dtype=torch.long)
    return run_pt2e("RoBERTa-tiny (2L)", model, (ids, mask),
                    quantizer=AllOpsQuantizer())


def test_gemma_tiny():
    from transformers import GemmaForCausalLM, GemmaConfig
    config = GemmaConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, intermediate_size=512,
        num_key_value_heads=4, head_dim=64,
    )
    model = NoCacheWrapper(GemmaForCausalLM(config))
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Gemma-tiny (2L)", model, (ids,), quantizer=AllOpsQuantizer())


# ── VLM ───────────────────────────────────────────────────────────────────────

def test_clip_tiny():
    from transformers import CLIPModel, CLIPConfig
    from transformers import CLIPTextConfig, CLIPVisionConfig
    vision_cfg = CLIPVisionConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, image_size=224, patch_size=32,
    )
    text_cfg = CLIPTextConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, max_position_embeddings=32,
    )
    config = CLIPConfig(text_config=text_cfg, vision_config=vision_cfg)
    model = CLIPModel(config).eval()

    class CLIPWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, pixel_values, input_ids, attention_mask):
            out = self.m(pixel_values=pixel_values, input_ids=input_ids,
                         attention_mask=attention_mask)
            return out.logits_per_image

    wrapped = CLIPWrapper(model)
    pixel = torch.randn(1, 3, 224, 224)
    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("CLIP-tiny", wrapped, (pixel, ids, mask),
                    quantizer=AllOpsQuantizer())


def test_llava_tiny():
    from transformers import LlavaForConditionalGeneration, LlavaConfig
    from transformers import CLIPVisionConfig, LlamaConfig
    vision_cfg = CLIPVisionConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, image_size=224, patch_size=32,
    )
    text_cfg = LlamaConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, num_key_value_heads=4,
    )
    config = LlavaConfig(vision_config=vision_cfg, text_config=text_cfg)
    model = LlavaForConditionalGeneration(config).eval()

    class LlavaWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, pixel_values):
            return self.m(input_ids=input_ids, pixel_values=pixel_values,
                          use_cache=False).logits

    wrapped = LlavaWrapper(model)
    # LLaVA expects image tokens embedded in input_ids (use dummy ids)
    ids = torch.ones(1, 20, dtype=torch.long)
    pixel = torch.randn(1, 3, 224, 224)
    return run_pt2e("LLaVA-tiny", wrapped, (ids, pixel),
                    quantizer=AllOpsQuantizer())


def test_paligemma_tiny():
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig
    from transformers import SiglipVisionConfig, GemmaConfig
    vision_cfg = SiglipVisionConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, image_size=224, patch_size=32,
    )
    text_cfg = GemmaConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, num_key_value_heads=4, head_dim=64,
    )
    config = PaliGemmaConfig(vision_config=vision_cfg, text_config=text_cfg)
    model = PaliGemmaForConditionalGeneration(config).eval()

    class PGWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, pixel_values):
            return self.m(input_ids=input_ids, pixel_values=pixel_values,
                          use_cache=False).logits

    wrapped = PGWrapper(model)
    ids = torch.ones(1, 20, dtype=torch.long)
    pixel = torch.randn(1, 3, 224, 224)
    return run_pt2e("PaliGemma-tiny", wrapped, (ids, pixel),
                    quantizer=AllOpsQuantizer())


def test_qwen2vl_tiny():
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
    config = Qwen2VLConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, num_key_value_heads=4,
    )
    model = Qwen2VLForConditionalGeneration(config).eval()

    class Qwen2VLWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).logits

    wrapped = Qwen2VLWrapper(model)
    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Qwen2-VL-tiny", wrapped, (ids, mask),
                    quantizer=AllOpsQuantizer())


def test_blip2_tiny():
    from transformers import Blip2ForConditionalGeneration, Blip2Config
    from transformers import Blip2VisionConfig, Blip2QFormerConfig, OPTConfig
    vision_cfg = Blip2VisionConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512, image_size=224, patch_size=32,
    )
    qformer_cfg = Blip2QFormerConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=512,
    )
    text_cfg = OPTConfig(
        hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
        ffn_dim=512, word_embed_proj_dim=256,
    )
    config = Blip2Config(vision_config=vision_cfg, qformer_config=qformer_cfg,
                         text_config=text_cfg)
    model = Blip2ForConditionalGeneration(config).eval()

    class Blip2Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, pixel_values, input_ids):
            return self.m(pixel_values=pixel_values, input_ids=input_ids,
                          use_cache=False).logits

    wrapped = Blip2Wrapper(model)
    pixel = torch.randn(1, 3, 224, 224)
    ids = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("BLIP-2-tiny", wrapped, (pixel, ids),
                    quantizer=AllOpsQuantizer())


def test_qwen2vl_local():
    import os
    path = "/home/user/claude_test/vlm_eval/models/qwen/Qwen2___5-VL-2B-Instruct"
    if not os.path.exists(path) or not any(
        f.endswith((".bin", ".safetensors")) for f in os.listdir(path)
    ):
        r = Result("Qwen2.5-VL-2B (local)")
        r.note = "model weights not found"
        return r
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(path).eval()

    class Qwen2VLWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).logits

    wrapped = Qwen2VLWrapper(model)
    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    return run_pt2e("Qwen2.5-VL-2B (local)", wrapped, (ids, mask),
                    quantizer=AllOpsQuantizer())


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    # Vision CNN
    test_resnet50,
    test_mobilenet_v2,
    test_efficientnet_b0,
    # Vision Transformer
    test_vit_b16,
    test_swin_t,
    test_convnext_tiny,
    # NLP Encoder
    test_bert,
    test_roberta_tiny,
    # Encoder-Decoder
    test_t5_tiny,
    # Decoder LLM
    test_gpt2_tiny,
    test_opt_tiny,
    test_bloom_tiny,
    test_llama_tiny,
    test_mistral_tiny,
    test_falcon_tiny,
    test_phi2_tiny,
    test_gemma_tiny,
    # Local LLM weights
    test_qwen2_local,
    test_qwen1_local,
    test_gpt2_local,
    # VLM
    test_clip_tiny,
    test_llava_tiny,
    test_paligemma_tiny,
    test_qwen2vl_tiny,
    test_blip2_tiny,
    test_qwen2vl_local,
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
