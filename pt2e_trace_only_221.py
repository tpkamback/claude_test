"""
PT2E trace-only minimal test — torch 2.2.1

torch.export.export の成否のみを測定する。
量子化 (prepare/convert) は行わない。

torch 2.2.1 固有の注意:
  - torch.compiler.is_compiling() が未実装のため stub が必要
  - transformers 4.44.0
  - strict=False を使う (transformers モデルは strict モードで失敗する)
"""

import torch
import traceback
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass

# ── torch 2.2.1 パッチ ────────────────────────────────────────────────────────
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False


@dataclass
class Result:
    model_name: str
    trace_ok: bool = False
    n_nodes: int = 0
    note: str = ""


def trace(model_name: str, model, example_inputs: tuple) -> Result:
    result = Result(model_name)
    model.eval()
    try:
        ep = torch.export.export(model, example_inputs, strict=False)
        gm = ep.module()
        result.trace_ok = True
        result.n_nodes = len(list(gm.graph.nodes))
        result.note = f"nodes={result.n_nodes}"
    except Exception as e:
        result.note = f"{type(e).__name__}: {str(e)[:100]}"
    return result


# ── Vision CNN ────────────────────────────────────────────────────────────────

def test_resnet50():
    import torchvision.models as m
    return trace("ResNet50", m.resnet50(weights=None),
                 (torch.randn(1, 3, 224, 224),))

def test_mobilenet_v2():
    import torchvision.models as m
    return trace("MobileNetV2", m.mobilenet_v2(weights=None),
                 (torch.randn(1, 3, 224, 224),))

def test_efficientnet_b0():
    import torchvision.models as m
    return trace("EfficientNet-B0", m.efficientnet_b0(weights=None),
                 (torch.randn(1, 3, 224, 224),))

# ── Vision Transformer ────────────────────────────────────────────────────────

def test_vit_b16():
    import torchvision.models as m
    return trace("ViT-B/16", m.vit_b_16(weights=None),
                 (torch.randn(1, 3, 224, 224),))

def test_swin_t():
    import torchvision.models as m
    return trace("Swin-T", m.swin_t(weights=None),
                 (torch.randn(1, 3, 224, 224),))

def test_convnext_tiny():
    import timm
    return trace("ConvNeXt-Tiny", timm.create_model("convnext_tiny", pretrained=False),
                 (torch.randn(1, 3, 224, 224),))

# ── NLP Encoder ───────────────────────────────────────────────────────────────

def test_bert():
    from transformers import BertModel, BertConfig
    config = BertConfig(num_hidden_layers=2, num_attention_heads=4,
                        hidden_size=256, intermediate_size=512)
    ids = torch.ones(1, 32, dtype=torch.long)
    mask = torch.ones(1, 32, dtype=torch.long)
    return trace("BERT-tiny", BertModel(config), (ids, mask))

def test_roberta_tiny():
    from transformers import RobertaModel, RobertaConfig
    config = RobertaConfig(num_hidden_layers=2, num_attention_heads=4,
                           hidden_size=256, intermediate_size=512)
    ids = torch.ones(1, 32, dtype=torch.long)
    mask = torch.ones(1, 32, dtype=torch.long)
    return trace("RoBERTa-tiny", RobertaModel(config), (ids, mask))

# ── Encoder-Decoder ───────────────────────────────────────────────────────────

def test_t5_tiny():
    from transformers import T5ForConditionalGeneration, T5Config
    config = T5Config(num_layers=2, num_heads=4, d_model=256,
                      d_ff=512, d_kv=64, num_decoder_layers=2)
    model = T5ForConditionalGeneration(config).eval()
    class T5Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, decoder_input_ids):
            return self.m(input_ids=input_ids,
                          decoder_input_ids=decoder_input_ids,
                          use_cache=False).logits
    enc = torch.ones(1, 16, dtype=torch.long)
    dec = torch.ones(1, 8, dtype=torch.long)
    return trace("T5-tiny", T5Wrapper(model), (enc, dec))

# ── LLM / Decoder ─────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, input_ids): return self.model(input_ids, use_cache=False).logits

def test_gpt2_tiny():
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config(n_layer=2, n_head=4, n_embd=256)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("GPT-2-tiny", NoCacheWrapper(GPT2LMHeadModel(config)), (ids,))

def test_opt_tiny():
    from transformers import OPTForCausalLM, OPTConfig
    config = OPTConfig(num_hidden_layers=2, num_attention_heads=4,
                       hidden_size=256, ffn_dim=512, word_embed_proj_dim=256)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("OPT-tiny", NoCacheWrapper(OPTForCausalLM(config)), (ids,))

def test_bloom_tiny():
    from transformers import BloomForCausalLM, BloomConfig
    config = BloomConfig(n_layer=2, n_head=4, hidden_size=256)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("BLOOM-tiny", NoCacheWrapper(BloomForCausalLM(config)), (ids,))

def test_llama_tiny():
    from transformers import LlamaForCausalLM, LlamaConfig
    config = LlamaConfig(num_hidden_layers=2, num_attention_heads=4,
                         hidden_size=256, intermediate_size=512,
                         num_key_value_heads=4)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("LLaMA-tiny", NoCacheWrapper(LlamaForCausalLM(config)), (ids,))

def test_mistral_tiny():
    from transformers import MistralForCausalLM, MistralConfig
    config = MistralConfig(num_hidden_layers=2, num_attention_heads=4,
                           hidden_size=256, intermediate_size=512,
                           num_key_value_heads=4, sliding_window=16)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("Mistral-tiny", NoCacheWrapper(MistralForCausalLM(config)), (ids,))

def test_falcon_tiny():
    from transformers import FalconForCausalLM, FalconConfig
    config = FalconConfig(num_hidden_layers=2, num_attention_heads=4, hidden_size=256)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("Falcon-tiny", NoCacheWrapper(FalconForCausalLM(config)), (ids,))

def test_phi2_tiny():
    from transformers import PhiForCausalLM, PhiConfig
    config = PhiConfig(num_hidden_layers=2, num_attention_heads=4,
                       hidden_size=256, intermediate_size=512)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("Phi-2-tiny", NoCacheWrapper(PhiForCausalLM(config)), (ids,))

def test_gemma_tiny():
    from transformers import GemmaForCausalLM, GemmaConfig
    config = GemmaConfig(num_hidden_layers=2, num_attention_heads=4,
                         hidden_size=256, intermediate_size=512,
                         num_key_value_heads=4, head_dim=64)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("Gemma-tiny", NoCacheWrapper(GemmaForCausalLM(config)), (ids,))

def test_qwen2_local():
    import os
    path = "/home/user/claude_test/eval_ppl/models/qwen2-tiny"
    if not os.path.exists(path):
        r = Result("Qwen2-tiny (local)"); r.note = "not found"; return r
    from transformers import AutoModelForCausalLM
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("Qwen2-tiny (local)",
                 NoCacheWrapper(AutoModelForCausalLM.from_pretrained(path)), (ids,))

def test_qwen1_local():
    import os
    path = "/home/user/claude_test/eval_ppl/models/qwen-tiny"
    if not os.path.exists(path):
        r = Result("Qwen1-tiny (local)"); r.note = "not found"; return r
    from transformers import AutoModelForCausalLM
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("Qwen1-tiny (local)",
                 NoCacheWrapper(AutoModelForCausalLM.from_pretrained(
                     path, trust_remote_code=True)), (ids,))

def test_gpt2_local():
    import os
    path = "/home/user/claude_test/eval_ppl/models/gpt2-tiny"
    if not os.path.exists(path):
        r = Result("GPT2-tiny (local)"); r.note = "not found"; return r
    from transformers import AutoModelForCausalLM
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("GPT2-tiny (local)",
                 NoCacheWrapper(AutoModelForCausalLM.from_pretrained(path)), (ids,))

# ── VLM ───────────────────────────────────────────────────────────────────────

def test_clip_tiny():
    from transformers import CLIPModel, CLIPConfig, CLIPTextConfig, CLIPVisionConfig
    vision_cfg = CLIPVisionConfig(hidden_size=256, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=512,
                                  image_size=224, patch_size=32)
    text_cfg = CLIPTextConfig(hidden_size=256, num_hidden_layers=2,
                              num_attention_heads=4, intermediate_size=512,
                              max_position_embeddings=32)
    config = CLIPConfig(text_config=text_cfg, vision_config=vision_cfg)
    model = CLIPModel(config).eval()
    class CLIPWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, pixel_values, input_ids, attention_mask):
            return self.m(pixel_values=pixel_values, input_ids=input_ids,
                          attention_mask=attention_mask).logits_per_image
    pixel = torch.randn(1, 3, 224, 224)
    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    return trace("CLIP-tiny", CLIPWrapper(model), (pixel, ids, mask))

def test_llava_tiny():
    from transformers import LlavaForConditionalGeneration, LlavaConfig
    from transformers import CLIPVisionConfig, LlamaConfig
    vision_cfg = CLIPVisionConfig(hidden_size=256, num_hidden_layers=2,
                                  num_attention_heads=4, intermediate_size=512,
                                  image_size=224, patch_size=32)
    text_cfg = LlamaConfig(hidden_size=256, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=512,
                           num_key_value_heads=4)
    config = LlavaConfig(vision_config=vision_cfg, text_config=text_cfg)
    model = LlavaForConditionalGeneration(config).eval()
    class LlavaWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, pixel_values):
            return self.m(input_ids=input_ids, pixel_values=pixel_values,
                          use_cache=False).logits
    ids = torch.ones(1, 20, dtype=torch.long)
    pixel = torch.randn(1, 3, 224, 224)
    return trace("LLaVA-tiny", LlavaWrapper(model), (ids, pixel))

def test_paligemma_tiny():
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig
    from transformers import SiglipVisionConfig, GemmaConfig
    vision_cfg = SiglipVisionConfig(hidden_size=256, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=512,
                                    image_size=224, patch_size=32)
    text_cfg = GemmaConfig(hidden_size=256, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=512,
                           num_key_value_heads=4, head_dim=64)
    config = PaliGemmaConfig(vision_config=vision_cfg, text_config=text_cfg)
    model = PaliGemmaForConditionalGeneration(config).eval()
    class PGWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, pixel_values):
            return self.m(input_ids=input_ids, pixel_values=pixel_values,
                          use_cache=False).logits
    ids = torch.ones(1, 20, dtype=torch.long)
    pixel = torch.randn(1, 3, 224, 224)
    return trace("PaliGemma-tiny", PGWrapper(model), (ids, pixel))

def test_qwen2vl_tiny():
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
    config = Qwen2VLConfig(hidden_size=256, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=512,
                           num_key_value_heads=4)
    model = Qwen2VLForConditionalGeneration(config).eval()
    class Qwen2VLWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).logits
    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    return trace("Qwen2-VL-tiny", Qwen2VLWrapper(model), (ids, mask))

def test_blip2_tiny():
    from transformers import Blip2ForConditionalGeneration, Blip2Config
    from transformers import Blip2VisionConfig, Blip2QFormerConfig, OPTConfig
    vision_cfg = Blip2VisionConfig(hidden_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, intermediate_size=512,
                                   image_size=224, patch_size=32)
    qformer_cfg = Blip2QFormerConfig(hidden_size=256, num_hidden_layers=2,
                                     num_attention_heads=4, intermediate_size=512)
    text_cfg = OPTConfig(hidden_size=256, num_hidden_layers=2,
                         num_attention_heads=4, ffn_dim=512, word_embed_proj_dim=256)
    config = Blip2Config(vision_config=vision_cfg, qformer_config=qformer_cfg,
                         text_config=text_cfg)
    model = Blip2ForConditionalGeneration(config).eval()
    class Blip2Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, pixel_values, input_ids):
            return self.m(pixel_values=pixel_values, input_ids=input_ids,
                          use_cache=False).logits
    pixel = torch.randn(1, 3, 224, 224)
    ids = torch.ones(1, 16, dtype=torch.long)
    return trace("BLIP-2-tiny", Blip2Wrapper(model), (pixel, ids))

# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    # Vision CNN
    test_resnet50, test_mobilenet_v2, test_efficientnet_b0,
    # Vision Transformer
    test_vit_b16, test_swin_t, test_convnext_tiny,
    # NLP Encoder
    test_bert, test_roberta_tiny,
    # Encoder-Decoder
    test_t5_tiny,
    # Decoder LLM
    test_gpt2_tiny, test_opt_tiny, test_bloom_tiny, test_llama_tiny,
    test_mistral_tiny, test_falcon_tiny, test_phi2_tiny, test_gemma_tiny,
    # Local weights
    test_qwen2_local, test_qwen1_local, test_gpt2_local,
    # VLM
    test_clip_tiny, test_llava_tiny, test_paligemma_tiny,
    test_qwen2vl_tiny, test_blip2_tiny,
]


def print_results(results):
    ok = sum(r.trace_ok for r in results)
    print(f"\n{'='*70}")
    print(f"torch {torch.__version__}  |  torch.export.export  |  {ok}/{len(results)} OK")
    print(f"{'='*70}")
    print(f"{'Model':<28} {'Trace':^6}  Note")
    print(f"{'-'*70}")
    for r in results:
        mark = "OK  " if r.trace_ok else "FAIL"
        print(f"{r.model_name:<28} {mark:^6}  {r.note[:40]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print(f"torch {torch.__version__} / torch.export.export trace-only test")
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
        mark = "OK" if r.trace_ok else "FAIL"
        print(f"    {mark}  {r.note[:70]}")

    print_results(results)
