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

# ── mmlab / Detection & Segmentation ─────────────────────────────────────────

def test_unet_seg():
    import torch.nn as nn
    class DoubleConv(nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
        def forward(self, x): return self.net(x)
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = DoubleConv(3,64), DoubleConv(64,128), DoubleConv(128,256)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.d2 = DoubleConv(256, 128)
            self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.d1 = DoubleConv(128, 64)
            self.out = nn.Conv2d(64, 3, 1)
        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            b  = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(b), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)
    try:
        from mmseg.models.backbones import UNet as MMUNet
        class MMW(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bb = MMUNet(in_channels=3, base_channels=32, num_stages=4,
                                 strides=(1,1,1,1), enc_num_convs=(2,2,2,2),
                                 dec_num_convs=(2,2,2), enc_dilations=(1,1,1,1),
                                 dec_dilations=(1,1,1), with_cp=False,
                                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),
                                 upsample_cfg=dict(type='InterpConv'), norm_eval=False)
                self.head = nn.Conv2d(32, 3, 1)
            def forward(self, x): return self.head(self.bb(x)[-1])
        return trace("UNet (mmseg)", MMW(), (torch.randn(1, 3, 256, 256),))
    except Exception:
        return trace("UNet-seg", UNet(), (torch.randn(1, 3, 256, 256),))


def test_deeplab_v3():
    from torchvision.models.segmentation import deeplabv3_resnet50
    class W(torch.nn.Module):
        def __init__(self): super().__init__(); self.m = deeplabv3_resnet50(weights=None, num_classes=21)
        def forward(self, x): return self.m(x)["out"]
    return trace("DeepLabV3-R50", W(), (torch.randn(1, 3, 224, 224),))


def test_fcn_r50():
    from torchvision.models.segmentation import fcn_resnet50
    class W(torch.nn.Module):
        def __init__(self): super().__init__(); self.m = fcn_resnet50(weights=None, num_classes=21)
        def forward(self, x): return self.m(x)["out"]
    return trace("FCN-R50", W(), (torch.randn(1, 3, 224, 224),))


def test_yolov3():
    try:
        from mmdet.registry import MODELS as MMDET_MODELS
        cfg = dict(
            type="YOLOV3",
            backbone=dict(type="Darknet", depth=53, out_indices=(3, 4, 5)),
            neck=dict(type="YOLOV3Neck", num_scales=3,
                      in_channels=[256, 512, 1024], out_channels=[128, 256, 512]),
            bbox_head=dict(
                type="YOLOV3Head", num_classes=80,
                in_channels=[128, 256, 512], out_channels=[256, 512, 1024],
                anchor_generator=dict(
                    type="YOLOAnchorGenerator",
                    base_sizes=[[(10,13),(16,30),(33,23)],
                                [(30,61),(62,45),(59,119)],
                                [(116,90),(156,198),(373,326)]],
                    strides=[32, 16, 8]),
                bbox_coder=dict(type="YOLOBBoxCoder"),
                featmap_strides=[32, 16, 8]),
            train_cfg=None, test_cfg=None)
        m = MMDET_MODELS.build(cfg).eval()
        class W(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.m = inner
            def forward(self, x): return self.m.bbox_head(self.m.extract_feat(x))
        return trace("YOLOv3 (mmdet)", W(m), (torch.randn(1, 3, 416, 416),))
    except Exception:
        import torch.nn as nn
        class MinYOLOv3(nn.Module):
            def __init__(self, nc=80, na=3):
                super().__init__()
                def blk(ic, oc, s=1): return nn.Sequential(
                    nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(oc), nn.LeakyReLU(0.1, inplace=True))
                self.stem = blk(3, 32, 2); self.s2 = blk(32, 64, 2)
                self.s3 = blk(64, 128, 2); self.s4 = blk(128, 256, 2); self.s5 = blk(256, 512, 2)
                self.h3 = nn.Conv2d(128, na*(5+nc), 1)
                self.h4 = nn.Conv2d(256, na*(5+nc), 1)
                self.h5 = nn.Conv2d(512, na*(5+nc), 1)
            def forward(self, x):
                f3 = self.s3(self.s2(self.stem(x)))
                f4 = self.s4(f3); f5 = self.s5(f4)
                return self.h3(f3), self.h4(f4), self.h5(f5)
        return trace("YOLOv3-mini", MinYOLOv3(), (torch.randn(1, 3, 416, 416),))


def test_yolov5():
    try:
        import torch.hub
        m = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=False, verbose=False)
        class W(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.m = inner
            def forward(self, x): return self.m(x)
        return trace("YOLOv5s (hub)", W(m.model), (torch.randn(1, 3, 640, 640),))
    except Exception as e:
        r = Result("YOLOv5s (hub)"); r.note = f"{type(e).__name__}: {str(e)[:60]}"; return r


def test_centernet():
    try:
        from mmdet.registry import MODELS as MMDET_MODELS
        cfg = dict(
            type="CenterNet",
            backbone=dict(type="ResNet", depth=18, norm_cfg=dict(type="BN")),
            neck=dict(type="CTResNetNeck", in_channels=512,
                      num_deconv_filters=(256, 128, 64),
                      num_deconv_kernels=(4, 4, 4), use_dcn=False),
            bbox_head=dict(type="CenterNetHead", num_classes=80,
                           in_channels=64, feat_channels=64),
            train_cfg=None, test_cfg=None)
        m = MMDET_MODELS.build(cfg).eval()
        class W(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.m = inner
            def forward(self, x): return self.m.bbox_head(self.m.extract_feat(x))
        return trace("CenterNet (mmdet)", W(m), (torch.randn(1, 3, 512, 512),))
    except Exception:
        import torch.nn as nn
        class MinCenterNet(nn.Module):
            def __init__(self, nc=80):
                super().__init__()
                def blk(ic, oc, s=1): return nn.Sequential(
                    nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
                def up(ic, oc): return nn.Sequential(
                    nn.ConvTranspose2d(ic, oc, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
                self.enc = nn.Sequential(blk(3,64,2), blk(64,128,2), blk(128,256,2), blk(256,512,2))
                self.dec = nn.Sequential(up(512,256), up(256,128), up(128,64))
                self.hm = nn.Conv2d(64, nc, 1)
                self.wh = nn.Conv2d(64, 2, 1)
                self.off = nn.Conv2d(64, 2, 1)
            def forward(self, x):
                f = self.dec(self.enc(x))
                return self.hm(f), self.wh(f), self.off(f)
        return trace("CenterNet-mini", MinCenterNet(), (torch.randn(1, 3, 512, 512),))


def test_mobilenet_v3():
    import torchvision.models as m
    return trace("MobileNetV3-L", m.mobilenet_v3_large(weights=None),
                 (torch.randn(1, 3, 224, 224),))


def test_efficientnet_b4():
    import timm
    return trace("EfficientNet-B4", timm.create_model("efficientnet_b4", pretrained=False),
                 (torch.randn(1, 3, 380, 380),))


def test_segnext():
    try:
        from mmseg.registry import MODELS as MMSEG_MODELS
        cfg = dict(
            type="EncoderDecoder",
            backbone=dict(type="MSCAN", in_channels=3,
                          embed_dims=[32, 64, 160, 256], mlp_ratios=[8,8,4,4],
                          drop_rate=0.0, drop_path_rate=0.1, depths=[3,3,5,2],
                          norm_cfg=dict(type="BN2d")),
            decode_head=dict(type="LightHamHead", in_channels=[64,160,256],
                             in_index=[1,2,3], channels=256, ham_channels=256,
                             num_classes=150, dropout_ratio=0.1,
                             norm_cfg=dict(type="GN", num_groups=32),
                             align_corners=False,
                             loss_decode=dict(type="CrossEntropyLoss")),
            auxiliary_head=None, train_cfg=None, test_cfg=dict(mode="whole"))
        m = MMSEG_MODELS.build(cfg).eval()
        class W(torch.nn.Module):
            def __init__(self, inner): super().__init__(); self.m = inner
            def forward(self, x): return self.m.decode_head(self.m.extract_feat(x))
        return trace("SegNeXt-T (mmseg)", W(m), (torch.randn(1, 3, 512, 512),))
    except Exception:
        import torch.nn as nn, torch.nn.functional as F
        class MinSegNeXt(nn.Module):
            def __init__(self, nc=150):
                super().__init__()
                def blk(ic, oc, s=1): return nn.Sequential(
                    nn.Conv2d(ic, oc, 3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(oc), nn.GELU())
                self.s1 = blk(3, 32, 4); self.s2 = blk(32, 64, 2)
                self.s3 = blk(64, 160, 2); self.s4 = blk(160, 256, 2)
                self.fuse = nn.Conv2d(160+256, 256, 1)
                self.head = nn.Conv2d(256, nc, 1)
            def forward(self, x):
                f2 = self.s2(self.s1(x))
                f3 = self.s3(f2); f4 = self.s4(f3)
                f4u = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
                fused = self.fuse(torch.cat([f3, f4u], 1))
                return self.head(F.interpolate(fused, scale_factor=4, mode='bilinear', align_corners=False))
        return trace("SegNeXt-mini", MinSegNeXt(), (torch.randn(1, 3, 512, 512),))


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
    # mmlab / Detection & Segmentation
    test_unet_seg, test_deeplab_v3, test_fcn_r50,
    test_yolov3, test_yolov5, test_centernet,
    test_mobilenet_v3, test_efficientnet_b4, test_segnext,
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
