"""
MediaTek executorch LLaMA 量子化フロー再現スクリプト
https://github.com/pytorch/executorch/blob/main/examples/mediatek/model_export_scripts/llama.py

【オリジナルとの主な違い】
  - 入力: input_ids ではなく inputs_embeds + mask + pos_emb + cache
      → モデルを複数 chunk に分割し、各 chunk が hidden_state を受け取る
  - 量子化: A16W4 / A16W8 (activations はfloat16、weights のみ int4/int8)
  - dynamic_shapes: 可変 seq_len でエクスポート
  - 各固定形状ごとに再エクスポート → MethodProgramsPartitionerSpec で多メソッド委譲

【このスクリプト】
  NeuropilotQuantizer → 独自 WeightOnlyQuantizer (A16W8 相当) で代替
  MediaTek edge 変換 (.pte 生成) は省略
  tiny LLaMA デコーダ chunk で動作確認
"""

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer import (
    Quantizer, QuantizationAnnotation, QuantizationSpec,
)
from torchao.quantization.pt2e.observer import MinMaxObserver

print(f"torch   : {torch.__version__}")
import torchao; print(f"torchao : {torchao.__version__}")
print()

# ── A16W8 相当 Quantizer (重みのみ int8) ─────────────────────────────────────
# オリジナル:
#   quantizer = NeuropilotQuantizer()
#   quantizer.setup_precision(Precision.A16W8)

_FLOAT = (torch.float32, torch.float16, torch.bfloat16)

class WeightOnlyInt8Quantizer(Quantizer):
    """activations は float のまま、weights のみ per-tensor symmetric int8 量子化。
    MediaTek の A16W8 モード相当。"""

    WEIGHT_OPS = {
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }

    def _is_float(self, node):
        val = node.meta.get("val")
        return val is not None and getattr(val, "dtype", None) in _FLOAT

    def annotate(self, model):
        w_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128, quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            observer_or_fake_quant_ctr=MinMaxObserver,
        )
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in self.WEIGHT_OPS:
                continue
            if not self._is_float(node):
                continue
            # linear の arg[1] が weight (2次元 float tensor)
            args = node.args
            weight_node = args[1] if len(args) > 1 else None
            if weight_node is None or not isinstance(weight_node, torch.fx.Node):
                continue
            if not self._is_float(weight_node):
                continue
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={weight_node: w_spec},
                output_qspec=None,   # activation は量子化しない
            )
        return model

    def validate(self, model):
        pass


# ── tiny LLaMA デコーダ chunk ─────────────────────────────────────────────────
# オリジナルは LlamaForCausalLM をレイヤー単位で chunk 分割し、
# 各 chunk が (inputs_embeds, mask, pos_emb, *cache) を受け取る。
# ここでは transformers の LlamaDecoderLayer を使った最小 chunk を再現。

def make_llama_chunk(n_layers=2, hidden=256, heads=4, kv_heads=4, ffn=512):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=ffn,
        num_hidden_layers=n_layers,
        max_position_embeddings=256,
    )

    class LlamaChunk(nn.Module):
        """hidden_state を受け取り hidden_state を返す chunk。
        KV cache は use_cache=False で除去。"""
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [LlamaDecoderLayer(config, i) for i in range(n_layers)])
            self.norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)

        def forward(self, hidden_states, attention_mask=None, position_ids=None):
            for layer in self.layers:
                out = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                )
                hidden_states = out[0]
            return self.norm(hidden_states)

    return LlamaChunk().eval()


# ── 再現フロー ─────────────────────────────────────────────────────────────────

def run_llama_pt2e(seq_len=16):
    print(f"{'='*60}")
    print(f"[LLaMA chunk] PT2E A16W8 量子化  (seq_len={seq_len})")
    print(f"{'='*60}")

    HIDDEN, HEADS = 256, 4
    model = make_llama_chunk()

    hidden_states  = torch.randn(1, seq_len, HIDDEN)
    attention_mask = torch.zeros(1, 1, seq_len, seq_len)  # causal mask (additive)
    position_ids   = torch.arange(seq_len).unsqueeze(0)
    inputs = (hidden_states, attention_mask, position_ids)

    # ── STEP 1: torch.export (dynamic_shapes で可変 seq_len) ───────
    # オリジナル:
    #   example_inputs, dynamic_shapes = model.get_example_inputs(max_num_token, max_cache_size, True)
    #   pre_autograd = torch.export.export(model, example_inputs,
    #                                      dynamic_shapes=dynamic_shapes, strict=True).module()
    print("STEP 1: torch.export.export(dynamic_shapes, strict=False) ...")
    L = torch.export.Dim("seq_len", min=1, max=128)
    dynamic_shapes = ({0: None, 1: L, 2: None},   # hidden_states [B, L, H]
                      {0: None, 1: None, 2: L, 3: L},  # mask [B,1,L,L]
                      {0: None, 1: L})              # position_ids [B, L]
    try:
        ep = torch.export.export(model, inputs,
                                 dynamic_shapes=dynamic_shapes, strict=False)
        captured = ep.module()
        print(f"  → {len(list(captured.graph.nodes))} nodes (dynamic export OK)")
    except Exception as e:
        print(f"  dynamic export FAIL ({type(e).__name__}: {str(e)[:80]})")
        print("  → falling back to static export ...")
        ep = torch.export.export(model, inputs, strict=False)
        captured = ep.module()
        print(f"  → {len(list(captured.graph.nodes))} nodes (static export)")

    # ── STEP 2: prepare_pt2e ───────────────────────────────────────
    # オリジナル:
    #   quantizer = NeuropilotQuantizer()
    #   quantizer.setup_precision(Precision.A16W8)
    #   prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    print("STEP 2: prepare_pt2e (WeightOnlyInt8 / A16W8 相当) ...")
    quantizer = WeightOnlyInt8Quantizer()
    annotated = prepare_pt2e(captured, quantizer)
    annotated_w = [n for n in annotated.graph.nodes
                   if "observer" in str(n.target) or "fake_quant" in str(n.target)]
    print(f"  → observer 挿入数: {len(annotated_w)}")

    # ── STEP 3: calibration ────────────────────────────────────────
    # オリジナル:
    #   if cal_dataset: calibrate_model(prepared_graph, cal_dataset, chunk_idx)
    #   else:           prepared_graph(*example_inputs)
    print("STEP 3: calibration (dummy 3 samples) ...")
    with torch.no_grad():
        for _ in range(3):
            h = torch.randn(1, seq_len, HIDDEN)
            m = torch.zeros(1, 1, seq_len, seq_len)
            p = torch.arange(seq_len).unsqueeze(0)
            annotated(h, m, p)
    print("  → done")

    # ── STEP 4: convert_pt2e ───────────────────────────────────────
    # オリジナル: convert_pt2e(prepared_graph, fold_quantize=False)
    print("STEP 4: convert_pt2e(fold_quantize=False) ...")
    quantized = convert_pt2e(annotated, fold_quantize=False)
    all_nodes = list(quantized.graph.nodes)
    qdq_nodes = [n for n in all_nodes
                 if "quantize_per_tensor" in str(n.target)
                 or "dequantize_per_tensor" in str(n.target)]
    linear_nodes = [n for n in all_nodes if n.target == torch.ops.aten.linear.default]
    print(f"  → total nodes   : {len(all_nodes)}")
    print(f"  → QDQ nodes     : {len(qdq_nodes)}")
    print(f"  → linear nodes  : {len(linear_nodes)}")

    # ── STEP 5: 推論確認 ────────────────────────────────────────────
    print("STEP 5: 推論確認 ...")
    with torch.no_grad():
        out_f = model(*inputs)
        out_q = quantized(*inputs)
    diff = (out_f - out_q).abs().mean().item()
    print(f"  float output : {out_f.shape}, quant output : {out_q.shape}")
    print(f"  mean |float - quant| : {diff:.4f}")

    # ── STEP 6 (省略): 固定形状ごとの再エクスポート + MediaTek backend
    # オリジナル:
    #   for shape, ntok_and_cache in export_shapes.items():
    #       aten_dialect = torch.export.export(converted_graph, example_inputs, strict=True)
    #       method_to_edge_program[fname] = exir.to_edge(aten_dialect).exported_program()
    #       method_to_partitioner[fname] = NeuropilotPartitioner(compile_spec)
    #   delegated = to_backend(MethodProgramsPartitionerSpec(...))
    #   edge_manager.to_executorch(...)  →  .pte 書き出し
    print("STEP 6: 固定形状再エクスポート + NeuropilotPartitioner → 省略")
    print()


run_llama_pt2e(seq_len=16)
print("完了")
