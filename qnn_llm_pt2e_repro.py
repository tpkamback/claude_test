#!/usr/bin/env python3
"""
Qualcomm QNN PT2E LLM activation quantization 再現スクリプト。
examples/qualcomm/oss_scripts/llm_utils/qnn_decoder_model_manager.py の
pt2e_quantize() フローを直接再現。

オリジナルフロー:
  quantizer = make_quantizer(quant_dtype=QuantDtype.use_8a8w,
                              per_channel_linear=True, act_observer=MinMaxObserver)
  graph_module = prepare_pt2e(graph_module, quantizer)
  pt2e_calibrate(...)          # LLM forward でキャリブレーション
  graph_module = convert_pt2e(graph_module)
  # → to_edge_transform_and_lower_to_qnn / .pte (省略)

Run: QNN_SDK_ROOT=/fake python3 qnn_llm_pt2e_repro.py
"""
import os, warnings
os.environ.setdefault("QNN_SDK_ROOT", "/fake")
warnings.filterwarnings("ignore")

import torch
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

print(f"torch   : {torch.__version__}")
import torchao; print(f"torchao : {torchao.__version__}")
print()

# QnnQuantizer インポート (Pure Python; SDK 不要)
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType, QcomChipset)
print(f"QnnQuantizer : {QnnQuantizer}")
print(f"QuantDtype   : {list(QuantDtype.__members__)}")
print()

# ── パッチ: LiftConstantScalarOperands が _schema を持たない OP をスキップ ─────
# torch 2.11 の torch.export では WrapWithSetGradEnabled など _schema 非持ちノードが
# 出現することがあり、QNN の変換パスが AttributeError を出す。
import executorch.backends.qualcomm._passes.lift_constant_scalar_operands as _lc
_orig_create = _lc.LiftConstantScalarOperands._create_tensor_args

def _safe_create_tensor_args(self, node, gm):
    if not hasattr(node.target, "_schema"):
        return {}
    return _orig_create(self, node, gm)

_lc.LiftConstantScalarOperands._create_tensor_args = _safe_create_tensor_args
# ────────────────────────────────────────────────────────────────────────────────


def run_llm(precision_name: str, quant_dtype):
    from transformers import LlamaForCausalLM, LlamaConfig

    config = LlamaConfig(
        num_hidden_layers=2, num_attention_heads=4,
        hidden_size=256, intermediate_size=512, num_key_value_heads=4)

    class NoCacheWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, ids): return self.m(ids, use_cache=False).logits

    model = NoCacheWrapper(LlamaForCausalLM(config)).eval()
    ids   = torch.ones(1, 16, dtype=torch.long)
    params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print(f"[tiny LLaMA] QNN PT2E {precision_name}")
    print(f"  params={params:,}  (hidden=256, 2-layer, 15 linears)")
    print("=" * 60)

    # STEP 1
    print("STEP 1: torch.export.export(strict=False) ...")
    exp = torch.export.export(model, (ids,), strict=False).module()
    ns0  = list(exp.graph.nodes)
    lin0 = [n for n in ns0 if n.target == torch.ops.aten.linear.default]
    print(f"  → {len(ns0)} nodes  ({len(lin0)} linear)")

    # STEP 2
    print(f"STEP 2: prepare_pt2e (QnnQuantizer / {precision_name}) ...")
    quantizer = QnnQuantizer(
        backend=QnnExecuTorchBackendType.kHtpBackend,
        soc_model=QcomChipset.SM8750,
        strict=False,   # QNN backend constraint validation をスキップ
    )
    quantizer.set_default_quant_config(
        quant_dtype=quant_dtype,
        is_qat=False,
        is_linear_per_channel=True,
        act_observer=MinMaxObserver,
    )
    prepared = prepare_pt2e(exp, quantizer)
    obs = [n for n in prepared.graph.nodes
           if "observer" in str(n.target) or "fake_quant" in str(n.target)]
    print(f"  → observer/fake_quant: {len(obs)}")

    # STEP 3
    print("STEP 3: calibration (dummy 3 pass) ...")
    with torch.no_grad():
        for _ in range(3): prepared(torch.randint(0, 256, (1, 16)))
    print("  → done")

    # STEP 4
    print("STEP 4: convert_pt2e(fold_quantize=False) ...")
    quantized = convert_pt2e(prepared, fold_quantize=False)
    ns1  = list(quantized.graph.nodes)
    q_n  = [n for n in ns1 if "quantize_per"   in str(n.target)]
    dq_n = [n for n in ns1 if "dequantize_per" in str(n.target)]
    lin_n= [n for n in ns1 if n.target == torch.ops.aten.linear.default]
    print(f"  → total nodes      : {len(ns1)}")
    print(f"  → quantize nodes   : {len(q_n)}")
    print(f"  → dequantize nodes : {len(dq_n)}")
    print(f"  → linear nodes     : {len(lin_n)}")

    # activation 量子化確認
    act_q = [n for n in q_n
             if any(u.target == torch.ops.aten.linear.default for u in n.users)]
    print(f"\n  act-quant → linear: {len(act_q)}")
    if act_q:
        print(f"  ✓ LLM ACTIVATIONS QUANTIZED ({precision_name})")
        for n in act_q[:4]:
            dtype_arg = n.args[1] if len(n.args) > 1 else "?"
            print(f"    {n.name}  dtype={dtype_arg}")
    else:
        print("  ✗ activation quant への linear 接続が確認できなかった")

    # STEP 5
    print("\nSTEP 5: 推論確認 ...")
    with torch.no_grad():
        out_f = model(ids); out_q = quantized(ids)
    diff = (out_f - out_q).abs().mean().item()
    print(f"  mean|float-quant| = {diff:.4f}")
    print(f"\nSummary [{precision_name}]: {len(ns0)} float → {len(ns1)} QDQ")
    print(f"  quant={len(q_n)}, dequant={len(dq_n)}, act-quant-to-linear={len(act_q)}")
    print("STEP 6: to_edge_transform_and_lower_to_qnn / .pte → 省略\n")


run_llm("A8W8  (int8  act + int8 wt)", QuantDtype.use_8a8w)
run_llm("A16W8 (int16 act + int8 wt)", QuantDtype.use_16a8w)
print("完了")
