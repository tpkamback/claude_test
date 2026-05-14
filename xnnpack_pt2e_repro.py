#!/usr/bin/env python3
"""
executorch examples/xnnpack/quantization/example.py の PT2E フロー再現。
XNNPACK A8W8 static quantization (activation + weight 両方 int8)。
.pte 書き出しは省略。
"""
import warnings; warnings.filterwarnings("ignore")
import torch, torchvision.models as tvm
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

print(f"torch   : {torch.__version__}")
import torchao; print(f"torchao : {torchao.__version__}")
print()

def run(model_name, model, ex):
    model.eval()
    print("=" * 60)
    print(f"[{model_name}] XNNPACK PT2E A8W8 static quantization")
    print("=" * 60)

    # STEP 1
    print("STEP 1: torch.export.export(strict=True) ...")
    try:   exp = torch.export.export(model, ex, strict=True).module()
    except: exp = torch.export.export(model, ex, strict=False).module()
    ns0 = list(exp.graph.nodes)
    lin0  = sum(1 for n in ns0 if n.target == torch.ops.aten.linear.default)
    conv0 = sum(1 for n in ns0 if n.target == torch.ops.aten.convolution.default)
    print(f"  → {len(ns0)} nodes  ({conv0} conv, {lin0} linear)")

    # STEP 2
    print("STEP 2: prepare_pt2e (XNNPACKQuantizer, per-ch symmetric int8) ...")
    q = XNNPACKQuantizer()
    q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prep = prepare_pt2e(exp, q)
    obs = sum(1 for n in prep.graph.nodes
              if "observer" in str(n.target) or "fake_quant" in str(n.target))
    print(f"  → observer/fake_quant: {obs}")

    # STEP 3
    print("STEP 3: calibration (5 random samples) ...")
    with torch.no_grad():
        for _ in range(5): prep(*[torch.randn_like(x) for x in ex])
    print("  → done")

    # STEP 4
    print("STEP 4: convert_pt2e(fold_quantize=False) ...")
    conv = convert_pt2e(prep, fold_quantize=False)
    ns1  = list(conv.graph.nodes)
    q_n  = [n for n in ns1 if "quantize_per"   in str(n.target)]
    dq_n = [n for n in ns1 if "dequantize_per" in str(n.target)]
    print(f"  → total nodes     : {len(ns1)}")
    print(f"  → quantize nodes  : {len(q_n)}")
    print(f"  → dequantize nodes: {len(dq_n)}")

    ops = {torch.ops.aten.linear.default, torch.ops.aten.convolution.default}
    act_q = [n for n in q_n if any(u.target in ops for u in n.users)]
    print(f"  → act-quant → linear/conv: {len(act_q)}")
    if act_q:
        print("  ✓ ACTIVATIONS QUANTIZED (int8)")
        for n in act_q[:3]:
            print(f"    {n.name} → {[u.target for u in n.users]}")

    # STEP 5
    print("STEP 5: 推論確認 ...")
    with torch.no_grad():
        out_f = model(*ex); out_q = conv(*ex)
    diff = (out_f - out_q).abs().mean().item()
    print(f"  mean|float-quant| = {diff:.4f}")
    print(f"\nSummary: {len(ns0)} float → {len(ns1)} QDQ  (act-quant={len(act_q)})")
    print("STEP 6: export_to_edge / .pte → 省略\n")

run("MobileNetV2", tvm.mobilenet_v2(weights=None), (torch.randn(1,3,224,224),))
run("ResNet50",    tvm.resnet50(weights=None),       (torch.randn(1,3,224,224),))
print("完了")
