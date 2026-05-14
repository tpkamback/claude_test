"""
MediaTek executorch example の PT2E 量子化フロー再現スクリプト
https://github.com/pytorch/executorch/tree/main/examples/mediatek

【オリジナルの流れ】
  torch.export.export(strict=True)
  → prepare_pt2e(NeuropilotQuantizer)   ← MediaTek 独自 (SDK 必要)
  → calibration loop
  → convert_pt2e(fold_quantize=False)
  → NeuropilotPartitioner / to_edge_transform_and_lower
  → .pte ファイル書き出し               ← MediaTek NPU 用バイナリ

【このスクリプト】
  NeuropilotQuantizer → XNNPACKQuantizer で代替
  .pte 生成は省略 (MediaTek SDK 不要)
  ResNet50 / MobileNetV2 で動作確認
"""

import torch
import torchvision.models as tv

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.testing.pt2e._xnnpack_quantizer import XNNPACKQuantizer
from torchao.testing.pt2e._xnnpack_quantizer_utils import (
    QuantizationConfig, QuantizationSpec,
)
from torchao.quantization.pt2e.observer import MinMaxObserver

def _sym_int8_spec():
    return QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128, quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=MinMaxObserver,
    )

def get_a8w8_config():
    spec = _sym_int8_spec()
    return QuantizationConfig(
        input_activation=spec,
        output_activation=spec,
        weight=spec,
        bias=None,
    )

print(f"torch    : {torch.__version__}")
import torchao; print(f"torchao  : {torchao.__version__}")
print()


def run_pt2e(model_name: str, model: torch.nn.Module, inputs: tuple, n_calib: int = 5):
    print(f"{'='*60}")
    print(f"[{model_name}] PT2E 量子化")
    print(f"{'='*60}")

    model.eval()

    # ── STEP 1: torch.export.export ────────────────────────────────
    # オリジナル: torch.export.export(model, inputs, strict=True).module()
    print("STEP 1: torch.export.export(strict=False) ...")
    ep = torch.export.export(model, inputs, strict=False)
    captured = ep.module()
    float_nodes = len(list(captured.graph.nodes))
    print(f"  → {float_nodes} nodes (float graph)")

    # ── STEP 2: prepare_pt2e ───────────────────────────────────────
    # オリジナル: NeuropilotQuantizer().setup_precision(Precision.A8W8)
    # ここでは XNNPACKQuantizer (per-tensor symmetric int8, 同等設定)
    print("STEP 2: prepare_pt2e (XNNPACKQuantizer / A8W8 相当) ...")
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_a8w8_config())
    annotated = prepare_pt2e(captured, quantizer)
    obs_nodes = [n for n in annotated.graph.nodes if "observer" in str(n.target).lower()
                 or "fake_quant" in str(n.target).lower()]
    print(f"  → observer/fake_quant nodes: {len(obs_nodes)}")

    # ── STEP 3: キャリブレーション ─────────────────────────────────
    # オリジナル: for data in dataset: annotated_model(*data)
    print(f"STEP 3: calibration ({n_calib} samples) ...")
    for i in range(n_calib):
        dummy = tuple(torch.randn_like(x) for x in inputs)
        annotated(*dummy)
    print(f"  → done")

    # ── STEP 4: convert_pt2e ───────────────────────────────────────
    # オリジナル: convert_pt2e(annotated_model, fold_quantize=False)
    print("STEP 4: convert_pt2e(fold_quantize=False) ...")
    quantized = convert_pt2e(annotated, fold_quantize=False)
    all_nodes = list(quantized.graph.nodes)
    qdq_nodes  = [n for n in all_nodes
                  if "quantize_per_tensor" in str(n.target)
                  or "dequantize_per_tensor" in str(n.target)]
    print(f"  → total nodes : {len(all_nodes)}")
    print(f"  → QDQ nodes   : {len(qdq_nodes)}")

    # ── STEP 5: 推論確認 ────────────────────────────────────────────
    print("STEP 5: 推論確認 ...")
    with torch.no_grad():
        out_float = model(*inputs)
        out_quant = quantized(*inputs)
    print(f"  float output shape : {out_float.shape}")
    print(f"  quant output shape : {out_quant.shape}")
    diff = (out_float - out_quant).abs().mean().item()
    print(f"  mean |float - quant|: {diff:.4f}")

    # ── STEP 6 (省略): MediaTek edge 変換 ──────────────────────────
    # オリジナル:
    #   aten_dialect = torch.export.export(quantized_model, inputs, strict=True)
    #   edge_prog = to_edge_transform_and_lower(aten_dialect, partitioner=[NeuropilotPartitioner(...)])
    #   exec_prog = edge_prog.to_executorch(...)
    #   open("model.pte", "wb").write(exec_prog.buffer)
    print("STEP 6: MediaTek edge変換 → 省略 (SDK 不要部分はここまで)")
    print()


# ResNet50 (MediaTek 例と同一モデル)
run_pt2e(
    "ResNet50",
    tv.resnet50(weights=None),
    (torch.randn(1, 3, 224, 224),),
)

# MobileNetV2 (MediaTek 例と同一モデル)
run_pt2e(
    "MobileNetV2",
    tv.mobilenet_v2(weights=None),
    (torch.randn(1, 3, 224, 224),),
)

print("完了")
