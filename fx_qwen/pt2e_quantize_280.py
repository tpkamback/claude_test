"""
pt2e_quantize_280.py — PT2E QDQ 量子化の最小実装 (torch 2.8)

=== 動作確認済み環境 ===
  torch:        2.8.0+cu128
  transformers: 5.8.0
  Python:       3.11

=== torch 2.8 での PT2E import パス ===
  from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
  from torch.export import export_for_training  # ← 2.8 での推奨 capture 方法

=== torch 2.2 → 2.8 の変更点 ===
  capture_pre_autograd_graph (torch._export):
    - torch 2.2.1: 利用可能
    - torch 2.8.0: 削除済み (ImportError が発生する)
  export_for_training (torch.export):
    - torch 2.8.0: capture_pre_autograd_graph の後継
    - aten.linear.default を保持したまま capture できる ← 重要
    - .module() を呼んで GraphModule を取り出す必要がある
  torch.export.export:
    - 2.8 でも利用可能だが、strict モードで失敗するモデルあり
    - export_for_training のほうが量子化パイプラインには適している

=== aten.linear が保持されるか ===
  torch.export.export        → OK (aten.linear.default 保持確認)
  export_for_training        → OK (aten.linear.default 保持確認)

=== ハマりポイント ===
  1. transformers >= 5.x で dtype= 引数が変更
       旧: torch_dtype=torch.float32  (deprecated警告が出る)
       新: dtype=torch.float32
  2. KV cache を use_cache=False で無効化しないと export が失敗する
       (DynamicCache が pytree 未登録のため)
  3. Qwen-tiny (6.08M params) は lm_head のみ nn.Linear
       attention 層はカスタム実装のため export グラフに aten.linear は 1 つのみ
       → QDQ ノードは 5 個 (input dequant + weight q/dequant + output q/dequant)
  4. MinMaxObserver は Long (int64) テンソルに適用するとエラーになる
       input_qspec_map で Linear の入力のみ指定することで回避できる
  5. torch 2.8 では torch.ao.quantization に DeprecationWarning が出る
       "will be removed in 2.10" → 2.10 以降は torchao の pt2e API へ移行が必要
       torchao: https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e

=== 実行方法 ===
  python pt2e_quantize_280.py
  python pt2e_quantize_280.py --model ../eval_ppl/models/qwen-tiny
  python pt2e_quantize_280.py --seq_len 32
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM

# ── PT2E API (torch 2.8) ──────────────────────────────────────────────────────
# torch 2.8 での capture: export_for_training を使う
# torch._export.capture_pre_autograd_graph は 2.8 で削除されている
from torch.export import export_for_training
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer import Quantizer


# ─────────────────────────────────────────────────────────────────────────────
# Quantizer 定義: aten.linear.default に per-tensor symmetric int8 を適用
# ─────────────────────────────────────────────────────────────────────────────

class LinearInt8Quantizer(Quantizer):
    """
    aten.linear.default ノードに対して per-tensor symmetric int8 の
    QDQ (Quantize-Dequantize) アノテーションを付与する最小 Quantizer。

    XNNPACKQuantizer のような完全実装と異なり、aten.linear のみを対象とする。
    カスタム Quantizer の実装パターンの参考として使用できる。
    """

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.ao.quantization.quantizer import QuantizationAnnotation, QuantizationSpec
        from torch.ao.quantization.observer import MinMaxObserver

        act_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_symmetric,
            observer_or_fake_quant_ctr=MinMaxObserver,
        )

        annotated = 0
        for node in model.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        node.args[0]: act_spec,  # activation
                        node.args[1]: act_spec,  # weight
                    },
                    output_qspec=act_spec,
                )
                annotated += 1

        print(f"      annotated {annotated} aten.linear node(s)")
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ラッパーモジュール
# ─────────────────────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    """
    KV キャッシュを無効化し logits のみ返すラッパー。
    DynamicCache が pytree 未登録のため export 時にエラーになる問題を回避。
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False).logits


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: ロード
# ─────────────────────────────────────────────────────────────────────────────

def load(model_id: str, seq_len: int):
    print(f"[1/5] Loading: {model_id}")
    # transformers 5.x では dtype= を使う (torch_dtype= は deprecated)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    linears = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
    print(f"      {params:.2f}M params, {linears} nn.Linear module(s)")
    wrapped = NoCacheWrapper(model)
    example = (torch.zeros(1, seq_len, dtype=torch.long),)
    return wrapped, example


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: エクスポート (export_for_training)
# ─────────────────────────────────────────────────────────────────────────────

def export(model: torch.nn.Module, example: tuple) -> torch.fx.GraphModule:
    """
    torch 2.8 での PT2E 推奨キャプチャ方法。

    export_for_training は capture_pre_autograd_graph (2.2.x) の後継。
    aten.linear.default を保持したまま capture するため、
    Quantizer の annotate でパターンマッチできる。

    .module() を呼ぶことで GraphModule を取り出す。
    """
    print("[2/5] export_for_training ...")
    gm = export_for_training(model, example).module()

    ops = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    linear_ops = sorted(o for o in ops if any(k in o for k in ["linear", "mm", "addmm"]))
    total = len(list(gm.graph.nodes))
    print(f"      linear-related ops: {linear_ops}")
    print(f"      total nodes: {total}")
    return gm


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: prepare_pt2e
# ─────────────────────────────────────────────────────────────────────────────

def prepare(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    prepare_pt2e で observer (fake quantize) を挿入する。
    LinearInt8Quantizer は aten.linear ノードのみをアノテーションするため、
    Long 型テンソルへの observer 挿入エラーを避けられる。
    """
    print("[3/5] prepare_pt2e ...")
    quantizer = LinearInt8Quantizer()
    prepared = prepare_pt2e(gm, quantizer)
    obs = sum(1 for n in prepared.graph.nodes if "activation_post_process" in n.name)
    print(f"      observer nodes inserted: {obs}")
    return prepared


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: キャリブレーション
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(prepared: torch.fx.GraphModule, seq_len: int) -> None:
    """
    代表的な入力データで forward を呼び observer に統計を収集させる。
    本実装では 1 サンプルのみ。本番では実データを複数サンプル使用する。
    """
    print("[4/5] calibrating ...")
    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    with torch.no_grad():
        prepared(dummy)
    print("      calibration done (1 sample)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: convert_pt2e + 推論確認
# ─────────────────────────────────────────────────────────────────────────────

def convert_and_verify(prepared: torch.fx.GraphModule, seq_len: int) -> torch.fx.GraphModule:
    """
    observer を QDQ (quantize_per_tensor / dequantize_per_tensor) ノードに変換する。
    変換後に推論を実行して動作確認する。
    """
    print("[5/5] convert_pt2e + inference ...")
    quantized = convert_pt2e(prepared)

    qdq_nodes = [
        n for n in quantized.graph.nodes
        if "quantize" in n.name or "dequantize" in n.name
    ]
    print(f"      QDQ nodes: {len(qdq_nodes)}")
    print(f"      QDQ node names: {[n.name for n in qdq_nodes]}")

    dummy = torch.zeros(1, seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = quantized(dummy)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"      logits: {out.shape}  |  inference: {elapsed:.1f} ms")
    return quantized


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PT2E QDQ quantization on torch 2.8 (minimal example)"
    )
    parser.add_argument("--model", default="../eval_ppl/models/qwen-tiny")
    parser.add_argument("--seq_len", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"torch: {torch.__version__}")

    wrapped, example = load(args.model, args.seq_len)

    try:
        gm = export(wrapped, example)
    except Exception as e:
        print(f"[ERROR] export: {type(e).__name__}: {e}")
        return

    try:
        prepared = prepare(gm)
    except Exception as e:
        print(f"[ERROR] prepare_pt2e: {type(e).__name__}: {e}")
        return

    try:
        calibrate(prepared, args.seq_len)
    except Exception as e:
        print(f"[ERROR] calibrate: {type(e).__name__}: {e}")
        return

    try:
        convert_and_verify(prepared, args.seq_len)
    except Exception as e:
        print(f"[ERROR] convert_pt2e: {type(e).__name__}: {e}")
        return

    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
