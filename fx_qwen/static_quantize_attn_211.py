"""
static_quantize_attn_211.py — Attention + RMSNorm + SiLU + Softmax + RoPE の
                               Static Fake Quantization（torch 2.11 / forward hook 方式）

量子化対象:
  - Attention matmul  （QK^T, AV の両 matmul 入力）
  - RMSNorm           （Qwen2RMSNorm の入力）
  - SiLU              （Qwen2MLP の act_fn 入力）
  - Softmax           （attn_weights → 確率変換の入力）
  - RoPE              （apply_rotary_pos_emb の q/k 入力）

量子化方式: Fake Quantization（対称 int8）
  入力テンソルを int8 に量子化 → float に逆量子化 → 演算を実行
  calibration フェーズで per-tensor abs-max から静的 scale を収集する。

Usage:
    python static_quantize_attn_211.py
    python static_quantize_attn_211.py --model ../eval_ppl/models/qwen2-tiny \\
        --seq_len 16 --n_calib 4
"""

import argparse
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake Quantization ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

INT8_MAX = 127.0


def compute_scale(tensor: torch.Tensor) -> torch.Tensor:
    """
    対称量子化の per-tensor scale を計算する。
      scale = abs_max / INT8_MAX
    ゼロ除算を避けるため eps を加算する。
    """
    return tensor.abs().max().clamp(min=1e-8) / INT8_MAX


def fake_quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Fake Quantization:
      q = clamp(round(x / scale), -127, 127)   # quantize → int8 相当
      x̂ = q * scale                             # dequantize → float
    勾配は straight-through estimator で通過する（推論専用なので不要だが一貫性のため）。
    """
    q = torch.clamp(torch.round(tensor / scale), -INT8_MAX, INT8_MAX)
    return q * scale


# ─────────────────────────────────────────────────────────────────────────────
# キャリブレーション統計コレクター
# ─────────────────────────────────────────────────────────────────────────────

class CalibStats:
    """
    キャリブレーション中に abs_max を running-max で蓄積し、
    最終的な静的 scale を保持する。
    """

    def __init__(self, name: str, kind: str):
        self.name = name          # モジュール名 or 識別子
        self.kind = kind          # "rmsnorm" / "silu" / "softmax" / "rope_q" / "rope_k" / "attn_qk" / "attn_av"
        self.running_max: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.n_samples = 0

    def update(self, tensor: torch.Tensor):
        val = tensor.detach().abs().max()
        if self.running_max is None:
            self.running_max = val.clone()
        else:
            self.running_max = torch.max(self.running_max, val)
        self.n_samples += 1

    def finalize(self):
        if self.running_max is not None:
            self.scale = self.running_max.clamp(min=1e-8) / INT8_MAX
        else:
            self.scale = torch.tensor(1.0 / INT8_MAX)


# ─────────────────────────────────────────────────────────────────────────────
# Hook ハンドラ群
# ─────────────────────────────────────────────────────────────────────────────

def make_rmsnorm_hook(stats: CalibStats):
    """
    Qwen2RMSNorm の forward hook。
    入力 hidden_states の abs-max を蓄積する（calibration フェーズのみ統計収集）。
    """
    def hook(module, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        if x is not None:
            stats.update(x)
        return None  # 入力変更なし（calibration のみ）
    return hook


def make_rmsnorm_fake_quant_hook(stats: CalibStats):
    """
    Qwen2RMSNorm の forward pre-hook（量子化フェーズ）。
    入力を fake quantize して返す。
    """
    def hook(module, args, kwargs):
        x = args[0] if args else kwargs.get("hidden_states")
        if x is not None and stats.scale is not None:
            x_fq = fake_quantize(x, stats.scale.to(x.device))
            if args:
                return (x_fq, *args[1:]), kwargs
            else:
                kwargs["hidden_states"] = x_fq
                return args, kwargs
        return None
    return hook


def make_silu_hook(stats: CalibStats):
    """
    Qwen2MLP.act_fn（SiLU）を monkey-patch するためのラッパークラス用統計収集 hook。
    act_fn は nn.Module ではなく callable なため、MLP 自体の hook から対処する。
    """
    def hook(module, args, kwargs):
        # MLP forward の入力 x の統計を収集
        x = args[0] if args else kwargs.get("x")
        if x is not None:
            stats.update(x)
        return None
    return hook


def make_mlp_fake_quant_hook(stats: CalibStats):
    """
    Qwen2MLP の forward pre-hook（量子化フェーズ）。
    gate_proj に渡す入力（= act_fn の入力に対応）を fake quantize する。
    SiLU の実際の入力は gate_proj(x) だが、ここでは act_fn 直前の入力 x を量子化する。
    """
    def hook(module, args, kwargs):
        x = args[0] if args else kwargs.get("x")
        if x is not None and stats.scale is not None:
            x_fq = fake_quantize(x, stats.scale.to(x.device))
            if args:
                return (x_fq, *args[1:]), kwargs
            else:
                kwargs["x"] = x_fq
                return args, kwargs
        return None
    return hook


class FakeQuantSoftmax(nn.Module):
    """
    Softmax の入力（attn_weights logits）を fake quantize してから
    通常の softmax を適用するラッパーモジュール。
    calibration フェーズでは統計のみ収集。
    """

    def __init__(self, stats: CalibStats):
        super().__init__()
        self.stats = stats
        self.calibrating = True

    def forward(self, attn_weights: torch.Tensor, dim: int = -1, dtype=None) -> torch.Tensor:
        if self.calibrating:
            self.stats.update(attn_weights)
        elif self.stats.scale is not None:
            attn_weights = fake_quantize(attn_weights, self.stats.scale.to(attn_weights.device))
        return F.softmax(attn_weights, dim=dim, dtype=dtype)


class FakeQuantRoPE:
    """
    apply_rotary_pos_emb を差し替えるための callable ラッパー。
    q, k の両テンソルそれぞれの abs-max を収集し、fake quantize する。
    """

    def __init__(self, stats_q: CalibStats, stats_k: CalibStats):
        self.stats_q = stats_q
        self.stats_k = stats_k
        self.calibrating = True

    def __call__(self, q, k, cos, sin, unsqueeze_dim=1):
        if self.calibrating:
            self.stats_q.update(q)
            self.stats_k.update(k)
        else:
            if self.stats_q.scale is not None:
                q = fake_quantize(q, self.stats_q.scale.to(q.device))
            if self.stats_k.scale is not None:
                k = fake_quantize(k, self.stats_k.scale.to(k.device))
        # 元の apply_rotary_pos_emb ロジックをインライン実行
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class FakeQuantAttnForward:
    """
    eager_attention_forward を差し替えるための callable ラッパー。
    QK matmul の入力（query, key_states）と
    AV matmul の入力（attn_weights, value_states）をそれぞれ fake quantize する。
    Softmax は FakeQuantSoftmax で処理済みなので attn_weights はすでに fake quantized。
    """

    def __init__(self, stats_qk: CalibStats, stats_av: CalibStats,
                 softmax_wrapper: FakeQuantSoftmax):
        self.stats_qk = stats_qk
        self.stats_av = stats_av
        self.softmax_wrapper = softmax_wrapper
        self.calibrating = True

    def __call__(self, module, query, key, value, attention_mask, scaling,
                 dropout=0.0, **kwargs):
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        # QK matmul の入力を fake quantize（calibration 時は統計収集のみ）
        if self.calibrating:
            self.stats_qk.update(query)
            self.stats_qk.update(key_states)
        else:
            if self.stats_qk.scale is not None:
                scale_qk = self.stats_qk.scale.to(query.device)
                query = fake_quantize(query, scale_qk)
                key_states = fake_quantize(key_states, scale_qk)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax（FakeQuantSoftmax で fake quantize 済み）
        attn_weights = self.softmax_wrapper(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

        # AV matmul の入力を fake quantize
        if self.calibrating:
            self.stats_av.update(attn_weights)
            self.stats_av.update(value_states)
        else:
            if self.stats_av.scale is not None:
                scale_av = self.stats_av.scale.to(attn_weights.device)
                attn_weights = fake_quantize(attn_weights, scale_av)
                value_states = fake_quantize(value_states, scale_av)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# メイン量子化インストルメンテーション
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantInstrument:
    """
    モデル全体に fake quantization フック / ラッパーを設置し、
    calibration → finalize → 量子化推論の状態遷移を管理する。
    """

    def __init__(self):
        self.all_stats: Dict[str, CalibStats] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        # 置き換えた関数を後で参照するためのリスト
        self._softmax_wrappers: List[FakeQuantSoftmax] = []
        self._rope_wrappers: List[FakeQuantRoPE] = []
        self._attn_wrappers: List[FakeQuantAttnForward] = []

    # ── 統計登録ヘルパー ─────────────────────────────────────────────────────

    def _new_stats(self, name: str, kind: str) -> CalibStats:
        stats = CalibStats(name, kind)
        self.all_stats[f"{name}::{kind}"] = stats
        return stats

    # ── インストルメンテーション ─────────────────────────────────────────────

    def instrument(self, model: nn.Module):
        """
        モデルの各モジュールに対してキャリブレーション統計収集 hook を設置し、
        FakeQuant ラッパーを差し込む。
        """
        for name, mod in model.named_modules():

            # ── RMSNorm ────────────────────────────────────────────────────
            if isinstance(mod, Qwen2RMSNorm):
                stats = self._new_stats(name, "rmsnorm")
                h = mod.register_forward_pre_hook(
                    make_rmsnorm_hook(stats), with_kwargs=True
                )
                self._hooks.append(h)

            # ── SiLU（MLP 単位で入力を量子化）────────────────────────────
            elif isinstance(mod, Qwen2MLP):
                stats = self._new_stats(name, "silu")
                h = mod.register_forward_pre_hook(
                    make_silu_hook(stats), with_kwargs=True
                )
                self._hooks.append(h)

            # ── Attention（Softmax / RoPE / matmul を内包） ───────────────
            elif isinstance(mod, Qwen2Attention):
                # Softmax wrapper
                stats_sm = self._new_stats(name, "softmax")
                softmax_wrapper = FakeQuantSoftmax(stats_sm)
                self._softmax_wrappers.append(softmax_wrapper)

                # RoPE wrapper
                stats_rq = self._new_stats(name, "rope_q")
                stats_rk = self._new_stats(name, "rope_k")
                rope_wrapper = FakeQuantRoPE(stats_rq, stats_rk)
                self._rope_wrappers.append(rope_wrapper)

                # Attn matmul wrapper
                stats_qk = self._new_stats(name, "attn_qk")
                stats_av = self._new_stats(name, "attn_av")
                attn_wrapper = FakeQuantAttnForward(stats_qk, stats_av, softmax_wrapper)
                self._attn_wrappers.append(attn_wrapper)

                # Attention モジュールに wrapper を注入
                # （forward 内で参照されるように属性として保持）
                mod._fq_rope = rope_wrapper
                mod._fq_attn = attn_wrapper

                # forward を monkey-patch
                _patch_attention_forward(mod)

        n = {
            "rmsnorm": sum(1 for s in self.all_stats.values() if s.kind == "rmsnorm"),
            "silu":    sum(1 for s in self.all_stats.values() if s.kind == "silu"),
            "softmax": sum(1 for s in self.all_stats.values() if s.kind == "softmax"),
            "rope":    sum(1 for s in self.all_stats.values() if s.kind in ("rope_q", "rope_k")) // 2,
            "attn_matmul": sum(1 for s in self.all_stats.values() if s.kind == "attn_qk"),
        }
        return n

    # ── calibration 終了後に scale を確定 ────────────────────────────────────

    def finalize_calibration(self):
        """
        全 CalibStats を finalize（scale を確定）し、
        全 wrapper / hook を「量子化モード」に切り替える。
        """
        for stats in self.all_stats.values():
            stats.finalize()

        # calibrating フラグを False に切り替え
        for sw in self._softmax_wrappers:
            sw.calibrating = False
        for rw in self._rope_wrappers:
            rw.calibrating = False
        for aw in self._attn_wrappers:
            aw.calibrating = False

        # RMSNorm / SiLU hook を fake-quant hook に差し替え
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # 収集完了後、fake quant hook を再登録
        for key, stats in self.all_stats.items():
            mod_name = key.split("::")[0]
            kind = stats.kind
            mod = _get_module(self._model_ref, mod_name)
            if mod is None:
                continue
            if kind == "rmsnorm":
                h = mod.register_forward_pre_hook(
                    make_rmsnorm_fake_quant_hook(stats), with_kwargs=True
                )
                self._hooks.append(h)
            elif kind == "silu":
                h = mod.register_forward_pre_hook(
                    make_mlp_fake_quant_hook(stats), with_kwargs=True
                )
                self._hooks.append(h)

    def set_model_ref(self, model: nn.Module):
        self._model_ref = model

    def scale_summary(self) -> Dict[str, Dict]:
        """各種量子化の scale 統計をまとめて返す。"""
        summary = defaultdict(list)
        for key, stats in self.all_stats.items():
            if stats.scale is not None:
                summary[stats.kind].append(stats.scale.item())
        result = {}
        for kind, scales in summary.items():
            arr = torch.tensor(scales)
            result[kind] = {
                "count": len(scales),
                "min": arr.min().item(),
                "max": arr.max().item(),
                "mean": arr.mean().item(),
            }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Attention forward monkey-patch
# ─────────────────────────────────────────────────────────────────────────────

def _patch_attention_forward(attn_mod: Qwen2Attention):
    """
    Qwen2Attention.forward を差し替えて、
    _fq_rope / _fq_attn ラッパーを呼ぶようにする。
    """
    import types
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention as _Qwen2Attn
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        # RoPE を fake-quant ラッパー経由で適用
        query_states, key_states = self._fq_rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # attention matmul + softmax を fake-quant ラッパー経由で実行
        attn_output, attn_weights = self._fq_attn(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_mod.forward = types.MethodType(patched_forward, attn_mod)


def _get_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    """ドット区切りのモジュール名からサブモジュールを返す。"""
    parts = name.split(".")
    mod = model
    for p in parts:
        if p == "":
            return mod
        if not hasattr(mod, p):
            return None
        mod = getattr(mod, p)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../eval_ppl/models/qwen2-tiny")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length (fixed for static quant)")
    parser.add_argument("--n_calib", type=int, default=4,
                        help="Number of calibration samples")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── [1/4] Load ────────────────────────────────────────────────────────────
    print(f"[1/4] Loading: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params")

    # ── [2/4] Instrument ──────────────────────────────────────────────────────
    print("[2/4] Instrumenting fake-quant hooks ...")
    instrument = FakeQuantInstrument()
    instrument.set_model_ref(model)
    counts = instrument.instrument(model)
    print(f"      RMSNorm      : {counts['rmsnorm']:3d} layers")
    print(f"      SiLU (MLP)   : {counts['silu']:3d} layers")
    print(f"      Softmax      : {counts['softmax']:3d} layers")
    print(f"      RoPE         : {counts['rope']:3d} layers")
    print(f"      Attn matmul  : {counts['attn_matmul']:3d} layers")

    # ── [3/4] Calibration ─────────────────────────────────────────────────────
    print(f"[3/4] Calibrating ({args.n_calib} samples, seq_len={args.seq_len}) ...")
    with torch.no_grad():
        for i in range(args.n_calib):
            dummy = torch.randint(0, 100, (1, args.seq_len))
            model(dummy, use_cache=False)
            print(f"      sample {i + 1}/{args.n_calib} done")

    instrument.finalize_calibration()
    print("      Calibration finalized. Scale summary:")
    summary = instrument.scale_summary()
    kind_labels = {
        "rmsnorm":  "RMSNorm input",
        "silu":     "SiLU input (MLP x)",
        "softmax":  "Softmax input (attn logits)",
        "rope_q":   "RoPE query input",
        "rope_k":   "RoPE key input",
        "attn_qk":  "Attn QK matmul input",
        "attn_av":  "Attn AV matmul input",
    }
    for kind, info in summary.items():
        label = kind_labels.get(kind, kind)
        print(f"        {label:30s} count={info['count']:3d}  "
              f"scale=[{info['min']:.4f}, {info['max']:.4f}]  mean={info['mean']:.4f}")

    # ── [4/4] Fake-quant inference ────────────────────────────────────────────
    print("[4/4] Running fake-quant inference ...")
    dummy = torch.zeros(1, args.seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy, use_cache=False)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Logits : {out.logits.shape}")
    print(f"  Time   : {elapsed:.1f} ms")

    total_quant_layers = sum(counts.values())
    print(f"\n  Quantized layer summary:")
    print(f"    RMSNorm      : {counts['rmsnorm']:3d}")
    print(f"    SiLU (MLP)   : {counts['silu']:3d}")
    print(f"    Softmax      : {counts['softmax']:3d}")
    print(f"    RoPE         : {counts['rope']:3d}")
    print(f"    Attn matmul  : {counts['attn_matmul']:3d}")
    print(f"    ─────────────────")
    print(f"    Total        : {total_quant_layers:3d}")

    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
