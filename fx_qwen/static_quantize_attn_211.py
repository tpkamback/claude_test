"""
static_quantize_attn_211.py — Full Static Quantization
  (nn.Linear + Attn matmul + RMSNorm + SiLU + Softmax + RoPE)

Quantization targets:
  1. nn.Linear (169 layers)     - static_quantize_211.py logic (torchao)
  2. Attention matmul           - Q@K^T and attn_weights@V (fake quant on inputs)
  3. RMSNorm (Qwen2RMSNorm)     - fake quant on input
  4. SiLU                       - fake quant on input
  5. Softmax                    - fake quant on attn_weights before softmax
  6. RoPE (RotaryEmbedding)     - fake quant on x input

Fake quantization = quantize_affine (int8) -> dequantize_affine (float) -> original op.

Usage:
    python static_quantize_attn_211.py
    python static_quantize_attn_211.py --model ../eval_ppl/models/Qwen2.5-0.5B-random --seq_len 16
"""

import argparse
import time
import types
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from torchao.quantization import (
    AffineQuantizedMinMaxObserver,
    MappingType,
)
from torchao.quantization.granularity import PerTensor
from torchao.quantization.quant_primitives import quantize_affine, dequantize_affine

# Import Linear-only pipeline from the existing module
from static_quantize_211 import (
    insert_observers,
    calibrate as calibrate_linear,
    remove_observers,
    apply_static_quant,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: per-tensor symmetric fake quantization via torchao API
# ─────────────────────────────────────────────────────────────────────────────

def fake_quantize_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric fake-quantization to int8 using torchao primitives."""
    orig_dtype = x.dtype
    x_f = x.float()
    amax = x_f.abs().max()
    if amax == 0:
        return x
    scale = (amax / 127.0).reshape(1)
    zero_point = torch.zeros(1, dtype=torch.int32, device=x.device)
    block_size = tuple(x_f.shape)
    q = quantize_affine(
        x_f,
        block_size=block_size,
        scale=scale,
        zero_point=zero_point,
        output_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
    )
    dq = dequantize_affine(
        q,
        block_size=block_size,
        scale=scale,
        zero_point=zero_point,
        input_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        output_dtype=torch.float32,
    )
    return dq.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Attention calibration observers
# ─────────────────────────────────────────────────────────────────────────────

def insert_attn_observers(model: nn.Module):
    """
    Register a forward pre-hook (with_kwargs=True) on each Qwen2Attention.
    Collects min/max for Q, K_expanded, attn_weights (post-softmax), V_expanded.
    Returns (attn_observers, handles).
    """
    attn_observers = {}
    handles = []

    for name, mod in model.named_modules():
        if not isinstance(mod, Qwen2Attention):
            continue

        obs_q = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC, torch.int8, granularity=PerTensor(), keepdim=True)
        obs_k = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC, torch.int8, granularity=PerTensor(), keepdim=True)
        obs_a = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC, torch.int8, granularity=PerTensor(), keepdim=True)
        obs_v = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC, torch.int8, granularity=PerTensor(), keepdim=True)
        attn_observers[name] = {"q": obs_q, "k": obs_k, "a": obs_a, "v": obs_v}

        def make_hook(oq, ok, oa, ov):
            def hook(module, args, kwargs):
                hidden_states = kwargs.get("hidden_states")
                position_embeddings = kwargs.get("position_embeddings")
                if hidden_states is None or position_embeddings is None:
                    return
                cos, sin = position_embeddings
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                with torch.no_grad():
                    q = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    k_exp = repeat_kv(k, module.num_key_value_groups)
                    v_exp = repeat_kv(v, module.num_key_value_groups)
                    scaling = module.head_dim ** -0.5
                    raw_scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scaling
                    attn_w = F.softmax(raw_scores, dim=-1, dtype=torch.float32).to(q.dtype)
                    oq(q)
                    ok(k_exp)
                    oa(attn_w)
                    ov(v_exp)
            return hook

        h = mod.register_forward_pre_hook(
            make_hook(obs_q, obs_k, obs_a, obs_v), with_kwargs=True)
        handles.append(h)

    return attn_observers, handles


# ─────────────────────────────────────────────────────────────────────────────
# Apply quantization: Attention matmuls
# ─────────────────────────────────────────────────────────────────────────────

def _quant_dequant(t: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    bs = tuple(t.shape)
    t_int8 = quantize_affine(t, bs, scale, zp, torch.int8)
    return dequantize_affine(t_int8, bs, scale, zp, torch.int8, output_dtype=torch.float32)


def apply_attn_quant(model: nn.Module, attn_observers: dict) -> int:
    """
    Patch each Qwen2Attention.forward to fake-quantize:
      Q and K before Q@K^T, attn_weights and V before attn@V.
    """
    n_patched = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, Qwen2Attention):
            continue
        if name not in attn_observers:
            continue

        obs = attn_observers[name]
        sq, zq = obs["q"].calculate_qparams()
        sk, zk = obs["k"].calculate_qparams()
        sa, za = obs["a"].calculate_qparams()
        sv, zv = obs["v"].calculate_qparams()
        zq = zq.to(torch.int32)
        zk = zk.to(torch.int32)
        za = za.to(torch.int32)
        zv = zv.to(torch.int32)

        def make_patched_forward(the_mod, sq_, zq_, sk_, zk_, sa_, za_, sv_, zv_):
            def patched_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                **kwargs,
            ):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, the_mod.head_dim)

                q = the_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                k = the_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                v = the_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, the_mod.layer_idx)

                k_exp = repeat_kv(k, the_mod.num_key_value_groups)
                v_exp = repeat_kv(v, the_mod.num_key_value_groups)

                scaling = the_mod.head_dim ** -0.5

                # Q @ K^T with int8 fake quantization
                q_dq = _quant_dequant(q,     sq_, zq_)
                k_dq = _quant_dequant(k_exp, sk_, zk_)
                attn_weights = torch.matmul(q_dq, k_dq.transpose(-2, -1)) * scaling

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                # Softmax: fake-quant attn_weights before softmax
                attn_weights = fake_quantize_int8(attn_weights)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

                # attn_weights @ V with int8 fake quantization
                a_dq = _quant_dequant(attn_weights, sa_, za_)
                v_dq = _quant_dequant(v_exp,        sv_, zv_)
                attn_output = torch.matmul(a_dq, v_dq)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = the_mod.o_proj(attn_output)
                return attn_output, None

            return patched_forward

        mod.forward = make_patched_forward(mod, sq, zq, sk, zk, sa, za, sv, zv)
        n_patched += 1

    return n_patched


# ─────────────────────────────────────────────────────────────────────────────
# Apply quantization: RMSNorm, SiLU, RoPE
# ─────────────────────────────────────────────────────────────────────────────

def apply_rmsnorm_quant(model: nn.Module) -> int:
    """Patch Qwen2RMSNorm.forward to fake-quantize input."""
    orig_forward = Qwen2RMSNorm.forward
    count = 0

    def quantized_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = fake_quantize_int8(hidden_states)
        return orig_forward(self, hidden_states)

    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2RMSNorm):
            mod.forward = types.MethodType(quantized_forward, mod)
            count += 1
    return count


def apply_silu_quant(model: nn.Module) -> int:
    """Patch SiLUActivation.forward to fake-quantize input."""
    count = 0

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fake_quantize_int8(x)
        return F.silu(x)

    for name, mod in model.named_modules():
        if "SiLU" in type(mod).__name__:
            mod.forward = types.MethodType(quantized_forward, mod)
            count += 1
    return count


def apply_rope_quant(model: nn.Module) -> int:
    """Patch Qwen2RotaryEmbedding.forward to fake-quantize x input."""
    count = 0

    def quantized_forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        x = fake_quantize_int8(x)
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    for name, mod in model.named_modules():
        if isinstance(mod, Qwen2RotaryEmbedding):
            mod.forward = types.MethodType(quantized_forward, mod)
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../eval_ppl/models/Qwen2.5-0.5B-random")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length (fixed for static quant)")
    parser.add_argument("--n_calib", type=int, default=4,
                        help="Number of calibration samples")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # [1/5] Load
    print(f"[1/5] Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params")

    # [2/5] Insert observers
    print("[2/5] Inserting observers (Linear / Attn / RMSNorm / SiLU / Softmax / RoPE) ...")
    linear_observers = insert_observers(model)
    attn_observers, attn_hooks = insert_attn_observers(model)
    rmsnorm_count = sum(1 for _, m in model.named_modules() if isinstance(m, Qwen2RMSNorm))
    silu_count = sum(1 for _, m in model.named_modules() if "SiLU" in type(m).__name__)
    rope_count = sum(1 for _, m in model.named_modules() if isinstance(m, Qwen2RotaryEmbedding))
    print(f"      Linear   : {len(linear_observers)} layers")
    print(f"      Attention: {len(attn_observers)} layers")
    print(f"      RMSNorm  : {rmsnorm_count} layers")
    print(f"      SiLU     : {silu_count} layers")
    print(f"      RoPE     : {rope_count} layers")

    # [3/5] Calibrate
    print(f"[3/5] Calibrating ({args.n_calib} samples, seq_len={args.seq_len}) ...")
    calibrate_linear(model, args.seq_len, args.n_calib)
    print("      done")

    # [4/5] Remove observers
    print("[4/5] Removing observers ...")
    for h in attn_hooks:
        h.remove()
    remove_observers(model)
    print("      done")

    # [5/5] Apply quantization
    print("[5/5] Applying quantization ...")
    apply_static_quant(model, linear_observers)
    n_attn = apply_attn_quant(model, attn_observers)
    n_rmsnorm = apply_rmsnorm_quant(model)
    n_silu = apply_silu_quant(model)
    n_rope = apply_rope_quant(model)
    # softmax is inside each attention block (one per block)
    n_softmax = n_attn

    print()
    print(f"  Linear     : {len(linear_observers)} layers")
    print(f"  Attn matmul:  {n_attn * 2} layers (Q@K^T, attn@V)")
    print(f"  RMSNorm    :  {n_rmsnorm} layers")
    print(f"  SiLU       :  {n_silu} layers")
    print(f"  Softmax    :  {n_softmax} layers")
    print(f"  RoPE       :  {n_rope} layers")

    # Inference check
    dummy = torch.zeros(1, args.seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy, use_cache=False)
    elapsed = (time.perf_counter() - t0) * 1000

    print()
    print(f"  Logits : {out.logits.shape}")
    print(f"  Time   : {elapsed:.1f} ms")
    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
