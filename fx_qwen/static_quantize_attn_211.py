"""
static_quantize_attn_211.py — Static Quantization (Linear + Attention matmul) on torch 2.11 via torchao

Pipeline:
  1. Loading model
  2. Insert Linear observers (from static_quantize_211.py)
  3. Insert Attention observers (register_forward_pre_hook on Qwen2Attention)
  4. Calibrate (collect min/max for Linear activations and Q/K/attn_weights/V tensors)
  5. Apply quantization:
     - Linear: Int8StaticActivationInt8WeightConfig (169 layers)
     - Attention matmuls: monkey-patch each Qwen2Attention.forward to
       quantize Q, K before Q@K^T and attn_weights, V before attn_weights@V

Qwen2.5-0.5B uses _attn_implementation="sdpa" which delegates to
F.scaled_dot_product_attention. We replace each Qwen2Attention.forward with a
custom eager path that inserts quantize_affine / dequantize_affine around the
two matmuls.  Element-wise ops (RMSNorm, SiLU, Softmax, RoPE) stay in float32.

Usage:
    python static_quantize_attn_211.py
    python static_quantize_attn_211.py --model ../eval_ppl/models/Qwen2.5-0.5B-random --seq_len 16
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
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
# Step 3: Register calibration hooks on Qwen2Attention
# ─────────────────────────────────────────────────────────────────────────────

def insert_attn_observers(model: nn.Module):
    """
    Register a forward pre-hook (with_kwargs=True) on each Qwen2Attention.

    Qwen2Attention.forward receives all its arguments as **kwargs
    (hidden_states, position_embeddings, attention_mask, ...).
    The pre-hook captures hidden_states and position_embeddings,
    re-runs the Q/K/V projections and RoPE, then feeds
    Q, K_expanded, attn_weights (post-softmax), V_expanded
    into per-tensor AffineQuantizedMinMaxObservers.

    Returns:
        attn_observers: dict[layer_name -> {"q","k","a","v"} -> observer]
        handles:        list of removable hook handles
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
                    return  # safety guard

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
# Step 5: Monkey-patch Qwen2Attention.forward with quantized matmuls
# ─────────────────────────────────────────────────────────────────────────────

def _quant_dequant(t: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    """Quantize t to int8 (per-tensor) then dequantize back to float32."""
    bs = tuple(t.shape)
    t_int8 = quantize_affine(t, bs, scale, zp, torch.int8)
    return dequantize_affine(t_int8, bs, scale, zp, torch.int8, output_dtype=torch.float32)


def apply_attn_quant(model: nn.Module, attn_observers: dict) -> int:
    """
    For each Qwen2Attention module:
      1. Extract calibrated per-tensor scales from the observers.
      2. Replace module.forward with a patched version that performs:
           Q_int8  = quantize(Q,  scale_q)  -> dequant -> use in Q@K^T
           K_int8  = quantize(K,  scale_k)  -> dequant
           A_int8  = quantize(A,  scale_a)  -> dequant -> use in A@V
           V_int8  = quantize(V,  scale_v)  -> dequant
         where A = softmax(Q_dq @ K_dq^T / sqrt(head_dim)).
         Softmax and all element-wise ops remain in float32.

    Returns the number of patched layers.
    """
    n_patched = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, Qwen2Attention):
            continue
        if name not in attn_observers:
            continue

        obs = attn_observers[name]

        sq, zq = obs["q"].calculate_qparams()   # shape (1,1,1,1)
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

                # ── Q @ K^T with int8 quantization ────────────────────────
                q_dq  = _quant_dequant(q,     sq_, zq_)
                k_dq  = _quant_dequant(k_exp, sk_, zk_)

                attn_weights = torch.matmul(q_dq, k_dq.transpose(-2, -1)) * scaling

                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

                # ── attn_weights @ V with int8 quantization ───────────────
                a_dq  = _quant_dequant(attn_weights, sa_, za_)
                v_dq  = _quant_dequant(v_exp,        sv_, zv_)

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

    print(f"[1/5] Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      {params:.2f}M params, attn_impl={model.config._attn_implementation}")

    print("[2/5] Inserting Linear observers ...")
    linear_observers = insert_observers(model)
    print(f"      {len(linear_observers)} Linear layers instrumented")

    print("[3/5] Inserting Attention observers ...")
    attn_observers, attn_hooks = insert_attn_observers(model)
    print(f"      {len(attn_observers)} Attention layers instrumented")

    print(f"[4/5] Calibrating ({args.n_calib} samples, seq_len={args.seq_len}) ...")
    calibrate_linear(model, args.seq_len, args.n_calib)
    for h in attn_hooks:
        h.remove()
    remove_observers(model)
    print("      done")

    print("[5/5] Applying quantization (Linear + Attention matmul) ...")
    apply_static_quant(model, linear_observers)
    n_linear = len(linear_observers)

    n_attn = apply_attn_quant(model, attn_observers)

    print(f"\n  Linear layers  : {n_linear} quantized")
    print(f"  Attention layers: {n_attn} quantized (Q@K^T, attn@V)")

    # Inference check
    dummy = torch.zeros(1, args.seq_len, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(dummy, use_cache=False)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Logits : {out.logits.shape}")
    print(f"  Time   : {elapsed:.1f} ms")
    print(f"\nAll steps succeeded on torch {torch.__version__}")


if __name__ == "__main__":
    main()
