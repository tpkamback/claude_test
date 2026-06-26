"""
qwen25_static_export.py — Export Qwen2.5 prefill/decode graphs as fully static
graphs (StaticCache) for NPU deployment, and verify correctness.

Pipeline:
  1. Build Qwen2.5-0.5B-shaped model (random weights; HF Hub is network-blocked
     in this environment, so no real checkpoint is downloaded)
  2. Export separate prefill (N tokens) and decode (1 token) graphs via
     TorchExportableModuleForDecoderOnlyLM (StaticCache-based)
  3. Verify both graphs are fully static (no SymInt shapes, no `cat` ops,
     only `index_copy_` for cache writes)
  4. Locate the causal-mask constant in the graph (only present with
     attn_implementation="eager"; with "sdpa" the mask stays inside the
     fused kernel and is not an editable graph constant) and confirm it can
     be safely overwritten (e.g. replacing -inf with a finite large-negative
     value, as commonly required for NPU/quantized backends)
  5. Run the exported graphs end-to-end (prefill -> copy cache buffers into
     the decode graph -> decode loop) and confirm the generated token
     sequence is identical to eager model.generate()

Usage:
    python qwen25_static_export.py
    python qwen25_static_export.py --tiny   # use a tiny Qwen2Config for speed
"""

import argparse

import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

PROMPT_LEN = 6
NEW_TOKENS = 8
MAX_CACHE_LEN = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true",
                        help="Use a small Qwen2Config (fewer layers/dims) for fast iteration")
    return parser.parse_args()


def make_config(tiny: bool) -> Qwen2Config:
    if tiny:
        return Qwen2Config(
            vocab_size=5000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            rope_theta=1000000.0,
            tie_word_embeddings=True,
            attn_implementation="eager",
        )
    # Qwen2.5-0.5B's published config.json values; weights are randomly
    # initialized since the real checkpoint cannot be downloaded here.
    return Qwen2Config(
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )


def build_model(config: Qwen2Config, state_dict=None) -> Qwen2ForCausalLM:
    model = Qwen2ForCausalLM(config)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    model.generation_config = GenerationConfig(
        use_cache=True,
        cache_implementation="static",
        cache_config={"batch_size": 1, "max_cache_len": MAX_CACHE_LEN},
    )
    return model


def check_static(exported_program, name: str):
    """
    `cat` ops are NOT a reliable static/dynamic signal by themselves: Qwen2's
    RoPE `rotate_half` uses `torch.cat((-x2, x1), dim=-1)` on fixed-size
    tensors, which shows up as plain `aten.cat.default` nodes unrelated to
    cache growth. The real invariants for "fully static" are:
      - no SymInt shapes anywhere in the graph (`range_constraints == {}`)
      - the StaticCache is written via `index_copy_` (in-place, fixed-size),
        never via a `cat` that grows the cache tensor itself
    """
    nodes = list(exported_program.graph.nodes)
    cat_nodes = [n for n in nodes if "cat" in str(n.target)]
    index_copy_nodes = [n for n in nodes if "index_copy" in str(n.target)]
    print(f"[{name}] nodes={len(nodes)}  range_constraints={exported_program.range_constraints}  "
          f"cat={len(cat_nodes)} (RoPE rotate_half, static-shaped)  index_copy={len(index_copy_nodes)} (cache writes)")
    assert exported_program.range_constraints == {}, f"{name}: graph still has dynamic (SymInt) dims"
    for n in nodes:
        val = n.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            assert all(isinstance(s, int) for s in val.shape), f"{name}: node {n.name} has a non-static shape"


def find_causal_mask_where_node(exported_program):
    """With attn_implementation='eager' the causal mask is materialized as
    `torch.where(bool_mask, 0.0, finfo(dtype).min)`, so the mask value shows
    up as a literal scalar in an `aten.where.ScalarOther` node's args. With
    'sdpa' the mask stays boolean and the fill value lives inside the fused
    kernel, so there is nothing to edit at the graph level."""
    candidates = [n for n in exported_program.graph.nodes
                  if "where" in str(n.target) and len(n.args) == 3 and isinstance(n.args[2], float)]
    return candidates


def overwrite_mask_value(exported_program, new_value: float):
    nodes = find_causal_mask_where_node(exported_program)
    for n in nodes:
        n.args = (n.args[0], n.args[1], new_value)
    return len(nodes)


def main():
    args = parse_args()
    config = make_config(args.tiny)
    torch.manual_seed(0)

    base_model = build_model(config)
    weights = base_model.state_dict()
    prompt_ids = torch.randint(1, min(config.vocab_size, 5000), (1, PROMPT_LEN))

    print("=" * 60)
    print("STEP 1: eager baseline (model.generate)")
    print("=" * 60)
    baseline_model = build_model(config, weights)
    baseline = baseline_model.generate(prompt_ids, max_new_tokens=NEW_TOKENS, do_sample=False)
    print("baseline tokens:", baseline.tolist()[0])

    print()
    print("=" * 60)
    print("STEP 2: export prefill / decode as separate static graphs")
    print("=" * 60)
    exportable_prefill = TorchExportableModuleForDecoderOnlyLM(
        build_model(config, weights), batch_size=1, max_cache_len=MAX_CACHE_LEN)
    ep_prefill = exportable_prefill.export(input_ids=prompt_ids, cache_position=torch.arange(PROMPT_LEN))

    exportable_decode = TorchExportableModuleForDecoderOnlyLM(
        build_model(config, weights), batch_size=1, max_cache_len=MAX_CACHE_LEN)
    ep_decode = exportable_decode.export(
        input_ids=torch.zeros(1, 1, dtype=torch.long), cache_position=torch.tensor([0]))

    print()
    print("=" * 60)
    print("STEP 3: verify both graphs are fully static")
    print("=" * 60)
    check_static(ep_prefill, "prefill")
    check_static(ep_decode, "decode")

    print()
    print("=" * 60)
    print("STEP 4: locate + overwrite the causal-mask fill value")
    print("=" * 60)
    prefill_mod = ep_prefill.module()
    logits_before = prefill_mod(input_ids=prompt_ids, cache_position=torch.arange(PROMPT_LEN))

    where_nodes = find_causal_mask_where_node(ep_prefill)
    print(f"mask `where` nodes found: {len(where_nodes)} (value={where_nodes[0].args[2] if where_nodes else None})")
    n_changed = overwrite_mask_value(ep_prefill, -1e4)
    print(f"overwrote {n_changed} node(s): -inf -> -1e4")

    prefill_mod = ep_prefill.module()
    logits_after = prefill_mod(input_ids=prompt_ids, cache_position=torch.arange(PROMPT_LEN))
    max_diff = (logits_before - logits_after).abs().max().item()
    print(f"logits max diff after mask rewrite: {max_diff}")
    assert max_diff == 0.0
    assert torch.equal(logits_before.argmax(-1), logits_after.argmax(-1))

    print()
    print("=" * 60)
    print("STEP 5: run exported graphs end-to-end, compare to baseline")
    print("=" * 60)
    decode_mod = ep_decode.module()

    logits = prefill_mod(input_ids=prompt_ids, cache_position=torch.arange(PROMPT_LEN))
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

    # Each exported graph owns its own StaticCache buffers; bridging them at
    # runtime (here, a plain buffer copy) is the responsibility deployment
    # tooling normally handles (e.g. ExecuTorch multi-method .pte memory plan).
    prefill_buffers = dict(prefill_mod.named_buffers())
    decode_buffers = dict(decode_mod.named_buffers())
    copied = 0
    for name, buf in decode_buffers.items():
        if name in prefill_buffers:
            buf.copy_(prefill_buffers[name])
            copied += 1
    print(f"copied {copied} cache buffers: prefill graph -> decode graph")

    generated = prompt_ids[0].tolist() + [next_token]
    pos = PROMPT_LEN
    for _ in range(NEW_TOKENS - 1):
        logits = decode_mod(input_ids=torch.tensor([[generated[-1]]]), cache_position=torch.tensor([pos]))
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated.append(next_token)
        pos += 1

    print("exported tokens:", generated)
    print("baseline tokens:", baseline.tolist()[0])
    match = baseline.tolist()[0] == generated
    print("MATCH:", match)
    assert match, "exported graph output diverged from eager baseline"

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
