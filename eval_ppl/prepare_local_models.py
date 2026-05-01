"""
Save tiny local models (GPT-2 like, Qwen-2.5 like) to disk.
Run once before compare_ppl.py.
"""

from pathlib import Path
from transformers import GPT2Config, GPT2LMHeadModel

MODELS = {
    "gpt2-tiny": dict(
        n_layer=4, n_head=4, n_embd=128, n_positions=256,
        vocab_size=5000,
    ),
    "qwen-tiny": dict(
        n_layer=6, n_head=8, n_embd=256, n_positions=256,
        vocab_size=5000,
    ),
}

for name, kwargs in MODELS.items():
    path = Path(f"models/{name}")
    path.mkdir(parents=True, exist_ok=True)
    config = GPT2Config(**kwargs)
    model = GPT2LMHeadModel(config)
    model.save_pretrained(path)
    config.save_pretrained(path)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Saved {name}: {params:.2f}M params → {path}")
