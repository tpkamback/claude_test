"""
Toy PPL evaluation — no internet, no GPU required.

Creates a tiny randomly-initialized GPT-2 model and evaluates PPL
on a small hardcoded text corpus. Used to verify the sliding-window
pipeline end-to-end.

Usage:
    python eval_ppl_toy.py
"""

import math
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel


# ── Toy corpus (simulates WikiText structure) ─────────────────────────────────
TOY_TEXT = """
The history of artificial intelligence began in antiquity with myths and stories
of artificial beings endowed with intelligence or consciousness by master craftsmen.
The seeds of modern artificial intelligence were planted by classical philosophers
who attempted to describe the process of human thinking as the mechanical manipulation
of symbols.

Natural language processing is a subfield of linguistics, computer science, and
artificial intelligence concerned with the interactions between computers and human
language, in particular how to program computers to process and analyze large amounts
of natural language data. The goal is a computer capable of understanding the contents
of documents, including the contextual nuances of the language within them.

A language model is a probability distribution over sequences of words. Given any
sequence of words of length m, a language model assigns a probability to the whole
sequence. Language models generate probabilities by training on text corpora.
Perplexity is a measurement of how well a probability model predicts a sample.
A low perplexity indicates the model is good at predicting the sample.
""".strip()


def build_toy_model(vocab_size: int = 1000, n_layer: int = 2, n_head: int = 2,
                    n_embd: int = 64, n_positions: int = 128) -> GPT2LMHeadModel:
    """Return a randomly-initialized tiny GPT-2 model."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def compute_ppl(model, input_ids: torch.Tensor, max_length: int, stride: int):
    seq_len = input_ids.size(1)
    nlls, total = [], 0
    prev_end = 0

    t0 = time.perf_counter()
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        window = input_ids[:, begin:end]
        labels = window.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            loss = model(window, labels=labels).loss
        nlls.append(loss * target_len)
        total += target_len
        prev_end = end
        if end == seq_len:
            break

    elapsed = time.perf_counter() - t0
    ppl = math.exp(torch.stack(nlls).sum().item() / total)
    return ppl, elapsed, total


def main():
    print("=== Toy PPL Evaluation (CPU, no internet) ===\n")

    # Simple word-level tokenizer (no download needed)
    print("[1/3] Building vocabulary from corpus...")
    words = TOY_TEXT.split()
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    vocab_size = len(vocab)
    print(f"      Vocab size : {vocab_size}")

    def tokenize(text):
        return torch.tensor([[vocab.get(w, 0) for w in text.split()]])

    # Toy model
    print("[2/3] Building toy model (random weights)...")
    model = build_toy_model(vocab_size=vocab_size)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      Parameters : {param_count:.2f}M")

    # Tokenize
    input_ids = tokenize(TOY_TEXT)
    print(f"      Tokens     : {input_ids.size(1)}")

    # Evaluate with two stride settings
    print("[3/3] Computing PPL...\n")
    max_len = model.config.n_positions  # 128

    results = []
    for stride in [max_len, max_len // 2]:
        label = "non-overlapping" if stride == max_len else "overlapping (stride=max/2)"
        ppl, elapsed, tokens = compute_ppl(model, input_ids, max_len, stride)
        results.append((label, stride, ppl, elapsed, tokens))

    print(f"{'Setting':<30} {'Stride':>6} {'Tokens':>7} {'PPL':>10} {'Time(s)':>8}")
    print("-" * 65)
    for label, stride, ppl, elapsed, tokens in results:
        print(f"{label:<30} {stride:>6} {tokens:>7} {ppl:>10.2f} {elapsed:>8.3f}")

    print("\nNote: PPL is high because the model has random weights.")
    print("      Stride < max_length gives each token more context → lower PPL.")


if __name__ == "__main__":
    main()
