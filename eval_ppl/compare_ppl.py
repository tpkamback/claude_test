"""
Compare PPL computed by:
  (A) Custom eval_ppl.py implementation  → token perplexity
  (B) lm-eval reference algorithm        → word perplexity  (same NLL, different denominator)

Both run on local toy models (no internet required).

Usage:
    python compare_ppl.py
"""

import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

# ── Corpus (simulates WikiText test split) ────────────────────────────────────
CORPUS = """
= Natural Language Processing =

Natural language processing is a subfield of linguistics, computer science, and
artificial intelligence concerned with the interactions between computers and human
language. The goal is a computer capable of understanding the contents of documents,
including the contextual nuances of the language within them. The technology can then
accurately extract information and insights contained in the documents as well as
categorize and organize the documents themselves.

= Language Models =

A language model is a probability distribution over sequences of words. Given any
sequence of words of length m, a language model assigns a probability to the whole
sequence. Language models generate probabilities by training on text corpora in one
or more languages. The quality of the model can be measured by perplexity on a held
out test corpus.

= Perplexity =

Perplexity is a measurement of how well a probability model predicts a sample.
A low perplexity indicates the model is good at predicting the sample. A language
model can be used to estimate the probability of a word sequence. Perplexity is
defined as the exponentiation of the average negative log-likelihood per token.
Formally, for a tokenized sequence x of N tokens, perplexity is computed as the
exponential of the mean negative log-likelihood across all tokens.

= Transformer Architecture =

The transformer is a deep learning architecture developed by researchers at Google.
It relies on self-attention mechanisms to process sequences of data. Unlike recurrent
neural networks, transformers process the entire sequence simultaneously, allowing for
much greater parallelism during training. Transformers have become the dominant
architecture for large language models, enabling remarkable advances in natural
language understanding and generation tasks.

= Evaluation Benchmarks =

Standard benchmarks for language models include WikiText-2 and WikiText-103, which
contain text from verified Wikipedia articles. These benchmarks are widely used to
compare the perplexity of different language models. A well-trained language model
with one billion parameters typically achieves a perplexity below twenty on WikiText-2.
Larger models generally achieve lower perplexity, indicating better language modeling
capability across a wide range of topics and writing styles.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Simple word-level tokenizer (offline, no HuggingFace download)
# ─────────────────────────────────────────────────────────────────────────────

class LocalTokenizer:
    """Word-level tokenizer built from the corpus vocabulary."""

    def __init__(self, text: str):
        words = text.split()
        vocab = sorted(set(words))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def __call__(self, text: str, return_tensors="pt"):
        ids = [self.w2i.get(w, 0) for w in text.split()]
        t = torch.tensor([ids])
        class Out:
            input_ids = t
        return Out()

    def count_words(self, text: str) -> int:
        return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Shared NLL computation (sliding window)
# ─────────────────────────────────────────────────────────────────────────────

def compute_nll_sequence(model, input_ids: torch.Tensor, max_length: int, stride: int):
    """
    Returns (total_nll, total_tokens, elapsed_sec).
    stride = max_length  →  non-overlapping  (lm-eval default)
    stride < max_length  →  overlapping      (more accurate)
    """
    seq_len = input_ids.size(1)
    total_nll, total_tokens = 0.0, 0
    prev_end = 0

    t0 = time.perf_counter()
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        window = input_ids[:, begin:end]
        labels = window.clone()
        labels[:, :-target_len] = -100  # mask context tokens

        with torch.no_grad():
            loss = model(window, labels=labels).loss  # mean NLL over unmasked tokens

        total_nll += loss.item() * target_len
        total_tokens += target_len
        prev_end = end
        if end == seq_len:
            break

    elapsed = time.perf_counter() - t0
    return total_nll, total_tokens, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Two evaluation methods
# ─────────────────────────────────────────────────────────────────────────────

def eval_custom(model, tokenizer, text: str, max_length: int):
    """
    (A) Custom eval_ppl.py style:
        PPL = exp(total_NLL / total_tokens)   ← token perplexity
        stride = max_length  (lm-eval default, non-overlapping)
    """
    input_ids = tokenizer(text).input_ids
    nll, tokens, elapsed = compute_nll_sequence(model, input_ids, max_length, stride=max_length)
    ppl = math.exp(nll / tokens)
    return ppl, tokens, elapsed


def eval_lmeval_style(model, tokenizer, text: str, max_length: int):
    """
    (B) lm-eval wikitext style:
        word_perplexity = exp(total_NLL / total_words)   ← word perplexity
        stride = max_length  (same non-overlapping window)
        NLL computation is identical to (A); only the denominator differs.
    """
    input_ids = tokenizer(text).input_ids
    nll, tokens, elapsed = compute_nll_sequence(model, input_ids, max_length, stride=max_length)
    words = tokenizer.count_words(text)
    word_ppl = math.exp(nll / words)
    return word_ppl, words, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, vocab_size: int):
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model


def main():
    tokenizer = LocalTokenizer(CORPUS)

    model_dirs = sorted(Path("models").glob("*-tiny"))
    if not model_dirs:
        print("ERROR: No models found. Run prepare_local_models.py first.")
        return

    # Header
    print("=" * 90)
    print(f"{'Model':<14} {'Params':>7} {'Method':<22} {'Tokens/Words':>13} {'PPL':>10} {'Time(s)':>8}")
    print("-" * 90)

    for model_dir in model_dirs:
        model = load_model(str(model_dir), tokenizer.vocab_size)
        # swap embedding vocab to match our tokenizer
        model.resize_token_embeddings(tokenizer.vocab_size)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        max_len = model.config.n_positions

        # (A) Custom token PPL
        ppl_a, tokens, t_a = eval_custom(model, tokenizer, CORPUS, max_len)
        print(f"{model_dir.name:<14} {params:>6.2f}M  {'(A) custom  token-PPL':<22} {tokens:>13,} {ppl_a:>10.2f} {t_a:>8.3f}")

        # (B) lm-eval word PPL
        ppl_b, words, t_b = eval_lmeval_style(model, tokenizer, CORPUS, max_len)
        print(f"{'':>22}  {'(B) lm-eval word-PPL':<22} {words:>13,} {ppl_b:>10.2f} {t_b:>8.3f}")
        print()

    print("=" * 90)
    print("\nNotes:")
    print("  stride = max_length (non-overlapping) — matches lm-eval wikitext default")
    print("  (A) token-PPL : denominator = number of tokens")
    print("  (B) word-PPL  : denominator = number of words  (lm-eval reports this)")
    print("  NLL is identical between (A) and (B); only the denominator differs.")
    print("  Random weights → high PPL. Real models (GPT-2, Qwen) give PPL < 30.")


if __name__ == "__main__":
    main()
