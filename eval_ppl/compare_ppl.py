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

Early natural language processing systems were based on hand-written rules. Although
this approach provides a precise and interpretable representation, it does not scale
well in practice. Starting in the late 1980s, however, there was a revolution in
natural language processing with the introduction of machine learning algorithms.
This was due both to the steady increase in computational power and the gradual
increase in the dominance of statistical methods in NLP research.

= Language Models =

A language model is a probability distribution over sequences of words. Given any
sequence of words of length m, a language model assigns a probability to the whole
sequence. Language models generate probabilities by training on text corpora in one
or more languages. The quality of the model can be measured by perplexity on a held
out test corpus.

Statistical language models were the dominant approach before the rise of neural
methods. N-gram models estimate the probability of a word given the preceding n-1
words. Neural language models use dense word embeddings and recurrent or transformer
architectures to capture long-range dependencies in text. Large pretrained language
models trained on massive corpora have demonstrated remarkable abilities in few-shot
and zero-shot generalization across many NLP tasks.

= Perplexity =

Perplexity is a measurement of how well a probability model predicts a sample.
A low perplexity indicates the model is good at predicting the sample. A language
model can be used to estimate the probability of a word sequence. Perplexity is
defined as the exponentiation of the average negative log-likelihood per token.
Formally, for a tokenized sequence x of N tokens, perplexity is computed as the
exponential of the mean negative log-likelihood across all tokens.

Perplexity can be understood as the weighted average branching factor of a language.
If the perplexity of a language model is k, then on average the model is as uncertain
as if it had to choose uniformly at random between k equally likely words at each
position. Lower perplexity models assign higher probabilities to held-out test data
and are considered better at capturing the statistical structure of language.

= Transformer Architecture =

The transformer is a deep learning architecture developed by researchers at Google.
It relies on self-attention mechanisms to process sequences of data. Unlike recurrent
neural networks, transformers process the entire sequence simultaneously, allowing for
much greater parallelism during training. Transformers have become the dominant
architecture for large language models, enabling remarkable advances in natural
language understanding and generation tasks.

The self-attention mechanism allows each token to attend to all other tokens in the
input sequence, weighted by learned attention scores. Multi-head attention applies
several attention functions in parallel, allowing the model to attend to different
representation subspaces at different positions. Positional encodings are added to
the input embeddings to give the model information about the relative or absolute
position of tokens in the sequence.

= Evaluation Benchmarks =

Standard benchmarks for language models include WikiText-2 and WikiText-103, which
contain text from verified Wikipedia articles. These benchmarks are widely used to
compare the perplexity of different language models. A well-trained language model
with one billion parameters typically achieves a perplexity below twenty on WikiText-2.
Larger models generally achieve lower perplexity, indicating better language modeling
capability across a wide range of topics and writing styles.

WikiText-2 contains approximately two million training tokens extracted from the set
of verified good and featured articles on Wikipedia. WikiText-103 is a larger version
containing over one hundred million training tokens. Both datasets preserve the
original text, including capitalization, punctuation, and numbers, making them
challenging and realistic benchmarks for language model evaluation.

= Tokenization =

Tokenization is the process of breaking text into smaller units called tokens.
In modern language models, subword tokenization methods such as byte-pair encoding
are commonly used. These methods strike a balance between character-level and
word-level tokenization, allowing the model to handle rare and out-of-vocabulary
words by breaking them into frequent subword units.

Byte-pair encoding starts with individual characters and iteratively merges the most
frequent pair of consecutive tokens into a new token. This process is repeated until
the desired vocabulary size is reached. The resulting vocabulary contains common words
as single tokens and rare words split into subword units. Most large language models
use vocabularies of thirty thousand to one hundred thousand tokens.

= Scaling Laws =

Scaling laws describe how model performance improves as a function of the number of
parameters, the amount of training data, and the amount of compute. Empirical studies
have shown that language model perplexity decreases smoothly as a power law with
respect to each of these factors. These findings have motivated training increasingly
large models on increasingly large datasets.

The Chinchilla scaling laws suggest that for a given compute budget, the optimal
strategy is to train a model with roughly the same number of parameters as the number
of training tokens. This implies that many large language models are undertrained
relative to their size. Following these guidelines leads to smaller but more efficient
models that achieve competitive performance at reduced inference cost.

= Attention Mechanisms =

Attention mechanisms allow models to selectively focus on different parts of the input
when producing an output. In the encoder-decoder attention used in machine translation,
the decoder learns to align each output token with the most relevant positions in the
source sequence. Self-attention extends this idea to allow each token in a sequence
to attend to all other tokens within the same sequence.

The scaled dot-product attention computes attention scores by taking the dot product
of query and key vectors, scaling by the square root of the key dimension, and applying
a softmax to obtain attention weights. The output is a weighted sum of value vectors.
This operation is efficient to compute on modern hardware and forms the core building
block of transformer models.
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
