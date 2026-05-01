"""
eval_quant_ppl.py — PPL comparison: Float32 vs Static Int8 quantization

Evaluates Qwen2.5-0.5B (random weights) on an offline corpus.
Computes token perplexity before and after static int8 quantization.

Usage:
    python eval_quant_ppl.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

# Make ppl module importable
sys.path.insert(0, "../eval_ppl")

import torch
from transformers import AutoModelForCausalLM

from ppl import compute_nll, token_perplexity
from static_quantize_211 import insert_observers, calibrate, remove_observers, apply_static_quant

# ─────────────────────────────────────────────────────────────────────────────
# Corpus (from compare_ppl.py)
# ─────────────────────────────────────────────────────────────────────────────

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
# Simple word-level tokenizer (no HuggingFace tokenizer required)
# ─────────────────────────────────────────────────────────────────────────────

class LocalTokenizer:
    """Word-level tokenizer built from the corpus vocabulary."""

    def __init__(self, text: str):
        words = text.split()
        vocab = sorted(set(words))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> torch.Tensor:
        """Return (1, seq_len) tensor of word ids."""
        ids = [self.w2i.get(w, 0) for w in text.split()]
        return torch.tensor([ids])

    def count_words(self, text: str) -> int:
        return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "../eval_ppl/models/Qwen2.5-0.5B-random"
MAX_LENGTH = 128
STRIDE = 128
N_CALIB = 4
CALIB_SEQ_LEN = 128


def load_model() -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    tokenizer = LocalTokenizer(CORPUS)
    input_ids = tokenizer.encode(CORPUS)
    n_tokens = input_ids.size(1)
    n_words = tokenizer.count_words(CORPUS)

    # ── Float32 PPL ────────────────────────────────────────────────────────────
    print("Loading float32 model ...")
    model_fp32 = load_model()
    params = sum(p.numel() for p in model_fp32.parameters()) / 1e6

    with torch.no_grad():
        nll_fp32, tokens_fp32 = compute_nll(model_fp32, input_ids, MAX_LENGTH, STRIDE)
    ppl_fp32 = token_perplexity(nll_fp32, tokens_fp32)
    del model_fp32

    # ── Static Int8 PPL ────────────────────────────────────────────────────────
    print("Loading model for static int8 quantization ...")
    model_int8 = load_model()

    observers = insert_observers(model_int8)
    calibrate(model_int8, CALIB_SEQ_LEN, N_CALIB)
    remove_observers(model_int8)
    apply_static_quant(model_int8, observers)

    with torch.no_grad():
        nll_int8, tokens_int8 = compute_nll(model_int8, input_ids, MAX_LENGTH, STRIDE)
    ppl_int8 = token_perplexity(nll_int8, tokens_int8)
    del model_int8

    # ── Results ────────────────────────────────────────────────────────────────
    degradation = ppl_int8 - ppl_fp32
    degradation_pct = (degradation / ppl_fp32) * 100

    print()
    print(f"Model : Qwen2.5-0.5B (random weights)")
    print(f"Corpus: {n_tokens} tokens, {n_words} words")
    print()
    print(f"Float32  token-PPL: {ppl_fp32:.2f}")
    print(f"Int8 Static  token-PPL: {ppl_int8:.2f}")
    print(f"Degradation: +{degradation:.2f} ({degradation_pct:.1f}%)")


if __name__ == "__main__":
    main()
