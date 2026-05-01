"""
ppl.py — Perplexity calculation core

Implements the same sliding-window perplexity used by lm-evaluation-harness.

lm-eval reference:
  - Rolling loglikelihood:
      lm_eval/models/huggingface.py  loglikelihood_rolling()
      https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/models/huggingface.py#L680
  - WikiText word_perplexity metric aggregation:
      lm_eval/tasks/wikitext/wikitext.yaml  (metric: word_perplexity)
      https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/wikitext/wikitext.yaml
"""

import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str,
    dtype: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal language model and its tokenizer from HuggingFace Hub
    (or a local directory path).

    dtype="auto" lets HuggingFace pick bfloat16 on modern GPUs, float32 on CPU.
    This matches lm-eval's default:
      lm_eval/models/huggingface.py  __init__()  dtype="auto"
      https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/models/huggingface.py#L160
    """
    torch_dtype = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
        "auto":     "auto",
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use GPU if available, otherwise CPU.
    # lm-eval equivalent: --device cuda  or  --device cpu
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()  # disable dropout etc. — required for deterministic PPL
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Text → token ids
# ─────────────────────────────────────────────────────────────────────────────

def encode(tokenizer: AutoTokenizer, text: str) -> torch.Tensor:
    """
    Convert a plain-text string into a (1, seq_len) tensor of token ids.

    lm-eval encodes WikiText the same way before passing to loglikelihood_rolling():
      lm_eval/tasks/wikitext/wikitext.yaml  doc_to_target: "{{page}}"
    """
    return tokenizer(text, return_tensors="pt").input_ids  # shape: (1, seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# Core: sliding-window negative log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def compute_nll(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_length: int,
    stride: int,
) -> tuple[float, int]:
    """
    Compute total negative log-likelihood (NLL) and token count for a sequence
    that may be longer than the model's context window.

    The key challenge
    -----------------
    Transformers have a fixed context window (e.g. 2048 tokens for Qwen2.5).
    A test corpus like WikiText-2 has ~245,000 tokens — far longer than any
    single forward pass.  We therefore slide a window across the sequence.

    Sliding-window algorithm
    ------------------------
    Each iteration we feed a window of `max_length` tokens to the model, but
    only accumulate the loss for the rightmost `stride` tokens.  The left part
    serves as context so those tokens get a long, realistic history.

    Visual example (max_length=6, stride=3):

        Full sequence:  [A B C D E F G H I]
        Window 1:       [A B C D E F]          loss on [D E F]
                               ↑ context ↑         ↑ target ↑
        Window 2:             [D E F G H I]    loss on [G H I]

    Setting stride = max_length gives non-overlapping windows (faster, less
    accurate).  This is the lm-eval default for the wikitext task.

    lm-eval equivalent
    ------------------
    lm_eval/models/huggingface.py  loglikelihood_rolling()
    https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/models/huggingface.py#L680

    The loop structure mirrors lm-eval's implementation:
      - context tokens → label = -100  (ignored by CrossEntropyLoss)
      - target tokens  → label = token id

    Parameters
    ----------
    model      : loaded causal LM (eval mode)
    input_ids  : (1, seq_len) tensor
    max_length : context window size (tokens)
    stride     : how many tokens to advance per step
                 stride == max_length → non-overlapping (lm-eval default)
                 stride <  max_length → overlapping (more accurate, slower)

    Returns
    -------
    total_nll   : sum of -log P(x_i | context) over all target tokens
    total_tokens: number of target tokens accumulated
    """
    # Determine which device the model lives on so we can move tensors there.
    device = next(model.parameters()).device

    seq_len = input_ids.size(1)
    total_nll    = 0.0
    total_tokens = 0
    prev_end     = 0  # end position of the previous window

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)

        # How many NEW tokens does this window contribute?
        # (The rest are context repeated from the previous window.)
        target_len = end - prev_end

        # Slice the window and move to the model's device.
        window_ids = input_ids[:, begin:end].to(device)  # (1, window_len)

        # Build label tensor.
        # CrossEntropyLoss ignores positions where label == -100.
        # We mask the context (left) part so only the `target_len` rightmost
        # tokens contribute to the loss.
        #
        # lm-eval does exactly this masking in loglikelihood_rolling():
        #   cont_toks = rolling_token_windows[...]  (target tokens only)
        labels = window_ids.clone()
        labels[:, :-target_len] = -100  # mask context tokens

        with torch.no_grad():
            # model(..., labels=...) returns mean cross-entropy over unmasked tokens.
            # Multiply by target_len to recover the *sum* of NLL.
            #
            # PyTorch CrossEntropyLoss reduction="mean" (default):
            #   loss = -1/N * sum log P(x_i | context)
            # So: sum NLL = loss * N
            loss = model(window_ids, labels=labels).loss
            total_nll    += loss.item() * target_len
            total_tokens += target_len

        prev_end = end
        if end == seq_len:
            break

    return total_nll, total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# PPL metrics
# ─────────────────────────────────────────────────────────────────────────────

def token_perplexity(total_nll: float, total_tokens: int) -> float:
    """
    Token perplexity (our custom metric):

        PPL_token = exp( total_NLL / total_tokens )

    This is the standard definition used in most NLP papers and textbooks.
    The denominator counts *tokens* (subwords after BPE tokenization).
    """
    return math.exp(total_nll / total_tokens)


def word_perplexity(total_nll: float, total_words: int) -> float:
    """
    Word perplexity (lm-eval's reported metric for WikiText):

        PPL_word = exp( total_NLL / total_words )

    lm-eval divides by the number of *words* (space-separated), not tokens.
    Because BPE tokenizers split rare words into multiple subwords,
    total_tokens >= total_words, so:

        PPL_word >= PPL_token  (word PPL is always higher)

    lm-eval reference:
      lm_eval/tasks/wikitext/wikitext.yaml  metric_list: word_perplexity
      https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/wikitext/wikitext.yaml
    """
    return math.exp(total_nll / total_words)


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    stride: int | None = None,
    max_length: int | None = None,
) -> dict:
    """
    Evaluate perplexity of `text` under `model`.

    stride defaults to max_length (non-overlapping), matching lm-eval's
    wikitext task default.

    Returns a dict with keys:
      token_ppl  : token perplexity (our metric)
      word_ppl   : word perplexity  (lm-eval metric)
      tokens     : number of tokens
      words      : number of words
      elapsed    : wall-clock seconds
    """
    if max_length is None:
        max_length = model.config.max_position_embeddings
    if stride is None:
        stride = max_length  # lm-eval default: non-overlapping

    input_ids = encode(tokenizer, text)
    words = len(text.split())

    t0 = time.perf_counter()
    total_nll, total_tokens = compute_nll(model, input_ids, max_length, stride)
    elapsed = time.perf_counter() - t0

    return {
        "token_ppl": token_perplexity(total_nll, total_tokens),
        "word_ppl":  word_perplexity(total_nll, words),
        "tokens":    total_tokens,
        "words":     words,
        "elapsed":   elapsed,
    }
