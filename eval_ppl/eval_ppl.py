"""
Perplexity (PPL) evaluation on WikiText-2 / WikiText-103 dataset.

Reference implementation based on lm-evaluation-harness (lm-eval).
Usage:
    python eval_ppl.py --model Qwen/Qwen2.5-0.5B
    python eval_ppl.py --model Qwen/Qwen2.5-3B --dataset wikitext-103
    python eval_ppl.py --model Qwen/Qwen2.5-3B --num_gpus 2 --stride 512
"""

import argparse
import math
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPL on WikiText dataset")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2",
        choices=["wikitext-2", "wikitext-103"],
        help="WikiText variant to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding window stride (tokens). Set equal to max_length for non-overlapping.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Context window length. Defaults to model's max position embeddings.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs for model parallel (via device_map).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of windows processed per forward pass (increase for speed on large GPU memory).",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, num_gpus: int, dtype: str):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if num_gpus > 1:
        device_map = "auto"
    elif torch.cuda.is_available():
        device_map = "cuda:0"
    else:
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def get_wikitext_text(dataset_name: str, split: str) -> str:
    """Load WikiText and concatenate all text into a single string."""
    hf_name = "wikitext-2-raw-v1" if dataset_name == "wikitext-2" else "wikitext-103-raw-v1"
    dataset = load_dataset("wikitext", hf_name, split=split)
    # Join all non-empty lines
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    return text


def compute_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int,
    stride: int,
    batch_size: int,
) -> tuple[float, float]:
    """
    Compute perplexity with a sliding window approach.

    PPL = exp( -1/N * sum_i log P(x_i | x_{<i}) )

    For sequences longer than max_length we slide a window of size max_length
    with step `stride`, accumulating only the NLL of the new tokens in each window
    (the first max_length - stride tokens are context only).

    Returns:
        ppl  : perplexity value
        elapsed: wall-clock seconds
    """
    # Determine device for inputs
    first_param = next(model.parameters())
    device = first_param.device if first_param.device.type != "meta" else torch.device("cuda:0")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids: torch.Tensor = encodings.input_ids  # (1, seq_len)
    seq_len = input_ids.size(1)

    if max_length is None:
        max_length = model.config.max_position_embeddings

    nlls: list[torch.Tensor] = []
    total_tokens = 0

    start = time.perf_counter()

    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        # Tokens whose loss we accumulate in this window
        target_len = end - prev_end

        window_ids = input_ids[:, begin:end].to(device)
        target_ids = window_ids.clone()
        # Mask context tokens (left part repeated from previous window)
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(window_ids, labels=target_ids)
            # outputs.loss is mean NLL over non-masked tokens
            nll = outputs.loss * target_len

        nlls.append(nll.cpu())
        total_tokens += target_len
        prev_end = end

        if end == seq_len:
            break

    elapsed = time.perf_counter() - start

    mean_nll = torch.stack(nlls).sum() / total_tokens
    ppl = math.exp(mean_nll.item())
    return ppl, elapsed


def main() -> None:
    args = parse_args()

    print(f"Model   : {args.model}")
    print(f"Dataset : {args.dataset} / {args.split}")
    print(f"Stride  : {args.stride}")
    print(f"GPUs    : {args.num_gpus}")

    print("\n[1/3] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.num_gpus, args.dtype)

    print("[2/3] Loading dataset...")
    text = get_wikitext_text(args.dataset, args.split)
    token_count = len(tokenizer(text).input_ids)
    print(f"      Tokens in corpus: {token_count:,}")

    max_length = args.max_length or model.config.max_position_embeddings
    print(f"      Max context length: {max_length}")

    print("[3/3] Computing PPL...")
    ppl, elapsed = compute_ppl(
        model, tokenizer, text, max_length, args.stride, args.batch_size
    )

    print("\n" + "=" * 50)
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {args.dataset}-{args.split}")
    print(f"  PPL     : {ppl:.4f}")
    print(f"  Time    : {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
