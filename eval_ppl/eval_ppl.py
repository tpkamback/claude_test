"""
eval_ppl.py — CLI for evaluating LLM perplexity on WikiText.

PPL calculation logic lives in ppl.py.

Usage:
    python eval_ppl.py --model Qwen/Qwen2.5-0.5B
    python eval_ppl.py --model Qwen/Qwen2.5-0.5B --dataset wikitext-103
    python eval_ppl.py --model /path/to/local/model --stride 256
"""

import argparse

from datasets import load_dataset

from ppl import evaluate, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPL on WikiText dataset")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g. Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2",
        choices=["wikitext-2", "wikitext-103"],
        help="WikiText variant (default: wikitext-2)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Sliding window stride in tokens. Defaults to max_length (lm-eval default).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Context window length. Defaults to model's max_position_embeddings.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    return parser.parse_args()


def load_wikitext(dataset_name: str, split: str) -> str:
    """Download WikiText and concatenate all non-empty lines into one string."""
    hf_name = (
        "wikitext-2-raw-v1" if dataset_name == "wikitext-2"
        else "wikitext-103-raw-v1"
    )
    dataset = load_dataset("wikitext", hf_name, split=split)
    return "\n\n".join(row["text"] for row in dataset if row["text"].strip())


def main() -> None:
    args = parse_args()

    print(f"Model   : {args.model}")
    print(f"Dataset : {args.dataset} / {args.split}")

    print("\n[1/3] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)

    print("[2/3] Loading dataset...")
    text = load_wikitext(args.dataset, args.split)
    print(f"      Words in corpus : {len(text.split()):,}")

    print("[3/3] Computing PPL...")
    result = evaluate(model, tokenizer, text, args.stride, args.max_length)

    print("\n" + "=" * 55)
    print(f"  Model      : {args.model}")
    print(f"  Dataset    : {args.dataset}-{args.split}")
    print(f"  Tokens     : {result['tokens']:,}")
    print(f"  Words      : {result['words']:,}")
    print(f"  token_PPL  : {result['token_ppl']:.4f}  (our metric)")
    print(f"  word_PPL   : {result['word_ppl']:.4f}  (lm-eval metric)")
    print(f"  Time       : {result['elapsed']:.1f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()
