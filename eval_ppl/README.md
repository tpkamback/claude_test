# eval_ppl — LLM Perplexity Evaluation on WikiText

WikiText データセットを使って LLM の Perplexity (PPL) を評価するサンプルコードです。
LLM 初学者がすぐ試せるよう、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) をリファレンスにしたカスタム実装を用意しています。

---

## ディレクトリ構成

```
eval_ppl/
├── .venv/                  # Python 仮想環境（git 管理外）
├── docs/
│   └── ppl_and_wikitext.md # PPL と WikiText の解説ドキュメント
├── results/                # 評価結果ログ・サマリ（実行後生成）
├── eval_ppl.py             # カスタム PPL 評価スクリプト（本実装）
├── eval_ppl_lmeval.sh      # lm-eval を使ったリファレンス実行スクリプト
├── run_all.sh              # 全モデルをまとめて評価するスクリプト
├── requirements.txt        # pip パッケージ一覧
└── README.md               # 本ファイル
```

---

## セットアップ

### 1. Python 仮想環境の作成

```bash
cd eval_ppl
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. パッケージのインストール

```bash
pip install -r requirements.txt
```

主なパッケージ：

| パッケージ | 用途 |
|-----------|------|
| `lm-eval` | PPL 評価のリファレンス実装 |
| `transformers` | モデル・トークナイザのロード |
| `torch` | テンソル演算・GPU 実行 |
| `accelerate` | マルチ GPU 対応 (`device_map="auto"`) |
| `datasets` | WikiText データセットのロード |

### 3. （オプション）GPU 確認

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), 'GPUs available')"
```

---

## 使い方

### カスタム実装で PPL を計算する

```bash
source .venv/bin/activate

# 基本
python eval_ppl.py --model Qwen/Qwen2.5-0.5B

# GPU 2 枚使用
python eval_ppl.py --model Qwen/Qwen2.5-3B --num_gpus 2

# stride を変更（小さいほど精度向上・低速）
python eval_ppl.py --model Qwen/Qwen2.5-0.5B --stride 256

# WikiText-103 で評価
python eval_ppl.py --model Qwen/Qwen2.5-0.5B --dataset wikitext-103
```

主なオプション：

| オプション | デフォルト | 説明 |
|-----------|---------|------|
| `--model` | 必須 | HuggingFace モデル ID |
| `--dataset` | `wikitext-2` | `wikitext-2` or `wikitext-103` |
| `--split` | `test` | `test` or `validation` |
| `--stride` | `512` | スライディングウィンドウのストライド（トークン数） |
| `--max_length` | モデル依存 | コンテキスト長の上限 |
| `--num_gpus` | `1` | 使用 GPU 数 |
| `--dtype` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |

### lm-eval リファレンスで計算する

```bash
bash eval_ppl_lmeval.sh Qwen/Qwen2.5-0.5B 1
bash eval_ppl_lmeval.sh Qwen/Qwen2.5-3B   2
```

### 全モデルをまとめて評価

```bash
bash run_all.sh 2   # 2 GPU で全モデル評価
```

結果は `results/summary.tsv` にまとめられます。

---

## 評価結果

**設定：** WikiText-2 test split、stride=512、GPU×2（A100 等を想定）

### カスタム実装 (eval_ppl.py)

| モデル | PPL | 計算時間 |
|--------|-----|---------|
| Qwen/Qwen2.5-0.5B | *(実行して記入)* | *(実行して記入)* |
| Qwen/Qwen2.5-3B   | *(実行して記入)* | *(実行して記入)* |

### lm-eval リファレンス

| モデル | word_perplexity | byte_perplexity | 計算時間 |
|--------|----------------|----------------|---------|
| Qwen/Qwen2.5-0.5B | *(実行して記入)* | *(実行して記入)* | *(実行して記入)* |
| Qwen/Qwen2.5-3B   | *(実行して記入)* | *(実行して記入)* | *(実行して記入)* |

> GPU 環境で `bash run_all.sh 2` を実行後、`results/summary.tsv` の値をここに記入してください。

---

## PPL の仕組みについて

→ [`docs/ppl_and_wikitext.md`](docs/ppl_and_wikitext.md) を参照

---

## 参考

- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HuggingFace Transformers — Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [WikiText dataset (HuggingFace)](https://huggingface.co/datasets/wikitext)
