# Perplexity (PPL) と WikiText データセット解説

LLM 初学者向けに、**Perplexity（困惑度）** の定義・計算方法と、評価に使う **WikiText データセット** の概要をまとめます。

---

## 1. Perplexity (PPL) とは

### 直感的な理解

Perplexity は「言語モデルがテキストをどれだけ驚かずに予測できるか」を示す指標です。

- **低い PPL** → モデルがテキストをよく予測できている（良いモデル）
- **高い PPL** → モデルが次のトークンを予測しにくい（悪いモデル）

### 数式

テキスト $x = (x_1, x_2, \ldots, x_N)$ に対して：

$$
\text{PPL}(x) = \exp\!\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_1, \ldots, x_{i-1}) \right)
$$

| 記号 | 意味 |
|------|------|
| $N$ | トークン総数 |
| $x_i$ | $i$ 番目のトークン |
| $P(x_i \mid x_1,\ldots,x_{i-1})$ | モデルが予測した条件付き確率 |

### Cross-Entropy Loss との関係

Transformer モデルの学習・評価で用いる **Cross-Entropy Loss** は：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i})
$$

これは **negative log-likelihood (NLL) の平均** であり、PPL はそれの指数：

$$
\text{PPL} = e^{\mathcal{L}}
$$

HuggingFace の `AutoModelForCausalLM` で `labels` を渡すと `outputs.loss` として $\mathcal{L}$ が返ってくるため、`math.exp(outputs.loss.item())` で PPL を計算できます。

---

## 2. スライディングウィンドウによる PPL 計算

### 問題

Transformer はコンテキスト長 $L_{\max}$（例：2048 トークン）の制限があります。
Wikipedia などの長いドキュメント全体に対して PPL を測るには工夫が必要です。

### 方法

**stride $s$ のスライディングウィンドウ**を使います：

```
|←─── max_length ───→|
[  context  |← target →]  window 1  (begin=0)
        [  context  |← target →]    window 2  (begin=stride)
                [  context  |← target →]      window 3  ...
```

- 各ウィンドウで左側 `max_length - stride` トークンは **コンテキストのみ**（損失計算対象外）
- 右側 `stride` トークンのみ **NLL を累積**
- stride = max_length にすると **non-overlapping**（精度やや低下）
- stride が小さいほど各トークンに長いコンテキストが与えられ **精度向上**（計算コスト増）

### 実装のポイント（PyTorch）

```python
target_ids = input_ids.clone()
target_ids[:, :-target_len] = -100   # コンテキスト部分をマスク
outputs = model(input_ids, labels=target_ids)
nll = outputs.loss * target_len      # loss はマスク外の平均 → トークン数をかける
```

最終 PPL:

```python
ppl = math.exp(total_nll / total_tokens)
```

---

## 3. WikiText データセット

### 概要

| 項目 | WikiText-2 | WikiText-103 |
|------|-----------|--------------|
| 出典 | Wikipedia の featured/good articles | 同左（大規模版） |
| 訓練トークン数 | 約 2.1M | 約 103M |
| テストトークン数 | 約 245K | 約 246K |
| HuggingFace ID | `wikitext-2-raw-v1` | `wikitext-103-raw-v1` |
| 言語 | 英語 | 英語 |

PPL 評価の標準ベンチマークとして広く使われており、GPT-2 や LLaMA などの論文でも報告されています。

### HuggingFace での読み込み

```python
from datasets import load_dataset

# WikiText-2
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# WikiText-103
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
```

### データ構造

```python
ds[0]
# {'text': ' = Valkyria Chronicles III = \n'}
ds[1]
# {'text': ''}
ds[2]
# {'text': ' Senjō no Valkyria 3 : ...'}
```

各行は `{"text": "..."}` の辞書。空行（セクション区切り）を除いて結合すると一続きのコーパスになります。

### lm-eval でのタスク名

```
lm_eval --tasks wikitext
```

`wikitext` タスクは WikiText-2 test split の PPL (`word_perplexity`, `byte_perplexity`, `bits_per_byte`) を報告します。

---

## 4. PPL の目安（参考値）

以下は WikiText-2 test における報告値の例です（実際の値は実行環境・stride 設定で変わります）：

| モデル | PPL (WikiText-2) |
|--------|-----------------|
| GPT-2 (117M) | ~29 |
| GPT-2 XL (1.5B) | ~18 |
| Qwen2.5-0.5B | ~20 前後 |
| Qwen2.5-3B | ~12 前後 |

PPL は **モデルサイズが大きいほど低くなる**傾向があります。

---

## 5. 参考リンク

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 本プロジェクトのリファレンス実装
- [HuggingFace: WikiText](https://huggingface.co/datasets/wikitext)
- [HuggingFace: Perplexity of Fixed-Length Models](https://huggingface.co/docs/transformers/perplexity)
- [Merity et al. 2016: Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) — WikiText 論文
