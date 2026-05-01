# FX量子化トライアル — エラー記録と対処法

Qwen2.5（および GPT-2系 toy モデル）に対して PyTorch FX量子化を適用した際に発生したエラーと、それぞれの原因・対処法の記録。

---

## 環境

| 項目 | バージョン |
|------|-----------|
| Python | 3.11 |
| PyTorch | 2.11.0+cu130 |
| transformers | 5.7.0 |
| torchao | 0.17.0 |

---

## エラー① — `co_varnames is too small`

### 発生箇所

```python
torch.fx.symbolic_trace(model)
```

### フルエラー

```
File "torch/fx/_symbolic_trace.py", line 210, in _patch_function
    new_code = CodeType(*co_args)
ValueError: code: co_varnames is too small
```

### 原因

`torch.fx.symbolic_trace` は内部でモデルの `forward()` 関数のバイトコードを書き換えてトレースする。
transformers の CausalLM は `forward()` の引数が非常に多い（`input_ids`, `attention_mask`, `past_key_values`, `use_cache`, ... など20以上）ため、Python のコードオブジェクト内の `co_varnames` サイズ上限を超えてしまう。

### 対処法

`torch.fx.symbolic_trace` の代わりに **`torch.export.export`** を使う。
`torch.export` は dynamo（torch.compile）ベースのトレーサーを使用しており、複雑な関数シグネチャに対して robust。

```python
# NG
traced = torch.fx.symbolic_trace(model)

# OK
exported = torch.export.export(model, example_inputs)
```

---

## エラー② — `transformers.utils.fx` が存在しない

### 発生箇所

```python
from transformers.utils.fx import symbolic_trace
```

### フルエラー

```
ModuleNotFoundError: No module named 'transformers.utils.fx'
```

### 原因

`transformers.utils.fx` は transformers 4.x 系に存在したモジュールだが、**transformers 5.x で削除**された。

### 対処法

transformers 独自の FX tracing は使わず、`torch.export.export` を直接使う（エラー①の対処法と同じ）。

---

## エラー③ — `DynamicCache` が pytree 未登録

### 発生箇所

```python
torch.export.export(model, example_inputs)
```

### フルエラー

```
RuntimeError: Found <class 'transformers.cache_utils.DynamicCache'>
in output, which is not a known type.
If this type holds tensors, you need to register a pytree for it.
```

### 原因

transformers 4.36 以降、CausalLM の `forward()` はデフォルトで KV キャッシュを `DynamicCache` オブジェクトとして返す（`use_cache=True` が default）。
`torch.export` はグラフの出力を pytree（torch が認識できる型のツリー）として処理する必要があるが、`DynamicCache` は pytree として登録されていないため失敗する。

### 対処法

`use_cache=False` を固定し、`logits` テンソルだけを返すラッパーモジュールを噛ませる。

```python
class NoCacheWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False).logits

wrapped = NoCacheWrapper(model)
exported = torch.export.export(wrapped, (torch.zeros(1, 16, dtype=torch.long),))
# → 成功: 362 nodes
```

PPL評価など推論用途では KV キャッシュは不要なため、この制約は問題にならない。

---

## エラー④ — `torch.ao.quantization` が deprecated

### 発生箇所

```python
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
```

### フルエラー / 警告

```
DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10.
For migrations of users:
2. FX graph mode quantization (prepare_fx, convert_fx),
   please migrate to use torchao pt2e quantization API instead (prepare_pt2e, convert_pt2e)
   https://github.com/pytorch/ao/issues/2259
```

さらに `torchao.quantization.pt2e` に `prepare_pt2e` が存在しないエラーも発生:

```
ImportError: cannot import name 'prepare_pt2e' from 'torchao.quantization.pt2e'
```

### 原因

PyTorch の量子化 API は世代交代の過渡期にある:

| 世代 | API | 状態 |
|------|-----|------|
| 旧 (Eager mode) | `torch.ao.quantization.quantize` | deprecated |
| 旧 (FX mode) | `prepare_fx` / `convert_fx` | deprecated |
| 新 (PT2E) | `prepare_pt2e` / `convert_pt2e` | torchao 0.17 では torch.ao に移動済み |
| **現行推奨** | `torchao.quantization.quantize_()` | ✅ 安定 |

torchao 0.17 時点では `prepare_pt2e` の場所が流動的で、`torchao.quantization.pt2e` には存在しない。

### 対処法

PT2E パイプライン（`prepare_pt2e` → `convert_pt2e`）は使わず、**`torchao.quantization.quantize_()`** の eager mode API を使う。こちらはモデルの `nn.Linear` 層を in-place で量子化するシンプルな API で、FX グラフのエクスポートが不要。

```python
from torchao.quantization import quantize_, Int8WeightOnlyConfig

quantize_(model, Int8WeightOnlyConfig(version=2))
# → INT8 weight-only 量子化が完了、推論可能
```

---

## エラー⑤ — `Int8WeightOnlyConfig` v1 deprecated 警告

### 発生箇所

```python
quantize_(model, Int8WeightOnlyConfig())  # version=1 (default)
```

### 警告

```
UserWarning: Config Deprecation: version 1 of Int8WeightOnlyConfig
is deprecated and will no longer be supported in a future release,
please use version 2
```

### 原因

`Int8WeightOnlyConfig` のデフォルト（`version=1`）は `PlainLayout` / `AffineQuantizedTensor` を使う旧実装で、torchao 0.17 で deprecated になっている。

### 対処法

`version=2` を明示する。

```python
quantize_(model, Int8WeightOnlyConfig(version=2))  # ✅ 警告なし
```

---

## 最終的に動作した構成

```python
import torch
from transformers import AutoModelForCausalLM
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# 1. モデルロード
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
model.eval()

# 2. FX グラフ export（DynamicCache 回避のためラッパー使用）
class NoCacheWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, input_ids): return self.model(input_ids, use_cache=False).logits

exported = torch.export.export(NoCacheWrapper(model),
                               (torch.zeros(1, 16, dtype=torch.long),))

# 3. INT8 量子化（eager mode, in-place）
quantize_(model, Int8WeightOnlyConfig(version=2))

# 4. 推論確認
with torch.no_grad():
    out = model(torch.zeros(1, 16, dtype=torch.long), use_cache=False)
# → Logits shape: torch.Size([1, 16, vocab_size])
```

---

---

# torch 2.2.1 での PT2E パイプライン

torch 2.2.1 + transformers 4.44.0 で `prepare_pt2e` / `convert_pt2e` を使った場合のエラーと対処。
実装: `pt2e_quantize_221.py`

---

## エラー⑥ — `torch.compiler.is_compiling` が未実装（2.2.1）

### 発生箇所

```python
torch.export.export(NoCacheWrapper(model), example)
```

### エラー

```
InternalTorchDynamoError: is_compiling
...
transformers/utils/import_utils.py: return torch.compiler.is_compiling()
```

### 原因

transformers 4.44 が内部で `torch.compiler.is_compiling()` を呼ぶが、
この関数は **torch 2.3 以降**で追加されたため 2.2.1 には存在しない。

### 対処法

```python
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
```

---

## エラー⑦ — `ExportedProgram` に `.meta` がない（2.2.1）

### 発生箇所

```python
prepare_pt2e(exported_program, quantizer)
```

### エラー

```
AttributeError: 'ExportedProgram' object has no attribute 'meta'
```

### 原因

torch 2.3 以降では `prepare_pt2e` が `ExportedProgram` を直接受け取れるが、
2.2.1 では `GraphModule` を期待する。

### 対処法

```python
gm = exported_program.module()   # GraphModule を取り出す
prepare_pt2e(gm, quantizer)      # ✅
```

---

## エラー⑧ — `prepared.eval()` が未サポート（2.2.1）

### 発生箇所

```python
prepared.eval()
```

### エラー

```
NotImplementedError: Calling eval() is not supported yet.
```

### 原因

`prepare_pt2e` が返す `GraphModule` は 2.2.1 時点で `eval()` 未対応。

### 対処法

`eval()` を呼ばずそのままキャリブレーションする。

---

## エラー⑨ — HistogramObserver が Long tensor で失敗（2.2.1）

### 発生箇所

キャリブレーション実行時。

### エラー

```
RuntimeError: torch.histogram: input tensor and hist tensor should have
the same dtype, but got input long int and hist float
```

### 原因

`set_global(static config)` では activation observer が全ノードに挿入される。
embedding への入力 (`input_ids`) など Long 型テンソルを持つノードにも挿入されてしまい、
`HistogramObserver` が `histc` を Long tensor に対して呼び出す。

2.2.1 では observer 挿入時の dtype フィルタが未実装のため発生する。

### 対処法

`is_dynamic=True` (Dynamic Quantization) を使う。
weight のみ量子化し activation observer を挿入しないため回避できる。

```python
cfg = get_symmetric_quantization_config(is_dynamic=True)
quantizer = XNNPACKQuantizer().set_global(cfg)
```

---

## エラー⑩ — `torch.export.export` が `aten.linear` を分解する（2.2.1）

### 発生箇所

`torch.export.export` 後のグラフで `prepare_pt2e` を呼んでも observer が 0 になる。

### 原因

`torch.export.export` は `aten.linear.default` を `aten.addmm.default` に完全分解する。
XNNPACKQuantizer の "linear" パターンマッチャーは `aten.linear.default` を探すため
マッチしなくなり observer が挿入されない。

### 対処法

`torch._export.capture_pre_autograd_graph` を使う。
これは `aten.linear.default` を保持したまま graph を capture する。

```python
from torch._export import capture_pre_autograd_graph
gm = capture_pre_autograd_graph(model, example)
# → aten.linear.default が保持される → XNNPACKQuantizer が正しく動作
```

> torch 2.3 以降では `torch.export.export` + decomposition table 指定が推奨になる。

---

## torch 2.2.1 で動作した最終構成

```python
import torch
from transformers import AutoModelForCausalLM
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config,
)

# パッチ: torch.compiler.is_compiling が 2.2.1 に未実装
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32)
model.eval()

class NoCacheWrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.model = m
    def forward(self, x): return self.model(x, use_cache=False).logits

example = (torch.zeros(1, 16, dtype=torch.long),)
gm = capture_pre_autograd_graph(NoCacheWrapper(model), example)  # linear 保持

quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config(is_dynamic=True)  # Long tensor 問題を回避
)
prepared = prepare_pt2e(gm, quantizer)   # observers: 2
quantized = convert_pt2e(prepared)       # quantize/dequantize nodes: 4

with torch.no_grad():
    out = quantized(torch.zeros(1, 16, dtype=torch.long))
# → Logits shape: torch.Size([1, 16, vocab_size])  ✅
```

---

## バージョン別 API 対応まとめ

| API | torch 2.2.1 | torch 2.11 |
|-----|------------|-----------|
| `torch.export.export` | ✅ | ✅ |
| `capture_pre_autograd_graph` | ✅ (推奨) | △ (deprecated) |
| `prepare_pt2e` / `convert_pt2e` | ✅ (`torch.ao`) | ❌ (torchao に移管) |
| `XNNPACKQuantizer` | ✅ (`torch.ao`) | ❌ (torchao に移管) |
| `torchao.quantize_()` | △ (別途インストール) | ✅ |
| `torch.compiler.is_compiling` | ❌ (手動パッチ必要) | ✅ |
| `prepared.eval()` | ❌ (未サポート) | ✅ |

---

## 残課題

- Qwen2.5-0.5B / 3B での実測（HuggingFace へのアクセスが必要）
- INT8量子化後の PPL 変化を `eval_ppl/eval_ppl.py` と組み合わせて評価
- Static Quantization の Long dtype 問題の根本解決（カスタム Quantizer で dtype フィルタ追加）
