# fx_qwen — FX-based Quantization for Qwen2.5

PyTorch FX + torchao を使って Qwen2.5 に量子化を適用するサンプルコードです。

## ディレクトリ構成

```
fx_qwen/
├── .venv/              # Python 仮想環境（git 管理外）
├── docs/
│   └── fx_quantization_errors.md  # エラー記録と対処法
├── fx_quantize.py      # メインスクリプト
├── requirements.txt    # pip パッケージ一覧
└── README.md
```

## セットアップ

```bash
cd fx_qwen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

```bash
# toy モデル（ネット不要）
python fx_quantize.py --model ../eval_ppl/models/qwen-tiny

# Qwen2.5-0.5B（HuggingFace アクセス要）
python fx_quantize.py --model Qwen/Qwen2.5-0.5B

# INT4 量子化
python fx_quantize.py --model Qwen/Qwen2.5-0.5B --quant int4
```

## 量子化パイプライン

```
CausalLM
  └─[NoCacheWrapper]─→ torch.export.export ─→ FX graph (362 nodes)
                              ↓
                    torchao.quantize_()
                    Int8WeightOnlyConfig(version=2)
                              ↓
                    推論確認 (use_cache=False)
```

## エラー対処記録

発生したエラーと対処法の詳細は [`docs/fx_quantization_errors.md`](docs/fx_quantization_errors.md) を参照。

| エラー | 原因 | 対処 |
|--------|------|------|
| `co_varnames is too small` | `torch.fx.symbolic_trace` が複雑な forward を処理できない | `torch.export.export` に切り替え |
| `symbolic_trace + concrete_args` でも TraceError | concrete_args はシグネチャ引数のみ concrete 化。派生テンソル (position_ids 等) が Proxy になり transformers 5.x の masking_utils 内の if 文で Proxy.__bool__() が呼ばれる | `torch.export.export` を使う（dynamo ベースで制御フロー対応） |
| `No module named transformers.utils.fx` | transformers 5.x で削除済み | `torch.export.export` を直接使用 |
| `DynamicCache` not a known type | KV キャッシュが pytree 未登録 | `NoCacheWrapper` で `use_cache=False` に固定 |
| `torch.ao.quantization` deprecated | PyTorch 2.10 以降廃止 | `torchao.quantize_()` に移行 |
| `Int8WeightOnlyConfig` v1 deprecated | torchao 0.17 で旧実装廃止 | `version=2` を明示 |
