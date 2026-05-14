#!/usr/bin/env python3
"""
Bootstrap: オリジナル llama.py をそのまま実行するためのモンキーパッチスクリプト。
MediaTek SDK (mtk_converter / mtk_neuron) と backend 以降の処理をスタブ化し、
prepare_pt2e → convert_pt2e まで本物の NeuropilotQuantizer で実行する。

使い方:
  cd /home/user/executorch_sparse/examples/mediatek
  python run_llama_orig.py tiny_llama/config.json -p A16W8 -n 2 -shapes 16t64c
"""

import sys
import os
import types
import warnings
warnings.filterwarnings("ignore")

# llama.py は cwd を sys.path に追加するので先に合わせる
MEDIATEK_DIR = os.path.dirname(os.path.abspath(__file__))
if MEDIATEK_DIR not in sys.path:
    sys.path.insert(0, MEDIATEK_DIR)

# ── STEP 0: datasets スタブ (llama.py が import するが dataset=None 時は使わない) ──
_datasets = types.ModuleType("datasets")
_datasets.load_dataset = lambda *a, **kw: []
sys.modules["datasets"] = _datasets

# ── STEP A: mtk_converter / mtk_neuron スタブ ────────────────────────────────
print("[bootstrap] mtk_converter / mtk_neuron スタブを挿入 ...")

mtk_converter_mod = types.ModuleType("mtk_converter")

class _FakeConverter:
    quantize = True
    input_quantization_bitwidths = None
    allow_missing_quantization_ranges = True
    prepend_input_quantize_ops = True
    prepend_input_quantize_ops_indices = []
    append_output_dequantize_ops = True
    append_output_dequantize_ops_indices = []

    @classmethod
    def from_exported_program(cls, ep):
        return cls()

    def convert_to_mlir(self):
        return b""

mtk_converter_mod.PyTorchV2Converter = _FakeConverter

# サブモジュール階層
_python  = types.ModuleType("mtk_converter.python")
_convs   = types.ModuleType("mtk_converter.python.converters")
_pytorch = types.ModuleType("mtk_converter.python.converters.pytorch")
_imp2    = types.ModuleType("mtk_converter.python.converters.pytorch.importer_v2")
_imp2.is_fx_node_supported = lambda node: True

mtk_converter_mod.python = _python
_python.converters        = _convs
_convs.pytorch            = _pytorch
_pytorch.importer_v2      = _imp2

sys.modules["mtk_converter"]                                     = mtk_converter_mod
sys.modules["mtk_converter.python"]                              = _python
sys.modules["mtk_converter.python.converters"]                   = _convs
sys.modules["mtk_converter.python.converters.pytorch"]           = _pytorch
sys.modules["mtk_converter.python.converters.pytorch.importer_v2"] = _imp2

# mtk_neuron スタブ
_mtk_neuron = types.ModuleType("mtk_neuron")
_mtk_neuron.compile = lambda mlir_str, options="": b"\x00" * 13
_mtk_neuron.extract_shared_data = lambda models, options="": (
    b"\x00" * 13, [b"\x00" * 13] * len(models)
)
sys.modules["mtk_neuron"] = _mtk_neuron

# ── STEP B: executorch.backends.mediatek インポート ───────────────────────────
print("[bootstrap] executorch.backends.mediatek をインポート ...")
from executorch.backends.mediatek import (       # noqa: E402
    NeuropilotQuantizer, NeuropilotPartitioner, Precision
)
from executorch.backends.mediatek.preprocess import NeuropilotBackend
from executorch.exir.backend.backend_details import PreprocessResult
from executorch.exir.backend.backend_api import (
    MethodProgramsPartitionerSpec, to_backend
)
from executorch import exir
print(f"  → NeuropilotQuantizer: {NeuropilotQuantizer}")
print(f"  → Precision members  : {list(Precision.__members__)}")

# ── STEP C: Backend 処理スタブ ────────────────────────────────────────────────
print("[bootstrap] Backend (NeuropilotBackend / exir / to_backend) をスタブ化 ...")

@classmethod                                    # type: ignore[misc]
def _fake_preprocess(cls, edge_program, module_compile_spec):
    print("    [stub] NeuropilotBackend.preprocess → dummy bytes")
    return PreprocessResult(processed_bytes=b"\x00" * 13)

NeuropilotBackend.preprocess = _fake_preprocess  # type: ignore[method-assign]

_orig_to_edge = exir.to_edge
def _fake_to_edge(ep, *args, **kwargs):
    print("    [stub] exir.to_edge → no-op wrapper")
    class _FakeEdgeProg:
        def exported_program(self_inner):
            return ep
    return _FakeEdgeProg()
exir.to_edge = _fake_to_edge

def _fake_to_backend(spec):
    print("    [stub] to_backend(MethodProgramsPartitionerSpec) → no-op")
    return {}
import executorch.exir.backend.backend_api as _ba
_ba.to_backend = _fake_to_backend

class _FakeEdgeManager:
    def __init__(self, delegated, compile_config=None): pass
    def to_executorch(self, config=None):
        class _FakeExecProg:
            buffer = b"\x00" * 13
        return _FakeExecProg()

exir.EdgeProgramManager = _FakeEdgeManager       # type: ignore[attr-defined]

# ── STEP D: export_to_et_ir パッチ (convert_pt2e 後で停止) ─────────────────
import torch
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

def _patched_export_to_et_ir(
    output_folder, exp_name, model, precision,
    max_num_token, max_cache_size, chunk_idx, export_shapes,
    platform_b, cal_dataset=None,
):
    print(f"\n{'='*60}")
    print(f"  Chunk {chunk_idx}  |  precision={precision}")
    print(f"{'='*60}")

    example_inputs, dynamic_shapes = model.get_example_inputs(
        max_num_token, max_cache_size, True
    )

    # STEP 1
    print("  STEP 1: torch.export.export ...")
    try:
        pre_autograd = torch.export.export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
        ).module()
        print(f"    → strict=True OK  ({len(list(pre_autograd.graph.nodes))} nodes)")
    except Exception as e:
        print(f"    → strict=True FAIL ({type(e).__name__}: {str(e)[:80]})")
        print("    → strict=False でリトライ ...")
        pre_autograd = torch.export.export(
            model, example_inputs, strict=False
        ).module()
        print(f"    → strict=False OK ({len(list(pre_autograd.graph.nodes))} nodes)")

    # STEP 2
    print(f"  STEP 2: prepare_pt2e  (NeuropilotQuantizer / Precision.{precision}) ...")
    quantizer = NeuropilotQuantizer()
    quantizer.setup_precision(getattr(Precision, precision))
    prepared = prepare_pt2e(pre_autograd, quantizer)
    obs_nodes = [n for n in prepared.graph.nodes
                 if "observer" in str(n.target) or "fake_quant" in str(n.target)]
    print(f"    → observer / fake_quant ノード: {len(obs_nodes)}")

    # STEP 3
    print("  STEP 3: calibration (dummy 1 pass) ...")
    with torch.no_grad():
        prepared(*example_inputs)
    print("    → done")

    # STEP 4
    print("  STEP 4: convert_pt2e(fold_quantize=False) ...")
    converted = convert_pt2e(prepared, fold_quantize=False)
    all_n = list(converted.graph.nodes)
    qdq_n  = [n for n in all_n
               if "quantize_per_tensor" in str(n.target)
               or "dequantize_per_tensor" in str(n.target)]
    lin_n  = [n for n in all_n if n.target == torch.ops.aten.linear.default]
    print(f"    → total nodes  : {len(all_n)}")
    print(f"    → QDQ nodes    : {len(qdq_n)}")
    print(f"    → linear nodes : {len(lin_n)}")

    print("  STEP 5-6: NeuropilotPartitioner / .pte 書き出し → スタブ (省略)")
    print()


# llama モジュールをインポートして関数を差し替え (model_export_scripts/ に存在)
import importlib.util as _ilu
_llama_spec = _ilu.spec_from_file_location(
    "llama",
    os.path.join(MEDIATEK_DIR, "model_export_scripts", "llama.py"),
)
_llama_mod = _ilu.module_from_spec(_llama_spec)
sys.modules["llama"] = _llama_mod
_llama_spec.loader.exec_module(_llama_mod)
_llama_mod.export_to_et_ir = _patched_export_to_et_ir

# ── STEP E: サニティチェック / トークナイザ パッチ ───────────────────────────
print("[bootstrap] sanity checks / tokenizer をパッチ ...")

import aot_utils.llm_utils.sanity_checks as _sc
_sc.check_tokenizer_exist = lambda folder: None
_sc.check_weights_exist   = lambda folder: None
_llama_mod.check_tokenizer_exist = lambda folder: None
_llama_mod.check_weights_exist   = lambda folder: None

import aot_utils.llm_utils.utils as _llm_utils

# embedding LUT 書き出し → スタブ
_llm_utils.dump_embedding_lut_for_cmdline = lambda wd, sd, cfg: None
_llama_mod.dump_embedding_lut_for_cmdline  = lambda wd, sd, cfg: None

# ダミートークナイザ (dataset=None のとき実際には使わない)
class _DummyTokenizer:
    eos_token_id = 2
    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls()
    def __call__(self, text, **kw):
        return {"input_ids": torch.zeros(1, 4, dtype=torch.int32)}

_orig_resolve = _llm_utils.resolve_model_classes

def _patched_resolve(config_filepath, bypass_tokenizer=False, response_handler=None):
    result = _orig_resolve(
        config_filepath, bypass_tokenizer=True,
        response_handler=response_handler
    )
    config, weight_dir, chunk_class = result
    return config, weight_dir, _DummyTokenizer, chunk_class

_llm_utils.resolve_model_classes = _patched_resolve
_llama_mod.resolve_model_classes  = _patched_resolve

# ── STEP F: main() 実行 ───────────────────────────────────────────────────────
print("[bootstrap] パッチ完了。llama.main() を実行 ...")
print()

if __name__ == "__main__":
    _llama_mod.main()
