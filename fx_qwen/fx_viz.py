"""
fx_viz.py — Block-level HTML visualizer for LLM FX graphs.

Parses a torch.export FX graph from a Qwen-style Transformer and generates
a single-file HTML with collapsible per-layer blocks, op color-coding, and
summary stats.

Usage:
    python fx_viz.py --model ../eval_ppl/models/Qwen2.5-0.5B-random --output fx_graph.html
"""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Model loading & export
# ─────────────────────────────────────────────────────────────────────────────

class NoCacheWrapper(torch.nn.Module):
    """Wraps a CausalLM to disable KV cache and return only logits."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, use_cache=False).logits


def load_and_export(model_path: str, seq_len: int) -> Tuple[torch.nn.Module, object]:
    """Load model and export to FX graph via torch.export."""
    print(f"[1/3] Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"      Parameters: {params:.2f}M")

    print(f"[2/3] Exporting FX graph (seq_len={seq_len}) ...")
    t0 = time.perf_counter()
    wrapped = NoCacheWrapper(model)
    example = (torch.zeros(1, seq_len, dtype=torch.long),)
    with torch.no_grad():
        ep = torch.export.export(wrapped, example)
    gm = ep.module()
    elapsed = time.perf_counter() - t0
    nodes = len(list(gm.graph.nodes))
    print(f"      Export done in {elapsed:.1f}s — {nodes} nodes")
    return gm, ep


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Graph parsing
# ─────────────────────────────────────────────────────────────────────────────

# Map substrings in node names / targets to display categories
OP_CATEGORY_RULES: List[Tuple[str, str]] = [
    # attention
    ("q_proj", "attention"),
    ("k_proj", "attention"),
    ("v_proj", "attention"),
    ("o_proj", "attention"),
    ("attn", "attention"),
    ("sdpa", "attention"),
    ("scaled_dot_product", "attention"),
    ("rotary", "attention"),
    ("rope", "attention"),
    ("cos", "attention"),
    ("sin", "attention"),
    # norm
    ("norm", "norm"),
    ("layer_norm", "norm"),
    ("rms", "norm"),
    # activation
    ("silu", "activation"),
    ("gelu", "activation"),
    ("relu", "activation"),
    ("act", "activation"),
    ("sigmoid", "activation"),
    # linear / projection
    ("gate_proj", "linear"),
    ("up_proj", "linear"),
    ("down_proj", "linear"),
    ("lm_head", "linear"),
    ("embed_tokens", "embed"),
    ("linear", "linear"),
    ("mm", "linear"),
    ("addmm", "linear"),
    # arithmetic
    ("add", "arithmetic"),
    ("mul", "arithmetic"),
    ("sub", "arithmetic"),
    ("div", "arithmetic"),
    ("pow", "arithmetic"),
    ("rsqrt", "arithmetic"),
    ("mean", "arithmetic"),
]

LAYER_RE = re.compile(r"(?:^|_)layers[_.](\d+)[_.]", re.IGNORECASE)
EMBED_RE = re.compile(r"embed", re.IGNORECASE)
NORM_FINAL_RE = re.compile(r"(?:^|_)(?:final_layernorm|model_norm|norm)(?:$|_)")
LMHEAD_RE = re.compile(r"lm_head")


def categorize_op(name: str, target: str) -> str:
    """Return display category for a node based on name/target strings."""
    combined = (name + " " + str(target)).lower()
    for keyword, cat in OP_CATEGORY_RULES:
        if keyword in combined:
            return cat
    return "other"


def assign_group(name: str) -> Tuple[str, Optional[int]]:
    """
    Returns (group_name, layer_index).
    group_name is one of: 'embed', 'layer_N', 'norm', 'lm_head', 'other'.
    """
    m = LAYER_RE.search(name)
    if m:
        n = int(m.group(1))
        return f"layer_{n}", n

    if LMHEAD_RE.search(name):
        return "lm_head", None
    if NORM_FINAL_RE.search(name):
        return "norm", None
    if EMBED_RE.search(name):
        return "embed", None

    return "other", None


def _get_meta_shape(node) -> str:
    """Extract tensor shape from node metadata if available."""
    val = node.meta.get("val", None)
    if val is None:
        val = node.meta.get("tensor_meta", None)
    if val is None:
        return ""
    try:
        if hasattr(val, "shape"):
            return str(tuple(val.shape))
    except Exception:
        pass
    return ""


def parse_graph(gm: torch.nn.Module, max_layers: Optional[int] = None) -> dict:
    """
    Walk gm.graph.nodes and group them by Transformer block.

    Returns a dict:
        {
          "total_nodes": int,
          "groups": {
              "embed": {"nodes": [...], "op_counts": {...}},
              "layer_0": {"nodes": [...], "op_counts": {...}, "layer_idx": 0},
              ...
              "norm": {...},
              "lm_head": {...},
              "other": {...},
          },
          "layer_indices": [0, 1, 2, ...],   # sorted layer numbers found
        }

    Each node entry:
        {"name": str, "op": str, "target": str, "category": str, "shape": str}
    """
    print("[3/3] Parsing graph ...")
    groups: Dict[str, dict] = {}
    layer_indices = set()

    for node in gm.graph.nodes:
        name: str = node.name
        op: str = node.op          # placeholder, call_function, call_method, get_attr, output
        target = node.target

        target_str = str(target) if not callable(target) else getattr(target, "__name__", str(target))
        # For call_function, use qualified name
        if op == "call_function" and callable(target):
            target_str = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))

        category = categorize_op(name, target_str)
        shape = _get_meta_shape(node)

        group_name, layer_idx = assign_group(name)

        # Respect max_layers filter (still keep non-layer groups)
        if layer_idx is not None and max_layers is not None and layer_idx >= max_layers:
            continue

        if group_name not in groups:
            groups[group_name] = {
                "nodes": [],
                "op_counts": defaultdict(int),
                "layer_idx": layer_idx,
            }

        entry = {
            "name": name,
            "op": op,
            "target": target_str,
            "category": category,
            "shape": shape,
        }
        groups[group_name]["nodes"].append(entry)
        groups[group_name]["op_counts"][category] += 1

        if layer_idx is not None:
            layer_indices.add(layer_idx)

    total = sum(len(g["nodes"]) for g in groups.values())
    layer_indices_sorted = sorted(layer_indices)

    print(f"      Groups found: {list(groups.keys())[:6]}{'...' if len(groups) > 6 else ''}")
    print(f"      Layers: {len(layer_indices_sorted)},  Total nodes: {total}")

    return {
        "total_nodes": total,
        "groups": groups,
        "layer_indices": layer_indices_sorted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: HTML rendering
# ─────────────────────────────────────────────────────────────────────────────

# CSS color per category
CAT_COLORS = {
    "linear":     "#3b82f6",   # blue
    "norm":       "#22c55e",   # green
    "activation": "#f97316",   # orange
    "attention":  "#a855f7",   # purple
    "embed":      "#06b6d4",   # cyan
    "arithmetic": "#94a3b8",   # slate
    "other":      "#6b7280",   # gray
}

CAT_BG = {
    "linear":     "#eff6ff",
    "norm":       "#f0fdf4",
    "activation": "#fff7ed",
    "attention":  "#faf5ff",
    "embed":      "#ecfeff",
    "arithmetic": "#f8fafc",
    "other":      "#f9fafb",
}


def _op_badge(category: str, count: int = 1) -> str:
    color = CAT_COLORS.get(category, "#6b7280")
    bg = CAT_BG.get(category, "#f9fafb")
    label = f"{category}" if count == 1 else f"{category} ×{count}"
    return (
        f'<span class="badge" style="background:{bg};color:{color};'
        f'border:1px solid {color};">{label}</span>'
    )


def _node_row(node: dict) -> str:
    color = CAT_COLORS.get(node["category"], "#6b7280")
    bg = CAT_BG.get(node["category"], "#f9fafb")
    shape_html = (
        f'<span class="shape">{node["shape"]}</span>' if node["shape"] else ""
    )
    return (
        f'<tr style="background:{bg};">'
        f'<td class="node-name">{node["name"]}</td>'
        f'<td><span class="op-tag" style="color:{color};">{node["op"]}</span></td>'
        f'<td class="target">{node["target"]}</td>'
        f'<td>{shape_html}</td>'
        f'</tr>\n'
    )


def _layer_sub_section(title: str, nodes: List[dict]) -> str:
    if not nodes:
        return ""
    rows = "".join(_node_row(n) for n in nodes)
    return (
        f'<div class="subsection">'
        f'<div class="subsection-title">{title} ({len(nodes)} nodes)</div>'
        f'<table class="node-table"><thead><tr>'
        f'<th>name</th><th>op</th><th>target</th><th>shape</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
        f'</div>\n'
    )


def _split_layer_nodes(nodes: List[dict]) -> dict:
    """Split a layer's nodes into sub-categories for display."""
    buckets: Dict[str, List[dict]] = {
        "Self Attention": [],
        "MLP": [],
        "LayerNorm / RMSNorm": [],
        "RoPE": [],
        "Other": [],
    }
    for n in nodes:
        nm = n["name"].lower()
        tgt = n["target"].lower()
        combined = nm + " " + tgt
        if any(k in combined for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn", "sdpa", "scaled_dot")):
            buckets["Self Attention"].append(n)
        elif any(k in combined for k in ("gate_proj", "up_proj", "down_proj", "silu", "gelu", "mlp")):
            buckets["MLP"].append(n)
        elif any(k in combined for k in ("rope", "rotary", "cos", "sin", "embed_positions")):
            buckets["RoPE"].append(n)
        elif any(k in combined for k in ("norm", "rms", "layer_norm")):
            buckets["LayerNorm / RMSNorm"].append(n)
        else:
            buckets["Other"].append(n)
    return buckets


def _group_card(group_name: str, group_data: dict, is_layer: bool) -> str:
    """Generate HTML for one collapsible group card."""
    nodes = group_data["nodes"]
    op_counts = group_data["op_counts"]
    n_nodes = len(nodes)

    if is_layer:
        layer_idx = group_data["layer_idx"]
        title = f"Layer {layer_idx}"
        card_id = f"card-layer-{layer_idx}"
    else:
        title = group_name.replace("_", " ").title()
        card_id = f"card-{group_name}"

    # Badge summary
    badges = " ".join(_op_badge(cat, cnt) for cat, cnt in sorted(op_counts.items()))

    # Node content
    if is_layer:
        buckets = _split_layer_nodes(nodes)
        content_html = "".join(
            _layer_sub_section(sec, bucket_nodes)
            for sec, bucket_nodes in buckets.items()
            if bucket_nodes
        )
    else:
        rows = "".join(_node_row(n) for n in nodes)
        content_html = (
            f'<table class="node-table"><thead><tr>'
            f'<th>name</th><th>op</th><th>target</th><th>shape</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
        )

    return f"""
<div class="group-card" id="{card_id}">
  <div class="group-header" onclick="toggleCard('{card_id}')">
    <span class="toggle-icon" id="icon-{card_id}">▶</span>
    <span class="group-title">{title}</span>
    <span class="node-count">({n_nodes} nodes)</span>
    <span class="badges">{badges}</span>
  </div>
  <div class="group-body" id="body-{card_id}" style="display:none;">
    {content_html}
  </div>
</div>
"""


def render_html(parsed: dict, output_path: str) -> None:
    """Write the visualizer HTML to output_path."""
    groups = parsed["groups"]
    layer_indices = parsed["layer_indices"]
    total_nodes = parsed["total_nodes"]
    n_layers = len(layer_indices)

    # Build ordered group list: embed → layer_0..N → norm → lm_head → other
    ordered_groups = []
    for gname in ["embed", "other"]:
        if gname in groups and gname != "other":
            ordered_groups.append((gname, groups[gname], False))

    if "embed" in groups:
        ordered_groups.append(("embed", groups["embed"], False))

    for li in layer_indices:
        gname = f"layer_{li}"
        if gname in groups:
            ordered_groups.append((gname, groups[gname], True))

    for gname in ["norm", "lm_head"]:
        if gname in groups:
            ordered_groups.append((gname, groups[gname], False))

    if "other" in groups:
        ordered_groups.append(("other", groups["other"], False))

    # Build the legend
    legend_items = "".join(
        f'<span class="legend-item">'
        f'<span class="legend-dot" style="background:{CAT_COLORS[cat]};"></span>'
        f'{cat}</span>'
        for cat in ["linear", "norm", "activation", "attention", "embed", "arithmetic", "other"]
    )

    # Build all cards
    cards_html = "".join(
        _group_card(gname, gdata, is_layer)
        for gname, gdata, is_layer in ordered_groups
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FX Graph Visualizer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f1f5f9;
    color: #1e293b;
    padding: 16px;
  }}
  .summary-box {{
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }}
  .summary-box h1 {{
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #f8fafc;
  }}
  .summary-stats {{
    display: flex;
    gap: 24px;
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 10px;
  }}
  .summary-stats strong {{ color: #e2e8f0; }}
  .controls {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  .btn {{
    padding: 6px 14px;
    border: 1px solid #475569;
    border-radius: 6px;
    background: #334155;
    color: #e2e8f0;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.15s;
  }}
  .btn:hover {{ background: #475569; }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 8px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.82rem;
    color: #94a3b8;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .group-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
  }}
  .group-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    cursor: pointer;
    user-select: none;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    transition: background 0.12s;
  }}
  .group-header:hover {{ background: #f1f5f9; }}
  .toggle-icon {{
    font-size: 0.75rem;
    color: #64748b;
    width: 12px;
    transition: transform 0.2s;
  }}
  .toggle-icon.open {{ transform: rotate(90deg); }}
  .group-title {{
    font-weight: 600;
    font-size: 0.95rem;
    color: #0f172a;
  }}
  .node-count {{
    font-size: 0.8rem;
    color: #94a3b8;
  }}
  .badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-left: auto;
  }}
  .badge {{
    font-size: 0.72rem;
    padding: 2px 7px;
    border-radius: 99px;
    font-weight: 500;
  }}
  .group-body {{
    padding: 12px 16px;
  }}
  .subsection {{
    margin-bottom: 14px;
  }}
  .subsection-title {{
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e2e8f0;
  }}
  .node-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }}
  .node-table th {{
    text-align: left;
    padding: 5px 8px;
    background: #f1f5f9;
    color: #64748b;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #e2e8f0;
  }}
  .node-table td {{
    padding: 4px 8px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
  }}
  .node-table tr:last-child td {{ border-bottom: none; }}
  .node-name {{
    font-family: 'Consolas', 'Fira Code', monospace;
    color: #1e293b;
    white-space: nowrap;
  }}
  .op-tag {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
  }}
  .target {{
    font-family: monospace;
    color: #475569;
    font-size: 0.76rem;
    word-break: break-all;
  }}
  .shape {{
    font-family: monospace;
    color: #94a3b8;
    font-size: 0.75rem;
  }}
  .flow-line {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #64748b;
    margin: 4px 0 8px;
    flex-wrap: wrap;
  }}
  .flow-arrow {{
    color: #94a3b8;
  }}
  .flow-block {{
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
    font-size: 0.78rem;
  }}
  @media (max-width: 600px) {{
    .badges {{ display: none; }}
  }}
</style>
</head>
<body>

<div class="summary-box">
  <h1>FX Graph Visualizer</h1>
  <div class="summary-stats">
    <span>Total nodes: <strong>{total_nodes}</strong></span>
    <span>Transformer layers: <strong>{n_layers}</strong></span>
    <span>Groups: <strong>{len(ordered_groups)}</strong></span>
  </div>
  <div class="controls">
    <button class="btn" onclick="expandAll()">Expand All</button>
    <button class="btn" onclick="collapseAll()">Collapse All</button>
  </div>
  <div class="legend">{legend_items}</div>
</div>

<div id="cards-container">
{cards_html}
</div>

<script>
function toggleCard(id) {{
  var body = document.getElementById('body-' + id);
  var icon = document.getElementById('icon-' + id);
  if (body.style.display === 'none') {{
    body.style.display = 'block';
    icon.classList.add('open');
  }} else {{
    body.style.display = 'none';
    icon.classList.remove('open');
  }}
}}

function expandAll() {{
  document.querySelectorAll('.group-body').forEach(function(b) {{
    b.style.display = 'block';
  }});
  document.querySelectorAll('.toggle-icon').forEach(function(i) {{
    i.classList.add('open');
  }});
}}

function collapseAll() {{
  document.querySelectorAll('.group-body').forEach(function(b) {{
    b.style.display = 'none';
  }});
  document.querySelectorAll('.toggle-icon').forEach(function(i) {{
    i.classList.remove('open');
  }});
}}
</script>
</body>
</html>
"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"      Wrote {output_path}  ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM FX graph as block-level collapsible HTML"
    )
    parser.add_argument(
        "--model",
        default="../eval_ppl/models/Qwen2.5-0.5B-random",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length for export (default: 16)")
    parser.add_argument("--output", default="fx_graph.html",
                        help="Output HTML path (default: fx_graph.html)")
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Show only the first N layers (default: all)",
    )
    args = parser.parse_args()

    gm, ep = load_and_export(args.model, args.seq_len)
    parsed = parse_graph(gm, max_layers=args.layers)
    render_html(parsed, args.output)

    print()
    print(f"Done!  Layers: {len(parsed['layer_indices'])},  "
          f"Total nodes: {parsed['total_nodes']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
