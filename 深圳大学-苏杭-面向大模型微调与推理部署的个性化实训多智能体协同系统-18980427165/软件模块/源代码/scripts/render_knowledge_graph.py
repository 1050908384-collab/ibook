"""Render the current knowledge graph as a self-contained static SVG page."""
from __future__ import annotations


# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------

import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import json
from html import escape
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
BLOCKS_PATH = ROOT_DIR / "data" / "seeds" / "knowledge_blocks.json"
SUBTOPICS_PATH = ROOT_DIR / "data" / "seeds" / "knowledge_subtopics.json"
OUTPUT_PATH = ROOT_DIR / "docs" / "知识图谱可视化.html"


def _shorten(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[:limit - 1]}..."


def _rect(x: int, y: int, width: int, height: int, css_class: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="4" class="{css_class}"/>'


def _text(x: int, y: int, value: str, css_class: str) -> str:
    return f'<text x="{x}" y="{y}" class="{css_class}">{escape(value)}</text>'


def render() -> Path:
    blocks = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    subtopics = json.loads(SUBTOPICS_PATH.read_text(encoding="utf-8"))
    topic_by_block = {
        block["block_id"]: sorted(
            [item for item in subtopics if item["block_id"] == block["block_id"]],
            key=lambda item: item["sequence_index"],
        )
        for block in blocks
    }

    column_width, column_gap, left, top = 218, 20, 28, 82
    card_height, card_gap = 42, 12
    width = left * 2 + len(blocks) * column_width + (len(blocks) - 1) * column_gap
    height = top + 60 + max(len(items) for items in topic_by_block.values()) * (card_height + card_gap) + 40
    block_positions: dict[str, tuple[int, int]] = {}
    topic_positions: dict[str, tuple[int, int]] = {}
    edge_parts: list[str] = []
    node_parts: list[str] = []

    for index, block in enumerate(blocks):
        x = left + index * (column_width + column_gap)
        block_positions[block["block_id"]] = (x + column_width // 2, top)

    for block in blocks:
        target_x, target_y = block_positions[block["block_id"]]
        for prerequisite_id in block.get("prerequisites", []):
            source = block_positions.get(prerequisite_id)
            if source:
                source_x, source_y = source
                edge_parts.append(
                    f'<path d="M {source_x} {source_y - 22} C {source_x} 14, {target_x} 14, {target_x} {target_y - 22}" class="block-edge"/>'
                )

    for index, block in enumerate(blocks):
        x = left + index * (column_width + column_gap)
        node_parts.append(_rect(x, top - 32, column_width, 46, "block"))
        node_parts.append(_text(x + 9, top - 10, f"{block['block_id']} {_shorten(block['block_name'], 14)}", "block-label"))
        for topic_index, topic in enumerate(topic_by_block[block["block_id"]]):
            y = top + 34 + topic_index * (card_height + card_gap)
            topic_positions[topic["subtopic_id"]] = (x + column_width // 2, y + card_height // 2)
            node_parts.append(_rect(x, y, column_width, card_height, "install" if topic["sequence_index"] == 0 else "topic"))
            node_parts.append(_text(x + 8, y + 16, topic["subtopic_id"], "topic-id"))
            node_parts.append(_text(x + 8, y + 31, _shorten(topic["subtopic_name"], 27), "topic-label"))
            if topic_index:
                center_x = x + column_width // 2
                edge_parts.append(f'<path d="M {center_x} {y - card_gap} L {center_x} {y - 3}" class="sequence-edge"/>')

    for topic in subtopics:
        if topic["sequence_index"] != 0:
            continue
        target = topic_positions[topic["subtopic_id"]]
        for prerequisite_id in topic.get("prerequisites", []):
            source = topic_positions.get(prerequisite_id)
            if source:
                source_x, source_y = source
                target_x, target_y = target
                edge_parts.append(
                    f'<path d="M {source_x} {source_y} C {source_x} {source_y + 76}, {target_x} {target_y - 76}, {target_x} {target_y}" class="install-edge"/>'
                )

    svg = "\n".join(edge_parts + node_parts)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>知识图谱可视化</title>
<style>
body{{margin:0;background:#f4f6fa;color:#1f2937;font-family:Arial,"Microsoft YaHei",sans-serif}}
header{{padding:18px 24px;background:#fff;border-bottom:1px solid #d8dee9}} h1{{font-size:20px;margin:0 0 8px}}
.meta,.legend{{font-size:13px;color:#526070}} .legend{{display:flex;gap:18px;margin-top:10px;flex-wrap:wrap}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}} .viewport{{overflow:auto;height:calc(100vh - 116px);padding:20px}}
svg{{display:block;background:#fff;border:1px solid #d8dee9}} .block{{fill:#1d4ed8}} .block-label{{fill:#fff;font-size:13px;font-weight:bold}}
.topic{{fill:#f8fafc;stroke:#cbd5e1}} .install{{fill:#fff1f2;stroke:#d9485f;stroke-width:1.5}} .topic-id{{fill:#475569;font-size:10px}} .topic-label{{fill:#1e293b;font-size:11px}}
.block-edge{{fill:none;stroke:#64748b;stroke-width:1.5;stroke-dasharray:5 4}} .sequence-edge{{fill:none;stroke:#94a3b8;stroke-width:1.5}} .install-edge{{fill:none;stroke:#d9485f;stroke-width:2;stroke-dasharray:6 4}}
</style></head><body><header><h1>大模型微调与部署实训知识图谱</h1>
<div class="meta">15 个知识板块，73 个知识点。学习状态由运行时表 <code>subtopic_status</code> 保存：<code>-1</code> 未掌握，<code>0</code> 正在学习，<code>1</code> 已掌握。</div>
<div class="legend"><span><i class="dot" style="background:#1d4ed8"></i>知识板块</span><span><i class="dot" style="background:#fff1f2;border:1px solid #d9485f"></i>安装/环境准备</span><span>灰色虚线：板块先修</span><span>红色虚线：跨板块安装依赖</span><span>纵向连线：板块内顺序</span></div></header>
<main class="viewport"><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="知识图谱">{svg}</svg></main></body></html>"""
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(render())
