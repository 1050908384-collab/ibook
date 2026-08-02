# -*- coding: utf-8 -*-
"""
视频资源检索工具 —— 从 知识清单-学习视频链接收集.md 解析视频链接，
按知识板块（block_id / block_name）检索对应视频资源列表。

数据来源: data/知识库原始数据/知识清单-学习视频链接收集.md
格式:
  ###### **板块名称-总时长**
  **子主题-时长**
  【视频标题】URL

解析后返回统一结构:
{
  "video_id": "vid_01",
  "block_id": "KB-01",
  "block_name": "Linux环境与命令行操作",
  "topic": "常用Shell命令",
  "title": "【2026年版】速学Linux核心命令",
  "url": "https://www.bilibili.com/video/BV1iwgY6jEjs",
  "duration_hours": 1.0,
  "source_type": "video"
}
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VIDEO_MD_PATH = None
_BLOCK_NAME_TO_ID = {
    "Linux环境与命令行操作": "KB-01",
    "Python编程基础": "KB-02",
    "深度学习框架PyTorch基础": "KB-03",
    "自然语言处理（NLP）基础概念": "KB-04",
    "注意力机制": "KB-05",
    "子词切分与分词器（Tokenizer）": "KB-06",
    "Transformer架构（Decoder-Only）": "KB-07",
    "HuggingFace生态工具链": "KB-08",
    "参数高效微调（LoRA与QLoRA）": "KB-12",
    "参数高效微调(LoRA与QLoRA)": "KB-12",
    "PyTorch进阶训练技巧": "KB-13",
    "分布式训练（DeepSpeed）": "KB-14",
    "模型推理部署（vLLM）": "KB-15",
}


def _get_video_md_path() -> Path:
    global _VIDEO_MD_PATH
    if _VIDEO_MD_PATH is not None:
        return _VIDEO_MD_PATH
    here = Path(__file__).resolve()
    root = here.parent.parent.parent.parent
    candidate = root / "data" / "知识库原始数据" / "知识清单-学习视频链接收集.md"
    if candidate.exists():
        _VIDEO_MD_PATH = candidate
        return _VIDEO_MD_PATH
    alt = Path(__file__).resolve()
    for _ in range(6):
        alt = alt.parent
        p = alt / "data" / "知识库原始数据" / "知识清单-学习视频链接收集.md"
        if p.exists():
            _VIDEO_MD_PATH = p
            return p
    logger.warning("视频清单 Markdown 未找到: %s", candidate)
    return candidate


def _parse_duration(text: str) -> float:
    """从 '1.5h' / '30min' / '1小时' 等格式解析小时数。"""
    text = text.strip().lower()
    m = re.search(r"([\d.]+)\s*h", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*(?:hour|小时|hr)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*min", text)
    if m:
        return float(m.group(1)) / 60.0
    return 0.0


def _parse_video_markdown(md_path: Path) -> List[Dict[str, Any]]:
    """解析视频 Markdown，返回所有视频条目列表。"""
    if not md_path.exists():
        logger.warning("视频清单文件不存在: %s", md_path)
        return []

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    videos: List[Dict[str, Any]] = []
    current_block_name = ""
    current_block_id = ""
    current_topic = ""
    current_topic_hours = 0.0
    video_idx = 0

    block_header_re = re.compile(r"^######\s+\*\*(.+?)\*\*")
    topic_re = re.compile(r"^\*\*(.+?)\*\*")
    video_line_re = re.compile(r"^【(.+?)】(https?://\S+)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Block header: ###### **Linux环境与命令行操作-4h**
        m = block_header_re.match(line)
        if m:
            raw = m.group(1)
            # Extract duration if present: "名称-4h" or "名称-4.5h"
            duration_match = re.search(r"-([\d.]+)h$", raw)
            if duration_match:
                current_block_name = raw[: duration_match.start()].strip()
                current_topic = ""
            else:
                current_block_name = raw
                current_topic = ""
            current_block_id = _BLOCK_NAME_TO_ID.get(current_block_name, "")
            continue

        # Topic: **理解LoRA的低秩分解原理-1.5h**
        m = topic_re.match(line)
        if m and line.startswith("**(") or (m and not line.startswith("【")):
            raw = m.group(1)
            duration_match = re.search(r"-([\d.]+)h$", raw)
            if duration_match:
                current_topic = raw[: duration_match.start()].strip()
                current_topic_hours = float(duration_match.group(1))
            else:
                current_topic = raw
                current_topic_hours = 0.0
            continue

        # Video: 【视频标题】https://...
        m = video_line_re.match(line)
        if m and current_block_name:
            title = m.group(1).strip()
            url = m.group(2).strip().rstrip("\\")
            # Remove strikethrough markers from title
            if title.startswith("~~") and title.endswith("~~"):
                title = title[2:-2].strip()
            if not title or not url:
                continue
            video_idx += 1
            videos.append({
                "video_id": f"vid_{current_block_id}_{video_idx:03d}" if current_block_id else f"vid_unknown_{video_idx:03d}",
                "block_id": current_block_id or "",
                "block_name": current_block_name,
                "topic": current_topic,
                "title": title,
                "url": url,
                "duration_hours": current_topic_hours,
                "source_type": "video",
            })

    logger.info("解析视频清单完成: %d 条视频", len(videos))
    return videos


_CACHE: Optional[List[Dict[str, Any]]] = None


def _get_all_videos() -> List[Dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    md_path = _get_video_md_path()
    _CACHE = _parse_video_markdown(md_path)
    return _CACHE


def search_videos(
    block_id: Optional[str] = None,
    block_name: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """按 block_id 或 block_name 检索视频资源。"""
    all_videos = _get_all_videos()
    if not all_videos:
        return []

    results = all_videos
    if block_id:
        results = [v for v in results if v["block_id"] == block_id]
    if block_name:
        bn_norm = block_name.replace("（", "(").replace("）", ")").strip()
        results = [
            v for v in results
            if bn_norm in v["block_name"].replace("（", "(").replace("）", ")")
            or v["block_name"].replace("（", "(").replace("）", ")") in bn_norm
        ]

    return results[:top_k]


def search_videos_by_query(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """按关键词检索视频资源。"""
    all_videos = _get_all_videos()
    if not all_videos:
        return []

    query_lower = query.lower()
    scored: List[tuple[float, Dict[str, Any]]] = []
    for v in all_videos:
        score = 0.0
        if query_lower in v["title"].lower():
            score += 10.0
        if query_lower in v["topic"].lower():
            score += 5.0
        if query_lower in v["block_name"].lower():
            score += 3.0
        if v["block_id"] and query_lower in v["block_id"].lower():
            score += 2.0
        if score > 0:
            scored.append((score, v))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in scored[:top_k]]


def clear_cache() -> None:
    global _CACHE
    _CACHE = None


class VideoSearchTool:
    """Agent3 视频检索工具，供 deterministic / hybrid Agent 调用。"""

    def __init__(self) -> None:
        pass

    async def search_by_block(
        self, block_id: str = "", block_name: str = "", top_k: int = 5
    ) -> List[Dict[str, Any]]:
        return search_videos(block_id=block_id, block_name=block_name, top_k=top_k)

    async def search_by_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return search_videos_by_query(query, top_k=top_k)
