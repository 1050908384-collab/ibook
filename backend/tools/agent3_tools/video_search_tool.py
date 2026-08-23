# -*- coding: utf-8 -*-
"""个性化学习资源生成 Agent video lookup backed by the canonical video resource catalogue.

The old implementation parsed a Markdown file with a hard-coded block-ID map.
That map drifted after the curriculum was expanded.  Both 个性化学习资源生成 Agent paths now use
the same catalogue that is imported into SQLite, so returned URLs are stable
``canonical_url`` values and remain aligned with the knowledge graph.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "video_resources.json"
BLOCKS_PATH = PROJECT_ROOT / "data" / "seeds" / "knowledge_blocks.json"
_CACHE: list[dict[str, Any]] | None = None


def _normalise(value: str) -> str:
    return value.replace("（", "(").replace("）", ")").strip().lower()


def _load_videos() -> list[dict[str, Any]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        resources = {item["resource_id"]: item for item in catalog.get("resources", [])}
        blocks = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        _CACHE = []
        return _CACHE

    block_names = {item["block_id"]: item["block_name"] for item in blocks}
    videos: list[dict[str, Any]] = []
    for scope, mappings in (
        ("subtopic", catalog.get("subtopic_mappings", [])),
        ("block", catalog.get("block_mappings", [])),
    ):
        for mapping in mappings:
            if mapping.get("mapping_status") != "confirmed":
                continue
            resource = resources.get(mapping.get("resource_id"))
            if not resource or resource.get("resource_type") != "video":
                continue
            block_id = mapping.get("block_id", "")
            videos.append({
                "video_id": resource["resource_id"],
                "resource_id": resource["resource_id"],
                "block_id": block_id,
                "block_name": block_names.get(block_id, ""),
                "subtopic_id": mapping.get("subtopic_id"),
                "topic": mapping.get("mapping_notes", ""),
                "title": resource["title"],
                "url": resource["canonical_url"],
                "duration_hours": 0.0,
                "source_type": "video_catalog",
                "scope": scope,
                "role": mapping.get("role", "supplementary"),
                "sequence_index": mapping.get("sequence_index", 1),
                "relevance_score": mapping.get("relevance_score", 0.0),
            })
    _CACHE = videos
    return _CACHE


def search_videos(
    block_id: str | None = None,
    block_name: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return confirmed canonical video URLs for a graph block."""
    results = _load_videos()
    if block_id:
        results = [item for item in results if item["block_id"] == block_id]
    if block_name:
        wanted = _normalise(block_name)
        results = [item for item in results if _normalise(item["block_name"]) == wanted]
    unique: dict[str, dict[str, Any]] = {}
    for item in sorted(
        results,
        key=lambda row: (0 if row["scope"] == "block" else 1, 0 if row["role"] == "primary" else 1, row["sequence_index"]),
    ):
        unique.setdefault(item["resource_id"], item)
    return list(unique.values())[:top_k]


def search_videos_by_query(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    wanted = _normalise(query)
    matches = [
        item for item in _load_videos()
        if wanted in _normalise(item["title"])
        or wanted in _normalise(item["block_name"])
        or wanted in _normalise(item["topic"])
    ]
    return matches[:top_k]


def clear_cache() -> None:
    global _CACHE
    _CACHE = None


class VideoSearchTool:
    async def search_by_block(self, block_id: str = "", block_name: str = "", top_k: int = 5) -> list[dict[str, Any]]:
        return search_videos(block_id=block_id, block_name=block_name, top_k=top_k)

    async def search_by_query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return search_videos_by_query(query, top_k=top_k)
