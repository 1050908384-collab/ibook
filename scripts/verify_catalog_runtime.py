"""Smoke-check the graph and video paths used by 学情诊断与阶段规划 Agent and 个性化学习资源生成 Agent."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path[:0] = [str(ROOT_DIR), str(BACKEND_DIR)]

from tools.agent1_tools.graph_query_tool import GraphQueryTool
from tools.agent3_tools.video_search_tool import VideoSearchTool
from tools.learning_resource_tool import LearningResourceTool


async def verify() -> dict:
    graph_tool = GraphQueryTool()
    direct_dependencies = await graph_tool.query_direct_prerequisites("KB-12")
    dependency_ids = {item["block_id"] for item in direct_dependencies}
    assert dependency_ids == {"KB-04", "KB-11"}, dependency_ids

    subtopics = json.loads((ROOT_DIR / "data" / "seeds" / "knowledge_subtopics.json").read_text(encoding="utf-8"))
    blocks = json.loads((ROOT_DIR / "data" / "seeds" / "knowledge_blocks.json").read_text(encoding="utf-8"))
    db_videos = await LearningResourceTool().get_recommended_videos("KB-12", "KBS-12-00")
    catalog_videos = await VideoSearchTool().search_by_block("KB-12", top_k=5)
    assert db_videos, "数据库未返回 LoRA/QLoRA 安装节点视频"
    assert catalog_videos, "目录兜底未返回 LoRA/QLoRA 板块视频"
    assert len({video["resource_id"] for video in db_videos}) == len(db_videos), "数据库视频存在重复推荐"
    for video in db_videos + catalog_videos:
        assert str(video.get("url", "")).startswith("https://"), video

    checked_blocks = 0
    for block in blocks:
        block_id = block["block_id"]
        first_subtopic = min(
            (item for item in subtopics if item["block_id"] == block_id),
            key=lambda item: item["sequence_index"],
        )
        stage_videos = await LearningResourceTool().get_recommended_videos(block_id, first_subtopic["subtopic_id"])
        fallback_videos = await VideoSearchTool().search_by_block(block_id, top_k=1)
        assert stage_videos, f"数据库未返回 {block_id} 阶段视频"
        assert fallback_videos, f"目录兜底未返回 {block_id} 视频"
        checked_blocks += 1

    return {
        "agent1_direct_prerequisites": sorted(dependency_ids),
        "agent3_database_video_count": len(db_videos),
        "agent3_catalog_video_count": len(catalog_videos),
        "sample_url": db_videos[0]["url"],
        "agent3_verified_blocks": checked_blocks,
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))
