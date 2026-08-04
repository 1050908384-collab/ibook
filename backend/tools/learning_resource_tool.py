# -*- coding: utf-8 -*-
"""学习资源的只读查询与审核辅助。"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from config import BACKEND_DIR, settings


def _get_db_path() -> Path:
    raw_path = settings.DATABASE_URL.replace("sqlite:///", "", 1)
    path = Path(raw_path)
    return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


class LearningResourceTool:
    """按学习阶段读取已确认的视频资源，数据库异常时安全降级为空列表。"""

    async def get_recommended_videos(self, block_id: str, subtopic_id: str | None) -> list[dict[str, Any]]:
        if not block_id:
            return []
        try:
            with sqlite3.connect(f"file:{_get_db_path()}?mode=ro", uri=True) as connection:
                block_rows = connection.execute(
                    """
                    SELECT r.resource_id, r.title, r.canonical_url, r.content_summary,
                           r.review_status, br.block_id, NULL, br.role,
                           br.sequence_index, br.relevance_score, br.mapping_status, 'block'
                    FROM block_resources AS br
                    JOIN learning_resources AS r ON r.resource_id = br.resource_id
                    WHERE br.block_id = ?
                      AND br.mapping_status = 'confirmed'
                      AND r.resource_type = 'video'
                    """,
                    (block_id,),
                ).fetchall()
                subtopic_rows = []
                if subtopic_id:
                    subtopic_rows = connection.execute(
                        """
                        SELECT r.resource_id, r.title, r.canonical_url, r.content_summary,
                               r.review_status, sr.block_id, sr.subtopic_id, sr.role,
                               sr.sequence_index, sr.relevance_score, sr.mapping_status, 'subtopic'
                        FROM subtopic_resources AS sr
                        JOIN learning_resources AS r ON r.resource_id = sr.resource_id
                        WHERE sr.block_id = ?
                          AND sr.subtopic_id = ?
                          AND sr.mapping_status = 'confirmed'
                          AND r.resource_type = 'video'
                        """,
                        (block_id, subtopic_id),
                    ).fetchall()
        except sqlite3.Error:
            return []

        rows = block_rows + subtopic_rows
        # A resource can intentionally be both a block fallback and a precise
        # subtopic mapping.  Return it once, preferring the precise mapping.
        rows.sort(key=lambda row: (0 if row[11] == "subtopic" else 1, 0 if row[7] == "primary" else 1, row[8], row[0]))
        unique_rows = []
        seen_resource_ids = set()
        for row in rows:
            if row[0] in seen_resource_ids:
                continue
            seen_resource_ids.add(row[0])
            unique_rows.append(row)
        return [
            {
                "resource_id": row[0],
                "title": row[1],
                "url": row[2],
                "content_summary": row[3],
                "review_status": row[4],
                "block_id": row[5],
                "subtopic_id": row[6],
                "role": row[7],
                "sequence_index": row[8],
                "relevance_score": row[9],
                "mapping_status": row[10],
                "scope": row[11],
            }
            for row in unique_rows
        ]

    async def validate_recommended_videos(
        self,
        videos: list[dict[str, Any]],
        block_id: str,
        subtopic_id: str | None,
    ) -> list[dict[str, Any]]:
        """校验 Agent3 输出的视频仍属于当前阶段，并提示未完成内容审核的资源。"""
        if not videos or not block_id:
            return []
        expected = {
            item["resource_id"]: item
            for item in await self.get_recommended_videos(block_id, subtopic_id)
        }
        errors: list[dict[str, Any]] = []
        for video in videos:
            resource_id = str(video.get("resource_id") or "")
            matched = expected.get(resource_id)
            if not resource_id or not matched:
                errors.append({
                    "error_type": "video_resource_mismatch",
                    "severity": "major",
                    "description": f"推荐视频 {resource_id or '未提供资源ID'} 不属于当前学习阶段。",
                    "suggestion": "仅返回当前 block_id 与 subtopic_id 已确认关联的视频资源。",
                    "source_ref": f"src_video_{resource_id}" if resource_id else None,
                    "change_request": {"required": True, "action": "replace_video_resource"},
                })
                continue
            if video.get("url") != matched["url"]:
                errors.append({
                    "error_type": "video_url_mismatch",
                    "severity": "major",
                    "description": f"推荐视频 {resource_id} 的 URL 与资源库记录不一致。",
                    "suggestion": "使用资源库中的 canonical_url，禁止由模型自行改写视频链接。",
                    "source_ref": f"src_video_{resource_id}",
                    "change_request": {"required": True, "action": "replace_video_url"},
                })
            if matched["review_status"] != "confirmed":
                errors.append({
                    "error_type": "video_content_unverified",
                    "severity": "minor",
                    "description": f"推荐视频 {resource_id} 尚未完成内容摘要或字幕审核。",
                    "suggestion": "补充人工摘要或字幕后，将资源 review_status 更新为 confirmed。",
                    "source_ref": f"src_video_{resource_id}",
                })
        return errors
