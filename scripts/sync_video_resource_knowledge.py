# -*- coding: utf-8 -*-
"""将已人工确认的视频摘要或字幕同步到 knowledge_docs 与 knowledge_chunks。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "video_resources.json"
DATABASE_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"
PLACEHOLDER = "导入后需补充视频内容摘要或字幕"


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    resources = {item["resource_id"]: item for item in catalog["resources"]}
    inserted_docs = 0
    skipped_resources = 0

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for mapping in catalog["subtopic_mappings"]:
            resource = resources[mapping["resource_id"]]
            summary = (resource.get("content_summary") or "").strip()
            transcript = (resource.get("metadata") or {}).get("transcript_text", "").strip()
            if (
                resource.get("review_status") != "confirmed"
                or mapping.get("mapping_status") != "confirmed"
                or not summary
                or PLACEHOLDER in summary
            ):
                skipped_resources += 1
                continue

            content = summary if not transcript else f"视频摘要：\n{summary}\n\n字幕/转写：\n{transcript}"
            doc_id = f"VID-{resource['resource_id']}-{mapping['subtopic_id']}"
            chunk_id = f"CHK-{doc_id}-001"
            connection.execute(
                """
                INSERT INTO knowledge_docs (
                    doc_id, block_id, subtopic_id, title, content, doc_type, source, difficulty
                ) VALUES (?, ?, ?, ?, ?, 'tutorial', ?, '初级')
                ON CONFLICT(doc_id) DO UPDATE SET
                    content=excluded.content, source=excluded.source, title=excluded.title
                """,
                (
                    doc_id, mapping["block_id"], mapping["subtopic_id"], resource["title"],
                    content, resource["canonical_url"],
                ),
            )
            connection.execute(
                """
                INSERT INTO knowledge_chunks (
                    chunk_id, doc_id, block_id, chunk_index, content, chunk_type, char_start, char_end
                ) VALUES (?, ?, ?, 0, ?, 'text', 0, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET content=excluded.content, char_end=excluded.char_end
                """,
                (chunk_id, doc_id, mapping["block_id"], content, len(content)),
            )
            inserted_docs += 1

    print(f"synced_docs={inserted_docs} skipped_unreviewed_mappings={skipped_resources}")


if __name__ == "__main__":
    main()
