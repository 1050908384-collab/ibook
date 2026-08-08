"""Import reviewed local Markdown lessons into the runtime knowledge database.

The source directory remains the editable, version-controlled source of truth.
This importer owns only ``LOCAL-*`` documents and their chunks, so existing seed
documents, videos, questions, and user data are left untouched.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "知识库原始数据"
SUBTOPICS_FILE = PROJECT_ROOT / "data" / "seeds" / "knowledge_subtopics.json"
DATABASE_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"
ACTIVE_BLOCKS = {f"KB-{number:02d}" for number in (*range(1, 7), *range(9, 13))}


@dataclass(frozen=True)
class LessonFile:
    path: Path
    block_id: str
    subtopic_id: str
    difficulty: str


def _directory_order(name: str) -> int:
    match = re.match(r"(\d+)(?:[_.、]|$)", name)
    if not match:
        raise ValueError(f"目录缺少数字顺序前缀: {name}")
    return int(match.group(1))


def _load_subtopics() -> dict[str, list[dict]]:
    records = json.loads(SUBTOPICS_FILE.read_text(encoding="utf-8"))
    by_block: dict[str, list[dict]] = {}
    for record in records:
        if record["block_id"] in ACTIVE_BLOCKS:
            by_block.setdefault(record["block_id"], []).append(record)
    for records_in_block in by_block.values():
        records_in_block.sort(key=lambda item: item["sequence_index"])
    return by_block


def discover_lessons() -> list[LessonFile]:
    subtopics_by_block = _load_subtopics()
    lessons: list[LessonFile] = []

    for block_dir in sorted(SOURCE_DIR.iterdir()):
        if not block_dir.is_dir():
            continue
        block_number = _directory_order(block_dir.name)
        block_id = f"KB-{block_number:02d}"
        if block_id not in ACTIVE_BLOCKS:
            continue
        subtopics = subtopics_by_block.get(block_id, [])
        for subtopic_dir in sorted((item for item in block_dir.iterdir() if item.is_dir()), key=lambda item: _directory_order(item.name)):
            index = _directory_order(subtopic_dir.name) - 1
            if index < 0 or index >= len(subtopics):
                raise ValueError(f"{subtopic_dir} 无法映射到 {block_id} 的子知识点")
            markdown_files = sorted(subtopic_dir.glob("*.md"))
            if len(markdown_files) != 1:
                raise ValueError(f"{subtopic_dir} 应包含且只包含一个 Markdown 文件")
            markdown_file = markdown_files[0]
            if markdown_file.stat().st_size == 0:
                continue
            lessons.append(LessonFile(
                path=markdown_file,
                block_id=block_id,
                subtopic_id=subtopics[index]["subtopic_id"],
                difficulty=subtopics[index].get("difficulty") or "中级",
            ))
    return lessons


def _title_and_content(path: Path) -> tuple[str, str]:
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError(f"资料为空: {path}")
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip(), content
    return path.stem, content


def _reference_url(content: str) -> str:
    """Use the first curated external link as the learner-facing article URL."""
    match = re.search(r"https?://[^\s)>\]}`]+", content)
    return match.group(0).rstrip(".,，。;；") if match else ""


def _chunk_markdown(content: str, size: int = 1500) -> list[tuple[str, str]]:
    """Split at paragraph boundaries without splitting fenced code blocks."""
    units: list[tuple[str, str]] = []
    buffer: list[str] = []
    in_code_block = False
    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        buffer.append(line)
        if not in_code_block and not line.strip():
            units.append(("code" if any(item.lstrip().startswith("```") for item in buffer) else "text", "".join(buffer).strip()))
            buffer = []
    if buffer:
        units.append(("code" if any(item.lstrip().startswith("```") for item in buffer) else "text", "".join(buffer).strip()))

    chunks: list[tuple[str, str]] = []
    current_type = "text"
    current = ""
    for unit_type, unit in units:
        if not unit:
            continue
        if current and len(current) + len(unit) + 2 > size:
            chunks.append((current_type, current))
            current = ""
        if not current:
            current_type = unit_type
        current = f"{current}\n\n{unit}".strip() if current else unit
    if current:
        chunks.append((current_type, current))
    return chunks


def import_lessons(database_path: Path | None = None) -> tuple[int, int]:
    lessons = discover_lessons()
    if not lessons:
        raise RuntimeError("未发现可导入的本地学习资料")

    doc_count = 0
    chunk_count = 0
    connection = sqlite3.connect(database_path or DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            for lesson in lessons:
                title, content = _title_and_content(lesson.path)
                doc_id = f"LOCAL-{lesson.subtopic_id}"
                source_url = _reference_url(content)

                connection.execute("DELETE FROM knowledge_chunks WHERE doc_id = ?", (doc_id,))
                connection.execute("""
                    INSERT INTO knowledge_docs (
                        doc_id, block_id, subtopic_id, title, content, doc_type, source, difficulty
                    ) VALUES (
                        ?, ?, ?, ?, ?, 'tutorial', ?, ?
                    )
                    ON CONFLICT(doc_id) DO UPDATE SET
                        block_id = excluded.block_id,
                        subtopic_id = excluded.subtopic_id,
                        title = excluded.title,
                        content = excluded.content,
                        source = excluded.source,
                        difficulty = excluded.difficulty
                """, (doc_id, lesson.block_id, lesson.subtopic_id, title, content,
                         source_url, lesson.difficulty))
                for index, (chunk_type, chunk_content) in enumerate(_chunk_markdown(content), start=1):
                    chunk_id = f"{doc_id}-C{index:03d}"
                    connection.execute("""
                        INSERT INTO knowledge_chunks (
                            chunk_id, doc_id, block_id, chunk_index, content, chunk_type, char_start, char_end
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, NULL, NULL
                        )
                    """, (chunk_id, doc_id, lesson.block_id, index, chunk_content, chunk_type))
                    chunk_count += 1
                doc_count += 1
    finally:
        connection.close()
    return doc_count, chunk_count


if __name__ == "__main__":
    documents, chunks = import_lessons()
    print(f"[DONE] 已导入 {documents} 个本地课程文档，生成 {chunks} 个知识分块。")
