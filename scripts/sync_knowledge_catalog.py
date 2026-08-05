"""Synchronize the learning catalogue with the authoritative curriculum Markdown.

The Markdown file is the source of truth for names and durations.  This script
keeps stable IDs for existing teaching topics, creates a ``-00`` installation
node for every environment-preparation item, and removes legacy topics that
are not in the curriculum.  Installation nodes carry explicit cross-block
prerequisites so they participate in the DAG rather than being display-only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SEEDS_DIR = ROOT_DIR / "data" / "seeds"
CURRICULUM_FILE = ROOT_DIR / "知识清单-带学习时长版（新版）.md"
BLOCKS_FILE = SEEDS_DIR / "knowledge_blocks.json"
SUBTOPICS_FILE = SEEDS_DIR / "knowledge_subtopics.json"
DOCS_FILE = SEEDS_DIR / "knowledge_docs.json"
QUESTIONS_FILE = SEEDS_DIR / "test_questions.json"
VIDEO_CATALOG_FILE = ROOT_DIR / "data" / "knowledge_base" / "video_resources.json"
RELATIONSHIP_REPORT_FILE = ROOT_DIR / "docs" / "知识图谱关系说明.md"

HEADING_RE = re.compile(r"^\*\*(.+?)-(\d+(?:\.\d+)?)h\*\*$")
TOPIC_RE = re.compile(r"^(.*?)(?:-)?(\d+(?:\.\d+)?)h$")

# Each key is an installation node.  These dependencies represent the actual
# software/environment condition, not merely the display order of a block.
INSTALLATION_PREREQUISITES = {
    "KB-02": [],
    "KB-03": ["KBS-02-00"],
    "KB-04": ["KBS-01-04", "KBS-02-00"],
    "KB-07": ["KBS-02-00"],
    "KB-08": ["KBS-07-00"],
    "KB-11": ["KBS-07-00"],
    "KB-12": ["KBS-04-00", "KBS-11-00"],
    "KB-13": ["KBS-04-00"],
    "KB-14": ["KBS-13-00"],
    "KB-15": ["KBS-04-00", "KBS-11-00"],
}

# These legacy nodes were added outside the authoritative curriculum.  Keep
# historical learning evidence by moving it to the final valid topic of the
# same block before the nodes are removed from the database.
RETIRED_SUBTOPIC_REPLACEMENTS = {
    "KBS-01-05": "KBS-01-04",
    "KBS-02-06": "KBS-02-05",
    "KBS-03-05": "KBS-03-04",
    "KBS-09-06": "KBS-09-05",
    "KBS-10-04": "KBS-10-03",
    "KBS-12-06": "KBS-12-05",
}


def _normalise(text: str) -> str:
    return text.replace("\\_", "_").strip()


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_stable_id(
    block_id: str,
    topic_name: str,
    stable_ids_by_name: dict[tuple[str, str], str],
) -> str | None:
    """Match a concise legacy title to its expanded curriculum title."""
    direct_match = stable_ids_by_name.get((block_id, topic_name))
    if direct_match:
        return direct_match
    candidates = [
        subtopic_id
        for (candidate_block_id, candidate_name), subtopic_id in stable_ids_by_name.items()
        if candidate_block_id == block_id
        and (topic_name.startswith(candidate_name) or candidate_name.startswith(topic_name))
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _replace_subtopic_references(path: Path, rows: list[dict]) -> int:
    changed = 0
    for row in rows:
        old_id = row.get("subtopic_id")
        replacement_id = RETIRED_SUBTOPIC_REPLACEMENTS.get(old_id)
        if replacement_id:
            row["subtopic_id"] = replacement_id
            changed += 1
    if changed:
        _write_json(path, rows)
    return changed


def _sync_references() -> dict[str, int]:
    docs = _load_json(DOCS_FILE)
    questions = _load_json(QUESTIONS_FILE)
    video_catalog = json.loads(VIDEO_CATALOG_FILE.read_text(encoding="utf-8"))
    video_mappings = video_catalog.get("subtopic_mappings", [])
    changed = {
        "knowledge_docs": _replace_subtopic_references(DOCS_FILE, docs),
        "test_questions": _replace_subtopic_references(QUESTIONS_FILE, questions),
        "video_mappings": _replace_subtopic_references(VIDEO_CATALOG_FILE, video_mappings),
    }
    if changed["video_mappings"]:
        _write_json(VIDEO_CATALOG_FILE, video_catalog)
    return changed


def _write_relationship_report(blocks: list[dict], subtopics: list[dict]) -> None:
    """Export direct edges in a review-friendly form for curriculum editing."""
    block_names = {item["block_id"]: item["block_name"] for item in blocks}
    topic_names = {item["subtopic_id"]: item["subtopic_name"] for item in subtopics}
    lines = [
        "# 知识图谱关系说明",
        "",
        "本文件由 `scripts/sync_knowledge_catalog.py` 生成。关系均为直接边；传递依赖由运行时图谱自动计算。",
        "",
        "## 关系类型",
        "",
        "- `CONTAINS`：知识板块包含知识点。",
        "- `PROGRESSES_TO`：同一板块中按 `sequence_index` 的相邻学习顺序。",
        "- `PREREQUISITE_OF`：源节点必须先完成，目标节点才可学习。",
        "",
        "## 板块级直接先修",
        "",
    ]
    for block in blocks:
        dependencies = block.get("prerequisites", [])
        if dependencies:
            names = "、".join(f"{item}（{block_names[item]}）" for item in dependencies)
            lines.append(f"- {block['block_id']}（{block['block_name']}） <- {names}")
        else:
            lines.append(f"- {block['block_id']}（{block['block_name']}）：无直接板块先修")

    lines.extend(["", "## 安装节点的跨板块直接先修", ""])
    for subtopic in subtopics:
        if subtopic["sequence_index"] != 0:
            continue
        dependencies = subtopic.get("prerequisites", [])
        if dependencies:
            names = "、".join(f"{item}（{topic_names[item]}）" for item in dependencies)
            lines.append(f"- {subtopic['subtopic_id']}（{subtopic['subtopic_name']}） <- {names}")
        else:
            lines.append(f"- {subtopic['subtopic_id']}（{subtopic['subtopic_name']}）：起始环境节点")

    lines.extend(["", "## 板块内顺序", ""])
    for block in blocks:
        chain = sorted(
            (item for item in subtopics if item["block_id"] == block["block_id"]),
            key=lambda item: item["sequence_index"],
        )
        nodes = " -> ".join(f"{item['subtopic_id']}（{item['subtopic_name']}）" for item in chain)
        lines.append(f"- {block['block_id']}：{nodes}")
    RELATIONSHIP_REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_curriculum() -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None
    for raw_line in CURRICULUM_FILE.read_text(encoding="utf-8").splitlines():
        line = _normalise(raw_line)
        if not line:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            current = {"name": heading.group(1), "hours": float(heading.group(2)), "topics": []}
            blocks.append(current)
            continue
        if current is None:
            continue
        topic = TOPIC_RE.match(line)
        if not topic:
            continue
        current["topics"].append(
            {"name": topic.group(1).rstrip("-").strip(), "hours": float(topic.group(2))}
        )
    if len(blocks) != 15:
        raise ValueError(f"课程清单应包含 15 个知识板块，实际解析到 {len(blocks)} 个")
    return blocks


def sync() -> dict:
    current_blocks = _load_json(BLOCKS_FILE)
    current_subtopics = _load_json(SUBTOPICS_FILE)
    curriculum = _parse_curriculum()

    blocks_by_id = {row["block_id"]: row for row in current_blocks}
    stable_ids_by_name = {
        (row["block_id"], _normalise(row["subtopic_name"])): row["subtopic_id"]
        for row in current_subtopics
    }
    synced_blocks: list[dict] = []
    synced_subtopics: list[dict] = []

    for position, source_block in enumerate(curriculum, start=1):
        block_id = f"KB-{position:02d}"
        existing_block = blocks_by_id[block_id]
        if _normalise(existing_block["block_name"]) != source_block["name"]:
            raise ValueError(f"{block_id} 与新版课程清单的板块名称不一致")

        topics = source_block["topics"]
        installation = next((topic for topic in topics if topic["name"].startswith("（环境准备：")), None)
        lesson_topics = [topic for topic in topics if topic is not installation]
        if sum(topic["hours"] for topic in topics) != source_block["hours"]:
            raise ValueError(f"{block_id} 的知识点时长之和与板块时长不一致")

        description = "\n".join(f"- {topic['name']}-{topic['hours']:g}h" for topic in topics)
        synced_blocks.append({**existing_block, "description": description, "estimated_hours": source_block["hours"]})

        previous_id: str | None = None
        if installation is not None:
            installation_id = f"KBS-{position:02d}-00"
            synced_subtopics.append(
                {
                    "subtopic_id": installation_id,
                    "block_id": block_id,
                    "subtopic_name": installation["name"],
                    "estimated_hours": installation["hours"],
                    "sequence_index": 0,
                    "difficulty": existing_block["difficulty"],
                    "prerequisites": INSTALLATION_PREREQUISITES.get(block_id, []),
                    "content_summary": installation["name"],
                }
            )
            previous_id = installation_id

        for sequence_index, topic in enumerate(lesson_topics, start=1):
            stable_id = _find_stable_id(block_id, topic["name"], stable_ids_by_name)
            if stable_id is None:
                raise ValueError(f"{block_id} 缺少已有知识点 ID: {topic['name']}")
            synced_subtopics.append(
                {
                    "subtopic_id": stable_id,
                    "block_id": block_id,
                    "subtopic_name": topic["name"],
                    "estimated_hours": topic["hours"],
                    "sequence_index": sequence_index,
                    "difficulty": existing_block["difficulty"],
                    "prerequisites": [previous_id] if previous_id else [],
                    "content_summary": topic["name"],
                }
            )
            previous_id = stable_id

    _write_json(BLOCKS_FILE, synced_blocks)
    _write_json(SUBTOPICS_FILE, synced_subtopics)
    reference_changes = _sync_references()
    _write_relationship_report(synced_blocks, synced_subtopics)
    return {
        "blocks": len(synced_blocks),
        "subtopics": len(synced_subtopics),
        "installation_nodes": sum(item["sequence_index"] == 0 for item in synced_subtopics),
        "total_hours": sum(item["estimated_hours"] for item in synced_blocks),
        "reference_changes": reference_changes,
    }


if __name__ == "__main__":
    print(json.dumps(sync(), ensure_ascii=False, indent=2))
