# -*- coding: utf-8 -*-
"""
knowledge_block_query_tool — 学情诊断与阶段规划 Agent 知识板块查询工具（原Agent2工具，已合并至学情诊断与阶段规划 Agent）
========================================================================================
功能：按技能名称/板块ID查询知识板块详情（含 estimated_hours、difficulty、
prerequisites、subtopics）。
    - 真实模式：主数据来自 knowledge_blocks 表，关系数据来自 graph_data
    - Mock 模式：返回 ./data/test_data/mock_knowledge_block_query_tool.json 预设板块数据
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

from config import BACKEND_DIR, DATA_DIR, settings
from db.graph_client import GraphClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_knowledge_block_query_tool.json"
BLOCKS_SEED_FILE = DATA_DIR / "seeds" / "knowledge_blocks.json"


def knowledge_block_query(input_data: dict, config: dict = None) -> dict:
    config = config or {}
    use_mock = config.get("use_mock", True)

    skill_name = (input_data or {}).get("skill_name") or (input_data or {}).get("block_id") or ""
    if not skill_name.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 skill_name 或 block_id 字段，无法查询",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_query(skill_name, input_data)

        block, db_available = _query_block(skill_name)
        return {
            "status": "success",
            "data": {
                "block": block,
                "db_available": db_available,
            },
            "message": _build_message(block),
            "mock_used": False,
        }
    except Exception as exc:
        logger.exception("knowledge_block_query 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"knowledge_block_query 执行异常: {exc}",
            "mock_used": use_mock,
        }


def _get_db_path() -> Path:
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    path = Path(raw)
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    return path


def _query_block(skill_name: str) -> tuple[Dict[str, Any] | None, bool]:
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，knowledge_block_query 无可用板块", db_path)
        return None, False

    try:
        graph_client = GraphClient()
        seed_index = _load_seed_index()
        block_id = _resolve_block_id(skill_name, seed_index) or skill_name
        graph_block = _find_block_in_graph(graph_client, block_id) or _find_block_in_graph(graph_client, skill_name)

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                SELECT block_id, block_name, description, estimated_hours,
                       difficulty, prerequisites, corresponding_gaps
                FROM knowledge_blocks
                WHERE block_id = ? OR block_name = ?
                """,
                (block_id, skill_name),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            logger.warning("未找到板块: %s", skill_name)
            return None, True

        block_id = graph_block.get("block_id", row[0]) if graph_block else row[0]
        prerequisite_names = _get_block_prerequisite_names(graph_client, block_id)
        subtopics = _get_ordered_subtopics(graph_client, block_id)

        gaps = []
        if row[6]:
            try:
                gaps = json.loads(row[6])
                if not isinstance(gaps, list):
                    gaps = []
            except json.JSONDecodeError:
                gaps = []

        return {
            "block_id": block_id,
            "block_name": seed_index.get(block_id, {}).get("block_name") or graph_block.get("block_name", row[1]) if graph_block else seed_index.get(block_id, {}).get("block_name", row[1]),
            "description": seed_index.get(block_id, {}).get("description") or graph_block.get("description", row[2]) if graph_block else seed_index.get(block_id, {}).get("description", row[2]),
            "estimated_hours": seed_index.get(block_id, {}).get("estimated_hours") or graph_block.get("estimated_hours", row[3]) if graph_block else seed_index.get(block_id, {}).get("estimated_hours", row[3]),
            "difficulty": seed_index.get(block_id, {}).get("difficulty") or graph_block.get("difficulty", row[4]) if graph_block else seed_index.get(block_id, {}).get("difficulty", row[4]),
            "prerequisites": prerequisite_names,
            "corresponding_gaps": gaps,
            "subtopics": subtopics,
        }, True
    except sqlite3.Error as exc:
        logger.warning("查询 knowledge_blocks 失败: %s", exc)
        return None, False


def _find_block_in_graph(graph_client: GraphClient, skill_name: str) -> dict[str, Any] | None:
    normalized = skill_name.strip()
    for node_id, node_data in graph_client.graph.nodes(data=True):
        if node_data.get("node_type") != "block":
            continue
        if node_id == normalized or node_data.get("block_name") == normalized:
            payload = dict(node_data)
            payload["block_id"] = node_id
            return payload
    return None


def _load_seed_index() -> dict[str, dict[str, Any]]:
    if not BLOCKS_SEED_FILE.exists():
        return {}
    try:
        rows = json.loads(BLOCKS_SEED_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        block_id = row.get("block_id")
        if block_id:
            index[block_id] = row
    return index


def _resolve_block_id(skill_name: str, seed_index: dict[str, dict[str, Any]]) -> str | None:
    normalized = skill_name.strip().lower()
    if not normalized:
        return None
    for block_id, row in seed_index.items():
        block_name = str(row.get("block_name", "")).strip().lower()
        gaps = [str(item).strip().lower() for item in row.get("corresponding_gaps", []) if str(item).strip()]
        if normalized == block_id.lower() or normalized == block_name:
            return block_id
        if normalized in gaps:
            return block_id
        if normalized and (normalized in block_name or any(normalized in gap for gap in gaps)):
            return block_id
    return None


def _get_block_prerequisite_names(graph_client: GraphClient, block_id: str) -> list[str]:
    names: list[str] = []
    for prerequisite_id in graph_client.get_prerequisites(block_id):
        node_data = graph_client.graph.nodes.get(prerequisite_id, {})
        names.append(node_data.get("block_name", prerequisite_id))
    return names


def _get_ordered_subtopics(graph_client: GraphClient, block_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtopic_id in graph_client.get_contained_subtopics(block_id):
        node_data = dict(graph_client.graph.nodes.get(subtopic_id, {}))
        rows.append(
            {
                "subtopic_id": subtopic_id,
                "subtopic_name": node_data.get("subtopic_name", subtopic_id),
                "sequence_index": node_data.get("sequence_index", 0),
                "estimated_hours": node_data.get("estimated_hours", 0),
                "difficulty": node_data.get("difficulty", ""),
            }
        )
    return sorted(rows, key=lambda item: (item["sequence_index"], item["subtopic_id"]))


def _build_message(block: Dict[str, Any] | None) -> str:
    if not block:
        return "未找到匹配的知识板块"
    return f"找到板块: {block['block_name']}（难度: {block['difficulty']}，时长: {block['estimated_hours']}小时）"


def _mock_query(skill_name: str, input_data: dict) -> dict:
    scenario = (input_data or {}).get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    data = dict(scenarios[scenario])
    data.setdefault("block", None)
    data.setdefault("db_available", False)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] {_build_message(data.get('block'))}",
        "mock_used": True,
    }


def _load_mock() -> dict:
    if not MOCK_FILE.exists():
        logger.warning("Mock 文件不存在: %s", MOCK_FILE)
        return {}
    try:
        with MOCK_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取 Mock 文件失败: %s", exc)
        return {}


class KnowledgeBlockQueryTool:
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def query_block(self, skill_name: str) -> Dict[str, Any]:
        result = knowledge_block_query({"skill_name": skill_name}, config=self._default_config)
        if result.get("status") != "success":
            return {}
        return result["data"].get("block") or {}
