# -*- coding: utf-8 -*-
"""
knowledge_block_query_tool — Agent2 知识板块查询工具
================================================
功能：按技能名称/板块ID查询知识板块详情（含estimated_hours/difficulty/prerequisites）。
    - 真实模式：查 knowledge_blocks 表，返回完整板块信息
    - Mock 模式：返回 ./data/test_data/mock_knowledge_block_query_tool.json 预设板块数据

统一接口：
    def knowledge_block_query(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（stage_planning.skill.md 规则3）：
    - block_name = gap.skill，estimated_hours 固定2.0（框架默认，实际从表读取）
    - prerequisites 沿用 gap.prerequisites
    - 字段须符合 StageInfo 契约

保留 KnowledgeBlockQueryTool 类作为向后兼容适配器。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

from config import BACKEND_DIR, DATA_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_knowledge_block_query_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def knowledge_block_query(input_data: dict, config: dict = None) -> dict:
    """
    查询知识板块详情。

    参数：
        input_data: dict
            - skill_name: str   技能名称（优先，匹配 block_name）
            - block_id: str     板块ID（备选）
            - scenario: str     Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool    是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "block": {
                    "block_id": str,
                    "block_name": str,
                    "description": str,
                    "estimated_hours": float,
                    "difficulty": str,
                    "prerequisites": List[str],
                    "corresponding_gaps": List[str]
                } | None,
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
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
    except Exception as e:
        logger.exception("knowledge_block_query 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"knowledge_block_query 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：数据库查询
# ------------------------------------------------------------------
def _get_db_path() -> Path:
    """解析 settings.DATABASE_URL 为绝对路径。"""
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _query_block(skill_name: str) -> tuple[Dict[str, Any] | None, bool]:
    """
    SQL: 从 knowledge_blocks 表查询板块信息（先按 block_name，再按 block_id）。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，knowledge_block_query 无可用板块", db_path)
        return None, False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                SELECT block_id, block_name, description, estimated_hours, 
                       difficulty, prerequisites, corresponding_gaps
                FROM knowledge_blocks
                WHERE block_name = ? OR block_id = ?
                """,
                (skill_name, skill_name),
            )
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            logger.warning("未找到板块: %s", skill_name)
            return None, True

        prereqs = []
        if row[5]:
            try:
                prereqs = json.loads(row[5])
                if not isinstance(prereqs, list):
                    prereqs = []
            except json.JSONDecodeError:
                prereqs = []

        gaps = []
        if row[6]:
            try:
                gaps = json.loads(row[6])
                if not isinstance(gaps, list):
                    gaps = []
            except json.JSONDecodeError:
                gaps = []

        return {
            "block_id": row[0],
            "block_name": row[1],
            "description": row[2],
            "estimated_hours": row[3],
            "difficulty": row[4],
            "prerequisites": prereqs,
            "corresponding_gaps": gaps,
        }, True
    except sqlite3.Error as e:
        logger.warning("查询 knowledge_blocks 失败: %s", e)
        return None, False


def _build_message(block: Dict[str, Any] | None) -> str:
    if not block:
        return "未找到匹配的知识板块"
    return f"找到板块: {block['block_name']}（难度: {block['difficulty']}，时长: {block['estimated_hours']}小时）"


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_query(skill_name: str, input_data: dict) -> dict:
    """从 mock JSON 读取预设场景。"""
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
        with MOCK_FILE.open("r", encoding="utf-8") as f:
            return json.load(f).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("读取 Mock 文件失败: %s", e)
        return {}


# ------------------------------------------------------------------
# 向后兼容适配器
# ------------------------------------------------------------------
class KnowledgeBlockQueryTool:
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def query_block(self, skill_name: str) -> Dict[str, Any]:
        result = knowledge_block_query({"skill_name": skill_name}, config=self._default_config)
        if result.get("status") != "success":
            return {}
        return result["data"].get("block", {})
