# -*- coding: utf-8 -*-
"""
graph_query_tool — Agent1 知识图谱依赖查询工具
================================================
功能：按技能名称/板块ID查询知识图谱节点与先修依赖，递归推导学习先后顺序。
    - 真实模式：查 knowledge_blocks 表的 prerequisites 字段（JSON数组）做递归依赖查询
    - Mock 模式：返回 ./data/test_data/mock_graph_query_tool.json 预设依赖链

统一接口：
    def graph_query(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（knowledge_graph_mapping.skill.md）：
    - 规则1：先修依赖必须是图谱中已存在标准板块，不生成临时节点
    - 规则2：依赖具备传递性，自动去重
    - 规则3：依赖链超3层生成分层提示
    - 规则4：检测循环依赖，截断后返回部分有效依赖并警告

保留 GraphQueryTool 类作为向后兼容适配器。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config import BACKEND_DIR, DATA_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_graph_query_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def graph_query(input_data: dict, config: dict = None) -> dict:
    """
    查询技能的先修依赖链。

    参数：
        input_data: dict
            - skill_name: str   技能名称（优先）
            - block_id: str     板块ID（备选）
            - scenario: str     Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool    是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "skill": str,
                "dependencies": [
                    {"block_id", "block_name", "difficulty", "estimated_hours", "level"}
                ],
                "max_depth": int,
                "warning": str | None,
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

        blocks, db_available = _load_blocks()
        if not blocks:
            return {
                "status": "success",
                "data": {
                    "skill": skill_name,
                    "dependencies": [],
                    "max_depth": 0,
                    "warning": None,
                    "db_available": False,
                },
                "message": f"知识库不可用，无{skill_name}的依赖信息",
                "mock_used": False,
            }

        result = _resolve_dependencies(skill_name, blocks)
        result["db_available"] = db_available
        return {
            "status": "success",
            "data": result,
            "message": _build_message(result),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("graph_query 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"graph_query 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：数据库依赖查询
# ------------------------------------------------------------------
def _get_db_path() -> Path:
    """解析 settings.DATABASE_URL 为绝对路径（与 sqlite_client 一致）。"""
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _load_blocks() -> tuple[Dict[str, Dict[str, Any]], bool]:
    """
    SQL: 从 knowledge_blocks 表读取全部板块（含 prerequisites JSON字段）。
    返回：{block_name: block_data, ...} 的索引字典，便于快速查找。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，graph_query 无可用知识板块", db_path)
        return {}, False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                SELECT block_id, block_name, description, estimated_hours, difficulty, prerequisites
                FROM knowledge_blocks
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        blocks: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            prereqs = []
            if r[5]:
                try:
                    prereqs = json.loads(r[5])
                    if not isinstance(prereqs, list):
                        prereqs = []
                except json.JSONDecodeError:
                    prereqs = []
            blocks[r[1]] = {
                "block_id": r[0],
                "block_name": r[1],
                "description": r[2],
                "estimated_hours": r[3],
                "difficulty": r[4],
                "prerequisites": prereqs,
            }
        return blocks, True
    except sqlite3.Error as e:
        logger.warning("读取 knowledge_blocks 失败: %s", e)
        return {}, False


def _resolve_dependencies(skill_name: str, blocks: Dict[str, Dict[str, Any]],
                          visited: set = None, depth: int = 1) -> dict:
    """递归解析依赖链，检测循环依赖。"""
    if visited is None:
        visited = set()

    dependencies: List[Dict[str, Any]] = []
    warning = None
    max_depth = depth

    if skill_name in visited:
        return {
            "skill": skill_name,
            "dependencies": [],
            "max_depth": depth,
            "warning": f"circular_dependency: {skill_name} 出现在依赖链中形成循环",
        }
    visited.add(skill_name)

    block = blocks.get(skill_name)
    if not block:
        visited.remove(skill_name)
        return {
            "skill": skill_name,
            "dependencies": [],
            "max_depth": depth,
            "warning": None,
        }

    for prereq_name in block.get("prerequisites", []):
        if prereq_name in visited:
            warning = f"circular_dependency: {prereq_name} ↔ {skill_name} 形成循环，已截断"
            continue

        child_result = _resolve_dependencies(prereq_name, blocks, visited, depth + 1)
        if child_result.get("warning"):
            warning = child_result["warning"]
        max_depth = max(max_depth, child_result["max_depth"])

        child_block = blocks.get(prereq_name)
        if child_block:
            dependencies.append({
                "block_id": child_block["block_id"],
                "block_name": prereq_name,
                "difficulty": child_block["difficulty"],
                "estimated_hours": child_block["estimated_hours"],
                "level": depth,
            })

    visited.remove(skill_name)

    return {
        "skill": skill_name,
        "dependencies": dependencies,
        "max_depth": max_depth,
        "warning": warning,
    }


def _build_message(result: dict) -> str:
    skill = result["skill"]
    deps = result["dependencies"]
    max_depth = result["max_depth"]
    warning = result["warning"]

    if warning:
        return f"{skill} 的依赖查询完成，{len(deps)}个依赖，最大深度{max_depth}，警告：{warning}"
    if not deps:
        return f"{skill} 无先修依赖"
    if max_depth > 3:
        return f"{skill} 的依赖查询完成，{len(deps)}个依赖，最大深度{max_depth}，建议拆分为多个阶段学习"
    return f"{skill} 的依赖查询完成，{len(deps)}个依赖，最大深度{max_depth}"


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
    data["skill"] = skill_name
    data.setdefault("warning", None)
    data.setdefault("db_available", False)

    return {
        "status": "success",
        "data": data,
        "message": f"[Mock] {_build_message(data)}",
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
class GraphQueryTool:
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def query_prerequisites(self, skill_name: str) -> List[str]:
        result = graph_query({"skill_name": skill_name}, config=self._default_config)
        if result.get("status") != "success":
            return []
        deps = result["data"].get("dependencies", [])
        return [d["block_name"] for d in deps]

    async def query_dependencies(self, skill_name: str) -> List[str]:
        return await self.query_prerequisites(skill_name)
