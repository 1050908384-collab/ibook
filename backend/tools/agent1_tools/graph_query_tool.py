# -*- coding: utf-8 -*-
"""
graph_query_tool — Agent1 知识图谱依赖查询工具
================================================
功能：按技能名称/板块ID查询知识图谱节点与先修依赖，递归推导学习先后顺序。
    - 真实模式：读取 data/graph_data/runtime 的图谱结果
    - Mock 模式：返回 ./data/test_data/mock_graph_query_tool.json 预设依赖链

统一接口：
    def graph_query(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from config import DATA_DIR
from db.graph_client import GraphClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_graph_query_tool.json"
BLOCKS_SEED_FILE = DATA_DIR / "seeds" / "knowledge_blocks.json"


def graph_query(input_data: dict, config: dict = None) -> dict:
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

        graph_client = GraphClient()
        result = _resolve_dependencies(skill_name, graph_client)
        result["db_available"] = bool(graph_client.graph.nodes)
        return {
            "status": "success",
            "data": result,
            "message": _build_message(result),
            "mock_used": False,
        }
    except Exception as exc:
        logger.exception("graph_query 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"graph_query 执行异常: {exc}",
            "mock_used": use_mock,
        }


def _resolve_dependencies(skill_name: str, graph_client: GraphClient) -> dict:
    graph = graph_client.graph
    seed_index = _load_seed_index()
    block_id = _resolve_block_id(skill_name, seed_index) or _find_block_id(graph, skill_name)
    if not block_id:
        return {
            "skill": skill_name,
            "dependencies": [],
            "max_depth": 0,
            "warning": None,
        }

    prerequisite_graph = nx.DiGraph()
    block_nodes: dict[str, dict[str, Any]] = {}

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") != "block":
            continue
        block_nodes[node_id] = dict(node_data)
        prerequisite_graph.add_node(node_id)

    for source_id, target_id, edge_data in graph.edges(data=True):
        relation_types = edge_data.get("relation_types", [])
        if (
            (edge_data.get("relation_type") == "PREREQUISITE_OF" or "PREREQUISITE_OF" in relation_types)
            and edge_data.get("scope") == "block"
        ):
            prerequisite_graph.add_edge(source_id, target_id)

    if block_id not in prerequisite_graph:
        return {
            "skill": block_nodes.get(block_id, {}).get("block_name", skill_name),
            "dependencies": [],
            "max_depth": 0,
            "warning": None,
        }

    ancestors = nx.ancestors(prerequisite_graph, block_id)
    dependencies: list[dict[str, Any]] = []
    for ancestor_id in sorted(
        ancestors,
        key=lambda node_id: (
            _distance_or_default(prerequisite_graph, node_id, block_id),
            node_id,
        ),
    ):
        node_data = block_nodes.get(ancestor_id, {})
        dependencies.append(
            {
                "block_id": ancestor_id,
                "block_name": _canonical_block_name(ancestor_id, node_data, seed_index),
                "difficulty": node_data.get("difficulty", ""),
                "estimated_hours": node_data.get("estimated_hours", 0),
                "level": _distance_or_default(prerequisite_graph, ancestor_id, block_id),
            }
        )

    max_depth = max((item["level"] for item in dependencies), default=0)
    return {
        "skill": _canonical_block_name(block_id, block_nodes.get(block_id, {}), seed_index) or skill_name,
        "dependencies": dependencies,
        "max_depth": max_depth,
        "warning": None,
    }


def _find_block_id(graph: nx.DiGraph, skill_name: str) -> str | None:
    normalized = skill_name.strip()
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") != "block":
            continue
        if node_id == normalized or node_data.get("block_name") == normalized:
            return node_id
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


def _canonical_block_name(block_id: str, node_data: dict[str, Any], seed_index: dict[str, dict[str, Any]]) -> str:
    seed_row = seed_index.get(block_id, {})
    return seed_row.get("block_name") or node_data.get("block_name") or block_id


def _distance_or_default(graph: nx.DiGraph, source_id: str, target_id: str) -> int:
    try:
        return nx.shortest_path_length(graph, source_id, target_id)
    except nx.NetworkXNoPath:
        return 0


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


def _mock_query(skill_name: str, input_data: dict) -> dict:
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
        with MOCK_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取 Mock 文件失败: %s", exc)
        return {}


class GraphQueryTool:
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def query_prerequisites(self, skill_name: str) -> List[str]:
        result = graph_query({"skill_name": skill_name}, config=self._default_config)
        if result.get("status") != "success":
            return []
        deps = result["data"].get("dependencies", [])
        return [item["block_name"] for item in deps]

    async def query_direct_prerequisites(self, skill_name: str) -> List[Dict[str, Any]]:
        """Return only direct block predecessors for Agent1 stage ordering."""
        try:
            graph_client = GraphClient()
            seed_index = _load_seed_index()
            block_id = _resolve_block_id(skill_name, seed_index) or _find_block_id(graph_client.graph, skill_name)
            if not block_id:
                return []
            return [
                {
                    "block_id": prerequisite_id,
                    "block_name": _canonical_block_name(
                        prerequisite_id,
                        dict(graph_client.graph.nodes[prerequisite_id]),
                        seed_index,
                    ),
                }
                for prerequisite_id in graph_client.get_prerequisites(block_id)
            ]
        except Exception:
            return []

    async def query_dependencies(self, skill_name: str) -> List[str]:
        return await self.query_prerequisites(skill_name)
