# -*- coding: utf-8 -*-
"""
vector_search_tool — Agent3 语义检索工具
================================================
功能：对用户问题或关键词进行语义检索，返回相关知识文档片段。
    - 真实模式：查 knowledge_chunks 表，按相似度排序返回top_k片段
    - Mock 模式：返回 ./data/test_data/mock_vector_search_tool.json 预设检索结果

统一接口：
    def vector_search(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（question_answering.skill.md）：
    - 检索结果用于回答用户问题时提供上下文支撑
    - 返回结果按相似度降序排列
    - 数量限制：1~5条

保留 VectorSearchTool 类作为向后兼容适配器。
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

MOCK_FILE = DATA_DIR / "test_data" / "mock_vector_search_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def vector_search(input_data: dict, config: dict = None) -> dict:
    """
    语义检索相关知识片段。

    参数：
        input_data: dict
            - query: str              查询关键词/问题
            - block_id: str           板块ID（可选过滤条件）
            - top_k: int              返回数量（默认3，范围1~5）
            - scenario: str           Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool          是否使用 Mock（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "results": [
                    {
                        "chunk_id": str,
                        "block_id": str,
                        "content": str,
                        "source": str,
                        "similarity": float,
                        "title": str
                    }
                ],
                "db_available": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)

    query = input_data.get("query", "")
    block_id = input_data.get("block_id", "")
    top_k = min(max(input_data.get("top_k", 3), 1), 5)

    if not query.strip() and not block_id:
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 query 或 block_id 字段，无法检索",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_search(input_data, top_k)

        results, db_available = _search_chunks(query, block_id, top_k)
        return {
            "status": "success",
            "data": {
                "results": results,
                "db_available": db_available,
            },
            "message": _build_message(results),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("vector_search 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"vector_search 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 真实模式：数据库检索（关键词匹配模拟）
# ------------------------------------------------------------------
def _search_chunks(query: str, block_id: str, top_k: int) -> tuple[List[Dict[str, Any]], bool]:
    """
    SQL: 从 knowledge_chunks 表检索相关片段（关键词匹配+按更新时间排序）。
    注：真实向量检索需集成向量数据库（如 Pinecone/Chroma），当前用关键词匹配模拟。
    通过 JOIN knowledge_docs 获取 source、title、created_at 字段。
    """
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s，vector_search 无可用片段", db_path)
        return [], False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            query_words = query.split()
            if not query_words:
                conn.close()
                return [], True

            like_clauses = " OR ".join(["c.content LIKE ?" for _ in query_words])
            params = [f"%{word}%" for word in query_words]

            if block_id:
                sql = f"""
                    SELECT c.chunk_id, c.block_id, c.content, d.source, d.created_at, d.title
                    FROM knowledge_chunks c
                    LEFT JOIN knowledge_docs d ON c.doc_id = d.doc_id
                    WHERE c.block_id = ? AND ({like_clauses})
                    ORDER BY d.created_at DESC
                    LIMIT ?
                    """
                params = [block_id] + params + [top_k]
            else:
                sql = f"""
                    SELECT c.chunk_id, c.block_id, c.content, d.source, d.created_at, d.title
                    FROM knowledge_chunks c
                    LEFT JOIN knowledge_docs d ON c.doc_id = d.doc_id
                    WHERE {like_clauses}
                    ORDER BY d.created_at DESC
                    LIMIT ?
                    """
                params = params + [top_k]

            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        finally:
            conn.close()

        results = []
        for i, r in enumerate(rows):
            similarity = max(0.6, 1.0 - i * 0.1)
            results.append({
                "chunk_id": r[0],
                "block_id": r[1],
                "content": r[2][:500] + "..." if len(r[2]) > 500 else r[2],
                "source": r[3] or "",
                "similarity": round(similarity, 4),
                "title": r[5] or "",
            })

        return results, True
    except sqlite3.Error as e:
        logger.warning("查询 knowledge_chunks 失败: %s", e)
        return [], False


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _get_db_path() -> Path:
    """解析 settings.DATABASE_URL 为绝对路径。"""
    url = settings.DATABASE_URL
    raw = url.replace("sqlite:///", "", 1)
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


def _build_message(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "未找到相关知识片段"
    avg_sim = sum(r["similarity"] for r in results) / len(results)
    return f"检索到 {len(results)} 条相关片段，平均相似度 {avg_sim:.2f}"


# ------------------------------------------------------------------
# Mock 模式
# ------------------------------------------------------------------
def _mock_search(input_data: dict, top_k: int) -> dict:
    """从 mock JSON 读取预设场景。"""
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    all_results = scenarios.get(scenario, {}).get("results", [])
    block_id = input_data.get("block_id", "")
    
    if block_id:
        all_results = [r for r in all_results if r.get("block_id") == block_id]
    
    results = all_results[:top_k]

    return {
        "status": "success",
        "data": {
            "results": results,
            "db_available": False,
        },
        "message": f"[Mock] {_build_message(results)}",
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
class VectorSearchTool:
    def __init__(self):
        self._default_config = {"use_mock": True}

    async def search(self, query: str, block_id: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        result = vector_search({
            "query": query,
            "block_id": block_id,
            "top_k": top_k,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])

    async def search_by_block(self, block_id: str) -> List[Dict[str, Any]]:
        result = vector_search({
            "query": "",
            "block_id": block_id,
            "top_k": 3,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])

    async def search_by_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        result = vector_search({
            "query": query,
            "block_id": "",
            "top_k": top_k,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])
