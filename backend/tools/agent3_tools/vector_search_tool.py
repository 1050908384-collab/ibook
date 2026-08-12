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
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    block_ids = list(dict.fromkeys(input_data.get("block_ids", []) or ([] if not block_id else [block_id])))
    subtopic_id = input_data.get("subtopic_id", "")
    subtopic_ids = list(dict.fromkeys(input_data.get("subtopic_ids", []) or ([] if not subtopic_id else [subtopic_id])))
    top_k = min(max(input_data.get("top_k", 3), 1), 12)

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

        results, db_available = _search_chunks(query, block_ids, subtopic_ids, top_k)
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
# 真实模式：Chroma 向量检索
# ------------------------------------------------------------------
def _search_chunks(query: str, block_ids: List[str], subtopic_ids: List[str], top_k: int) -> tuple[List[Dict[str, Any]], bool]:
    """
    Chroma 向量检索：从 knowledge_base 集合做语义检索。
    注：SQLite knowledge_chunks 表作为回退（Chroma 无数据时）。
    """
    chroma_results = _search_chroma(query, block_ids, subtopic_ids, top_k)
    if chroma_results:
        return chroma_results, True

    logger.info("Chroma 无结果，回退 SQLite FTS5/BM25 检索")
    return _search_sqlite_fallback(query, block_ids, subtopic_ids, top_k)


def _search_chroma(query: str, block_ids: List[str], subtopic_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Chroma 向量检索 knowledge_base 集合（使用配置的 Ollama embedding 模型）。"""
    try:
        from chromadb import PersistentClient

        chroma_path = _get_chroma_path()
        if not chroma_path.exists():
            return []

        chroma_client = PersistentClient(path=str(chroma_path))
        # 必须传入与写入时相同的 embedding function，否则 query 和 document
        # 不在同一向量空间。
        from backend.db.chroma_client import CHROMA_COLLECTION, get_embedding_function
        ef = get_embedding_function()
        # EF 为 None 时绝不用 None 创建 collection——Chroma 会静默回退英文模型，
        # 导致中文检索 sim 全部失真。直接降级到 SQLite 关键词匹配。
        if ef is None:
            logger.warning("Embedding function 不可用，跳过 Chroma 检索，回退 SQLite")
            return []
        collection = chroma_client.get_or_create_collection(
            CHROMA_COLLECTION,
            embedding_function=ef,
        )
        if collection.count() == 0:
            return []

        filters = []
        if block_ids:
            filters.append({"block_id": {"$in": block_ids}})
        if subtopic_ids:
            filters.append({"subtopic_id": {"$in": subtopic_ids}})
        where_filter = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        output = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0] if results.get("metadatas") and results["metadatas"] else [{}] * len(results["documents"][0]),
            results["distances"][0],
        ):
            similarity = round(max(0.0, 1.0 - float(distance)), 4) if distance is not None else 0.5
            output.append({
                "chunk_id": metadata.get("chunk_id", "") if metadata else "",
                "block_id": metadata.get("block_id", "") if metadata else "",
                "content": (doc[:500] + "...") if len(doc) > 500 else doc,
                "source": metadata.get("source", "") if metadata else "",
                "similarity": similarity,
                "title": metadata.get("title", "") if metadata else "",
            })
        return sorted(output, key=lambda x: x["similarity"], reverse=True)
    except ImportError:
        logger.info("chromadb not installed, skip vector search")
        return []
    except Exception as e:
        logger.warning("Chroma 向量检索失败: %s", e)
        return []


def _build_fts_match_query(query: str) -> str:
    """构造适用于 trigram FTS5 的安全查询，兼容中文连续文本和英文技术词。"""
    terms: list[str] = []
    terms.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", query))
    for segment in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(segment) < 3:
            terms.append(segment)
        else:
            terms.extend(segment[index:index + 3] for index in range(len(segment) - 2))
    unique_terms = list(dict.fromkeys(term.replace('"', '""') for term in terms if term.strip()))[:24]
    return " OR ".join(f'"{term}"' for term in unique_terms)


def _search_sqlite_fallback(query: str, block_ids: List[str], subtopic_ids: List[str], top_k: int) -> tuple[List[Dict[str, Any]], bool]:
    """SQLite FTS5/BM25 回退（Chroma 无数据时使用）。"""
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning("数据库不存在: %s", db_path)
        return [], False

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            match_query = _build_fts_match_query(query)
            scope_clauses = []
            scope_params: list[str] = []
            if block_ids:
                scope_clauses.append(f"f.block_id IN ({', '.join('?' for _ in block_ids)})")
                scope_params.extend(block_ids)
            if subtopic_ids:
                scope_clauses.append(f"f.subtopic_id IN ({', '.join('?' for _ in subtopic_ids)})")
                scope_params.extend(subtopic_ids)
            where_scope = " AND ".join(scope_clauses) if scope_clauses else "1 = 1"

            if not match_query:
                rows = conn.execute(f"""
                    SELECT c.chunk_id, c.block_id, c.content, d.source, d.created_at, d.title
                    FROM knowledge_chunks c
                    JOIN knowledge_docs d ON c.doc_id = d.doc_id
                    WHERE {where_scope.replace('f.', 'c.')}
                    ORDER BY CASE WHEN d.doc_id LIKE 'LOCAL-%' THEN 0 ELSE 1 END, c.chunk_index
                    LIMIT ?
                """, scope_params + [top_k]).fetchall()
                return _rows_to_results(rows, []), True

            rows = conn.execute(f"""
                SELECT c.chunk_id, c.block_id, c.content, d.source, d.created_at, d.title,
                       bm25(knowledge_chunks_fts) AS score
                FROM knowledge_chunks_fts f
                JOIN knowledge_chunks c ON c.chunk_id = f.chunk_id
                LEFT JOIN knowledge_docs d ON c.doc_id = d.doc_id
                WHERE knowledge_chunks_fts MATCH ? AND {where_scope}
                ORDER BY score ASC
                LIMIT ?
            """, [match_query, *scope_params, top_k]).fetchall()
        finally:
            conn.close()

        return _fts_rows_to_results(rows), True
    except sqlite3.Error as e:
        logger.warning("SQLite 回退检索失败: %s", e)
        return [], False


def _rows_to_results(rows: list[tuple], query_words: list[str]) -> List[Dict[str, Any]]:
    results = []
    for r in rows:
        overlap = sum(1 for word in query_words if word.lower() in (r[2] or "").lower())
        similarity = round(min(1.0, 0.3 + overlap / max(len(query_words), 1) * 0.7), 4) if query_words else 1.0
        results.append({
            "chunk_id": r[0], "block_id": r[1],
            "content": r[2][:500] + "..." if len(r[2]) > 500 else r[2],
            "source": r[3] or "", "similarity": similarity, "title": r[5] or "",
        })
    return results


def _fts_rows_to_results(rows: list[tuple]) -> List[Dict[str, Any]]:
    """转换 BM25 结果；BM25 值越小代表越相关。"""
    results = []
    for rank, row in enumerate(rows):
        results.append({
            "chunk_id": row[0], "block_id": row[1],
            "content": row[2][:500] + "..." if len(row[2]) > 500 else row[2],
            "source": row[3] or "", "similarity": round(max(0.1, 1.0 - rank * 0.1), 4),
            "title": row[5] or "", "bm25_score": row[6],
        })
    return results


def _get_chroma_path() -> Path:
    """解析 Chroma 路径。"""
    raw = settings.CHROMA_PATH
    p = Path(raw)
    if not p.is_absolute():
        p = (BACKEND_DIR / p).resolve()
    return p


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


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """PRAGMA 探测某表实际列名集合（用于新字段缺失即降级）。"""
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}
    except sqlite3.Error:
        return set()


# knowledge_chunks 数据侧新增字段（他人负责入库；见《知识库分块新增字段说明.md》）。
_CHUNK_OPTIONAL_COLS = [
    "heading_path", "heading_level", "source_line_start",
    "source_line_end", "chunk_strategy",
]


def _select_chunk_columns(available: set[str]) -> tuple[str, list[str]]:
    """构造 knowledge_chunks 的防御式 SELECT 列表达式。

    始终选取基础列（chunk_id/doc_id/block_id/chunk_index/content），
    可选新列存在则 SELECT，不存在则 `NULL AS col`。
    返回 (select_expr, ordered_column_names)。
    """
    base = ["c.chunk_id", "c.doc_id", "c.block_id", "c.chunk_index", "c.content"]
    names = ["chunk_id", "doc_id", "block_id", "chunk_index", "content"]
    exprs = list(base)
    for col in _CHUNK_OPTIONAL_COLS:
        if col in available:
            exprs.append(f"c.{col}")
        else:
            exprs.append(f"NULL AS {col}")
        names.append(col)
    return ", ".join(exprs), names


def _chunk_row_to_dict(row: tuple, names: list[str], title: str = "", source: str = "",
                       subtopic_id: str = "") -> Dict[str, Any]:
    data = {names[i]: row[i] for i in range(len(names))}
    data["title"] = title or ""
    data["source"] = source or ""
    if subtopic_id:
        data["subtopic_id"] = subtopic_id
    return data


# ------------------------------------------------------------------
# 向后兼容适配器
# ------------------------------------------------------------------
class VectorSearchTool:
    def __init__(self):
        self._default_config = {"use_mock": False}

    async def search(
        self,
        query: str,
        block_id: str = "",
        block_ids: Optional[List[str]] = None,
        subtopic_ids: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        result = vector_search({
            "query": query,
            "block_id": block_id,
            "block_ids": block_ids or [],
            "subtopic_ids": subtopic_ids or [],
            "top_k": top_k,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])

    async def search_by_block(
        self, block_id: str, subtopic_id: str = "", top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        result = vector_search({
            "query": "",
            "block_id": block_id,
            "subtopic_id": subtopic_id,
            "top_k": top_k,
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

    def get_local_lessons(self, block_id: str, subtopic_ids: list[str]) -> List[Dict[str, Any]]:
        """Load complete imported lessons for a learning package, not search snippets."""
        if not block_id or not subtopic_ids:
            return []
        db_path = _get_db_path()
        if not db_path.exists():
            return []
        placeholders = ", ".join("?" for _ in subtopic_ids)
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                rows = connection.execute(f"""
                    SELECT doc_id, block_id, subtopic_id, title, content, source
                    FROM knowledge_docs
                    WHERE block_id = ?
                      AND subtopic_id IN ({placeholders})
                      AND doc_id LIKE 'LOCAL-%'
                    ORDER BY subtopic_id, doc_id
                """, [block_id, *subtopic_ids]).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            logger.warning("读取本地学习资料失败: %s", error)
            return []
        return [
            {
                "doc_id": row[0], "block_id": row[1], "subtopic_id": row[2],
                "title": row[3] or "", "content": row[4] or "", "source": row[5] or "",
            }
            for row in rows
        ]

    def get_block_ids_for_subtopics(self, subtopic_ids: list[str]) -> List[str]:
        """按阶段知识点解析全部所属板块，兼容旧的单 block_id 阶段。"""
        if not subtopic_ids:
            return []
        db_path = _get_db_path()
        if not db_path.exists():
            return []
        placeholders = ", ".join("?" for _ in subtopic_ids)
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                rows = connection.execute(
                    f"SELECT DISTINCT block_id FROM knowledge_subtopics WHERE subtopic_id IN ({placeholders})",
                    subtopic_ids,
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            logger.warning("解析阶段知识点所属板块失败: %s", error)
            return []
        return [row[0] for row in rows if row[0]]

    def get_stage_chunks(self, block_id: str, subtopic_ids: list[str]) -> List[Dict[str, Any]]:
        """读取某阶段（block + 子知识点集合）的原始 knowledge_chunks（供 Agent3 块级生成）。

        - 只读 sqlite，按 (subtopic_id, chunk_index) 稳定排序；
        - PRAGMA 探测新列（heading_path 等），缺失即以 NULL 降级，不崩溃；
        - 数据库不存在或空库返回 []（由上层降级到 local lessons / pseudo chunks）。
        """
        if not block_id:
            return []
        db_path = _get_db_path()
        if not db_path.exists():
            return []
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                available = _table_columns(connection, "knowledge_chunks")
                if not available:
                    return []
                select_expr, names = _select_chunk_columns(available)
                if subtopic_ids:
                    placeholders = ", ".join("?" for _ in subtopic_ids)
                    where = f"c.block_id = ? AND d.subtopic_id IN ({placeholders})"
                    params = [block_id, *subtopic_ids]
                else:
                    where = "c.block_id = ?"
                    params = [block_id]
                sql = f"""
                    SELECT {select_expr}, d.title, d.source, d.subtopic_id
                    FROM knowledge_chunks c
                    JOIN knowledge_docs d ON c.doc_id = d.doc_id
                    WHERE {where}
                    ORDER BY d.subtopic_id, c.chunk_index
                """
                rows = connection.execute(sql, params).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as error:
            logger.warning("读取阶段 chunks 失败: %s", error)
            return []
        n = len(names)
        return [
            _chunk_row_to_dict(row, names, title=row[n], source=row[n + 1], subtopic_id=row[n + 2])
            for row in rows
        ]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """按 chunk_id 回查单个原始 knowledge_chunk（供 Agent4 回查来源块）。

        同样防御式列选择；未找到或库不可用返回 None（不产生假数据）。
        """
        if not chunk_id:
            return None
        db_path = _get_db_path()
        if not db_path.exists():
            return None
        try:
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                available = _table_columns(connection, "knowledge_chunks")
                if not available:
                    return None
                select_expr, names = _select_chunk_columns(available)
                sql = f"""
                    SELECT {select_expr}, d.title, d.source, d.subtopic_id
                    FROM knowledge_chunks c
                    LEFT JOIN knowledge_docs d ON c.doc_id = d.doc_id
                    WHERE c.chunk_id = ?
                    LIMIT 1
                """
                row = connection.execute(sql, [chunk_id]).fetchone()
            finally:
                connection.close()
        except sqlite3.Error as error:
            logger.warning("回查 chunk 失败: %s", error)
            return None
        if not row:
            return None
        n = len(names)
        return _chunk_row_to_dict(row, names, title=row[n], source=row[n + 1], subtopic_id=row[n + 2])
