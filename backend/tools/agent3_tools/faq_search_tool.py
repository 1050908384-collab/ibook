# -*- coding: utf-8 -*-
"""
faq_search_tool — Agent3 FAQ混合检索工具
================================================
功能：对用户问题进行混合型FAQ检索。
    - 真实模式：LLM提取关键词 → Chroma向量检索 → SQL LIKE兜底
    - Mock 模式：返回 ./data/test_data/mock_faq_search_tool.json 预设检索结果

统一接口：
    def faq_search(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

混合检索流程：
    用户问题 → LLM提取3-5个关键词 → Chroma向量检索FAQ → 返回结果
                                          ↓ (未命中)
                                    SQL LIKE关键词匹配 → 返回结果
                                          ↓ (仍未命中)
                                    返回空列表（由上层调用LLM生成）

规则对齐（question_answering.skill.md）：
    - 仅在用户提问时调用，其他场景不触发
    - 优先返回向量检索命中结果
    - SQL LIKE作为兜底方案
    - 返回结果按相似度/热度降序排列
    - 数量限制：1~5条

保留 FAQSearchTool 类作为向后兼容适配器。
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config import BACKEND_DIR, DATA_DIR, settings
from db.sqlite_client import SQLiteClient
from db.chroma_client import ChromaClient
from llm.client import LLMClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_faq_search_tool.json"


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def faq_search(input_data: dict, config: dict = None) -> dict:
    """
    混合检索FAQ条目。

    参数：
        input_data: dict
            - query: str              用户问题
            - block_id: str           板块ID（可选过滤条件）
            - top_k: int              返回数量（默认3，范围1~5）
            - scenario: str           Mock 模式下选择场景：normal|boundary|exception
        config: dict
            - use_mock: bool          是否使用 Mock（默认 True）
            - enable_llm_keyword: bool 是否启用LLM关键词提取（默认 True）

    返回：
        {
            "status": "success" | "error",
            "data": {
                "results": [
                    {
                        "faq_id": str,
                        "block_id": str,
                        "question": str,
                        "answer": str,
                        "difficulty": str,
                        "tags": list,
                        "views": int,
                        "similarity": float
                    }
                ],
                "db_available": bool,
                "vector_search_used": bool,
                "keyword_search_used": bool
            },
            "message": str,
            "mock_used": bool
        }
    """
    config = config or {}
    use_mock = config.get("use_mock", True)
    enable_llm_keyword = config.get("enable_llm_keyword", True)

    query = input_data.get("query", "")
    block_id = input_data.get("block_id", "")
    top_k = min(max(input_data.get("top_k", 3), 1), 5)

    if not query.strip():
        return {
            "status": "error",
            "data": None,
            "message": "输入未包含 query 字段，无法检索",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_search(input_data, top_k)

        results, db_available, vector_used, keyword_used = _hybrid_search(
            query, block_id, top_k, enable_llm_keyword
        )
        return {
            "status": "success",
            "data": {
                "results": results,
                "db_available": db_available,
                "vector_search_used": vector_used,
                "keyword_search_used": keyword_used,
            },
            "message": _build_message(results, vector_used, keyword_used),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("faq_search 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"faq_search 执行异常: {e}",
            "mock_used": use_mock,
        }


# ------------------------------------------------------------------
# 混合检索流程
# ------------------------------------------------------------------
def _hybrid_search(
    query: str,
    block_id: str,
    top_k: int,
    enable_llm_keyword: bool = True,
) -> tuple[List[Dict[str, Any]], bool, bool, bool]:
    """
    混合检索流程：LLM提取关键词 → Chroma向量检索 → SQL LIKE兜底
    """
    db_available = True
    vector_used = False
    keyword_used = False

    keywords = _extract_keywords(query) if enable_llm_keyword else [query]

    vector_results = _search_chroma(keywords, block_id, top_k)
    if vector_results:
        vector_used = True
        return vector_results, db_available, vector_used, keyword_used

    keyword_results = _search_sql(keywords, block_id, top_k)
    if keyword_results:
        keyword_used = True
        return keyword_results, db_available, vector_used, keyword_used

    return [], db_available, vector_used, keyword_used


def _extract_keywords(query: str) -> List[str]:
    """
    LLM提取关键词（轻量prompt，控制token消耗）。
    """
    try:
        llm_client = LLMClient()
        system_prompt = "你是一个关键词提取器。从用户问题中提取3-5个最相关的搜索关键词。仅返回关键词，用逗号分隔，不要解释。"
        prompt = f"用户问题：{query}\n提取的关键词："

        response = llm_client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=50,
            temperature=0.1,
        )

        keywords = [k.strip() for k in response.split(",") if k.strip()]
        if not keywords:
            keywords = [query]
        return keywords[:5]
    except Exception as e:
        logger.warning(f"LLM关键词提取失败，使用原始问题作为关键词: {e}")
        return [query]


def _search_chroma(keywords: List[str], block_id: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Chroma向量检索FAQ。
    使用专门的"faq_collection"集合进行向量检索。
    """
    try:
        from chromadb import PersistentClient

        chroma_client = PersistentClient(path=str(DATA_DIR / "chroma"))
        collection = chroma_client.get_or_create_collection("faq_collection")

        query_text = " ".join(keywords)
        where = {"block_id": block_id} if block_id else None

        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        faq_results = []
        for i, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            similarity = max(0.0, 1.0 - distance)
            faq_results.append({
                "faq_id": metadata.get("faq_id", ""),
                "block_id": metadata.get("block_id", ""),
                "question": metadata.get("question", doc),
                "answer": metadata.get("answer", ""),
                "difficulty": metadata.get("difficulty", ""),
                "tags": metadata.get("tags", []),
                "views": metadata.get("views", 0),
                "similarity": round(similarity, 4),
            })

        return sorted(faq_results, key=lambda x: x["similarity"], reverse=True)[:top_k]
    except ImportError:
        logger.warning("ChromaDB未安装，跳过向量检索")
        return []
    except Exception as e:
        logger.warning(f"Chroma向量检索失败: {e}")
        return []


def _search_sql(keywords: List[str], block_id: str, top_k: int) -> List[Dict[str, Any]]:
    """
    SQL LIKE关键词匹配兜底。
    """
    try:
        sqlite_client = SQLiteClient()
        query_text = " ".join(keywords)
        results = sqlite_client.search_faq(block_id=block_id, query=query_text, limit=top_k)

        for i, result in enumerate(results):
            result["similarity"] = round(max(0.5, 1.0 - i * 0.1), 4)

        return results
    except Exception as e:
        logger.warning(f"SQL LIKE检索失败: {e}")
        return []


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _build_message(results: List[Dict[str, Any]], vector_used: bool, keyword_used: bool) -> str:
    if not results:
        return "FAQ库未找到相关条目"

    search_method = "向量检索" if vector_used else "关键词匹配" if keyword_used else "未知"
    avg_sim = sum(r["similarity"] for r in results) / len(results)
    return f"[FAQ检索-{search_method}] 检索到 {len(results)} 条相关条目，平均相似度 {avg_sim:.2f}"


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
            "vector_search_used": True,
            "keyword_search_used": False,
        },
        "message": f"[Mock] {_build_message(results, True, False)}",
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
class FAQSearchTool:
    def __init__(self):
        self._default_config = {"use_mock": True, "enable_llm_keyword": True}

    async def search(self, query: str, block_id: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        result = faq_search({
            "query": query,
            "block_id": block_id,
            "top_k": top_k,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])

    async def search_by_block(self, block_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        result = faq_search({
            "query": "",
            "block_id": block_id,
            "top_k": top_k,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])

    async def search_by_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        result = faq_search({
            "query": query,
            "block_id": "",
            "top_k": top_k,
        }, config=self._default_config)
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])