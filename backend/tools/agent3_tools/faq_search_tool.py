# -*- coding: utf-8 -*-
"""
Agent3 FAQ 向量检索工具

运行链路：
1. 仅在用户提问时调用
2. 直接使用完整问题做 FAQ 向量检索
3. 仅当最高相似度达到阈值时返回 FAQ 标准答案
4. 未达到阈值时返回空结果，由上层继续走普通答疑 / LLM

不再包含：
- LLM 提取关键词
- SQL LIKE FAQ 兜底
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from config import BACKEND_DIR, DATA_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MOCK_FILE = DATA_DIR / "test_data" / "mock_faq_search_tool.json"
DEFAULT_MIN_SIMILARITY = 0.78


def faq_search(input_data: dict, config: dict = None) -> dict:
    """
    FAQ 向量检索统一入口。

    input_data:
        - query: str
        - block_id: str
        - top_k: int
        - scenario: str

    config:
        - use_mock: bool
        - min_similarity: float
    """
    config = config or {}
    use_mock = config.get("use_mock", True)
    min_similarity = float(config.get("min_similarity", DEFAULT_MIN_SIMILARITY))

    query = (input_data.get("query") or "").strip()
    block_id = (input_data.get("block_id") or "").strip()
    top_k = min(max(int(input_data.get("top_k", 3)), 1), 5)

    if not query:
        return {
            "status": "error",
            "data": None,
            "message": "query is required",
            "mock_used": use_mock,
        }

    try:
        if use_mock:
            return _mock_search(input_data, top_k, min_similarity)

        results = _search_chroma(query=query, block_id=block_id, top_k=top_k)
        accepted = _filter_by_similarity(results, min_similarity=min_similarity)
        return {
            "status": "success",
            "data": {
                "results": accepted,
                "db_available": True,
                "vector_search_used": True,
                "min_similarity": min_similarity,
                "top_similarity": results[0]["similarity"] if results else None,
                "accepted": bool(accepted),
            },
            "message": _build_message(accepted, results, min_similarity),
            "mock_used": False,
        }
    except Exception as exc:
        logger.exception("faq_search failed")
        return {
            "status": "error",
            "data": None,
            "message": f"faq_search failed: {exc}",
            "mock_used": use_mock,
        }


def _search_chroma(query: str, block_id: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        from chromadb import PersistentClient

        chroma_path = Path(settings.CHROMA_PATH)
        if not chroma_path.is_absolute():
            chroma_path = (BACKEND_DIR / chroma_path).resolve()

        chroma_client = PersistentClient(path=str(chroma_path))
        collection = chroma_client.get_or_create_collection("faq_collection")
        where = {"block_id": block_id} if block_id else None

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        faq_results = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = max(0.0, 1.0 - float(distance))
            faq_results.append(
                {
                    "faq_id": metadata.get("faq_id", ""),
                    "block_id": metadata.get("block_id", ""),
                    "question": metadata.get("question", doc),
                    "answer": metadata.get("answer", ""),
                    "difficulty": metadata.get("difficulty", ""),
                    "tags": metadata.get("tags", []),
                    "views": metadata.get("views", 0),
                    "similarity": round(similarity, 4),
                }
            )

        return sorted(faq_results, key=lambda item: item["similarity"], reverse=True)[:top_k]
    except ImportError:
        logger.warning("chromadb not installed, skip FAQ vector search")
        return []
    except Exception as exc:
        logger.warning("FAQ vector search failed: %s", exc)
        return []


def _filter_by_similarity(results: List[Dict[str, Any]], min_similarity: float) -> List[Dict[str, Any]]:
    if not results:
        return []
    if results[0]["similarity"] < min_similarity:
        return []
    return results


def _build_message(
    accepted: List[Dict[str, Any]],
    raw_results: List[Dict[str, Any]],
    min_similarity: float,
) -> str:
    if not raw_results:
        return "FAQ vector search found no candidates"
    top_similarity = raw_results[0]["similarity"]
    if not accepted:
        return f"FAQ candidate rejected: top_similarity={top_similarity:.2f} < threshold={min_similarity:.2f}"
    return f"FAQ matched: {len(accepted)} result(s), top_similarity={top_similarity:.2f}"


def _mock_search(input_data: dict, top_k: int, min_similarity: float) -> dict:
    scenario = input_data.get("scenario") or "normal"
    scenarios = _load_mock()
    if scenario not in scenarios:
        scenario = "normal"

    results = scenarios.get(scenario, {}).get("results", [])[:top_k]
    accepted = _filter_by_similarity(results, min_similarity=min_similarity)

    return {
        "status": "success",
        "data": {
            "results": accepted,
            "db_available": False,
            "vector_search_used": True,
            "min_similarity": min_similarity,
            "top_similarity": results[0]["similarity"] if results else None,
            "accepted": bool(accepted),
        },
        "message": f"[Mock] {_build_message(accepted, results, min_similarity)}",
        "mock_used": True,
    }


def _load_mock() -> dict:
    if not MOCK_FILE.exists():
        logger.warning("Mock file not found: %s", MOCK_FILE)
        return {}
    try:
        with MOCK_FILE.open("r", encoding="utf-8") as file:
            return json.load(file).get("scenarios", {})
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read mock FAQ file: %s", exc)
        return {}


class FAQSearchTool:
    def __init__(self, min_similarity: float = DEFAULT_MIN_SIMILARITY):
        self._default_config = {
            "use_mock": False,
            "min_similarity": min_similarity,
        }

    async def search(self, query: str, block_id: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        result = faq_search(
            {
                "query": query,
                "block_id": block_id,
                "top_k": top_k,
            },
            config=self._default_config,
        )
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])

    async def search_by_block(self, block_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return []

    async def search_by_query(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        result = faq_search(
            {
                "query": query,
                "block_id": "",
                "top_k": top_k,
            },
            config=self._default_config,
        )
        if result.get("status") != "success":
            return []
        return result["data"].get("results", [])
