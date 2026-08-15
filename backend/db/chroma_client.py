# -*- coding: utf-8 -*-
# Chroma 向量库连接
# Embedding 使用 Ollama 的 qwen3-embedding:0.6b 模型（中文语义检索）
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from chromadb import PersistentClient
    from chromadb.api.types import D, EmbeddingFunction, Embeddings, Space
    import numpy as np  # noqa: F401  # Chroma EF 依赖 numpy
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config import BACKEND_DIR, settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# 统一 collection 名，避免写入端 / 读取端名称不一致导致读写分离。
# init_knowledge_base.py 与 vector_search_tool.py 均引用此常量。
CHROMA_COLLECTION = "knowledge_base"


class _OllamaEFProxySafe(EmbeddingFunction):
    """绕过系统代理的 Ollama EmbeddingFunction。

    Chroma 自带的 OllamaEmbeddingFunction 调用 ollama-python，其 httpx.Client
    默认 ``trust_env=True``，会读取 HTTP_PROXY 等环境变量导致 502。
    本实现直接使用 ``httpx.Client(trust_env=False)`` 调用 /api/embed。
    """

    def __init__(self, url: str, model_name: str, timeout: int = 180, max_retries: int = 3):
        import httpx
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        # 显式禁止读取环境代理，避免 502
        self._client = httpx.Client(trust_env=False, timeout=timeout)

    def __call__(self, input: D) -> Embeddings:  # noqa: D401 - 匹配 Chroma 基类签名
        import numpy as np
        inputs = list(input)
        if not inputs:
            return []
        # truncate=True：超长文本由 Ollama 自动截断，避免静默失败
        # Truncate oversized chunks at the embedding model's context limit.
        payload = {"model": self.model_name, "input": inputs, "truncate": True}
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(f"{self.url}/api/embed", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return [np.array(e, dtype=np.float32) for e in data["embeddings"]]
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.warning(
                        "Ollama embed 第 %d/%d 次失败，%ds 后重试: %s",
                        attempt, self.max_retries, backoff, e,
                    )
                    time.sleep(backoff)
        # 全部重试失败：向上抛出，由调用方降级（init 崩溃 / search 回退 SQLite）
        raise RuntimeError(
            f"Ollama embed 连续 {self.max_retries} 次失败 (url={self.url}, "
            f"model={self.model_name}): {last_err}"
        )

    @staticmethod
    def name() -> str:
        return "ollama_proxy_safe"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "model_name": self.model_name,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_OllamaEFProxySafe":
        return _OllamaEFProxySafe(
            url=config["url"],
            model_name=config["model_name"],
            timeout=config.get("timeout", 180),
            max_retries=config.get("max_retries", 3),
        )


def get_embedding_function():
    """返回代理安全的 Ollama embedding function。

    使用 settings.OLLAMA_BASE_URL + settings.EMBEDDING_MODEL（默认 qwen3-embedding:0.6b）。
    若 chromadb / httpx 未安装或构造失败，返回 None 并记录错误日志。

    ⚠ 调用方注意：返回 None 时**绝不能**用 None 作为 embedding_function
    创建 collection——Chroma 会静默回退到英文 all-MiniLM-L6-v2 模型，
    导致中文检索 sim 全部失真。调用方必须据此降级（回退 SQLite 或报错）。
    """
    if not CHROMA_AVAILABLE:
        return None
    try:
        return _OllamaEFProxySafe(
            url=settings.OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL,
        )
    except Exception as e:
        logger.error(
            "创建 Ollama embedding function 失败 (url=%s, model=%s): %s。"
            "Chroma 向量检索将不可用，回退 SQLite 关键词匹配。",
            settings.OLLAMA_BASE_URL, settings.EMBEDDING_MODEL, e,
        )
        return None


class ChromaClient:
    """Chroma 向量库客户端封装。

    降级策略：chromadb 未安装或 embedding function 不可用时，
    ``self.collection`` 置 None，所有操作 no-op，调用方回退 SQLite。
    """

    def __init__(self):
        if CHROMA_AVAILABLE:
            chroma_path = Path(settings.CHROMA_PATH)
            if not chroma_path.is_absolute():
                chroma_path = (BACKEND_DIR / chroma_path).resolve()
            self.client = PersistentClient(path=str(chroma_path))
            self._embedding_function = get_embedding_function()
            # 关键防御：EF 为 None 时绝不用 None 创建 collection，
            # 否则 Chroma 会静默回退到英文 all-MiniLM-L6-v2 模型，
            # 导致中文检索 sim 全部失真（正是此前修复问题的隐性版本）。
            if self._embedding_function is None:
                logger.error(
                    "Embedding function 不可用，ChromaClient 进入降级模式"
                    "（collection 不创建，所有操作 no-op）"
                )
                self.collection = None
            else:
                self.collection = self.client.get_or_create_collection(
                    CHROMA_COLLECTION,
                    embedding_function=self._embedding_function,
                )
        else:
            self.client = None
            self.collection = None
            self._embedding_function = None

    def _available(self) -> bool:
        """Chroma 是否可用（已安装且 EF 就绪且 collection 已创建）。"""
        return CHROMA_AVAILABLE and self.collection is not None

    def add_documents(self, documents: list, metadatas: list = None, ids: list = None) -> None:
        if not self._available():
            return
        # upsert：重复 ID 覆盖而非报错，幂等写入（init 脚本可安全重跑）
        # metadatas/ids 传 None 时由 Chroma 处理，不再用 [] 触发长度不匹配
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, n_results: int = 5, where: dict = None) -> list:
        if not self._available():
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results

    def get_by_id(self, doc_id: str) -> dict:
        if not self._available():
            return {}
        results = self.collection.get(ids=[doc_id])
        return results

    def delete_by_id(self, doc_id: str) -> None:
        if not self._available():
            return
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        if not self._available():
            return 0
        return self.collection.count()
