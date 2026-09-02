# -*- coding: utf-8 -*-
"""学情诊断与阶段规划 Agent 专用的知识文档语义召回工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

# chromadb / fastembed 是重依赖（向量库 + 本地 embedding），仅 学情诊断与阶段规划 Agent 语义召回真正需要。
# 为了让不依赖向量库的链路（如 专业知识与教学质量审核 Agent/6 审核与决策测试）能在未安装这两个包的环境下
# 正常 import，这里改为惰性加载：模块导入不再强依赖，只有真正调用到时才导入并报清晰错误。
if TYPE_CHECKING:  # 仅供类型检查，运行期不导入
    from chromadb import PersistentClient
    from fastembed import TextEmbedding


def _load_persistent_client():
    try:
        from chromadb import PersistentClient
    except ModuleNotFoundError as exc:  # 缺依赖时给出可操作的提示
        raise ModuleNotFoundError(
            "语义召回需要 chromadb，请先安装：pip install chromadb"
        ) from exc
    return PersistentClient


def _load_text_embedding():
    try:
        from fastembed import TextEmbedding
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "语义召回需要 fastembed，请先安装：pip install fastembed"
        ) from exc
    return TextEmbedding

try:
    from config import BACKEND_DIR, DATA_DIR, settings
except ModuleNotFoundError:
    from backend.config import BACKEND_DIR, DATA_DIR, settings


COLLECTION_NAME = "agent1_knowledge_docs"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dimension": EMBEDDING_DIMENSION,
}
MODEL_CACHE_DIR = DATA_DIR / "fastembed_models"
MODEL_READY_FILE = MODEL_CACHE_DIR / ".agent1_bge_small_zh_ready"
INTENT_EXPANSIONS = (
    (("模型变小", "模型压缩", "更小的模型"), "量化 QLoRA 4bit 低显存微调"),
    (("显存更少", "显存不足", "减少显存"), "QLoRA 量化 4bit 低显存微调"),
    (("跑得更快", "推理加速", "推理更快"), "vLLM 推理加速 高并发部署"),
    (("生产环境", "服务化部署", "线上部署"), "vLLM 生产环境 模型服务 推理部署"),
    (("多卡", "分布式训练"), "DeepSpeed ZeRO 多卡分布式训练加速"),
)


class FastEmbedChineseEmbeddingFunction:
    """Lazy FastEmbed adapter that prevents normal users from implicit downloads."""

    def __init__(self, allow_model_load: bool = False) -> None:
        self.allow_model_load = allow_model_load
        self._model: TextEmbedding | None = None

    @staticmethod
    def name() -> str:
        return "fastembed-bge-small-zh-v1.5"

    def get_config(self) -> dict[str, Any]:
        return {"model_name": EMBEDDING_MODEL}

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, query=False)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, query=True)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input, query=False)

    def _embed(self, texts: list[str], query: bool) -> list[list[float]]:
        if not self.allow_model_load and not MODEL_READY_FILE.exists():
            raise RuntimeError(
                "学情诊断与阶段规划 Agent FastEmbed model is not initialized. Run the explicit index initialization script."
            )
        if self._model is None:
            model_kwargs: dict[str, Any] = {}
            cached_model_path = self._cached_model_path()
            if cached_model_path:
                model_kwargs["specific_model_path"] = str(cached_model_path)
            elif not self.allow_model_load:
                raise RuntimeError("学情诊断与阶段规划 Agent FastEmbed model files are unavailable.")
            self._model = _load_text_embedding()(
                model_name=EMBEDDING_MODEL,
                cache_dir=str(MODEL_CACHE_DIR),
                cuda=False,
                **model_kwargs,
            )
        embeddings = self._model.query_embed(texts) if query else self._model.embed(texts)
        return [embedding.tolist() for embedding in embeddings]

    @staticmethod
    def _cached_model_path() -> Path | None:
        snapshots_dir = MODEL_CACHE_DIR / "models--Qdrant--bge-small-zh-v1.5" / "snapshots"
        for model_file in snapshots_dir.glob("*/model_optimized.onnx"):
            return model_file.parent
        return None


class Agent1ChromaClient:
    """Own the 学情诊断与阶段规划 Agent collection and embedding configuration only."""

    def __init__(self) -> None:
        chroma_path = Path(settings.CHROMA_PATH)
        if not chroma_path.is_absolute():
            chroma_path = (BACKEND_DIR / chroma_path).resolve()
        self.client = _load_persistent_client()(path=str(chroma_path))
        self.embedding_function = FastEmbedChineseEmbeddingFunction()

    def get_collection(self):
        try:
            collection = self.client.get_collection(
                COLLECTION_NAME,
                embedding_function=self.embedding_function,
            )
        except Exception:
            return None
        if (collection.metadata or {}).get("hnsw:space") != "cosine":
            return None
        return collection

    def rebuild_collection(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return self.client.create_collection(
            COLLECTION_NAME,
            embedding_function=FastEmbedChineseEmbeddingFunction(allow_model_load=True),
            metadata=COLLECTION_METADATA,
        )


class SemanticKnowledgeRecallTool:
    """Query the 学情诊断与阶段规划 Agent-only semantic index without changing other agents' stores."""

    def __init__(self) -> None:
        try:
            self.collection = Agent1ChromaClient().get_collection()
        except Exception:
            self.collection = None

    def is_ready(self) -> bool:
        try:
            return bool(
                self.collection
                and self.collection.count() > 0
                and MODEL_READY_FILE.exists()
                and FastEmbedChineseEmbeddingFunction._cached_model_path()
            )
        except Exception:
            return False

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Return semantic evidence grouped later by the knowledge catalog."""
        if not query.strip() or not self.is_ready():
            return []
        try:
            results = self.collection.query(
                query_texts=[self._expand_query(query)],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        recalled: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            try:
                similarity = 1.0 - min(1.0, max(0.0, float(distance)))
            except (TypeError, ValueError):
                similarity = 0.0
            try:
                direct_prerequisite_ids = json.loads(metadata.get("direct_prerequisite_ids", "[]"))
            except (TypeError, ValueError, json.JSONDecodeError):
                direct_prerequisite_ids = []
            recalled.append({
                "doc_id": str(metadata.get("doc_id", "")),
                "block_id": str(metadata.get("block_id", "")),
                "subtopic_id": str(metadata.get("subtopic_id", "")),
                "title": str(metadata.get("title", "")),
                "difficulty": str(metadata.get("difficulty", "")),
                "content_summary": str(metadata.get("content_summary") or document or "")[:500],
                "sequence_index": int(metadata.get("sequence_index") or 0),
                "direct_prerequisite_ids": direct_prerequisite_ids,
                "similarity": round(similarity, 4),
            })
        best_by_block: dict[str, dict[str, Any]] = {}
        for hit in recalled:
            block_id = hit["block_id"]
            if not block_id or float(hit["similarity"]) <= float(best_by_block.get(block_id, {}).get("similarity", -1)):
                continue
            best_by_block[block_id] = hit
        return sorted(
            best_by_block.values(),
            key=lambda item: -float(item["similarity"]),
        )[:max(1, top_k)]

    @staticmethod
    def _expand_query(query: str) -> str:
        lowered = query.casefold()
        expansions = [
            expansion
            for terms, expansion in INTENT_EXPANSIONS
            if any(term in lowered for term in terms)
        ]
        return " ".join([query, *expansions])
