# -*- coding: utf-8 -*-
"""Agent1 专用的知识文档语义召回工具。"""

from __future__ import annotations

from typing import Any

try:
    from db.chroma_client import CHROMA_AVAILABLE
    from db.chroma_client import ChromaClient
except ModuleNotFoundError:
    from backend.db.chroma_client import CHROMA_AVAILABLE
    from backend.db.chroma_client import ChromaClient


COLLECTION_NAME = "agent1_knowledge_docs"


class SemanticKnowledgeRecallTool:
    """Query the Agent1-only semantic index without changing other agents' stores."""

    def __init__(self) -> None:
        self.collection = None
        if not CHROMA_AVAILABLE:
            return
        try:
            client = ChromaClient().client
            self.collection = client.get_or_create_collection(COLLECTION_NAME)
        except Exception:
            self.collection = None

    def is_ready(self) -> bool:
        try:
            return bool(self.collection and self.collection.count() > 0)
        except Exception:
            return False

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Return semantic evidence grouped later by the knowledge catalog."""
        if not query.strip() or not self.is_ready():
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
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
                similarity = 1.0 / (1.0 + max(0.0, float(distance)))
            except (TypeError, ValueError):
                similarity = 0.0
            recalled.append({
                "doc_id": str(metadata.get("doc_id", "")),
                "block_id": str(metadata.get("block_id", "")),
                "subtopic_id": str(metadata.get("subtopic_id", "")),
                "title": str(metadata.get("title", "")),
                "difficulty": str(metadata.get("difficulty", "")),
                "content_summary": str(document or "")[:500],
                "similarity": round(similarity, 4),
            })
        return recalled
