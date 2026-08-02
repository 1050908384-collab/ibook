# -*- coding: utf-8 -*-
"""Build the Agent1-only Chroma index from the canonical knowledge_docs seed."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path[:0] = [str(ROOT_DIR), str(BACKEND_DIR)]

from db.chroma_client import CHROMA_AVAILABLE, ChromaClient
from tools.agent1_tools.semantic_knowledge_recall_tool import COLLECTION_NAME


def build_agent1_semantic_index() -> int:
    if not CHROMA_AVAILABLE:
        raise RuntimeError("chromadb is not installed")

    rows = json.loads((ROOT_DIR / "data" / "seeds" / "knowledge_docs.json").read_text(encoding="utf-8"))
    client = ChromaClient().client
    collection = client.get_or_create_collection(COLLECTION_NAME)
    ids = [str(row["doc_id"]) for row in rows]
    documents = [f"{row.get('title', '')}\n{row.get('content', '')}" for row in rows]
    metadatas = [
        {
            "doc_id": str(row["doc_id"]),
            "block_id": str(row.get("block_id", "")),
            "subtopic_id": str(row.get("subtopic_id", "")),
            "title": str(row.get("title", "")),
            "difficulty": str(row.get("difficulty", "")),
        }
        for row in rows
    ]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return collection.count()


if __name__ == "__main__":
    print(f"Agent1 semantic index documents: {build_agent1_semantic_index()}")
