# -*- coding: utf-8 -*-
"""Build the Agent1-only Chroma index from the canonical knowledge_docs seed."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path[:0] = [str(ROOT_DIR), str(BACKEND_DIR)]

from tools.agent1_tools.semantic_knowledge_recall_tool import (
    Agent1ChromaClient,
    MODEL_CACHE_DIR,
    MODEL_READY_FILE,
)


RETRIEVAL_HINTS = {
    "KB-12": "显存更少，模型变小，量化，4bit，低显存微调，参数高效微调",
    "KB-14": "多卡训练，分布式训练，训练加速，DeepSpeed，ZeRO",
    "KB-15": "生产环境部署，模型服务化，推理加速，跑得更快，vLLM，高并发推理",
}


def build_agent1_semantic_index() -> int:
    rows = json.loads((ROOT_DIR / "data" / "seeds" / "knowledge_docs.json").read_text(encoding="utf-8"))
    blocks = json.loads((ROOT_DIR / "data" / "seeds" / "knowledge_blocks.json").read_text(encoding="utf-8"))
    subtopics = json.loads((ROOT_DIR / "data" / "seeds" / "knowledge_subtopics.json").read_text(encoding="utf-8"))
    block_index = {str(row.get("block_id", "")): row for row in blocks}
    subtopic_index = {str(row.get("subtopic_id", "")): row for row in subtopics}
    collection = Agent1ChromaClient().rebuild_collection()
    ids = [str(row["doc_id"]) for row in rows]
    documents = []
    for row in rows:
        block = block_index.get(str(row.get("block_id", "")), {})
        documents.append("\n".join([
            str(row.get("title", "")),
            str(row.get("content", "")),
            f"知识板块：{block.get('block_name', '')}",
            f"板块说明：{block.get('description', '')}",
            f"相关目标：{'、'.join(block.get('corresponding_gaps', []))}",
            f"同义需求：{RETRIEVAL_HINTS.get(str(row.get('block_id', '')), '')}",
        ]))
    metadatas = [
        {
            "doc_id": str(row["doc_id"]),
            "block_id": str(row.get("block_id", "")),
            "subtopic_id": str(row.get("subtopic_id", "")),
            "title": str(row.get("title", "")),
            "difficulty": str(row.get("difficulty", "")),
            "sequence_index": int(subtopic_index.get(str(row.get("subtopic_id", "")), {}).get("sequence_index") or 0),
            "direct_prerequisite_ids": json.dumps(
                subtopic_index.get(str(row.get("subtopic_id", "")), {}).get("prerequisites", []),
                ensure_ascii=False,
            ),
            "content_summary": str(subtopic_index.get(str(row.get("subtopic_id", "")), {}).get("content_summary", "")),
        }
        for row in rows
    ]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_FILE.write_text("initialized\n", encoding="utf-8")
    return collection.count()


if __name__ == "__main__":
    print(f"Agent1 semantic index documents: {build_agent1_semantic_index()}")
