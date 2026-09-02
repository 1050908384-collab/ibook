# -*- coding: utf-8 -*-
"""Evaluate 学情诊断与阶段规划 Agent's lightweight Chinese semantic retrieval against intent cases."""

from __future__ import annotations


import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path[:0] = [str(ROOT_DIR), str(BACKEND_DIR)]

from tools.agent1_tools.semantic_knowledge_recall_tool import SemanticKnowledgeRecallTool


CASES = [
    ("怎么让大模型占用显存更少跑得更快", {"KB-12", "KB-15"}),
    ("我想把模型变小一点", {"KB-12", "KB-15"}),
    ("部署大模型到生产环境", {"KB-15"}),
    ("多卡训练加速", {"KB-14"}),
    ("QLoRA 4bit 量化微调", {"KB-12"}),
]


def main() -> None:
    recall = SemanticKnowledgeRecallTool()
    if not recall.is_ready():
        raise RuntimeError(
            "学情诊断与阶段规划 Agent FastEmbed index is unavailable. Run "
            "python scripts/init_agent1_semantic_index.py."
        )

    failures = []
    for query, expected_block_ids in CASES:
        hits = recall.search(query, top_k=3)
        actual_block_ids = [hit["block_id"] for hit in hits]
        passed = bool(actual_block_ids and actual_block_ids[0] in expected_block_ids)
        print({
            "query": query,
            "expected_top1": sorted(expected_block_ids),
            "actual_top1": actual_block_ids[0] if actual_block_ids else None,
            "passed": passed,
        })
        if not passed:
            failures.append(query)
    if failures:
        raise SystemExit(f"Semantic retrieval evaluation failed: {failures}")


if __name__ == "__main__":
    main()
