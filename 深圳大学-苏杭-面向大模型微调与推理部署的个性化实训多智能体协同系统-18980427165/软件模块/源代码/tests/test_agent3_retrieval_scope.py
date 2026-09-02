# -*- coding: utf-8 -*-
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]

from agents.deterministic import DeterministicContentAgent
from models.contracts import AnswerGenerationInput
from models.profile import UserProfile
from models.stage import StageInfo
from tools.agent3_tools.vector_search_tool import _search_sqlite_fallback


class Agent3RetrievalScopeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = DeterministicContentAgent()
        self.stage = StageInfo(
            stage_id="cross-block-stage",
            stage_num=1,
            block_id="KB-12",
            block_name="混合阶段",
            block_ids=["KB-12", "KB-15"],
            subtopic_id="KBS-12-01",
            subtopic_ids=["KBS-12-01", "KBS-15-03"],
            subtopic_names=["LoRA", "PagedAttention"],
            gap_skill="参数高效微调与推理部署",
        )

    def test_stage_resolves_all_blocks_from_subtopics(self):
        self.assertEqual(
            {"KB-12", "KB-15"},
            set(self.agent._stage_block_ids(self.stage)),
        )

    def test_bm25_fallback_handles_continuous_chinese_and_scope(self):
        results, available = _search_sqlite_fallback(
            "为什么PagedAttention可以减少KVCache显存碎片",
            ["KB-12", "KB-15"],
            ["KBS-12-01", "KBS-15-03"],
            3,
        )
        self.assertTrue(available)
        self.assertTrue(results)
        self.assertTrue(all(item["block_id"] == "KB-15" for item in results))

    async def test_technical_answer_passes_multiblock_scope_to_search(self):
        profile = UserProfile(session_id="cross-block-session", goal="掌握微调和推理")
        request = AnswerGenerationInput(
            session_id="cross-block-session",
            question="PagedAttention 为什么减少显存碎片？",
            stage=self.stage,
            profile=profile,
            block_id="KB-12",
        )
        with patch.object(self.agent.vector_search_tool, "search", return_value=[]) as search:
            with patch.object(self.agent.vector_search_tool, "search_by_query", return_value=[]):
                await self.agent.generate_answer(request)

        search.assert_awaited_once()
        _, kwargs = search.call_args
        self.assertEqual({"KB-12", "KB-15"}, set(kwargs["block_ids"]))
        self.assertEqual({"KBS-12-01", "KBS-15-03"}, set(kwargs["subtopic_ids"]))


if __name__ == "__main__":
    unittest.main()
