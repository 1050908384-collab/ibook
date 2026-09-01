# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]

from agents.deterministic import HybridContentAgent
from models.contracts import AnswerGenerationInput
from models.profile import UserProfile
from models.stage import StageInfo
from tools.agent4_tools.cognitive_load_tool import cognitive_load


class Agent3OutputBudgetTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stage = StageInfo(
            stage_id="budget-stage",
            stage_num=1,
            block_id="KB-12",
            block_name="Budget Test",
            gap_skill="LoRA",
        )

    async def test_block_falls_back_to_source_when_model_exceeds_budget(self):
        agent = HybridContentAgent()
        source = "A concise source paragraph about low-rank adaptation. " * 8
        agent.llm_client.complete = AsyncMock(return_value="Expanded output. " * 200)

        block = await agent.generate_block(
            self.stage,
            {"chunk_id": "budget-chunk", "content": source, "title": "Budget"},
        )

        # Budget fallback preserves the source inside the deterministic
        # learning-card wrapper rather than returning raw source verbatim.
        self.assertIn(source.strip(), block.personalized_content)
        self.assertTrue(block.personalized_content.strip())

    async def test_daily_answer_has_hard_character_limit(self):
        agent = HybridContentAgent()
        agent.llm_client.complete = AsyncMock(return_value=("A。" * 200))
        request = AnswerGenerationInput(
            session_id="budget-session",
            question="I feel tired today.",
            stage=self.stage,
            profile=UserProfile(session_id="budget-session", goal="Learn LoRA"),
        )

        answer = await agent._generate_daily_answer_llm(request)

        self.assertLessEqual(len(answer), 150)

    def test_review_tool_reports_source_budget_violation(self):
        result = cognitive_load({"content": "x" * 101, "max_output_length": 100})
        issue_types = {issue["issue_type"] for issue in result["data"]["issues"]}
        self.assertIn("verbosity_over_budget", issue_types)

    def test_agent3_loads_runtime_policies_from_skills(self):
        agent = HybridContentAgent()

        self.assertIn("source chunk", agent.content_skill_context)
        self.assertIn("150 characters", agent.answer_skill_context)
        self.assertNotIn("附录", agent.content_skill_context)


if __name__ == "__main__":
    unittest.main()
