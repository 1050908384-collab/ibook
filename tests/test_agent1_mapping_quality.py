# -*- coding: utf-8 -*-
import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]
os.environ.setdefault("EXECUTION_MODE", "deterministic")

from agents.deterministic import DeterministicDiagnosticAgent
from models.contracts import DiagnosticInput
from tools.agent1_tools.knowledge_catalog_tool import KnowledgeCatalogTool
from tools.agent6_tools.profile_update_tool import ProfileUpdateTool


class Agent1MappingQualityTest(unittest.IsolatedAsyncioTestCase):
    async def test_multitarget_mapping_preserves_direct_prerequisite_frontier(self):
        agent = DeterministicDiagnosticAgent()
        result = await agent.run(DiagnosticInput(
            session_id="agent1-mapping-test",
            user_input="I know Python and PyTorch. I want to learn LoRA and deploy models with vLLM.",
        ))

        gap_ids = [gap.block_id for gap in result.profile.knowledge_gaps]
        self.assertIn("KB-12", gap_ids)
        self.assertIn("KB-15", gap_ids)
        self.assertNotIn("KB-09", gap_ids)
        self.assertNotIn("KB-04", gap_ids)
        self.assertLess(gap_ids.index("KB-11"), gap_ids.index("KB-12"))
        self.assertLess(gap_ids.index("KB-08"), gap_ids.index("KB-15"))

    def test_prompt_candidates_include_decision_context(self):
        catalog = KnowledgeCatalogTool()
        candidate = catalog.prompt_candidates("LoRA", top_k=1)[0]
        self.assertIn("description", candidate)
        self.assertIn("difficulty", candidate)
        self.assertIn("prerequisite_ids", candidate)
        self.assertIn("direct_prerequisite_ids", candidate["subtopics"][0])

    async def test_mastered_transformer_skips_vllm_transformer_prerequisite(self):
        agent = DeterministicDiagnosticAgent()
        result = await agent.run(DiagnosticInput(
            session_id="mastered-transformer-test",
            user_input="I know Transformer and want to learn LoRA and vLLM deployment.",
        ))
        self.assertEqual({"KB-08"}, agent._resolve_mastered_block_ids(result.profile.skill_tags))
        self.assertNotIn("KB-08", [gap.block_id for gap in result.profile.knowledge_gaps])

    async def test_qlora_maps_to_lora_block(self):
        agent = DeterministicDiagnosticAgent()
        result = await agent.run(DiagnosticInput(
            session_id="qlora-test",
            user_input="I want to learn QLoRA 4bit quantization finetuning.",
        ))
        target_gap = next(gap for gap in result.profile.knowledge_gaps if gap.block_id == "KB-12")
        self.assertEqual("QLoRA", target_gap.skill)

    async def test_wireless_lora_requests_disambiguation(self):
        agent = DeterministicDiagnosticAgent()
        result = await agent.run(DiagnosticInput(
            session_id="wireless-lora-test",
            user_input="I want to learn LoRa wireless communication technology.",
        ))
        self.assertLess(result.profile.confidence, 0.7)
        self.assertFalse(result.profile.knowledge_gaps[0].block_id)
        self.assertTrue(any("无线通信" in question for question in result.followup_questions))

    async def test_unknown_goal_is_low_confidence_even_with_structured_fields(self):
        agent = DeterministicDiagnosticAgent()
        text = (
            "我的身份背景：研究生\n"
            "我的已有基础：Python\n"
            "我的学习目标：robotics\n"
            "我每天可以学习：2小时\n"
            "我的学习方式：动手实操\n"
            "我的设备条件：RTX 4060"
        )
        result = await agent.run(DiagnosticInput(session_id="unknown-goal-test", user_input=text))
        self.assertLess(result.profile.confidence, 0.7)
        self.assertFalse(result.profile.knowledge_gaps[0].block_id)

    async def test_direct_frontier_defines_first_stage_subtopic(self):
        agent = DeterministicDiagnosticAgent()
        result = await agent.run(DiagnosticInput(
            session_id="zero-foundation-lora-test",
            user_input="I want to learn LoRA.",
        ))
        self.assertEqual("KB-04", result.profile.knowledge_gaps[0].block_id)
        block = agent.knowledge_catalog_tool.get_block("KB-04")
        self.assertEqual("KBS-04-01", block["subtopics"][0]["subtopic_id"])

    def test_semantic_evidence_retains_subtopic_direct_prerequisites(self):
        catalog = KnowledgeCatalogTool()
        candidates = catalog.merge_recall_results([], [{
            "block_id": "KB-15",
            "subtopic_id": "KBS-15-04",
            "title": "vLLM deployment",
            "difficulty": "advanced",
            "content_summary": "Deploy inference service",
            "direct_prerequisite_ids": ["KBS-15-01"],
            "similarity": 0.9,
        }])
        self.assertEqual("KB-15", candidates[0]["block_id"])
        self.assertEqual(
            ["KBS-15-01"],
            candidates[0]["matched_subtopics"][0]["direct_prerequisite_ids"],
        )

    async def test_agent6_update_transparently_preserves_agent1_metadata(self):
        gap = {
            "skill": "LoRA",
            "block_id": "KB-12",
            "block_name": "LoRA",
            "target_subtopic_ids": ["KBS-12-01"],
            "mapping_confidence": 0.91,
            "level": "beginner",
            "status": "unmastered",
            "progress": 0.0,
            "prerequisites": ["Transformer"],
        }
        updated = await ProfileUpdateTool().update_gaps([gap], accuracy=0.8)
        self.assertEqual("KB-12", updated[0]["block_id"])
        self.assertEqual("LoRA", updated[0]["block_name"])
        self.assertEqual(["KBS-12-01"], updated[0]["target_subtopic_ids"])
        self.assertEqual(0.91, updated[0]["mapping_confidence"])


if __name__ == "__main__":
    unittest.main()
