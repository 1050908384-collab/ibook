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
        self.assertLess(gap_ids.index("KB-11"), gap_ids.index("KB-12"))
        self.assertLess(gap_ids.index("KB-08"), gap_ids.index("KB-15"))

    def test_prompt_candidates_include_decision_context(self):
        catalog = KnowledgeCatalogTool()
        candidate = catalog._prompt_candidate(catalog.search("LoRA", top_k=1)[0])
        self.assertIn("description", candidate)
        self.assertIn("difficulty", candidate)
        self.assertIn("prerequisite_ids", candidate)

    async def test_agent6_update_transparently_preserves_agent1_metadata(self):
        gap = {
            "skill": "LoRA",
            "block_id": "KB-11",
            "block_name": "LoRA",
            "target_subtopic_ids": ["KBS-11-01"],
            "mapping_confidence": 0.91,
            "level": "beginner",
            "status": "unmastered",
            "progress": 0.0,
            "prerequisites": ["Transformer"],
        }
        updated = await ProfileUpdateTool().update_gaps([gap], accuracy=0.8)
        self.assertEqual("KB-11", updated[0]["block_id"])
        self.assertEqual("LoRA", updated[0]["block_name"])
        self.assertEqual(["KBS-11-01"], updated[0]["target_subtopic_ids"])
        self.assertEqual(0.91, updated[0]["mapping_confidence"])


if __name__ == "__main__":
    unittest.main()
