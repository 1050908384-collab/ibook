# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path[:0] = [str(PROJECT_ROOT), str(BACKEND_DIR)]
os.environ.setdefault("EXECUTION_MODE", "deterministic")

from agents.deterministic import DeterministicDiagnosticAgent, HybridDiagnosticAgent
from db.sqlite_client import SQLiteClient
from models.contracts import StagePlanningInput
from models.profile import KnowledgeGap, UserProfile
from scripts.insert_knowledge_data import (
    BLOCKS_FILE_PATH,
    SUBTOPICS_FILE_PATH,
    insert_knowledge_blocks,
    insert_knowledge_subtopics,
    load_required_list,
)
from tools.agent1_tools.graph_query_tool import GraphQueryTool
from models.contracts import DiagnosticInput


class KnowledgeGraphUpgradeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.client = SQLiteClient(f"sqlite:///{(Path(self.temp_dir.name) / 'graph.db').as_posix()}")
        insert_knowledge_blocks(self.client, load_required_list(BLOCKS_FILE_PATH))
        insert_knowledge_subtopics(self.client, load_required_list(SUBTOPICS_FILE_PATH))

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    async def test_graph_has_existing_ids_and_valid_hard_edges(self):
        blocks = load_required_list(BLOCKS_FILE_PATH)
        subtopics = load_required_list(SUBTOPICS_FILE_PATH)
        self.assertEqual(15, len(blocks))
        self.assertEqual(73, len(subtopics))
        self.assertEqual({f"KB-{index:02d}" for index in range(1, 16)}, {row["block_id"] for row in blocks})
        report = json.loads((PROJECT_ROOT / "data/graph_data/current/derived/validation_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["missing_subtopic_prerequisites"])
        self.assertFalse(report["subtopic_cycles"])

        graph = GraphQueryTool()
        lora = await graph.minimal_hard_prerequisite_subgraph(["KBS-12-03"], "basic LoRA")
        self.assertIn("KBS-12-03", lora)
        self.assertNotIn("KBS-12-04", lora)
        self.assertNotIn("KBS-13-01", lora)

    async def test_soft_and_goal_relations_do_not_block_default_paths(self):
        graph = GraphQueryTool()
        general_sft = await graph.minimal_hard_prerequisite_subgraph(["KBS-10-02"], "general sft")
        chat_sft = await graph.minimal_hard_prerequisite_subgraph(["KBS-10-02"], "对话SFT")
        general_vllm = await graph.minimal_hard_prerequisite_subgraph(["KBS-15-04"], "deploy vLLM")
        tuned_vllm = await graph.minimal_hard_prerequisite_subgraph(["KBS-15-04"], "部署微调后模型")

        self.assertNotIn("KBS-07-04", general_sft)
        self.assertIn("KBS-07-04", chat_sft)
        self.assertNotIn("KBS-12-05", general_vllm)
        self.assertIn("KBS-12-05", tuned_vllm)

    async def test_training_optimization_is_goal_recommended_not_lora_blocker(self):
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        result = await agent.run(DiagnosticInput(
            session_id="optimization-goal",
            user_input="I want high efficiency finetuning with stable training.",
        ))
        self.assertIn("KB-13", [gap.block_id for gap in result.profile.knowledge_gaps])

    async def test_agent1_selects_unlocked_node_from_hard_closure(self):
        session_id = "hard-closure-stage"
        self.client.create_session(session_id)
        graph = GraphQueryTool()
        levels = await graph.minimal_hard_prerequisite_subgraph(["KBS-12-03"], "basic LoRA")
        for subtopic_id in levels:
            if subtopic_id != "KBS-12-03":
                self.client.set_subtopic_status(session_id, subtopic_id, 1, 1.0, 1.0, False)

        agent = DeterministicDiagnosticAgent(db_client=self.client)
        profile = UserProfile(
            session_id=session_id,
            goal="basic LoRA",
            knowledge_gaps=[KnowledgeGap(skill="LoRA", block_id="KB-12", target_subtopic_ids=["KBS-12-03"])],
        )
        result = await agent.generate_stage(StagePlanningInput(session_id=session_id, profile=profile, stage_num=1))
        self.assertEqual("KBS-12-03", result.stage.subtopic_id)

    async def test_lora_stage_resumes_user_level_in_progress_prerequisite(self):
        session_id = "lora-with-in-progress-prerequisites"
        self.client.create_session(session_id)
        self.client.set_subtopic_status(session_id, "KBS-01-00", 0)
        self.client.set_subtopic_status(session_id, "KBS-02-00", 0)

        agent = DeterministicDiagnosticAgent(db_client=self.client)
        profile = UserProfile(
            session_id=session_id,
            goal="掌握Lora",
            daily_hours=2,
            knowledge_gaps=[KnowledgeGap(
                skill="LoRA",
                block_id="KB-12",
                target_subtopic_ids=["KBS-12-03"],
            )],
        )

        result = await agent.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=profile,
            stage_num=1,
        ))

        self.assertEqual("KBS-02-00", result.stage.subtopic_id)
        self.assertTrue(result.stage.subtopic_ids)

    async def test_stage_planning_falls_back_when_all_candidates_are_blocked(self):
        session_id = "blocked-stage-candidates"
        self.client.create_session(session_id)
        graph = GraphQueryTool()
        levels = await graph.minimal_hard_prerequisite_subgraph(["KBS-12-03"], "掌握Lora")
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        stored_statuses = self.client.list_subtopic_statuses(session_id)
        agent.db_client.list_subtopic_statuses = lambda _: [
            item for item in stored_statuses
            if levels.get(item["subtopic_id"], -1) != 0
        ]

        async def empty_fallback(*_args, **_kwargs):
            return {}

        agent.stage_generator_tool.generate_stage = empty_fallback
        profile = UserProfile(
            session_id=session_id,
            goal="掌握Lora",
            daily_hours=2,
            knowledge_gaps=[KnowledgeGap(
                skill="LoRA",
                block_id="KB-12",
                target_subtopic_ids=["KBS-12-03"],
            )],
        )

        result = await agent.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=profile,
            stage_num=1,
        ))

        self.assertEqual("KB-12", result.stage.block_id)
        self.assertIsNone(result.stage.subtopic_id)

    async def test_agent1_parses_zero_foundation_and_calculates_stage_days(self):
        session_id = "linux-time-budget"
        self.client.create_session(session_id)
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        diagnostic = await agent.run(DiagnosticInput(
            session_id=session_id,
            user_input=(
                "学习目标：我想要掌握 Linux\n"
                "已掌握基础：0基础\n"
                "每日学习时长：0.5小时\n"
                "学习方式：混合"
            ),
        ))
        self.assertEqual([], diagnostic.profile.skill_tags)
        self.assertEqual("KB-01", diagnostic.profile.knowledge_gaps[0].block_id)

        first_stage = (await agent.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=1,
        ))).stage
        self.assertEqual("KBS-01-00", first_stage.subtopic_id)
        self.assertEqual(0.5, first_stage.estimated_hours)
        self.assertEqual(0.5, first_stage.recommended_daily_hours)
        self.assertEqual(1, first_stage.suggested_study_days)

        self.client.set_subtopic_status(session_id, "KBS-01-00", 1, 1.0, 1.0, False)
        second_stage = (await agent.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=2,
        ))).stage
        self.assertEqual("KBS-01-01", second_stage.subtopic_id)
        self.assertEqual(1.0, second_stage.estimated_hours)
        self.assertEqual(0.5, second_stage.recommended_daily_hours)
        self.assertEqual(2, second_stage.suggested_study_days)

    async def test_partial_foundation_is_visible_in_profile_tags(self):
        session_id = "partial-linux-foundation"
        self.client.create_session(session_id)
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        block = agent.knowledge_catalog_tool.get_block("KB-01")
        selected = next(
            item for item in block["subtopics"]
            if item["subtopic_id"] == "KBS-01-01"
        )

        diagnostic = await agent.run_with_foundation(
            session_id=session_id,
            goal="掌握 Linux",
            mastered_subtopic_ids=[selected["subtopic_id"]],
        )

        self.assertIn(selected["subtopic_name"], diagnostic.profile.skill_tags)
        self.assertNotIn(block["block_name"], diagnostic.profile.skill_tags)

    async def test_python_path_does_not_require_linux_environment(self):
        session_id = "python-without-linux"
        self.client.create_session(session_id)
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        diagnostic = await agent.run(DiagnosticInput(
            session_id=session_id,
            user_input="学习目标：我想掌握 Python\n已掌握基础：无\n每日学习时长：2小时",
        ))
        gap = diagnostic.profile.knowledge_gaps[0]
        self.assertEqual("KB-02", gap.block_id)
        self.assertEqual([], gap.prerequisites)

        stage = (await agent.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=1,
        ))).stage
        self.assertEqual("KBS-02-00", stage.subtopic_id)
        self.assertEqual([], stage.prerequisites)

    async def test_stale_semantic_id_is_filtered_and_lora_path_stays_in_closure(self):
        class StaleSemanticRecall:
            def search(self, query, top_k=8):
                return [{
                    "block_id": "KB-12", "subtopic_id": "KBS-12-06",
                    "title": "retired LoRA document", "similarity": 0.99,
                }]

        session_id = "stale-semantic-lora"
        self.client.create_session(session_id)
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        agent.semantic_knowledge_recall_tool = StaleSemanticRecall()
        diagnostic = await agent.run(DiagnosticInput(session_id=session_id, user_input="I want to learn LoRA."))
        gap = next(item for item in diagnostic.profile.knowledge_gaps if item.block_id == "KB-12")
        self.assertEqual(["KBS-12-03"], gap.target_subtopic_ids)

        levels = await agent.graph_query_tool.minimal_hard_prerequisite_subgraph(
            gap.target_subtopic_ids, diagnostic.profile.goal
        )
        planned_ids = []
        for stage_num in range(1, len(levels) + 2):
            stage = (await agent.generate_stage(StagePlanningInput(
                session_id=session_id, profile=diagnostic.profile, stage_num=stage_num,
            ))).stage
            self.assertIn(stage.subtopic_id, levels)
            planned_ids.append(stage.subtopic_id)
            self.client.set_subtopic_status(session_id, stage.subtopic_id, 1, 1.0, 1.0, False)
            if stage.subtopic_id == "KBS-12-03":
                break
        self.assertEqual("KBS-12-03", planned_ids[-1])
        self.assertNotIn("KBS-12-04", planned_ids)

    async def test_plain_lora_targets_basic_lora_config_after_index_rebuild(self):
        agent = DeterministicDiagnosticAgent(db_client=self.client)
        result = await agent.run(DiagnosticInput(
            session_id="plain-lora-target", user_input="I want to learn LoRA."
        ))
        gap = next(item for item in result.profile.knowledge_gaps if item.block_id == "KB-12")
        self.assertEqual(["KBS-12-03"], gap.target_subtopic_ids)

    async def test_hybrid_agent_does_not_require_hardware_profile_field(self):
        class FixedClient:
            async def complete(self, *args, **kwargs):
                return '{"knowledge_gaps": [], "low_confidence": false}'

        agent = HybridDiagnosticAgent(db_client=self.client)
        agent.llm_client = FixedClient()
        result = await agent.run(DiagnosticInput(
            session_id="hybrid-no-hardware", user_input="I want to learn LoRA."
        ))
        self.assertFalse(hasattr(result.profile, "hardware"))

    def test_sqlite_uses_corrected_hard_prerequisites(self):
        self.client.create_session("sqlite-hard-edge")
        rows = {row["subtopic_id"]: row for row in self.client.list_subtopic_statuses("sqlite-hard-edge")}
        self.assertEqual(["KBS-12-00", "KBS-12-02"], json.loads(rows["KBS-12-03"]["prerequisites"]))
        self.assertEqual(["KBS-08-02"], json.loads(rows["KBS-15-01"]["prerequisites"]))


if __name__ == "__main__":
    unittest.main()
