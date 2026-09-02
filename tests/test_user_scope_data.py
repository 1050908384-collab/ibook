import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from db.sqlite_client import DEFAULT_USER_ID, SQLiteClient
from models.stage import StageInfo
from orchestrator.state_manager import StateManager
from services.learning_evidence_service import LearningEvidenceService
from tools.agent4_tools.personalization_check_tool import PersonalizationCheckTool


class UserScopeDataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = Path(self.temp_dir.name) / "user-scope.db"
        self.database_url = f"sqlite:///{database.as_posix()}"
        self.client = SQLiteClient(self.database_url)
        with self.client.Session() as session:
            session.execute(text("""
                INSERT INTO knowledge_blocks (block_id, block_name, estimated_hours)
                VALUES ('KB-01', 'Python基础', 3.0)
            """))
            session.execute(text("""
                INSERT INTO knowledge_subtopics (
                    subtopic_id, block_id, subtopic_name, estimated_hours,
                    sequence_index, prerequisites
                ) VALUES
                    ('KBS-01-01', 'KB-01', '语法基础', 1.0, 1, '[]'),
                    ('KBS-01-02', 'KB-01', '函数与模块', 2.0, 2, '["KBS-01-01"]')
            """))
            session.commit()

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def _profile(self, session_id: str, goal: str) -> dict:
        return {
            "session_id": session_id,
            "user_id": DEFAULT_USER_ID,
            "grade": "开发工程师",
            "skill_tags": ["Python"],
            "knowledge_gaps": [{
                "skill": "Python基础",
                "block_id": "KB-01",
                "target_subtopic_ids": ["KBS-01-01", "KBS-01-02"],
            }],
            "goal": goal,
            "daily_hours": 1.5,
            "learning_style": "practice",
            "hardware": "笔记本电脑",
        }

    def test_profile_is_user_scoped_and_target_data_is_session_scoped(self):
        self.client.create_session("goal-a")
        self.client.create_session("goal-b")
        self.client.save_profile("goal-a", self._profile("goal-a", "学习 Python"))
        self.client.save_profile("goal-b", {
            "session_id": "goal-b",
            "goal": "完成自动化项目",
            "knowledge_gaps": [],
            "daily_hours": 2.0,
        })

        user = self.client.get_user_profile(DEFAULT_USER_ID)
        first = self.client.get_latest_profile("goal-a")
        second = self.client.get_latest_profile("goal-b")
        self.assertEqual("开发工程师", user["grade"])
        self.assertEqual(["Python"], second["skill_tags"])
        self.assertEqual("学习 Python", first["goal"])
        self.assertEqual("完成自动化项目", second["goal"])
        with self.client.Session() as session:
            legacy_count = session.execute(text("SELECT COUNT(*) FROM user_profiles")).scalar_one()
        self.assertEqual(2, legacy_count)

    def test_foundation_confirmation_persists_even_without_skill_tags(self):
        self.client.create_session("goal-a")
        self.client.save_profile("goal-a", {
            **self._profile("goal-a", "学习 Python"),
            "skill_tags": [],
            "foundation_confirmed": True,
        })

        user = self.client.get_user_profile(DEFAULT_USER_ID)
        composed = self.client.get_latest_profile("goal-a")

        self.assertTrue(user["foundation_confirmed"])
        self.assertEqual([], user["skill_tags"])
        self.assertTrue(composed["foundation_confirmed"])
        self.assertEqual([], composed["skill_tags"])

    def test_subtopic_mastery_is_shared_across_goals(self):
        self.client.create_session("goal-a")
        self.client.create_session("goal-b")
        self.client.set_subtopic_status(
            "goal-a", "KBS-01-01", status=1, mastery_score=0.95,
            last_accuracy=0.9, increment_attempt=True,
        )
        shared = self.client.get_subtopic_status("goal-b", "KBS-01-01")
        self.assertEqual(1, shared["status"])
        self.assertEqual(0.95, shared["mastery_score"])
        self.assertEqual(1, shared["attempt_count"])

    def test_initial_foundation_can_be_replaced_without_erasing_learned_mastery(self):
        self.client.replace_user_initial_foundation(DEFAULT_USER_ID, ["KBS-01-01"])
        self.client.set_user_subtopic_status(
            DEFAULT_USER_ID, "KBS-01-01", status=1, mastery_score=0.85,
            is_initial_foundation=True, attempt_count=2,
        )
        self.client.set_user_subtopic_status(
            DEFAULT_USER_ID, "KBS-01-02", status=1, mastery_score=0.9,
            is_initial_foundation=False, attempt_count=2,
        )

        self.client.replace_user_initial_foundation(DEFAULT_USER_ID, ["KBS-01-02"])

        removed = self.client.get_user_subtopic_status(DEFAULT_USER_ID, "KBS-01-01")
        retained = self.client.get_user_subtopic_status(DEFAULT_USER_ID, "KBS-01-02")
        self.assertEqual(1, removed["status"])
        self.assertEqual(0.85, removed["mastery_score"])
        self.assertFalse(removed["is_initial_foundation"])
        self.assertEqual(1, retained["status"])
        self.assertTrue(retained["is_initial_foundation"])
        self.assertEqual(2, retained["attempt_count"])

    def test_target_and_stage_durations_can_be_updated_independently(self):
        self.client.create_session("goal-a")
        self.client.save_profile("goal-a", self._profile("goal-a", "学习 Python"))
        self.client.save_stage("goal-a", {
            "stage_id": "stage-a",
            "stage_num": 1,
            "block_id": "KB-01",
            "block_name": "Python基础",
            "subtopic_id": "KBS-01-01",
            "estimated_hours": 1.25,
        })
        target = self.client.update_target_settings("goal-a", {"target_daily_hours": 2.5})
        stage = self.client.update_stage_duration("goal-a", "stage-a", 1.75)
        self.assertEqual(3.0, target["target_total_hours"])
        self.assertEqual(2.5, target["target_daily_hours"])
        self.assertEqual(1.75, stage["estimated_hours"])

    def test_target_total_hours_only_redistributes_unfinished_stages(self):
        self.client.create_session("goal-a")
        self.client.save_profile("goal-a", self._profile("goal-a", "学习 Python"))
        self.client.save_stage("goal-a", {
            "stage_id": "stage-completed",
            "stage_num": 1,
            "block_name": "语法基础",
            "estimated_hours": 1.5,
            "status": "completed",
        })
        self.client.save_stage("goal-a", {
            "stage_id": "stage-learning",
            "stage_num": 2,
            "block_name": "函数与模块",
            "estimated_hours": 1.0,
            "status": "learning",
        })
        self.client.save_stage("goal-a", {
            "stage_id": "stage-pending",
            "stage_num": 3,
            "block_name": "综合练习",
            "estimated_hours": 2.0,
            "status": "pending",
        })

        target = self.client.update_target_settings("goal-a", {"target_total_hours": 7.5})
        stages = {stage["stage_id"]: stage for stage in self.client.list_stage_progress("goal-a")}

        self.assertEqual(7.5, target["target_total_hours"])
        self.assertEqual(1.5, stages["stage-completed"]["estimated_hours"])
        self.assertEqual(2.0, stages["stage-learning"]["estimated_hours"])
        self.assertEqual(4.0, stages["stage-pending"]["estimated_hours"])

    def test_refresh_target_metrics_keeps_existing_stage_durations(self):
        self.client.create_session("goal-a")
        self.client.save_profile("goal-a", self._profile("goal-a", "学习 Python"))
        self.client.save_stage("goal-a", {
            "stage_id": "stage-completed",
            "stage_num": 1,
            "block_name": "语法基础",
            "estimated_hours": 8.0,
            "status": "completed",
        })
        self.client.save_stage("goal-a", {
            "stage_id": "stage-pending",
            "stage_num": 2,
            "block_name": "函数与模块",
            "estimated_hours": 9.0,
            "status": "pending",
        })

        target = self.client.refresh_target_metrics("goal-a")
        stages = {stage["stage_id"]: stage for stage in self.client.list_stage_progress("goal-a")}

        self.assertEqual(17.0, target["target_total_hours"])
        self.assertEqual(8.0, stages["stage-completed"]["estimated_hours"])
        self.assertEqual(9.0, stages["stage-pending"]["estimated_hours"])

    def test_zero_target_hours_keeps_unfinished_stage_duration(self):
        self.client.create_session("goal-a")
        self.client.save_profile("goal-a", self._profile("goal-a", "学习 Python"))
        self.client.save_stage("goal-a", {
            "stage_id": "stage-learning",
            "stage_num": 1,
            "block_name": "函数与模块",
            "estimated_hours": 1.5,
            "status": "learning",
        })

        target = self.client.update_target_settings("goal-a", {"target_total_hours": 0})
        stage = self.client.get_stage_duration("goal-a", "stage-learning")

        self.assertEqual(0.0, target["target_total_hours"])
        self.assertEqual(1.5, stage["estimated_hours"])

    def test_zero_duration_system_stage_survives_target_refresh(self):
        self.client.create_session("goal-a")
        profile = {
            **self._profile("goal-a", "Python 基础已掌握"),
            "knowledge_gaps": [],
        }
        self.client.save_profile("goal-a", profile)
        self.client.save_stage("goal-a", {
            "stage_id": "stage-empty",
            "stage_num": 1,
            "block_name": "已完成全部知识点",
            "estimated_hours": 0.0,
            "status": "retreat",
        })

        self.client.save_profile("goal-a", profile)
        target = self.client.get_target_settings("goal-a")
        stage = self.client.get_stage_duration("goal-a", "stage-empty")

        self.assertEqual(0.0, target["target_total_hours"])
        self.assertEqual(0.0, stage["estimated_hours"])

    def test_unstable_prerequisite_creates_remediation_evidence_and_task(self):
        self.client.create_session("goal-a")
        self.client.set_subtopic_status(
            "goal-a", "KBS-01-01", status=1, mastery_score=1.0,
            last_accuracy=0.5, increment_attempt=True,
        )
        self.client.save_stage("goal-a", {
            "stage_id": "stage-syntax", "stage_num": 1,
            "block_name": "Python 语法", "estimated_hours": 1.0,
            "status": "completed", "accuracy": 0.5,
        })
        stage = StageInfo(
            stage_id="stage-data", stage_num=2, block_id="KB-01",
            block_name="Python 数据结构", gap_skill="Python",
            subtopic_id="KBS-01-02", subtopic_ids=["KBS-01-02"],
        )

        evidence = LearningEvidenceService(self.client).build("goal-a", stage)
        from agents.deterministic import DeterministicContentAgent
        action, reason, tasks, refs = DeterministicContentAgent._build_teaching_layer(
            stage, "列表用于组织多个值。", None, None, evidence, block_index=0,
        )

        self.assertEqual("remediate", action)
        self.assertTrue(evidence.preflight_review)
        self.assertIn("KBS-01-01", refs[0])
        self.assertTrue(tasks)
        self.assertIn("前置知识需要补救", reason)
        import asyncio
        issues = asyncio.run(PersonalizationCheckTool().check({
            "personalization_action": action,
            "personalization_reason": reason,
            "evidence_refs": refs,
            "fact_source_chunk_ids": ["chunk-1"],
            "teaching_tasks": tasks,
        }))
        self.assertEqual([], issues)

    def test_delete_last_stage_removes_max_stage_when_runtime_is_previous_stage(self):
        self.client.create_session("goal-a")
        for stage_num in range(1, 5):
            self.client.save_stage("goal-a", {
                "stage_id": f"stage-{stage_num}",
                "stage_num": stage_num,
                "block_name": f"Python 阶段 {stage_num}",
                "estimated_hours": 1.0,
                "status": "completed" if stage_num < 4 else "pending",
            })

        # 第 4 阶段已规划但未开始时，运行态仍在第 3 阶段。
        with self.client.Session() as session:
            session.execute(text("""
                UPDATE session_states
                SET state_json = :state_json
                WHERE session_id = :session_id
            """), {
                "session_id": "goal-a",
                "state_json": '{"stage_info":{"stage_id":"stage-3","stage_num":3},"next_stage":{"stage_id":"stage-4","stage_num":4}}',
            })
            session.commit()

        details = self.client.delete_last_stage("goal-a")
        remaining_ids = [stage["stage_id"] for stage in self.client.list_stage_progress("goal-a")]

        self.assertEqual("stage-4", details["stage_id"])
        self.assertEqual(3, details["remaining_stage_num"])
        self.assertEqual(["stage-1", "stage-2", "stage-3"], remaining_ids)

    def test_session_state_does_not_persist_combined_profile(self):
        manager = StateManager(self.client)
        manager.create_session("goal-a")
        manager.update_profile("goal-a", self._profile("goal-a", "学习 Python"))
        with self.client.Session() as session:
            payload = session.execute(
                text("SELECT state_json FROM session_states WHERE session_id = 'goal-a'")
            ).scalar_one()
        self.assertIn('"profile": {', payload)
        self.assertNotIn('"practice_profile"', payload)
        reloaded_client = SQLiteClient(self.database_url)
        try:
            restored = StateManager(reloaded_client).get_session("goal-a")
            self.assertEqual("开发工程师", restored["profile"]["grade"])
            self.assertEqual("学习 Python", restored["profile"]["goal"])
        finally:
            reloaded_client.close()

    def test_practice_profile_is_rebuilt_from_user_hardware(self):
        manager = StateManager(self.client)
        manager.create_session("goal-a")
        self.client.upsert_user_profile(DEFAULT_USER_ID, {"hardware": "RTX 4090"})
        manager.refresh_practice_profile("goal-a")

        with self.client.Session() as session:
            payload = session.execute(
                text("SELECT state_json FROM session_states WHERE session_id = 'goal-a'")
            ).scalar_one()
        self.assertNotIn('"practice_profile"', payload)

        reloaded_client = SQLiteClient(self.database_url)
        try:
            restored = StateManager(reloaded_client).get_session("goal-a")
            self.assertEqual("RTX 4090", restored["practice_profile"]["hardware"])
        finally:
            reloaded_client.close()


if __name__ == "__main__":
    unittest.main()
