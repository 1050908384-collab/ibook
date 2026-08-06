import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from db.sqlite_client import DEFAULT_USER_ID, SQLiteClient
from orchestrator.state_manager import StateManager


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
        self.assertEqual(0, legacy_count)

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

    def test_session_state_does_not_persist_combined_profile(self):
        manager = StateManager(self.client)
        manager.create_session("goal-a")
        manager.update_profile("goal-a", self._profile("goal-a", "学习 Python"))
        with self.client.Session() as session:
            payload = session.execute(
                text("SELECT state_json FROM session_states WHERE session_id = 'goal-a'")
            ).scalar_one()
        self.assertIn('"profile": null', payload)
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
