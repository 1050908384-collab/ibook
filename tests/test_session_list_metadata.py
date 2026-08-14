import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db.sqlite_client import SQLiteClient


class SessionListMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database = Path(self.temp_dir.name) / "sessions.db"
        self.client = SQLiteClient(f"sqlite:///{database.as_posix()}")

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_list_sessions_includes_database_creation_time(self):
        self.client.create_session("goal-a")

        sessions = self.client.list_sessions()

        self.assertEqual(1, len(sessions))
        self.assertEqual("goal-a", sessions[0]["session_id"])
        self.assertIsNotNone(sessions[0]["created_at"])

    def test_list_sessions_only_returns_the_requested_users_sessions(self):
        self.client.create_session("goal-a", "user-a-123")
        self.client.create_session("goal-b", "user-b-456")

        user_a_sessions = self.client.list_sessions("user-a-123")
        user_b_sessions = self.client.list_sessions("user-b-456")

        self.assertEqual(["goal-a"], [item["session_id"] for item in user_a_sessions])
        self.assertEqual(["goal-b"], [item["session_id"] for item in user_b_sessions])


if __name__ == "__main__":
    unittest.main()
