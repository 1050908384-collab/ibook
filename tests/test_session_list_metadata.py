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


if __name__ == "__main__":
    unittest.main()
