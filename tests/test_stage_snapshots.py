import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from db.sqlite_client import SQLiteClient
from orchestrator.state_manager import StateManager


def stage(number: int) -> dict:
    return {
        "stage_id": f"stage-{number}",
        "stage_num": number,
        "block_id": f"KB-{number:02d}",
        "block_name": f"Block {number}",
        "subtopic_id": f"KBS-{number:02d}-00",
        "subtopic_name": f"Topic {number}",
        "subtopic_ids": [f"KBS-{number:02d}-00"],
        "subtopic_names": [f"Topic {number}"],
        "gap_skill": f"Skill {number}",
    }


def package(session_id: str, number: int) -> dict:
    return {
        "session_id": session_id,
        "stage_id": f"stage-{number}",
        "block_id": f"KB-{number:02d}",
        "block_name": f"Block {number}",
        "tutorial": f"package {number}",
    }


class StageSnapshotTest(unittest.TestCase):
    def test_stage_package_and_qa_are_kept_per_stage_after_progressing(self):
        with tempfile.TemporaryDirectory() as directory:
            database_url = f"sqlite:///{(Path(directory) / 'snapshots.db').as_posix()}"
            manager = StateManager(SQLiteClient(database_url))
            session_id = manager.create_session()["session_id"]

            manager.update_stage(session_id, stage(1))
            manager.update_learning_package(session_id, package(session_id, 1))
            manager.add_qa_record(session_id, "question one", "answer one")

            manager.update_stage(session_id, stage(2))
            manager.update_learning_package(session_id, package(session_id, 2))
            manager.add_qa_record(session_id, "question two", "answer two")

            restored = StateManager(SQLiteClient(database_url))
            first = restored.get_stage_snapshot(session_id, 1)
            second = restored.get_stage_snapshot(session_id, 2)

            self.assertEqual("package 1", first["learning_package"]["tutorial"])
            self.assertEqual(["question one"], [item["question"] for item in first["qa_history"]])
            self.assertEqual("package 2", second["learning_package"]["tutorial"])
            self.assertEqual(["question two"], [item["question"] for item in second["qa_history"]])

            # Legacy state_json can retain untagged records from stage 1 after
            # the session advances. The current stage must still use the
            # stage-scoped database history, rather than that stale cache.
            restored.update_field(session_id, "qa_history", [{
                "question": "legacy stage one question",
                "answer": "legacy stage one answer",
            }])
            current_after_legacy_restore = restored.get_stage_snapshot(session_id, 2)
            self.assertEqual(
                ["question two"],
                [item["question"] for item in current_after_legacy_restore["qa_history"]],
            )

            # Asking a question in a completed stage must not append it to
            # the active stage's runtime dialogue.
            restored.add_qa_record(session_id, "follow-up one", "answer follow-up", stage_num=1)
            first_after_follow_up = restored.get_stage_snapshot(session_id, 1)
            second_after_follow_up = restored.get_stage_snapshot(session_id, 2)
            self.assertEqual(
                ["question one", "follow-up one"],
                [item["question"] for item in first_after_follow_up["qa_history"]],
            )
            self.assertEqual(
                ["question two"],
                [item["question"] for item in second_after_follow_up["qa_history"]],
            )

            # Existing sessions from before this change have no stage snapshot,
            # but stage_progress and user_qa still make them navigable.
            restored.update_field(session_id, "stage_snapshots", {})
            legacy_first = restored.get_stage_snapshot(session_id, 1)
            self.assertEqual("stage-1", legacy_first["stage_info"]["stage_id"])
            self.assertIsNone(legacy_first["learning_package"])
            self.assertEqual(
                ["question one", "follow-up one"],
                [item["question"] for item in legacy_first["qa_history"]],
            )
            restored.db_client.close()
            manager.db_client.close()


if __name__ == "__main__":
    unittest.main()
