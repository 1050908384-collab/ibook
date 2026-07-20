import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from db.sqlite_client import SQLiteClient
from orchestrator.harness import Harness
from orchestrator.state_manager import StateManager


class DeterministicFlowTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{(Path(self.temp_dir.name) / 'flow.db').as_posix()}"
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.temp_dir.cleanup()

    def create_harness(self) -> Harness:
        client = SQLiteClient(self.database_url)
        self.clients.append(client)
        return Harness(StateManager(client), execution_mode="deterministic")

    async def test_complete_flow_survives_new_harness(self):
        first = self.create_harness()
        session_id = first.state_manager.create_session()["session_id"]

        initial = await first.process_initial_input(
            session_id,
            "I know Python and want to learn LoRA",
        )
        self.assertEqual("ready", initial["status"])
        self.assertEqual("LoRA", initial["stage_info"]["block_name"])

        started = await first.start_stage(session_id)
        self.assertTrue(started["review_passed"])
        question = started["learning_package"]["questions"][0]
        submitted = await first.submit_answer(
            session_id,
            question["question_id"],
            "LoRA",
        )
        self.assertTrue(submitted["is_correct"])

        restored = self.create_harness()
        restored_session = restored.state_manager.get_session(session_id)
        self.assertEqual("LoRA", restored_session["learning_package"]["block_name"])

        completed = await restored.complete_stage(session_id)
        self.assertEqual("completed", completed["status"])
        self.assertEqual(1.0, completed["accuracy"])

        final = self.create_harness().state_manager.get_session(session_id)
        self.assertEqual("completed", final["status"])
        self.assertEqual("已掌握", final["profile"]["knowledge_gaps"][0]["status"])

    async def test_incorrect_answer_repeats_stage(self):
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        await harness.process_initial_input(session_id, "want to learn Transformer")
        started = await harness.start_stage(session_id)
        question_id = started["learning_package"]["questions"][0]["question_id"]

        submitted = await harness.submit_answer(session_id, question_id, "wrong")
        self.assertFalse(submitted["is_correct"])
        result = await harness.complete_stage(session_id)

        self.assertEqual("repeat", result["status"])
        self.assertEqual(0.0, result["accuracy"])
        self.assertEqual(1, result["next_stage"]["stage_num"])


if __name__ == "__main__":
    unittest.main()
