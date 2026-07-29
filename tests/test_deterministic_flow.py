import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from db.sqlite_client import SQLiteClient
from orchestrator.harness import Harness
from orchestrator.state_manager import StateManager
from models.review import ReviewError, TechnicalReviewResult
from tools.agent4_tools.rule_check_tool import RuleCheckTool


class RejectOnceTechnicalReviewer:
    def __init__(self):
        self.calls = 0

    async def run(self, request):
        self.calls += 1
        if self.calls == 1:
            return TechnicalReviewResult(
                passed=False,
                errors=[ReviewError(
                    error_type="test_rejection",
                    description="缺少关键技术说明",
                    suggestion="补充关键技术说明",
                )],
            )
        return TechnicalReviewResult(passed=True)


class ExplodingReviewer:
    async def run(self, request):
        raise RuntimeError("review unavailable")


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

    async def test_learning_package_is_revised_and_reviewed_again(self):
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        await harness.process_initial_input(session_id, "want to learn LoRA")
        reviewer = RejectOnceTechnicalReviewer()
        harness.agent4 = reviewer

        result = await harness.start_stage(session_id)

        self.assertTrue(result["review_passed"])
        self.assertEqual(2, result["iterations"])
        self.assertEqual(2, result["learning_package"]["version"])
        self.assertFalse(result["review_history"][0]["passed"])
        self.assertTrue(result["review_history"][1]["passed"])
        self.assertEqual(2, reviewer.calls)

    async def test_answer_is_revised_and_reviewed_again(self):
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        await harness.process_initial_input(session_id, "want to learn LoRA")
        await harness.start_stage(session_id)
        reviewer = RejectOnceTechnicalReviewer()
        harness.agent4 = reviewer

        result = await harness.handle_question(session_id, "What is LoRA?")

        self.assertTrue(result["tech_pass"])
        self.assertTrue(result["quality_pass"])
        self.assertEqual(2, result["iterations"])
        self.assertIn("补充修订", result["answer"])
        self.assertFalse(result["review_history"][0]["passed"])
        self.assertTrue(result["review_history"][1]["passed"])
        self.assertEqual(2, reviewer.calls)

    async def test_reviewer_failure_becomes_failed_review(self):
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        await harness.process_initial_input(session_id, "want to learn LoRA")
        # Agent5已合并至Agent4，质量审核通过合并后的agent4处理
        harness.agent4 = ExplodingReviewer()

        result = await harness.handle_question(session_id, "What is LoRA?")

        self.assertFalse(result["tech_pass"])  # 合并后质量错误也影响tech_pass
        self.assertEqual(3, result["iterations"])
        # 合并后错误存储在technical.errors中
        error = result["review_history"][-1]["technical"]["errors"][0]
        self.assertEqual("review_system_error", error["error_type"])

    async def test_lora_rule_tool_rejects_mechanism_and_parameter_errors(self):
        content = (
            "LoRA将原始权重W分解为U*Sigma*V，并只训练U和V。"
            "原模型有10万个参数，微调参数量仅为50万，显著减少了参数。"
        )

        errors = await RuleCheckTool().check_rules(content)

        error_types = {error["error_type"] for error in errors}
        self.assertIn("lora_mechanism_error", error_types)
        self.assertIn("parameter_count_error", error_types)


if __name__ == "__main__":
    unittest.main()
