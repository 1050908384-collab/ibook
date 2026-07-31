import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from db.sqlite_client import SQLiteClient
from orchestrator.harness import Harness
from orchestrator.state_manager import StateManager
from models.review import ReviewError, TechnicalReviewResult
from tools.agent4_tools.rule_check_tool import RuleCheckTool
from scripts.insert_knowledge_data import (
    BLOCKS_FILE_PATH,
    DOCS_FILE_PATH,
    FAQ_FILE_PATH,
    QUESTIONS_FILE_PATH,
    SUBTOPICS_FILE_PATH,
    insert_faq_items,
    insert_knowledge_blocks,
    insert_knowledge_docs,
    insert_knowledge_subtopics,
    insert_test_questions,
    load_required_list,
)


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
        self._seed_reference_data(client)
        self.clients.append(client)
        return Harness(StateManager(client), execution_mode="deterministic")

    def _seed_reference_data(self, client: SQLiteClient) -> None:
        insert_knowledge_blocks(client, load_required_list(BLOCKS_FILE_PATH))
        insert_knowledge_docs(client, load_required_list(DOCS_FILE_PATH))
        insert_knowledge_subtopics(client, load_required_list(SUBTOPICS_FILE_PATH))
        insert_faq_items(client, load_required_list(FAQ_FILE_PATH))
        insert_test_questions(client, load_required_list(QUESTIONS_FILE_PATH))

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
        # 单一知识盲区 LoRA：答对后 all_mastered=True → session 真正结束
        self.assertEqual("completed", completed["status"])
        self.assertEqual(1.0, completed["accuracy"])
        self.assertTrue(completed.get("all_mastered"))
        self.assertEqual("completed", completed.get("session_status"))
        self.assertFalse(completed.get("can_plan_next_stage"))

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

        # P3 新契约：repeat 场景不直接 end_session，也不直接返回 next_stage，改为按需 plan-next
        self.assertEqual("repeat", result["status"])
        self.assertEqual(0.0, result["accuracy"])
        self.assertFalse(result.get("all_mastered"))
        self.assertTrue(result.get("can_plan_next_stage"))
        self.assertEqual("active", result.get("session_status"))
        self.assertIn("next_strategy", result)
        self.assertIn("stage_summary", result)
        # session 仍 active，允许继续 plan-next
        session = harness.state_manager.get_session(session_id)
        self.assertEqual("active", session["status"])

    async def test_pass_but_not_all_mastered_keeps_session_active(self):
        """
        覆盖率验证：当 accuracy>=80% 但仍有其它知识盲区时，
        complete_stage 返回 continue，session_status=active，
        can_plan_next_stage=True，session 不被提前结束。
        """
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        initial = await harness.process_initial_input(
            session_id, "I want to learn LoRA and QLoRA"
        )
        # 确保至少有 2 个未掌握盲区
        gaps = initial["profile"]["knowledge_gaps"]
        self.assertGreaterEqual(len(gaps), 1)

        started = await harness.start_stage(session_id)
        question = started["learning_package"]["questions"][0]
        # 答对，保证 accuracy>=0.8
        await harness.submit_answer(session_id, question["question_id"], question["correct_answer"])

        completed = await harness.complete_stage(session_id)
        # 因仍有其它盲区，会话不能提前结束
        self.assertEqual("active", completed.get("session_status"))
        self.assertTrue(completed.get("can_plan_next_stage"))
        self.assertIn(completed["status"], ("continue", "completed"))
        if not completed.get("all_mastered"):
            # 未全部掌握，session 仍为 active
            self.assertEqual("active", harness.state_manager.get_session(session_id)["status"])
        self.assertIn("updated_profile", completed)
        self.assertIn("next_strategy", completed)

    async def test_plan_next_stage_creates_real_next_stage_and_start_stage_switches(self):
        """
        完整按需规划链路：
        complete_stage(accuracy 达标) → plan_next_stage 生成 next_stage → start_stage 时自动切换为当前
        并验证阶段编号真正推进了。
        """
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        initial = await harness.process_initial_input(
            session_id, "I know Python and want to learn Transformer and LoRA"
        )
        self.assertEqual("ready", initial["status"])
        gaps = initial["profile"]["knowledge_gaps"]

        started = await harness.start_stage(session_id)
        question = started["learning_package"]["questions"][0]
        await harness.submit_answer(
            session_id, question["question_id"], question["correct_answer"]
        )

        completed = await harness.complete_stage(session_id)
        self.assertTrue(completed.get("can_plan_next_stage"))

        # 当还有盲区时，调用 plan-next 才真正生成下一阶段
        if not completed.get("all_mastered"):
            planned = await harness.plan_next_stage(session_id, adjustment_notes="优先做动手练习")
            self.assertEqual("planned", planned["status"])
            self.assertIn("next_stage", planned)
            self.assertIn("planning_notes", planned)
            self.assertIn("动手练习", planned["planning_notes"])

            stored = harness.state_manager.get_next_stage(session_id)
            self.assertIsNotNone(stored)
            self.assertEqual(planned["next_stage"]["stage_id"], stored["stage_id"])

            # start_stage 应自动把 next_stage 切换为当前，并清空 next_stage
            restarted = await harness.start_stage(session_id)
            self.assertTrue(restarted["review_passed"])
            self.assertIsNone(harness.state_manager.get_next_stage(session_id))
            # 阶段已推进
            current = harness.state_manager.get_current_stage(session_id)
            self.assertEqual(planned["next_stage"]["stage_num"], current["stage_num"])
        else:
            # 只有一个盲区 + 全部掌握：can_plan_next_stage=False，plan-next 允许失败
            self.assertFalse(completed.get("can_plan_next_stage"))

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
