import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("EXECUTION_MODE", "deterministic")

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
    """第 1 轮拒绝（带完整辩论协议字段），第 2 轮通过。用于验证审核→修改→重审循环。"""

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
                    canonical_error_id="test_reject_001",
                    must_fix=True,
                    must_fix_reason="测试用：关键技术缺失",
                    change_request={"required": True, "action": "补充关键技术说明"},
                )],
            )
        return TechnicalReviewResult(passed=True)

    async def rebut(self, session_id, issues, author_replies):
        # 第 1 轮坚持 issue（persist），迫使 Agent3 修改后进第 2 轮审核
        return {issue.get("canonical_error_id", ""): "persist" for issue in issues}


class ExplodingReviewer:
    """审核总是抛异常，用于验证异常兜底。"""

    async def run(self, request):
        raise RuntimeError("review unavailable")

    async def rebut(self, session_id, issues, author_replies):
        # 异常路径不会真正走到 rebut，提供空实现避免 AttributeError
        return {}


class AlwaysPersistReviewer:
    """每轮都报 must_fix 错误且 rebut 永远 persist，用于验证超轮次 needs_human_review 兜底。"""

    def __init__(self):
        self.calls = 0

    async def run(self, request):
        self.calls += 1
        return TechnicalReviewResult(
            passed=False,
            errors=[ReviewError(
                error_type="stubborn_error",
                description="永远不解决的问题",
                suggestion="无法解决",
                canonical_error_id="stubborn_001",
                must_fix=True,
                must_fix_reason="测试用：死锁兜底",
                change_request={"required": True, "action": "无法解决"},
            )],
        )

    async def rebut(self, session_id, issues, author_replies):
        return {issue.get("canonical_error_id", ""): "persist" for issue in issues}


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
            "I know Python, PyTorch, and HuggingFace and want to learn LoRA",
        )
        self.assertEqual("ready", initial["status"])
        # LoRA 目标现在显式展开其环境先修，第一阶段必须先完成 Linux 安装。
        self.assertEqual("Linux环境与命令行操作", initial["stage_info"]["block_name"])
        self.assertEqual("KBS-01-00", initial["stage_info"]["subtopic_id"])

        started = await first.start_stage(session_id)
        self.assertTrue(started["review_passed"])
        question = started["learning_package"]["questions"][0]
        submitted = await first.submit_answer(
            session_id,
            question["question_id"],
            question["correct_answer"],
        )
        self.assertTrue(submitted["is_correct"])

        restored = self.create_harness()
        restored_session = restored.state_manager.get_session(session_id)
        self.assertEqual("Linux环境与命令行操作", restored_session["learning_package"]["block_name"])

        completed = await restored.complete_stage(session_id)
        # 环境节点完成后仍有后续先修，必须保持会话并允许继续规划。
        self.assertEqual("continue", completed["status"])
        self.assertEqual(1.0, completed["accuracy"])
        self.assertFalse(completed.get("all_mastered"))
        self.assertEqual("active", completed.get("session_status"))
        self.assertTrue(completed.get("can_plan_next_stage"))

        final = self.create_harness().state_manager.get_session(session_id)
        self.assertEqual("active", final["status"])

    async def test_subtopic_status_persists_after_database_client_reloads(self):
        client = SQLiteClient(self.database_url)
        self._seed_reference_data(client)
        self.clients.append(client)
        session_id = "subtopic-status-persistence"
        client.create_session(session_id)

        client.set_subtopic_status(
            session_id=session_id,
            subtopic_id="KBS-01-00",
            status=1,
            mastery_score=1.0,
            last_accuracy=1.0,
            increment_attempt=True,
        )

        reloaded = SQLiteClient(self.database_url)
        self.clients.append(reloaded)
        persisted = reloaded.get_subtopic_status(session_id, "KBS-01-00")
        self.assertIsNotNone(persisted)
        self.assertEqual(1, persisted["status"])
        self.assertEqual(1.0, persisted["mastery_score"])
        self.assertEqual(1.0, persisted["last_accuracy"])
        self.assertEqual(1, persisted["attempt_count"])

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
        # 辩论协议返回字段（P3）
        self.assertIn("review_trace_id", result)
        self.assertIn("package_status", result)
        self.assertEqual("approved", result["package_status"])
        self.assertFalse(result["needs_human_review"])

    async def test_debate_protocol_with_real_agents(self):
        """用真实 deterministic agent 验证完整辩论协议链路。

        验证点：
        - Agent3 首轮包带 sources + claims
        - Agent4 审核结果带 canonical_error_id + must_fix
        - harness 返回 review_trace_id / package_status / needs_human_review
        - DB 6 张表有记录（review_traces / learning_package_claims 至少有数据）
        """
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        await harness.process_initial_input(session_id, "want to learn LoRA")

        result = await harness.start_stage(session_id)

        # 1. Agent3 产出带 claims（sources 依赖知识库数据，可能为空）
        pkg = result["learning_package"]
        self.assertGreater(len(pkg.get("claims", [])), 0, "Agent3 首轮包必须带 claims")
        claim = pkg["claims"][0]
        self.assertIn("claim_id", claim)
        self.assertIn("claim_type", claim)
        self.assertIn("section_ref", claim)
        self.assertIn("status", claim)

        # 2. 辩论协议返回字段
        self.assertIn("review_trace_id", result)
        self.assertIsNotNone(result["review_trace_id"])
        self.assertIn("package_status", result)
        self.assertIn("needs_human_review", result)
        self.assertIn("unresolved_issue_count", result)

        # 3. DB 落库验证：review_traces 有记录
        trace_id = result["review_trace_id"]
        db_path = harness.state_manager.db_client.engine.url.database
        conn = sqlite3.connect(db_path)
        try:
            trace_count = conn.execute(
                "SELECT COUNT(*) FROM review_traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()[0]
            self.assertGreater(trace_count, 0, "review_traces 必须有记录")

            claims_count = conn.execute(
                "SELECT COUNT(*) FROM learning_package_claims WHERE trace_id = ?", (trace_id,)
            ).fetchone()[0]
            self.assertGreater(claims_count, 0, "learning_package_claims 必须有记录")
        finally:
            conn.close()

    async def test_needs_human_review_deadlock(self):
        """Agent4 永远 persist + must_fix，验证超轮次后 needs_human_review=True。

        这是死循环停止机制的核心验证：达到 MAX_REVIEW_ITERATIONS 后不阻塞，
        返回 needs_human_review=True + package_status=provisional。
        """
        harness = self.create_harness()
        session_id = harness.state_manager.create_session()["session_id"]
        await harness.process_initial_input(session_id, "want to learn LoRA")
        harness.agent4 = AlwaysPersistReviewer()

        result = await harness.start_stage(session_id)

        # 达到最大轮次，未决 must_fix 仍存在
        from config import settings
        self.assertEqual(settings.MAX_REVIEW_ITERATIONS, result["iterations"])
        # 死锁兜底：needs_human_review=True，package 降级为 provisional
        self.assertTrue(result["needs_human_review"])
        self.assertEqual("provisional", result["package_status"])
        # 学习包仍可用（A 方案：降级可用 + 待审标记），不阻塞学习流
        self.assertTrue(result["review_passed"])
        self.assertGreater(result["unresolved_must_fix_count"], 0)
        # DB trace 最终状态
        trace_id = result["review_trace_id"]
        db_path = harness.state_manager.db_client.engine.url.database
        conn = sqlite3.connect(db_path)
        try:
            trace = conn.execute(
                "SELECT final_status, package_status FROM review_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            self.assertEqual("needs_human_review", trace[0])
            self.assertEqual("provisional", trace[1])
        finally:
            conn.close()

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
