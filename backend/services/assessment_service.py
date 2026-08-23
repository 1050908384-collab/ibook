# -*- coding: utf-8 -*-
"""AssessmentService — 测试服务（专业知识与教学质量审核 Agent6_API接口任务说明.md 第 2 节）。

由 专业知识与教学质量审核 Agent/6 队友承担工程实现，负责创建测试、保存答案、判题和计算正确率，
不属于 专业知识与教学质量审核 Agent 的运行职责（专业知识与教学质量审核 Agent 只审核学习包与答疑回答）。

严格落实第 2 节的测试模块约束：
1. 每次测试有独立 attempt_id。
2. 测试开始时冻结当前学习包的题目（快照存入 assessment_attempt_questions）。
3. 同一题可以在提交前修改，但数据库只保存一条最终答案（PRIMARY KEY 覆盖）。
4. 未完成全部必答题时不能提交。
5. 提交后不能再修改答案。
6. 只有整份测试提交后，才能返回正确答案和解析。
7. 正确率只能从已提交的完整 attempt 计算，不能从零散答题记录计算。
"""
from typing import Any, Dict, List, Optional
import logging

from db.sqlite_client import SQLiteClient
from orchestrator.harness import Harness

logger_assessment = logging.getLogger(__name__)
if not logger_assessment.handlers:
    logger_assessment.addHandler(logging.NullHandler())


class AssessmentError(ValueError):
    """测试服务业务错误（映射为 4xx）。"""

    def __init__(self, message: str, code: str = "ASSESSMENT_ERROR"):
        super().__init__(message)
        self.code = code


NO_QUESTIONS_AVAILABLE_CODE = "NO_QUESTIONS_AVAILABLE"


class AssessmentService:
    def __init__(self, db_client: SQLiteClient, state_manager=None, harness=None):
        self.db = db_client
        # state_manager 用于读取"当前学习包"的题目以冻结；缺省时退化为按当前阶段读题
        self.state_manager = state_manager
        # harness 引用用于异步 LLM 语义判分兜底（可空；空时退化为纯规则）
        self._harness = harness

    def attach_harness(self, harness) -> None:
        """后续绑定 harness 实例，使 LLM 语义兜底判分生效。"""
        self._harness = harness

    # ------------------------------------------------------------------
    # 判题：复用 Harness 的多级答案匹配策略，保持与 submit_answer 一致口径
    # ------------------------------------------------------------------
    @staticmethod
    def _grade(user_answer: str, correct_answer: str, question_type: str = "") -> bool:
        """同步规则判分（快速路径）。"""
        if user_answer is None:
            return False
        return Harness._match_answer(user_answer, correct_answer, question_type=question_type)

    async def _grade_async(
        self,
        user_answer: str,
        correct_answer: str,
        question_type: str = "",
        question_text: str = "",
    ) -> bool:
        """
        异步判分（推荐路径）：
        1. 先走同步规则（跨平台归一化 + 注释剥离 + 多候选 + 多级匹配）
        2. 规则失败时，若题型合适（fill_blank/short_answer/practical）且 harness 可用，
           调用 LLM 语义等价兜底
        """
        if user_answer is None:
            return False
        # 1) 规则快速路径
        rule_ok = Harness._match_answer(user_answer, correct_answer, question_type=question_type)
        if rule_ok:
            return True
        # 2) LLM 语义兜底（仅当 harness 可用时）
        if self._harness is not None and question_text:
            try:
                llm_ok = await self._harness._match_answer_async(
                    user_answer=user_answer,
                    correct_answer=correct_answer,
                    question_type=question_type,
                    question_text=question_text,
                )
                if llm_ok:
                    return True
            except Exception as exc:
                logger_assessment.warning(
                    "AssessmentService async grading LLM fallback skipped: %s", exc
                )
        return False

    @staticmethod
    def _is_answered(user_answer: Optional[str]) -> bool:
        return user_answer is not None and str(user_answer).strip() != ""

    # ------------------------------------------------------------------
    # 1) 创建测试：冻结当前学习包的题目
    # ------------------------------------------------------------------
    def create_attempt(self, session_id: str) -> Dict[str, Any]:
        frozen_questions, stage_meta = self._collect_current_package_questions(session_id)
        if not frozen_questions:
            raise AssessmentError(
                "当前阶段没有可测试的题目，无法创建测试",
                code=NO_QUESTIONS_AVAILABLE_CODE,
            )

        # 幂等复用：同一 session+stage 未正式交卷前只保留一份 in_progress attempt。
        # 前端双击“开始测试”或刷新重试时返回原 attempt（resumed=True），不重复建卷。
        existing = self.db.get_in_progress_attempt(
            session_id, stage_id=stage_meta.get("stage_id")
        )
        if existing:
            result = self.get_attempt(existing["attempt_id"])
            result["resumed"] = True
            return result

        attempt_id = self.db.create_assessment_attempt(
            session_id=session_id,
            frozen_questions=frozen_questions,
            stage_id=stage_meta.get("stage_id"),
            stage_num=stage_meta.get("stage_num"),
            block_id=stage_meta.get("block_id"),
            subtopic_id=stage_meta.get("subtopic_id"),
        )
        result = self.get_attempt(attempt_id)
        result["resumed"] = False
        return result

    def _collect_current_package_questions(self, session_id: str):
        """从当前学习包收集题目快照及阶段元信息。"""
        stage_meta: Dict[str, Any] = {}
        questions: List[Dict[str, Any]] = []

        if self.state_manager is not None:
            package = self.state_manager.get_learning_package(session_id)
            stage = self.state_manager.get_current_stage(session_id)
            if stage:
                stage_meta = {
                    "stage_id": stage.get("stage_id"),
                    "stage_num": stage.get("stage_num"),
                    "block_id": stage.get("block_id"),
                    "subtopic_id": stage.get("subtopic_id"),
                }

            # 收集 subtopic 数组用于旧快照题量判定
            subtopic_ids: List[str] = []
            if stage:
                subtopic_ids = list(stage.get("subtopic_ids") or [])
                if not subtopic_ids and stage.get("subtopic_id"):
                    subtopic_ids = [stage["subtopic_id"]]
            block_id = stage_meta.get("block_id")

            # ---- 旧快照题量不足时动态重抽（与 /current/package 同标准）----
            stale = False
            if package and isinstance(package.get("questions"), list):
                pkg_q = package["questions"]
                if subtopic_ids and len(pkg_q) <= max(2, len(subtopic_ids) - 1):
                    stale = True
                if not stale and block_id and pkg_q:
                    first_block = pkg_q[0].get("block_id") if isinstance(pkg_q[0], dict) else None
                    if first_block and first_block != block_id:
                        stale = True
            if stale and block_id:
                try:
                    refreshed = self.db.list_questions_for_stage(
                        block_id, subtopic_ids=subtopic_ids, limit=max(8, len(subtopic_ids) * 2)
                    )
                    if refreshed:
                        package["questions"] = refreshed
                        self.state_manager.update_learning_package(session_id, package)
                        logger_assessment.info(
                            "AssessmentService refreshed stale questions: session=%s old=%d new=%d",
                            session_id, len(pkg_q), len(refreshed),
                        )
                except Exception as exc:
                    logger_assessment.warning("AssessmentService refresh questions failed: %s", exc)

            if package and package.get("questions"):
                for q in package["questions"]:
                    question_id = q.get("question_id")
                    db_question = self.db.get_question(question_id) if question_id else None
                    # The assessment is a frozen snapshot of test_questions,
                    # never an 个性化学习资源生成 Agent-generated question that happens to be in
                    # the learning package.
                    if not db_question:
                        continue
                    if stage_meta.get("block_id") and db_question.get("block_id") != stage_meta["block_id"]:
                        continue
                    questions.append({
                        "question_id": db_question["question_id"],
                        "block_id": db_question["block_id"],
                        "subtopic_id": db_question.get("subtopic_id") or stage_meta.get("subtopic_id"),
                        "difficulty": db_question.get("difficulty"),
                        "question_type": db_question.get("question_type"),
                        "question_text": db_question["question_text"],
                        "options": db_question.get("options"),
                        "correct_answer": db_question["correct_answer"],
                        "explanation": db_question.get("explanation"),
                        "hint": db_question.get("hint"),
                        "is_required": True,
                    })
            if not questions and stage_meta.get("block_id"):
                questions = [
                    {**question, "is_required": True}
                    for question in self.db.list_questions_for_stage(
                        stage_meta["block_id"], subtopic_ids, limit=max(8, len(subtopic_ids) * 2)
                    )
                ]
        return questions, stage_meta

    # ------------------------------------------------------------------
    # 2) 刷新后恢复进度
    # ------------------------------------------------------------------
    def get_attempt(self, attempt_id: str) -> Dict[str, Any]:
        attempt = self._require_attempt(attempt_id)
        questions = self.db.get_assessment_attempt_questions(attempt_id)
        answers = self.db.get_assessment_answers(attempt_id)
        submitted = attempt["status"] == "submitted"

        question_views: List[Dict[str, Any]] = []
        for q in questions:
            saved = answers.get(q["question_id"], {})
            view: Dict[str, Any] = {
                "question_id": q["question_id"],
                "sequence_index": q["sequence_index"],
                "block_id": q["block_id"],
                "subtopic_id": q["subtopic_id"],
                "difficulty": q["difficulty"],
                "question_type": q["question_type"],
                "question_text": q["question_text"],
                "options": q["options"],
                "hint": q["hint"],
                "is_required": q["is_required"],
                "user_answer": saved.get("user_answer"),
            }
            # 约束 6：只有整份提交后才公布正确答案与解析
            if submitted:
                view["correct_answer"] = q["correct_answer"]
                view["explanation"] = q["explanation"]
                view["is_correct"] = saved.get("is_correct")
            question_views.append(view)

        result: Dict[str, Any] = {
            "attempt_id": attempt["attempt_id"],
            "session_id": attempt["session_id"],
            "stage_id": attempt["stage_id"],
            "status": attempt["status"],
            "total_questions": attempt["total_questions"],
            "answered_count": attempt["answered_count"],
            "questions": question_views,
            "created_at": attempt["created_at"],
        }
        if submitted:
            result["correct_count"] = attempt["correct_count"]
            result["accuracy"] = attempt["accuracy"]
            result["submitted_at"] = attempt["submitted_at"]
        return result

    # ------------------------------------------------------------------
    # 3) 保存/覆盖单题答案
    # ------------------------------------------------------------------
    def save_answer(self, attempt_id: str, question_id: str, user_answer: str, time_spent: int = 0) -> Dict[str, Any]:
        attempt = self._require_attempt(attempt_id)
        # 约束 5：提交后不能再修改
        if attempt["status"] == "submitted":
            raise AssessmentError("测试已提交，不能再修改答案")
        # 只能作答冻结快照内的题目
        question = self._find_frozen_question(attempt_id, question_id)
        if question is None:
            raise AssessmentError(f"题目不在本次测试范围内: {question_id}")

        # 约束 3：只保存一条最终答案（暂不写 is_correct，避免中途泄漏判题结果）
        self.db.upsert_assessment_answer(
            attempt_id=attempt_id,
            question_id=question_id,
            user_answer=user_answer,
            is_correct=None,
            time_spent=time_spent,
        )
        refreshed = self._require_attempt(attempt_id)
        return {
            "attempt_id": attempt_id,
            "question_id": question_id,
            "saved": True,
            "answered_count": refreshed["answered_count"],
            "total_questions": refreshed["total_questions"],
            "status": refreshed["status"],
        }

    # ------------------------------------------------------------------
    # 4) 提交：完整性校验 + 统一判题 + 公布结果
    # ------------------------------------------------------------------
    def submit_attempt(self, attempt_id: str) -> Dict[str, Any]:
        """同步版本：纯规则判分（兼容旧调用方，不使用 LLM 兜底）。"""
        attempt = self._require_attempt(attempt_id)
        if attempt["status"] == "submitted":
            raise AssessmentError("测试已提交，不能重复提交")

        questions = self.db.get_assessment_attempt_questions(attempt_id)
        answers = self.db.get_assessment_answers(attempt_id)

        missing = [
            q["question_id"]
            for q in questions
            if q["is_required"] and not self._is_answered(answers.get(q["question_id"], {}).get("user_answer"))
        ]
        if missing:
            raise AssessmentError(f"还有必答题未完成，不能提交：{missing}")

        graded_answers: List[Dict[str, Any]] = []
        correct_count = 0
        for q in questions:
            saved = answers.get(q["question_id"], {})
            user_answer = saved.get("user_answer")
            is_correct = self._grade(user_answer, q["correct_answer"], q["question_type"] or "")
            if is_correct:
                correct_count += 1
            graded_answers.append({
                "question_id": q["question_id"],
                "user_answer": user_answer,
                "is_correct": is_correct,
                "time_spent": saved.get("time_spent", 0),
            })

        total = len(questions)
        accuracy = round(correct_count / total, 4) if total else 0.0
        self.db.finalize_assessment_attempt(
            attempt_id=attempt_id,
            graded_answers=graded_answers,
            correct_count=correct_count,
            accuracy=accuracy,
        )
        return self.get_attempt(attempt_id)

    async def submit_attempt_async(self, attempt_id: str) -> Dict[str, Any]:
        """
        异步版本（推荐）：规则判分 + fill_blank/实操题失败时 LLM 语义兜底。
        与 submit_attempt 流程一致，仅在判题环节使用增强版 _grade_async。
        """
        attempt = self._require_attempt(attempt_id)
        if attempt["status"] == "submitted":
            raise AssessmentError("测试已提交，不能重复提交")

        questions = self.db.get_assessment_attempt_questions(attempt_id)
        answers = self.db.get_assessment_answers(attempt_id)

        missing = [
            q["question_id"]
            for q in questions
            if q["is_required"] and not self._is_answered(answers.get(q["question_id"], {}).get("user_answer"))
        ]
        if missing:
            raise AssessmentError(f"还有必答题未完成，不能提交：{missing}")

        # 统一异步判题：对每条自由文本题，规则失败时尝试 LLM 语义等价
        graded_answers: List[Dict[str, Any]] = []
        correct_count = 0
        for q in questions:
            saved = answers.get(q["question_id"], {})
            user_answer = saved.get("user_answer")
            is_correct = await self._grade_async(
                user_answer=user_answer,
                correct_answer=q["correct_answer"],
                question_type=q.get("question_type") or "",
                question_text=q.get("question_text") or "",
            )
            if is_correct:
                correct_count += 1
            graded_answers.append({
                "question_id": q["question_id"],
                "user_answer": user_answer,
                "is_correct": is_correct,
                "time_spent": saved.get("time_spent", 0),
            })

        total = len(questions)
        accuracy = round(correct_count / total, 4) if total else 0.0
        self.db.finalize_assessment_attempt(
            attempt_id=attempt_id,
            graded_answers=graded_answers,
            correct_count=correct_count,
            accuracy=accuracy,
        )
        return self.get_attempt(attempt_id)

    # ------------------------------------------------------------------
    # 正确率：只从已提交的完整 attempt 计算（约束 7）
    # ------------------------------------------------------------------
    def get_accuracy_from_submitted(self, session_id: str, stage_id: Optional[str] = None) -> Optional[float]:
        row = self.db.get_submitted_attempt_accuracy(session_id, stage_id=stage_id)
        return row["accuracy"] if row else None

    def has_submitted_attempt(self, session_id: str, stage_id: Optional[str] = None) -> bool:
        """当前阶段是否存在已提交（status=submitted）的完整 attempt。

        供 v1 阶段完成接口做前置硬校验：严格按 stage_id 过滤，不退回会话级，
        避免用户未提交当前阶段测试就绕过完成阶段。stage_id 为空或无已提交时返回 False。
        """
        if not stage_id:
            return False
        return self.db.get_submitted_attempt_accuracy(session_id, stage_id=stage_id) is not None

    def get_submitted_attempt(self, session_id: str, stage_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return the submitted attempt, including answer review data, for a stage."""
        row = self.db.get_submitted_attempt_accuracy(session_id, stage_id=stage_id)
        return self.get_attempt(row["attempt_id"]) if row else None

    def get_submitted_attempt_for_stage_num(self, session_id: str, stage_num: int) -> Optional[Dict[str, Any]]:
        """Compatibility lookup for historical stages whose snapshot lost its stage_id."""
        attempts = self.db.list_assessment_attempts(session_id)
        submitted = [
            item for item in attempts
            if item.get("stage_num") == stage_num and item.get("status") == "submitted"
        ]
        if not submitted:
            return None
        latest = max(submitted, key=lambda item: item.get("submitted_at") or item.get("created_at") or "")
        return self.get_attempt(latest["attempt_id"])

    @staticmethod
    def stage_allows_persisted_result(stage: Optional[Dict[str, Any]]) -> bool:
        """Only ended stages may restore a submitted assessment as their result.

        A low-score retry intentionally reuses the same ``stage_id``. Once the
        regenerated stage is pending/learning, an older submitted attempt with
        that id belongs to the previous learning cycle and must not reopen the
        previous summary after refresh.
        """
        status = str((stage or {}).get("status") or "").strip().lower()
        return not status or status in {"completed", "retreat"}

    def get_submitted_attempt_for_result(
        self,
        session_id: str,
        stage: Optional[Dict[str, Any]],
        stage_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Return the assessment review only when the requested stage has ended."""
        if not self.stage_allows_persisted_result(stage):
            return None
        stage_id = (stage or {}).get("stage_id")
        attempt = self.get_submitted_attempt(session_id, stage_id)
        if attempt:
            return attempt
        return self.get_submitted_attempt_for_stage_num(session_id, stage_num)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _require_attempt(self, attempt_id: str) -> Dict[str, Any]:
        attempt = self.db.get_assessment_attempt(attempt_id)
        if not attempt:
            raise AssessmentError(f"测试不存在: {attempt_id}")
        return attempt

    def _find_frozen_question(self, attempt_id: str, question_id: str) -> Optional[Dict[str, Any]]:
        for q in self.db.get_assessment_attempt_questions(attempt_id):
            if q["question_id"] == question_id:
                return q
        return None
