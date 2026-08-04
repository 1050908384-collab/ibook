# -*- coding: utf-8 -*-
"""AssessmentService — 测试服务（Agent46_API接口任务说明.md 第 2 节）。

由 Agent4/6 队友承担工程实现，负责创建测试、保存答案、判题和计算正确率，
不属于 Agent4 的运行职责（Agent4 只审核学习包与答疑回答）。

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

from db.sqlite_client import SQLiteClient
from orchestrator.harness import Harness


class AssessmentError(ValueError):
    """测试服务业务错误（映射为 4xx）。"""


class AssessmentService:
    def __init__(self, db_client: SQLiteClient, state_manager=None):
        self.db = db_client
        # state_manager 用于读取“当前学习包”的题目以冻结；缺省时退化为按当前阶段读题
        self.state_manager = state_manager

    # ------------------------------------------------------------------
    # 判题：复用 Harness 的多级答案匹配策略，保持与 submit_answer 一致口径
    # ------------------------------------------------------------------
    @staticmethod
    def _grade(user_answer: str, correct_answer: str, question_type: str = "") -> bool:
        if user_answer is None:
            return False
        return Harness._match_answer(user_answer, correct_answer, question_type=question_type)

    @staticmethod
    def _is_answered(user_answer: Optional[str]) -> bool:
        return user_answer is not None and str(user_answer).strip() != ""

    # ------------------------------------------------------------------
    # 1) 创建测试：冻结当前学习包的题目
    # ------------------------------------------------------------------
    def create_attempt(self, session_id: str) -> Dict[str, Any]:
        frozen_questions, stage_meta = self._collect_current_package_questions(session_id)
        if not frozen_questions:
            raise AssessmentError("当前学习包没有可测试的题目，无法创建测试")

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
            if package and package.get("questions"):
                for q in package["questions"]:
                    question_id = q.get("question_id")
                    db_question = self.db.get_question(question_id) if question_id else None
                    # The assessment is a frozen snapshot of test_questions,
                    # never an Agent3-generated question that happens to be in
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
        attempt = self._require_attempt(attempt_id)
        # 约束 5：重复提交拒绝
        if attempt["status"] == "submitted":
            raise AssessmentError("测试已提交，不能重复提交")

        questions = self.db.get_assessment_attempt_questions(attempt_id)
        answers = self.db.get_assessment_answers(attempt_id)

        # 约束 4：未完成全部必答题时不能提交
        missing = [
            q["question_id"]
            for q in questions
            if q["is_required"] and not self._is_answered(answers.get(q["question_id"], {}).get("user_answer"))
        ]
        if missing:
            raise AssessmentError(f"还有必答题未完成，不能提交：{missing}")

        # 统一判题
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
        # 约束 6：提交后返回带正确答案与解析的完整结果
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
