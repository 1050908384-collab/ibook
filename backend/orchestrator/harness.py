# -*- coding: utf-8 -*-
import asyncio
import json
import re
import uuid
from typing import Any, Dict, List, Optional

from agents.deterministic import (
    DeterministicContentAgent,
    DeterministicDiagnosticAgent,
    DeterministicStageDecisionAgent,
    DeterministicTechnicalReviewAgent,
    HybridContentAgent,
    HybridDiagnosticAgent,
    HybridTechnicalReviewAgent,
)
from config import settings
from models.contracts import (
    AnswerGenerationInput,
    AnswerRevisionInput,
    ContentGenerationInput,
    ContentRevisionInput,
    DiagnosticInput,
    ReviewInput,
    ReviewOutput,
    StageDecisionInput,
    StagePlanningInput,
)
from models.profile import PracticeProfile, UserProfile
from models.review import QualityReviewResult, ReviewError, ReviewRound, TechnicalReviewResult
from models.stage import LearningPackage, StageInfo
from orchestrator.state_manager import StateManager

# 审核辩论协议常量（协议第 7 节）
MAX_DEBATE_ROUNDS = 2  # 内层答辩复核轮次上限


class Harness:
    def __init__(self, state_manager: StateManager | None = None, execution_mode: str | None = None):
        self.state_manager = state_manager or StateManager()
        self.execution_mode = execution_mode or settings.EXECUTION_MODE
        if self.execution_mode not in {"deterministic", "hybrid"}:
            raise ValueError("EXECUTION_MODE must be 'deterministic' or 'hybrid'")
        self.agent1 = (
            HybridDiagnosticAgent(db_client=self.state_manager.db_client)
            if self.execution_mode == "hybrid"
            else DeterministicDiagnosticAgent(db_client=self.state_manager.db_client)
        )
    
        self.agent3 = HybridContentAgent() if self.execution_mode == "hybrid" else DeterministicContentAgent()
        self.agent4 = HybridTechnicalReviewAgent() if self.execution_mode == "hybrid" else DeterministicTechnicalReviewAgent()
        # Agent5已合并至Agent4，质量审核功能已集成
        self.agent6 = DeterministicStageDecisionAgent()

    def _log(self, session_id: str, event: str, detail: str = "") -> None:
        suffix = f" | {detail}" if detail else ""
        print(f"[Harness][session={session_id}] {event}{suffix}", flush=True)

    @staticmethod
    def _extract_user_facing_answer(raw_answer: str) -> str:
        """Keep model prompt examples and review traces out of the learner UI."""
        text = (raw_answer or "").strip()
        if not text:
            return ""

        # Some local models copy the JSON example embedded in a skill file.
        # Prefer the final fenced JSON answer when it is present.
        for block in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)):
            try:
                value = json.loads(block)
            except json.JSONDecodeError:
                continue
            answer = value.get("answer") if isinstance(value, dict) else None
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

        # Also accept an unfenced JSON object without exposing its wrapper.
        matches = re.findall(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
        for value in reversed(matches):
            try:
                answer = json.loads(f'"{value}"')
            except json.JSONDecodeError:
                answer = value.replace(r"\\n", "\n").replace(r'\\"', '"')
            if answer.strip():
                return answer.strip()

        return text

    @staticmethod
    def _is_greeting(question: str) -> bool:
        return bool(re.fullmatch(r"(?:你好|您好|嗨|哈喽|hello|hi)[！!。．，,？?\\s]*", question.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _is_greeting(question: str) -> bool:
        return bool(re.fullmatch(r"(?:\u4f60\u597d|\u60a8\u597d|\u55e8|\u54c8\u558e|hello|hi)[!\uff01.\u3002,\uff0c?\uff1f\s]*", question.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _greeting_answer(stage_name: str) -> str:
        return (
            f"你好，我是你的实训导师。你当前正在学习「{stage_name or '当前阶段'}」。"
            "可以直接问我概念、实操步骤或当前学习材料中的疑问。"
        )

    async def process_initial_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        self._log(session_id, "process_initial_input:start", f"message={user_input[:60]}")
        self.state_manager.require_session(session_id)
        previous_inputs = []
        for interaction in self.state_manager.get_interactions(session_id):
            if interaction.get("type") != "initial_diagnostic":
                continue
            data = interaction.get("data", {})
            previous_input = data.get("user_input") if isinstance(data, dict) else None
            if previous_input:
                previous_inputs.append(previous_input)
        diagnostic_input = "\n".join([*previous_inputs, user_input])
        diagnostic = await self.agent1.run(
            DiagnosticInput(session_id=session_id, user_input=diagnostic_input)
        )
        self.state_manager.update_practice_profile(
            session_id, PracticeProfile.from_user_input(user_input)
        )
        # 无论置信度是否达标，都保存 profile（草稿），允许用户通过 PATCH /profile 修改
        self.state_manager.update_profile(session_id, diagnostic.profile)
        self.state_manager.add_interaction(session_id, "initial_diagnostic", {"user_input": user_input})
        if diagnostic.profile.confidence < settings.CONFIDENCE_THRESHOLD:
            self._log(
                session_id,
                "process_initial_input:needs_more_info",
                f"confidence={diagnostic.profile.confidence:.2f}",
            )
            return {
                "status": "needs_more_info",
                "questions": diagnostic.followup_questions,
                "profile": diagnostic.profile.model_dump(mode="json"),
            }

        stage = await self.agent1.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=1,
        ))
        self.state_manager.update_stage(session_id, stage.stage)
        self._log(
            session_id,
            "process_initial_input:ready",
            f"goal={diagnostic.profile.goal}, stage={stage.stage.block_name}",
        )
        return {
            "status": "ready",
            "profile": diagnostic.profile.model_dump(mode="json"),
            "stage_info": stage.stage.model_dump(mode="json"),
        }

    async def start_stage(self, session_id: str) -> Dict[str, Any]:
        # 若已通过 plan-next 规划过 next_stage，则在真正启动阶段前把 next_stage 切换为当前 stage_info
        planned = self.state_manager.get_next_stage(session_id)
        if planned:
            self._log(session_id, "start_stage:apply_planned_next_stage", f"planned_stage_num={planned.get('stage_num')}, block={planned.get('block_name')}")
            self.state_manager.update_stage(session_id, StageInfo.model_validate(planned))
            # 切换完成后清理 next_stage / planning_notes，避免重复套用
            self.state_manager.clear_next_stage(session_id)

        stage = StageInfo.model_validate(self._require_stage(session_id))
        profile = UserProfile.model_validate(self._require_profile(session_id))
        practice_profile = PracticeProfile.model_validate(
            self.state_manager.get_practice_profile(session_id)
        )
        self._log(session_id, "start_stage:start", f"stage={stage.stage_num}, block={stage.block_name}")
        generated = await self.agent3.run(ContentGenerationInput(
            session_id=session_id,
            stage=stage,
            learning_profile=profile,
            practice_profile=practice_profile,
        ))
        learning_package, review_history = await self._review_learning_package(
            session_id,
            generated.learning_package,
            stage,
            profile,
        )
        # 从 _review_learning_package 获取审核 trace 信息（P3）
        review_trace = getattr(self, "_last_review_trace", None) or {}
        final_status = review_trace.get("final_status", "passed")
        package_status = review_trace.get("package_status", "approved")
        # 拍板：needs_human_review 下学习包降级可用（A 方案），不阻塞学习流
        review_passed = final_status in ("passed", "needs_human_review")
        needs_human_review = final_status == "needs_human_review"

        stage.status = "learning" if review_passed else "pending"
        self.state_manager.update_stage(session_id, stage)
        if review_passed and stage.subtopic_id:
            self.state_manager.db_client.set_subtopic_status(
                session_id=session_id,
                subtopic_id=stage.subtopic_id,
                status=0,
            )
        self.state_manager.update_learning_package(session_id, learning_package)
        self.state_manager.add_interaction(session_id, "stage_review_completed", {
            "stage_id": stage.stage_id,
            "passed": review_passed,
            "iterations": len(review_history),
            "review_trace_id": review_trace.get("trace_id"),
            "final_status": final_status,
            "needs_human_review": needs_human_review,
        })
        self._log(
            session_id,
            "start_stage:done",
            f"passed={review_passed}, iterations={len(review_history)}, "
            f"final_status={final_status}, needs_human_review={needs_human_review}",
        )
        return {
            "learning_package": learning_package.model_dump(mode="json"),
            "review_passed": review_passed,
            "iterations": len(review_history),
            "review_history": [item.model_dump(mode="json") for item in review_history],
            # P3：审核辩论协议返回字段（前端可据此展示"待审"标记）
            "review_trace_id": review_trace.get("trace_id"),
            "package_status": package_status,
            "needs_human_review": needs_human_review,
            "unresolved_issue_count": review_trace.get("unresolved_issue_count", 0),
            "unresolved_must_fix_count": review_trace.get("unresolved_must_fix_count", 0),
        }

    async def handle_question(self, session_id: str, question: str) -> Dict[str, Any]:
        stage = StageInfo.model_validate(self._require_stage(session_id))
        profile = UserProfile.model_validate(self._require_profile(session_id))
        self._log(session_id, "handle_question:start", f"question={question[:60]}")
        is_greeting = self._is_greeting(question)
        if is_greeting:
            raw_answer = self._greeting_answer(stage.block_name)
        else:
            generated = await self.agent3.generate_answer(AnswerGenerationInput(
                session_id=session_id,
                question=question,
                stage=stage,
                profile=profile,
            ))
            raw_answer = generated.answer
        answer, review_history = await self._review_answer(
            session_id,
            question,
            self._extract_user_facing_answer(raw_answer),
            stage,
            profile,
        )
        # A greeting is a conversational turn, not a technical answer. Agent4
        # may classify it as off-topic and ask Agent3 to rewrite it; preserve
        # the deterministic greeting instead of exposing that rewrite.
        answer = self._extract_user_facing_answer(raw_answer if is_greeting else answer)
        latest_review = review_history[-1]
        self.state_manager.add_qa_record(
            session_id,
            question,
            answer,
            latest_review.technical.passed,
            latest_review.quality.passed,
        )
        self.state_manager.add_interaction(session_id, "question_asked", {"question": question})
        # 答疑审核摘要（Agent46_API接口任务说明.md 第 1 节）：由合并后的 Agent4 产出
        # technical_passed / quality_passed / iterations / status / needs_human_review，
        # 供前端 qa 返回体的 review_summary 使用（审核逻辑仍归属 Agent4，harness 仅透传）。
        review_summary = self._build_answer_review_summary(
            latest_review.technical,
            iterations=len(review_history),
        )
        self._log(
            session_id,
            "handle_question:done",
            f"tech_pass={latest_review.technical.passed}, quality_pass={latest_review.quality.passed}, "
            f"review_status={review_summary['status']}, needs_human_review={review_summary['needs_human_review']}",
        )
        return {
            "answer": answer,
            "tech_pass": latest_review.technical.passed,
            "quality_pass": latest_review.quality.passed,
            "iterations": len(review_history),
            "review_history": [item.model_dump(mode="json") for item in review_history],
            # 第 1 节答疑审核摘要（前端 review_summary 字段来源）
            "review_summary": review_summary,
        }

    def _build_answer_review_summary(
        self,
        technical_result: TechnicalReviewResult,
        iterations: int = 1,
    ) -> Dict[str, Any]:
        """解析并调用 Agent4 的答疑审核摘要生成器（第 1 节 review_summary）。

        build_answer_review_summary 是无状态方法（仅由 TechnicalReviewResult + iterations
        推导 5 个固定字段），故按以下优先级健壮解析，避免对具体 reviewer 实现的强耦合：
          1. self.agent4._protocol_agent（真实 Deterministic/Hybrid 审核体持有的协议 Agent）
          2. self.agent4 本身直接暴露 build_answer_review_summary（直接注入 TechnicalReviewAgent）
          3. 兜底：用 TechnicalReviewAgent 的实现（stateless），兼容测试注入的精简/桩 reviewer
        """
        builder = None
        protocol_agent = getattr(self.agent4, "_protocol_agent", None)
        if protocol_agent is not None and hasattr(protocol_agent, "build_answer_review_summary"):
            builder = protocol_agent.build_answer_review_summary
        elif hasattr(self.agent4, "build_answer_review_summary"):
            builder = self.agent4.build_answer_review_summary
        if builder is None:
            from agents.agent_04_tech_review import TechnicalReviewAgent
            # build_answer_review_summary 不访问 self，直接以 None 作 self 调用底层函数即可，
            # 避免实例化 TechnicalReviewAgent（会加载全部工具链）。
            return TechnicalReviewAgent.build_answer_review_summary(
                None, technical_result, iterations=iterations
            )
        return builder(technical_result, iterations=iterations)

    async def submit_answer(self, session_id: str, question_id: str, user_answer: str) -> Dict[str, Any]:
        package = LearningPackage.model_validate(self._require_package(session_id))
        question = next((item for item in package.questions if item.question_id == question_id), None)
        if question is None:
            raise ValueError(f"Question not found in current stage: {question_id}")
        is_correct = self._match_answer(
            user_answer, question.correct_answer,
            question_type=getattr(question, 'question_type', '')
        )
        self.state_manager.add_answer_record(session_id, question_id, user_answer, is_correct)
        self._log(session_id, "submit_answer", f"question_id={question_id}, correct={is_correct}")
        return {"is_correct": is_correct, "correct_answer": question.correct_answer}

    def _resolve_stage_accuracy(self, session_id: str, stage: StageInfo) -> float:
        """确定本阶段用于 Agent6 决策的正确率（Agent46_API接口任务说明.md 第 2/3 节）。

        - §3 complete 流程要求「读取已提交测试的准确率」，§2 约束 7 要求正确率只能来自
          已提交的完整 attempt，不能从零散答题记录计算。
        - 因此优先取当前阶段已提交 attempt 的正确率；找不到时（例如仍走旧的 submit_answer
          零散答题链路）再退化为 StateManager.calculate_accuracy，保证向后兼容不阻断流程。
        """
        try:
            stage_id = getattr(stage, "stage_id", None)
            submitted = self.state_manager.db_client.get_submitted_attempt_accuracy(
                session_id, stage_id=stage_id
            )
            # 阶段维度没有已提交 attempt 时，退回会话维度最近一次已提交 attempt
            if submitted is None and stage_id is not None:
                submitted = self.state_manager.db_client.get_submitted_attempt_accuracy(session_id)
            if submitted and submitted.get("accuracy") is not None:
                self._log(
                    session_id,
                    "complete_stage:accuracy_source",
                    f"submitted_attempt={submitted.get('attempt_id')}, accuracy={submitted['accuracy']}",
                )
                return float(submitted["accuracy"])
        except Exception as exc:
            self._log(session_id, "complete_stage:accuracy_fallback", f"reason={exc}")
        # 兜底：旧的零散答题链路
        return self.state_manager.calculate_accuracy(session_id)

    def _resolve_stage_records(self, session_id: str, stage: StageInfo):
        """取本阶段已提交测试的逐题记录与聚合用时，供 Agent6 决策使用。

        - 与 _resolve_stage_accuracy 同源同口径（Agent46_API接口任务说明.md 第 2/3 节）：
          优先当前阶段已提交 attempt，退回会话维度最近一次已提交 attempt。
        - 逐题记录含 time_spent（前端 PUT .../answers/{qid} 提交的答题用时），
          聚合用时换算为小时透传给 Agent6，替代仅从 interactions 估算的耗时。
        - 无已提交 attempt 时返回 ([], 0.0)，Agent6 退化为旧的 interactions 估算，不阻断流程。
        """
        try:
            stage_id = getattr(stage, "stage_id", None)
            records = self.state_manager.db_client.get_submitted_attempt_records(
                session_id, stage_id=stage_id
            )
            if records and records.get("question_records"):
                total_seconds = float(records.get("total_time_spent", 0) or 0)
                time_spent_hours = round(total_seconds / 3600.0, 2)
                self._log(
                    session_id,
                    "complete_stage:records_source",
                    f"submitted_attempt={records.get('attempt_id')}, "
                    f"questions={len(records['question_records'])}, time_spent_hours={time_spent_hours}",
                )
                return records["question_records"], time_spent_hours
        except Exception as exc:
            self._log(session_id, "complete_stage:records_fallback", f"reason={exc}")
        return [], 0.0

    async def complete_stage(self, session_id: str) -> Dict[str, Any]:
        """
        complete_stage 新职责（P3 按需阶段规划）：
        1. 只负责调用 Agent6 决策
        2. 保存 Agent6 更新后的用户画像 / 个性化数据
        3. 只有在全部知识盲区都掌握时，才真正结束 session
        4. 返回 status/session_id/accuracy/updated_profile/stage_summary/
           next_strategy/can_plan_next_stage/session_status/all_mastered/
           final_report/phase/available_actions
        5. 不直接推进当前阶段到下一个阶段编号，不生成 next_stage
        """
        profile = UserProfile.model_validate(self._require_profile(session_id))
        stage = StageInfo.model_validate(self._require_stage(session_id))
        accuracy = self._resolve_stage_accuracy(session_id, stage)
        assessment_records, time_spent_hours = self._resolve_stage_records(session_id, stage)
        self._log(session_id, "complete_stage:start", f"stage={stage.stage_num}, accuracy={accuracy:.2f}")
        decision = await self.agent6.run(StageDecisionInput(
            session_id=session_id,
            profile=profile,
            stage=stage,
            accuracy=accuracy,
            qa_history=self.state_manager.get_qa_history(session_id),
            interactions=self.state_manager.get_interactions(session_id),
            assessment_records=assessment_records,
            time_spent_hours=time_spent_hours,
        ))

        # 更新当前阶段的结果状态（只改当前 stage，不推进编号）
        stage.status = "completed" if accuracy >= 0.8 else "retreat"
        stage.accuracy = accuracy
        stage.strategy_used = decision.strategy

        if stage.subtopic_id:
            self.state_manager.db_client.set_subtopic_status(
                session_id=session_id,
                subtopic_id=stage.subtopic_id,
                status=1 if accuracy >= 0.8 else 0,
                mastery_score=1.0 if accuracy >= 0.8 else max(0.4, round(accuracy, 3)),
                last_accuracy=accuracy,
                increment_attempt=True,
            )

        synced_profile = await self._sync_profile_from_subtopic_status(decision.profile)
        # A stage can be an inserted prerequisite node for a later target
        # block.  Its ``gap_skill`` still names that target, so marking the
        # whole gap as mastered here would prematurely end the session.
        # ``_sync_profile_from_subtopic_status`` is the sole authority for
        # block completion and only reports mastery after every subtopic in
        # the target block has actually been completed.
        self.state_manager.update_stage(session_id, stage)
        self.state_manager.update_profile(session_id, synced_profile)

        # 只有全部掌握才真正结束 session
        all_mastered = all(gap.progress >= 1.0 for gap in synced_profile.knowledge_gaps)
        if all_mastered:
            self.state_manager.end_session(session_id)
            self._log(session_id, "complete_stage:all_mastered -> end_session")
            final_report = self._generate_final_report(synced_profile, accuracy)
        else:
            # 未全部掌握：session 保持 active，等用户按需触发 plan-next
            final_report = None
            self._log(
                session_id,
                f"complete_stage:keep_active | accuracy={accuracy:.2f} | all_mastered=False",
            )

        # 可规划下一阶段的条件：session 仍 active 即可；即便 repeat 场景也允许重新规划
        session_state = self.state_manager.get_session(session_id)
        session_status = session_state.get("status") if session_state else "active"
        can_plan_next_stage = not all_mastered and session_status == "active"

        result_status = "completed" if all_mastered else ("repeat" if accuracy < 0.8 else "continue")
        # 阶段结果的 phase / available_actions（Agent46_API接口任务说明.md 第 3 节）：
        # - 结业（all_mastered）：phase=completed，无后续动作
        # - 未结业：phase=stage_result，开放 plan_next_stage（前端据此展示“生成下一阶段”）
        if all_mastered:
            phase = "completed"
            available_actions: List[str] = []
        else:
            phase = "stage_result"
            available_actions = ["plan_next_stage"] if can_plan_next_stage else []
        return {
            "status": result_status,
            "session_id": session_id,
            "accuracy": accuracy,
            "updated_profile": synced_profile.model_dump(mode="json"),
            "stage_summary": decision.stage_summary,
            "next_strategy": decision.strategy,
            "can_plan_next_stage": can_plan_next_stage,
            "session_status": session_status,
            "all_mastered": all_mastered,
            "final_report": final_report,
            "phase": phase,
            "available_actions": available_actions,
        }

    async def plan_next_stage(self, session_id: str, adjustment_notes: str | None = None) -> Dict[str, Any]:
        """
        按需生成下一阶段（plan-next 接口）：
        - 基于 complete_stage 后最新的画像，调用 Agent1 的规划链路
        - 生成真正的 next_stage 并保存到 session state
        - 不立刻把 next_stage 覆盖到当前 stage_info（留待 start_stage 或前端确认后切换）

        stage_num 选择规则：
        - accuracy >= 0.8（已掌握本阶段）：stage_num = current_stage + 1
        - accuracy < 0.8（retreat/repeat）：stage_num = current_stage（重试当前阶段）
        """
        profile = UserProfile.model_validate(self._require_profile(session_id))
        current_stage_info = StageInfo.model_validate(self._require_stage(session_id))
        self._log(
            session_id,
            "plan_next_stage:start",
            f"current_stage={current_stage_info.stage_num}, accuracy={current_stage_info.accuracy or 0:.2f}",
        )

        accuracy = current_stage_info.accuracy or 0.0
        if accuracy >= 0.8:
            next_stage_num = current_stage_info.stage_num + 1
            default_notes = f"本阶段（{current_stage_info.stage_num}）已达标，推进到下一阶段"
        else:
            next_stage_num = current_stage_info.stage_num
            default_notes = f"本阶段（{current_stage_info.stage_num}）正确率不足 80%，重试当前阶段"

        user_notes = adjustment_notes.strip() if adjustment_notes and adjustment_notes.strip() else None
        planning_notes = (
            f"{default_notes}；{user_notes}" if user_notes else default_notes
        )

        planned = await self.agent1.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=profile,
            stage_num=next_stage_num,
            adjustment_notes=user_notes,
        ))
        next_stage = planned.stage
        self.state_manager.update_next_stage(session_id, next_stage)
        self.state_manager.update_planning_notes(session_id, planning_notes)

        session_state = self.state_manager.get_session(session_id)
        session_status = session_state.get("status") if session_state else "active"
        self._log(
            session_id,
            "plan_next_stage:done",
            f"next_stage_num={next_stage.stage_num}, block={next_stage.block_name}",
        )
        return {
            "status": "planned",
            "session_id": session_id,
            "next_stage": next_stage.model_dump(mode="json"),
            "updated_profile": profile.model_dump(mode="json"),
            "planning_notes": planning_notes,
            "session_status": session_status,
            # plan-next 成功后的 phase / available_actions（Agent46_API接口任务说明.md 第 3 节）
            "phase": "next_stage_ready",
            "available_actions": ["start_stage"],
        }

    async def _review_learning_package(
        self,
        session_id: str,
        package: LearningPackage,
        stage: StageInfo,
        profile: UserProfile,
    ) -> tuple[LearningPackage, list[ReviewRound]]:
        """审核学习包：审核 → 答辩 → 复核 → 裁决（协议第 4~5 节）。

        外层 review iteration（MAX_REVIEW_ITERATIONS）内嵌 debate 子循环。
        每个 iteration = 一次完整辩论轮（Agent4 审核 → Agent3 答辩 → Agent4 复核 → Orchestrator 裁决 → Agent3 修改）。
        达到最大轮次仍有未决 must_fix → needs_human_review（非阻塞，package 降级为 provisional）。
        """
        history: list[ReviewRound] = []
        current_package = package
        # 循环外创建一次 trace_id，全程复用（协议：同一 trace 内迭代）
        trace_id = f"tr_{uuid.uuid4().hex[:16]}"
        current_package.review_trace_id = trace_id
        final_status = "in_progress"
        package_status = "draft"

        # 落库：创建 trace + 首轮 sources + claims（P2）
        self._persist_trace_init(trace_id, session_id, stage, current_package)

        for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
            print(f"=== 学习包审核循环：第{iteration}轮（trace={trace_id}）===")

            # 步骤1：Agent4 首轮审核（传 sources/claims/trace_id/round_index）
            reviews = await self._review(ReviewInput(
                session_id=session_id,
                target_type="learning_package",
                content=current_package.tutorial,
                stage=stage,
                profile=profile,
                code_examples=current_package.code_examples,
                trace_id=trace_id,
                sources=[s.model_dump(mode="json") for s in current_package.sources],
                claims=[c.model_dump(mode="json") for c in current_package.claims],
                recommended_videos=current_package.recommended_videos,
                round_index=iteration,
            ))

            # 如果审核通过，直接结束
            if reviews.passed:
                round_result = ReviewRound(
                    iteration=iteration,
                    technical=reviews.technical,
                    quality=reviews.quality,
                )
                history.append(round_result)
                self._record_review_round(session_id, "learning_package", round_result)
                final_status = "passed"
                package_status = "approved"
                print(f"=== 学习包第{iteration}轮审核通过 ===")
                break

            # 步骤2：Agent3 答辩（对每条 issue 逐条回复）
            issues = [e.model_dump(mode="json") for e in reviews.technical.errors]
            package_dict = current_package.model_dump(mode="json")
            print(f"=== 调用 Agent3.reply_to_review（{len(issues)} 条 issue）===")
            author_replies = await self.agent3.reply_to_review(
                session_id, issues, package_dict
            )

            # 步骤3：Agent4 复核裁决（对 Agent3 的回复做 persist/withdraw/downgrade）
            print(f"=== 调用 Agent4.rebut ===")
            verdicts = await self.agent4.rebut(session_id, issues, author_replies)
            print(f"=== Agent4 复核裁决结果：{verdicts} ===")

            # 步骤4：Orchestrator 裁决 — 根据复核结果分类 issues
            accepted_issues, unresolved_must_fix_count = self._adjudicate_issues(
                reviews.technical.errors, verdicts
            )

            # 落库：issues + round + responses（P2）
            self._persist_debate_round(
                trace_id, iteration, reviews.technical.errors,
                author_replies, verdicts, accepted_issues, unresolved_must_fix_count,
            )

            # 步骤5：Agent3 修改学习包（只处理被裁决通过/降级的 issue）
            if accepted_issues:
                print(f"=== 调用 Agent3.modify_package（{len(accepted_issues)} 条已裁决 issue）===")
                revised_package_dict = await self.agent3.modify_package(
                    session_id,
                    package_dict,
                    {"tech": reviews.technical.errors, "quality": reviews.quality.errors},
                    accepted_issues,
                )
                current_package = LearningPackage.model_validate(revised_package_dict)
                current_package.review_trace_id = trace_id
                # 落库：更新 claims（status 变为 revised/dropped）（P2）
                self._persist_claims_update(trace_id, current_package)

            round_result = ReviewRound(
                iteration=iteration,
                technical=reviews.technical,
                quality=reviews.quality,
            )
            history.append(round_result)
            self._record_review_round(session_id, "learning_package", round_result)

            # 步骤6：判断是否通过 — 所有 must_fix 已解决则通过
            if unresolved_must_fix_count == 0 and not reviews.technical.errors:
                final_status = "passed"
                package_status = "approved"
                print(f"=== 学习包第{iteration}轮辩论后通过 ===")
                break

            if unresolved_must_fix_count == 0:
                # 无未决 must_fix，但有 minor 问题 → 视为通过
                final_status = "passed"
                package_status = "approved"
                print(f"=== 学习包第{iteration}轮辩论后通过（剩余非 must_fix 问题）===")
                break

            if iteration == settings.MAX_REVIEW_ITERATIONS:
                # 达到最大轮次仍有未决 must_fix → needs_human_review（协议第 7 节）
                final_status = "needs_human_review"
                package_status = "provisional"
                print(
                    f"=== 学习包达到最大审核轮次，{unresolved_must_fix_count} 个 must_fix 未决，"
                    f"标记为 needs_human_review（package 降级为 provisional）==="
                )
                break

            print(f"=== 第{iteration}轮仍有 {unresolved_must_fix_count} 个 must_fix 未决，进入下一轮 ===")

        # 设置 package 最终状态
        current_package.review_trace_id = trace_id
        unresolved_count = sum(
            1 for r in history for e in r.technical.errors if e.must_fix
        )
        self._last_review_trace = {
            "trace_id": trace_id,
            "final_status": final_status,
            "package_status": package_status,
            "unresolved_issue_count": unresolved_count,
            "unresolved_must_fix_count": unresolved_count,
        }
        # 落库：更新 trace 最终状态（P2）
        self._persist_trace_finalize(trace_id, final_status, package_status, unresolved_count)
        return current_package, history

    def _persist_trace_init(
        self, trace_id: str, session_id: str, stage: StageInfo, package: LearningPackage
    ) -> None:
        """落库：创建 trace + sources + claims（循环外调用一次）。"""
        try:
            db = self.state_manager.db_client
            db.save_review_trace(
                trace_id=trace_id,
                session_id=session_id,
                stage_id=stage.stage_id,
                target_type="learning_package",
                package_version=package.version,
                package_status="draft",
                review_mode="debate",
                final_status="in_progress",
            )
            db.save_learning_package_sources(
                trace_id, [s.model_dump(mode="json") for s in package.sources]
            )
            db.save_learning_package_claims(
                trace_id, [c.model_dump(mode="json") for c in package.claims]
            )
        except Exception as e:
            print(f"[Harness] 落库失败（trace init）: {e}")

    def _persist_debate_round(
        self,
        trace_id: str,
        iteration: int,
        errors: List[ReviewError],
        author_replies: List[Dict[str, Any]],
        verdicts: Dict[str, str],
        accepted_issues: List[Dict[str, Any]],
        unresolved_must_fix_count: int,
    ) -> None:
        """落库：issues + round + responses（每轮辩论后调用）。"""
        try:
            db = self.state_manager.db_client
            round_id = f"rd_{trace_id}_{iteration}"

            # 1. issues（带 opened_in_round_num）
            issues_for_db = []
            for e in errors:
                iss = e.model_dump(mode="json")
                iss["opened_in_round_num"] = iteration
                issues_for_db.append(iss)
            db.save_review_debate_issues(trace_id, issues_for_db)

            # 2. round（开轮并立即关闭）
            verdict_label = "all_resolved" if unresolved_must_fix_count == 0 else "partial_resolved"
            db.save_review_debate_round(
                trace_id=trace_id,
                round_id=round_id,
                round_num=iteration,
                review_iteration=iteration,
                round_status="closed",
                orchestrator_verdict=verdict_label,
                verdict_summary=f"unresolved_must_fix={unresolved_must_fix_count}",
            )

            # 3. Agent3 答辩 responses
            author_responses = []
            for reply in author_replies:
                cid = reply.get("canonical_error_id", "")
                author_responses.append({
                    "round_id": round_id,
                    "issue_id": f"iss_{trace_id}_{cid}",
                    "actor": "agent3",
                    "response_type": "author_reply",
                    "disposition": reply.get("disposition", "accepted"),
                    "rationale": reply.get("reason"),
                    "source_refs": reply.get("source_refs", []),
                    "patched_sections": reply.get("patched_sections", []),
                })
            db.save_review_debate_responses(round_id, author_responses)

            # 4. Agent4 复核 responses
            reviewer_responses = []
            for cid, verdict in verdicts.items():
                reviewer_responses.append({
                    "round_id": round_id,
                    "issue_id": f"iss_{trace_id}_{cid}",
                    "actor": "agent4",
                    "response_type": "reviewer_verdict",
                    "disposition": verdict,
                    "rationale": None,
                })
            db.save_review_debate_responses(round_id, reviewer_responses)

            # 5. Orchestrator 裁决 responses + 更新 issue 状态
            orchestrator_responses = []
            for iss in accepted_issues:
                cid = iss.get("canonical_error_id", "")
                issue_id = f"iss_{trace_id}_{cid}"
                # withdraw → accepted, downgrade → partially_accepted
                verdict = verdicts.get(cid, "withdraw")
                disposition = "accepted" if verdict == "withdraw" else "partially_accepted"
                orchestrator_responses.append({
                    "round_id": round_id,
                    "issue_id": issue_id,
                    "actor": "orchestrator",
                    "response_type": "orchestrator_ruling",
                    "disposition": disposition,
                    "rationale": f"Agent4 verdict={verdict}",
                })
                db.update_review_debate_issue_status(
                    issue_id=issue_id,
                    current_status=disposition,
                    final_disposition=disposition,
                    resolved_in_round_num=iteration,
                )
            db.save_review_debate_responses(round_id, orchestrator_responses)

            # persist 的 issue（persist 且 must_fix）标记为 open
            for e in errors:
                cid = e.canonical_error_id or ""
                if verdicts.get(cid) == "persist":
                    db.update_review_debate_issue_status(
                        issue_id=f"iss_{trace_id}_{cid}",
                        current_status="open",
                    )
        except Exception as e:
            print(f"[Harness] 落库失败（debate round）: {e}")

    def _persist_claims_update(self, trace_id: str, package: LearningPackage) -> None:
        """落库：modify_package 后更新 claims（status 变为 revised/dropped）。"""
        try:
            db = self.state_manager.db_client
            db.save_learning_package_claims(
                trace_id, [c.model_dump(mode="json") for c in package.claims]
            )
        except Exception as e:
            print(f"[Harness] 落库失败（claims update）: {e}")

    def _persist_trace_finalize(
        self, trace_id: str, final_status: str, package_status: str, unresolved_count: int
    ) -> None:
        """落库：更新 trace 最终状态（循环结束调用）。"""
        try:
            db = self.state_manager.db_client
            db.update_review_trace_status(
                trace_id=trace_id,
                final_status=final_status,
                package_status=package_status,
                unresolved_issue_count=unresolved_count,
                unresolved_must_fix_count=unresolved_count,
            )
        except Exception as e:
            print(f"[Harness] 落库失败（trace finalize）: {e}")

    def _adjudicate_issues(
        self,
        errors: List[ReviewError],
        verdicts: Dict[str, str],
    ) -> tuple[List[Dict[str, Any]], int]:
        """Orchestrator 裁决：根据 Agent4 复核结果分类 issues。

        返回 (accepted_issues, unresolved_must_fix_count)：
        - accepted_issues：Agent3 需要 modify 的 issues（withdraw + downgrade）
        - unresolved_must_fix_count：仍坚持(persist)且 must_fix=true 的数量
        """
        accepted_issues: List[Dict[str, Any]] = []
        unresolved_must_fix_count = 0

        for err in errors:
            cid = err.canonical_error_id or ""
            verdict = verdicts.get(cid, "persist")

            if verdict == "withdraw":
                # Agent4 撤销该 issue → Agent3 接受处理
                accepted_issues.append(err.model_dump(mode="json"))
            elif verdict == "downgrade":
                # Agent4 降级 → 降级 severity，若原 must_fix 则解除
                err.severity = "minor" if err.severity == "major" else err.severity
                if err.must_fix:
                    err.must_fix = False
                    err.must_fix_reason = "Agent4 复核降级，解除 must_fix"
                accepted_issues.append(err.model_dump(mode="json"))
            else:
                # persist → Agent4 坚持该 issue
                if err.must_fix:
                    # must_fix=True：强制 Agent3 修改，但仍计入未决（进下一轮重新审核）
                    accepted_issues.append(err.model_dump(mode="json"))
                    unresolved_must_fix_count += 1
                # must_fix=False：minor 问题，允许通过，不强制修改

        return accepted_issues, unresolved_must_fix_count

    async def _review_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        stage: StageInfo,
        profile: UserProfile,
    ) -> tuple[str, list[ReviewRound]]:
        history = []
        current_answer = answer
        for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
            print(f"=== 问答审核循环：第{iteration}轮 ===")
            reviews = await self._review(ReviewInput(
                session_id=session_id,
                target_type="answer",
                content=current_answer,
                question=question,
                stage=stage,
                profile=profile,
            ))
            round_result = ReviewRound(
                iteration=iteration,
                technical=reviews.technical,
                quality=reviews.quality,
            )
            history.append(round_result)
            self._record_review_round(session_id, "answer", round_result)
            if round_result.passed:
                print(f"=== 回答第{iteration}轮审核通过 ===")
                break
            if iteration == settings.MAX_REVIEW_ITERATIONS:
                print("=== 回答达到最大审核轮次，返回未通过状态 ===")
                break
            print(f"=== Orchestrator汇总第{iteration}轮意见，调用Agent3修改回答 ===")
            revised = await self.agent3.revise_answer(AnswerRevisionInput(
                session_id=session_id,
                question=question,
                answer=current_answer,
                stage=stage,
                profile=profile,
                technical_errors=reviews.technical.errors,
                quality_errors=reviews.quality.errors,
            ))
            current_answer = revised.answer
        return current_answer, history

    async def _review(self, request: ReviewInput) -> ReviewOutput:
        # Agent5已合并至Agent4，统一调用合并后的TechnicalReviewAgent
        try:
            technical = await self.agent4.run(request)
        except Exception as exc:
            technical = TechnicalReviewResult(
                passed=False,
                errors=[ReviewError(
                    error_type="review_system_error",
                    severity="critical",
                    description=f"Agent4 review failed: {exc}",
                    suggestion="重新执行审核，若仍失败请检查合并后的 Agent4、skill 与 tool 链路",
                )],
                review_details={"exception": str(exc)},
            )

        # 质量审核结果已合并在technical结果中
        quality = QualityReviewResult(
            passed=technical.passed,
            errors=technical.errors,
            scores=getattr(technical, 'scores', None),
            review_details=getattr(technical, 'review_details', None),
        )

        return ReviewOutput(technical=technical, quality=quality)

    def _record_review_round(self, session_id: str, target_type: str, review_round: ReviewRound) -> None:
        print(
            f"=== Orchestrator审核汇总：target={target_type}, round={review_round.iteration}, "
            f"merged_errors={len(review_round.technical.errors)}, passed={review_round.passed} ==="
        )
        for error in review_round.technical.errors:
            print(f"  [合并审核] [{error.severity}] {error.description} -> {error.suggestion or '未提供建议'}")
        self.state_manager.add_interaction(session_id, "review_round", {
            "target_type": target_type,
            **review_round.model_dump(mode="json"),
            "passed": review_round.passed,
        })

    async def _sync_profile_from_subtopic_status(self, profile: UserProfile) -> UserProfile:
        synced_profile = profile.model_copy(deep=True)
        for gap in synced_profile.knowledge_gaps:
            block_info = {}
            try:
                block_info = await self.agent1.knowledge_block_query_tool.query_block(gap.skill)
            except Exception:
                block_info = {}
            block_id = block_info.get("block_id")
            if not block_id:
                continue
            summary = self.state_manager.db_client.get_block_status_summary(
                session_id=synced_profile.session_id,
                block_id=block_id,
            )
            if not summary:
                continue
            gap.status = summary["status"]
            gap.progress = summary["progress"]
            if not gap.prerequisites:
                gap.prerequisites = block_info.get("prerequisites", [])
        synced_profile.version += 1
        return synced_profile

    def _require_profile(self, session_id: str) -> dict:
        profile = self.state_manager.get_profile(session_id)
        if not profile:
            raise ValueError("Session has no user profile")
        return profile

    def _require_stage(self, session_id: str) -> dict:
        stage = self.state_manager.get_current_stage(session_id)
        if not stage:
            raise ValueError("Session has no current stage")
        return stage

    def _require_package(self, session_id: str) -> dict:
        package = self.state_manager.get_learning_package(session_id)
        if not package:
            raise ValueError("Session has no learning package")
        return package

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """标准化答案：去空格、转小写、去标点符号。"""
        import re
        cleaned = re.sub(r'[^\w.\u4e00-\u9fff/]', '', answer.strip().lower())
        return cleaned

    @staticmethod
    def _match_answer(user_answer: str, correct_answer: str, question_type: str = '') -> bool:
        """
        多级答案匹配策略（P2 模糊匹配）。
        1. 精确匹配（标准化后相等）
        2. 选择题：单字母选项匹配
        3. 包含匹配（正确答案含用户答案或反之）
        4. 数值近似（误差 < 5%）
        """
        user_norm = Harness._normalize_answer(user_answer)
        correct_norm = Harness._normalize_answer(correct_answer)

        # 1. 精确匹配
        if user_norm == correct_norm:
            return True

        # 2. 选择题选项字母匹配
        if question_type in ('选择', '选择题', '单选', 'multiple_choice'):
            if len(user_norm) == 1 and user_norm.isalpha() and \
               correct_norm.startswith(user_norm):
                return True

        # 3. 包含匹配（双向，至少2字符防止单字母误判）
        if len(user_norm) >= 3 and len(correct_norm) >= 3:
            if user_norm in correct_norm or correct_norm in user_norm:
                return True

        # 4. 数值近似
        try:
            user_num = float(user_norm.replace(',', '').replace('，', ''))
            correct_num = float(correct_norm.replace(',', '').replace('，', ''))
            if abs(user_num - correct_num) / max(abs(correct_num), 1.0) < 0.05:
                return True
        except (ValueError, ZeroDivisionError):
            pass

        return False

    def _generate_final_report(self, profile: UserProfile, accuracy: float) -> Dict[str, Any]:
        return {
            "goal": profile.goal,
            "accuracy": accuracy,
            "mastered_skills": [gap.skill for gap in profile.knowledge_gaps if gap.progress >= 1.0],
            "profile_version": profile.version,
        }

    # ===== P1: 公共 Session 管理方法 =====
    def get_session(self, session_id: str) -> dict | None:
        return self.state_manager.get_session(session_id)

    def get_runtime_state(self, session_id: str) -> dict | None:
        return self.state_manager.get_session(session_id)

    def end_session(self, session_id: str) -> None:
        self.state_manager.end_session(session_id)
        self._log(session_id, "session:end")

    def abandon_session(self, session_id: str) -> None:
        self.state_manager.abandon_session(session_id)
        self._log(session_id, "session:abandon")

    def create_session(self, session_id: str | None = None) -> dict:
        session = self.state_manager.create_session(session_id)
        self._log(session["session_id"], "session:create")
        return session
