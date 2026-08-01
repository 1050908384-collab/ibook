# -*- coding: utf-8 -*-
import asyncio
from typing import Any, Dict

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
from models.profile import UserProfile
from models.review import QualityReviewResult, ReviewError, ReviewRound, TechnicalReviewResult
from models.stage import LearningPackage, StageInfo
from orchestrator.state_manager import StateManager


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

    async def process_initial_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        self._log(session_id, "process_initial_input:start", f"message={user_input[:60]}")
        self.state_manager.require_session(session_id)
        diagnostic = await self.agent1.run(DiagnosticInput(session_id=session_id, user_input=user_input))
        if diagnostic.profile.confidence < settings.CONFIDENCE_THRESHOLD:
            self._log(
                session_id,
                "process_initial_input:needs_more_info",
                f"confidence={diagnostic.profile.confidence:.2f}",
            )
            return {"status": "needs_more_info", "questions": diagnostic.followup_questions}

        self.state_manager.update_profile(session_id, diagnostic.profile)
        stage = await self.agent1.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=1,
        ))
        self.state_manager.update_stage(session_id, stage.stage)
        self.state_manager.add_interaction(session_id, "initial_diagnostic", {"user_input": user_input})
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
        self._log(session_id, "start_stage:start", f"stage={stage.stage_num}, block={stage.block_name}")
        generated = await self.agent3.run(ContentGenerationInput(session_id=session_id, stage=stage))
        learning_package, review_history = await self._review_learning_package(
            session_id,
            generated.learning_package,
            stage,
            profile,
        )
        review_passed = review_history[-1].passed

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
        })
        self._log(
            session_id,
            "start_stage:done",
            f"passed={review_passed}, iterations={len(review_history)}",
        )
        return {
            "learning_package": learning_package.model_dump(mode="json"),
            "review_passed": review_passed,
            "iterations": len(review_history),
            "review_history": [item.model_dump(mode="json") for item in review_history],
        }

    async def handle_question(self, session_id: str, question: str) -> Dict[str, Any]:
        stage = StageInfo.model_validate(self._require_stage(session_id))
        profile = UserProfile.model_validate(self._require_profile(session_id))
        self._log(session_id, "handle_question:start", f"question={question[:60]}")
        generated = await self.agent3.generate_answer(AnswerGenerationInput(
            session_id=session_id,
            question=question,
            stage=stage,
            profile=profile,
        ))
        answer, review_history = await self._review_answer(
            session_id,
            question,
            generated.answer,
            stage,
            profile,
        )
        latest_review = review_history[-1]
        self.state_manager.add_qa_record(
            session_id,
            question,
            answer,
            latest_review.technical.passed,
            latest_review.quality.passed,
        )
        self.state_manager.add_interaction(session_id, "question_asked", {"question": question})
        self._log(
            session_id,
            "handle_question:done",
            f"tech_pass={latest_review.technical.passed}, quality_pass={latest_review.quality.passed}",
        )
        return {
            "answer": answer,
            "tech_pass": latest_review.technical.passed,
            "quality_pass": latest_review.quality.passed,
            "iterations": len(review_history),
            "review_history": [item.model_dump(mode="json") for item in review_history],
        }

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

    async def complete_stage(self, session_id: str) -> Dict[str, Any]:
        """
        complete_stage 新职责（P3 按需阶段规划）：
        1. 只负责调用 Agent6 决策
        2. 保存 Agent6 更新后的用户画像 / 个性化数据
        3. 只有在全部知识盲区都掌握时，才真正结束 session
        4. 返回 status/session_id/accuracy/updated_profile/stage_summary/
           next_strategy/can_plan_next_stage/session_status/all_mastered
        5. 不直接推进当前阶段到下一个阶段编号，不生成 next_stage
        """
        profile = UserProfile.model_validate(self._require_profile(session_id))
        stage = StageInfo.model_validate(self._require_stage(session_id))
        accuracy = self.state_manager.calculate_accuracy(session_id)
        self._log(session_id, "complete_stage:start", f"stage={stage.stage_num}, accuracy={accuracy:.2f}")
        decision = await self.agent6.run(StageDecisionInput(
            session_id=session_id,
            profile=profile,
            stage=stage,
            accuracy=accuracy,
            qa_history=self.state_manager.get_qa_history(session_id),
            interactions=self.state_manager.get_interactions(session_id),
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
        if accuracy >= 0.8:
            for gap in synced_profile.knowledge_gaps:
                if gap.skill == stage.gap_skill:
                    gap.status = "已掌握"
                    gap.progress = 1.0
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
        }

    async def _review_learning_package(
        self,
        session_id: str,
        package: LearningPackage,
        stage: StageInfo,
        profile: UserProfile,
    ) -> tuple[LearningPackage, list[ReviewRound]]:
        history = []
        current_package = package
        for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
            print(f"=== 学习包审核循环：第{iteration}轮 ===")
            reviews = await self._review(ReviewInput(
                session_id=session_id,
                target_type="learning_package",
                content=current_package.tutorial,
                stage=stage,
                profile=profile,
                code_examples=current_package.code_examples,
            ))
            round_result = ReviewRound(
                iteration=iteration,
                technical=reviews.technical,
                quality=reviews.quality,
            )
            history.append(round_result)
            self._record_review_round(session_id, "learning_package", round_result)
            if round_result.passed:
                print(f"=== 学习包第{iteration}轮审核通过 ===")
                break
            if iteration == settings.MAX_REVIEW_ITERATIONS:
                print("=== 学习包达到最大审核轮次，保留未通过状态 ===")
                break
            print(f"=== Orchestrator汇总第{iteration}轮意见，调用Agent3修改学习包 ===")
            revised = await self.agent3.revise_learning_package(ContentRevisionInput(
                session_id=session_id,
                learning_package=current_package,
                stage=stage,
                profile=profile,
                technical_errors=reviews.technical.errors,
                quality_errors=reviews.quality.errors,
            ))
            current_package = revised.learning_package
        return current_package, history

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
