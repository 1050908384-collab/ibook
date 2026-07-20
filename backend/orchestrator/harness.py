from typing import Any, Dict

from agents.deterministic import (
    DeterministicContentAgent,
    DeterministicDiagnosticAgent,
    DeterministicQualityReviewAgent,
    DeterministicStageDecisionAgent,
    DeterministicStagePlannerAgent,
    DeterministicTechnicalReviewAgent,
)
from config import settings
from models.contracts import (
    ContentGenerationInput,
    DiagnosticInput,
    ReviewInput,
    ReviewOutput,
    StageDecisionInput,
    StagePlanningInput,
)
from models.profile import UserProfile
from models.stage import LearningPackage, StageInfo
from orchestrator.state_manager import StateManager


class Harness:
    def __init__(self, state_manager: StateManager | None = None, execution_mode: str | None = None):
        self.state_manager = state_manager or StateManager()
        self.execution_mode = execution_mode or settings.EXECUTION_MODE
        if self.execution_mode != "deterministic":
            raise ValueError("Only deterministic execution mode is available in the minimal closed loop")
        self.agent1 = DeterministicDiagnosticAgent()
        self.agent2 = DeterministicStagePlannerAgent()
        self.agent3 = DeterministicContentAgent()
        self.agent4 = DeterministicTechnicalReviewAgent()
        self.agent5 = DeterministicQualityReviewAgent()
        self.agent6 = DeterministicStageDecisionAgent()

    async def process_initial_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        self.state_manager.require_session(session_id)
        diagnostic = await self.agent1.run(DiagnosticInput(session_id=session_id, user_input=user_input))
        if diagnostic.profile.confidence < settings.CONFIDENCE_THRESHOLD:
            return {"status": "needs_more_info", "questions": diagnostic.followup_questions}

        self.state_manager.update_profile(session_id, diagnostic.profile)
        stage = await self.agent2.run(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=1,
        ))
        self.state_manager.update_stage(session_id, stage.stage)
        self.state_manager.add_interaction(session_id, "initial_diagnostic", {"user_input": user_input})
        return {
            "status": "ready",
            "profile": diagnostic.profile.model_dump(mode="json"),
            "stage_info": stage.stage.model_dump(mode="json"),
        }

    async def start_stage(self, session_id: str) -> Dict[str, Any]:
        stage = StageInfo.model_validate(self._require_stage(session_id))
        generated = await self.agent3.run(ContentGenerationInput(session_id=session_id, stage=stage))
        reviews = await self._review(session_id, generated.learning_package)
        if not reviews.passed:
            raise ValueError("Deterministic learning package failed validation")

        stage.status = "learning"
        self.state_manager.update_stage(session_id, stage)
        self.state_manager.update_learning_package(session_id, generated.learning_package)
        self.state_manager.add_interaction(session_id, "stage_started", {"stage_id": stage.stage_id})
        return {
            "learning_package": generated.learning_package.model_dump(mode="json"),
            "review_passed": True,
            "iterations": 1,
            "reviews": reviews.model_dump(mode="json"),
        }

    async def handle_question(self, session_id: str, question: str) -> Dict[str, Any]:
        stage = StageInfo.model_validate(self._require_stage(session_id))
        answer = await self.agent3.answer_question(question, stage)
        self.state_manager.add_qa_record(session_id, question, answer, True, True)
        self.state_manager.add_interaction(session_id, "question_asked", {"question": question})
        return {"answer": answer, "tech_pass": True, "quality_pass": True, "iterations": 1}

    async def submit_answer(self, session_id: str, question_id: str, user_answer: str) -> Dict[str, Any]:
        package = LearningPackage.model_validate(self._require_package(session_id))
        question = next((item for item in package.questions if item.question_id == question_id), None)
        if question is None:
            raise ValueError(f"Question not found in current stage: {question_id}")
        is_correct = self._normalize_answer(user_answer) == self._normalize_answer(question.correct_answer)
        self.state_manager.add_answer_record(session_id, question_id, user_answer, is_correct)
        return {"is_correct": is_correct, "correct_answer": question.correct_answer}

    async def complete_stage(self, session_id: str) -> Dict[str, Any]:
        profile = UserProfile.model_validate(self._require_profile(session_id))
        stage = StageInfo.model_validate(self._require_stage(session_id))
        accuracy = self.state_manager.calculate_accuracy(session_id)
        decision = await self.agent6.run(StageDecisionInput(
            session_id=session_id,
            profile=profile,
            stage=stage,
            accuracy=accuracy,
            qa_history=self.state_manager.get_qa_history(session_id),
            interactions=self.state_manager.get_interactions(session_id),
        ))
        stage.status = "completed" if accuracy >= 0.8 else "retreat"
        stage.accuracy = accuracy
        stage.strategy_used = decision.strategy
        self.state_manager.update_stage(session_id, stage)
        self.state_manager.update_profile(session_id, decision.profile)

        if decision.decision == "completed":
            self.state_manager.end_session(session_id)
            return {
                "status": "completed",
                "accuracy": accuracy,
                "updated_profile": decision.profile.model_dump(mode="json"),
                "final_report": self._generate_final_report(decision.profile, accuracy),
            }

        next_stage_num = stage.stage_num + 1 if decision.decision == "continue" else stage.stage_num
        next_stage = await self.agent2.run(StagePlanningInput(
            session_id=session_id,
            profile=decision.profile,
            stage_num=next_stage_num,
        ))
        self.state_manager.update_stage(session_id, next_stage.stage)
        return {
            "status": decision.decision,
            "accuracy": accuracy,
            "updated_profile": decision.profile.model_dump(mode="json"),
            "next_stage": next_stage.stage.model_dump(mode="json"),
        }

    async def _review(self, session_id: str, package: LearningPackage) -> ReviewOutput:
        request = ReviewInput(session_id=session_id, learning_package=package)
        technical = await self.agent4.run(request)
        quality = await self.agent5.run(request)
        return ReviewOutput(technical=technical, quality=quality)

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

    def _normalize_answer(self, answer: str) -> str:
        return "".join(answer.strip().lower().split())

    def _generate_final_report(self, profile: UserProfile, accuracy: float) -> Dict[str, Any]:
        return {
            "goal": profile.goal,
            "accuracy": accuracy,
            "mastered_skills": [gap.skill for gap in profile.knowledge_gaps if gap.status == "已掌握"],
            "profile_version": profile.version,
        }
