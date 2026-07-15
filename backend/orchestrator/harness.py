from typing import Dict, Any, Optional
from agents.agent_01_diagnostic import DiagnosticAgent
from agents.agent_02_stage_planner import StagePlannerAgent
from agents.agent_03_content_gen import ContentGeneratorAgent
from agents.agent_04_tech_review import TechnicalReviewAgent
from agents.agent_05_quality_review import QualityReviewAgent
from agents.agent_06_stage_decision import StageDecisionAgent
from orchestrator.state_manager import StateManager
from config import settings

class Harness:
    def __init__(self):
        self.state_manager = StateManager()
        self.agent1 = DiagnosticAgent()
        self.agent2 = StagePlannerAgent()
        self.agent3 = ContentGeneratorAgent()
        self.agent4 = TechnicalReviewAgent()
        self.agent5 = QualityReviewAgent()
        self.agent6 = StageDecisionAgent()

    async def process_initial_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        raw_data = self._extract_raw_data(user_input)
        profile = await self.agent1.generate_profile(session_id, raw_data)
        
        if profile["confidence"] < settings.CONFIDENCE_THRESHOLD:
            return {"status": "needs_more_info", "questions": self._generate_followup_questions(profile)}
        
        self.state_manager.update_profile(session_id, profile)
        stage_info = await self.agent2.generate_stage(session_id, profile)
        self.state_manager.update_stage(session_id, stage_info)
        
        return {
            "status": "ready",
            "profile": profile,
            "stage_info": stage_info
        }

    async def start_stage(self, session_id: str) -> Dict[str, Any]:
        stage_info = self.state_manager.get_current_stage(session_id)
        learning_package = await self.agent3.generate_learning_package(session_id, stage_info)
        
        tech_errors, quality_errors = await self._parallel_review(session_id, learning_package)
        
        for iteration in range(settings.MAX_REVIEW_ITERATIONS):
            if not tech_errors and not quality_errors:
                break
            combined_errors = {"tech": tech_errors, "quality": quality_errors}
            learning_package = await self.agent3.modify_learning_package(session_id, learning_package, combined_errors)
            tech_errors, quality_errors = await self._parallel_review(session_id, learning_package)
        
        self.state_manager.update_learning_package(session_id, learning_package)
        
        return {
            "learning_package": learning_package,
            "review_passed": not tech_errors and not quality_errors,
            "iterations": iteration + 1
        }

    async def handle_question(self, session_id: str, question: str) -> Dict[str, Any]:
        answer = await self.agent3.answer_question(session_id, question)
        
        tech_errors, quality_errors = await self._parallel_review(session_id, {"answer": answer})
        
        if tech_errors or quality_errors:
            combined_errors = {"tech": tech_errors, "quality": quality_errors}
            answer = await self.agent3.modify_answer(session_id, answer, combined_errors)
        
        self.state_manager.add_qa_record(session_id, question, answer)
        
        return {
            "answer": answer,
            "tech_pass": not tech_errors,
            "quality_pass": not quality_errors
        }

    async def submit_answer(self, session_id: str, question_id: str, user_answer: str) -> Dict[str, Any]:
        is_correct = self._evaluate_answer(session_id, question_id, user_answer)
        self.state_manager.add_answer_record(session_id, question_id, user_answer, is_correct)
        
        return {"is_correct": is_correct}

    async def complete_stage(self, session_id: str) -> Dict[str, Any]:
        accuracy = self.state_manager.calculate_accuracy(session_id)
        profile = self.state_manager.get_profile(session_id)
        qa_history = self.state_manager.get_qa_history(session_id)
        interactions = self.state_manager.get_interactions(session_id)
        
        updated_data = await self.agent6.update_user_data(session_id, accuracy, profile, qa_history, interactions)
        new_profile = await self.agent1.generate_profile(session_id, updated_data)
        
        self.state_manager.update_profile(session_id, new_profile)
        
        if self._all_gaps_mastered(new_profile):
            return {
                "status": "completed",
                "final_report": self._generate_final_report(session_id)
            }
        
        new_stage = await self.agent2.generate_stage(session_id, new_profile)
        self.state_manager.update_stage(session_id, new_stage)
        
        return {
            "status": "continue",
            "accuracy": accuracy,
            "updated_profile": new_profile,
            "next_stage": new_stage
        }

    async def _parallel_review(self, session_id: str, content: Dict[str, Any]) -> tuple:
        tech_errors = await self.agent4.review(session_id, content)
        quality_errors = await self.agent5.review(session_id, content)
        return tech_errors, quality_errors

    def _extract_raw_data(self, user_input: str) -> Dict[str, Any]:
        return {"user_input": user_input}

    def _generate_followup_questions(self, profile: Dict[str, Any]) -> list:
        return []

    def _evaluate_answer(self, session_id: str, question_id: str, user_answer: str) -> bool:
        return False

    def _all_gaps_mastered(self, profile: Dict[str, Any]) -> bool:
        return False

    def _generate_final_report(self, session_id: str) -> Dict[str, Any]:
        return {}
