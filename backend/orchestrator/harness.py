from typing import Dict, Any, Optional
import logging
from backend.agents import AGENTS
from backend.orchestrator.state import WorkflowState
from backend.orchestrator.termination import check_termination
from backend.orchestrator.exception_handler import ExceptionHandler
from backend.orchestrator.human_intervention import HumanIntervention
from backend.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentHarness:
    def __init__(self):
        self.agents = {agent_id: agent_cls() for agent_id, agent_cls in AGENTS.items()}
        self.exception_handler = ExceptionHandler()
        self.human_intervention = HumanIntervention()
    
    def run(self, session_id: str, user_input: str, 
            learner_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = WorkflowState(session_id)
        
        if learner_profile:
            state.learner_profile = learner_profile
        
        state.add_conversation("user", user_input)
        
        diagnostic_result = self._run_diagnostic(state, user_input)
        state.learner_profile = diagnostic_result.get("learner_profile")
        
        if not diagnostic_result.get("completed"):
            return {
                "status": "diagnostic_pending",
                "next_questions": diagnostic_result.get("next_questions"),
                "learner_profile": state.learner_profile
            }
        
        while True:
            retrieval_result = self._run_retrieval(state)
            state.knowledge_points = retrieval_result.get("knowledge_points", "")
            
            state.reset_debate_round()
            debate_result = self._run_debate_cycle(state)
            
            if debate_result.get("circuit_breaker_tripped"):
                return {
                    "status": "circuit_breaker",
                    "content": debate_result.get("content"),
                    "reason": debate_result.get("reason")
                }
            
            state.generated_content = debate_result.get("content", "")
            state.tech_review_result = debate_result.get("tech_review_result", {})
            
            quality_result = self._run_quality_review(state)
            state.quality_review_result = quality_result
            
            iteration_result = self._run_iteration(state)
            state.iteration_state.strategy = iteration_result.get("strategy")
            
            if iteration_result.get("terminated"):
                return {
                    "status": "completed",
                    "content": state.generated_content,
                    "learner_profile": state.learner_profile,
                    "iteration_round": state.iteration_state.iteration_round,
                    "accuracy": state.iteration_state.accuracy,
                    "coverage": state.iteration_state.coverage,
                    "reason": iteration_result.get("termination_reason")
                }
            
            state.increment_iteration_round()
            
            terminated, reason, _ = check_termination(state.iteration_state.model_dump())
            if terminated:
                return {
                    "status": "terminated",
                    "content": state.generated_content,
                    "reason": reason,
                    "iteration_round": state.iteration_state.iteration_round
                }
    
    def _run_diagnostic(self, state: WorkflowState, user_input: str) -> Dict[str, Any]:
        logger.info("Running diagnostic agent...")
        agent = self.agents["agent_01_diagnostic"]
        return self.exception_handler.handle(
            agent.process,
            {"user_input": user_input, "conversation_history": state.conversation_history}
        )
    
    def _run_retrieval(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Running retrieval agent...")
        agent = self.agents["agent_02_retrieval"]
        return self.exception_handler.handle(
            agent.process,
            {
                "query": state.knowledge_points or "LoRA微调",
                "learner_profile": state.learner_profile
            }
        )
    
    def _run_debate_cycle(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info(f"Starting debate cycle (max {settings.MAX_DEBATE_ROUNDS} rounds)...")
        
        generation_agent = self.agents["agent_03_generation"]
        review_agent = self.agents["agent_04_tech_review"]
        
        revision_instructions = ""
        
        while state.iteration_state.debate_round < settings.MAX_DEBATE_ROUNDS:
            logger.info(f"Debate round {state.iteration_state.debate_round + 1}")
            
            generation_result = self.exception_handler.handle(
                generation_agent.process,
                {
                    "knowledge_points": state.knowledge_points,
                    "learner_profile": state.learner_profile,
                    "revision_instructions": revision_instructions
                }
            )
            
            tech_review_result = self.exception_handler.handle(
                review_agent.process,
                {
                    "content": generation_result.get("content", ""),
                    "references": []
                }
            )
            
            state.increment_debate_round()
            
            if not tech_review_result.get("hallucination_detected", False):
                logger.info("Tech review passed, exiting debate cycle")
                return {
                    "content": generation_result.get("content", ""),
                    "tech_review_result": tech_review_result,
                    "debate_rounds": state.iteration_state.debate_round
                }
            
            logger.warning(f"Hallucination detected in round {state.iteration_state.debate_round}")
            revision_instructions = tech_review_result.get("suggestions", "")
        
        logger.warning("Debate rounds exhausted, triggering circuit breaker")
        return {
            "content": generation_result.get("content", ""),
            "tech_review_result": tech_review_result,
            "circuit_breaker_tripped": True,
            "reason": f"辩论{settings.MAX_DEBATE_ROUNDS}轮后仍存在幻觉，接受当前版本"
        }
    
    def _run_quality_review(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Running quality review agent...")
        agent = self.agents["agent_05_quality_review"]
        return self.exception_handler.handle(
            agent.process,
            {
                "content": state.generated_content,
                "learner_profile": state.learner_profile
            }
        )
    
    def _run_iteration(self, state: WorkflowState) -> Dict[str, Any]:
        logger.info("Running iteration agent...")
        agent = self.agents["agent_06_iteration"]
        return self.exception_handler.handle(
            agent.process,
            {
                "accuracy": state.iteration_state.accuracy,
                "coverage": state.iteration_state.coverage,
                "learner_profile": state.learner_profile,
                "tech_review_result": state.tech_review_result,
                "quality_review_result": state.quality_review_result,
                "iteration_round": state.iteration_state.iteration_round,
                "last_improvement": state.iteration_state.last_improvement
            }
        )