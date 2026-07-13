from typing import Dict, Any, List, Optional
from backend.models.iteration_state import IterationState
from backend.models.learner_profile import LearnerProfile


class WorkflowState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.iteration_state = IterationState(session_id=session_id)
        self.learner_profile: Optional[LearnerProfile] = None
        self.knowledge_points: str = ""
        self.generated_content: str = ""
        self.tech_review_result: Dict[str, Any] = {}
        self.quality_review_result: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.current_step: str = "diagnostic"
        self.hallucination_count: int = 0
    
    def reset_debate_round(self):
        self.iteration_state.debate_round = 0
    
    def increment_debate_round(self):
        self.iteration_state.debate_round += 1
    
    def increment_iteration_round(self):
        self.iteration_state.iteration_round += 1
    
    def update_accuracy(self, accuracy: float):
        self.iteration_state.last_improvement = accuracy - self.iteration_state.accuracy
        self.iteration_state.accuracy = accuracy
        
        if self.iteration_state.last_improvement < 0.03:
            self.iteration_state.stagnant_count += 1
        else:
            self.iteration_state.stagnant_count = 0
    
    def update_coverage(self, coverage: float):
        self.iteration_state.coverage = coverage
    
    def add_conversation(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "iteration_state": self.iteration_state.model_dump(),
            "learner_profile": self.learner_profile.model_dump() if self.learner_profile else None,
            "knowledge_points": self.knowledge_points,
            "generated_content": self.generated_content,
            "tech_review_result": self.tech_review_result,
            "quality_review_result": self.quality_review_result,
            "current_step": self.current_step,
            "hallucination_count": self.hallucination_count
        }