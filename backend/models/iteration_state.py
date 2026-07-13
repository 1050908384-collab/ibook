from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from enum import Enum


class IterationStrategy(str, Enum):
    ADVANCE = "advance"
    RETREAT = "retreat"
    REINFORCE = "reinforce"
    CORRECT = "correct"
    TERMINATE = "terminate"


class IterationState(BaseModel):
    session_id: str
    iteration_round: int = 0
    debate_round: int = 0
    accuracy: float = 0.0
    coverage: float = 0.0
    strategy: Optional[IterationStrategy] = None
    last_improvement: float = 0.0
    stagnant_count: int = 0
    terminated: bool = False
    termination_reason: Optional[str] = None
    history: List[Dict[str, Any]] = []
    
    class Config:
        use_enum_values = True