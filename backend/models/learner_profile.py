from pydantic import BaseModel
from typing import Dict, List, Optional
from enum import Enum


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearnerProfile(BaseModel):
    user_id: str
    name: str
    major: str
    programming_experience: str
    background: Dict[str, SkillLevel] = {}
    knowledge_gaps: List[str] = []
    learning_goals: List[str] = []
    available_time: str
    preferred_learning_style: str = "hands_on"
    environment: Dict[str, str] = {}
    progress: Dict[str, float] = {}
    
    class Config:
        use_enum_values = True