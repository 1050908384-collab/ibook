from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from models.profile import UserProfile
from models.stage import LearningPackage, StageInfo

class QARecord(BaseModel):
    qa_id: str
    session_id: str
    stage: int
    question: str
    answer: str
    tech_pass: bool = True
    quality_pass: bool = True

class AnswerRecord(BaseModel):
    answer_id: str
    session_id: str
    stage: int
    question_id: str
    user_answer: str
    is_correct: bool
    time_spent: int = 0

class SessionData(BaseModel):
    session_id: str
    current_stage: int = 1
    status: Literal["active", "completed", "abandoned"] = "active"
    profile: Optional[UserProfile] = None
    stage_info: Optional[StageInfo] = None
    learning_package: Optional[LearningPackage] = None
    qa_history: List[Dict[str, Any]] = Field(default_factory=list)
    answer_records: List[Dict[str, Any]] = Field(default_factory=list)
    interactions: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
