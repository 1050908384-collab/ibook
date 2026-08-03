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
    title: Optional[str] = None
    current_stage: int = 1
    status: Literal["active", "completed", "abandoned"] = "active"
    phase: str = "diagnosing"
    profile: Optional[UserProfile] = None
    profile_confirmed: bool = False
    profile_modified: bool = False
    stage_info: Optional[StageInfo] = None
    learning_package: Optional[LearningPackage] = None
    next_stage: Optional[StageInfo] = None
    next_stage_planning_notes: Optional[str] = None
    qa_history: List[Dict[str, Any]] = Field(default_factory=list)
    answer_records: List[Dict[str, Any]] = Field(default_factory=list)
    interactions: List[Dict[str, Any]] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
