from pydantic import BaseModel
from typing import List, Optional

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
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
