from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional, List, Any

router = APIRouter()


class SubmitRequest(BaseModel):
    session_id: str
    user_id: str
    question_id: str
    answers: Dict[str, Any]
    code_submission: Optional[str] = None
    timestamp: Optional[str] = None


class SubmitResponse(BaseModel):
    session_id: str
    question_id: str
    is_correct: bool
    score: float
    feedback: Optional[str] = None
    correct_answer: Optional[str] = None


@router.post("/test", response_model=SubmitResponse)
async def submit_test(request: SubmitRequest):
    return SubmitResponse(
        session_id=request.session_id,
        question_id=request.question_id,
        is_correct=True,
        score=0.8,
        feedback="Good job!",
        correct_answer=None
    )


@router.post("/code")
async def submit_code(request: SubmitRequest):
    return {
        "session_id": request.session_id,
        "question_id": request.question_id,
        "executed": True,
        "output": "Hello, World!",
        "passed": True,
        "score": 1.0
    }