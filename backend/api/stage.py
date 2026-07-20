from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_harness

router = APIRouter()
harness = get_harness()

class StartStageRequest(BaseModel):
    session_id: str

class SubmitAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    user_answer: str

@router.post("/start")
async def start_stage(request: StartStageRequest):
    try:
        result = await harness.start_stage(request.session_id)
        return result
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))

@router.post("/submit-answer")
async def submit_answer(request: SubmitAnswerRequest):
    try:
        result = await harness.submit_answer(request.session_id, request.question_id, request.user_answer)
        return result
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))

@router.post("/complete/{session_id}")
async def complete_stage(session_id: str):
    try:
        result = await harness.complete_stage(session_id)
        return result
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))
