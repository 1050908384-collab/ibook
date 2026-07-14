from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from orchestrator.harness import Harness

router = APIRouter()
harness = Harness()

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
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-answer")
async def submit_answer(request: SubmitAnswerRequest):
    try:
        result = await harness.submit_answer(request.session_id, request.question_id, request.user_answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/complete/{session_id}")
async def complete_stage(session_id: str):
    try:
        result = await harness.complete_stage(session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
