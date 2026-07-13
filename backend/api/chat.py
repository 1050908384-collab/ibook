from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from backend.orchestrator.harness import AgentHarness

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    user_input: str
    learner_profile: Optional[Dict] = None


class ChatResponse(BaseModel):
    status: str
    content: Optional[str] = None
    next_questions: Optional[list] = None
    learner_profile: Optional[Dict] = None
    iteration_round: Optional[int] = None
    accuracy: Optional[float] = None
    coverage: Optional[float] = None
    reason: Optional[str] = None


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        harness = AgentHarness()
        result = harness.run(
            session_id=request.session_id,
            user_input=request.user_input,
            learner_profile=request.learner_profile
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))