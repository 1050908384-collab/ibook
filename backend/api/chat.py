from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel, Field

from api.dependencies import get_harness

router = APIRouter()
harness = get_harness()

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    response: dict

@router.post("/")
async def chat(request: ChatRequest):
    try:
        if not request.session_id:
            request.session_id = harness.state_manager.create_session(None)["session_id"]
        print(f"[API][POST /api/chat] session_id={request.session_id} | message={request.message[:60]}", flush=True)
        
        result = await harness.process_initial_input(request.session_id, request.message)
        return {
            "session_id": request.session_id,
            "response": result,
            "state": harness.get_runtime_state(request.session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))

@router.post("/question")
async def handle_question(request: ChatRequest):
    try:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        print(f"[API][POST /api/chat/question] session_id={request.session_id} | message={request.message[:60]}", flush=True)
        result = await harness.handle_question(request.session_id, request.message)
        return {
            "session_id": request.session_id,
            "response": result,
            "state": harness.get_runtime_state(request.session_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))
