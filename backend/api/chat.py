from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from orchestrator.harness import Harness

router = APIRouter()
harness = Harness()

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: dict

@router.post("/")
async def chat(request: ChatRequest):
    try:
        if not request.session_id:
            request.session_id = harness.state_manager.create_session(None)["session_id"]
        
        result = await harness.process_initial_input(request.session_id, request.message)
        return {"session_id": request.session_id, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/question")
async def handle_question(request: ChatRequest):
    try:
        result = await harness.handle_question(request.session_id, request.message)
        return {"session_id": request.session_id, "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
