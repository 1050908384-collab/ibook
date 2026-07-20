from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_harness

router = APIRouter()
harness = get_harness()

class SessionResponse(BaseModel):
    session_id: str
    status: str

@router.post("/")
async def create_session():
    try:
        session = harness.state_manager.create_session(None)
        return {"session_id": session["session_id"], "status": "active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}")
async def get_session(session_id: str):
    try:
        session = harness.state_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{session_id}/end")
async def end_session(session_id: str):
    try:
        harness.state_manager.end_session(session_id)
        return {"session_id": session_id, "status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{session_id}/abandon")
async def abandon_session(session_id: str):
    try:
        harness.state_manager.abandon_session(session_id)
        return {"session_id": session_id, "status": "abandoned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
