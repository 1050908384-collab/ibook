from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uuid

router = APIRouter()

sessions = {}


class SessionCreate(BaseModel):
    user_id: str
    user_name: str


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    user_name: str
    created_at: str


@router.post("/", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user_id": request.user_id,
        "user_name": request.user_name,
        "created_at": str(uuid.uuid1().time)
    }
    return SessionResponse(
        session_id=session_id,
        user_id=request.user_id,
        user_name=request.user_name,
        created_at=sessions[session_id]["created_at"]
    )


@router.get("/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"message": "Session deleted successfully"}


@router.get("/")
async def list_sessions():
    return list(sessions.values())