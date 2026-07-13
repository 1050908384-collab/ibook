from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional

router = APIRouter()


class FeedbackRequest(BaseModel):
    session_id: str
    user_id: str
    content_id: str
    rating: int
    comments: Optional[str] = None
    timestamp: Optional[str] = None


@router.post("/")
async def submit_feedback(request: FeedbackRequest):
    return {
        "message": "Feedback received",
        "data": request.model_dump()
    }


@router.get("/{session_id}")
async def get_feedback(session_id: str):
    return {"session_id": session_id, "feedbacks": []}