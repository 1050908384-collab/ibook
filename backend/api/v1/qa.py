"""Learning Q&A endpoints backed by the existing 个性化学习资源生成 Agent -> 专业知识与教学质量审核 Agent harness flow."""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_harness, require_owned_session

router = APIRouter(prefix="/sessions/{session_id}/qa", tags=["Q&A"], dependencies=[Depends(require_owned_session)])
harness = get_harness()


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    stage_num: Optional[int] = Field(default=None, ge=1)


class QAResponse(BaseModel):
    qa_id: str
    session_id: str
    question: str
    answer: str
    stage_id: Optional[str] = None
    review_summary: dict[str, Any] = Field(default_factory=dict)
    source_summary: list[dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[str] = None


class QAHistoryResponse(BaseModel):
    items: list[QAResponse]


@router.post("", response_model=QAResponse)
async def ask_question(session_id: str, request: QARequest):
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        result = await harness.handle_question(session_id, request.question, request.stage_num)
        session = harness.get_session(session_id) or {}
        current_stage = session.get("stage_info") or {}
        requested_stage_num = request.stage_num if request.stage_num is not None else current_stage.get("stage_num")
        snapshot = harness.state_manager.get_stage_snapshot(session_id, requested_stage_num)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Stage not found")
        history = snapshot.get("qa_history", [])
        return {
            "qa_id": f"qa_{len(history)}",
            "session_id": session_id,
            "question": request.question,
            "answer": result["answer"],
            "stage_id": (snapshot.get("stage_info") or {}).get("stage_id"),
            "review_summary": result.get("review_summary", {}),
            "source_summary": [],
            "created_at": session.get("updated_at"),
        }
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("", response_model=QAHistoryResponse)
async def get_qa_history(session_id: str, stage_num: Optional[int] = None):
    session = harness.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    current_stage = session.get("stage_info") or {}
    requested_stage_num = stage_num if stage_num is not None else current_stage.get("stage_num")
    snapshot = harness.state_manager.get_stage_snapshot(session_id, requested_stage_num)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Stage not found")
    stage_info = snapshot.get("stage_info") or {}
    stage_id = stage_info.get("stage_id")
    items = []
    for index, record in enumerate(snapshot.get("qa_history", []), start=1):
        question = record.get("question", "")
        answer = record.get("answer", "")
        if harness._is_greeting(question):
            answer = harness._greeting_answer(stage_info.get("block_name", ""))
        else:
            answer = harness._extract_user_facing_answer(answer)
        items.append({
            "qa_id": f"qa_{index}",
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "stage_id": stage_id,
            "review_summary": {
                "technical_passed": record.get("tech_pass", True),
                "quality_passed": record.get("quality_pass", True),
            },
            "source_summary": [],
            "created_at": session.get("updated_at"),
        })
    return {"items": items}
