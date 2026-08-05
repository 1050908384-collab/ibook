# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_harness
from db.sqlite_client import DEFAULT_USER_ID

router = APIRouter(tags=["User Scope"])
harness = get_harness()


class UserProfileUpdate(BaseModel):
    grade: Optional[str] = None
    skill_tags: Optional[list[str]] = None
    learning_style: Optional[str] = None
    hardware: Optional[str] = None
    default_daily_hours: Optional[float] = Field(default=None, gt=0, le=24)
    learning_rhythm: Optional[str] = None


class KnowledgeStatusUpdate(BaseModel):
    status: int = Field(ge=-1, le=1)
    mastery_score: float = Field(ge=0, le=1)
    is_initial_foundation: Optional[bool] = None
    attempt_count: Optional[int] = Field(default=None, ge=0)


class TargetSettingsUpdate(BaseModel):
    goal: Optional[str] = None
    target_total_hours: Optional[float] = Field(default=None, ge=0)
    target_daily_hours: Optional[float] = Field(default=None, gt=0, le=24)
    target_progress: Optional[float] = Field(default=None, ge=0, le=1)


class StageDurationUpdate(BaseModel):
    estimated_hours: float = Field(gt=0, le=1000)


def _require_local_user(user_id: str) -> None:
    if user_id != DEFAULT_USER_ID:
        raise HTTPException(status_code=404, detail="User not found")


@router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    _require_local_user(user_id)
    profile = harness.state_manager.db_client.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return profile


@router.patch("/users/{user_id}/profile")
async def update_user_profile(user_id: str, request: UserProfileUpdate):
    _require_local_user(user_id)
    updates = request.model_dump(exclude_none=True)
    existing = harness.state_manager.db_client.get_user_profile(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="User profile not found")
    profile = harness.state_manager.db_client.upsert_user_profile(user_id, updates)
    harness.state_manager.refresh_cached_user_profile(user_id)
    return profile


@router.get("/users/{user_id}/knowledge-network")
async def get_knowledge_network(user_id: str):
    _require_local_user(user_id)
    rows = harness.state_manager.db_client.list_user_subtopic_statuses(user_id)
    blocks: dict[str, dict] = {}
    for row in rows:
        block = blocks.setdefault(row["block_id"], {
            "block_id": row["block_id"],
            "subtopics": [],
            "mastered_count": 0,
        })
        block["subtopics"].append(row)
        if int(row["status"]) == 1:
            block["mastered_count"] += 1
    return {
        "user_id": user_id,
        "total_subtopics": len(rows),
        "mastered_subtopics": sum(1 for row in rows if int(row["status"]) == 1),
        "blocks": list(blocks.values()),
    }


@router.patch("/users/{user_id}/knowledge-network/{subtopic_id}")
async def update_knowledge_status(user_id: str, subtopic_id: str, request: KnowledgeStatusUpdate):
    _require_local_user(user_id)
    try:
        return harness.state_manager.db_client.set_user_subtopic_status(
            user_id,
            subtopic_id,
            **request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/target-settings")
async def get_target_settings(session_id: str):
    target = harness.state_manager.db_client.get_target_settings(session_id)
    if not target:
        raise HTTPException(status_code=404, detail="Session not found")
    return target


@router.patch("/sessions/{session_id}/target-settings")
async def update_target_settings(session_id: str, request: TargetSettingsUpdate):
    try:
        target = harness.state_manager.db_client.update_target_settings(
            session_id, request.model_dump(exclude_none=True)
        )
        harness.state_manager.refresh_cached_target(session_id)
        return target
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/stages/{stage_id}/duration")
async def get_stage_duration(session_id: str, stage_id: str):
    stage = harness.state_manager.db_client.get_stage_duration(session_id, stage_id)
    if not stage:
        raise HTTPException(status_code=404, detail="Stage not found")
    return stage


@router.patch("/sessions/{session_id}/stages/{stage_id}/duration")
async def update_stage_duration(session_id: str, stage_id: str, request: StageDurationUpdate):
    try:
        stage = harness.state_manager.db_client.update_stage_duration(
            session_id, stage_id, request.estimated_hours
        )
        harness.state_manager.refresh_cached_stage_duration(session_id, stage_id, request.estimated_hours)
        return stage
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
