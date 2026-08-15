# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_current_user_id, get_harness

router = APIRouter(tags=["User Scope"])
harness = get_harness()


class UserProfileUpdate(BaseModel):
    grade: Optional[str] = None
    skill_tags: Optional[list[str]] = None
    foundation_confirmed: Optional[bool] = None
    learning_style: Optional[str] = None
    hardware: Optional[str] = None
    default_daily_hours: Optional[float] = Field(default=None, gt=0, le=24)
    learning_rhythm: Optional[str] = None
    mastered_subtopic_ids: Optional[list[str]] = None


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


def _require_user(user_id: str, current_user_id: str) -> None:
    if user_id != current_user_id:
        raise HTTPException(status_code=404, detail="User not found")


def _require_owned_session(session_id: str, user_id: str) -> None:
    try:
        owner_id = harness.state_manager.db_client.get_session_user_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    if owner_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str, current_user_id: str = Depends(get_current_user_id)):
    _require_user(user_id, current_user_id)
    profile = harness.state_manager.db_client.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    return profile


@router.patch("/users/{user_id}/profile")
async def update_user_profile(user_id: str, request: UserProfileUpdate, current_user_id: str = Depends(get_current_user_id)):
    _require_user(user_id, current_user_id)
    updates = request.model_dump(exclude_none=True)
    # 不再拒绝已确认 foundation 的 PATCH：upsert_user_profile 使用 ON CONFLICT DO UPDATE，
    # 是安全的覆盖语义；之前的 409 拦截过严，导致用户重新选择基础知识点、
    # 或其他人在继承数据库的机器上重新提交 profile 时被永久拒绝（前端反复重试 409）。
    mastered_subtopic_ids = updates.pop("mastered_subtopic_ids", None)
    try:
        profile = harness.state_manager.db_client.upsert_user_profile(user_id, updates)
        if mastered_subtopic_ids is not None:
            harness.state_manager.db_client.replace_user_initial_foundation(user_id, mastered_subtopic_ids)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    harness.state_manager.refresh_cached_user_profile(user_id)
    return profile


@router.get("/users/{user_id}/knowledge-network")
async def get_knowledge_network(user_id: str, current_user_id: str = Depends(get_current_user_id)):
    _require_user(user_id, current_user_id)
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
async def update_knowledge_status(user_id: str, subtopic_id: str, request: KnowledgeStatusUpdate, current_user_id: str = Depends(get_current_user_id)):
    _require_user(user_id, current_user_id)
    try:
        return harness.state_manager.db_client.set_user_subtopic_status(
            user_id,
            subtopic_id,
            **request.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/target-settings")
async def get_target_settings(session_id: str, user_id: str = Depends(get_current_user_id)):
    _require_owned_session(session_id, user_id)
    target = harness.state_manager.db_client.get_target_settings(session_id)
    if not target:
        raise HTTPException(status_code=404, detail="Session not found")
    return target


@router.patch("/sessions/{session_id}/target-settings")
async def update_target_settings(session_id: str, request: TargetSettingsUpdate, user_id: str = Depends(get_current_user_id)):
    _require_owned_session(session_id, user_id)
    try:
        target = harness.state_manager.db_client.update_target_settings(
            session_id, request.model_dump(exclude_none=True)
        )
        harness.state_manager.refresh_cached_target(session_id)
        return target
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/stages/{stage_id}/duration")
async def get_stage_duration(session_id: str, stage_id: str, user_id: str = Depends(get_current_user_id)):
    _require_owned_session(session_id, user_id)
    stage = harness.state_manager.db_client.get_stage_duration(session_id, stage_id)
    if not stage:
        raise HTTPException(status_code=404, detail="Stage not found")
    return stage


@router.patch("/sessions/{session_id}/stages/{stage_id}/duration")
async def update_stage_duration(session_id: str, stage_id: str, request: StageDurationUpdate, user_id: str = Depends(get_current_user_id)):
    _require_owned_session(session_id, user_id)
    try:
        stage = harness.state_manager.db_client.update_stage_duration(
            session_id, stage_id, request.estimated_hours
        )
        harness.state_manager.refresh_cached_stage_duration(session_id, stage_id, request.estimated_hours)
        return stage
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# === 知识板块/知识点查询（分步诊断用） ===

@router.get("/knowledge-blocks/{block_id}/subtopics")
async def get_block_subtopics(block_id: str):
    """
    获取指定知识板块下的所有知识点。

    GET /api/v1/knowledge-blocks/{block_id}/subtopics

    前端流程第 2 步：用户选择已掌握的板块后，
    调用此接口获取该板块下的知识点列表，供用户选择。
    """
    from tools.agent1_tools.knowledge_catalog_tool import KnowledgeCatalogTool
    catalog = KnowledgeCatalogTool()
    block = catalog.get_block(block_id)
    if not block:
        raise HTTPException(status_code=404, detail=f"Knowledge block '{block_id}' not found")
    return {
        "block_id": block_id,
        "block_name": block.get("block_name", block_id),
        "subtopics": block.get("subtopics", []),
    }
