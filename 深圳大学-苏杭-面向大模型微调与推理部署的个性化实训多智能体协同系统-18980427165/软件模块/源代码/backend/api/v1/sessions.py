# -*- coding: utf-8 -*-
"""
v1/sessions.py - 学情诊断与阶段规划 Agent/3 前端 API：会话管理接口
================================================
对应文档：学情诊断与阶段规划 Agent3_API接口任务说明.md 第 1 节
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.dependencies import get_current_user_id, get_harness
from api.v1.learning_resources import backfill_stage_tutorial, build_stage_document_resources
from api.v1.schemas import PublicLearningPackage, to_public_learning_package
from tools.learning_resource_tool import LearningResourceTool

router = APIRouter(prefix="/sessions", tags=["Sessions"])
harness = get_harness()


def _require_owned_session(session_id: str, user_id: str) -> None:
    try:
        owner_id = harness.state_manager.db_client.get_session_user_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    if owner_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str
    title: str = ""
    status: str
    phase: str
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionSnapshot(BaseModel):
    session_id: str
    title: str = ""
    status: str
    phase: str
    profile: Optional[dict[str, Any]] = None
    profile_modified: bool = False
    current_stage: Optional[dict[str, Any]] = None
    next_stage: Optional[dict[str, Any]] = None
    learning_package: Optional[PublicLearningPackage] = None
    assessment_summary: dict[str, Any] = Field(default_factory=dict)
    progress_summary: dict[str, Any] = Field(default_factory=dict)
    recent_qa: list[dict[str, Any]] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: str = "local-user"
    user_profile_exists: bool = False
    requires_full_profile: bool = True
    target_settings: dict[str, Any] = Field(default_factory=dict)


async def _get_snapshot_resources(package: Optional[dict[str, Any]], stage: Optional[dict[str, Any]]) -> Optional[list[dict[str, Any]]]:
    """Backfill document and video resources for legacy packages."""
    if not package or not stage:
        return None

    resources = list(package.get("learning_resources") or [])
    known_ids = {item.get("resource_id") for item in resources if isinstance(item, dict)}
    for resource in build_stage_document_resources(stage):
        if resource.get("resource_id") not in known_ids:
            resources.append(resource)
            known_ids.add(resource.get("resource_id"))

    if package.get("recommended_videos"):
        return resources

    block_id = stage.get("block_id")
    if not block_id:
        return resources or None
    videos = await LearningResourceTool().get_recommended_videos(block_id, stage.get("subtopic_id"))
    resources.extend([
        {
            "resource_id": video["resource_id"],
            "resource_type": "video",
            "title": video["title"],
            "url": video["url"],
            "platform": "bilibili" if "bilibili.com" in video["url"] else "",
            "description": video.get("content_summary", ""),
            "recommendation_level": video.get("role", "supplementary"),
            "role": video.get("role", "supplementary"),
            "order": video.get("sequence_index", 1),
            "subtopic_id": video.get("subtopic_id") or "",
        }
        for video in videos
    ])
    return resources


@router.post("", response_model=SessionSnapshot)
async def create_session(user_id: str = Depends(get_current_user_id)):
    """
    创建新会话
    
    POST /api/v1/sessions
    """
    try:
        session = harness.create_session(None, user_id)
        user_profile = harness.state_manager.db_client.get_user_profile(user_id)
        target_settings = harness.state_manager.db_client.get_target_settings(session["session_id"]) or {}
        return {
            "session_id": session["session_id"],
            "title": session.get("title") or "",
            "status": session.get("status", "active"),
            "phase": session.get("phase", "diagnosing"),
            "profile": None,
            "current_stage": None,
            "next_stage": None,
            "learning_package": None,
            "assessment_summary": {},
            "progress_summary": {},
            "recent_qa": [],
            "available_actions": ["submit_diagnosis"],
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "user_id": user_id,
            "user_profile_exists": bool(user_profile),
            "requires_full_profile": not bool(user_profile),
            "target_settings": target_settings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=SessionListResponse)
async def list_sessions(user_id: str = Depends(get_current_user_id)):
    """
    获取所有会话列表（左侧对话栏 + 账号级"所有学习画像/所有项目"列表）
    
    GET /api/v1/sessions
    
    返回字段（文档要求 items[]）：
    session_id, title, goal, grade, skill_tags[], knowledge_gaps[],
    daily_hours, learning_style, version, status, phase,
    current_stage_num, current_stage_name, progress, updated_at
    
    其中 profile_summary 嵌套对象包含画像摘要信息。
    """
    try:
        # 使用 harness 的 state_manager 确保一致性
        sm = harness.state_manager
        sessions = sm.list_sessions(user_id)
        items = []
        for s in sessions:
            session_id = s.get("session_id")
            # Old imports may contain incomplete rows.  They cannot form a
            # usable API session and must not invalidate the complete list.
            if not isinstance(session_id, str) or not session_id.strip():
                continue
            ps = s.get("profile_summary", {}) or {}
            items.append({
                "session_id": session_id,
                "title": s.get("title") or "",
                # profile_summary 字段（按文档要求平铺到 items[]）
                "goal": ps.get("goal", ""),
                "grade": ps.get("grade", ""),
                "skill_tags": ps.get("skill_tags", []),
                "knowledge_gaps": ps.get("knowledge_gaps", []),
                "daily_hours": ps.get("daily_hours"),
                "target_total_hours": ps.get("target_total_hours", 0),
                "target_daily_hours": ps.get("target_daily_hours", ps.get("daily_hours")),
                "learning_style": ps.get("learning_style", ""),
                "version": ps.get("version", 1),
                "status": s.get("status") or "active",
                "phase": s.get("phase") or "diagnosing",
                "current_stage_num": ps.get("current_stage_num", 0),
                "current_stage_name": ps.get("current_stage_name", ""),
                "progress": ps.get("progress", 0),
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                # 保留 profile_summary 嵌套对象，便于前端按需读取
                "profile_summary": ps,
            })
        return {"sessions": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}", response_model=SessionSnapshot)
async def get_session_snapshot(session_id: str, user_id: str = Depends(get_current_user_id)):
    """
    获取完整会话快照（页面恢复）
    
    GET /api/v1/sessions/{session_id}
    """
    try:
        _require_owned_session(session_id, user_id)
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 构建完整快照
        stage = session.get("stage_info")
        learning_package = session.get("learning_package")

        # 修复旧会话：从数据库补全 subtopic_ids/subtopic_names 数组
        # GET /sessions/{session_id} 是页面恢复时调用的第一个接口，
        # 如果不在这里补全，前端 resolveStageSubtopics 会拿不到分页数据，
        # 导致所有知识点内容堆在一页里、分页控件消失。
        if stage:
            from api.v1.learning_resources import ensure_stage_subtopics
            stage = ensure_stage_subtopics(stage)

        snapshot_resources = await _get_snapshot_resources(learning_package, stage)
        snapshot = {
            "session_id": session["session_id"],
            "title": session.get("title") or "",
            "status": session.get("status") or "active",
            "phase": session.get("phase") or "diagnosing",
            "profile": session.get("profile"),
            "profile_modified": session.get("profile_modified", False),
            "current_stage": stage,
            "next_stage": session.get("next_stage"),
            "learning_package": to_public_learning_package(
                backfill_stage_tutorial(learning_package, stage) if learning_package and stage else learning_package,
                stage,
                snapshot_resources,
            ),
            "assessment_summary": {
                "answer_records": session.get("answer_records", []),
                "current_stage": session.get("current_stage", 1),
            },
            "progress_summary": {
                "total_stages": 1 if session.get("stage_info") is not None else 0,
                "current_stage": session.get("current_stage", 1),
                "qa_count": len(session.get("qa_history", [])),
            },
            "recent_qa": session.get("qa_history", [])[-5:] if session.get("qa_history") else [],
            "available_actions": session.get("available_actions", []),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "user_id": user_id,
            "user_profile_exists": bool(harness.state_manager.db_client.get_user_profile(user_id)),
            "requires_full_profile": not bool(harness.state_manager.db_client.get_user_profile(user_id)),
            "target_settings": harness.state_manager.db_client.get_target_settings(session_id) or {},
        }
        
        # 确保个性化字段在 current_stage 中
        if snapshot["current_stage"]:
            stage = snapshot["current_stage"]
            # 个性化字段已在 StageInfo 模型中定义，会自动包含
            # recommended_daily_hours, suggested_study_days, learning_style_notes,
            # hardware_notes, personalization_summary
        
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/abandon")
async def abandon_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    """
    放弃当前学习

    POST /api/v1/sessions/{session_id}/abandon
    """
    try:
        # 先检查会话是否存在，给出明确的 404
        _require_owned_session(session_id, user_id)
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        harness.abandon_session(session_id)
        return {
            "session_id": session_id,
            "status": "abandoned",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    """物理删除一个 session（不可逆）：删除 DB 中该 session 的全部快照、画像、
    阶段进度、测试记录、审核链路、问答与状态缓存。

    DELETE /api/v1/sessions/{session_id}
    """
    try:
        # 所有权校验：防止跨用户删除（db_client.delete_session 也会再验一次）
        _require_owned_session(session_id, user_id)
        deletes = harness.state_manager.delete_session(session_id, user_id)
        return {
            "session_id": session_id,
            "deleted": True,
            "details": deletes,
        }
    except HTTPException:
        raise
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
