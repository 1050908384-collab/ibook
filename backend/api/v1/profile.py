# -*- coding: utf-8 -*-
"""
v1/profile.py - Agent1/3 前端 API：用户画像接口
================================================
对应文档：Agent13_API接口任务说明.md 第 2 节（画像相关）
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.dependencies import get_harness, require_owned_session
from api.v1.schemas import ProfileResponse, ProfileUpdateResponse

router = APIRouter(prefix="/sessions/{session_id}", tags=["Profile"], dependencies=[Depends(require_owned_session)])
harness = get_harness()


class ProfileConfirmRequest(BaseModel):
    confirm: bool
    notes: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    goal: Optional[str] = None
    daily_hours: Optional[float] = None
    learning_style: Optional[str] = None
    hardware: Optional[str] = None


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(session_id: str):
    """
    获取用户画像
    
    GET /api/v1/sessions/{session_id}/profile
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        profile = session.get("profile")
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        practice_profile = session.get("practice_profile") or {}
        return {
            "profile_id": f"profile_{session_id[:8]}",
            "session_id": session_id,
            "goal": profile.get("goal", ""),
            "grade": profile.get("grade", ""),
            "skill_tags": profile.get("skill_tags", []),
            "daily_hours": profile.get("daily_hours", 2.0),
            "learning_style": profile.get("learning_style", "mixed"),
            "hardware": practice_profile.get("hardware", ""),
            "device_type": practice_profile.get("device_type", ""),
            "gpu_model": practice_profile.get("gpu_model", ""),
            "vram_gb": practice_profile.get("vram_gb"),
            "gpu_architecture": practice_profile.get("gpu_architecture", ""),
            "knowledge_gaps": profile.get("knowledge_gaps", []),
            "learning_rhythm": profile.get("learning_rhythm", "medium"),
            "confidence_index": profile.get("confidence_index", 0.5),
            "confusion_markers": profile.get("confusion_markers", []),
            "confidence": profile.get("confidence", 0.0),
            "version": profile.get("version", 1),
            "confirmed": session.get("profile_confirmed", False),
            "created_at": session.get("updated_at"),
            "updated_at": session.get("updated_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profile/confirm")
async def confirm_profile(
    session_id: str,
    request: ProfileConfirmRequest,
):
    """
    确认或质疑画像
    
    POST /api/v1/sessions/{session_id}/profile/confirm
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 会话状态检查：已放弃/已完成的会话不允许再确认画像
        status = session.get("status", "active")
        if status != "active":
            raise HTTPException(status_code=409, 
                                detail=f"会话状态为 {status}，无法确认画像，请创建新会话")

        # 检查画像是否存在
        existing_profile = session.get("profile")
        if not existing_profile:
            raise HTTPException(status_code=404, detail="Profile not found, 请先完成诊断")

        # 使用 harness 的 state_manager 确保缓存共享
        sm = harness.state_manager

        if request.confirm:
            # 确认画像
            sm.update_field(session_id, "profile_confirmed", True)

            # 判断是否需要生成/重新生成阶段：
            # 1. 还没有阶段 -> 生成
            # 2. 画像被修改过（profile_modified=True）-> 重新生成以同步个性化字段
            profile_modified = session.get("profile_modified", False)
            need_generate = not session.get("stage_info") or profile_modified

            if need_generate:
                from models.contracts import StagePlanningInput
                profile = session.get("profile", {})
                from models.profile import UserProfile
                user_profile = UserProfile.model_validate(profile)

                try:
                    stage_output = await harness.agent1.generate_stage(
                        StagePlanningInput(
                            session_id=session_id,
                            profile=user_profile,
                            stage_num=1,
                        )
                    )
                    sm.update_stage(session_id, stage_output.stage)
                    sm.update_field(session_id, "phase", "stage_ready")
                    sm.update_field(session_id, "available_actions", ["start_stage", "edit_profile"])
                    sm.update_field(session_id, "profile_modified", False)  # 重置标记

                    action = "学习阶段已重新生成（画像已更新）" if profile_modified else "学习阶段已生成"
                    return {
                        "profile_id": f"profile_{session_id[:8]}",
                        "session_id": session_id,
                        "status": "confirmed",
                        "phase": "stage_ready",
                        "next_step": "开始学习",
                        "stage_generated": True,
                        "stage_info": stage_output.stage.model_dump(mode="json"),
                        "message": f"画像已确认，{action}",
                    }
                except Exception as e:
                    sm.update_field(session_id, "phase", "stage_ready")
                    sm.update_field(session_id, "available_actions", ["edit_profile"])
                    return {
                        "profile_id": f"profile_{session_id[:8]}",
                        "session_id": session_id,
                        "status": "confirmed",
                        "phase": "stage_ready",
                        "next_step": "编辑画像",
                        "stage_generated": False,
                        "message": f"画像已确认，但生成阶段失败: {str(e)}",
                    }
            else:
                sm.update_field(session_id, "phase", "stage_ready")
                sm.update_field(session_id, "available_actions", ["start_stage", "edit_profile"])
                return {
                    "profile_id": f"profile_{session_id[:8]}",
                    "session_id": session_id,
                    "status": "confirmed",
                    "phase": "stage_ready",
                    "next_step": "开始学习",
                    "stage_generated": False,
                    # 返回当前阶段信息（含 subtopic_ids/subtopic_names），
                    # 前端 confirmProfile 后可直接用于学习包分页过滤
                    "stage_info": session.get("stage_info"),
                    "message": "画像已确认",
                }
        else:
            # 用户提出质疑
            sm.update_field(session_id, "profile_confirmed", False)
            sm.update_field(session_id, "phase", "diagnosing")
            sm.update_field(session_id, "available_actions", ["submit_diagnosis"])
            
            return {
                "profile_id": f"profile_{session_id[:8]}",
                "session_id": session_id,
                "status": "disputed",
                "phase": "diagnosing",
                "next_step": "重新诊断",
                "message": f"已记录您的质疑，备注：{request.notes or '无'}。请在下方输入框补充更多信息。",
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/profile", response_model=ProfileUpdateResponse)
async def update_profile(
    session_id: str,
    request: ProfileUpdateRequest,
):
    """
    修改画像字段（设置页偏好持久化）
    
    PATCH /api/v1/sessions/{session_id}/profile
    
    请求体字段：
    - daily_hours: 每阶段学习时长
    - learning_style: 学习风格
    - hardware: 硬件条件
    - goal: 学习目标
    
    行为：只更新画像，不重新诊断、不重新规划当前阶段。
    如果用户希望阶段同步新的个性化字段，需再次调用 POST /profile/confirm 触发重生成。
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        profile = session.get("profile")
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found, 请先完成诊断")

        # 使用 harness 的 state_manager 确保缓存共享
        sm = harness.state_manager
        
        updated_fields = {}
        if request.goal is not None:
            profile["goal"] = request.goal
            updated_fields["goal"] = request.goal
        if request.daily_hours is not None:
            profile["daily_hours"] = request.daily_hours
            updated_fields["daily_hours"] = request.daily_hours
        if request.learning_style is not None:
            profile["learning_style"] = request.learning_style
            updated_fields["learning_style"] = request.learning_style
        if request.hardware is not None:
            # update_practice_profile persists to learner_profiles and then
            # refreshes the runtime DTO used by Agent3.
            sm.update_practice_profile(session_id, {"hardware": request.hardware})
            updated_fields["hardware"] = request.hardware
        
        learning_profile_updated = any(
            field in updated_fields for field in ("goal", "daily_hours", "learning_style")
        )
        if learning_profile_updated:
            sm.update_profile(session_id, profile)
            # 标记画像已修改：下次 confirm 时会重新生成阶段以同步个性化字段
            # 但 PATCH 本身不触发重新规划，保持当前阶段不变。
            sm.update_field(session_id, "profile_modified", True)
        
        return {
            "profile_id": f"profile_{session_id[:8]}",
            "session_id": session_id,
            "updated_fields": updated_fields,
            "profile_modified": learning_profile_updated,
            "message": f"已更新 {len(updated_fields)} 个字段，如需同步到学习阶段请重新确认画像",
        }
    except HTTPException:
        raise
    except ValueError as e:
        if "not active" in str(e):
            raise HTTPException(status_code=409, detail=f"会话状态异常: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
