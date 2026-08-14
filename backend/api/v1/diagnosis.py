# -*- coding: utf-8 -*-
"""
v1/diagnosis.py - Agent1/3 前端 API：诊断接口
==============================================
对应文档：Agent13_API接口任务说明.md 第 2 节（诊断消息）
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from api.dependencies import get_harness, require_owned_session
from models.contracts import StagePlanningInput

router = APIRouter(prefix="/sessions/{session_id}/diagnosis", tags=["Diagnosis"], dependencies=[Depends(require_owned_session)])
harness = get_harness()


# === 分步诊断模型 ===

class GoalBlocksRequest(BaseModel):
    goal: str


class GoalBlockItem(BaseModel):
    block_id: str
    block_name: str
    description: str = ""
    difficulty: str = ""
    estimated_hours: float = 0
    subtopics: list[dict] = Field(default_factory=list)


class GoalBlocksResponse(BaseModel):
    goal: str
    blocks: list[GoalBlockItem] = Field(default_factory=list)


class FoundationSubmitRequest(BaseModel):
    goal: str
    mastered_subtopic_ids: list[str] = Field(default_factory=list)
    daily_hours: float = 2.0
    learning_style: str = "mixed"


class FoundationSubmitResponse(BaseModel):
    status: str
    profile: Optional[dict] = None
    stage: Optional[dict] = None
    followup_questions: list[str] = Field(default_factory=list)


class DiagnosisMessageRequest(BaseModel):
    content: str
    message_type: str = "user"


class DiagnosisMessage(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: Optional[str] = None


class DiagnosisResultResponse(BaseModel):
    status: str
    assistant_message: str
    missing_fields: list[str] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    profile_draft: dict = Field(default_factory=dict)
    confidence: float = 0.0


class DiagnosisHistoryResponse(BaseModel):
    messages: list[DiagnosisMessage] = Field(default_factory=list)
    total: int = 0


@router.post("/messages", response_model=DiagnosisResultResponse)
async def submit_diagnosis_message(
    session_id: str,
    request: DiagnosisMessageRequest,
):
    """
    提交诊断消息（用户输入）
    
    POST /api/v1/sessions/{session_id}/diagnosis/messages
    
    返回字段（文档要求）：
    - status: needs_more_info | profile_ready
    - assistant_message
    - missing_fields
    - followup_questions
    - profile_draft
    - confidence
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 如果已有 profile，且置信度足够，直接生成阶段
        existing_profile = session.get("profile")
        if existing_profile and existing_profile.get("goal") and existing_profile.get("confidence", 0) >= 0.7:
            print(f"[API] Profile already exists with confidence {existing_profile.get('confidence')}")
            
            from models.contracts import StagePlanningInput
            from models.profile import UserProfile
            
            user_profile = UserProfile.model_validate(existing_profile)
            
            # 如果还没有阶段，生成阶段
            if not session.get("stage_info"):
                try:
                    stage_output = await harness.agent1.generate_stage(
                        StagePlanningInput(
                            session_id=session_id,
                            profile=user_profile,
                            stage_num=1,
                        )
                    )
                    # 使用 harness 的 state_manager，确保缓存共享
                    sm = harness.state_manager
                    sm.update_stage(session_id, stage_output.stage)
                    sm.update_field(session_id, "phase", "stage_ready")
                    sm.update_field(session_id, "available_actions", ["start_stage", "edit_profile"])
                    
                    print(f"[API] Stage generated: {stage_output.stage.block_name}")
                    
                    return {
                        "status": "profile_ready",
                        "assistant_message": "画像已完善，学习阶段已生成。",
                        "missing_fields": [],
                        "followup_questions": [],
                        "profile_draft": existing_profile,
                        "confidence": existing_profile.get("confidence", 0.8),
                    }
                except Exception as e:
                    print(f"[API] Stage generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "status": "profile_ready",
                        "assistant_message": f"画像已完善，但生成学习阶段失败: {str(e)}",
                        "missing_fields": [],
                        "followup_questions": [],
                        "profile_draft": existing_profile,
                        "confidence": existing_profile.get("confidence", 0.8),
                    }
            else:
                return {
                    "status": "profile_ready",
                    "assistant_message": "画像已完善，请开始学习。",
                    "missing_fields": [],
                    "followup_questions": [],
                    "profile_draft": existing_profile,
                    "confidence": existing_profile.get("confidence", 0.8),
                }
        
        # 否则调用 harness 进行初始诊断
        print(f"[API] Running initial diagnosis for: {request.content[:50]}...")
        result = await harness.process_initial_input(session_id, request.content)
        print(f"[API] Diagnosis result: {result.get('status')}")
        
        status = result.get("status", "needs_more_info")
        
        if status == "ready":
            response = {
                "status": "profile_ready",
                "assistant_message": "诊断完成，画像已生成。",
                "missing_fields": [],
                "followup_questions": [],
                "profile_draft": result.get("profile"),
                "confidence": 0.8,
            }
        else:
            questions = result.get("questions", [])
            # 确定缺失字段
            missing_fields = []
            profile = result.get("profile") or {}
            if not profile.get("goal"):
                missing_fields.append("goal")
            if profile.get("daily_hours", 0) <= 0:
                missing_fields.append("daily_hours")
            if not profile.get("learning_style"):
                missing_fields.append("learning_style")
            
            response = {
                "status": "needs_more_info",
                "assistant_message": "需要更多信息来完善您的画像。",
                "missing_fields": missing_fields,
                "followup_questions": questions,
                "profile_draft": profile,
                "confidence": profile.get("confidence", 0.0),
            }
        harness.state_manager.add_interaction(session_id, "diagnosis_assistant", {
            "content": response["assistant_message"],
            "followup_questions": response["followup_questions"],
        })
        return response
    except HTTPException:
        raise
    except ValueError as e:
        print(f"[API] ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[API] Exception: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/messages", response_model=DiagnosisHistoryResponse)
async def get_diagnosis_messages(session_id: str):
    """
    获取诊断消息历史
    
    GET /api/v1/sessions/{session_id}/diagnosis/messages
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        interactions = session.get("interactions", [])
        messages = []
        for idx, interaction in enumerate(interactions):
            msg_type = interaction.get("type", "")
            data = interaction.get("data", {})
            if not isinstance(data, dict):
                continue
            if msg_type == "initial_diagnostic" and data.get("user_input"):
                messages.append({
                    "message_id": f"msg_{idx}",
                    "role": "user",
                    "content": str(data["user_input"]),
                    "created_at": session.get("updated_at"),
                })
            elif msg_type == "diagnosis_assistant" and data.get("content"):
                messages.append({
                    "message_id": f"msg_{idx}",
                    "role": "assistant",
                    "content": str(data["content"]),
                    "created_at": session.get("updated_at"),
                })
        
        return {
            "messages": messages,
            "total": len(messages),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================================
# 分步诊断：目标 → 知识板块 → 知识点 → 确定
# ==================================================================

@router.post("/goal-blocks", response_model=GoalBlocksResponse)
async def get_goal_blocks(session_id: str, request: GoalBlocksRequest):
    """
    根据学习目标返回其下所有知识板块（含知识点列表）。

    POST /api/v1/sessions/{session_id}/diagnosis/goal-blocks

    前端流程第 1 步：用户输入大目标后调用此接口，
    获取目标匹配的知识板块列表，每块含 subtopics 供前端展开。
    """
    try:
        blocks = harness.agent1.get_goal_blocks(request.goal)
        return {"goal": request.goal, "blocks": blocks}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/foundation", response_model=FoundationSubmitResponse)
async def submit_foundation(session_id: str, request: FoundationSubmitRequest):
    """
    用户确认已掌握知识点后提交，生成画像并进入阶段规划。

    POST /api/v1/sessions/{session_id}/diagnosis/foundation

    前端流程最后一步：用户选好已掌握的知识点后点击"确定"，
    将 mastered_subtopic_ids 提交给 Agent1，生成 profile 和第一个学习阶段。
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        diagnostic = await harness.agent1.run_with_foundation(
            session_id=session_id,
            goal=request.goal,
            mastered_subtopic_ids=request.mastered_subtopic_ids,
            daily_hours=request.daily_hours,
            learning_style=request.learning_style,
        )
        profile = diagnostic.profile
        harness.state_manager.update_profile(session_id, profile)
        harness.state_manager.add_interaction(session_id, "initial_diagnostic", {
            "user_input": f"目标：{request.goal}；已掌握知识点：{len(request.mastered_subtopic_ids)}个",
        })

        if profile.confidence < 0.7:
            return {
                "status": "needs_more_info",
                "profile": profile.model_dump(mode="json"),
                "stage": None,
                "followup_questions": diagnostic.followup_questions,
            }

        # 生成第一个学习阶段
        stage_output = await harness.agent1.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=profile,
            stage_num=1,
        ))
        harness.state_manager.update_stage(session_id, stage_output.stage)
        harness.state_manager.update_field(session_id, "phase", "stage_ready")
        harness.state_manager.update_field(session_id, "available_actions", ["start_stage", "edit_profile"])

        return {
            "status": "profile_ready",
            "profile": profile.model_dump(mode="json"),
            "stage": stage_output.stage.model_dump(mode="json"),
            "followup_questions": [],
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
