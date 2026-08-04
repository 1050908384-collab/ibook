# -*- coding: utf-8 -*-
"""测试服务 API（Agent46_API接口任务说明.md 第 2 节）。

由 Agent4/6 队友承担工程实现。路径遵循文档定义的 /api/v1 前缀。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_assessment_service, get_harness
from services.assessment_service import AssessmentError

router = APIRouter()
assessment_service = get_assessment_service()
harness = get_harness()


class SaveAnswerRequest(BaseModel):
    user_answer: str = Field(..., description="本题的作答内容，提交前可反复覆盖")
    time_spent: int = Field(default=0, description="本题作答用时（秒），可选")


def _raise(error: Exception):
    status_code = 400 if isinstance(error, AssessmentError) else 500
    raise HTTPException(status_code=status_code, detail=str(error))


@router.post("/sessions/{session_id}/stages/current/assessment-attempts")
async def create_assessment_attempt(session_id: str):
    """根据当前学习包创建一次测试（冻结题目，返回独立 attempt_id）。"""
    try:
        print(f"[API][POST /assessment-attempts] session_id={session_id}", flush=True)
        return assessment_service.create_attempt(session_id)
    except Exception as e:
        _raise(e)


@router.get("/assessment-attempts/{attempt_id}")
async def get_assessment_attempt(attempt_id: str):
    """刷新后恢复测试进度（未提交不返回答案/解析，提交后返回完整结果）。"""
    try:
        return assessment_service.get_attempt(attempt_id)
    except Exception as e:
        _raise(e)


@router.put("/assessment-attempts/{attempt_id}/answers/{question_id}")
async def save_assessment_answer(attempt_id: str, question_id: str, request: SaveAnswerRequest):
    """保存或覆盖单题答案（提交前可改，只保存一条最终答案）。"""
    try:
        return assessment_service.save_answer(
            attempt_id=attempt_id,
            question_id=question_id,
            user_answer=request.user_answer,
            time_spent=request.time_spent,
        )
    except Exception as e:
        _raise(e)


@router.post("/assessment-attempts/{attempt_id}/submit")
async def submit_assessment_attempt(attempt_id: str):
    """检查完整性、统一判题并公布结果（提交后不可再改）。"""
    try:
        print(f"[API][POST /assessment-attempts/submit] attempt_id={attempt_id}", flush=True)
        return assessment_service.submit_attempt(attempt_id)
    except Exception as e:
        _raise(e)


@router.post("/sessions/{session_id}/stages/current/complete")
async def complete_current_stage_v1(session_id: str):
    """v1 统一链路的阶段完成接口。

    硬前置：当前阶段必须存在 status=submitted 的 assessment_attempt，否则返回 409
    ASSESSMENT_NOT_SUBMITTED，杜绝未交卷绕过完成阶段。校验仅在此接口层完成，
    harness.complete_stage 及其内部 fallback 不改，旧 /api/stage/complete 行为不变。
    """
    print(f"[API][POST /api/v1/.../stages/current/complete] session_id={session_id}", flush=True)
    # 1) 取当前阶段
    stage = harness.state_manager.get_current_stage(session_id)
    if not stage:
        raise HTTPException(status_code=404, detail="当前会话没有进行中的阶段")
    stage_id = stage.get("stage_id")

    # 2) 前置校验：当前阶段必须已交卷
    if not assessment_service.has_submitted_attempt(session_id, stage_id):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ASSESSMENT_NOT_SUBMITTED",
                "message": "当前阶段测试未提交，无法完成阶段",
            },
        )

    # 3) 交卷已满足，走既有阶段完成链路
    try:
        result = await harness.complete_stage(session_id)
        phase = result.get("phase", "stage_result")
        available_actions = result.get("available_actions", ["plan_next_stage"])
        # This route is registered before the legacy stage router, so it is
        # the effective /complete handler and must persist the phase itself.
        harness.state_manager.update_field(session_id, "phase", phase)
        harness.state_manager.update_field(session_id, "available_actions", available_actions)
        return {
            **result,
            "phase": phase,
            "available_actions": available_actions,
            "state": harness.get_runtime_state(session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))
