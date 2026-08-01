from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_harness

router = APIRouter()
harness = get_harness()


class StartStageRequest(BaseModel):
    session_id: str


class SubmitAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    user_answer: str


class PlanNextStageRequest(BaseModel):
    session_id: str
    adjustment_notes: Optional[str] = Field(
        default=None,
        description="用户对下一阶段的手动调整说明，例如：想先学某块、跳过某个板块、调整难度等",
    )


@router.post("/start")
async def start_stage(request: StartStageRequest):
    try:
        print(f"[API][POST /api/stage/start] session_id={request.session_id}", flush=True)
        result = await harness.start_stage(request.session_id)
        return {
            **result,
            "state": harness.get_runtime_state(request.session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))


@router.post("/submit-answer")
async def submit_answer(request: SubmitAnswerRequest):
    try:
        print(f"[API][POST /api/stage/submit-answer] session_id={request.session_id} | question_id={request.question_id}", flush=True)
        result = await harness.submit_answer(
            request.session_id, request.question_id, request.user_answer
        )
        return {
            **result,
            "state": harness.get_runtime_state(request.session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))


@router.post("/complete/{session_id}")
async def complete_stage(session_id: str):
    """
    complete_stage 职责（P3 按需阶段规划）：
    - 只调用 Agent6 决策，保存更新后的用户画像
    - 只在全部知识盲区都掌握时才真正结束 session
    - 返回 can_plan_next_stage / all_mastered / session_status
    - 不在此接口内生成 next_stage，交由 /plan-next 按需生成
    """
    try:
        print(f"[API][POST /api/stage/complete] session_id={session_id}", flush=True)
        result: Dict[str, Any] = await harness.complete_stage(session_id)
        return {
            **result,
            "state": harness.get_runtime_state(session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))


@router.post("/plan-next")
async def plan_next_stage(request: PlanNextStageRequest):
    """
    按需生成下一阶段：
    - 由前端在用户点击“生成下一阶段”后调用
    - 基于 complete_stage 后最新的画像 + (可选) 用户 adjustment_notes
    - 通过 Agent1.generate_stage 规划出真正的 next_stage
    - 保存到当前 session 的 next_stage 字段中（start_stage 时自动切换为当前）
    """
    try:
        print(
            f"[API][POST /api/stage/plan-next] session_id={request.session_id}"
            + (f" | notes={request.adjustment_notes[:60]}" if request.adjustment_notes else ""),
            flush=True,
        )
        result = await harness.plan_next_stage(
            session_id=request.session_id,
            adjustment_notes=request.adjustment_notes,
        )
        return {
            **result,
            "state": harness.get_runtime_state(request.session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))
