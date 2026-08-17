# -*- coding: utf-8 -*-
"""测试服务 API（Agent46_API接口任务说明.md 第 2 节）。

由 Agent4/6 队友承担工程实现。路径遵循文档定义的 /api/v1 前缀。
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_assessment_service, get_harness, require_owned_attempt, require_owned_session
from services.assessment_service import AssessmentError

router = APIRouter()
assessment_service = get_assessment_service()
harness = get_harness()


class SaveAnswerRequest(BaseModel):
    user_answer: str = Field(..., description="本题的作答内容，提交前可反复覆盖")
    time_spent: int = Field(default=0, description="本题作答用时（秒），可选")


def _raise(error: Exception):
    if isinstance(error, AssessmentError):
        # 结构化业务错误：前端可按 code 做差异化流程（例如 NO_QUESTIONS_AVAILABLE 走"直接结束阶段"弹窗）
        detail = {
            "code": error.code,
            "message": str(error),
        }
        # 业务侧"无题"统一用 400（Bad Request / 业务前置不满足），
        # 避免与 FastAPI 原生参数校验失败的 422（Unprocessable Entity）混淆。
        status_code = 400
        raise HTTPException(status_code=status_code, detail=detail)
    raise HTTPException(status_code=500, detail=str(error))


@router.post("/sessions/{session_id}/stages/current/assessment-attempts")
async def create_assessment_attempt(session_id: str, _user_id: str = Depends(require_owned_session)):
    """根据当前学习包创建一次测试（冻结题目，返回独立 attempt_id）。"""
    try:
        print(f"[API][POST /assessment-attempts] session_id={session_id}", flush=True)
        return assessment_service.create_attempt(session_id)
    except Exception as e:
        _raise(e)


@router.get("/assessment-attempts/{attempt_id}")
async def get_assessment_attempt(attempt_id: str, _user_id: str = Depends(require_owned_attempt)):
    """刷新后恢复测试进度（未提交不返回答案/解析，提交后返回完整结果）。"""
    try:
        return assessment_service.get_attempt(attempt_id)
    except Exception as e:
        _raise(e)


@router.get("/sessions/{session_id}/stages/{stage_num}/result")
async def get_stage_result(
    session_id: str,
    stage_num: int,
    _user_id: str = Depends(require_owned_session),
):
    """Read a persisted Agent6 summary and the submitted assessment review."""
    try:
        summary = harness.state_manager.get_stage_result(session_id, stage_num)
        snapshot = harness.state_manager.get_stage_snapshot(session_id, stage_num) or {}
        stage = snapshot.get("stage_info") or {}
        attempt = assessment_service.get_submitted_attempt(session_id, stage.get("stage_id"))
        if not attempt:
            attempt = assessment_service.get_submitted_attempt_for_stage_num(session_id, stage_num)
        # Historical attempts created before Agent6 summaries were persisted
        # still need to open as a completed-stage summary, never as a new test.
        if not summary and attempt:
            accuracy = attempt.get("accuracy")
            summary = {
                "status": "completed",
                "stage_num": stage_num,
                "stage_id": attempt.get("stage_id") or stage.get("stage_id"),
                "accuracy": accuracy,
                "stage_summary": (
                    "该阶段测试已完成，已根据保存的测试记录恢复阶段总结。"
                    f"本次正确率为 {round(float(accuracy or 0) * 100)}%。"
                ),
                "next_strategy": stage.get("strategy_used") or "review",
            }
        if not summary and not attempt:
            raise HTTPException(status_code=404, detail="Stage result not found")
        return {
            "session_id": session_id,
            "stage_num": stage_num,
            "stage_result": summary,
            "assessment_attempt": attempt,
        }
    except HTTPException:
        raise
    except Exception as e:
        _raise(e)


@router.put("/assessment-attempts/{attempt_id}/answers/{question_id}")
async def save_assessment_answer(attempt_id: str, question_id: str, request: SaveAnswerRequest, _user_id: str = Depends(require_owned_attempt)):
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
async def submit_assessment_attempt(attempt_id: str, _user_id: str = Depends(require_owned_attempt)):
    """检查完整性、统一判题并公布结果（提交后不可再改）。"""
    try:
        print(f"[API][POST /assessment-attempts/submit] attempt_id={attempt_id}", flush=True)
        return assessment_service.submit_attempt(attempt_id)
    except Exception as e:
        _raise(e)


@router.post("/sessions/{session_id}/stages/current/complete")
async def complete_current_stage_v1(
    session_id: str,
    bypass_assessment: bool = Query(False, description="若该阶段无测试题，传 true 直接完成阶段"),
    _user_id: str = Depends(require_owned_session),
):
    """v1 统一链路的阶段完成接口。

    默认硬前置：当前阶段必须存在 status=submitted 的 assessment_attempt。
    例外：bypass_assessment=true 时，只允许"该阶段实际无题（DB 查不出任何题目，且 package.questions 也为空）"的
    情况直接走完成链路；若该阶段其实有题目（仅用户未交卷或题目量很少），会拒绝，避免绕过测试。
    """
    print(f"[API][POST /api/v1/.../stages/current/complete] session_id={session_id} bypass={bypass_assessment}", flush=True)
    # 1) 取当前阶段
    stage = harness.state_manager.get_current_stage(session_id)
    if not stage:
        raise HTTPException(status_code=404, detail="当前会话没有进行中的阶段")
    stage_id = stage.get("stage_id")

    # 2) 前置校验
    if bypass_assessment:
        # 只允许"确实无题"时绕过：先尝试抽题，若抽不到任何才允许 bypass
        try:
            block_id = stage.get("block_id")
            subtopic_ids = list(stage.get("subtopic_ids") or []) or (
                [stage["subtopic_id"]] if stage.get("subtopic_id") else []
            )
            any_question = False
            db_client = harness.state_manager.db_client
            if block_id:
                any_question = bool(db_client.list_questions_for_stage(
                    block_id, subtopic_ids=subtopic_ids or None, limit=1, questions_per_subtopic=1,
                ))
            if not any_question:
                # 兜底：再检查 package 里的 questions
                package = harness.state_manager.get_learning_package(session_id)
                pkg_qs = package.get("questions") if isinstance(package, dict) else None
                if isinstance(pkg_qs, list) and pkg_qs:
                    any_question = True
            if any_question:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "HAS_QUESTIONS_CANNOT_BYPASS",
                        "message": "当前阶段存在测试题，请完成测试后再提交",
                    },
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"检测题目失败：{exc}") from exc
    else:
        if not assessment_service.has_submitted_attempt(session_id, stage_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "ASSESSMENT_NOT_SUBMITTED",
                    "message": "当前阶段测试未提交，无法完成阶段",
                },
            )

    # 3) 前置 OK，走既有阶段完成链路
    try:
        result = await harness.complete_stage(session_id)
        # 无测试题直接完成时：修正 accuracy/策略判断，避免 Agent6 把 accuracy=0 当成"测试失败"
        if bypass_assessment:
            # accuracy 字段直接置为 null 表示无测试数据，前端据此隐藏正确率
            result["accuracy"] = None
            result["next_strategy"] = "advance_due_to_no_questions"
            # 确保当前阶段状态为 completed，subtopic 状态不被当作失败
            try:
                synced = result.get("updated_profile")
                if isinstance(synced, dict) and isinstance(synced.get("knowledge_gaps"), list):
                    # 不改画像内容，仅用于展示
                    pass
                stage_obj = harness.state_manager.get_current_stage(session_id)
                if stage_obj and stage_obj.get("status") != "completed":
                    stage_obj["status"] = "completed"
                    stage_obj["strategy_used"] = result.get("next_strategy")
                    harness.state_manager.update_stage(session_id, stage_obj)
                # 再刷新一次 runtime state，确保 UI 看到 status=completed
                result["status"] = result.get("status") or "continue"
            except Exception:
                pass
        phase = result.get("phase", "stage_result")
        available_actions = result.get("available_actions", ["plan_next_stage"])
        harness.state_manager.update_field(session_id, "phase", phase)
        harness.state_manager.update_field(session_id, "available_actions", available_actions)
        return {
            **result,
            "phase": phase,
            "available_actions": available_actions,
            "bypass_assessment": bypass_assessment,
            "state": harness.get_runtime_state(session_id),
        }
    except Exception as e:
        status_code = 404 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(e))
