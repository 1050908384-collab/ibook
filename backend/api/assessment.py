# -*- coding: utf-8 -*-
"""测试服务 API（专业知识与教学质量审核 Agent6_API接口任务说明.md 第 2 节）。

由 专业知识与教学质量审核 Agent/6 队友承担工程实现。路径遵循文档定义的 /api/v1 前缀。
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_assessment_service, get_harness, require_owned_attempt, require_owned_session
from services.assessment_service import AssessmentError
from tools.agent6_tools.stage_summary_generator_tool import StageSummaryGeneratorTool

router = APIRouter()
assessment_service = get_assessment_service()
harness = get_harness()
stage_summary_generator = StageSummaryGeneratorTool()


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


async def _rebuild_agent6_summary(
    session_id: str,
    stage: dict,
    attempt: dict,
) -> str:
    """Recreate a legacy 学情评估与动态决策 Agent report from immutable submitted answers."""
    profile = harness.state_manager.get_profile(session_id) or {}
    question_records = [
        {
            "question_id": question.get("question_id"),
            "skill": question.get("subtopic_id") or stage.get("subtopic_id") or stage.get("block_name"),
            "question_type": question.get("question_type"),
            "is_correct": bool(question.get("is_correct")),
            "time_spent": question.get("time_spent", 0),
        }
        for question in attempt.get("questions", [])
    ]
    return await stage_summary_generator.generate_summary(
        stage_name=stage.get("block_name") or "当前学习阶段",
        accuracy=float(attempt.get("accuracy") or 0),
        knowledge_gaps=profile.get("knowledge_gaps", []),
        confusion_tags=[],
        question_records=question_records,
        emotion_score=None,
        pace_adjustment=profile.get("learning_rhythm", "medium"),
        time_spent_hours=0,
        expected_time_hours=float(stage.get("estimated_hours") or 0),
        next_strategy=stage.get("strategy_used") or "reinforce",
        total_knowledge_points=len(stage.get("subtopic_ids") or []) or 1,
        total_answers=int(attempt.get("answered_count") or len(question_records)),
    )


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
    stage_id: str | None = None,
    _user_id: str = Depends(require_owned_session),
):
    """Read a persisted 学情评估与动态决策 Agent summary and the submitted assessment review."""
    try:
        summary = None
        snapshot = None
        if stage_id:
            summary = harness.state_manager.get_stage_result_by_id(session_id, stage_id)
            snapshot = harness.state_manager.get_stage_snapshot_by_id(session_id, stage_id)
        # Dashboard/history caches can retain an earlier stage_id after a
        # deterministic stage snapshot is rebuilt. Fall back to stage_num so
        # an existing completed stage does not incorrectly return 404.
        if not summary:
            summary = harness.state_manager.get_stage_result(session_id, stage_num)
        if not snapshot:
            snapshot = harness.state_manager.get_stage_snapshot(session_id, stage_num)
        snapshot = snapshot or {}
        stage = snapshot.get("stage_info") or {}
        # A retry keeps the same stage_id but resets the stage to pending/learning.
        # Never let the previous cycle's persisted summary or submitted attempt
        # make that regenerated learning package look completed after refresh.
        if summary and not assessment_service.stage_allows_persisted_result(stage):
            summary = None
        attempt = assessment_service.get_submitted_attempt_for_result(
            session_id, stage, stage_num
        )
        # Historical attempts created before 学情评估与动态决策 Agent summaries were persisted
        # still need to open as a completed-stage summary, never as a new test.
        if not summary and attempt:
            accuracy = attempt.get("accuracy")
            summary = {
                "status": "completed",
                "stage_num": stage_num,
                "stage_id": attempt.get("stage_id") or stage.get("stage_id"),
                "accuracy": accuracy,
                "stage_summary": await _rebuild_agent6_summary(session_id, stage, attempt),
                "next_strategy": stage.get("strategy_used") or "review",
                "summary_source": "agent6_rebuilt_from_saved_assessment",
                "can_plan_next_stage": True,
                "available_actions": ["plan_next_stage"],
            }
            harness.state_manager.save_stage_result(session_id, stage_num, summary)
        # Some earlier no-question stages were marked completed by 学情评估与动态决策 Agent but
        # their textual result was only returned to the browser, never stored.
        # Recreate a durable archive summary from the completed stage itself.
        if not summary and stage.get("status") in {"completed", "retreat"}:
            bypass = stage.get("strategy_used") == "advance_due_to_no_questions"
            topic = stage.get("block_name") or "当前学习主题"
            subtopic = stage.get("subtopic_name") or stage.get("gap_skill") or topic
            summary = {
                "status": "completed" if stage.get("status") == "completed" else "repeat",
                "stage_num": stage_num,
                "stage_id": stage.get("stage_id"),
                "accuracy": None if bypass else stage.get("accuracy"),
                "bypass_assessment": bypass,
                "stage_summary": (
                    f"# 阶段学习总结\n\n"
                    f"你已完成“{topic}”阶段，当前学习重点为“{subtopic}”。\n\n"
                    + ("本阶段没有匹配的测试题目，系统已按无题完成流程归档。"
                       if bypass else "本阶段已完成，建议结合学习资料继续巩固薄弱知识点。")
                ),
                "next_strategy": stage.get("strategy_used") or "review",
                "can_plan_next_stage": True,
                "available_actions": ["plan_next_stage"],
            }
            harness.state_manager.save_stage_result(session_id, stage_num, summary)
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
    """
    检查完整性、统一判题并公布结果（提交后不可再改）。
    异步路径：规则判分（快速）+ fill_blank/实操题失败时 LLM 语义兜底（可选，hybrid 模式且 harness 可用时生效）。
    """
    try:
        print(f"[API][POST /assessment-attempts/submit] attempt_id={attempt_id}", flush=True)
        return await assessment_service.submit_attempt_async(attempt_id)
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
        result = await harness.complete_stage(session_id, bypass_assessment=bypass_assessment)
        # 无测试题直接完成时：修正 accuracy/策略判断，避免 学情评估与动态决策 Agent 把 accuracy=0 当成"测试失败"
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
