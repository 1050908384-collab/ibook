# -*- coding: utf-8 -*-
"""仪表盘 API（Agent46_API接口任务说明.md 第 4 节）。

由 Agent4/6 队友承担工程实现：后端聚合 user_profiles / stage_progress /
subtopic_status / assessment_attempts / user_qa，不属于 Agent4 的运行职责。
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_harness, require_owned_session

router = APIRouter()
harness = get_harness()


def _build_dashboard(session_id: str, db=None) -> Dict[str, Any]:
    db = db or harness.state_manager.db_client

    # 会话不存在时直接 404：必须在任何聚合调用之前判断，因为部分聚合
    # （get_subtopic_mastery_counts / get_block_progress）会触发
    # ensure_subtopic_status → INSERT subtopic_status，对不存在的 session
    # 会因外键约束抛 IntegrityError（被上层误判为 500）。
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session dashboard not found: {session_id}")

    profile = db.get_latest_profile(session_id)
    stages = db.list_stage_progress(session_id)
    # 同一 stage_num 可能因低分重试产生多条 stage_progress 记录；它们是
    # 审计/恢复数据，不应在学习导航中被渲染成多个阶段。保留该编号最后
    # 写入的一条作为当前可见阶段记录。
    visible_by_num: Dict[int, Dict[str, Any]] = {}
    for item in stages:
        num = int(item.get("stage_num") or 0)
        if num > 0:
            visible_by_num[num] = item
    stages = [visible_by_num[num] for num in sorted(visible_by_num)]
    mastered_count, total_count = db.get_subtopic_mastery_counts(session_id)
    block_progress = db.get_block_progress(session_id)
    attempts = db.list_assessment_attempts(session_id)
    recent_qa = db.get_recent_qa(session_id, limit=5)

    if profile is None and not stages and total_count == 0:
        raise HTTPException(status_code=404, detail=f"Session dashboard not found: {session_id}")

    # 当前阶段摘要：取最大 stage_num 的阶段
    current_stage: Optional[Dict[str, Any]] = stages[-1] if stages else None
    current_stage_summary: Optional[Dict[str, Any]] = None
    if current_stage:
        current_stage_summary = {
            "stage_id": current_stage["stage_id"],
            "stage_num": current_stage["stage_num"],
            "block_id": current_stage["block_id"],
            "block_name": current_stage["block_name"],
            "subtopic_id": current_stage["subtopic_id"],
            "status": current_stage["status"],
            "accuracy": current_stage["accuracy"],
            "strategy_used": current_stage["strategy_used"],
        }

    # 画像摘要
    profile_summary: Optional[Dict[str, Any]] = None
    confidence_index = 0.5
    learning_rhythm = "medium"
    if profile:
        confidence_index = profile.get("confidence_index", 0.5) or 0.5
        learning_rhythm = profile.get("learning_rhythm", "medium") or "medium"
        profile_summary = {
            "goal": profile.get("goal", ""),
            "grade": profile.get("grade", ""),
            "skill_tags": profile.get("skill_tags", []),
            "knowledge_gaps": profile.get("knowledge_gaps", []),
            "daily_hours": profile.get("daily_hours"),
            "learning_style": profile.get("learning_style"),
            "version": profile.get("version"),
        }

    # 总体进度：已掌握子知识点占比
    overall_progress = round(mastered_count / total_count, 3) if total_count else 0.0

    # 仪表盘正确率按阶段聚合，每个已有测试结果的阶段计一次。
    stage_accuracies = [
        float(stage["accuracy"])
        for stage in stages
        if stage.get("accuracy") is not None
    ]
    average_accuracy = (
        round(sum(stage_accuracies) / len(stage_accuracies), 4)
        if stage_accuracies
        else None
    )

    # 阶段历史：所有阶段的精简记录
    stage_history: List[Dict[str, Any]] = [
        {
            "stage_id": s["stage_id"],
            "stage_num": s["stage_num"],
            "block_id": s["block_id"],
            "block_name": s["block_name"],
            "subtopic_id": s["subtopic_id"],
            "estimated_hours": s["estimated_hours"],
            "status": s["status"],
            "accuracy": s["accuracy"],
            "strategy_used": s["strategy_used"],
            "completed_at": s["completed_at"],
        }
        for s in stages
    ]

    # 正确率趋势：只取已提交 attempt，按提交顺序
    accuracy_trend: List[Dict[str, Any]] = [
        {
            "attempt_id": a["attempt_id"],
            "stage_num": a["stage_num"],
            "accuracy": a["accuracy"],
            "correct_count": a["correct_count"],
            "total_questions": a["total_questions"],
            "submitted_at": a["submitted_at"],
        }
        for a in attempts
        if a["status"] == "submitted"
    ]

    return {
        "session_id": session_id,
        "profile_summary": profile_summary,
        "current_stage_summary": current_stage_summary,
        "overall_progress": overall_progress,
        # 保留旧字段，避免已有调用方失效。
        "current_accuracy": average_accuracy,
        "average_accuracy": average_accuracy,
        "confidence_index": confidence_index,
        "learning_rhythm": learning_rhythm,
        "mastered_count": mastered_count,
        "total_count": total_count,
        "block_progress": block_progress,
        "stage_history": stage_history,
        "accuracy_trend": accuracy_trend,
        "recent_qa": recent_qa,
    }


@router.get("/sessions/{session_id}/dashboard")
async def get_dashboard(session_id: str, _user_id: str = Depends(require_owned_session)):
    """获取仪表盘主数据（聚合多表）。"""
    try:
        print(f"[API][GET /sessions/{session_id}/dashboard]", flush=True)
        return _build_dashboard(session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
