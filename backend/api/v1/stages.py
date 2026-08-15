# -*- coding: utf-8 -*-
"""
v1/stages.py - Agent1/3 前端 API：阶段与学习包接口
================================================
对应文档：Agent13_API接口任务说明.md 第 3 节
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.dependencies import get_harness, require_owned_session
from api.v1.learning_resources import backfill_stage_tutorial, build_stage_document_resources
from api.v1.schemas import to_public_learning_package, to_public_question

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

router = APIRouter(prefix="/sessions/{session_id}/stages", tags=["Stages"], dependencies=[Depends(require_owned_session)])
harness = get_harness()


class StartStageRequest(BaseModel):
    pass


class AnswerRequest(BaseModel):
    question_id: str
    answer: str


class CompleteStageRequest(BaseModel):
    pass


class PlanNextStageRequest(BaseModel):
    adjustment_notes: Optional[str] = None


def _sanitize_question(question: dict) -> dict:
    """移除问题中的敏感字段（correct_answer, explanation, expected_output）"""
    return to_public_question(question)


def _build_learning_resources(stage_info: dict, package: dict) -> list[dict]:
    """
    构建学习资源列表（文档要求字段）。
    
    数据源优先级：
    1. package 中已有的 learning_resources（来自 Agent3 生成）
    2. 通过 LearningResourceTool 从数据库读取已确认视频资源
    3. 兜底：基于 tutorial/code_examples 生成默认资源
    
    返回字段（文档要求）：
    resource_id, resource_type, title, url, platform, description,
    recommendation_level, role, order
    """
    # 防御性处理：None 输入转为空 dict
    stage_info = stage_info or {}
    package = package or {}

    # 1. 优先用 package 已有资源（Agent3 生成），补全字段
    existing_raw = package.get("learning_resources") or []
    existing = [r for r in existing_raw if isinstance(r, dict)]  # 跳过 None/非 dict 项
    if existing:
        normalized = []
        for idx, res in enumerate(existing):
            url = res.get("url", res.get("canonical_url")) or ""
            normalized.append({
                "resource_id": res.get("resource_id") or f"res_{idx}",
                "resource_type": res.get("resource_type") or "video",
                "title": res.get("title") or "",
                "url": url,
                "platform": (res.get("platform") or _extract_platform(url)) or "",
                "description": res.get("description", res.get("content_summary")) or "",
                "recommendation_level": res.get("recommendation_level", res.get("role")) or "supplementary",
                "role": res.get("role") or "supplementary",
                "order": res.get("order", res.get("sequence_index", idx + 1)) or idx + 1,
                # 重要：dict.get(key, default) 仅在 key 不存在时给默认值；若显式存 None 仍会是 None。
                # 使用 "x or ''" 模式确保 None → ""，避免下游 Pydantic LearningResource.subtopic_id 的
                # str 类型校验因 None 失败导致整个接口 500。
                "subtopic_id": (res.get("subtopic_id") or ""),
            })
        # 按 order 排序
        normalized.sort(key=lambda r: (0 if r["role"] == "primary" else 1, r["order"]))
        return normalized

    # 2. 从数据库读取已确认的视频资源
    block_id = stage_info.get("block_id") if stage_info else None
    subtopic_id = stage_info.get("subtopic_id") if stage_info else None
    if block_id:
        try:
            import asyncio
            from tools.learning_resource_tool import LearningResourceTool
            tool = LearningResourceTool()
            # 同步调用 async 方法
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 已在事件循环中，创建 task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        videos = pool.submit(
                            asyncio.run, tool.get_recommended_videos(block_id, subtopic_id)
                        ).result()
                else:
                    videos = loop.run_until_complete(tool.get_recommended_videos(block_id, subtopic_id))
            except RuntimeError:
                videos = asyncio.run(tool.get_recommended_videos(block_id, subtopic_id))

            if videos:
                resources = []
                for v in videos:
                    resources.append({
                        "resource_id": v.get("resource_id", ""),
                        "resource_type": "video",  # 目前库中只有 video
                        "title": v.get("title", ""),
                        "url": v.get("url", v.get("canonical_url", "")),
                        "platform": _extract_platform(v.get("url", "")),
                        "description": v.get("content_summary", v.get("description", "")),
                        "recommendation_level": v.get("role", "supplementary"),
                        "role": v.get("role", "supplementary"),
                        "order": v.get("sequence_index", 1),
                        "subtopic_id": v.get("subtopic_id") or "",
                    })
                # 按 order 排序，primary 优先
                resources.sort(key=lambda r: (0 if r["role"] == "primary" else 1, r["order"]))
                return resources
        except Exception:
            pass  # 数据库读取失败，走兜底

    # 3. 兜底：基于 tutorial 生成默认资源
    block_name = stage_info.get("block_name", "") if stage_info else ""
    subtopic_name = stage_info.get("subtopic_name", "") if stage_info else ""
    stage_id = stage_info.get("stage_id", "") if stage_info else ""
    tutorial = package.get("tutorial", "") if package else ""
    return [
        {
            "resource_id": f"res_tutorial_{stage_id}",
            "resource_type": "article",
            "title": f"{subtopic_name or block_name} 教程",
            "url": "",
            "platform": "",
            "description": (tutorial[:200] if tutorial else "基础教程"),
            "recommendation_level": "primary",
            "role": "primary",
            "order": 1,
        },
        {
            "resource_id": f"res_code_{stage_id}",
            "resource_type": "code",
            "title": f"{subtopic_name or block_name} 代码示例",
            "url": "",
            "platform": "",
            "description": "包含完整的代码示例和练习",
            "recommendation_level": "supplementary",
            "role": "supplementary",
            "order": 2,
        },
    ]


def _extract_platform(url) -> str:
    """从 URL 提取平台名（支持带/不带 www、短链，兼容 None/非 str）"""
    if not url or not isinstance(url, str):
        return ""
    url_lower = url.lower()
    if "bilibili" in url_lower or "b23.tv" in url_lower:
        return "bilibili"
    if "youtube" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if "github" in url_lower:
        return "github"
    if "zhihu" in url_lower:
        return "zhihu"
    if "csdn" in url_lower:
        return "csdn"
    return ""


@router.get("/current")
async def get_current_stage(session_id: str):
    """
    获取当前阶段详情（不返回题目）
    
    GET /api/v1/sessions/{session_id}/stages/current
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        stage = session.get("stage_info")
        if not stage:
            raise HTTPException(status_code=404, detail="No current stage")

        # 修复旧会话：从数据库补全 subtopic_ids/subtopic_names 数组
        from api.v1.learning_resources import ensure_stage_subtopics
        stage = ensure_stage_subtopics(stage)

        # 确保个性化字段在响应中
        stage_response = {
            "stage_id": stage.get("stage_id"),
            "stage_num": stage.get("stage_num"),
            "block_id": stage.get("block_id"),
            "block_name": stage.get("block_name"),
            "subtopic_id": stage.get("subtopic_id"),
            "subtopic_name": stage.get("subtopic_name"),
            # 返回数组形式的知识点列表，前端按知识点分页过滤学习资源时依赖这两个字段
            "subtopic_ids": stage.get("subtopic_ids", []),
            "subtopic_names": stage.get("subtopic_names", []),
            "estimated_hours": stage.get("estimated_hours"),
            "gap_skill": stage.get("gap_skill"),
            "prerequisites": stage.get("prerequisites", []),
            "status": stage.get("status", "pending"),
            "title": stage.get("title"),
            "description": stage.get("description"),
            "objectives": stage.get("objectives", []),
            "difficulty": stage.get("difficulty"),
            # 个性化字段
            "recommended_daily_hours": stage.get("recommended_daily_hours"),
            "suggested_study_days": stage.get("suggested_study_days"),
            "learning_style_notes": stage.get("learning_style_notes"),
            "hardware_notes": stage.get("hardware_notes"),
            "personalization_summary": stage.get("personalization_summary"),
        }
        
        return {
            "stage": stage_response,
            "phase": session.get("phase", "stage_ready"),
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current/package")
async def get_current_stage_package(session_id: str):
    """
    获取当前阶段学习包（含题目）
    
    GET /api/v1/sessions/{session_id}/stages/current/package
    
    返回字段（文档要求）：
    - stage_id, stage_num, block_name, subtopic_name, estimated_hours
    - tutorial, code_examples, learning_resources, questions, review_summary
    """
    try:
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 如果学习包还未生成，返回提示
        if session.get("phase") == "stage_ready":
            raise HTTPException(
                status_code=409, 
                detail="当前阶段尚未开始，请调用 POST /api/v1/sessions/{session_id}/stages/current/start 开始学习"
            )
        
        package = session.get("learning_package")
        stage_info = session.get("stage_info", {})

        if not package:
            raise HTTPException(status_code=404, detail="No learning package available")

        # 修复旧会话：从数据库补全 subtopic_ids/subtopic_names 数组
        from api.v1.learning_resources import ensure_stage_subtopics
        stage_info = ensure_stage_subtopics(stage_info)

        # ---- 旧快照题量不足时动态重抽并更新 session ----
        stids = list(stage_info.get("subtopic_ids") or [])
        if not stids and stage_info.get("subtopic_id"):
            stids = [stage_info["subtopic_id"]]
        raw_questions = package.get("questions", []) or []
        # 判定标准：
        #   - 有多个 subtopic (>=2) 但题目只有 <=2 → 旧快照
        #   - 题目数 < subtopic 数（每个 subtopic 连 1 题都保证不了）
        need_refresh = False
        if stids and len(raw_questions) <= max(2, len(stids) - 1):
            need_refresh = True
        # 题目来自其他 block 也需要重抽
        block_id = stage_info.get("block_id")
        if not need_refresh and block_id and raw_questions:
            first_q_block = (raw_questions[0].get("block_id") if isinstance(raw_questions[0], dict) else None)
            if first_q_block and first_q_block != block_id:
                need_refresh = True
        if need_refresh and block_id:
            try:
                refreshed = harness.state_manager.db_client.list_questions_for_stage(
                    block_id, subtopic_ids=stids, limit=max(8, len(stids) * 2)
                )
                if refreshed:
                    # 转成 package.questions 存储的 dict 形式（包含 options JSON 处理）
                    package["questions"] = refreshed
                    # 回写到 session state（learning_package）
                    harness.state_manager.update_learning_package(session_id, package)
                    logger.info(
                        "get_current_stage_package refreshed stale questions: session=%s old=%d new=%d stids=%s",
                        session_id, len(raw_questions), len(refreshed), stids,
                    )
            except Exception as exc:
                logger.warning("get_current_stage_package refresh failed, use snapshot: %s", exc)

        # 清理题目中的敏感字段
        questions = package.get("questions", [])
        sanitized_questions = [_sanitize_question(q) for q in questions]

        # 学习资源：使用统一构建函数，从数据库读取已确认视频并补全文档要求字段
        learning_resources = _build_learning_resources(stage_info, package)
        # 先把快照里的旧 article（不带 subtopic_id）过滤掉——它们会导致前端 matchesSubtopic
        # 按名称不匹配而全部被隐藏，使页面显示"暂无匹配文章链接"。
        # 真正的文章资源由 build_stage_document_resources 从本地 LOCAL- 文档动态生成。
        learning_resources = [
            r for r in learning_resources
            if not isinstance(r, dict)
            or str(r.get("resource_type", "")).lower() not in {"article", "documentation"}
            or str(r.get("subtopic_id") or "").strip() != ""
        ]
        known_resource_ids = {
            resource.get("resource_id") for resource in learning_resources if isinstance(resource, dict)
        }
        dynamic_articles = list(build_stage_document_resources(stage_info))
        learning_resources.extend(
            resource for resource in dynamic_articles
            if resource.get("resource_id") not in known_resource_ids
        )

        # 学习包响应（不含正确答案）
        package_response = {
            "stage_id": stage_info.get("stage_id", package.get("stage_id")),
            "stage_num": stage_info.get("stage_num", 1),
            "block_name": stage_info.get("block_name", ""),
            "subtopic_name": stage_info.get("subtopic_name", ""),
            # 确保 package 也返回 subtopic 数组，前端 LearningPackage 依赖它们
            "subtopic_ids": list(stage_info.get("subtopic_ids") or []),
            "subtopic_names": list(stage_info.get("subtopic_names") or []),
            "estimated_hours": stage_info.get("estimated_hours", 0),
            "tutorial": package.get("tutorial"),
            "code_examples": package.get("code_examples", []),
            "learning_resources": learning_resources,
            "questions": sanitized_questions,  # 已清理敏感字段
            "review_summary": {
                "status": package.get("status", "draft"),
                "reviewed_by": "Agent4",
                "review_notes": package.get("review_notes", "待审核"),
            },
            # 个性化字段
            "recommended_daily_hours": stage_info.get("recommended_daily_hours"),
            "suggested_study_days": stage_info.get("suggested_study_days"),
            "learning_style_notes": stage_info.get("learning_style_notes"),
            "hardware_notes": stage_info.get("hardware_notes"),
            "personalization_summary": stage_info.get("personalization_summary"),
        }
        
        package_response = to_public_learning_package(
            backfill_stage_tutorial(package, stage_info),
            stage_info,
            learning_resources,
        )

        # 诊断日志：确认最终响应包含文章资源
        final_resources = (package_response or {}).get("learning_resources") or []
        article_count = sum(1 for r in final_resources if isinstance(r, dict) and r.get("resource_type") == "article")
        logger.info(
            "get_current_stage_package: session=%s stage.block_id=%s stage.subtopic_ids=%s "
            "final_resources=%d article_resources=%d package.subtopic_ids=%s",
            session_id, stage_info.get("block_id"), stage_info.get("subtopic_ids"),
            len(final_resources), article_count,
            (package_response or {}).get("subtopic_ids"),
        )

        return {
            "package": package_response,
            "total_questions": len(sanitized_questions),
            "phase": session.get("phase", "learning"),
            "session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/current/start")
async def start_current_stage(session_id: str):
    """
    开始当前阶段学习

    POST /api/v1/sessions/{session_id}/stages/current/start
    """
    try:
        # 先检查会话是否存在
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if not session.get("stage_info"):
            raise HTTPException(status_code=409, detail="当前没有可开始的阶段，请先完成诊断与画像确认")
        if not session.get("profile_confirmed", False):
            raise HTTPException(status_code=409, detail="Profile must be confirmed before starting a stage")

        result = await harness.start_stage(session_id)

        # 更新 session 状态
        harness.state_manager.update_field(session_id, "phase", "learning")
        harness.state_manager.update_field(session_id, "available_actions", ["ask_question", "start_assessment"])

        # 获取当前阶段信息（含个性化字段）
        session = harness.get_session(session_id)
        stage = session.get("stage_info", {})

        return {
            "session_id": session_id,
            "stage_id": result.get("stage_id", stage.get("stage_id")),
            "status": "started",
            "phase": "learning",
            "package_id": result.get("package_id"),
            "available_actions": ["ask_question", "start_assessment"],
            # 个性化字段
            "stage_info": {
                "recommended_daily_hours": stage.get("recommended_daily_hours"),
                "suggested_study_days": stage.get("suggested_study_days"),
                "learning_style_notes": stage.get("learning_style_notes"),
                "hardware_notes": stage.get("hardware_notes"),
                "personalization_summary": stage.get("personalization_summary"),
            },
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def submit_stage_answer(
    session_id: str,
    request: AnswerRequest,
):
    """
    提交题目答案

    POST /api/v1/sessions/{session_id}/stages/current/answer
    """
    try:
        # 先检查会话是否存在
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        result = await harness.submit_answer(session_id, request.question_id, request.answer)
        if not result:
            raise HTTPException(status_code=404, detail="题目或学习包不存在")

        # 文档要求：correct_answer 等敏感字段只能在完整测试提交后由 AssessmentService 返回
        # 单题提交只返回对错结果，不泄露正确答案
        return {
            "question_id": request.question_id,
            "is_correct": result.get("is_correct", False),
            "message": "回答正确！" if result.get("is_correct") else "回答不正确，请继续努力",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def complete_current_stage(
    session_id: str,
    request: CompleteStageRequest = None,
):
    """
    完成当前阶段

    POST /api/v1/sessions/{session_id}/stages/current/complete
    """
    try:
        # 先检查会话是否存在
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 检查阶段是否已开始：只有 learning 阶段才能 complete
        phase = session.get("phase", "")
        if phase != "learning":
            detail_map = {
                "diagnosing": "当前处于诊断阶段，请先完成诊断并确认画像后开始学习",
                "stage_ready": "当前阶段尚未开始，请先调用 POST /stages/current/start 开始学习",
                "stage_result": "阶段已完成，无需重复提交",
                "next_stage_ready": "下一阶段已就绪，请 start 后再 complete",
            }
            msg = detail_map.get(phase, "当前状态不允许完成阶段")
            raise HTTPException(status_code=409, detail=msg)

        result = await harness.complete_stage(session_id)

        # 更新 session 状态
        phase = result.get("phase", "stage_result")
        harness.state_manager.update_field(session_id, "phase", phase)
        harness.state_manager.update_field(session_id, "available_actions", result.get("available_actions", ["plan_next_stage"]))

        # 获取当前阶段信息以补充 stage_id
        session = harness.get_session(session_id)
        stage = session.get("stage_info", {}) if session else {}

        return {
            "session_id": session_id,
            "stage_id": stage.get("stage_id"),
            "status": result.get("status", "completed"),
            "phase": phase,
            "accuracy": result.get("accuracy"),
            "stage_summary": result.get("stage_summary"),
            "next_strategy": result.get("next_strategy"),
            "can_plan_next_stage": result.get("can_plan_next_stage", False),
            "all_mastered": result.get("all_mastered", False),
            "final_report": result.get("final_report"),
            "available_actions": result.get("available_actions", ["plan_next_stage"]),
            "message": "阶段完成" if result.get("all_mastered") else "阶段评估完成，可规划下一阶段",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/next/plan")
async def plan_next_stage(
    session_id: str,
    request: PlanNextStageRequest,
):
    """
    规划下一阶段

    POST /api/v1/sessions/{session_id}/stages/current/plan-next
    """
    try:
        # 先检查会话是否存在
        session = harness.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # 检查状态：必须先 complete 得到评估结果后才能 plan-next
        phase = session.get("phase", "")
        if phase not in ("stage_result", "next_stage_ready"):
            detail_map = {
                "diagnosing": "当前处于诊断阶段，请先完成诊断并确认画像、开始并完成当前阶段",
                "stage_ready": "当前阶段尚未开始，请先 start 再 complete 后规划下一阶段",
                "learning": "当前阶段仍在学习中，请先完成（POST /complete）后再规划下一阶段",
            }
            msg = detail_map.get(phase, "请先完成当前阶段（POST /complete）后再规划下一阶段")
            raise HTTPException(status_code=409, detail=msg)

        result = await harness.plan_next_stage(session_id, request.adjustment_notes)
        
        # 更新 session 状态
        phase = result.get("phase", "next_stage_ready")
        harness.state_manager.update_field(session_id, "phase", phase)
        harness.state_manager.update_field(session_id, "available_actions", result.get("available_actions", ["start_stage"]))
        
        # 获取下一阶段信息（含个性化字段）
        next_stage = result.get("next_stage", {})
        
        return {
            "session_id": session_id,
            "next_stage_id": next_stage.get("stage_id"),
            "status": "planned",
            "phase": phase,
            "next_stage": next_stage,
            "updated_profile": result.get("updated_profile"),
            "planning_notes": result.get("planning_notes"),
            "available_actions": result.get("available_actions", ["start_stage"]),
            "message": "下一阶段规划完成",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
