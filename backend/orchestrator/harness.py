# -*- coding: utf-8 -*-
import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.deterministic import (
    DeterministicContentAgent,
    DeterministicDiagnosticAgent,
    DeterministicStageDecisionAgent,
    DeterministicTechnicalReviewAgent,
    HybridContentAgent,
    HybridDiagnosticAgent,
    HybridTechnicalReviewAgent,
)
from config import settings
from models.contracts import (
    AnswerGenerationInput,
    AnswerRevisionInput,
    ContentGenerationInput,
    ContentRevisionInput,
    DiagnosticInput,
    ReviewInput,
    ReviewOutput,
    StageDecisionInput,
    StagePlanningInput,
)
from models.profile import PracticeProfile, UserProfile
from models.review import QualityReviewResult, ReviewError, ReviewRound, TechnicalReviewResult
from models.stage import LearningPackage, Source, StageInfo
from models.material import TestQuestion
from orchestrator.state_manager import StateManager

# 审核辩论协议常量（协议第 7 节）
MAX_DEBATE_ROUNDS = 2  # 内层答辩复核轮次上限

# 块级流水线并发上限：Agent3 生成/修改、Agent4 复核均限制为同时 4 个块在飞，
# 兼顾低配设备（不做成 settings 开关——属内部调度细节，非运维旋钮）。
BLOCK_CONCURRENCY = 4


@dataclass
class BlockState:
    """块级流水线中单个内容块的运行态（可变、不落库）。

    - open_issues 为空 ⇒ 该块已 done；
    - block.version 即版本追踪（modify_block 自增），无需另存；
    - rounds_sent 记录该块被交还 Agent3 的次数（每块二级防死循环）。
    - last_issue_fp 记录上一轮「修改+复核后」的 open_issues 指纹，用于收敛检测：
      若某块连续两轮复核后的问题集完全一致（同一批注定改不动的问题原地打转），
      即判定该块无法收敛，冻结它、不再对它继续 modify+recheck，省掉注定失败的轮次。
    - frozen 为 True ⇒ 该块已被收敛检测冻结，直接留待人工，不再进入修改队列。
    """
    cid: str
    block: Any
    open_issues: List[ReviewError]
    last_review: TechnicalReviewResult
    rounds_sent: int = 0
    recheck_task: Optional[asyncio.Task] = None
    last_issue_fp: Optional[frozenset] = None
    frozen: bool = False


def _issue_fingerprint(issues: List[ReviewError]) -> frozenset:
    """把一组 issue 归一化为稳定指纹（用于收敛检测「同一批问题原地打转」）。

    以「问题种类」为身份，而非 canonical_error_id：后者的 seed 含 description，
    而认知负荷/难度这类质量问题的描述常嵌入随正文变化的数值（如负荷分数、字数），
    导致同一类改不动的问题每轮 canonical_error_id 都在漂移、永远判不出收敛。
    故收敛检测只关心「这一类问题（可细到绑定的 claim）是否还在」：
      key = error_type（+ claim_id，若有）
    同一批 error_type 集合连续两轮不变，即认定该块无法靠修订继续收敛。
    """
    keys = set()
    for e in issues or []:
        et = getattr(e, "error_type", "") or ""
        claim_id = getattr(e, "claim_id", None) or ""
        keys.add(f"{et}|{claim_id}" if claim_id else et)
    return frozenset(keys)



class Harness:
    def __init__(self, state_manager: StateManager | None = None, execution_mode: str | None = None):
        self.state_manager = state_manager or StateManager()
        self.execution_mode = execution_mode or settings.EXECUTION_MODE
        if self.execution_mode not in {"deterministic", "hybrid"}:
            raise ValueError("EXECUTION_MODE must be 'deterministic' or 'hybrid'")
        self.agent1 = (
            HybridDiagnosticAgent(db_client=self.state_manager.db_client)
            if self.execution_mode == "hybrid"
            else DeterministicDiagnosticAgent(db_client=self.state_manager.db_client)
        )
    
        self.agent3 = (
            HybridContentAgent(database_url=self.state_manager.db_client.database_url)
            if self.execution_mode == "hybrid"
            else DeterministicContentAgent(database_url=self.state_manager.db_client.database_url)
        )
        self.agent4 = HybridTechnicalReviewAgent() if self.execution_mode == "hybrid" else DeterministicTechnicalReviewAgent()
        # Agent5已合并至Agent4，质量审核功能已集成
        self.agent6 = DeterministicStageDecisionAgent()
        # Short-lived event buffer for the generation modal. These records are
        # operational evidence only and never become learner-facing content.
        self._generation_events: Dict[str, List[Dict[str, Any]]] = {}
        self._generation_event_seq: Dict[str, int] = {}

    def _reset_generation_events(self, session_id: str) -> None:
        self._generation_events[session_id] = []
        self._generation_event_seq[session_id] = 0

    def get_generation_events(self, session_id: str, after: int = 0) -> Dict[str, Any]:
        events = self._generation_events.get(session_id, [])
        return {
            "events": [item for item in events if item["id"] > after],
            "latest_id": self._generation_event_seq.get(session_id, 0),
        }

    def _log(self, session_id: str, event: str, detail: str = "") -> None:
        suffix = f" | {detail}" if detail else ""
        print(f"[Harness][session={session_id}] {event}{suffix}", flush=True)
        event_id = self._generation_event_seq.get(session_id, 0) + 1
        self._generation_event_seq[session_id] = event_id
        events = self._generation_events.setdefault(session_id, [])
        events.append({
            "id": event_id,
            "timestamp": int(time.time() * 1000),
            "event": event,
            "detail": detail,
        })
        if len(events) > 120:
            del events[:-120]

    @staticmethod
    def _extract_user_facing_answer(raw_answer: str) -> str:
        """Keep model prompt examples and review traces out of the learner UI."""
        text = (raw_answer or "").strip()
        if not text:
            return ""

        if text.startswith("Error: DeepSeek API authentication failed"):
            return "当前 DeepSeek 配置无效：API Key 认证失败（HTTP 401）。请检查 backend/.env 中的 DEEPSEEK_API_KEY。"
        if text.startswith("Error: DeepSeek API access denied"):
            return "当前 DeepSeek 请求被拒绝（HTTP 403）。请检查账号权限、接口地址或服务开通状态。"
        if text.startswith("Error: DeepSeek API endpoint or model is invalid"):
            return "当前 DeepSeek 配置无效：接口地址或模型名不可用（HTTP 404）。请检查 DEEPSEEK_API_BASE 和 DEEPSEEK_MODEL。"
        if text.startswith("Error: DeepSeek API rate limit exceeded"):
            return "当前 DeepSeek 请求过于频繁（HTTP 429），请稍后重试。"
        if text.startswith("Error: DeepSeek API request failed"):
            return "当前 DeepSeek 请求失败，请检查网络连通性或接口配置。"
        if text.startswith("Error: DeepSeek API transport failure"):
            return "当前 DeepSeek 云端连接不稳定，系统已自动重试仍未成功。请稍后重新提问。"
        if text.startswith("Error: DeepSeek returned empty content"):
            return "当前 DeepSeek 未返回可展示的最终回答，系统已自动重试。请稍后重新提问。"
        if text.startswith("Error: DeepSeek API response format invalid"):
            return "当前 DeepSeek 返回格式异常，系统已自动重试。请稍后重新提问。"

        # The deterministic emergency path used to concatenate retrieval
        # excerpts, history, and profile context. Those are prompt inputs,
        # never a learner-facing answer.
        internal_context_markers = (
            "可以先参考这些与问题相关的知识片段",
            "最近对话记录：",
            "学习画像：",
        )
        if all(marker in text for marker in internal_context_markers):
            return "当前回答服务暂时不可用，请稍后重试。"

        # Agent3 may occasionally expose the revision workflow before the
        # actual answer, e.g. "已根据审核意见修复……以下是修改后的完整回答：" or
        # "修改后内容如下：" / "修订后答复：". That is internal orchestration
        # metadata, not learner-facing content, so strip the leading marker.
        # 载体词（回答/内容/答复/版本）必选，避免正文里"最终结果："被误匹配而剥掉前半段。
        revision_answer_markers = re.findall(
            r"(?:以下是|下面是)?(?:的)?"
            r"(?:修改后|修复后|修订后|更新后|最终)(?:的)?(?:完整)?"
            r"(?:回答|内容|答复|版本)\s*(?:如下|请见)?\s*[：:]",
            text,
            flags=re.IGNORECASE,
        )
        if revision_answer_markers:
            marker = revision_answer_markers[-1]
            candidate = text.rsplit(marker, 1)[-1].strip()
            if candidate:
                text = candidate

        # Some local models copy the JSON example embedded in a skill file.
        # Prefer the final fenced JSON answer when it is present.
        for block in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)):
            try:
                value = json.loads(block)
            except json.JSONDecodeError:
                continue
            answer = value.get("answer") if isinstance(value, dict) else None
            if isinstance(answer, str) and answer.strip():
                return answer.strip()

        # Also accept an unfenced JSON object without exposing its wrapper.
        matches = re.findall(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
        for value in reversed(matches):
            try:
                answer = json.loads(f'"{value}"')
            except json.JSONDecodeError:
                answer = value.replace(r"\\n", "\n").replace(r'\\"', '"')
            if answer.strip():
                return answer.strip()

        return text

    @staticmethod
    def _is_greeting(question: str) -> bool:
        return bool(re.fullmatch(r"(?:\u4f60\u597d|\u60a8\u597d|\u55e8|\u54c8\u558e|hello|hi)[!\uff01.\u3002,\uff0c?\uff1f\s]*", question.strip(), flags=re.IGNORECASE))

    @staticmethod
    def _greeting_answer(stage_name: str) -> str:
        return (
            f"你好，我是你的实训导师。你当前正在学习「{stage_name or '当前阶段'}」。"
            "可以直接问我概念、实操步骤或当前学习材料中的疑问。"
        )

    async def process_initial_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        self._log(session_id, "process_initial_input:start", f"message={user_input[:60]}")
        self.state_manager.require_session(session_id)
        previous_inputs = []
        for interaction in self.state_manager.get_interactions(session_id):
            if interaction.get("type") != "initial_diagnostic":
                continue
            data = interaction.get("data", {})
            previous_input = data.get("user_input") if isinstance(data, dict) else None
            if previous_input:
                previous_inputs.append(previous_input)
        diagnostic_input = "\n".join([*previous_inputs, user_input])
        diagnostic = await self.agent1.run(
            DiagnosticInput(session_id=session_id, user_input=diagnostic_input)
        )
        user_id = self.state_manager.db_client.get_session_user_id(session_id)
        reusable_profile = self.state_manager.db_client.get_user_profile(user_id)
        if reusable_profile:
            # New goals only need target-specific input. Stable learner facts
            # are reused from the single user-level profile.
            diagnostic.profile.user_id = user_id
            diagnostic.profile.grade = reusable_profile.get("grade", diagnostic.profile.grade)
            diagnostic.profile.skill_tags = reusable_profile.get("skill_tags", diagnostic.profile.skill_tags)
            diagnostic.profile.learning_style = reusable_profile.get("learning_style", diagnostic.profile.learning_style)
            diagnostic.profile.learning_rhythm = reusable_profile.get("learning_rhythm", diagnostic.profile.learning_rhythm)
            diagnostic.profile.confidence_index = reusable_profile.get("confidence_index", diagnostic.profile.confidence_index)
            diagnostic.profile.confusion_markers = reusable_profile.get("confusion_markers", diagnostic.profile.confusion_markers)
            diagnostic.profile.confidence = max(
                diagnostic.profile.confidence,
                reusable_profile.get("confidence", 0.7),
            )
            if diagnostic.profile.daily_hours <= 0:
                diagnostic.profile.daily_hours = reusable_profile.get("default_daily_hours", 2.0)
        else:
            diagnostic.profile.user_id = user_id
        # 无论置信度是否达标，都保存 profile（草稿），允许用户通过 PATCH /profile 修改
        self.state_manager.update_profile(session_id, diagnostic.profile)
        # PracticeProfile remains Agent3's execution DTO, while hardware has a
        # single durable source in learner_profiles. Only first-time input may
        # seed that source; later goals always reuse it.
        if not reusable_profile:
            inferred_practice_profile = PracticeProfile.from_user_input(user_input)
            if inferred_practice_profile.hardware:
                self.state_manager.db_client.upsert_user_profile(
                    user_id, {"hardware": inferred_practice_profile.hardware}
                )
        self.state_manager.refresh_practice_profile(session_id)
        self.state_manager.add_interaction(session_id, "initial_diagnostic", {"user_input": user_input})
        if diagnostic.profile.confidence < settings.CONFIDENCE_THRESHOLD:
            self._log(
                session_id,
                "process_initial_input:needs_more_info",
                f"confidence={diagnostic.profile.confidence:.2f}",
            )
            return {
                "status": "needs_more_info",
                "questions": diagnostic.followup_questions,
                "profile": diagnostic.profile.model_dump(mode="json"),
            }

        stage = await self.agent1.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=diagnostic.profile,
            stage_num=1,
        ))
        self.state_manager.update_stage(session_id, stage.stage)
        self._log(
            session_id,
            "process_initial_input:ready",
            f"goal={diagnostic.profile.goal}, stage={stage.stage.block_name}",
        )
        return {
            "status": "ready",
            "profile": diagnostic.profile.model_dump(mode="json"),
            "stage_info": stage.stage.model_dump(mode="json"),
        }

    async def start_stage(self, session_id: str) -> Dict[str, Any]:
        self._reset_generation_events(session_id)
        # 若已通过 plan-next 规划过 next_stage，则在真正启动阶段前把 next_stage 切换为当前 stage_info
        planned = self.state_manager.get_next_stage(session_id)
        if planned:
            self._log(session_id, "start_stage:apply_planned_next_stage", f"planned_stage_num={planned.get('stage_num')}, block={planned.get('block_name')}")
            self.state_manager.update_stage(session_id, StageInfo.model_validate(planned))
            # 切换完成后清理 next_stage / planning_notes，避免重复套用
            self.state_manager.clear_next_stage(session_id)

        stage = StageInfo.model_validate(self._require_stage(session_id))
        profile = UserProfile.model_validate(self._require_profile(session_id))
        practice_profile = PracticeProfile.model_validate(
            self.state_manager.get_practice_profile(session_id)
        )
        self._log(session_id, "start_stage:start", f"stage={stage.stage_num}, block={stage.block_name}")
        if settings.BLOCK_PIPELINE_ENABLED:
            # 块级并行流水线：Agent3 逐块生成、Agent4 逐块审核并行推进（任务 5）。
            learning_package, review_history = await self._review_learning_package_pipeline(
                session_id, stage, profile, practice_profile,
            )
        else:
            generated = await self.agent3.run(ContentGenerationInput(
                session_id=session_id,
                stage=stage,
                learning_profile=profile,
                practice_profile=practice_profile,
            ))
            learning_package, review_history = await self._review_learning_package(
                session_id,
                generated.learning_package,
                stage,
                profile,
            )
        # 从 _review_learning_package 获取审核 trace 信息（P3）
        review_trace = getattr(self, "_last_review_trace", None) or {}
        final_status = review_trace.get("final_status", "passed")
        package_status = review_trace.get("package_status", "approved")
        # 拍板：needs_human_review 下学习包降级可用（A 方案），不阻塞学习流
        review_passed = final_status in ("passed", "needs_human_review")
        needs_human_review = final_status == "needs_human_review"

        stage.status = "learning" if review_passed else "pending"
        self.state_manager.update_stage(session_id, stage)
        if review_passed:
            stage_subtopic_ids = list(dict.fromkeys(stage.subtopic_ids or []))
            if not stage_subtopic_ids and stage.subtopic_id:
                stage_subtopic_ids = [stage.subtopic_id]
            for subtopic_id in stage_subtopic_ids:
                self.state_manager.db_client.set_subtopic_status(
                    session_id=session_id,
                    subtopic_id=subtopic_id,
                    status=0,
                )
        self.state_manager.update_learning_package(session_id, learning_package)
        self.state_manager.add_interaction(session_id, "stage_review_completed", {
            "stage_id": stage.stage_id,
            "passed": review_passed,
            "iterations": len(review_history),
            "review_trace_id": review_trace.get("trace_id"),
            "final_status": final_status,
            "needs_human_review": needs_human_review,
        })
        self._log(
            session_id,
            "start_stage:done",
            f"passed={review_passed}, iterations={len(review_history)}, "
            f"final_status={final_status}, needs_human_review={needs_human_review}",
        )
        return {
            "learning_package": learning_package.model_dump(mode="json"),
            "review_passed": review_passed,
            "iterations": len(review_history),
            "review_history": [item.model_dump(mode="json") for item in review_history],
            # P3：审核辩论协议返回字段（前端可据此展示"待审"标记）
            "review_trace_id": review_trace.get("trace_id"),
            "package_status": package_status,
            "needs_human_review": needs_human_review,
            "unresolved_issue_count": review_trace.get("unresolved_issue_count", 0),
            "unresolved_must_fix_count": review_trace.get("unresolved_must_fix_count", 0),
        }

    async def handle_question(
        self, session_id: str, question: str, stage_num: int | None = None
    ) -> Dict[str, Any]:
        current_stage = StageInfo.model_validate(self._require_stage(session_id))
        requested_stage_num = stage_num if stage_num is not None else current_stage.stage_num
        snapshot = self.state_manager.get_stage_snapshot(session_id, requested_stage_num)
        if not snapshot:
            raise ValueError(f"Stage not found: {requested_stage_num}")
        stage = StageInfo.model_validate(snapshot["stage_info"])
        profile = UserProfile.model_validate(self._require_profile(session_id))
        practice_profile = PracticeProfile.model_validate(
            self.state_manager.get_practice_profile(session_id)
        )
        self._log(session_id, "handle_question:start", f"question={question[:60]}")
        is_greeting = self._is_greeting(question)

        # === 构建 memory 上下文（供 Agent3 生成更贴合用户的回答）===
        # 当前阶段 block_id，用于技术类回答时检索本阶段相关的知识文档
        block_id = stage.block_id if stage else None
        # 历史问答记录（过滤非 dict 项，防止 Pydantic 验证失败）
        raw_qa = snapshot.get("qa_history", [])
        qa_history = [item for item in raw_qa if isinstance(item, dict)]
        # 用户画像摘要文本
        profile_summary = self._build_profile_summary(profile)

        if is_greeting:
            raw_answer = self._greeting_answer(stage.block_name)
            question_type = "daily"  # 问候属于日常类，不走 Agent4
        elif self._is_profile_fact_question(question):
            # 画像事实查询不应交给知识库检索或通用闲聊模型猜测；直接读取
            # 实践画像中的结构化设备字段，保证“我的设备型号是什么”这类问题稳定可答。
            model = (getattr(practice_profile, "gpu_model", "") or "").strip()
            hardware = (getattr(practice_profile, "hardware", "") or "").strip()
            device_type = (getattr(practice_profile, "device_type", "") or "").strip()
            vram = getattr(practice_profile, "vram_gb", None)
            if model:
                detail = f"（{vram}GB 显存）" if vram else ""
                raw_answer = f"你的设备型号是 {model}{detail}。"
            elif hardware:
                raw_answer = f"你当前画像中记录的设备条件是：{hardware}。"
            elif device_type:
                raw_answer = f"你当前画像中记录的设备类型是：{device_type}。"
            else:
                raw_answer = "你的用户画像中暂未记录设备型号，可以在设置中补充。"
            question_type = "profile_fact"
        else:
            generated = await self.agent3.generate_answer(AnswerGenerationInput(
                session_id=session_id,
                question=question,
                stage=stage,
                profile=profile,
                practice_profile=practice_profile,
                block_id=block_id,
                qa_history=qa_history,
                profile_summary=profile_summary,
            ))
            raw_answer = generated.answer
            question_type = getattr(generated, "question_type", "technical")

        # === 根据问题类型决定是否调用 Agent4 审核 ===
        if question_type in {"daily", "profile_fact"}:
            # 日常类：跳过 Agent4 审核，直接使用 Agent3 的回答
            self._log(session_id, "handle_question:daily_skip_review", f"question_type={question_type}")
            answer = self._extract_user_facing_answer(raw_answer)
            # 构造一个"通过"的审核结果，保持返回体结构一致
            from models.review import ReviewRound, TechnicalReviewResult, QualityReviewResult
            passed_round = ReviewRound(
                iteration=1,
                technical=TechnicalReviewResult(passed=True, errors=[]),
                quality=QualityReviewResult(passed=True, errors=[]),
            )
            review_history = [passed_round]
            review_summary = {
                "status": "skipped",
                "technical_passed": True,
                "quality_passed": True,
                "iterations": 0,
                "needs_human_review": False,
                "reason": "profile_fact_skipped_agent4" if question_type == "profile_fact" else "daily_question_skipped_agent4",
            }
        else:
            # 技术类：走原 Agent4 审核流程
            answer, review_history = await self._review_answer(
                session_id,
                question,
                self._extract_user_facing_answer(raw_answer),
                stage,
                profile,
                practice_profile=practice_profile,
            )
            # A greeting is a conversational turn, not a technical answer. Agent4
            # may classify it as off-topic and ask Agent3 to rewrite it; preserve
            # the deterministic greeting instead of exposing that rewrite.
            answer = self._extract_user_facing_answer(raw_answer if is_greeting else answer)
            latest_review = review_history[-1]
            review_summary = self._build_answer_review_summary(
                latest_review.technical,
                iterations=len(review_history),
            )
            # 库外/超纲问题：直接输出 Agent3 内容，向前端呈现为非失败的独立状态，
            # 避免把「知识库未覆盖」误报成审核未通过。
            if self._is_out_of_scope_review(latest_review.technical):
                review_summary = {
                    **review_summary,
                    "status": "out_of_scope",
                    "technical_passed": True,
                    "needs_human_review": False,
                    "reason": "question_out_of_knowledge_base",
                }
            self._log(
                session_id,
                "handle_question:done",
                f"tech_pass={latest_review.technical.passed}, quality_pass={latest_review.quality.passed}, "
                f"review_status={review_summary['status']}, needs_human_review={review_summary['needs_human_review']}",
            )

        latest_review = review_history[-1]
        self.state_manager.add_qa_record(
            session_id,
            question,
            answer,
            latest_review.technical.passed,
            latest_review.quality.passed,
            stage_num=stage.stage_num,
        )
        self.state_manager.add_interaction(session_id, "question_asked", {
            "question": question,
            "question_type": question_type,
            "stage_num": stage.stage_num,
        })
        return {
            "answer": answer,
            "question_type": question_type,
            "tech_pass": latest_review.technical.passed,
            "quality_pass": latest_review.quality.passed,
            "iterations": len(review_history),
            "review_history": [item.model_dump(mode="json") for item in review_history],
            # 第 1 节答疑审核摘要（前端 review_summary 字段来源）
            "review_summary": review_summary,
        }

    @staticmethod
    def _build_profile_summary(profile: UserProfile) -> str:
        """将用户画像转为摘要文本，供 Agent3 日常类回答使用。"""
        parts = []
        if getattr(profile, "goal", None):
            parts.append(f"学习目标：{profile.goal}")
        if getattr(profile, "grade", None):
            parts.append(f"水平：{profile.grade}")
        if getattr(profile, "daily_hours", None):
            parts.append(f"每阶段学习时长：{profile.daily_hours}小时")
        if getattr(profile, "learning_style", None):
            parts.append(f"学习风格：{profile.learning_style}")
        if getattr(profile, "hardware", None):
            parts.append(f"硬件条件：{profile.hardware}")
        # 知识盲区
        gaps = getattr(profile, "knowledge_gaps", [])
        if gaps:
            gap_skills = [g.skill for g in gaps if getattr(g, "skill", None)]
            if gap_skills:
                parts.append(f"知识盲区：{', '.join(gap_skills[:3])}")
        return "；".join(parts) if parts else ""

    @staticmethod
    def _is_profile_fact_question(question: str) -> bool:
        """识别只需读取用户画像的事实问题，避免被误判为知识库问答。"""
        text = (question or "").strip().lower()
        return bool(text) and any(keyword in text for keyword in (
            "我的设备", "设备型号", "我的显卡", "我的硬件", "我的gpu",
            "what device", "what gpu", "my hardware", "device model",
        ))

    def _build_answer_review_summary(
        self,
        technical_result: TechnicalReviewResult,
        iterations: int = 1,
    ) -> Dict[str, Any]:
        """解析并调用 Agent4 的答疑审核摘要生成器（第 1 节 review_summary）。

        build_answer_review_summary 是无状态方法（仅由 TechnicalReviewResult + iterations
        推导 5 个固定字段），故按以下优先级健壮解析，避免对具体 reviewer 实现的强耦合：
          1. self.agent4._protocol_agent（真实 Deterministic/Hybrid 审核体持有的协议 Agent）
          2. self.agent4 本身直接暴露 build_answer_review_summary（直接注入 TechnicalReviewAgent）
          3. 兜底：用 TechnicalReviewAgent 的实现（stateless），兼容测试注入的精简/桩 reviewer
        """
        builder = None
        protocol_agent = getattr(self.agent4, "_protocol_agent", None)
        if protocol_agent is not None and hasattr(protocol_agent, "build_answer_review_summary"):
            builder = protocol_agent.build_answer_review_summary
        elif hasattr(self.agent4, "build_answer_review_summary"):
            builder = self.agent4.build_answer_review_summary
        if builder is None:
            from agents.agent_04_tech_review import TechnicalReviewAgent
            # build_answer_review_summary 不访问 self，直接以 None 作 self 调用底层函数即可，
            # 避免实例化 TechnicalReviewAgent（会加载全部工具链）。
            return TechnicalReviewAgent.build_answer_review_summary(
                None, technical_result, iterations=iterations
            )
        return builder(technical_result, iterations=iterations)

    async def submit_answer(self, session_id: str, question_id: str, user_answer: str) -> Dict[str, Any]:
        package = LearningPackage.model_validate(self._require_package(session_id))
        question = next((item for item in package.questions if item.question_id == question_id), None)
        if question is None:
            raise ValueError(f"Question not found in current stage: {question_id}")
        is_correct = self._match_answer(
            user_answer, question.correct_answer,
            question_type=getattr(question, 'question_type', '')
        )
        self.state_manager.add_answer_record(session_id, question_id, user_answer, is_correct)
        self._log(session_id, "submit_answer", f"question_id={question_id}, correct={is_correct}")
        return {"is_correct": is_correct, "correct_answer": question.correct_answer}

    def _resolve_stage_accuracy(self, session_id: str, stage: StageInfo) -> float:
        """确定本阶段用于 Agent6 决策的正确率（Agent46_API接口任务说明.md 第 2/3 节）。

        - §3 complete 流程要求「读取已提交测试的准确率」，§2 约束 7 要求正确率只能来自
          已提交的完整 attempt，不能从零散答题记录计算。
        - 因此优先取当前阶段已提交 attempt 的正确率；找不到时（例如仍走旧的 submit_answer
          零散答题链路）再退化为 StateManager.calculate_accuracy，保证向后兼容不阻断流程。
        """
        try:
            stage_id = getattr(stage, "stage_id", None)
            submitted = self.state_manager.db_client.get_submitted_attempt_accuracy(
                session_id, stage_id=stage_id
            )
            # 阶段维度没有已提交 attempt 时，退回会话维度最近一次已提交 attempt
            if submitted is None and stage_id is not None:
                submitted = self.state_manager.db_client.get_submitted_attempt_accuracy(session_id)
            if submitted and submitted.get("accuracy") is not None:
                self._log(
                    session_id,
                    "complete_stage:accuracy_source",
                    f"submitted_attempt={submitted.get('attempt_id')}, accuracy={submitted['accuracy']}",
                )
                return float(submitted["accuracy"])
        except Exception as exc:
            self._log(session_id, "complete_stage:accuracy_fallback", f"reason={exc}")
        # 兜底：旧的零散答题链路
        return self.state_manager.calculate_accuracy(session_id)

    def _resolve_stage_records(self, session_id: str, stage: StageInfo):
        """取本阶段已提交测试的逐题记录与聚合用时，供 Agent6 决策使用。

        - 与 _resolve_stage_accuracy 同源同口径（Agent46_API接口任务说明.md 第 2/3 节）：
          优先当前阶段已提交 attempt，退回会话维度最近一次已提交 attempt。
        - 逐题记录含 time_spent（前端 PUT .../answers/{qid} 提交的答题用时），
          聚合用时换算为小时透传给 Agent6，替代仅从 interactions 估算的耗时。
        - 无已提交 attempt 时返回 ([], 0.0)，Agent6 退化为旧的 interactions 估算，不阻断流程。
        """
        try:
            stage_id = getattr(stage, "stage_id", None)
            records = self.state_manager.db_client.get_submitted_attempt_records(
                session_id, stage_id=stage_id
            )
            if records and records.get("question_records"):
                total_seconds = float(records.get("total_time_spent", 0) or 0)
                time_spent_hours = round(total_seconds / 3600.0, 2)
                self._log(
                    session_id,
                    "complete_stage:records_source",
                    f"submitted_attempt={records.get('attempt_id')}, "
                    f"questions={len(records['question_records'])}, time_spent_hours={time_spent_hours}",
                )
                return records["question_records"], time_spent_hours
        except Exception as exc:
            self._log(session_id, "complete_stage:records_fallback", f"reason={exc}")
        return [], 0.0

    async def complete_stage(self, session_id: str, bypass_assessment: bool = False) -> Dict[str, Any]:
        """
        complete_stage 新职责（P3 按需阶段规划）：
        1. 只负责调用 Agent6 决策
        2. 保存 Agent6 更新后的用户画像 / 个性化数据
        3. 只有在全部知识盲区都掌握时，才真正结束 session
        4. 返回 status/session_id/accuracy/updated_profile/stage_summary/
           next_strategy/can_plan_next_stage/session_status/all_mastered/
           final_report/phase/available_actions
        5. 不直接推进当前阶段到下一个阶段编号，不生成 next_stage
        """
        profile = UserProfile.model_validate(self._require_profile(session_id))
        stage = StageInfo.model_validate(self._require_stage(session_id))
        # A stage without a matching assessment is explicitly completed by
        # the learner. Treat it as passed so planning advances to the next
        # knowledge point instead of reopening the same stage.
        accuracy = 1.0 if bypass_assessment else self._resolve_stage_accuracy(session_id, stage)
        assessment_records, time_spent_hours = self._resolve_stage_records(session_id, stage)
        self._log(session_id, "complete_stage:start", f"stage={stage.stage_num}, accuracy={accuracy:.2f}")
        decision = await self.agent6.run(StageDecisionInput(
            session_id=session_id,
            profile=profile,
            stage=stage,
            accuracy=accuracy,
            qa_history=self.state_manager.get_qa_history(session_id),
            interactions=self.state_manager.get_interactions(session_id),
            assessment_records=assessment_records,
            time_spent_hours=time_spent_hours,
        ))

        # 更新当前阶段的结果状态（只改当前 stage，不推进编号）
        # 通过阈值：正确率 ≥ 70% 即视为本阶段通过（用户要求）
        stage.status = "completed" if accuracy >= 0.7 else "retreat"
        stage.accuracy = accuracy
        stage.strategy_used = decision.strategy

        stage_subtopic_ids = list(getattr(stage, "subtopic_ids", []) or [])
        if not stage_subtopic_ids and stage.subtopic_id:
            stage_subtopic_ids = [stage.subtopic_id]
        for subtopic_id in stage_subtopic_ids:
            self.state_manager.db_client.set_subtopic_status(
                session_id=session_id,
                subtopic_id=subtopic_id,
                status=1 if accuracy >= 0.7 else 0,
                mastery_score=1.0 if accuracy >= 0.7 else max(0.4, round(accuracy, 3)),
                last_accuracy=accuracy,
                increment_attempt=True,
            )
            self.state_manager.db_client.refresh_target_metrics(session_id)
            self.state_manager.refresh_cached_target(session_id)

        synced_profile = await self._sync_profile_from_subtopic_status(decision.profile)
        # A stage can be an inserted prerequisite node for a later target
        # block.  Its ``gap_skill`` still names that target, so marking the
        # whole gap as mastered here would prematurely end the session.
        # ``_sync_profile_from_subtopic_status`` is the sole authority for
        # block completion and only reports mastery after every subtopic in
        # the target block has actually been completed.
        self.state_manager.update_stage(session_id, stage)
        self.state_manager.update_profile(session_id, synced_profile)

        # 只有全部掌握才真正结束 session
        all_mastered = all(gap.progress >= 1.0 for gap in synced_profile.knowledge_gaps)
        if all_mastered:
            self.state_manager.end_session(session_id)
            self._log(session_id, "complete_stage:all_mastered -> end_session")
            final_report = self._generate_final_report(synced_profile, accuracy)
        else:
            # 未全部掌握：session 保持 active，等用户按需触发 plan-next
            final_report = None
            self._log(
                session_id,
                f"complete_stage:keep_active | accuracy={accuracy:.2f} | all_mastered=False",
            )

        # 可规划下一阶段的条件：session 仍 active 即可；即便 repeat 场景也允许重新规划
        session_state = self.state_manager.get_session(session_id)
        session_status = session_state.get("status") if session_state else "active"
        can_plan_next_stage = not all_mastered and session_status == "active"

        result_status = "completed" if all_mastered else ("repeat" if accuracy < 0.7 else "continue")
        # 阶段结果的 phase / available_actions（Agent46_API接口任务说明.md 第 3 节）：
        # - 结业（all_mastered）：phase=completed，无后续动作
        # - 未结业：phase=stage_result，开放 plan_next_stage（前端据此展示“生成下一阶段”）
        if all_mastered:
            phase = "completed"
            available_actions: List[str] = []
        else:
            phase = "stage_result"
            available_actions = ["plan_next_stage"] if can_plan_next_stage else []
        result = {
            "status": result_status,
            "session_id": session_id,
            "stage_id": stage.stage_id,
            "stage_num": stage.stage_num,
            "accuracy": accuracy,
            "bypass_assessment": bypass_assessment,
            "updated_profile": synced_profile.model_dump(mode="json"),
            "stage_summary": decision.stage_summary,
            "next_strategy": decision.strategy,
            "can_plan_next_stage": can_plan_next_stage,
            "session_status": session_status,
            "all_mastered": all_mastered,
            "final_report": final_report,
            "phase": phase,
            "available_actions": available_actions,
        }
        # The result view must remain available after a refresh or after a
        # later stage becomes active, so do not leave this only in the HTTP response.
        self.state_manager.save_stage_result(session_id, stage.stage_num, result)
        return result

    async def plan_next_stage(
        self,
        session_id: str,
        adjustment_notes: str | None = None,
        force_advance: bool = False,
    ) -> Dict[str, Any]:
        """
        按需生成下一阶段（plan-next 接口）：
        - 基于 complete_stage 后最新的画像，调用 Agent1 的规划链路
        - 生成真正的 next_stage 并保存到 session state
        - 不立刻把 next_stage 覆盖到当前 stage_info（留待 start_stage 或前端确认后切换）

        stage_num 选择规则：
        - accuracy >= 0.7（已掌握本阶段，正确率 ≥ 70%）：stage_num = current_stage + 1
        - accuracy < 0.7（retreat/repeat）：stage_num = current_stage（重试当前阶段）
        """
        profile = UserProfile.model_validate(self._require_profile(session_id))
        current_stage_info = StageInfo.model_validate(self._require_stage(session_id))
        self._log(
            session_id,
            "plan_next_stage:start",
            f"current_stage={current_stage_info.stage_num}, accuracy={current_stage_info.accuracy or 0:.2f}",
        )

        accuracy = current_stage_info.accuracy or 0.0
        if force_advance or accuracy >= 0.7:
            next_stage_num = current_stage_info.stage_num + 1
            default_notes = f"本阶段（{current_stage_info.stage_num}）已达标，推进到下一阶段"
        else:
            next_stage_num = current_stage_info.stage_num
            default_notes = f"本阶段（{current_stage_info.stage_num}）正确率不足 70%，重试当前阶段"

        user_notes = adjustment_notes.strip() if adjustment_notes and adjustment_notes.strip() else None
        planning_notes = (
            f"{default_notes}；{user_notes}" if user_notes else default_notes
        )

        planned = await self.agent1.generate_stage(StagePlanningInput(
            session_id=session_id,
            profile=profile,
            stage_num=next_stage_num,
            adjustment_notes=user_notes,
            excluded_subtopic_ids=(
                [current_stage_info.subtopic_id]
                if force_advance and current_stage_info.subtopic_id
                else []
            ),
        ))
        next_stage = planned.stage
        # A below-threshold result is a retry of the same stage instance, not
        # a second stage with the same number. Reuse the immutable stage_id so
        # stage_progress and snapshots update in place and cannot hide an
        # earlier knowledge point (for example installation -> syntax).
        if next_stage_num == current_stage_info.stage_num and not force_advance:
            next_stage.stage_id = current_stage_info.stage_id
        self.state_manager.update_next_stage(session_id, next_stage)
        self.state_manager.update_planning_notes(session_id, planning_notes)

        session_state = self.state_manager.get_session(session_id)
        session_status = session_state.get("status") if session_state else "active"
        self._log(
            session_id,
            "plan_next_stage:done",
            f"next_stage_num={next_stage.stage_num}, block={next_stage.block_name}",
        )
        return {
            "status": "planned",
            "session_id": session_id,
            "next_stage": next_stage.model_dump(mode="json"),
            "updated_profile": profile.model_dump(mode="json"),
            "planning_notes": planning_notes,
            "session_status": session_status,
            # plan-next 成功后的 phase / available_actions（Agent46_API接口任务说明.md 第 3 节）
            "phase": "next_stage_ready",
            "available_actions": ["start_stage"],
        }

    async def _review_learning_package(
        self,
        session_id: str,
        package: LearningPackage,
        stage: StageInfo,
        profile: UserProfile,
    ) -> tuple[LearningPackage, list[ReviewRound]]:
        """审核学习包：审核 → 答辩 → 复核 → 裁决（协议第 4~5 节）。

        外层 review iteration（MAX_REVIEW_ITERATIONS）内嵌 debate 子循环。
        每个 iteration = 一次完整辩论轮（Agent4 审核 → Agent3 答辩 → Agent4 复核 → Orchestrator 裁决 → Agent3 修改）。
        达到最大轮次仍有未决 must_fix → needs_human_review（非阻塞，package 降级为 provisional）。
        """
        history: list[ReviewRound] = []
        current_package = package
        # 循环外创建一次 trace_id，全程复用（协议：同一 trace 内迭代）
        trace_id = f"tr_{uuid.uuid4().hex[:16]}"
        current_package.review_trace_id = trace_id
        final_status = "in_progress"
        package_status = "draft"

        # 落库：创建 trace + 首轮 sources + claims（P2）
        self._persist_trace_init(trace_id, session_id, stage, current_package)

        for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
            print(f"=== 学习包审核循环：第{iteration}轮（trace={trace_id}）===")

            # 步骤1：Agent4 首轮审核（传 sources/claims/trace_id/round_index）
            reviews = await self._review(ReviewInput(
                session_id=session_id,
                target_type="learning_package",
                content=current_package.tutorial,
                stage=stage,
                profile=profile,
                code_examples=current_package.code_examples,
                trace_id=trace_id,
                sources=[s.model_dump(mode="json") for s in current_package.sources],
                claims=[c.model_dump(mode="json") for c in current_package.claims],
                recommended_videos=current_package.recommended_videos,
                round_index=iteration,
            ))

            # 如果审核通过，直接结束
            if reviews.passed:
                round_result = ReviewRound(
                    iteration=iteration,
                    technical=reviews.technical,
                    quality=reviews.quality,
                )
                history.append(round_result)
                self._record_review_round(session_id, "learning_package", round_result)
                final_status = "passed"
                package_status = "approved"
                print(f"=== 学习包第{iteration}轮审核通过 ===")
                break

            # 步骤2：Agent3 答辩（对每条 issue 逐条回复）
            issues = [e.model_dump(mode="json") for e in reviews.technical.errors]
            package_dict = current_package.model_dump(mode="json")
            print(f"=== 调用 Agent3.reply_to_review（{len(issues)} 条 issue）===")
            author_replies = await self.agent3.reply_to_review(
                session_id, issues, package_dict
            )

            # 步骤3：Agent4 复核裁决（对 Agent3 的回复做 persist/withdraw/downgrade）
            print(f"=== 调用 Agent4.rebut ===")
            verdicts = await self.agent4.rebut(session_id, issues, author_replies)
            print(f"=== Agent4 复核裁决结果：{verdicts} ===")

            # 步骤4：Orchestrator 裁决 — 根据复核结果分类 issues
            accepted_issues, unresolved_must_fix_count = self._adjudicate_issues(
                reviews.technical.errors, verdicts
            )

            # 落库：issues + round + responses（P2）
            self._persist_debate_round(
                trace_id, iteration, reviews.technical.errors,
                author_replies, verdicts, accepted_issues, unresolved_must_fix_count,
            )

            # 步骤5：Agent3 修改学习包（只处理被裁决通过/降级的 issue）
            if accepted_issues:
                print(f"=== 调用 Agent3.modify_package（{len(accepted_issues)} 条已裁决 issue）===")
                revised_package_dict = await self.agent3.modify_package(
                    session_id,
                    package_dict,
                    {"tech": reviews.technical.errors, "quality": reviews.quality.errors},
                    accepted_issues,
                )
                current_package = LearningPackage.model_validate(revised_package_dict)
                current_package.review_trace_id = trace_id
                # 落库：更新 claims（status 变为 revised/dropped）（P2）
                self._persist_claims_update(trace_id, current_package)

            round_result = ReviewRound(
                iteration=iteration,
                technical=reviews.technical,
                quality=reviews.quality,
            )
            history.append(round_result)
            self._record_review_round(session_id, "learning_package", round_result)

            # 步骤6：判断是否通过 — 所有 must_fix 已解决则通过
            if unresolved_must_fix_count == 0 and not reviews.technical.errors:
                final_status = "passed"
                package_status = "approved"
                print(f"=== 学习包第{iteration}轮辩论后通过 ===")
                break

            if unresolved_must_fix_count == 0:
                # 无未决 must_fix，但有 minor 问题 → 视为通过
                final_status = "passed"
                package_status = "approved"
                print(f"=== 学习包第{iteration}轮辩论后通过（剩余非 must_fix 问题）===")
                break

            if iteration == settings.MAX_REVIEW_ITERATIONS:
                # 达到最大轮次仍有未决 must_fix → needs_human_review（协议第 7 节）
                final_status = "needs_human_review"
                package_status = "provisional"
                print(
                    f"=== 学习包达到最大审核轮次，{unresolved_must_fix_count} 个 must_fix 未决，"
                    f"标记为 needs_human_review（package 降级为 provisional）==="
                )
                break

            print(f"=== 第{iteration}轮仍有 {unresolved_must_fix_count} 个 must_fix 未决，进入下一轮 ===")

        # 设置 package 最终状态
        current_package.review_trace_id = trace_id
        unresolved_count = sum(
            1 for r in history for e in r.technical.errors if e.must_fix
        )
        self._last_review_trace = {
            "trace_id": trace_id,
            "final_status": final_status,
            "package_status": package_status,
            "unresolved_issue_count": unresolved_count,
            "unresolved_must_fix_count": unresolved_count,
        }
        # 落库：更新 trace 最终状态（P2）
        self._persist_trace_finalize(trace_id, final_status, package_status, unresolved_count)
        return current_package, history

    # ==================================================================
    # 块级真并发流水线：Agent3 生成/修改与 Agent4 审核/复核并发推进。
    # Phase A：Agent3 受控生成，Agent4 独立消费已生成块
    # 首轮后：一次性构造完整 issue_package（按 content_block_id 分组）交给 Agent3
    # Phase C：并发修改（Semaphore(2)），每块改完立即 create_task 派发独立复核
    #          （块1复核不阻塞块2修改）；双层防死循环（全局轮次 + 每块 rounds_sent ≤ MAX）
    # Phase D：聚合所有块为一个 LearningPackage
    # ==================================================================
    async def _generate_and_review_block(
        self,
        session_id: str,
        stage: StageInfo,
        profile: UserProfile,
        practice_profile: PracticeProfile,
        source_chunk: Dict[str, Any],
        block_index: int,
        round_index: int,
        generation_sem: asyncio.Semaphore,
        review_sem: asyncio.Semaphore,
    ) -> tuple[Any, TechnicalReviewResult]:
        """单块生产/消费：Agent3 生成后立即提交 Agent4 审核。

        两个 Semaphore 有意分离：审核等待或执行时不占用 Agent3 的生成槽，
        因此 Agent3 可继续生产后续块，Agent4 独立消费已完成的块。
        """
        chunk_label = str(source_chunk.get("chunk_id") or source_chunk.get("source_chunk_id") or block_index + 1)
        self._log(session_id, "agent3:block_generate_start", f"block={block_index + 1}, chunk={chunk_label}")
        block = await self._run_capped(
            generation_sem,
            lambda: self.agent3.generate_block(
                stage=stage,
                source_chunk=source_chunk,
                profile=profile,
                practice=practice_profile,
                block_index=block_index,
                session_id=session_id,
            ),
        )
        self._log(session_id, "agent3:block_generate_done", f"block={block_index + 1}")
        try:
            self._log(session_id, "agent4:first_review_start", f"block={block_index + 1}, round={round_index}")
            review = await self._run_capped(
                review_sem,
                lambda: self.agent4.review_block(
                    block=block, stage=stage, profile=profile, round_index=round_index,
                ),
            )
            self._log(session_id, "agent4:first_review_done", f"block={block_index + 1}, issues={len(review.errors)}")
        except Exception as exc:
            review = TechnicalReviewResult(
                passed=False,
                errors=[ReviewError(
                    error_type="review_system_error",
                    severity="critical",
                    description=f"Agent4 review_block failed: {exc}",
                    content_block_id=getattr(block, "content_block_id", None),
                )],
            )
            self._log(session_id, "agent4:first_review_failed", f"block={block_index + 1}")
        return block, review

    def _aggregate_blocks(
        self,
        session_id: str,
        stage: StageInfo,
        blocks: List[Any],
        trace_id: str,
        questions: Optional[List[Any]] = None,
        recommended_videos: Optional[List[dict]] = None,
    ) -> LearningPackage:
        """把已生成/修订的内容块按 block_index 聚合成一个 LearningPackage。"""
        ordered = sorted(blocks, key=lambda b: getattr(b, "block_index", 0))
        tutorial_parts: List[str] = [f"# {stage.block_name}"]
        all_sources: List[Any] = []
        all_claims: List[Any] = []
        seen_source_ids: set = set()
        for blk in ordered:
            heading = blk.heading_path or blk.block_name or ""
            if heading:
                tutorial_parts.append(f"## {heading}")
            if blk.personalized_content:
                tutorial_parts.append(blk.personalized_content)
            for src in blk.sources or []:
                if src.source_id not in seen_source_ids:
                    seen_source_ids.add(src.source_id)
                    all_sources.append(src)
            all_claims.extend(blk.claims or [])
        videos = list(recommended_videos or [])
        for video in videos:
            source_id = f"src_video_{video['resource_id']}"
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            all_sources.append(Source(
                source_id=source_id,
                source_type="video",
                title=video["title"],
                locator=f"learning_resources:{video['resource_id']}",
                content_ref=video.get("url"),
                metadata_json={
                    "block_id": video.get("block_id"),
                    "subtopic_id": video.get("subtopic_id"),
                    "review_status": video.get("review_status"),
                },
            ))
        return LearningPackage(
            session_id=session_id,
            stage_id=stage.stage_id,
            block_id=stage.block_id,
            block_name=stage.block_name,
            tutorial="\n\n".join(p for p in tutorial_parts if p),
            sources=all_sources,
            claims=all_claims,
            questions=list(questions or []),
            recommended_videos=videos,
            review_trace_id=trace_id,
        )

    async def _populate_stage_questions(
        self,
        package: LearningPackage,
        stage: StageInfo,
    ) -> None:
        """为聚合后的学习包补齐自测题（块级流水线路径）。

        整篇路径由 Agent3.run 直接从 test_questions 表提取题目，块级路径以块为最小
        生成单位、不产出整阶段题目，故聚合后需与整篇路径一致地按 block_id + subtopic_ids
        提取题目，保证学习包可作答（提交答案→掌握判定→阶段推进依赖此题目）。
        题库不可用或无匹配题目时优雅降级为空列表，不中断学习流。
        """
        try:
            extracted = await self.agent3.question_extract_tool.extract_questions(
                package.tutorial,
                {
                    "block_id": stage.block_id,
                    "block_name": stage.block_name,
                    "subtopic_ids": self.agent3._stage_subtopic_ids(stage),
                    # 阶段内有多个 subtopic 时，题库抽取内部会按每个 subtopic 抽 2 题
                    # 再整体补齐；这里的 question_count 作为总量下限避免题量不足。
                    "question_count": 8,
                },
            )
        except Exception:
            extracted = []
        package.questions = [
            TestQuestion.model_validate(question)
            for question in extracted
            if isinstance(question, dict)
        ]

    @staticmethod
    async def _run_capped(sem: asyncio.Semaphore, coro_factory):
        """并发上限原语：持有 sem 期间执行一个协程（生成/修改两阶段共用）。"""
        async with sem:
            return await coro_factory()

    async def _recheck_block(
        self,
        session_id: str,
        st: "BlockState",
        stage: StageInfo,
        profile: UserProfile,
        round_index: int,
    ) -> None:
        """只复核单个已修改块，并把结果覆盖回该块的 open_issues（任务 4）。

        覆盖语义即「关闭/重开/新增」：不再返回的问题=关闭，仍返回的=保留，新出现的=新增。
        作为独立 task 派发，故不阻塞其它块的修改（REQ-C）。异常兜底为 critical 留在 open_issues。

        省 token：复核只跑确定性工具（use_llm_semantic=False），不再每块每轮花一次
        LLM 语义审核；LLM 语义审核已在 Phase A 首轮全量执行，复核只需判定修订是否
        消解了首轮暴露的规则/事实/质量问题。
        """
        try:
            self._log(session_id, "agent4:recheck_start", f"round={round_index}")
            rr = await self.agent4.review_block(
                block=st.block, stage=stage, profile=profile, round_index=round_index,
                use_llm_semantic=False,
            )
        except Exception as exc:
            rr = TechnicalReviewResult(
                passed=False,
                errors=[ReviewError(
                    error_type="review_system_error", severity="critical",
                    description=f"re-review failed: {exc}", content_block_id=st.cid,
                    content_version=getattr(st.block, "version", None),
                )],
            )
        st.last_review = rr
        st.open_issues = list(rr.errors)
        self._log(session_id, "agent4:recheck_done", f"round={round_index}, issues={len(rr.errors)}")

    async def _review_learning_package_pipeline(
        self,
        session_id: str,
        stage: StageInfo,
        profile: UserProfile,
        practice_profile: PracticeProfile,
    ) -> tuple[LearningPackage, list[ReviewRound]]:
        """块级真并发生产者/消费者流水线（任务 5 + 真并发改造）。

        - 生成：并发上限 BLOCK_CONCURRENCY（=4），每块生成后立即提交独立审核；
        - 首轮后一次性构造完整 issue_package（按块分组），交给 Agent3；
        - 修改：并发上限 BLOCK_CONCURRENCY（=4），每块改完立即派发独立复核 task
          （块1复核不阻塞块2修改）；
        - 全部块无 open 问题才 passed；Agent4 交还问题清单最多 MAX_REVIEW_ITERATIONS 次；
        - 无可用原始 chunk 时安全降级为原整篇链路。

        省 token / 省耗时（避免 Agent3↔Agent4 无谓来回扯皮）：
        - 收敛检测：某块「修改+复核后」问题集与上一轮完全一致（同一批注定改不动的
          问题原地打转，如认知负荷/难度红线）即冻结该块、留待人工，不再对它继续
          modify+recheck，砍掉注定失败的重复轮次；
        - 复核只跑确定性工具（use_llm_semantic=False），不再每块每轮花一次 LLM 语义审核，
          LLM 语义审核仅在 Phase A 首轮全量执行，核心审核能力不缩水。
        """
        history: list[ReviewRound] = []
        trace_id = f"tr_{uuid.uuid4().hex[:16]}"

        # 读取本阶段原始 knowledge_chunks（分块由数据侧负责入库）。
        source_chunks = self.agent3.iter_stage_chunks(stage)
        if not source_chunks:
            # 无块可用：降级回整篇链路，保证不因缺块中断学习流。
            self._log(session_id, "pipeline:no_chunks -> fallback_full", f"stage={stage.stage_num}")
            generated = await self.agent3.run(ContentGenerationInput(
                session_id=session_id,
                stage=stage,
                learning_profile=profile,
                practice_profile=practice_profile,
            ))
            return await self._review_learning_package(
                session_id, generated.learning_package, stage, profile,
            )

        # ---- Phase A：生成（并发上限 2）× 每块生成后立即首轮审核 ----
        self._log(session_id, "pipeline:phaseA_start", f"chunks={len(source_chunks)}, cap={BLOCK_CONCURRENCY}")
        gen_sem = asyncio.Semaphore(BLOCK_CONCURRENCY)
        review_sem = asyncio.Semaphore(BLOCK_CONCURRENCY)
        pairs = await asyncio.gather(*[
            self._generate_and_review_block(
                session_id, stage, profile, practice_profile, chunk, idx, round_index=1,
                generation_sem=gen_sem, review_sem=review_sem,
            )
            for idx, chunk in enumerate(source_chunks)
        ])
        # 每块建立运行态 BlockState（当前版本、open 问题、复核结果、交还次数）
        states: Dict[str, BlockState] = {}
        for blk, review in pairs:
            states[blk.content_block_id] = BlockState(
                cid=blk.content_block_id,
                block=blk,
                open_issues=list(review.errors),
                last_review=review,
                # 首轮问题集作为收敛检测基线：后续某轮复核后指纹与之相同即视为原地打转。
                last_issue_fp=_issue_fingerprint(review.errors),
            )

        first_round_errors = [e for st in states.values() for e in st.open_issues]
        history.append(ReviewRound(
            iteration=1,
            technical=TechnicalReviewResult(passed=not first_round_errors, errors=list(first_round_errors)),
            quality=QualityReviewResult(passed=not first_round_errors, errors=list(first_round_errors)),
        ))
        self._log(
            session_id, "pipeline:phaseA_done",
            f"blocks={len(states)}, issues={len(first_round_errors)}",
        )

        # ---- 构造完整 issue_package（按 content_block_id 分组，一次性交付）----
        issue_package: Dict[str, List[ReviewError]] = {
            cid: st.open_issues for cid, st in states.items() if st.open_issues
        }
        self._log(session_id, "pipeline:issue_package", f"blocks_with_issues={len(issue_package)}")

        stage_questions, stage_videos = await self.agent3.get_stage_supplements(stage)

        # ---- 落库：首轮聚合出一个 draft 学习包并创建 trace（sources/claims）----
        # trace init 需要 sources/claims，故此处先聚合一次 draft 包用于建 trace；
        # 学习包正式聚合在 Phase D 结束时再做一次（含最终修订块）。
        draft_package = self._aggregate_blocks(
            session_id, stage, [st.block for st in states.values()], trace_id,
            questions=stage_questions, recommended_videos=stage_videos,
        )
        self._persist_trace_init(trace_id, session_id, stage, draft_package)
        # 首轮审核结果作为第 1 轮 review_round 落库（Agent4 异步审核结果统一留痕）
        self._record_review_round(session_id, "learning_package", history[0])

        # ---- Phase C：修改（并发上限 2）+ 每块改完并发复核 ----
        final_status = "passed"
        package_status = "approved"
        total_unresolved = 0

        for round_num in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
            # Agent3 先接收本轮完整问题列表，再从已建立的待修改队列中取块。
            # 后续轮次同样以完整列表重新交付，已通过的块不会进入该列表。
            revision_queue = await self.agent3.receive_issue_package(
                session_id,
                issue_package,
                {cid: st.block for cid, st in states.items()},
            )
            pending = [
                cid for cid in revision_queue
                if not states[cid].frozen
                and states[cid].rounds_sent < settings.MAX_REVIEW_ITERATIONS
            ]
            if not pending:
                # 无待修改块：要么全部通过，要么剩余的都已被收敛检测冻结（留待人工）。
                any_frozen_open = any(
                    st.frozen and st.open_issues for st in states.values()
                )
                if any_frozen_open:
                    final_status = "needs_human_review"
                    package_status = "provisional"
                else:
                    final_status = "passed"
                    package_status = "approved"
                break

            modify_sem = asyncio.Semaphore(BLOCK_CONCURRENCY)
            # 收集本轮各块辩论产物，供统一落库（_persist_debate_round）
            round_debate: List[Dict[str, Any]] = []

            async def _fix_one(cid: str) -> int:
                """单块：答辩→复核裁决→定点修改→派发独立复核 task。返回该块未决 must_fix 数。"""
                st = states[cid]
                st.rounds_sent += 1
                st.recheck_task = None
                # 记录本轮该块的原始 issues（修改前快照，用于辩论落库）
                pre_errors = list(st.open_issues)
                # 从 Agent3 已接收的完整 issue_package 队列取本块问题；状态快照
                # 仍保留 ReviewError 实例，供 Orchestrator 做裁决和落库。
                issues = self.agent3.queued_issues(session_id, cid)
                block_dict = st.block.model_dump(mode="json")

                # Agent3 答辩（块作为最小上下文）
                try:
                    self._log(session_id, "agent3:review_reply_start", f"round={round_num}")
                    author_replies = await self.agent3.reply_to_review(session_id, issues, block_dict)
                    self._log(session_id, "agent3:review_reply_done", f"round={round_num}, replies={len(author_replies)}")
                except Exception:
                    author_replies = []
                # Agent4 复核裁决
                try:
                    self._log(session_id, "agent4:verdict_start", f"round={round_num}")
                    verdicts = await self.agent4.rebut(session_id, issues, author_replies)
                    self._log(session_id, "agent4:verdict_done", f"round={round_num}, decisions={len(verdicts)}")
                except Exception:
                    verdicts = {}
                accepted_issues, unresolved_must_fix_count = self._adjudicate_issues(
                    pre_errors, verdicts,
                )
                # 缓存辩论产物（轮末统一落库）
                round_debate.append({
                    "errors": pre_errors,
                    "author_replies": author_replies,
                    "verdicts": verdicts,
                    "accepted_issues": accepted_issues,
                    "unresolved_must_fix_count": unresolved_must_fix_count,
                })

                if accepted_issues:
                    # 定点修改本块（version+1、revision_note），修改槽只在此期间持有
                    try:
                        self._log(session_id, "agent3:revision_start", f"round={round_num}")
                        revised = await self.agent3.modify_block(st.block, accepted_issues)
                        st.block = revised
                        self._log(session_id, "agent3:revision_done", f"round={round_num}")
                    except Exception as exc:
                        # 修改失败：保留原 open_issues，不派复核，交由后续轮次/人工
                        self._log(session_id, "pipeline:modify_failed", f"cid={cid}, err={exc}")
                        return unresolved_must_fix_count
                    # 立即派发独立复核 task（不占修改槽 → 块1复核不阻塞块2修改）
                    st.recheck_task = asyncio.create_task(
                        self._recheck_block(session_id, st, stage, profile, round_num + 1)
                    )
                else:
                    # 无接受的修改：该块视为已解决，清空残留 open 问题
                    st.open_issues = []
                return unresolved_must_fix_count

            # 并发修改（上限 2）
            unresolved_list = await asyncio.gather(*[
                self._run_capped(modify_sem, lambda c=cid: _fix_one(c)) for cid in pending
            ])
            round_unresolved = sum(unresolved_list)

            # 汇合本轮所有并发复核（复核内部已兜底异常，不会抛）
            recheck_tasks = [states[cid].recheck_task for cid in pending if states[cid].recheck_task]
            if recheck_tasks:
                await asyncio.gather(*recheck_tasks)

            # 收敛检测：本轮「修改+复核后」若某块问题集与上一轮完全一致（同一批注定
            # 改不动的问题原地打转，如认知负荷/难度红线无法靠追加式修订消解），则冻结该块，
            # 停止对它继续 modify+recheck，省掉注定失败的后续轮次（省 token 与耗时）。
            frozen_this_round = 0
            for cid in pending:
                st = states[cid]
                new_fp = _issue_fingerprint(st.open_issues)
                if st.open_issues and new_fp == st.last_issue_fp:
                    st.frozen = True
                    frozen_this_round += 1
                st.last_issue_fp = new_fp
            if frozen_this_round:
                self._log(
                    session_id, "pipeline:converge_freeze",
                    f"round={round_num}, frozen_blocks={frozen_this_round}",
                )

            # Agent4 完成本轮复核后，再构造下一轮完整问题列表；只含仍未通过的块，
            # 且排除已冻结块（冻结块留待人工，不再交还 Agent3 空转）。
            issue_package = {
                cid: st.open_issues
                for cid, st in states.items()
                if st.open_issues and not st.frozen
            }

            # 落库：把本轮所有块的辩论产物合并成一条 review_debate_round（P2）
            self._persist_pipeline_round(trace_id, round_num, round_debate)

            round_errors = [e for st in states.values() for e in st.open_issues]
            round_result = ReviewRound(
                iteration=round_num + 1,
                technical=TechnicalReviewResult(passed=not round_errors, errors=list(round_errors)),
                quality=QualityReviewResult(passed=not round_errors, errors=list(round_errors)),
            )
            history.append(round_result)
            self._record_review_round(session_id, "learning_package", round_result)
            self._log(
                session_id, "pipeline:phaseC_round",
                f"round={round_num}, unresolved_must_fix={round_unresolved}, open_errors={len(round_errors)}",
            )

            # 终止判定统一以「修改+复核后」的真实状态 any_open 为准（修复问题A）。
            # round_unresolved 是「修改前」快照，仅用于日志/统计，不再参与终止判定，
            # 避免本轮已被 recheck 解决的问题在末轮被误判为 needs_human_review。
            any_open = any(st.open_issues for st in states.values())
            if not any_open:
                # 全部块复核后无 open 问题 → 完成（任务 5）
                final_status = "passed"
                package_status = "approved"
                break
            # 仍有 open 的块若全部已被收敛检测冻结，继续循环也只是空转，直接判需人工。
            active_open = any(
                st.open_issues and not st.frozen for st in states.values()
            )
            if not active_open:
                final_status = "needs_human_review"
                package_status = "provisional"
                break
            # 交还次数封顶（任务 6：≤ MAX_REVIEW_ITERATIONS）仍有残留 open → 需人工
            if round_num == settings.MAX_REVIEW_ITERATIONS:
                final_status = "needs_human_review"
                package_status = "provisional"
                break

        # ---- Phase D：聚合块为学习包 ----
        package = self._aggregate_blocks(
            session_id, stage, [st.block for st in states.values()], trace_id,
            questions=stage_questions, recommended_videos=stage_videos,
        )
        # 块级路径以块为最小单位，不产出整阶段自测题；聚合后与整篇路径一致地补齐题目，
        # 保证学习包可作答（提交答案/掌握判定依赖题目）。
        await self._populate_stage_questions(package, stage)
        # 修订后 claims 状态更新落库（P2）
        self._persist_claims_update(trace_id, package)
        total_unresolved = sum(
            1 for st in states.values() for e in st.open_issues if getattr(e, "must_fix", False)
        )
        self._last_review_trace = {
            "trace_id": trace_id,
            "final_status": final_status,
            "package_status": package_status,
            "unresolved_issue_count": total_unresolved,
            "unresolved_must_fix_count": total_unresolved,
        }
        # 落库：更新 trace 最终状态（P2）
        self._persist_trace_finalize(trace_id, final_status, package_status, total_unresolved)
        self._log(
            session_id, "pipeline:done",
            f"final_status={final_status}, blocks={len(states)}, unresolved={total_unresolved}",
        )
        return package, history

    def _persist_trace_init(
        self, trace_id: str, session_id: str, stage: StageInfo, package: LearningPackage
    ) -> None:
        """落库：创建 trace + sources + claims（循环外调用一次）。"""
        try:
            db = self.state_manager.db_client
            db.save_review_trace(
                trace_id=trace_id,
                session_id=session_id,
                stage_id=stage.stage_id,
                target_type="learning_package",
                package_version=package.version,
                package_status="draft",
                review_mode="debate",
                final_status="in_progress",
            )
            db.save_learning_package_sources(
                trace_id, [s.model_dump(mode="json") for s in package.sources]
            )
            db.save_learning_package_claims(
                trace_id, [c.model_dump(mode="json") for c in package.claims]
            )
        except Exception as e:
            print(f"[Harness] 落库失败（trace init）: {e}")

    def _persist_debate_round(
        self,
        trace_id: str,
        iteration: int,
        errors: List[ReviewError],
        author_replies: List[Dict[str, Any]],
        verdicts: Dict[str, str],
        accepted_issues: List[Dict[str, Any]],
        unresolved_must_fix_count: int,
    ) -> None:
        """落库：issues + round + responses（每轮辩论后调用）。"""
        try:
            db = self.state_manager.db_client
            round_id = f"rd_{trace_id}_{iteration}"

            # 1. issues（带 opened_in_round_num）
            issues_for_db = []
            for e in errors:
                iss = e.model_dump(mode="json")
                iss["opened_in_round_num"] = iteration
                issues_for_db.append(iss)
            db.save_review_debate_issues(trace_id, issues_for_db)

            # 2. round（开轮并立即关闭）
            verdict_label = "all_resolved" if unresolved_must_fix_count == 0 else "partial_resolved"
            db.save_review_debate_round(
                trace_id=trace_id,
                round_id=round_id,
                round_num=iteration,
                review_iteration=iteration,
                round_status="closed",
                orchestrator_verdict=verdict_label,
                verdict_summary=f"unresolved_must_fix={unresolved_must_fix_count}",
            )

            # 3. Agent3 答辩 responses
            author_responses = []
            for reply in author_replies:
                cid = reply.get("canonical_error_id", "")
                author_responses.append({
                    "round_id": round_id,
                    "issue_id": f"iss_{trace_id}_{cid}",
                    "actor": "agent3",
                    "response_type": "author_reply",
                    "disposition": reply.get("disposition", "accepted"),
                    "rationale": reply.get("reason"),
                    "source_refs": reply.get("source_refs", []),
                    "patched_sections": reply.get("patched_sections", []),
                })
            db.save_review_debate_responses(round_id, author_responses)

            # 4. Agent4 复核 responses
            reviewer_responses = []
            for cid, verdict in verdicts.items():
                reviewer_responses.append({
                    "round_id": round_id,
                    "issue_id": f"iss_{trace_id}_{cid}",
                    "actor": "agent4",
                    "response_type": "reviewer_verdict",
                    "disposition": verdict,
                    "rationale": None,
                })
            db.save_review_debate_responses(round_id, reviewer_responses)

            # 5. Orchestrator 裁决 responses + 更新 issue 状态
            orchestrator_responses = []
            for iss in accepted_issues:
                cid = iss.get("canonical_error_id", "")
                issue_id = f"iss_{trace_id}_{cid}"
                # withdraw → accepted, downgrade → partially_accepted
                verdict = verdicts.get(cid, "withdraw")
                disposition = "accepted" if verdict == "withdraw" else "partially_accepted"
                orchestrator_responses.append({
                    "round_id": round_id,
                    "issue_id": issue_id,
                    "actor": "orchestrator",
                    "response_type": "orchestrator_ruling",
                    "disposition": disposition,
                    "rationale": f"Agent4 verdict={verdict}",
                })
                db.update_review_debate_issue_status(
                    issue_id=issue_id,
                    current_status=disposition,
                    final_disposition=disposition,
                    resolved_in_round_num=iteration,
                )
            db.save_review_debate_responses(round_id, orchestrator_responses)

            # persist 的 issue（persist 且 must_fix）标记为 open
            for e in errors:
                cid = e.canonical_error_id or ""
                if verdicts.get(cid) == "persist":
                    db.update_review_debate_issue_status(
                        issue_id=f"iss_{trace_id}_{cid}",
                        current_status="open",
                    )
        except Exception as e:
            print(f"[Harness] 落库失败（debate round）: {e}")

    def _persist_pipeline_round(
        self, trace_id: str, round_num: int, round_debate: List[Dict[str, Any]],
    ) -> None:
        """块级流水线专用：把「本轮多个块」的辩论产物合并为一条 review_debate_round 落库。

        每个块的 _fix_one 产出一份辩论片段（errors/author_replies/verdicts/accepted_issues/
        unresolved_must_fix_count）；此处合并后复用 _persist_debate_round（本身已 try/except 兜底）。
        无辩论片段（本轮无块进入修改）则跳过。
        """
        if not round_debate:
            return
        merged_errors: List[ReviewError] = []
        merged_replies: List[Dict[str, Any]] = []
        merged_verdicts: Dict[str, str] = {}
        merged_accepted: List[Dict[str, Any]] = []
        total_unresolved = 0
        for frag in round_debate:
            merged_errors.extend(frag.get("errors", []))
            merged_replies.extend(frag.get("author_replies", []))
            merged_verdicts.update(frag.get("verdicts", {}))
            merged_accepted.extend(frag.get("accepted_issues", []))
            total_unresolved += int(frag.get("unresolved_must_fix_count", 0))
        self._persist_debate_round(
            trace_id, round_num, merged_errors,
            merged_replies, merged_verdicts, merged_accepted, total_unresolved,
        )

    def _persist_claims_update(self, trace_id: str, package: LearningPackage) -> None:
        """落库：modify_package 后更新 claims（status 变为 revised/dropped）。"""
        try:
            db = self.state_manager.db_client
            db.save_learning_package_claims(
                trace_id, [c.model_dump(mode="json") for c in package.claims]
            )
        except Exception as e:
            print(f"[Harness] 落库失败（claims update）: {e}")

    def _persist_trace_finalize(
        self, trace_id: str, final_status: str, package_status: str, unresolved_count: int
    ) -> None:
        """落库：更新 trace 最终状态（循环结束调用）。"""
        try:
            db = self.state_manager.db_client
            db.update_review_trace_status(
                trace_id=trace_id,
                final_status=final_status,
                package_status=package_status,
                unresolved_issue_count=unresolved_count,
                unresolved_must_fix_count=unresolved_count,
            )
        except Exception as e:
            print(f"[Harness] 落库失败（trace finalize）: {e}")

    def _adjudicate_issues(
        self,
        errors: List[ReviewError],
        verdicts: Dict[str, str],
    ) -> tuple[List[Dict[str, Any]], int]:
        """Orchestrator 裁决：根据 Agent4 复核结果分类 issues。

        返回 (accepted_issues, unresolved_must_fix_count)：
        - accepted_issues：Agent3 需要 modify 的 issues（withdraw + downgrade）
        - unresolved_must_fix_count：仍坚持(persist)且 must_fix=true 的数量
        """
        accepted_issues: List[Dict[str, Any]] = []
        unresolved_must_fix_count = 0

        for err in errors:
            cid = err.canonical_error_id or ""
            verdict = verdicts.get(cid, "persist")

            if verdict == "withdraw":
                # Agent4 撤销该 issue → Agent3 接受处理
                accepted_issues.append(err.model_dump(mode="json"))
            elif verdict == "downgrade":
                # Agent4 降级 → 降级 severity，若原 must_fix 则解除
                err.severity = "minor" if err.severity == "major" else err.severity
                if err.must_fix:
                    err.must_fix = False
                    err.must_fix_reason = "Agent4 复核降级，解除 must_fix"
                accepted_issues.append(err.model_dump(mode="json"))
            else:
                # persist → Agent4 坚持该 issue
                if err.must_fix:
                    # must_fix=True：强制 Agent3 修改，但仍计入未决（进下一轮重新审核）
                    accepted_issues.append(err.model_dump(mode="json"))
                    unresolved_must_fix_count += 1
                # must_fix=False：minor 问题，允许通过，不强制修改

        return accepted_issues, unresolved_must_fix_count

    @staticmethod
    def _is_out_of_scope_review(technical_result: TechnicalReviewResult) -> bool:
        """判断本轮技术审核是否为「库外/超纲」——即唯一阻塞项来自 out_of_scope。

        触发条件（同时满足）：
          - 至少存在一个 error_type == out_of_scope 的错误（知识库未覆盖该主题）；
          - 不存在其他会强制改写的阻塞错误（must_fix=True 或 severity=critical）。
        满足时上层直接输出 Agent3 内容，不再逼迫改写（库外问题无从溯源，改写无益）。
        """
        errors = getattr(technical_result, "errors", None) or []
        if not errors:
            return False
        has_out_of_scope = False
        for err in errors:
            et = (getattr(err, "error_type", None) or "").lower()
            if et == "out_of_scope":
                has_out_of_scope = True
                continue
            # 其余任何 must_fix / critical 错误都属于真实需改写项，不可短路
            if getattr(err, "must_fix", False) or getattr(err, "severity", None) == "critical":
                return False
        return has_out_of_scope

    async def _review_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        stage: StageInfo,
        profile: UserProfile,
        practice_profile: PracticeProfile | None = None,
    ) -> tuple[str, list[ReviewRound]]:
        history = []
        current_answer = answer
        for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
            print(f"=== 问答审核循环：第{iteration}轮 ===")
            reviews = await self._review(ReviewInput(
                session_id=session_id,
                target_type="answer",
                content=current_answer,
                question=question,
                stage=stage,
                profile=profile,
            ))
            round_result = ReviewRound(
                iteration=iteration,
                technical=reviews.technical,
                quality=reviews.quality,
            )
            history.append(round_result)
            self._record_review_round(session_id, "answer", round_result)
            if round_result.passed:
                print(f"=== 回答第{iteration}轮审核通过 ===")
                break
            # 库外/超纲：问题不在知识库覆盖范围内。此时无从溯源，改写也无益，
            # 直接输出 Agent3 内容而非逼迫反复改写。判据：本轮技术错误全部为 out_of_scope。
            if self._is_out_of_scope_review(round_result.technical):
                print(f"=== 回答第{iteration}轮判定为库外问题（out_of_scope），直接输出 Agent3 内容 ===")
                break
            if iteration == settings.MAX_REVIEW_ITERATIONS:
                print("=== 回答达到最大审核轮次，返回未通过状态 ===")
                break
            print(f"=== Orchestrator汇总第{iteration}轮意见，调用Agent3修改回答 ===")
            revised = await self.agent3.revise_answer(AnswerRevisionInput(
                session_id=session_id,
                question=question,
                answer=current_answer,
                stage=stage,
                profile=profile,
                practice_profile=practice_profile or PracticeProfile(),
                technical_errors=reviews.technical.errors,
                quality_errors=reviews.quality.errors,
            ))
            current_answer = revised.answer
        return current_answer, history

    async def _review(self, request: ReviewInput) -> ReviewOutput:
        # Agent5已合并至Agent4，统一调用合并后的TechnicalReviewAgent
        try:
            technical = await self.agent4.run(request)
        except Exception as exc:
            technical = TechnicalReviewResult(
                passed=False,
                errors=[ReviewError(
                    error_type="review_system_error",
                    severity="critical",
                    description=f"Agent4 review failed: {exc}",
                    suggestion="重新执行审核，若仍失败请检查合并后的 Agent4、skill 与 tool 链路",
                )],
                review_details={"exception": str(exc)},
            )

        # 质量审核结果已合并在technical结果中
        quality = QualityReviewResult(
            passed=technical.passed,
            errors=technical.errors,
            scores=getattr(technical, 'scores', None),
            review_details=getattr(technical, 'review_details', None),
        )

        return ReviewOutput(technical=technical, quality=quality)

    def _record_review_round(self, session_id: str, target_type: str, review_round: ReviewRound) -> None:
        print(
            f"=== Orchestrator审核汇总：target={target_type}, round={review_round.iteration}, "
            f"merged_errors={len(review_round.technical.errors)}, passed={review_round.passed} ==="
        )
        for error in review_round.technical.errors:
            print(f"  [合并审核] [{error.severity}] {error.description} -> {error.suggestion or '未提供建议'}")
        self.state_manager.add_interaction(session_id, "review_round", {
            "target_type": target_type,
            **review_round.model_dump(mode="json"),
            "passed": review_round.passed,
        })

    async def _sync_profile_from_subtopic_status(self, profile: UserProfile) -> UserProfile:
        synced_profile = profile.model_copy(deep=True)
        for gap in synced_profile.knowledge_gaps:
            block_info = {}
            try:
                block_info = await self.agent1.knowledge_block_query_tool.query_block(gap.skill)
            except Exception:
                block_info = {}
            block_id = block_info.get("block_id")
            if not block_id:
                continue
            summary = self.state_manager.db_client.get_block_status_summary(
                session_id=synced_profile.session_id,
                block_id=block_id,
            )
            if not summary:
                continue
            gap.status = summary["status"]
            gap.progress = summary["progress"]
            if not gap.prerequisites:
                gap.prerequisites = block_info.get("prerequisites", [])
        synced_profile.version += 1
        return synced_profile

    def _require_profile(self, session_id: str) -> dict:
        profile = self.state_manager.get_profile(session_id)
        if not profile:
            raise ValueError("Session has no user profile")
        return profile

    def _require_stage(self, session_id: str) -> dict:
        stage = self.state_manager.get_current_stage(session_id)
        if not stage:
            raise ValueError("Session has no current stage")
        return stage

    def _require_package(self, session_id: str) -> dict:
        package = self.state_manager.get_learning_package(session_id)
        if not package:
            raise ValueError("Session has no learning package")
        return package

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """标准化答案：去空格、转小写、去标点符号。"""
        import re
        cleaned = re.sub(r'[^\w.\u4e00-\u9fff/]', '', answer.strip().lower())
        return cleaned

    @staticmethod
    def _match_answer(user_answer: str, correct_answer: str, question_type: str = '') -> bool:
        """
        多级答案匹配策略（P2 模糊匹配）。
        1. 精确匹配（标准化后相等）
        2. 判断题：A/B/√/× 等多种形式与 正确/错误 文本归一
        3. 选择题：单字母选项匹配
        4. 包含匹配（正确答案含用户答案或反之）
        5. 数值近似（误差 < 5%）
        """
        TRUE_FALSE_TYPES = {
            'true_false', '判断', '判断题', 'yes_no', 'truefalse', 'tf', '对错', '正误'
        }
        user_norm = Harness._normalize_answer(user_answer)
        correct_norm = Harness._normalize_answer(correct_answer)
        qt_norm = (question_type or '').strip().lower()

        # 1. 精确匹配
        if user_norm == correct_norm:
            return True

        # 2. 判断题：前端点击 A="正确" / B="错误" 选项键提交；correct_answer 可能存
        #    "正确"/"错误"/"T"/"F"/"true"/"false"/"是"/"否"/"√"/"×" 等多种文本。
        #    先做归一化再比较。
        def _tf_map(text: str) -> str | None:
            t = text.strip().lower() if isinstance(text, str) else ''
            t_plain = t  # 保留原始去标点前版本用于判断
            # 去标点版本（_normalize_answer 已做过，但这里更安全再做一次）
            import re as _re
            t_clean = _re.sub(r'[^\w.\u4e00-\u9fff/]', '', t)
            true_aliases = {
                '正确', '对', '对的', '是', '是的', '正确的',
                't', 'true', 'yes', 'y', '1',
                'a',  # 前端选项 A 对应 ["正确","错误"] 中的第一个
                '√', '✔', '✓',
            }
            false_aliases = {
                '错误', '错', '不对', '否', '不是', '假的', '错误的', '不正确',
                'f', 'false', 'no', 'n', '0',
                'b',  # 前端选项 B 对应第二个选项
                '×', '✗', '✘', 'x',
            }
            if t_clean in true_aliases or t_plain in true_aliases:
                return '正确'
            if t_clean in false_aliases or t_plain in false_aliases:
                return '错误'
            return None

        if qt_norm in {Harness._normalize_answer(t) for t in TRUE_FALSE_TYPES} or \
           (not qt_norm and (_tf_map(user_norm) or _tf_map(correct_norm))):
            user_tf = _tf_map(user_answer)
            correct_tf = _tf_map(correct_answer)
            if user_tf and correct_tf and user_tf == correct_tf:
                return True

        # 3. 选择题选项字母匹配
        if question_type in ('选择', '选择题', '单选', 'multiple_choice', 'single_choice', '多选', 'multi_choice'):
            if len(user_norm) == 1 and user_norm.isalpha() and \
               correct_norm.startswith(user_norm):
                return True

        # 4. 包含匹配（双向，至少2字符防止单字母误判）
        if len(user_norm) >= 3 and len(correct_norm) >= 3:
            if user_norm in correct_norm or correct_norm in user_norm:
                return True

        # 5. 数值近似
        try:
            user_num = float(user_norm.replace(',', '').replace('，', ''))
            correct_num = float(correct_norm.replace(',', '').replace('，', ''))
            if abs(user_num - correct_num) / max(abs(correct_num), 1.0) < 0.05:
                return True
        except (ValueError, ZeroDivisionError):
            pass

        return False

    def _generate_final_report(self, profile: UserProfile, accuracy: float) -> Dict[str, Any]:
        return {
            "goal": profile.goal,
            "accuracy": accuracy,
            "mastered_skills": [gap.skill for gap in profile.knowledge_gaps if gap.progress >= 1.0],
            "profile_version": profile.version,
        }

    # ===== P1: 公共 Session 管理方法 =====
    def get_session(self, session_id: str) -> dict | None:
        return self.state_manager.get_session(session_id)

    def get_runtime_state(self, session_id: str) -> dict | None:
        return self.state_manager.get_session(session_id)

    def end_session(self, session_id: str) -> None:
        self.state_manager.end_session(session_id)
        self._log(session_id, "session:end")

    def abandon_session(self, session_id: str) -> None:
        self.state_manager.abandon_session(session_id)
        self._log(session_id, "session:abandon")

    def create_session(self, session_id: str | None = None, user_id: str = "local-user") -> dict:
        session = self.state_manager.create_session(session_id, user_id)
        self._log(session["session_id"], "session:create")
        return session
