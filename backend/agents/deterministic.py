# -*- coding: utf-8 -*-
# ← 实际运行的 Agent 实现（含确定性版+混合版）
import json
import math
import re
import uuid
from typing import Any, Dict, List, Optional

try:
    from db.sqlite_client import SQLiteClient
except ModuleNotFoundError:
    from backend.db.sqlite_client import SQLiteClient
from llm.client import LLMClient
from models.contracts import (
    AnswerGenerationInput,
    AnswerGenerationOutput,
    AnswerRevisionInput,
    ContentGenerationInput,
    ContentGenerationOutput,
    ContentRevisionInput,
    DiagnosticInput,
    DiagnosticOutput,
    ReviewInput,
    ReviewOutput,
    StageDecisionInput,
    StageDecisionOutput,
    StagePlanningInput,
    StagePlanningOutput,
)
from models.material import TestQuestion
from models.profile import KnowledgeGap, PracticeProfile, UserProfile
from models.review import QualityReviewResult, ReviewError, TechnicalReviewResult
from models.stage import Claim, ContentBlock, LearningPackage, Source, StageInfo
from tools.learning_resource_tool import LearningResourceTool
from agents.agent_04_tech_review import TechnicalReviewAgent, _select_key_claims
from skills import load_skills
from tools.agent1_tools.calculate_confidence_tool import CalculateConfidenceTool
from tools.agent1_tools.extract_skills_tool import ExtractSkillsTool
from tools.agent1_tools.graph_query_tool import GraphQueryTool
from tools.agent1_tools.knowledge_catalog_tool import KnowledgeCatalogTool
from tools.agent1_tools.semantic_knowledge_recall_tool import SemanticKnowledgeRecallTool
from tools.agent1_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent1_tools.stage_generator_tool import StageGeneratorTool
from tools.agent3_tools.question_extract_tool import QuestionExtractTool
from tools.agent3_tools.vector_search_tool import VectorSearchTool
from tools.agent3_tools.video_search_tool import VideoSearchTool
from tools.agent4_tools.cognitive_load_tool import CognitiveLoadTool
from tools.agent4_tools.difficulty_analyzer_tool import DifficultyAnalyzerTool
from tools.agent4_tools.style_match_tool import StyleMatchTool
from tools.agent6_tools.profile_update_tool import ProfileUpdateTool
from tools.agent6_tools.stage_summary_generator_tool import StageSummaryGeneratorTool
from tools.agent6_tools.strategy_selector_tool import StrategySelectorTool
from tools.agent4_tools.knowledge_verify_tool import KnowledgeVerifyTool
from tools.agent4_tools.rule_check_tool import RuleCheckTool


SKILL_NAMES = (
    "Python",
    "PyTorch",
    "Transformer",
    "注意力机制",
    "LoRA",
    "QLoRA",
    "机器学习",
    "深度学习",
    "大模型微调",
    "模型部署",
)


DAILY_ANSWER_LIMIT = 150
# deepseek-r1 spends part of num_predict on internal reasoning. These are
# generation budgets, not user-facing length targets.
TECHNICAL_ANSWER_TOKENS = 3072  # 原 1536 对详细中文技术回答（原理+公式+示例）不够，导致 length 截断后保留半截回答
DAILY_ANSWER_TOKENS = 768


def _build_sourced_claim(
    *,
    claim_id: str,
    text: str,
    section_ref: str,
    source_id: Optional[str],
    chunk_id: str,
    claim_type: str = "concept",
    confidence: float = 0.8,
) -> Claim:
    """构造一条「强制溯源」的 Claim（辩论方案 第 1 点）。

    规则：
      - 有来源块（chunk_id 非空）→ source_ref 指向该来源，is_key_technical=True，
        进入 Agent4 的 KB 事实核验（可被核验/辩论）。
      - 无来源块 → 该客观陈述无引用，直接标 is_key_technical=False 且 source_ref=None，
        并在 metadata_json 记 sourcing="unsupported"。此类「无依据」主张不作为可核验关键
        Claim 进入辩论（_select_key_claims 只挑 is_key_technical=True），从源头减少幻觉。
    """
    has_source = bool(chunk_id and source_id)
    metadata = {"source_chunk_id": chunk_id}
    if not has_source:
        metadata["sourcing"] = "unsupported"
    return Claim(
        claim_id=claim_id,
        text=(text or "").strip()[:500],
        source_ref=source_id if has_source else None,
        section_ref=section_ref,
        claim_type=claim_type,
        is_key_technical=has_source,
        confidence=confidence if has_source else min(confidence, 0.3),
        metadata_json=metadata,
    )


def get_skill_context(skill_names: list[str]) -> str:
    return "\n\n".join(skill.content for skill in load_skills(skill_names))


def get_runtime_skill_context(skill_names: list[str]) -> str:
    """Load only the runtime rules declared by a skill, not its examples."""
    policies = []
    for skill in load_skills(skill_names):
        match = re.search(
            r"<!-- RUNTIME_POLICY_START -->(.*?)<!-- RUNTIME_POLICY_END -->",
            skill.content,
            flags=re.DOTALL,
        )
        if match:
            policies.append(match.group(1).strip())
    return "\n\n".join(policies)


def _agent_log(agent_name: str, message: str) -> None:
    print(f"[{agent_name}] {message}", flush=True)


def _shorten_subtopic_name(name: str) -> str:
    """清理并截断 subtopic_name，供阶段生成与结构化回答复用。

    - 仅当首尾成对包裹括号时才去掉外层（如“（环境准备：...）”），避免误删
      “常用Shell命令（cd/ls/mkdir/cp/mv）”这类内部内容括号。
    - 过长时截断到第一个分隔符或30字。
    - 去掉截断后可能悬挂的末尾开放括号。
    """
    name = name or ""
    while len(name) >= 2 and name[0] in "（(" and name[-1] in "）)":
        name = name[1:-1]
    if len(name) > 30:
        for sep in ["。", "，", ",", ".", "；", ";"]:
            idx = name.find(sep)
            if 10 < idx <= 35:
                name = name[:idx]
                break
        else:
            name = name[:30] + "..."
    name = name.rstrip("（(").rstrip()
    return name


class DeterministicDiagnosticAgent:
    def __init__(self, db_client: SQLiteClient | None = None):
        self.extract_skills_tool = ExtractSkillsTool()
        self.calculate_confidence_tool = CalculateConfidenceTool()
        self.graph_query_tool = GraphQueryTool()
        self.knowledge_catalog_tool = KnowledgeCatalogTool()
        self.semantic_knowledge_recall_tool = SemanticKnowledgeRecallTool()
        self.knowledge_block_query_tool = KnowledgeBlockQueryTool()
        self.stage_generator_tool = StageGeneratorTool()
        self.db_client = db_client or SQLiteClient()

    async def run(self, request: DiagnosticInput) -> DiagnosticOutput:
        _agent_log("Agent1", f"diagnostic:start | session={request.session_id}")
        text = request.user_input.strip()
        structured = self._parse_structured_input(text)
        target_text = structured.get("goal") or self._target_text(text)
        mastered = structured.get("skill_tags") or self._extract_mastered_skills(text, target_text)
        needs_domain_clarification = self._needs_domain_clarification(target_text)
        target_candidates = [] if needs_domain_clarification else self._select_target_candidates(target_text)
        gaps = await self._build_grounded_gaps(target_text, target_candidates, mastered)
        if self._needs_training_optimization(target_text):
            optimization_block = self.knowledge_catalog_tool.get_block("KB-13")
            if optimization_block and all(gap.block_id != "KB-13" for gap in gaps):
                gaps.append(KnowledgeGap(
                    skill=str(optimization_block["block_name"]),
                    block_id="KB-13",
                    block_name=str(optimization_block["block_name"]),
                    mapping_confidence=0.85,
                    level="advanced",
                ))
        if not gaps and not needs_domain_clarification:
            fallback_targets = [skill for skill in SKILL_NAMES if skill.lower() in target_text.lower()]
            gaps = await self._build_grounded_gaps(
                target_text,
                [
                    resolved
                    for skill in fallback_targets
                    if (resolved := self.knowledge_catalog_tool.resolve_block(skill))
                ],
                mastered,
            )
        if not gaps:
            gaps = [KnowledgeGap(skill=target_text or "通用学习目标")]
        # Stages expand the target's subtopic-level hard closure. Do not inject
        # coarse block prerequisites into the diagnostic profile.

        explicit_hours = self._has_daily_hours(text)
        explicit_style = self._has_learning_style(text)
        confidence = 0.75 if target_text else 0.5
        try:
            confidence_result = await self.calculate_confidence_tool.calculate(
                text,
                mastered,
                [gap.model_dump(mode="json") for gap in gaps],
                goal=target_text,
                daily_hours=structured.get("daily_hours") if explicit_hours else None,
                learning_style=structured.get("learning_style", "") if explicit_style else "",
            )
            confidence = max(confidence, confidence_result)
        except Exception:
            pass
        required_structured_fields = [
            structured.get("grade"),
            structured.get("skill_tags"),
            structured.get("goal"),
            structured.get("daily_hours"),
            structured.get("learning_style"),
        ]
        if (
            all(required_structured_fields)
            and gaps
            and all(gap.block_id and gap.mapping_confidence >= 0.8 for gap in gaps)
        ):
            confidence = max(confidence, 0.9)
        if not gaps or any(
            not gap.block_id or float(gap.mapping_confidence or 0.0) < 0.6
            for gap in gaps
        ):
            confidence = min(confidence, 0.65)
        followup_questions = self._build_followup_questions(
            structured=structured,
            has_hours=explicit_hours,
            has_style=explicit_style,
            mapped=any(gap.block_id for gap in gaps),
        ) if confidence < 0.7 else []
        if confidence < 0.7 and needs_domain_clarification:
            followup_questions = self._with_domain_clarification_question(followup_questions)
        profile = UserProfile(
            session_id=request.session_id,
            grade=structured.get("grade", ""),
            skill_tags=mastered,
            knowledge_gaps=gaps,
            goal=target_text or text,
            daily_hours=structured.get("daily_hours", self._extract_daily_hours(text)),
            learning_style=structured.get("learning_style", self._extract_learning_style(text)),
            confidence=confidence,
        )
        _agent_log("Agent1", f"diagnostic:done | goal={profile.goal} | gaps={len(profile.knowledge_gaps)} | confidence={profile.confidence:.2f}")
        return DiagnosticOutput(profile=profile, followup_questions=followup_questions)

    # ------------------------------------------------------------------
    # 分步诊断：目标 → 知识板块 → 知识点 → 确定
    # ------------------------------------------------------------------

    def get_goal_blocks(self, goal: str) -> list[dict[str, Any]]:
        """根据学习目标返回其下所有知识板块（含知识点列表）。

        仅返回与目标语义匹配的板块；无匹配时返回空列表。
        每个板块都附带 subtopics 列表，供前端展开选择。
        """
        target_text = (goal or "").strip()
        candidates = self._select_target_candidates(target_text) if target_text else []
        if candidates:
            blocks = []
            for candidate in candidates:
                block_id = str(candidate.get("block_id", ""))
                resolved = self.knowledge_catalog_tool.get_block(block_id) or candidate
                blocks.append({
                    "block_id": block_id,
                    "block_name": resolved.get("block_name", block_id),
                    "description": resolved.get("description", ""),
                    "difficulty": resolved.get("difficulty", ""),
                    "estimated_hours": resolved.get("estimated_hours", 0),
                    "subtopics": resolved.get("subtopics", []),
                })
            return blocks
        return []

    async def run_with_foundation(
        self,
        session_id: str,
        goal: str,
        mastered_subtopic_ids: list[str],
        daily_hours: float = 2.0,
        learning_style: str = "mixed",
    ) -> DiagnosticOutput:
        """接收用户已掌握知识点，直接构建画像，不依赖文本解析。

        流程：
        1. 按 goal 搜索目标板块（同 run 中的 _select_target_candidates）
        2. 将 mastered_subtopic_ids 按 block_id 分组，找出完全掌握的板块
        3. 构建 gaps（目标板块中未全部掌握的）
        4. 生成 UserProfile
        """
        target_text = (goal or "").strip()
        mastered_set = set(mastered_subtopic_ids or [])

        # 找目标板块
        candidates = self._select_target_candidates(target_text) if target_text else []
        gaps = await self._build_grounded_gaps(
            target_text, candidates, [],  # mastered_skills 通过知识点级别处理
        )

        # 标记已掌握的知识点：将 mastered_subtopic_ids 对应的 subtopic 状态设为已掌握
        # 同时构建 session 级别的已掌握状态映射（供候选筛选使用）
        for subtopic_id in mastered_set:
            try:
                self.db_client.set_subtopic_status(
                    session_id,
                    subtopic_id,
                    status=1,
                    mastery_score=1.0,
                    last_accuracy=1.0,
                    is_initial_foundation=True,
                )
            except Exception:
                pass

        # 修复 run_with_foundation 的 gaps：
        # 1) 对每个 gap，target_subtopic_ids = 板块中「所有 subtopic - 已掌握 subtopic」
        # 2) 如果板块中所有 subtopic 都掌握了，删掉该 gap
        filtered_gaps: list[KnowledgeGap] = []
        for gap in gaps:
            if not gap.block_id:
                filtered_gaps.append(gap)
                continue
            block = self.knowledge_catalog_tool.get_block(gap.block_id) or {}
            subtopics = block.get("subtopics", []) or []
            all_block_sub_ids: list[str] = []
            for s in subtopics:
                sid = str(s.get("subtopic_id") or s.get("id") or "")
                if sid:
                    all_block_sub_ids.append(sid)
            unmastered_ids = [sid for sid in all_block_sub_ids if sid not in mastered_set]
            if not unmastered_ids:
                # 此板块所有知识点都掌握了，不再纳入 gap
                continue
            # 按原始子知识点的相对顺序写回 target_subtopic_ids
            existing_targets = list(getattr(gap, "target_subtopic_ids", []) or [])
            if existing_targets:
                # 保留原 target 顺序，但过滤掉已掌握的 + 补齐未包含的未掌握
                preserved = [sid for sid in existing_targets if sid not in mastered_set and sid in all_block_sub_ids]
                extra = [sid for sid in unmastered_ids if sid not in preserved]
                gap.target_subtopic_ids = preserved + extra
            else:
                gap.target_subtopic_ids = list(unmastered_ids)
            filtered_gaps.append(gap)
        gaps = filtered_gaps

        # 找出完全掌握的板块（该板块所有知识点都在 mastered_set 中）
        mastered_blocks: list[str] = []
        all_blocks = self.knowledge_catalog_tool.list_blocks()
        for block in all_blocks:
            block_id = str(block.get("block_id", ""))
            subtopics = block.get("subtopics", [])
            if not subtopics:
                continue
            all_subtopic_ids = {s.get("subtopic_id", "") for s in subtopics}
            if all_subtopic_ids and all_subtopic_ids.issubset(mastered_set):
                mastered_blocks.append(block_id)

        # 完整掌握时记录板块名；部分掌握时记录具体知识点，避免用户已勾选
        # 但画像仍显示“未识别”。
        skill_tags: list[str] = []
        mastered_block_ids = set(mastered_blocks)
        for block in all_blocks:
            block_id = str(block.get("block_id", ""))
            resolved_block = self.knowledge_catalog_tool.get_block(block_id) or block
            block_name = str(resolved_block.get("block_name", block_id))
            if block_id in mastered_block_ids:
                skill_tags.append(block_name)
                continue
            for subtopic in resolved_block.get("subtopics", []) or []:
                subtopic_id = str(subtopic.get("subtopic_id") or subtopic.get("id") or "")
                if subtopic_id in mastered_set:
                    skill_tags.append(str(
                        subtopic.get("subtopic_name")
                        or subtopic.get("name")
                        or subtopic_id
                    ))

        # 构建置信度：有目标匹配 + 有已掌握知识点 → 高置信度
        confidence = 0.75
        if candidates and gaps and all(gap.block_id and gap.mapping_confidence >= 0.8 for gap in gaps):
            confidence = max(confidence, 0.9)
        if mastered_set:
            confidence = max(confidence, 0.8)

        profile = UserProfile(
            session_id=session_id,
            skill_tags=skill_tags,
            knowledge_gaps=gaps if gaps else [KnowledgeGap(skill=target_text or "通用学习目标")],
            goal=target_text or "学习",
            daily_hours=max(0.5, min(8.0, float(daily_hours or 2.0))),
            learning_style=learning_style or "mixed",
            confidence=confidence,
        )
        _agent_log(
            "Agent1",
            f"foundation:done | goal={profile.goal} | mastered_subtopics={len(mastered_set)} | "
            f"mastered_blocks={len(mastered_blocks)} | gaps={len(profile.knowledge_gaps)} | confidence={profile.confidence:.2f}",
        )
        return DiagnosticOutput(profile=profile, followup_questions=[])

    def _select_target_candidates(self, target_text: str) -> list[dict[str, Any]]:
        candidates = self._recall_candidates(target_text)
        if not candidates:
            return []
        top_score = float(candidates[0].get("score") or 0)
        threshold = max(20.0, top_score * 0.35)
        return [candidate for candidate in candidates if float(candidate.get("score") or 0) >= threshold][:4]

    def _recall_candidates(self, target_text: str) -> list[dict[str, Any]]:
        lexical_candidates = self.knowledge_catalog_tool.search(target_text, top_k=8)
        try:
            semantic_hits = self.semantic_knowledge_recall_tool.search(target_text, top_k=8)
        except Exception:
            semantic_hits = []
        return self.knowledge_catalog_tool.merge_recall_results(
            lexical_candidates,
            semantic_hits,
            top_k=8,
        )

    async def _build_grounded_gaps(
        self,
        target_text: str,
        candidates: list[dict[str, Any]],
        mastered_skills: list[str],
    ) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        seen_block_ids: set[str] = set()
        for candidate in candidates:
            block_id = str(candidate.get("block_id", ""))
            if not block_id or block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)
            resolved = self.knowledge_catalog_tool.get_block(block_id) or candidate
            skill_label = self.knowledge_catalog_tool.canonical_skill_label(resolved, target_text)
            matched_subtopics = candidate.get("matched_subtopics", [])
            target_subtopic_ids = [
                str(item.get("subtopic_id", ""))
                for item in matched_subtopics
                if item.get("subtopic_id")
            ]
            recalled_subtopic_ids = self.knowledge_catalog_tool.resolve_subtopic_ids(
                block_id, target_subtopic_ids
            )
            canonical_target_ids = self.knowledge_catalog_tool.default_target_subtopic_ids(
                block_id, target_text
            )
            # Intent-level targets are more precise than a semantically similar
            # article such as a LoRA export guide. Fall back to recalled IDs for
            # blocks without an explicit canonical target.
            target_subtopic_ids = canonical_target_ids or recalled_subtopic_ids
            prerequisites = self._immediate_prerequisite_names(resolved, mastered_skills)
            raw_score = float(candidate.get("score") or 0)
            mapping_confidence = float(candidate.get("mapping_confidence") or min(0.99, 0.7 + raw_score / 200))
            gaps.append(KnowledgeGap(
                skill=skill_label,
                block_id=block_id,
                block_name=str(resolved.get("block_name", skill_label)),
                target_subtopic_ids=target_subtopic_ids,
                mapping_confidence=round(mapping_confidence, 3),
                level=self._infer_gap_level(target_text, str(resolved.get("difficulty", ""))),
                prerequisites=prerequisites,
            ))
        return gaps

    def _immediate_prerequisite_names(self, block: dict[str, Any], mastered_skills: list[str]) -> list[str]:
        mastered_block_ids = self._resolve_mastered_block_ids(mastered_skills)
        names: list[str] = []
        for prerequisite_id in block.get("prerequisites", []):
            if prerequisite_id in mastered_block_ids:
                continue
            prerequisite = self.knowledge_catalog_tool.get_block(str(prerequisite_id))
            names.append(str(prerequisite.get("block_name", prerequisite_id)) if prerequisite else str(prerequisite_id))
        return names

    async def _insert_unmet_direct_prerequisite_gaps(
        self,
        gaps: list[KnowledgeGap],
        mastered_skills: list[str],
    ) -> list[KnowledgeGap]:
        """Insert only the target's unmet direct prerequisites before that target."""
        mastered_block_ids = self._resolve_mastered_block_ids(mastered_skills)
        known_block_ids = {gap.block_id for gap in gaps if gap.block_id}
        ordered: list[KnowledgeGap] = []
        inserted_block_ids: set[str] = set()
        for gap in gaps:
            if not gap.block_id:
                ordered.append(gap)
                continue
            try:
                prerequisites = await self.graph_query_tool.query_direct_prerequisites(gap.block_id)
            except Exception:
                prerequisites = []
            for prerequisite in prerequisites:
                prerequisite_id = str(prerequisite.get("block_id", ""))
                if not prerequisite_id or prerequisite_id in mastered_block_ids or prerequisite_id in known_block_ids or prerequisite_id in inserted_block_ids:
                    continue
                block = self.knowledge_catalog_tool.get_block(prerequisite_id) or prerequisite
                ordered.append(KnowledgeGap(
                    skill=str(block.get("block_name", prerequisite_id)),
                    block_id=prerequisite_id,
                    block_name=str(block.get("block_name", prerequisite_id)),
                    mapping_confidence=0.85,
                    level="beginner",
                    prerequisites=[],
                ))
                inserted_block_ids.add(prerequisite_id)
            ordered.append(gap)
        return ordered

    def _resolve_mastered_block_ids(self, mastered_skills: list[str]) -> set[str]:
        """Resolve only catalog-backed mastered skills before skipping prerequisites."""
        block_ids: set[str] = set()
        for skill in mastered_skills:
            resolved = self.knowledge_catalog_tool.resolve_block(skill)
            if resolved and resolved.get("block_id"):
                block_ids.add(str(resolved["block_id"]))
                continue
            candidates = self.knowledge_catalog_tool.search(skill, top_k=1)
            if candidates and candidates[0].get("matched_terms"):
                block_ids.add(str(candidates[0].get("block_id", "")))
        return {block_id for block_id in block_ids if block_id}

    @staticmethod
    def _needs_domain_clarification(target_text: str) -> bool:
        lowered = target_text.casefold()
        return "lora" in lowered and any(
            term in lowered
            for term in ("wireless", "radio", "lorawan", "iot", "通信", "无线", "物联网")
        )

    @staticmethod
    def _with_domain_clarification_question(questions: list[str]) -> list[str]:
        question = "你提到的 LoRa 是指大模型参数高效微调 LoRA，还是物联网无线通信 LoRa？"
        return list(dict.fromkeys([question, *questions]))

    @staticmethod
    def _needs_training_optimization(target_text: str) -> bool:
        normalized = target_text.casefold()
        return any(keyword in normalized for keyword in (
            "high efficiency", "efficient finetuning", "stable training",
            "production training", "高效微调", "稳定训练", "生产训练",
        ))

    def _extract_mastered_skills(self, text: str, target_text: str) -> list[str]:
        mastery_text = text
        cue = re.search(
            r"(?:want to learn|想学习|想学|学习目标|目标|不会|不熟悉|希望掌握|计划学习)",
            text,
            flags=re.IGNORECASE,
        )
        if cue:
            mastery_text = text[:cue.start()]
        elif target_text and target_text != text:
            mastery_text = text.replace(target_text, "")

        target_block_ids = {
            candidate.get("block_id")
            for candidate in self.knowledge_catalog_tool.search(target_text, top_k=8)
        }
        mastered: list[str] = []
        for candidate in self.knowledge_catalog_tool.search(mastery_text, top_k=8):
            if candidate.get("block_id") in target_block_ids:
                continue
            resolved = self.knowledge_catalog_tool.get_block(str(candidate.get("block_id", ""))) or candidate
            mastered.append(self.knowledge_catalog_tool.canonical_skill_label(resolved, mastery_text))
        if not mastered:
            mastered = [skill for skill in SKILL_NAMES if skill.lower() in mastery_text.lower()]
        return list(dict.fromkeys(mastered))

    @staticmethod
    def _infer_gap_level(target_text: str, difficulty: str) -> str:
        lowered = target_text.lower()
        if any(keyword in lowered for keyword in ("零基础", "入门", "beginner", "从头学")):
            return "beginner"
        if any(keyword in lowered for keyword in ("精通", "深入", "高级", "advanced", "生产环境", "性能优化")):
            return "advanced"
        if any(keyword in lowered for keyword in ("进阶", "提升", "intermediate", "有基础")):
            return "intermediate"
        return {"初级": "beginner", "中级": "intermediate", "高级": "advanced"}.get(difficulty, "beginner")

    @staticmethod
    def _has_daily_hours(text: str) -> bool:
        return bool(re.search(r"(?:每天|每日|每阶段)?.{0,8}?[0-9]+(?:\.[0-9]+)?\s*(?:小时|h|H)", text))

    @staticmethod
    def _has_learning_style(text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in ("实操", "实践", "动手", "理论", "混合", "practice", "theory", "mixed"))

    @staticmethod
    def _build_followup_questions(
        structured: dict,
        has_hours: bool,
        has_style: bool,
        mapped: bool,
    ) -> list[str]:
        questions: list[str] = []
        if not structured.get("goal") and not mapped:
            questions.append("你希望最终完成哪一种微调或部署任务？")
        if not structured.get("skill_tags"):
            questions.append("你目前已经掌握哪些相关技能？例如 Python、PyTorch、Transformer。")
        if not has_hours:
            questions.append("你每个学习阶段大约能安排多少小时？")
        if not has_style:
            questions.append("你更偏好理论讲解、动手实操，还是两者结合？")
        return questions

    def _target_text(self, text: str) -> str:
        return self._extract_goal_from_sentences(text) or text

    async def generate_stage(self, request: StagePlanningInput) -> StagePlanningOutput:
        """Stage planning functionality merged from Agent2 into Agent1."""
        _agent_log("Agent1", f"stage_planning:start | session={request.session_id} | stage={request.stage_num}")

        gap = next(
            (item for item in request.profile.knowledge_gaps if float(item.progress or 0.0) < 1.0),
            None,
        )
        if gap is None:
            raise ValueError("No unmastered knowledge gap remains")

        preferred_block = {}
        try:
            preferred_block = await self.knowledge_block_query_tool.query_block(gap.block_id or gap.skill)
        except Exception:
            preferred_block = {}
        if not preferred_block and gap.block_id:
            preferred_block = self.knowledge_catalog_tool.get_block(gap.block_id) or {}
        preferred_block = preferred_block or {}

        # Plan from the target's reverse hard-prerequisite closure.  The DB
        # supplies the currently unlocked nodes; neither block order nor soft
        # graph relations are allowed to make a node blocking.
        target_subtopic_ids = list(gap.target_subtopic_ids)
        if not target_subtopic_ids:
            target_subtopic_ids = [
                item["subtopic_id"] for item in preferred_block.get("subtopics", [])
                if item.get("subtopic_id")
            ]
        hard_levels = await self.graph_query_tool.minimal_hard_prerequisite_subgraph(
            target_subtopic_ids,
            request.profile.goal,
        )

        candidates = self.db_client.list_learnable_subtopics(
            session_id=request.session_id,
            preferred_block_id=preferred_block.get("block_id") or gap.block_id,
            max_hours=request.profile.daily_hours,
        )
        if hard_levels:
            # Include nodes that will become unlocked after an earlier item in
            # this same stage. Their prerequisites are checked again below.
            candidates = [
                item for item in self.db_client.list_subtopic_statuses(request.session_id)
                if int(item.get("status", -1)) in (-1, 0) and item["subtopic_id"] in hard_levels
            ]
            candidates.sort(
                key=lambda item: (
                    hard_levels[item["subtopic_id"]],
                    0 if item.get("estimated_hours", 0) <= request.profile.daily_hours else 1,
                    item.get("sequence_index", 0),
                    item["subtopic_id"],
                )
            )
            # A goal that directly names its own block (for example, "learn
            # Python") should fill the daily budget with the following
            # sequential subtopics in that same block. Do not do this while
            # traversing prerequisites for a different final target such as
            # LoRA, because those blocks are only preparatory.
            if candidates and candidates[0].get("block_id") == preferred_block.get("block_id"):
                planned_ids = {item["subtopic_id"] for item in candidates}
                for item in self.db_client.list_subtopic_statuses(request.session_id):
                    if (
                        int(item.get("status", -1)) in (-1, 0)
                        and item.get("block_id") == preferred_block.get("block_id")
                        and item["subtopic_id"] not in planned_ids
                    ):
                        candidates.append(item)
                candidates.sort(key=lambda item: (
                    int(item.get("sequence_index") or 0), item["subtopic_id"],
                ))
        
        # A learner may explicitly continue after a low score. Keep the current
        # subtopic unfinished, but do not create a new numbered stage for it.
        excluded_subtopic_ids = set(request.excluded_subtopic_ids)
        if excluded_subtopic_ids:
            candidates = [
                item for item in candidates
                if item.get("subtopic_id") not in excluded_subtopic_ids
            ]

        # Fall back to a block-level stage when the catalog has no candidate.
        if not candidates:
            return await self._generate_fallback_stage(request, gap, preferred_block)

        # A stage consumes the learner's daily budget. Keep catalog order and
        # only group currently unlocked nodes from the same planned closure.
        selected_subtopics = []
        budget = max(0.5, min(8.0, float(request.profile.daily_hours or 2.0)))
        total_hours = 0.0
        # Build this from the complete status list rather than using the
        # block-scoped helper: prerequisites can belong to an earlier block.
        completed_ids = {
            item["subtopic_id"]
            for item in self.db_client.list_subtopic_statuses(request.session_id)
            if int(item.get("status", -1)) == 1
        }
        for candidate in candidates:
            if selected_subtopics and candidate.get("block_id") != selected_subtopics[0].get("block_id"):
                continue
            prerequisites = candidate.get("prerequisites") or []
            if isinstance(prerequisites, str):
                try:
                    prerequisites = json.loads(prerequisites)
                except json.JSONDecodeError:
                    prerequisites = []
            if any(prerequisite not in completed_ids for prerequisite in prerequisites):
                continue
            hours = max(0.0, float(candidate.get("estimated_hours") or 0.0))
            if selected_subtopics and hours > 0 and total_hours + hours > budget:
                break
            selected_subtopics.append(candidate)
            total_hours += hours
            completed_ids.add(candidate["subtopic_id"])
        if not selected_subtopics:
            return await self._generate_fallback_stage(request, gap, preferred_block)
        subtopic = selected_subtopics[0]
        block_info = preferred_block
        if subtopic.get("block_id") and subtopic.get("block_id") != preferred_block.get("block_id"):
            try:
                block_info = await self.knowledge_block_query_tool.query_block(subtopic["block_id"])
            except Exception:
                block_info = preferred_block
        if not block_info and subtopic.get("block_id"):
            block_info = self.knowledge_catalog_tool.get_block(subtopic["block_id"]) or {}
        block_info = block_info or preferred_block or {}

        # 阶段内容必须以最终选中的子知识点为准。gap_skill 保留目标缺口，
        # 用于后续阶段推进；它不能覆盖当前阶段的教学主题。
        stage_gap = gap.model_dump(mode="json")
        stage_gap["skill"] = _shorten_subtopic_name(subtopic.get("subtopic_name") or "") or block_info.get("block_name") or gap.skill
        stage_block_info = dict(block_info)
        if subtopic.get("estimated_hours") is not None:
            stage_block_info["estimated_hours"] = subtopic["estimated_hours"]

        try:
            generated_stage = await self.stage_generator_tool.generate_stage(
                request.session_id,
                stage_gap,
                stage_block_info,
                request.profile.goal,
                request.stage_num,
                profile=request.profile.model_dump(mode="json"),
            )
        except Exception:
            generated_stage = {}
        generated_stage = generated_stage or {}

        stable_key = f"{request.session_id}:{request.stage_num}:{subtopic['subtopic_id']}"
        stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        block_id = subtopic.get("block_id") or block_info.get("block_id") or generated_stage.get("block_id") or f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, gap.skill).hex[:8].upper()}"
        block_name = block_info.get("block_name") or subtopic.get("block_name") or generated_stage.get("block_name") or gap.skill
        estimated_hours = round(total_hours, 1)
        # Environment setup may be catalogued as 0h. When it is the only
        # selected item, present a practical minimum rather than 0 hours.
        if estimated_hours <= 0:
            estimated_hours = 0.5
        estimated_hours, recommended_daily_hours, suggested_study_days = self._stage_time_plan(
            estimated_hours,
            request.profile.daily_hours,
        )
        prerequisites = subtopic.get("prerequisites")
        if prerequisites is None:
            prerequisites = block_info.get("prerequisites") or generated_stage.get("prerequisites") or gap.prerequisites
        if isinstance(prerequisites, str):
            try:
                prerequisites = json.loads(prerequisites)
            except json.JSONDecodeError:
                prerequisites = []

        stage = StageInfo(
            stage_id=stage_id,
            stage_num=request.stage_num,
            block_id=block_id,
            block_name=block_name,
            block_ids=list(dict.fromkeys(
                item.get("block_id") for item in selected_subtopics if item.get("block_id")
            )) or [block_id],
            subtopic_id=subtopic.get("subtopic_id"),
            subtopic_name=subtopic.get("subtopic_name"),
            subtopic_ids=[item.get("subtopic_id") for item in selected_subtopics if item.get("subtopic_id")],
            subtopic_names=[item.get("subtopic_name") for item in selected_subtopics if item.get("subtopic_name")],
            gap_skill=gap.skill,
            estimated_hours=estimated_hours,
            prerequisites=prerequisites,
            title=generated_stage.get("title"),
            description=generated_stage.get("description"),
            objectives=generated_stage.get("objectives", []),
            difficulty=generated_stage.get("difficulty"),
            recommended_daily_hours=recommended_daily_hours,
            suggested_study_days=suggested_study_days,
            learning_style_notes=generated_stage.get("learning_style_notes"),
            hardware_notes=generated_stage.get("hardware_notes"),
            personalization_summary=generated_stage.get("personalization_summary"),
        )
        _agent_log(
            "Agent1",
            f"stage_planning:done | block={stage.block_name} | block_id={stage.block_id} | subtopics={','.join(stage.subtopic_ids) or '-'} | hours={stage.estimated_hours} | days={stage.suggested_study_days}",
        )
        return StagePlanningOutput(stage=stage)

    async def _generate_fallback_stage(
        self,
        request: StagePlanningInput,
        gap: KnowledgeGap,
        preferred_block: dict[str, Any],
    ) -> StagePlanningOutput:
        # 修复：fallback 前先过滤 target_subtopic_ids 里已完成的 subtopic，
        # 避免反复返回同一个已掌握 subtopic 导致阶段无限重复。
        # 先查询当前 session 的 subtopic 状态，找出已完成集合
        try:
            all_statuses = self.db_client.list_subtopic_statuses(request.session_id)
            completed_ids = {
                s.get("subtopic_id", "")
                for s in all_statuses
                if int(s.get("status", -1)) == 1
            }
        except Exception:
            completed_ids = set()

        # 将 gap.target_subtopic_ids 过滤为「未完成」集合
        target_ids = [sid for sid in list(getattr(gap, "target_subtopic_ids", []) or []) if sid not in completed_ids]

        # 如果 target_subtopic_ids 为空或全部完成，再从 block.subtopics 中找第一个未完成的
        block_subtopics = preferred_block.get("subtopics", []) or []
        if not target_ids:
            for sub in block_subtopics:
                sid = str(sub.get("subtopic_id") or sub.get("id") or "")
                if sid and sid not in completed_ids:
                    target_ids = [sid]
                    break

        # 如果仍无候选 → 此 gap 下无未掌握知识点，返回空 stage，让调用方停止
        if not target_ids:
            empty_stage = StageInfo(
                stage_id=f"empty-{request.session_id}-{request.stage_num}",
                stage_num=request.stage_num,
                block_id=gap.block_id or preferred_block.get("block_id") or "",
                block_name=gap.skill or gap.block_name or preferred_block.get("block_name") or "",
                subtopic_id="",
                subtopic_name="",
                subtopic_ids=[],
                subtopic_names=[],
                gap_skill=gap.skill,
                estimated_hours=0.0,
                prerequisites=[],
                title="已完成全部知识点",
                description="当前 gap 下没有需要学习的知识点了",
                objectives=[],
                difficulty="初级",
                recommended_daily_hours=0.0,
                suggested_study_days=0,
            )
            _agent_log("Agent1", f"stage_planning:fallback | gap fully mastered, stop")
            return StagePlanningOutput(stage=empty_stage)

        # 构造一个临时 gap，target_subtopic_ids 为未完成集合（确保 stage_generator_tool 选对）
        gap_dict = gap.model_dump(mode="json")
        gap_dict["target_subtopic_ids"] = target_ids

        try:
            fallback_stage = await self.stage_generator_tool.generate_stage(
                request.session_id,
                gap_dict,
                preferred_block,
                request.profile.goal,
                request.stage_num,
                profile=request.profile.model_dump(mode="json"),
            )
        except Exception:
            fallback_stage = {}
        fallback_stage = fallback_stage or {}

        stage_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{request.session_id}:{request.stage_num}:{gap.skill}",
        ))
        block_id = (
            gap.block_id
            or preferred_block.get("block_id")
            or f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, gap.skill).hex[:8].upper()}"
        )
        block_name = gap.skill or gap.block_name or preferred_block.get("block_name")

        # subtopic 单值 + 列表字段：从 stage_generator_tool 返回值提取
        # （stage_generator_tool 已按知识点级生成 subtopic_ids/subtopic_names 列表）
        fb_sub_id = fallback_stage.get("subtopic_id")
        fb_sub_name = fallback_stage.get("subtopic_name")
        fb_sub_ids = list(fallback_stage.get("subtopic_ids", []) or [])
        fb_sub_names = list(fallback_stage.get("subtopic_names", []) or [])
        if not fb_sub_ids and fb_sub_id:
            fb_sub_ids = [fb_sub_id]
        if not fb_sub_names and fb_sub_name:
            fb_sub_names = [fb_sub_name]
        if not fb_sub_id and fb_sub_ids:
            fb_sub_id = fb_sub_ids[0]
        if not fb_sub_name and fb_sub_names:
            fb_sub_name = fb_sub_names[0]

        # estimated_hours 兜底：对 0 小时 subtopic 给 0.5 小时（避免预估时长 0 问题）
        raw_hours = fallback_stage.get("estimated_hours")
        if raw_hours is None or float(raw_hours or 0.0) < 0.1:
            raw_hours = max(0.5, float(len(fb_sub_ids or []) or 1) * 0.5)
        estimated_hours, recommended_daily_hours, suggested_study_days = self._stage_time_plan(
            raw_hours,
            request.profile.daily_hours,
        )
        stage = StageInfo(
            stage_id=stage_id,
            stage_num=request.stage_num,
            block_id=block_id,
            block_name=block_name,
            subtopic_id=fb_sub_id,
            subtopic_name=fb_sub_name,
            subtopic_ids=fb_sub_ids,
            subtopic_names=fb_sub_names,
            gap_skill=gap.skill,
            estimated_hours=estimated_hours,
            prerequisites=gap.prerequisites,
            title=fallback_stage.get("title"),
            description=fallback_stage.get("description"),
            objectives=fallback_stage.get("objectives", []),
            difficulty=fallback_stage.get("difficulty"),
            recommended_daily_hours=recommended_daily_hours,
            suggested_study_days=suggested_study_days,
            learning_style_notes=fallback_stage.get("learning_style_notes"),
            hardware_notes=fallback_stage.get("hardware_notes"),
            personalization_summary=fallback_stage.get("personalization_summary"),
        )
        _agent_log(
            "Agent1",
            f"stage_planning:fallback | block={stage.block_name} | subtopics={','.join(fb_sub_ids)} | hours={stage.estimated_hours} | days={stage.suggested_study_days}",
        )
        return StagePlanningOutput(stage=stage)

    @staticmethod
    def _stage_time_plan(estimated_hours: float, daily_hours: float) -> tuple[float, float, int]:
        """Turn one atomic knowledge point into a learner-visible time plan."""
        effective_hours = max(0.0, round(float(estimated_hours or 0.0), 1))
        daily_budget = max(0.5, min(8.0, float(daily_hours or 2.0)))
        recommended_daily_hours = min(effective_hours, daily_budget)
        suggested_study_days = max(1, math.ceil(effective_hours / daily_budget))
        return effective_hours, recommended_daily_hours, suggested_study_days

    def _parse_structured_input(self, text: str) -> dict:
        goal = self._labeled_value(
            text,
            ("我的学习目标", "学习目标", "learning goal", "goal"),
        ) or self._extract_goal_from_sentences(text)
        return {
            "grade": self._labeled_value(text, ("我的身份背景", "身份背景", "学习背景", "background")),
            "skill_tags": self._parse_tags(self._labeled_value(
                text,
                ("我的已有基础", "已有基础", "知识基础", "mastered skills", "mastered"),
            )),
            "goal": goal,
            "daily_hours": self._extract_daily_hours(self._labeled_value(
                text,
                ("我每天可以学习", "每天可以学习", "每日学习时长", "我每阶段可以学习", "每阶段学习时长", "每阶段可以学习", "daily study hours", "daily hours"),
            ) or text),
            "learning_style": self._extract_learning_style(self._labeled_value(
                text,
                ("我的学习方式", "学习方式", "learning style"),
            ) or text),
        }

    @staticmethod
    def _labeled_value(text: str, labels: tuple[str, ...]) -> str:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[：:]\s*([^\n]+)"
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" \t，。,.；;")
        return ""

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        if not raw:
            return []
        normalized = raw.strip().casefold()
        if normalized in {"none", "no foundation", "no prior knowledge", "0基础", "零基础", "无", "无基础"}:
            return []
        parts = re.split(r"[、,，/|；;\n]+", raw)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _extract_goal_from_sentences(text: str) -> str:
        patterns = (
            r"(?:我想要掌握|我想掌握|我想学习|我想学|我希望掌握|我希望学习)\s*(.+)",
            r"\b(?:i\s+)?want\s+to\s+(?:learn|master)\s+(.+)",
        )
        for line in text.splitlines():
            for pattern in patterns:
                match = re.search(pattern, line, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip(" \t，。,.；;")
        return ""

    @staticmethod
    def _extract_daily_hours(text: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:小时|h|H)", text)
        if not match:
            match = re.search(r"(?:每天|每日|每阶段).{0,8}?([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return 2.0
        hours = float(match.group(1))
        return max(0.5, min(8.0, hours))

    @staticmethod
    def _extract_learning_style(text: str) -> str:
        lowered = text.lower()
        if "practice" in lowered or "实操" in text or "实践" in text or "动手" in text:
            return "practice"
        if "theory" in lowered or "理论" in text or "原理" in text:
            return "theory"
        if "mixed" in lowered or "混合" in text or "结合" in text:
            return "mixed"
        return "mixed"


class HybridDiagnosticAgent(DeterministicDiagnosticAgent):
    def __init__(self, db_client: SQLiteClient | None = None):
        super().__init__(db_client=db_client)
        self.llm_client = LLMClient()

    async def run(self, request: DiagnosticInput) -> DiagnosticOutput:
        fallback = await super().run(request)
        candidate_query = fallback.profile.goal or request.user_input
        recalled_candidates = self._recall_candidates(candidate_query)
        candidates = [self.knowledge_catalog_tool._prompt_candidate(item) for item in recalled_candidates]
        authoritative_profile = {
            "grade": fallback.profile.grade,
            "skill_tags": fallback.profile.skill_tags,
            "goal": fallback.profile.goal,
            "daily_hours": fallback.profile.daily_hours,
            "learning_style": fallback.profile.learning_style,
        }
        prompt = f"""你只负责在候选知识目录中细化学习者的知识盲区，不得改写用户已经明确填写的画像字段。

用户原始描述：
{request.user_input}

已由确定性解析得到的事实字段（只读，不得修改）：
{json.dumps(authoritative_profile, ensure_ascii=False)}

候选知识板块与子知识点：
{json.dumps(candidates, ensure_ascii=False)}

只返回严格 JSON：
{{"knowledge_gaps":[{{"block_id":"KB-12","level":"beginner","target_subtopic_ids":["KBS-12-01"]}}],"low_confidence":false}}

规则：
1. block_id 只能从候选目录选择，禁止创造新板块或自由改名。
2. 保留目标里明确提到的多个独立方向，例如 LoRA/QLoRA 与 vLLM 部署应分别覆盖。
3. target_subtopic_ids 只能填写该板块候选中真实存在的 ID；无法确定时返回空数组。
4. level 只能是 beginner、intermediate、advanced。
5. 已有基础不能误判为知识盲区，除非目标明确要求继续进阶。
6. 若候选中没有与目标明确匹配的板块，或用户语境属于候选目录外领域，返回 {{"knowledge_gaps":[],"low_confidence":true}}，不要硬选。
"""
        response = await self.llm_client.complete(
            prompt,
            system_prompt="你是知识目录分类器。只做候选选择和细化，必须输出严格JSON。",
            temperature=0.2,
            json_mode=True,
        )
        data = self._parse_json(response) or {}
        low_confidence = bool(data.get("low_confidence", False)) or self._needs_domain_clarification(candidate_query)
        normalized_gaps = await self._merge_grounded_gaps(
            fallback.profile.knowledge_gaps,
            data.get("knowledge_gaps", []),
            candidates,
            candidate_query,
            low_confidence=low_confidence,
        )
        profile = fallback.profile.model_copy(update={"knowledge_gaps": normalized_gaps})
        confidence = self._hybrid_profile_confidence(
            request.user_input,
            profile,
            low_confidence=low_confidence,
        )
        profile = profile.model_copy(update={"confidence": confidence})
        structured = self._parse_structured_input(request.user_input)
        structured["skill_tags"] = structured.get("skill_tags") or profile.skill_tags
        followup_questions = self._build_followup_questions(
            structured=structured,
            has_hours=self._has_daily_hours(request.user_input),
            has_style=self._has_learning_style(request.user_input),
            mapped=any(gap.block_id for gap in normalized_gaps),
        ) if confidence < 0.7 else []
        if confidence < 0.7 and self._needs_domain_clarification(candidate_query):
            followup_questions = self._with_domain_clarification_question(followup_questions)
        return DiagnosticOutput(profile=profile, followup_questions=followup_questions)

    async def _merge_grounded_gaps(
        self,
        fallback_gaps: list[KnowledgeGap],
        llm_gaps: list[Any],
        candidates: list[dict[str, Any]],
        target_text: str,
        low_confidence: bool = False,
    ) -> list[KnowledgeGap]:
        by_block_id = {gap.block_id: gap for gap in fallback_gaps if gap.block_id}
        ungrounded = [gap for gap in fallback_gaps if not gap.block_id]
        allowed_ids = {
            str(candidate.get("block_id", ""))
            for candidate in candidates
            if candidate.get("block_id")
        }

        llm_candidates = [] if low_confidence else llm_gaps
        for raw_gap in llm_candidates if isinstance(llm_candidates, list) else []:
            if isinstance(raw_gap, str):
                raw_gap = {"skill": raw_gap}
            if not isinstance(raw_gap, dict):
                continue
            raw_block = str(raw_gap.get("block_id") or raw_gap.get("skill") or raw_gap.get("block_name") or "")
            resolved = self.knowledge_catalog_tool.resolve_block(raw_block, candidates)
            if not resolved:
                continue
            block_id = str(resolved.get("block_id", ""))
            if allowed_ids and block_id not in allowed_ids:
                continue
            subtopic_values = raw_gap.get("target_subtopic_ids") or raw_gap.get("subtopics") or []
            target_subtopic_ids = self.knowledge_catalog_tool.resolve_subtopic_ids(block_id, subtopic_values)
            level = self._normalize_level(raw_gap.get("level"), target_text, str(resolved.get("difficulty", "")))
            existing = by_block_id.get(block_id)
            candidate = next((item for item in candidates if item.get("block_id") == block_id), {})
            candidate_confidence = float(candidate.get("mapping_confidence") or resolved.get("mapping_confidence") or 0.7)
            if existing:
                by_block_id[block_id] = existing.model_copy(update={
                    "level": level,
                    "target_subtopic_ids": target_subtopic_ids or existing.target_subtopic_ids,
                    "mapping_confidence": max(existing.mapping_confidence, candidate_confidence),
                })
                continue
            try:
                prerequisites = await self.graph_query_tool.query_prerequisites(block_id)
            except Exception:
                prerequisites = []
            by_block_id[block_id] = KnowledgeGap(
                skill=self.knowledge_catalog_tool.canonical_skill_label(resolved, target_text),
                block_id=block_id,
                block_name=str(resolved.get("block_name", block_id)),
                target_subtopic_ids=target_subtopic_ids,
                mapping_confidence=candidate_confidence,
                level=level,
                prerequisites=prerequisites,
            )

        grounded = list(by_block_id.values())
        return grounded or ungrounded

    def _hybrid_profile_confidence(
        self,
        text: str,
        profile: UserProfile,
        low_confidence: bool = False,
    ) -> float:
        structured = self._parse_structured_input(text)
        score = 0.15
        if profile.goal and any(gap.block_id for gap in profile.knowledge_gaps):
            score += 0.3
        if structured.get("grade"):
            score += 0.1
        if profile.skill_tags:
            score += 0.15
        if self._has_daily_hours(text):
            score += 0.15
        if self._has_learning_style(text):
            score += 0.1
        if profile.knowledge_gaps and all(gap.mapping_confidence >= 0.8 for gap in profile.knowledge_gaps):
            score += 0.1
        score = round(max(0.3, min(0.95, score)), 3)
        has_low_evidence = low_confidence or not profile.knowledge_gaps or any(
            not gap.block_id or float(gap.mapping_confidence or 0.0) < 0.6
            for gap in profile.knowledge_gaps
        )
        return min(score, 0.6) if has_low_evidence else score

    def _normalize_level(self, value: Any, target_text: str, difficulty: str) -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "beginner": "beginner",
            "intermediate": "intermediate",
            "advanced": "advanced",
            "初级": "beginner",
            "中级": "intermediate",
            "高级": "advanced",
            "入门": "beginner",
            "进阶": "intermediate",
        }
        return aliases.get(normalized) or self._infer_gap_level(target_text, difficulty)

    def _parse_json(self, response: str) -> dict | None:
        if not response or response.startswith("Error:"):
            return None
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None


class DeterministicContentAgent:
    def __init__(self, database_url: str = None):
        self.vector_search_tool = VectorSearchTool()
        self.question_extract_tool = QuestionExtractTool(database_url=database_url)
        self.video_search_tool = VideoSearchTool()
        self.learning_resource_tool = LearningResourceTool()
        # 以 session 为键保存当前轮完整问题列表。块级流水线先一次性交付
        # issue_package，再由 Agent3 从该队列中并发取块修改。
        self._revision_queues: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    @staticmethod
    def _stage_subtopic_ids(stage: StageInfo) -> list[str]:
        ids = list(getattr(stage, "subtopic_ids", []) or [])
        if not ids and stage.subtopic_id:
            ids = [stage.subtopic_id]
        return list(dict.fromkeys(item for item in ids if item))

    def _stage_block_ids(self, stage: StageInfo) -> list[str]:
        """解析阶段全部板块，兼容旧的单 block_id 数据。"""
        explicit_ids = list(getattr(stage, "block_ids", []) or [])
        resolved_ids = self.vector_search_tool.get_block_ids_for_subtopics(
            self._stage_subtopic_ids(stage),
        )
        return list(dict.fromkeys(
            item for item in [*explicit_ids, *resolved_ids, stage.block_id] if item
        ))

    @staticmethod
    def _stage_subtopic_names(stage: StageInfo) -> list[str]:
        """返回阶段内所有清理过括号/截断后的知识点名称（按 subtopic_ids 对齐）。

        优先使用 subtopic_names 列表，否则降级到单个 subtopic_name，最后兜底 block_name。
        """
        names = list(getattr(stage, "subtopic_names", []) or [])
        if not names:
            single = getattr(stage, "subtopic_name", "") or ""
            names = [single] if single else []
        cleaned = [_shorten_subtopic_name(n) for n in names if n]
        if not cleaned:
            fallback = getattr(stage, "block_name", "") or "当前阶段"
            cleaned = [fallback]
        return cleaned

    async def _search_stage_docs(self, stage: StageInfo) -> list[dict]:
        subtopic_ids = self._stage_subtopic_ids(stage)
        local_lessons = self.vector_search_tool.get_local_lessons(
            stage.block_id, subtopic_ids,
        )
        local_subtopic_ids = {
            item.get("subtopic_id") for item in local_lessons if item.get("subtopic_id")
        }

        # Local lessons are the authoritative learner-facing materials.  Fall
        # back only for a stage subtopic that has no imported local lesson.
        docs = list(local_lessons)
        for subtopic_id in subtopic_ids:
            if subtopic_id in local_subtopic_ids:
                continue
            subtopic_docs = await self.vector_search_tool.search_by_block(
                stage.block_id, subtopic_id, top_k=6,
            )
            for doc in subtopic_docs:
                doc["subtopic_id"] = subtopic_id
            docs.extend(subtopic_docs)

        unique_docs = []
        seen_doc_keys = set()
        for doc in docs:
            doc_key = doc.get("doc_id") or doc.get("chunk_id")
            if not doc_key or doc_key in seen_doc_keys:
                continue
            seen_doc_keys.add(doc_key)
            unique_docs.append(doc)
        return unique_docs

    async def _get_stage_videos(self, stage: StageInfo) -> list[dict]:
        videos = []
        for subtopic_id in self._stage_subtopic_ids(stage):
            videos.extend(await self.learning_resource_tool.get_recommended_videos(stage.block_id, subtopic_id))
        return list({item.get("resource_id"): item for item in videos if item.get("resource_id")}.values())

    async def get_stage_supplements(self, stage: StageInfo) -> tuple[list[TestQuestion], list[dict]]:
        """Load database-backed assessment questions and videos for any package pipeline."""
        videos = await self._get_stage_videos(stage)
        try:
            extracted = await self.question_extract_tool.extract_questions(
                "",
                {
                    "block_id": stage.block_id,
                    "block_name": stage.block_name,
                    "subtopic_ids": self._stage_subtopic_ids(stage),
                    # 每个 subtopic 抽 2 题；这里作为总量下限
                    "question_count": 8,
                },
            )
        except Exception:
            extracted = []
        questions = [
            TestQuestion.model_validate(question)
            for question in extracted
            if isinstance(question, dict)
        ]
        return questions, videos

    async def run(self, request: ContentGenerationInput) -> ContentGenerationOutput:
        _agent_log("Agent3", f"content:start | block={request.stage.block_name}")
        stage = request.stage
        related_docs = []
        try:
            related_docs = await self._search_stage_docs(stage)
        except Exception:
            related_docs = []
        questions, recommended_videos = await self.get_stage_supplements(stage)
        tutorial = self._build_retrieved_material_tutorial(stage, related_docs)
        practice_note = self._practice_note(request.practice_profile)
        if practice_note:
            tutorial += f"\n## 实训提示\n\n{practice_note}\n"
        sources = []
        if related_docs:
            for i, doc in enumerate(related_docs):
                doc_id = doc.get("doc_id") or doc.get("id") or f"doc_{i}"
                sources.append(Source(
                    source_id=f"src_{doc_id}",
                    source_type="knowledge_doc",
                    title=doc.get("title", ""),
                    locator=doc.get("source") or doc.get("chunk_id") or doc_id,
                    content_ref=doc.get("chunk_id") or doc_id,
                    metadata_json={
                        "block_id": doc.get("block_id", stage.block_id),
                        "subtopic_id": doc.get("subtopic_id"),
                        "doc_id": doc_id,
                    },
                ))

        for video in recommended_videos:
            sources.append(Source(
                source_id=f"src_video_{video['resource_id']}",
                source_type="video",
                title=video["title"],
                locator=f"learning_resources:{video['resource_id']}",
                content_ref=video["url"],
                metadata_json={
                    "block_id": video["block_id"],
                    "subtopic_id": video["subtopic_id"],
                    "review_status": video["review_status"],
                },
            ))

        # 视频资源兜底：若 learning_resource_tool 未返回视频，用 video_search_tool 补充
        if not recommended_videos:
            try:
                video_results = await self.video_search_tool.search_by_block(
                    block_id=stage.block_id,
                    block_name=stage.block_name,
                    top_k=5,
                )
                for v in video_results:
                    sources.append(Source(
                        source_id=v["video_id"],
                        source_type="video",
                        title=v["title"],
                        locator=v["url"],
                        content_ref=v.get("topic", ""),
                        metadata_json={
                            "topic": v.get("topic", ""),
                            "duration_hours": v.get("duration_hours", 0),
                            "block_id": v.get("block_id", ""),
                        },
                    ))
            except Exception:
                pass

        technical_source = next(
            (source for source in sources if source.source_type == "knowledge_doc"),
            None,
        )

        # 强制溯源（辩论方案 第 1 点）：有知识库来源才作为可核验关键 Claim，
        # 否则标 unsupported（无依据），不进入辩论。
        _tech_chunk_id = ""
        if technical_source is not None:
            _tech_chunk_id = technical_source.content_ref or technical_source.source_id or ""
        claims = [
            _build_sourced_claim(
                claim_id="clm_001",
                text=f"{stage.block_name}的核心概念和学习步骤",
                section_ref="tutorial:sec_0",
                source_id=technical_source.source_id if technical_source else None,
                chunk_id=_tech_chunk_id,
                claim_type="concept",
                confidence=0.8,
            )
        ]

        package = LearningPackage(
            session_id=request.session_id,
            stage_id=stage.stage_id,
            block_id=stage.block_id,
            block_name=stage.block_name,
            tutorial=tutorial,
            questions=questions,
            sources=sources,
            claims=claims,
            recommended_videos=recommended_videos,
        )
        _agent_log("Agent3", f"content:done | materials={len(related_docs)} | videos={len(recommended_videos)} | questions={len(package.questions)} | tutorial_len={len(package.tutorial)}")
        return ContentGenerationOutput(learning_package=package)

    def _build_retrieved_material_tutorial(self, stage: StageInfo, docs: list[dict]) -> str:
        """Build a usable package directly from the local lessons when LLM output is unavailable."""
        # 延迟导入，避免与 api.v1.learning_resources 形成循环依赖
        from api.v1.learning_resources import _nest_document_markdown

        names_by_id = dict(zip(stage.subtopic_ids, stage.subtopic_names))
        parts = [
            f"# {stage.block_name}",
            f"本阶段预计 {stage.estimated_hours:g} 小时，覆盖：" + "、".join(
                stage.subtopic_names or [stage.subtopic_name or stage.block_name]
            ),
        ]
        for subtopic_id in self._stage_subtopic_ids(stage):
            topic_docs = [item for item in docs if item.get("subtopic_id") == subtopic_id]
            heading = names_by_id.get(subtopic_id, subtopic_id)
            parts.append(f"## {heading}")
            if not topic_docs:
                parts.append("本知识点暂未匹配到本地学习资料。")
                continue
            source = topic_docs[0].get("source")
            title = topic_docs[0].get("title") or heading
            if source:
                parts.append(f"资料来源：[{title}]({source})")
            # 用 _nest_document_markdown 处理 content：清理面包屑/（①）标记、
            # 去掉参考链接行、并把文章内部的 H2→H3、H3→H4，避免与顶层 ## heading
            # 混淆导致前端 getTopicTutorial 切片错位。
            nested_contents = [
                _nest_document_markdown(str(item.get("content") or ""))
                for item in topic_docs[:3]
                if item.get("content")
            ]
            parts.extend(nested_contents)
        return "\n\n".join(parts)

    @staticmethod
    def _practice_note(practice_profile: PracticeProfile) -> str:
        """Adapt executable examples to stated conditions without selecting a setup."""
        hardware = practice_profile.hardware.strip()
        if not hardware:
            return ""
        normalized = hardware.casefold()
        if any(marker in normalized for marker in ("rtx", "gtx", "nvidia", "cuda")):
            return (
                f"已记录你的实训条件：{hardware}。示例会按本地 NVIDIA GPU 可执行的"
                "参数和验证方式组织，并保留与本知识点直接相关的最小步骤。"
            )
        if any(marker in normalized for marker in ("cloud", "云", "colab")):
            return (
                f"已记录你的实训条件：{hardware}。示例会按你已使用的云端环境给出"
                "可执行参数和验证方式，不改变本阶段的知识点目标。"
            )
        return (
            f"已记录你的实训条件：{hardware}。示例会据此调整可执行范围和验证方式，"
            "不改变本阶段的知识点目标。"
        )

    async def generate_answer(self, request: AnswerGenerationInput) -> AnswerGenerationOutput:
        _agent_log("Agent3", f"answer:start | question={request.question[:60]}")

        # === 问题分类：technical（技术类）| daily（日常类）===
        # Deterministic 模式无 LLM，用关键词兜底分类
        question_type = self._classify_question_by_keywords(request.question)
        _agent_log("Agent3", f"answer:classified | type={question_type}")

        # 检测是否为"学习内容/阶段信息"询问 — 结构化回答，不依赖 LLM
        question = request.question
        learning_content_keywords = [
            "学习阶段", "阶段内容", "当前学", "知识点", "学习内容", "当前阶段",
            "学什么", "学习什么", "阶段是什么", "现在学", "我的学习",
            "什么阶段", "在什么阶段", "哪个阶段", "学到哪", "学到哪里",
            "进度", "在学", "现在在", "目前在", "当前进度",
        ]
        is_learning_content_query = any(kw in question for kw in learning_content_keywords)

        if is_learning_content_query:
            subtopic_names = self._stage_subtopic_names(request.stage)
            subtopic_count = len(subtopic_names)
            objectives = getattr(request.stage, "objectives", []) or []
            goal = getattr(request.profile, "goal", "") or "学习"
            block_name = getattr(request.stage, "block_name", "") or "当前阶段"
            estimated_hours = getattr(request.stage, "estimated_hours", 0) or 0
            difficulty = getattr(request.stage, "difficulty", "") or ""
            daily_hours = getattr(request.profile, "daily_hours", 2.0) or 2.0

            learning_points = []
            try:
                if request.block_id:
                    docs = await self._search_stage_docs(request.stage)
                    for doc in docs[:5]:
                        content = doc.get("content", "") or ""
                        if content and len(content) > 10:
                            learning_points.append(content[:120])
            except Exception:
                pass

            stage_num = getattr(request.stage, "stage_num", "?")
            answer_lines = [f"## 当前学习阶段：第{stage_num}阶段 — {block_name}"]
            # 多知识点编号列表展示（阶段是知识点级，一个阶段可含多个知识点）
            label = "本阶段知识点" if subtopic_count > 1 else "知识点"
            answer_lines.append(f"**{label}**（{subtopic_count}个）：")
            for i, name in enumerate(subtopic_names, 1):
                answer_lines.append(f"  {i}. {name}")
            if difficulty:
                answer_lines.append(f"**难度等级**：{difficulty}")
            answer_lines.append(f"**预估时长**：{estimated_hours} 小时（每天学习 {daily_hours} 小时）")

            if objectives:
                answer_lines.append("\n**本阶段学习目标**：")
                for i, obj in enumerate(objectives[:5], 1):
                    answer_lines.append(f"  {i}. {obj}")

            if learning_points:
                answer_lines.append("\n**核心学习内容**：")
                for point in learning_points:
                    answer_lines.append(f"  • {point}")

            answer_lines.append(f"\n**学习目标**：{goal}")

            answer_text = "\n".join(answer_lines)

            # 追加历史 memory
            memory_context = self._build_memory_context(request)
            if memory_context:
                answer_text += f"\n\n{memory_context}"

            _agent_log("Agent3", f"answer:structured_learning_content | len={len(answer_text)}")
            return AnswerGenerationOutput(answer=answer_text, question_type="technical")

        if question_type == "daily":
            # 日常类：只用历史对话 memory + 画像，跳过 Agent4
            answer = self._generate_daily_answer(request)
            return AnswerGenerationOutput(answer=answer, question_type="daily")

        # 自测/检验掌握程度 意图：走专门路径返回本阶段练习题，不依赖 LLM 自由发挥
        # （本地 LLM 易把"如何检验掌握程度"误解为"用代码捕捉错误"）
        if self._is_self_assessment_query(question):
            answer_text = await self._build_self_assessment_answer(request)
            memory_context = self._build_memory_context(request)
            if memory_context:
                answer_text += f"\n\n{memory_context}"
            _agent_log("Agent3", f"answer:structured_self_assessment | len={len(answer_text)}")
            return AnswerGenerationOutput(answer=answer_text, question_type="technical")

        # 技术类：问题分析 + 本阶段相关知识文档检索 + 历史 memory 整合
        # 按用户问题语义检索本阶段相关文档（不是取阶段所有文档，避免 LLM 被带偏）
        related_docs = []
        try:
            # 解析本阶段所有 block_ids（包含 subtopic_ids 推导 + stage.block_ids + stage.block_id）
            stage_block_ids = self._stage_block_ids(request.stage)
            # 合入请求显式传入的单个 block_id（如果有）
            if request.block_id and request.block_id not in stage_block_ids:
                stage_block_ids = [*stage_block_ids, request.block_id]
            stage_subtopic_ids = self._stage_subtopic_ids(request.stage)
            if stage_block_ids or stage_subtopic_ids:
                related_docs = await self.vector_search_tool.search(
                    request.question,
                    block_ids=stage_block_ids,
                    subtopic_ids=stage_subtopic_ids,
                    top_k=5,
                )
            if not related_docs:
                related_docs = await self.vector_search_tool.search_by_query(request.question, top_k=3)
        except Exception:
            related_docs = []

        # 构建历史 memory 上下文
        memory_context = self._build_memory_context(request)

        # 阶段知识点概览（让学习者知道当前阶段覆盖哪些知识点）
        stage_topic_names = self._stage_subtopic_names(request.stage)
        stage_topic_text = "、".join(stage_topic_names)
        if related_docs:
            references = "\n".join(f"- {item.get('content', '')[:160]}" for item in related_docs)
            answer = (
                f"你当前学习的是 {request.stage.block_name}（本阶段知识点：{stage_topic_text}）。"
                f"针对“{request.question}”，可以先参考这些与问题相关的知识片段：\n{references}"
            )
        else:
            answer = (
                f"你当前学习的是 {request.stage.block_name}（本阶段知识点：{stage_topic_text}）。"
                f"针对“{request.question}”，建议先回到本阶段核心概念，再结合练习验证理解。"
            )

        # 如果有历史 memory 上下文，追加到回答中
        if memory_context:
            answer = f"{answer}\n\n{memory_context}"

        return AnswerGenerationOutput(answer=answer, question_type="technical")

    # ------------------------------------------------------------------
    # 自测/检验掌握程度 意图识别与结构化回答
    # ------------------------------------------------------------------
    # 本地 LLM 容易把"如何检验掌握程度"误解为"用代码捕捉错误（try-except）"，
    # 因此在技术类 LLM 回答之前用规则识别这类意图，走专门路径返回本阶段练习题
    # 作为自测方案，结果可控且可操作。
    SELF_ASSESSMENT_ACTION_WORDS = (
        "检验", "测试", "检查", "评估", "考察", "验证", "衡量", "考查",
    )
    SELF_ASSESSMENT_TARGET_WORDS = (
        "掌握程度", "掌握情况", "学习效果", "理解程度", "理解情况",
        "学会", "学得怎样", "学得怎么样", "水平", "掌握",
    )
    SELF_ASSESSMENT_STRONG_WORDS = ("自测", "测验", "自我检测", "自我测试")

    @classmethod
    def _is_self_assessment_query(cls, question: str) -> bool:
        """检测是否为"检验/测试掌握程度"类自测意图。"""
        q = question or ""
        if any(w in q for w in cls.SELF_ASSESSMENT_STRONG_WORDS):
            return True
        has_action = any(w in q for w in cls.SELF_ASSESSMENT_ACTION_WORDS)
        has_target = any(w in q for w in cls.SELF_ASSESSMENT_TARGET_WORDS)
        return has_action and has_target

    async def _build_self_assessment_answer(self, request: AnswerGenerationInput) -> str:
        """构建自测方案回答：提取本阶段练习题作为自测题目。

        不依赖 LLM 自由发挥，直接用 test_questions 表的本阶段题目构建结构化
        自测方案；无题目时降级为通用自测建议。
        """
        stage = request.stage
        block_name = getattr(stage, "block_name", "") or "当前阶段"
        goal = getattr(request.profile, "goal", "") or "学习"
        subtopic_names = self._stage_subtopic_names(stage)
        stage_num = getattr(stage, "stage_num", "?")

        questions: list[dict] = []
        try:
            extracted = await self.question_extract_tool.extract_questions(
                "",
                {"block_id": stage.block_id, "block_name": block_name, "question_count": 3},
            )
            questions = [q for q in extracted if isinstance(q, dict)] if extracted else []
        except Exception:
            questions = []

        answer_lines = [f"## 检验本阶段掌握程度（第{stage_num}阶段 — {block_name}）"]

        label = "本阶段知识点" if len(subtopic_names) > 1 else "知识点"
        answer_lines.append(f"**{label}**（{len(subtopic_names)}个）：")
        for i, name in enumerate(subtopic_names, 1):
            answer_lines.append(f"  {i}. {name}")

        if questions:
            answer_lines.append(f"\n**自测题目**（{len(questions)}道，建议先独立作答）：")
            for i, q in enumerate(questions, 1):
                q_text = (q.get("question_text", "") or "").strip()
                if not q_text:
                    continue
                difficulty = q.get("difficulty", "") or ""
                header = f"### 题目{i}"
                if difficulty:
                    header += f"（{difficulty}）"
                answer_lines.append(header)
                answer_lines.append(q_text)
                options_text = self._format_question_options(q.get("options", ""))
                if options_text:
                    answer_lines.append(options_text)
            answer_lines.append("\n**自测建议**：")
            answer_lines.append("- 先独立作答，不要查资料；")
            answer_lines.append("- 作答后可以问我某道题的答案或解析；")
            answer_lines.append("- 如果某题完全没思路，说明对应知识点需要复习。")
        else:
            answer_lines.append("\n**自测建议**：")
            answer_lines.append("- 尝试独立写一个小程序，练习本阶段知识点；")
            answer_lines.append("- 对照上面的知识点列表，用自己的话复述每个知识点的核心概念；")
            answer_lines.append("- 完成后可以问我具体知识点的解析或示例。")

        answer_lines.append(f"\n**学习目标**：{goal}")
        return "\n".join(answer_lines)

    @staticmethod
    def _format_question_options(options: Any) -> str:
        """将题目选项格式化为用户友好的多行文本。

        支持以下输入：
        - list[dict]：如 [{"key": "A", "text": "..."}, ...]
        - JSON 字符串：如 '[{"key":"A","text":"..."}]'
        - 普通字符串：直接返回
        - 其他/空：返回空串
        """
        if not options:
            return ""
        # list[dict] 情况
        if isinstance(options, list):
            lines = []
            for opt in options:
                if isinstance(opt, dict):
                    key = str(opt.get("key", "")).strip()
                    text = str(opt.get("text", "")).strip()
                    if key and text:
                        lines.append(f"{key}. {text}")
                    elif text:
                        lines.append(text)
                elif isinstance(opt, str):
                    lines.append(opt)
            return "\n".join(lines) if lines else ""
        # 字符串情况：尝试解析为 JSON
        if isinstance(options, str):
            stripped = options.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    return DeterministicContentAgent._format_question_options(parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
            return stripped if stripped else ""
        return ""

    @staticmethod
    def _classify_question_by_keywords(question: str) -> str:
        """Deterministic 模式下用规则分类问题（无 LLM）。

        返回 "technical" 或 "daily"。明确技术诉求优先；高置信问候、闲聊、
        情绪表达和学习状态咨询归为 daily，避免进入知识检索与 Agent4 审核。
        """
        q = (question or "").strip()
        q_lower = q.lower()

        stage_learning_markers = (
            "学习难点", "阶段难点", "本阶段难点", "这个阶段的难点", "这一阶段的难点",
            "学习重点", "阶段重点", "易错点", "容易错", "掌握情况", "薄弱点",
        )
        if any(marker in q for marker in stage_learning_markers):
            return "technical"

        # 情绪词可以和技术问题同时出现，例如“CUDA 报错让我很焦虑”。
        # 只要有明确技术诉求，仍应进入知识检索和 Agent4 审核。
        tech_keywords = [
            "代码", "报错", "error", "bug", "算法", "原理", "公式", "实现",
            "函数", "类", "接口", "api", "框架", "配置", "安装", "部署",
            "python", "pytorch", "transformer", "怎么写", "如何实现",
            "区别", "什么是", "解释", "运行", "命令", "参数", "显存", "模型",
            "lora", "qlora", "vllm", "cuda", "deepspeed", "微调", "推理",
            "学习阶段", "阶段内容", "当前学", "知识点", "学习内容", "当前阶段",
        ]
        if any(keyword in q_lower for keyword in tech_keywords):
            return "technical"

        daily_patterns = [
            r"^(你好|您好|嗨|哈喽|hello|hi|早上好|中午好|下午好|晚上好)[啊呀嘛吗！!。,.，\s]*$",
            r"^(谢谢|感谢|多谢|辛苦了|再见|拜拜|晚安|回头见)[你了啊呀！!。,.，\s]*$",
            r"(心情|情绪|焦虑|紧张|难过|沮丧|烦躁|压力|疲惫|好累|累死|崩溃|害怕|担心|没信心|想放弃)",
            r"(学不进去|不想学|不想学习|没有动力|没动力|注意力不集中|状态不好|看不下去|坚持不住)",
            r"(鼓励我|安慰我|陪我聊|聊聊天|闲聊|讲个笑话|你是谁|你叫什么|你在吗)",
            r"(今天过得|今天怎么样|最近怎么样|你好吗|心情怎么样|吃饭了吗|在干嘛)",
            r"(学习计划|学习路线|怎么学|如何学|每天学多久|时间怎么安排|怎么坚持|为什么要学|学这个有什么用)",
        ]
        if any(re.search(pattern, q_lower) for pattern in daily_patterns):
            return "daily"

        # 无法高置信识别时保持技术类，避免漏掉没有显式关键词的技术问题。
        return "technical"

    def _build_memory_context(self, request: AnswerGenerationInput) -> str:
        """构建历史 memory 上下文文本，追加到技术类回答中。

        只使用历史问答和画像摘要，不使用当前学习包内容。
        """
        parts = []
        # 历史问答
        if request.qa_history:
            recent = request.qa_history[-3:]  # 只取最近3轮
            history_text = "\n".join([
                f"- 问：{item.get('question', '')[:80]}\n  答：{item.get('answer', '')[:100]}"
                for item in recent
                if isinstance(item, dict)
            ])
            if history_text:
                parts.append(f"💬 最近对话记录：\n{history_text}")

        # 用户画像摘要
        if request.profile_summary:
            parts.append(f"👤 学习画像：{request.profile_summary}")

        return "\n\n".join(parts)

    def _generate_daily_answer(self, request: AnswerGenerationInput) -> str:
        """生成日常类回答（Deterministic 模式，无 LLM）。

        使用历史对话 memory + 用户画像 + profile_summary，不使用当前学习包内容。
        """
        profile = request.profile
        stage = request.stage

        # 个性化开头
        goal = getattr(profile, "goal", "") or "你的学习目标"
        block_name = getattr(stage, "block_name", "") or "当前阶段"

        # 主体回答
        answer = f"我理解你的感受。你目前的学习目标是「{goal}」，正在学习「{block_name}」。"

        # 引用历史对话
        if request.qa_history:
            last_qa = request.qa_history[-1] if request.qa_history else None
            if isinstance(last_qa, dict) and last_qa.get("question"):
                answer += f"\n\n结合你之前问的「{last_qa.get('question', '')[:50]}」，建议先梳理好基础概念，再逐步深入。"

        # 引用画像信息（profile 对象字段）
        daily_hours = getattr(profile, "daily_hours", None)
        if daily_hours:
            answer += f"\n\n你每天有 {daily_hours} 小时学习时间，合理规划好节奏就好。"

        learning_style = getattr(profile, "learning_style", "")
        if learning_style:
            style_hint = "多动手实践" if learning_style == "practice" else "注重原理理解" if learning_style == "theory" else "理论结合实践"
            answer += f"你的学习风格偏向{style_hint}，可以按这个方向调整。"

        # 引用画像摘要文本（补充 profile 对象未覆盖的信息）
        if request.profile_summary:
            answer += f"\n\n关于你：{request.profile_summary}"

        answer += "\n\n如果遇到具体的技术问题，随时可以问我，我们一起解决。"
        return answer

    async def revise_learning_package(self, request: ContentRevisionInput) -> ContentGenerationOutput:
        package = request.learning_package.model_copy(deep=True)
        suggestions = [
            error.suggestion or error.description
            for error in request.technical_errors + request.quality_errors
        ]
        if suggestions:
            package.tutorial += "\n\n## 审核修订\n\n" + "\n".join(f"- {item}" for item in suggestions)
        package.version += 1
        return ContentGenerationOutput(learning_package=package)

    async def revise_answer(self, request: AnswerRevisionInput) -> AnswerGenerationOutput:
        suggestions = [
            error.suggestion or error.description
            for error in request.technical_errors + request.quality_errors
        ]
        suffix = "；".join(suggestions)
        answer = request.answer if not suffix else f"{request.answer}\n\n补充修订：{suffix}"
        return AnswerGenerationOutput(answer=answer)

    async def reply_to_review(
        self,
        session_id: str,
        issues: List[Dict[str, Any]],
        package: Dict[str, Any],
        debate_history: Optional[List] = None,
    ) -> List[Dict[str, Any]]:
        """对 Agent4 的每条 issue 逐条回复。

        边界规则（协议强制）：
        - must_fix=true 的 issue：只能 accepted 或 partially_accepted，不能 rejected
        - rejected 必须带 source_refs 反证，否则无效
        """
        replies: List[Dict[str, Any]] = []
        sources = package.get("sources", [])

        for issue in issues:
            must_fix = issue.get("must_fix", False)
            canonical_id = issue.get("canonical_error_id", "")

            if must_fix:
                disposition = "partially_accepted"
                reason = f"[{issue.get('error_type', '')}] 根据审核建议修正: {issue.get('suggestion') or issue.get('description', '')}"
                source_refs = issue.get("source_refs") or [
                    s.get("source_id") for s in sources[:1] if isinstance(s, dict)
                ] or [s.source_id for s in sources[:1]]
                patched_sections = [issue.get("location") or "tutorial:sec_0"]
            else:
                disposition = "accepted"
                reason = f"接受审核意见: {issue.get('description', '')}"
                source_refs = issue.get("source_refs") or []
                patched_sections = [issue.get("location") or "tutorial:sec_0"]

            replies.append({
                "canonical_error_id": canonical_id,
                "disposition": disposition,
                "reason": reason,
                "source_refs": source_refs,
                "patched_sections": patched_sections,
            })
        return replies

    async def receive_issue_package(
        self,
        session_id: str,
        issue_package: Dict[str, List[Any]],
        blocks_by_id: Dict[str, Any],
    ) -> List[str]:
        """一次接收 Agent4 的完整问题列表，并建立本轮待修改块队列。

        ``issue_package`` 以 ``content_block_id`` 分组，故 Agent3 在开始修改
        任一块之前已经掌握本轮所有待处理块和对应问题。返回值按块顺序稳定排列，
        Orchestrator 可据此以受控并发执行 ``modify_block``。
        """
        normalized: Dict[str, List[Dict[str, Any]]] = {}
        for content_block_id, raw_issues in (issue_package or {}).items():
            if content_block_id not in blocks_by_id:
                continue
            issues: List[Dict[str, Any]] = []
            for issue in raw_issues or []:
                if hasattr(issue, "model_dump"):
                    issues.append(issue.model_dump(mode="json"))
                elif isinstance(issue, dict):
                    issues.append(dict(issue))
            if issues:
                normalized[content_block_id] = issues

        self._revision_queues[session_id] = normalized
        return sorted(
            normalized,
            key=lambda cid: (
                getattr(blocks_by_id[cid], "block_index", 0),
                cid,
            ),
        )

    def queued_issues(self, session_id: str, content_block_id: str) -> List[Dict[str, Any]]:
        """读取 Agent3 已接收的当前问题队列中的一个块，供修改阶段消费。"""
        return list(self._revision_queues.get(session_id, {}).get(content_block_id, []))

    async def modify_package(
        self,
        session_id: str,
        package: Dict[str, Any],
        errors: Dict[str, Any],
        accepted_issues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """修改学习包内容，只处理 accepted/partially_accepted 的 issue.

        - 内容改写 → claim status="revised"
        - 内容删除 → claim status="dropped"
        - 新增的 claim → status="active"
        - 保留 review_trace_id 不变（同一 trace 内迭代）
        - version 递增
        """
        tech_errors = errors.get("tech", [])

        def _get_attr(item: Any, key: str, default: Any = None) -> Any:
            if hasattr(item, key):
                return getattr(item, key)
            if isinstance(item, dict):
                return item.get(key, default)
            return default

        if accepted_issues is not None:
            accepted_canonical_ids = {
                _get_attr(i, "canonical_error_id") for i in accepted_issues
            }
            tech_errors = [
                e for e in tech_errors
                if _get_attr(e, "canonical_error_id") in accepted_canonical_ids
                or not _get_attr(e, "must_fix", False)
            ]
        if tech_errors and "tutorial" in package:
            suggestions = [
                _get_attr(e, "suggestion") or _get_attr(e, "description", "")
                for e in tech_errors
            ]
            package["tutorial"] += "\n\n## 辩论修订\n\n" + "\n".join(f"- {s}" for s in suggestions)

        # 更新 claims 的 status
        if "claims" in package and isinstance(package["claims"], list):
            accepted_claim_ids = {
                _get_attr(i, "claim_id") for i in (accepted_issues or [])
                if _get_attr(i, "claim_id")
            }
            for claim in package["claims"]:
                claim_id = _get_attr(claim, "claim_id")
                if claim_id in accepted_claim_ids:
                    # 内容改写 → revised
                    if isinstance(claim, dict):
                        claim["status"] = "revised"
                    elif hasattr(claim, "status"):
                        claim.status = "revised"
                elif any(
                    _get_attr(e, "claim_id") == claim_id
                    for e in tech_errors
                ):
                    # 内容删除 → dropped
                    if isinstance(claim, dict):
                        claim["status"] = "dropped"
                    elif hasattr(claim, "status"):
                        claim.status = "dropped"

        # version 递增
        if "version" in package:
            package["version"] = package.get("version", 1) + 1
        return package

    # ==================================================================
    # 块级流水线（任务 3 / 任务 6）：Agent3 逐块生成个性化内容 + 定点修改回传
    # ==================================================================
    def iter_stage_chunks(self, stage: StageInfo) -> List[Dict[str, Any]]:
        """读取当前阶段（block + 子知识点集合）的原始 knowledge_chunk 列表。

        - 按 (subtopic_id, chunk_index) 稳定排序，保证块顺序可复现；
        - 库不可用/为空返回 []（上层据此降级，不臆造内容）。
        每个元素即一个原始 chunk dict，供 generate_block 逐块个性化。
        """
        subtopic_ids = self._stage_subtopic_ids(stage)
        try:
            return self.vector_search_tool.get_stage_chunks(stage.block_id, subtopic_ids)
        except Exception:
            return []

    @staticmethod
    def _chunk_source_note(source_chunk: Dict[str, Any], title: str) -> str:
        """构造人类可读的来源说明（标题路径 + 原文行号，缺失优雅省略）。"""
        parts: List[str] = []
        heading_path = source_chunk.get("heading_path")
        if heading_path:
            parts.append(f"标题路径：{heading_path}")
        elif title:
            parts.append(f"资料：{title}")
        start = source_chunk.get("source_line_start")
        end = source_chunk.get("source_line_end")
        if start is not None and end is not None:
            parts.append(f"原文行号：{start}-{end}")
        chunk_id = source_chunk.get("chunk_id")
        if chunk_id:
            parts.append(f"来源块：{chunk_id}")
        return "；".join(parts) if parts else "来源：本地学习资料"

    def _personalize_chunk_text(
        self,
        stage: StageInfo,
        source_text: str,
        profile: Optional[UserProfile],
        practice: Optional[PracticeProfile],
    ) -> str:
        """确定性块级个性化：不改写事实，只在原文块外围补充可执行/降维提示。

        deterministic 模式无 LLM，保留原始块正文作为事实主体（避免引入幻觉），
        再依据画像/实训条件追加一段个性化说明。Hybrid 版会覆盖为 LLM 改写。
        """
        body = (source_text or "").strip()
        practice_note = self._practice_note(practice) if practice else ""
        style_note = ""
        if profile is not None:
            style = getattr(profile, "learning_style", "") or ""
            level = getattr(profile, "level", "") or getattr(profile, "current_level", "") or ""
            hints = []
            if style:
                hints.append(f"学习风格：{style}")
            if level:
                hints.append(f"当前水平：{level}")
            if hints:
                style_note = "（个性化提示：" + "，".join(hints) + "）"
        parts = [body] if body else []
        if style_note:
            parts.append(style_note)
        if practice_note:
            parts.append(f"实训提示：{practice_note}")
        return "\n\n".join(parts).strip()

    async def generate_block(
        self,
        stage: StageInfo,
        source_chunk: Dict[str, Any],
        profile: Optional[UserProfile] = None,
        practice: Optional[PracticeProfile] = None,
        block_index: int = 0,
        session_id: str = "",
    ) -> ContentBlock:
        """对单个原始 knowledge_chunk 生成一个个性化内容块（ContentBlock）。

        完成条件（任务 3）：每完成一个块即可独立提交给 Agent4 审核。输出保留
        source_chunk_id / 标题路径 / 来源说明 / Claim，Claim 绑定来源块以便块级事实核验。
        异步签名统一：Hybrid 版覆盖为 LLM 改写，deterministic 版仅做保守个性化。
        """
        chunk_id = source_chunk.get("chunk_id") or source_chunk.get("source_chunk_id") or ""
        doc_id = source_chunk.get("doc_id")
        subtopic_id = source_chunk.get("subtopic_id")
        title = source_chunk.get("title") or ""
        source_locator = source_chunk.get("source") or ""
        source_text = source_chunk.get("content") or ""

        content_block_id = "_".join(
            str(p) for p in ("cb", session_id, stage.stage_id, chunk_id or block_index) if p != ""
        )
        content_block_id = re.sub(r"_+", "_", content_block_id).strip("_")

        personalized = self._personalize_chunk_text(stage, source_text, profile, practice)
        source_note = self._chunk_source_note(source_chunk, title)

        # 来源：回指原始块，Agent4 据 content_ref/metadata 的 source_chunk_id 回查核验
        source = Source(
            source_id=f"src_{chunk_id or block_index}",
            source_type="knowledge_doc",
            title=title,
            locator=source_locator or chunk_id,
            content_ref=chunk_id,
            metadata_json={
                "block_id": source_chunk.get("block_id", stage.block_id),
                "subtopic_id": subtopic_id,
                "doc_id": doc_id,
                "source_chunk_id": chunk_id,
                "heading_path": source_chunk.get("heading_path"),
                "source_line_start": source_chunk.get("source_line_start"),
                "source_line_end": source_chunk.get("source_line_end"),
            },
        )

        # 每块一条关键技术 Claim，claim_id 绑定 content_block_id，便于 Agent4 逐块核验。
        # 强制溯源（辩论方案 第 1 点）：有来源块才作为可核验关键 Claim；无来源块的客观陈述
        # 标为 unsupported（无依据），不进入辩论，从源头减少幻觉。
        claim_id = f"clm_{content_block_id}"
        claim = _build_sourced_claim(
            claim_id=claim_id,
            text=source_text or f"{stage.block_name}的核心概念",
            section_ref=f"block:{content_block_id}",
            source_id=source.source_id,
            chunk_id=chunk_id,
            claim_type="concept",
            confidence=0.8,
        )

        return ContentBlock(
            content_block_id=content_block_id,
            session_id=session_id or "",
            stage_id=stage.stage_id,
            block_id=stage.block_id,
            subtopic_id=subtopic_id,
            block_name=stage.block_name,
            block_index=block_index,
            source_chunk_id=chunk_id or None,
            source_doc_id=doc_id,
            heading_path=source_chunk.get("heading_path"),
            heading_level=source_chunk.get("heading_level"),
            source_line_start=source_chunk.get("source_line_start"),
            source_line_end=source_chunk.get("source_line_end"),
            source_note=source_note,
            personalized_content=personalized,
            claims=[claim],
            sources=[source],
            version=1,
            status="draft",
        )

    async def modify_block(
        self,
        block: ContentBlock,
        issues: List[Dict[str, Any]],
        revision_note: Optional[str] = None,
    ) -> ContentBlock:
        """定点修改单个内容块（任务 6：只回传修订后的问题块、版本号和修改说明）。

        - 只处理传入的当前块 issue（块级审核结果），不触碰其它块；
        - version 递增；status 置 revised；写入 revision_note（自述修改说明）；
        - Claim 状态按 issue 更新（有事实类 issue 的 Claim → revised）。
        deterministic 版做保守修订（在正文追加修订说明），Hybrid 版覆盖为 LLM 改写。
        """
        updated = block.model_copy(deep=True)
        applied_notes: List[str] = []
        touched_claim_ids = set()
        for issue in issues or []:
            desc = issue.get("suggestion") or issue.get("description") or ""
            if desc:
                applied_notes.append(desc)
            cid = issue.get("claim_id")
            if cid:
                touched_claim_ids.add(cid)

        if applied_notes:
            updated.personalized_content = (
                (updated.personalized_content or "").rstrip()
                + "\n\n## 修订说明\n\n"
                + "\n".join(f"- {n}" for n in applied_notes)
            )

        for claim in updated.claims:
            if claim.claim_id in touched_claim_ids and claim.status == "active":
                claim.status = "revised"

        updated.version = (block.version or 1) + 1
        updated.status = "revised"
        updated.revision_note = revision_note or (
            "已按审核建议定点修订本块：" + "；".join(applied_notes)
            if applied_notes else "本块无需修改（无有效 issue）"
        )
        return updated


class HybridContentAgent(DeterministicContentAgent):
    def __init__(self, database_url: str = None):
        super().__init__(database_url=database_url)
        self.llm_client = LLMClient()
        self.content_skill_context = get_runtime_skill_context(["content_generation"])
        self.answer_skill_context = get_runtime_skill_context(["question_answering"])

    @staticmethod
    def _effective_length(text: str) -> int:
        """Count visible content consistently for source/output budgets."""
        return len(re.sub(r"\s+", "", text or ""))

    @classmethod
    def _fits_budget(cls, text: str, budget: int) -> bool:
        return bool(text and text.strip()) and budget > 0 and cls._effective_length(text) <= budget

    @staticmethod
    def _source_max_tokens(source_text: str, floor: int = 160, ceiling: int = 1600) -> int:
        # Chinese text is typically encoded in substantially fewer tokens than
        # characters. The post-generation budget check below remains decisive.
        return max(floor, min(ceiling, max(1, HybridContentAgent._effective_length(source_text) // 2)))

    @staticmethod
    def _clip_daily_answer(answer: str, limit: int = DAILY_ANSWER_LIMIT) -> str:
        """Keep a concise daily reply without cutting through a sentence."""
        text = (answer or "").strip()
        if len(text) <= limit:
            return text
        candidate = text[:limit]
        boundary = max(candidate.rfind(mark) for mark in ("\n", "。", "！", "？", ".", "!", "?"))
        if boundary >= 0:
            return candidate[:boundary + 1].strip()
        # A hard character cut would expose a visibly incomplete sentence.
        # Let the prompt limit control this exceptional no-punctuation case.
        return text

    async def _complete_answer_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Never pass an Ollama length-truncated answer to the user."""
        response = await self.llm_client.complete(
            prompt,
            system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if getattr(self.llm_client, "last_done_reason", None) != "length":
            return response

        retry_prompt = (
            f"{prompt}\n\nThe previous attempt was cut off. Return a complete, "
            "concise answer only. Finish every sentence and omit optional detail."
        )
        response = await self.llm_client.complete(
            retry_prompt,
            system_prompt,
            temperature=min(temperature, 0.3),
            max_tokens=max_tokens + 1024,  # 重试时增加 token 预算，给"更简洁但完整"的回答留足空间
        )
        if getattr(self.llm_client, "last_done_reason", None) == "length":
            # A local reasoning model can use most of its token budget before
            # finishing. Keep a non-empty second response instead of replacing
            # a usable partial answer with the emergency fallback.
            if response.strip():
                _agent_log("Agent3", "answer:llm_truncated_twice -> keep_partial_answer")
                return response.strip()
            _agent_log("Agent3", "answer:llm_truncated_twice -> deterministic_fallback")
            return ""
        return response

    async def _compress_to_budget(self, text: str, budget: int) -> str:
        """Give a good response one concise retry before using a safe fallback."""
        if self._fits_budget(text, budget):
            return text.strip()
        if not text or budget <= 0:
            return ""
        prompt = f"""Condense the following content without adding facts. Keep its key explanation and preserve Markdown when present. The result must contain no more than {budget} non-whitespace characters.\n\n{text}"""
        try:
            response = await self.llm_client.complete(
                prompt,
                system_prompt="You are a precise technical editor. Return only the condensed content.",
                max_tokens=self._source_max_tokens(text, floor=96, ceiling=900),
                temperature=0.2,
            )
        except Exception:
            return ""
        return response.strip() if self._fits_budget(response, budget) else ""

    async def run(self, request: ContentGenerationInput) -> ContentGenerationOutput:
        output = await super().run(request)
        _agent_log("Agent3", f"content:llm_enhance | block={request.stage.block_name}")
        stage = request.stage
        related_docs = await self._search_stage_docs(stage)
        learning_profile = request.learning_profile.model_dump(mode="json") if request.learning_profile else {}
        material_context = "\n\n".join(
            f"## 资料：{item.get('title', '')}\n{item.get('content', '')}"
            for item in related_docs
        )
        source_material = "\n\n".join(str(item.get("content") or "") for item in related_docs)
        # 阶段知识点列表（让 LLM 知道本阶段要覆盖的所有知识点，教程结构对齐 agent1 的知识点级阶段）
        stage_topic_names = self._stage_subtopic_names(stage)
        stage_topic_text = "、".join(stage_topic_names)
        prompt = f"""请为学习阶段“{stage.block_name}”生成一份Markdown教程。

本阶段包含 {len(stage_topic_names)} 个知识点：{stage_topic_text}
教程需要覆盖上述所有知识点，按顺序组织章节。

学习者画像：{json.dumps(learning_profile, ensure_ascii=False)}
实训条件：{json.dumps(request.practice_profile.model_dump(mode="json"), ensure_ascii=False)}

要求：
1. 保留原资料中的核心概念、事实与已有结构；
2. 仅根据学习者画像调整表达难度、术语解释和重点顺序；
3. 不得新增代码、示例、练习、总结、额外章节或跨知识点内容；
4. 实训条件仅用于表达适配，不得改变 Agent1 的知识点路径，也不得将本地或云端选择作为教程目标。

本节经检索确认的知识库资料：
{material_context or '无可用资料，不得补充未验证的技术细节。'}

内容生成规范：
{self.content_skill_context}
"""
        tutorial = await self.llm_client.complete(
            prompt,
            system_prompt="你是严谨、清晰的个性化学习助手。",
            temperature=0.4,
            max_tokens=self._source_max_tokens(source_material),
        )
        source_budget = self._effective_length(source_material)
        if source_budget and not self._fits_budget(tutorial, source_budget):
            tutorial = await self._compress_to_budget(tutorial, source_budget)
        if self._is_usable_tutorial(tutorial, stage, related_docs):
            output.learning_package.tutorial = tutorial
            _agent_log("Agent3", f"content:llm_done | tutorial_len={len(tutorial)}")
        else:
            # LLM 输出不可用时，保留 super().run() 生成的带 ## 子标题结构化 tutorial。
            # 不能用 source_material（所有知识点内容的纯拼接，无 ## 分隔）覆盖——
            # 否则前端 getTopicTutorial 无法按知识点切片，所有页都会显示同一份合并内容。
            _agent_log("Agent3", "content:llm_invalid -> keep structured tutorial from base agent")
        return output

    async def generate_block(
        self,
        stage: StageInfo,
        source_chunk: Dict[str, Any],
        profile: Optional[UserProfile] = None,
        practice: Optional[PracticeProfile] = None,
        block_index: int = 0,
        session_id: str = "",
    ) -> ContentBlock:
        """块级个性化（Hybrid）：先构造确定性块骨架，再用 LLM 在原文范围内改写表达。

        仅改写 personalized_content（表达层），不改动 source_chunk_id / 标题路径 /
        Claim 等溯源字段，保证 Agent4 仍能回查来源块做事实核验。LLM 失败或产出
        不可用时保留确定性个性化正文兜底。
        """
        block = await super().generate_block(
            stage, source_chunk, profile, practice, block_index, session_id,
        )
        source_text = (source_chunk.get("content") or "").strip()
        if not source_text:
            return block

        learning_profile = profile.model_dump(mode="json") if profile is not None else {}
        practice_profile = practice.model_dump(mode="json") if practice is not None else {}
        heading_path = source_chunk.get("heading_path") or block.block_name
        prompt = f"""请对以下原始学习资料「块」进行个性化改写，使其贴合学习者画像与实训条件。

所属知识点/标题路径：{heading_path}
学习者画像：{json.dumps(learning_profile, ensure_ascii=False)}
实训条件：{json.dumps(practice_profile, ensure_ascii=False)}

原始资料（本块，禁止引入原文之外未经核验的技术事实）：
{source_text}

要求：
1. 仅对原文进行表达降维、术语释义和重点重排，不得新增事实；
2. 不得新增代码、示例、练习、总结、新章节或跨块内容；
3. 依据画像调整表达难度与重点；实训条件仅用于表达适配；
4. 输出 Markdown 正文，不要输出标题层级之外的元信息。
5. 输出的有效字符数不得超过原始材料的 {self._effective_length(source_text)} 字符。

内容生成规范：
{self.content_skill_context}
"""
        try:
            rewritten = await self.llm_client.complete(
                prompt,
                system_prompt="你是严谨、清晰的个性化学习助手，只在给定资料范围内改写。",
                temperature=0.4,
                max_tokens=self._source_max_tokens(source_text),
            )
        except Exception:
            rewritten = ""
        budget = self._effective_length(source_text)
        if rewritten and not self._fits_budget(rewritten, budget):
            rewritten = await self._compress_to_budget(rewritten, budget)
        if rewritten and self._fits_budget(rewritten, budget):
            block.personalized_content = rewritten.strip()
            _agent_log("Agent3", f"block:llm_done | cb={block.content_block_id} | len={len(rewritten)}")
        else:
            # Deterministic personalization adds notes, so use source_text here
            # to keep the user-facing material within the original budget.
            block.personalized_content = source_text
            _agent_log("Agent3", f"block:llm_invalid -> source_fallback | cb={block.content_block_id}")
        return block

    async def modify_block(
        self,
        block: ContentBlock,
        issues: List[Dict[str, Any]],
        revision_note: Optional[str] = None,
    ) -> ContentBlock:
        """块级定点修改（Hybrid）：用 LLM 依据本块 issue 重写正文，版本递增。

        仅回传修订后的单块（任务 6）；LLM 失败时回退确定性修订。溯源字段不变。
        """
        original_content = (block.personalized_content or "").strip()
        base = await super().modify_block(block, issues, revision_note)
        if not issues:
            return base

        issue_lines = []
        for issue in issues:
            loc = issue.get("in_chunk_position") or issue.get("location") or ""
            desc = issue.get("description") or ""
            sug = issue.get("suggestion") or ""
            issue_lines.append(f"- 位置[{loc}] 问题：{desc}；建议：{sug}")
        issues_text = "\n".join(issue_lines)
        prompt = f"""请依据审核意见对以下个性化内容「块」做定点修订，只改有问题的部分，其余保持不变。

块标识：{block.content_block_id}（来源块 {block.source_chunk_id or '未知'}）
当前正文：
{block.personalized_content}

审核意见（本块）：
{issues_text}

要求：
1. 逐条落实审核意见，不得引入新的未核验技术事实；
2. 保持与来源块一致，不跨块扩展；
3. 输出修订后的完整 Markdown 正文。
4. 修订后的有效字符数不得超过原正文的 {self._effective_length(original_content)} 字符。

内容生成规范：
{self.content_skill_context}
"""
        try:
            rewritten = await self.llm_client.complete(
                prompt,
                system_prompt="你是严谨的个性化学习助手，按审核意见定点修订，不越界。",
                temperature=0.3,
                max_tokens=self._source_max_tokens(original_content),
            )
        except Exception:
            rewritten = ""
        budget = self._effective_length(original_content)
        if rewritten and not self._fits_budget(rewritten, budget):
            rewritten = await self._compress_to_budget(rewritten, budget)
        if rewritten and self._fits_budget(rewritten, budget):
            base.personalized_content = rewritten.strip()
        else:
            base.personalized_content = original_content
        return base

    async def generate_answer(self, request: AnswerGenerationInput) -> AnswerGenerationOutput:
        _agent_log("Agent3", f"answer:hybrid_start | question={request.question[:60]}")

        # === 问题分类：用 LLM 判断 technical / daily ===
        question_type = await self._classify_question_by_llm(request.question)
        _agent_log("Agent3", f"answer:classified | type={question_type}")

        if question_type == "daily":
            # 日常类：整合 memory + 对话历史 + 学习包内容，跳过 Agent4
            answer = await self._generate_daily_answer_llm(request)
            return AnswerGenerationOutput(answer=answer, question_type="daily")

        # 检测是否为"学习内容/阶段信息"询问（虽然被分类为 technical，但需要结构化回答）
        question = request.question
        learning_content_keywords = [
            "学习阶段", "阶段内容", "当前学", "知识点", "学习内容", "当前阶段",
            "学什么", "学习什么", "阶段是什么", "现在学", "我的学习",
            "什么阶段", "在什么阶段", "哪个阶段", "学到哪", "学到哪里",
            "进度", "在学", "现在在", "目前在", "当前进度",
        ]
        is_learning_content_query = any(kw in question for kw in learning_content_keywords)

        if is_learning_content_query:
            # 学习内容询问：直接用阶段数据构建回答，不依赖 LLM 自由发挥
            subtopic_names = self._stage_subtopic_names(request.stage)
            subtopic_count = len(subtopic_names)
            objectives = getattr(request.stage, "objectives", []) or []
            goal = getattr(request.profile, "goal", "") or "学习"
            block_name = getattr(request.stage, "block_name", "") or "当前阶段"
            estimated_hours = getattr(request.stage, "estimated_hours", 0) or 0
            difficulty = getattr(request.stage, "difficulty", "") or ""
            daily_hours = getattr(request.profile, "daily_hours", 2.0) or 2.0

            # 收集本阶段所有知识点的相关文档摘要（按多知识点检索）
            learning_points = []
            try:
                if request.block_id:
                    docs = await self._search_stage_docs(request.stage)
                    for doc in docs[:5]:
                        content = doc.get("content", "") or ""
                        if content and len(content) > 10:
                            learning_points.append(content[:120])
            except Exception:
                pass

            # 构建结构化的学习内容介绍
            stage_num = getattr(request.stage, "stage_num", "?")
            answer_lines = [f"## 当前学习阶段：第{stage_num}阶段 — {block_name}"]
            label = "本阶段知识点" if subtopic_count > 1 else "知识点"
            answer_lines.append(f"**{label}**（{subtopic_count}个）：")
            for i, name in enumerate(subtopic_names, 1):
                answer_lines.append(f"  {i}. {name}")
            if difficulty:
                answer_lines.append(f"**难度等级**：{difficulty}")
            answer_lines.append(f"**预估时长**：{estimated_hours} 小时（每天学习 {daily_hours} 小时）")

            if objectives:
                answer_lines.append("\n**本阶段学习目标**：")
                for i, obj in enumerate(objectives[:5], 1):
                    answer_lines.append(f"  {i}. {obj}")

            if learning_points:
                answer_lines.append("\n**核心学习内容**：")
                for point in learning_points:
                    answer_lines.append(f"  • {point}")

            answer_lines.append(f"\n**学习目标**：{goal}")

            answer_text = "\n".join(answer_lines)

            _agent_log("Agent3", f"answer:structured_learning_content | len={len(answer_text)}")
            return AnswerGenerationOutput(answer=answer_text, question_type="technical")

        # 自测/检验掌握程度 意图：走专门路径返回本阶段练习题，不依赖 LLM 自由发挥
        # （本地 LLM 易把"如何检验掌握程度"误解为"用代码捕捉错误"）
        if self._is_self_assessment_query(question):
            answer_text = await self._build_self_assessment_answer(request)
            _agent_log("Agent3", f"answer:structured_self_assessment | len={len(answer_text)}")
            return AnswerGenerationOutput(answer=answer_text, question_type="technical")

        # 技术类（非学习内容询问）：按用户问题语义检索本阶段相关文档 + LLM 回答
        # 关键：必须按 query 检索与问题相关的文档，而不是取阶段所有文档，
        # 否则 LLM 会被不相关文档带偏（如问"语法"却答"环境准备"）
        related_docs = []
        try:
            stage_block_ids = self._stage_block_ids(request.stage)
            if stage_block_ids:
                # 优先按问题语义 + 当前阶段全部板块/知识点范围检索。
                related_docs = await self.vector_search_tool.search(
                    request.question,
                    block_ids=stage_block_ids,
                    subtopic_ids=self._stage_subtopic_ids(request.stage),
                    top_k=5,
                )
            if not related_docs:
                related_docs = await self.vector_search_tool.search_by_query(request.question, top_k=3)
        except Exception:
            related_docs = []
        context = ""
        if related_docs:
            context = "\n\n以下知识库资料仅供参考（与问题相关时使用，无关时忽略）：\n" + "\n".join(
                f"- {item.get('content', '')[:200]}" for item in related_docs
            )

        # 构建 memory 上下文
        memory_context = self._build_memory_context_prompt(request)

        # 构建硬件/平台信息，让 LLM 生成平台适配的命令
        hardware_note = ""
        hardware = getattr(request.practice_profile, "hardware", "") if request.practice_profile else ""
        if hardware:
            # 判断是否包含明确的操作系统/平台信息
            has_platform = any(kw in hardware.lower() for kw in (
                "windows", "win", "mac", "osx", "linux", "ubuntu", "wsl", "centos", "debian",
            ))
            if has_platform:
                hardware_note = f"\n用户设备/平台：{hardware}（请据此生成对应平台的命令和操作步骤）"
            else:
                # 只有硬件信息无平台信息，提示 LLM 给出跨平台方案
                hardware_note = f"\n用户硬件条件：{hardware}（请优先给出跨平台通用方法；如需命令，请同时提供 Windows 和 Linux 两种写法）"

        # 阶段知识点列表（让 LLM 知道当前阶段覆盖的所有知识点，回答更聚焦）
        stage_topic_names = self._stage_subtopic_names(request.stage)
        stage_topic_text = "、".join(stage_topic_names)
        prompt = f"""当前学习阶段：{request.stage.block_name}
本阶段知识点（{len(stage_topic_names)}个）：{stage_topic_text}
用户学习目标：{request.profile.goal}
用户问题：{request.question}{hardware_note}

请聚焦回答用户提出的实际问题，先给出简明结论，再解释原理；需要时提供示例。
重要：知识库资料仅供参考，如果与用户问题无关请忽略，不要复述资料中与问题无关的内容。不要编造来源。

意图识别提示：
- 若用户询问"如何检验/测试掌握程度、怎么知道学会了、学习效果如何"等自测类问题，应推荐自测方式（做练习题、写小项目、对照知识点复述），不要回答具体的异常捕获或错误处理语法。
- 若用户询问某个具体技术点（语法、数据类型、函数等），直接解释该技术点。
{context}"""
        if memory_context:
            prompt += f"\n\n{memory_context}"
        prompt += f"\n\n答疑规范：\n{self.answer_skill_context}"
        answer = await self._complete_answer_with_retry(
            prompt,
            "你是技术准确、适合教学的学习助手。",
            temperature=0.5,
            max_tokens=TECHNICAL_ANSWER_TOKENS,
        )
        if answer.startswith("Error: DeepSeek"):
            _agent_log("Agent3", "answer:llm_error -> surface_deepseek_error")
            return AnswerGenerationOutput(answer=answer, question_type="technical")
        if self._is_valid_response(answer):
            _agent_log("Agent3", f"answer:llm_done | answer_len={len(answer)}")
            return AnswerGenerationOutput(answer=answer, question_type="technical")
        _agent_log("Agent3", "answer:llm_invalid -> deterministic_fallback")
        return await super().generate_answer(request)

    async def _classify_question_by_llm(self, question: str) -> str:
        """Hybrid 模式下用 LLM 分类问题，失败时回退到关键词分类。"""
        import json as _json
        rule_type = self._classify_question_by_keywords(question)
        # 高置信日常表达直接短路，避免分类模型波动导致问候、闲聊或情绪表达
        # 误入知识库检索和 Agent4 技术审核。
        if rule_type == "daily":
            return "daily"
        # Questions about the learner's current-stage difficulty are technical
        # learning-content queries even if a general classifier calls them daily.
        if any(marker in question for marker in (
            "学习难点", "阶段难点", "本阶段难点", "这个阶段的难点", "这一阶段的难点",
            "学习重点", "阶段重点", "易错点", "容易错", "掌握情况", "薄弱点",
        )):
            return "technical"
        system_prompt = """你是一个问题分类器。请判断用户的问题属于"技术类"还是"日常类"。

分类标准：
- technical（技术类）：涉及代码、算法、概念解释、技术原理、API使用、报错排查、编程实现、框架工具、学习内容/阶段知识询问（如"当前学什么"、"阶段内容是什么"、"知识点有哪些"）等
- daily（日常类）：学习计划/路线咨询（如"我该怎么学"、"学习路线"）、学习状态/情绪（如"今天状态不好"、"学不进去"）、学习意义咨询（如"学这个有什么用"）、非技术闲聊/问候（如"你好"、"谢谢"）

关键区分：问"当前学习内容/知识/阶段"归 technical；问"学习计划/路线/情绪"归 daily。

请只输出一个JSON：{"type": "technical"} 或 {"type": "daily"}"""
        prompt = f"请判断以下问题属于技术类还是日常类：\n用户问题：{question}\n请输出JSON格式结果。"
        try:
            response = await self.llm_client.complete(
                prompt, system_prompt, temperature=0.1, json_mode=True
            )
            if response and not response.startswith("Error:"):
                result = _json.loads(response)
                q_type = result.get("type", "technical").lower()
                if q_type in ("technical", "daily"):
                    return q_type
        except Exception:
            pass
        # LLM 失败回退到父类的关键词分类
        return self._classify_question_by_keywords(question)

    def _build_memory_context_prompt(self, request: AnswerGenerationInput) -> str:
        """构建历史 memory 上下文 prompt 片段，供 LLM 技术类回答参考。

        只使用历史问答和画像摘要，不使用当前学习包内容。
        """
        parts = []
        if request.qa_history:
            recent = request.qa_history[-5:]
            history_text = "\n".join([
                f"- 问：{item.get('question', '')[:80]}\n  答：{item.get('answer', '')[:150]}"
                for item in recent
                if isinstance(item, dict)
            ])
            if history_text:
                parts.append(f"最近对话记录：\n{history_text}")
        if request.profile_summary:
            parts.append(f"学习画像：{request.profile_summary}")
        if parts:
            return "请结合以下用户历史上下文生成更贴合的回答：\n" + "\n\n".join(parts)
        return ""

    async def _generate_daily_answer_llm(self, request: AnswerGenerationInput) -> str:
        """Hybrid 模式下用 LLM 生成日常类回答，只用历史对话 memory + 画像。"""
        # 检测是否为学习内容询问（虽然被分类为 daily，但需要给出实质内容）
        question = request.question
        learning_content_keywords = ["学习阶段", "阶段内容", "当前学", "知识点", "学习内容", "当前阶段", "学什么"]
        is_learning_content_query = any(kw in question for kw in learning_content_keywords)

        if is_learning_content_query:
            # 学习内容询问：给出结构化的当前阶段内容介绍
            subtopic_name = getattr(request.stage, "subtopic_name", "") or ""
            objectives = getattr(request.stage, "objectives", []) or []
            goal = getattr(request.profile, "goal", "") or "学习"
            block_name = getattr(request.stage, "block_name", "") or "当前阶段"

            stage_intro = f"你当前正在学习「{block_name}」"
            if subtopic_name:
                stage_intro += f"（知识点：{subtopic_name}）"

            objectives_text = ""
            if objectives:
                objectives_text = "\n\n本阶段学习目标：\n" + "\n".join(
                    f"  {i+1}. {obj}" for i, obj in enumerate(objectives[:5])
                )

            prompt = f"""用户问：{question}

用户学习目标：{goal}
当前学习阶段：{block_name}
{stage_intro}
{objectives_text}

请基于以上信息，清晰地告诉用户当前阶段的学习内容是什么。用简洁明了的语言列出核心知识点和学习目标。"""

            answer = await self._complete_answer_with_retry(
                prompt,
                "你是一个专业的学习助手，能够清晰地介绍当前学习阶段的内容。",
                temperature=0.3,
                max_tokens=TECHNICAL_ANSWER_TOKENS,
            )
            return answer or self._generate_daily_answer(request)

        # 普通日常类问题：情感支持/学习咨询
        history_text = ""
        if request.qa_history:
            recent = request.qa_history[-5:]
            history_text = "\n".join([
                f"用户问：{item.get('question', '')}\n答：{item.get('answer', '')[:200]}"
                for item in recent
                if isinstance(item, dict)
            ])

        goal = getattr(request.profile, "goal", "") or "学习"
        block_name = getattr(request.stage, "block_name", "") or "当前阶段"

        system_prompt = """你是一个温暖、专业的学习导师。用户提出的是日常类问题（非技术问题）。

请基于以下信息生成有温度、有针对性的回答：
1. 结合用户画像（学习目标、水平、偏好），给出个性化建议
2. 参考历史对话，理解用户当前的学习状态和情绪
3. 不要引用技术学习材料，保持轻松自然的对话风格
4. 语气要亲切自然，像朋友间的对话，避免说教
5. 回答要简洁，不超过150字，最多3句，不重复安慰或追问。"""

        prompt = f"""用户问题：{request.question}

用户当前学习目标：{goal}
当前学习阶段：{block_name}

用户画像摘要：
{request.profile_summary or '暂无'}

历史问答记录：
{history_text or '暂无'}

请生成一个温暖、有针对性的回答。"""

        try:
            response = await self._complete_answer_with_retry(
                prompt,
                system_prompt,
                temperature=0.5,
                max_tokens=DAILY_ANSWER_TOKENS,
            )
            if response.startswith("Error: DeepSeek"):
                return response
            if response and not response.startswith("Error:"):
                return self._clip_daily_answer(response)
        except Exception:
            pass
        # LLM 失败回退到父类的确定性日常回答
        return self._generate_daily_answer(request)

    async def revise_learning_package(self, request: ContentRevisionInput) -> ContentGenerationOutput:
        feedback = self._format_feedback(request.technical_errors, request.quality_errors)
        prompt = f"""请根据审核意见修改学习教程。

学习阶段：{request.stage.block_name}
用户目标：{request.profile.goal}

原教程：
{request.learning_package.tutorial}

审核意见：
{feedback}

要求：逐条解决审核问题，保留正确内容，返回修改后的完整Markdown教程。"""
        tutorial = await self.llm_client.complete(
            prompt,
            system_prompt="你是Agent3内容生成专家，负责依据合并审核意见严谨改稿。",
            temperature=0.3,
            max_tokens=self._source_max_tokens(request.learning_package.tutorial),
        )
        original_budget = self._effective_length(request.learning_package.tutorial)
        if tutorial and not self._fits_budget(tutorial, original_budget):
            tutorial = await self._compress_to_budget(tutorial, original_budget)
        if not self._is_valid_response(tutorial):
            # Keep the reviewed source unchanged instead of using the older
            # deterministic revision path, which appends a revision note.
            return ContentGenerationOutput(learning_package=request.learning_package.model_copy(deep=True))
        package = request.learning_package.model_copy(deep=True)
        package.tutorial = tutorial
        package.version += 1
        return ContentGenerationOutput(learning_package=package)

    async def revise_answer(self, request: AnswerRevisionInput) -> AnswerGenerationOutput:
        feedback = self._format_feedback(request.technical_errors, request.quality_errors)
        hardware = getattr(request.practice_profile, "hardware", "") if request.practice_profile else ""
        if hardware:
            has_platform = any(kw in hardware.lower() for kw in (
                "windows", "win", "mac", "osx", "linux", "ubuntu", "wsl",
            ))
            hardware_note = (
                f"\n用户设备/平台：{hardware}（请据此生成对应平台的命令）"
                if has_platform
                else f"\n用户硬件条件：{hardware}（如需命令，请同时提供 Windows 和 Linux 两种写法）"
            )
        else:
            hardware_note = ""
        prompt = f"""请根据审核意见修改回答。

用户问题：{request.question}
当前阶段：{request.stage.block_name}{hardware_note}

原回答：
{request.answer}

审核意见：
{feedback}

要求：逐条修复问题，保持技术准确和教学清晰，返回完整修改后回答。"""
        answer = await self._complete_answer_with_retry(
            prompt,
            "你是Agent3答疑专家，负责依据合并审核意见修改回答。",
            temperature=0.3,
            max_tokens=TECHNICAL_ANSWER_TOKENS,
        )
        if not self._is_valid_response(answer):
            return AnswerGenerationOutput(answer=request.answer)
        return AnswerGenerationOutput(answer=answer)

    def _format_feedback(self, technical_errors: list[ReviewError], quality_errors: list[ReviewError]) -> str:
        # Agent5已合并至Agent4，所有错误统一在technical_errors中
        lines = []
        for error in technical_errors:
            lines.append(
                f"- [审核][{error.severity}] {error.description}；建议：{error.suggestion or '请修正'}"
            )
        return "\n".join(lines) or "无"

    def _is_valid_response(self, response: str) -> bool:
        return bool(response.strip()) and not response.startswith("Error:")

    def _is_usable_tutorial(
        self, response: str, stage: StageInfo, related_docs: list[dict],
    ) -> bool:
        """Reject successful but off-task local-model responses."""
        if not self._is_valid_response(response):
            return False
        text = response.strip()
        review_markers = (
            "审查结果", "顶层函数", "严重性", "代码审查", "审核结果",
            "输入输出规范", "异常处理规范", "配置化与安全规范", "代码风格参考标准",
            "错误类型及其处理", "Google Python Style Guide", "OWASP",
        )
        if any(marker in text for marker in review_markers):
            return False
        return text.startswith("#") or stage.block_name in text


class DeterministicTechnicalReviewAgent:
    def __init__(self):
        self.rule_check_tool = RuleCheckTool()
        self.knowledge_verify_tool = KnowledgeVerifyTool()
        self.difficulty_analyzer_tool = DifficultyAnalyzerTool()
        self.cognitive_load_tool = CognitiveLoadTool()
        self.style_match_tool = StyleMatchTool()
        self.learning_resource_tool = LearningResourceTool()
        # 委托 Agent4 协议字段补全（对接需求清单 2.1）
        self._protocol_agent = TechnicalReviewAgent()

    async def run(self, request: ReviewInput) -> TechnicalReviewResult:
        errors = []
        video_resource_errors = []
        raw_hallucinations: List[Dict[str, Any]] = []
        if not request.content.strip():
            errors.append(ReviewError(error_type="missing_content", severity="critical", description="待审核内容为空"))
            return TechnicalReviewResult(passed=False, errors=errors)

        # 步骤1：硬性检查（规则检查 — LoRA机制、参数量、安全红线）
        try:
            rule_errors = normalize_review_errors(
                await self.rule_check_tool.check_rules(request.content)
            )
            errors.extend(rule_errors)
        except Exception:
            pass

        # 步骤2：事实检查（幻觉检测 — 知识依据验证，sim<0.65判幻觉）
        # Claim 回退修正（任务 2）：存在有效关键 Claim 时，只对 Claim 核验（结果可为空，
        # 不再回退整篇），避免整篇送检重新产生全文级误报；无有效关键 Claim 时才回退整篇。
        # 保留原始 dict（raw_hallucinations，含显式 claim_id/块溯源）传入协议补全，
        # 避免 normalize 丢弃 claim_id/块字段后仅靠 location 匹配（同 section 多 claim 时会误配）。
        try:
            key_claims = _select_key_claims(request.claims)
            if key_claims:
                claim_hallucinations = await self._protocol_agent._verify_claims_hallucination(
                    key_claims
                )
                raw_hallucinations = claim_hallucinations
                errors.extend(normalize_review_errors(claim_hallucinations))
            else:
                hallucination_errors = normalize_review_errors(
                    await self.knowledge_verify_tool.verify_knowledge(request.content)
                )
                errors.extend(hallucination_errors)
        except Exception:
            pass

        # 步骤3-4：质量审核（难度/认知负荷/风格匹配 — 原Agent5功能）
        if request.target_type == "learning_package":
            profile = request.profile.model_dump(mode="json") if hasattr(request.profile, "model_dump") else request.profile
            try:
                tool_errors = await self.difficulty_analyzer_tool.analyze(request.content, profile=profile)
                tool_errors.extend(await self.cognitive_load_tool.analyze(request.content))
                tool_errors.extend(await self.style_match_tool.check(request.content, profile=profile))
                errors.extend(normalize_review_errors(tool_errors))
            except Exception:
                pass
            # 步骤5：推荐视频资源审核（technical_review.skill.md 规则5）— 异常兜底，
            # tool 内部仅捕获 sqlite3.Error，此处再兜住 OSError 等非 sqlite3 异常，避免中断审核链路。
            try:
                video_resource_errors = await self.learning_resource_tool.validate_recommended_videos(
                    request.recommended_videos,
                    request.stage.block_id,
                    self._stage_subtopic_ids(request.stage),
                )
                errors.extend(normalize_review_errors(video_resource_errors))
            except Exception:
                video_resource_errors = []
        # 步骤5：审核辩论协议字段补全（对接需求清单 2.1，委托 Agent4 实现）
        # raw_errors 汇入视频错误与逐 Claim 幻觉原始 dict（后者含显式 claim_id，供精确归属）
        errors = self._protocol_agent._enrich_debate_fields(
            errors,
            sources=request.sources,
            claims=request.claims,
            round_index=request.round_index,
            raw_errors=list(video_resource_errors) + raw_hallucinations,
        )

        # 步骤6：综合裁决（有critical错误或errors非空则不通过）
        return TechnicalReviewResult(passed=not errors, errors=errors)

    async def rebut(
        self,
        session_id: str,
        issues: List[Dict[str, Any]],
        author_replies: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """委托 Agent4 实现复核裁决（对接需求清单 2.2）。"""
        return await self._protocol_agent.rebut(session_id, issues, author_replies)

    # ==================================================================
    # 块级审核（任务 4）：输入一个个性化内容块，回查来源块做事实核验，
    # 审核范围限定当前块，issue 绑定 content_chunk_id/content_version/claim_id/块内位置。
    # ==================================================================
    async def review_block(
        self,
        block: Any,
        stage: StageInfo,
        profile: Any = None,
        round_index: int = 1,
        trace_id: Optional[str] = None,
        use_llm_semantic: bool = True,
    ) -> TechnicalReviewResult:
        # Deterministic 审核本就不调 LLM，use_llm_semantic 仅为与 Hybrid 保持同签名。
        return await _review_block_impl(
            self, block, stage, profile, round_index, trace_id, use_llm=False,
        )


# DeterministicQualityReviewAgent 和 HybridQualityReviewAgent 已合并至 Agent4
# 质量审核功能已集成到 DeterministicTechnicalReviewAgent 和 HybridTechnicalReviewAgent


class HybridTechnicalReviewAgent:
    def __init__(self):
        self.llm_client = LLMClient()
        self.rule_check_tool = RuleCheckTool()
        self.knowledge_verify_tool = KnowledgeVerifyTool()
        self.difficulty_analyzer_tool = DifficultyAnalyzerTool()
        self.cognitive_load_tool = CognitiveLoadTool()
        self.style_match_tool = StyleMatchTool()
        self.learning_resource_tool = LearningResourceTool()
        self.skill_context = get_skill_context([
            "technical_review",
            "industry_standards",
            "code_error_patterns",
            "pedagogy",
            "learner_model",
            "quality_metrics",
        ])
        # 委托 Agent4 协议字段补全（对接需求清单 2.1）
        self._protocol_agent = TechnicalReviewAgent()

    async def run(self, request: ReviewInput) -> TechnicalReviewResult:
        print(f"=== Agent4(合并后) 技术+质量审核开始：{request.target_type} ===")
        video_resource_errors = []
        # 步骤1：硬性检查（规则检查）— 异常兜底
        try:
            rule_errors = normalize_review_errors(
                await self.rule_check_tool.check_rules(request.content)
            )
        except Exception:
            rule_errors = []

        # 步骤2：事实检查（幻觉检测 — 知识依据验证）— 异常兜底
        # Claim 回退修正（任务 2）：存在有效关键 Claim 时，只对 Claim 核验（结果可为空，
        # 不再回退整篇）；无有效关键 Claim 时才回退整篇。保留原始 dict（raw_hallucinations，
        # 含显式 claim_id/块溯源）供协议补全精确匹配。
        raw_hallucinations: List[Dict[str, Any]] = []
        try:
            key_claims = _select_key_claims(request.claims)
            if key_claims:
                claim_hallucinations = await self._protocol_agent._verify_claims_hallucination(
                    key_claims
                )
                raw_hallucinations = claim_hallucinations
                hallucination_errors = normalize_review_errors(claim_hallucinations)
            else:
                hallucination_errors = normalize_review_errors(
                    await self.knowledge_verify_tool.verify_knowledge(request.content)
                )
        except Exception:
            hallucination_errors = []

        # 步骤3-4：质量审核（原Agent5功能）
        quality_tool_errors = []
        if request.target_type == "learning_package":
            profile = request.profile.model_dump(mode="json") if hasattr(request.profile, "model_dump") else request.profile
            try:
                quality_tool_errors.extend(await self.difficulty_analyzer_tool.analyze(request.content, profile=profile))
                quality_tool_errors.extend(await self.cognitive_load_tool.analyze(request.content))
                quality_tool_errors.extend(await self.style_match_tool.check(request.content, profile=profile))
            except Exception:
                pass
            # 步骤5：推荐视频资源审核（technical_review.skill.md 规则5）— 异常兜底，
            # tool 内部仅捕获 sqlite3.Error，此处再兜住 OSError 等非 sqlite3 异常，避免中断审核链路。
            try:
                video_resource_errors = await self.learning_resource_tool.validate_recommended_videos(
                    request.recommended_videos,
                    request.stage.block_id,
                    DeterministicContentAgent._stage_subtopic_ids(request.stage),
                )
                video_errors = normalize_review_errors(video_resource_errors)
            except Exception:
                video_resource_errors = []
                video_errors = []
        else:
            video_errors = []

        quality_tool_errors_normalized = normalize_review_errors(quality_tool_errors)

        if video_errors:
            quality_tool_errors_normalized.extend(video_errors)

        prompt = self._build_prompt(request)
        response = await self.llm_client.complete(
            prompt,
            system_prompt="你是Agent4(合并后)技术+质量审核专家。必须独立审核，不得因为内容表达流畅就默认正确。",
            temperature=0.1,
            json_mode=True,
        )
        result = self._parse_result(response)
        if rule_errors:
            result.errors = rule_errors + result.errors
            result.passed = False
            print(f"=== 硬性规则检查发现错误：{len(rule_errors)}个 ===")
        if hallucination_errors:
            result.errors = hallucination_errors + result.errors
            result.passed = False
            print(f"=== 事实检查(幻觉)发现错误：{len(hallucination_errors)}个 ===")
        if quality_tool_errors_normalized:
            result.errors = quality_tool_errors_normalized + result.errors
            result.passed = False
            print(f"=== 质量审核发现错误：{len(quality_tool_errors_normalized)}个 ===")

        # 步骤5：审核辩论协议字段补全（对接需求清单 2.1，委托 Agent4 实现）
        # 强制为每个 error 补全 canonical_error_id/error_id/source_refs/change_request/
        # must_fix/must_fix_reason/claim_id（协议 2.3 / 2.4 / 第 6 节）
        result.errors = self._protocol_agent._enrich_debate_fields(
            result.errors,
            sources=request.sources,
            claims=request.claims,
            round_index=request.round_index,
            raw_errors=list(video_resource_errors) + raw_hallucinations,
        )

        print(f"=== Agent4(合并后) 审核完成：passed={result.passed}, errors={len(result.errors)} ===")
        return result

    async def rebut(
        self,
        session_id: str,
        issues: List[Dict[str, Any]],
        author_replies: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """委托 Agent4 实现复核裁决（对接需求清单 2.2）。

        返回 {canonical_error_id: "persist" | "withdraw" | "downgrade"}。
        """
        return await self._protocol_agent.rebut(session_id, issues, author_replies)

    async def review_block(
        self,
        block: Any,
        stage: StageInfo,
        profile: Any = None,
        round_index: int = 1,
        trace_id: Optional[str] = None,
        use_llm_semantic: bool = True,
    ) -> TechnicalReviewResult:
        """块级审核（Hybrid）：确定性块级事实/质量核验 + LLM 语义补充（范围限定当前块）。

        ``use_llm_semantic`` 用于省 token：修改后复核（recheck）只需确定性工具
        （规则/难度/认知负荷/事实核验）判定修订是否消解了原问题，无需每块每轮再花
        一次 LLM 语义审核；首轮 Phase A 仍全量执行 LLM 语义审核，能力不缩水。
        """
        return await _review_block_impl(
            self, block, stage, profile, round_index, trace_id, use_llm=use_llm_semantic,
        )

    def _build_prompt(self, request: ReviewInput) -> str:
        domain_reference = self._domain_reference(request.stage.block_name)
        video_context = "无"
        if request.recommended_videos:
            video_context = "\n".join(
                "- {resource_id}: {title} | {url} | 审核状态={review_status} | 摘要={content_summary}".format(
                    resource_id=video.get("resource_id", "未提供ID"),
                    title=video.get("title", "未提供标题"),
                    url=video.get("url", "未提供URL"),
                    review_status=video.get("review_status", "unknown"),
                    content_summary=video.get("content_summary", "未提供摘要"),
                )
                for video in request.recommended_videos
            )
        return f"""请审核以下{request.target_type}的技术正确性。

学习阶段：{request.stage.block_name}
用户问题：{request.question or '无'}
待审核内容：
{request.content[:12000]}

当前阶段推荐视频：
{video_context}

审核规范：
{self.skill_context}

强制审核步骤：
1. 独立提取并核验每个技术声明，不得沿用原文结论；
2. 重新计算所有参数量、比例和数值示例；
3. 区分“分解原权重”和“增加可训练低秩更新”等不同机制；
4. 任何无法确认的关键声明不得直接判定通过。
5. 推荐视频必须属于当前知识点；视频摘要未审核时，明确指出其不能作为技术事实依据。

当前领域校验基线：
{domain_reference}

只返回严格JSON对象：
{{"passed":true,"errors":[{{"error_type":"fact_error","severity":"major","location":"具体位置","description":"问题说明","evidence":"判断依据","suggestion":"修正建议"}}],"review_details":"审核总结"}}
没有问题时errors必须是空数组。"""

    def _domain_reference(self, block_name: str) -> str:
        if "lora" in block_name.lower():
            return (
                "LoRA冻结原权重W，训练低秩增量ΔW=(α/r)BA；它不是先对W执行SVD后再训练U、Σ、V。"
                "若W维度为d_out×d_in，LoRA可训练参数量为r(d_in+d_out)，示例必须据此计算。"
            )
        return "暂无内置领域基线；关键事实应依据可靠知识或标记为需要复核。"

    def _parse_result(self, response: str) -> TechnicalReviewResult:
        data = parse_json_object(response)
        if data is None:
            error = ReviewError(
                error_type="review_system_error",
                severity="critical",
                description="Agent4审核结果无法解析",
                suggestion="重新执行技术审核",
            )
            return TechnicalReviewResult(passed=False, errors=[error])
        errors = normalize_review_errors(data.get("errors", []))
        return TechnicalReviewResult(
            passed=bool(data.get("passed", not errors)) and not errors,
            errors=errors,
            review_details=data.get("review_details"),
        )


# HybridQualityReviewAgent 已合并至 HybridTechnicalReviewAgent
# 质量审核功能已集成到 HybridTechnicalReviewAgent


def parse_json_object(response: str) -> dict | None:
    if not response or response.startswith("Error:"):
        return None
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def normalize_review_errors(raw_errors: list) -> list[ReviewError]:
    errors = []
    for raw in raw_errors if isinstance(raw_errors, list) else []:
        # 已是 ReviewError 实例则直接收录（跳过二次构造）
        if isinstance(raw, ReviewError):
            errors.append(raw)
            continue
        if not isinstance(raw, dict):
            continue
        severity = str(raw.get("severity", "major")).lower()
        if severity not in {"critical", "major", "minor"}:
            severity = "major"
        errors.append(ReviewError(
            error_type=raw.get("error_type") or raw.get("type") or "unknown",
            severity=severity,
            location=raw.get("location"),
            description=raw.get("description") or raw.get("message") or "未提供问题说明",
            evidence=raw.get("evidence"),
            suggestion=raw.get("suggestion"),
        ))
    return errors


def _block_to_dict(block: Any) -> Dict[str, Any]:
    """把 ContentBlock（模型或 dict）统一转成 dict，容错缺字段。"""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json")
    return {}


async def _review_block_impl(
    agent: Any,
    block: Any,
    stage: StageInfo,
    profile: Any,
    round_index: int,
    trace_id: Optional[str],
    use_llm: bool,
) -> TechnicalReviewResult:
    """块级审核共享实现（Deterministic / Hybrid 共用）。

    任务 4 约束：
      - 输入一个个性化内容块 + 其来源块信息；
      - 回查 source_chunk_id 对应的原始资料做事实核验（verify_against_chunk）；
      - 每条 issue 绑定 content_chunk_id / content_version / claim_id / 块内位置 / 原因 / 证据 / 建议；
      - 保留难度/认知负荷/风格/代码（规则）审核，但审核范围限定为当前块正文。
    """
    bd = _block_to_dict(block)
    content = (bd.get("personalized_content") or "").strip()
    content_block_id = bd.get("content_block_id")
    content_version = bd.get("version", 1)
    source_chunk_id = bd.get("source_chunk_id")
    block_claims = bd.get("claims") or []
    sources = bd.get("sources") or []

    _agent_log(
        "Agent4",
        f"review_block:start | cb={content_block_id} | src_chunk={source_chunk_id} | v={content_version}",
    )

    if not content:
        err = ReviewError(
            error_type="missing_content",
            severity="critical",
            description="内容块正文为空",
            content_block_id=content_block_id,
            content_version=content_version,
        )
        return TechnicalReviewResult(passed=False, errors=[err])

    errors: List[ReviewError] = []
    raw_facts: List[Dict[str, Any]] = []

    # 步骤1：硬性规则检查（限定本块正文）
    try:
        errors.extend(normalize_review_errors(await agent.rule_check_tool.check_rules(content)))
    except Exception:
        pass

    # 步骤2：事实核验（Claim 优先，回查来源块）。块级只对本块 Claim 核验，
    # 不回退整篇；Claim 已带 source_chunk_id，_verify_claims_hallucination 会走 verify_against_chunk。
    key_claims = _select_key_claims(block_claims)
    try:
        if key_claims:
            claim_facts = await agent._protocol_agent._verify_claims_hallucination(key_claims)
            raw_facts = claim_facts
            errors.extend(normalize_review_errors(claim_facts))
        elif source_chunk_id:
            # 无显式关键 Claim，但有来源块：对本块正文按来源块核验（仍绑定确切块，不整篇送检）
            chunk_facts = await agent.knowledge_verify_tool.verify_against_chunk(content, source_chunk_id)
            raw_facts = chunk_facts
            errors.extend(normalize_review_errors(chunk_facts))
        # 既无关键 Claim 又无来源块时，不做全库整篇送检（块级不制造全文级误报）
    except Exception:
        pass

    # 步骤3-4：质量审核（难度/认知负荷/风格），范围限定本块正文
    profile_json = profile.model_dump(mode="json") if hasattr(profile, "model_dump") else profile
    try:
        quality: List[Any] = []
        quality.extend(await agent.difficulty_analyzer_tool.analyze(content, profile=profile_json))
        quality.extend(await agent.cognitive_load_tool.analyze(content))
        quality.extend(await agent.style_match_tool.check(content, profile=profile_json))
        errors.extend(normalize_review_errors(quality))
    except Exception:
        pass

    # 步骤4.5（Hybrid）：LLM 语义审核，范围限定本块，补充事实/表达类问题
    if use_llm:
        try:
            llm_errors = await _llm_review_block(agent, block=bd, stage=stage)
            errors.extend(normalize_review_errors(llm_errors))
        except Exception:
            pass

    # 步骤5：审核辩论协议字段补全（复用现有 enrich，raw_facts 携带 claim_id/块溯源）
    errors = agent._protocol_agent._enrich_debate_fields(
        errors,
        sources=sources,
        claims=block_claims,
        round_index=round_index,
        raw_errors=raw_facts,
    )

    # 步骤6：块级溯源强制回填——每条 issue 必须携带块标识与版本（任务 4 强约束）。
    # in_chunk_position / claim_id 由 enrich/事实核验尽力填充；此处兜底补齐块级字段。
    for e in errors:
        if getattr(e, "content_block_id", None) is None:
            e.content_block_id = content_block_id
        if getattr(e, "content_version", None) is None:
            e.content_version = content_version
        if getattr(e, "content_chunk_id", None) is None and source_chunk_id:
            e.content_chunk_id = source_chunk_id

    _agent_log(
        "Agent4",
        f"review_block:done | cb={content_block_id} | errors={len(errors)}",
    )
    return TechnicalReviewResult(passed=not errors, errors=errors)


async def _llm_review_block(agent: Any, block: Dict[str, Any], stage: StageInfo) -> List[Dict[str, Any]]:
    """Hybrid 块级 LLM 审核：只针对当前块正文找技术/表达问题，输出结构化 issue 列表。

    返回 raw dict issue 列表（error_type/severity/description/evidence/suggestion/in_chunk_position），
    由调用方 normalize + enrich 补齐协议字段。LLM 失败或解析失败返回 []。
    """
    content = (block.get("personalized_content") or "").strip()
    heading_path = block.get("heading_path") or block.get("block_name") or ""
    source_chunk_id = block.get("source_chunk_id") or ""
    prompt = f"""请只审核下面这一个「内容块」的技术正确性与表达质量，不要扩展到其它内容。

所属知识点/标题路径：{heading_path}
来源块：{source_chunk_id or '未知'}

待审核内容块：
{content[:8000]}

审核规范：
{agent.skill_context}

要求：以 JSON 返回 {{"issues": [{{"error_type": "...", "severity": "critical|major|minor", "description": "...", "evidence": "...", "suggestion": "...", "in_chunk_position": "..."}}]}}。
若本块无问题返回 {{"issues": []}}。只针对本块，不要臆造原文之外的事实。"""
    response = await agent.llm_client.complete(
        prompt,
        system_prompt="你是Agent4块级审核专家，只审核给定内容块，不越界，不因表达流畅默认正确。",
        temperature=0.1,
        json_mode=True,
    )
    if not response or not isinstance(response, str):
        return []
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return []
    issues = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(issues, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    for it in issues:
        if isinstance(it, dict) and (it.get("description") or it.get("message")):
            cleaned.append(it)
    return cleaned


class DeterministicStageDecisionAgent:
    def __init__(self):
        self.strategy_selector_tool = StrategySelectorTool()
        self.profile_update_tool = ProfileUpdateTool()
        self.stage_summary_generator_tool = StageSummaryGeneratorTool()
        self.db_client = SQLiteClient()

    async def run(self, request: StageDecisionInput) -> StageDecisionOutput:
        profile = request.profile.model_copy(deep=True)

        # 预先提取原始答题记录与情绪均分，供 BKT 引擎的 Wt/We 修正使用
        # （须在 update_gaps 之前计算，避免 BKT 引擎因缺少原始记录退化为简化公式）
        # 只读层 state_manager.add_answer_record 存储的记录不含 skill/knowledge_point 字段，
        # 若不补齐则 _group_by_skill 会全部归入 "unknown"，导致 update_gaps 按 gap.skill
        # 匹配失败、BKT 引擎退化为聚合兜底公式（Wt/P(G) 失效）。故在 Agent6 层用当前阶段
        # 的 gap_skill 给缺失 skill 的记录补齐归属，使 BKT 引擎真正作用于当前知识点。
        current_skill = request.stage.gap_skill
        pre_question_records: list[dict] = []
        # 优先使用已提交测试的逐题记录（含 time_spent），来源：assessment_answers.time_spent，
        # 由前端 PUT .../answers/{qid} 的 time_spent 提供；缺失时退回从 qa_history 提取。
        assessment_records = getattr(request, "assessment_records", None) or []
        source_records = assessment_records if assessment_records else request.qa_history
        for qa in source_records:
            if isinstance(qa, dict) and ("question_id" in qa or "skill" in qa or "is_correct" in qa):
                rec = dict(qa)
                if not rec.get("skill") and not rec.get("knowledge_point") and current_skill:
                    rec["skill"] = current_skill
                pre_question_records.append(rec)
        pre_emotion_score: float | None = None
        _pre_emotion_list: list[float] = []
        for qa in request.qa_history:
            if isinstance(qa, dict) and "emotion_score" in qa:
                try:
                    _pre_emotion_list.append(float(qa["emotion_score"]))
                except (ValueError, TypeError):
                    pass
        if _pre_emotion_list:
            pre_emotion_score = sum(_pre_emotion_list) / len(_pre_emotion_list)

        # 步骤2：精算掌握度（逐知识点BKT更新）— 异常兜底
        try:
            updated_gaps = await self.profile_update_tool.update_gaps(
                [gap.model_dump(mode="json") for gap in profile.knowledge_gaps],
                request.accuracy,
                question_records=pre_question_records,
                emotion_score=pre_emotion_score,
                skill_tags=list(profile.skill_tags or []),
            )
            profile.knowledge_gaps = [KnowledgeGap.model_validate(gap) for gap in updated_gaps]
        except Exception:
            pass  # 保留原 gaps

        current_gap = next((gap for gap in profile.knowledge_gaps if gap.skill == request.stage.gap_skill), None)
        if current_gap is not None:
            # 用该知识点自身的掌握度判定，不再使用全局accuracy强制覆盖（避免错误地把薄弱知识点标为已掌握）
            if current_gap.progress >= 0.75 and current_gap.status in ("学习中", "未掌握"):
                current_gap.status = "已掌握"
                current_gap.progress = 1.0

        # 步骤2b：画像维度更新 — 各项独立兜底
        try:
            profile.learning_rhythm = await self.profile_update_tool.infer_rhythm(request.interactions)
        except Exception:
            profile.learning_rhythm = "medium"
        try:
            profile.confidence_index = await self.profile_update_tool.infer_confidence(request.qa_history, request.accuracy)
        except Exception:
            profile.confidence_index = max(0.0, min(1.0, request.accuracy))
        try:
            profile.confusion_markers = await self.profile_update_tool.detect_confusion(request.qa_history)
        except Exception:
            profile.confusion_markers = []
        profile.version += 1

        # 步骤3：策略选择（透传 iteration_history，激活 backtrack/终止逻辑）
        # StageDecisionInput 未携带 iteration_history 字段，尝试从 profile 的 model_extra 或 qa_history 推断
        iteration_history = []
        try:
            profile_extra = profile.model_dump(mode="json") if hasattr(profile, "model_dump") else {}
            iteration_history = profile_extra.get("iteration_history") or []
        except Exception:
            iteration_history = []
        try:
            strategy_full = await self.strategy_selector_tool.select_strategy_full(
                request.accuracy,
                profile.model_dump(mode="json"),
                iteration_history=iteration_history,
            )
            strategy = strategy_full.get("strategy_label", "reinforce")
        except Exception:
            strategy = "reinforce" if request.accuracy >= 0.6 else "simplify"
            strategy_full = {"strategy_label": strategy, "backtrack_target": None,
                             "should_terminate": False, "decision": None}
        # 回溯目标（前置知识）：StageDecisionOutput/UserProfile 无独立字段承载，
        # 故拼入阶段总结文本透传给用户，避免目标被静默丢弃（iteration_strategy.skill.md 规则2）
        backtrack_target = strategy_full.get("backtrack_target") if strategy == "backtrack" else None
        # 终止信号：达到最大轮次 / 死循环停滞 / 全部掌握均会触发 should_terminate，
        # 此时策略工具已给出权威 decision（通常为 completed）。之前 run() 忽略该信号、
        # 仅用 accuracy 重算，会丢弃规范的结业结果（iteration_strategy.skill.md 规则3）。
        should_terminate = bool(strategy_full.get("should_terminate", False))
        tool_decision = strategy_full.get("decision")

        # 决策判定（优先级：终止信号/工具权威 decision > backtrack > accuracy 推进）
        # strategy 为 backtrack（回溯）时，decision 映射为 repeat（回退重学前置知识）
        # strategy 为 finish（结业）时，decision 映射为 completed
        # 注意：此处不再用全局 accuracy 强制把 current_gap 置为“已掌握”
        # （与上方 L1064-1068 的逐知识点判定保持一致，避免把薄弱知识点误标为已掌握；
        #  continue/repeat 的推进决策仍由下方 accuracy>=0.7 分支负责，不依赖 gap 状态覆盖）
        all_mastered = all(gap.status == "已掌握" for gap in profile.knowledge_gaps)
        if strategy == "finish" or all_mastered or should_terminate:
            # 尊重策略工具的终止裁决；工具 decision 缺省时兜底为 completed
            decision = tool_decision or "completed"
        elif strategy == "backtrack":
            decision = "repeat"
        elif request.accuracy >= 0.7:
            decision = "continue"
        else:
            decision = "repeat"

        # === 步骤4-5：生成阶段总结报告 ===
        # 从qa_history和interactions推断情绪得分和耗时
        emotion_score: float | None = None
        emotion_scores_list: list[float] = []
        for qa in request.qa_history:
            if isinstance(qa, dict) and "emotion_score" in qa:
                try:
                    emotion_scores_list.append(float(qa["emotion_score"]))
                except (ValueError, TypeError):
                    pass
        if emotion_scores_list:
            emotion_score = sum(emotion_scores_list) / len(emotion_scores_list)

        # 统计实际耗时（秒 → 小时）
        total_seconds = 0.0
        expected_seconds = 0.0
        for interaction in request.interactions:
            if not isinstance(interaction, dict):
                continue
            try:
                total_seconds += float(interaction.get("time_spent", 0) or 0)
                expected_seconds += float(interaction.get("expected_time", 100) or 100)
            except (ValueError, TypeError):
                continue
        # 优先使用已提交测试聚合用时（来自 assessment_answers.time_spent 汇总），
        # 无测试用时（=0）时退回 interactions 估算，保持向后兼容。
        request_time_spent_hours = float(getattr(request, "time_spent_hours", 0) or 0)
        if request_time_spent_hours > 0:
            time_spent_hours = round(request_time_spent_hours, 2)
        else:
            time_spent_hours = round(total_seconds / 3600.0, 2)
        expected_time_hours = round(expected_seconds / 3600.0, 2)

        # 生成qa历史转化为question_records（复用已补齐 skill 归属的 pre_question_records，
        # 保证总结统计与 BKT 引擎输入一致）
        question_records: list[dict] = pre_question_records

        stage_name = request.stage.block_name or f"阶段{request.stage.stage_num}"
        knowledge_gaps_json = [g.model_dump(mode="json") for g in profile.knowledge_gaps]
        confusion_tags = list(profile.confusion_markers) if profile.confusion_markers else []
        pace = profile.learning_rhythm or "medium"

        # 步骤5：生成总结报告 — 异常时降级为简短报告
        try:
            stage_summary = await self.stage_summary_generator_tool.generate_summary(
                stage_name=stage_name,
                accuracy=request.accuracy,
                knowledge_gaps=knowledge_gaps_json,
                confusion_tags=confusion_tags,
                question_records=question_records,
                emotion_score=emotion_score,
                pace_adjustment=pace,
                time_spent_hours=time_spent_hours,
                expected_time_hours=expected_time_hours,
                next_strategy=strategy,
                total_knowledge_points=len(profile.knowledge_gaps),
                total_answers=len(question_records) or len(request.qa_history),
            )
        except Exception:
            stage_summary = (
                f"## 📊 本阶段学习总结\n\n"
                f"### 📈 阶段概览\n【{stage_name}】阶段已结束，正确率{request.accuracy*100:.0f}%。\n\n"
                f"### 💪 鼓励与建议\n你的专属AI导师始终在这里陪你！"
            )
        # 回溯策略时把前置知识目标拼入总结，确保 backtrack_target 不被丢弃
        if strategy == "backtrack" and backtrack_target:
            stage_summary = (stage_summary or "") + f"\n\n### 🔁 回溯目标\n建议回到前置知识【{backtrack_target}】重新学习。"
        return StageDecisionOutput(
            decision=decision,
            profile=profile,
            strategy=strategy,
            stage_summary=stage_summary,
        )


async def run_deterministic_reviews(request: ReviewInput) -> ReviewOutput:
    # Agent5已合并至Agent4，统一由合并后的TechnicalReviewAgent处理
    technical = await DeterministicTechnicalReviewAgent().run(request)
    # 质量审核结果已合并在technical结果中
    quality = QualityReviewResult(
        passed=technical.passed,
        errors=technical.errors,
    )
    return ReviewOutput(technical=technical, quality=quality)
