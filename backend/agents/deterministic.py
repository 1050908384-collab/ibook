# -*- coding: utf-8 -*-
# ← 实际运行的 Agent 实现（含确定性版+混合版）
import json
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
from models.profile import KnowledgeGap, UserProfile
from models.review import QualityReviewResult, ReviewError, TechnicalReviewResult
from models.stage import Claim, LearningPackage, Source, StageInfo
from tools.learning_resource_tool import LearningResourceTool
from agents.agent_04_tech_review import TechnicalReviewAgent
from skills import load_skills
from tools.agent1_tools.calculate_confidence_tool import CalculateConfidenceTool
from tools.agent1_tools.extract_skills_tool import ExtractSkillsTool
from tools.agent1_tools.graph_query_tool import GraphQueryTool
from tools.agent1_tools.knowledge_catalog_tool import KnowledgeCatalogTool
from tools.agent1_tools.semantic_knowledge_recall_tool import SemanticKnowledgeRecallTool
from tools.agent1_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent1_tools.stage_generator_tool import StageGeneratorTool
from tools.agent3_tools.faq_search_tool import FAQSearchTool
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


def get_skill_context(skill_names: list[str]) -> str:
    return "\n\n".join(skill.content for skill in load_skills(skill_names))


def _agent_log(agent_name: str, message: str) -> None:
    print(f"[{agent_name}] {message}", flush=True)


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
        gaps = await self._insert_unmet_direct_prerequisite_gaps(gaps, mastered)

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
            hardware=structured.get("hardware", ""),
            confidence=confidence,
        )
        _agent_log("Agent1", f"diagnostic:done | goal={profile.goal} | gaps={len(profile.knowledge_gaps)} | confidence={profile.confidence:.2f}")
        return DiagnosticOutput(profile=profile, followup_questions=followup_questions)

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
        return bool(re.search(r"(?:每天|每日)?.{0,8}?[0-9]+(?:\.[0-9]+)?\s*(?:小时|h|H)", text))

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
            questions.append("你每天大约能投入多少小时学习？")
        if not has_style:
            questions.append("你更偏好理论讲解、动手实操，还是两者结合？")
        return questions

    def _target_text(self, text: str) -> str:
        match = re.search(
            r"(?:想学习|想学|学习目标(?:是|为)?|目标(?:是|为)?|want to learn|learn)(.+)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip(" ，。,.！!") if match else text

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

        candidates = self.db_client.list_learnable_subtopics(
            session_id=request.session_id,
            preferred_block_id=preferred_block.get("block_id"),
            max_hours=request.profile.daily_hours,
        )
        
        # Fallback: 当没有可学习的 subtopic 时，直接使用 gap 信息生成阶段
        if not candidates:
            profile_dict = request.profile.model_dump(mode="json")
            fallback_stage = await self.stage_generator_tool.generate_stage(
                request.session_id,
                gap.model_dump(mode="json"),
                preferred_block,
                request.profile.goal,
                request.stage_num,
                profile=profile_dict,
            )
            if not fallback_stage:
                fallback_stage = {}
            
            stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{request.session_id}:{request.stage_num}:{gap.skill}"))
            block_id = gap.block_id or preferred_block.get("block_id") or f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, gap.skill).hex[:8].upper()}"
            block_name = gap.skill or gap.block_name or preferred_block.get("block_name")
            
            stage = StageInfo(
                stage_id=stage_id,
                stage_num=request.stage_num,
                block_id=block_id,
                block_name=block_name,
                subtopic_id=None,
                subtopic_name=None,
                gap_skill=gap.skill,
                estimated_hours=fallback_stage.get("estimated_hours", 2.0),
                prerequisites=gap.prerequisites,
                title=fallback_stage.get("title"),
                description=fallback_stage.get("description"),
                objectives=fallback_stage.get("objectives", []),
                difficulty=fallback_stage.get("difficulty"),
                recommended_daily_hours=fallback_stage.get("recommended_daily_hours"),
                suggested_study_days=fallback_stage.get("suggested_study_days"),
                learning_style_notes=fallback_stage.get("learning_style_notes"),
                hardware_notes=fallback_stage.get("hardware_notes"),
                personalization_summary=fallback_stage.get("personalization_summary"),
            )
            _agent_log(
                "Agent1",
                f"stage_planning:fallback | block={stage.block_name} | hours={stage.estimated_hours} | days={stage.suggested_study_days}",
            )
            return StagePlanningOutput(stage=stage)

        preferred_subtopic_ids = set(gap.target_subtopic_ids)
        subtopic = next(
            (item for item in candidates if item.get("subtopic_id") in preferred_subtopic_ids),
            candidates[0],
        )
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
        stage_gap["skill"] = subtopic.get("subtopic_name") or block_info.get("block_name") or gap.skill
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
        if block_id == (gap.block_id or preferred_block.get("block_id")):
            block_name = gap.skill or gap.block_name or block_info.get("block_name") or generated_stage.get("block_name")
        else:
            block_name = block_info.get("block_name") or subtopic.get("block_name") or generated_stage.get("block_name") or gap.skill
        subtopic_hours = subtopic.get("estimated_hours")
        estimated_hours = float(
            subtopic_hours
            if subtopic_hours is not None
            else block_info.get("estimated_hours")
            or generated_stage.get("estimated_hours")
            or 2.0
        )
        prerequisites = subtopic.get("prerequisites")
        if prerequisites is None:
            prerequisites = block_info.get("prerequisites") or generated_stage.get("prerequisites") or gap.prerequisites

        stage = StageInfo(
            stage_id=stage_id,
            stage_num=request.stage_num,
            block_id=block_id,
            block_name=block_name,
            subtopic_id=subtopic.get("subtopic_id"),
            subtopic_name=subtopic.get("subtopic_name"),
            gap_skill=gap.skill,
            estimated_hours=estimated_hours,
            prerequisites=prerequisites,
            title=generated_stage.get("title"),
            description=generated_stage.get("description"),
            objectives=generated_stage.get("objectives", []),
            difficulty=generated_stage.get("difficulty"),
            recommended_daily_hours=generated_stage.get("recommended_daily_hours"),
            suggested_study_days=generated_stage.get("suggested_study_days"),
            learning_style_notes=generated_stage.get("learning_style_notes"),
            hardware_notes=generated_stage.get("hardware_notes"),
            personalization_summary=generated_stage.get("personalization_summary"),
        )
        _agent_log(
            "Agent1",
            f"stage_planning:done | block={stage.block_name} | block_id={stage.block_id} | subtopic={stage.subtopic_id or '-'} | hours={stage.estimated_hours} | days={stage.suggested_study_days}",
        )
        return StagePlanningOutput(stage=stage)

    def _parse_structured_input(self, text: str) -> dict:
        goal = self._labeled_value(text, ("我的学习目标", "学习目标"))
        return {
            "grade": self._labeled_value(text, ("我的身份背景", "身份背景", "学习背景")),
            "skill_tags": self._parse_tags(self._labeled_value(text, ("我的已有基础", "已有基础", "知识基础"))),
            "goal": goal,
            "daily_hours": self._extract_daily_hours(self._labeled_value(text, ("我每天可以学习", "每天可以学习", "每日学习时长")) or text),
            "learning_style": self._extract_learning_style(self._labeled_value(text, ("我的学习方式", "学习方式")) or text),
            "hardware": self._labeled_value(text, ("我的设备条件", "设备条件")),
        }

    @staticmethod
    def _labeled_value(text: str, labels: tuple[str, ...]) -> str:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[：:]\s*(.+)"
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" \t，。,.；;")
        return ""

    @staticmethod
    def _parse_tags(raw: str) -> list[str]:
        if not raw:
            return []
        parts = re.split(r"[、,，/|；;\s]+", raw)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _extract_daily_hours(text: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:小时|h|H)", text)
        if not match:
            match = re.search(r"(?:每天|每日).{0,8}?([0-9]+(?:\.[0-9]+)?)", text)
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
            "hardware": fallback.profile.hardware,
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
        normalized_gaps = await self._insert_unmet_direct_prerequisite_gaps(
            normalized_gaps,
            fallback.profile.skill_tags,
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
        if structured.get("hardware"):
            score += 0.05
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
    def __init__(self):
        self.vector_search_tool = VectorSearchTool()
        self.question_extract_tool = QuestionExtractTool()
        self.faq_search_tool = FAQSearchTool()
        self.video_search_tool = VideoSearchTool()
        self.learning_resource_tool = LearningResourceTool()

    async def run(self, request: ContentGenerationInput) -> ContentGenerationOutput:
        _agent_log("Agent3", f"content:start | block={request.stage.block_name}")
        stage = request.stage
        related_docs = []
        try:
            related_docs = await self.vector_search_tool.search_by_block(stage.block_id)
        except Exception:
            related_docs = []
        recommended_videos = await self.learning_resource_tool.get_recommended_videos(
            stage.block_id, stage.subtopic_id
        )
        tutorial = (
            f"# {stage.block_name}\n\n"
            f"本阶段目标是理解 {stage.block_name} 的核心概念，并完成一次基础练习。\n\n"
            "## 学习步骤\n\n"
            "1. 阅读核心概念。\n"
            "2. 用自己的语言复述。\n"
            "3. 完成阶段测试题。\n"
        )
        if related_docs:
            excerpts = "\n".join(f"- {item.get('content', '')[:120]}" for item in related_docs[:3])
            tutorial += f"\n\n## 相关知识片段\n\n{excerpts}\n"

        extracted_questions = []
        try:
            extracted_questions = await self.question_extract_tool.extract_questions(
                tutorial,
                {"block_id": stage.block_id, "block_name": stage.block_name, "question_count": 2},
            )
        except Exception:
            extracted_questions = []

        questions = [TestQuestion(
            question_id=f"Q-{stage.block_id.replace('KB-', '')}-01",
            block_id=stage.block_id,
            question_type="填空",
            question_text="本阶段的学习主题是什么？",
            correct_answer=stage.block_name,
            hint="答案就是当前阶段名称。",
        )]
        questions.extend(
            TestQuestion.model_validate(question)
            for question in extracted_questions
            if isinstance(question, dict)
        )

        sources = []
        if related_docs:
            for i, doc in enumerate(related_docs[:3]):
                doc_id = doc.get("doc_id") or doc.get("id") or f"doc_{i}"
                sources.append(Source(
                    source_id=f"src_{doc_id}",
                    source_type="knowledge_doc",
                    title=doc.get("title", ""),
                    locator=doc.get("chunk_id") or doc_id,
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

        claims = [
            Claim(
                claim_id="clm_001",
                text=f"{stage.block_name}的核心概念和学习步骤",
                source_ref=technical_source.source_id if technical_source else None,
                section_ref="tutorial:sec_0",
                claim_type="concept",
                is_key_technical=technical_source is not None,
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
        _agent_log("Agent3", f"content:done | docs={len(related_docs)} | videos={len(recommended_videos)} | questions={len(package.questions)} | tutorial_len={len(package.tutorial)}")
        return ContentGenerationOutput(learning_package=package)

    async def generate_answer(self, request: AnswerGenerationInput) -> AnswerGenerationOutput:
        _agent_log("Agent3", f"answer:start | question={request.question[:60]}")
        faq_results = []
        try:
            faq_results = await self.faq_search_tool.search(
                request.question,
                block_id=request.stage.block_id,
                top_k=1,
            )
        except Exception:
            faq_results = []
        if faq_results:
            answer = faq_results[0].get("answer") or faq_results[0].get("question") or ""
            if answer:
                _agent_log("Agent3", f"answer:faq_hit | similarity={faq_results[0].get('similarity')}")
                return AnswerGenerationOutput(answer=answer)

        related_docs = []
        try:
            related_docs = await self.vector_search_tool.search_by_query(request.question, top_k=3)
        except Exception:
            related_docs = []
        if related_docs:
            references = "\n".join(f"- {item.get('content', '')[:160]}" for item in related_docs)
            answer = (
                f"你当前学习的是 {request.stage.block_name}。针对“{request.question}”，"
                f"可以先参考这些知识片段：\n{references}"
            )
        else:
            answer = (
                f"你当前学习的是 {request.stage.block_name}。针对“{request.question}”，"
                "建议先回到本阶段核心概念，再结合练习验证理解。"
            )
        return AnswerGenerationOutput(answer=answer)

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


class HybridContentAgent(DeterministicContentAgent):
    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()
        self.content_skill_context = get_skill_context(["content_generation", "code_style"])
        self.answer_skill_context = get_skill_context(["question_answering"])

    async def run(self, request: ContentGenerationInput) -> ContentGenerationOutput:
        output = await super().run(request)
        _agent_log("Agent3", f"content:llm_enhance | block={request.stage.block_name}")
        stage = request.stage
        prompt = f"""请为学习阶段“{stage.block_name}”生成一份Markdown教程。

要求：
1. 解释核心概念和适用场景；
2. 给出清晰的学习步骤；
3. 提供一个简短示例；
4. 面向初学者，避免没有解释的术语。

内容生成规范：
{self.content_skill_context}
"""
        tutorial = await self.llm_client.complete(
            prompt,
            system_prompt="你是严谨、清晰的个性化学习助手。",
            temperature=0.4,
        )
        if self._is_valid_response(tutorial):
            output.learning_package.tutorial = tutorial
            _agent_log("Agent3", f"content:llm_done | tutorial_len={len(tutorial)}")
        return output

    async def generate_answer(self, request: AnswerGenerationInput) -> AnswerGenerationOutput:
        _agent_log("Agent3", f"answer:hybrid_start | question={request.question[:60]}")
        faq_results = []
        try:
            faq_results = await self.faq_search_tool.search(
                request.question,
                block_id=request.stage.block_id,
                top_k=1,
            )
        except Exception:
            faq_results = []
        if faq_results:
            answer = faq_results[0].get("answer") or faq_results[0].get("question") or ""
            if answer:
                _agent_log("Agent3", f"answer:faq_hit | similarity={faq_results[0].get('similarity')}")
                return AnswerGenerationOutput(answer=answer)

        related_docs = []
        try:
            related_docs = await self.vector_search_tool.search_by_query(request.question, top_k=3)
        except Exception:
            related_docs = []
        context = ""
        if related_docs:
            context = "\n\n相关知识片段：\n" + "\n".join(
                f"- {item.get('content', '')[:200]}" for item in related_docs
            )
        prompt = f"""当前学习阶段：{request.stage.block_name}
用户学习目标：{request.profile.goal}
用户问题：{request.question}

请直接回答问题。先给出简明结论，再解释原理；需要时提供示例。不要编造来源。{context}"""
        prompt += f"\n\n答疑规范：\n{self.answer_skill_context}"
        answer = await self.llm_client.complete(
            prompt,
            system_prompt="你是技术准确、适合教学的学习助手。",
            temperature=0.5,
        )
        if self._is_valid_response(answer):
            _agent_log("Agent3", f"answer:llm_done | answer_len={len(answer)}")
            return AnswerGenerationOutput(answer=answer)
        _agent_log("Agent3", "answer:llm_invalid -> deterministic_fallback")
        return await super().generate_answer(request)

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
        )
        if not self._is_valid_response(tutorial):
            return await super().revise_learning_package(request)
        package = request.learning_package.model_copy(deep=True)
        package.tutorial = tutorial
        package.version += 1
        return ContentGenerationOutput(learning_package=package)

    async def revise_answer(self, request: AnswerRevisionInput) -> AnswerGenerationOutput:
        feedback = self._format_feedback(request.technical_errors, request.quality_errors)
        prompt = f"""请根据审核意见修改回答。

用户问题：{request.question}
当前阶段：{request.stage.block_name}

原回答：
{request.answer}

审核意见：
{feedback}

要求：逐条修复问题，保持技术准确和教学清晰，返回完整修改后回答。"""
        answer = await self.llm_client.complete(
            prompt,
            system_prompt="你是Agent3答疑专家，负责依据合并审核意见修改回答。",
            temperature=0.3,
        )
        if not self._is_valid_response(answer):
            return await super().revise_answer(request)
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
        try:
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
                    request.stage.subtopic_id,
                )
                errors.extend(normalize_review_errors(video_resource_errors))
            except Exception:
                video_resource_errors = []
        # 步骤5：审核辩论协议字段补全（对接需求清单 2.1，委托 Agent4 实现）
        errors = self._protocol_agent._enrich_debate_fields(
            errors,
            sources=request.sources,
            claims=request.claims,
            round_index=request.round_index,
            raw_errors=video_resource_errors,
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
        try:
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
                    request.stage.subtopic_id,
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
            raw_errors=video_resource_errors,
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
        #  continue/repeat 的推进决策仍由下方 accuracy>=0.8 分支负责，不依赖 gap 状态覆盖）
        all_mastered = all(gap.status == "已掌握" for gap in profile.knowledge_gaps)
        if strategy == "finish" or all_mastered or should_terminate:
            # 尊重策略工具的终止裁决；工具 decision 缺省时兜底为 completed
            decision = tool_decision or "completed"
        elif strategy == "backtrack":
            decision = "repeat"
        elif request.accuracy >= 0.8:
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
