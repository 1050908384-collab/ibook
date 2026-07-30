# -*- coding: utf-8 -*-
# ← 实际运行的 Agent 实现（含确定性版+混合版）
import json
import re
import uuid

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
from models.stage import LearningPackage, StageInfo
from skills import load_skills
from tools.agent1_tools.calculate_confidence_tool import CalculateConfidenceTool
from tools.agent1_tools.extract_skills_tool import ExtractSkillsTool
from tools.agent1_tools.graph_query_tool import GraphQueryTool
from tools.agent1_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent1_tools.stage_generator_tool import StageGeneratorTool
from tools.agent3_tools.faq_search_tool import FAQSearchTool
from tools.agent3_tools.question_extract_tool import QuestionExtractTool
from tools.agent3_tools.vector_search_tool import VectorSearchTool
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
    def __init__(self):
        self.extract_skills_tool = ExtractSkillsTool()
        self.calculate_confidence_tool = CalculateConfidenceTool()
        self.graph_query_tool = GraphQueryTool()
        self.knowledge_block_query_tool = KnowledgeBlockQueryTool()
        self.stage_generator_tool = StageGeneratorTool()
        self.db_client = SQLiteClient()

    async def run(self, request: DiagnosticInput) -> DiagnosticOutput:
        _agent_log("Agent1", f"diagnostic:start | session={request.session_id}")
        text = request.user_input.strip()
        target_text = self._target_text(text)
        targets = [skill for skill in SKILL_NAMES if skill.lower() in target_text.lower()]
        mentioned = [skill for skill in SKILL_NAMES if skill.lower() in text.lower()]
        mastered = [skill for skill in mentioned if skill not in targets]
        followup_questions: list[str] = []
        try:
            extracted_tags = await self.extract_skills_tool.extract_skills(text)
            extracted_gaps = await self.extract_skills_tool.extract_gaps(text)
            if extracted_tags:
                mastered = extracted_tags
            if extracted_gaps:
                targets = [gap.get("skill", "") for gap in extracted_gaps if gap.get("skill")]
        except Exception:
            extracted_gaps = []

        if not targets:
            targets = [mentioned[-1]] if mentioned else ["通用学习目标"]
            mastered = [skill for skill in mentioned if skill not in targets]

        gaps = []
        for skill in targets:
            prerequisites: list[str] = []
            try:
                prerequisites = await self.graph_query_tool.query_prerequisites(skill)
            except Exception:
                prerequisites = []
            gaps.append(KnowledgeGap(skill=skill, prerequisites=prerequisites))

        confidence = 0.9 if mentioned and target_text != text else 0.75
        try:
            confidence_result = await self.calculate_confidence_tool.calculate(
                text,
                mastered,
                [gap.model_dump(mode="json") for gap in gaps],
            )
            confidence = max(confidence, confidence_result)
        except Exception:
            pass
        if confidence < 0.7 and not mentioned:
            followup_questions = [
                "你希望优先解决哪一类问题？",
                "你目前有哪些相关基础？",
                "你每天大约能投入多久学习？",
            ]
        profile = UserProfile(
            session_id=request.session_id,
            skill_tags=mastered,
            knowledge_gaps=gaps,
            goal=target_text or text,
            confidence=confidence,
        )
        _agent_log("Agent1", f"diagnostic:done | goal={profile.goal} | gaps={len(profile.knowledge_gaps)} | confidence={profile.confidence:.2f}")
        return DiagnosticOutput(profile=profile, followup_questions=followup_questions)

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
            (item for item in request.profile.knowledge_gaps if item.status != "已掌握"),
            None,
        )
        if gap is None:
            raise ValueError("No unmastered knowledge gap remains")

        block_info = {}
        try:
            block_info = await self.knowledge_block_query_tool.query_block(gap.skill)
        except Exception:
            block_info = {}
        block_info = block_info or {}

        generated_stage = {}
        try:
            generated_stage = await self.stage_generator_tool.generate_stage(
                request.session_id,
                gap.model_dump(mode="json"),
                block_info,
                request.profile.goal,
                request.stage_num,
            )
        except Exception:
            generated_stage = {}
        generated_stage = generated_stage or {}

        subtopic = None
        try:
            subtopic = self.db_client.get_next_subtopic(
                request.session_id,
                block_info.get("block_id", ""),
                max_hours=request.profile.daily_hours,
            ) if block_info.get("block_id") else None
        except Exception:
            subtopic = None

        stable_key = f"{request.session_id}:{request.stage_num}:{gap.skill}"
        stage_id = generated_stage.get("stage_id") or str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        stage_name = gap.skill or block_info.get("block_name") or generated_stage.get("block_name")
        stage_hours = (
            subtopic.get("estimated_hours")
            if subtopic and subtopic.get("estimated_hours") is not None else
            block_info.get("estimated_hours") or generated_stage.get("estimated_hours") or 2.0
        )
        stage = StageInfo(
            stage_id=stage_id,
            stage_num=request.stage_num,
            block_id=block_info.get("block_id") or generated_stage.get("block_id") or f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, gap.skill).hex[:8].upper()}",
            block_name=stage_name,
            subtopic_id=subtopic.get("subtopic_id") if subtopic else None,
            gap_skill=gap.skill,
            estimated_hours=stage_hours,
            prerequisites=(
                subtopic.get("prerequisites")
                if subtopic and subtopic.get("prerequisites") is not None else
                block_info.get("prerequisites") or generated_stage.get("prerequisites") or gap.prerequisites
            ),
        )
        _agent_log("Agent1", f"stage_planning:done | block={stage.block_name} | block_id={stage.block_id} | subtopic={stage.subtopic_id or '-'}")
        return StagePlanningOutput(stage=stage)


class HybridDiagnosticAgent(DeterministicDiagnosticAgent):
    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()

    async def run(self, request: DiagnosticInput) -> DiagnosticOutput:
        fallback = await super().run(request)
        prompt = f"""请分析下面的学习者描述，并只返回一个JSON对象，不要添加说明文字。

学习者描述：{request.user_input}

JSON字段要求：
- skill_tags: 已掌握技能字符串数组
- knowledge_gaps: 待学习知识数组，每项包含 skill 和 level
- goal: 学习目标
- daily_hours: 每日学习小时数，未知时为2
- learning_style: theory、practice 或 mixed

示例：
{{"skill_tags":["Python"],"knowledge_gaps":[{{"skill":"LoRA","level":"beginner"}}],"goal":"掌握LoRA微调","daily_hours":2,"learning_style":"mixed"}}
"""
        response = await self.llm_client.complete(
            prompt,
            system_prompt="你是学习者画像诊断专家，必须输出严格JSON。",
            temperature=0.2,
            json_mode=True,
        )
        data = self._parse_json(response)
        if not data:
            return fallback

        try:
            gaps = data.get("knowledge_gaps", [])
            normalized_gaps = [
                {"skill": gap, "level": "beginner"} if isinstance(gap, str) else gap
                for gap in gaps
            ]
            if not normalized_gaps:
                return fallback
            profile = UserProfile(
                session_id=request.session_id,
                skill_tags=data.get("skill_tags", []),
                knowledge_gaps=normalized_gaps,
                goal=data.get("goal") or fallback.profile.goal,
                daily_hours=data.get("daily_hours", 2.0),
                learning_style=data.get("learning_style", "mixed"),
                confidence=0.9,
            )
            return DiagnosticOutput(profile=profile)
        except (TypeError, ValueError):
            return fallback

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

    async def run(self, request: ContentGenerationInput) -> ContentGenerationOutput:
        _agent_log("Agent3", f"content:start | block={request.stage.block_name}")
        stage = request.stage
        related_docs = []
        try:
            related_docs = await self.vector_search_tool.search_by_block(stage.block_id)
        except Exception:
            related_docs = []
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
        package = LearningPackage(
            session_id=request.session_id,
            stage_id=stage.stage_id,
            block_id=stage.block_id,
            block_name=stage.block_name,
            tutorial=tutorial,
            questions=questions,
        )
        _agent_log("Agent3", f"content:done | docs={len(related_docs)} | questions={len(package.questions)} | tutorial_len={len(package.tutorial)}")
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

    async def run(self, request: ReviewInput) -> TechnicalReviewResult:
        errors = []
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

        # 步骤5：综合裁决（有critical错误或errors非空则不通过）
        return TechnicalReviewResult(passed=not errors, errors=errors)


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
        self.skill_context = get_skill_context([
            "technical_review",
            "industry_standards",
            "pedagogy",
            "learner_model",
            "quality_metrics",
        ])

    async def run(self, request: ReviewInput) -> TechnicalReviewResult:
        print(f"=== Agent4(合并后) 技术+质量审核开始：{request.target_type} ===")
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

        quality_tool_errors_normalized = normalize_review_errors(quality_tool_errors)

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
        print(f"=== Agent4(合并后) 审核完成：passed={result.passed}, errors={len(result.errors)} ===")
        return result

    def _build_prompt(self, request: ReviewInput) -> str:
        domain_reference = self._domain_reference(request.stage.block_name)
        return f"""请审核以下{request.target_type}的技术正确性。

学习阶段：{request.stage.block_name}
用户问题：{request.question or '无'}
待审核内容：
{request.content[:12000]}

审核规范：
{self.skill_context}

强制审核步骤：
1. 独立提取并核验每个技术声明，不得沿用原文结论；
2. 重新计算所有参数量、比例和数值示例；
3. 区分“分解原权重”和“增加可训练低秩更新”等不同机制；
4. 任何无法确认的关键声明不得直接判定通过。

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

        # 步骤2：精算掌握度（逐知识点BKT更新）— 异常兜底
        try:
            updated_gaps = await self.profile_update_tool.update_gaps(
                [gap.model_dump(mode="json") for gap in profile.knowledge_gaps],
                request.accuracy,
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
            strategy = await self.strategy_selector_tool.select_strategy(
                request.accuracy,
                profile.model_dump(mode="json"),
                iteration_history=iteration_history,
            )
        except Exception:
            strategy = "reinforce" if request.accuracy >= 0.6 else "simplify"

        # 决策判定
        # strategy 为 backtrack（回溯）时，decision 映射为 repeat（回退重学前置知识）
        # strategy 为 finish（结业）时，decision 映射为 completed
        if current_gap is not None and request.accuracy >= 0.8:
            current_gap.status = "已掌握"
            current_gap.progress = 1.0

        all_mastered = all(gap.status == "已掌握" for gap in profile.knowledge_gaps)
        if strategy == "finish" or all_mastered:
            decision = "completed"
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
        time_spent_hours = round(total_seconds / 3600.0, 2)
        expected_time_hours = round(expected_seconds / 3600.0, 2)

        # 生成qa历史转化为question_records
        question_records: list[dict] = []
        for qa in request.qa_history:
            if isinstance(qa, dict) and ("question_id" in qa or "skill" in qa or "is_correct" in qa):
                question_records.append(qa)

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
