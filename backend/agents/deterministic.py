import re
import uuid

from models.contracts import (
    ContentGenerationInput,
    ContentGenerationOutput,
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


class DeterministicDiagnosticAgent:
    async def run(self, request: DiagnosticInput) -> DiagnosticOutput:
        text = request.user_input.strip()
        target_text = self._target_text(text)
        targets = [skill for skill in SKILL_NAMES if skill.lower() in target_text.lower()]
        mentioned = [skill for skill in SKILL_NAMES if skill.lower() in text.lower()]
        mastered = [skill for skill in mentioned if skill not in targets]

        if not targets:
            targets = [mentioned[-1]] if mentioned else ["通用学习目标"]
            mastered = [skill for skill in mentioned if skill not in targets]

        gaps = [KnowledgeGap(skill=skill) for skill in targets]
        confidence = 0.9 if mentioned and target_text != text else 0.75
        profile = UserProfile(
            session_id=request.session_id,
            skill_tags=mastered,
            knowledge_gaps=gaps,
            goal=target_text or text,
            confidence=confidence,
        )
        return DiagnosticOutput(profile=profile)

    def _target_text(self, text: str) -> str:
        match = re.search(
            r"(?:想学习|想学|学习目标(?:是|为)?|目标(?:是|为)?|want to learn|learn)(.+)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip(" ，。,.！!") if match else text


class DeterministicStagePlannerAgent:
    async def run(self, request: StagePlanningInput) -> StagePlanningOutput:
        gap = next(
            (item for item in request.profile.knowledge_gaps if item.status != "已掌握"),
            None,
        )
        if gap is None:
            raise ValueError("No unmastered knowledge gap remains")

        stable_key = f"{request.session_id}:{request.stage_num}:{gap.skill}"
        stage_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        stage = StageInfo(
            stage_id=stage_id,
            stage_num=request.stage_num,
            block_id=f"KB-{uuid.uuid5(uuid.NAMESPACE_DNS, gap.skill).hex[:8].upper()}",
            block_name=gap.skill,
            gap_skill=gap.skill,
            estimated_hours=2.0,
            prerequisites=gap.prerequisites,
        )
        return StagePlanningOutput(stage=stage)


class DeterministicContentAgent:
    async def run(self, request: ContentGenerationInput) -> ContentGenerationOutput:
        stage = request.stage
        tutorial = (
            f"# {stage.block_name}\n\n"
            f"本阶段目标是理解 {stage.block_name} 的核心概念，并完成一次基础练习。\n\n"
            "## 学习步骤\n\n"
            "1. 阅读核心概念。\n"
            "2. 用自己的语言复述。\n"
            "3. 完成阶段测试题。\n"
        )
        question = TestQuestion(
            question_id=f"Q-{stage.stage_id[:8]}-1",
            block_id=stage.block_id,
            question_type="填空",
            question_text="本阶段的学习主题是什么？",
            correct_answer=stage.block_name,
            hint="答案就是阶段标题。",
        )
        package = LearningPackage(
            session_id=request.session_id,
            stage_id=stage.stage_id,
            block_id=stage.block_id,
            block_name=stage.block_name,
            tutorial=tutorial,
            questions=[question],
        )
        return ContentGenerationOutput(learning_package=package)

    async def answer_question(self, question: str, stage: StageInfo) -> str:
        return f"你当前学习的是 {stage.block_name}。针对“{question}”，建议先回到本阶段核心概念，再结合练习验证理解。"


class DeterministicTechnicalReviewAgent:
    async def run(self, request: ReviewInput) -> TechnicalReviewResult:
        errors = []
        package = request.learning_package
        if not package.tutorial.strip():
            errors.append(ReviewError(error_type="missing_tutorial", description="教程内容为空"))
        if not package.questions:
            errors.append(ReviewError(error_type="missing_questions", description="学习包没有测试题"))
        return TechnicalReviewResult(passed=not errors, errors=errors)


class DeterministicQualityReviewAgent:
    async def run(self, request: ReviewInput) -> QualityReviewResult:
        errors = []
        if len(request.learning_package.tutorial.strip()) < 40:
            errors.append(ReviewError(error_type="too_short", description="教程内容不足以支持阶段学习"))
        return QualityReviewResult(
            passed=not errors,
            errors=errors,
            scores={"clarity": 1.0 if not errors else 0.0},
        )


class DeterministicStageDecisionAgent:
    async def run(self, request: StageDecisionInput) -> StageDecisionOutput:
        profile = request.profile.model_copy(deep=True)
        for gap in profile.knowledge_gaps:
            if gap.skill != request.stage.gap_skill:
                continue
            if request.accuracy >= 0.8:
                gap.status = "已掌握"
                gap.progress = 1.0
            else:
                gap.status = "部分掌握"
                gap.progress = max(gap.progress, request.accuracy)

        profile.confidence_index = min(1.0, max(0.0, request.accuracy))
        profile.version += 1
        all_mastered = all(gap.status == "已掌握" for gap in profile.knowledge_gaps)
        if all_mastered:
            decision = "completed"
            strategy = "finish"
        elif request.accuracy >= 0.8:
            decision = "continue"
            strategy = "advance"
        else:
            decision = "repeat"
            strategy = "reinforce"
        return StageDecisionOutput(decision=decision, profile=profile, strategy=strategy)


async def run_deterministic_reviews(request: ReviewInput) -> ReviewOutput:
    technical = await DeterministicTechnicalReviewAgent().run(request)
    quality = await DeterministicQualityReviewAgent().run(request)
    return ReviewOutput(technical=technical, quality=quality)
