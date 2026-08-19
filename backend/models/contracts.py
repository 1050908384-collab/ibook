from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from models.profile import PracticeProfile, UserProfile
from models.review import ReviewError, ReviewResult
from models.stage import ContentBlock, LearningPackage, StageInfo


class DiagnosticInput(BaseModel):
    session_id: str
    user_input: str


class DiagnosticOutput(BaseModel):
    profile: UserProfile
    followup_questions: List[str] = Field(default_factory=list)


# === 分步诊断：目标 → 知识板块 → 知识点 → 确定 ===

class GoalBlocksRequest(BaseModel):
    """按学习目标返回其下所有知识板块。"""
    goal: str


class GoalBlockItem(BaseModel):
    """知识板块选项（含知识点列表供前端展开）。"""
    block_id: str
    block_name: str
    description: str = ""
    difficulty: str = ""
    estimated_hours: float = 0
    subtopics: List[dict] = Field(default_factory=list)


class GoalBlocksResponse(BaseModel):
    goal: str
    blocks: List[GoalBlockItem] = Field(default_factory=list)


class FoundationSubmitRequest(BaseModel):
    """用户确认已掌握知识点后提交，生成画像并进入阶段规划。"""
    session_id: str
    goal: str
    mastered_subtopic_ids: List[str] = Field(default_factory=list)
    daily_hours: float = 2.0
    learning_style: str = "mixed"


class FoundationSubmitResponse(BaseModel):
    status: str  # "profile_ready" | "needs_more_info"
    profile: Optional[dict] = None
    stage: Optional[dict] = None
    followup_questions: List[str] = Field(default_factory=list)


class StagePlanningInput(BaseModel):
    session_id: str
    profile: UserProfile
    stage_num: int
    adjustment_notes: Optional[str] = None
    excluded_subtopic_ids: List[str] = Field(default_factory=list)


class StagePlanningOutput(BaseModel):
    stage: StageInfo


class LearningEvidence(BaseModel):
    """Learner facts that must cause observable teaching decisions."""

    completed_stages: List[dict] = Field(default_factory=list)
    mastered_subtopic_ids: List[str] = Field(default_factory=list)
    unstable_prerequisites: List[dict] = Field(default_factory=list)
    weak_topics: List[dict] = Field(default_factory=list)
    recent_questions: List[str] = Field(default_factory=list)
    completed_topic_names: List[str] = Field(default_factory=list)
    preflight_review: Optional[dict] = None


class ContentGenerationInput(BaseModel):
    session_id: str
    stage: StageInfo
    learning_profile: UserProfile | None = None
    practice_profile: PracticeProfile = Field(default_factory=PracticeProfile)
    learning_evidence: LearningEvidence = Field(default_factory=LearningEvidence)


class ContentGenerationOutput(BaseModel):
    learning_package: LearningPackage
    blocks: List[ContentBlock] = Field(default_factory=list)  # 块级流水线产出（旧整篇路径留空）


class ReviewInput(BaseModel):
    session_id: str
    target_type: Literal["learning_package", "answer"]
    content: str
    stage: StageInfo
    profile: UserProfile
    question: Optional[str] = None
    code_examples: List[dict] = Field(default_factory=list)
    # === Agent3/4 审核辩论协议扩展（对接需求清单 1.1）===
    trace_id: Optional[str] = None                       # 本次审核生命周期 ID（review_traces.trace_id）
    sources: List[dict] = Field(default_factory=list)    # Agent3 产出的来源清单（协议 2.1）
    claims: List[dict] = Field(default_factory=list)     # Agent3 产出的可验证陈述（协议 2.2）
    recommended_videos: List[dict] = Field(default_factory=list)  # Agent3 按当前子知识点推荐的视频资源
    round_index: int = 1                                 # 当前审核轮次，用于 error_id 后缀 _r{round_index}
    # === 块级审核扩展（块级并行流水线）：整篇路径留空，块级路径由 review_block 填充 ===
    content_block_id: Optional[str] = None
    content_version: Optional[int] = None
    source_chunk_id: Optional[str] = None
    heading_path: Optional[str] = None
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None


class BlockRevisionInput(BaseModel):
    """Agent3 块级定点修改输入：一个待修订块 + 该块相关的 issues。"""
    session_id: str
    block: ContentBlock
    issues: List[dict] = Field(default_factory=list)


class RebutInput(BaseModel):
    """Agent4 复核输入：Agent4 上一轮 issues + Agent3 逐条回复（对接需求清单 1.2）。"""
    session_id: str
    trace_id: str
    issues: List[dict]           # Agent4 上一轮 issues（canonical_error_id/must_fix/severity...）
    author_replies: List[dict]   # Agent3 逐条回复（canonical_error_id/disposition/reason/patched_sections/source_refs）


class RebutOutput(BaseModel):
    """Agent4 复核输出：每个 canonical_error_id 的裁决（对接需求清单 1.2）。"""
    verdicts: Dict[str, str]     # {canonical_error_id: "persist" | "withdraw" | "downgrade"}


class ContentRevisionInput(BaseModel):
    session_id: str
    learning_package: LearningPackage
    stage: StageInfo
    profile: UserProfile
    technical_errors: List[ReviewError] = Field(default_factory=list)
    quality_errors: List[ReviewError] = Field(default_factory=list)


class AnswerGenerationInput(BaseModel):
    session_id: str
    question: str
    stage: StageInfo
    profile: UserProfile
    practice_profile: PracticeProfile = Field(default_factory=PracticeProfile)
    block_id: Optional[str] = None
    qa_history: List[dict] = Field(default_factory=list)
    profile_summary: Optional[str] = None


class AnswerGenerationOutput(BaseModel):
    answer: str
    # 问题类型标记：technical（技术类）| daily（日常类）
    # harness 据此决定是否调用 Agent4 审核
    question_type: Literal["technical", "daily"] = "technical"


class AnswerRevisionInput(BaseModel):
    session_id: str
    question: str
    answer: str
    stage: StageInfo
    profile: UserProfile
    practice_profile: PracticeProfile = Field(default_factory=PracticeProfile)
    technical_errors: List[ReviewError] = Field(default_factory=list)
    quality_errors: List[ReviewError] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    technical: ReviewResult
    quality: ReviewResult

    @computed_field
    @property
    def passed(self) -> bool:
        return self.technical.passed and self.quality.passed


class StageDecisionInput(BaseModel):
    session_id: str
    profile: UserProfile
    stage: StageInfo
    accuracy: float
    qa_history: List[dict] = Field(default_factory=list)
    interactions: List[dict] = Field(default_factory=list)
    # 已提交测试逐题记录（含 time_spent），优先于 qa_history 作为 BKT/总结的答题记录来源
    assessment_records: List[dict] = Field(default_factory=list)
    # 已提交测试聚合实际用时（小时），>0 时优先于 interactions 估算的耗时
    time_spent_hours: float = 0.0


class StageDecisionOutput(BaseModel):
    decision: Literal["continue", "repeat", "completed"]
    profile: UserProfile
    strategy: str
    stage_summary: Optional[str] = None
