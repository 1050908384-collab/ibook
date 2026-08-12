from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from models.material import TestQuestion


class StageInfo(BaseModel):
    stage_id: str
    stage_num: int
    block_id: str
    block_name: str
    # 一个阶段可覆盖多个知识板块；block_id 保留为旧接口的主板块。
    block_ids: List[str] = Field(default_factory=list)
    subtopic_id: Optional[str] = None
    subtopic_name: Optional[str] = None
    subtopic_ids: List[str] = Field(default_factory=list)
    subtopic_names: List[str] = Field(default_factory=list)
    estimated_hours: float = 1.0
    gap_skill: str
    prerequisites: List[str] = Field(default_factory=list)
    status: str = "pending"
    accuracy: Optional[float] = None
    strategy_used: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    difficulty: Optional[str] = None
    recommended_daily_hours: Optional[float] = None
    suggested_study_days: Optional[int] = None
    learning_style_notes: Optional[str] = None
    hardware_notes: Optional[str] = None
    personalization_summary: Optional[str] = None


class Source(BaseModel):
    source_id: str
    source_type: Literal["knowledge_doc", "best_practice", "manual", "external_note", "code_example", "video"]
    title: Optional[str] = None
    locator: Optional[str] = None
    content_ref: Optional[str] = None
    source_version: Optional[str] = None
    metadata_json: Optional[dict] = None


class Claim(BaseModel):
    claim_id: str
    text: str
    source_ref: Optional[str] = None
    source_span: Optional[str] = None
    section_ref: str
    claim_type: Literal["fact", "procedure", "code", "config", "concept"]
    is_key_technical: bool = False
    confidence: float = 0.5
    status: Literal["active", "revised", "dropped"] = "active"
    metadata_json: Optional[dict] = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_source_ref(self) -> "Claim":
        if self.is_key_technical and not self.source_ref:
            raise ValueError("source_ref cannot be null when is_key_technical=True")
        return self


class ContentBlock(BaseModel):
    """块级个性化内容单元（对应一个原始 knowledge_chunk）。

    Agent3 逐块生成、Agent4 逐块审核的最小交换单位。所有来源溯源字段可选，
    数据侧新字段（heading_path 等）缺失时优雅降级，不影响块级流水线运行。
    """
    content_block_id: str                                   # 个性化内容块唯一 ID
    session_id: str
    stage_id: str
    block_id: str                                           # 所属知识板块（KB-xx）
    subtopic_id: Optional[str] = None                       # 所属子知识点
    block_name: str
    block_index: int                                        # 块在阶段内的顺序（用于聚合拼装）
    # 来源溯源（回指原始 knowledge_chunk；字段缺失即降级）
    source_chunk_id: Optional[str] = None
    source_doc_id: Optional[str] = None
    heading_path: Optional[str] = None
    heading_level: Optional[int] = None
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None
    source_note: str = ""                                   # 人类可读来源说明
    personalized_content: str = ""                          # 个性化后的正文
    claims: List[Claim] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    version: int = 1
    revision_note: Optional[str] = None                     # 修改说明（modify_block 回传）
    status: Literal["draft", "reviewing", "revised", "approved", "needs_human_review"] = "draft"


class LearningPackage(BaseModel):
    session_id: str
    stage_id: str
    block_id: str
    block_name: str
    tutorial: str = ""
    code_examples: List[dict] = Field(default_factory=list)
    questions: List[TestQuestion] = Field(default_factory=list)
    format: str = "markdown"
    version: int = 1
    review_trace_id: Optional[str] = None
    sources: List[Source] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    recommended_videos: List[dict] = Field(default_factory=list)
