from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from models.material import TestQuestion


class StageInfo(BaseModel):
    stage_id: str
    stage_num: int
    block_id: str
    block_name: str
    subtopic_id: Optional[str] = None
    subtopic_name: Optional[str] = None
    estimated_hours: float = 1.0
    gap_skill: str
    prerequisites: List[str] = Field(default_factory=list)
    status: str = "pending"
    accuracy: Optional[float] = None
    strategy_used: Optional[str] = None


class Source(BaseModel):
    source_id: str
    source_type: Literal["knowledge_doc", "faq", "best_practice", "manual", "external_note", "code_example"]
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
