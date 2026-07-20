from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

from models.profile import UserProfile
from models.review import ReviewError, ReviewResult
from models.stage import LearningPackage, StageInfo


class DiagnosticInput(BaseModel):
    session_id: str
    user_input: str


class DiagnosticOutput(BaseModel):
    profile: UserProfile
    followup_questions: List[str] = Field(default_factory=list)


class StagePlanningInput(BaseModel):
    session_id: str
    profile: UserProfile
    stage_num: int


class StagePlanningOutput(BaseModel):
    stage: StageInfo


class ContentGenerationInput(BaseModel):
    session_id: str
    stage: StageInfo


class ContentGenerationOutput(BaseModel):
    learning_package: LearningPackage


class ReviewInput(BaseModel):
    session_id: str
    target_type: Literal["learning_package", "answer"]
    content: str
    stage: StageInfo
    profile: UserProfile
    question: Optional[str] = None
    code_examples: List[dict] = Field(default_factory=list)


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


class AnswerGenerationOutput(BaseModel):
    answer: str


class AnswerRevisionInput(BaseModel):
    session_id: str
    question: str
    answer: str
    stage: StageInfo
    profile: UserProfile
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


class StageDecisionOutput(BaseModel):
    decision: Literal["continue", "repeat", "completed"]
    profile: UserProfile
    strategy: str
