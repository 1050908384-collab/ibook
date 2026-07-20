from typing import List, Literal

from pydantic import BaseModel, Field

from models.profile import UserProfile
from models.review import ReviewResult
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
    learning_package: LearningPackage


class ReviewOutput(BaseModel):
    technical: ReviewResult
    quality: ReviewResult

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
