from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, computed_field

class ReviewError(BaseModel):
    error_type: str
    severity: Literal["critical", "major", "minor"] = "major"
    location: Optional[str] = None
    description: str
    evidence: Optional[str] = None
    suggestion: Optional[str] = None

class ReviewResult(BaseModel):
    reviewer: Literal["technical", "quality"]
    passed: bool
    errors: List[ReviewError] = Field(default_factory=list)
    review_details: Optional[Any] = None


class TechnicalReviewResult(ReviewResult):
    reviewer: Literal["technical"] = "technical"


class QualityReviewResult(ReviewResult):
    reviewer: Literal["quality"] = "quality"
    scores: Optional[dict] = None


class ReviewRound(BaseModel):
    iteration: int
    technical: TechnicalReviewResult
    quality: QualityReviewResult

    @computed_field
    @property
    def passed(self) -> bool:
        return self.technical.passed and self.quality.passed
