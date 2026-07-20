from typing import List, Literal, Optional

from pydantic import BaseModel, Field

class ReviewError(BaseModel):
    error_type: str
    location: Optional[str] = None
    description: str
    suggestion: Optional[str] = None

class ReviewResult(BaseModel):
    reviewer: Literal["technical", "quality"]
    passed: bool
    errors: List[ReviewError] = Field(default_factory=list)
    review_details: Optional[str] = None


class TechnicalReviewResult(ReviewResult):
    reviewer: Literal["technical"] = "technical"


class QualityReviewResult(ReviewResult):
    reviewer: Literal["quality"] = "quality"
    scores: Optional[dict] = None
