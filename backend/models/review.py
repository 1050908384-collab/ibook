from pydantic import BaseModel
from typing import List, Optional

class ReviewError(BaseModel):
    error_type: str
    location: Optional[str] = None
    description: str
    suggestion: Optional[str] = None

class TechnicalReviewResult(BaseModel):
    passed: bool
    errors: List[ReviewError] = []
    review_details: Optional[str] = None

class QualityReviewResult(BaseModel):
    passed: bool
    errors: List[ReviewError] = []
    scores: Optional[dict] = None
    review_details: Optional[str] = None
