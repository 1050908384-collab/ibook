from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class ReviewResult(BaseModel):
    review_id: str
    passed: bool
    score: float = 0.0
    issues: List[str] = []
    suggestions: List[str] = []
    hallucination_evidence: List[str] = []
    metadata: Dict[str, Any] = {}


class QualityMetrics(BaseModel):
    pedagogical_score: float = 0.0
    cognitive_load: float = 0.0
    engagement_score: float = 0.0
    environment_fit: float = 0.0
    personalization_score: float = 0.0