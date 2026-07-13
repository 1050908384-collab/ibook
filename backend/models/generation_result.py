from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class GenerationResult(BaseModel):
    result_id: str
    content: str
    format: str = "markdown"
    difficulty: str = "medium"
    coverage: float = 0.0
    hallucination_detected: bool = False
    references: List[str] = []
    suggestions: List[str] = []
    metadata: Dict[str, Any] = {}


class ResourceItem(BaseModel):
    title: str
    type: str
    content: str
    estimated_time: str
    prerequisites: List[str] = []