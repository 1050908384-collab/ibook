from typing import List, Optional

from pydantic import BaseModel, Field

from models.material import TestQuestion

class StageInfo(BaseModel):
    stage_id: str
    stage_num: int
    block_id: str
    block_name: str
    estimated_hours: float = 1.0
    gap_skill: str
    prerequisites: List[str] = Field(default_factory=list)
    status: str = "pending"
    accuracy: Optional[float] = None
    strategy_used: Optional[str] = None

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
