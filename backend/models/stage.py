from pydantic import BaseModel
from typing import List, Dict, Optional

class StageInfo(BaseModel):
    stage_id: str
    stage_num: int
    block_id: str
    block_name: str
    estimated_hours: float = 1.0
    prerequisites: List[str] = []
    status: str = "pending"
    accuracy: Optional[float] = None
    strategy_used: Optional[str] = None

class LearningPackage(BaseModel):
    session_id: str
    stage_id: str
    block_id: str
    block_name: str
    tutorial: str = ""
    code_examples: List[Dict] = []
    questions: List[Dict] = []
    version: int = 1
