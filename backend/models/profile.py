from pydantic import BaseModel
from typing import List, Dict, Optional

class KnowledgeGap(BaseModel):
    skill: str
    status: str = "未开始"
    progress: float = 0.0
    prerequisites: List[str] = []

class UserProfile(BaseModel):
    session_id: str
    skill_tags: List[str] = []
    knowledge_gaps: List[KnowledgeGap] = []
    goal: str = ""
    daily_hours: float = 2.0
    learning_style: str = "mixed"
    learning_rhythm: str = "medium"
    confidence_index: float = 0.5
    confusion_markers: List[str] = []
    confidence: float = 0.7
    version: int = 1
