# -*- coding: utf-8 -*-
from typing import List

from pydantic import BaseModel, Field


class KnowledgeGap(BaseModel):
    skill: str
    block_id: str = ""
    block_name: str = ""
    target_subtopic_ids: List[str] = Field(default_factory=list)
    mapping_confidence: float = 0.0
    level: str = "beginner"
    status: str = "未掌握"
    progress: float = 0.0
    prerequisites: List[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    session_id: str
    user_id: str = "local-user"
    grade: str = ""
    skill_tags: List[str] = Field(default_factory=list)
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)
    goal: str = ""
    daily_hours: float = 2.0
    learning_style: str = "mixed"
    hardware: str = ""
    learning_rhythm: str = "medium"
    confidence_index: float = 0.5
    confusion_markers: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    version: int = 1
