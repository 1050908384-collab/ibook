# -*- coding: utf-8 -*-
import re
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
    foundation_confirmed: bool = False
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)
    goal: str = ""
    daily_hours: float = 2.0
    learning_style: str = "mixed"
    learning_rhythm: str = "medium"
    confidence_index: float = 0.5
    confusion_markers: List[str] = Field(default_factory=list)
    confidence: float = 0.7
    version: int = 1


class PracticeProfile(BaseModel):
    """Execution context supplied by orchestration to Agent3 only."""
    hardware: str = ""

    @classmethod
    def from_user_input(cls, text: str) -> "PracticeProfile":
        labeled_match = re.search(
            r"(?:我的\s*)?(?:设备条件|硬件条件|显卡|hardware|device)\s*(?:是|为|:|：)\s*([^\n，。；;]+)",
            text,
            re.IGNORECASE,
        )
        if labeled_match:
            return cls(hardware=labeled_match.group(1).strip())

        # Initial free-text diagnostics commonly describe a GPU without a
        # label, for example "I have an RTX 4090 with 24GB VRAM".
        gpu_match = re.search(
            r"\b(?:nvidia\s+)?(?:rtx|gtx)\s*\d{3,4}(?:\s+with\s+\d+\s*gb(?:\s+vram)?)?\b",
            text,
            re.IGNORECASE,
        )
        return cls(hardware=gpu_match.group(0).strip() if gpu_match else "")
