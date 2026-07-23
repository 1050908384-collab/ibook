# -*- coding: utf-8 -*-
from typing import List, Optional

from pydantic import BaseModel, Field

class KnowledgeBlock(BaseModel):
    block_id: str
    block_name: str
    description: Optional[str] = None
    estimated_hours: float = 1.0
    difficulty: str = "初级"
    prerequisites: List[str] = Field(default_factory=list)
    corresponding_gaps: List[str] = Field(default_factory=list)

class KnowledgeDoc(BaseModel):
    doc_id: str
    block_id: str
    title: str
    content: str
    doc_type: str = "tutorial"
    source: Optional[str] = None
    difficulty: str = "初级"

class TestQuestion(BaseModel):
    question_id: str
    block_id: str
    difficulty: str = "初级"
    question_type: str = "选择"
    question_text: str
    options: Optional[List[str]] = None
    correct_answer: str
    hint: Optional[str] = None
