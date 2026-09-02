# -*- coding: utf-8 -*-
from typing import Any, List, Optional

from pydantic import BaseModel, Field

class KnowledgeBlock(BaseModel):
    block_id: str
    block_name: str
    description: Optional[str] = None
    estimated_hours: float = 1.0
    difficulty: str = "??"
    prerequisites: List[str] = Field(default_factory=list)
    corresponding_gaps: List[str] = Field(default_factory=list)

class KnowledgeDoc(BaseModel):
    doc_id: str
    block_id: str
    title: str
    content: str
    doc_type: str = "tutorial"
    source: Optional[str] = None
    difficulty: str = "??"

class TestQuestion(BaseModel):
    question_id: str
    block_id: str
    subtopic_id: Optional[str] = None
    difficulty: str = "??"
    question_type: str = "single_choice"
    question_text: str
    options: Optional[List[Any]] = None
    correct_answer: str
    hint: Optional[str] = None
    explanation: Optional[str] = None
    expected_output: Optional[str] = None
    grading_mode: Optional[str] = None
    source: Optional[str] = None
