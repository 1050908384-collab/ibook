from pydantic import BaseModel
from typing import Dict, Optional, Any
from enum import Enum


class AgentRole(str, Enum):
    DIAGNOSTIC = "agent_01_diagnostic"
    RETRIEVAL = "agent_02_retrieval"
    GENERATION = "agent_03_generation"
    TECH_REVIEW = "agent_04_tech_review"
    QUALITY_REVIEW = "agent_05_quality_review"
    ITERATION = "agent_06_iteration"


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    DEBATE = "debate"
    FEEDBACK = "feedback"


class AgentMessage(BaseModel):
    message_id: str
    sender: AgentRole
    receiver: Optional[AgentRole] = None
    message_type: MessageType
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: float = 0.0
    
    class Config:
        use_enum_values = True