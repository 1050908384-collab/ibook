from .base_agent import BaseAgent
from .agent_01_diagnostic import DiagnosticAgent
from .agent_02_retrieval import RetrievalAgent
from .agent_03_generation import GenerationAgent
from .agent_04_tech_review import TechReviewAgent
from .agent_05_quality_review import QualityReviewAgent
from .agent_06_iteration import IterationAgent

AGENTS = {
    "agent_01_diagnostic": DiagnosticAgent,
    "agent_02_retrieval": RetrievalAgent,
    "agent_03_generation": GenerationAgent,
    "agent_04_tech_review": TechReviewAgent,
    "agent_05_quality_review": QualityReviewAgent,
    "agent_06_iteration": IterationAgent,
}

__all__ = [
    "BaseAgent", "DiagnosticAgent", "RetrievalAgent",
    "GenerationAgent", "TechReviewAgent", "QualityReviewAgent",
    "IterationAgent", "AGENTS"
]