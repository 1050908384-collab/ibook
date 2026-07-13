from typing import Tuple
from backend.config import settings


def route_model(agent_id: str) -> Tuple[str, str]:
    model = settings.AGENT_MODEL_ROUTING.get(agent_id, settings.OLLAMA_MODEL)
    provider = settings.LLM_PROVIDER
    
    if provider == "deepseek":
        model = "deepseek-chat"
    
    return model, provider