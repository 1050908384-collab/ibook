from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from backend.llm.client import LLMClient
from backend.llm.router import route_model


class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.llm_client = LLMClient()
        self.model, self.provider = route_model(agent_id)
        self.available_tools: List[str] = []
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def call_llm(self, prompt: str, **kwargs) -> str:
        return self.llm_client.generate(
            model=self.model,
            prompt=prompt,
            provider=self.provider,
            **kwargs
        )
    
    def call_llm_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return self.llm_client.chat(
            model=self.model,
            messages=messages,
            provider=self.provider,
            **kwargs
        )
    
    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.available_tools