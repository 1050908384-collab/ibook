from typing import Dict, Any, List, Optional
from llm.client import LLMClient
from skills import load_skills

class BaseAgent:
    def __init__(self, agent_name: str, skill_names: List[str] = None):
        self.agent_name = agent_name
        self.llm_client = LLMClient()
        self.skills = load_skills(skill_names or [])
        self.tools = []

    def load_tools(self, tools: List[Any]) -> None:
        self.tools.extend(tools)

    async def call_llm(self, prompt: str, system_prompt: str = "", **kwargs) -> str:
        return await self.llm_client.complete(prompt, system_prompt, **kwargs)

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        for tool in self.tools:
            if hasattr(tool, tool_name):
                return getattr(tool, tool_name)(**kwargs)
        raise ValueError(f"Tool {tool_name} not found")

    def build_prompt(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)

    def get_skill_context(self) -> str:
        return "\n".join([skill.content for skill in self.skills])
