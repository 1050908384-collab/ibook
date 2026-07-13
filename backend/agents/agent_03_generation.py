from backend.agents.base_agent import BaseAgent
from typing import Dict, Any


class GenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_03_generation", "内容生成Agent")
        self.available_tools = [
            "code_generation_tool",
            "template_render_tool",
            "format_converter_tool"
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        knowledge_points = input_data.get("knowledge_points", "")
        learner_profile = input_data.get("learner_profile", {})
        revision_instructions = input_data.get("revision_instructions", "")
        
        prompt = f"""你是一个内容生成专家，负责为学习者生成个性化学习资源。
        
知识点：{knowledge_points}
学习者画像：{learner_profile}
修改指令：{revision_instructions}

请生成完整的学习资源，包括：
1. 前置知识（如果有）
2. 核心概念讲解
3. 实践代码（可运行）
4. 常见误区
5. 自测题

要求：
- 内容必须覆盖所有知识点
- 难度匹配学习者水平
- 语言通俗易懂
"""
        
        response = self.call_llm(prompt)
        
        return {
            "content": response,
            "format": "markdown",
            "difficulty": self._estimate_difficulty(response),
            "coverage": self._calculate_coverage(response, knowledge_points)
        }
    
    def _estimate_difficulty(self, content: str) -> str:
        word_count = len(content.split())
        if word_count < 500:
            return "easy"
        elif word_count < 1500:
            return "medium"
        else:
            return "hard"
    
    def _calculate_coverage(self, content: str, knowledge_points: str) -> float:
        points = knowledge_points.split("、")
        covered = sum(1 for point in points if point.strip() in content)
        return covered / len(points) if points else 0.0