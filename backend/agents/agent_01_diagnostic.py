from backend.agents.base_agent import BaseAgent
from typing import Dict, Any
from backend.models.learner_profile import LearnerProfile


class DiagnosticAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_01_diagnostic", "学情诊断Agent")
        self.available_tools = [
            "vector_search_tool",
            "graph_query_tool",
            "question_generation_tool"
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = input_data.get("user_input", "")
        conversation_history = input_data.get("conversation_history", [])
        
        prompt = f"""你是一个学情诊断专家，负责了解学习者的背景和需求。
        
对话历史：{conversation_history}
用户当前输入：{user_input}

请分析并输出学习者画像（JSON格式），包括：
1. 专业背景和编程经验
2. 已掌握的技能（Python/深度学习/分布式等）
3. 知识盲区
4. 学习目标
5. 可用时间
6. 学习偏好
"""
        
        response = self.call_llm(prompt)
        
        return {
            "learner_profile": response,
            "next_questions": self._generate_next_questions(response),
            "completed": self._is_diagnosis_complete(response)
        }
    
    def _generate_next_questions(self, profile: str) -> list:
        prompt = f"""根据以下学习者画像，生成1-2个追问问题以完善画像：
{profile}
"""
        result = self.call_llm(prompt)
        return result.split("\n") if result else []
    
    def _is_diagnosis_complete(self, profile: str) -> bool:
        required_fields = ["专业背景", "编程经验", "学习目标", "可用时间"]
        return all(field in profile for field in required_fields)