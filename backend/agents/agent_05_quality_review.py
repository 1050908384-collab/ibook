from backend.agents.base_agent import BaseAgent
from typing import Dict, Any


class QualityReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_05_quality_review", "质量评审Agent")
        self.available_tools = [
            "difficulty_analyzer_tool",
            "cognitive_load_tool",
            "environment_check_tool",
            "engagement_analyzer_tool"
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        learner_profile = input_data.get("learner_profile", {})
        
        prompt = f"""你是一个质量评审专家，从教学适配性和学习者体验角度评估资源。
        
生成内容：{content}
学习者画像：{learner_profile}

请进行以下评估：
1. 教学逻辑检查：评估内容编排是否符合"从已知到未知"的渐进原则
2. 认知负荷评估：评估单次学习内容的复杂度是否超过该学习者的承载能力
3. 动机与参与度评估：检查是否有足够的情境化、案例化内容激发学习动机
4. 实践可行性评估：评估所需的软硬件环境是否匹配学习者的实际情况
5. 个性化匹配度评估：评估资源是否真正命中学习者的知识盲区

输出评估结果（JSON格式）：
- pedagogical_score: 教学分数(0-1)
- cognitive_load: 认知负荷(0-1)
- engagement_score: 参与度分数(0-1)
- environment_fit: 环境匹配度(0-1)
- personalization_score: 个性化匹配度(0-1)
- overall_score: 综合分数(0-1)
- suggestions: 改进建议
"""
        
        response = self.call_llm(prompt)
        
        return {
            "scores": response,
            "overall_score": self._extract_overall_score(response),
            "suggestions": self._extract_suggestions(response)
        }
    
    def _extract_overall_score(self, response: str) -> float:
        import re
        match = re.search(r"overall_score[\s:]+([0-1](?:\.\d+)?)", response)
        return float(match.group(1)) if match else 0.5
    
    def _extract_suggestions(self, response: str) -> list:
        suggestions = []
        for line in response.split("\n"):
            if "suggestion" in line.lower() or "建议" in line:
                suggestions.append(line.strip())
        return suggestions