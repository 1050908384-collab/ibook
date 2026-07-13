from backend.agents.base_agent import BaseAgent
from typing import Dict, Any


class TechReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_04_tech_review", "技术审核Agent")
        self.available_tools = [
            "code_parser_tool",
            "knowledge_verify_tool",
            "dependency_check_tool",
            "rule_check_tool"
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        content = input_data.get("content", "")
        references = input_data.get("references", [])
        
        prompt = f"""你是一个技术审核专家，负责检查生成内容的技术准确性。
        
生成内容：{content}
参考资料：{references}

请进行以下检查：
1. 技术事实核查：检查概念、参数配置是否与知识库一致
2. 代码可执行性检查：检查语法和依赖是否正确（不实际运行）
3. 行业规范合规检查：检查是否有违规表述
4. 幻觉标记：如发现"知识库中无依据"的表述，标记为潜在幻觉

输出审核结果（JSON格式）：
- passed: 是否通过
- issues: 问题列表
- hallucination_evidence: 幻觉证据
- suggestions: 修改建议
"""
        
        response = self.call_llm(prompt)
        
        return {
            "passed": self._parse_passed(response),
            "issues": self._extract_issues(response),
            "hallucination_detected": self._detect_hallucination(response),
            "suggestions": self._extract_suggestions(response)
        }
    
    def _parse_passed(self, response: str) -> bool:
        return "passed: true" in response.lower() or "通过" in response
    
    def _extract_issues(self, response: str) -> list:
        issues = []
        for line in response.split("\n"):
            if "issue" in line.lower() or "问题" in line:
                issues.append(line.strip())
        return issues
    
    def _detect_hallucination(self, response: str) -> bool:
        return "幻觉" in response or "hallucination" in response.lower()
    
    def _extract_suggestions(self, response: str) -> list:
        suggestions = []
        for line in response.split("\n"):
            if "suggestion" in line.lower() or "建议" in line:
                suggestions.append(line.strip())
        return suggestions