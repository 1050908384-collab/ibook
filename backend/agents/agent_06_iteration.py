from backend.agents.base_agent import BaseAgent
from typing import Dict, Any
from backend.config import settings


class IterationAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_06_iteration", "迭代决策Agent")
        self.available_tools = [
            "accuracy_calculator_tool",
            "coverage_calculator_tool",
            "strategy_selector_tool",
            "instruction_generator_tool"
        ]
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        accuracy = input_data.get("accuracy", 0.0)
        coverage = input_data.get("coverage", 0.0)
        learner_profile = input_data.get("learner_profile", {})
        tech_review_result = input_data.get("tech_review_result", {})
        quality_review_result = input_data.get("quality_review_result", {})
        iteration_round = input_data.get("iteration_round", 0)
        last_improvement = input_data.get("last_improvement", 0.0)
        
        strategy = self._determine_strategy(
            accuracy, coverage, iteration_round, last_improvement
        )
        
        instructions = ""
        if strategy != "terminate":
            instructions = self._generate_revision_instructions(
                strategy, tech_review_result, quality_review_result, learner_profile
            )
        
        return {
            "strategy": strategy,
            "revision_instructions": instructions,
            "terminated": strategy == "terminate",
            "termination_reason": self._get_termination_reason(strategy, accuracy, coverage)
        }
    
    def _determine_strategy(self, accuracy: float, coverage: float, 
                          iteration_round: int, last_improvement: float) -> str:
        if accuracy >= settings.ACCURACY_THRESHOLD and coverage >= settings.COVERAGE_THRESHOLD:
            return "terminate"
        
        if iteration_round >= settings.MAX_ITERATIONS:
            return "terminate"
        
        if last_improvement < settings.STAGNATION_THRESHOLD and iteration_round > 2:
            return "reinforce"
        
        if accuracy >= 0.8:
            return "advance"
        elif accuracy >= 0.5:
            return "retreat"
        else:
            return "reinforce"
    
    def _generate_revision_instructions(self, strategy: str, 
                                       tech_review: Dict[str, Any],
                                       quality_review: Dict[str, Any],
                                       learner_profile: Dict[str, Any]) -> str:
        prompt = f"""你是一个迭代决策专家，负责生成具体的修改指令。

当前策略：{strategy}
技术审核结果：{tech_review}
质量评审结果：{quality_review}
学习者画像：{learner_profile}

请生成具体的修改指令：
- advance策略：生成更难/更综合的任务
- retreat策略：用更通俗的方式重新解释
- reinforce策略：推荐补充前置知识模块
- correct策略：修正具体错误
"""
        
        return self.call_llm(prompt)
    
    def _get_termination_reason(self, strategy: str, accuracy: float, coverage: float) -> str:
        if strategy == "terminate":
            if accuracy >= settings.ACCURACY_THRESHOLD and coverage >= settings.COVERAGE_THRESHOLD:
                return "达到目标：正确率>85%且覆盖率>90%"
            else:
                return f"达到最大迭代次数，当前正确率{accuracy:.2%}，覆盖率{coverage:.2%}"
        return ""