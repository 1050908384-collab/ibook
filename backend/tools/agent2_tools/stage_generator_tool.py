from typing import Dict, Any
from llm.client import LLMClient
import json

class StageGeneratorTool:
    def __init__(self):
        self.llm_client = LLMClient()

    async def generate_stage(self, session_id: str, gap: Dict[str, Any], block_info: Dict[str, Any], goal: str) -> Dict[str, Any]:
        skill_name = gap.get("skill", "")
        level = gap.get("level", "beginner")
        
        system_prompt = """你是一个学习规划专家。请根据用户的知识缺口和学习目标，生成一个学习阶段计划。
        返回JSON格式，包含：stage_id（阶段ID）、title（阶段标题）、description（阶段描述）、
        objectives（学习目标列表）、duration_hours（预计学时）、difficulty（难度：easy/medium/hard）。"""
        
        prompt = f"""知识缺口：{skill_name}
当前水平：{level}
学习目标：{goal}

请生成一个学习阶段计划，以JSON格式返回："""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        
        try:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        
        return {
            "stage_id": f"stage_{session_id[:8]}",
            "title": f"学习{skill_name}",
            "description": f"掌握{skill_name}的核心概念和实践技能",
            "objectives": [f"理解{skill_name}的基本概念", f"掌握{skill_name}的核心技术", f"能够独立完成{skill_name}相关任务"],
            "duration_hours": 4,
            "difficulty": "medium"
        }