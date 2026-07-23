# -*- coding: utf-8 -*-
from typing import Dict, Any
from llm.client import LLMClient
import json
import re

class StageGeneratorTool:
    def __init__(self):
        self.llm_client = LLMClient()

    def _clean_response(self, response: str) -> str:
        response = response.replace("<think>", "").replace("</think>", "")
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'\s*```', '', response)
        return response.strip()

    async def generate_stage(self, session_id: str, gap: Dict[str, Any], block_info: Dict[str, Any], goal: str) -> Dict[str, Any]:
        skill_name = gap.get("skill", "未知技能")
        level = gap.get("level", "beginner")
        
        system_prompt = """你是一个学习规划专家。请根据用户的知识缺口和学习目标，生成一个学习阶段计划。"""
        
        prompt = f"""知识缺口：{skill_name}
当前水平：{level}
学习目标：{goal}

请生成一个学习阶段计划，以JSON对象格式返回，包含：stage_id、title、description、objectives、duration_hours、difficulty。"""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        print(f"=== generate_stage LLM响应 ===\n{response}\n=== 响应结束 ===")
        response = self._clean_response(response)
        
        try:
            array_match = re.search(r'\[.*\]', response, re.DOTALL)
            if array_match:
                result = json.loads(array_match.group())
                if isinstance(result, list) and len(result) > 0:
                    print(f"=== generate_stage 解析结果（数组取第一个）===\n{result[0]}\n=== 解析结束 ===")
                    return result[0]
            
            obj_match = re.search(r'\{.*\}', response, re.DOTALL)
            if obj_match:
                result = json.loads(obj_match.group())
                print(f"=== generate_stage 解析结果（单个对象）===\n{result}\n=== 解析结束 ===")
                return result
        except Exception as e:
            print(f"generate_stage JSON解析失败: {e}, 使用默认值")
        
        return {
            "stage_id": f"stage_{session_id[:8]}",
            "title": f"学习{skill_name}",
            "description": f"掌握{skill_name}的核心概念和实践技能",
            "objectives": [f"理解{skill_name}的基本概念", f"掌握{skill_name}的核心技术", f"能够独立完成{skill_name}相关任务"],
            "duration_hours": 4,
            "difficulty": "medium"
        }