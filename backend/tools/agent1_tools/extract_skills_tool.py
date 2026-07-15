from typing import List, Dict, Any
from llm.client import LLMClient
import json

class ExtractSkillsTool:
    def __init__(self):
        self.llm_client = LLMClient()

    async def extract_skills(self, user_input: str) -> List[str]:
        if not user_input:
            return []
        
        system_prompt = """你是一个学习诊断专家。请分析用户的输入，提取用户已经掌握的技能标签。
        技能标签应该是具体的技术或知识领域，如：Python编程、机器学习、深度学习等。
        只返回技能标签列表，不要解释。"""
        
        prompt = f"用户输入：{user_input}\n\n请提取用户已掌握的技能标签，以JSON数组格式返回："
        
        response = await self.llm_client.complete(prompt, system_prompt)
        
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return []
        except Exception:
            return []

    async def extract_gaps(self, user_input: str) -> List[Dict[str, Any]]:
        if not user_input:
            return []
        
        system_prompt = """你是一个学习诊断专家。请分析用户的输入，识别用户想要学习但尚未掌握的知识领域（知识缺口）。
        每个知识缺口应该包含：skill（技能名称）和 level（当前水平：beginner/intermediate/advanced）。
        只返回JSON格式，不要解释。"""
        
        prompt = f"用户输入：{user_input}\n\n请识别用户的知识缺口，以JSON数组格式返回，每个元素包含skill和level字段："
        
        response = await self.llm_client.complete(prompt, system_prompt)
        
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return []
        except Exception:
            return []