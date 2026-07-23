#LLM提取→技能标签+盲区
from typing import List, Dict, Any
from llm.client import LLMClient
import json
import re

class ExtractSkillsTool:
    def __init__(self):
        self.llm_client = LLMClient()

    def _clean_response(self, response: str) -> str:
        response = response.replace("<think>", "").replace("</think>", "")
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()

    async def extract_skills(self, user_input: str) -> List[str]:
        if not user_input:
            return []
        
        system_prompt = """你是一个学习诊断专家。请分析用户的输入，提取用户已经掌握的技能标签。
        技能标签应该是具体的技术或知识领域，如：Python编程、机器学习、深度学习等。"""
        
        example = '["Python", "机器学习"]'
        prompt = f"用户输入：{user_input}\n\n请提取用户已掌握的技能标签，以JSON数组格式返回，例如：{example}"
        
        response = await self.llm_client.complete(prompt, system_prompt)
        print(f"=== extract_skills LLM响应 ===\n{response}\n=== 响应结束 ===")
        response = self._clean_response(response)
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                print(f"=== extract_skills 解析结果 ===\n{result}\n=== 解析结束 ===")
                return result
            return ["编程基础"]
        except Exception as e:
            print(f"extract_skills JSON解析失败: {e}, 使用默认值")
            return ["编程基础"]

    async def extract_gaps(self, user_input: str) -> List[Dict[str, Any]]:
        if not user_input:
            return []
        
        system_prompt = """你是一个学习诊断专家。请分析用户的输入，识别用户想要学习但尚未掌握的知识领域（知识缺口）。"""
        
        example = '[{"skill": "大模型微调", "level": "beginner"}]'
        prompt = f"用户输入：{user_input}\n\n请识别用户的知识缺口，以JSON数组格式返回，每个元素包含skill和level字段，例如：{example}"
        
        response = await self.llm_client.complete(prompt, system_prompt)
        print(f"=== extract_gaps LLM响应 ===\n{response}\n=== 响应结束 ===")
        response = self._clean_response(response)
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                print(f"=== extract_gaps 解析结果 ===\n{result}\n=== 解析结束 ===")
                return result
            return [{"skill": "大模型微调", "level": "beginner"}]
        except Exception as e:
            print(f"extract_gaps JSON解析失败: {e}, 使用默认值")
            return [{"skill": "大模型微调", "level": "beginner"}]