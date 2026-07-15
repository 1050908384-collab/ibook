from typing import List, Dict, Any
from llm.client import LLMClient
import json
import re

class DifficultyAnalyzerTool:
    def __init__(self):
        self.llm_client = LLMClient()

    def _clean_response(self, response: str) -> str:
        response = response.replace("<think>", "").replace("</think>", "")
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'\s*```', '', response)
        return response.strip()

    async def analyze(self, content: str) -> List[Dict[str, Any]]:
        if not content:
            return []
        
        system_prompt = """你是一个专业的教学质量评估专家。请分析以下内容的难度是否适合初学者学习。
        如果内容过于复杂或过于简单，请指出问题并给出建议。"""
        
        prompt = f"""请分析以下内容的学习难度：
        
        {content[:2000]}
        
        请以JSON数组格式返回发现的问题，每个问题包含：
        - type: 问题类型（too_hard/too_easy/poor_structure）
        - message: 问题描述和改进建议
        - location: 问题位置（可描述为"内容开头/中间/结尾"）
        
        如果内容难度适中，返回[]"""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        response = self._clean_response(response)
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                errors = json.loads(match.group())
                if errors:
                    print(f"=== DifficultyAnalyzerTool发现问题 === {errors}")
                return errors
        except Exception as e:
            print(f"DifficultyAnalyzerTool解析失败: {e}")
        
        return []