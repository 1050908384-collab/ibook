from typing import List, Dict, Any
from llm.client import LLMClient
import json
import re

class KnowledgeVerifyTool:
    def __init__(self):
        self.llm_client = LLMClient()

    def _clean_response(self, response: str) -> str:
        response = response.replace("<think>", "").replace("</think>", "")
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'\s*```', '', response)
        return response.strip()

    async def verify_knowledge(self, content: str) -> List[Dict[str, Any]]:
        if not content:
            return []
        
        system_prompt = """你是一个专业的技术审核专家。请检查以下内容是否存在技术错误、知识幻觉或不准确的信息。
        如果发现错误，请返回错误列表；如果没有错误，返回空列表。"""
        
        prompt = f"""请检查以下内容的技术准确性：
        
        {content[:4000]}
        
        请以JSON数组格式返回发现的错误，每个错误包含：
        - type: 错误类型（hallucination/fact_error/concept_error）
        - message: 错误描述
        - location: 错误位置（可描述为"内容开头/中间/结尾"）
        
        如果没有错误，返回[]"""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        response = self._clean_response(response)
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                errors = json.loads(match.group())
                if errors:
                    print(f"=== KnowledgeVerifyTool发现错误 === {errors}")
                return errors
        except Exception as e:
            print(f"KnowledgeVerifyTool解析失败: {e}")
        
        return []