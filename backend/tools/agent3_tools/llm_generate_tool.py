from typing import List, Dict, Any
from llm.client import LLMClient

class LLMGenerateTool:
    def __init__(self):
        self.llm_client = LLMClient()

    async def generate_tutorial(self, block_id: str, gap: Dict[str, Any], related_docs: List[Dict[str, Any]]) -> str:
        skill_name = gap.get("skill", block_id)
        
        system_prompt = """你是一个专业的技术教程编写专家。请用通俗易懂的语言编写一篇技术教程。
        教程应该包含：概念介绍、核心原理、实践步骤、示例代码和常见问题。
        目标读者是希望学习该技术的初学者到中级学习者。"""
        
        prompt = f"""请编写一篇关于"{skill_name}"的技术教程，包括：
1. 概念介绍：什么是{skill_name}？为什么需要学习它？
2. 核心原理：{skill_name}的工作原理是什么？
3. 实践步骤：如何开始学习和使用{skill_name}？
4. 示例代码：提供一个简单的代码示例
5. 常见问题：学习{skill_name}时常见的问题和解决方案

请用markdown格式编写，内容要详细但易于理解。"""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        print(f"=== generate_tutorial LLM响应（前500字符）===\n{response[:500]}\n=== 响应结束 ===")
        return response

    async def fix_tutorial(self, tutorial: str, errors: List[Dict[str, Any]]) -> str:
        error_str = "\n".join([f"- {e.get('message', '')}" for e in errors])
        
        system_prompt = """你是一个专业的技术编辑。请根据反馈意见修改教程内容，确保技术准确性。"""
        
        prompt = f"""原教程内容：
{tutorial}

需要修复的问题：
{error_str}

请修改教程，解决以上问题，并保持内容的完整性和可读性。"""
        
        return await self.llm_client.complete(prompt, system_prompt)

    async def adjust_tutorial_difficulty(self, tutorial: str, errors: List[Dict[str, Any]]) -> str:
        return tutorial

    async def generate_answer(self, question: str, related_docs: List[Dict[str, Any]]) -> str:
        system_prompt = """你是一个专业的技术问答助手。请准确、详细地回答用户的问题。
        回答应该包括：问题解答、相关概念解释、示例代码（如果适用）、参考资料。"""
        
        prompt = f"""问题：{question}

请详细回答这个问题，确保技术准确性和易懂性。如果涉及代码，请提供示例代码。"""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        print(f"=== generate_answer LLM响应（前500字符）===\n{response[:500]}\n=== 响应结束 ===")
        return response

    async def fix_answer(self, answer: str, errors: List[Dict[str, Any]], question: str = "") -> str:
        return self._modify_answer(answer, errors, [], question)

    async def adjust_answer(self, answer: str, errors: List[Dict[str, Any]], question: str = "") -> str:
        return self._modify_answer(answer, [], errors, question)

    async def _modify_answer(self, answer: str, tech_errors: List[Dict[str, Any]], quality_errors: List[Dict[str, Any]], question: str = "") -> str:
        tech_error_str = "\n".join([f"- [技术错误] [{e.get('type', 'unknown')}] {e.get('message', '')}" for e in tech_errors])
        quality_error_str = "\n".join([f"- [质量问题] [{e.get('type', 'unknown')}] {e.get('message', '')}" for e in quality_errors])
        all_errors = []
        if tech_error_str:
            all_errors.append(f"技术错误：\n{tech_error_str}")
        if quality_error_str:
            all_errors.append(f"质量问题：\n{quality_error_str}")
        error_str = "\n\n".join(all_errors)
        
        system_prompt = """你是一个专业的技术编辑。请根据反馈意见修改回答。
        重要：只修改被指出有问题的部分，保持其他正确内容不变，不要删减任何信息。
        修改后的回答长度应该与原回答相当或更长，确保所有技术细节都被保留。"""
        
        question_part = f"原始问题：{question}\n\n" if question else ""
        
        prompt = f"""{question_part}原回答：
{answer}

需要修复的问题：
{error_str}

请修改回答，只针对上述问题进行修正。保持回答的完整性和详细程度，不要删减正确的内容。
修改后的回答应该包含所有原始信息，只修正错误的部分。如果没有需要修复的问题，请原样返回原回答。"""
        
        response = await self.llm_client.complete(prompt, system_prompt)
        print(f"=== _modify_answer LLM响应（前300字符）===\n{response[:300]}\n=== 响应结束 ===")
        return response