from typing import Dict, Any, List
import re
from agents.base_agent import BaseAgent
from tools.agent3_tools.vector_search_tool import VectorSearchTool
from tools.agent3_tools.llm_generate_tool import LLMGenerateTool
from tools.agent3_tools.code_generation_tool import CodeGenerationTool
from tools.agent3_tools.question_extract_tool import QuestionExtractTool
from tools.agent3_tools.template_render_tool import TemplateRenderTool
from tools.agent3_tools.format_converter_tool import FormatConverterTool
from tools.agent3_tools.code_executor_tool import CodeExecutorTool

class ContentGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("content_generator_agent", ["knowledge_graph", "content_strategy", "language_models"])
        self.vector_search_tool = VectorSearchTool()
        self.llm_generate_tool = LLMGenerateTool()
        self.code_generation_tool = CodeGenerationTool()
        self.question_extract_tool = QuestionExtractTool()
        self.template_render_tool = TemplateRenderTool()
        self.format_converter_tool = FormatConverterTool()
        self.code_executor_tool = CodeExecutorTool()
        self.load_tools([self.vector_search_tool, self.llm_generate_tool, self.code_generation_tool, 
                        self.question_extract_tool, self.template_render_tool, self.format_converter_tool,
                        self.code_executor_tool])

    async def generate_learning_package(self, session_id: str, stage_info: Dict[str, Any]) -> Dict[str, Any]:
        block_id = stage_info.get("block_id", "")
        gap = stage_info.get("gap", {})
        
        related_docs = await self.vector_search_tool.search_by_block(block_id)
        tutorial = await self.llm_generate_tool.generate_tutorial(block_id, gap, related_docs)
        
        code_examples = await self.code_generation_tool.generate_code(block_id, stage_info)
        
        questions = await self.question_extract_tool.extract_questions(tutorial, stage_info)
        
        return {
            "tutorial": tutorial,
            "code_examples": code_examples,
            "questions": questions,
            "format": "markdown"
        }

    async def modify_package(self, session_id: str, package: Dict[str, Any], errors: Dict[str, Any]) -> Dict[str, Any]:
        tech_errors = errors.get("tech", [])
        quality_errors = errors.get("quality", [])
        
        if tech_errors and "tutorial" in package:
            package["tutorial"] = await self.llm_generate_tool.fix_tutorial(package["tutorial"], tech_errors)
        
        if quality_errors and "tutorial" in package:
            package["tutorial"] = await self.llm_generate_tool.adjust_tutorial_difficulty(package["tutorial"], quality_errors)
        
        return package

    async def answer_question(self, session_id: str, question: str) -> str:
        code_snippets = self._extract_code_from_question(question)
        
        if code_snippets:
            return await self._handle_code_correction(session_id, question, code_snippets)
        
        related_docs = await self.vector_search_tool.search_by_query(question, top_k=3)
        answer = await self.llm_generate_tool.generate_answer(question, related_docs)
        return answer

    def _extract_code_from_question(self, question: str) -> List[str]:
        patterns = [
            r'```(\w*)\n(.*?)```',
            r'```\n(.*?)```',
            r'`([^`]+)`',
        ]
        code_snippets = []
        for pattern in patterns:
            matches = re.findall(pattern, question, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    code = match[1].strip()
                else:
                    code = match.strip()
                if code and len(code) > 5:
                    code_snippets.append(code)
        return code_snippets

    async def _handle_code_correction(self, session_id: str, question: str, code_snippets: List[str]) -> str:
        execution_results = []
        for code in code_snippets[:3]:
            result = await self.code_executor_tool.execute_code(code)
            execution_results.append({
                "code": code,
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "return_code": result.get("return_code", 0),
                "timeout": result.get("timeout", False),
            })

        execution_context = "\n\n".join([
            f"代码片段{i+1}执行结果:\n输出: {r['output']}\n错误: {r['error']}\n返回码: {r['return_code']}"
            for i, r in enumerate(execution_results)
        ])

        prompt = f"""你是一个专业的代码纠错老师。请按照以下结构回答：

## 错误分析
分析代码中的错误原因

## 修改步骤
分步骤说明修改方案

## 修改后完整代码
提供修改后的完整代码

## 修改说明
总结修改要点和学习建议

用户问题：{question}

代码执行结果：
{execution_context}

请先肯定用户的尝试，再指出问题，最后给出修改方案。"""

        return await self.llm_generate_tool.generate_answer(prompt, [])

    async def modify_answer(self, session_id: str, answer: str, errors: Dict[str, Any], question: str = "") -> str:
        tech_errors = errors.get("tech", [])
        quality_errors = errors.get("quality", [])
        
        if tech_errors or quality_errors:
            answer = await self.llm_generate_tool._modify_answer(answer, tech_errors, quality_errors, question)
        
        return answer