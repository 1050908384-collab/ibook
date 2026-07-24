from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from tools.agent3_tools.vector_search_tool import VectorSearchTool
from tools.agent3_tools.llm_generate_tool import LLMGenerateTool
from tools.agent3_tools.code_generation_tool import CodeGenerationTool
from tools.agent3_tools.question_extract_tool import QuestionExtractTool
from tools.agent3_tools.template_render_tool import TemplateRenderTool
from tools.agent3_tools.format_converter_tool import FormatConverterTool

class ContentGeneratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("content_generator_agent", ["knowledge_graph", "content_strategy", "language_models"])
        self.vector_search_tool = VectorSearchTool()
        self.llm_generate_tool = LLMGenerateTool()
        self.code_generation_tool = CodeGenerationTool()
        self.question_extract_tool = QuestionExtractTool()
        self.template_render_tool = TemplateRenderTool()
        self.format_converter_tool = FormatConverterTool()
        self.load_tools([self.vector_search_tool, self.llm_generate_tool, self.code_generation_tool, 
                        self.question_extract_tool, self.template_render_tool, self.format_converter_tool])

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
        related_docs = await self.vector_search_tool.search_by_query(question, top_k=3)
        answer = await self.llm_generate_tool.generate_answer(question, related_docs)
        return answer

    async def modify_answer(self, session_id: str, answer: str, errors: Dict[str, Any], question: str = "") -> str:
        tech_errors = errors.get("tech", [])
        quality_errors = errors.get("quality", [])
        
        if tech_errors or quality_errors:
            answer = await self.llm_generate_tool._modify_answer(answer, tech_errors, quality_errors, question)
        
        return answer