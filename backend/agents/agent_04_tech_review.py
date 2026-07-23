# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from tools.agent4_tools.code_parser_tool import CodeParserTool
from tools.agent4_tools.knowledge_verify_tool import KnowledgeVerifyTool
from tools.agent4_tools.rule_check_tool import RuleCheckTool

class TechnicalReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("technical_review_agent", ["technical_review", "industry_standards"])
        self.code_parser_tool = CodeParserTool()
        self.knowledge_verify_tool = KnowledgeVerifyTool()
        self.rule_check_tool = RuleCheckTool()
        self.load_tools([self.code_parser_tool, self.knowledge_verify_tool, self.rule_check_tool])

    async def review(self, session_id: str, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors = []
        print(f"=== Agent4 TechnicalReviewAgent 开始审核 ===")
        
        if "tutorial" in content:
            print(f"=== 审核教程内容（前200字符）===\n{content['tutorial'][:200]}")
            hallucination_errors = await self.knowledge_verify_tool.verify_knowledge(content["tutorial"])
            errors.extend(hallucination_errors)
            
            rule_errors = await self.rule_check_tool.check_rules(content["tutorial"])
            errors.extend(rule_errors)
        
        if "code_examples" in content:
            print(f"=== 审核代码示例 ===")
            for code in content["code_examples"]:
                syntax_errors = await self.code_parser_tool.parse_code(code.get("code", ""))
                errors.extend(syntax_errors)
                
                code_rule_errors = await self.rule_check_tool.check_code_rules(code.get("code", ""))
                errors.extend(code_rule_errors)
        
        if "answer" in content:
            print(f"=== 审核回答内容（前200字符）===\n{content['answer'][:200]}")
            hallucination_errors = await self.knowledge_verify_tool.verify_knowledge(content["answer"])
            errors.extend(hallucination_errors)
        
        print(f"=== Agent4 TechnicalReviewAgent 审核完成 ===")
        print(f"=== 发现技术错误: {len(errors)}个 ===")
        if errors:
            for i, e in enumerate(errors, 1):
                print(f"  {i}. [{e.get('type', 'unknown')}] {e.get('message', '')}")
        
        return errors