from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from tools.agent5_tools.difficulty_analyzer_tool import DifficultyAnalyzerTool
from tools.agent5_tools.cognitive_load_tool import CognitiveLoadTool
from tools.agent5_tools.style_match_tool import StyleMatchTool

class QualityReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__("quality_review_agent", ["pedagogy", "learner_model", "quality_metrics"])
        self.difficulty_analyzer_tool = DifficultyAnalyzerTool()
        self.cognitive_load_tool = CognitiveLoadTool()
        self.style_match_tool = StyleMatchTool()
        self.load_tools([self.difficulty_analyzer_tool, self.cognitive_load_tool, self.style_match_tool])

    async def review(self, session_id: str, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        errors = []
        
        if "tutorial" in content:
            difficulty_errors = await self.difficulty_analyzer_tool.analyze(content["tutorial"])
            errors.extend(difficulty_errors)
            
            load_errors = await self.cognitive_load_tool.analyze(content["tutorial"])
            errors.extend(load_errors)
            
            style_errors = await self.style_match_tool.check(content["tutorial"])
            errors.extend(style_errors)
        
        if "answer" in content:
            style_errors = await self.style_match_tool.check(content["answer"])
            errors.extend(style_errors)
        
        return errors
