from typing import Dict, Any
from agents.base_agent import BaseAgent
from tools.agent1_tools.extract_skills_tool import ExtractSkillsTool
from tools.agent1_tools.calculate_confidence_tool import CalculateConfidenceTool
from tools.agent1_tools.graph_query_tool import GraphQueryTool

class DiagnosticAgent(BaseAgent):
    def __init__(self):
        super().__init__("diagnostic_agent", ["learner_profiling", "knowledge_graph_mapping"])
        self.extract_skills_tool = ExtractSkillsTool()
        self.calculate_confidence_tool = CalculateConfidenceTool()
        self.graph_query_tool = GraphQueryTool()
        self.load_tools([self.extract_skills_tool, self.calculate_confidence_tool, self.graph_query_tool])

    async def generate_profile(self, session_id: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = raw_data.get("user_input", "")
        
        skill_tags = await self.extract_skills_tool.extract_skills(user_input)
        knowledge_gaps = await self.extract_skills_tool.extract_gaps(user_input)
        
        for gap in knowledge_gaps:
            prerequisites = await self.graph_query_tool.query_prerequisites(gap["skill"])
            gap["prerequisites"] = prerequisites
        
        confidence = await self.calculate_confidence_tool.calculate(user_input, skill_tags, knowledge_gaps)
        
        profile = {
            "session_id": session_id,
            "skill_tags": skill_tags,
            "knowledge_gaps": knowledge_gaps,
            "goal": raw_data.get("goal", ""),
            "daily_hours": raw_data.get("daily_hours", 2.0),
            "learning_style": raw_data.get("learning_style", "mixed"),
            "learning_rhythm": raw_data.get("learning_rhythm", "medium"),
            "confidence_index": raw_data.get("confidence_index", 0.5),
            "confusion_markers": [],
            "confidence": confidence
        }
        
        return profile

    async def update_profile(self, session_id: str, profile: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if key in profile:
                profile[key] = value
        
        return profile
