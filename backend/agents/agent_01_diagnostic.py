from typing import Dict, Any
from agents.base_agent import BaseAgent
from tools.agent1_tools.extract_skills_tool import ExtractSkillsTool
from tools.agent1_tools.calculate_confidence_tool import CalculateConfidenceTool
from tools.agent1_tools.graph_query_tool import GraphQueryTool
from tools.agent1_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent1_tools.stage_generator_tool import StageGeneratorTool

class DiagnosticAgent(BaseAgent):
    def __init__(self):
        super().__init__("diagnostic_agent", ["learner_profiling", "knowledge_graph_mapping", "stage_planning"])
        self.extract_skills_tool = ExtractSkillsTool()
        self.calculate_confidence_tool = CalculateConfidenceTool()
        self.graph_query_tool = GraphQueryTool()
        self.knowledge_block_query_tool = KnowledgeBlockQueryTool()
        self.stage_generator_tool = StageGeneratorTool()
        self.load_tools([
            self.extract_skills_tool,
            self.calculate_confidence_tool,
            self.graph_query_tool,
            self.knowledge_block_query_tool,
            self.stage_generator_tool,
        ])

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

    async def generate_stage(self, session_id: str, profile: Dict[str, Any], stage_num: int = 1) -> Dict[str, Any]:
        """生成下一学习阶段计划（原Agent2功能，已合并至Agent1）。"""
        knowledge_gaps = profile.get("knowledge_gaps", [])
        unmastered_gaps = [g for g in knowledge_gaps if g.get("status") != "已掌握"]
        
        if not unmastered_gaps:
            raise ValueError("No unmastered knowledge gap remains")
        
        gap = unmastered_gaps[0]
        skill_name = gap.get("skill", "")
        
        block_info = {}
        try:
            block_info = await self.knowledge_block_query_tool.query_block(skill_name)
        except Exception:
            block_info = {}
        
        generated_stage = {}
        try:
            generated_stage = await self.stage_generator_tool.generate_stage(
                session_id, gap, block_info, profile.get("goal", ""), stage_num
            )
        except Exception:
            generated_stage = {}
        
        return {
            "stage_id": generated_stage.get("stage_id", ""),
            "stage_num": stage_num,
            "block_id": block_info.get("block_id", ""),
            "block_name": block_info.get("block_name", skill_name),
            "estimated_hours": generated_stage.get("estimated_hours", 2.0),
            "gap_skill": skill_name,
            "prerequisites": block_info.get("prerequisites", []),
            "status": "pending",
        }
