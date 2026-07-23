# -*- coding: utf-8 -*-
from typing import Dict, Any
from agents.base_agent import BaseAgent
from tools.agent2_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent2_tools.stage_generator_tool import StageGeneratorTool

class StagePlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("stage_planner_agent", ["stage_planning", "knowledge_graph_mapping"])
        self.knowledge_block_query_tool = KnowledgeBlockQueryTool()
        self.stage_generator_tool = StageGeneratorTool()
        self.load_tools([self.knowledge_block_query_tool, self.stage_generator_tool])

    async def generate_stage(self, session_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        knowledge_gaps = profile.get("knowledge_gaps", [])
        goal = profile.get("goal", "")
        
        unstarted_gaps = [g for g in knowledge_gaps if g.get("status", "未开始") == "未开始"]
        
        if not unstarted_gaps:
            return {"status": "completed", "message": "所有知识盲区已掌握"}
        
        next_gap = unstarted_gaps[0]
        block_info = await self.knowledge_block_query_tool.query_block(next_gap["skill"])
        
        stage_info = await self.stage_generator_tool.generate_stage(
            session_id=session_id,
            gap=next_gap,
            block_info=block_info,
            goal=goal
        )
        
        return stage_info

    async def generate_full_path(self, session_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        knowledge_gaps = profile.get("knowledge_gaps", [])
        goal = profile.get("goal", "")
        
        stages = []
        for gap in knowledge_gaps:
            if gap.get("status") != "已掌握":
                block_info = await self.knowledge_block_query_tool.query_block(gap["skill"])
                stage = await self.stage_generator_tool.generate_stage(
                    session_id=session_id,
                    gap=gap,
                    block_info=block_info,
                    goal=goal
                )
                stages.append(stage)
        
        return {"total_stages": len(stages), "stages": stages}
