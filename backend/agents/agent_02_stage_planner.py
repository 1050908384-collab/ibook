# -*- coding: utf-8 -*-
from typing import Any, Dict

from agents.base_agent import BaseAgent
from db.sqlite_client import SQLiteClient
from tools.agent2_tools.knowledge_block_query_tool import KnowledgeBlockQueryTool
from tools.agent2_tools.stage_generator_tool import StageGeneratorTool


class StagePlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("stage_planner_agent", ["stage_planning", "knowledge_graph_mapping"])
        self.knowledge_block_query_tool = KnowledgeBlockQueryTool()
        self.stage_generator_tool = StageGeneratorTool()
        self.db_client = SQLiteClient()
        self.load_tools([self.knowledge_block_query_tool, self.stage_generator_tool])

    async def generate_stage(self, session_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        knowledge_gaps = profile.get("knowledge_gaps", [])
        goal = profile.get("goal", "")
        daily_hours = profile.get("daily_hours")

        preferred_gap = next((gap for gap in knowledge_gaps if gap.get("progress", 0.0) < 1.0), None)
        if preferred_gap is None:
            return {"status": "completed", "message": "所有知识板块均已掌握"}

        preferred_block = await self.knowledge_block_query_tool.query_block(preferred_gap["skill"])
        preferred_block = preferred_block or {}

        candidates = self.db_client.list_learnable_subtopics(
            session_id=session_id,
            preferred_block_id=preferred_block.get("block_id"),
            max_hours=daily_hours,
        )
        if not candidates:
            return {"status": "blocked", "message": "当前没有满足前置条件的未掌握知识点"}

        next_subtopic = candidates[0]
        block_info = preferred_block
        if next_subtopic.get("block_id") != preferred_block.get("block_id"):
            block_info = await self.knowledge_block_query_tool.query_block(next_subtopic["block_id"])
            block_info = block_info or preferred_block

        return await self.stage_generator_tool.generate_stage(
            session_id=session_id,
            gap={
                **preferred_gap,
                "skill": block_info.get("block_name", preferred_gap["skill"]),
                "prerequisites": next_subtopic.get("prerequisites", []),
            },
            block_info={
                **block_info,
                "block_id": next_subtopic.get("block_id", block_info.get("block_id")),
                "estimated_hours": next_subtopic.get("estimated_hours", block_info.get("estimated_hours", 2.0)),
                "subtopic_id": next_subtopic.get("subtopic_id"),
                "subtopic_name": next_subtopic.get("subtopic_name"),
            },
            goal=goal,
        )

    async def generate_full_path(self, session_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        knowledge_gaps = profile.get("knowledge_gaps", [])
        preferred_gap = next((gap for gap in knowledge_gaps if gap.get("progress", 0.0) < 1.0), None)
        if preferred_gap is None:
            return {"total_stages": 0, "stages": []}

        preferred_block = await self.knowledge_block_query_tool.query_block(preferred_gap["skill"])
        preferred_block = preferred_block or {}
        candidates = self.db_client.list_learnable_subtopics(
            session_id=session_id,
            preferred_block_id=preferred_block.get("block_id"),
            max_hours=profile.get("daily_hours"),
        )

        stages = [
            {
                "block_id": candidate.get("block_id"),
                "subtopic_id": candidate.get("subtopic_id"),
                "subtopic_name": candidate.get("subtopic_name"),
                "estimated_hours": candidate.get("estimated_hours"),
                "prerequisites": candidate.get("prerequisites", []),
            }
            for candidate in candidates
        ]
        return {"total_stages": len(stages), "stages": stages}
