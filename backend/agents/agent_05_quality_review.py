# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
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

    def _profile_to_dict(self, profile: Any) -> Dict[str, Any]:
        if hasattr(profile, "model_dump"):
            return profile.model_dump(mode="json")
        if hasattr(profile, "dict"):
            return profile.dict()
        if isinstance(profile, dict):
            return profile
        return {}

    async def review(
        self,
        session_id: str,
        content: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        errors = []
        print(f"=== Agent5 QualityReviewAgent 开始审核 ===")

        profile_dict = self._profile_to_dict(profile) if profile else {}

        if "tutorial" in content:
            print(f"=== 分析教程难度（前200字符）===\n{content['tutorial'][:200]}")
            difficulty_errors = await self.difficulty_analyzer_tool.analyze(
                content["tutorial"], profile=profile_dict
            )
            errors.extend(difficulty_errors)

            load_errors = await self.cognitive_load_tool.analyze(content["tutorial"])
            errors.extend(load_errors)

            style_errors = await self.style_match_tool.check(
                content["tutorial"], profile=profile_dict
            )
            errors.extend(style_errors)

        if "answer" in content:
            print(f"=== 分析回答质量（前200字符）===\n{content['answer'][:200]}")
            style_errors = await self.style_match_tool.check(
                content["answer"], profile=profile_dict
            )
            errors.extend(style_errors)

            difficulty_errors = await self.difficulty_analyzer_tool.analyze(
                content["answer"], profile=profile_dict
            )
            errors.extend(difficulty_errors)

        print(f"=== Agent5 QualityReviewAgent 审核完成 ===")
        print(f"=== 发现质量问题: {len(errors)}个 ===")
        if errors:
            for i, e in enumerate(errors, 1):
                print(f"  {i}. [{e.get('type', 'unknown')}] {e.get('message', '')}")

        return errors