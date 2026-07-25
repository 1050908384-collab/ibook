from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from tools.agent6_tools.accuracy_calculator_tool import AccuracyCalculatorTool
from tools.agent6_tools.profile_update_tool import ProfileUpdateTool
from tools.agent6_tools.strategy_selector_tool import StrategySelectorTool

class StageDecisionAgent(BaseAgent):
    def __init__(self):
        super().__init__("stage_decision_agent", ["iteration_strategy", "user_model_update"])
        self.accuracy_calculator_tool = AccuracyCalculatorTool()
        self.profile_update_tool = ProfileUpdateTool()
        self.strategy_selector_tool = StrategySelectorTool()
        self.load_tools([self.accuracy_calculator_tool, self.profile_update_tool, self.strategy_selector_tool])

    def _profile_to_dict(self, profile: Any) -> Dict[str, Any]:
        if hasattr(profile, "model_dump"):
            return profile.model_dump(mode="json")
        if hasattr(profile, "dict"):
            return profile.dict()
        if isinstance(profile, dict):
            return profile
        return {}

    async def update_user_data(
        self,
        session_id: str,
        accuracy: float,
        profile: Dict[str, Any],
        qa_history: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        updated_data = {}

        profile_dict = self._profile_to_dict(profile) if profile else {}

        strategy = await self.strategy_selector_tool.select_strategy(accuracy, profile_dict)
        updated_data["strategy"] = strategy

        knowledge_gaps = profile_dict.get("knowledge_gaps", [])
        updated_gaps = await self.profile_update_tool.update_gaps(knowledge_gaps, accuracy)
        updated_data["knowledge_gaps"] = updated_gaps

        rhythm = await self.profile_update_tool.infer_rhythm(interactions)
        updated_data["learning_rhythm"] = rhythm

        confidence_index = await self.profile_update_tool.infer_confidence(qa_history, accuracy)
        updated_data["confidence_index"] = confidence_index

        confusion_markers = await self.profile_update_tool.detect_confusion(qa_history)
        updated_data["confusion_markers"] = confusion_markers

        return updated_data