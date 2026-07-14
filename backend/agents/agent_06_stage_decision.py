from typing import Dict, Any, List
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

    async def update_user_data(self, session_id: str, accuracy: float, profile: Dict[str, Any], 
                              qa_history: List[Dict[str, Any]], interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        updated_data = {}
        
        strategy = await self.strategy_selector_tool.select_strategy(accuracy, profile)
        updated_data["strategy"] = strategy
        
        updated_gaps = await self.profile_update_tool.update_gaps(profile["knowledge_gaps"], accuracy)
        updated_data["knowledge_gaps"] = updated_gaps
        
        rhythm = await self.profile_update_tool.infer_rhythm(interactions)
        updated_data["learning_rhythm"] = rhythm
        
        confidence_index = await self.profile_update_tool.infer_confidence(qa_history, accuracy)
        updated_data["confidence_index"] = confidence_index
        
        confusion_markers = await self.profile_update_tool.detect_confusion(qa_history)
        updated_data["confusion_markers"] = confusion_markers
        
        return updated_data
