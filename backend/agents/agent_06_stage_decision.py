from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from tools.agent6_tools.accuracy_calculator_tool import AccuracyCalculatorTool
from tools.agent6_tools.profile_update_tool import ProfileUpdateTool
from tools.agent6_tools.strategy_selector_tool import StrategySelectorTool
from tools.agent6_tools.stage_summary_generator_tool import StageSummaryGeneratorTool

class StageDecisionAgent(BaseAgent):
    def __init__(self):
        super().__init__("stage_decision_agent", ["iteration_strategy", "user_model_update"])
        self.accuracy_calculator_tool = AccuracyCalculatorTool()
        self.profile_update_tool = ProfileUpdateTool()
        self.strategy_selector_tool = StrategySelectorTool()
        self.stage_summary_generator_tool = StageSummaryGeneratorTool()
        self.load_tools([
            self.accuracy_calculator_tool,
            self.profile_update_tool,
            self.strategy_selector_tool,
            self.stage_summary_generator_tool,
        ])

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

    async def process(
        self,
        session_id: str,
        accuracy: float,
        profile: Dict[str, Any],
        qa_history: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        stage_name: str = "",
        question_records: Optional[List[Dict[str, Any]]] = None,
        emotion_score: Optional[float] = None,
        time_spent_hours: float = 0,
        expected_time_hours: float = 0,
        total_knowledge_points: int = 0,
        total_answers: int = 0,
    ) -> Dict[str, Any]:
        """
        处理阶段结束后的全部更新工作，包括画像更新和阶段总结生成。

        返回结构：
        {
            "profile_updates": {...},
            "stage_summary": str,  # Markdown格式的阶段总结
            "strategy": str,
            "update_rationale": str,
        }
        """
        # 1. 执行原有画像更新逻辑
        updated_data = await self.update_user_data(
            session_id, accuracy, profile, qa_history, interactions
        )

        # 2. 准备总结生成所需的数据
        profile_dict = self._profile_to_dict(profile) if profile else {}
        
        # 从更新后的数据中提取关键指标
        knowledge_gaps = updated_data.get("knowledge_gaps", [])
        confusion_tags = updated_data.get("confusion_markers", [])
        pace_adjustment = updated_data.get("learning_rhythm", "medium")
        next_strategy = updated_data.get("strategy", "reinforce")
        
        # 从输入数据推断情绪（如果未直接提供）
        if emotion_score is None and qa_history:
            emotion_scores = []
            for qa in qa_history:
                if isinstance(qa, dict) and "emotion_score" in qa:
                    try:
                        emotion_scores.append(float(qa["emotion_score"]))
                    except (ValueError, TypeError):
                        pass
            if emotion_scores:
                emotion_score = sum(emotion_scores) / len(emotion_scores)
        
        # 如果未提供答题记录，从qa_history推断
        if question_records is None:
            question_records = qa_history or []
        
        # 3. 生成阶段总结报告
        stage_summary = await self.stage_summary_generator_tool.generate_summary(
            stage_name=stage_name or "当前学习",
            accuracy=accuracy,
            knowledge_gaps=knowledge_gaps,
            confusion_tags=confusion_tags,
            question_records=question_records,
            emotion_score=emotion_score,
            pace_adjustment=pace_adjustment,
            time_spent_hours=time_spent_hours,
            expected_time_hours=expected_time_hours,
            next_strategy=next_strategy,
            total_knowledge_points=total_knowledge_points or len(knowledge_gaps),
            total_answers=total_answers or len(question_records),
        )

        # 4. 组装最终输出
        profile_updates = {
            "knowledge_gaps": knowledge_gaps,
            "dynamic_state": {
                "confidence_index": updated_data.get("confidence_index", 0.5),
            },
            "pace_adjustment": pace_adjustment,
            "confusion_tags": confusion_tags,
            "next_strategy": next_strategy,
        }

        return {
            "profile_updates": profile_updates,
            "stage_summary": stage_summary,
            "strategy": next_strategy,
            "update_rationale": f"阶段{stage_name}完成：正确率{accuracy:.0%}，策略{next_strategy}，共{len(knowledge_gaps)}个知识点",
        }