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
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        emotion_score: Optional[float] = None,
        gap_skill: str = "",
    ) -> Dict[str, Any]:
        updated_data = {}

        profile_dict = self._profile_to_dict(profile) if profile else {}

        # 策略选择：透传 iteration_history，激活 backtrack/终止逻辑（iteration_strategy.skill.md 规则2/3）
        # 使用 full 版本，回溯目标（backtrack_target）与终止/重置信号一并透传，不再丢弃
        try:
            strategy_full = await self.strategy_selector_tool.select_strategy_full(
                accuracy, profile_dict, iteration_history=iteration_history
            )
            strategy = strategy_full.get("strategy_label", "reinforce")
        except Exception:
            strategy = "reinforce" if accuracy >= 0.6 else "simplify"
            strategy_full = {
                "strategy_label": strategy,
                "backtrack_target": None,
                "iteration_reset": False,
                "should_terminate": False,
            }
        updated_data["strategy"] = strategy
        updated_data["backtrack_target"] = strategy_full.get("backtrack_target")
        updated_data["iteration_reset"] = strategy_full.get("iteration_reset", False)
        updated_data["should_terminate"] = strategy_full.get("should_terminate", False)
        updated_data["decision"] = strategy_full.get("decision")

        try:
            knowledge_gaps = profile_dict.get("knowledge_gaps", [])
            # 透传原始答题记录、情绪均分与学习者经验标签，激活完整 BKT 引擎（Wt/We 修正）
            # 只读层记录不含 skill 字段，用当前阶段 gap_skill 补齐归属，避免 _group_by_skill
            # 全部归入 "unknown" 导致 BKT 退化为聚合兜底（与 DeterministicStageDecisionAgent 对齐）
            bkt_records = []
            for qa in (qa_history or []):
                if isinstance(qa, dict):
                    rec = dict(qa)
                    if not rec.get("skill") and not rec.get("knowledge_point") and gap_skill:
                        rec["skill"] = gap_skill
                    bkt_records.append(rec)
            updated_gaps = await self.profile_update_tool.update_gaps(
                knowledge_gaps, accuracy,
                question_records=bkt_records,
                emotion_score=emotion_score,
                skill_tags=profile_dict.get("skill_tags", []),
            )
            updated_data["knowledge_gaps"] = updated_gaps
        except Exception:
            updated_data["knowledge_gaps"] = profile_dict.get("knowledge_gaps", [])

        try:
            rhythm = await self.profile_update_tool.infer_rhythm(interactions)
        except Exception:
            rhythm = "medium"
        updated_data["learning_rhythm"] = rhythm

        try:
            confidence_index = await self.profile_update_tool.infer_confidence(qa_history, accuracy)
        except Exception:
            confidence_index = max(0.0, min(1.0, accuracy))
        updated_data["confidence_index"] = confidence_index

        try:
            confusion_markers = await self.profile_update_tool.detect_confusion(qa_history)
        except Exception:
            confusion_markers = []
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
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        gap_skill: str = "",
    ) -> Dict[str, Any]:
        """
        处理阶段结束后的全部更新工作，包括画像更新和阶段总结生成。

        返回结构：
        {
            "decision": str,            # continue/repeat/completed
            "profile_updates": {...},   # 含 knowledge_gaps(含status/mastery)、dynamic_state、pace_adjustment、confusion_tags、next_strategy
            "stage_summary": str,       # Markdown格式的阶段总结
            "strategy": str,            # 可能含 backtrack（回溯）策略标签
            "update_rationale": str,
        }
        """
        # 1. 执行原有画像更新逻辑（透传 iteration_history 激活 backtrack/终止逻辑）
        # 若调用方未传 iteration_history，尝试从 profile dict 中读取（上游可携带）
        if iteration_history is None:
            profile_dict_preview = self._profile_to_dict(profile) if profile else {}
            iteration_history = profile_dict_preview.get("iteration_history") or []

        # 先推断情绪均分，供 BKT 的 We 修正因子使用（须在 update_user_data 之前）
        if emotion_score is None and qa_history:
            _emotion_scores = []
            for qa in qa_history:
                if isinstance(qa, dict) and "emotion_score" in qa:
                    try:
                        _emotion_scores.append(float(qa["emotion_score"]))
                    except (ValueError, TypeError):
                        pass
            if _emotion_scores:
                emotion_score = sum(_emotion_scores) / len(_emotion_scores)

        updated_data = await self.update_user_data(
            session_id, accuracy, profile, qa_history, interactions,
            iteration_history=iteration_history,
            emotion_score=emotion_score,
            gap_skill=gap_skill,
        )

        # 2. 准备总结生成所需的数据
        profile_dict = self._profile_to_dict(profile) if profile else {}
        
        # 从更新后的数据中提取关键指标
        knowledge_gaps = updated_data.get("knowledge_gaps", [])
        confusion_tags = updated_data.get("confusion_markers", [])
        pace_adjustment = updated_data.get("learning_rhythm", "medium")
        next_strategy = updated_data.get("strategy", "reinforce")
        backtrack_target = updated_data.get("backtrack_target")
        iteration_reset = updated_data.get("iteration_reset", False)

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

        # 4. 组装最终输出（与 StageDecisionOutput 核心字段对齐）
        # 补充 mastery 字段（= progress），保证与画像Schema一致
        gaps_with_mastery = []
        for gap in knowledge_gaps:
            gap_copy = dict(gap) if isinstance(gap, dict) else {}
            if "mastery" not in gap_copy:
                gap_copy["mastery"] = gap_copy.get("progress", 0.0)
            gaps_with_mastery.append(gap_copy)

        profile_updates = {
            "knowledge_gaps": gaps_with_mastery,
            "dynamic_state": {
                "confidence_index": updated_data.get("confidence_index", 0.5),
            },
            "pace_adjustment": pace_adjustment,
            "confusion_tags": confusion_tags,
            "next_strategy": next_strategy,
        }
        # 回溯策略时透传前置知识目标与迭代重置信号（iteration_strategy.skill.md 规则2）
        if next_strategy == "backtrack":
            profile_updates["backtrack_target"] = backtrack_target
            profile_updates["iteration_reset"] = iteration_reset

        # 决策判定（与 DeterministicStageDecisionAgent.run 逻辑一致）
        # 优先级：终止信号/工具权威 decision > backtrack > accuracy 推进
        # next_strategy 为 backtrack（回溯）时，decision 映射为 repeat（回退重学前置知识）
        # next_strategy 为 finish（结业）时，decision 映射为 completed
        # 达到最大轮次/死循环停滞会触发 should_terminate，此时尊重策略工具的终止裁决，
        # 不再仅用 accuracy 重算而丢弃规范的结业结果（iteration_strategy.skill.md 规则3）。
        should_terminate = bool(updated_data.get("should_terminate", False))
        tool_decision = updated_data.get("decision")
        all_mastered = all(
            gap.get("status") == "已掌握" for gap in knowledge_gaps if isinstance(gap, dict)
        ) if knowledge_gaps else False
        if next_strategy == "finish" or all_mastered or should_terminate:
            decision = tool_decision or "completed"
        elif next_strategy == "backtrack":
            decision = "repeat"
        elif accuracy >= 0.8:
            decision = "continue"
        else:
            decision = "repeat"

        return {
            "decision": decision,
            "profile_updates": profile_updates,
            "stage_summary": stage_summary,
            "strategy": next_strategy,
            "update_rationale": f"阶段{stage_name}完成：正确率{accuracy:.0%}，策略{next_strategy}，共{len(knowledge_gaps)}个知识点",
        }