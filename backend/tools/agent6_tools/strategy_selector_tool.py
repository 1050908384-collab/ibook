# -*- coding: utf-8 -*-
"""
strategy_selector_tool — Agent6 策略选择工具
==============================================
功能：输入正确率、当前画像、迭代历史，输出策略标签
（进阶/巩固/降维/回溯/结业）及决策理由。规则引擎，读取
iteration_strategy.skill.md 中的策略表做匹配，无 Mock。

统一接口：
    def strategy_selector(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

依赖：iteration_strategy.skill.md 规则1-3（五档策略表、方向切换、终止条件）。
保留 StrategySelectorTool 类作为向后兼容适配器。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# ------------------------------------------------------------------
# 策略阈值常量（iteration_strategy.skill.md 规则1）
# ------------------------------------------------------------------
ACCURACY_FINISH = 0.85       # ≥0.85 且 coverage≥0.90 且全部已掌握 → 结业
ACCURACY_ADVANCE = 0.80      # ≥0.80 且非全部已掌握 → 进阶
ACCURACY_REINFORCE_LOW = 0.60  # 0.60 ≤ accuracy < 0.80 → 巩固
# accuracy < 0.60 → 降维

COVERAGE_FINISH = 0.90       # 结业所需的覆盖率阈值

# 方向切换阈值（规则2）
BACKTRACK_IMPROVEMENT_THRESHOLD = 0.05  # 连续2轮提升<5% → 回溯
BACKTRACK_CONSECUTIVE_ROUNDS = 2        # 连续降维无效轮数

# 终止条件阈值（规则3）
MAX_ITERATIONS = 5                    # 最大迭代轮次
STAGNATION_IMPROVEMENT_THRESHOLD = 0.03  # 停滞判定：提升<3%
STAGNATION_CONSECUTIVE_ROUNDS = 2        # 连续停滞轮数

# 策略标签
STRATEGY_FINISH = "finish"
STRATEGY_ADVANCE = "advance"
STRATEGY_REINFORCE = "reinforce"
STRATEGY_SIMPLIFY = "simplify"
STRATEGY_BACKTRACK = "backtrack"

# 决策类型
DECISION_CONTINUE = "continue"
DECISION_REPEAT = "repeat"
DECISION_COMPLETED = "completed"


# ------------------------------------------------------------------
# 核心规则引擎
# ------------------------------------------------------------------
def _compute_coverage(profile: Dict[str, Any]) -> float:
    """计算知识覆盖率（已掌握gaps数/总gaps数）。"""
    knowledge_gaps = profile.get("knowledge_gaps", [])
    if not knowledge_gaps:
        return 1.0
    mastered = sum(1 for g in knowledge_gaps if g.get("status") == "已掌握")
    return mastered / len(knowledge_gaps)


def _check_all_mastered(profile: Dict[str, Any]) -> bool:
    """检查是否所有知识盲区均已掌握。"""
    knowledge_gaps = profile.get("knowledge_gaps", [])
    if not knowledge_gaps:
        return True
    return all(g.get("status") == "已掌握" for g in knowledge_gaps)


def _check_backtrack_trigger(iteration_history: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    检查是否触发方向切换（降维→回溯）。
    
    触发条件：当策略为"降维"且连续2轮降维后正确率提升 < 5%。
    返回 (是否触发, 原因描述)
    """
    if len(iteration_history) < 3:
        return False, ""
    
    # 找最近的降维记录
    simplify_records = []
    for i, record in enumerate(reversed(iteration_history)):
        strategy = record.get("strategy_used", "")
        if strategy == STRATEGY_SIMPLIFY:
            accuracy = record.get("accuracy", 0)
            simplify_records.append((len(iteration_history) - 1 - i, accuracy))
        elif strategy == STRATEGY_BACKTRACK:
            # 已触发回溯，不再检测
            return False, ""
        else:
            break
    
    if len(simplify_records) < BACKTRACK_CONSECUTIVE_ROUNDS + 1:
        return False, ""
    
    # 检查最近2次降维的提升幅度
    # simplify_records 由 reversed(iteration_history) 构建，顺序为「新→旧」（按轮次降序）；
    # 上方 break 已保证这些记录是「连续」的降维轮次。
    recent = simplify_records[:BACKTRACK_CONSECUTIVE_ROUNDS + 1]  # 3条记录（新→旧）
    consecutive_invalid = 0
    
    for i in range(len(recent) - 1):
        if i + 1 < len(recent):
            improvement = recent[i][1] - recent[i + 1][1]  # 新一轮 accuracy − 上一轮 accuracy = 提升幅度
            if improvement < BACKTRACK_IMPROVEMENT_THRESHOLD:
                consecutive_invalid += 1
    
    if consecutive_invalid >= BACKTRACK_CONSECUTIVE_ROUNDS:
        return True, f"连续{consecutive_invalid}轮降维正确率提升<{BACKTRACK_IMPROVEMENT_THRESHOLD:.0%}"
    
    return False, ""


def _check_termination(
    accuracy: float,
    coverage: float,
    all_mastered: bool,
    iteration_history: List[Dict[str, Any]],
    max_iterations: int,
) -> Tuple[bool, str]:
    """
    检查是否满足终止条件。
    
    返回 (should_terminate, reason)
    """
    current_iteration = len(iteration_history)
    reasons = []
    
    # 条件1：结业判定
    if all_mastered:
        return True, "所有知识盲区已掌握，触发结业"
    
    # 条件2：最大迭代次数
    if current_iteration >= max_iterations:
        reasons.append(f"达到最大迭代次数{max_iterations}")
    
    # 条件3：死循环检测
    if len(iteration_history) >= STAGNATION_CONSECUTIVE_ROUNDS + 1:
        recent = iteration_history[-(STAGNATION_CONSECUTIVE_ROUNDS + 1):]
        # 检查是否已触发过回溯
        backtrack_triggered = any(
            r.get("strategy_used") == STRATEGY_BACKTRACK for r in iteration_history
        )
        if backtrack_triggered:
            consecutive_stagnation = 0
            for i in range(len(recent) - 1):
                if i + 1 < len(recent):
                    improvement = recent[i + 1].get("accuracy", 0) - recent[i].get("accuracy", 0)
                    if improvement < STAGNATION_IMPROVEMENT_THRESHOLD:
                        consecutive_stagnation += 1
            if consecutive_stagnation >= STAGNATION_CONSECUTIVE_ROUNDS:
                reasons.append(
                    f"连续{STAGNATION_CONSECUTIVE_ROUNDS}轮改进<{STAGNATION_IMPROVEMENT_THRESHOLD:.0%}，判定为停滞"
                )
    
    if reasons:
        return True, "、".join(reasons)
    
    return False, ""


def _select_strategy(
    accuracy: float,
    coverage: float,
    all_mastered: bool,
    backtrack_triggered: bool,
) -> Tuple[str, str]:
    """
    根据五档策略表选择策略标签（规则1）。
    
    优先级：结业 > 回溯 > 进阶 > 巩固 > 降维
    """
    if all_mastered and coverage >= COVERAGE_FINISH and accuracy >= ACCURACY_FINISH:
        return STRATEGY_FINISH, f"accuracy={accuracy:.2f}≥{ACCURACY_FINISH}且coverage={coverage:.2f}≥{COVERAGE_FINISH}且全部已掌握"
    
    if backtrack_triggered:
        return STRATEGY_BACKTRACK, "降维连续无效，触发回溯"
    
    if accuracy >= ACCURACY_ADVANCE:
        return STRATEGY_ADVANCE, f"accuracy={accuracy:.2f}≥{ACCURACY_ADVANCE}，匹配进阶策略"
    
    if accuracy >= ACCURACY_REINFORCE_LOW:
        return STRATEGY_REINFORCE, f"accuracy={accuracy:.2f}∈[{ACCURACY_REINFORCE_LOW},{ACCURACY_ADVANCE})，匹配巩固策略"
    
    return STRATEGY_SIMPLIFY, f"accuracy={accuracy:.2f}<{ACCURACY_REINFORCE_LOW}，匹配降维策略"


def _compute_decision(strategy_label: str, should_terminate: bool) -> str:
    """根据策略标签和终止标志计算 decision。"""
    if should_terminate or strategy_label == STRATEGY_FINISH:
        return DECISION_COMPLETED
    if strategy_label == STRATEGY_ADVANCE:
        return DECISION_CONTINUE
    return DECISION_REPEAT


def _update_profile(
    profile: Dict[str, Any],
    strategy_label: str,
    accuracy: float,
) -> Dict[str, Any]:
    """根据策略更新画像中的知识盲区状态（规则1 掌握状态更新）。"""
    updated = {}
    updated["knowledge_gaps"] = []
    
    for gap in profile.get("knowledge_gaps", []):
        gap_copy = dict(gap)
        if accuracy >= ACCURACY_ADVANCE:
            gap_copy["status"] = "已掌握"
            gap_copy["progress"] = 1.0
        else:
            current_progress = gap_copy.get("progress", 0.0)
            gap_copy["status"] = "学习中" if accuracy >= ACCURACY_REINFORCE_LOW else "未掌握"
            gap_copy["progress"] = max(current_progress, accuracy)
        updated["knowledge_gaps"].append(gap_copy)
    
    # 更新 confidence_index
    old_confidence = profile.get("confidence_index", 0.5)
    updated["confidence_index"] = round(min(1.0, max(0.0, accuracy)), 3)
    
    # 版本号递增
    updated["version"] = profile.get("version", 0) + 1
    
    # 保留其他字段
    for key, value in profile.items():
        if key not in updated:
            updated[key] = value
    
    return updated


def _find_backtrack_target(profile: Dict[str, Any]) -> Optional[str]:
    """查找回溯目标（前置知识）。"""
    knowledge_gaps = profile.get("knowledge_gaps", [])
    for gap in knowledge_gaps:
        prerequisites = gap.get("prerequisites", [])
        if prerequisites:
            return prerequisites[0]
    return None


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def strategy_selector(input_data: dict, config: dict = None) -> dict:
    """
    根据正确率、覆盖率和迭代历史选择下一阶段策略。

    参数：
        input_data: dict
            - accuracy: float              本阶段正确率（0~1）
            - coverage: float              知识覆盖率（0~1，可选）
            - current_profile: dict        当前用户画像
            - iteration_history: list      迭代历史列表
            - max_iterations: int          最大迭代次数（默认5）
            - user_completed: bool         用户主动完成（可选）
        config: dict                       忽略，规则引擎无 Mock

    返回：
        {
            "status": "success" | "error",
            "data": {
                "strategy_label": "finish"|"advance"|"reinforce"|"simplify"|"backtrack",
                "reason": str,
                "should_terminate": bool,
                "decision": "continue"|"repeat"|"completed",
                "profile": {...},
                "backtrack_target": str|null,
                "iteration_reset": bool
            },
            "message": str,
            "mock_used": False
        }
    """
    _ = config or {}
    
    try:
        input_data = input_data or {}
        
        accuracy = input_data.get("accuracy")
        coverage = input_data.get("coverage")
        current_profile = input_data.get("current_profile") or {}
        iteration_history = input_data.get("iteration_history") or []
        max_iterations = input_data.get("max_iterations", MAX_ITERATIONS)
        user_completed = input_data.get("user_completed", False)
        
        # 数据完整性检查
        if accuracy is None:
            return {
                "status": "success",
                "data": {
                    "strategy_label": STRATEGY_REINFORCE,
                    "reason": "数据不完整（accuracy缺失），采用默认巩固策略",
                    "should_terminate": False,
                    "decision": DECISION_REPEAT,
                    "profile": current_profile,
                    "backtrack_target": None,
                    "iteration_reset": False,
                },
                "message": "数据不完整",
                "mock_used": False,
            }
        
        # 规范化 accuracy
        accuracy = min(1.0, max(0.0, float(accuracy)))
        
        # 计算覆盖率
        if coverage is None:
            coverage = _compute_coverage(current_profile)
        
        # 检查是否全部掌握
        all_mastered = _check_all_mastered(current_profile)
        
        # 用户主动完成
        if user_completed:
            updated_profile = _update_profile(current_profile, STRATEGY_FINISH, accuracy)
            return {
                "status": "success",
                "data": {
                    "strategy_label": STRATEGY_FINISH,
                    "reason": "学习者主动完成，触发结业",
                    "should_terminate": True,
                    "decision": DECISION_COMPLETED,
                    "profile": updated_profile,
                    "backtrack_target": None,
                    "iteration_reset": False,
                },
                "message": "用户主动完成学习",
                "mock_used": False,
            }
        
        # 答题记录不足
        if not iteration_history and accuracy == 0.0:
            updated_profile = _update_profile(current_profile, STRATEGY_SIMPLIFY, accuracy)
            return {
                "status": "success",
                "data": {
                    "strategy_label": STRATEGY_SIMPLIFY,
                    "reason": "答题记录不足，accuracy=0.0，匹配降维策略",
                    "should_terminate": False,
                    "decision": DECISION_REPEAT,
                    "profile": updated_profile,
                    "backtrack_target": None,
                    "iteration_reset": False,
                },
                "message": "降维策略（答题不足）",
                "mock_used": False,
            }
        
        # 检查方向切换
        backtrack_triggered, backtrack_reason = _check_backtrack_trigger(iteration_history)
        
        # 检查终止条件
        should_terminate, termination_reason = _check_termination(
            accuracy, coverage, all_mastered, iteration_history, max_iterations
        )
        
        # 选择策略
        strategy_label, strategy_reason = _select_strategy(
            accuracy, coverage, all_mastered, backtrack_triggered
        )
        
        # 组装理由
        reason_parts = [strategy_reason]
        if backtrack_triggered:
            reason_parts.append(f"方向切换：{backtrack_reason}")
        if should_terminate:
            reason_parts.append(f"终止条件：{termination_reason}")
        if coverage < COVERAGE_FINISH and accuracy >= ACCURACY_FINISH:
            reason_parts.append(f"覆盖率{coverage:.2f}<{COVERAGE_FINISH}，未满足结业条件")
        if strategy_label != STRATEGY_FINISH and not all_mastered:
            reason_parts.append("存在未掌握的知识盲区")
        
        reason = "。".join(reason_parts)
        
        # 计算 decision
        decision = _compute_decision(strategy_label, should_terminate)
        
        # 更新画像
        updated_profile = _update_profile(current_profile, strategy_label, accuracy)
        
        # 回溯目标
        backtrack_target = None
        iteration_reset = False
        if strategy_label == STRATEGY_BACKTRACK:
            backtrack_target = _find_backtrack_target(updated_profile)
            if backtrack_target is None:
                backtrack_target = "入门阶段"
            iteration_reset = True
        
        return {
            "status": "success",
            "data": {
                "strategy_label": strategy_label,
                "reason": reason,
                "should_terminate": should_terminate,
                "decision": decision,
                "profile": updated_profile,
                "backtrack_target": backtrack_target,
                "iteration_reset": iteration_reset,
            },
            "message": f"策略选择完成：{strategy_label}",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("strategy_selector 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"strategy_selector 执行异常: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 向后兼容适配器
# 保留原 StageDecisionAgent 调用的 async 类接口。
# ------------------------------------------------------------------
class StrategySelectorTool:
    def __init__(self):
        pass

    async def select_strategy(
        self,
        accuracy: float,
        profile: Dict[str, Any],
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        coverage: Optional[float] = None,
        max_iterations: int = MAX_ITERATIONS,
        user_completed: bool = False,
    ) -> str:
        """
        根据正确率、画像和迭代历史选择策略标签。

        委托给函数式 strategy_selector() 完整版，激活 iteration_strategy.skill.md
        规则2（降维→回溯方向切换）和规则3（最大轮次/死循环终止）。

        参数：
            accuracy: 本阶段正确率（0~1）
            profile: 当前用户画像
            iteration_history: 迭代历史列表（可选，用于 backtrack/终止检测）
            coverage: 知识覆盖率（可选，缺省时自动计算）
            max_iterations: 最大迭代次数（默认5）
            user_completed: 用户是否主动完成

        返回：策略标签字符串
            finish / advance / reinforce / simplify / backtrack
        """
        input_data = {
            "accuracy": accuracy,
            "current_profile": profile or {},
            "iteration_history": iteration_history or [],
            "max_iterations": max_iterations,
            "user_completed": user_completed,
        }
        if coverage is not None:
            input_data["coverage"] = coverage

        result = strategy_selector(input_data)
        if result.get("status") != "success" or not result.get("data"):
            # 异常兜底：回退到简化版阈值判定
            accuracy = min(1.0, max(0.0, float(accuracy)))
            knowledge_gaps = (profile or {}).get("knowledge_gaps", [])
            all_mastered = all(
                g.get("status") == "已掌握" for g in knowledge_gaps
            ) if knowledge_gaps else True
            if accuracy >= ACCURACY_FINISH and all_mastered:
                return STRATEGY_FINISH
            elif accuracy >= ACCURACY_ADVANCE:
                return STRATEGY_ADVANCE
            elif accuracy >= ACCURACY_REINFORCE_LOW:
                return STRATEGY_REINFORCE
            return STRATEGY_SIMPLIFY

        return result["data"].get("strategy_label", STRATEGY_REINFORCE)

    async def select_strategy_full(
        self,
        accuracy: float,
        profile: Dict[str, Any],
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        coverage: Optional[float] = None,
        max_iterations: int = MAX_ITERATIONS,
        user_completed: bool = False,
    ) -> Dict[str, Any]:
        """
        完整版策略选择，返回包含 strategy_label/decision/backtrack_target/should_terminate
        等全部字段的 dict。供需要回溯目标或终止信号的调用方使用。
        """
        input_data = {
            "accuracy": accuracy,
            "current_profile": profile or {},
            "iteration_history": iteration_history or [],
            "max_iterations": max_iterations,
            "user_completed": user_completed,
        }
        if coverage is not None:
            input_data["coverage"] = coverage

        result = strategy_selector(input_data)
        if result.get("status") != "success" or not result.get("data"):
            # 异常兜底
            label = await self.select_strategy(
                accuracy, profile, iteration_history, coverage, max_iterations, user_completed
            )
            return {
                "strategy_label": label,
                "reason": "函数式接口异常，回退简化判定",
                "should_terminate": False,
                "decision": DECISION_REPEAT,
                "backtrack_target": None,
                "iteration_reset": False,
            }
        return result["data"]