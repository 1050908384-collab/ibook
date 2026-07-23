# -*- coding: utf-8 -*-
"""
profile_update_tool — Agent6 画像更新工具
==========================================
功能：输入掌握度数据和策略标签，输出结构化画像更新字段
（knowledge_gaps、confidence_index、learning_rhythm、confusion_markers、next_strategy），
用于写入 user_profiles 表新版本。纯 Python 字段映射，无 Mock。

统一接口：
    def profile_update(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

依赖：user_model_update.skill.md 规则1-6（BKT模型、修正因子、状态映射、节奏推断、
信心指数、困惑标记）。
保留 ProfileUpdateTool 类作为向后兼容适配器。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# ------------------------------------------------------------------
# BKT 四参数常量（user_model_update.skill.md 规则1）
# ------------------------------------------------------------------
BKT_DEFAULT_P_T = 0.4       # 学习跃迁概率
BKT_DEFAULT_P_S = 0.1       # 粗心概率
BKT_P_G_BY_TYPE = {         # 猜对概率（按题型）
    "选择题": 0.25,
    "问答题": 0.1,
    "实操题": 0.05,
}
BKT_P_G_DEFAULT = 0.15      # 题型未知时的默认值
BKT_P_L0_HAS_EXPERIENCE = 0.6   # 有相似经验时的初始掌握概率
BKT_P_L0_NO_EXPERIENCE = 0.2    # 无经验时的初始掌握概率

# 状态映射阈值（规则3）
MASTERED_THRESHOLD = 0.8      # ≥0.8 → 已掌握
LEARNING_LOW = 0.4            # 0.4~0.8 → 学习中

# 时长修正因子 Wt（规则2）
W_FAST_THRESHOLD = 0.5        # <0.5倍 → Wt=0.7
W_SLOW_THRESHOLD = 2.0        # >2.0倍 → Wt=0.7
W_MILD_LOW = 0.5              # 0.5~0.8倍 → Wt=0.85
W_MILD_HIGH = 1.2             # 1.2~2.0倍 → Wt=0.85
W_NORMAL_LOW = 0.8            # 0.8~1.2倍 → Wt=1.0

# 情绪修正因子 We（规则2）
WE_POSITIVE_THRESHOLD = 0.5   # >0.5 → We=1.1
WE_NEGATIVE_THRESHOLD = -0.3  # <-0.3 → We=0.9

# 信心指数权重（规则5）
CONFIDENCE_ACCURACY_WEIGHT = 0.6
CONFIDENCE_EMOTION_WEIGHT = 0.4

# 困惑标记阈值（规则6）
CONFUSION_WRONG_THRESHOLD = 2   # 答错次数 ≥ 2
CONFUSION_QUESTION_THRESHOLD = 3  # 提问次数 ≥ 3


# ------------------------------------------------------------------
# BKT 模型核心计算
# ------------------------------------------------------------------
def _get_p_l0(skill: str, skill_tags: List[str]) -> float:
    """根据用户 skill_tags 判断初始掌握概率 P(L₀)。"""
    if not skill or not skill_tags:
        return BKT_P_L0_NO_EXPERIENCE
    skill_lower = skill.lower()
    for tag in skill_tags:
        if tag.lower() in skill_lower or skill_lower in tag.lower():
            return BKT_P_L0_HAS_EXPERIENCE
    return BKT_P_L0_NO_EXPERIENCE


def _get_p_g(question_type: str) -> float:
    """根据题型获取猜对概率 P(G)。"""
    return BKT_P_G_BY_TYPE.get(question_type, BKT_P_G_DEFAULT)


def _bkt_update(p_l: float, is_correct: bool, p_g: float, p_s: float, p_t: float) -> float:
    """
    BKT 单次更新公式（user_model_update.skill.md 规则1）。
    
    答对后：P(L|correct) = P(L)×(1-P(S)) / [P(L)×(1-P(S)) + (1-P(L))×P(G)]
    答错后：P(L|incorrect) = P(L)×P(S) / [P(L)×P(S) + (1-P(L))×(1-P(G))]
    学习跃迁：P(L)_new = P(L|response) + (1-P(L|response))×P(T)
    """
    try:
        if is_correct:
            numerator = p_l * (1.0 - p_s)
            denominator = p_l * (1.0 - p_s) + (1.0 - p_l) * p_g
        else:
            numerator = p_l * p_s
            denominator = p_l * p_s + (1.0 - p_l) * (1.0 - p_g)
        
        if denominator == 0:
            return p_l
        
        p_l_given_response = numerator / denominator
        p_l_new = p_l_given_response + (1.0 - p_l_given_response) * p_t
        return min(1.0, max(0.0, p_l_new))
    except (ZeroDivisionError, OverflowError):
        return p_l


def _compute_wt(avg_time_spent: float, expected_avg_time: float, sample_count: int) -> float:
    """计算时长修正因子 Wt（规则2）。"""
    if sample_count < 3:
        return 1.0
    if expected_avg_time <= 0:
        return 1.0
    ratio = avg_time_spent / expected_avg_time
    if ratio < W_FAST_THRESHOLD or ratio > W_SLOW_THRESHOLD:
        return 0.7
    if ratio < W_NORMAL_LOW or ratio > W_MILD_HIGH:
        return 0.85
    return 1.0


def _compute_we(emotion_score: Optional[float]) -> float:
    """计算情绪修正因子 We（规则2）。"""
    if emotion_score is None:
        return 1.0
    if emotion_score > WE_POSITIVE_THRESHOLD:
        return 1.1
    if emotion_score < WE_NEGATIVE_THRESHOLD:
        return 0.9
    return 1.0


def _compute_corrected_p(b_bkt: float, wt: float, we: float) -> float:
    """计算修正后的掌握概率 P_corrected = min(1.0, max(0.0, P_BKT × Wt × We))。"""
    return min(1.0, max(0.0, b_bkt * wt * we))


def _map_status(p_corrected: float) -> Tuple[str, float]:
    """将 P_corrected 映射为 status 和 progress（规则3）。"""
    if p_corrected >= MASTERED_THRESHOLD:
        return "已掌握", 1.0
    elif p_corrected >= LEARNING_LOW:
        return "学习中", round(p_corrected, 3)
    else:
        return "未掌握", round(p_corrected, 3)


def _infer_rhythm(actual_total_time: float, expected_total_time: float) -> str:
    """推断学习节奏（规则4）。"""
    if expected_total_time <= 0:
        return "medium"
    ratio = actual_total_time / expected_total_time
    if ratio < 0.8:
        return "fast"
    elif ratio > 1.2:
        return "slow"
    return "medium"


def _infer_confidence(accuracy: float, emotion_score: Optional[float]) -> float:
    """计算信心指数（规则5）。"""
    accuracy = min(1.0, max(0.0, accuracy))
    if emotion_score is None:
        emotion_normalized = 0.5
    else:
        emotion_normalized = (emotion_score + 1.0) / 2.0
    confidence = accuracy * CONFIDENCE_ACCURACY_WEIGHT + emotion_normalized * CONFIDENCE_EMOTION_WEIGHT
    return min(1.0, max(0.0, round(confidence, 3)))


def _detect_confusion(
    question_records: List[Dict[str, Any]],
    skill_questions: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    """生成困惑标记（规则6）。"""
    confusion_tags: List[str] = []
    for skill, records in skill_questions.items():
        wrong_count = sum(1 for r in records if not r.get("is_correct", False))
        question_count = sum(1 for r in records if r.get("is_question", False))
        
        if wrong_count >= CONFUSION_WRONG_THRESHOLD or question_count >= CONFUSION_QUESTION_THRESHOLD:
            detail = _extract_confusion_detail(skill, records)
            confusion_tags.append(detail)
    
    return confusion_tags


def _extract_confusion_detail(skill: str, records: List[Dict[str, Any]]) -> str:
    """从错题记录中提取困惑细节。"""
    wrong_skills = set()
    for r in records:
        if not r.get("is_correct", False):
            detail = r.get("concept_detail") or r.get("question_concept") or ""
            if detail:
                wrong_skills.add(detail)
    
    if wrong_skills:
        return f"{skill}（{', '.join(sorted(wrong_skills))}）"
    return skill


def _group_by_skill(question_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """将答题记录按知识点分组。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in question_records:
        skill = record.get("skill") or record.get("knowledge_point") or "unknown"
        if skill not in grouped:
            grouped[skill] = []
        grouped[skill].append(record)
    return grouped


def _compute_skill_bkt(
    skill: str,
    records: List[Dict[str, Any]],
    skill_tags: List[str],
    expected_avg_time: float,
) -> Tuple[float, float, float, int]:
    """
    计算单个知识点的 BKT 掌握概率和修正因子。
    返回：(P_corrected, Wt, We, sample_count)
    """
    p_l = _get_p_l0(skill, skill_tags)
    p_g = BKT_P_G_DEFAULT
    
    total_time = 0.0
    time_count = 0
    
    for record in records:
        question_type = record.get("question_type", "")
        p_g = _get_p_g(question_type)
        is_correct = record.get("is_correct", False)
        p_l = _bkt_update(p_l, is_correct, p_g, BKT_DEFAULT_P_S, BKT_DEFAULT_P_T)
        
        time_spent = record.get("time_spent")
        if time_spent is not None:
            try:
                total_time += float(time_spent)
                time_count += 1
            except (ValueError, TypeError):
                pass
    
    avg_time = total_time / time_count if time_count > 0 else 0.0
    wt = _compute_wt(avg_time, expected_avg_time, time_count)
    we = 1.0  # 情绪修正在外层统一处理
    
    return p_l, wt, we, time_count


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def profile_update(input_data: dict, config: dict = None) -> dict:
    """
    根据答题记录和情绪得分，计算 BKT 掌握概率并生成画像更新字段。

    参数：
        input_data: dict
            - question_records: list    答题记录列表
            - emotion_score: float      情绪均分（-1~+1）
            - current_profile: dict     当前用户画像
            - expected_avg_time: float  预期平均耗时（秒，可选）
            - accuracy: float           阶段正确率（可选，用于信心指数计算）
        config: dict                   忽略，纯计算无 Mock

    返回：
        {
            "status": "success" | "error",
            "data": {
                "profile_updates": {
                    "knowledge_gaps": [...],
                    "dynamic_state": {"mastery_probability", "confidence_index"},
                    "pace_adjustment": "fast"|"medium"|"slow",
                    "confusion_tags": [...],
                    "next_strategy": "reinforce"|"advance"|...
                },
                "update_rationale": str
            },
            "message": str,
            "mock_used": False
        }
    """
    _ = config or {}
    
    try:
        input_data = input_data or {}
        question_records = input_data.get("question_records") or []
        emotion_score = input_data.get("emotion_score")
        current_profile = input_data.get("current_profile") or {}
        expected_avg_time = input_data.get("expected_avg_time", 100.0)
        accuracy = input_data.get("accuracy", 0.0)
        
        skill_tags = current_profile.get("skill_tags", [])
        existing_gaps = current_profile.get("knowledge_gaps", [])
        
        if not question_records:
            return {
                "status": "success",
                "data": {
                    "profile_updates": {},
                    "update_rationale": "无答题记录，画像未更新",
                },
                "message": "无答题记录",
                "mock_used": False,
            }
        
        skill_groups = _group_by_skill(question_records)
        
        # 计算平均耗时
        all_times = []
        for r in question_records:
            t = r.get("time_spent")
            if t is not None:
                try:
                    all_times.append(float(t))
                except (ValueError, TypeError):
                    pass
        
        total_time_spent = sum(all_times)
        avg_time_spent = sum(all_times) / len(all_times) if all_times else 0.0
        sample_count = len(all_times)
        
        # 计算各知识点 BKT
        knowledge_gaps_updated: List[Dict[str, Any]] = []
        bkt_results: Dict[str, float] = {}
        wt_values: List[float] = []
        rationale_parts: List[str] = []
        
        for skill, records in skill_groups.items():
            p_bkt, wt, we_local, sc = _compute_skill_bkt(
                skill, records, skill_tags, expected_avg_time
            )
            
            bkt_results[skill] = p_bkt
            wt_values.append(wt)
            
            # 计算修正后的掌握概率
            we_global = _compute_we(emotion_score)
            p_corrected = _compute_corrected_p(p_bkt, wt, we_global)
            status, progress = _map_status(p_corrected)
            
            # 查找已有 gap 或新建
            existing_gap = None
            for g in existing_gaps:
                if g.get("skill") == skill:
                    existing_gap = g
                    break
            
            gap_entry = {
                "skill": skill,
                "level": existing_gap.get("level", "beginner") if existing_gap else "beginner",
                "status": status,
                "progress": progress,
                "prerequisites": existing_gap.get("prerequisites", []) if existing_gap else [],
            }
            knowledge_gaps_updated.append(gap_entry)
            
            rationale_parts.append(
                f"{skill}: P_BKT={p_bkt:.3f}, Wt={wt:.2f}, We={we_global:.1f}, "
                f"P_corrected={p_corrected:.3f}, status={status}"
            )
        
        # 计算整体掌握概率（取所有知识点的最大值）
        max_mastery = max(bkt_results.values()) if bkt_results else 0.0
        overall_mastery = round(max_mastery * _compute_we(emotion_score), 3)
        overall_mastery = min(1.0, max(0.0, overall_mastery))
        
        # 学习节奏
        total_expected_time = expected_avg_time * len(question_records)
        rhythm = _infer_rhythm(total_time_spent, total_expected_time)
        
        # 信心指数
        confidence = _infer_confidence(accuracy, emotion_score)
        
        # 困惑标记
        confusion_tags = _detect_confusion(question_records, skill_groups)
        
        # 下一策略建议（基于掌握概率）
        if overall_mastery >= 0.85:
            next_strategy = "finish"
        elif overall_mastery >= 0.80:
            next_strategy = "advance"
        elif overall_mastery >= 0.60:
            next_strategy = "reinforce"
        else:
            next_strategy = "simplify"
        
        # 组装理由
        wt_desc = f"Wt取{sum(wt_values)/len(wt_values):.2f}" if wt_values else "Wt=1.0"
        we_desc = f"We={_compute_we(emotion_score):.1f}"
        emotion_desc = "情绪数据缺失，We=1.0" if emotion_score is None else f"情绪={emotion_score}"
        
        if sample_count < 3:
            wt_desc += "（样本量不足<3，Wt采用fallback=1.0）"
        
        update_rationale = (
            f"BKT计算：{'; '.join(rationale_parts)}。"
            f"修正因子：{wt_desc}、{we_desc}（{emotion_desc}）。"
            f"P_corrected汇总={overall_mastery}，"
            f"节奏={rhythm}（耗时比={total_time_spent/total_expected_time:.1f}），"
            f"信心指数={confidence}，"
            f"困惑标记={confusion_tags}。"
            f"建议下一策略：{next_strategy}。"
        )
        
        return {
            "status": "success",
            "data": {
                "profile_updates": {
                    "knowledge_gaps": knowledge_gaps_updated,
                    "dynamic_state": {
                        "mastery_probability": overall_mastery,
                        "confidence_index": confidence,
                    },
                    "pace_adjustment": rhythm,
                    "confusion_tags": confusion_tags,
                    "next_strategy": next_strategy,
                },
                "update_rationale": update_rationale,
            },
            "message": f"画像更新完成：{len(knowledge_gaps_updated)} 个知识点，信心指数 {confidence}",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("profile_update 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"profile_update 执行异常: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 向后兼容适配器
# 保留原 StageDecisionAgent 调用的 async 类接口，委托给函数式核心。
# ------------------------------------------------------------------
class ProfileUpdateTool:
    def __init__(self):
        pass

    async def update_gaps(self, gaps: List[Dict[str, Any]], accuracy: float) -> List[Dict[str, Any]]:
        """
        更新知识点掌握状态（简化版）。
        基于 accuracy 直接映射（与 BKT 结果兼容）：
        - accuracy >= 0.8 → 已掌握, progress=1.0
        - accuracy < 0.8  → 部分掌握/学习中, progress=max(progress, accuracy)
        """
        updated = []
        for gap in gaps:
            gap_copy = dict(gap)
            if accuracy >= 0.8:
                gap_copy["status"] = "已掌握"
                gap_copy["progress"] = 1.0
            else:
                current_progress = gap_copy.get("progress", 0.0)
                gap_copy["status"] = "学习中" if accuracy >= 0.4 else "未掌握"
                gap_copy["progress"] = max(current_progress, accuracy)
            updated.append(gap_copy)
        return updated

    async def infer_rhythm(self, interactions: List[Dict[str, Any]]) -> str:
        """
        推断学习节奏。
        从 interactions 中提取耗时数据，计算实际/预期比值。
        """
        if not interactions:
            return "medium"
        
        total_time = 0.0
        expected_time = 0.0
        
        for interaction in interactions:
            time_spent = interaction.get("time_spent", 0)
            expected = interaction.get("expected_time", 100)
            try:
                total_time += float(time_spent)
                expected_time += float(expected)
            except (ValueError, TypeError):
                continue
        
        if expected_time <= 0:
            return "medium"
        
        ratio = total_time / expected_time
        if ratio < 0.8:
            return "fast"
        elif ratio > 1.2:
            return "slow"
        return "medium"

    async def infer_confidence(self, qa_history: List[Dict[str, Any]], accuracy: float) -> float:
        """
        计算信心指数。
        优先使用情绪加权公式；若 qa_history 中无情绪数据则回退到纯 accuracy。
        """
        accuracy = min(1.0, max(0.0, accuracy))
        
        emotion_scores = []
        for qa in qa_history:
            if isinstance(qa, dict) and "emotion_score" in qa:
                try:
                    emotion_scores.append(float(qa["emotion_score"]))
                except (ValueError, TypeError):
                    pass
        
        if emotion_scores:
            avg_emotion = sum(emotion_scores) / len(emotion_scores)
            return _infer_confidence(accuracy, avg_emotion)
        else:
            return _infer_confidence(accuracy, None)

    async def detect_confusion(self, qa_history: List[Dict[str, Any]]) -> List[str]:
        """
        检测困惑标记。
        统计每个知识点的答错次数和提问次数。
        """
        if not qa_history:
            return []
        
        skill_records: Dict[str, List[Dict[str, Any]]] = {}
        for qa in qa_history:
            if not isinstance(qa, dict):
                continue
            skill = qa.get("skill") or qa.get("knowledge_point") or "unknown"
            if skill not in skill_records:
                skill_records[skill] = []
            skill_records[skill].append(qa)
        
        confusion = []
        for skill, records in skill_records.items():
            wrong_count = sum(1 for r in records if not r.get("is_correct", False))
            question_count = sum(1 for r in records if r.get("is_question", False))
            
            if wrong_count >= CONFUSION_WRONG_THRESHOLD or question_count >= CONFUSION_QUESTION_THRESHOLD:
                confusion.append(skill)
        
        return confusion