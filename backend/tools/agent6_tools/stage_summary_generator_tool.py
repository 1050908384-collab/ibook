# -*- coding: utf-8 -*-
"""
stage_summary_generator_tool — Agent6 阶段总结报告生成工具
===========================================================
功能：输入正确率、画像更新数据、交互数据、策略标签，输出面向用户的
Markdown格式阶段学习总结报告（8个维度）。

统一接口：
    def stage_summary_generator(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

依赖：user_model_update.skill.md 规则10（阶段总结报告生成规则）。
保留 StageSummaryGeneratorTool 类作为向后兼容适配器。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ------------------------------------------------------------------
# 策略标签中文映射
# ------------------------------------------------------------------
STRATEGY_LABEL_CN = {
    "finish": "结业",
    "advance": "进阶",
    "reinforce": "巩固",
    "simplify": "降维",
    "backtrack": "回溯",
}

# 策略说明映射（规则10.2 维度7）
STRATEGY_DESCRIPTIONS = {
    "finish": "恭喜你完成全部学习目标！🎉 你可以导出学习报告，或进入进阶模块继续学习。",
    "advance": "下一阶段策略：【进阶】。你已掌握本阶段内容，将推送难度更高的任务，挑战一下自己吧！",
    "reinforce": "下一阶段策略：【巩固】。你已掌握大部分内容，但仍有提升空间，将推送同难度补充练习帮助你巩固。",
    "simplify": "下一阶段策略：【降维】。部分知识点掌握度低于60%，将从更基础的版本重新讲解，帮你打下坚实基础。",
    "backtrack": "下一阶段策略：【回溯】。降维学习效果不佳，建议回到前置知识模块重新学习。",
}


# ------------------------------------------------------------------
# 正确率定性评价
# ------------------------------------------------------------------
def _evaluate_accuracy(accuracy: float) -> str:
    """根据正确率返回定性评价文字。"""
    if accuracy >= 0.85:
        return "表现优秀！说明你对本阶段内容掌握得非常好！"
    elif accuracy >= 0.70:
        return "表现良好！大部分内容已掌握，建议针对薄弱点进行巩固。"
    elif accuracy >= 0.60:
        return "表现一般，有几个知识点掌握不够扎实，建议重新学习。"
    else:
        return "本阶段正确率较低，建议从基础内容重新学习，不要急于进入下一阶段。"


# ------------------------------------------------------------------
# 情绪状态评价
# ------------------------------------------------------------------
def _evaluate_emotion(emotion_score: Optional[float]) -> str:
    """根据情绪得分返回评价文字。"""
    if emotion_score is None:
        return "学习过程中未检测到明显情绪特征。保持专注继续加油！"
    
    score = round(emotion_score, 2)
    if score >= 0.5:
        return f"学习过程中你表现积极（情绪得分{score}），保持好心态继续加油！😊"
    elif score >= 0:
        return f"学习过程中你的情绪偏中性（情绪得分{score}），偶尔有疑惑。遇到不懂的别担心，及时提问就好。💪"
    else:
        return f"学习过程中你有些挫败感（情绪得分{score}）。别灰心！学习本身就有挑战，适当休息后再继续效果会更好。🌟"


# ------------------------------------------------------------------
# 学习节奏评价
# ------------------------------------------------------------------
def _evaluate_pace(pace_adjustment: str, time_diff_hours: float = 0) -> str:
    """根据节奏标签返回评价文字。"""
    if pace_adjustment == "fast":
        extra = f"（用时比预期少了约{abs(time_diff_hours):.1f}小时）" if time_diff_hours > 0 else ""
        return f"本阶段你用时比预期快了{extra}，学得有点快。建议放慢节奏，确保每个知识点都能理解透彻再进入下一阶段。"
    elif pace_adjustment == "slow":
        extra = f"（用时比预期多了约{abs(time_diff_hours):.1f}小时）" if time_diff_hours > 0 else ""
        return f"本阶段你用时比预期多了{extra}，可能是内容难度较大。如有需要可以使用'降维'策略重新学习。"
    else:
        return "学习节奏很合适，继续保持这样的进度。"


# ------------------------------------------------------------------
# 知识点掌握情况分组
# ------------------------------------------------------------------
def _group_knowledge_status(knowledge_gaps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """将知识点按掌握状态分组。"""
    grouped = {
        "mastered": [],
        "learning": [],
        "not_mastered": [],
    }
    
    for gap in knowledge_gaps:
        skill = gap.get("skill", gap.get("topic", "未知知识点"))
        status = gap.get("status", "")
        
        if status == "已掌握":
            grouped["mastered"].append(skill)
        elif status == "学习中":
            grouped["learning"].append(skill)
        elif status == "未掌握":
            grouped["not_mastered"].append(skill)
    
    return grouped


# ------------------------------------------------------------------
# 知识盲区分析
# ------------------------------------------------------------------
def _analyze_knowledge_blind_spots(
    knowledge_gaps: List[Dict[str, Any]],
    confusion_tags: List[str],
    question_records: List[Dict[str, Any]] = None,
) -> str:
    """生成知识盲区分析文字。"""
    not_mastered = [
        gap for gap in knowledge_gaps
        if gap.get("status") == "未掌握"
    ]
    
    parts = []
    
    # 分析连续答错次数
    if not_mastered and question_records:
        for gap in not_mastered:
            skill = gap.get("skill", "")
            wrong_count = sum(
                1 for r in question_records
                if r.get("skill") == skill and not r.get("is_correct", False)
            )
            if wrong_count >= 2:
                parts.append(f"你在【{skill}】上连续答错{wrong_count}次，建议重新学习该知识点。")
            elif wrong_count > 0:
                parts.append(f"你在【{skill}】上答错{wrong_count}次，需要加强练习。")
    
    # 分析困惑标签
    if confusion_tags:
        tags_str = "、".join(confusion_tags[:3])  # 最多3个
        parts.append(f"我们注意到你对【{tags_str}】经常提问或出错，可能需要额外练习。")
    
    if not parts:
        return "本阶段没有明显的知识盲区，继续保持！"
    
    return " ".join(parts)


# ------------------------------------------------------------------
# 鼓励与建议
# ------------------------------------------------------------------
def _generate_encouragement(
    stage_name: str,
    accuracy: float,
    emotion_score: Optional[float],
    knowledge_gaps: List[Dict[str, Any]],
) -> str:
    """生成鼓励与建议文字。"""
    parts = [f"你已经完成了【{stage_name}】阶段的学习，离目标又近了一步！💪"]
    
    not_mastered = [
        gap for gap in knowledge_gaps
        if gap.get("status") == "未掌握"
    ]
    
    if accuracy < 0.60:
        parts.append("不用气馁，学习是一个反复打磨的过程。调整好状态，我们重新来过。")
    
    if emotion_score is not None and emotion_score < 0:
        parts.append("记得适当休息，学习效率比时长更重要。休息好了我们再继续！")
    
    if not_mastered:
        topics = [g.get("skill", "") for g in not_mastered[:2]]  # 最多2个
        topics_str = "、".join(topics)
        parts.append(f"建议在进入下一阶段前，先回顾一下【{topics_str}】的内容。")
    
    parts.append("有任何问题随时提问，你的专属AI导师始终在这里陪你！")
    
    return " ".join(parts)


# ------------------------------------------------------------------
# 统一函数式接口
# ------------------------------------------------------------------
def stage_summary_generator(input_data: dict, config: dict = None) -> dict:
    """
    生成阶段学习总结报告（Markdown格式）。

    参数：
        input_data: dict
            - stage_name: str           阶段名称
            - accuracy: float           阶段正确率（0~1）
            - knowledge_gaps: list      知识点掌握状态列表
            - confusion_tags: list      困惑标签列表
            - question_records: list    答题记录列表
            - emotion_score: float      情绪得分（-1~+1）
            - emotion_state: str        情绪状态描述
            - pace_adjustment: str      学习节奏（fast/medium/slow）
            - time_spent_hours: float   本阶段实际用时（小时）
            - expected_time_hours: float 预期用时（小时）
            - next_strategy: str        下一阶段策略标签
            - total_knowledge_points: int 本阶段知识点总数
            - total_answers: int        本阶段答题总数
        config: dict                   忽略，纯计算无 Mock

    返回：
        {
            "status": "success" | "error",
            "data": {
                "stage_summary": str,       # Markdown格式的总结报告
                "summary_length": int,      # 报告字数
                "dimensions_covered": int  # 覆盖的维度数
            },
            "message": str,
            "mock_used": False
        }
    """
    _ = config or {}
    
    try:
        input_data = input_data or {}
        
        stage_name = input_data.get("stage_name", "当前学习")
        accuracy = float(input_data.get("accuracy", 0.0))
        knowledge_gaps = input_data.get("knowledge_gaps", []) or []
        confusion_tags = input_data.get("confusion_tags", []) or []
        question_records = input_data.get("question_records", []) or []
        emotion_score = input_data.get("emotion_score")
        emotion_state = input_data.get("emotion_state", "")
        pace_adjustment = input_data.get("pace_adjustment", "medium")
        time_spent_hours = float(input_data.get("time_spent_hours", 0))
        expected_time_hours = float(input_data.get("expected_time_hours", 0))
        next_strategy = input_data.get("next_strategy", "reinforce")
        total_knowledge_points = int(input_data.get("total_knowledge_points", len(knowledge_gaps)))
        total_answers = int(input_data.get("total_answers", len(question_records)))
        
        # 计算时间差
        time_diff_hours = abs(time_spent_hours - expected_time_hours) if expected_time_hours > 0 else 0
        
        # 计算正确率百分比
        accuracy_pct = round(accuracy * 100)
        
        # 维度1：阶段概览
        overview = f"【{stage_name}】阶段已结束，你完成了{total_knowledge_points}个知识点的学习，共答题{total_answers}道"
        if time_spent_hours > 0:
            overview += f"，用时约{time_spent_hours:.1f}小时"
        overview += "。"
        
        # 维度2：正确率评价
        accuracy_eval = _evaluate_accuracy(accuracy)
        accuracy_section = f"整体正确率：{accuracy_pct}%——{accuracy_eval}"
        if accuracy < 0.60:
            not_mastered_skills = [
                g.get("skill", "") for g in knowledge_gaps
                if g.get("status") == "未掌握"
            ]
            if not_mastered_skills:
                accuracy_section += f"\n\n建议优先回顾以下知识点：{'、'.join(not_mastered_skills[:3])}"
        
        # 维度3：知识点掌握情况
        grouped = _group_knowledge_status(knowledge_gaps)
        knowledge_lines = []
        if grouped["mastered"]:
            knowledge_lines.append(f"- ✅ 已掌握：{'、'.join(grouped['mastered'])}")
        if grouped["learning"]:
            knowledge_lines.append(f"- 🔄 学习中：{'、'.join(grouped['learning'])}")
        if grouped["not_mastered"]:
            knowledge_lines.append(f"- ❌ 待加强：{'、'.join(grouped['not_mastered'])}")
        knowledge_section = "\n".join(knowledge_lines) if knowledge_lines else "- 📝 暂无知识点记录"
        
        # 维度4：知识盲区分析
        blind_spot_analysis = _analyze_knowledge_blind_spots(
            knowledge_gaps, confusion_tags, question_records
        )
        
        # 维度5：学习节奏评价
        pace_section = _evaluate_pace(pace_adjustment, time_diff_hours)
        
        # 维度6：情绪状态
        emotion_section = _evaluate_emotion(emotion_score)
        
        # 维度7：下一阶段策略
        strategy_cn = STRATEGY_LABEL_CN.get(next_strategy, next_strategy)
        strategy_desc = STRATEGY_DESCRIPTIONS.get(next_strategy, f"下一阶段策略：【{strategy_cn}】。")
        strategy_section = strategy_desc
        
        # 维度8：鼓励与建议
        encouragement = _generate_encouragement(
            stage_name, accuracy, emotion_score, knowledge_gaps
        )
        
        # 组装完整报告
        report = f"""## 📊 本阶段学习总结

### 📈 阶段概览
{overview}

### ✅ 正确率
{accuracy_section}

### 📚 知识点掌握情况
{knowledge_section}

### 🔍 知识盲区分析
{blind_spot_analysis}

### ⏱️ 学习节奏评价
{pace_section}

### 😊 情绪状态
{emotion_section}

### 🎯 下一阶段策略
{strategy_section}

### 💪 鼓励与建议
{encouragement}"""
        
        # 统计字数和覆盖维度数
        char_count = len(report)
        dimensions_covered = 8  # 始终覆盖全部8个维度
        
        return {
            "status": "success",
            "data": {
                "stage_summary": report,
                "summary_length": char_count,
                "dimensions_covered": dimensions_covered,
            },
            "message": f"阶段总结报告生成完成：{char_count}字，覆盖{dimensions_covered}个维度",
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("stage_summary_generator 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"stage_summary_generator 执行异常: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 向后兼容适配器
# ------------------------------------------------------------------
class StageSummaryGeneratorTool:
    def __init__(self):
        pass

    async def generate_summary(
        self,
        stage_name: str,
        accuracy: float,
        knowledge_gaps: List[Dict[str, Any]],
        confusion_tags: List[str],
        question_records: List[Dict[str, Any]],
        emotion_score: Optional[float],
        pace_adjustment: str,
        time_spent_hours: float,
        expected_time_hours: float,
        next_strategy: str,
        total_knowledge_points: int = 0,
        total_answers: int = 0,
        emotion_state: str = "",
    ) -> str:
        """
        异步接口：生成阶段总结报告。

        返回Markdown格式的总结报告字符串。
        """
        input_data = {
            "stage_name": stage_name,
            "accuracy": accuracy,
            "knowledge_gaps": knowledge_gaps,
            "confusion_tags": confusion_tags,
            "question_records": question_records,
            "emotion_score": emotion_score,
            "emotion_state": emotion_state,
            "pace_adjustment": pace_adjustment,
            "time_spent_hours": time_spent_hours,
            "expected_time_hours": expected_time_hours,
            "next_strategy": next_strategy,
            "total_knowledge_points": total_knowledge_points or len(knowledge_gaps),
            "total_answers": total_answers or len(question_records),
        }

        result = stage_summary_generator(input_data)
        if result.get("status") != "success":
            return _generate_fallback_report(stage_name, accuracy)
        return result["data"]["stage_summary"]


def _generate_fallback_report(stage_name: str, accuracy: float) -> str:
    """生成降级报告（当正式报告生成失败时）。"""
    accuracy_pct = round(accuracy * 100)
    eval_text = _evaluate_accuracy(accuracy)
    
    return f"""## 📊 本阶段学习总结

### 📈 阶段概览
【{stage_name}】阶段已结束。

### ✅ 正确率
整体正确率：{accuracy_pct}%——{eval_text}

### 💪 鼓励与建议
继续努力，你的专属AI导师始终在这里陪你！"""
