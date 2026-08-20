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

# 策略"决策理由"——解释「为什么」给这个策略，替代机械播报
STRATEGY_RATIONALE = {
    "finish": lambda ctx: "所有阶段目标已闭环，已具备独立完成真实任务的基础能力。",
    "advance": lambda ctx: (
        f"这次 {ctx.get('accuracy_pct', 0)}% 正确率，"
        f"且{ctx.get('_mastered_tip', '核心练习')}达标，"
        "具备带着现有能力处理更复杂场景的条件。"
    ),
    "reinforce": lambda ctx: (
        "现在不能直接推进——"
        f"{ctx.get('_not_mastered_tip', '有部分知识点尚未稳定达标')}，"
        "先补齐再往后走，避免越学越吃力。"
    ),
    "simplify": lambda ctx: (
        f"这次正确率仅 {ctx.get('accuracy_pct', 0)}%，"
        "说明当前讲解深度的跨度偏大。从更基础版本重讲是打地基，不是否定进度。"
    ),
    "backtrack": lambda ctx: (
        "连续两轮降维后正确率仍偏低，根因在前置知识断层。"
        "先回到前置模块打底，等下次再回来时会顺很多。"
    ),
}

# 策略"下一阶段必学提示"——告诉学员「接下来具体学什么、为什么是它」
STRATEGY_NEXT_STEP_TIP = {
    "finish": "🎉 现在可导出完整学习报告存档；之后想挑战更高目标，新建更高阶的学习项目即可。",
    "advance": "接下来的重点：在【{stage_name}】之上，引入真实边界条件与多步骤组合场景——这一步是从「看懂」到「会用」的关键门槛。",
    "reinforce": "接下来的重点：回到【{_reinforce_focus}】做补充练习，把卡住的 1-2 个点先打通，比推进新内容更划算。",
    "simplify": "接下来的重点：从【{_simplify_focus}】的基础版重新学起。步子会拆得更小、例子更多、提示更明确，帮你把之前模糊的地方逐个厘清。",
    "backtrack": "接下来的重点：先回到前置模块【{_backtrack_focus}】，把概念框架搭起来。等前置掌握了，再回到当前模块会快很多。",
}


def _pick_context_hints(
    knowledge_gaps: List[Dict[str, Any]], accuracy: float
) -> Dict[str, str]:
    """根据掌握情况生成几段上下文提示，用于策略文案的个性化填充。"""
    mastered = [g.get("skill", "核心知识点") for g in knowledge_gaps if g.get("status") == "已掌握"]
    learning = [g.get("skill", "") for g in knowledge_gaps if g.get("status") == "学习中"]
    not_mastered = [g.get("skill", "") for g in knowledge_gaps if g.get("status") == "未掌握"]

    hints: Dict[str, str] = {}

    if mastered:
        # 最多列2个，再多用「等」字收束
        if len(mastered) <= 2:
            hints["_mastered_tip"] = "「" + "、".join(mastered) + "」"
        else:
            hints["_mastered_tip"] = "「" + "、".join(mastered[:2]) + f"」等{len(mastered)}项"
    else:
        hints["_mastered_tip"] = "关键练习"

    if not_mastered:
        hints["_not_mastered_tip"] = (
            "「" + "、".join(not_mastered[:2])
            + ("」等" + str(len(not_mastered)) + "个知识点仍未稳定达标" if len(not_mastered) > 1 else "」这块还没稳定达标")
        )
        hints["_reinforce_focus"] = "、".join(not_mastered[:2])
        hints["_simplify_focus"] = "、".join(not_mastered[:2])
    elif learning:
        hints["_not_mastered_tip"] = "「" + "、".join(learning[:2]) + "」还处在学习中，尚未通过测试闭环"
        hints["_reinforce_focus"] = "、".join(learning[:2])
        hints["_simplify_focus"] = "、".join(learning[:2])
    else:
        hints["_not_mastered_tip"] = "测试覆盖的知识点数量偏少，当前结论的置信度不足"
        hints["_reinforce_focus"] = "本阶段核心知识点"
        hints["_simplify_focus"] = "本阶段核心知识点"

    hints["_backtrack_focus"] = hints.get("_reinforce_focus", "前置基础")
    return hints


# ------------------------------------------------------------------
# 正确率定性评价 — 从「贴标签」升级为「价值解读」
#   差评：整体正确率：100%——表现优秀！说明你对本阶段内容掌握得非常好！
#   好评（100%，2题）：虽然本轮只检测了2道题，但这2道题刚好覆盖了阶段的核心考核点，你全部避开了常见坑，说明基础动作很扎实。100%只代表「本轮没有暴露盲区」，不代表可以停止练习——下一阶段会在这个基础上加大变式难度。
# ------------------------------------------------------------------
def _evaluate_accuracy(accuracy: float, total_answers: int, total_knowledge_points: int) -> str:
    """正确率评价：每档给出「结论 + 现象归因 + 用户能直接用的判断标准」。"""
    accuracy_pct = round(accuracy * 100)

    # 题量极少时，主动暴露「样本量不足」的置信度问题
    small_sample = total_answers <= 2
    few_points = total_knowledge_points <= 2

    if accuracy >= 0.95:
        if small_sample:
            return (
                f"正确率 {accuracy_pct}%（{total_answers} 题全对）。"
                f"这 {total_answers} 题覆盖了 {total_knowledge_points} 个核心考核点，基础动作过关。"
                f"不过样本量偏少，100% 只代表「本轮没暴露盲区」——"
                "判断真实掌握度的标准是「能不能做对变式题」，后面会用变式题继续验证。"
            )
        return (
            f"正确率 {accuracy_pct}%，稳定。"
            "从错题分布看，你对关键概念不是停留在「眼熟」，而是能稳定调用——"
            "这是后面处理组合场景的底气。"
            "给你一个自测标准：如果能不看讲义把分散的点串成一条完整链路，就可以放心推进。"
        )
    elif accuracy >= 0.85:
        return (
            f"正确率 {accuracy_pct}%，达标。"
            "错题通常集中在细节（边界条件/步骤顺序），而不是「完全不会」——"
            "这种题丢分最可惜，因为补起来最快。"
            "建议：每次做新内容前花 5 分钟，专门回看 2 个最近的细节类错题即可。"
        )
    elif accuracy >= 0.70:
        return (
            f"正确率 {accuracy_pct}%，处于「会但不稳」区间。"
            "问题一般两类：要么概念边界易混，要么关键步骤差一步。"
            "这种状态最忌讳直接冲进度——表面 70 分，后面叠加新内容后会跌到 50 分以下。"
            "判断拐点是否出现的标准：同一类题连续做对 2 道变式，才算真打通。"
        )
    elif accuracy >= 0.60:
        return (
            f"正确率 {accuracy_pct}%，刚过及格线，危险。"
            f"按经验推算，这 {total_knowledge_points} 个知识点里至少有 3-4 个处于「蒙对/半懂」。"
            "这里不能赌运气——半懂的点不打通，后面每次用到它都会翻车。"
            "动作：先把半懂的点用 1-2 个补充例题跑通，再谈推进。"
        )
    else:
        if few_points:
            return (
                f"正确率 {accuracy_pct}%，基础概念还没建立。"
                "别紧张——知识点少时，通常只要卡住的 1-2 个概念打通，正确率会直接跳 30% 以上。"
                "后续会换更基础的讲法重讲一遍，步子拆小后会明显顺。"
            )
        return (
            f"正确率 {accuracy_pct}%，需要重新打底。"
            "更可能的原因不是「你学不会」，而是前置知识有断层，或者讲法不匹配你的节奏。"
            "后续会从更基础的版本、更小的步子重新讲起——先拿到「我能做对」的体验，再慢慢加量。"
        )


# ------------------------------------------------------------------
# 情绪状态评价 — 中性状态直接跳过展示，不占位
#   差评：学习过程中未检测到明显情绪特征。保持专注继续加油！
#   好评（无情绪数据）：该维度不输出任何内容，由上游空字符串判定跳过
# ------------------------------------------------------------------
def _evaluate_emotion(emotion_score: Optional[float]) -> str:
    """根据情绪得分返回评价文字；无情绪时输出中性说明（保证8个section不缺）。"""
    if emotion_score is None:
        return (
            "未检测到明显的情绪波动——这是一种比较稳的状态，说明内容难度和吸收节奏基本匹配。"
            "如果后面遇到卡点或感觉顺得不正常，随时反馈，我可以据此微调后面的讲解深度。"
        )

    score = round(emotion_score, 2)
    if score >= 0.5:
        return (
            f"学习状态较积极（情绪得分 {score}），这种状态下吸收新知效率最高。"
            "建议：把「能轻松拿下」的内容快速过掉，把更多精力留给会卡顿的点——"
            "趁着状态好啃硬骨头，是提效最明显的方式。😊"
        )
    elif score >= 0:
        return (
            f"情绪总体偏中性（得分 {score}），偶尔出现卡顿或回头翻讲义——这是正常的吸收过程。"
            "给你一个判断标准：如果同一道错题错了第2次，就停下来，不要硬做新题；"
            "这通常代表背后有没打通的概念，而不是你「粗心」。💪"
        )
    else:
        return (
            f"学习中有一些挫败感（得分 {score}）。"
            "通常不是能力问题，更可能是题目难度跨度或讲解风格不匹配。"
            "现在最不划算的是硬扛——先停 5-10 分钟，或者回到上一个你能稳定做对的练习，"
            "从「我能搞定」的小成功里重新拿节奏。🌟"
        )


# ------------------------------------------------------------------
# 学习节奏评价 — 从一句「学得快/慢」升级为「明确风险 + 具体动作」
#   差评：学得有点快。建议放慢节奏，确保每个知识点都能理解透彻再进入下一阶段。
#   好评：⚠️ 节奏预警：本阶段比预期快了约 2.0 小时。更可能的原因不是「你掌握得快」，而是「教程看得粗、测试题量偏少」——在 100% 正确率下尤其要注意这个陷阱。建议在进入下一阶段前，抽 5 分钟做「随机变式自测」3 道（薄弱点突击功能），如果仍然全对，再放心推进。
# ------------------------------------------------------------------
def _evaluate_pace(
    pace_adjustment: str,
    time_diff_hours: float = 0,
    accuracy: float = 1.0,
) -> str:
    """节奏评价：给出「信号解读 + 为什么危险/没问题 + 明确动作（含阈值）」。"""
    if pace_adjustment == "fast":
        if time_diff_hours > 0:
            head = f"⚠️ 节奏预警：比预期快 {abs(time_diff_hours):.1f}h。"
        else:
            head = "⚠️ 节奏预警：节奏偏快。"

        if accuracy >= 0.95:
            return (
                head
                + "高正确率下偏快 ≠「学得快」——更可能「教程看得粗/测试题量少」。"
                "这是最常见的隐性盲区来源：当时觉得懂了，一周后遇到变式就掉链子。"
                "行动：进入下一轮前，先用「薄弱点突击」随机抽 3 道变式题自测；"
                "判断标准很简单——全对再推进，只要错 1 道就立刻回头补对应讲解。"
            )
        elif accuracy >= 0.70:
            return (
                head
                + "当前错误多属细节类失误，说明概念本身能理解，但输入信息的颗粒度不够。"
                "动作：放慢 10% 节奏、每学完一个小节停 30 秒用自己的话复述一次，"
                "正确率通常能再抬一截，而且后面不需要返工，整体更省时间。"
            )
        else:
            return (
                head
                + "正确率也偏低，组合信号很明确——跳过讲解直接刷题，导致基础概念连框架都没建立。"
                "行动：先把教程里「常见坑/注意事项」段从头读一遍，读完再做 2 道最基础的概念题，"
                "能稳定做对了，再开始做进阶题。"
            )

    elif pace_adjustment == "slow":
        if time_diff_hours > 0:
            head = f"ℹ️ 节奏提示：比预期慢 {abs(time_diff_hours):.1f}h。"
        else:
            head = "ℹ️ 节奏提示：节奏偏慢。"

        if accuracy >= 0.85:
            return (
                head
                + "属于「稳扎稳打」型：慢换来了正确率和扎实基础，后面反而会越学越快。"
                "给你一个判断标准：单个知识点反复纠结超过 15 分钟时，先标「待回看」继续推进，"
                "很多卡点在学完关联内容后会自然解开，没必要在单点死磕。"
            )
        elif accuracy >= 0.60:
            return (
                head
                + "慢 + 及格：大概率卡在某几个关键概念上，而不是整体节奏问题。"
                "这种场景硬扛最不划算——与其在原地纠结 30 分钟，"
                "不如切到对应薄弱点的基础版讲解，或直接执行「降维」重学，卡点一般 5 分钟能解开。"
            )
        else:
            return (
                head
                + "投入多、效果低：大概率是讲法不匹配（或者前置概念缺失），不是你学得不认真。"
                "后续会自动切到「降维」，用更小的步子、更具象的例子重讲一遍。"
                "先停在当前位置，不要往后面继续翻——硬推进只会越来越挫败。"
            )

    else:
        return (
            "✓ 节奏合适：既不囫囵吞枣，也不在单点过度纠结。"
            "这是吸收效率最高的状态——保持，后面遇到卡点再微调即可。"
        )


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
# 知识盲区分析 — 占位文案改为「无盲区≠没问题」的审慎表述
#   差评：本阶段没有明显的知识盲区，继续保持！
#   好评（100% + 0盲区）：本轮没有暴露可定位的知识盲区。但需要注意：0盲区更常见的解释是「本轮测试尚未覆盖到边界条件/组合场景」，而不是「所有变式都已掌握」。下一阶段我们会主动引入更复杂的组合题目，用更难的样本去继续找盲区。
# ------------------------------------------------------------------
def _analyze_knowledge_blind_spots(
    knowledge_gaps: List[Dict[str, Any]],
    confusion_tags: List[str],
    question_records: List[Dict[str, Any]] = None,
    total_answers: int = 0,
    accuracy: float = 1.0,
) -> str:
    """盲区分析：对每类错题给出「原因层级定位 + 针对性动作（明确做到什么程度算完成）」。"""
    not_mastered = [
        gap for gap in knowledge_gaps
        if gap.get("status") == "未掌握"
    ]

    parts = []

    # 每档原因层级 + 动作完成标准
    if not_mastered and question_records:
        for gap in not_mastered:
            skill = gap.get("skill", "")
            wrong_count = sum(
                1 for r in question_records
                if r.get("skill") == skill and not r.get("is_correct", False)
            )
            if wrong_count >= 3:
                parts.append(
                    f"📌 【{skill}】连续错 {wrong_count} 次：这不是粗心，是底层概念框架还没搭起来。"
                    "「能看答案看懂」≠「自己能推导出来」。"
                    "动作完成标准：回到该知识点的「概念引入」段，读完合上书，能用自己的话不看讲义复述核心逻辑，才算过。"
                )
            elif wrong_count >= 2:
                parts.append(
                    f"📌 【{skill}】错 {wrong_count} 次：说明主流程能走通，但边界条件/概念边界的某个分支理解有偏差。"
                    "这种点不需要整段重学——精准定位卡点就行。"
                    "动作完成标准：对照错题，一步一步标出具体错在第几步/哪个分支，把那一步改正，同类题再做 2 道变式全对，就算过。"
                )
            elif wrong_count > 0:
                parts.append(
                    f"· 【{skill}】错 1 次：大概率是参数名/边界条件/格式这类单点失误，不算概念盲区。"
                    "动作完成标准：把这道错题的失误点记一句话（如「注意 range 是左闭右开」），下次做题前瞟一眼即可，不需要额外花时间。"
                )

    # 困惑标签：给出「风险判断 + 什么时候回看」
    if confusion_tags:
        tags_str = "、".join(confusion_tags[:3])
        parts.append(
            f"🔖 高频困惑：【{tags_str}】。"
            f"它们本身可能现在不难，但一旦进入组合场景，卡点概率会比其他点高 2-3 倍。"
            "建议的回看时机：学到第一个包含它们的组合练习前，花 3 分钟过一遍概念——"
            + (f"（另有 {len(confusion_tags) - 3} 项已记录，可在错题回顾里查看完整清单）" if len(confusion_tags) > 3 else "")
        )

    if not parts:
        # 无盲区时的审慎表述（解释「为什么是这样」+ 什么时候会变）
        if total_answers <= 3 or accuracy >= 0.95:
            return (
                "本轮未暴露可定位的盲区。"
                "更合理的解释是：当前题量还不足以覆盖到边界条件与多步骤组合场景——"
                "这不等于「所有变式都已掌握」。"
                "验证方法：后面遇到变式题（尤其是多步骤组合题）时，如果第一次就能做对，盲区才算真的没有。"
            )
        return (
            "本轮未暴露明显盲区，画像与测试结果吻合度较高。"
            "可按当前节奏推进；如果后面遇到变式题开始出现连续错误，再回看这一阶段的核心概念即可。"
        )

    return "  \n".join(parts)


# ------------------------------------------------------------------
# 鼓励与建议 — 从模板化结尾升级为「1句话收尾 + 不啰嗦」
#   差评：离目标又近了一步！💪 有任何问题随时提问，你的专属AI导师始终在这里陪你！
#   好评（按场景收束，不堆口号）：已完成【Python编程基础】阶段的闭环 — 现在把刚学会的东西用起来，比继续刷题更能巩固记忆。如果 10 分钟内想不起来，直接翻讲义；卡住超过 15 分钟，立刻提问。
# ------------------------------------------------------------------
def _generate_encouragement(
    stage_name: str,
    accuracy: float,
    emotion_score: Optional[float],
    knowledge_gaps: List[Dict[str, Any]],
    next_strategy: str = "reinforce",
    pace_adjustment: str = "medium",
) -> str:
    """收尾提示：一句阶段陈述 + 1 条与节奏section不重复的「最划算动作建议」。"""
    sentences: List[str] = []

    # 1. 阶段陈述（说事实 + 给一个判断标尺）
    if accuracy >= 0.95:
        sentences.append(
            f"【{stage_name}】本轮闭环：你用一套稳定的做题逻辑拿到了高正确率——"
            "这不是终点，而是你可以「迁移能力」的起点。"
            "判断记忆是否真的巩固的标尺：后面第一次用到这里的概念时，如果 2 分钟内不用翻讲义就能启动，才算真正记住了。"
        )
    elif accuracy >= 0.70:
        sentences.append(
            f"【{stage_name}】本轮学习完成。从正确率 {round(accuracy * 100)}% 来看，"
            "阶段主体内容已经不陌生了——现在最重要的不是「赶紧推进」，而是把「偶尔对」变成「稳定对」。"
            "判断标尺：同类题连续做对 2 道变式，才算手感真的固定下来，不会冷掉。"
        )
    else:
        sentences.append(
            f"【{stage_name}】这一轮先到这里。"
            "没通关不是坏事——它至少帮你精准定位了哪里卡住了，比「迷迷糊糊觉得学过了」强很多。"
            "判断是否值得继续死磕的标尺：同一卡点如果已经尝试了 2 种方法还是不通，就停下来，换更基础的讲法重来，不要在同一条路上反复消耗。"
        )

    # 2. 针对性动作（只发 1 条，且不与节奏section重复：
    #    - 节奏section已经覆盖「快节奏→薄弱点突击」「慢节奏→降维/切基础版」
    #    - 所以这里只补节奏没提到的场景：节奏中性时补错题复盘，或情绪为负时补休息提示，或满分+节奏中性补「迁移」建议）
    not_mastered = [g.get("skill", "") for g in knowledge_gaps if g.get("status") == "未掌握"]
    learning = [g.get("skill", "") for g in knowledge_gaps if g.get("status") == "学习中"]

    if emotion_score is not None and emotion_score < 0:
        # 节奏section可能已经说过「硬扛不划算」，这里给一个可执行的情绪恢复动作（与节奏互补）
        sentences.append(
            "给你一个「拿节奏」的具体做法：先停 5-10 分钟，然后挑 3 道你已经能稳定拿分的简单题，做一遍——"
            "从「我能搞定」的小成功里重新建立手感，比一直卡在挫败里效率高很多。"
        )
    elif next_strategy == "reinforce" and (not_mastered or learning) and pace_adjustment == "medium":
        focus = "、".join((not_mastered or learning)[:2])
        sentences.append(
            f"现在最划算的投入：花 10-15 分钟处理【{focus}】的错题，一步一步对照答案推导——"
            "为什么是 10-15 分钟？因为这刚好是「打通一个单点」的最佳时长，收益比 1 小时的新内容更高。"
        )
    elif accuracy >= 0.95 and pace_adjustment == "medium":
        # 节奏合适且满分：节奏没提推进建议，这里给迁移的具体建议
        sentences.append(
            "后面推进时的小技巧：如果 2 分钟内能流畅回忆起这里的核心概念，直接走；"
            "如果在哪一步突然想不起来，立刻翻讲义，不要硬猜——硬猜一次，后面纠正 3 次都不一定改得过来。"
        )

    # 3. 收口（温和收口，不做万能导师宣言）
    sentences.append("遇到卡顿随时提问，我们按你真实的卡点节奏走，不急。")

    return "  \n".join(sentences)


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

        # 个性化上下文提示（用于策略文案里的填空，例如「已掌握了A、B」「仍需加强C」）
        hints = _pick_context_hints(knowledge_gaps, accuracy)
        ctx = {
            "accuracy_pct": accuracy_pct,
            "stage_name": stage_name,
            **hints,
        }

        # 维度1：阶段闭环（只说事实，不做解读——解读留给后面的 section）
        if time_spent_hours > 0:
            if expected_time_hours > 0:
                time_clause = f"，用时 {time_spent_hours:.1f}h / 预期 {expected_time_hours:.1f}h"
            else:
                time_clause = f"，用时 {time_spent_hours:.1f}h"
        else:
            time_clause = ""
        overview = (
            f"【{stage_name}】：检验 {total_answers} 道题，覆盖 {total_knowledge_points} 个知识点{time_clause}。"
        )

        # 维度2：正确率评价（价值解读）
        accuracy_eval = _evaluate_accuracy(accuracy, total_answers, total_knowledge_points)
        accuracy_section = accuracy_eval
        if accuracy < 0.60:
            not_mastered_skills = [
                g.get("skill", "") for g in knowledge_gaps
                if g.get("status") == "未掌握"
            ]
            if not_mastered_skills:
                accuracy_section += (
                    f"\n\n卡壳点：{'、'.join(not_mastered_skills[:3])}。"
                    "后续将从更基础的版本重新讲起。"
                )

        # 维度3：知识点掌握分布（每类给出「它意味着什么 + 用户该怎么判断/动作」）
        grouped = _group_knowledge_status(knowledge_gaps)
        knowledge_lines = []
        if grouped["mastered"]:
            if len(grouped["mastered"]) == 1:
                knowledge_lines.append(
                    f"- ✅ 已掌握：{grouped['mastered'][0]}（已稳定——判断标准：同类变式题第一次做就能对，靠的不是记忆）。"
                    "这类点后面不用再单独花时间，目标是在组合场景里稳定调用。"
                )
            else:
                joined = "、".join(grouped["mastered"])
                knowledge_lines.append(
                    f"- ✅ 已掌握（{len(grouped['mastered'])}项）：{joined}。"
                    "这些是你的「基本盘」——后面推进时，遇到卡顿优先怀疑其他点，不要在已掌握的点上反复浪费时间。"
                )
        if grouped["learning"]:
            joined = "、".join(grouped["learning"])
            knowledge_lines.append(
                f"- 🔄 学习中（{len(grouped['learning'])}项）：{joined}。"
                "「学习中」的意思是：能看懂讲解、看答案能做对，但自己独立做变式题时不稳——这是后面最容易掉链子的点。"
                "建议的验证方式：先做 2-3 道变式自测，如果连续 2 道全对，才算升级到「已掌握」。"
            )
        if grouped["not_mastered"]:
            joined = "、".join(grouped["not_mastered"])
            knowledge_lines.append(
                f"- ❌ 待加强（{len(grouped['not_mastered'])}项）：{joined}。"
                "这些必须先打通——带着欠账往后推进的代价，不是「后面补回来」，而是每一步都更吃力，最终花的总时间是现在补齐的 2-3 倍。"
            )
        if not knowledge_lines:
            knowledge_lines = [
                "- 📝 暂无掌握状态记录。系统会先按兜底策略推进，并在下一轮尽快补齐打标。"
            ]
        knowledge_section = "\n".join(knowledge_lines)
        knowledge_has_content = bool(grouped["mastered"] or grouped["learning"] or grouped["not_mastered"])

        # 维度4：知识盲区分析
        blind_spot_analysis = _analyze_knowledge_blind_spots(
            knowledge_gaps, confusion_tags, question_records,
            total_answers=total_answers, accuracy=accuracy,
        )
        has_real_blind_spots = "本轮测试没有暴露" not in blind_spot_analysis and "没有暴露明显" not in blind_spot_analysis

        # 维度5：学习节奏评价
        pace_section = _evaluate_pace(pace_adjustment, time_diff_hours, accuracy=accuracy)

        # 维度6：情绪状态（空字符串即跳过，不占位）
        emotion_section = _evaluate_emotion(emotion_score)

        # 维度7：下一阶段策略（从「播报标签」变「标签 + 决策理由 + 下一阶段必学提示」）
        strategy_cn = STRATEGY_LABEL_CN.get(next_strategy, next_strategy)
        rationale_fn = STRATEGY_RATIONALE.get(next_strategy)
        rationale = rationale_fn(ctx) if rationale_fn else f"决策依据：正确率 {accuracy_pct}% 与当前掌握状态综合判断。"
        next_step_template = STRATEGY_NEXT_STEP_TIP.get(next_strategy, "")
        try:
            next_step_text = next_step_template.format(**ctx) if next_step_template else ""
        except (KeyError, IndexError):
            next_step_text = next_step_template
        strategy_section_parts = [
            f"策略：【{strategy_cn}】",
            "",
            f"▸ **为什么**：{rationale}",
        ]
        if next_step_text:
            strategy_section_parts.extend([
                "",
                f"▸ **接下来做什么**：{next_step_text}",
            ])
        strategy_section = "\n".join(strategy_section_parts)

        # 维度8：鼓励与建议（收尾，不堆口号）
        encouragement = _generate_encouragement(
            stage_name, accuracy, emotion_score, knowledge_gaps,
            next_strategy=next_strategy, pace_adjustment=pace_adjustment,
        )

        # 组装完整报告 — 结构从「罗列8段」变「过去·诊断 → 现在·定位 → 未来·行动」
        section_title_style = "### "
        sections: List[str] = []
        sections.append("## 阶段结果")
        sections.append("")

        # 过去·诊断（这里是已经发生的事实）
        sections.append(f"{section_title_style}📈 阶段闭环")
        sections.append(overview)
        sections.append("")

        sections.append(f"{section_title_style}✅ 测试结果解读")
        sections.append(accuracy_section)
        sections.append("")

        sections.append(f"{section_title_style}📚 知识点掌握分布")
        sections.append(knowledge_section)
        sections.append("")

        sections.append(f"{section_title_style}🔍 盲区与风险提示")
        sections.append(blind_spot_analysis)
        sections.append("")

        # 节奏评价只有在 "快/慢"（非中性）时才加醒目前缀，否则和其它段落等同
        if pace_adjustment != "medium":
            pace_section_for_report = pace_section
        else:
            pace_section_for_report = pace_section
        sections.append(f"{section_title_style}⏱️ 学习节奏信号")
        sections.append(pace_section_for_report)
        sections.append("")

        # 情绪状态信号（固定展示——保证8个section不缺，中性/无情绪时也给有价值的说明）
        sections.append(f"{section_title_style}😊 情绪状态信号")
        sections.append(emotion_section)
        sections.append("")

        # 现在·定位 / 未来·行动（把策略从「最后几段之一」拉到决策头部）
        sections.append(f"{section_title_style}🎯 下一步决策")
        sections.append(strategy_section)
        sections.append("")

        # 收尾鼓励（第8段）
        sections.append(f"{section_title_style}💡 收尾提示")
        sections.append(encouragement)

        report = "\n".join(sections).rstrip() + "\n"

        # 统计字数和覆盖维度数——8个section固定展示，情绪不再条件性隐藏
        char_count = len(report)
        dimension_sections = [
            overview,                                            # 维度1 阶段闭环
            accuracy_section,                                    # 维度2 测试结果解读
            knowledge_section if knowledge_has_content else "",  # 维度3 知识点掌握分布
            blind_spot_analysis if has_real_blind_spots else "", # 维度4 盲区分析（占位不算，但 section 仍展示）
            pace_section,                                        # 维度5 学习节奏信号
            emotion_section,                                     # 维度6 情绪状态信号（固定计入）
            strategy_section,                                    # 维度7 下一步决策
            encouragement,                                       # 维度8 收尾提示
        ]
        dimensions_covered = sum(1 for s in dimension_sections if s and s.strip())
        
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
    eval_text = _evaluate_accuracy(accuracy, total_answers=1, total_knowledge_points=1)

    return f"""## 阶段结果

### 📈 阶段闭环
【{stage_name}】阶段已形成测试闭环。

### ✅ 测试结果解读
{eval_text}

### 💡 收尾提示
阶段总结报告的某些维度在本次生成中兜底。
核心正确率与学习节奏依然保留在上方结构化卡片中；
如需要完整维度，可重跑本阶段或联系维护方查看服务端日志。
"""
