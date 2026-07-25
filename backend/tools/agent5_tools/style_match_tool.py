# -*- coding: utf-8 -*-
"""
style_match_tool — Agent5 风格匹配评估工具
================================================
功能：输入内容风格特征与画像中 learning_style，输出匹配度评分（0~1）。
特性：纯规则匹配，无需 Mock（mock_used 始终为 False）。

统一接口：
    def style_match(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐（learner_model.skill.md 规则3 风格适配）：
    | learning_style | 目标理论占比 | 匹配的内容特征 |
    | practice / 动手操作 | 理论20%+实践80% | 案例驱动、代码实操占主体 |
    | theory  / 理论讲解  | 理论70%+实践30% | 概念拆解、原理推导 |
    | mixed   / 混合型    | 理论40%+实践60% | 先概念后代码 |

匹配度计算（离散映射，对齐 skill 规则3）：
    - |实际理论占比 - 目标理论占比| <= 0.2 → style_match = 1.0（在偏好区间内）
    - 偏离 <= 0.4                                     → style_match = 0.5（偏离±20%）
    - 严重偏离                                         → style_match < 0.5，标记 style_mismatch(major)

输入兼容：
    - input_data["content"]：原始文本，本工具自动提取风格特征
    - input_data["style_features"]：预算特征 {"theory_ratio": 0~1, "code_ratio": 0~1}
    - input_data["learning_style"] 或 input_data["profile"]["learning_style"]
保留 StyleMatchTool 类作为向后兼容适配器。
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# 风格偏好目标理论占比（learner_model.skill.md 规则3）
_STYLE_TARGET_THEORY = {
    "practice": 0.20,
    "theory": 0.70,
    "mixed": 0.40,
}

# 实操段落关键词（用于把含实操描述的prose也算入实践部分）
_PRACTICE_KEYWORDS = (
    "示例", "代码", "实操", "动手", "实践", "运行", "输出", "执行", "命令",
    "安装", "配置", "调用", "示例代码", "demo", "case",
)


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def style_match(input_data: dict, config: dict = None) -> dict:
    """
    评估内容风格与用户 learning_style 的匹配度。

    参数：
        input_data: dict
            - content: str                 原始文本（与 style_features 二选一）
            - style_features: dict         预算特征 {"theory_ratio","code_ratio"}
            - learning_style: str          用户学习风格
            - profile: dict                画像（从中取 learning_style）
        config: dict                       纯规则，忽略 use_mock

    返回：
        {
            "status": "success" | "error",
            "data": {
                "learning_style": str,        # 归一化后的 practice/theory/mixed
                "raw_learning_style": str,    # 原始输入值
                "theory_ratio": float,        # 实际理论占比
                "practice_ratio": float,      # 实际实践占比
                "target_theory_ratio": float, # 目标理论占比
                "deviation": float,           # |实际-目标|
                "style_match": float,         # 0~1
                "issues": [{issue_type, severity, location, description, suggestion}]
            },
            "message": str,
            "mock_used": False
        }
    """
    _ = config or {}
    try:
        if not input_data:
            return _err("输入为空")

        raw_style = _extract_learning_style(input_data)
        style, style_note = _normalize_style(raw_style)
        target = _STYLE_TARGET_THEORY[style]

        theory_ratio, practice_ratio, feat_note = _compute_theory_ratio(input_data)
        if theory_ratio is None:
            return _err("未提供 content 或 style_features，无法计算理论占比")

        deviation = abs(theory_ratio - target)
        match = _match_score(deviation)

        issues: List[Dict[str, Any]] = []
        if match < 0.5:
            issues.append({
                "issue_type": "style_mismatch",
                "severity": "major",
                "location": "内容整体",
                "description": (
                    f"用户 learning_style 为 {style}（目标理论占比 {target:.0%}），"
                    f"但内容理论占比 {theory_ratio:.0%}，严重偏离用户偏好。"
                ),
                "suggestion": _style_suggestion(style),
            })

        note = " ".join(n for n in (style_note, feat_note) if n)
        return {
            "status": "success",
            "data": {
                "learning_style": style,
                "raw_learning_style": raw_style or "",
                "theory_ratio": round(theory_ratio, 4),
                "practice_ratio": round(practice_ratio, 4),
                "target_theory_ratio": target,
                "deviation": round(deviation, 4),
                "style_match": match,
                "issues": issues,
                "note": note or None,
            },
            "message": _build_message(style, match, theory_ratio, target),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("style_match 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"style_match 执行异常: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 风格特征提取
# ------------------------------------------------------------------
def _extract_learning_style(input_data: dict) -> str:
    """从 input_data 中提取 learning_style，兼容顶层与 profile 嵌套。"""
    style = (input_data or {}).get("learning_style")
    if not style:
        profile = (input_data or {}).get("profile") or {}
        style = profile.get("learning_style")
    return style or ""


def _normalize_style(raw: str) -> tuple[str, str]:
    """
    归一化 learning_style 为 practice/theory/mixed。
    兼容中文：动手操作/实操型→practice，理论讲解/理论型→theory，混合型→mixed。
    缺失或无法识别默认 mixed（learner_model.skill.md 异常处理约定）。
    """
    if not raw:
        return "mixed", "learning_style 缺失，默认 mixed"
    s = str(raw).strip().lower()
    if s in ("practice", "动手操作", "实操型", "实践型", "操作型"):
        return "practice", ""
    if s in ("theory", "理论讲解", "理论型", "理论"):
        return "theory", ""
    if s in ("mixed", "混合型", "混合"):
        return "mixed", ""
    # 模糊匹配
    if "实践" in s or "操作" in s or "practice" in s:
        return "practice", f"learning_style='{raw}' 模糊识别为 practice"
    if "理论" in s or "theory" in s:
        return "theory", f"learning_style='{raw}' 模糊识别为 theory"
    if "混合" in s or "mixed" in s:
        return "mixed", f"learning_style='{raw}' 模糊识别为 mixed"
    return "mixed", f"learning_style='{raw}' 无法识别，默认 mixed"


def _compute_theory_ratio(input_data: dict) -> tuple[float | None, float, str]:
    """
    计算内容理论占比。
    优先用预算 style_features；否则从 content 提取：
      实践字符 = 代码块内字符 + 含实操关键词的prose行字符
      理论字符 = 其余prose字符
      theory_ratio = 理论 / (理论+实践)
    返回 (theory_ratio, practice_ratio, note)。
    """
    feats = (input_data or {}).get("style_features")
    if isinstance(feats, dict) and feats.get("theory_ratio") is not None:
        tr = float(feats["theory_ratio"])
        return tr, 1.0 - tr, "使用预算 style_features"

    content = (input_data or {}).get("content") or ""
    if not content.strip():
        return None, 0.0, "未提供 content"

    # 分离代码块与prose
    code_chars = 0
    prose_parts: List[str] = []
    pos = 0
    for m in re.finditer(r"```.*?```", content, re.DOTALL):
        prose_parts.append(content[pos:m.start()])
        code_chars += len(re.sub(r"\s", "", m.group()))
        pos = m.end()
    prose_parts.append(content[pos:])
    prose = "\n".join(prose_parts)

    practice_prose_chars = 0
    theory_prose_chars = 0
    for line in prose.splitlines():
        ns = len(re.sub(r"\s", "", line))
        if ns == 0:
            continue
        if any(kw in line for kw in _PRACTICE_KEYWORDS):
            practice_prose_chars += ns
        else:
            theory_prose_chars += ns

    practice_chars = code_chars + practice_prose_chars
    total = theory_prose_chars + practice_chars
    if total == 0:
        return None, 0.0, "内容无可统计字符"
    theory_ratio = theory_prose_chars / total
    return theory_ratio, practice_chars / total, ""


def _match_score(deviation: float) -> float:
    """离散映射（对齐 learner_model.skill.md 规则3）。"""
    if deviation <= 0.2:
        return 1.0
    if deviation <= 0.4:
        return 0.5
    return max(0.2, round(1.0 - deviation, 2))


def _style_suggestion(style: str) -> str:
    if style == "practice":
        return "减少公式推导部分，增加可直接运行的完整代码示例（含数据加载、训练、保存、推理全流程）。"
    if style == "theory":
        return "增加原理推导与文献引用，减少纯代码示例，补充概念拆解与公式说明。"
    return "平衡理论与实践比例（理论40%+实践60%），先概念后代码，避免极端偏向。"


def _build_message(style: str, match: float, theory_ratio: float, target: float) -> str:
    status = "匹配" if match >= 0.5 else "不匹配"
    return f"风格{status}：learning_style={style}，理论占比 {theory_ratio:.0%}（目标 {target:.0%}），匹配度 {match}"


def _err(msg: str) -> dict:
    return {"status": "error", "data": None, "message": msg, "mock_used": False}


# ------------------------------------------------------------------
# 向后兼容适配器
# 保留原 QualityReviewAgent 调用的 async 类接口，委托给函数式核心。
# 支持画像传入：profile 可包含 learning_style
# check 返回 issues 列表（兼容旧调用方 print 的 type/message 字段）。
# ------------------------------------------------------------------
class StyleMatchTool:
    def __init__(self):
        pass

    async def check(self, content: str, profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        input_data: Dict[str, Any] = {"content": content}
        if profile is not None:
            input_data["profile"] = profile
        result = style_match(input_data)
        if result.get("status") != "success":
            return []
        issues = result["data"].get("issues", [])
        for it in issues:
            it["type"] = it.get("issue_type", "style_mismatch")
            it["message"] = it.get("description", "")
        return issues
