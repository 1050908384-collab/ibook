# -*- coding: utf-8 -*-
"""
cognitive_load_tool — 认知负荷评估工具（原Agent5，合并至专业知识与教学质量审核 Agent）
================================================
功能：输入内容文本，输出新概念密度（个/千字）、总篇幅、逻辑复杂度（高/中/低）、负荷评分（0~100）。
特性：纯计算，无需 Mock（mock_used 始终为 False）。

统一接口：
    def cognitive_load(input_data: dict, config: dict = None) -> dict
    返回：{"status", "data", "message", "mock_used"}

规则对齐：
    - pedagogy.skill.md 规则3（认知负荷理论）：新概念密度阈值 3~5适中 / 6~7偶密集 / 8~10部分超载 / >10严重超载
    - quality_metrics.skill.md 规则4.2：提供两个互逆映射函数
        * cognitive_load_score（合理度，0-100，越高越好，用于加权总分）
        * cognitive_load_level （负荷水平，0-100，越高越糟，用于红线检查，= 100 - score）
    - quality_metrics.skill.md 规则3（单项红线）：cognitive_load_level > 85 直接判不通过

概念识别采用启发式（大写首字母 + 中文术语后缀 + 已知缩写），为估算值（pedagogy.skill.md 异常处理约定）。
extract_concepts / count_chars 等模块级函数供 difficulty_analyzer_tool 复用。
保留 CognitiveLoadTool 类作为向后兼容适配器。
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# 认知负荷红线阈值（quality_metrics.skill.md 规则3）
RED_LINE_LEVEL = 85

# 中文技术术语后缀（命中即视为一个新概念）
_CHINESE_SUFFIXES = [
    "矩阵", "系数", "函数", "模型", "机制", "参数", "向量", "张量", "梯度",
    "损失", "优化器", "优化", "层", "注意力", "编码器", "解码器", "嵌入",
    "归一化", "正则化", "量化", "微调", "推理", "训练", "部署", "分解",
    "增量", "显存", "权重", "低秩", "激活", "池化", "卷积", "残差", "掩码",
    "词表", "分词", "语料", "学习率", "过拟合", "欠拟合", "泛化", "收敛",
    "分布", "采样", "概率", "交叉熵", "反向传播", "前向传播",
]

# 英文非概念词（句首词 / Python 关键字 / 内置类型），不计为新概念
_ENGLISH_NON_CONCEPTS = {
    # 句首/常见词
    "the", "in", "a", "an", "it", "we", "you", "this", "that", "for", "to",
    "if", "as", "by", "on", "and", "or", "but", "when", "while", "is", "are",
    "was", "were", "be", "been", "with", "from", "at", "not", "no", "do",
    "does", "did", "has", "have", "had", "can", "could", "will", "would",
    "should", "may", "might", "there", "here", "then", "than", "so", "such",
    "each", "all", "both", "more", "most", "other", "some", "only", "own",
    "same", "very", "just", "now", "of",
    # Python 关键字 / 内置（大写形式）
    "true", "false", "none", "self", "class", "def", "return", "import",
    "pass", "with", "try", "except", "finally", "raise", "yield", "lambda",
    "global", "nonlocal", "assert", "del", "break", "continue",
    # 常见内置类型
    "int", "str", "list", "dict", "float", "bool", "tuple", "set", "bytes",
}


# ------------------------------------------------------------------
# 模块级工具函数（供 difficulty_analyzer_tool 复用）
# ------------------------------------------------------------------
def count_chars(text: str) -> Dict[str, int]:
    """统计字符数：total 含空白，effective 不含空白（用于密度计算）。"""
    total = len(text)
    effective = len(re.sub(r"\s", "", text))
    return {"total": total, "effective": effective}


def extract_concepts(text: str) -> List[str]:
    """
    启发式提取首次出现的技术概念（去重，保留出现顺序）。
    - 英文：大写首字母词 / 连字符词 / 全大写缩写，排除非概念词
    - 中文：以技术术语后缀结尾的 1~6 字序列
    返回去重后的概念列表（每个概念计一次，对应"首次出现"）。
    """
    if not text:
        return []
    seen: List[str] = []
    seen_set = set()

    # 英文概念：首字母大写或全大写，允许连字符
    for m in re.finditer(r"[A-Z][a-zA-Z0-9]*(?:-[A-Za-z0-9]+)*", text):
        word = m.group()
        if word.lower() in _ENGLISH_NON_CONCEPTS:
            continue
        if len(word) < 2 and word.upper() != word:
            continue
        if word not in seen_set:
            seen_set.add(word)
            seen.append(word)

    # 中文概念：后缀匹配
    suffix_alt = "|".join(sorted(_CHINESE_SUFFIXES, key=len, reverse=True))
    for m in re.finditer(rf"[\u4e00-\u9fff]{{1,6}}(?:{suffix_alt})", text):
        term = m.group()
        if term not in seen_set:
            seen_set.add(term)
            seen.append(term)

    return seen


def compute_density(text: str) -> tuple[float, List[str]]:
    """计算新概念密度（个/千字）与概念列表。density = 概念数 / 非空白字符数 × 1000。"""
    concepts = extract_concepts(text)
    eff = count_chars(text)["effective"]
    density = (len(concepts) / eff * 1000) if eff > 0 else 0.0
    return density, concepts


# ------------------------------------------------------------------
# 评分映射函数（直接实现 quality_metrics.skill.md 规则4.2）
# ------------------------------------------------------------------
def map_cognitive_load_score(concept_density: float) -> int:
    """返回维度评分（合理度，0-100，越高越好，用于加权总分）。"""
    if concept_density <= 5:
        return 100 - round((5 - concept_density) * 2)      # 3~5个: 96~100
    elif concept_density <= 7:
        return 90 - round((concept_density - 5) * 10)      # 6~7个: 80~70
    elif concept_density <= 10:
        return 69 - round((concept_density - 7) * 6)       # 8~10个: 51~69
    else:
        return max(0, 49 - round((concept_density - 10) * 5))  # >10个: <49


def map_cognitive_load_level(concept_density: float) -> int:
    """返回负荷水平（0-100，越高越糟，用于红线检查），与合理度互逆。"""
    return 100 - map_cognitive_load_score(concept_density)


# ------------------------------------------------------------------
# 统一函数式接口（规范实现）
# ------------------------------------------------------------------
def cognitive_load(input_data: dict, config: dict = None) -> dict:
    """
    评估内容的认知负荷。

    参数：
        input_data: dict
            - content: str   待评估文本（tutorial / answer）
        config: dict         纯计算，忽略 use_mock 等字段

    返回：
        {
            "status": "success" | "error",
            "data": {
                "concept_density": float,       # 新概念密度（个/千字）
                "total_length": int,            # 总篇幅（字符数，含空白）
                "effective_chars": int,         # 非空白字符数（密度计算依据）
                "logical_complexity": "高"|"中"|"低",
                "cognitive_load_score": int,    # 合理度（0-100，越高越好）
                "cognitive_load_level": int,    # 负荷水平（0-100，越高越糟）
                "load_score": int,              # 负荷评分（=cognitive_load_level）
                "red_line_triggered": bool,     # cognitive_load_level > 85
                "concept_count": int,
                "new_concepts": [str],
                "metrics": {"formula_count","nesting_depth","code_block_count"},
                "issues": [{issue_type, severity, location, description, suggestion}]
            },
            "message": str,
            "mock_used": False
        }
    """
    _ = config or {}
    try:
        content = (input_data or {}).get("content") or ""
        max_output_length = (input_data or {}).get("max_output_length")
        if not content.strip():
            return {
                "status": "error",
                "data": None,
                "message": "输入未包含 content 字段，无法评估认知负荷",
                "mock_used": False,
            }

        density, concepts = compute_density(content)
        chars = count_chars(content)
        score = map_cognitive_load_score(density)
        level = map_cognitive_load_level(density)
        complexity, metrics = _analyze_complexity(content)

        red_line = level > RED_LINE_LEVEL
        issues = _build_issues(density, level, red_line, complexity)
        if isinstance(max_output_length, int) and max_output_length > 0 and chars["effective"] > max_output_length:
            issues.append({
                "issue_type": "verbosity_over_budget",
                "severity": "major",
                "location": "full_text",
                "description": (
                    f"Output length {chars['effective']} exceeds the approved source budget "
                    f"of {max_output_length} characters."
                ),
                "suggestion": "Remove repetition and unsupported expansion; keep the revised text within the source budget.",
            })

        return {
            "status": "success",
            "data": {
                "concept_density": round(density, 2),
                "total_length": chars["total"],
                "effective_chars": chars["effective"],
                "logical_complexity": complexity,
                "cognitive_load_score": score,
                "cognitive_load_level": level,
                "load_score": level,
                "red_line_triggered": red_line,
                "concept_count": len(concepts),
                "new_concepts": concepts,
                "metrics": metrics,
                "issues": issues,
            },
            "message": _build_message(density, level, complexity, red_line),
            "mock_used": False,
        }
    except Exception as e:
        logger.exception("cognitive_load 执行失败")
        return {
            "status": "error",
            "data": None,
            "message": f"cognitive_load 执行异常: {e}",
            "mock_used": False,
        }


# ------------------------------------------------------------------
# 逻辑复杂度分析（pedagogy.skill.md 步骤4：嵌套层级深度 + 公式数量）
# ------------------------------------------------------------------
def _analyze_complexity(content: str) -> tuple[str, Dict[str, int]]:
    """返回逻辑复杂度等级与原始指标。"""
    # 公式数量：$$ 块 + 行内 $ + LaTeX 命令 + 数学符号行
    formula_count = len(re.findall(r"\$\$", content)) // 2
    formula_count += len(re.findall(r"\\(?:frac|sum|int|sqrt|cdot|times|alpha|beta|sigma|delta|theta|mu|lambda)\b", content))
    formula_count += len(re.findall(r"[αβγσΔΣθμλ∂∇∑∫√≈≤≥≠]", content))

    # 括号嵌套深度
    nesting_depth = _max_bracket_depth(content)

    # 代码块数量
    code_block_count = len(re.findall(r"```", content)) // 2

    raw = formula_count * 6 + nesting_depth * 8 + code_block_count * 4
    if raw >= 30:
        complexity = "高"
    elif raw >= 15:
        complexity = "中"
    else:
        complexity = "低"
    return complexity, {
        "formula_count": formula_count,
        "nesting_depth": nesting_depth,
        "code_block_count": code_block_count,
    }


def _max_bracket_depth(content: str) -> int:
    """计算括号 ( [ { 的最大嵌套深度。"""
    depth = max_depth = 0
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in content:
        if ch in "([{":
            stack.append(ch)
            depth = len(stack)
            max_depth = max(max_depth, depth)
        elif ch in ")]}":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
                depth = len(stack)
            else:
                # 不匹配，跳过避免误判
                pass
    return max_depth


# ------------------------------------------------------------------
# 问题清单与消息
# ------------------------------------------------------------------
def _build_issues(density: float, level: int, red_line: bool, complexity: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if density > 10:
        severity = "critical" if red_line else "major"
        issues.append({
            "issue_type": "cognitive_load",
            "severity": severity,
            "location": "全文",
            "description": f"新概念密度 {density:.1f} 个/千字，负荷水平 {level}，{'超过85红线' if red_line else '超过10个/千字上限'}，认知负荷超载。",
            "suggestion": "将内容拆分为多个阶段，每阶段新概念控制在5个以内，或补充辅助说明。",
        })
    elif density > 7:
        issues.append({
            "issue_type": "concept_dense",
            "severity": "minor",
            "location": "全文",
            "description": f"新概念密度 {density:.1f} 个/千字，部分超载。",
            "suggestion": "适当分散概念引入节奏，每段新概念不超过2个。",
        })
    if complexity == "高":
        issues.append({
            "issue_type": "high_complexity",
            "severity": "minor",
            "location": "全文",
            "description": "逻辑复杂度高（公式/嵌套/代码块密集）。",
            "suggestion": "增加分步骤图示或过渡说明，降低外在认知负荷。",
        })
    return issues


def _build_message(density: float, level: int, complexity: str, red_line: bool) -> str:
    if red_line:
        return f"认知负荷红线触发：密度 {density:.1f}/千字，负荷水平 {level} > {RED_LINE_LEVEL}，建议拆分阶段"
    return f"概念密度 {density:.1f}/千字，合理度 {100 - level}，负荷水平 {level}，复杂度{complexity}"


# ------------------------------------------------------------------
# 向后兼容适配器
# 保留原调用的 async 类接口，委托给函数式核心。
# analyze 返回 issues 列表（兼容旧调用方 print 的 type/message 字段）。
# ------------------------------------------------------------------
class CognitiveLoadTool:
    def __init__(self):
        pass

    async def analyze(self, content: str, max_output_length: int | None = None) -> List[Dict[str, Any]]:
        result = cognitive_load({
            "content": content,
            "max_output_length": max_output_length,
        })
        if result.get("status") != "success":
            return []
        issues = result["data"].get("issues", [])
        for it in issues:
            it["type"] = it.get("issue_type", "cognitive_load")
            it["message"] = it.get("description", "")
        return issues
