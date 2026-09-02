# -*- coding: utf-8 -*-
"""专业知识与教学质量审核 Agent checker for whether a block applies 个性化学习资源生成 Agent's personalization plan."""
from __future__ import annotations

import re
from typing import Any


_PRACTICE_MARKERS = ("实践", "实操", "动手", "示例", "practice")
_VALID_ACTIONS = {
    "keep", "remediate",
    "explain",
    "explain_and_practice",
    "compress",
    "reorder",
    "contextualize",
}

# grade → 目标术语密度（每千字预期 ≥ N 个 "进阶/专业类术语" 命中）
# 低于下限 → 大一写得太水；高于上限 → 研究生版术语密度溢出（大一看不懂）。
# 语义与 learner_model.skill.md 规则 3 / grade 粒度匹配。
_GRADE_TERMINOLOGY_TARGETS = {
    # 关键字子串匹配（忽略大小写）。key = 归一化 grade 关键词；value = (lower, upper)
    "本科一年级": (2, 6),
    "大一": (2, 6),
    "freshman": (2, 6),
    "undergraduate year 1": (2, 6),
    "本科二年级": (3, 8),
    "大二": (3, 8),
    "本科三年级": (4, 10),
    "大三": (4, 10),
    "本科四年级": (5, 12),
    "大四": (5, 12),
    "本科": (3, 10),
    "undergraduate": (3, 10),
    "硕士": (7, 16),
    "研究生": (7, 16),
    "master": (7, 16),
    "ms": (7, 16),
    "博士": (10, 22),
    "phd": (10, 22),
    "doctor": (10, 22),
}

# 「学术进阶/密度型术语」词表——只选"大一 vs 研究生"差异非常显著的、
# 几乎不会出现在入门铺垫里的硬核词。中英文覆盖，子串匹配即可（避免枚举）。
_ADVANCED_TERMINOLOGY_KEYWORDS = (
    # 通用 CS / 工程进阶
    "渐近复杂度", "渐近复杂度", "时间复杂度", "空间复杂度", "大o记号",
    "下界", "上界", "worst-case", "amortized", "不动点", "不动点定理",
    "微分同胚", "流形", "测度论", "凸松弛", "对偶性", "lagrangian",
    "regularization", "泛化界", "泛化误差界", "generalization bound",
    "pac learnable", "probably approximately correct", "vc维", "vc dimension",
    "radon-nikodym", "gaussian isoperimetric", "chernoff bound",
    "hoeffding", "mcdiarmid", "union bound", "convergence in distribution",
    "鞅收敛", "martingale", "kl散度", "f散度", "wasserstein",
    "stein discrepancy", "maximum mean discrepancy", "mmd",
    "hessian", "雅可比", "jacobian", "奇异值分解", "谱分解", "qr分解",
    "cholesky", "condition number", "kkt", "kuhn-tucker", "拉格朗日对偶",
    "一阶必要性", "二阶充分性", "非凸非光滑", "subgradient", "proximal",
    "alternating direction method of multipliers", "admm",
    "warm start", "oracle complexity", "样本复杂度", "communication complexity",
    "信息论下界", "信息论上界", "data-processing inequality",
    "rate-distortion", "shannon entropy", "互信息", "mutual information",
    "bregman divergence", "pinsker", "cramér-rao", "fisher information",
    "重参数化", "reparameterization", "score matching", "score-based",
    "stochastic differential equation", "sde", "fokker-planck",
    "reverse-time sde", " Langevin", "mcmc", "markov chain monte carlo",
    "metropolis-hastings", "gibbs sampling", "变分推断", "variational inference",
    "elbo", "evidence lower bound", "mean-field", "structured variational",
    "stability", "generalization gap", "empirical risk minimization",
    "structural risk minimization", "bias-variance tradeoff", "trade-off",
    # LLM / ML 进阶
    "rope (rotary position)", "rotary position embedding", "grouped-query attention",
    "gqa", "multi-query attention", "mqa", "sliding window attention",
    "flash attention", "paged attention", "prefix lm", "causal lm",
    "alignment tax", "preference optimization", "dpo", "ipo", "kto", "ppo",
    "proximal policy optimization", "direct preference optimization",
    "rejection sampling", "best-of-n", "speculative decoding", "medusa",
    "eagle", "tree-structured decoding", "kv cache", "paged kv",
    "tensor parallel", "pipeline parallel", "data parallel", "fsdp",
    "zero redundancy optimizer", "zero-1", "zero-2", "zero-3",
    "gradient checkpointing", "mixed precision", "bfloat16", "fp8",
    "per-layer quant", "activation-aware quant", "awq", "gptq",
    "qlora", "double quant", "nf4", "dora", "lora alpha",
    "rank stabilization", "residual stream", "attn sink", "streaming llm",
    "sycophancy", "truthfulqa", "hallucination rate", "faithfulness",
    "in-context learning", "icl", "chain-of-thought", "cot",
    "least-to-most", "tree of thoughts", "graph of thoughts",
    "tool-use grounding", "function calling", "structured output",
    # Linux / 系统进阶（入门铺垫里不会出现）
    "slab allocator", "slub", "slob", "rcu", "read-copy-update",
    "cgroup v2", "control group", "namespace user", "pid namespace",
    "network namespace", "overlayfs", "coalescing writes",
    "io_uring", "splice(2)", "sendfile(2)", "zero-copy",
    "eBPF", "bpf verifier", "tracepoint", "kprobe", "uprobes",
    "ftrace", "perf_event", "cfs bandwidth", "cfs quota",
    "completely fair scheduler", "oom killer", "page frame reclaiming",
    "thrashing", "page cache thrash", "reverse mapping", "rmap",
    "transparent hugepage", "thp", "memory compaction", "numa balancing",
    "iptables conntrack", "connection tracking", "nf_conntrack",
    "tc rules", "htb qdisc", "netfilter hooks",
    "systemd journald", "dbus activation", "polkit",
)

# "入门铺垫 / 定义什么是…"类语句词表——只有当用户 skill_tags 已经掌握相关主题时，
# 这些才算"废话"；否则入门用户依然需要。
_INTRO_FILLER_HINTS = (
    "什么是", "什么叫", "是什么", "介绍一下", "简单介绍", "快速了解",
    "入门指南", "基础入门", "顾名思义", "基本概念", "基本定义",
    "定义", "我们首先了解", "让我们首先", "我们先了解", "首先学习",
    "首先我们来了解", "首先我们介绍", "概述", "简介", "概要",
    "在深入之前", "正式介绍之前", "在学习", "what is",
    "introduction to", "getting started", "let's first understand",
    "let's start with", "before we dive", "before diving",
    "基础知识", "基本知识", "基本用法", "基本语法", "背景",
)


def _match_grade_band(grade: str):
    g = (grade or "").strip()
    if not g:
        return None
    g_low = g.lower()
    for kw, (lo, hi) in _GRADE_TERMINOLOGY_TARGETS.items():
        if kw.lower() in g_low:
            return g, lo, hi
    return None


def _count_advanced_terms(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    return sum(1 for kw in _ADVANCED_TERMINOLOGY_KEYWORDS if kw.lower() in t)


def _char_count(text: str) -> int:
    if not text:
        return 0
    # 按"可见字符"（排除 markdown 格式符）近似估算；空白不计入
    return len(re.sub(r"\s+", "", text or ""))


def _intro_filler_ratio(text: str) -> float:
    """估算入门铺垫句占比：按句子切，命中 _INTRO_FILLER_HINTS 子句的比例。"""
    if not text:
        return 0.0
    sentences = re.split(r"(?<=[。！？!?\n])", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        return 0.0
    hits = 0
    for s in sentences:
        sl = s.lower()
        if any(h.lower() in sl for h in _INTRO_FILLER_HINTS):
            hits += 1
    return hits / len(sentences)


def _skill_tag_matches_filler(tag: str, text: str) -> bool:
    """判断用户已掌握 skill_tag 是否在块"入门铺垫"里被重复介绍过。

    命中条件（两个都满足）：
      1. 正文出现 ≥ 1 处 skill tag 关键词（同义扩展）；
      2. 正文 ≥ 2 处 _INTRO_FILLER_HINTS 命中，或 入门铺垫占比 ≥ 0.2。
    """
    tag = (tag or "").strip().lower()
    if not tag or not text:
        return False
    t_low = text.lower()
    tag_forms = {tag}
    # 常见英文连写/缩写变体
    for variant in (
        tag.replace(" ", ""),
        tag.replace("_", " "),
        tag.replace("_", ""),
    ):
        tag_forms.add(variant)
    # 去掉过短的 tag（长度 1 基本不可能作为语义块）
    tag_forms = {t for t in tag_forms if len(t) >= 2}
    if not any(t in t_low for t in tag_forms):
        return False
    # 入门铺垫占比高（≥ 20% 或绝对 ≥ 2 句）
    ratio = _intro_filler_ratio(text)
    sentences = re.split(r"(?<=[。！？!?\n])", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    fillers_hit = sum(
        1 for s in sentences
        if any(h.lower() in s.lower() for h in _INTRO_FILLER_HINTS)
    )
    return ratio >= 0.20 or fillers_hit >= 2


class PersonalizationCheckTool:
    """Validate structured teaching-layer metadata without judging source facts."""

    async def check(self, block: dict[str, Any], profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        profile = profile or {}
        block = block or {}
        issues: list[dict[str, Any]] = []
        action = str(block.get("personalization_action") or "").strip()
        reason = str(block.get("personalization_reason") or "").strip()
        source_ids = [item for item in block.get("fact_source_chunk_ids", []) if item]
        evidence_refs = [item for item in block.get("evidence_refs", []) if item]
        style = str(profile.get("learning_style") or "").lower()

        # ★★2 画像形式红线：需要完整正文做轻量术语/句式统计。
        # block 在不同路径上可能携带 3 种不同正文字段：优先 personalized_content
        #（块级），其次 tutorial/content（兜底）。仅在有正文时做 grade/skill_tags 校验，
        # 缺失不硬报错（避免空块误触发）。
        content_text = (
            str(block.get("personalized_content") or "")
            or str(block.get("content") or "")
            or str(block.get("tutorial") or "")
        ).strip()

        if action not in _VALID_ACTIONS:
            issues.append({
                "error_type": "personalization_action_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "内容块未声明有效的个性化处理动作。",
                "suggestion": "声明 keep、compress、explain、reorder、contextualize 或 explain_and_practice。",
            })
        if not reason:
            issues.append({
                "error_type": "personalization_reason_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "内容块未记录本次个性化的用户画像依据。",
                "suggestion": "记录学习目标、学习方式或实训条件中至少一项作为生成依据。",
            })
        if not source_ids:
            issues.append({
                "error_type": "teaching_source_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "个性化教学层未声明对应的事实来源块。",
                "suggestion": "在 fact_source_chunk_ids 中保留当前 source_chunk_id。",
            })
        if action == "remediate" and not evidence_refs:
            issues.append({
                "error_type": "remediation_evidence_missing",
                "severity": "major",
                "location": "教学层元数据",
                "description": "前置补救内容未绑定学习证据。",
                "suggestion": "在 evidence_refs 中记录对应的前置知识点或低分阶段。",
            })
        # ★★2 规则①：grade 术语密度匹配。
        # 场景：大一版写出了研究生版的术语密度 → 打回降难度。
        # 触发：profile.grade 能匹配目标档 + 块正文足够长（≥ 400 字）
        # → 统计每千字进阶术语个数。< 下限 or > 上限 → grade_terminology_density_mismatch
        grade_raw = str(profile.get("grade") or "").strip()
        grade_band = _match_grade_band(grade_raw) if grade_raw else None
        if grade_band and content_text:
            grade_name, lo_target, hi_target = grade_band
            char_n = _char_count(content_text)
            # 正文 ≥ 200 字时才启用（单个学习块的最小合理长度）
            if char_n >= 200:
                # 换算成"每千字"
                n_terms = _count_advanced_terms(content_text)
                terms_per_k = (n_terms / char_n) * 1000
                if terms_per_k < lo_target or terms_per_k > hi_target:
                    reason_desc = (
                        f"密度过高（已掌握者读起来更合适）"
                        if terms_per_k > hi_target
                        else f"密度过低（讲得像入门水课，目标年级缺乏挑战）"
                    )
                    issues.append({
                        "error_type": "grade_terminology_density_mismatch",
                        "severity": "major",
                        "location": "年级匹配",
                        "description": (
                            f"画像 grade={grade_name}（目标进阶术语 {lo_target}~{hi_target} 个/千字），"
                            f"但块正文中实际为 {terms_per_k:.1f} 个/千字（共 {n_terms} 个进阶术语 / "
                            f"{char_n} 字正文）——{reason_desc}。"
                        ),
                        "suggestion": (
                            "按 grade 调优术语密度：过高时把大段研究生级术语换成定性解释+超链接/脚注；"
                            "过低时在原文骨架基础上补充 2~3 段本年级应掌握的进阶内容（必须来自 source 原文）。"
                        ),
                    })

        # ★★2 规则②：已掌握 skill_tags 仍写入门铺垫废话。
        # 场景：用户 skill_tags 已掌握 Linux / LoRA 等，但块正文里仍然出现大量
        # "什么是 Linux / 什么是 LoRA"类的入门句 → 定量打回改（压缩或直接跳过铺垫）。
        mastered_tags_raw = profile.get("skill_tags") or []
        if isinstance(mastered_tags_raw, (list, tuple)) and content_text:
            char_n = _char_count(content_text)
            if char_n >= 200:
                filler_ratio = _intro_filler_ratio(content_text)
                matching_tags = []
                for tag in mastered_tags_raw:
                    ts = str(tag).strip()
                    if not ts:
                        continue
                    if _skill_tag_matches_filler(ts, content_text):
                        matching_tags.append(ts)
                if matching_tags and filler_ratio >= 0.18:
                    issues.append({
                        "error_type": "skill_tags_mastered_intro_filler",
                        "severity": "major",
                        "location": "已掌握技能匹配",
                        "description": (
                            f"画像已掌握 skill_tags={matching_tags}，但本块正文里入门铺垫句占比 "
                            f"≈{filler_ratio:.0%}，仍然在为这些已掌握的概念写『什么是 X / 基本概念 X』"
                            f"类入门废话——不符合『已掌握 → 简单化而非重复铺垫』的个性化规则。"
                        ),
                        "suggestion": (
                            "压缩或跳过已掌握技能的入门铺垫段：保留原文骨架中与"
                            "本次块目标直接相关的段，其余定义性段落改为 1 句『读者已掌握 X，"
                            "如有疑问参考：原文来源章节 Y』的引用形式即可。"
                        ),
                    })

        return issues
