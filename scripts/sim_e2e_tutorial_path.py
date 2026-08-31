"""E2E 验证：真实生产 HybridContentAgent.run 路径（整 tutorial 字段）
两会话 Linux 画像：practice（大一） / theory（研一）

- 直接 new HybridContentAgent + ContentGenerationInput（不启动 HTTP）
- 看 tutorial 的字符相似度 / 内容级差异（不强制 3 板块外壳）
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import sys
import uuid
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from models.contracts import (  # noqa: E402
    ContentGenerationInput,
    LearningEvidence,
    StageInfo,
)
from models.profile import (  # noqa: E402
    PracticeProfile,
    UserProfile,
)
from agents.deterministic import HybridContentAgent  # noqa: E402


def char_sim(a: str, b: str) -> float:
    """字符级 Jaccard 近似（3-gram）。"""
    def ngram(s: str, n=3):
        s = re.sub(r"\s+", "", s or "")
        if len(s) < n:
            return {s}
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    A, B = ngram(a), ngram(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def content_markers(t: str, style: str) -> Dict[str, bool]:
    """不看 3 骨架外壳了，看真实内容表达里的风格差异。"""
    s = t or ""
    if style == "practice":
        return {
            "出现『我怎么用 / 我什么时候需要它 / 先明确三件事』等实操口吻": any(
                w in s for w in ("我怎么用", "我什么时候需要", "先明确", "先别急", "避坑", "工作台", "黑底白字", "直接输入")
            ),
            "出现『概念摘要 / 一句话版』式压缩": any(w in s for w in ("概念摘要", "一句话版", "一句话解释", "一句话")),
            "出现代码块 OR 终端字样/黑窗口字样/命令行": (
                s.count("```") >= 2
                or any(w in s for w in ("终端", "命令行", "Shell", "敲命令", "一行命令", "黑窗口", "工作台"))
            ),
        }
    # theory
    return {
        "出现『内核 / 架构 / 模块化 / 内核 vs 发行版 / trade-off / 复杂度』理论术语": any(
            w in s for w in ("内核", "架构", "模块化", "发行版", "trade-off", "复杂度", "边界条件", "核心要素", "数据结构", "流程")
        ),
        "出现『为什么』表达或概念动机 ≥1 条": any(w in s for w in ("为什么", "动机", "核心动机", "归因于", "原因是")),
        "出现 A vs B / 两者 / 与 ... 的差异 / 对比 / 显著差异 等对比句式": any(
            w in s for w in ("vs", "Vs", "两者", "差异", "对比", "不同之处", "区分", "关键区分", "而不是")
        ),
    }


async def main() -> int:
    agent = HybridContentAgent()
    # Linux 真实 block：KB-01-00-C001（与 diagnose 同一 source，直接调 agent._search_stage_docs）
    stage = StageInfo(
        stage_id="stg-e2e-sim",
        stage_num=1,
        block_id="LOCAL-KBS-01-00-C001",
        block_name="环境准备：安装 Linux 系统（或 WSL2 / 云服务器），熟悉终端使用，确保能执行 Shell 命令 > 一、先搞清楚三个问题（看完再动手，避免常见踩坑）（①） > 1.1 什么是 Linux？我为什么要装它？（①）",
        gap_name="Linux 基础",
        gap_level="beginner",
        gap_skill="Linux 命令操作基础",
        subtopic_ids=["KB-01-00-001"],
        teaching_objective="了解 Linux 基本概念，准备学习环境",
    )

    profiles = {
        "practice": UserProfile(
            session_id=f"e2e-prac-{uuid.uuid4().hex[:8]}",
            goal="掌握Linux（实操）",
            grade="大一",
            skill_tags=["打字基础"],
            knowledge_gaps=[],
            daily_hours=2.0,
            learning_style="practice",
            learning_rhythm="break_down_small",
            confidence=0.78,
        ),
        "theory": UserProfile(
            session_id=f"e2e-the-{uuid.uuid4().hex[:8]}",
            goal="掌握Linux（理论）",
            grade="研一",
            skill_tags=["Python", "计算机基础", "操作系统概念"],
            knowledge_gaps=[],
            daily_hours=3.0,
            learning_style="theory",
            learning_rhythm="deep_dive_long",
            confidence=0.88,
        ),
    }
    practice_profile = PracticeProfile(
        hardware="联想拯救者 Y7000P / RTX 4060 8GB",
        device_type="laptop",
        gpu_model="RTX 4060",
        vram_gb=8,
    )
    # LearningEvidence 空壳 + 简单预检查占位
    learning_evidence = LearningEvidence(
        session_id=stage.stage_id,
        stage_id=stage.stage_id,
        block_id=stage.block_id,
        unstable_prerequisites=[],
        weak_topics=[],
        recent_questions=[],
        preflight_review=None,
    )

    outs: Dict[str, Any] = {}
    for style, up in profiles.items():
        req = ContentGenerationInput(
            session_id=up.session_id,
            stage=stage,
            learning_profile=up,
            practice_profile=practice_profile,
            learning_evidence=learning_evidence,
        )
        out = await agent.run(req)
        tut = out.learning_package.tutorial or ""
        outs[style] = {
            "tutorial": tut,
            "profile": up.model_dump(mode="json"),
            "markers": content_markers(tut, style),
        }
        print(f"\n── style={style} tutorial 前 680 字 ──")
        print(tut[:680])
        print(f"  markers = {json.dumps(outs[style]['markers'], ensure_ascii=False)}")

    a = outs["practice"]["tutorial"]
    b = outs["theory"]["tutorial"]
    sim = char_sim(a, b)
    pm = outs["practice"]["markers"]
    tm = outs["theory"]["markers"]
    practice_hit = sum(1 for v in pm.values() if v)
    theory_hit = sum(1 for v in tm.values() if v)

    print("\n" + "=" * 72)
    print("E2E 结论（HybridContentAgent.run 真生产路径 · 整篇 tutorial 字段）")
    print(f"  字符相似度 = {sim:.4f}  （目标 ≤ 0.70；≥ 0.90 = 假个性化）")
    print(f"  practice 内容标记命中 {practice_hit}/3：{json.dumps(pm, ensure_ascii=False)}")
    print(f"  theory   内容标记命中 {theory_hit}/3：{json.dumps(tm, ensure_ascii=False)}")
    print(f"  tutorial 长度：practice={len(a)} chars, theory={len(b)} chars")

    if sim > 0.70:
        print("❌ 相似度仍太高：真个性化未生效。")
        return 1
    if practice_hit < 2:
        print("❌ practice 版内容特征不足")
        return 2
    if theory_hit < 2:
        print("❌ theory 版内容特征不足")
        return 3
    print("✅ E2E_OK：整篇 tutorial 路径已有差异化。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
