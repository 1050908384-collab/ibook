# -*- coding: utf-8 -*-
"""Agent4 ★ 立刻要做 落地验证脚本（纯规则，0 LLM / 0 DB 调用，0 token）

验证 5 组用例：
  Case A. Baseline：合法实操块 + 轻度教学建议  →  passed=True
  Case B. 实操型画像 + 块缺 teaching_tasks（缺动手验证段）
         → PersonalizationCheckTool 打回 practice_task_missing，并入 rule_errors → passed=False
  Case C. 块 personalization_action 无效（= ""）
         → personalization_action_missing 阻断 passed=False
  Case D. 严重风格错配（practice 画像 + 几乎 100% 理论废话 / 几乎 0% 实践）
         → StyleMatchTool 报 style_mismatch(major) → 移到 hallucination_errors / result.errors
         → passed=False AND must_fix=True
  Case E. 轻度风格偏离（match ≥ 0.5） + 轻微难度建议
         → 仅 personalization_errors 非空、passed=True、must_fix=False（不阻断）
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
for p in (str(ROOT), str(BACKEND)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ------------------------------------------------------------------
# 1) 环境补丁：避免 deterministic.py 里 LLMClient / 数据库依赖在 import 阶段崩溃
# ------------------------------------------------------------------
os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-placeholder-nocall")   # 占位即可，不会被真实调用
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")
os.environ.setdefault("EXECUTION_MODE", "hybrid")
os.environ.setdefault("BLOCK_PIPELINE_ENABLED", "true")
os.environ.setdefault("ALLOW_MODEL_LOAD", "False")

# ------------------------------------------------------------------
# 2) 工具级 Unit：直接调 PersonalizationCheckTool / StyleMatchTool，校验原始 rule 信号
# ------------------------------------------------------------------
async def tool_level_cases() -> None:
    from tools.agent4_tools.personalization_check_tool import PersonalizationCheckTool
    from tools.agent4_tools.style_match_tool import style_match

    pct = PersonalizationCheckTool()

    # Case B 工具级：practice learner + 缺 teaching_tasks
    # PersonalizationCheckTool 用的是模糊匹配（_PRACTICE_MARKERS 含 "practice"/"实操"），
    # 这里用 "动手实操型" 中文也能命中；agent 层 UserProfile 只接受 practice/theory/mixed。
    practice_profile = {"learning_style": "动手实操型"}
    bad_block_no_tasks = {
        "personalization_action": "explain_and_practice",
        "personalization_reason": "match practice learner",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [],  # ← 违规
    }
    issues = await pct.check(bad_block_no_tasks, profile=practice_profile)
    err_types = [i["error_type"] for i in issues]
    assert "practice_task_missing" in err_types, f"[B-tool] practice_task_missing 未命中: {err_types}"
    assert any(i["severity"] == "major" for i in issues if i["error_type"] == "practice_task_missing"), \
        f"[B-tool] severity 必须 major"
    print(f"[PASS][B-tool] 实操缺动手任务: {err_types}")

    # Case C 工具级：action 无效
    bad_block_bad_action = {
        "personalization_action": "",       # ← 违规
        "personalization_reason": "r",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [],
    }
    issues = await pct.check(bad_block_bad_action, profile={"learning_style": "理论讲解型"})
    err_types = [i["error_type"] for i in issues]
    assert "personalization_action_missing" in err_types, \
        f"[C-tool] action_missing 未命中: {err_types}"
    print(f"[PASS][C-tool] action 无效: {err_types}")

    # Case D 工具级：practice 用户 + 纯理论文（完全没代码/命令）
    pure_theory = "\n".join([
        "Linux 操作系统的概念与设计动机",
        "我们首先深入剖析 Unix 哲学的宏体系结构与内核演化脉络",
        "从宏观视角理解分时系统调度原理、进程抽象与地址空间设计",
        "然后讨论 POSIX 标准对可移植操作系统的约束与语义层含义",
        "最后对比 Monolithic Kernel 与 Micro Kernel 的学术争议与权衡利弊",
    ])
    r = style_match({"content": pure_theory, "profile": {"learning_style": "practice"}})
    assert r["status"] == "success"
    issue_types = [i.get("issue_type") for i in r["data"]["issues"]]
    severities = [i.get("severity") for i in r["data"]["issues"]]
    assert "style_mismatch" in issue_types and "major" in severities, \
        f"[D-tool] style_mismatch major 未命中: {r['data']['issues']}"
    print(f"[PASS][D-tool] 实操画像+纯理论风格错配: score={r['data']['style_match']}, "
          f"theory={r['data']['theory_ratio']:.0%}, issues={r['data']['issues']}")

    # Case E 工具级：practice 用户 + 中等实践占比（在宽容区间内，无 major style_mismatch）
    mixed = "\n".join([
        "Linux 命令基础概述。",
        "### 立即动手·最简操作",
        "执行 `ls -l` 查看目录：",
        "```",
        "ls -l",
        "```",
        "输出结果的第三列表示文件所有者，第四列表示所属组。",
        "再执行 `pwd` 打印当前目录，动手验证输出。",
    ])
    r = style_match({"content": mixed, "profile": {"learning_style": "practice"}})
    assert r["status"] == "success"
    assert not any(i.get("severity") == "major" for i in r["data"]["issues"]), \
        f"[E-tool] 轻度匹配不应产生 major: {r['data']['issues']}"
    print(f"[PASS][E-tool] 轻度匹配无阻断: score={r['data']['style_match']}, "
          f"theory={r['data']['theory_ratio']:.0%}, issues={len(r['data']['issues'])}")


# ------------------------------------------------------------------
# 3) Agent 级 Integration：调 HybridTechnicalReviewAgent.run()，
#    走整条 rule_errors / personalization_errors / hallucination_errors 通道。
#    Mock 掉会访问 DB / 调 LLM 的工具 → 只验证 ★ 新增规则的集成链路。
# ------------------------------------------------------------------
async def agent_level_cases() -> None:
    from models.contracts import ReviewInput
    from models.profile import UserProfile
    from models.stage import StageInfo
    from agents.deterministic import HybridTechnicalReviewAgent

    agent = HybridTechnicalReviewAgent()

    # ---- Patch：彻底切断 DB / LLM / 外部依赖工具 ----
    async def _fake_rule_check(*a, **kw): return []
    async def _fake_knowledge(*a, **kw): return []
    async def _fake_diff(*a, **kw): return []
    async def _fake_cog(*a, **kw): return []
    async def _fake_llm_complete(*a, **kw):
        # 只返回空 errors，不干扰我们测 deterministic 工具链
        return '{"errors": [], "passed": true}'

    agent.rule_check_tool.check_rules = _fake_rule_check
    agent.knowledge_verify_tool.verify_knowledge = _fake_knowledge
    agent.knowledge_verify_tool.verify_against_chunk = _fake_knowledge
    agent.knowledge_verify_tool.retrieve_evidence = lambda *a, **kw: []
    agent.difficulty_analyzer_tool.analyze = _fake_diff
    agent.cognitive_load_tool.analyze = _fake_cog
    agent.llm_client.complete = _fake_llm_complete

    def _mk_stage():
        return StageInfo(
            stage_id="stg-unit",
            stage_num=1,
            block_id="KB-LINUX-001",
            block_name="Linux基础",
            gap_skill="linux_fundamentals",
            subtopic_ids=["sub-1"],
        )

    def _run_case(name, inp: ReviewInput, expect_passed: bool, must_have_error_types=None,
                  must_not_have_in_errors=None, must_fix_on=None):
        """must_fix_on: {error_type: True|False}  要求 result.errors 里的每条指定 error_type 必须满足 must_fix"""
        print(f"\n=== Agent-Level Case {name} ===")
        res = asyncio.get_event_loop().run_until_complete(agent.run(inp))
        # workaround: 上面外层已经是 async，这里直接用 await 的方式我们在外层调用。
        return res

    # 因为上面嵌套 asyncio 不方便，直接在外面 await：
    def _make_req(*, profile, blocks, content):
        return ReviewInput(
            session_id="sess-unit",
            target_type="learning_package",
            content=content,
            stage=_mk_stage(),
            profile=profile,
            content_blocks=blocks,
        )

    # --- Case A (baseline 合法) ---
    profile_ok = UserProfile(session_id="sess-unit", learning_style="practice",
                             skill_tags=["linux_basics"])
    blocks_ok = [{
        "content_block_id": "cb-1",
        "personalization_action": "explain_and_practice",
        "personalization_reason": "匹配实操画像，动手验证+观察输出",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [
            {"type": "verify", "minutes": 3, "prompt": "运行 ls -l 观察所有者列"},
            {"type": "self_check", "minutes": 2, "prompt": "回答 pwd 输出了什么"},
        ],
    }]
    content_ok = (
        "# Linux ls 命令实操\n"
        "首先理解：`ls` 列出目录内容。\n"
        "## 立即动手·最简操作\n"
        "运行 `ls -l` 观察输出。\n"
        "```\nls -l\n```\n"
        "## 动手验证\n"
        "修改为 `ls -a`，对比有无隐藏文件的差异。\n"
    )
    res = await agent.run(_make_req(profile=profile_ok, blocks=blocks_ok, content=content_ok))
    # baseline 可能会因 style / difficulty 产生 minor 建议，但必须 passed=True
    print(f"[A] passed={res.passed}  errors={[e.error_type for e in res.errors]}  "
          f"personalization_errors={[e.error_type+'|'+e.severity+'|must_fix='+str(e.must_fix) for e in res.personalization_errors]}")
    assert res.passed is True, f"[A] baseline 不该阻断。passed={res.passed} errors={res.errors}"
    for e in res.personalization_errors:
        assert e.must_fix is False, f"[A] personalization 建议 must_fix 必须 False，got {e.error_type}={e.must_fix}"
    print("[PASS][A] 合法 baseline 全部通过，personalization 建议不阻断。")

    # --- Case B (缺 teaching_tasks → practice_task_missing 阻断) ---
    profile_practice = UserProfile(session_id="sess-unit", learning_style="practice", skill_tags=[])
    blocks_bad_tasks = [{
        "content_block_id": "cb-1",
        "personalization_action": "explain_and_practice",
        "personalization_reason": "匹配实操画像",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [],   # ← 违规
    }]
    res = await agent.run(_make_req(profile=profile_practice, blocks=blocks_bad_tasks, content=content_ok))
    err_types = [e.error_type for e in res.errors]
    print(f"[B] passed={res.passed}  error_types={err_types}")
    assert res.passed is False, f"[B] 缺 practice 任务应该 passed=False"
    assert "practice_task_missing" in err_types, f"[B] 缺 practice_task_missing: {err_types}"
    must_fix_flags = {e.error_type: e.must_fix for e in res.errors}
    assert must_fix_flags.get("practice_task_missing") is True, \
        f"[B] practice_task_missing 必须 must_fix=True，got {must_fix_flags}"
    print("[PASS][B] PersonalizationCheckTool.practice_task_missing 并入 rule_errors，阻断发布，must_fix=True。")

    # --- Case C (action 空 → personalization_action_missing 阻断) ---
    blocks_bad_action = [{
        "content_block_id": "cb-1",
        "personalization_action": "",      # ← 违规
        "personalization_reason": "r",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [{"type": "verify", "minutes": 2, "prompt": "x"}],
    }]
    res = await agent.run(_make_req(profile=profile_practice, blocks=blocks_bad_action, content=content_ok))
    err_types = [e.error_type for e in res.errors]
    print(f"[C] passed={res.passed}  error_types={err_types}")
    assert res.passed is False and "personalization_action_missing" in err_types, \
        f"[C] action_missing 应阻断: passed={res.passed} errors={err_types}"
    print("[PASS][C] PersonalizationCheckTool.personalization_action_missing 阻断发布。")

    # --- Case D (严重风格错配：practice 用户 + 100% 纯理论文章) ---
    pure_theory_content = (
        "# Linux 内核设计哲学\n"
        "Linux 是一种类 Unix 的单体内核操作系统。\n"
        "首先，我们从宏观上剖析 Unix 哲学的宏体系结构：一切皆文件、机制与策略分离、"
        "模块化内核子系统、可移植性设计理念与 POSIX 标准语义的深层含义。\n"
        "其次，深入讨论 Monolithic Kernel 与 Micro Kernel 在学术研究历史中的演进脉络、"
        "性能权衡与可维护性争议，并对比 Mach、Minix、Linux、BSD 在内核架构选型上的历史抉择。\n"
        "最后，我们通过宏观概念层面，梳理进程抽象、虚拟地址空间、系统调用接口、VFS 抽象层、"
        "调度类与 CFS 算法的理论基础与设计动机。\n"
        "综上，内核设计的整体理论框架具有重要的参考价值。\n"
    )
    res = await agent.run(_make_req(profile=profile_practice, blocks=blocks_ok,
                                    content=pure_theory_content))
    err_types = [e.error_type for e in res.errors]
    per_types = [e.error_type for e in res.personalization_errors]
    print(f"[D] passed={res.passed}  errors={err_types}  personalization_errors={per_types}")
    # style_mismatch(major) 应该在 errors 里、且不在 personalization_errors 里
    has_blocking_style_in_errors = any(
        e.error_type == "style_mismatch" and e.severity in {"major", "critical"} and e.must_fix is True
        for e in res.errors
    )
    style_still_in_personalization = any(
        e.error_type == "style_mismatch" and e.severity in {"major", "critical"}
        for e in res.personalization_errors
    )
    assert res.passed is False, f"[D] 严重风格错配必须 passed=False"
    assert has_blocking_style_in_errors, f"[D] style_mismatch major 必须升级进入 errors、must_fix=True。errors={[(e.error_type,e.severity,e.must_fix) for e in res.errors]}"
    assert not style_still_in_personalization, f"[D] style_mismatch major 不应仍留在 personalization_errors。per={res.personalization_errors}"
    print("[PASS][D] style_mismatch(major) 从 personalization 通道升级为 hallucination/主errors 通道，passed=False + must_fix=True。")

    # --- Case E (轻度教学建议不阻断：mixed 用户 + 轻度不匹配 + 难度建议) ---
    async def _minor_diff(*a, **kw):
        # 产生一个 minor 级别的 difficulty_analyzer 建议，但不该阻断
        return [{
            "error_type": "difficulty_adjustment",
            "severity": "minor",
            "location": "段落 2",
            "description": "部分术语可补充基础释义",
            "suggestion": "补充 1~2 条通俗解释",
        }]
    agent.difficulty_analyzer_tool.analyze = _minor_diff

    profile_mixed = UserProfile(session_id="sess-unit", learning_style="mixed", skill_tags=[])
    res = await agent.run(_make_req(profile=profile_mixed, blocks=blocks_ok, content=content_ok))
    print(f"[E] passed={res.passed}  errors={[e.error_type for e in res.errors]}  "
          f"personalization_errors={[(e.error_type,e.severity,'must_fix='+str(e.must_fix)) for e in res.personalization_errors]}")
    # 仅有 minor 教学建议 → personalization 非空但 passed=True
    assert res.passed is True, f"[E] mild 建议不该阻断，passed={res.passed}"
    assert len(res.personalization_errors) > 0, f"[E] 应该保留 personalization_errors 以便 UI 呈现"
    for e in res.personalization_errors:
        assert e.must_fix is False, f"[E] mild 建议 must_fix=False。违反: {e.error_type}={e.must_fix}"
    print("[PASS][E] 轻度个性化建议进入 personalization_errors，不阻断发布。")


async def main() -> None:
    print("=" * 70)
    print("Agent4 ★ 立刻要做 落地验证 (tool-level + agent-level integration)")
    print("=" * 70)
    await tool_level_cases()
    await agent_level_cases()
    print("\n✅ ALL_CASES_PASSED: Agent4 ★ 立刻要做 全部落地验证通过。")


if __name__ == "__main__":
    asyncio.run(main())
