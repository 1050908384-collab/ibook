# -*- coding: utf-8 -*-
"""Part G: 快速回归验证（按 scripts/ 目录规范插入 sys.path，避免 import 失败）。"""
from __future__ import annotations
import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent  # repo root (scripts/../)
_BE_ = Path(_PROJ_ROOT_, "backend")
for _p in (str(_PROJ_ROOT_), str(_BE_)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
os.chdir(str(_BE_))

import asyncio, json, difflib

results = {}
failures = []

# ──────────────────────────────────────────────────────────────
# Test 1: 编译/导入所有修改过的模块（syntax check）
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 1: Import smoke tests")
try:
    from tools.agent1_tools.stage_generator_tool import _generate_summary, _generate_style_notes
    from agents.deterministic import DeterministicContentAgent
    from models.stage import ContentBlock, PersonalizationMeta
    from models.profile import (
        LEARNING_STYLE_THEORY, LEARNING_STYLE_PRACTICE, LEARNING_STYLE_MIXED,
        LearningStyle, normalize_learning_style, UserProfile, PracticeProfile,
    )
    from models.stage import StageInfo
    from orchestrator.state_manager import StateManager
    from orchestrator import harness as _harness_mod
    from db import sqlite_client as _sc_mod
    print("  [PASS] All modified modules import OK")
    results["imports"] = True
except Exception as exc:
    print(f"  [FAIL] Import error: {type(exc).__name__}: {exc}")
    import traceback; traceback.print_exc()
    failures.append(("imports", str(exc)))
    results["imports"] = False
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# Test 2: Part A consistency — theory/practice/mixed style produces matching labels in both notes & summary
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 2: Part A — _generate_style_notes / _generate_summary 一致性")
# (style_input, keyword_expected_in_notes, keyword_expected_in_summary)
#   For 'mixed' → notes intentionally says 混合型学习者（显示"混合"）summary says 综合型
#   (一个描述学习者，一个用于条目措辞)，两者归一化的风格值一致。
#   For aliases: reading / visual / auditory → theory兼容; kinesthetic → practice兼容
style_cases: list[tuple[object, str, str]] = [
    # (style, notes_must_contain, summary_must_contain)
    (LEARNING_STYLE_THEORY,  "理论型", "理论型"),
    (LEARNING_STYLE_PRACTICE, "动手型", "动手型"),
    (LEARNING_STYLE_MIXED,    "混合",   "综合型"),
    # 旧分类别名兼容（Part A + profile.py 的 keywords 扩展）
    ("reading",               "理论",   "理论型"),
    ("visual",                "理论",   "理论型"),
    ("auditory",              "理论",   "理论型"),
    ("kinesthetic",           "动手",   "动手型"),
    # 中文自由输入
    ("理论型",                 "理论型", "理论型"),
    ("动手",                   "动手",   "动手型"),
    ("综合",                   "混合",   "综合型"),
]
for style_val, exp_note, exp_sum in style_cases:
    n1 = _generate_style_notes(style_val, "nvidia-smi显存监控")
    s1 = _generate_summary("大一", 2.0, style_val, 3.5, 2)
    ok_n = exp_note in n1
    ok_s = exp_sum in s1
    passed = ok_n and ok_s
    print(f"  style={style_val!r:15s} notes[{exp_note:3s}]={ok_n} summary[{exp_sum:3s}]={ok_s} → {passed}")
    if not passed:
        failures.append((f"style_consistency[{style_val}]",
                         f"notes_has={ok_n} summary_has={ok_s} / notes={n1[:50]!r} / summary={s1[:70]!r}"))
results["A_consistency"] = len(failures) == 0

# ──────────────────────────────────────────────────────────────
# Test 3: Part D — Deterministic _personalize_chunk_text 三风格字符相似度 ≤0.70
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 3: Part D — tutorial similarity (theory vs practice ≤ 0.70 required)")

agent = DeterministicContentAgent()

SAMPLE_SOURCE = (
    "# 2.1 nvidia-smi 最简命令（①）\n\n"
    "**nvidia-smi**（NVIDIA System Management Interface）是 NVIDIA 官方提供的 GPU 系统管理接口，"
    "通过它可以**实时查看显卡的工作状态**。对训练与推理来说，这是你每天要打开的第一个监控窗口，"
    "也是排查 GPU 问题的第一道防线。\n\n"
    "## 为什么需要它\n\n"
    "在深度学习领域，GPU 显存（VRAM）是最稀缺的资源之一。训练前你需要确认：\n"
    "1. 显卡是否被正确识别；\n2. 当前有多少显存空闲；\n3. 有没有其他进程正在占用 GPU。\n\n"
    "如果你直接启动训练脚本而没有先检查 GPU 状态，常见的报错包括：\n"
    "- ``RuntimeError: CUDA out of memory. Tried to allocate ...``（最常见）\n"
    "- ``RuntimeError: No CUDA GPUs are available``（驱动/环境问题）\n"
    "- ``CUDA error: initialization error``（权限或 cuda 版本不匹配）\n\n"
    "## 最简命令\n\n"
    "在 Linux 终端里，输入下面这行命令，然后按回车：\n"
    "```bash\nnvidia-smi\n```\n\n"
    "就这么简单。执行后，屏幕上会输出类似下面的信息表：\n\n"
    "| 字段         | 含义                                                            |\n"
    "|-------------|------------------------------------------------------------------|\n"
    "| Driver Version | 本机安装的 NVIDIA 显卡驱动版本                                   |\n"
    "| CUDA Version | 该驱动支持的最高 CUDA 版本号                                     |\n"
    "| GPU Name    | 物理显卡型号                                                     |\n"
    "| Memory-Usage | 当前已使用显存 / 总显存                                          |\n"
    "| GPU-Util    | GPU 计算利用率（0% 表示空闲）                                   |\n\n"
    "## 常见报错与解决\n\n"
    "- **\"nvidia-smi: command not found\"**：说明 nvidia 驱动未正确安装。\n"
    "  修复：先执行 `lspci | grep -i nvidia` 确认硬件识别；再重装驱动。\n"
    "- **\"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver\"**：\n"
    "  通常是内核升级后驱动模块未加载。修复：重启机器或 `sudo modprobe nvidia`。\n"
)


def make_profile(style: LearningStyle, hours: float):
    # LEARNING_STYLE_* 常量是字符串字面量（"theory"/"practice"/"mixed"），
    # 不是枚举，没有 .value。normalize 返回的也是 str，直接使用即可。
    normalized: str = normalize_learning_style(style)
    up = UserProfile(
        profile_id="t1", session_id="sid-1", user_id="u1",
        goal="掌握Linux", grade="大一", learning_style=normalized,
        daily_hours=hours,
        skill_tags=[], hardware="联想拯救者Y7000P（RTX 4060，8G显存）",
    )
    pp = PracticeProfile.from_hardware_text("联想拯救者Y7000P（RTX 4060，8G显存）")
    return up, pp

stage = StageInfo(
    stage_id="st1", stage_num=1, block_id="KB-01",
    block_name="Linux环境与命令行操作",
    subtopic_name="nvidia-smi 显存监控",
    gap_skill="nvidia-smi 显存监控",
)

prof_t, prac_t = make_profile(LEARNING_STYLE_THEORY,    2.0)
prof_p, prac_p = make_profile(LEARNING_STYLE_PRACTICE, 3.0)
prof_m, prac_m = make_profile(LEARNING_STYLE_MIXED,    2.5)

out_t = agent._personalize_chunk_text(stage, SAMPLE_SOURCE, prof_t, prac_t)
out_p = agent._personalize_chunk_text(stage, SAMPLE_SOURCE, prof_p, prac_p)
out_m = agent._personalize_chunk_text(stage, SAMPLE_SOURCE, prof_m, prac_m)
sim_tp = difflib.SequenceMatcher(None, out_t, out_p).ratio()
sim_tm = difflib.SequenceMatcher(None, out_t, out_m).ratio()
sim_mp = difflib.SequenceMatcher(None, out_m, out_p).ratio()

tmp = _PROJ_ROOT_ / "tmp"
tmp.mkdir(exist_ok=True)
(tmp / "partd_theory.md").write_text(out_t, encoding='utf-8')
(tmp / "partd_practice.md").write_text(out_p, encoding='utf-8')
(tmp / "partd_mixed.md").write_text(out_m, encoding='utf-8')

print(f"  Theory   len = {len(out_t):5d}  → tmp/partd_theory.md")
print(f"  Practice len = {len(out_p):5d}  → tmp/partd_practice.md")
print(f"  Mixed    len = {len(out_m):5d}  → tmp/partd_mixed.md")
print()
print(f"  similarity(theory ↔ practice) = {sim_tp:.3f}  {'✅ ≤0.70 PASS' if sim_tp <= 0.70 else '❌ >0.70 FAIL'}")
print(f"  similarity(theory ↔ mixed)    = {sim_tm:.3f}")
print(f"  similarity(mixed ↔ practice)  = {sim_mp:.3f}")

results["D_similarity_tp_le_0p7"] = sim_tp <= 0.70
if sim_tp > 0.70:
    failures.append(("D_similarity_tp", f"{sim_tp:.3f} > 0.70"))

# 结构锚点检查
checks = []
def _must(cond, msg):
    checks.append((cond, msg))
_must("立即动手" in out_p or "最简操作" in out_p, "practice 版含『立即动手 / 最简操作』小标题")
_must("动手验证" in out_p, "practice 版含『动手验证』任务")
_must("概念与设计动机" in out_t, "theory 版含『概念与设计动机』")
_must("思考题" in out_t, "theory 版含『思考题』")
_must("术语" in out_t or "参数对比表" in out_t, "theory 版含『术语/参数对比表』")
_must("讲解段" in out_m or "混合推进" in out_m or "实操段" in out_m, "mixed 版含讲解段/实操段交替标记")
_must(len(out_p) >= 0.7 * len(SAMPLE_SOURCE), "practice 版非空且与原料长度同数量级")
_must(len(out_t) >= 0.7 * len(SAMPLE_SOURCE), "theory 版非空且与原料长度同数量级")
for ok, msg in checks:
    print(f"  {'✅' if ok else '❌'}  {msg}")
all_ok = all(ok for ok, _ in checks)
results["D_structure_anchors"] = all_ok
if not all_ok:
    failures.append(("D_structure_anchors",
                     ", ".join(msg for ok, msg in checks if not ok)))

# ──────────────────────────────────────────────────────────────
# Test 4: Part E — ContentBlock.personalization_meta OK after deterministic.generate_block
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 4: Part E — Deterministic generate_block fills personalization_meta correctly")
src = {"content": SAMPLE_SOURCE, "chunk_id": "ch001", "title": "nvidia-smi 最简命令",
       "heading_path": "Linux环境与命令行操作 / nvidia-smi 显存监控 / 2.1 最简命令",
       "block_id": "KB-01", "subtopic_id": "KBS-01-04", "doc_id": "doc-linux"}
async def run_dm_blk():
    return await agent.generate_block(
        stage, src, prof_p, prac_p, block_index=0, session_id="sid-regression-001",
    )
try:
    blk = asyncio.run(run_dm_blk())
    pm = blk.personalization_meta
    print(f"  meta.path             = {pm.path!r:15s} (expected 'deterministic')")
    print(f"  meta.llm_fail_reason  = {pm.llm_fail_reason!r}")
    print(f"  meta.style_applied    = {pm.style_applied!r} (expected 'practice')")
    ok_pm = pm.path == "deterministic" and pm.style_applied == LEARNING_STYLE_PRACTICE
    results["E_meta_deterministic"] = ok_pm
    if not ok_pm:
        failures.append(("E_meta", f"path={pm.path!r} style_applied={pm.style_applied!r}"))
    print(f"  PASS? {ok_pm}")
    # 同时确认 personalized_content 不再是原文 1:1 输出（应为 D 节重排后的版本）
    sim_vs_source = difflib.SequenceMatcher(None, blk.personalized_content, SAMPLE_SOURCE).ratio()
    print(f"  personalized_content vs SAMPLE_SOURCE similarity = {sim_vs_source:.3f}")
    if sim_vs_source > 0.92:
        print(f"  ⚠ WARNING: 个性化内容与原文过于相似 (>0.92)，D 节重排可能未生效。")
except Exception as exc:
    results["E_meta_run"] = False
    failures.append(("E_meta_run", f"{type(exc).__name__}: {exc}"))
    import traceback; traceback.print_exc()

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("REGRESSION SUMMARY")
for k, v in results.items():
    if isinstance(v, bool):
        tag = "✅" if v else "❌"
        print(f"  {tag}  {k:32s}  =  {v}")
    else:
        print(f"  -  {k:32s}  =  {v}")
if failures:
    print(f"\n❌  FAILURES ({len(failures)}):")
    for name, detail in failures:
        print(f"    * {name:<32s}  : {detail}")
    sys.exit(1)
else:
    print("\n✅  All inline regression checks PASSED.")
    sys.exit(0)
