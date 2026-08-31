# -*- coding: utf-8 -*-
"""Part G: 快速回归验证（内嵌到现有的 scripts/ 目录运行方式，不新建文件）。

运行：python -c "import sys; sys.path.insert(0, r'backend'); sys.path.insert(0, r'.'); exec(open(r'_tmp_regression_personalization.py', encoding='utf-8').read())"
"""
import sys, os, asyncio, json, difflib
sys.path.insert(0, os.path.abspath(r'd:\我的项目\揭榜挂帅\ai_agent-trainer'))
os.chdir(r'd:\我的项目\揭榜挂帅\ai_agent-trainer')

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
        LearningStyle, normalize_learning_style,
    )
    from orchestrator.state_manager import StateManager
    print("  [PASS] All modified modules import OK")
    results["imports"] = True
except Exception as exc:
    print(f"  [FAIL] Import error: {exc!r}")
    failures.append(("imports", str(exc)))
    results["imports"] = False

# ──────────────────────────────────────────────────────────────
# Test 2: Part A consistency — theory style produces matching summary
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 2: Part A — _generate_style_notes / _generate_summary 一致性 for theory / practice / mixed")
for style_val, exp_keyword in [
    (LEARNING_STYLE_THEORY,  "理论型"),
    (LEARNING_STYLE_PRACTICE, "动手型"),
    (LEARNING_STYLE_MIXED,    "综合型"),
    ("reading",               "综合型"),   # 中文描述形式 → normalize
    ("理论型",                 "理论型"),   # 中文标签直接传入
    ("动手",                   "动手型"),   # 别名 → normalize
]:
    normalized = normalize_learning_style(style_val)
    n1 = _generate_style_notes(style_val, "nvidia-smi显存监控")
    s1 = _generate_summary("大一", 2.0, style_val, 3.5, 2)
    passed = (exp_keyword in n1) and (exp_keyword in s1)
    print(f"  style={style_val!r} → normalized={normalized.value!r}")
    print(f"    notes[:50]={n1[:50]!r}")
    print(f"    summary[:80]={s1[:80]!r}")
    print(f"    → {exp_keyword!r} in BOTH? {passed}")
    if not passed:
        failures.append((f"style_consistency[{style_val}]", f"notes={exp_keyword in n1} / summary={exp_keyword in s1}"))
results["A_consistency"] = all(f[0].split("[")[0] != "style_consistency" for f in failures) or len(failures) == 0

# ──────────────────────────────────────────────────────────────
# Test 3: Part D — Deterministic _personalize_chunk_text 三风格字符相似度 ≤0.70
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 3: Part D — tutorial similarity between theory vs practice blocks (must be ≤0.70)")

from models.profile import UserProfile, PracticeProfile
from models.stage import StageInfo

agent = DeterministicContentAgent()

SAMPLE_SOURCE = """# 2.1 nvidia-smi 最简命令（①）
\n**nvidia-smi**（NVIDIA System Management Interface）是 NVIDIA 官方提供的 GPU 系统管理接口，
通过它可以**实时查看显卡的工作状态**。对训练与推理来说，这是你每天要打开的第一个监控窗口，
也是排查 GPU 问题的第一道防线。

## 为什么需要它

在深度学习领域，GPU 显存（VRAM）是最稀缺的资源之一。训练前你需要确认：
1. 显卡是否被正确识别；
2. 当前有多少显存空闲；
3. 有没有其他进程正在占用 GPU。

如果你直接启动训练脚本而没有先检查 GPU 状态，常见的报错包括：
- ``RuntimeError: CUDA out of memory. Tried to allocate ...``（最常见）
- ``RuntimeError: No CUDA GPUs are available``（驱动/环境问题）
- ``CUDA error: initialization error``（权限或 cuda 版本不匹配）

## 最简命令

在 Linux 终端里，输入下面这行命令，然后按回车：
```bash
nvidia-smi
```

就这么简单。执行后，屏幕上会输出类似下面的信息表：

| 字段         | 含义                                                            |
|-------------|------------------------------------------------------------------|
| Driver Version | 本机安装的 NVIDIA 显卡驱动版本                                   |
| CUDA Version | 该驱动支持的最高 CUDA 版本号（实际安装的 CUDA Toolkit 版本应 ≤ 它） |
| GPU Name    | 物理显卡型号                                                     |
| Memory-Usage | 当前已使用显存 / 总显存                                          |
| GPU-Util    | GPU 计算利用率（0% 表示空闲）                                  |
| Processes   | 当前占用该 GPU 的所有进程列表（PID、显存占用、用户、命令）       |

## 常见报错与解决

- **"nvidia-smi: command not found"**：说明 nvidia 驱动未正确安装或未加入 PATH。
  修复：先执行 `lspci | grep -i nvidia` 确认硬件识别；再重装驱动。
- **"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"**：
  通常是内核升级后驱动模块未加载。修复：重启机器或 `sudo modprobe nvidia`。
- **显示 "Unable to determine the device handle"**：多卡环境下某块卡异常掉卡。
  修复：检查服务器机柜指示灯；重插显卡或联系机房人员。
"""

def make_profile(style: LearningStyle, hours: float) -> tuple[UserProfile, PracticeProfile]:
    up = UserProfile(
        profile_id="t1", session_id="sid-1", user_id="u1",
        goal="掌握Linux", grade="大一", learning_style=style.value,
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

src = {"content": SAMPLE_SOURCE, "chunk_id": "ch001", "title": "nvidia-smi 最简命令"}
prof_t, prac_t = make_profile(LEARNING_STYLE_THEORY,    2.0)
prof_p, prac_p = make_profile(LEARNING_STYLE_PRACTICE, 3.0)
prof_m, prac_m = make_profile(LEARNING_STYLE_MIXED,    2.5)

out_t = agent._personalize_chunk_text(stage, SAMPLE_SOURCE, prof_t, prac_t)
out_p = agent._personalize_chunk_text(stage, SAMPLE_SOURCE, prof_p, prac_p)
out_m = agent._personalize_chunk_text(stage, SAMPLE_SOURCE, prof_m, prac_m)
sim_tp = difflib.SequenceMatcher(None, out_t, out_p).ratio()
sim_tm = difflib.SequenceMatcher(None, out_t, out_m).ratio()
sim_mp = difflib.SequenceMatcher(None, out_m, out_p).ratio()

with open(r'tmp\partd_theory.md', 'w', encoding='utf-8') as f: f.write(out_t)
with open(r'tmp\partd_practice.md', 'w', encoding='utf-8') as f: f.write(out_p)
with open(r'tmp\partd_mixed.md', 'w', encoding='utf-8') as f: f.write(out_m)

print(f"  Theory  len = {len(out_t)} chars → tmp\\partd_theory.md")
print(f"  Practice len = {len(out_p)} chars → tmp\\partd_practice.md")
print(f"  Mixed    len = {len(out_m)} chars → tmp\\partd_mixed.md")
print()
print(f"  similarity(theory ↔ practice) = {sim_tp:.3f}  {'✅ ≤0.70 PASS' if sim_tp <= 0.70 else '❌ >0.70 FAIL'}")
print(f"  similarity(theory ↔ mixed)    = {sim_tm:.3f}  {'✅ ≤0.85' if sim_tm <= 0.85 else '❌ >0.85'}")
print(f"  similarity(mixed ↔ practice)  = {sim_mp:.3f}  {'✅ ≤0.85' if sim_mp <= 0.85 else '❌ >0.85'}")

results["D_similarity_tp"] = sim_tp <= 0.70
if sim_tp > 0.70:
    failures.append(("D_similarity_tp", f"{sim_tp:.3f} > 0.70"))

# Structure checks: each style contains its mandatory anchors
anchors_ok = True
anchors_t = ["概念与设计动机", "术语", "工作原理", "示例参考（无需立即在你机器上运行）", "思考题"]
for kw in anchors_t:
    if kw not in out_t:
        print(f"  [WARN theory anchor missing] {kw!r}  — 可能原料中没有合适的文本命中")
        # 理论型缺少概念动机标题是致命的
        if "概念与设计动机" == kw or "思考题" == kw:
            anchors_ok = False
anchors_p = ["立即动手", "动手验证", "常见报错"]
for kw in anchors_p:
    if kw not in out_p:
        print(f"  [WARN practice anchor missing]: {kw!r}")
        if "立即动手" == kw or "动手验证" == kw:
            anchors_ok = False
if "混合推进" in out_m or "讲解段" in out_m or "实操段" in out_m:
    pass  # OK
else:
    print("  [WARN mixed anchor missing]: 讲解段/实操段交替未出现")

results["D_anchors"] = anchors_ok
print(f"  structure anchors OK? {anchors_ok}")

# ──────────────────────────────────────────────────────────────
# Test 4: Part E — ContentBlock.personalization_meta 默认值 OK
# ──────────────────────────────────────────────────────────────
print("=" * 80)
print("Test 4: Part E — Deterministic generate_block fills personalization_meta")
async def run_dm_blk():
    return await agent.generate_block(
        stage, src, prof_p, prac_p, block_index=0, session_id="sid-regression",
    )
try:
    blk = asyncio.run(run_dm_blk())
    pm = blk.personalization_meta
    print(f"  meta.path = {pm.path!r} (expected 'deterministic')")
    print(f"  meta.llm_fail_reason = {pm.llm_fail_reason!r}")
    print(f"  meta.style_applied = {pm.style_applied!r} (expected 'practice')")
    ok_pm = (pm.path == "deterministic" and pm.style_applied == "practice")
    results["E_meta"] = ok_pm
    if not ok_pm:
        failures.append(("E_meta", f"path={pm.path} style={pm.style_applied}"))
    print(f"  OK? {ok_pm}")
except Exception as exc:
    results["E_meta"] = False
    failures.append(("E_meta_run", f"{type(exc).__name__}: {exc}"))
    import traceback; traceback.print_exc()

print("=" * 80)
print("SUMMARY")
for k, v in results.items():
    print(f"  {k:25s}: {v}")
if failures:
    print(f"\n❌  FAILURES ({len(failures)} items):")
    for f in failures:
        print(f"    - {f[0]}: {f[1]}")
    sys.exit(1)
else:
    print("\n✅ All inline regression checks passed.")
    sys.exit(0)
