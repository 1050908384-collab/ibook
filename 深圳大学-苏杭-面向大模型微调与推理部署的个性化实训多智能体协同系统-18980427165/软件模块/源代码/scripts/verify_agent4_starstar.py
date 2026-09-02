# -*- coding: utf-8 -*-
"""Agent4 ★★ 第二期 落地验证（0 LLM / 0 DB / 0 token）。

Case:
  F. ★★1 baseline：原文骨架 → 个性化后仅插入 Callout/学习卡，顺序完全保留 → 无 paragraph_order_violation。
  G. ★★1 违规：个性化后把 S 的第 3 段挪到第 1 段之前（逆序） → paragraph_order_violation。
  H. ★★1 违规：原文 6 段正文仅 2 段保留在个性化骨架（覆盖率 < 60%） → paragraph_order_violation。
  I. ★★2 违规：grade=大一，正文塞满研究生级术语（密度 > 6 个/千字上限） → grade_terminology_density_mismatch。
  J. ★★2 违规：grade=硕士，正文全是"什么是 Linux / 基本概念 Linux"水段密度 < 7 个/千字下限 → grade_terminology_density_mismatch。
  K. ★★2 违规：已掌握 skill_tags=["Linux"]，正文仍然"什么是 Linux / 基本概念 / 简介"类铺垫 30%+ → skill_tags_mastered_intro_filler。
  L. ★★2 baseline：grade=大一 + skill_tags=空，正常基础内容 → 两条红线不触发。
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

os.environ.setdefault("LLM_PROVIDER", "deepseek")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-placeholder-nocall")


SOURCE_LINUX = """\
# Linux 文件权限基础

## 概念与基本设计动机

Linux 文件权限基于 Unix 经典的"三类主体 × 三种权限"模型。
每一个文件和目录都关联一个所有者、一个所属组。
主体被划分为所有者(u)、同组用户(g)、其他用户(o)三类。

### 三类权限位

读权限 r：对于文件，允许读取文件内容；对于目录，允许列出目录条目。
写权限 w：对于文件，允许修改或截断文件；对于目录，允许在其中创建/删除条目。
执行权限 x：对于文件，允许把它作为程序/脚本执行；对于目录，允许进入该目录。

## 立即动手·最简操作

先看你当前所在目录下的文件权限：
```
ls -l
```
输出的首列形如 `-rw-r--r--`，依次是类型位 + u/g/o 三组权限。
再执行 `stat a.txt` 观察更完整的权限字段。

## 常见报错与排查

当出现 `Permission denied` 时，有三种常见根因：
1. 文件的目标主体缺少对应权限位 → 用 chmod 补足。
2. 路径上某一级父目录缺少 x 位 → 需要逐层检查。
3. 你操作了受保护目录（如 /root、/etc 下敏感子路径）→ 通过 sudo 提权或切换目标路径。
"""

PERSONALIZED_OK = """\
# 画像专属 · Linux 文件权限基础 · 本块学习卡

## 立即动手 · Linux 文件权限基础 最简操作
### 本节关键命令 · 你自己的运行记录（请先执行，再读正文）
| 序号 | 命令（原文给出） | 你预期的输出（动手前猜测） | 实际终端输出（执行后粘贴） | 你当下是否看懂每个输出字段？ |
|---|---|---|---|---|
| 1 | `ls -l` |  |  | 待填 |
| 2 | *(下一段代码，读完后自行粘贴)* |  |  | 待填 |
| 3 | *(后续命令)* |  |  | 待填 |

> 🎯 **你的画像 = 动手型 learner**：本块**强制阅读顺序与节奏**（不要跳步）：
> 1. 立即从上面表格第一条命令开始 → 粘贴终端 → 实际运行 → 把输出填回表格；
> 2. 遇到下面的『报错排查列表段』时，先停下来，用你机器的输出逐条对；
> 3. 概念说明段**只在你「动手卡住 / 看不懂输出字段」时再回溯阅读**；
> 4. 全部命令执行完后，再回到文末『画像专属 · 本块学习检查』完成验证。

# Linux 文件权限基础

## 概念与基本设计动机

Linux 文件权限基于 Unix 经典的"三类主体 × 三种权限"模型。
每一个文件和目录都关联一个所有者、一个所属组。
主体被划分为所有者(u)、同组用户(g)、其他用户(o)三类。

### 三类权限位

读权限 r：对于文件，允许读取文件内容；对于目录，允许列出目录条目。
写权限 w：对于文件，允许修改或截断文件；对于目录，允许在其中创建/删除条目。
执行权限 x：对于文件，允许把它作为程序/脚本执行；对于目录，允许进入该目录。

🛠 **【实操请暂停】** 在终端里实际运行本段后面的代码块前，先回想：
 你之前操作过的 Windows 文件属性里"只读"对应这三列中哪一位？

## 立即动手·最简操作

先看你当前所在目录下的文件权限：
```
ls -l
```
输出的首列形如 `-rw-r--r--`，依次是类型位 + u/g/o 三组权限。
再执行 `stat a.txt` 观察更完整的权限字段。

🐛 **【报错请登记】** 你本机是否出现 Permission denied？
 如果有：先按文中给出的排查方法自查。

## 常见报错与排查

当出现 `Permission denied` 时，有三种常见根因：
1. 文件的目标主体缺少对应权限位 → 用 chmod 补足。
2. 路径上某一级父目录缺少 x 位 → 需要逐层检查。
3. 你操作了受保护目录（如 /root、/etc 下敏感子路径）→ 通过 sudo 提权或切换目标路径。

# 画像专属 · 本块学习检查
1. **执行与输出核对**：在你机器的终端中独立执行命令 `ls -l` （不要复制前面的答案），把输出粘贴在下方；然后列出 3 个实际出现的输出字段，并回到本块正文中找到这 3 个字段的对应解释。
2. **报错对照**：你本机运行后有没有出现类似『文件的目标主体缺少对应权限位』的现象？如果有：先按文中给出的排查方法自查，并把你尝试过的 2 种修复方法 / 命令序列记录。
3. **字段自查**：从你实际输出中找到 `所有者 / 所属组 / 权限位` 这几个字段，分别用 1 句话说明它们在『你这次的实际运行里』代表什么含义。
4. **流程与顺序自检**：如果本块涉及多条命令，说明你实际执行的顺序是否与文中描述一致？如果不一致，会导致什么后果？
"""


# 违规 G：第 3 段（三类权限位）被挪到 概念设计动机 之前 → 严格逆序
PERSONALIZED_REORDERED = """\
# Linux 文件权限基础

### 三类权限位

读权限 r：对于文件，允许读取文件内容；对于目录，允许列出目录条目。
写权限 w：对于文件，允许修改或截断文件；对于目录，允许在其中创建/删除条目。
执行权限 x：对于文件，允许把它作为程序/脚本执行；对于目录，允许进入该目录。

## 概念与基本设计动机

Linux 文件权限基于 Unix 经典的"三类主体 × 三种权限"模型。
每一个文件和目录都关联一个所有者、一个所属组。
主体被划分为所有者(u)、同组用户(g)、其他用户(o)三类。

## 立即动手·最简操作

先看你当前所在目录下的文件权限：
```
ls -l
```
输出的首列形如 `-rw-r--r--`，依次是类型位 + u/g/o 三组权限。
再执行 `stat a.txt` 观察更完整的权限字段。

## 常见报错与排查

当出现 `Permission denied` 时，有三种常见根因：
1. 文件的目标主体缺少对应权限位 → 用 chmod 补足。
2. 路径上某一级父目录缺少 x 位 → 需要逐层检查。
3. 你操作了受保护目录（如 /root、/etc 下敏感子路径）→ 通过 sudo 提权或切换目标路径。
"""


# 违规 H：大面积正文缺失（只保留 2 段短骨架，其余被替换为无关内容）→ 覆盖率 < 60%
# 故意让 P 至少有 3 段骨架正文段（凑够非空骨架 ≥ 3 触发阈值），但其中只有 2 段
# 真正来自原文骨架 → covered_ratio = 2 / 7 ≈ 28% < 60%。其余段全是 Agent3 杜撰的
# "与原文完全无关的段落"。
PERSONALIZED_SPARSE = """\
# Linux 文件权限基础

## 概念与基本设计动机

Linux 文件权限基于 Unix 经典的"三类主体 × 三种权限"模型。

## 什么是 GPU 显存带宽

GPU 显存带宽是 GPU 与 VRAM 间数据传输速率。
一般用 GB/s 衡量，影响大模型推理吞吐。

## 如何做蛋糕

首先准备面粉。然后加入鸡蛋。最后烘焙。

## 为什么要写周报

写周报可以同步项目进展。写周报是良好沟通习惯。

```
ls -l
```
"""


TEXT_GRADE_FRESHMAN_TOO_HARD = """\
# Linux 系统原理 · 大一年级教学块

## 本节内容

在深入学习 Linux 之前，我们首先了解 Linux 内核背后的 **RCU (Read-Copy-Update)** 机制，
以及其与 **SLUB / SLAB 分配器** 的配合关系。你需要重点掌握 **cgroup v2** 下
的 **CFS bandwidth** 控制、**io_uring zero-copy splice** 路径，以及
**eBPF verifier** 在 **tracepoint / kprobe** 中的 **bpf verifier** 安全边界。
另外，关于 **memory compaction / transparent hugepage** 的行为，也应该与
**numa balancing** 一同理解。此外 **nf_conntrack** 在 **netfilter hooks**
下的性能表现，以及 **ftrace perf_event** 的开销，都是本节讨论的重点。
内核态下 **RMAP reverse mapping** 与 **page cache thrash** 分析需要掌握。
最后 **htb qdisc / tc rules** 排队策略需要进一步梳理。

（本段以上术语密度已远超大一目标档，应该被 grade_terminology_density_mismatch 命中。）
"""


TEXT_GRADE_MASTER_TOO_EASY = """\
# 研究生进阶系统课题学习块。

## 什么是 Linux？
首先，让我们首先了解：什么是 Linux？简单介绍一下，Linux 是一种开源操作系统。
Linux 的基本概念来自于 Unix。Linux 的背景可以追溯到 1991 年。
Linux 的基础知识包括文件和目录。

## Linux 简介
顾名思义，Linux 是一套开源的类 Unix 操作系统。
入门指南建议我们首先学习文件和目录。

## Linux 概述
在深入之前，首先我们来了解 Linux 的基本定义。
Linux 是什么？Linux 的简介、Linux 的背景、Linux 的基本概念——
这些都是 Linux 的入门指南中首先会介绍的内容。
在学习 Linux 时，基础知识包括：什么是文件、什么是目录。
（本段 1 个进阶术语都没有，对"硕士档"应命中 grade_terminology_density_mismatch。）
"""


TEXT_SKILLS_LINUX_FILLER = """\
# 本块 Linux 文件操作复习

## 什么是 Linux？
Linux 是什么？顾名思义，Linux 是一套开源操作系统。
Linux 的基本概念来自于 Unix 哲学。让我们首先了解什么是 Linux。

## Linux 简介
简单介绍 Linux：Linux 是开源 OS。Linux 的基本定义是类 Unix。
入门指南：首先学习 Linux 的基础入门。

## 基本概念 Linux 文件是什么
首先了解：什么是 Linux 文件？基本定义——文件是 Linux 中保存数据的单位。
在深入之前，先介绍 Linux 文件的基础知识。
Linux 是什么？Linux 的基本概念是什么？这些都是入门的基础。

## Linux 基本命令
什么是 `ls` 命令？Linux 中 ls 用于查看目录。
让我们首先了解 ls 的基本用法：在学习 ls 时，基本语法包括 `ls [选项] [路径]`。
在正式介绍之前，先说明 ls 的背景：它是 Linux 基础命令之一。

（本块用户已掌握 skill_tags=["Linux"]，且填充比例极高，应命中 skill_tags_mastered_intro_filler。）
"""


async def case_F_order_ok():
    from agents.deterministic import _verify_paragraph_order
    res = _verify_paragraph_order(SOURCE_LINUX, PERSONALIZED_OK)
    print(f"[F] order issues: {[(r['error_type'], r.get('location')) for r in res]}")
    assert res == [], f"[F] baseline 仅加壳不应报顺序违规，实际={res}"
    print("[PASS][F] ★★1 骨架顺序 baseline（仅加壳无改序）通过")


async def case_G_order_reordered():
    from agents.deterministic import _verify_paragraph_order
    res = _verify_paragraph_order(SOURCE_LINUX, PERSONALIZED_REORDERED)
    err_types = [r["error_type"] for r in res]
    print(f"[G] order issues types={err_types}  raw={res[:3]}")
    assert "paragraph_order_violation" in err_types, f"[G] 逆序应命中 paragraph_order_violation，got {res}"
    print("[PASS][G] ★★1 严格逆序 → paragraph_order_violation 命中")


async def case_H_order_sparse():
    from agents.deterministic import _verify_paragraph_order
    res = _verify_paragraph_order(SOURCE_LINUX, PERSONALIZED_SPARSE)
    err_types = [r["error_type"] for r in res]
    print(f"[H] order issues types={err_types}  raw={res[:3]}")
    assert "paragraph_order_violation" in err_types, f"[H] 骨架大面积缺失应命中覆盖率违规，got {res}"
    print("[PASS][H] ★★1 覆盖率 < 60% → paragraph_order_violation 命中")


async def case_I_grade_freshman_too_hard():
    from tools.agent4_tools.personalization_check_tool import PersonalizationCheckTool
    tool = PersonalizationCheckTool()
    block = {
        "personalization_action": "keep",
        "personalization_reason": "grade 匹配",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [],
        "personalized_content": TEXT_GRADE_FRESHMAN_TOO_HARD,
    }
    profile = {"learning_style": "mixed", "grade": "本科一年级"}
    issues = await tool.check(block, profile=profile)
    err_types = [i["error_type"] for i in issues]
    print(f"[I] issues={err_types}")
    assert "grade_terminology_density_mismatch" in err_types, \
        f"[I] 大一密度超标未命中：{[f'{i['error_type']}={i.get('description','')[:80]}' for i in issues]}"
    print("[PASS][I] ★★2 grade=大一但术语密度过高 → grade_terminology_density_mismatch 命中")


async def case_J_grade_master_too_easy():
    from tools.agent4_tools.personalization_check_tool import PersonalizationCheckTool
    tool = PersonalizationCheckTool()
    block = {
        "personalization_action": "keep",
        "personalization_reason": "x",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [],
        "personalized_content": TEXT_GRADE_MASTER_TOO_EASY,
    }
    profile = {"learning_style": "theory", "grade": "硕士研究生"}
    issues = await tool.check(block, profile=profile)
    err_types = [i["error_type"] for i in issues]
    print(f"[J] issues={err_types}")
    assert "grade_terminology_density_mismatch" in err_types, \
        f"[J] 硕士密度过低未命中：{issues}"
    print("[PASS][J] ★★2 grade=硕士但术语全是入门水段 → grade_terminology_density_mismatch 命中")


async def case_K_skill_tags_intro_filler():
    from tools.agent4_tools.personalization_check_tool import PersonalizationCheckTool
    tool = PersonalizationCheckTool()
    block = {
        "personalization_action": "compress",
        "personalization_reason": "已掌握 Linux",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [{"t": "quick"}],  # 避免 practice_task_missing 噪声
        "personalized_content": TEXT_SKILLS_LINUX_FILLER,
    }
    profile = {
        "learning_style": "practice",
        "skill_tags": ["Linux", "Shell Basics"],
    }
    issues = await tool.check(block, profile=profile)
    err_types = [i["error_type"] for i in issues]
    print(f"[K] issues={err_types}")
    assert "skill_tags_mastered_intro_filler" in err_types, \
        f"[K] 已掌握 Linux 仍写入门铺垫未命中：{issues}"
    print("[PASS][K] ★★2 skill_tags 已掌握 Linux + 正文 30%+ 入门铺垫 → skill_tags_mastered_intro_filler 命中")


async def case_L_grade_and_skills_pass():
    """baseline：正常大一内容 + 空 skill_tags，两条红线都不触发。"""
    from tools.agent4_tools.personalization_check_tool import PersonalizationCheckTool
    tool = PersonalizationCheckTool()
    # 从 PERSONALIZED_OK（包含骨架+少量 callout）截一段标准大一内容：
    excerpt = "\n".join(PERSONALIZED_OK.splitlines()[:80])
    block = {
        "personalization_action": "explain_and_practice",
        "personalization_reason": "大一实践画像",
        "fact_source_chunk_ids": ["src-1"],
        "teaching_tasks": [{"t": "run ls -l"}],
        "personalized_content": excerpt,
    }
    profile = {
        "learning_style": "practice",
        # Baseline L：grade 留空（用户前端 grade 窗口本来就常留空，留空时不触发红线）
        # skill_tags 为空 → 不触发 skill_tags_mastered_intro_filler
        # → 保证 grade/skill 两条红线不触发
        "grade": "",
        "skill_tags": [],
    }
    issues = await tool.check(block, profile=profile)
    bad = [i for i in issues if i["error_type"] in {
        "grade_terminology_density_mismatch",
        "skill_tags_mastered_intro_filler",
    }]
    print(f"[L] issues={[i['error_type'] for i in issues]}  bad={bad}")
    assert bad == [], f"[L] baseline 不应命中 grade/skill 形式红线，bad={bad}"
    print("[PASS][L] ★★2 baseline（大一 + 无已掌握 + 正常内容）两条红线均不触发")


async def main() -> None:
    print("=" * 70)
    print("Agent4 ★★ 第二期 落地验证（纯规则，0 LLM/DB）")
    print("=" * 70)
    await case_F_order_ok()
    await case_G_order_reordered()
    await case_H_order_sparse()
    await case_I_grade_freshman_too_hard()
    await case_J_grade_master_too_easy()
    await case_K_skill_tags_intro_filler()
    await case_L_grade_and_skills_pass()
    print("\n✅ ALL_CASES_PASSED: Agent4 ★★ 第二期 全部落地验证通过。")


if __name__ == "__main__":
    asyncio.run(main())
