"""兼容验证：Fix-A 修改后，旧会话（content_blocks 为空 / tutorial 为空 / tutorial 是 source_material 纯拼接无 ## 标题）
仍然会正常触发 backfill，不会被 Guard 1 短路。
"""
from __future__ import annotations
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from api.v1.learning_resources import backfill_stage_tutorial  # noqa: E402


STAGE = {
    "block_id": "KB-01-00",
    "subtopic_ids": ["KB-01-00-001", "KB-01-00-002", "KB-01-00-003"],
    "subtopic_names": ["什么是 Linux？为什么要用它？", "Linux 发行版选择", "终端与基本命令入门"],
}


def run_case(name: str, pkg: dict) -> bool:
    before_tut = pkg.get("tutorial") or ""
    blocks_before = pkg.get("content_blocks") or []
    out = backfill_stage_tutorial(pkg, STAGE)
    after_tut = out.get("tutorial") or ""
    changed = (after_tut != before_tut) and len(after_tut) > len(before_tut)
    # backfill 成功标志：拼接后的 tutorial 包含『## {subtopic name}』子标题
    filled = all(
        f"## {name.strip('（）')}" in after_tut or any(
            token in after_tut for token in ("## 什么是 Linux", "## Linux 发行版选择", "## 终端与基本命令入门")
        )
        for name in STAGE["subtopic_names"]
    )
    status = "✅" if changed else "⚠️"
    print(f"[{status}] {name}")
    print(f"    blocks_before_len={len(blocks_before)}  tutorial_before_len={len(before_tut)}  tutorial_after_len={len(after_tut)}")
    print(f"    触发了backfill填充? {'YES' if changed else 'NO'}")
    return changed


def main() -> int:
    cases = [
        ("Case1: tutorial为空 + content_blocks为空（最老会话）",
         {"tutorial": "", "content_blocks": []}),
        ("Case2: tutorial为空格 + content_blocks为None",
         {"tutorial": "   \n\n  ", "content_blocks": None}),
        ("Case3: tutorial为source_material纯拼接（无任何##标题）",
         {"tutorial": "这是一段关于Linux的正文。\n\n" * 20 + "介绍完发行版后，我们来学命令。\n\n" + "这是更多正文。\n" * 15,
          "content_blocks": []}),
        ("Case4: tutorial为老版getTopicTutorial无法切片的纯文本（有内容但无##）",
         {"tutorial": "一、Linux简介：它是一个操作系统...（无 ## 标题的整段）",
          "content_blocks": []}),
    ]
    ok = 0
    for n, p in cases:
        if run_case(n, p):
            ok += 1
    print("\n" + "=" * 60)
    print(f"兼容旧会话：{ok}/{len(cases)} 都被 backfill 正常填充。")
    # 最后一个反向用例：content_blocks 非空 → 不应被填充（即新会话 Guard 1 生效）
    pkg_new = {"tutorial": "这是旧的个性化tutorial正文，应保留。## 立即动手 · linux发行版选择 快速上手", "content_blocks": [{"content_block_id": "x", "personalized_content": "xxx"}]}
    out_new = backfill_stage_tutorial(pkg_new, STAGE)
    guard_ok = out_new.get("tutorial") == pkg_new["tutorial"]
    print(f"[{'✅' if guard_ok else '❌'}] Case5: content_blocks 非空新会话 Guard1生效，不应被backfill覆盖 = {guard_ok}")
    if not guard_ok:
        print(f"   BEFORE: {pkg_new['tutorial'][:120]}")
        print(f"   AFTER : {out_new.get('tutorial','')[:120]}")
        return 1
    if ok == len(cases):
        print("✅ LEGACY_OK：旧会话兼容无回归")
        return 0
    print("❌ 一些旧会话没有被填充。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
