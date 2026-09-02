"""验证 hypothesis: backfill_stage_tutorial 是否把个性化 tutorial 覆盖回原文通用版。

用真实 DB 数据：session b87e（theory）和 a20c（practice）调用 backfill_stage_tutorial，
比较 before/after tutorial 内容、相似度、标记。
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from db.sqlite_client import SQLiteClient  # noqa: E402
from api.v1.learning_resources import backfill_stage_tutorial  # noqa: E402


def char_sim(a: str, b: str, n: int = 3) -> float:
    def ngram(s: str):
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


def markers(s: str) -> Dict[str, Any]:
    return {
        "practice口吻": any(w in s for w in ("立即动手 ·", "本节关键命令", "先别急", "概念摘要", "一句话版", "避坑", "工作台", "实操请暂停")),
        "theory口吻": any(w in s for w in ("概念与设计动机", "概念卡片", "思考题", "架构哲学", "内核", "模块化", "trade-off", "三个层面", "为什么必须装")),
        "代码块": s.count("```") >= 2,
    }


def main() -> int:
    db = SQLiteClient()
    try:
        with db.Session() as s:
            from sqlalchemy import text
            rows = s.execute(text("""
                SELECT session_id, goal FROM sessions ORDER BY datetime(created_at) DESC
            """)).mappings().fetchall()
        print(f"会话数 = {len(rows)}")
        for r in rows:
            sid = r["session_id"]
            payload = db.load_session_state(sid) or {}
            pkg = payload.get("learning_package")
            stage = payload.get("stage_info")
            if not pkg or not stage:
                continue
            style = ((payload.get("profile") or {}).get("learning_style")
                     or db.get_latest_profile(sid).get("learning_style") if db.get_latest_profile(sid) else None)
            before = str(pkg.get("tutorial") or "")
            before_mark = markers(before)
            after_pkg = backfill_stage_tutorial(dict(pkg), dict(stage))
            after = str(after_pkg.get("tutorial") or "")
            after_mark = markers(after)
            sim_before_vs_after = char_sim(before, after)
            print("\n" + "=" * 72)
            print(f"Session {sid[:10]}… goal={r['goal']!r} 画像学习风格={style!r}")
            print(f"  BEFORE tutorial chars={len(before)}  markers={before_mark}")
            print(f"  AFTER  tutorial chars={len(after)}  markers={after_mark}")
            print(f"  BEFORE vs AFTER tutorial 字符相似度 = {sim_before_vs_after:.4f}  (<0.85 = 被改写)")
            if sim_before_vs_after < 0.85:
                print(f"  ⚠️  BACKFILL_COVERED: 个性化 tutorial 被 backfill_stage_tutorial 覆盖回原文！")
                print(f"     BEFORE 前 260 字: {before[:260].replace(chr(10),' / ')}")
                print(f"     AFTER  前 260 字: {after[:260].replace(chr(10),' / ')}")
                # 解释为什么：subtopic_names vs ## 正则命中情况
                names = list(stage.get("subtopic_names") or [])
                print(f"     stage.subtopic_names 前 3 = {names[:3]}")
                print(f"     tutorial 中实际出现的前 5 行 ## 标题：")
                for line in before.splitlines()[:50]:
                    if line.startswith("##"):
                        print(f"       {line[:120]}")
            else:
                print(f"  ✅ 保持原样，未被覆盖")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
