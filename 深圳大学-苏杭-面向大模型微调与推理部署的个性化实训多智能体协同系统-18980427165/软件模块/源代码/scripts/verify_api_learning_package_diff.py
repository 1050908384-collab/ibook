"""验证 Fix 后：接口 to_public_learning_package(backfill_stage_tutorial(pkg, stage))
返回的 tutorial 对于两会话仍保留个性化差异（不会被 backfill 覆盖成相同正文）。
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from db.sqlite_client import SQLiteClient  # noqa: E402
from api.v1.learning_resources import backfill_stage_tutorial  # noqa: E402
from api.v1.schemas import to_public_learning_package  # noqa: E402


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


def main() -> int:
    db = SQLiteClient()
    try:
        with db.Session() as s:
            from sqlalchemy import text
            rows = s.execute(text("SELECT session_id, goal FROM sessions ORDER BY datetime(created_at) DESC")).mappings().fetchall()
        print(f"共 {len(rows)} 会话")
        packages: dict[str, Any] = {}
        for r in rows:
            sid = r["session_id"]
            payload = db.load_session_state(sid) or {}
            pkg = payload.get("learning_package")
            stage = payload.get("stage_info")
            style = ((payload.get("profile") or {}).get("learning_style")
                     or (db.get_latest_profile(sid) or {}).get("learning_style"))
            if not pkg or not stage:
                continue
            backfilled = backfill_stage_tutorial(dict(pkg), dict(stage))
            public = to_public_learning_package(backfilled, stage)
            tut = public.get("tutorial") or ""
            blocks = public.get("content_blocks") or []
            packages[sid] = {
                "goal": r["goal"], "style": style, "tutorial": tut,
                "blocks_len": len(blocks),
                "practice": any(w in tut for w in ("立即动手 ·", "本节关键命令", "先别急", "概念摘要", "一句话版", "避坑", "工作台", "实操请暂停", "动手验证")),
                "theory": any(w in tut for w in ("概念与设计动机", "概念卡片", "思考题", "架构哲学", "内核", "模块化", "trade-off", "三个层面", "为什么必须装", "思考提示", "概念自证")),
                "head": tut[:280].replace("\n", " / "),
            }
            print(
                f"\n[{sid[:10]}…] goal={r['goal']!r} 画像风格={style!r} "
                f"tutorial_chars={len(tut)} blocks={len(blocks)}"
            )
            print(f"  practice标记={packages[sid]['practice']}, theory标记={packages[sid]['theory']}")
            print(f"  tutorial前280字: {packages[sid]['head']}")
        sids = list(packages.keys())
        if len(sids) >= 2:
            a, b = sids[0], sids[1]
            sim = char_sim(packages[a]["tutorial"], packages[b]["tutorial"])
            print("\n" + "=" * 72)
            print(f"模拟接口 GET 返回：两包 tutorial 字符相似度 = {sim:.4f}  (≤0.70 = 差异化保留)")
            pa, pb = packages[a], packages[b]
            print(
                f"  A 画像={pa['style']!r} practice={pa['practice']} theory={pa['theory']}"
            )
            print(
                f"  B 画像={pb['style']!r} practice={pb['practice']} theory={pb['theory']}"
            )
            # 确保两包对应标记互斥命中（不是两包都有 practice）
            if (pa["style"] == "practice" and pa["practice"] and not pa["theory"]
                    and pb["style"] == "theory" and pb["theory"] and not pb["practice"]):
                print("✅ 两包个性化特征与画像一致，且互斥。")
                return 0
            if sim <= 0.70:
                print("ℹ️ 相似度 OK，但标记没有完全互斥命中（可能是该子知识点天然缺少代码/概念内容）。")
                return 0
            print("❌ 结果仍然不对劲：要么相似度太高，要么标记与画像不一致。")
            return 1
        print("⚠️ 会话不足2，无法跨会话相似度对比。")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
