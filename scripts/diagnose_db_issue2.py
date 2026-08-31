# -*- coding: utf-8 -*-
"""
诊断问题2：直接查数据库，对比两个Linux学习目标的content_blocks是否完全一样。
同时诊断问题1：查看tutorial的排版结构。
"""
import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def ngram_sim(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    a_set = set(a[i:i + n] for i in range(len(a) - n + 1))
    b_set = set(b[i:i + n] for i in range(len(b) - n + 1))
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def main():
    with engine.connect() as conn:
        # 1. 列出所有 sessions
        rows = conn.execute(text("""
            SELECT s.session_id, s.user_id, s.goal, s.status, s.current_stage,
                   s.created_at, s.updated_at
            FROM sessions s
            ORDER BY s.created_at DESC
        """)).fetchall()
        print("=" * 90)
        print(f"[DB] 共 {len(rows)} 个 sessions（学习目标）：")
        print("=" * 90)
        linux_sessions = []
        for r in rows:
            rid = dict(r._mapping)
            is_linux = "linux" in (rid.get("goal") or "").lower()
            marker = "  [LINUX目标]" if is_linux else ""
            print(f"  session_id={rid['session_id'][:12]}... "
                  f"status={rid['status']} stage={rid['current_stage']}"
                  f" goal={str(rid.get('goal'))[:60]}{marker}")
            if is_linux:
                linux_sessions.append(rid["session_id"])

        if len(linux_sessions) < 2:
            print(f"\n[WARN] Linux目标不足2个（找到{len(linux_sessions)}），无法对比。")
            return 1

        s1, s2 = linux_sessions[0], linux_sessions[1]
        print(f"\n对比对象：\n  S1 = {s1}\n  S2 = {s2}")

        # 2. 取两个 session 的 state_json，检查 learning_package
        results = {}
        for sid in (s1, s2):
            row = conn.execute(text("""
                SELECT state_json FROM session_states WHERE session_id = :sid
            """), {"sid": sid}).first()
            if not row:
                print(f"  [{sid[:12]}] session_states 为空！")
                continue
            state = json.loads(row[0])
            lp = state.get("learning_package") or {}
            cb = lp.get("content_blocks") or []
            tutorial = lp.get("tutorial") or ""
            profile = state.get("profile") or {}
            learning_style = profile.get("learning_style", "N/A")
            results[sid] = {
                "learning_style": learning_style,
                "tutorial_len": len(tutorial),
                "tutorial_head": tutorial[:500],
                "cb_count": len(cb),
                "cb_heading_paths": [b.get("heading_path") for b in cb] if isinstance(cb, list) else None,
                "cb_personalized_first300": [
                    (b.get("heading_path", "")[:40], (b.get("personalized_content") or "")[:300])
                    for b in (cb[:3] if isinstance(cb, list) else [])
                ],
                "raw_cb": cb,
            }

        # 3. 打印各自画像、标题前导
        print("\n" + "=" * 90)
        print("画像 & 学习包概览")
        print("=" * 90)
        for sid, info in results.items():
            print(f"\n[{sid[:12]}] learning_style={info['learning_style']}")
            print(f"  tutorial长度={info['tutorial_len']} 块数={info['cb_count']}")
            print(f"  tutorial前500字：")
            print("    " + info["tutorial_head"].replace("\n", "\n    "))

        # 4. content_blocks 个性化内容 相似度对比（逐块 & 整体）
        print("\n" + "=" * 90)
        print("逐块 personalized_content 相似度")
        print("=" * 90)
        r1, r2 = list(results.values())
        if not isinstance(r1["raw_cb"], list) or not isinstance(r2["raw_cb"], list):
            print("  [ERROR] content_blocks 格式异常，不是列表")
            return 2

        cb1 = r1["raw_cb"]
        cb2 = r2["raw_cb"]
        # 按 block_index 排序
        cb1_sorted = sorted(cb1, key=lambda b: b.get("block_index", 0) if isinstance(b, dict) else getattr(b, "block_index", 0))
        cb2_sorted = sorted(cb2, key=lambda b: b.get("block_index", 0) if isinstance(b, dict) else getattr(b, "block_index", 0))
        print(f"  S1块数={len(cb1_sorted)}  S2块数={len(cb2_sorted)}")

        # 找配对（heading_path匹配）
        hp1 = {b.get("heading_path") if isinstance(b, dict) else getattr(b, "heading_path", None): b for b in cb1_sorted}
        hp2 = {b.get("heading_path") if isinstance(b, dict) else getattr(b, "heading_path", None): b for b in cb2_sorted}
        common_hp = [h for h in hp1.keys() if h in hp2 and h]
        print(f"  共同heading_path数={len(common_hp)}")

        sims = []
        for hp in common_hp:
            b1 = hp1[hp]
            b2 = hp2[hp]
            pc1 = b1.get("personalized_content") if isinstance(b1, dict) else getattr(b1, "personalized_content", "")
            pc2 = b2.get("personalized_content") if isinstance(b2, dict) else getattr(b2, "personalized_content", "")
            sim = ngram_sim(pc1 or "", pc2 or "")
            sims.append(sim)
            marker = "  <<< 完全相同 >>>" if sim >= 0.98 else ("  [高度相似]" if sim >= 0.8 else "")
            print(f"  相似度={sim:.4f}  heading={str(hp)[:60]}{marker}")

        if sims:
            avg = sum(sims) / len(sims)
            mx = max(sims)
            print(f"\n  平均相似度={avg:.4f}  最大相似度={mx:.4f}")
            if avg >= 0.95:
                print("  结论：两个学习包的 content_blocks.personalized_content 基本完全相同！")
            elif avg >= 0.70:
                print("  结论：两个学习包高度相似（≥0.70），个性化不足。")
            else:
                print(f"  结论：两个学习包有明显差异（avg={avg:.4f} < 0.70）。")

        # 5. 整体 tutorial 相似度
        tut1 = r1["tutorial_head"] + ""
        tut2 = r2["tutorial_head"] + ""
        # 取整体前3000字
        state_s1 = json.loads(conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:s"), {"s": s1}).first()[0])
        state_s2 = json.loads(conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:s"), {"s": s2}).first()[0])
        full_tut1 = (state_s1.get("learning_package") or {}).get("tutorial", "")
        full_tut2 = (state_s2.get("learning_package") or {}).get("tutorial", "")
        tut_sim = ngram_sim(full_tut1[:5000], full_tut2[:5000])
        print(f"\n  整体tutorial相似度(前5000字)={tut_sim:.4f}")

        # 6. 画像对比：各自的 learning_style 是否真的不同？
        print("\n" + "=" * 90)
        print("画像学习风格核对（session级 user_profiles + session_states.profile）")
        print("=" * 90)
        for sid in (s1, s2):
            # session_states.profile
            st = json.loads(conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:s"), {"s": sid}).first()[0])
            sp = st.get("profile") or {}
            # user_profiles 会话级
            up_row = conn.execute(text("""
                SELECT learning_style, daily_hours, confidence, goal, version, updated_at
                FROM user_profiles WHERE session_id = :s ORDER BY version DESC LIMIT 1
            """), {"s": sid}).first()
            # learner_profiles 全局
            lp_row = conn.execute(text("""
                SELECT learning_style, grade, daily_hours, confidence_index
                FROM learner_profiles ORDER BY updated_at DESC LIMIT 1
            """)).first()
            print(f"\n[{sid[:12]}]")
            print(f"  session_states.profile.learning_style = {sp.get('learning_style')}")
            print(f"  user_profiles (session级 latest) learning_style = {up_row[0] if up_row else '无数据'} version={up_row[4] if up_row else '-'}")
            print(f"  learner_profiles (全局 latest) learning_style = {lp_row[0] if lp_row else '无数据'}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
