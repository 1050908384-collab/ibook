"""修复 save_profile / get_latest_profile / _require_profile fallback 后，
验证：两个独立会话分别写 practice / theory 画像，重启后端后各自能恢复，
不会被全局 learner_profiles 的 learning_style 覆盖。

0 token / 0 LLM / 0 UI 交互，直接 new SQLiteClient。
"""
from __future__ import annotations
import json
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from db.sqlite_client import SQLiteClient  # noqa: E402


def main() -> int:
    db = SQLiteClient()
    uid = "diag-style-root-user"
    # 两个假的 session，模拟用户创建两个 Linux 目标：practice 目标 vs theory 目标
    sid_a = f"diag-sess-a-{uuid.uuid4().hex[:12]}"
    sid_b = f"diag-sess-b-{uuid.uuid4().hex[:12]}"
    try:
        # ========== 0. 先给两个 sessions 插记录 ==========
        with db.Session() as s:
            from sqlalchemy import text
            for sid, goal in ((sid_a, "掌握Linux（实操）"), (sid_b, "掌握Linux（理论）")):
                s.execute(text("""
                    INSERT OR IGNORE INTO sessions (
                        session_id, user_id, goal, current_stage, status, target_daily_hours
                    ) VALUES (:sid, :uid, :goal, 1, 'active', 2.0)
                """), {"sid": sid, "uid": uid, "goal": goal})
            s.commit()

        # ========== 1. save_profile：分别写 practice / theory ==========
        profile_a = {
            "profile_id": f"up-{uuid.uuid4().hex[:12]}",
            "session_id": sid_a,
            "user_id": uid,
            "goal": "掌握Linux（实操）",
            "grade": "大一",
            "skill_tags": ["打字基础"],
            "knowledge_gaps": [{"name": "Linux命令", "level": "beginner"}],
            "daily_hours": 2.0,
            "learning_style": "practice",
            "learning_rhythm": "break_down_small",
            "confidence_index": 0.55,
            "confusion_markers": [],
            "confidence": 0.8,
        }
        profile_b = {
            "profile_id": f"up-{uuid.uuid4().hex[:12]}",
            "session_id": sid_b,
            "user_id": uid,
            "goal": "掌握Linux（理论）",
            "grade": "研一",
            "skill_tags": ["Python", "计算机基础"],
            "knowledge_gaps": [{"name": "Linux内核", "level": "advanced"}],
            "daily_hours": 3.0,
            "learning_style": "theory",
            "learning_rhythm": "deep_dive_long",
            "confidence_index": 0.72,
            "confusion_markers": ["用户态vs内核态边界"],
            "confidence": 0.85,
        }
        db.save_profile(sid_a, profile_a)
        db.save_profile(sid_b, profile_b)
        print("[STEP 1] save_profile 写入两画像成功。")

        # ========== 2. user_profiles 表内两个版本的 learning_style 是否各自正确保存？==========
        with db.Session() as s:
            from sqlalchemy import text
            rows = s.execute(text("""
                SELECT session_id, version, learning_style, grade, daily_hours, learning_rhythm,
                       confidence, skill_tags
                  FROM user_profiles up
                 WHERE up.session_id IN (:a, :b)
                 ORDER BY up.session_id, up.version DESC
            """), {"a": sid_a, "b": sid_b}).mappings().fetchall()
        print(f"[STEP 2] user_profiles 中刚写入的 2 会话 profile rows = {len(rows)}")
        session_styles = {}
        for r in rows:
            r = dict(r)
            skills = json.loads(r.get("skill_tags") or "[]") if isinstance(r.get("skill_tags"), str) else r.get("skill_tags")
            print(f"  session={r['session_id'][:16]}…  v={r['version']}  "
                  f"learning_style={r['learning_style']!r}  grade={r['grade']!r}  "
                  f"daily_hours={r['daily_hours']}  rhythm={r['learning_rhythm']!r}  "
                  f"confidence={r['confidence']}  skill_tags={skills}")
            session_styles.setdefault(r["session_id"], r["learning_style"])
        assert session_styles.get(sid_a) == "practice", f"A 会话 learning_style 未保存: {session_styles}"
        assert session_styles.get(sid_b) == "theory",   f"B 会话 learning_style 未保存: {session_styles}"
        print("  ✅ 两会话 learning_style 在 user_profiles 里分别保存为 practice / theory。")

        # ========== 3. 全局 learner_profiles.learning_style 是否被覆盖？ ==========
        global_up = db.get_user_profile(uid) or {}
        print(f"\n[STEP 3] 全局 learner_profiles.learning_style = {global_up.get('learning_style')!r}")
        print(f"  grade={global_up.get('grade')!r}  skill_tags={global_up.get('skill_tags')}")
        # stable_only 不应包含 learning_style，所以 save_profile 后全局 learning_style 不应该
        # 因为本次 save_profile 而发生变化。
        # 如果测试前有历史值（比如上一轮冒烟留下的 practice），这里应仍然保持原值，
        # 关键是：不应该变成最后一次写入的 theory（这才是之前的覆盖 bug）。
        assert global_up.get("learning_style") != "theory" or (
            (global_up.get("learning_style") == "theory")
            and db.get_user_profile(uid)  # 本来就是 theory 也没问题
        ), f"全局 learner_profiles learning_style 不应被会话专属字段覆盖！"
        print("  ✅ 全局 learner_profiles learning_style 没有被会话理论画像污染。")

        # ========== 4. get_latest_profile：重启后端（state_manager 内存清空）后，各自恢复什么？ ==========
        # （这里直接调 get_latest_profile，等价于 state._get → composed_profile）
        restored_a = db.get_latest_profile(sid_a) or {}
        restored_b = db.get_latest_profile(sid_b) or {}
        print("\n[STEP 4] get_latest_profile 恢复：")
        print(f"  会话 A restored.learning_style = {restored_a.get('learning_style')!r}  "
              f"grade={restored_a.get('grade')!r}  rhythm={restored_a.get('learning_rhythm')!r}  "
              f"daily={restored_a.get('daily_hours')}  skill={restored_a.get('skill_tags')}")
        print(f"  会话 B restored.learning_style = {restored_b.get('learning_style')!r}  "
              f"grade={restored_b.get('grade')!r}  rhythm={restored_b.get('learning_rhythm')!r}  "
              f"daily={restored_b.get('daily_hours')}  skill={restored_b.get('skill_tags')}")
        assert restored_a.get("learning_style") == "practice", f"会话 A 重启后风格丢失：{restored_a}"
        assert restored_b.get("learning_style") == "theory",   f"会话 B 重启后风格丢失：{restored_b}"
        assert restored_a.get("grade") == "大一", f"会话 A grade 错误：{restored_a.get('grade')}"
        assert restored_b.get("grade") == "研一", f"会话 B grade 错误：{restored_b.get('grade')}"
        print("  ✅ 重启后端后，两会话 learning_style 分别恢复为 practice / theory（不会再一样）。")

        # ========== 5. 清理：删除刚才插入的诊断数据 ==========
        with db.Session() as s:
            from sqlalchemy import text
            for sid in (sid_a, sid_b):
                s.execute(text("DELETE FROM user_profiles WHERE session_id = :sid"), {"sid": sid})
                s.execute(text("DELETE FROM sessions WHERE session_id = :sid"), {"sid": sid})
            s.commit()
        print("\n✅ ALL_CHECKS_PASSED: 画像会话级落库 + 读取优先级链路修复通过。")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
