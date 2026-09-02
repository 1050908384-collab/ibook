# -*- coding: utf-8 -*-
"""清理评测运行时产生的个人数据（user 级画像 + 掌握记录 + sessions）。

用法: python scripts/eval_utils/clean_personal_data.py

保留: 知识库、知识图谱、向量库、测试题库
清除: sessions, learner_profiles, user_subtopic_status, subtopic_status,
      session_states, stage_progress, user_answers, assessment_answers,
      assessment_attempts, review_traces 等运行时表
"""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

# 只清空运行时数据，绝不碰知识侧表
RUNTIME_TABLES = [
    "sessions",
    "learner_profiles",
    "user_subtopic_status",
    "subtopic_status",
    "session_states",
    "stage_progress",
    "user_answers",
    "assessment_answers",
    "assessment_attempts",
    "assessment_attempt_questions",
    "user_qa",
    "review_traces",
    "review_debate_issues",
    "review_debate_responses",
    "review_debate_rounds",
    "learning_package_claims",
    "learning_package_sources",
    "verified_claims",
    "user_profiles",
]


def main():
    if not DB_PATH.exists():
        print(f"[CLEAN] 数据库不存在: {DB_PATH}")
        return 1

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # 清空前的计数
    print("[CLEAN] 清理前:")
    for t in RUNTIME_TABLES:
        try:
            cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            if cnt:
                print(f"  {t}: {cnt} 行")
        except sqlite3.OperationalError:
            pass

    for t in RUNTIME_TABLES:
        try:
            cur.execute(f"DELETE FROM {t}")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    print("\n[CLEAN] 清理后:")
    all_zero = True
    for t in RUNTIME_TABLES:
        try:
            cnt = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            mark = "✅" if cnt == 0 else "⚠️"
            if cnt:
                all_zero = False
            print(f"  {t}: {cnt} {mark}")
        except sqlite3.OperationalError:
            print(f"  {t}: (表不存在，跳过)")

    conn.close()
    if all_zero:
        print("\n[CLEAN] ✅ 所有运行时数据已清空，知识库保留完整。")
        return 0
    else:
        print("\n[CLEAN] ⚠️ 仍有残留数据未清除，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
