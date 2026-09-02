# -*- coding: utf-8 -*-
"""清理 SQLite 数据库中的个人运行时数据，保留公共知识库。

使用场景（方案 1 - 发布前清理）：
  1. 先备份正在使用的 project.db（你自己继续用备份）
  2. 对项目目录里的副本运行此脚本
  3. 清理后的副本可安全发给他人，不含你的学习记录 / 画像 / 对话 / 答题

默认作用于项目 data/sqlite_data/project.db；可通过 --db 指定其他路径。
--no-backup 跳过自动备份（默认始终先复制一份时间戳备份到同目录）。
"""
from __future__ import annotations


# ---------- 路径设置：python scripts\xxx.py 直接运行时能正确找到 backend / data ----------

import sys
from pathlib import Path
_PROJ_ROOT_ = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in sys.path: sys.path.insert(0, str(_PROJ_ROOT_))
_BE_ = Path(_PROJ_ROOT_, "backend")
if str(_BE_) not in sys.path: sys.path.insert(0, str(_BE_))

import sys as _sys_py
from pathlib import Path as _Path_py
_PROJ_ROOT_ = _Path_py(__file__).resolve().parent.parent
if str(_PROJ_ROOT_) not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_))
if str(_PROJ_ROOT_ / "backend") not in _sys_py.path:
    _sys_py.path.insert(0, str(_PROJ_ROOT_ / "backend"))
# -------------------------------------------------------------------------------------

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Windows 控制台中文输出兜底：若 stdout/stderr 无 reconfigure（Python 3.7+）
# 则强制 UTF-8，避免 GBK 终端报 UnicodeEncodeError 或乱码。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

# 公共知识库表 —— 严格保留，一行不动（FTS 虚表也归此类）
PUBLIC_TABLE_PREFIXES = (
    "knowledge_",
    "test_questions",
    "block_resources",
    "subtopic_resources",
    "learning_resources",
    "schema_migrations",
)

# 明确的个人运行时数据表（按 session / user / trace 归属）。
# 按"有外键依赖的先删子表"顺序排列，避免违反外键约束（SQLite 默认 FK 关闭，
# 但保持顺序仍便于未来启用 FK 时不出错）。
PERSONAL_TABLES = [
    # 子表：学习包 claim / source / 审核辩论记录（按 trace_id）
    "learning_package_claims",
    "learning_package_sources",
    "review_debate_responses",
    "review_debate_issues",
    "review_debate_rounds",
    "review_traces",
    # 答题 / QA
    "assessment_answers",
    "assessment_attempt_questions",
    "assessment_attempts",
    "user_answers",
    "user_qa",
    # 进度 / 状态
    "stage_progress",
    "subtopic_status",
    "user_subtopic_status",
    # 会话 / 画像
    "session_states",
    "user_profiles",
    "sessions",
    "learner_profiles",
]


def _collect_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def _is_public(table: str) -> bool:
    if any(table.startswith(pfx) for pfx in PUBLIC_TABLE_PREFIXES):
        return True
    # FTS 虚表及其影子表（knowledge_chunks_fts*）
    if table.startswith("knowledge_chunks_fts"):
        return True
    return False


def _report(conn: sqlite3.Connection, title: str) -> None:
    print(f"\n=== {title} ===")
    tables = _collect_tables(conn)
    total_rows = 0
    personal_rows = 0
    public_rows = 0
    for t in tables:
        n = _row_count(conn, t)
        total_rows += n
        tag = ""
        if _is_public(t):
            public_rows += n
            tag = " [公共-保留]"
        elif t in PERSONAL_TABLES:
            personal_rows += n
            tag = " [个人-清理]"
        else:
            tag = " [未知-跳过]"
        if n > 0 or (t in PERSONAL_TABLES and n == 0):
            print(f"  {t:45s} {n:>7d} 行{tag}")
    print(
        f"  合计: {total_rows} 行  (公共 {public_rows} / 个人 {personal_rows} / "
        f"未知 {total_rows - public_rows - personal_rows})"
    )


def _backup(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}.before_clean_{stamp}.bak")
    shutil.copy2(db_path, backup_path)
    print(f"已备份原数据库 -> {backup_path}")
    return backup_path


def main() -> int:
    parser = argparse.ArgumentParser(description="清理 project.db 的个人运行时数据")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="数据库路径")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="跳过备份（不推荐；默认始终先在同目录生成 .bak 备份）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印统计表，不实际删除任何数据",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过交互确认（给批处理 .bat 用）",
    )
    args = parser.parse_args()

    db_path = args.db.resolve()
    if not db_path.exists():
        print(f"[ERROR] 数据库不存在: {db_path}")
        return 2

    print(f"目标数据库: {db_path}")

    if not args.no_backup and not args.dry_run:
        _backup(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        # 清理前统计
        _report(conn, "清理前统计")

        existing_tables = set(_collect_tables(conn))
        to_delete: list[str] = [t for t in PERSONAL_TABLES if t in existing_tables]
        unknown_personal = [t for t in existing_tables if not _is_public(t) and t not in PERSONAL_TABLES]
        if unknown_personal:
            print(
                "\n[提示] 以下表未归类（不自动清理），如需加入清理列表请修改 PERSONAL_TABLES："
            )
            for t in unknown_personal:
                print(f"  - {t}  ({_row_count(conn, t)} 行)")

        if args.dry_run:
            print("\n--dry-run 模式，不做任何删除。上面是清理前统计。")
            return 0

        if not args.yes:
            answer = input(
                f"\n即将从 {db_path.name} 删除以上 {len(to_delete)} 张个人数据表中的全部记录，"
                f"公共知识库完全保留。继续? [y/N]: "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                print("已取消。")
                return 1

        print("\n开始删除...")
        cur = conn.cursor()
        # 优先使用 DELETE（保留表结构，FTS 虚表不会因 DROP 出问题）
        for table in to_delete:
            before = _row_count(cur, table)
            if before == 0:
                print(f"  - {table}: 已是空表，跳过")
                continue
            cur.execute(f'DELETE FROM "{table}"')
            after = cur.rowcount
            print(f"  - {table}: 删除 {after} 行（原为 {before} 行）")
        conn.commit()

        # VACUUM 回收空间（不能在事务内执行；PRAGMA auto_vacuum 默认关闭）
        print("\n执行 VACUUM 回收磁盘空间...")
        conn.close()
        conn = sqlite3.connect(str(db_path))
        # 显式关闭 WAL 模式并检查点，避免留下 -wal / -journal 临时文件锁定 db
        try:
            conn.execute("PRAGMA journal_mode = DELETE")
        except Exception:
            pass
        conn.execute("VACUUM")
        conn.commit()
        conn.close()

        size_before = args.db.stat().st_size if args.no_backup or args.dry_run else None
        size_after = db_path.stat().st_size
        print(f"数据库文件大小: {size_after / 1024 / 1024:.2f} MB")

        # 清理后统计单独打开，用完立刻关闭，避免 Windows 留下文件句柄
        report_conn = sqlite3.connect(str(db_path))
        try:
            _report(report_conn, "清理后统计")
        finally:
            report_conn.close()
    finally:
        # 无论异常路径如何，确保主连接已经关闭
        try:
            conn.close()
        except Exception:
            pass

    # 给 Windows 一点时间彻底释放文件句柄
    import time as _time
    _time.sleep(0.3)

    # 尝试验证是否还有 -wal / -journal 临时文件残留（有就说明还有进程占句柄）
    wal = db_path.with_suffix(db_path.suffix + "-wal")
    journal = db_path.with_suffix(db_path.suffix + "-journal")
    shm = db_path.with_suffix(db_path.suffix + "-shm")
    leftover = [p for p in (wal, journal, shm) if p.exists()]
    if leftover:
        print("\n[WARNING] 检测到 SQLite 临时文件残留，说明还有进程正在占用数据库：")
        for p in leftover:
            print(f"    - {p.name}")
        print("    如果需要手动删除 project.db，请先关闭所有后端进程和清理脚本窗口。")
        print("    任务管理器 → 详细信息 → 结束 python.exe / uvicorn.exe 后再试。")

    print("\n[DONE] 个人数据已清理，公共知识库完整保留。")
    print("       可安全打包发送给他人。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
