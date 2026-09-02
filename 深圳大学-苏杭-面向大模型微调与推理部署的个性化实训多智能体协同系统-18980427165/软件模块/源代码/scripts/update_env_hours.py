"""
迁移脚本：将所有"环境准备"知识点的学习时长从 0h 更新为 0.5h
========================================================================

变更范围：
1. seed 文件（knowledge_blocks.json / knowledge_subtopics.json）
2. SQLite 数据库（knowledge_blocks / knowledge_subtopics 表）—— 同时更新两个 project.db
3. （随后由 build_knowledge_graph.py 重建图谱 nodes.jsonl / edges.jsonl / runtime pickle）

设计原则：
- 仅修改 estimated_hours 与 description 字段，不触碰 prerequisites / sequence_index / 关系结构
- 仅修改 sequence_index=0 且 estimated_hours=0.0 的"环境准备"subtopic（共 11 个）
- 所属 block 的 estimated_hours 同步 +0.5h，description 内对应行 -0h 替换为 -0.5h
- 修改前自动备份原文件，可随时回滚
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

import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# ---------- 配置 ----------
ROOT_DIR = Path(r"d:\19137\学习\项目\学习助手\ai_agent-trainer")
SEEDS_DIR = ROOT_DIR / "data" / "seeds"
BLOCKS_SEED = SEEDS_DIR / "knowledge_blocks.json"
SUBTOPICS_SEED = SEEDS_DIR / "knowledge_subtopics.json"

# 两个 project.db 都要更新（避免工作目录不同导致读到旧库）
DB_PATHS = [
    ROOT_DIR / "data" / "sqlite_data" / "project.db",
    ROOT_DIR / "backend" / "data" / "sqlite_data" / "project.db",
]

NEW_ENV_HOURS = 0.5
TIMESTAMP_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def _backup(src: Path) -> None:
    if not src.exists():
        print(f"[SKIP] 备份跳过（文件不存在）: {src}")
        return
    dst = src.with_suffix(src.suffix + f".before_hours_update_{TIMESTAMP_TAG}.bak")
    shutil.copy2(src, dst)
    print(f"[BACKUP] {src} -> {dst}")


def _load_json(path: Path) -> list | dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _is_env_subtopic(subtopic: dict) -> bool:
    """识别"环境准备"subtopic：sequence_index=0 且名称含"环境准备"。"""
    return (
        subtopic.get("sequence_index") == 0
        and "环境准备" in subtopic.get("subtopic_name", "")
    )


def _update_description_env_line(description: str) -> tuple[str, bool]:
    """将 description 中"环境准备"那一行的 -0h 替换为 -0.5h。"""
    if not description:
        return description, False
    lines = description.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if "环境准备" not in line:
            continue
        # 行尾的 -0h 替换为 -0.5h（避免影响行内其他 0h 片段）
        if line.rstrip().endswith("-0h"):
            lines[i] = line[:-3] + "-0.5h"
            changed = True
            break
    return "\n".join(lines), changed


def update_seed_files() -> dict:
    """更新 seed JSON 文件，返回变更统计。"""
    print("\n===== [Step 1] 更新 seed JSON 文件 =====")
    blocks = _load_json(BLOCKS_SEED)
    subtopics = _load_json(SUBTOPICS_SEED)

    env_subtopic_ids = {s["subtopic_id"] for s in subtopics if _is_env_subtopic(s)}
    affected_block_ids = {s["block_id"] for s in subtopics if _is_env_subtopic(s)}

    sub_count = 0
    for s in subtopics:
        if s["subtopic_id"] in env_subtopic_ids and s.get("estimated_hours") == 0.0:
            old = s["estimated_hours"]
            s["estimated_hours"] = NEW_ENV_HOURS
            print(f"  [SUBTOPIC] {s['subtopic_id']} {s['subtopic_name'][:30]}... {old} -> {NEW_ENV_HOURS}")
            sub_count += 1

    block_count = 0
    for b in blocks:
        if b["block_id"] not in affected_block_ids:
            continue
        old_h = b.get("estimated_hours", 0)
        b["estimated_hours"] = old_h + NEW_ENV_HOURS
        new_desc, desc_changed = _update_description_env_line(b.get("description", ""))
        b["description"] = new_desc
        print(f"  [BLOCK]    {b['block_id']} {b['block_name'][:30]}... {old_h} -> {b['estimated_hours']} (desc_changed={desc_changed})")
        block_count += 1

    _write_json(BLOCKS_SEED, blocks)
    _write_json(SUBTOPICS_SEED, subtopics)
    print(f"[OK] seed 文件已更新: blocks={block_count}, subtopics={sub_count}")
    return {
        "env_subtopic_ids": sorted(env_subtopic_ids),
        "affected_block_ids": sorted(affected_block_ids),
        "subtopic_changes": sub_count,
        "block_changes": block_count,
    }


def update_database(db_path: Path, env_subtopic_ids: list[str], affected_block_ids: list[str]) -> dict:
    """更新单个 SQLite 数据库文件。"""
    print(f"\n===== [Step 2] 更新数据库: {db_path} =====")
    if not db_path.exists():
        print(f"[SKIP] 数据库文件不存在: {db_path}")
        return {"subtopic_changes": 0, "block_changes": 0, "skipped": True}

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = OFF")
    cur = conn.cursor()

    # 更新 subtopic
    cur.execute(
        "UPDATE knowledge_subtopics SET estimated_hours = ? "
        "WHERE subtopic_id IN ({}) AND estimated_hours = 0.0".format(
            ",".join(["?"] * len(env_subtopic_ids))
        ),
        [NEW_ENV_HOURS] + list(env_subtopic_ids),
    )
    sub_count = cur.rowcount

    # 更新 block：estimated_hours += 0.5
    cur.execute(
        "UPDATE knowledge_blocks SET estimated_hours = estimated_hours + ? "
        "WHERE block_id IN ({})".format(",".join(["?"] * len(affected_block_ids))),
        [NEW_ENV_HOURS] + list(affected_block_ids),
    )
    block_count = cur.rowcount

    # 更新 block：description 内环境准备行的 -0h -> -0.5h
    cur.execute(
        "SELECT block_id, description FROM knowledge_blocks WHERE block_id IN ({})".format(
            ",".join(["?"] * len(affected_block_ids))
        ),
        list(affected_block_ids),
    )
    rows = cur.fetchall()
    desc_updated = 0
    for block_id, description in rows:
        if not description:
            continue
        new_desc, changed = _update_description_env_line(description)
        if changed:
            cur.execute(
                "UPDATE knowledge_blocks SET description = ? WHERE block_id = ?",
                (new_desc, block_id),
            )
            desc_updated += 1

    conn.commit()
    conn.close()
    print(f"[OK] 数据库已更新: subtopics={sub_count}, blocks={block_count}, descriptions={desc_updated}")
    return {
        "subtopic_changes": sub_count,
        "block_changes": block_count,
        "description_changes": desc_updated,
        "skipped": False,
    }


def verify_database(db_path: Path) -> None:
    """验证数据库内容。"""
    print(f"\n===== [VERIFY] {db_path} =====")
    if not db_path.exists():
        print(f"[SKIP] 文件不存在")
        return
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute(
        "SELECT subtopic_id, block_id, subtopic_name, estimated_hours, sequence_index "
        "FROM knowledge_subtopics WHERE sequence_index = 0 ORDER BY subtopic_id"
    )
    print("\n-- 环境准备类 subtopic（sequence_index=0）:")
    for row in cur.fetchall():
        flag = "OK" if row[3] == NEW_ENV_HOURS else "FAIL"
        print(f"  [{flag}] {row[0]} {row[1]} hours={row[3]} name={row[2][:50]}")

    cur.execute(
        "SELECT COUNT(*) FROM knowledge_subtopics WHERE sequence_index = 0 AND estimated_hours = 0.0"
    )
    zero_count = cur.fetchone()[0]
    print(f"\n-- 仍是 0h 的环境准备 subtopic 数量: {zero_count}（应为 0）")

    cur.execute(
        "SELECT block_id, block_name, estimated_hours FROM knowledge_blocks ORDER BY block_id"
    )
    print("\n-- 各 block 时长:")
    total = 0.0
    for row in cur.fetchall():
        print(f"  {row[0]} {row[1][:30]:<32} {row[2]}h")
        total += row[2]
    print(f"  {'合计':<35} {total}h")

    conn.close()


def main():
    print(f"环境准备 subtopic 时长迁移脚本 - {TIMESTAMP_TAG}")
    print(f"目标：将所有环境准备 subtopic 由 0h 改为 {NEW_ENV_HOURS}h\n")

    # 备份
    print("===== [Step 0] 备份原文件 =====")
    _backup(BLOCKS_SEED)
    _backup(SUBTOPICS_SEED)
    for db in DB_PATHS:
        _backup(db)

    # 更新 seed 文件
    result = update_seed_files()

    # 更新数据库（每个 project.db）
    for db_path in DB_PATHS:
        update_database(db_path, result["env_subtopic_ids"], result["affected_block_ids"])

    # 验证
    for db_path in DB_PATHS:
        verify_database(db_path)

    print("\n===== [Done] 迁移完成 =====")
    print("下一步：运行 build_knowledge_graph.py 重建知识图谱")
    print("  python scripts/build_knowledge_graph.py")


if __name__ == "__main__":
    sys.exit(main() or 0)
