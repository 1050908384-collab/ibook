# -*- coding: utf-8 -*-
"""查 schema 里是否有 learning_package_sources_legacy 的引用 + 所有表的 FK"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sqlalchemy import text
from db.sqlite_client import SQLiteClient

client = SQLiteClient()

with client.Session() as s:
    # 查 learning_package_sources_legacy 是否存在
    rows = s.execute(text("SELECT name FROM sqlite_master WHERE name LIKE '%legacy%' OR name LIKE '%sources_legacy%'")).fetchall()
    print(f"legacy tables: {rows}")

    # 查 learning_package_claims 的 schema（CREATE 语句）
    row = s.execute(text("SELECT sql FROM sqlite_master WHERE type='table' AND name='learning_package_claims'")).fetchone()
    print(f"\nlearning_package_claims CREATE:\n{row[0] if row else 'NOT FOUND'}")

    # 查所有触发器
    triggers = s.execute(text("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='trigger'")).fetchall()
    print(f"\ntriggers ({len(triggers)}):")
    for t in triggers:
        print(f"  {t[0]} on {t[1]}")
        print(f"    {t[2]}")

    # 查所有 FK 关系（PRAGMA foreign_key_list for each table）
    tables = s.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")).fetchall()
    print(f"\nFKs per table:")
    for (tbl,) in tables:
        fks = s.execute(text(f"PRAGMA foreign_key_list('{tbl}')")).fetchall()
        if fks:
            print(f"  {tbl}:")
            for fk in fks:
                print(f"    {fk}")
