"""Mock 数据加载器：agent1-3 tool 开发期用的最小脚手架。

约定：
- mock 数据目录为 tests/test_mock_data/（文件夹名带 test 以便 git 跟踪且醒目）。
- DB 表 mock 文件名直接用 <table>.json（如 knowledge_blocks.json、test_questions.json），
  顶层是数组，字段对齐 backend/db/sqlite_client.py 的 CREATE TABLE
  （created_at 等带 DEFAULT 的字段省略）。
- 派生 mock（无对应 DB 表）顶层是对象：
  skill_prerequisites.json / templates.json / code_examples.json。
- 在 DB 里以 TEXT 存 JSON 的字段（prerequisites/corresponding_gaps/options），
  在 mock 里用原生 list；seed 时由本模块 json.dumps，tool 读出需 json.loads。
- mock/真实切换：env AGENT_TOOL_DATA_SOURCE=mock|db，默认 mock。
"""
import json
import os
from pathlib import Path

MOCK_DATA_DIR = Path(__file__).resolve().parent / "test_mock_data"

# 表 -> 需要在 seed 前 json.dumps 的字段（DB 中以 TEXT 存 JSON 的列）
_JSON_FIELDS = {
    "knowledge_blocks": ["prerequisites", "corresponding_gaps"],
    "test_questions": ["options"],
    "knowledge_docs": [],
    "knowledge_chunks": [],
}


def _path(name: str) -> Path:
    """文件名 = <name>.json。name 传表名或派生名（如 knowledge_blocks / test_questions / skill_prerequisites）。"""
    return MOCK_DATA_DIR / f"{name}.json"


def load_mock(name: str) -> list[dict]:
    """读 DB 表 mock（顶层是数组）。name 为表名（如 knowledge_blocks、test_questions）。"""
    with open(_path(name), encoding="utf-8") as f:
        return json.load(f)


def load_mock_dict(name: str, key_field: str) -> dict:
    """读 DB 表 mock 并按 key_field 索引成 {key: row}。"""
    return {row[key_field]: row for row in load_mock(name)}


def load_mock_named(name: str) -> dict:
    """读派生 mock（顶层是对象）。name 为派生名（如 skill_prerequisites、templates）。"""
    with open(_path(name), encoding="utf-8") as f:
        return json.load(f)


def seed_mock_to_sqlite(client, table_name: str) -> int:
    """把 mock 行写入 SQLiteClient 已建好的表。

    - INSERT OR IGNORE 跳过已存在 PK，可重复 seed。
    - TEXT 存 JSON 的字段自动 json.dumps。
    - 返回实际写入的行数（注意：INSERT OR IGNORE 静默跳过冲突，返回值为
      准备的行数；调用方应配合 assertGreater 判断是否真的插入）。
    """
    rows = load_mock(table_name)
    if not rows:
        return 0
    from sqlalchemy import text

    json_fields = set(_JSON_FIELDS.get(table_name, []))
    cols = list(rows[0].keys())
    col_sql = ", ".join(cols)
    param_sql = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT OR IGNORE INTO {table_name} ({col_sql}) VALUES ({param_sql})"

    prepared = []
    for row in rows:
        r = dict(row)
        for f in json_fields:
            if isinstance(r.get(f), (list, dict)):
                r[f] = json.dumps(r[f], ensure_ascii=False)
        prepared.append(r)

    with client.engine.connect() as conn:
        result = conn.execute(text(sql), prepared)
        conn.commit()
        # rowcount 在 INSERT OR IGNORE 下反映实际插入行数（被忽略的不计）
        return result.rowcount or len(prepared)


def get_data_source() -> str:
    """返回当前 tool 数据来源：mock（默认）或 db。"""
    return os.environ.get("AGENT_TOOL_DATA_SOURCE", "mock").lower()
