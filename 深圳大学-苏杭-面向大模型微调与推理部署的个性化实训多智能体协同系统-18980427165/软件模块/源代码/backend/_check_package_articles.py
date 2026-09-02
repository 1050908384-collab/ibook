# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.v1.learning_resources import build_stage_document_resources
from tools.agent3_tools.vector_search_tool import VectorSearchTool

# 先确认 db 路径
import config as c
raw = c.settings.DATABASE_URL.replace("sqlite:///", "", 1)
db_path = Path(raw)
if not db_path.is_absolute():
    db_path = (c.BACKEND_DIR / db_path).resolve()
print(f"[DB] {db_path}  exists={db_path.exists()}")

# 用 stage like { block_id: KB-02, subtopic_ids: [KBS-02-00, ...] }
# 先查询整个 block 下所有 subtopic 排序，模拟真实 stage
import sqlite3
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
blocks = conn.execute("SELECT DISTINCT block_id FROM knowledge_subtopics ORDER BY block_id").fetchall()
print(f"\nBlocks in DB: {[b[0] for b in blocks]}")

for (block_id,) in blocks:
    sub_rows = conn.execute("""
        SELECT subtopic_id, subtopic_name, sequence_index
        FROM knowledge_subtopics WHERE block_id = ? ORDER BY sequence_index
    """, (block_id,)).fetchall()
    print(f"\n=== Block {block_id}: {len(sub_rows)} subtopics ===")
    ids = [r[0] for r in sub_rows]
    names = [r[1] for r in sub_rows]
    stage = {"block_id": block_id, "subtopic_ids": ids, "subtopic_names": names}
    dyn = build_stage_document_resources(stage)
    print(f"build_stage_document_resources: {len(dyn)} articles total")
    by_sub: dict[str, int] = {}
    empty_subs = []
    for sid in ids:
        c = sum(1 for r in dyn if r.get("subtopic_id") == sid)
        if c == 0: empty_subs.append(sid)
        by_sub[sid] = c
    # 只展示每条 block 的前 3 条 + 有没有 0 的
    for sid, cnt in list(by_sub.items())[:5]:
        sname = next((n for i, n in zip(ids, names) if i == sid), "")
        print(f"   {sid} ({sname[:45]}...): {cnt}")
    if len(by_sub) > 5:
        print(f"   ... (showing first 5 of {len(by_sub)})")
    if empty_subs:
        print(f"   !!! {len(empty_subs)} subtopics with 0 articles:")
        for sid in empty_subs:
            sname = next((n for i, n in zip(ids, names) if i == sid), "")
            print(f"     - {sid}: {sname[:60]}")
            # Check if local doc even exists for it
            dr = conn.execute("""
                SELECT doc_id, source, length(content) FROM knowledge_docs
                WHERE block_id=? AND subtopic_id=? AND doc_id LIKE 'LOCAL-%' LIMIT 1
            """, (block_id, sid)).fetchone()
            if dr:
                doc_id, source, clen = dr
                print(f"        => doc={doc_id} source={ (source or '')[:60]!r} content_len={clen}")
                # Extract URLs by simple regex
                import re
                content = conn.execute("SELECT content FROM knowledge_docs WHERE doc_id=?", (doc_id,)).fetchone()[0] or ""
                urls = re.findall(r"https?://[^\s)>\"]+", content)
                print(f"        => regex-unique URL count in content = {len(set(urls))}, source_is_http={bool(source and source.startswith('http'))}")
            else:
                print(f"        => NO LOCAL knowledge_docs for this subtopic!!!")
conn.close()
