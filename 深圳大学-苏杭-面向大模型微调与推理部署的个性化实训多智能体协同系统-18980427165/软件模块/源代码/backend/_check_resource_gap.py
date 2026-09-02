# -*- coding: utf-8 -*-
import sqlite3, re, sys
from pathlib import Path

# ---------- find DB ----------
candidates = [Path("project.db"), Path("data/project.db"), Path("../data/project.db")]
db_path = None
for p in candidates:
    if p.exists():
        db_path = p.resolve()
        break
if db_path is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as c
    raw = c.settings.DATABASE_URL.replace("sqlite:///", "", 1)
    db_path = Path(raw)
    if not db_path.is_absolute():
        db_path = (c.BACKEND_DIR / db_path).resolve()

print("[DB]", db_path)
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

# ---------- 1. find Python install-like subtopics ----------
rows = conn.execute("""
    SELECT subtopic_id, subtopic_name, block_id
    FROM knowledge_subtopics
    WHERE subtopic_name LIKE '%Python%安装%'
       OR subtopic_name LIKE '%安装%Python%'
       OR subtopic_name LIKE '%环境搭建%'
       OR subtopic_name LIKE '%开发环境%'
       OR subtopic_name LIKE '%Python安装%'
    ORDER BY block_id, sequence_index
""").fetchall()
print("\n[1] 匹配的知识点:")
targets = []
for r in rows:
    print("  ", r)
    targets.append(r)

# Also list the whole block's subtopics if user meant "other stages"
if not targets:
    # Fallback: any knowledge_subtopics name containing 安装
    rows = conn.execute("SELECT subtopic_id, subtopic_name, block_id FROM knowledge_subtopics WHERE subtopic_name LIKE '%安装%' LIMIT 10").fetchall()
    for r in rows:
        print("  (fuzzy)", r)
        targets.append(r)

# ---------- for each, look in learning_resources (via SR / BR) for article-type ----------
seen_blocks = {}
for sub, sname, block in targets:
    print(f"\n=== subtopic {sub} ({sname!r}), block {block} ===")
    # subtopic_resources + block_resources joined to learning_resources (non-video)
    q = conn.execute("""
        SELECT 'SR', sr.subtopic_id, sr.role, sr.mapping_status,
               r.resource_id, r.resource_type, r.title, r.canonical_url
        FROM subtopic_resources sr
        JOIN learning_resources r ON r.resource_id = sr.resource_id
        WHERE sr.block_id = ? AND sr.subtopic_id = ?
        UNION ALL
        SELECT 'BR', NULL, br.role, br.mapping_status,
               r.resource_id, r.resource_type, r.title, r.canonical_url
        FROM block_resources br
        JOIN learning_resources r ON r.resource_id = br.resource_id
        WHERE br.block_id = ?
        ORDER BY 1, 2, 3
    """, (block, sub, block)).fetchall()
    print(f"  [2a] subtopic/block 关联的学习资源 (共 {len(q)})：")
    for r in q:
        kind, sid, role, mstatus, rid, rtype, title, url = r
        print(f"    {kind} sid={sid!s:12} role={role!s:8} map={mstatus!s:9} id={rid:18} type={rtype:6} title={str(title)[:50]!s}")
        if url: print(f"         URL: {url[:120]}")
    seen_blocks.setdefault(block, []).append(sub)

    # knowledge_docs source & content urls
    drows = conn.execute("""
        SELECT doc_id, title, source, content
        FROM knowledge_docs
        WHERE block_id = ? AND subtopic_id = ? AND doc_id LIKE 'LOCAL-%'
        ORDER BY doc_id
    """, (block, sub)).fetchall()
    print(f"  [2b] knowledge_docs 共 {len(drows)} 篇 LOCAL 文档:")
    for doc_id, title, source, content in drows:
        content = content or ""
        urls = sorted(set(re.findall(r"https?://[^\s)>\"]+", content or "")))
        print(f"    {doc_id}  title={title[:50]!r}  source={ (source or '')[:80]!r}  source_is_http={bool(source and source.startswith('http'))}  content_refs={len(urls)}")
        if source and source.startswith("http"):
            print(f"      SOURCE URL: {source}")
        for u in urls[:5]:
            print(f"      REF URL:   {u[:140]}")

# ---------- Also print total resource types per block ----------
print("\n[3] 各 block 学习资源类型概览:")
for block in sorted(seen_blocks.keys()):
    totals = conn.execute("""
        SELECT r.resource_type, COUNT(*)
        FROM learning_resources r
        JOIN (
            SELECT resource_id FROM subtopic_resources WHERE block_id = ?
            UNION
            SELECT resource_id FROM block_resources WHERE block_id = ?
        ) x ON x.resource_id = r.resource_id
        GROUP BY r.resource_type
    """, (block, block)).fetchall()
    print(f"  block {block}: {totals}")

conn.close()
