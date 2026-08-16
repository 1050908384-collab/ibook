# -*- coding: utf-8 -*-
import sqlite3, re, sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api.v1.learning_resources import extract_all_reference_urls

conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
# Check all LOCAL docs in blocks that have install-like subtopics
rows = conn.execute("""
    SELECT doc_id, block_id, subtopic_id, title, source, content
    FROM knowledge_docs
    WHERE doc_id LIKE 'LOCAL-%'
      AND (
        subtopic_id IN ('KBS-02-00','KBS-03-00','KBS-07-00')
        OR title LIKE '%安装%' OR title LIKE '%环境准备%'
      )
""").fetchall()
for doc_id, blk, sid, title, source, content in rows:
    print(f"\n===== {doc_id}  blk={blk} sid={sid}  title={title!r} =====")
    print(f"  source = {(source or '')[:120]!r}")
    refs = extract_all_reference_urls(content or "")
    print(f"  extract_all_reference_urls 返回 {len(refs)} 条:")
    for label, url in refs:
        print(f"    [ {label[:70]} ] -> {url[:140]}")
    if not refs:
        # Print first 600 chars of content to analyze
        print("  !!! EMPTY. First 800 chars of content:")
        print("  |" + (content or "")[:800].replace("\n", "\n  |"))
conn.close()
