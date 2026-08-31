import sqlite3, json
c = sqlite3.connect("data/sqlite_data/project.db")
# 查 KB-01-02 chunks
print("=== KB-01-02 (环境变量) knowledge_chunks 列表 ===")
for r in c.execute("""
    SELECT chunk_id, chunk_index, heading_path, length(content)
    FROM knowledge_chunks WHERE doc_id='LOCAL-KBS-01-02' ORDER BY chunk_index
""").fetchall():
    print(f"  {r[0]:28s} idx={str(r[1]):>4s} len={str(r[3]):>6s}  hp={r[2] or ''}")

# 检查 「优先级」 字样是否在 heading_path / content 任意一个里
print("\n=== 检查 heading_path 或 content 包含'优先级'字样的 chunks  ===")
for r in c.execute("""
    SELECT chunk_id, heading_path, substr(content,1,200)
    FROM knowledge_chunks WHERE doc_id='LOCAL-KBS-01-02'
      AND (heading_path LIKE '%优先级%' OR content LIKE '%优先级%')
""").fetchall():
    print(f"  {r[0]:28s}  hp={r[1] or ''}")
    print(f"    content[:200]={r[2]}")
