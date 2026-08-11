# -*- coding: utf-8 -*-
"""从 SQLite knowledge_chunks 表重建 Chroma 向量索引。

数据源：data/sqlite_data/project.db 的 knowledge_chunks 表（JOIN knowledge_docs
取 title/source/subtopic_id）。每个 chunk 作为一个 document 写入 Chroma
的 knowledge_base collection，metadata 包含 chunk_id / block_id / subtopic_id
/ source / title，供 Agent3 vector_search_tool 检索使用。

重复运行安全：写入前先删除旧 collection 重建，避免脏数据残留。
"""
import os
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from backend.db.chroma_client import CHROMA_COLLECTION, ChromaClient  # noqa: E402
from config import BACKEND_DIR as BACKEND_PATH, settings  # noqa: E402


DB_PATH = Path(ROOT_DIR) / "data" / "sqlite_data" / "project.db"


def _resolve_chroma_path() -> Path:
    p = Path(settings.CHROMA_PATH)
    if not p.is_absolute():
        p = (BACKEND_PATH / p).resolve()
    return p


def init_knowledge_base():
    print("重建 Chroma 向量索引...")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite 数据库不存在: {DB_PATH}")

    # 1. 清空旧 collection（避免脏数据残留）
    chroma_path = _resolve_chroma_path()
    try:
        from chromadb import PersistentClient
        client = PersistentClient(path=str(chroma_path))
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"已删除旧 collection: {CHROMA_COLLECTION}")
        except Exception:
            # collection 不存在时忽略
            pass
    except ImportError:
        print("chromadb 未安装，无法重建向量索引")
        return

    # 2. 从 SQLite 读取所有 knowledge_chunks（JOIN knowledge_docs 拿元数据）
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT c.chunk_id, c.block_id, c.content,
                   d.subtopic_id, d.source, d.title, d.doc_id
            FROM knowledge_chunks c
            LEFT JOIN knowledge_docs d ON c.doc_id = d.doc_id
            ORDER BY c.doc_id, c.chunk_index
        """).fetchall()
    finally:
        conn.close()

    if not rows:
        print("警告: knowledge_chunks 表为空，未写入任何向量")
        return

    # 3. 组装 documents / metadatas / ids
    documents = []
    metadatas = []
    ids = []
    for chunk_id, block_id, content, subtopic_id, source, title, doc_id in rows:
        # Chroma metadata 值必须是 str/int/float/bool，None 转空串
        documents.append(content or "")
        metadatas.append({
            "chunk_id": chunk_id or "",
            "block_id": block_id or "",
            "subtopic_id": subtopic_id or "",
            "source": source or "",
            "title": title or "",
            "doc_id": doc_id or "",
        })
        ids.append(chunk_id)

    # 4. 写入 Chroma（分批，每批最多 200 条，避免单次请求过大）
    chroma_client = ChromaClient()
    BATCH_SIZE = 200
    total = len(documents)
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        chroma_client.add_documents(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"  写入进度: {end}/{total}")

    # 5. 验证
    count = chroma_client.count()
    print(f"\n[DONE] 已写入 {count} 条向量到 Chroma (collection={CHROMA_COLLECTION})")
    print(f"  Chroma 路径: {chroma_path}")


if __name__ == "__main__":
    init_knowledge_base()
