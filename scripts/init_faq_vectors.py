# -*- coding: utf-8 -*-
"""
FAQ数据向量化初始化脚本
================================================
功能：将faq_items表中的FAQ数据向量化，存入Chroma向量库。
执行方式：python scripts/init_faq_vectors.py

流程：
    1. 从SQLite的faq_items表读取所有FAQ条目
    2. 将每个FAQ的question和answer合并为文档内容
    3. 将文档向量化并存入Chroma的"faq_collection"集合
    4. 保留metadata（faq_id, block_id, question, answer, difficulty, tags, views）

依赖：
    - ChromaDB (需提前安装：pip install chromadb)
    - SQLite数据库中已有faq_items表
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.sqlite_client import SQLiteClient
from backend.db.chroma_client import ChromaClient
from chromadb import PersistentClient


def init_faq_vectors():
    print("[INFO] 开始FAQ数据向量化初始化...")

    sqlite_client = SQLiteClient()

    try:
        with sqlite_client.Session() as session:
            rows = session.execute("""
                SELECT faq_id, block_id, question, answer, difficulty, tags, views
                FROM faq_items
            """).fetchall()

        if not rows:
            print("[WARN] faq_items表为空，跳过向量化")
            return

        print(f"[INFO] 读取到 {len(rows)} 条FAQ条目")

        chroma_client = PersistentClient(path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma"))
        collection = chroma_client.get_or_create_collection("faq_collection")

        documents = []
        metadatas = []
        ids = []

        for row in rows:
            faq_id, block_id, question, answer, difficulty, tags, views = row

            documents.append(f"{question}\n{answer}")

            try:
                tags_list = eval(tags) if tags else []
            except:
                tags_list = []

            metadatas.append({
                "faq_id": faq_id,
                "block_id": block_id,
                "question": question,
                "answer": answer,
                "difficulty": difficulty or "",
                "tags": tags_list,
                "views": views or 0,
            })

            ids.append(faq_id)

        print(f"[INFO] 准备向量化 {len(documents)} 条文档...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"[OK] FAQ向量初始化完成！")
        print(f"[INFO] Chroma集合 'faq_collection' 包含 {collection.count()} 条向量")

    except Exception as e:
        print(f"[ERROR] FAQ向量初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    init_faq_vectors()