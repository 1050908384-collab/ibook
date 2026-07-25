# -*- coding: utf-8 -*-
"""
FAQ 数据向量化初始化脚本

作用：
1. 从 `faq_items` 读取 FAQ 数据
2. 将 `question + answer` 写入 Chroma 向量库
3. 统一使用 `DATABASE_URL` 与 `CHROMA_PATH`
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BACKEND_DIR)

from sqlalchemy import text

from backend.config import BACKEND_DIR as BACKEND_PATH, settings
from backend.db.sqlite_client import SQLiteClient

try:
    from chromadb import PersistentClient
except ImportError:
    PersistentClient = None


def init_faq_vectors():
    if PersistentClient is None:
        print("[ERROR] chromadb is not installed. Run `pip install chromadb` first.")
        return

    print("[INFO] Starting FAQ vector initialization...")
    sqlite_client = SQLiteClient()

    try:
        with sqlite_client.Session() as session:
            rows = session.execute(
                text(
                    """
                    SELECT faq_id, block_id, question, answer, difficulty, tags, views
                    FROM faq_items
                    """
                )
            ).fetchall()

        if not rows:
            print("[WARN] faq_items is empty, skipping vector initialization.")
            return

        print(f"[INFO] Loaded {len(rows)} FAQ items")

        chroma_path = Path(settings.CHROMA_PATH)
        if not chroma_path.is_absolute():
            chroma_path = (BACKEND_PATH / chroma_path).resolve()

        chroma_client = PersistentClient(path=str(chroma_path))
        collection = chroma_client.get_or_create_collection("faq_collection")

        documents = []
        metadatas = []
        ids = []

        for faq_id, block_id, question, answer, difficulty, tags, views in rows:
            documents.append(f"{question}\n{answer}")

            try:
                tags_list = json.loads(tags) if tags else []
                if not isinstance(tags_list, list):
                    tags_list = []
            except Exception:
                tags_list = []

            metadatas.append(
                {
                    "faq_id": faq_id,
                    "block_id": block_id,
                    "question": question,
                    "answer": answer,
                    "difficulty": difficulty or "",
                    "tags": tags_list,
                    "views": views or 0,
                }
            )
            ids.append(faq_id)

        print(f"[INFO] Writing {len(documents)} FAQ vectors...")
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

        print("[OK] FAQ vector initialization completed")
        print(f"[INFO] faq_collection count: {collection.count()}")
    except Exception as exc:
        print(f"[ERROR] FAQ vector initialization failed: {exc}")
        raise
    finally:
        sqlite_client.close()


if __name__ == "__main__":
    init_faq_vectors()
