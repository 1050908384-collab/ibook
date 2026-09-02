# -*- coding: utf-8 -*-
"""精查 KB-01（Linux 5 篇）的 knowledge_docs.source_file_path 与 每个 doc 的 chunks ##大节覆盖完整度。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"
MD_ROOT = PROJECT_ROOT / "data" / "知识库原始数据"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def main():
    with engine.connect() as conn:
        print("=" * 110)
        print("[KB-01] knowledge_docs 表（Linux 5个子主题文档）详情")
        print("=" * 110)
        docs = conn.execute(text("""
            SELECT doc_id, block_id, title, source, source_file_path, LENGTH(content) AS doc_len
            FROM knowledge_docs WHERE block_id='KB-01' ORDER BY doc_id
        """)).fetchall()
        for d in docs:
            rm = d._mapping
            print(f"\n  doc_id={rm['doc_id']:20s}  title={str(rm['title'])[:60]}")
            print(f"    source={str(rm.get('source') or '')[:80]}")
            sfp = rm.get("source_file_path") or ""
            print(f"    source_file_path = {sfp[-120:] if len(sfp)>120 else sfp}")
            # chunks 归属
            chs = conn.execute(text("""
                SELECT chunk_id, chunk_index, heading_path, LENGTH(content)
                FROM knowledge_chunks WHERE doc_id=:did ORDER BY chunk_index, chunk_id
            """), {"did": rm["doc_id"]}).fetchall()
            print(f"    chunks: {len(chs)} 个 (doc_len={rm['doc_len']})")
            for ch in chs:
                hp = ch[2] or ""
                leaf = hp.split(">")[-1].strip() if ">" in hp else hp
                print(f"      {ch[0]:25s} idx={str(ch[1]):>4s}  len={ch[3]:>5d}  leaf={leaf[:80]}")
            # 检查 heading_path 大节覆盖（> 后的第一级）
            top_levels = sorted({(ch[2].split(">")[-2].strip() if ch[2] and ">" in ch[2] else "?") for ch in chs if ch[2]})
            print(f"    对应原文的 ## 级大节: {top_levels}")

        print("\n" + "=" * 110)
        print("[KB-01] 原始 md 文件清单")
        print("=" * 110)
        kb01 = MD_ROOT / "01_Linux环境与命令行操作"
        for md in sorted(kb01.glob("**/*.md")):
            rel = str(md.relative_to(MD_ROOT))
            lines = md.read_text(encoding="utf-8").splitlines()
            h2s = []
            for ln in lines:
                if ln.startswith("## "):
                    h2s.append(ln[3:].strip())
            print(f"  {rel:<80s}  body_len={sum(len(l) for l in lines):>6d}  ##大节={h2s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
