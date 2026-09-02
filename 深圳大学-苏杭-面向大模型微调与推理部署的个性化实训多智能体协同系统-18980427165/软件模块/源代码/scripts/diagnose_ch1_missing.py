# -*- coding: utf-8 -*-
"""诊断"第一章不见了"：查当前 stage block 的全部原始 knowledge_chunks
和 content_blocks 的 source_chunk_id 溯源，定位第一大节（"一、xxx/概述"）是否存在但被跳过。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def main():
    with engine.connect() as conn:
        # 取两个 session 的 stage block_id
        sessions = conn.execute(text("""
            SELECT s.session_id, s.goal, sp.block_id, sp.block_name, sp.subtopic_id, sp.stage_num
            FROM sessions s JOIN stage_progress sp USING(session_id)
            WHERE LOWER(s.goal) LIKE '%linux%'
            ORDER BY s.created_at DESC, sp.stage_num
        """)).fetchall()
        target_blocks = {}
        for r in sessions:
            rm = r._mapping
            print(f"session {rm['session_id'][:12]} stage#{rm['stage_num']} "
                  f"block_id={rm['block_id']} block_name={rm['block_name']} "
                  f"subtopic_id={rm['subtopic_id']}")
            target_blocks.setdefault(rm["block_id"], rm["block_name"])

        # 对每个 block，列出全部原始 chunk
        for bid, bname in target_blocks.items():
            print("\n" + "=" * 90)
            print(f"[原始分块] block={bid} ({bname}) - 全部 knowledge_chunks")
            print("=" * 90)
            chunks = conn.execute(text("""
                SELECT chunk_id, heading_path, LENGTH(content) AS clen,
                       substr(content, 1, 160) AS preview
                FROM knowledge_chunks WHERE block_id = :bid ORDER BY chunk_id
            """), {"bid": bid}).fetchall()
            for i, c in enumerate(chunks):
                m = c._mapping
                stars = ""
                hp = str(m.get("heading_path") or "")
                leaf = hp.split(">")[-1].strip() if ">" in hp else hp
                if i == 0 or leaf.startswith("一") or "概述" in leaf or "简介" in leaf or "介绍" in leaf or "定位" in leaf:
                    stars = "  ⭐ [候选第一部分/第一章内容]"
                print(f"  {m['chunk_id']:25s}  len={m['clen']:>5d}  leaf_heading={leaf[:90]}{stars}")
            print(f"  ---- 共 {len(chunks)} 个 ----")

        # session 的 content_blocks 溯源
        print("\n" + "=" * 90)
        print("[content_blocks 溯源] 两个 Linux session")
        print("=" * 90)
        for r in sessions[:2]:  # 只看第一个 session 的 stage
            sid = r._mapping["session_id"]
            row = conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:sid"), {"sid": sid}).first()
            if not row:
                continue
            st = json.loads(row[0])
            lp = st.get("learning_package") or {}
            cbs = lp.get("content_blocks") or []
            print(f"\n[{sid[:12]}] content_blocks ({len(cbs)} 块):")
            chunk_ids_present = set()
            for i, b in enumerate(cbs if isinstance(cbs, list) else []):
                if not isinstance(b, dict):
                    continue
                scid = b.get("source_chunk_id") or "?"
                chunk_ids_present.add(scid)
                # 查 DB 对应 chunk 的 leaf heading
                r2 = conn.execute(text("SELECT heading_path FROM knowledge_chunks WHERE chunk_id=:cid"), {"cid": scid}).first()
                db_hp = r2[0] if r2 else "NOT_FOUND"
                db_leaf = (db_hp.split(">")[-1].strip() if db_hp and ">" in db_hp else (db_hp or ""))
                marker = "  ⭐ [第一章候选]" if (("一" in db_leaf and "简" not in db_leaf) or "概述" in db_leaf or "定位" in db_leaf) else ""
                print(f"  [{i:2d}] src_chunk={scid:25s}  leaf_hp={db_leaf[:80]}{marker}")

            # 对比：列出 block 里有的 chunk 但 content_blocks 里没用到的
            bid = r._mapping["block_id"]
            all_chunks = [row2["chunk_id"] for row2 in conn.execute(
                text("SELECT chunk_id FROM knowledge_chunks WHERE block_id=:bid ORDER BY chunk_id"),
                {"bid": bid}
            ).mappings().all()]
            missing = [cid for cid in all_chunks if cid not in chunk_ids_present]
            print(f"\n  [DB有但content_blocks里缺失的 chunks] 共 {len(missing)} 个：")
            for cid in missing:
                r3 = conn.execute(text("SELECT heading_path, LENGTH(content) FROM knowledge_chunks WHERE chunk_id=:cid"), {"cid": cid}).first()
                hp = r3[0] if r3 else ""
                leaf = (hp.split(">")[-1].strip() if hp and ">" in hp else hp)
                ln = r3[1] if r3 else "-"
                marker = "  ⭐⭐ 如果是第一章这里丢了！" if ("一" in leaf and "简" not in leaf) or "概述" in leaf or "定位" in leaf else ""
                print(f"    - {cid:25s}  len={ln:>5s}  leaf={leaf[:90]}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
