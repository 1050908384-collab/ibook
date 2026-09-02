# -*- coding: utf-8 -*-
"""查 KBS-01-04（nvidia-smi显存监控）子主题本身含不含"一、什么是 nvidia-smi"的概述章。
同时查 stage/fact_source_chunk_ids 的选择逻辑是否过滤掉了前面的 chunks。
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
        # 1. 查 KBS-01-04 子主题详情
        row = conn.execute(text("""
            SELECT subtopic_id, subtopic_name, sequence_index, difficulty,
                   prerequisites, content_summary
            FROM knowledge_subtopics WHERE subtopic_id = 'KBS-01-04'
        """)).first()
        print("=" * 90)
        print("[子主题] KBS-01-04:")
        print("=" * 90)
        if row:
            for k, v in row._mapping.items():
                print(f"  {k:20s} = {str(v)[:200]}")
        else:
            print("  不存在！")

        # 2. 全部 KB-01 子主题排序（确认 sequence）
        print("\n" + "=" * 90)
        print("[KB-01 子主题列表（按 sequence_index）]")
        print("=" * 90)
        rows = conn.execute(text("""
            SELECT subtopic_id, sequence_index, difficulty, subtopic_name,
                   prerequisites, content_summary
            FROM knowledge_subtopics
            WHERE block_id = 'KB-01' ORDER BY sequence_index
        """)).fetchall()
        for r in rows:
            rm = r._mapping
            print(f"  [{rm['sequence_index']:>2d}] {rm['subtopic_id']:15s} {rm['difficulty']:5s} "
                  f"name={str(rm['subtopic_name'])[:60]}")

        # 3. 查 KBS-01-04 下的全部 chunks（heading_path 前缀匹配 & block_id+chunk_index）
        print("\n" + "=" * 90)
        print("[KBS-01-04 子主题归属的 chunks（看 heading_path 是否包含 nvidia-smi 显存监控）]")
        print("=" * 90)
        rows = conn.execute(text("""
            SELECT chunk_id, chunk_index, heading_path, LENGTH(content)
            FROM knowledge_chunks
            WHERE block_id='KB-01' AND heading_path LIKE '%nvidia-smi%显存监控%'
            ORDER BY chunk_index, chunk_id
        """)).fetchall()
        for r in rows:
            rm = r._mapping
            leaf = rm["heading_path"].split(">")[-1].strip() if rm["heading_path"] and ">" in str(rm["heading_path"]) else str(rm["heading_path"])
            stars = "  ⭐⭐⭐ [第一大节！应该在2.1之前出现！]" if (
                "一、" in str(rm["heading_path"]) or "概述" in leaf or "定位" in leaf or "是什么" in leaf
            ) else ""
            print(f"  {rm['chunk_id']:25s} idx={rm['chunk_index']}  leaf={leaf[:80]}{stars}")
        print(f"  共 {len(rows)} 个 chunks 归属 KBS-01-04（nvidia-smi显存监控）")

        # 4. session 的 stage_info.fact_source_chunk_ids（stage 选择范围）
        print("\n" + "=" * 90)
        print("[两个 session 的 stage_info.fact_source_chunk_ids 全量打印]")
        print("=" * 90)
        sids = [r2[0] for r2 in conn.execute(text("SELECT session_id FROM sessions WHERE LOWER(goal) LIKE '%linux%'")).fetchall()]
        for sid in sids:
            st = json.loads(conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:sid"), {"sid": sid}).first()[0])
            si = st.get("stage_info") or {}
            fsc = si.get("fact_source_chunk_ids") or []
            print(f"\n[{sid[:12]}] fact_source_chunk_ids (len={len(fsc)}):")
            for cid in fsc:
                r2 = conn.execute(text("SELECT heading_path FROM knowledge_chunks WHERE chunk_id=:cid"), {"cid": cid}).first()
                hp = r2[0] if r2 else "NOT FOUND"
                leaf = (hp.split(">")[-1].strip() if hp and ">" in hp else hp)
                stars = "  ⭐⭐⭐ 第一大节候选" if ("一、" in str(hp) or "概述" in leaf) else ""
                print(f"    {cid:25s}  leaf={leaf[:80]}{stars}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
