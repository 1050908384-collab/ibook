# -*- coding: utf-8 -*-
"""精查当前两个 Linux session 的 content_blocks — 哪些块含你要删除的关键词"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text

engine = create_engine(f"sqlite:///{DB_PATH}")

with engine.connect() as conn:
    sids = [r[0] for r in conn.execute(text("SELECT session_id FROM sessions")).fetchall()]
    for sid in sids:
        row = conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:sid"), {"sid": sid}).first()
        if not row:
            continue
        st = json.loads(row[0])
        ls = (st.get("profile") or {}).get("learning_style") or "?"
        lp = st.get("learning_package") or {}
        cbs = lp.get("content_blocks") or []
        print(f"\n=== session {sid[:12]}  learning_style={ls}  blocks={len(cbs)} ===")
        for i, b in enumerate(cbs if isinstance(cbs, list) else []):
            if not isinstance(b, dict):
                continue
            pc = b.get("personalized_content") or ""
            hp = b.get("heading_path") or ""
            scid = b.get("source_chunk_id") or ""
            hits = [kw for kw in ["体系起点","思考题","概念对比","概念与设计动机","trade-off","GPU 监控体系的起点","黑盒"] if kw in pc]
            first_lines = [ln.strip() for ln in pc.splitlines()[:5] if ln.strip()]
            print(f"  [{i:2d}] scid={scid:28s} hp_leaf={hp.split('>')[-1].strip() if '>' in hp else hp[:50]:<45s} len={len(pc):>5d} suspect={hits}")
            for ln in first_lines[:4]:
                print(f"         >> {ln[:120]}")
        # 额外：全文 tutorial 查关键词
        tut = lp.get("tutorial") or ""
        lines = tut.splitlines()
        hits = [(i+1, line[:140]) for i, line in enumerate(lines) if any(kw in line for kw in ["这一行命令，就是整个 GPU 监控体系的起点","思考题","概念对比"])]
        if hits:
            print(f"  tutorial 中 suspect 关键词命中行：")
            for n, ln in hits:
                print(f"    L{n}: {ln}")
