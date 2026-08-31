# -*- coding: utf-8 -*-
"""验证修复效果：1. 标题不再堆叠（无重复 heading_path breadcrumb） 2. 块间有分隔 3. 个性化仍存在"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def ngram_sim(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    sa = set(a[i:i + n] for i in range(len(a) - n + 1))
    sb = set(b[i:i + n] for i in range(len(b) - n + 1))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def main():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.session_id, s.goal, sj.state_json
            FROM sessions s JOIN session_states sj USING(session_id)
            WHERE LOWER(s.goal) LIKE '%linux%'
            ORDER BY s.created_at DESC
        """)).fetchall()
        if len(rows) < 2:
            print("不足2个Linux会话"); return 1

        data = []
        for r in rows:
            rid = dict(r._mapping)
            st = json.loads(rid["state_json"])
            lp = st.get("learning_package") or {}
            tut = lp.get("tutorial", "")
            ls = (st.get("profile") or {}).get("learning_style", "?")
            cb = lp.get("content_blocks") or []
            print("\n" + "=" * 90)
            print(f"[{rid['session_id'][:12]}] learning_style={ls}  len_tutorial={len(tut)}  blocks={len(cb)}")
            print("=" * 90)
            head_lines = tut.splitlines()[:40]
            for i, ln in enumerate(head_lines):
                tag = "  [WARN: 长breadcrumb标题]" if (
                    ln.strip().startswith("## ") and ">" in ln
                ) else ""
                print(f"  L{i+1:2d}: {ln[:130]}{tag}")

            # 检查重复标题
            title_lines = [l.strip() for l in tut.splitlines() if l.strip().startswith("## ")]
            for i in range(1, len(title_lines)):
                # 检查是否出现 "heading_path 长标题" + "同内容 leaf 短标题" 连续重复
                if ">" in title_lines[i - 1]:
                    short_prev = title_lines[i - 1].split(">")[-1].strip().lstrip("#").strip()
                    short_cur = title_lines[i].lstrip("#").strip()
                    if short_prev and short_cur and (
                        short_prev == short_cur or short_cur in short_prev or short_prev in short_cur
                    ):
                        print(f"  !! [重复标题堆叠警告] L{i-1}='{title_lines[i-1][:80]}' / L{i}='{title_lines[i][:80]}'")
            data.append((rid["session_id"], ls, tut, cb))

        # 整体相似度
        (s1, l1, t1, cb1), (s2, l2, t2, cb2) = data[0], data[1]
        print("\n" + "=" * 90)
        print(f"整体 tutorial 相似度(3-gram) = {ngram_sim(t1[:8000], t2[:8000]):.4f}"
              f"   ({l1} vs {l2})")
        # 逐块 personalized 相似度
        hp1 = {}
        for b in (cb1 or []):
            if isinstance(b, dict):
                hp1[b.get("heading_path")] = b.get("personalized_content", "")
        hp2 = {}
        for b in (cb2 or []):
            if isinstance(b, dict):
                hp2[b.get("heading_path")] = b.get("personalized_content", "")
        commons = [h for h in hp1 if h in hp2 and h]
        sims = [ngram_sim(hp1[h] or "", hp2[h] or "") for h in commons]
        if sims:
            print(f"逐块 personalized_content 相似度：avg={sum(sims)/len(sims):.4f}  max={max(sims):.4f}")
        print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
