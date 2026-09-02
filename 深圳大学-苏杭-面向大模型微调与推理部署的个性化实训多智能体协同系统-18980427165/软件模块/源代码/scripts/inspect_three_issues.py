"""只取证，不改代码：

问题 1a："标题的 md 渲染格式非常不好"——抓：
  A) DB 中 source knowledge_chunks / catalog_tools 原始 heading_path
  B) content_blocks 保存的 block.heading_path / block_name
  C) _aggregate_blocks 生成的 tutorial 中 ## 标题行（含完整路径的丑陋格式）

问题 1b："第一章内容直接删完"——抓：
  A) knowledge_chunks 中每个块的原文长度
  B) DB 中对应 content_blocks 的 personalized_content 长度（含原文保留比例）
  C) 对比每个块的开头和结尾，判断是否"大段原文缺失 / 只剩外壳脚手架"

问题 2："相同知识点两目标还是一样的"——抓：
  两会话 content_blocks 一对一（按 block_id 对齐），算每块 char_sim + 表达层标记，
  输出相似度分布 + 找到相似度 >0.90 的块（真的一样的那些）列出原因。
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from db.sqlite_client import SQLiteClient  # noqa: E402


def char_sim(a: str, b: str, n: int = 3) -> float:
    def ngram(s: str):
        s = re.sub(r"\s+", "", s or "")
        if len(s) < n:
            return {s}
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    A, B = ngram(a), ngram(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def heading_lines(text: str) -> List[str]:
    return [l.rstrip() for l in (text or "").splitlines() if l.strip().startswith("#")][:12]


def get_session_ids(db: SQLiteClient) -> List[Dict[str, Any]]:
    with db.Session() as s:
        from sqlalchemy import text
        return [dict(r) for r in s.execute(text("""
            SELECT session_id, user_id, goal, status, current_stage, created_at
              FROM sessions ORDER BY datetime(created_at) DESC
        """)).mappings().fetchall()]


def get_state(db: SQLiteClient, sid: str) -> Dict[str, Any]:
    return db.load_session_state(sid) or {}


def get_block(db: SQLiteClient, block_id: str) -> Dict[str, Any]:
    # 用 KnowledgeCatalogTool 拿到原块内容/标题，优先看本地 knowledge_catalog/kbs 初始化好的块
    try:
        from tools.knowledge_catalog_tool import KnowledgeCatalogTool
        t = KnowledgeCatalogTool(database_url=db.database_url)
        b = t.get_block(block_id) or {}
    except Exception:
        b = {}
    if not b:
        # fallback: 查 knowledge_chunks 表（如果有）
        with db.Session() as s:
            from sqlalchemy import text
            try:
                rows = s.execute(text("""
                    SELECT chunk_id, block_id, heading_path, content, char_length
                      FROM knowledge_chunks WHERE block_id = :bid
                     LIMIT 1
                """), {"bid": block_id}).mappings().fetchall()
                if rows:
                    r = dict(rows[0])
                    b = {"block_id": r.get("block_id"), "chunk_id": r.get("chunk_id"),
                         "heading_path": r.get("heading_path"), "content": r.get("content"),
                         "char_length": r.get("char_length") or len(r.get("content") or "")}
            except Exception:
                pass
    return b or {}


def main() -> int:
    db = SQLiteClient()
    try:
        sessions = get_session_ids(db)
        print(f"共 {len(sessions)} 会话")
        # 最近两个 Linux 会话做对照
        sess_info: List[Dict[str, Any]] = []
        for r in sessions[:6]:
            sid = r["session_id"]
            state = get_state(db, sid)
            pkg = state.get("learning_package") or {}
            blocks = pkg.get("content_blocks") or []
            stage_info = state.get("stage_info") or state.get("current_stage_info") or {}
            profile = state.get("profile") or {}
            sess_info.append({
                **r,
                "profile": profile,
                "stage_info": stage_info,
                "learning_package": pkg,
                "blocks": blocks,
            })
        for s in sess_info:
            style = s["profile"].get("learning_style")
            print("\n" + "=" * 80)
            print(f"Session {s['session_id'][:12]}… goal={s['goal']!r} status={s['status']} style={style!r}")
            print("=" * 80)

            # ========== ISSUE 1a：标题 ==========
            pkg = s["learning_package"]
            tut = pkg.get("tutorial") or ""
            hl = heading_lines(tut)
            print("\n[ISSUE 1a] tutorial 前 12 行标题：")
            for line in hl:
                print(f"    {line[:200]}")
            # 看看丑陋的"nvidia-smi 显存监控 > 二、基础用法：..."是谁填的 heading_path：
            blocks = s["blocks"]
            print(f"\n  content_blocks 数 = {len(blocks)}")
            if blocks:
                print("  前 4 块的 heading_path（content_blocks 内保存的值）：")
                for b in blocks[:4]:
                    print(f"    block_id={str(b.get('block_id') or b.get('content_block_id'))[:20]}  "
                          f"heading_path={str(b.get('heading_path') or b.get('block_name') or '')[:120]!r}")
                # 原文 catalog 的 heading_path 是什么
                first = blocks[0]
                src_bid = str(first.get("source_chunk_id") or first.get("block_id") or "").strip()
                if src_bid:
                    cat = get_block(db, src_bid)
                    if not cat:
                        # 没 catalog 时按 learning_package 找 block_id
                        stage_bid = (s.get("stage_info") or {}).get("block_id") or ""
                        if stage_bid:
                            cat = get_block(db, stage_bid)
                    print(f"\n  对应原文 catalog 的 heading_path（source）：{str(cat.get('heading_path') or '')[:160]!r}")
                    print(f"  原文 catalog 标题 = {str(cat.get('block_name') or cat.get('name') or '')[:160]!r}")

            # ========== ISSUE 1b：第一章内容是否被删掉 ==========
            print("\n[ISSUE 1b] 第一章（block_index=0）原文长度 vs 个性化长度：")
            src_len_map: Dict[str, int] = {}
            # 先按 catalog 读每个原文块长度
            for b in blocks:
                bid = str(b.get("block_id") or b.get("content_block_id") or b.get("source_chunk_id") or "")
                if not bid:
                    continue
                cat = get_block(db, bid)
                txt = cat.get("content") or ""
                src_len_map[bid] = len(txt)
            rows = []
            for idx, b in enumerate(blocks[:6]):
                bid = str(b.get("block_id") or b.get("content_block_id") or b.get("source_chunk_id") or f"idx{idx}")
                pc = b.get("personalized_content") or ""
                src = src_len_map.get(bid) or 0
                ratio = (len(pc) / src * 100) if src else 0
                rows.append((idx, bid, src, len(pc), ratio))
                head = (pc[:200] or "").replace("\n", " / ")
                print(f"  idx={idx} src_len={src} → pc_len={len(pc)} 保留率={ratio:.0f}%")
                print(f"       前 200 字：{head}")
            if any(r[4] < 30 for r in rows if r[2] > 100):
                print("  ⚠️ 有块保留率 < 30%（原文超过 100 字但只留下不到 30%），Prompt 红线可能没挡住 LLM 过度删除。")

        # ========== ISSUE 2：两会话相同知识点逐块相似度 ==========
        if len(sess_info) >= 2:
            print("\n" + "#" * 80)
            print("[ISSUE 2] 最近两个 Linux 会话：逐块（按块序对齐）相似度 + 表达层标记对比")
            print("#" * 80)
            a, b = sess_info[0], sess_info[1]
            print(f"  A: {a['session_id'][:10]}… style={a['profile'].get('learning_style')!r} goal={a['goal']!r}")
            print(f"  B: {b['session_id'][:10]}… style={b['profile'].get('learning_style')!r} goal={b['goal']!r}")
            ab = a["blocks"]
            bb = b["blocks"]
            print(f"  A blocks={len(ab)}  B blocks={len(bb)}  对齐 = min")
            n = min(len(ab), len(bb))
            sims = []
            for i in range(n):
                a_pc = ab[i].get("personalized_content") or ""
                b_pc = bb[i].get("personalized_content") or ""
                sim = char_sim(a_pc, b_pc)
                sims.append(sim)
                a_hp = str(ab[i].get("heading_path") or ab[i].get("block_name") or "")[:80]
                prac_a = any(w in a_pc for w in ("先别急", "一句话版", "概念摘要", "验证方式", "我什么时候需要"))
                theo_a = any(w in a_pc for w in ("关键原因", "设计动机", "思考题", "概念对比", "架构哲学", "长期支持"))
                prac_b = any(w in b_pc for w in ("先别急", "一句话版", "概念摘要", "验证方式", "我什么时候需要"))
                theo_b = any(w in b_pc for w in ("关键原因", "设计动机", "思考题", "概念对比", "架构哲学", "长期支持"))
                warn = " ⚠️ SAME" if sim >= 0.90 else (" ❗ HIGH" if sim >= 0.75 else "")
                print(
                    f"  idx={i} sim={sim:.4f}{warn}  lenA={len(a_pc)} lenB={len(b_pc)}\n"
                    f"       heading: {a_hp!r}\n"
                    f"       A practice_markers={prac_a} theory_markers={theo_a}   "
                    f"B practice_markers={prac_b} theory_markers={theo_b}"
                )
            if sims:
                high_count = sum(1 for s in sims if s >= 0.90)
                print(f"\n  逐块相似度分布：max={max(sims):.4f}  avg={sum(sims)/len(sims):.4f}  ≥0.90 的块数 = {high_count}/{len(sims)}（=真的一样的块）")
                if high_count == len(sims):
                    print("  ❌ 所有块都一样：回退至 save_profile/get_latest_profile/Agent3 的 style 注入路径排查。")
                elif high_count > 0:
                    print("  ⚠️ 部分块一样：很可能是这些块 source 原文本身『既没代码也没概念对比素材可加』，LLM 无从差异化，或者 Prompt 仍在这些块上命中了旧版规则。")
                else:
                    print("  ✅ 所有块均未出现 ≥ 0.90 的雷同。若用户肉眼仍觉得一样，检查：① 两包 tutorial 仍被 backfill 覆盖；② 前端读的仍是旧 snapshot。")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
