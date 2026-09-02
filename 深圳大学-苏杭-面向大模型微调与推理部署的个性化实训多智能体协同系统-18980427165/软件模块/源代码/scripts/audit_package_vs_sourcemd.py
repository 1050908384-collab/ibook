# -*- coding: utf-8 -*-
"""逐块比对：Linux 会话的 personalized_content （学习包正文）
           vs knowledge_chunks.content（md原文切块）。
目标：
1. 逐块算 4-gram Jaccard-like 相似度。
2. 列出每块前 4 行原文 vs personalized，肉眼定位差异点。
3. 统计 practice/theory 两个会话的「整体相似度 / 改写字数比例 / 是否调用过LLM」证据。
4. 判断 Agent3 走了 LLM 分支还是 deterministic 纯规则回退（检查 profile 与调用标记）。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def ngram_sim(a: str, b: str, n: int = 4) -> float:
    if not a or not b:
        return 0.0
    sa = {a[i:i + n] for i in range(len(a) - n + 1)}
    sb = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def head_lines(text: str, k: int = 3):
    return [ln.strip() for ln in text.splitlines() if ln.strip()][:k]


def main():
    with engine.connect() as conn:
        # 1. 查最新 Linux 会话（按 created_at desc，要两个）
        rows = conn.execute(text("""
            SELECT s.session_id, s.goal, s.created_at, ss.state_json
            FROM sessions s LEFT JOIN session_states ss ON ss.session_id=s.session_id
            WHERE LOWER(s.goal) LIKE '%linux%' OR LOWER(s.goal) LIKE '%命令行%'
            ORDER BY s.created_at DESC LIMIT 10
        """)).fetchall()
        if not rows:
            # fallback：所有 sessions 按 createdAt desc 取最后几条，goal 里不包含其他关键词
            rows = conn.execute(text("""
                SELECT s.session_id, s.goal, s.created_at, ss.state_json
                FROM sessions s LEFT JOIN session_states ss ON ss.session_id=s.session_id
                ORDER BY s.created_at DESC LIMIT 8
            """)).fetchall()
        picked = []
        seen_styles = set()
        for r in rows:
            rm = r._mapping
            sj = json.loads(rm["state_json"]) if rm["state_json"] else {}
            ls = (sj.get("profile") or {}).get("learning_style") or ""
            ls = ls.strip() or "(none)"
            if ls in ("practice", "theory") and ls not in seen_styles:
                seen_styles.add(ls)
                picked.append((ls, rm["session_id"], sj))
            if len(picked) == 2:
                break
        if len(picked) < 2:
            # 退一步：只要有会话就用
            for r in rows:
                rm = r._mapping
                sj = json.loads(rm["state_json"]) if rm["state_json"] else {}
                ls = (sj.get("profile") or {}).get("learning_style") or "(none)"
                if any(ls == x for x, _, _ in picked):
                    continue
                picked.append((ls, rm["session_id"], sj))
                if len(picked) == 2:
                    break
        print(f"[PICKED] {len(picked)} sessions:")
        for ls, sid, _ in picked:
            print(f"  {ls:10s} sid={sid[:16]}…")

        def _load_source_cbs(cbs):
            """对每个块按 source_chunk_id 去 DB knowledge_chunks 拉 md 原文切块内容"""
            mapping = {}
            ids = []
            for b in (cbs or []):
                if isinstance(b, dict) and b.get("source_chunk_id"):
                    ids.append(b["source_chunk_id"])
            if not ids:
                return {}
            placeholders = ",".join(f":p{i}" for i in range(len(ids)))
            sql = f"SELECT chunk_id, doc_id, heading_path, content, chunk_strategy, heading_level FROM knowledge_chunks WHERE chunk_id IN ({placeholders})"
            rows_ch = conn.execute(text(sql), {f"p{i}": ids[i] for i in range(len(ids))}).fetchall()
            for r in rows_ch:
                rm = r._mapping
                mapping[rm["chunk_id"]] = {
                    "doc_id": rm["doc_id"],
                    "heading_path": rm["heading_path"] or "",
                    "content": rm["content"] or "",
                    "strategy": rm.get("chunk_strategy") or "",
                    "level": rm.get("heading_level") or 0,
                }
            return mapping

        # 先预读：查 harness / Agent3 是否打了 LLM 调用日志（直接读 state_json）
        for ls, sid, sj in picked:
            lp = sj.get("learning_package") or {}
            print(f"\n" + "=" * 100)
            print(f"[{ls}] session={sid[:14]}…  learning_package.tutorial 字符数：{len(lp.get('tutorial') or '')}  blocks={len(lp.get('content_blocks') or [])}")
            cbs = lp.get("content_blocks") or []
            mapping = _load_source_cbs(cbs)
            if not mapping:
                print("  ⚠️  content_blocks 的 source_chunk_id 在 knowledge_chunks 里找不到！（可能是旧补丁 chunk 被 import_local 删除了）")
            # 摘要统计
            sims = []
            len_ratio = []
            diff_chars = 0
            total_src_chars = 0
            print(f"  {'idx':>3s}  {'scid':28s}  {'hp_leaf':<50s}  {'src':>6s}  {'pers':>6s}  {'sim%':>6s}  首行diff标记")
            print("  " + "-" * 130)
            for i, b in enumerate(cbs if isinstance(cbs, list) else []):
                if not isinstance(b, dict):
                    continue
                scid = b.get("source_chunk_id") or ""
                pers = b.get("personalized_content") or ""
                hp = b.get("heading_path") or ""
                leaf = hp.split(">")[-1].strip() if ">" in hp else hp[:50]
                src = ""
                if scid in mapping:
                    src = mapping[scid]["content"]
                sim = ngram_sim(src, pers, 4) if src else -1.0
                src_len = len(src)
                pers_len = len(pers)
                sims.append(sim)
                total_src_chars += src_len
                diff_chars += abs(pers_len - src_len)
                len_ratio.append(pers_len / src_len if src_len else float("nan"))
                # 首行 diff：原文第一句非空 vs 改写后的第一句非空
                s1 = head_lines(src, 1)
                s2 = head_lines(pers, 1)
                if s1 and s2:
                    first_diff = "同" if s1[0] == s2[0] else f"异：…{s1[0][:30]} → {s2[0][:30]}"
                else:
                    first_diff = "(空)"
                if src:
                    print(f"  {i:>3d}  {scid:28s}  {leaf[:50]:<50s}  {src_len:>6d}  {pers_len:>6d}  {sim*100:>5.1f}%  {first_diff}")
                else:
                    print(f"  {i:>3d}  {scid:28s}  {leaf[:50]:<50s}  {src_len:>6d}  {pers_len:>6d}  ------  ❌ SOURCE NOT FOUND")
            # 汇总
            print(f"\n  相似度汇总：sims_count={len(sims)}")
            if sims:
                real = [s for s in sims if s >= 0]
                if real:
                    import statistics
                    print(f"    4gram相似度（越低改写越多，阈值 ≤0.70 算个性化有效）：")
                    print(f"      avg={sum(real)/len(real)*100:.1f}%   median={statistics.median(real)*100:.1f}%   min={min(real)*100:.1f}%   max={max(real)*100:.1f}%")
                    gt90 = sum(1 for s in real if s > 0.90)
                    gt80 = sum(1 for s in real if s > 0.80)
                    lt70 = sum(1 for s in real if s < 0.70)
                    print(f"      >90%（几乎原文照搬）块数：{gt90}/{len(real)}")
                    print(f"      >80%（轻微改写）块数：{gt80}/{len(real)}")
                    print(f"      <70%（个性化明显）块数：{lt70}/{len(real)}")
            print(f"  改写字节差(|pers-src|)：Σdiff={diff_chars:,d} / Σsrc={total_src_chars:,d} = {100*diff_chars/total_src_chars:.1f}%")
            # 肉眼 diff：找 2 个最有代表性的块
            if sims:
                pairs = sorted([(sims[i], i, cbs[i]) for i in range(len(cbs)) if isinstance(cbs[i], dict) and sims[i] >= 0],
                               key=lambda t: t[0])
                if pairs:
                    print(f"\n  [样例 A] 个性化程度最高的 1 块（sim 最低）：")
                    s, i, b = pairs[0]
                    scid = b.get("source_chunk_id") or ""
                    src = mapping.get(scid, {}).get("content", "")
                    pers = b.get("personalized_content") or ""
                    print(f"    idx={i} scid={scid} sim={s*100:.1f}%")
                    print(f"    原文前 4 行：")
                    for ln in head_lines(src, 4):
                        print(f"      |  {ln[:150]}")
                    print(f"    改写后前 4 行：")
                    for ln in head_lines(pers, 4):
                        print(f"      |  {ln[:150]}")
                    print(f"\n  [样例 B] 几乎没改的 1 块（sim 最高，>0.90 时挑）：")
                    for s, i, b in reversed(pairs):
                        if s > 0.90:
                            scid = b.get("source_chunk_id") or ""
                            src = mapping.get(scid, {}).get("content", "")
                            pers = b.get("personalized_content") or ""
                            print(f"    idx={i} scid={scid} sim={s*100:.1f}%")
                            print(f"    原文前 4 行：")
                            for ln in head_lines(src, 4):
                                print(f"      |  {ln[:150]}")
                            print(f"    改写后前 4 行：")
                            for ln in head_lines(pers, 4):
                                print(f"      |  {ln[:150]}")
                            break

        # 额外：查会话生成过程中是否有调用 DeepSeek 的证据
        print("\n" + "=" * 100)
        print("[Agent3 生成路径判断] — 检查 state_json 里 LLM 调用的 log 字段")
        for ls, sid, sj in picked:
            lp = sj.get("learning_package") or {}
            cbs = lp.get("content_blocks") or []
            # 检查 前 3 块的 personalized_content 里有没有明显 deterministic 纯规则特有的标识字符串：
            #   例如 "一句话记住它："  "先敲再解释："  "验证方式："  "本块必读锚点："  "概念摘要 + 我什么时候需要它"
            markers_theory = ["本块必读锚点", "设计动机", "概念摘要", "我什么时候需要它", "思考题", "术语自证"]
            markers_practice = ["先敲再解释", "一句话记住它", "验证方式", "立即动手", "动手验证", "报错登记"]
            hit_theory = 0; hit_practice = 0
            total_text_len = 0
            for b in (cbs or []):
                if isinstance(b, dict):
                    pc = b.get("personalized_content") or ""
                    total_text_len += len(pc)
                    hit_theory += sum(1 for m in markers_theory if m in pc)
                    hit_practice += sum(1 for m in markers_practice if m in pc)
            print(f"  [{ls}] 理论标记命中={hit_theory} 实操标记命中={hit_practice}（全文字符数={total_text_len:,d}）")
            # 结论：如果两边 marker 都=0，但相似度>90%的块很多 → 判定直接抄原文没改写
            # 如果 theory 有 markers → 走了 deterministic 纯规则（fallback）
            print(f"     → 判定：", end="")
            if hit_theory + hit_practice > 0:
                print("至少某些块走了 deterministic 纯规则改写（style 标记存在）")
            else:
                print("style 标记缺失 → 若相似度高，很可能是 LLM 保守回抄原文或没调 LLM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
