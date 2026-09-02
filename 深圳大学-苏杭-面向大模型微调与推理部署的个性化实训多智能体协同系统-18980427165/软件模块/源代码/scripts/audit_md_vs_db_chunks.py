# -*- coding: utf-8 -*-
"""对 data/知识库原始数据 下全部 .md 学习文章，与 DB knowledge_chunks 表逐篇比对。

核对维度（两篇文件：知识清单-*.md 和 测试题目.md 不纳入核对，只核对子目录下的正文 md）：
1. 文件总数：有多少篇正文 md？
2. 每篇文章：
   - 原文总字符数（去掉参考链接/版权等元数据后的正文部分，即首条 `---` 分隔符后或首条二级标题 `##` 起）
   - DB 中归属该 doc 的 chunks 总字符数（sum(LENGTH(content))）
   - **覆盖率** = chunks 拼接字符在原文中的 n-gram 命中比例
   - 缺失的关键二级大节（原文中每个 ## xxx，在所有 chunks 内容里找不到任何包含该标题句子的 → 视为缺失）
3. 汇总：全部正文 md 的"平均覆盖率 / 最差 10 篇 / 完全缺失的 doc 列表"。
"""
import json
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"
MD_ROOT = PROJECT_ROOT / "data" / "知识库原始数据"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def ngram_hit_ratio(src: str, chunks_joined: str, n: int = 4) -> float:
    """计算 chunks_joined 中 src 的 n-gram 覆盖率（用 src 做分母，即原文被覆盖了多少比例）。"""
    if not src:
        return 0.0
    src_grams = [src[i:i + n] for i in range(len(src) - n + 1)]
    if not src_grams:
        return 1.0
    # 为 O(n) 起见，构建 chunks 的 gram set
    cj_set = set(chunks_joined[i:i + n] for i in range(len(chunks_joined) - n + 1))
    hit = sum(1 for g in src_grams if g in cj_set)
    return hit / len(src_grams)


def extract_body(md_text: str) -> str:
    """取正文部分：跳过开头参考资料/元信息区（L1~适合人群这些）。
    规则：找到首次出现「## 」二级标题之前的前导元信息不做覆盖率硬要求，从第一个「## 」开始才算正文。
    若没有「## 」则从第一个「### 」起算；实在没有就取全文去掉首 10 行参考链接。
    """
    lines = md_text.splitlines()
    start_idx = 0
    for i, ln in enumerate(lines):
        if re.match(r"^##\s+\S", ln):
            start_idx = i
            break
    else:
        for i, ln in enumerate(lines):
            if re.match(r"^###\s+\S", ln):
                start_idx = i
                break
        else:
            start_idx = min(len(lines), 10)
    return "\n".join(lines[start_idx:])


def collect_h2_sections(md_body: str):
    """返回 [(title, content), ...] 按原文顺序。title 不包含 '## ' 前缀。"""
    res = []
    title = None
    buf = []
    for ln in md_body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", ln)
        if m:
            if title is not None:
                res.append((title, "\n".join(buf)))
            title = m.group(1)
            buf = []
        else:
            buf.append(ln)
    if title is not None:
        res.append((title, "\n".join(buf)))
    return res


def main():
    # 1. 先查 DB 中的 doc 清单（knowledge_docs）
    with engine.connect() as conn:
        docs_rows = conn.execute(text("""
            SELECT doc_id, block_id, title, source, source_file_path, LENGTH(content) AS raw_len
            FROM knowledge_docs ORDER BY block_id, doc_id
        """)).fetchall()
        docs_map = {}  # doc_id -> row
        for d in docs_rows:
            docs_map[d._mapping["doc_id"]] = dict(d._mapping)
        print(f"[DB] knowledge_docs 共 {len(docs_map)} 条记录")

        # 2. 收集 md 文件清单（只收二级子目录下/一级非清单文件，不含知识清单-* / *视频* / 测试题目.md）
        md_files: list[Path] = []
        for p in sorted(MD_ROOT.glob("**/*.md")):
            name = p.name
            if name.startswith("知识清单"):
                continue
            if "视频" in name:
                continue
            if name == "测试题目.md":
                continue
            md_files.append(p)
        print(f"[MD] 学习文章 md 共 {len(md_files)} 篇")

        # 3. 对每篇 md：尝试匹配 doc_id；再算覆盖率 & 缺失章节
        # 策略：匹配 doc_id 前缀 LOCAL-{block_idx}-{subidx} 用 block_idx 按 KB-xx 号推
        # 更稳妥：用 title 或 content 中 标志性 二级标题 去 docs_map 里模糊匹配（h2_set 比对）
        # 但更简单：原始 seed 入库时 doc_id 格式是 LOCAL-KBS-{KB}-{SUB}，所以按目录顺序对应
        # 直接用 heading_path 反推：doc_id 前缀对应的 chunk 都来自哪篇 md？
        # 这里用简单办法：对每个 md，把其 h2 的 title 列表做成集合，与 docs_map 中同 block 的 chunks 标题集合
        # 取 Jaccard 最高者作为匹配 doc。
        # 3. 对每篇 md：首选用 source_file_path 精确匹配 docs_map（绝对路径或相对匹配），无再回退 H2 集合 Jaccard 模糊匹配
        docs_by_sfp: dict[str, dict] = {}
        for did, drow in docs_map.items():
            sfp = str(drow.get("source_file_path") or "")
            if sfp:
                docs_by_sfp[sfp] = drow
        print(f"[DB] knowledge_docs 中 {len(docs_by_sfp)} 条记录带有 source_file_path")

        def _find_doc_by_sfp(p: Path) -> str | None:
            # 做 3 级匹配：绝对路径 → md文件名+stem 末尾 → md路径相对 MD_ROOT 的 parts 最后 3 节 与 docs.source_file_path 比对
            abs_path_str = str(p.resolve())
            for sfp, drow in docs_by_sfp.items():
                if sfp == abs_path_str or sfp.endswith(str(p)) or sfp.replace("/", "\\").endswith(str(p)):
                    return drow["doc_id"]
                # 按 MD_ROOT 相对路径
                try:
                    rel = p.relative_to(MD_ROOT).as_posix()
                except Exception:
                    rel = ""
                if rel and rel in sfp or sfp.endswith(rel):
                    return drow["doc_id"]
                # 仅用 3 段结尾（如 "5.nvidia-smi显存监控\\nvidia-smi显存监控.md"）做反匹配
                rel_parts = list(p.relative_to(MD_ROOT).parts) if str(p).startswith(str(MD_ROOT)) else list(p.parts[-3:])
                sfp_parts = [x for x in sfp.replace("\\", "/").split("/") if x]
                if len(sfp_parts) >= 2 and len(rel_parts) >= 2:
                    if tuple(sfp_parts[-2:]) == tuple(rel_parts[-2:]):
                        return drow["doc_id"]
            return None

        block_id_of_md = {}
        md_text_cache: dict[Path, str] = {}
        for p in md_files:
            txt = p.read_text(encoding="utf-8")
            md_text_cache[p] = txt
            parts = p.relative_to(MD_ROOT).parts
            # parts[0] = '01_Linux环境与命令行操作' -> KB-01
            kb_num = int(re.search(r"(\d+)", parts[0]).group(1)) if re.search(r"(\d+)", parts[0]) else None
            if kb_num:
                block_id_of_md[p] = f"KB-{kb_num:02d}"
            else:
                block_id_of_md[p] = None

        # 按 block_id 查 chunks
        kb_chunk_titles: dict[str, dict[str, list[str]]] = {}
        kb_chunks_content: dict[str, dict[str, str]] = {}
        for bid in sorted(set(v for v in block_id_of_md.values() if v)):
            kb_chunk_titles.setdefault(bid, {})
            kb_chunks_content.setdefault(bid, {})
            rows = conn.execute(text("""
                SELECT doc_id, heading_path, content FROM knowledge_chunks WHERE block_id=:bid ORDER BY chunk_index, chunk_id
            """), {"bid": bid}).fetchall()
            for r in rows:
                rm = r._mapping
                hp = rm["heading_path"] or ""
                leaf = hp.split(">")[-1].strip() if ">" in hp else hp
                kb_chunk_titles[bid].setdefault(rm["doc_id"], []).append(leaf)
                kb_chunks_content[bid].setdefault(rm["doc_id"], [])
                kb_chunks_content[bid][rm["doc_id"]].append(rm["content"] or "")
            for did in kb_chunks_content[bid]:
                kb_chunks_content[bid][did] = "\n\n".join(kb_chunks_content[bid][did])

        # 报告表
        print("\n" + "=" * 110)
        print(f"{'MD 文件':75s} {'匹配doc_id':22s}  {'KB':>5s}  {'md_body_len':>10s}  {'db_chunks_len':>12s}  {'4gram覆盖率':>11s}  缺失##大节")
        print("=" * 110)

        missing_docs = []  # 未匹配到 doc 的 md
        coverage_stats = []
        worst_10 = []

        for p in md_files:
            raw = md_text_cache[p]
            body = extract_body(raw)
            h2s = collect_h2_sections(body)  # [(title, content), ...]
            h2_titles = [t for t, _ in h2s]
            bid = block_id_of_md[p]
            matched_doc = _find_doc_by_sfp(p)
            best_jac = -1.0
            # 无命中再回退：H2 标题 Jaccard 模糊匹配（同 block 内）
            if not matched_doc and bid and bid in kb_chunk_titles:
                h2_titles = [t for t, _ in h2s]
                for did, leaves in kb_chunk_titles[bid].items():
                    s1, s2 = set(h2_titles), set(leaves)
                    if not s1 or not s2:
                        continue
                    jac = len(s1 & s2) / len(s1 | s2)
                    if jac > best_jac:
                        best_jac = jac
                        matched_doc = did
            # 再 fallback：按 docs_map title 对文件名匹配
            if not matched_doc and bid:
                fname_stem = Path(p).stem
                for did, drow in docs_map.items():
                    if drow.get("block_id") == bid and fname_stem in str(drow.get("title", "")):
                        matched_doc = did
                        break
            # 算覆盖率 & 缺失章节
            db_content = ""
            if matched_doc and bid and matched_doc in kb_chunks_content.get(bid, {}):
                db_content = kb_chunks_content[bid][matched_doc]
            cov = ngram_hit_ratio(body, db_content, n=4) if body else 0.0
            # 缺失 ## 大节判断：每个 h2 标题及其 content，如果 标题的 5-gram 都不在 db_content 算缺失
            missing_h2 = []
            for title, tcontent in h2s:
                title_in = title and (title in db_content or title[:10] in db_content)
                if not title_in:
                    # 进一步用正文 4gram 判断：若正文覆盖率低于 5% 且长度 >80 字才算缺失（避免小段落假阳性）
                    sub_cov = ngram_hit_ratio(tcontent, db_content, n=4) if len(tcontent) > 20 else 1.0
                    if sub_cov < 0.05 and len(tcontent) > 80:
                        missing_h2.append(title)
            # 打印
            md_rel = str(p.relative_to(MD_ROOT))[:72]
            did12 = (matched_doc[:20] + "…") if matched_doc else "无匹配"
            print(f"{md_rel:<75s} {did12:<22s} {str(bid or '-'):>5s}  "
                  f"{len(body):>10d}  {len(db_content):>12d}  "
                  f"{cov*100:>10.1f}%  "
                  f"{(' | '.join(missing_h2) if missing_h2 else '(无)')[:50]}")
            coverage_stats.append(cov)
            worst_10.append((cov, md_rel, matched_doc, missing_h2, len(body), len(db_content)))
            if not matched_doc:
                missing_docs.append(md_rel)

        worst_10.sort()
        print("\n" + "=" * 110)
        print(f"[汇总] {len(coverage_stats)} 篇 md：")
        if coverage_stats:
            avg_cov = sum(coverage_stats) / len(coverage_stats)
            mn_cov = min(coverage_stats)
            mx_cov = max(coverage_stats)
            median_cov = sorted(coverage_stats)[len(coverage_stats) // 2]
            print(f"  4gram原文覆盖率：avg={avg_cov*100:.1f}%  median={median_cov*100:.1f}%  min={mn_cov*100:.1f}%  max={mx_cov*100:.1f}%")
        print(f"  未匹配到 DB doc 的文件共 {len(missing_docs)} 篇：")
        for md in missing_docs[:10]:
            print(f"    - {md}")
        print("\n[覆盖率最低的 10 篇（重点检查）]")
        for cov, rel, did, missing_h2, blen, dlen in worst_10[:10]:
            print(f"  {cov*100:5.1f}%  body={blen:>6d}  db={dlen:>6d}  {rel[:65]}")
            if missing_h2:
                print(f"           缺失大节：{', '.join(missing_h2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
