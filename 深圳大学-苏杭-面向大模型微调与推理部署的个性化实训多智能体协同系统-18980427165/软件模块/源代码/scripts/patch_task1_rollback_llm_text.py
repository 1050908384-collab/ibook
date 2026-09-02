# -*- coding: utf-8 -*-
"""Task1：把现有两个 Linux session 中所有块的 personalized_content 里
你贴的那段「2.1 最简命令 + 体系起点 + 思考题 + 概念对比」的 LLM 风格改写
**全部替换成 nvidia-smi.md 原文 2.1 节的纯技术内容**（逐字，不做任何表达层扩写/补动机/加思考题）。

规则：对所有 content_blocks，如果它的 source_chunk_id 是
  - LOCAL-KBS-01-04-S01（原 patch 的第一章替代品）
  - LOCAL-KBS-01-04-C001 （2.1 最简命令 —— 用户指认要删的那段就在这里）
  - LOCAL-KBS-01-04-C002 （2.2 关键指标解读 下面内容如果有 LLM 额外补也顺便回退）
  - 以及任何 heading_path 含 "2.1 最简命令" 或 "1.1 nvidia-smi 是什么" 的块

一律用 nvidia-smi.md 对应原文行范围逐字替换（不调 LLM，纯逐字）：
  LOCAL-KBS-01-04-S01       ->  nvidia-smi.md L19-L36  （## 一、nvidia-smi 是什么？整大节）
  LOCAL-KBS-01-04-C001      ->  nvidia-smi.md L37-L59  （## 二、基础用法 2.1 最简命令 全文）
  LOCAL-KBS-01-04-C002      ->  nvidia-smi.md L60-L78  （2.2 关键指标解读 全文）

然后重跑 fix_existing_tutorials.py 用 HR-clean + 新聚合刷新 session tutorial。
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"
MD = PROJECT_ROOT / "data" / "知识库原始数据" / "01_Linux环境与命令行操作" / "5.nvidia-smi显存监控" / "nvidia-smi显存监控.md"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def _lines_range(lines, start_lineno_1based, end_lineno_1based_inclusive):
    return "\n".join(lines[start_lineno_1based - 1 : end_lineno_1based_inclusive])


def main():
    all_lines = MD.read_text(encoding="utf-8-sig").splitlines()
    # 按原文 ## 级大节切：
    # 一、nvidia-smi 是什么？    L19-L36（含 19行的 ## 标题 到 L36 的空行）
    ch1_raw = _lines_range(all_lines, 19, 36).strip()
    # 二、基础用法：单次查看 GPU 状态（①+④）下面的 2.1 最简命令（含 ### 2.1 标题 + 代码块 + 表格样例结束 到 2.2 标题前）
    # L37 = "## 二、..."   L39 = "### 2.1 最简命令"  L60 = "### 2.2 关键指标解读（①+④）"
    # 2.1 的范围：L37（二、基础用法 ## 标题的起始 + 下面 2.1 整节）到 L59 结束（2.2 标题前 1 行）
    ch21_raw = _lines_range(all_lines, 37, 59).strip()
    # 2.2 关键指标解读：L60-L77（L77 是 "调小 batch size；GPU-Util 低...。（④）"，L78是空行，L79=---）
    ch22_raw = _lines_range(all_lines, 60, 77).strip()

    # 映射：source_chunk_id -> (raw_content, leaf_heading 用于外层兜底标题)
    MAP = {
        "LOCAL-KBS-01-04-S01": ch1_raw,
        "LOCAL-KBS-01-04-C001": ch21_raw,
        "LOCAL-KBS-01-04-C002": ch22_raw,
    }

    with engine.connect() as conn:
        sids = [r[0] for r in conn.execute(text("SELECT session_id FROM sessions WHERE LOWER(goal) LIKE '%linux%'")).fetchall()]
        changed_any = False
        total_changes = 0
        for sid in sids:
            row = conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:sid"), {"sid": sid}).first()
            old_sj = row[0]
            st = json.loads(old_sj)
            lp = st.get("learning_package") or {}
            cbs = lp.get("content_blocks") or []
            if not isinstance(cbs, list):
                continue
            for b in cbs:
                if not isinstance(b, dict):
                    continue
                scid = b.get("source_chunk_id") or ""
                # 精确命中 S01 / C001 / C002 直接替换
                if scid in MAP:
                    new_content = MAP[scid]
                    old_content = b.get("personalized_content") or ""
                    if old_content != new_content:
                        b["personalized_content"] = new_content
                        total_changes += 1
                        changed_any = True
                        print(f"[{sid[:12]}] ✏️ 替换 source={scid:25s} （{len(old_content)} -> {len(new_content)} chars）")
                else:
                    # fallback：如果块的 personalized_content 中同时出现以下关键字之一（你指认的 LLM 风格内容），也回退为 C001 原文
                    #   "这一行命令，就是整个 GPU 监控体系的起点"  "设计动机"  "思考题"  "概念对比"  "零配置" + "trade-off"
                    suspect_keywords = [
                        "这一行命令，就是整个 GPU 监控体系的起点",
                        "思考题", "概念对比", "概念与设计动机",
                        "零配置", "trade-off",
                        "GPU 是一个黑盒，你需要一个官方认可的窗口去观察它",
                    ]
                    pc = b.get("personalized_content") or ""
                    hp = b.get("heading_path") or ""
                    if pc and any(kw in pc for kw in suspect_keywords) and ("2.1 最简命令" in hp or scid == "LOCAL-KBS-01-04-C001"):
                        old_content = pc
                        b["personalized_content"] = ch21_raw
                        total_changes += 1
                        changed_any = True
                        print(f"[{sid[:12]}] ✏️ （关键字命中）替换 hp={hp[:60]} （{len(old_content)} -> {len(ch21_raw)} chars）")
            if changed_any:
                lp["content_blocks"] = cbs
                st["learning_package"] = lp
                conn.execute(text("""
                    UPDATE session_states SET state_json=:sj, updated_at=CURRENT_TIMESTAMP WHERE session_id=:sid
                """), {"sj": json.dumps(st, ensure_ascii=False), "sid": sid})
        conn.commit()
    print(f"\n[DONE] 共替换 {total_changes} 个 content_blocks 的 personalized_content。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
