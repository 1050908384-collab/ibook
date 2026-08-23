# -*- coding: utf-8 -*-
"""
问答链路「库外专业知识」只读诊断脚本
=====================================
目的（只读、不改任何生产逻辑、不写生产库、不触发 个性化学习资源生成 Agent 生成 / 专业知识与教学质量审核 Agent 改写循环）：
  实测用户问答环节中，几类问题在当前 专业知识与教学质量审核 Agent 事实核验（verify_knowledge / knowledge_verify）
  下的 **相似度分布** 与 **分支走向**，验证「库里没有对应内容的专业领域提问」会走到：
    - hallucination（confidence < 0.65，critical，会逼 个性化学习资源生成 Agent 反复改写）
    - review_needed（0.65 ~ 0.80 灰区）
    - pass（>= 0.80，返回 []）
    - unverifiable（空库降级，fallback_reason=knowledge_chunks_empty，直接放行）
  同时打印 个性化学习资源生成 Agent 侧的问答分流结果（_classify_question_by_keywords → technical/daily），
  说明哪些问题根本不会进入 专业知识与教学质量审核 Agent 核验。

方法：
  用一个内存/临时 SQLite 建最小 knowledge_chunks（只放 LoRA / QLoRA 两条），
  模拟「知识库覆盖了 LoRA/QLoRA，但没有覆盖用户所问的库外领域」。
  然后对一批问题调用真实的 KnowledgeVerifyTool.verify_knowledge（use_mock=False）。

运行：
  python tests/check_qa_out_of_kb_diagnostic.py
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path[:0] = [str(ROOT_DIR), str(BACKEND_DIR)]

os.environ.setdefault("EXECUTION_MODE", "deterministic")

from config import settings  # noqa: E402
import tools.agent4_tools.knowledge_verify_tool as kv  # noqa: E402
from tools.agent4_tools.knowledge_verify_tool import (  # noqa: E402
    KnowledgeVerifyTool,
    THRESHOLD_CRITICAL,
    THRESHOLD_PASS,
)


# ----------------------- 最小知识库（只覆盖 LoRA / QLoRA） -----------------------
LORA_TEXT = (
    "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B，"
    "通过 ΔW=(alpha/r)BA 注入到线性层，从而大幅降低可训练参数量。"
)
QLORA_TEXT = (
    "QLoRA 在 LoRA 基础上把基座模型量化到 4-bit（NF4），"
    "配合双重量化与分页优化器，在单卡上微调大模型时显著降低显存占用。"
)


def _seed_kb(db_path: Path, with_chunks: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_docs (
                doc_id VARCHAR(50) PRIMARY KEY, block_id VARCHAR(50),
                subtopic_id VARCHAR(50), title VARCHAR(200),
                content TEXT, source VARCHAR(200)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS knowledge_chunks (
                chunk_id VARCHAR(50) PRIMARY KEY, doc_id VARCHAR(50),
                block_id VARCHAR(50), chunk_index INT, content TEXT,
                chunk_type VARCHAR(20), char_start INT, char_end INT,
                heading_path TEXT, heading_level INT,
                source_line_start INT, source_line_end INT, chunk_strategy VARCHAR(30)
            )"""
        )
        docs = [
            ("DOC-LORA", "KB-12", "KBS-12-01", "LoRA 原理", LORA_TEXT),
            ("DOC-QLORA", "KB-12", "KBS-12-01", "QLoRA 量化", QLORA_TEXT),
        ]
        for doc_id, block_id, sub, title, content in docs:
            conn.execute(
                "INSERT INTO knowledge_docs(doc_id, block_id, subtopic_id, title, content, source) "
                "VALUES(?,?,?,?,?,?)",
                (doc_id, block_id, sub, title, content, "local"),
            )
        if with_chunks:
            rows = [
                ("CH-LORA", "DOC-LORA", "KB-12", 0, LORA_TEXT, "text", 0, len(LORA_TEXT),
                 "LoRA 原理", 1, 1, 5, "heading"),
                ("CH-QLORA", "DOC-QLORA", "KB-12", 0, QLORA_TEXT, "text", 0, len(QLORA_TEXT),
                 "QLoRA 量化", 1, 6, 10, "heading"),
            ]
            conn.executemany(
                "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, "
                "chunk_type, char_start, char_end, heading_path, heading_level, "
                "source_line_start, source_line_end, chunk_strategy) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        conn.commit()
    finally:
        conn.close()


# ----------------------- 待测问题集 -----------------------
# label 标注人工预期归属，便于对照实测分支是否符合预期。
QUESTIONS = [
    # 库内：应能命中 LoRA/QLoRA
    ("库内-LoRA原样", "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B。"),
    ("库内-QLoRA原样", "QLoRA 把基座模型量化到 4-bit（NF4），配合双重量化降低显存。"),
    ("库内-同义改写", "LoRA 会锁定预训练参数，仅更新一对低秩矩阵来减少训练量。"),
    # 灰区候选：主题相关但表述差异大
    ("灰区候选-LoRA部分", "LoRA 的秩 r 一般设多少比较合适？"),
    # 库外专业领域：知识库完全没有覆盖
    ("库外-扩散模型", "扩散模型里 DDPM 的前向加噪和反向去噪过程是怎么推导的？"),
    ("库外-强化学习", "PPO 算法的 clip 目标函数为什么能稳定策略更新？"),
    ("库外-数据库", "PostgreSQL 的 MVCC 是如何避免读写锁冲突的？"),
    ("库外-完全无关", "唐诗宋词的平仄格律有哪些基本规则？"),
]


def _branch_of(confidence: float, errors: list) -> str:
    """根据 verify_knowledge 返回的 errors 列表推断分支走向。"""
    if not errors:
        return "pass_or_skipped(返回[]，直接通过)"
    et = errors[0].get("error_type")
    if et == "out_of_scope":
        return "out_of_scope(库外 → 直接输出 个性化学习资源生成 Agent 内容，不改写)"
    if et == "hallucination":
        return "hallucination(critical → 逼 个性化学习资源生成 Agent 反复改写)"
    if et == "review_needed":
        return "review_needed(灰区 → major，需复核)"
    return f"other({et})"


async def _run_verify_suite(title: str, with_chunks: bool) -> None:
    print("=" * 78)
    print(f"# {title}  (THRESHOLD_CRITICAL={THRESHOLD_CRITICAL}, THRESHOLD_PASS={THRESHOLD_PASS})")
    print("=" * 78)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    _seed_kb(db_path, with_chunks=with_chunks)

    # 只读地把工具指向临时库：覆盖 settings.DATABASE_URL（_get_db_path 从这里解析）。
    # _load_chunks 的进程内缓存 key 含文件 mtime，临时库路径唯一，天然不会命中旧缓存。
    orig_url = getattr(settings, "DATABASE_URL", None)
    settings.DATABASE_URL = f"sqlite:///{db_path.as_posix()}"

    tool = KnowledgeVerifyTool()
    print(f"{'label':<16} {'lex':>6} {'conf':>6} {'method':>10} {'level':>16}  分支走向")
    print("-" * 78)
    try:
        for label, q in QUESTIONS:
            # 直接调底层 knowledge_verify 拿完整字段（confidence/level/verify_method/lexical）
            raw = kv.knowledge_verify({"statement": q}, config={"use_mock": False})
            data = raw.get("data") or {}
            conf = data.get("confidence", 0.0)
            lex = data.get("lexical_confidence", conf)
            level = data.get("level", "-")
            method = data.get("verify_method", "-")
            fallback = data.get("fallback_reason")

            # 再调 专业知识与教学质量审核 Agent 实际使用的适配器 verify_knowledge，拿真实分支
            errors = await tool.verify_knowledge(q)
            branch = _branch_of(conf, errors)
            if fallback:
                branch = f"unverifiable(降级放行, {fallback})"
            print(f"{label:<16} {lex:>6.3f} {conf:>6.3f} {str(method):>10} {str(level):>16}  {branch}")
    finally:
        if orig_url is not None:
            settings.DATABASE_URL = orig_url
        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass


def _run_classify_suite() -> None:
    """打印 个性化学习资源生成 Agent 侧问答分流：哪些问题会被判 daily（跳过 专业知识与教学质量审核 Agent），哪些 technical。"""
    from agents.deterministic import DeterministicContentAgent

    print("=" * 78)
    print("# 个性化学习资源生成 Agent 问答分流 _classify_question_by_keywords（daily 直接跳过 专业知识与教学质量审核 Agent 核验）")
    print("=" * 78)
    agent = DeterministicContentAgent()
    classify = getattr(agent, "_classify_question_by_keywords", None)
    if classify is None:
        print("(当前 个性化学习资源生成 Agent 实现无 _classify_question_by_keywords，跳过分流诊断)")
        return
    extra = [("问候", "你好呀"), ("情绪", "我最近学得好累")]
    for label, q in QUESTIONS + extra:
        try:
            qtype = classify(q)
        except Exception as e:  # noqa: BLE001
            qtype = f"ERROR({e})"
        gate = "→ 进入 专业知识与教学质量审核 Agent 核验" if qtype == "technical" else "→ 跳过 专业知识与教学质量审核 Agent（直接返回）"
        print(f"  [{qtype:<9}] {label:<12} {gate}   {q}")


async def main() -> None:
    _run_classify_suite()
    print()
    await _run_verify_suite("场景 A：知识库非空（覆盖 LoRA/QLoRA，未覆盖库外领域）", with_chunks=True)
    print()
    await _run_verify_suite("场景 B：知识库为空（knowledge_chunks 无数据）", with_chunks=False)
    print()
    print("=" * 78)
    print("说明：本脚本为只读诊断，未修改任何生产代码，未触发 个性化学习资源生成 Agent 生成 / 专业知识与教学质量审核 Agent 改写循环。")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
