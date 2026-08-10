# -*- coding: utf-8 -*-
"""
校验 Agent3↔Agent4 学习包流水线的「并行语义」是否符合需求（模型 B）
================================================================
用户明确的并行模型：
  ❌ 模型A（交错流水线）：Agent3 在生成后续块的同时，修改 Agent4 已回馈的早期块。
  ✅ 模型B（当前需求）：Agent3 与 Agent4 同时处理各块（生成+首轮审核并发），
     但 Agent4 把问题「堆积到最后」再统一反馈给 Agent3；随后 Agent3 才逐块修改。

本脚本给真实 Harness 的 agent3/agent4 方法套上「带时间戳的记录器」，跑一遍
_review_learning_package_pipeline，然后对事件时间线做断言：

  P1 [Phase A 并发] 各块的「生成→首轮审核」在时间上重叠：
     存在「某块 generate_block 开始」早于「另一块 review_block 结束」——
     证明 asyncio.gather 让块间并发，而非逐块串行。
  P2 [问题堆积到最后] 第一次 modify_block 严格发生在「最后一个 Phase A review_block 结束」之后：
     证明修改不是边审边改，而是等所有块首轮审核堆积完毕后统一开始。
  P3 [修改阶段不再生成] 第一次 modify_block 之后不再出现任何 generate_block：
     证明不存在模型A的「一边生成板块4、一边改板块1」的交错。
  P4 [Phase B 逐块串行] Phase B 内 modify_block 之间不重叠（串行 await）。

运行：
  python tests/check_pipeline_parallel_semantics.py
"""
import asyncio
import itertools
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from config import settings  # noqa: E402

LORA_TEXT = (
    "LoRA 冻结原始预训练权重，只训练低秩增量矩阵 A 和 B，"
    "通过 ΔW=(alpha/r)BA 注入到线性层，从而大幅降低可训练参数量。"
)
QLORA_TEXT = (
    "QLoRA 在 LoRA 基础上把基座模型量化到 4-bit（NF4），"
    "配合双重量化与分页优化器，在单卡上微调大模型时显著降低显存占用。"
)
# 第三块：让并发窗口更明显（3 块 gather 更容易观测到重叠）。
ADAPTER_TEXT = (
    "Adapter 在 Transformer 层之间插入小型瓶颈网络，只训练这些新增模块，"
    "冻结原始权重，从而以少量参数完成下游任务适配。"
)


def _seed(db_path: Path) -> None:
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
            ("DOC-ADP", "KB-12", "KBS-12-01", "Adapter 微调", ADAPTER_TEXT),
        ]
        for doc_id, block_id, sub, title, content in docs:
            conn.execute(
                "INSERT INTO knowledge_docs(doc_id, block_id, subtopic_id, title, content, source) "
                "VALUES(?,?,?,?,?,?)",
                (doc_id, block_id, sub, title, content, "local"),
            )
        chunks = [
            ("CHK-LORA-0", "DOC-LORA", 0, LORA_TEXT, "LoRA 微调 > 原理", 10, 24),
            ("CHK-QLORA-0", "DOC-QLORA", 0, QLORA_TEXT, "QLoRA > 量化", 30, 41),
            ("CHK-ADP-0", "DOC-ADP", 0, ADAPTER_TEXT, "Adapter > 结构", 50, 60),
        ]
        for chunk_id, doc_id, idx, content, heading, ls, le in chunks:
            conn.execute(
                "INSERT INTO knowledge_chunks(chunk_id, doc_id, block_id, chunk_index, content, "
                "chunk_type, heading_path, heading_level, source_line_start, source_line_end, chunk_strategy) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (chunk_id, doc_id, "KB-12", idx, content, "text", heading, 2, ls, le, "heading"),
            )
        conn.commit()
    finally:
        conn.close()


class Timeline:
    """线程/协程安全无需：单事件循环内顺序记录带时间戳的事件。"""
    def __init__(self) -> None:
        self.events = []  # list[dict(seq, t, phase, kind, cid)]
        self._seq = itertools.count()

    def record(self, kind: str, cid: str) -> None:
        self.events.append({
            "seq": next(self._seq),
            "t": time.perf_counter(),
            "kind": kind,   # gen_start / gen_end / rev_start / rev_end / mod_start / mod_end
            "cid": cid,
        })

    def of(self, kind: str):
        return [e for e in self.events if e["kind"] == kind]

    def first(self, kind: str):
        xs = self.of(kind)
        return xs[0] if xs else None

    def last(self, kind: str):
        xs = self.of(kind)
        return xs[-1] if xs else None


def _install_recorders(harness, tl: Timeline, delay: float = 0.05):
    """把 harness.agent3/agent4 的关键方法包成「记录开始/结束 + 注入微小 await 延迟」的版本。

    延迟只为放大并发窗口，使 Phase A 的时间重叠可观测；不改变调用顺序与结果。
    """
    a3 = harness.agent3
    a4 = harness.agent4
    orig_gen = a3.generate_block
    orig_mod = a3.modify_block
    orig_rev = a4.review_block

    def _cid_of_block(block) -> str:
        return getattr(block, "content_block_id", "?")

    async def gen(*args, **kwargs):
        # 生成前我们还没有 content_block_id，用 source_chunk 的 chunk_id 作占位。
        src = kwargs.get("source_chunk") or {}
        placeholder = src.get("chunk_id") or f"idx{kwargs.get('block_index')}"
        tl.record("gen_start", placeholder)
        await asyncio.sleep(delay)
        block = await orig_gen(*args, **kwargs)
        tl.record("gen_end", _cid_of_block(block))
        return block

    async def rev(*args, **kwargs):
        block = kwargs.get("block") or (args[0] if args else None)
        cid = _cid_of_block(block)
        tl.record("rev_start", cid)
        await asyncio.sleep(delay)
        result = await orig_rev(*args, **kwargs)
        tl.record("rev_end", cid)
        return result

    async def mod(block, *args, **kwargs):
        cid = _cid_of_block(block)
        tl.record("mod_start", cid)
        await asyncio.sleep(delay)
        revised = await orig_mod(block, *args, **kwargs)
        tl.record("mod_end", cid)
        return revised

    a3.generate_block = gen
    a3.modify_block = mod
    a4.review_block = rev


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {mark}  {label}{(' | ' + detail) if detail else ''}")
        self.passed += int(bool(ok))
        self.failed += int(not ok)

    def summary(self) -> int:
        print(f"\n{'#' * 64}\n# 并行语义断言汇总：{self.passed} 通过 / {self.failed} 失败\n{'#' * 64}")
        return 0 if self.failed == 0 else 1


async def run(checker: Checker) -> None:
    from orchestrator.harness import Harness
    from models.profile import PracticeProfile, UserProfile
    from models.stage import StageInfo

    harness = Harness(execution_mode="deterministic")
    harness.state_manager.create_session("orch-parallel")
    tl = Timeline()
    _install_recorders(harness, tl, delay=0.05)

    stage = StageInfo(
        stage_id="stage-par-01", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    profile = UserProfile(session_id="orch-parallel", goal="掌握LoRA微调", learning_style="示例驱动")
    practice = PracticeProfile(hardware="RTX 4090 24GB")

    print("\n[并行语义] 运行 _review_learning_package_pipeline 并记录事件时间线 ...")
    await harness._review_learning_package_pipeline("orch-parallel", stage, profile, practice)

    gen_starts = tl.of("gen_start")
    rev_ends = tl.of("rev_end")
    mod_starts = tl.of("mod_start")
    mod_ends = tl.of("mod_end")

    print(f"  事件统计：gen_start={len(gen_starts)} rev_start={len(tl.of('rev_start'))} "
          f"rev_end={len(rev_ends)} mod_start={len(mod_starts)} mod_end={len(mod_ends)}")

    # 基本前提：确实是块级流水线（多块生成 + 有审核 + 有修改）。
    checker.check(len(gen_starts) >= 2, "Phase A 生成了多个块（块级流水线）", f"gen={len(gen_starts)}")
    checker.check(len(rev_ends) >= 2, "Phase A 每块都过了首轮审核", f"rev_end={len(rev_ends)}")

    # -------- P1：Phase A 生成×审核并发（块间时间重叠） --------
    # Phase A 的 review 是首轮（round_index=1）；Phase B 复审 review 在第一次 mod_start 之后。
    first_mod_t = mod_starts[0]["t"] if mod_starts else float("inf")
    phase_a_gen_starts = [e for e in gen_starts if e["t"] < first_mod_t]
    phase_a_rev_ends = [e for e in rev_ends if e["t"] < first_mod_t]
    # 若逐块串行：每个 gen_start 都应晚于前一块的 rev_end。并发则相反：
    # 存在「后一块 gen_start」早于「前面某块 rev_end」。
    last_gen_start_t = max((e["t"] for e in phase_a_gen_starts), default=0.0)
    earliest_rev_end_t = min((e["t"] for e in phase_a_rev_ends), default=float("inf"))
    overlap = last_gen_start_t < earliest_rev_end_t
    checker.check(
        overlap,
        "P1 Phase A 生成与审核在块间并发（时间重叠，非逐块串行）",
        f"last_gen_start={last_gen_start_t:.4f} < earliest_rev_end={earliest_rev_end_t:.4f}",
    )

    # -------- P2：问题堆积到最后才开始修改 --------
    # 第一次 modify_block 必须晚于「最后一个 Phase A(首轮) review 结束」。
    # 首轮 rev_end 即第一次 mod_start 之前的所有 rev_end。
    phase_a_last_rev_end_t = max((e["t"] for e in phase_a_rev_ends), default=0.0)
    if mod_starts:
        p2 = first_mod_t >= phase_a_last_rev_end_t
        checker.check(
            p2,
            "P2 首次修改发生在「所有块首轮审核堆积完毕」之后（Agent4 攒到最后再反馈）",
            f"first_mod_start={first_mod_t:.4f} >= last_phaseA_rev_end={phase_a_last_rev_end_t:.4f}",
        )
    else:
        # 无 mod 意味着首轮全通过；本用例特意构造了会触发问题的块，理应有 mod。
        checker.check(False, "P2 预期存在修改动作（构造了会触发问题的块）", "mod_starts 为空")

    # -------- P3：修改阶段不再生成（排除模型A交错） --------
    gen_after_first_mod = [e for e in gen_starts if e["t"] > first_mod_t]
    checker.check(
        len(gen_after_first_mod) == 0,
        "P3 首次修改之后不再有任何 generate_block（非模型A的边生成边改）",
        f"gen_after_mod={len(gen_after_first_mod)}",
    )

    # -------- P4：Phase B 逐块修改串行（modify_block 之间不重叠） --------
    # 串行 => 每个 mod_start 都不早于上一个 mod_end。
    serial = True
    detail = ""
    events = sorted([e for e in tl.events if e["kind"] in ("mod_start", "mod_end")], key=lambda e: e["seq"])
    active = 0
    max_active = 0
    for e in events:
        active += 1 if e["kind"] == "mod_start" else -1
        max_active = max(max_active, active)
    serial = max_active <= 1
    checker.check(
        serial,
        "P4 Phase B 逐块修改为串行（同一时刻至多 1 个 modify_block 在跑）",
        f"max_concurrent_modify={max_active}",
    )

    harness.state_manager.db_client.close()


def main() -> None:
    orig_url = settings.DATABASE_URL
    orig_flag = settings.BLOCK_PIPELINE_ENABLED
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    db_path = Path(temp_dir.name) / "par.db"
    settings.DATABASE_URL = f"sqlite:///{db_path.as_posix()}"
    settings.BLOCK_PIPELINE_ENABLED = True
    _seed(db_path)
    try:
        import tools.agent4_tools.knowledge_verify_tool as kv
        kv._CHUNK_CACHE.update(key=None, chunks=[], db_available=False)
    except Exception:
        pass

    checker = Checker()
    exit_code = 0
    try:
        asyncio.run(run(checker))
        exit_code = checker.summary()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 脚本执行异常：{exc}")
        exit_code = 2
    finally:
        settings.DATABASE_URL = orig_url
        settings.BLOCK_PIPELINE_ENABLED = orig_flag
        try:
            temp_dir.cleanup()
        except Exception:
            pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
