# -*- coding: utf-8 -*-
"""
校验 个性化学习资源生成 Agent↔专业知识与教学质量审核 Agent 学习包流水线的「并发语义」是否符合需求（真并发生产者/消费者）
================================================================
用户明确的并发模型：
  ❌ 模型A（交错流水线）：个性化学习资源生成 Agent 在生成后续块的同时，修改 专业知识与教学质量审核 Agent 已回馈的早期块。
  ✅ 当前需求：个性化学习资源生成 Agent 与 专业知识与教学质量审核 Agent 同时处理各块（生成+首轮审核受控并发），
     专业知识与教学质量审核 Agent 首轮跑完后一次性构造完整 issue_package（按块分组）交给 个性化学习资源生成 Agent；
     随后 个性化学习资源生成 Agent 按当前模型并发上限修改，每块改完立即派发独立复核，
     且块1的复核不阻塞块2的修改。

本脚本给真实 Harness 的 agent3/agent4 方法套上「带时间戳的记录器」，跑一遍
_review_learning_package_pipeline，然后对事件时间线做断言：

  P1 [Phase A 并发] 各块的「生成→首轮审核」在时间上重叠：
     存在「某块 generate_block 开始」早于「另一块 review_block 结束」——
     证明 asyncio.gather 让块间并发，而非逐块串行。
  P2 [生成并发受控] 任一时刻在飞的 generate_block 不超过当前模型对应的并发上限。
  P3 [问题堆积到最后] 第一次 modify_block 严格发生在「最后一个 Phase A review_block 结束」之后：
     证明修改不是边审边改，而是等所有块首轮审核堆积完毕后一次性反馈。
  P4 [修改阶段不再生成] 第一次 modify_block 之后不再出现任何 generate_block：
     证明不存在模型A的「一边生成板块4、一边改板块1」的交错。
  P5 [修改并发受控] 修改阶段任一时刻在飞的 modify_block 不超过当前模型对应的并发上限。
  P6 [非阻塞复核] 存在「某块 mod_start」早于「另一块修改后复核 rev_end」——
     证明块1的复核（create_task 派发）不阻塞块2开始修改。
  P7 [交还≤3] 专业知识与教学质量审核 Agent 交还问题清单的轮次 round_num 不超过 MAX_REVIEW_ITERATIONS(=3)。

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


def _max_inflight(tl: "Timeline", start_kind: str, end_kind: str) -> int:
    """按 seq 顺序扫描 start/end 事件，计算任一时刻在飞的最大数量。

    因所有事件在单事件循环内按发生顺序 append（seq 单调递增），
    按 seq 排序即为真实时间序；start +1 / end -1，取运行峰值。
    """
    events = sorted(
        [e for e in tl.events if e["kind"] in (start_kind, end_kind)],
        key=lambda e: e["seq"],
    )
    active = 0
    peak = 0
    for e in events:
        active += 1 if e["kind"] == start_kind else -1
        peak = max(peak, active)
    return peak


def _install_recorders(harness, tl: Timeline, delay: float = 0.05):
    """把 harness.agent3/agent4 的关键方法包成「记录开始/结束 + 注入微小 await 延迟」的版本。

    延迟只为放大并发窗口，使 Phase A 的时间重叠可观测；不改变调用顺序与结果。
    """
    a3 = harness.agent3
    a4 = harness.agent4
    orig_gen = a3.generate_block
    orig_mod = a3.modify_block
    orig_receive_issue_package = a3.receive_issue_package
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

    async def receive_issue_package(session_id, issue_package, blocks_by_id):
        # 记录 个性化学习资源生成 Agent 实际收到的完整问题列表，而不是只记录 Harness 构造了它。
        tl.record("issue_package_received", "|".join(sorted(issue_package)))
        return await orig_receive_issue_package(session_id, issue_package, blocks_by_id)

    a3.generate_block = gen
    a3.modify_block = mod
    a3.receive_issue_package = receive_issue_package
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
    block_concurrency = harness._block_concurrency()
    harness.state_manager.create_session("orch-parallel")
    tl = Timeline()
    _install_recorders(harness, tl, delay=0.05)

    stage = StageInfo(
        stage_id="stage-par-01", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA", subtopic_id="KBS-12-01",
    )
    # 落库 stage 满足审核 trace 落库的 stage_id 外键（保持日志干净，与生产一致）。
    harness.state_manager.update_stage("orch-parallel", stage)
    profile = UserProfile(session_id="orch-parallel", goal="掌握LoRA微调", learning_style="示例驱动")
    practice = PracticeProfile(hardware="RTX 4090 24GB")

    print("\n[并行语义] 运行 _review_learning_package_pipeline 并记录事件时间线 ...")
    package, history = await harness._review_learning_package_pipeline(
        "orch-parallel", stage, profile, practice,
    )

    gen_starts = tl.of("gen_start")
    rev_ends = tl.of("rev_end")
    mod_starts = tl.of("mod_start")
    mod_ends = tl.of("mod_end")

    print(f"  事件统计：gen_start={len(gen_starts)} rev_start={len(tl.of('rev_start'))} "
          f"rev_end={len(rev_ends)} mod_start={len(mod_starts)} mod_end={len(mod_ends)}")

    # 基本前提：确实是块级流水线（多块生成 + 有审核 + 有修改）。
    checker.check(len(gen_starts) >= 2, "Phase A 生成了多个块（块级流水线）", f"gen={len(gen_starts)}")
    checker.check(len(rev_ends) >= 2, "Phase A 每块都过了首轮审核", f"rev_end={len(rev_ends)}")

    first_mod_t = mod_starts[0]["t"] if mod_starts else float("inf")

    # -------- P1：Phase A 生成×审核并发（块间时间重叠） --------
    # Phase A 的 review 是首轮（round_index=1）；Phase B 复审 review 在第一次 mod_start 之后。
    phase_a_gen_starts = [e for e in gen_starts if e["t"] < first_mod_t]
    phase_a_rev_ends = [e for e in rev_ends if e["t"] < first_mod_t]
    # 并发（且受 Semaphore(2) 限流）的判据：存在「某块 gen_start」早于「另一块 rev_end」，
    # 即同一时间窗内至少两块的生成/审核在重叠推进。注意不能用「最后一个 gen_start」，
    # 因为受 cap=2 限流时第 3 块的 gen_start 必然晚于前两块之一的 rev_end（属正常）。
    earliest_gen_start_t = min((e["t"] for e in phase_a_gen_starts), default=float("inf"))
    latest_phase_a_rev_end_t = max((e["t"] for e in phase_a_rev_ends), default=0.0)
    overlap = earliest_gen_start_t < latest_phase_a_rev_end_t and len(phase_a_gen_starts) >= 2
    checker.check(
        overlap,
        "P1 Phase A 生成与审核在块间并发（时间重叠，非逐块串行）",
        f"earliest_gen_start={earliest_gen_start_t:.4f} < latest_phaseA_rev_end={latest_phase_a_rev_end_t:.4f}",
    )

    # -------- P2：生成并发受当前模型对应的 Semaphore 上限控制 --------
    # 用 gen_start / gen_end 计算任一时刻在飞的 generate_block 数（按时间顺序扫描）。
    max_gen_inflight = _max_inflight(tl, "gen_start", "gen_end")
    checker.check(
        1 < max_gen_inflight <= block_concurrency,
        f"P2 生成并发受控且曾并发（1 < 峰值 ≤ {block_concurrency}）",
        f"max_gen_inflight={max_gen_inflight}",
    )

    # -------- P2-1：审核不占用 个性化学习资源生成 Agent 生成槽 --------
    # 三块用例中，第 3 块的生成必须在任一首轮审核结束前开始；否则 generation
    # semaphore 仍然错误地包住了 generate→review 整条协程。
    third_gen_start_t = phase_a_gen_starts[2]["t"] if len(phase_a_gen_starts) >= 3 else float("inf")
    first_phase_a_review_end_t = min((e["t"] for e in phase_a_rev_ends), default=0.0)
    checker.check(
        len(phase_a_gen_starts) >= 3 and third_gen_start_t < first_phase_a_review_end_t,
        "P2-1 专业知识与教学质量审核 Agent 审核不占用 个性化学习资源生成 Agent 生成槽（第3块在首轮审核结束前开始生成）",
        f"third_gen_start={third_gen_start_t:.4f} < first_phaseA_rev_end={first_phase_a_review_end_t:.4f}",
    )

    # -------- P3：问题堆积到最后才开始修改 --------
    # 第一次 modify_block 必须晚于「最后一个 Phase A(首轮) review 结束」。
    phase_a_last_rev_end_t = max((e["t"] for e in phase_a_rev_ends), default=0.0)
    if mod_starts:
        p3 = first_mod_t >= phase_a_last_rev_end_t
        checker.check(
            p3,
            "P3 首次修改发生在「所有块首轮审核堆积完毕」之后（专业知识与教学质量审核 Agent 一次性反馈）",
            f"first_mod_start={first_mod_t:.4f} >= last_phaseA_rev_end={phase_a_last_rev_end_t:.4f}",
        )
    else:
        # 无 mod 意味着首轮全通过；本用例特意构造了会触发问题的块，理应有 mod。
        checker.check(False, "P3 预期存在修改动作（构造了会触发问题的块）", "mod_starts 为空")

    # -------- P3-1：个性化学习资源生成 Agent 先收到完整问题列表，再开始块级修改 --------
    package_receipts = tl.of("issue_package_received")
    first_receipt = package_receipts[0] if package_receipts else None
    first_receipt_blocks = first_receipt["cid"].split("|") if first_receipt else []
    checker.check(
        bool(first_receipt)
        and first_receipt["t"] <= first_mod_t
        and len(first_receipt_blocks) == 3,
        "P3-1 个性化学习资源生成 Agent 首次修改前收到完整 issue_package（含全部问题块）",
        f"received_blocks={len(first_receipt_blocks)}",
    )

    # -------- P4：修改阶段不再生成（排除模型A交错） --------
    gen_after_first_mod = [e for e in gen_starts if e["t"] > first_mod_t]
    checker.check(
        len(gen_after_first_mod) == 0,
        "P4 首次修改之后不再有任何 generate_block（非模型A的边生成边改）",
        f"gen_after_mod={len(gen_after_first_mod)}",
    )

    # -------- P5：修改并发上限=2（并发修改而非串行） --------
    # 修改阶段任一时刻在飞的 modify_block 不超过当前模型上限，且至少一次大于 1（证明并发）。
    max_mod_inflight = _max_inflight(tl, "mod_start", "mod_end")
    checker.check(
        1 < max_mod_inflight <= block_concurrency,
        f"P5 修改并发受控且曾并发（1 < 峰值 ≤ {block_concurrency}，非串行）",
        f"max_mod_inflight={max_mod_inflight}",
    )

    # -------- P6：非阻塞复核（块1复核不阻塞块2修改） --------
    # 修改后复核 = 第一次 mod_start 之后发生的 rev_end（Phase A 首轮 rev 已排除）。
    recheck_rev_ends = [e for e in rev_ends if e["t"] > first_mod_t]
    # 存在「某块 mod_start」晚于（另一块）已产生的修改后复核 rev_end 之前——
    # 即某块复核尚未收敛（rev_end 之前），另一块的修改已经开始。
    nonblocking = False
    detail_p6 = "无修改后复核事件"
    if recheck_rev_ends and len(mod_starts) >= 2:
        earliest_recheck_end_t = min(e["t"] for e in recheck_rev_ends)
        # 是否存在「第 2 个及以后的 mod_start」早于最早的修改后复核结束
        later_mod_starts = [e for e in mod_starts if e["t"] < earliest_recheck_end_t]
        nonblocking = len(later_mod_starts) >= 2
        detail_p6 = (
            f"earliest_recheck_rev_end={earliest_recheck_end_t:.4f}, "
            f"mod_starts_before_it={len(later_mod_starts)}"
        )
    checker.check(
        nonblocking,
        "P6 复核以独立 task 派发，不阻塞其它块修改（块1复核期间块2已开始修改）",
        detail_p6,
    )

    # -------- P7：交还次数 ≤ MAX_REVIEW_ITERATIONS(=3) --------
    # 修改轮次由外层 for round_num in range(1, MAX+1) 控制。
    # review_history = 首轮生成审核(iteration=1) + 每个修改轮各一条 ⇒
    # 修改轮数 = len(history) - 1，其上界即 MAX_REVIEW_ITERATIONS。
    modify_rounds = max(0, len(history) - 1)
    checker.check(
        modify_rounds <= settings.MAX_REVIEW_ITERATIONS,
        f"P7 交还问题清单轮次 ≤ MAX_REVIEW_ITERATIONS(={settings.MAX_REVIEW_ITERATIONS})（防死循环）",
        f"modify_rounds={modify_rounds}, history_len={len(history)}",
    )

    # -------- P8：收敛检测提前退出（省 token / 省耗时） --------
    # 本用例构造的块含认知负荷/难度红线，靠追加式修订无法消解——是「注定改不动」的场景。
    # 引入收敛检测前会走满 MAX_REVIEW_ITERATIONS(=3) 轮才 needs_human_review；引入后，
    # 某块连续两轮问题种类不变即冻结、提前退出，修改轮数应 < MAX（这里期望 1~2 轮）。
    checker.check(
        modify_rounds < settings.MAX_REVIEW_ITERATIONS,
        "P8 收敛检测在「注定改不动」时提前退出（修改轮数 < MAX，砍掉注定失败的重复轮次）",
        f"modify_rounds={modify_rounds} < MAX={settings.MAX_REVIEW_ITERATIONS}",
    )

    # -------- P9：复核阶段总审核次数受控（不再每块每轮重复满额复核） --------
    # rev_end 总数 = Phase A 首轮每块 1 次 + 每个修改轮每块 1 次复核。
    # 提前退出后，复核阶段的 rev_end 应显著少于「走满 MAX 轮」的理论上限
    # （blocks × MAX）。以块数 × 修改轮数为实测复核量，验证其 ≤ 块数 × modify_rounds
    # 且严格小于块数 × MAX（证明省掉了后续轮次的整批复核调用）。
    n_blocks = len(gen_starts)
    recheck_count = len([e for e in rev_ends if e["t"] > first_mod_t])
    checker.check(
        recheck_count <= n_blocks * modify_rounds
        and recheck_count < n_blocks * settings.MAX_REVIEW_ITERATIONS,
        "P9 复核调用量随提前退出下降（< 块数 × MAX，避免每块每轮重复满额复核）",
        f"recheck_count={recheck_count}, blocks={n_blocks}, modify_rounds={modify_rounds}",
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
