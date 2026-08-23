# -*- coding: utf-8 -*-
"""
个性化学习资源生成 Agent ↔ 专业知识与教学质量审核 Agent 辩论链路独立测试 —— 完整审核/答辩/复核/裁决/修订循环

复刻 backend/orchestrator/harness.py::_review_learning_package 的完整辩论逻辑，
但不实例化 Harness/StateManager（避免 DB 依赖），Orchestrator 裁决用本地精简函数实现。

一轮辩论 = 5 步（对齐审核辩论协议第 4~5 节）：
  1. 专业知识与教学质量审核 Agent.review              -> issues（含 canonical_error_id / must_fix）
  2. 个性化学习资源生成 Agent.reply_to_review     -> author_replies（accepted / partially_accepted / rejected）
  3. 专业知识与教学质量审核 Agent.rebut               -> verdicts（persist / withdraw / downgrade）
  4. adjudicate（本地裁决）     -> accepted_issues + unresolved_must_fix_count
  5. 个性化学习资源生成 Agent.modify_package      -> 修订后 package -> 进下一轮

终止条件（对齐 harness.py 与协议第 7 节）：
  - 某轮 专业知识与教学质量审核 Agent.review.passed=true                     -> passed
  - 某轮 unresolved_must_fix_count=0                   -> passed
  - 达到 MAX_REVIEW_ITERATIONS 仍有未决 must_fix       -> needs_human_review

运行方式（命令行参数优先，无参数时回退到交互提示）：
  # 交互模式（手动选择模式 / 输入 tutorial）
  python tests/check_agent34_debate.py

  # 非交互一键跑通（自动化调试常用）
  python tests/check_agent34_debate.py --mode deterministic
  python tests/check_agent34_debate.py --mode hybrid --tutorial "自定义教程..."
  python tests/check_agent34_debate.py --mode deterministic --non-interactive

参数：
  --mode {hybrid,deterministic}  执行模式；省略则交互询问（默认 hybrid）
  --tutorial TEXT                自定义待审核 tutorial；省略则用默认（含事实错误）
  --force-unresolved             调试用：把 个性化学习资源生成 Agent 对 must_fix issue 的答辩强制改为
                                 rejected（协议允许交 Orchestrator 裁决），触发 专业知识与教学质量审核 Agent
                                 rebut=persist，从而演示「多轮辩论 + needs_human_review」
                                 终止路径（默认 deterministic 链路总在第 1 轮解决，无法覆盖此分支）
  --non-interactive              强制不读 stdin，全部用参数/默认值（管道/CI 友好）
"""
import argparse
import asyncio
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, ROOT_DIR)

from config import settings
from agents.deterministic import (
    DeterministicContentAgent,
    DeterministicTechnicalReviewAgent,
    HybridContentAgent,
    HybridTechnicalReviewAgent,
)
from models.contracts import ReviewInput
from models.profile import UserProfile
from models.review import ReviewError
from models.stage import Claim, LearningPackage, Source, StageInfo


def pretty(label: str, obj) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _can_prompt(non_interactive: bool) -> bool:
    """仅当未强制非交互、且 stdin 是 TTY 时才读取用户输入，避免管道/CI 下 EOFError。"""
    if non_interactive:
        return False
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _safe_input(prompt: str, non_interactive: bool, default: str = "") -> str:
    """交互可用时读取输入；否则返回默认值（不抛 EOFError）。"""
    if not _can_prompt(non_interactive):
        return default
    try:
        return input(prompt).strip()
    except EOFError:
        return default


def choose_mode(cli_mode: str | None, non_interactive: bool) -> str:
    if cli_mode:
        return cli_mode
    if not _can_prompt(non_interactive):
        print("\n[非交互] 未指定 --mode，默认使用 hybrid。")
        return "hybrid"
    print("\n选择执行模式：")
    print("  1) hybrid        个性化学习资源生成 Agent/4 使用本地 LLM（Ollama），查看真实辩论（默认）")
    print("  2) deterministic 个性化学习资源生成 Agent/4 纯规则/工具链路，不调 LLM，速度快")
    choice = _safe_input("请输入 1 或 2（回车=1）: ", non_interactive, "1")
    return "deterministic" if choice == "2" else "hybrid"


def adjudicate_issues(
    errors: list[ReviewError],
    verdicts: dict[str, str],
) -> tuple[list[dict], int]:
    """Orchestrator 裁决（复刻 harness.py::_adjudicate_issues，去掉 DB 落库）。

    返回 (accepted_issues, unresolved_must_fix_count)：
    - withdraw  -> accepted_issues（个性化学习资源生成 Agent 接受处理）
    - downgrade -> 降级 severity + 解除 must_fix，进 accepted_issues
    - persist   -> must_fix=true 强制 个性化学习资源生成 Agent 修改并计入未决；must_fix=false 跳过
    """
    accepted: list[dict] = []
    unresolved_must_fix_count = 0
    for err in errors:
        cid = err.canonical_error_id or ""
        verdict = verdicts.get(cid, "persist")
        if verdict == "withdraw":
            accepted.append(err.model_dump(mode="json"))
        elif verdict == "downgrade":
            err.severity = "minor" if err.severity == "major" else err.severity
            if err.must_fix:
                err.must_fix = False
                err.must_fix_reason = "专业知识与教学质量审核 Agent 复核降级，解除 must_fix"
            accepted.append(err.model_dump(mode="json"))
        else:  # persist
            if err.must_fix:
                accepted.append(err.model_dump(mode="json"))
                unresolved_must_fix_count += 1
    return accepted, unresolved_must_fix_count


# 默认学习包：故意包含事实错误，用于触发 专业知识与教学质量审核 Agent 审核 → 辩论
DEFAULT_TUTORIAL = (
    "# LoRA 微调\n\n"
    "LoRA 的核心原理是对原始权重 W 执行 SVD 分解，再训练 U、Σ、V 三个矩阵。\n"
    "若 W 维度为 d_out×d_in，LoRA 可训练参数量为 d_out*d_in。\n"
    "LoRA 必须冻结原权重，只训练低秩增量 ΔW=(α/r)BA。\n"
    "QLoRA 是 LoRA 的扩展，必须使用 4-bit 量化。\n"
)


# 默认推荐视频（取自 data/knowledge_base/video_resources.json 真实 seed 记录 RES-VID-064，
# 映射到 KB-12/KBS-12-01）。随学习包进入辩论，触发 专业知识与教学质量审核 Agent「推荐视频资源审核」（规则5）：
#   资源库已 seed -> video_content_unverified(minor)；未 seed(当前 project.db) -> video_resource_mismatch(major,must_fix)。
DEFAULT_RECOMMENDED_VIDEOS = [{
    "resource_id": "RES-VID-064",
    "title": "LoRA微调：从原理到调参，7 个问题彻底理解 LoRA",
    "url": "https://www.bilibili.com/video/BV1waZ2YDEcp",
    "review_status": "pending",
    "content_summary": "用于学习“理解LoRA的低秩分解原理”的推荐视频；导入后需补充视频内容摘要或字幕。",
    "block_id": "KB-12",
    "subtopic_id": "KBS-12-01",
    "role": "primary",
}]


def build_default_package(session_id: str, stage: StageInfo) -> LearningPackage:
    sources = [
        Source(
            source_id="src_kb12_0",
            source_type="knowledge_doc",
            title="LoRA 原理文档",
            locator="doc_lora_01",
        ),
        Source(
            source_id="src_kb12_1",
            source_type="knowledge_doc",
            title="QLoRA 量化文档",
            locator="doc_qlora_01",
        ),
    ]
    # 为每个推荐视频补一条 source_type="video" 的 source（对齐 个性化学习资源生成 Agent 真实产出 src_video_{id}），
    # 使视频错误的 source_ref=src_video_RES-VID-064 能在辩论协议字段补全时命中 sources。
    for video in DEFAULT_RECOMMENDED_VIDEOS:
        sources.append(Source(
            source_id=f"src_video_{video['resource_id']}",
            source_type="video",
            title=video["title"],
            locator=f"learning_resources:{video['resource_id']}",
            content_ref=video["url"],
            metadata_json={
                "block_id": video.get("block_id"),
                "subtopic_id": video.get("subtopic_id"),
                "review_status": video.get("review_status"),
            },
        ))
    # 说明：Claim 模型强制「is_key_technical=True 时 source_ref 不可空」（models/stage.py 校验器，
    # 对齐协议 2.2 与数据库 CHECK 约束）。clm_002 用来模拟“已绑定来源、但陈述本身是事实错误”
    # 的关键技术声明——专业知识与教学质量审核 Agent 的错误检测来自 tutorial 正文的规则/幻觉命中，不依赖 claim 是否缺来源，
    # 因此这里为 clm_002 绑定一个真实 source 并给低 confidence，既合法又保留“待纠错”的语义。
    claims = [
        Claim(
            claim_id="clm_001",
            text="LoRA 冻结原权重，训练低秩增量 ΔW=(α/r)BA",
            source_ref="src_kb12_0",
            section_ref="tutorial:sec_1",
            claim_type="concept",
            is_key_technical=True,
            confidence=0.85,
        ),
        Claim(
            claim_id="clm_002",
            text="LoRA 对原始权重 W 执行 SVD 分解",  # 事实错误，应被 专业知识与教学质量审核 Agent 抓出
            source_ref="src_kb12_0",
            section_ref="tutorial:sec_1",
            claim_type="fact",
            is_key_technical=True,
            confidence=0.4,  # 低置信度：个性化学习资源生成 Agent 自评把握不足，语义上等价“待核验”
        ),
        Claim(
            claim_id="clm_003",
            text="QLoRA 必须使用 4-bit 量化",
            source_ref="src_kb12_1",
            section_ref="tutorial:sec_3",
            claim_type="fact",
            is_key_technical=True,
            confidence=0.8,
        ),
    ]
    return LearningPackage(
        session_id=session_id,
        stage_id=stage.stage_id,
        block_id=stage.block_id,
        block_name=stage.block_name,
        tutorial=DEFAULT_TUTORIAL,
        sources=sources,
        claims=claims,
        recommended_videos=[dict(v) for v in DEFAULT_RECOMMENDED_VIDEOS],
        version=1,
    )


def _print_exchange(iteration: int, issues: list[dict], author_replies: list[dict],
                    verdicts: dict[str, str]) -> None:
    """按 canonical_error_id 对齐打印 个性化学习资源生成 Agent↔专业知识与教学质量审核 Agent 的逐条交流，方便直观追踪辩论内容。"""
    print(f"\n{'-' * 60}\n★ 第{iteration}轮 个性化学习资源生成 Agent ↔ 专业知识与教学质量审核 Agent 逐条交流对照\n{'-' * 60}")
    reply_by_cid = {r.get("canonical_error_id"): r for r in author_replies}
    for idx, iss in enumerate(issues, 1):
        cid = iss.get("canonical_error_id")
        reply = reply_by_cid.get(cid, {})
        verdict = verdicts.get(cid, "—")
        print(f"\n[{idx}] canonical_error_id={cid}  ({iss.get('error_type')}/{iss.get('severity')}"
              f", must_fix={iss.get('must_fix')})")
        print(f"    ┌ 专业知识与教学质量审核 Agent 提出   : {iss.get('description')}")
        print(f"    │   change_request={iss.get('change_request')}  source_refs={iss.get('source_refs') or '未命中来源'}")
        print(f"    ├ 个性化学习资源生成 Agent 答辩   : disposition={reply.get('disposition', '（无回复）')}")
        print(f"    │   reason={reply.get('reason', '')}")
        print(f"    │   补充来源={reply.get('source_refs') or '无'}  patched={reply.get('patched_sections') or '无'}")
        print(f"    └ 专业知识与教学质量审核 Agent 复核   : verdict={verdict}  "
              f"({'坚持' if verdict == 'persist' else '撤销' if verdict == 'withdraw' else '降级' if verdict == 'downgrade' else '—'})")


async def run_debate(args: argparse.Namespace):
    non_interactive = bool(args.non_interactive)
    print("=" * 60)
    print("个性化学习资源生成 Agent ↔ 专业知识与教学质量审核 Agent 辩论链路独立测试")
    print(f"（最多 {settings.MAX_REVIEW_ITERATIONS} 轮，复刻 harness._review_learning_package）")
    print("=" * 60)

    mode = choose_mode(args.mode, non_interactive)
    print(f"\n>>> 执行模式：{mode}")
    if mode == "hybrid":
        agent3 = HybridContentAgent()
        agent4 = HybridTechnicalReviewAgent()
    else:
        agent3 = DeterministicContentAgent()
        agent4 = DeterministicTechnicalReviewAgent()

    session_id = "test-agent34-debate"
    stage = StageInfo(
        stage_id="stage-test-34", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
        subtopic_id="KBS-12-01",  # 对齐 RES-VID-064 映射的子主题，供 专业知识与教学质量审核 Agent 视频归属校验
    )
    profile = UserProfile(session_id=session_id, goal="掌握LoRA微调")

    package = build_default_package(session_id, stage)
    custom = args.tutorial
    if custom is None:
        print("\n请输入待审核的 tutorial 内容（直接回车使用默认示例，默认故意包含事实错误以触发辩论）：")
        print("-" * 60)
        print(DEFAULT_TUTORIAL)
        print("-" * 60)
        custom = _safe_input(">>> ", non_interactive, "")
    if custom:
        package.tutorial = custom

    pretty("【初始输入】LearningPackage（个性化学习资源生成 Agent 产出，进入辩论前）", {
        "block_name": package.block_name,
        "version": package.version,
        "tutorial_length": len(package.tutorial),
        "sources_count": len(package.sources),
        "claims_count": len(package.claims),
        "recommended_videos_count": len(package.recommended_videos),
        "sources": [s.model_dump(mode="json") for s in package.sources],
        "claims": [c.model_dump(mode="json") for c in package.claims],
        "recommended_videos": [dict(v) for v in package.recommended_videos],
    })

    history: list[dict] = []
    final_status = "in_progress"
    package_status = "draft"

    for iteration in range(1, settings.MAX_REVIEW_ITERATIONS + 1):
        print(f"\n{'#' * 60}\n# 辩论第 {iteration} 轮（最多 {settings.MAX_REVIEW_ITERATIONS} 轮）\n{'#' * 60}")

        # 步骤1：专业知识与教学质量审核 Agent 审核
        print(f"\n[Step 1] 专业知识与教学质量审核 Agent.review（round_index={iteration}）...")
        review_request = ReviewInput(
            session_id=session_id,
            target_type="learning_package",
            content=package.tutorial,
            stage=stage,
            profile=profile,
            sources=[s.model_dump(mode="json") for s in package.sources],
            claims=[c.model_dump(mode="json") for c in package.claims],
            recommended_videos=[dict(v) for v in package.recommended_videos],
            round_index=iteration,
        )
        review_result = await agent4.run(review_request)
        pretty(f"【专业知识与教学质量审核 Agent 输出】第{iteration}轮 issues（共{len(review_result.errors)}条）", {
            "passed": review_result.passed,
            "errors": [e.model_dump(mode="json") for e in review_result.errors],
        })

        if review_result.passed:
            final_status = "passed"
            package_status = "approved"
            print(f"\n=== 第{iteration}轮 专业知识与教学质量审核 Agent 直接通过，辩论结束 ===")
            history.append({"iteration": iteration, "passed": True, "errors": 0})
            break

        # 步骤2：个性化学习资源生成 Agent 答辩
        print(f"\n[Step 2] 个性化学习资源生成 Agent.reply_to_review（对每条 issue 逐条回复）...")
        issues = [e.model_dump(mode="json") for e in review_result.errors]
        package_dict = package.model_dump(mode="json")
        author_replies = await agent3.reply_to_review(session_id, issues, package_dict)

        # 调试路径：强制把 must_fix issue 的答辩改为 rejected（协议允许，交 Orchestrator 裁决），
        # 使 专业知识与教学质量审核 Agent._adjudicate_reply 命中 "must_fix && rejected -> persist"，从而让
        # unresolved_must_fix_count>0，演示「多轮辩论 + needs_human_review」终止分支。
        if args.force_unresolved:
            reply_by_cid = {r.get("canonical_error_id"): r for r in author_replies}
            forced = 0
            for iss in issues:
                if not iss.get("must_fix"):
                    continue
                cid = iss.get("canonical_error_id")
                reply = reply_by_cid.get(cid)
                if reply is None:
                    continue
                reply["disposition"] = "rejected"
                reply["reason"] = "[force-unresolved] 作者坚持原表述，不接受该 must_fix 修改意见"
                reply["source_refs"] = []  # 无反证：专业知识与教学质量审核 Agent 将坚持(persist)
                reply["patched_sections"] = []
                forced += 1
            if forced:
                print(f"    [--force-unresolved] 已将 {forced} 条 must_fix 答辩改为 rejected（触发 persist）")
        pretty(f"【个性化学习资源生成 Agent 输出】author_replies（共{len(author_replies)}条）", author_replies)

        # 步骤3：专业知识与教学质量审核 Agent 复核裁决
        print(f"\n[Step 3] 专业知识与教学质量审核 Agent.rebut（对 个性化学习资源生成 Agent 答辩做 persist/withdraw/downgrade）...")
        verdicts = await agent4.rebut(session_id, issues, author_replies)
        pretty("【专业知识与教学质量审核 Agent 输出】verdicts（persist=坚持 / withdraw=撤销 / downgrade=降级）", verdicts)

        # 逐条对照打印 个性化学习资源生成 Agent↔专业知识与教学质量审核 Agent 的交流内容
        _print_exchange(iteration, issues, author_replies, verdicts)

        # 步骤4：Orchestrator 裁决（本地精简版）
        print(f"\n[Step 4] Orchestrator 裁决（adjudicate_issues）...")
        accepted_issues, unresolved_must_fix_count = adjudicate_issues(
            review_result.errors, verdicts
        )
        pretty("【Orchestrator 输出】", {
            "accepted_issues_count": len(accepted_issues),
            "unresolved_must_fix_count": unresolved_must_fix_count,
            "accepted_issues": accepted_issues,
        })

        # 步骤5：个性化学习资源生成 Agent 修改学习包
        if accepted_issues:
            print(f"\n[Step 5] 个性化学习资源生成 Agent.modify_package（处理 {len(accepted_issues)} 条已裁决 issue）...")
            revised_dict = await agent3.modify_package(
                session_id,
                package_dict,
                {"tech": review_result.errors, "quality": []},
                accepted_issues,
            )
            package = LearningPackage.model_validate(revised_dict)
            pretty(f"【个性化学习资源生成 Agent 输出】修订后 package（version={package.version}）", {
                "version": package.version,
                "tutorial_length": len(package.tutorial),
                "claims_status": [
                    {"claim_id": c.claim_id, "status": c.status}
                    for c in package.claims
                ],
                "tutorial_tail": package.tutorial[-300:],
            })
        else:
            print(f"\n[Step 5] 无 accepted_issues，跳过 modify_package")

        history.append({
            "iteration": iteration,
            "passed": False,
            "errors": len(review_result.errors),
            "unresolved_must_fix": unresolved_must_fix_count,
        })

        # 终止判定
        if unresolved_must_fix_count == 0:
            final_status = "passed"
            package_status = "approved"
            print(f"\n=== 第{iteration}轮辩论后通过（无未决 must_fix）===")
            break
        if iteration == settings.MAX_REVIEW_ITERATIONS:
            final_status = "needs_human_review"
            package_status = "provisional"
            print(
                f"\n=== 达到最大轮次仍有 {unresolved_must_fix_count} 个 must_fix 未决，"
                f"标记为 needs_human_review（package 降级为 provisional）==="
            )
            break
        print(f"\n=== 第{iteration}轮仍有 {unresolved_must_fix_count} 个 must_fix 未决，进入下一轮 ===")

    # 最终总结
    pretty("【辩论最终结果】", {
        "final_status": final_status,
        "package_status": package_status,
        "total_rounds": len(history),
        "history": history,
    })
    print("\n状态说明：")
    print("  passed              专业知识与教学质量审核 Agent 通过或所有 must_fix 已解决")
    print("  needs_human_review  达到最大轮次仍有 must_fix 未决（非阻塞，package 降级为 provisional）")
    return final_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="个性化学习资源生成 Agent↔专业知识与教学质量审核 Agent 辩论链路独立测试（命令行参数优先，无参数回退交互）",
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "deterministic"], default=None,
        help="执行模式；省略则交互询问（非交互默认 hybrid）",
    )
    parser.add_argument(
        "--tutorial", type=str, default=None,
        help="自定义待审核 tutorial；省略则用默认示例（含事实错误）",
    )
    parser.add_argument(
        "--force-unresolved", action="store_true",
        help="调试用：把 个性化学习资源生成 Agent 对 must_fix issue 的答辩强制改为 rejected，触发 专业知识与教学质量审核 Agent persist，"
             "演示多轮辩论 + needs_human_review 终止分支",
    )
    parser.add_argument(
        "--non-interactive", action="store_true",
        help="强制不读 stdin，全部用参数/默认值（管道/CI 友好）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 说明：hybrid 模式下 asyncio.run() 关闭事件循环时，底层 httpx/Ollama 连接或
    # 子进程（code_executor）在解释器退出阶段清理可能抛出非致命异常，导致 Windows 上
    # 进程返回 -1，使 CI/自动化误判为“测试失败”——即便辩论已完整跑通并打印最终结果。
    # 因此这里显式区分：辩论逻辑正常结束 -> exit 0；仅在真正的运行期异常时 -> exit 1，
    # 从而让退出码忠实反映辩论结果，不受事件循环清理噪声干扰。
    exit_code = 0
    try:
        asyncio.run(run_debate(parse_args()))
    except KeyboardInterrupt:
        print("\n已中断。")
        exit_code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    # 强制以确定的退出码结束，屏蔽事件循环关闭阶段的清理噪声（Windows 上常见的 -1）。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
