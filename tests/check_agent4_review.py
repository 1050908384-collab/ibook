# -*- coding: utf-8 -*-
"""
Agent4 独立测试 —— 技术+质量审核 Agent（review + rebut 辩论协议）

职责（对齐 backend/agents/deterministic.py）：
  - 审核 learning_package / answer -> TechnicalReviewResult
    每个 error 强制携带：canonical_error_id / error_id / source_refs / change_request /
    must_fix / must_fix_reason / claim_id（审核辩论协议 2.3 / 2.4 / 第 6 节）
  - rebut 复核：对 Agent3 的逐条回复做裁决 -> {canonical_error_id: persist|withdraw|downgrade}

运行方式（命令行参数优先，无参数时回退到交互提示）：
  # 交互模式
  python tests/check_agent4_review.py

  # 非交互一键跑通（自动化调试常用）
  python tests/check_agent4_review.py --mode deterministic --rebut
  python tests/check_agent4_review.py --mode deterministic --target answer --rebut
  python tests/check_agent4_review.py --mode deterministic --verbose --rebut
  python tests/check_agent4_review.py --mode hybrid --content "自定义内容..." --non-interactive

参数：
  --mode {hybrid,deterministic}          执行模式；省略则交互询问（非交互默认 hybrid）
  --target {learning_package,answer}     审核目标类型；省略则交互询问（非交互默认 learning_package）
  --content TEXT                         自定义待审核内容；省略则用默认（含事实错误）
  --verbose                              逐步打印 Agent4 内部工具链的输入/输出（仅 deterministic 生效）：
                                         规则检查→幻觉检测→难度→认知负荷→风格→辩论协议字段补全
  --rebut / --no-rebut                   是否继续 rebut 复核；省略则交互询问（非交互默认执行 rebut）
  --non-interactive                      强制不读 stdin，全部用参数/默认值（管道/CI 友好）

说明：
  - 默认示例内容「故意包含事实错误」（LoRA 不是 SVD 分解；参数量算错），
    用于触发规则检查 / 幻觉检测，直观看到审核输出。
  - rebut 步骤会自动按 Agent3 风格构造 author_replies（must_fix->partially_accepted，其余->accepted），
    无需手写即可观察 persist/withdraw/downgrade 裁决。
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

from agents.deterministic import (
    DeterministicTechnicalReviewAgent,
    HybridTechnicalReviewAgent,
    normalize_review_errors,
)
from models.contracts import ReviewInput
from models.profile import UserProfile
from models.stage import StageInfo


def pretty(label: str, obj) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _dump_errors(errors) -> str:
    """把工具原始 dict 列表 / ReviewError 列表统一转为可读 JSON。"""
    out = []
    for e in errors or []:
        if hasattr(e, "model_dump"):
            out.append(e.model_dump(mode="json"))
        else:
            out.append(e)
    return json.dumps(out, ensure_ascii=False, indent=2, default=str)


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
    print("  1) hybrid        使用本地 LLM（Ollama），查看真实模型输出（默认）")
    print("  2) deterministic 纯规则/工具链路，不调用 LLM，速度快")
    choice = _safe_input("请输入 1 或 2（回车=1）: ", non_interactive, "1")
    return "deterministic" if choice == "2" else "hybrid"


def choose_target(cli_target: str | None, non_interactive: bool) -> str:
    if cli_target:
        return cli_target
    if not _can_prompt(non_interactive):
        print("[非交互] 未指定 --target，默认使用 learning_package。")
        return "learning_package"
    print("\n选择审核目标类型：")
    print("  1) learning_package  审核学习包 tutorial（默认）")
    print("  2) answer            审核问答回答")
    t = _safe_input("请输入 1 或 2（回车=1）: ", non_interactive, "1")
    return "answer" if t == "2" else "learning_package"


DEFAULT_CONTENT_LP = (
    "# LoRA 微调\n\n"
    "LoRA 的核心原理是对原始权重 W 执行 SVD 分解，再训练 U、Σ、V 三个矩阵。\n"
    "若 W 维度为 d_out×d_in，LoRA 可训练参数量为 d_out*d_in。\n"
    "LoRA 必须冻结原权重，只训练低秩增量 ΔW=(α/r)BA。\n"
)
DEFAULT_CONTENT_ANS = "LoRA 通过对权重矩阵做 SVD 分解来减少参数量，训练 U、Σ、V 三个矩阵。"

# 默认推荐视频（取自 data/knowledge_base/video_resources.json 的真实 seed 记录 RES-VID-064，
# 映射到 KB-12/KBS-12-01）。用于触发 Agent4 的「推荐视频资源审核」（technical_review.skill.md 规则5）：
#   - 资源库已 seed（learning_resources 有该记录，mapping_status=confirmed）时 -> video_content_unverified（minor，review_status=pending）
#   - 资源库未 seed（当前 project.db 尚无 learning_resources 表）时      -> video_resource_mismatch（major，must_fix）
# 两种资源库状态下都会命中视频校验路径，测试保持稳健。answer 场景不涉及视频，videos 传空。
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


async def show_tool_chain(
    agent: DeterministicTechnicalReviewAgent,
    request: ReviewInput,
) -> None:
    """逐个调用 Agent4 内部工具，打印链路中每一步的输入/输出（--verbose）。

    与 DeterministicTechnicalReviewAgent.run 的调用顺序严格一致，便于对照调试：
      1. rule_check_tool.check_rules          -> 硬性规则命中（LoRA机制/参数量/安全红线）
      2. knowledge_verify_tool.verify_knowledge -> 事实检查/幻觉（sim<0.65 判幻觉）
      3.（仅 learning_package）
         difficulty_analyzer_tool.analyze     -> 难度匹配
         cognitive_load_tool.analyze          -> 认知负荷
         style_match_tool.check               -> 风格匹配
      4. _protocol_agent._enrich_debate_fields -> 补全辩论协议字段
         （canonical_error_id / must_fix / change_request / source_refs / claim_id）

    仅在 deterministic 模式下逐步展示（hybrid 内部走 LLM 提示词，工具链不逐个暴露）。
    """
    print(f"\n{'*' * 60}\n★ Agent4 内部工具链逐步输入/输出（--verbose）\n{'*' * 60}")
    content = request.content
    profile = request.profile.model_dump(mode="json") if hasattr(request.profile, "model_dump") else request.profile
    raw_errors: list = []

    # 1. 硬性检查（规则检查）
    print("\n[Tool 1] rule_check_tool.check_rules")
    print(f"  输入 content（前120字）: {content[:120]}")
    try:
        rule_errors = await agent.rule_check_tool.check_rules(content)
    except Exception as exc:  # 兜底：与 run() 一致，单工具异常不应中断链路
        rule_errors = []
        print(f"  [异常兜底] {type(exc).__name__}: {exc}")
    print(f"  输出 rule_errors（{len(rule_errors)}条）: {_dump_errors(rule_errors)}")
    raw_errors.extend(rule_errors)

    # 2. 事实检查（幻觉检测）
    print("\n[Tool 2] knowledge_verify_tool.verify_knowledge")
    print(f"  输入 content（前120字）: {content[:120]}")
    try:
        hallucination_errors = await agent.knowledge_verify_tool.verify_knowledge(content)
    except Exception as exc:
        hallucination_errors = []
        print(f"  [异常兜底] {type(exc).__name__}: {exc}")
    print(f"  输出 hallucination_errors（{len(hallucination_errors)}条）: {_dump_errors(hallucination_errors)}")
    raw_errors.extend(hallucination_errors)

    # 3. 质量审核（仅 learning_package，与 run() 分支一致）
    video_raw_errors: list = []
    if request.target_type == "learning_package":
        print("\n[Tool 3] difficulty_analyzer_tool.analyze")
        print(f"  输入 profile: {json.dumps(profile, ensure_ascii=False)}")
        try:
            difficulty_errors = await agent.difficulty_analyzer_tool.analyze(content, profile=profile)
        except Exception as exc:
            difficulty_errors = []
            print(f"  [异常兜底] {type(exc).__name__}: {exc}")
        print(f"  输出 difficulty_errors（{len(difficulty_errors)}条）: {_dump_errors(difficulty_errors)}")
        raw_errors.extend(difficulty_errors)

        print("\n[Tool 4] cognitive_load_tool.analyze")
        try:
            load_errors = await agent.cognitive_load_tool.analyze(content)
        except Exception as exc:
            load_errors = []
            print(f"  [异常兜底] {type(exc).__name__}: {exc}")
        print(f"  输出 load_errors（{len(load_errors)}条）: {_dump_errors(load_errors)}")
        raw_errors.extend(load_errors)

        print("\n[Tool 5] style_match_tool.check")
        try:
            style_errors = await agent.style_match_tool.check(content, profile=profile)
        except Exception as exc:
            style_errors = []
            print(f"  [异常兜底] {type(exc).__name__}: {exc}")
        print(f"  输出 style_errors（{len(style_errors)}条）: {_dump_errors(style_errors)}")
        raw_errors.extend(style_errors)

        # 步骤5：推荐视频资源审核（technical_review.skill.md 规则5，与 run() 分支一致）
        print("\n[Tool 5.5] learning_resource_tool.validate_recommended_videos")
        print(f"  输入 recommended_videos（{len(request.recommended_videos)}个）: "
              f"{json.dumps(request.recommended_videos, ensure_ascii=False)}")
        print(f"  输入 block_id={request.stage.block_id} | subtopic_id={request.stage.subtopic_id}")
        try:
            video_raw_errors = await agent.learning_resource_tool.validate_recommended_videos(
                request.recommended_videos,
                request.stage.block_id,
                request.stage.subtopic_id,
            )
        except Exception as exc:
            video_raw_errors = []
            print(f"  [异常兜底] {type(exc).__name__}: {exc}")
        print(f"  输出 video_resource_errors（{len(video_raw_errors)}条）: {_dump_errors(video_raw_errors)}")
        print("  说明：资源库已 seed -> video_content_unverified(minor)；未 seed -> video_resource_mismatch(major,must_fix)")
        raw_errors.extend(video_raw_errors)
    else:
        print("\n[Tool 3-5.5] （target=answer，跳过 difficulty/cognitive_load/style/视频审核，与 run() 一致）")

    # 4. 综合裁决 + 辩论协议字段补全（raw_errors 含视频错误，与 run() 传 raw_errors=video_resource_errors 对齐）
    print("\n[Tool 6] normalize_review_errors + _protocol_agent._enrich_debate_fields")
    print(f"  输入 raw_errors（{len(raw_errors)}条，未归一化，含视频错误 {len(video_raw_errors)} 条）")
    normalized = normalize_review_errors(raw_errors)
    enriched = agent._protocol_agent._enrich_debate_fields(
        normalized,
        sources=request.sources,
        claims=request.claims,
        round_index=request.round_index,
        raw_errors=video_raw_errors,
    )
    print(f"  输出 enriched errors（{len(enriched)}条，含 canonical_error_id/must_fix/change_request）:")
    print(f"  {_dump_errors(enriched)}")


async def main(args: argparse.Namespace):
    non_interactive = bool(args.non_interactive)
    print("=" * 60)
    print("Agent4 技术+质量审核 Agent 独立测试（review + rebut）")
    print("=" * 60)

    mode = choose_mode(args.mode, non_interactive)
    print(f"\n>>> 执行模式：{mode}")
    agent = HybridTechnicalReviewAgent() if mode == "hybrid" else DeterministicTechnicalReviewAgent()

    target_type = choose_target(args.target, non_interactive)
    print(f">>> 审核目标类型：{target_type}")

    default_content = DEFAULT_CONTENT_LP if target_type == "learning_package" else DEFAULT_CONTENT_ANS
    content = args.content
    if content is None:
        print("\n请输入待审核内容（直接回车使用默认示例，默认故意包含事实错误以触发审核）：")
        print("-" * 60)
        print(default_content)
        print("-" * 60)
        content = _safe_input(">>> ", non_interactive, "")
    if not content:
        content = default_content

    session_id = "test-agent4-session"
    stage = StageInfo(
        stage_id="stage-test-04", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
        subtopic_id="KBS-12-01",
    )
    profile = UserProfile(session_id=session_id, goal="掌握LoRA微调")

    # 构造 sources / claims，让辩论协议字段 enrichment 生效
    sources = [{
        "source_id": "src_kb12_0",
        "source_type": "knowledge_doc",
        "title": "LoRA 原理文档",
        "locator": "doc_lora_01",
    }]
    claims = [{
        "claim_id": "clm_001",
        "text": "LoRA 冻结原权重，训练低秩增量 ΔW=(α/r)BA",
        "source_ref": "src_kb12_0",
        "section_ref": "tutorial:sec_0",
        "claim_type": "concept",
        "is_key_technical": True,
        "confidence": 0.85,
        "status": "active",
    }]

    # 推荐视频：仅 learning_package 场景传入（answer 场景不涉及视频）。
    # 同步为每个视频补一条 source_type="video" 的 source（对齐 Agent3 产出 src_video_{id}），
    # 使视频错误的 source_ref=src_video_RES-VID-064 能命中 sources，辩论协议字段可完整补全。
    recommended_videos = DEFAULT_RECOMMENDED_VIDEOS if target_type == "learning_package" else []
    for video in recommended_videos:
        sources.append({
            "source_id": f"src_video_{video['resource_id']}",
            "source_type": "video",
            "title": video["title"],
            "locator": f"learning_resources:{video['resource_id']}",
            "content_ref": video["url"],
        })

    request = ReviewInput(
        session_id=session_id,
        target_type=target_type,
        content=content,
        stage=stage,
        profile=profile,
        sources=sources,
        claims=claims,
        recommended_videos=recommended_videos,
        round_index=1,
    )
    pretty("【输入】ReviewInput（Agent4 审核入参）", {
        "session_id": session_id,
        "target_type": target_type,
        "content": content,
        "stage": stage.model_dump(mode="json"),
        "sources": sources,
        "claims": claims,
        "recommended_videos": recommended_videos,
        "round_index": 1,
    })

    # 可选：逐步展示内部工具链的输入/输出（仅 deterministic 模式暴露工具链）
    if args.verbose:
        if mode == "deterministic":
            await show_tool_chain(agent, request)
        else:
            print("\n[--verbose] hybrid 模式内部走 LLM 提示词，工具链不逐个暴露；"
                  "如需逐步查看工具输入/输出，请用 --mode deterministic --verbose。")

    # Step 1: 审核
    print("\n[Step 1] 运行 run() -> TechnicalReviewResult ...")
    result = await agent.run(request)
    pretty("【输出】TechnicalReviewResult 概要", {
        "passed": result.passed,
        "errors_count": len(result.errors),
        "must_fix_count": sum(1 for e in result.errors if e.must_fix),
        "critical_count": sum(1 for e in result.errors if e.severity == "critical"),
        "review_details": result.review_details,
    })
    print("\n--- errors（含辩论协议字段 canonical_error_id / must_fix / change_request / source_refs）---")
    print(json.dumps([e.model_dump(mode="json") for e in result.errors], ensure_ascii=False, indent=2, default=str))

    # Step 2: rebut 复核
    if not result.errors:
        print("\n无 errors，跳过 rebut 测试。")
        return

    do_rebut = args.rebut
    if do_rebut is None:
        if _can_prompt(non_interactive):
            ans = _safe_input("\n是否继续测试 rebut 复核裁决？(y/N): ", non_interactive, "n").lower()
            do_rebut = ans == "y"
        else:
            do_rebut = True  # 非交互默认走完整链路，便于观察 rebut 裁决
            print("\n[非交互] 未指定 --rebut/--no-rebut，默认执行 rebut 复核。")
    if not do_rebut:
        print("\n已跳过 rebut 复核。")
        return

    # 自动构造 Agent3 风格 author_replies（must_fix->partially_accepted，其余->accepted）
    issues = [e.model_dump(mode="json") for e in result.errors]
    author_replies = []
    for iss in issues:
        if iss.get("must_fix"):
            disposition = "partially_accepted"
            refs = iss.get("source_refs") or ["src_kb12_0"]
        else:
            disposition = "accepted"
            refs = iss.get("source_refs") or []
        author_replies.append({
            "canonical_error_id": iss.get("canonical_error_id"),
            "disposition": disposition,
            "reason": f"测试回复：{iss.get('description', '')}",
            "source_refs": refs,
            "patched_sections": [iss.get("location") or "tutorial:sec_0"],
        })
    pretty("【输入】rebut 的 author_replies（自动按 Agent3 风格构造）", author_replies)
    print("\n[Step 2] 运行 rebut() 复核裁决 ...")
    verdicts = await agent.rebut(session_id, issues, author_replies)
    pretty("【输出】rebut verdicts（persist=坚持 / withdraw=撤销 / downgrade=降级）", verdicts)

    # 逐条对照：issue -> Agent3 回复 -> Agent4 裁决
    print(f"\n{'-' * 60}\n★ issue ↔ Agent3 回复 ↔ Agent4 裁决 逐条对照\n{'-' * 60}")
    reply_by_cid = {r["canonical_error_id"]: r for r in author_replies}
    for idx, iss in enumerate(issues, 1):
        cid = iss.get("canonical_error_id")
        reply = reply_by_cid.get(cid, {})
        verdict = verdicts.get(cid, "—")
        print(f"\n[{idx}] {cid}  ({iss.get('error_type')}/{iss.get('severity')}, must_fix={iss.get('must_fix')})")
        print(f"    Agent4 提出 : {iss.get('description')}")
        print(f"    Agent3 回复 : {reply.get('disposition')}（source_refs={reply.get('source_refs')}）")
        print(f"    Agent4 裁决 : {verdict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent4 技术+质量审核 Agent 独立测试（命令行参数优先，无参数回退交互）",
    )
    parser.add_argument("--mode", choices=["hybrid", "deterministic"], default=None,
                        help="执行模式；省略则交互询问（非交互默认 hybrid）")
    parser.add_argument("--target", choices=["learning_package", "answer"], default=None,
                        help="审核目标类型；省略则交互询问（非交互默认 learning_package）")
    parser.add_argument("--content", type=str, default=None,
                        help="自定义待审核内容；省略则用默认示例（含事实错误）")
    parser.add_argument("--verbose", action="store_true",
                        help="额外逐步打印 Agent4 内部工具链的输入/输出"
                             "（规则/幻觉/难度/认知负荷/风格/协议字段补全），仅 deterministic 模式生效")
    rebut_group = parser.add_mutually_exclusive_group()
    rebut_group.add_argument("--rebut", dest="rebut", action="store_true", default=None,
                             help="继续执行 rebut 复核裁决")
    rebut_group.add_argument("--no-rebut", dest="rebut", action="store_false",
                             help="跳过 rebut 复核裁决")
    parser.add_argument("--non-interactive", action="store_true",
                        help="强制不读 stdin，全部用参数/默认值（管道/CI 友好）")
    return parser.parse_args()


if __name__ == "__main__":
    # 说明：hybrid 模式下 asyncio.run() 关闭事件循环时，底层 httpx/Ollama 连接或
    # 子进程（code_executor）在解释器退出阶段清理可能抛出非致命异常，导致 Windows 上
    # 进程返回 -1，使 CI/自动化误判为“测试失败”——即便 review/rebut 已完整跑通。
    # 因此显式区分：正常结束 -> exit 0；仅真正的运行期异常 -> exit 1。
    exit_code = 0
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\n已中断。")
        exit_code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
