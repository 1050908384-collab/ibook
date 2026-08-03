# -*- coding: utf-8 -*-
"""
Agent6 独立测试 —— 阶段决策 Agent

职责（对齐 backend/agents/deterministic.py）：
  - 阶段结束后，依据 accuracy / qa_history / interactions 更新画像（掌握度、节奏、信心、困惑标记）
  - 产出决策 decision（continue / repeat / completed）+ 策略 strategy + 阶段总结 stage_summary
  - 透传 iteration_history 激活 backtrack / finish 策略（iteration_strategy.skill.md）

运行方式（命令行参数优先，无参数时回退到交互提示）：
  # 交互模式
  python tests/check_agent6_stage_decision.py

  # 非交互一键跑通（自动化调试常用）
  python tests/check_agent6_stage_decision.py --accuracy 0.6
  python tests/check_agent6_stage_decision.py --accuracy 0.9 --verbose
  python tests/check_agent6_stage_decision.py --non-interactive

参数：
  --accuracy FLOAT     本阶段正确率（0~1）；省略则交互询问（非交互默认 0.85）
  --verbose            额外逐个打印 Agent6 内部工具链的输入/输出（BKT掌握度、节奏、信心、
                       困惑、策略、总结），方便查看链路中每一步的中间结果
  --non-interactive    强制不读 stdin，全部用参数/默认值（管道/CI 友好）

说明：
  - Agent6 仅有 deterministic 版本（无 hybrid），Harness 中即 DeterministicStageDecisionAgent，本测试与之一致。
  - accuracy >= 0.8 -> continue；< 0.8 -> repeat；全部掌握或 strategy=finish -> completed。
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

from agents.deterministic import DeterministicStageDecisionAgent
from models.contracts import StageDecisionInput
from models.profile import KnowledgeGap, UserProfile
from models.stage import StageInfo


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


def resolve_accuracy(cli_accuracy: float | None, non_interactive: bool) -> float:
    if cli_accuracy is not None:
        return max(0.0, min(1.0, cli_accuracy))
    acc_in = _safe_input("\n请输入本阶段正确率 accuracy（0~1，回车=0.85）: ", non_interactive, "")
    try:
        accuracy = float(acc_in) if acc_in else 0.85
    except ValueError:
        accuracy = 0.85
    return max(0.0, min(1.0, accuracy))


async def show_tool_chain(agent: DeterministicStageDecisionAgent, request: StageDecisionInput) -> None:
    """逐个调用 Agent6 内部工具，打印链路中每一步的输入/输出（--verbose）。

    与 DeterministicStageDecisionAgent.run 的调用顺序一致：
      profile_update_tool.update_gaps      -> 逐知识点 BKT 掌握度/状态
      profile_update_tool.infer_rhythm     -> 学习节奏 fast/medium/slow
      profile_update_tool.infer_confidence -> 信心指数（情绪加权）
      profile_update_tool.detect_confusion -> 困惑标记
      strategy_selector_tool.select_strategy_full -> 策略 + 决策 + 回溯/终止信号
      stage_summary_generator_tool.generate_summary -> Markdown 总结
    """
    print(f"\n{'*' * 60}\n★ Agent6 内部工具链逐步输入/输出（--verbose）\n{'*' * 60}")
    profile_dict = request.profile.model_dump(mode="json")
    gaps_in = [g.model_dump(mode="json") for g in request.profile.knowledge_gaps]

    # 与 run() 一致：先从 qa_history 预提取 BKT 所需的原始答题记录与情绪均分，
    # 否则 update_gaps 会缺少 question_records/emotion_score 而退化为简化聚合公式，
    # 导致 --verbose 展示的掌握度与真实 run() 不一致（失去调试意义）。
    pre_question_records = [
        qa for qa in request.qa_history
        if isinstance(qa, dict) and ("question_id" in qa or "skill" in qa or "is_correct" in qa)
    ]
    _emotions = [
        float(qa["emotion_score"]) for qa in request.qa_history
        if isinstance(qa, dict) and "emotion_score" in qa
        and isinstance(qa.get("emotion_score"), (int, float))
    ]
    emotion_score = sum(_emotions) / len(_emotions) if _emotions else None
    skill_tags = list(request.profile.skill_tags or [])

    # 1. BKT 掌握度更新（与 run() 完全同参：question_records/emotion_score/skill_tags）
    print("\n[Tool 1] profile_update_tool.update_gaps")
    print(f"  输入 gaps: {json.dumps(gaps_in, ensure_ascii=False)}")
    print(f"  输入 accuracy: {request.accuracy}")
    print(f"  输入 question_records: {json.dumps(pre_question_records, ensure_ascii=False)}")
    print(f"  输入 emotion_score: {emotion_score} | skill_tags: {skill_tags}")
    updated_gaps = await agent.profile_update_tool.update_gaps(
        gaps_in, request.accuracy,
        question_records=pre_question_records,
        emotion_score=emotion_score,
        skill_tags=skill_tags,
    )
    # 与 run() 一致：对当前阶段知识点应用 progress≥0.75 -> 已掌握 的收敛判定
    for gap in updated_gaps:
        if (gap.get("skill") == request.stage.gap_skill
                and gap.get("progress", 0.0) >= 0.75
                and gap.get("status") in ("学习中", "未掌握")):
            gap["status"] = "已掌握"
            gap["progress"] = 1.0
    print(f"  输出 gaps（含 current_gap 收敛判定）: {json.dumps(updated_gaps, ensure_ascii=False)}")

    # 2. 学习节奏
    print("\n[Tool 2] profile_update_tool.infer_rhythm")
    print(f"  输入 interactions: {json.dumps(request.interactions, ensure_ascii=False)}")
    rhythm = await agent.profile_update_tool.infer_rhythm(request.interactions)
    print(f"  输出 learning_rhythm: {rhythm}")

    # 3. 信心指数
    print("\n[Tool 3] profile_update_tool.infer_confidence")
    print(f"  输入 qa_history: {json.dumps(request.qa_history, ensure_ascii=False)}")
    confidence = await agent.profile_update_tool.infer_confidence(request.qa_history, request.accuracy)
    print(f"  输出 confidence_index: {confidence}")

    # 4. 困惑标记
    print("\n[Tool 4] profile_update_tool.detect_confusion")
    confusion = await agent.profile_update_tool.detect_confusion(request.qa_history)
    print(f"  输出 confusion_markers: {json.dumps(confusion, ensure_ascii=False)}")

    # 5. 策略选择（完整版，含决策与回溯/终止信号）
    # 与 run() 一致：用更新后的 gaps 回填 profile 再选策略，否则覆盖率/结业判定失真。
    profile_for_strategy = dict(profile_dict)
    profile_for_strategy["knowledge_gaps"] = updated_gaps
    print("\n[Tool 5] strategy_selector_tool.select_strategy_full")
    print(f"  输入 accuracy: {request.accuracy}")
    strategy_full = await agent.strategy_selector_tool.select_strategy_full(
        request.accuracy, profile_for_strategy, iteration_history=[]
    )
    strategy_label = strategy_full.get("strategy_label", "reinforce")
    print(f"  输出: {json.dumps(strategy_full, ensure_ascii=False)}")

    # 6. 阶段总结（与 run() 同参：真实 emotion_score + 收敛后的 gaps + 映射后的 strategy）
    print("\n[Tool 6] stage_summary_generator_tool.generate_summary")
    summary = await agent.stage_summary_generator_tool.generate_summary(
        stage_name=request.stage.block_name,
        accuracy=request.accuracy,
        knowledge_gaps=updated_gaps,
        confusion_tags=confusion,
        question_records=pre_question_records or request.qa_history,
        emotion_score=emotion_score,
        pace_adjustment=rhythm,
        time_spent_hours=1.5,
        expected_time_hours=2.0,
        next_strategy=strategy_label,
        total_knowledge_points=len(updated_gaps),
        total_answers=len(pre_question_records) or len(request.qa_history),
    )
    print(f"  输出 stage_summary（前 200 字）:\n  {summary[:200]}...")


async def main(args: argparse.Namespace):
    non_interactive = bool(args.non_interactive)
    print("=" * 60)
    print("Agent6 阶段决策 Agent 独立测试（画像更新 + 决策 + 总结）")
    print("=" * 60)
    print("说明：Agent6 仅 deterministic 版本（无 hybrid），与 Harness 实际使用一致。")

    agent = DeterministicStageDecisionAgent()

    accuracy = resolve_accuracy(args.accuracy, non_interactive)
    print(f"\n>>> accuracy = {accuracy}")

    # 构造 profile（含一个学习中、一个未掌握的 gap，便于观察决策差异）
    session_id = "test-agent6-session"
    profile = UserProfile(
        session_id=session_id,
        skill_tags=["Python", "PyTorch"],
        knowledge_gaps=[
            KnowledgeGap(skill="LoRA", status="学习中", progress=0.5, prerequisites=["PyTorch"]),
            KnowledgeGap(skill="QLoRA", status="未掌握", progress=0.0, prerequisites=["LoRA"]),
        ],
        goal="掌握LoRA微调",
    )
    stage = StageInfo(
        stage_id="stage-test-06", stage_num=1, block_id="KB-12",
        block_name="参数高效微调（LoRA与QLoRA）", gap_skill="LoRA",
        estimated_hours=2.0,
    )
    # qa_history 补充 skill / is_question 字段，便于观察 detect_confusion 触发逻辑
    qa_history = [
        {"question_id": "Q1", "skill": "LoRA", "is_correct": True, "emotion_score": 0.8},
        {"question_id": "Q2", "skill": "QLoRA", "is_correct": accuracy >= 0.5,
         "emotion_score": 0.6, "is_question": True},
    ]
    interactions = [
        {"time_spent": 5400, "expected_time": 7200},  # 1.5h 实际 / 2h 预期
    ]

    request = StageDecisionInput(
        session_id=session_id,
        profile=profile,
        stage=stage,
        accuracy=accuracy,
        qa_history=qa_history,
        interactions=interactions,
    )
    pretty("【输入】StageDecisionInput", {
        "accuracy": accuracy,
        "stage": stage.model_dump(mode="json"),
        "profile.knowledge_gaps": [g.model_dump(mode="json") for g in profile.knowledge_gaps],
        "qa_history": qa_history,
        "interactions": interactions,
    })

    # 可选：逐步展示内部工具链的输入/输出
    if args.verbose:
        await show_tool_chain(agent, request)

    # Step 1: 阶段决策（完整 run 链路）
    print("\n[Step 1] 运行 run() -> StageDecisionOutput ...")
    output = await agent.run(request)
    pretty("【输出】StageDecisionOutput（核心字段）", {
        "decision": output.decision,
        "strategy": output.strategy,
        "stage_summary": output.stage_summary,
    })
    pretty("【输出】更新后的 profile.knowledge_gaps（观察 status / progress 变化）",
           [g.model_dump(mode="json") for g in output.profile.knowledge_gaps])
    pretty("【输出】更新后的 profile（rhythm / confidence_index / confusion_markers）", {
        "learning_rhythm": output.profile.learning_rhythm,
        "confidence_index": output.profile.confidence_index,
        "confusion_markers": output.profile.confusion_markers,
        "version": output.profile.version,
    })

    print("\n决策参考：")
    print("  accuracy >= 0.8  -> continue（推进下一阶段）")
    print("  accuracy <  0.8  -> repeat（重试当前阶段）")
    print("  全部已掌握 / strategy=finish -> completed（结业）")
    print("  strategy=backtrack -> repeat（回退前置知识）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent6 阶段决策 Agent 独立测试（命令行参数优先，无参数回退交互）",
    )
    parser.add_argument("--accuracy", type=float, default=None,
                        help="本阶段正确率（0~1）；省略则交互询问（非交互默认 0.85）")
    parser.add_argument("--verbose", action="store_true",
                        help="额外逐步打印内部工具链的输入/输出（BKT/节奏/信心/困惑/策略/总结）")
    parser.add_argument("--non-interactive", action="store_true",
                        help="强制不读 stdin，全部用参数/默认值（管道/CI 友好）")
    return parser.parse_args()


if __name__ == "__main__":
    # 与 check_agent34_debate.py 一致：显式控制退出码，屏蔽事件循环关闭阶段的清理噪声
    # （Windows 上偶发的 -1），让退出码忠实反映 run() 的执行结果。
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
