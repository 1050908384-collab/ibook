# -*- coding: utf-8 -*-
"""从真实主评测结果抽取 4 组完整展示案例。

案例映射（同《测试数据建议方法》第八节）：
    A 初学者低分简化 + 第二阶段   <- case_040（真实主评测结果，多阶段）
    B 低显存 QLoRA 实操            <- case_018（RTX 4060 8GB 下的资源适配）
    C 工程部署进阶                <- case_026（24GB，权重合并+vLLM）
    D 自动回退案例                <- case_044 的受控故障注入重跑
                                      （与 A/B/C 保持为第四个独立画像）

用法:
    # 必须先运行过 02_50组 主评测
    python scripts/extract_complete_cases.py

    # 手动指定 A/B/C 来源
    python scripts/extract_complete_cases.py --A case_040 --B case_018 --C case_026

    # D（回退案例）可控制注入哪类故障：
    python scripts/extract_complete_cases.py --D-fault http_408_timeout
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
RAW_OUTPUTS_DIR = DELIVERY_DIR / "02_50组个性化主评测" / "raw_outputs"
CASES_DIR = DELIVERY_DIR / "01_四组完整案例"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def copy_stage_files(src_stage_dir: Path, dst_case_dir: Path,
                     file_names: list[str]) -> None:
    for fn in file_names:
        src = src_stage_dir / fn
        if src.exists():
            dst_case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_case_dir / fn)


def copy_generation_events_and_llm(src_case_raw_dir: Path, dst_case_dir: Path):
    for fn in ["generation_events.jsonl", "llm_calls.jsonl",
               "input.json", "run_manifest.json", "error.log"]:
        src = src_case_raw_dir / fn
        if src.exists():
            shutil.copy2(src, dst_case_dir / fn)


def extract_A_B_C(letter: str, case_id: str, run_id: str) -> Dict[str, Any]:
    """从真实主评测 raw_outputs 中抽取 A/B/C 案例。"""
    letter_dir_names = {
        "A": "case_01_beginner",
        "B": "case_02_low_vram_practice",
        "C": "case_03_engineer_deployment",
    }
    dst = CASES_DIR / letter_dir_names[letter]
    src = RAW_OUTPUTS_DIR / case_id
    if not src.exists():
        return {
            "letter": letter,
            "source_case_id": case_id,
            "status": "skipped",
            "reason": f"raw_outputs/{case_id} 不存在，请先运行 run_real_evaluation.py",
        }

    # 检查是否有 stage_2（A 必须有）
    stage_1_src = src / "stage_1"
    stage_2_src = src / "stage_2"
    has_stage_2 = stage_2_src.exists()
    if letter == "A" and not has_stage_2:
        return {
            "letter": letter,
            "source_case_id": case_id,
            "status": "skipped",
            "reason": f"A 要求两阶段闭环，但 {case_id} 无 stage_2 输出（需 run_real_evaluation.py 多阶段执行）",
        }

    # 拷贝 stage_1 标准 8 文件
    dst.mkdir(parents=True, exist_ok=True)
    stage_1_names = [
        "01_learner_profile.json",
        "02_diagnostic_output.json",
        "03_stage_planning_output.json",
        "06_learning_package.md",
        "07_review_trace.json",
        "08_assessment_questions.json",
        "09_learner_answers.json",
        "10_stage_decision.json",
    ]
    # 从源文件映射：
    source_map_1 = {
        "learner_profile.json": "01_learner_profile.json",
        "diagnostic_output.json": "02_diagnostic_output.json",
        "stage_planning_output.json": "03_stage_planning_output.json",
        "learning_package.md": "06_learning_package.md",
        "review_trace.json": "07_review_trace.json",
        "assessment_questions.json": "08_assessment_questions.json",
        "learner_answers.json": "09_learner_answers.json",
        "stage_decision.json": "10_stage_decision.json",
    }
    for src_fn, dst_fn in source_map_1.items():
        s = stage_1_src / src_fn
        if s.exists():
            shutil.copy2(s, dst / dst_fn)

    # stage_2（仅 A）
    if has_stage_2:
        stage2_dst = dst / "stage_2_extra"
        stage2_dst.mkdir(exist_ok=True)
        for src_fn in ["learner_profile_snapshot.json", "stage_planning_output.json",
                       "learning_package.md", "review_trace.json",
                       "assessment_questions.json", "learner_answers.json",
                       "stage_decision.json"]:
            s = stage_2_src / src_fn
            if s.exists():
                shutil.copy2(s, stage2_dst / src_fn)

    copy_generation_events_and_llm(src, dst)

    # README 和 screenshots 目录
    (dst / "11_screenshots").mkdir(exist_ok=True)
    readme = f"""# 完整展示案例 {letter}

来源：真实主评测场景 {case_id}
run_id：{run_id}
抽取时间：{_now_iso()}

## 说明
- 本案例所有文件从 `02_50组个性化主评测/raw_outputs/{case_id}/` 中抽取。
- `01_learner_profile.json` - 真实 Agent1 学情画像
- `02_diagnostic_output.json` - 诊断结果
- `03_stage_planning_output.json` - 阶段规划
- `04_generation_events.jsonl` - 生成事件时间线
- `05_llm_calls.jsonl` - 全量真实 LLM 调用（含模型名、耗时、回退、重试）
- `06_learning_package.md` - 真实 Agent3 生成 + Agent4 审核后的学习包
- `07_review_trace.json` - 审核修订闭环（含 iterations、must_fix issue）
- `08_assessment_questions.json` - 阶段测评题
- `09_learner_answers.json` - 受控答题结果
- `10_stage_decision.json` - 真实 Agent6 决策（accuracy、next_strategy）
"""
    with open(dst / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    # screenshots 占位说明
    with open(dst / "11_screenshots" / "README.md", "w", encoding="utf-8") as f:
        f.write("# 截图目录\n\n本目录用于存放真实运行过程中从前端界面截取的流程截图。\n"
                "建议顺序：\n"
                "1. Onboarding 画像录入页\n"
                "2. 学习计划确认页（Agent1 输出）\n"
                "3. 学习包浏览页（Agent3 内容 + 侧边栏审核修订状态）\n"
                "4. 测评题答题页\n"
                "5. 阶段结果与下一步建议页（Agent6 输出）\n")

    return {
        "letter": letter,
        "source_case_id": case_id,
        "status": "extracted",
        "target_dir": str(dst),
        "has_stage_2": has_stage_2,
    }


# =====================================================================
# D：受控故障注入重跑
# =====================================================================
async def extract_D_fallback(source_case_id: str, fault_type: str) -> Dict[str, Any]:
    """对 A/B/C 任一来源场景重跑，注入 DeepSeek 异常并让 Ollama 接管。"""
    dst = CASES_DIR / "case_04_fallback"
    dst.mkdir(parents=True, exist_ok=True)

    # 直接复用 run_fallback_tests.py 的逻辑（选指定底座 + 指定故障）
    from scripts.run_fallback_tests import run_single_fault

    print(f"  · D: 对 {source_case_id} 注入 fault={fault_type} 重跑 ...")
    fault_res = await run_single_fault(fault_type, source_case_id)

    # 把故障证据写进 D 的目录
    manifest = {
        "case_id": "case_04_fallback",
        "source_main_case_id": source_case_id,
        "run_type": "controlled_fallback_replay",
        "fault_type": fault_type,
        "fault_label": fault_res.get("fault_label", ""),
        "inject_method": fault_res.get("inject_method", ""),
        "actual_taken_model": fault_res.get("actual_taken_model", []),
        "fallback_triggered": fault_res.get("fallback_triggered", False),
        "flow_final_status": fault_res.get("flow_final_status", ""),
        "state_continuity": fault_res.get("state_continuity", False),
        "run_started_at": fault_res.get("run_started_at"),
        "run_finished_at": fault_res.get("run_finished_at"),
        "evidence": fault_res.get("evidence", {}),
        "error_message": fault_res.get("error_message"),
    }
    with open(dst / "controlled_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    # fallback_trace.jsonl
    with open(dst / "fallback_trace.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(fault_res, ensure_ascii=False, default=str) + "\n")
    # D 的调用日志必须来自本次受控回退重跑，不能沿用底座案例或留空。
    with open(dst / "05_llm_calls.jsonl", "w", encoding="utf-8") as f:
        for call in (fault_res.get("evidence", {}).get("llm_calls") or []):
            f.write(json.dumps(call, ensure_ascii=False, default=str) + "\n")
    # README：必须明确写明受控故障注入，不能说是自然发生
    readme = f"""# 完整展示案例 D：自动回退

source_main_case_id: {source_case_id}
run_type: controlled_fallback_replay

## 说明
本案例为**受控故障注入测试**，不是"自然发生的云端故障"。

## 注入方式
- fault_type: `{fault_type}`
- inject_method: `{manifest['inject_method']}`
- 故障描述: {manifest['fault_label']}

## 预期
- 云端 DeepSeek 调用失败后，系统自动回退到本地 Ollama
- 回退后流程仍完成（或 deterministic 兜底仍完成）
- 状态保持连续性（画像、阶段规划不落空）

## 实际结果
- 实际接管模型: {manifest['actual_taken_model']}
- 回退触发: {manifest['fallback_triggered']}
- 流程最终状态: {manifest['flow_final_status']}
- 状态连续性: {manifest['state_continuity']}

详见 `controlled_run_manifest.json` 和 `fallback_trace.jsonl`。
"""
    with open(dst / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    (dst / "11_screenshots").mkdir(exist_ok=True)
    with open(dst / "11_screenshots" / "README.md", "w", encoding="utf-8") as f:
        f.write("# 回退场景截图目录\n\n建议包含：\n"
                "1. 前端控制台显示 'DeepSeek 失败，切换到 Ollama' 的日志截图\n"
                "2. 学习包仍成功生成的界面截图（证明连续性）\n")
    # 再拷贝来源底座 case 的基本 9 文件（来自主评测成功版本）做对比
    src = RAW_OUTPUTS_DIR / source_case_id / "stage_1"
    if src.exists():
        map_ = {
            "learner_profile.json": "01_learner_profile.json",
            "diagnostic_output.json": "02_diagnostic_output.json",
            "stage_planning_output.json": "03_stage_planning_output.json",
            "learning_package.md": "06_learning_package.md",
            "review_trace.json": "07_review_trace.json",
            "assessment_questions.json": "08_assessment_questions.json",
            "learner_answers.json": "09_learner_answers.json",
            "stage_decision.json": "10_stage_decision.json",
        }
        for sfn, dfn in map_.items():
            s = src / sfn
            if s.exists():
                shutil.copy2(s, dst / dfn)
    # LLM 调用记录
    from scripts.run_real_evaluation import _llm_call_records as lcr
    recent = lcr[-50:]
    with open(dst / "05_llm_calls.jsonl", "w", encoding="utf-8") as f:
        for c in recent:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # 生成事件（空占位）
    with open(dst / "04_generation_events.jsonl", "w", encoding="utf-8") as f:
        f.write('{"id":1,"timestamp":' + str(int(time.time()*1000)) +
                ',"event":"controlled_fallback_replay","detail":"fault_type=' + fault_type + '"}\n')

    return {
        "letter": "D",
        "source_case_id": source_case_id,
        "fault_type": fault_type,
        "fallback_triggered": manifest["fallback_triggered"],
        "status": "extracted",
        "target_dir": str(dst),
    }


# =====================================================================
# 主
# =====================================================================
async def main():
    parser = argparse.ArgumentParser(description="从真实结果抽取 4 组完整展示案例")
    parser.add_argument("--A", default="case_040", help="A 初学者低分简化来源 case（默认 case_040）")
    parser.add_argument("--B", default="case_018", help="B 低显存 QLoRA 实操来源 case（默认 case_018）")
    parser.add_argument("--C", default="case_026", help="C 工程部署进阶来源 case（默认 case_026）")
    parser.add_argument("--D-source", default="case_044", help="D 回退重跑来源 case（默认 case_044，第四个独立画像）")
    parser.add_argument("--D-fault", default="connect_timeout", help="D 注入故障类型（默认 connect_timeout，见 run_fallback_tests 的 12 类）")
    args = parser.parse_args()

    CASES_DIR.mkdir(parents=True, exist_ok=True)

    # 读取全局 run_id
    run_id = ""
    rm = RAW_OUTPUTS_DIR / "run_manifest.json"
    if rm.exists():
        with open(rm, "r", encoding="utf-8") as f:
            run_id = json.load(f).get("run_id", "")

    results = []
    for letter, cid in [("A", args.A), ("B", args.B), ("C", args.C)]:
        print(f"抽取 {letter} <- {cid}")
        results.append(extract_A_B_C(letter, cid, run_id))
    # D 需要真运行（调用 harness）
    print("抽取 D （受控回退重跑）")
    results.append(await extract_D_fallback(args.D_source, args.D_fault))

    # 汇总打印
    print()
    print("=" * 60)
    for r in results:
        status = r["status"]
        letter = r["letter"]
        sid = r.get("source_case_id", "?")
        if status == "extracted":
            print(f"  [{letter}] OK  {sid} -> {r.get('target_dir')}")
        else:
            reason = r.get("reason") or r.get("error_message") or "未知原因"
            print(f"  [{letter}] SKIP  {sid}  {reason}")
    print()
    any_skipped = any(r["status"] == "skipped" for r in results)
    if any_skipped:
        print("提示：A/B/C 被跳过说明需要先运行主评测：")
        print("  python scripts/run_real_evaluation.py --all")
    return 0 if not any_skipped else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
