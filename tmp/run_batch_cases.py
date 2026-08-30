# -*- coding: utf-8 -*-
"""顺序批量评测：按 case_id 顺序逐个跑，每 case 前清洗 db + 冷却。

用法:
    # 默认跑 11-20
    python tmp/run_batch_cases.py

    # 指定范围
    python tmp/run_batch_cases.py --start 11 --end 20
    python tmp/run_batch_cases.py --start 21 --end 30 --cool-sec 10

    # 只做最后 5 项审核扫表（不重新跑）
    python tmp/run_batch_cases.py --audit-only --start 11 --end 20
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

DELIVERY_DIR = ROOT_DIR / "测试数据与评测材料"
MAIN_EVAL_DIR = DELIVERY_DIR / "02_50组个性化主评测"
RAW_OUTPUTS_DIR = MAIN_EVAL_DIR / "raw_outputs"
BATCH_LOG = RAW_OUTPUTS_DIR / "batch.log"


# ============================================================
# 工具函数
# ============================================================
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_personal_data() -> None:
    """调 clean_personal_data.py 清空运行时数据。"""
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "eval_utils" / "clean_personal_data.py")]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT_DIR))
    if result.returncode != 0:
        log(f"[WARN] 清洗失败，但继续执行:\n{result.stdout}\n{result.stderr}")
    else:
        # 只打印有内容的最后几行
        cleaned_lines = [l for l in (result.stdout or "").splitlines() if l.strip()]
        if cleaned_lines:
            tail = cleaned_lines[-1]
            if "所有运行时数据已清空" in tail or "✅" in tail:
                log("[CLEAN] ✅ runtime tables wiped")
            else:
                log(f"[CLEAN] {tail}")


def load_cases_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    cases = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(row)
    return cases


# ============================================================
# 顺序执行入口（复用 run_real_evaluation.run_single_case）
# ============================================================
async def run_batch(start: int, end: int, cool_sec: int, clean_each: bool) -> List[str]:
    from scripts.run_real_evaluation import (
        _patch_llm_client,
        run_single_case,
    )
    from orchestrator.harness import Harness

    csv_path = MAIN_EVAL_DIR / "evaluation_cases.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"场景文件不存在: {csv_path}")

    all_cases = load_cases_from_csv(csv_path)
    target_ids = [f"case_{i:03d}" for i in range(start, end + 1)]
    cases = [c for c in all_cases if c["case_id"] in target_ids]

    # 保持顺序：按 case_id 数字升序
    cases.sort(key=lambda c: int(c["case_id"].split("_")[-1]))
    if not cases:
        log(f"[ERROR] 未匹配到任何 case: {target_ids}")
        return []

    log(f"[BATCH] === 开始 batch {cases[0]['case_id']} ~ {cases[-1]['case_id']} "
        f"({len(cases)} cases), cool_sec={cool_sec}, clean_each={clean_each} ===")

    # 一次性 patch + 建 Harness
    _patch_llm_client()
    harness = Harness(execution_mode=None)

    executed: List[str] = []
    total = len(cases)

    for idx, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        log("")
        log(f"[BATCH] ── [{idx}/{total}] {case_id} ──")

        try:
            # Step 1: 清洗
            if clean_each:
                log(f"  [{case_id}] 清洗运行时数据...")
                clean_personal_data()

            # Step 2: 跑单 case
            log(f"  [{case_id}] Agent1 画像 → 阶段规划 → Agent3 学习包 → 出题 → 答题 → 策略决策...")
            t0 = time.time()
            run_id = f"{case_id}-{int(time.time())}"
            summary = await run_single_case(harness, case, run_id)
            elapsed = time.time() - t0

            status = summary.get("status", "?")
            llm_cnt = summary.get("llm_call_count", 0)
            exec_path = summary.get("execution_path", "")
            stage1_acc = summary.get("stage_1_accuracy")
            stage1_strategy = summary.get("stage_1_strategy", "")
            block_id = summary.get("stage_1_block", "") or summary.get("stage_1_block_id", "")

            log(f"  [{case_id}] DONE status={status} elapsed={elapsed:.1f}s "
                f"llm={llm_cnt} path={exec_path} block={block_id} "
                f"acc={stage1_acc} strategy={stage1_strategy}")
            executed.append(case_id)

        except Exception as exc:
            log(f"  [{case_id}] UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")
            log(traceback.format_exc())

        # Step 3: 冷却（最后一个 case 也冷却不划算，跳过）
        if idx < total and cool_sec > 0:
            log(f"  [{case_id}] 冷却 {cool_sec}s ...")
            time.sleep(cool_sec)

    log(f"[BATCH] === 完成 {len(executed)}/{total} cases: {executed} ===")
    return executed


# ============================================================
# 5 项审核（85%）自动扫表
# ============================================================
VALID_STYLES = {"theory", "practice", "mixed"}


def _load_case_metadata() -> Dict[str, Dict[str, str]]:
    csv_path = MAIN_EVAL_DIR / "evaluation_cases.csv"
    meta = {}
    for c in load_cases_from_csv(csv_path):
        meta[c["case_id"]] = c
    return meta


def audit_batch(start: int, end: int) -> List[Dict[str, Any]]:
    meta = _load_case_metadata()
    rows: List[Dict[str, Any]] = []

    for i in range(start, end + 1):
        cid = f"case_{i:03d}"
        case_dir = RAW_OUTPUTS_DIR / cid
        stage1_dir = case_dir / "stage_1"

        record: Dict[str, Any] = {"case_id": cid}

        # --- 加载文件 ---
        try:
            with open(case_dir / "run_summary.json", encoding="utf-8") as f:
                summary = json.load(f)
        except FileNotFoundError:
            record["status"] = "NO_RUN"
            rows.append(record)
            continue

        # learner_profile 实际保存在 stage_1 子目录（run_real_evaluation.py L723）
        try:
            with open(stage1_dir / "learner_profile.json", encoding="utf-8") as f:
                profile = json.load(f)
        except FileNotFoundError:
            profile = {}

        try:
            with open(stage1_dir / "stage_planning_output.json", encoding="utf-8") as f:
                stage_out = json.load(f)
        except FileNotFoundError:
            stage_out = {}

        # --- 85% 5 项 ---
        # 1) target_path: stage_1.block 非空 & 与 goal 的硬闭包对齐（不深入图谱，只看非空合理性）
        stage_1_block = (
            summary.get("stage_1_block")
            or summary.get("stage_1_block_id")
            or stage_out.get("block_id")
            or ""
        )
        profile_mastered = profile.get("mastered_block_ids") or summary.get("mastered_block_ids") or []
        ok_target = bool(stage_1_block) and (stage_1_block not in profile_mastered)

        # 2) difficulty: 有值 & 合法
        difficulty = (
            stage_out.get("difficulty")
            or stage_out.get("recommended_difficulty")
            or ""
        )
        ok_difficulty = difficulty in {"初级", "中级", "高级"}

        # 3) time_budget: estimated_hours <= daily_hours（画像 daily_hours）
        case_meta = meta.get(cid, {})
        daily_hours_raw = case_meta.get("daily_hours", "")
        try:
            daily_hours = float(daily_hours_raw) if daily_hours_raw else 0
        except ValueError:
            daily_hours = 0
        estimated_hours = stage_out.get("estimated_hours") or summary.get("stage_1_estimated_hours")
        try:
            est = float(estimated_hours) if estimated_hours else 0
        except (TypeError, ValueError):
            est = 0
        ok_time = (daily_hours == 0) or (est > 0 and est <= daily_hours)

        # 4) hardware: hardware_notes 非空
        hardware_notes = (
            stage_out.get("hardware_notes")
            or stage_out.get("hardware_requirements")
            or stage_out.get("hardware_summary")
            or ""
        )
        hw_in = case_meta.get("hardware", "")
        ok_hardware = bool(hardware_notes.strip())

        # 5) learning_style: learning_style 合法 + notes 前缀匹配
        # 优先级：profile(Agent1识别出的标准枚举) → stage_out → CSV(自然语言描述，需映射)
        style_alias = {
            "理论型": "theory", "实操型": "practice", "动手型": "practice",
            "混合型": "mixed", "理论": "theory", "实操": "practice", "混合": "mixed",
            # CSV中更丰富的自然语言描述
            "工程实操型": "practice", "边学边做": "mixed",
            "理论结合简单实操": "mixed", "理论加实操": "mixed",
            "以实操为主": "practice", "以理论为主": "theory",
        }
        raw_style = (
            profile.get("learning_style")
            or stage_out.get("learning_style")
            or summary.get("profile", {}).get("learning_style")
            or case_meta.get("learning_style", "")
        )
        raw_style = (raw_style or "").strip()
        # 先尝试作为枚举；否则查别名映射
        if raw_style in VALID_STYLES:
            learning_style = raw_style
        elif raw_style in style_alias:
            learning_style = style_alias[raw_style]
        else:
            # 模糊匹配：包含关键字
            low = raw_style.lower()
            if any(k in low for k in ("practice", "实操", "动手", "工程")):
                learning_style = "practice"
            elif any(k in low for k in ("mixed", "混合", "边学边做", "结合")):
                learning_style = "mixed"
            elif any(k in low for k in ("theory", "理论")):
                learning_style = "theory"
            else:
                learning_style = raw_style.lower() if raw_style else ""

        style_notes = (
            stage_out.get("learning_style_notes")
            or stage_out.get("style_suggestion")
            or ""
        )
        style_valid = learning_style in VALID_STYLES
        if style_valid and style_notes:
            prefix_map = {"theory": "理论型", "practice": "动手型", "mixed": "混合型"}
            expected_prefix = prefix_map.get(learning_style, "")
            # notes前缀匹配 OR notes中明确提及该风格关键词
            style_prefix_ok = (
                expected_prefix
                and (
                    style_notes.startswith(expected_prefix)
                    or expected_prefix in style_notes
                )
            )
        else:
            style_prefix_ok = False
        ok_style = style_valid and style_prefix_ok

        total_pass = sum([ok_target, ok_difficulty, ok_time, ok_hardware, ok_style])

        record.update({
            "status": summary.get("status", ""),
            "block_id": stage_1_block,
            "mastered_blocks": profile_mastered,
            "learning_style": learning_style,
            "difficulty": difficulty,
            "estimated_hours": est,
            "daily_hours": daily_hours,
            "hardware_input": hw_in,
            "hardware_notes_nonempty": ok_hardware,
            # 5 项单独结果
            "1_target_path":  "✅" if ok_target      else "❌",
            "2_difficulty":   "✅" if ok_difficulty  else "❌",
            "3_time_budget":  "✅" if ok_time        else "❌",
            "4_hardware":     "✅" if ok_hardware    else "❌",
            "5_learning_style":"✅" if ok_style       else "❌",
            "pass_5": f"{total_pass}/5",
        })
        rows.append(record)

    return rows


def print_audit_table(rows: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 110)
    print(f"{'case':<8} {'block':<8} {'mastered':<16} {'style':<9} "
          f"[1] [2] [3] [4] [5] → PASS")
    print("=" * 110)
    total_ok = {"1_target_path": 0, "2_difficulty": 0, "3_time_budget": 0,
                "4_hardware": 0, "5_learning_style": 0, "5_full": 0}
    for r in rows:
        if r.get("status") == "NO_RUN":
            print(f"{r['case_id']:<8} (未执行)")
            continue
        print(f"{r['case_id']:<8} {r.get('block_id',''):<8} "
              f"{str(r.get('mastered_blocks','')):<16} "
              f"{r.get('learning_style',''):<9} "
              f"{r['1_target_path']:>3} {r['2_difficulty']:>3} "
              f"{r['3_time_budget']:>3} {r['4_hardware']:>3} "
              f"{r['5_learning_style']:>3}   {r['pass_5']}")
        for k in total_ok:
            if k == "5_full":
                total_ok[k] += (1 if r["pass_5"] == "5/5" else 0)
            elif r.get(k) == "✅":
                total_ok[k] += 1

    n_run = sum(1 for r in rows if r.get("status") != "NO_RUN")
    print("=" * 110)
    if n_run:
        print(f"执行总数: {n_run} 个 case")
        for k, v in total_ok.items():
            name = k.ljust(20)
            pct = (v / n_run) * 100
            print(f"  {name}: {v}/{n_run}  ({pct:.1f}%)")
    print()


def save_audit_csv(rows: List[Dict[str, Any]], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"[AUDIT] 5 项审核表已保存 → {csv_out}")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="顺序批量评测 + 5 项审核")
    parser.add_argument("--start", type=int, default=11)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument("--cool-sec", type=int, default=5,
                        help="case 之间冷却秒数（最后一个 case 跳过）")
    parser.add_argument("--no-clean", action="store_true",
                        help="不要每个 case 前清洗（默认每次都洗）")
    parser.add_argument("--audit-only", action="store_true",
                        help="只做 5 项审核扫表，不重新跑 case")
    parser.add_argument("--audit-csv", type=str, default=None,
                        help="保存审核 CSV 到路径（默认 raw_outputs/audit_XX-YY.csv）")
    args = parser.parse_args()

    RAW_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.audit_only:
        asyncio.run(run_batch(
            start=args.start,
            end=args.end,
            cool_sec=args.cool_sec,
            clean_each=not args.no_clean,
        ))

    # 无论跑没跑，都产出审核表
    rows = audit_batch(args.start, args.end)
    print_audit_table(rows)

    audit_csv = Path(args.audit_csv) if args.audit_csv else (
        RAW_OUTPUTS_DIR / f"audit_{args.start:03d}-{args.end:03d}.csv"
    )
    save_audit_csv(rows, audit_csv)


if __name__ == "__main__":
    main()
