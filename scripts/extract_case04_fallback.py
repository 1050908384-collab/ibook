# -*- coding: utf-8 -*-
"""Extract case_04_fallback artifacts after run_fallback_tests.py completes the CONNECT_TIMEOUT-case_044 run.

Reads:
- sessions, user_profiles, stage_progress, user_qa, session_states tables for given session_id
- fallback_traces.jsonl for the controlled run record

Writes (overwrites existing) into case_04_fallback/:
    01_learner_profile.json / 02_diagnostic_output.json / 03_stage_planning_output.json
    04_generation_events.jsonl / 05_llm_calls.jsonl
    06_learning_package.md / 07_review_trace.json
    08_assessment_questions.json / 09_learner_answers.json / 10_stage_decision.json
    controlled_run_manifest.json (merge from fallback record + original shape)
    fallback_trace.jsonl (matching original structure)
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))
import os
os.chdir(str(BACKEND))
from db.sqlite_client import SQLiteClient
from sqlalchemy import text

TARGET = ROOT / "测试数据与评测材料" / "01_四组完整案例" / "case_04_fallback"
FALLBACK_DIR = ROOT / "测试数据与评测材料" / "03_回退可靠性测试"
RAW_044 = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs" / "case_044"


def _package_to_markdown(pkg: dict) -> str:
    """Mimic run_real_evaluation._package_to_markdown for 06 output."""
    md = []
    title = pkg.get("title") or pkg.get("block_name") or "学习包"
    md.append(f"# {title}\n")
    overview = pkg.get("overview") or pkg.get("description") or ""
    if overview.strip():
        md.append(overview.rstrip() + "\n")
    tutorial = pkg.get("tutorial") or ""
    if tutorial.strip():
        md.append(tutorial.rstrip() + "\n")
    subtopics = pkg.get("subtopics") or []
    if subtopics:
        md.append("## 子知识点清单\n")
        for s in subtopics:
            name = s.get("name") if isinstance(s, dict) else str(s)
            sid = s.get("id", "") if isinstance(s, dict) else ""
            md.append(f"- {sid} {name}".strip() if sid else f"- {name}")
        md.append("")
    resources = pkg.get("resources") or []
    if resources:
        md.append("## 拓展资源\n")
        for r in resources:
            if isinstance(r, dict):
                typ = r.get("type", "资源")
                url = r.get("url") or r.get("path") or ""
                name = r.get("title") or r.get("name") or typ
                md.append(f"- [{name}]({url}) （{typ}）" if url else f"- [{name}]（{typ}）")
            else:
                md.append(f"- {r}")
        md.append("")
    questions = pkg.get("questions") or []
    if questions:
        md.append("## 阶段测试题\n")
        for i, q in enumerate(questions, 1):
            qid = q.get("id") or q.get("question_id") or f"Q{i}"
            stem = q.get("stem") or q.get("question") or ""
            qtype = q.get("type") or q.get("question_type") or "choice"
            md.append(f"### {i}. [{qtype}] {qid}\n")
            md.append(stem.rstrip() + "\n")
            opts = q.get("options") or []
            if opts:
                labels = "ABCDEFGHIJ"
                for idx, opt in enumerate(opts):
                    label = labels[idx] if idx < len(labels) else str(idx + 1)
                    md.append(f"- {label}. {opt}")
                md.append("")
            ans = q.get("correct_answer") or q.get("answer") or ""
            if ans:
                md.append(f"**正确答案**: {ans}\n")
            expl = q.get("explanation") or q.get("analysis") or ""
            if expl:
                md.append(f"**解析**: {expl}\n")
        md.append("")
    return "\n".join(md)


def main(argv):
    if len(argv) < 2:
        print("Usage: extract_case04_fallback.py <session_id>")
        return 1
    session_id = argv[1]
    TARGET.mkdir(parents=True, exist_ok=True)

    # Load fallback record
    rec = None
    with open(FALLBACK_DIR / "fallback_traces.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ev = r.get("evidence") or {}
            if ev.get("session_id") == session_id or (
                r.get("fault_id") == "CONNECT_TIMEOUT-case_044"
            ):
                rec = r
                break
    if rec is None:
        # Try by fault_id directly
        with open(FALLBACK_DIR / "fallback_traces.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("fault_id") == "CONNECT_TIMEOUT-case_044":
                    rec = r
                    break
    if rec is None:
        print("FAIL: cannot find CONNECT_TIMEOUT-case_044 record in fallback_traces.jsonl")
        return 2

    ev = rec.get("evidence") or {}
    session_id = session_id or ev.get("session_id")

    cli = SQLiteClient()
    with cli.engine.connect() as conn:

        def _scalar(sql: str, **params):
            c = conn.execute(text(sql), params).fetchone()
            return c[0] if c else None

        def _row(sql: str, **params):
            r = conn.execute(text(sql), params).mappings().first()
            return dict(r) if r else {}

        def _rows(sql: str, **params):
            return [dict(r) for r in conn.execute(text(sql), params).mappings().fetchall()]

        # 01_learner_profile
        prof = _row(
            "SELECT * FROM user_profiles WHERE session_id=:sid ORDER BY version DESC LIMIT 1",
            sid=session_id,
        )
        if not prof:
            try:
                prof = _row(
                    "SELECT * FROM learner_profiles ORDER BY created_at DESC LIMIT 1"
                )
            except Exception:
                prof = {}
        with open(TARGET / "01_learner_profile.json", "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2, default=str)

        # session_state
        sstate = _row(
            "SELECT * FROM session_states WHERE session_id=:sid ORDER BY updated_at DESC LIMIT 1",
            sid=session_id,
        )
        state_json = {}
        raw_sj = sstate.get("state_json") if isinstance(sstate.get("state_json"), str) else ""
        if raw_sj:
            try:
                state_json = json.loads(raw_sj)
            except Exception:
                state_json = {"state_json_raw": raw_sj}

        # 02_diagnostic_output: from state_json.diagnostic or Agent1 process_initial_input keys
        diag = state_json.get("diagnostic") or state_json.get("profile") or state_json.get("last_diagnostic") or {}
        with open(TARGET / "02_diagnostic_output.json", "w", encoding="utf-8") as f:
            json.dump(diag or {"note": "diagnostic_inferred_from_profile"}, f, ensure_ascii=False, indent=2, default=str)

        # 03_stage_planning_output: from state_json.stage_info / current_stage / stages list
        plan = (
            state_json.get("stage_planning")
            or state_json.get("current_stage")
            or state_json.get("stage_info")
            or state_json.get("stages")
            or {}
        )
        with open(TARGET / "03_stage_planning_output.json", "w", encoding="utf-8") as f:
            json.dump(plan or {"note": "stage_plan_in_state"}, f, ensure_ascii=False, indent=2, default=str)

        # learning_package: session_states.state_json.learning_package or nested package key paths
        def _find_pkg(obj, max_depth=6):
            """Breadth-first search for a dict with tutorial/questions/content_blocks keys."""
            if not isinstance(obj, (dict, list)):
                return None
            stack = [(obj, 0)]
            while stack:
                cur, d = stack.pop(0)
                if d > max_depth:
                    continue
                if isinstance(cur, dict):
                    if isinstance(cur.get("tutorial"), str) and cur.get("tutorial"):
                        return cur
                    if isinstance(cur.get("content_blocks"), list) and cur["content_blocks"]:
                        return cur
                    for v in cur.values():
                        stack.append((v, d + 1))
                elif isinstance(cur, list):
                    for v in cur:
                        stack.append((v, d + 1))
            return None

        pkg = _find_pkg(state_json) or {}

        # 06_learning_package.md
        pkg_md = _package_to_markdown(pkg if isinstance(pkg, dict) else {})
        with open(TARGET / "06_learning_package.md", "w", encoding="utf-8") as f:
            f.write(pkg_md)

        # 07_review_trace.json: use fallback record evidence + iterations + review_passed
        review_trace = {
            "review_passed": ev.get("review_passed"),
            "iterations": ev.get("iterations", 0),
            "needs_human_review": bool(ev.get("needs_human_review")),
            "evidence": {
                k: v for k, v in ev.items() if k not in ("llm_calls",)
            },
            "fallback_route": rec.get("actual_taken_model"),
            "fallback_triggered": rec.get("fallback_triggered"),
        }
        with open(TARGET / "07_review_trace.json", "w", encoding="utf-8") as f:
            json.dump(review_trace, f, ensure_ascii=False, indent=2, default=str)

        # 08_assessment_questions.json: from learning_package.questions if present, else empty
        questions = pkg.get("questions") if isinstance(pkg, dict) else []
        with open(TARGET / "08_assessment_questions.json", "w", encoding="utf-8") as f:
            json.dump(questions or [], f, ensure_ascii=False, indent=2)

        # 09_learner_answers.json: from user_qa table
        qas = _rows(
            "SELECT * FROM user_qa WHERE session_id=:sid ORDER BY qa_id ASC, created_at ASC",
            sid=session_id,
        )
        with open(TARGET / "09_learner_answers.json", "w", encoding="utf-8") as f:
            json.dump(qas, f, ensure_ascii=False, indent=2, default=str)

        # 10_stage_decision.json: from stage_progress (last row) + Agent6 decision
        sps = _rows(
            "SELECT * FROM stage_progress WHERE session_id=:sid ORDER BY stage_num ASC, started_at ASC",
            sid=session_id,
        )
        decision_last = {}
        if sps:
            last = sps[-1]
            decision_last = {
                "stage_progress": last,
                "accuracy": last.get("accuracy"),
                "next_strategy": last.get("next_strategy"),
                "can_plan_next_stage": last.get("can_plan_next_stage"),
                "completed": bool(last.get("completed")),
            }
        with open(TARGET / "10_stage_decision.json", "w", encoding="utf-8") as f:
            json.dump(decision_last or {"note": "no_stage_progress"}, f, ensure_ascii=False, indent=2, default=str)

    cli.close()

    # 04_generation_events.jsonl: build minimal from evidence (fallback record itself + route samples) OR copy raw case_044 if present
    events_path = TARGET / "04_generation_events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "event_type": "fallback_run_start",
            "fault_id": rec.get("fault_id"),
            "at": rec.get("run_started_at"),
            "base_case_id": rec.get("base_case_id"),
            "inject_method": rec.get("inject_method"),
        }, ensure_ascii=False) + "\n")
        for s in (ev.get("llm_route_samples") or [])[:50]:
            f.write(json.dumps({
                "event_type": "llm_route_sample",
                **s,
                "at": rec.get("run_started_at"),
            }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "event_type": "fallback_run_done",
            "fault_id": rec.get("fault_id"),
            "at": rec.get("run_finished_at"),
            "flow_final_status": rec.get("flow_final_status"),
            "passed": rec.get("passed"),
        }, ensure_ascii=False) + "\n")

    # 05_llm_calls.jsonl: evidence.llm_calls list
    calls = ev.get("llm_calls") or []
    with open(TARGET / "05_llm_calls.jsonl", "w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # controlled_run_manifest.json (shaped after original pre-existing file):
    controlled = {
        "case_id": "case_04_fallback",
        "source_main_case_id": rec.get("base_case_id"),
        "run_type": "controlled_fallback_replay",
        "fault_type": rec.get("fault_type"),
        "fault_label": rec.get("fault_label"),
        "inject_method": rec.get("inject_method"),
        "actual_taken_model": rec.get("actual_taken_model"),
        "fallback_triggered": rec.get("fallback_triggered"),
        "flow_final_status": rec.get("flow_final_status"),
        "state_continuity": rec.get("state_continuity"),
        "run_started_at": rec.get("run_started_at"),
        "run_finished_at": rec.get("run_finished_at"),
        "passed": rec.get("passed"),
        "retry_count": rec.get("retry_count"),
        "expect_flow_failure": rec.get("expect_flow_failure"),
        "error_message": rec.get("error_message"),
        "evidence": {
            "session_id": ev.get("session_id"),
            "user_input_hash": ev.get("user_input_hash"),
            "profile_confidence": ev.get("profile_confidence"),
            "planned_block": ev.get("planned_block"),
            "review_passed": ev.get("review_passed"),
            "iterations": ev.get("iterations"),
            "llm_call_count": ev.get("llm_call_count"),
            "failed_calls": ev.get("failed_calls"),
            "successful_models": ev.get("successful_models"),
            "llm_route_samples": (ev.get("llm_route_samples") or [])[:10],
        },
    }
    with open(TARGET / "controlled_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(controlled, f, ensure_ascii=False, indent=2)

    # fallback_trace.jsonl = same record format, one line
    with open(TARGET / "fallback_trace.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # Also overwrite input.json using case_044 csv row to keep root file fresh
    csv_path = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "evaluation_cases.csv"
    row_044 = None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("case_id") == "case_044":
                row_044 = row
                break
    if row_044:
        input_json = {
            "case_id": "case_04_fallback",
            "primary_category": row_044.get("primary_category", ""),
            "tags": row_044.get("tags", ""),
            "background": row_044.get("background", ""),
            "existing_skills": row_044.get("existing_skills", ""),
            "learning_goal": row_044.get("learning_goal", ""),
            "daily_hours": row_044.get("daily_hours", ""),
            "learning_style": row_044.get("learning_style", ""),
            "hardware": row_044.get("hardware", ""),
            "expected_target_blocks": row_044.get("expected_target_blocks", ""),
            "expected_first_stage": row_044.get("expected_first_stage", ""),
            "expected_difficulty": row_044.get("expected_difficulty", ""),
            "expected_strategy": row_044.get("expected_strategy", ""),
            "run_stage_2": row_044.get("run_stage_2", ""),
            "answer_script_stage_1": row_044.get("answer_script_stage_1", ""),
            "user_input_text": (
                "我的学习背景：" + str(row_044.get("background", "")) + "\n"
                + "已掌握技能：" + str(row_044.get("existing_skills", "")) + "\n"
                + "我的学习目标：" + str(row_044.get("learning_goal", "")) + "\n"
                + "我每天能学：" + str(row_044.get("daily_hours", "")) + "小时\n"
                + "我喜欢怎么学：" + str(row_044.get("learning_style", "")) + "\n"
                + "我的硬件条件：" + str(row_044.get("hardware", ""))
            ),
        }
        with open(TARGET / "input.json", "w", encoding="utf-8") as f:
            json.dump(input_json, f, ensure_ascii=False, indent=2)

    # Copy over run_manifest/run_summary if present
    if (RAW_044 / "run_manifest.json").exists():
        shutil.copyfile(RAW_044 / "run_manifest.json", TARGET / "run_manifest.json")
    if (RAW_044 / "run_summary.json").exists():
        shutil.copyfile(RAW_044 / "run_summary.json", TARGET / "run_summary.json")

    print("CASE_04_FALLBACK_SYNC_OK session_id=", session_id, "passed=", rec.get("passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
