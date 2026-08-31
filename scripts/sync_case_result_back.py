# -*- coding: utf-8 -*-
"""Copy raw_outputs/<csv_case_id>/stage_1/* back to 01_四组完整案例/<target_case>/* with original 01-10 file names.
Overwrites target files (backup already taken in _backup_precover_20260831).

Usage: python scripts/sync_case_result_back.py <csv_case_id> <target_case_name> [primary_stage=stage_1]

primary_stage: which stage's content populates 06_learning_package.md etc. (case_01_beginner had run_stage_2=True, but historically it's stage_1 as primary 06 package — actually original 06 was a combined, but now we default stage_1 contents and put stage_2 contents under a subdir stage_2_secondary to preserve; learning_package.md uses first non-empty stage's or if stage_2 is multi-stage we use stage_1 for 06 learning_package, stage_2 for 06_stage2.md)
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs"
TARGET_ROOT = ROOT / "测试数据与评测材料" / "01_四组完整案例"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, txt: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")


def copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)


def main(argv):
    if len(argv) < 3:
        print("Usage: sync_case_result_back.py <csv_case_id> <target_case_name> [primary_stage]")
        return 1
    csv_id = argv[1]
    target_name = argv[2]
    primary = argv[3] if len(argv) > 3 else "stage_1"

    case_dir = RAW / csv_id
    target = TARGET_ROOT / target_name
    if not case_dir.exists():
        print(f"ERR raw case dir missing: {case_dir}")
        return 2
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)

    # Stage directory preference: primary stage populates 06-10 artifacts
    pri = case_dir / primary
    alt_stages = sorted([d for d in case_dir.iterdir() if d.is_dir() and d.name != primary and d.name.startswith("stage_")])

    # 01_learner_profile.json  <- stage_*/learner_profile.json (primary or first stage available)
    learner_profile_src = None
    for cand in [pri / "learner_profile.json"] + [s / "learner_profile.json" for s in alt_stages]:
        if cand.exists():
            learner_profile_src = cand
            break
    if learner_profile_src:
        shutil.copyfile(learner_profile_src, target / "01_learner_profile.json")

    # 02_diagnostic_output.json <- primary stage_diagnostic_output.json
    diag_src = pri / "diagnostic_output.json"
    if diag_src.exists():
        shutil.copyfile(diag_src, target / "02_diagnostic_output.json")

    # 03_stage_planning_output.json <- primary stage_planning_output.json
    plan_src = pri / "stage_planning_output.json"
    if plan_src.exists():
        shutil.copyfile(plan_src, target / "03_stage_planning_output.json")

    # 04_generation_events.jsonl (if exists separately, same filename as some targets name it)
    events_src = case_dir / "generation_events.jsonl"
    if events_src.exists():
        # Some targets (case_04) use 04_generation_events.jsonl naming; most of case_01_02_03 use root generation_events.jsonl -> also keep at root level under same name (overwrite)
        shutil.copyfile(events_src, target / "generation_events.jsonl")
        if (target / "04_generation_events.jsonl").exists() or target_name == "case_04_fallback":
            shutil.copyfile(events_src, target / "04_generation_events.jsonl")

    # 05_llm_calls.jsonl (same semantics)
    llm_src = case_dir / "llm_calls.jsonl"
    if llm_src.exists():
        shutil.copyfile(llm_src, target / "llm_calls.jsonl")
        if (target / "05_llm_calls.jsonl").exists() or target_name == "case_04_fallback":
            shutil.copyfile(llm_src, target / "05_llm_calls.jsonl")

    # 06_learning_package.md <- primary stage learning_package.md; if other stages exist, also save as 06_stage_<n>_learning_package.md
    pkg_src = pri / "learning_package.md"
    if pkg_src.exists():
        shutil.copyfile(pkg_src, target / "06_learning_package.md")
    for s in alt_stages:
        s_pkg = s / "learning_package.md"
        if s_pkg.exists():
            shutil.copyfile(s_pkg, target / f"06_{s.name}_learning_package.md")

    # 07_review_trace.json <- primary stage review_trace.json
    rev_src = pri / "review_trace.json"
    if rev_src.exists():
        shutil.copyfile(rev_src, target / "07_review_trace.json")
    for s in alt_stages:
        sp = s / "review_trace.json"
        if sp.exists():
            shutil.copyfile(sp, target / f"07_{s.name}_review_trace.json")

    # 08_assessment_questions.json <- primary stage assessment_questions.json
    aq_src = pri / "assessment_questions.json"
    if aq_src.exists():
        shutil.copyfile(aq_src, target / "08_assessment_questions.json")
    for s in alt_stages:
        sp = s / "assessment_questions.json"
        if sp.exists():
            shutil.copyfile(sp, target / f"08_{s.name}_assessment_questions.json")

    # 09_learner_answers.json <- primary stage learner_answers.json
    la_src = pri / "learner_answers.json"
    if la_src.exists():
        shutil.copyfile(la_src, target / "09_learner_answers.json")
    for s in alt_stages:
        sp = s / "learner_answers.json"
        if sp.exists():
            shutil.copyfile(sp, target / f"09_{s.name}_learner_answers.json")

    # 10_stage_decision.json <- primary stage_decision.json
    dec_src = pri / "stage_decision.json"
    if dec_src.exists():
        shutil.copyfile(dec_src, target / "10_stage_decision.json")
    for s in alt_stages:
        sp = s / "stage_decision.json"
        if sp.exists():
            shutil.copyfile(sp, target / f"10_{s.name}_stage_decision.json")

    # Also overwrite root-level manifest/summary/input.json with the CSV run output
    for name in ["run_manifest.json", "run_summary.json", "input.json"]:
        src = case_dir / name
        if src.exists():
            shutil.copyfile(src, target / name)

    print(f"SYNC_OK csv={csv_id} -> target={target_name} primary_stage={primary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
