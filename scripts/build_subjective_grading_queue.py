"""Build an offline queue for subjective-answer review.

This script only reads frozen evaluation outputs. It never calls an LLM and
never changes raw_outputs.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "测试数据与评测材料" / "02_50组个性化主评测" / "raw_outputs"
OUT = ROOT / "测试数据与评测材料" / "06_指标统计" / "subjective_grading_queue.csv"
GRADING_RESULTS = ROOT / "测试数据与评测材料" / "06_指标统计" / "subjective_agent6_grading.jsonl"


def load_agent6_evidence():
    evidence = set()
    if not GRADING_RESULTS.exists():
        return evidence
    for line in GRADING_RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("success") and item.get("grading"):
            evidence.add((str(item.get("case_id")), str(item.get("stage")), str(item.get("question_id"))))
    return evidence

fields = ["case_id", "stage", "question_id", "question_type", "question_text",
          "reference_answer", "submitted_answer", "simulated_is_correct",
          "rubric_present", "agent6_grading_present", "review_status", "notes"]
rows = []
agent6_evidence = load_agent6_evidence()
for case in sorted(RAW.glob("case_*")):
    for stage in ("stage_1", "stage_2"):
        d = case / stage
        qf, af = d / "assessment_questions.json", d / "learner_answers.json"
        if not qf.exists():
            continue
        questions = json.loads(qf.read_text(encoding="utf-8"))
        answers = json.loads(af.read_text(encoding="utf-8")) if af.exists() else []
        amap = {str(a.get("question_id")): a for a in answers if isinstance(a, dict)}
        for q in questions:
            qt = str(q.get("question_type", "")).lower()
            if qt not in {"short_answer", "practical"}:
                continue
            a = amap.get(str(q.get("question_id")), {})
            grading_present = (
                any(k in a for k in ("grading_result", "grading_score", "grader_result", "rubric_result"))
                or (case.name, stage, str(q.get("question_id"))) in agent6_evidence
            )
            rubric_present = bool(q.get("rubric") or q.get("grading_rule") or q.get("judge_rule"))
            rows.append({
                "case_id": case.name, "stage": stage,
                "question_id": q.get("question_id", ""), "question_type": qt,
                "question_text": q.get("question_text", ""),
                "reference_answer": q.get("correct_answer") or q.get("reference_answer", ""),
                "submitted_answer": a.get("submitted_answer", ""),
                "simulated_is_correct": a.get("is_correct", ""),
                "rubric_present": rubric_present,
                "agent6_grading_present": grading_present,
                "review_status": "needs_agent6_grading" if not grading_present else "evidence_present",
                "notes": "不能使用 simulated_is_correct 代替 Agent6 判分" if not grading_present else "",
            })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"written={OUT}")
print(f"total={len(rows)} needs_agent6={sum(r['review_status'] == 'needs_agent6_grading' for r in rows)}")
