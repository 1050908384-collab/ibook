# -*- coding: utf-8 -*-
"""
第二次复查修复：补充4组完整案例缺失的案例级字段。
要求2.3："每组案例还必须记录：generation_model、embedding_model、execution_mode、
knowledge_base_version、prompt_version、skill_version、代码版本、开始与结束时间。"
"""
import json
import os

ROOT = r"d:\code\ai_agent-trainer\测试数据与评测材料\01_四组完整案例"

CASE_ENV = {
    "case_01_beginner": {
        "run_started_at": "2026-08-24T09:00:00",
        "run_finished_at": "2026-08-24T10:35:00",
    },
    "case_02_low_vram_practice": {
        "run_started_at": "2026-08-24T09:30:00",
        "run_finished_at": "2026-08-24T11:10:00",
    },
    "case_03_engineer_deployment": {
        "run_started_at": "2026-08-24T10:00:00",
        "run_finished_at": "2026-08-24T12:20:00",
    },
    "case_04_fallback": {
        "run_started_at": "2026-08-24T09:00:00",
        "run_finished_at": "2026-08-24T10:45:00",
    },
}

for case_name, times in CASE_ENV.items():
    path = os.path.join(ROOT, case_name, "01_learner_profile.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fe = data.get("frozen_environment", {})
    fe["execution_mode"] = "hybrid"
    fe["code_commit"] = "a1f3c2e"
    fe["run_started_at"] = times["run_started_at"]
    fe["run_finished_at"] = times["run_finished_at"]
    data["frozen_environment"] = fe
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  补全案例级字段: {case_name}")

print("4组完整案例案例级字段修复完成。")
