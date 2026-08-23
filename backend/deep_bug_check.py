# -*- coding: utf-8 -*-
"""最刁钻 Bug 检测 - 异常码一致性 + 幂等性 + 字段完整性（只测学情诊断与阶段规划 Agent/3）"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fastapi.testclient import TestClient
from main import app

p = f = 0
errors = []

def check(n, c, d=""):
    global p, f
    if c: p += 1; print(f"  PASS  {n}")
    else: f += 1; print(f"  FAIL  {n} | {d}"); errors.append(n)

def make_client():
    return TestClient(app)

print("=" * 60)
print("最刁钻 Bug 检测 - 学情诊断与阶段规划 Agent/3 专属")
print("=" * 60)

# ============================================================
# [I] abandoned 会话调用所有写操作接口（异常码一致性）
# 期望：全部是 4xx，不得出现 500，且"not active"类语义应走 409
# ============================================================
print("\n[I] abandoned 会话全接口异常码检测")
print("-" * 60)
i1 = make_client()
sid_i1 = i1.post("/api/v1/sessions").json().get("session_id")
i1.post(f"/api/v1/sessions/{sid_i1}/diagnosis/messages", json={"content": "学习Vue，每天2小时"})
i1.post(f"/api/v1/sessions/{sid_i1}/profile/confirm", json={"confirm": True})
i1.post(f"/api/v1/sessions/{sid_i1}/abandon")
assert i1.get(f"/api/v1/sessions/{sid_i1}").json()["status"] == "abandoned"

# I-1: abandoned 再 PATCH profile
r = i1.patch(f"/api/v1/sessions/{sid_i1}/profile", json={"goal": "新目标x"})
check("I-1 abandoned PATCH/profile=4xx (非500)", 400 <= r.status_code < 500, 
      f"status={r.status_code}, body={r.text[:120]}")

# I-2: abandoned 再 POST profile/confirm（写操作）
r = i1.post(f"/api/v1/sessions/{sid_i1}/profile/confirm", json={"confirm": True})
check("I-2 abandoned CONFIRM/profile=4xx (非500)", 400 <= r.status_code < 500, 
      f"status={r.status_code}, body={r.text[:120]}")

# I-3: abandoned 再 POST stages/current/start
r = i1.post(f"/api/v1/sessions/{sid_i1}/stages/current/start")
check("I-3 abandoned START stage=4xx (非500)", 400 <= r.status_code < 500, 
      f"status={r.status_code}, body={r.text[:120]}")

# I-4: abandoned 再 POST stages/current/answer
r = i1.post(f"/api/v1/sessions/{sid_i1}/stages/current/answer",
            json={"question_id": "Q1", "answer": "x"})
check("I-4 abandoned SUBMIT answer=4xx (非500)", 400 <= r.status_code < 500, 
      f"status={r.status_code}, body={r.text[:120]}")

# I-5: abandoned 再 POST stages/current/complete
r = i1.post(f"/api/v1/sessions/{sid_i1}/stages/current/complete")
check("I-5 abandoned COMPLETE stage=4xx (非500)", 400 <= r.status_code < 500, 
      f"status={r.status_code}, body={r.text[:120]}")

# I-6: abandoned 再 POST stages/current/plan-next
r = i1.post(f"/api/v1/sessions/{sid_i1}/stages/current/plan-next",
            json={"adjustment_notes": "加难度"})
check("I-6 abandoned PLAN next=4xx (非500)", 400 <= r.status_code < 500, 
      f"status={r.status_code}, body={r.text[:120]}")

# I-7: abandoned 再 abandon → 幂等（不应崩溃）
r = i1.post(f"/api/v1/sessions/{sid_i1}/abandon")
check("I-7 abandoned 再 abandon 不崩溃(200/409)", 
      r.status_code in (200, 409), f"status={r.status_code}")

# I-8: abandoned 读操作 GET snapshot / GET profile / GET stages 应正常 200/404
# 读操作一般允许，GET sessions snapshot：
r = i1.get(f"/api/v1/sessions/{sid_i1}")
check("I-8 abandoned GET snapshot=200 (读操作开放)", r.status_code == 200, 
      f"status={r.status_code}")
# GET profile
r = i1.get(f"/api/v1/sessions/{sid_i1}/profile")
check("I-9 abandoned GET profile=200 (读操作开放)", r.status_code == 200, 
      f"status={r.status_code}")
# GET stages/current
r = i1.get(f"/api/v1/sessions/{sid_i1}/stages/current")
check("I-10 abandoned GET stages=200 (读操作开放)", r.status_code == 200, 
      f"status={r.status_code}")
i1.close()

# ============================================================
# [J] 幂等性（重复调用不崩、不产生副作用）
# ============================================================
print("\n[J] 幂等性检测（重复操作不崩）")
print("-" * 60)
j1 = make_client()

# J-1: 重复 start
sid_j1 = j1.post("/api/v1/sessions").json().get("session_id")
j1.post(f"/api/v1/sessions/{sid_j1}/diagnosis/messages", json={"content": "学习React，每天3小时"})
j1.post(f"/api/v1/sessions/{sid_j1}/profile/confirm", json={"confirm": True})
r1 = j1.post(f"/api/v1/sessions/{sid_j1}/stages/current/start")
check("J-1a start 第1次=200", r1.status_code == 200, f"status={r1.status_code}")
r2 = j1.post(f"/api/v1/sessions/{sid_j1}/stages/current/start")
check("J-1b start 第2次（重复）=非500", 
      200 <= r2.status_code < 500, f"status={r2.status_code}")

# J-2: 重复 complete（已在 G5 测过，但在 learning 状态下 double submit answer）
r = j1.get(f"/api/v1/sessions/{sid_j1}/stages/current/package")
questions = r.json().get("package", {}).get("questions", []) if r.status_code == 200 else []
if questions:
    qid = questions[0].get("question_id", "Q_FAKE")
    a1 = j1.post(f"/api/v1/sessions/{sid_j1}/stages/current/answer",
                 json={"question_id": qid, "answer": "第一次作答"})
    check("J-2a answer 第1次=200", a1.status_code == 200, f"status={a1.status_code}")
    a2 = j1.post(f"/api/v1/sessions/{sid_j1}/stages/current/answer",
                 json={"question_id": qid, "answer": "第二次作答（同题）"})
    check("J-2b answer 重复提交=非500", 200 <= a2.status_code < 500, 
          f"status={a2.status_code}")

# J-3: 不存在 question_id 提交答案 → 404 非 500
a3 = j1.post(f"/api/v1/sessions/{sid_j1}/stages/current/answer",
             json={"question_id": "Q_NOT_EXIST_12345", "answer": "x"})
check("J-3 不存在question_id=404/400 (非500)", 400 <= a3.status_code < 500,
      f"status={a3.status_code}, body={a3.text[:120]}")

# J-4: plan-next 在没 complete 之前调用 → 409 非 500
r = j1.post(f"/api/v1/sessions/{sid_j1}/stages/current/plan-next",
            json={"adjustment_notes": "我还没学完"})
check("J-4 未complete就plan-next=4xx (非500)", 400 <= r.status_code < 500,
      f"status={r.status_code}, body={r.text[:120]}")

j1.close()

# ============================================================
# [K] GET /sessions 列表 + profile_summary 字段完整性（每会话必15项）
# ============================================================
print("\n[K] 列表字段完整性检测")
print("-" * 60)
k1 = make_client()

# 准备 3 种状态会话
for goal in ["学习Unity游戏开发", "学习Go语言微服务", "学习Vue3前端"]:
    s = k1.post("/api/v1/sessions").json().get("session_id")
    k1.post(f"/api/v1/sessions/{s}/diagnosis/messages", json={"content": f"{goal}，每天2小时"})

r = k1.get("/api/v1/sessions")
check("K-1 list_sessions=200", r.status_code == 200)
sessions = r.json().get("sessions", [])
check("K-2 sessions 非空", len(sessions) > 0, f"len={len(sessions)}")

DOC_FIELDS = ["session_id", "title", "goal", "grade", "skill_tags", "knowledge_gaps",
              "daily_hours", "learning_style", "version", "status", "phase",
              "current_stage_num", "current_stage_name", "progress", "updated_at"]

for i, s in enumerate(sessions[:5]):  # 只抽前5条（够了）
    for field in DOC_FIELDS:
        check(f"K-3-{i} items[] 缺 {field}", field in s,
              f"session #{i} 缺字段 {field}, 现有={sorted(s.keys())}")
    # 额外：profile_summary 必须和平铺字段一致
    ps = s.get("profile_summary") or {}
    for field in ["goal", "grade", "skill_tags", "knowledge_gaps",
                  "daily_hours", "learning_style", "version",
                  "current_stage_num", "current_stage_name", "progress"]:
        if s.get(field) != ps.get(field):
            check(f"K-4-{i} 平铺≠summary {field}", False,
                  f"平铺={s.get(field)!r}, summary={ps.get(field)!r}")
            break
    else:
        check(f"K-4-{i} 平铺字段=profile_summary 10项", True)

    # 类型强校验
    check(f"K-5-{i} skill_tags 是 list", isinstance(s.get("skill_tags"), list))
    check(f"K-5-{i} knowledge_gaps 是 list", isinstance(s.get("knowledge_gaps"), list))
    check(f"K-5-{i} progress 是 int", isinstance(s.get("progress"), int),
          f"type={type(s.get('progress')).__name__}")
    check(f"K-5-{i} current_stage_num 是 int", isinstance(s.get("current_stage_num"), int),
          f"type={type(s.get('current_stage_num')).__name__}")
    check(f"K-5-{i} status 合法值", s.get("status") in ["active", "completed", "abandoned"])
    check(f"K-5-{i} progress 0..100", 0 <= int(s.get("progress", 0)) <= 100,
          f"progress={s.get('progress')}")

# GET snapshot 字段完整性
if sessions:
    sid_sample = sessions[0]["session_id"]
    snap = k1.get(f"/api/v1/sessions/{sid_sample}").json()
    SNAP_FIELDS = ["session_id", "title", "status", "phase", "profile", "profile_modified",
                   "current_stage", "next_stage", "learning_package",
                   "assessment_summary", "progress_summary", "recent_qa",
                   "available_actions", "created_at", "updated_at"]
    for field in SNAP_FIELDS:
        check(f"K-6 snapshot 缺 {field}", field in snap,
              f"snapshot 缺字段 {field}, 现有={sorted(snap.keys())}")

k1.close()

# ============================================================
# [L] GET stages/current/package 响应字段完整性（文档要求 10+项）
# ============================================================
print("\n[L] 学习包响应字段完整性检测")
print("-" * 60)
l1 = make_client()
sid_l1 = l1.post("/api/v1/sessions").json().get("session_id")
l1.post(f"/api/v1/sessions/{sid_l1}/diagnosis/messages", json={"content": "学习Flutter，每天1.5小时"})
l1.post(f"/api/v1/sessions/{sid_l1}/profile/confirm", json={"confirm": True})
l1.post(f"/api/v1/sessions/{sid_l1}/stages/current/start")
r = l1.get(f"/api/v1/sessions/{sid_l1}/stages/current/package")
if r.status_code == 200:
    pkg = r.json().get("package", {})
    PKG_FIELDS = ["stage_id", "stage_num", "block_name", "subtopic_name",
                  "estimated_hours", "tutorial", "code_examples",
                  "learning_resources", "questions", "review_summary",
                  "recommended_daily_hours", "suggested_study_days",
                  "learning_style_notes", "hardware_notes", "personalization_summary"]
    for field in PKG_FIELDS:
        check(f"L-1 package 缺 {field}", field in pkg,
              f"package 缺字段 {field}, 现有={sorted(pkg.keys())}")
    # learning_resources 每个资源 9 字段
    lr = pkg.get("learning_resources", [])
    if lr:
        LR9 = ["resource_id", "resource_type", "title", "url", "platform",
               "description", "recommendation_level", "role", "order"]
        for j, res in enumerate(lr[:3]):
            for field in LR9:
                if field not in res:
                    check(f"L-2 lr[{j}] 缺 {field}", False,
                          f"lr[{j}] 缺字段 {field}, 现有={sorted(res.keys())}")
                    break
            else:
                check(f"L-2 lr[{j}] 9字段齐全", True)
    # questions 不能含敏感字段（correct_answer等）
    for q in pkg.get("questions", []):
        leak = [f for f in ["correct_answer", "explanation", "expected_output", "reference_answer"] if f in q]
        if leak:
            check(f"L-3 questions 含敏感字段", False, f"泄露={leak}")
            break
    else:
        check("L-3 questions 无敏感字段泄露", True)
else:
    check("L-0 package=200", False, f"status={r.status_code}, body={r.text[:200]}")
l1.close()

print(f"\n{'='*60}")
print(f"结果: {p} 通过, {f} 失败")
print(f"{'='*60}")
if errors:
    for e in errors:
        print(f"  ❌ {e}")
else:
    print("\n✅ 全部最刁钻检测通过！")
sys.exit(0 if f == 0 else 1)
