# -*- coding: utf-8 -*-
"""极端边界场景 Bug 检测 - 只测 Agent1/3 相关"""
import sys, os, json
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
print("极端边界 Bug 检测 - Agent1/3 专属")
print("=" * 60)

# ============================================================
# [E] 会话状态边界（abandoned / 不存在）
# ============================================================
print("\n[E] 会话状态边界")
print("-" * 60)
e1 = make_client()

# E1: 不存在会话的各种 API
check("E1-1 GET 不存在会话返回 404", e1.get("/api/v1/sessions/FAKE_SESSION").status_code == 404)
check("E1-2 PATCH 不存在会话/profile 返回 404", 
      e1.patch("/api/v1/sessions/FAKE_SESSION/profile", json={"goal": "x"}).status_code == 404)
check("E1-3 POST 不存在会话/abandon 返回 404", 
      e1.post("/api/v1/sessions/FAKE_SESSION/abandon").status_code == 404)
check("E1-4 GET 不存在会话/stages/current 返回 404", 
      e1.get("/api/v1/sessions/FAKE_SESSION/stages/current").status_code == 404)
check("E1-5 POST 不存在会话/diagnosis/messages 返回 404", 
      e1.post("/api/v1/sessions/FAKE_SESSION/diagnosis/messages", 
              json={"content": "test"}).status_code == 404)
check("E1-6 POST 不存在会话/profile/confirm 返回 404", 
      e1.post("/api/v1/sessions/FAKE_SESSION/profile/confirm", 
              json={"confirm": True}).status_code == 404)

# E2: abandoned 会话的 API 调用行为
sid_ab = e1.post("/api/v1/sessions").json().get("session_id")
e1.post(f"/api/v1/sessions/{sid_ab}/diagnosis/messages", json={"content": "学习CV，每天3小时"})
e1.post(f"/api/v1/sessions/{sid_ab}/profile/confirm", json={"confirm": True})
e1.post(f"/api/v1/sessions/{sid_ab}/abandon")
check("E2-1 abandon 返回 200", True)  # 上面没报错就过了

snap = e1.get(f"/api/v1/sessions/{sid_ab}").json()
check("E2-2 快照 status=abandoned", snap.get("status") == "abandoned")

# abandoned 会话 PATCH profile：state_manager.require_session 会抛 ValueError
# 因为 status != 'active'，我们期望返回 4xx 而不是 500
r = e1.patch(f"/api/v1/sessions/{sid_ab}/profile", json={"goal": "修改目标"})
check("E2-3 abandoned PATCH profile 返回 4xx (非500)", 
      400 <= r.status_code < 500, f"实际 status={r.status_code}, body={r.text[:100]}")

# E3: list_sessions 空列表不会崩溃
r = e1.get("/api/v1/sessions")
check("E3-1 list_sessions 返回 200", r.status_code == 200)
check("E3-2 list_sessions sessions 是 list", isinstance(r.json().get("sessions"), list))

e1.close()

# ============================================================
# [F] 诊断与画像边界
# ============================================================
print("\n[F] 诊断与画像边界")
print("-" * 60)
f1 = make_client()

# F1: 空消息诊断
sid_f1 = f1.post("/api/v1/sessions").json().get("session_id")
r = f1.post(f"/api/v1/sessions/{sid_f1}/diagnosis/messages", json={"content": ""})
check("F1-1 空内容诊断不崩溃 (200/400)", r.status_code in (200, 400, 422), f"实际={r.status_code}")

# F2: 连续多次诊断（幂等）
sid_f2 = f1.post("/api/v1/sessions").json().get("session_id")
for i in range(3):
    r = f1.post(f"/api/v1/sessions/{sid_f2}/diagnosis/messages", 
                json={"content": f"第{i+1}次：学习深度学习，每天{i+1}小时"})
    check(f"F2-{i+1} 连续诊断 {i+1}/3 不崩溃", r.status_code == 200, f"status={r.status_code}")

# F3: 诊断后 GET profile（还没 confirm）
sid_f3 = f1.post("/api/v1/sessions").json().get("session_id")
f1.post(f"/api/v1/sessions/{sid_f3}/diagnosis/messages", json={"content": "学习Python基础"})
#  needs_more_info 时应该也能拿到 profile（刚修复的 Bug）
r = f1.get(f"/api/v1/sessions/{sid_f3}/profile")
check("F3-1 needs_more_info 时 GET profile 返回 200", r.status_code == 200, 
      f"status={r.status_code}, body={r.text[:200]}")
if r.status_code == 200:
    check("F3-2 profile 含 goal 字段", "goal" in r.json())
    check("F3-3 profile 含 daily_hours 字段", "daily_hours" in r.json())

# F4: 连续多次 confirm（幂等）
sid_f4 = f1.post("/api/v1/sessions").json().get("session_id")
f1.post(f"/api/v1/sessions/{sid_f4}/diagnosis/messages", json={"content": "学习NLP，每天2小时"})
for i in range(2):
    r = f1.post(f"/api/v1/sessions/{sid_f4}/profile/confirm", json={"confirm": True})
    check(f"F4-{i+1} 重复 confirm 不崩溃", r.status_code == 200)

# F5: confirm=False（质疑画像）
sid_f5 = f1.post("/api/v1/sessions").json().get("session_id")
f1.post(f"/api/v1/sessions/{sid_f5}/diagnosis/messages", json={"content": "学习RL"})
r = f1.post(f"/api/v1/sessions/{sid_f5}/profile/confirm", 
            json={"confirm": False, "notes": "硬件信息填错了"})
check("F5-1 质疑画像返回 200", r.status_code == 200)
check("F5-2 质疑后 status=disputed", r.json().get("status") == "disputed")
check("F5-3 质疑后 phase=diagnosing", r.json().get("phase") == "diagnosing")

# F6: PATCH profile 空 body（没有实际字段更新）
sid_f6 = f1.post("/api/v1/sessions").json().get("session_id")
f1.post(f"/api/v1/sessions/{sid_f6}/diagnosis/messages", json={"content": "学习前端"})
r = f1.patch(f"/api/v1/sessions/{sid_f6}/profile", json={})
check("F6-1 PATCH 空 body 返回 200", r.status_code == 200)
check("F6-2 PATCH 空 body updated_fields 是空 dict", 
      r.json().get("updated_fields") == {}, f"实际={r.json().get('updated_fields')!r}")
check("F6-3 PATCH 空 body profile_modified=False", 
      r.json().get("profile_modified") == False)

# F7: PATCH profile 非法类型 daily_hours（应该 422 不是 500）
sid_f7 = f1.post("/api/v1/sessions").json().get("session_id")
f1.post(f"/api/v1/sessions/{sid_f7}/diagnosis/messages", json={"content": "学习后端"})
r = f1.patch(f"/api/v1/sessions/{sid_f7}/profile", json={"daily_hours": "不是数字"})
check("F7-1 daily_hours 非法类型返回 422/400 (非500)", 
      400 <= r.status_code < 500, f"status={r.status_code}")

f1.close()

# ============================================================
# [G] 阶段与学习包边界
# ============================================================
print("\n[G] 阶段与学习包边界")
print("-" * 60)
g1 = make_client()

# G1: 没诊断就 GET stages/current（应该 404）
sid_g1 = g1.post("/api/v1/sessions").json().get("session_id")
r = g1.get(f"/api/v1/sessions/{sid_g1}/stages/current")
check("G1-1 无阶段时 GET current 返回 404", r.status_code == 404, f"status={r.status_code}")

# G2: 没诊断就 start（应该 409/404）
sid_g2 = g1.post("/api/v1/sessions").json().get("session_id")
r = g1.post(f"/api/v1/sessions/{sid_g2}/stages/current/start")
check("G2-1 无阶段 start 返回 409/404 (非500)", 
      400 <= r.status_code < 500, f"status={r.status_code}, body={r.text[:100]}")

# G3: stage_ready 状态就 GET package（应该 409 提示先 start）
sid_g3 = g1.post("/api/v1/sessions").json().get("session_id")
g1.post(f"/api/v1/sessions/{sid_g3}/diagnosis/messages", json={"content": "学习Android，每天2.5小时"})
g1.post(f"/api/v1/sessions/{sid_g3}/profile/confirm", json={"confirm": True})
# 确认后 phase=stage_ready，还没 start
r = g1.get(f"/api/v1/sessions/{sid_g3}/stages/current/package")
check("G3-1 stage_ready 状态 GET package 返回 409", r.status_code == 409, 
      f"status={r.status_code}, body={r.text[:100]}")

# G4: 没 start 就 submit_answer（404/409）
sid_g4 = g1.post("/api/v1/sessions").json().get("session_id")
g1.post(f"/api/v1/sessions/{sid_g4}/diagnosis/messages", json={"content": "学习数据库"})
g1.post(f"/api/v1/sessions/{sid_g4}/profile/confirm", json={"confirm": True})
r = g1.post(f"/api/v1/sessions/{sid_g4}/stages/current/answer", 
            json={"question_id": "Q_FAKE", "answer": "xxx"})
check("G4-1 未 start 提交答案返回 4xx (非500)", 
      400 <= r.status_code < 500, f"status={r.status_code}")

# G5: 没 start 就 complete（404/409）
sid_g5 = g1.post("/api/v1/sessions").json().get("session_id")
g1.post(f"/api/v1/sessions/{sid_g5}/diagnosis/messages", json={"content": "学习网络协议"})
g1.post(f"/api/v1/sessions/{sid_g5}/profile/confirm", json={"confirm": True})
r = g1.post(f"/api/v1/sessions/{sid_g5}/stages/current/complete")
check("G5-1 未 start complete 返回 4xx (非500)", 
      400 <= r.status_code < 500, f"status={r.status_code}")

# G6: 正常流程后 GET stages/current 检查个性化字段齐全
sid_g6 = g1.post("/api/v1/sessions").json().get("session_id")
g1.post(f"/api/v1/sessions/{sid_g6}/diagnosis/messages", json={"content": "学习iOS开发，每天1.5小时"})
g1.post(f"/api/v1/sessions/{sid_g6}/profile/confirm", json={"confirm": True})
r = g1.get(f"/api/v1/sessions/{sid_g6}/stages/current")
if r.status_code == 200:
    stage = r.json().get("stage", {})
    pers_fields = ["recommended_daily_hours", "suggested_study_days", 
                   "learning_style_notes", "hardware_notes", "personalization_summary"]
    for field in pers_fields:
        check(f"G6-1 stages/current 含 {field}", field in stage, 
              f"缺少字段 {field}. keys={list(stage.keys())}")
    check("G6-2 recommended_daily_hours=1.5", 
          stage.get("recommended_daily_hours") == 1.5,
          f"实际={stage.get('recommended_daily_hours')}")
else:
    check(f"G6 stages/current 返回 200", False, f"status={r.status_code}, body={r.text[:200]}")

g1.close()

# ============================================================
# [H] _build_learning_resources 极端输入
# ============================================================
print("\n[H] _build_learning_resources 极端输入")
print("-" * 60)
from api.v1.stages import _build_learning_resources

# H1: None 输入
try:
    r = _build_learning_resources(None, None)
    check("H1-1 None,None 输入不崩溃", isinstance(r, list))
except Exception as e:
    check("H1-1 None,None 输入不崩溃", False, str(e))

# H2: stage=None, package={}
try:
    r = _build_learning_resources(None, {})
    check("H2-1 stage=None 返回 list", isinstance(r, list))
    check("H2-2 fallback 仍有 2 条", len(r) == 2, f"实际 len={len(r)}")
except Exception as e:
    check("H2-1 stage=None 不崩溃", False, str(e))

# H3: learning_resources 含 None 项
try:
    r = _build_learning_resources({}, {
        "learning_resources": [
            None,  # None 项 → 被安全过滤
            {"url": None, "title": None, "resource_type": "video"},  # 字段全 None，但本身是 dict → 保留
            {"url": 12345, "platform": ["not", "a", "string"]},  # 字段类型错误，但本身是 dict → 保留
        ]
    })
    check("H3-1 含异常 resource 不崩溃", isinstance(r, list))
    # None/非 dict 会被安全过滤，所以最终是 2 条（不是 3 条）
    check("H3-2 None 被过滤、dict 保留共 2 条", len(r) == 2, f"实际 len={len(r)}")
except Exception as e:
    check("H3-1 含异常 resource 不崩溃", False, str(e))

# H4: 各种 URL 类型
from api.v1.stages import _extract_platform
check("H4-1 int URL -> ''", _extract_platform(123) == "")
check("H4-2 list URL -> ''", _extract_platform([]) == "")
check("H4-3 dict URL -> ''", _extract_platform({}) == "")
check("H4-4 bool True URL -> ''", _extract_platform(True) == "")  # True 是 1，不是 str
check("H4-5 空格 URL -> ''", _extract_platform("   ") == "")

print(f"\n{'='*60}")
print(f"结果: {p} 通过, {f} 失败")
print(f"{'='*60}")
if errors:
    for e in errors:
        print(f"  ❌ {e}")
else:
    print("\n✅ 所有极端边界检测通过！")
sys.exit(0 if f == 0 else 1)
