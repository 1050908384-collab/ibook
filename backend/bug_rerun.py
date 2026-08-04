# -*- coding: utf-8 -*-
"""全面 Bug 重检：3个新任务 + 4个已修复Bug 的所有边界"""
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
print("全面 Bug 重检：learning_resources / PATCH /profile / sessions")
print("=" * 60)

# ============================================================
# [A] learning_resources 深度 Bug 检测
# ============================================================
print("\n[A] learning_resources 深度检测")
print("-" * 60)

# --- A0: 函数直接调用（所有分支） ---
from api.v1.stages import _build_learning_resources, _extract_platform

# A0a: _extract_platform 全场景
check("A0a-1 bilibili.com", _extract_platform("https://bilibili.com/v/1") == "bilibili")
check("A0a-2 www.bilibili.com", _extract_platform("https://www.bilibili.com/video/BV1") == "bilibili")
check("A0a-3 m.bilibili.com", _extract_platform("https://m.bilibili.com/video/BV1") == "bilibili")
check("A0a-4 b23.tv 短链", _extract_platform("https://b23.tv/ABC123") == "bilibili")
check("A0a-5 youtube.com", _extract_platform("https://www.youtube.com/watch?v=abc") == "youtube")
check("A0a-6 music.youtube.com", _extract_platform("https://music.youtube.com/watch?v=1") == "youtube")
check("A0a-7 youtu.be 短链", _extract_platform("https://youtu.be/abc") == "youtube")
check("A0a-8 github.com", _extract_platform("https://github.com/user/repo") == "github")
check("A0a-9 gist.github.com", _extract_platform("https://gist.github.com/user/1") == "github")
check("A0a-10 github.io", _extract_platform("https://user.github.io/repo") == "github")
check("A0a-11 zhihu.com", _extract_platform("https://www.zhihu.com/question/1") == "zhihu")
check("A0a-12 zhuanlan.zhihu", _extract_platform("https://zhuanlan.zhihu.com/p/1") == "zhihu")
check("A0a-13 csdn.net", _extract_platform("https://blog.csdn.net/user/article/1") == "csdn")
check("A0a-14 www.csdn.net", _extract_platform("https://www.csdn.net/article/1") == "csdn")
check("A0a-15 空URL", _extract_platform("") == "")
check("A0a-16 None-like URL", _extract_platform(None) == "")
check("A0a-16b 数字非 str", _extract_platform(12345) == "")
check("A0a-17 未知站点", _extract_platform("https://example.com/page") == "")
check("A0a-18 带参数", _extract_platform("https://bilibili.com/v?aid=123&p=1") == "bilibili")

# A0b: _build_learning_resources 所有分支
# 分支1: package 已有 resources（各种字段缺失情况）
r1 = _build_learning_resources({}, {
    "learning_resources": [
        {"resource_id": "R1"},  # 几乎全缺
        {"canonical_url": "https://www.bilibili.com/video/BV1",
         "content_summary": "摘要内容",
         "sequence_index": 3,
         "role": "primary"},  # 无 resource_id
    ]
})
check("A0b-1 2 条资源", len(r1) == 2)
if len(r1) == 2:
    # primary 优先（role=primary 排前面）
    check("A0b-2 primary 优先", r1[0].get("role") == "primary", f"实际 role={r1[0].get('role')}")
    check("A0b-3 R1 resource_type 默认 video",
          [r for r in r1 if r["resource_id"] == "R1"][0].get("resource_type") == "video")
    check("A0b-4 R0(缺id) 补 resource_id",
          any(r.get("resource_id", "").startswith("res_") for r in r1),  # 缺 resource_id 时补 res_idx
          f"所有 resource_id: {[r.get('resource_id') for r in r1]}")
    check("A0b-5 R0 platform=bilibili", r1[0].get("platform") == "bilibili",
          f"实际 platform='{r1[0].get('platform')}'")
    check("A0b-6 R0 description 取 content_summary",
          "摘要" in r1[0].get("description", ""), f"实际 desc='{r1[0].get('description')}'")
    check("A0b-7 R0 order 取 sequence_index=3", r1[0].get("order") == 3)
    check("A0b-8 R1 order 取默认值=1",
          [r for r in r1 if r["resource_id"] == "R1"][0].get("order") == 1)
    check("A0b-9 recommendation_level 取 role",
          r1[0].get("recommendation_level") == "primary")

# 分支2: package 无 resources，但有 block_id（从 DB 读）
# 这里我们无法保证 DB 有数据，只测试不崩溃即可
try:
    r2 = _build_learning_resources({
        "block_id": "KB-12", "stage_id": "S1", "block_name": "Python"
    }, {})
    check("A0b-10 DB读取分支不崩溃", True)
    check("A0b-11 返回 list", isinstance(r2, list))
except Exception as e:
    check("A0b-10 DB读取分支不崩溃", False, f"异常: {e}")

# 分支3: 完全空输入（fallback）
r3 = _build_learning_resources({}, {})
check("A0b-12 fallback 2条", len(r3) == 2)
if len(r3) == 2:
    check("A0b-13 fallback[0] 是 article", r3[0].get("resource_type") == "article")
    check("A0b-14 fallback[1] 是 code", r3[1].get("resource_type") == "code")
    check("A0b-15 fallback[0] recommendation_level=primary",
          r3[0].get("recommendation_level") == "primary")
    check("A0b-16 fallback[1] recommendation_level=supplementary",
          r3[1].get("recommendation_level") == "supplementary")
    check("A0b-17 fallback[0] order=1", r3[0].get("order") == 1)
    check("A0b-18 fallback[1] order=2", r3[1].get("order") == 2)
    check("A0b-19 fallback[0] platform=''（空URL）", r3[0].get("platform") == "")
    check("A0b-20 fallback url 是 string（含空）", isinstance(r3[0].get("url", ""), str))

# --- A1: 真实 API 调用 ---
c1 = make_client()
sid = c1.post("/api/v1/sessions").json().get("session_id")
c1.post(f"/api/v1/sessions/{sid}/diagnosis/messages",
        json={"content": "学习LoRA微调，每天2小时，动手型"})
c1.post(f"/api/v1/sessions/{sid}/profile/confirm", json={"confirm": True})
c1.post(f"/api/v1/sessions/{sid}/stages/current/start")
r = c1.get(f"/api/v1/sessions/{sid}/stages/current/package")
check("A1-1 get package 200", r.status_code == 200)
if r.status_code == 200:
    pkg = r.json().get("package", {})
    lr = pkg.get("learning_resources", [])
    check("A1-2 learning_resources 非空", len(lr) > 0, f"len={len(lr)}")
    if lr:
        # 每个资源都做字段检查
        for idx, res in enumerate(lr[:2]):
            req9 = ["resource_id", "resource_type", "title", "url",
                    "platform", "description", "recommendation_level", "role", "order"]
            for field in req9:
                if field not in res:
                    check(f"A1-3-{idx} 缺 {field}", False)
                    break
            else:
                check(f"A1-3-{idx} 9字段齐全", True)
            # 类型检查
            check(f"A1-4-{idx} resource_type 合法",
                  res.get("resource_type") in ["video", "article", "documentation", "code"])
            check(f"A1-4-{idx} role 合法",
                  res.get("role") in ["primary", "supplementary"])
            check(f"A1-4-{idx} order 是 int", isinstance(res.get("order"), int))
            check(f"A1-4-{idx} level 合法",
                  res.get("recommendation_level") in ["primary", "supplementary"])
c1.close()

# ============================================================
# [B] PATCH /profile 深度检测
# ============================================================
print("\n[B] PATCH /profile 深度检测")
print("-" * 60)

# B1: 完整流程（PATCH 不改阶段 → confirm 才改）
b1 = make_client()
sid = b1.post("/api/v1/sessions").json().get("session_id")
b1.post(f"/api/v1/sessions/{sid}/diagnosis/messages",
        json={"content": "学习LoRA微调，每天2小时，动手型"})
b1.post(f"/api/v1/sessions/{sid}/profile/confirm", json={"confirm": True})

before = b1.get(f"/api/v1/sessions/{sid}/stages/current").json().get("stage", {})
check("B1-1 before daily_hours=2.0", before.get("recommended_daily_hours") == 2.0)

r = b1.patch(f"/api/v1/sessions/{sid}/profile", json={
    "daily_hours": 5.0,
    "learning_style": "visual",
    "hardware": "RTX 4090",
    "goal": "新目标",
})
check("B1-2 PATCH 200", r.status_code == 200)
d = r.json()
check("B1-3 updated_fields 4个", len(d.get("updated_fields", {})) == 4)
check("B1-4 profile_modified=True", d.get("profile_modified") == True)

# 快照验证
snap = b1.get(f"/api/v1/sessions/{sid}").json()
check("B1-5 快照 profile_modified=True", snap.get("profile_modified") == True)
check("B1-6 快照 profile.daily_hours=5.0", snap.get("profile", {}).get("daily_hours") == 5.0)

# 阶段未变
after_patch = b1.get(f"/api/v1/sessions/{sid}/stages/current").json().get("stage", {})
check("B1-7 after_patch stage_id 不变",
      after_patch.get("stage_id") == before.get("stage_id"))
check("B1-8 after_patch daily_hours 仍=2.0",
      after_patch.get("recommended_daily_hours") == 2.0)

# 确认后才变
b1.post(f"/api/v1/sessions/{sid}/profile/confirm", json={"confirm": True})
after_cf = b1.get(f"/api/v1/sessions/{sid}/stages/current").json().get("stage", {})
check("B1-9 confirm 后 daily_hours=5.0",
      after_cf.get("recommended_daily_hours") == 5.0)

# GET profile 接口验证
pr = b1.get(f"/api/v1/sessions/{sid}/profile").json()
check("B1-10 profile.daily_hours=5.0", pr.get("daily_hours") == 5.0)
check("B1-11 profile.learning_style=visual", pr.get("learning_style") == "visual")
check("B1-12 profile.hardware=RTX 4090", pr.get("hardware") == "RTX 4090")
check("B1-13 profile.goal=新目标", pr.get("goal") == "新目标")
b1.close()

# B2: 边界
b2 = make_client()
sid = b2.post("/api/v1/sessions").json().get("session_id")
b2.post(f"/api/v1/sessions/{sid}/diagnosis/messages", json={"content": "学习Python"})

# 空请求
r = b2.patch(f"/api/v1/sessions/{sid}/profile", json={})
check("B2-1 空 body 200", r.status_code == 200)
d = r.json()
check("B2-2 updated_fields 空", len(d.get("updated_fields", {})) == 0)
check("B2-3 profile_modified=False", d.get("profile_modified") == False)

# 单字段
r = b2.patch(f"/api/v1/sessions/{sid}/profile", json={"daily_hours": 3.5})
check("B2-4 单字段 200", r.status_code == 200)
d = r.json()
check("B2-5 updated_fields 1个", len(d.get("updated_fields")) == 1)
check("B2-6 profile_modified=True", d.get("profile_modified") == True)

# 非法 session
r = b2.patch("/api/v1/sessions/FAKE_ID/profile", json={"goal": "x"})
check("B2-7 不存在会话 404", r.status_code == 404)

# 浮点数 daily_hours
r = b2.patch(f"/api/v1/sessions/{sid}/profile", json={"daily_hours": 1.25})
check("B2-8 小数时长", r.status_code == 200)
b2.close()

# ============================================================
# [C] GET /api/v1/sessions + profile_summary 深度检测
# ============================================================
print("\n[C] GET /sessions + profile_summary 深度检测")
print("-" * 60)

# 准备各种状态的会话
c_ = make_client()
states = {}

# C1: 新建（未诊断）
s1 = c_.post("/api/v1/sessions").json().get("session_id")
states["new"] = s1

# C2: 已诊断但未 confirm
s2 = c_.post("/api/v1/sessions").json().get("session_id")
c_.post(f"/api/v1/sessions/{s2}/diagnosis/messages", json={"content": "学习NLP"})
states["diagnosed"] = s2

# C3: 已 confirm（有阶段）
s3 = c_.post("/api/v1/sessions").json().get("session_id")
c_.post(f"/api/v1/sessions/{s3}/diagnosis/messages", json={"content": "学习CV，每天3小时"})
c_.post(f"/api/v1/sessions/{s3}/profile/confirm", json={"confirm": True})
states["confirmed"] = s3

# C4: 已放弃
s4 = c_.post("/api/v1/sessions").json().get("session_id")
c_.post(f"/api/v1/sessions/{s4}/diagnosis/messages", json={"content": "学习RL"})
c_.post(f"/api/v1/sessions/{s4}/profile/confirm", json={"confirm": True})
c_.post(f"/api/v1/sessions/{s4}/abandon")
states["abandoned"] = s4

r = c_.get("/api/v1/sessions")
check("C-1 GET 200", r.status_code == 200)
all_s = {s["session_id"]: s for s in r.json().get("sessions", [])}

# 逐一检查
for state_name, sid_ in states.items():
    s = all_s.get(sid_)
    if not s:
        check(f"C-2 {state_name} 存在", False, f"session {sid_} 不在列表中")
        continue
    check(f"C-2 {state_name} 存在", True)

    # 15 文档字段都在
    req15 = ["session_id", "title", "goal", "grade", "skill_tags", "knowledge_gaps",
             "daily_hours", "learning_style", "version", "status", "phase",
             "current_stage_num", "current_stage_name", "progress", "updated_at"]
    for fld in req15:
        if fld not in s:
            check(f"C-3 {state_name} 缺 {fld}", False)
            break
    else:
        check(f"C-3 {state_name} 15字段齐全", True)

    # profile_summary 嵌套
    ps = s.get("profile_summary")
    check(f"C-4 {state_name} 含 profile_summary 嵌套", ps is not None)
    if ps:
        # 嵌套字段值 == 平铺值
        for k in ["goal", "grade", "skill_tags", "knowledge_gaps",
                  "daily_hours", "learning_style", "version",
                  "current_stage_num", "current_stage_name", "progress"]:
            if s.get(k) != ps.get(k):
                check(f"C-5 {state_name} 平铺{k} == summary.{k}", False,
                      f"平铺={s.get(k)!r}, summary={ps.get(k)!r}")
                break
        else:
            check(f"C-5 {state_name} 平铺值==summary值", True)

    # 状态特定断言
    if state_name == "new":
        check(f"C-6 new phase=diagnosing", s.get("phase") == "diagnosing")
        check(f"C-6 new progress=0", s.get("progress") == 0)
        check(f"C-6 new current_stage_num=0", s.get("current_stage_num") == 0)
        check(f"C-6 new current_stage_name=''", s.get("current_stage_name") == "")
    elif state_name == "confirmed":
        check(f"C-7 confirmed progress>0", s.get("progress") > 0,
              f"实际={s.get('progress')}")
        check(f"C-7 confirmed current_stage_num=1", s.get("current_stage_num") >= 1)
        check(f"C-7 confirmed 有 block_name", bool(s.get("current_stage_name")),
              f"实际='{s.get('current_stage_name')}'")
    elif state_name == "abandoned":
        check(f"C-8 abandoned status=abandoned", s.get("status") == "abandoned")
        check(f"C-8 abandoned progress=0", s.get("progress") == 0,
              f"实际={s.get('progress')}")

    # 类型检查
    check(f"C-9 {state_name} skill_tags 是 list", isinstance(s.get("skill_tags"), list))
    check(f"C-9 {state_name} knowledge_gaps 是 list", isinstance(s.get("knowledge_gaps"), list))
    check(f"C-9 {state_name} progress 是 int", isinstance(s.get("progress"), int),
          f"实际 type={type(s.get('progress')).__name__}")
    check(f"C-9 {state_name} current_stage_num 是 int", isinstance(s.get("current_stage_num"), int),
          f"实际 type={type(s.get('current_stage_num')).__name__}")

c_.close()

# ============================================================
# [D] 敏感字段 & 导入检查
# ============================================================
print("\n[D] 敏感字段 & 导入检查")
print("-" * 60)
d = make_client()
sid = d.post("/api/v1/sessions").json().get("session_id")
d.post(f"/api/v1/sessions/{sid}/diagnosis/messages", json={"content": "学习LoRA"})
d.post(f"/api/v1/sessions/{sid}/profile/confirm", json={"confirm": True})
d.post(f"/api/v1/sessions/{sid}/stages/current/start")
r = d.get(f"/api/v1/sessions/{sid}/stages/current/package")
if r.status_code == 200:
    qs = r.json().get("package", {}).get("questions", [])
    for q in qs:
        leak = [f for f in ["correct_answer", "explanation", "expected_output", "reference_answer"] if f in q]
        if leak:
            check("D-1 无敏感字段", False, f"泄露 {leak}")
            break
    else:
        if qs:
            check("D-1 无敏感字段", True)
        else:
            check("D-1 无敏感字段（无题目）", True)
d.close()

try:
    from api.v1 import sessions, diagnosis, profile, stages
    check("D-2 v1路由导入", True)
except Exception as e:
    check("D-2 v1路由导入", False, str(e))

try:
    from main import app
    from fastapi.testclient import TestClient
    _ = TestClient(app)
    check("D-3 FastAPI app 启动", True)
except Exception as e:
    check("D-3 FastAPI app 启动", False, str(e))

# ============================================================
# 总结
# ============================================================
print(f"\n{'='*60}")
print(f"结果: {p} 通过, {f} 失败")
print(f"{'='*60}")
if errors:
    for e in errors:
        print(f"  ❌ {e}")
else:
    print("\n✅ 所有 Bug 重检通过！")
sys.exit(0 if f == 0 else 1)
