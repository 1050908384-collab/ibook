"""INVESTIGATE - ONLY READ, NO MODIFY

只用项目真实数据（project.db + tmp/），不构造、不改代码，抓三件事：

STEP 1：所有 sessions 的画像真实来源
  - state JSON 里的 profile.learning_style（内存态/重启恢复态）
  - user_profiles（会话级表）最新 version 的 learning_style / grade / skill_tags
  - learner_profiles（全局 user 级）learning_style / grade / skill_tags
  打印出来一眼就能看出 learning_style 是不是没写进会话级 / 是不是被全局覆盖

STEP 2：真实 learning_package
  - state JSON 中 learning_package.tutorial 和 content_blocks 个性化标记 + 相似度
  - stage_snapshots（数字键/stage_id 键）最近一版同上
  - tmp/session*_learning_package.json 同上
  打印"一模一样"的具体发生位置（包 vs 包 / tutorial vs tutorial / blocks vs blocks）

STEP 3：任选两个 goal 含 Linux 或 LoRA/vLLM 的 session，重走 _require_profile → Harness 内部 profile，打印真实 learning_style 给 start_stage 时的值。不启动 HTTP，不生成学习包（省 token）。
"""
from __future__ import annotations
import json
import os
import re
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from db.sqlite_client import SQLiteClient  # noqa: E402


def char_sim(a: str, b: str, n: int = 3) -> float:
    def ngram(s: str):
        s = re.sub(r"\s+", "", s or "")
        if len(s) < n:
            return {s}
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    A, B = ngram(a), ngram(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def markers(content: str) -> Dict[str, bool]:
    s = content or ""
    return {
        "practice口吻(概念摘要/我怎么用/先别急/避坑/工作台)": any(
            w in s for w in ("概念摘要", "一句话版", "我什么时候需要", "我怎么用", "先别急", "避坑", "工作台", "黑底白字", "黑窗口")
        ),
        "theory口吻(内核/架构/设计动机/trade-off/思考题/为什么/三个层面)": any(
            w in s for w in ("内核", "架构", "设计动机", "trade-off", "思考题", "三个层面", "模块化", "发行版", "哲学")
        ),
        "代码块": s.count("```") >= 2,
    }


def dump_profile(tag: str, p: Dict[str, Any]) -> None:
    if not p:
        print(f"    [{tag}] <EMPTY>")
        return
    print(
        f"    [{tag}] learning_style={p.get('learning_style')!r}  "
        f"grade={p.get('grade')!r}  "
        f"daily_hours={p.get('daily_hours')}  "
        f"rhythm={p.get('learning_rhythm')!r}  "
        f"skill_tags={p.get('skill_tags')}  "
        f"goal={str(p.get('goal',''))[:40]!r}"
    )


def summarize_pkg(tag: str, pkg: Dict[str, Any]) -> Dict[str, Any]:
    tut = pkg.get("tutorial") or ""
    blocks = pkg.get("content_blocks") or []
    m = markers(tut)
    block_pc_list: List[str] = []
    block_any_personalized = False
    block_mark_summary = {"practice口吻": 0, "theory口吻": 0, "代码块": 0}
    for b in blocks:
        pc = b.get("personalized_content") or ""
        block_pc_list.append(pc[:120])
        bm = markers(pc)
        if bm["practice口吻(概念摘要/我怎么用/先别急/避坑/工作台)"]:
            block_mark_summary["practice口吻"] += 1
        if bm["theory口吻(内核/架构/设计动机/trade-off/思考题/为什么/三个层面)"]:
            block_mark_summary["theory口吻"] += 1
        if bm["代码块"]:
            block_mark_summary["代码块"] += 1
        if any(bm.values()):
            block_any_personalized = True
    print(f"    [{tag}] tutorial chars={len(tut)}")
    print(f"        tutorial markers={m}")
    print(f"        blocks={len(blocks)}  任一块含任一标记? {'✅' if block_any_personalized else '❌'}")
    print(f"        block累计标记: {block_mark_summary}")
    if tut:
        print(f"        tutorial 前 320 字: {tut[:320].replace(chr(10),' / ')}")
    return {
        "tutorial": tut, "blocks_len": len(blocks),
        "tutorial_markers": m, "block_mark_summary": block_mark_summary,
        "block_any_personalized": block_any_personalized,
    }


def main() -> int:
    db = SQLiteClient()
    try:
        with db.Session() as s:
            from sqlalchemy import text
            all_sessions = s.execute(text("""
                SELECT session_id, user_id, goal, status, current_stage, created_at
                  FROM sessions ORDER BY datetime(created_at) DESC
            """)).mappings().fetchall()
        print("=" * 80)
        print(f"[STEP 1] sessions 总数 = {len(all_sessions)}（按创建时间倒序）")
        print("=" * 80)
        # 打印全局 learner_profiles（每个 user_id 一行）
        user_ids = sorted({r["user_id"] for r in all_sessions})
        print("\n── 全局 learner_profiles 表（跨目标共享，不应该存 learning_style）──")
        for uid in user_ids:
            up = db.get_user_profile(uid) or {}
            dump_profile(f"GLOBAL user_id={uid[:14]}…", up)

        rows_for_check = []
        for r in all_sessions[:12]:
            sid = r["session_id"]
            print(f"\n[Session {r['session_id'][:10]}… status={r['status']} current_stage={r['current_stage']} goal={str(r.get('goal',''))[:60]!r}]")
            # 三个来源的画像：
            # (A) session_states.state_json 中的 state.profile
            state_payload = db.load_session_state(sid) or {}
            state_profile = (state_payload.get("profile") or {}) if isinstance(state_payload, dict) else {}
            dump_profile("A.state_json.profile", state_profile)
            # (B) user_profiles(session级) 最新 version 行
            with db.Session() as ss:
                from sqlalchemy import text as _t
                sp_rows = ss.execute(_t("""
                    SELECT * FROM user_profiles WHERE session_id = :sid ORDER BY version DESC LIMIT 1
                """), {"sid": sid}).mappings().fetchall()
            session_profile_latest = dict(sp_rows[0]) if sp_rows else {}
            if session_profile_latest:
                dump_profile(f"B.user_profiles(session级) v={session_profile_latest.get('version')}", session_profile_latest)
            else:
                print("    [B.user_profiles(session级)] 无行 ❌（会话专属 learning_style 根本没保存）")
            # (C) get_latest_profile（实际生产读取路径）
            composed = db.get_latest_profile(sid) or {}
            dump_profile("C.get_latest_profile(生产读取)", composed)

            # STEP 2: 真实 learning_package 特征
            # (1) state_json.learning_package
            state_pkg = (state_payload.get("learning_package") or {}) if isinstance(state_payload, dict) else {}
            # (2) stage_snapshots 里最新版本
            snapshots = (state_payload.get("stage_snapshots") or {}) if isinstance(state_payload, dict) else {}
            snapshot_pkg: Dict[str, Any] = {}
            snap_meta = ""
            if snapshots:
                keys = list(snapshots.keys())
                # 找一个含 learning_package 的最新 snapshot
                latest_snap: Dict[str, Any] = {}
                for k in reversed(keys):
                    v = snapshots.get(k) or {}
                    if isinstance(v, dict) and v.get("learning_package"):
                        latest_snap = v
                        snap_meta = f"snapshot_key={k!r} stage_info.stage_id={((v.get('stage_info') or {}).get('stage_id') or '')[:20]}"
                        snapshot_pkg = v["learning_package"] or {}
                        break
                if not latest_snap:
                    # 退而求其次取最近一个 key 的 learning_package
                    last = snapshots.get(keys[-1]) or {}
                    if isinstance(last, dict):
                        snapshot_pkg = last.get("learning_package") or {}
                        snap_meta = f"(fallback)snapshot_key={keys[-1]!r}"

            print(f"\n  ── STEP 2：真实 learning_package 特征 ──")
            s1 = summarize_pkg("state_json.learning_package", state_pkg)
            if snapshot_pkg:
                s2 = summarize_pkg(f"stage_snapshots {snap_meta}", snapshot_pkg)
            else:
                s2 = {"tutorial": "", "blocks_len": 0}
                print("    [stage_snapshots.learning_package] <EMPTY>")
            # 跨源相似度（同一个 session 内，state_json tutorial vs snapshot tutorial 是否相同？）
            if s1["tutorial"] and s2["tutorial"]:
                sim_same_sess = char_sim(s1["tutorial"], s2["tutorial"])
                print(f"    state_json.vs.snapshot.tutorial 字符相似度 = {sim_same_sess:.4f}  (>0.95=同一份快照)")

            rows_for_check.append({
                "session_id": sid,
                "goal": r.get("goal") or "",
                "status": r["status"],
                "A_style": state_profile.get("learning_style"),
                "B_style": session_profile_latest.get("learning_style"),
                "C_style": composed.get("learning_style"),
                "tutorial": s1["tutorial"] or s2["tutorial"],
                "blocks_len": s1["blocks_len"] or s2["blocks_len"],
            })

        # STEP 2b：如果有 ≥2 个 Linux 会话或 LoRA 会话，互相算相似度看是否真的一模一样
        print("\n" + "=" * 80)
        print("[STEP 2b] 会话之间 learning_package.tutorial 相似度（一模一样的直接定量证据）")
        print("=" * 80)
        target_rows = [r for r in rows_for_check if r["tutorial"]]
        for i in range(len(target_rows)):
            for j in range(i + 1, len(target_rows)):
                ra, rb = target_rows[i], target_rows[j]
                sim = char_sim(ra["tutorial"], rb["tutorial"])
                same_goal = (str(ra["goal"] or "")[:18] == str(rb["goal"] or "")[:18])
                flag = " ⚠️ SAME（一模一样或高度相同）" if same_goal and sim >= 0.85 else ""
                print(
                    f"  {ra['session_id'][:8]}…(goal={str(ra['goal'])[:25]!r} style={ra['C_style']}) "
                    f"vs {rb['session_id'][:8]}…(goal={str(rb['goal'])[:25]!r} style={rb['C_style']}) "
                    f"→ tutorial 字符相似度 = {sim:.4f}{flag}"
                )
                if flag:
                    # 打印两份 tutorial 前 240 字
                    print(f"     A 前 240 字: {ra['tutorial'][:240].replace(chr(10), ' / ')}")
                    print(f"     B 前 240 字: {rb['tutorial'][:240].replace(chr(10), ' / ')}")

        # STEP 3：真实读取 Harness._require_profile（不启动 HTTP、不调 LLM）
        print("\n" + "=" * 80)
        print("[STEP 3] Harness 内部 _require_profile 返回的 learning_style（start_stage 传给 Agent3 的源头）")
        print("=" * 80)
        import asyncio
        from orchestrator.state_manager import StateManager
        from models.profile import UserProfile
        sm = StateManager(db_client=db)
        for r in rows_for_check[:6]:
            sid = r["session_id"]
            state = sm._get(sid)
            profile_from_state = state.profile.model_dump(mode="json") if state and state.profile else None
            # 调生产读取函数 get_latest_profile 模拟 Harness._require_profile 的 fallback
            latest = db.get_latest_profile(sid) or {}
            print(f"\n  Session {sid[:10]}… status={r['status']} goal={str(r['goal'])[:50]!r}")
            print(f"    sm._get(state).profile exists? {'YES' if profile_from_state else 'NO'}")
            if profile_from_state:
                dump_profile("state.profile（start_stage 直接用这个）", profile_from_state)
            else:
                dump_profile("fallback → get_latest_profile()（实际会传入 Agent3）", latest)

        # STEP 4：tmp/ 下的旧快照（防止前端读的是这些旧文件）
        print("\n" + "=" * 80)
        print("[STEP 4] tmp/ 下 session*_profile.json / *_learning_package.json")
        print("=" * 80)
        TMP = os.path.join(ROOT, "tmp")
        if os.path.isdir(TMP):
            for name in sorted(os.listdir(TMP)):
                if not (name.endswith(".json") and ("profile" in name or "learning_package" in name)):
                    continue
                path = os.path.join(TMP, name)
                try:
                    data = json.load(open(path, "r", encoding="utf-8"))
                except Exception as e:
                    print(f"  {name}: <load_error {e}>")
                    continue
                if "profile" in name.lower():
                    dump_profile(name, data if isinstance(data, dict) else {})
                else:
                    summarize_pkg(name, data if isinstance(data, dict) else {})
        else:
            print("  tmp/ 目录不存在")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
