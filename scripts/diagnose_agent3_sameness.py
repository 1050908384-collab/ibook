"""=====================================
诊断脚本：根因排查「practice vs theory 生成完全相同」
=====================================
"""
from __future__ import annotations
import asyncio
import difflib
import json
import os
import sys
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

from config import settings  # noqa: E402
from models.profile import (  # noqa: E402
    UserProfile, PracticeProfile, normalize_learning_style,
    LEARNING_STYLE_PRACTICE, LEARNING_STYLE_THEORY,
)
# StageInfo 实际定义在 models/stage.py，从那里导入以避免 contract 版本差异。
from models.stage import StageInfo  # noqa: E402,F401
from db.sqlite_client import SQLiteClient as DatabaseClient  # noqa: E402


SOURCE_TEXT_SAMPLE = """## Linux 文件权限基础
在 Linux 中，每一个文件都有一组「权限位」用于控制谁能读、写、执行该文件。
常见的权限分为 3 类主体：文件所有者（Owner）、用户组（Group）、其他用户（Others）。
- **读（r）**：可以查看文件内容；
- **写（w）**：可以修改文件内容；
- **执行（x）**：可以把文件作为程序运行。

使用下面的命令查看当前目录下文件的权限：

```bash
ls -l
```

**典型输出字段说明：**
-rw-r--r-- 1 alice dev  4096 Aug 20 10:00 readme.md
drwxr-xr-x 2 alice dev  4096 Aug 20 10:05 project/

- 第 1 个字符 `-` 表示普通文件，`d` 表示目录；
- 后面 9 位每 3 位一组，依次是 Owner / Group / Others 的 rwx 权限；
- `alice` 是 Owner，`dev` 是 Group，`4096` 是字节数。

### 常见报错与排查
1. `Permission denied`：通常是你没有对应操作所需的权限位；
   解决：`chmod` 赋予对应权限，或切换到有权限的用户。
2. `Operation not permitted`：即使 Owner 是你自己，写系统目录 /tmp 之外也会遇到；
   解决：加 `sudo`，或改写到你自己的 `$HOME` 目录下。
3. `Is a directory`：你尝试用 `cat` 等文本命令去读一个目录；
   解决：用 `ls` / `cd` 代替。
"""


def char_sim(a: str, b: str) -> float:
    a_clean = re.sub(r"\s+", "", a or "")
    b_clean = re.sub(r"\s+", "", b or "")
    if not a_clean or not b_clean:
        return 0.0 if (a_clean or b_clean) else 1.0
    return difflib.SequenceMatcher(None, a_clean, b_clean).ratio()


def heading_markers(text: str) -> list[str]:
    text = text or ""
    return [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("#")][:6]


def has_three_headers(text: str) -> dict[str, bool]:
    return {
        "P_立即动手": ("立即动手" in (text or "")),
        "P_动手验证": ("动手验证" in (text or "")),
        "P_报错登记":   ("报错登记" in (text or "")),
        "T_概念与设计": ("概念与设计动机" in (text or "")),
        "T_思考提示":   ("思考提示" in (text or "")),
        "T_思考题":     ("思考题" in (text or "")),
        "T_术语自证":   ("术语自证" in (text or "")),
    }


def step0_db_snapshot() -> dict:
    print("=" * 72)
    print("[Step 0a] 先读 SQLite 实际表结构 → 再查会话/包/块")
    print("=" * 72)
    db = DatabaseClient(settings.DATABASE_URL)
    snapshot = {"sessions": [], "blocks_for_sessions": [], "linux_blocks": [], "schema": {}}
    try:
        with db.engine.connect() as conn:
            from sqlalchemy import text

            def cols(table: str) -> list[str]:
                try:
                    return [r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()]
                except Exception:
                    return []

            for t in ["sessions", "user_profiles", "learning_packages",
                      "content_blocks", "source_chunks", "knowledge_chunks",
                      "learner_profiles", "session_states"]:
                snapshot["schema"][t] = cols(t)
                print(f"  [schema] {t}: {snapshot['schema'][t]}")

            cs = snapshot["schema"]

            # 最近 sessions + user_profiles
            print("\n最近 10 个会话（sessions 表）：")
            try:
                rows = conn.execute(text(
                    "SELECT session_id, user_id, goal, current_stage, status, created_at"
                    "  FROM sessions ORDER BY created_at DESC LIMIT 10"
                )).mappings().fetchall()
            except Exception as exc:
                rows = []
                print(f"  SELECT sessions 失败: {exc}")
            for r in rows:
                rd = dict(r)
                sid = str(rd.get("session_id") or "")
                uid = str(rd.get("user_id") or "")
                style = "N/A"
                grade = ""
                skill = []
                # learner_profiles（user_id 级默认画像 + user_profiles（session 级））
                for tbl, where, key in (
                    ("user_profiles",
                     "session_id = :sid ORDER BY version DESC, created_at DESC LIMIT 1", None),
                    ("learner_profiles",
                     "user_id = :uid ORDER BY version DESC, updated_at DESC LIMIT 1", "uid"),
                ):
                    if tbl not in cs or not cs[tbl]:
                        continue
                    kwargs = {"sid": sid} if key is None else {"uid": uid}
                    try:
                        pf = conn.execute(text(f"SELECT * FROM {tbl} WHERE {where}"), kwargs).mappings().fetchone()
                    except Exception:
                        pf = None
                    if not pf:
                        continue
                    pd = dict(pf)
                    if "learning_style" in pd and pd.get("learning_style"):
                        style = str(pd["learning_style"]).strip() or style
                    if "grade" in pd and pd.get("grade"):
                        grade = str(pd["grade"]).strip() or grade
                    for key2 in ("skill_tags", "skills"):
                        if key2 in pd and pd.get(key2):
                            raw = pd[key2]
                            try:
                                val = json.loads(raw) if isinstance(raw, str) else list(raw or [])
                            except Exception:
                                val = str(raw)
                            skill = skill or val
                            break
                    if style != "N/A":
                        break
                print(f"  SESSION {sid[:16]}…  user={uid[:10]}  goal={str(rd.get('goal',''))[:30]!r} "
                      f"stage={rd.get('current_stage')} status={rd.get('status')} created={rd.get('created_at')}")
                print(f"    → learning_style={style!r}  grade={grade!r}  skill_tags={skill}")
                snapshot["sessions"].append({
                    "session_id": sid, "user_id": uid, "goal": rd.get("goal"),
                    "learning_style": style, "grade": grade, "skill_tags": skill,
                })

            # session_states 表可能存了 learning_package
            if "session_states" in cs and cs["session_states"]:
                print(f"\nsession_states 表存在！列：{cs['session_states']}")
                # 只查最近 2 个会话的 stage 状态
                for sess in snapshot["sessions"][:2]:
                    sid = sess["session_id"]
                    try:
                        st_rows = conn.execute(text(
                            "SELECT session_id, stage, state_json, updated_at"
                            "  FROM session_states WHERE session_id = :sid ORDER BY stage"
                        ), {"sid": sid}).mappings().fetchall()
                    except Exception as exc:
                        print(f"  查询 session_states 失败: {exc}")
                        st_rows = []
                    print(f"  SESSION {sid[:16]}…  session_states rows = {len(st_rows)}")
                    # state_json 里检查有没有 learning_package / content_blocks
                    for st in st_rows:
                        st = dict(st)
                        raw = st.get("state_json") or "{}"
                        try:
                            sj = json.loads(raw) if isinstance(raw, str) else (raw or {})
                        except Exception:
                            continue
                        tutorial = ""
                        if isinstance(sj, dict):
                            if "tutorial" in sj:
                                tutorial = str(sj["tutorial"])
                            for k, v in sj.items():
                                if isinstance(v, dict) and "tutorial" in v and not tutorial:
                                    tutorial = str(v["tutorial"])
                                if isinstance(v, dict) and "learning_package" in v:
                                    lp = v["learning_package"]
                                    if isinstance(lp, dict) and "tutorial" in lp:
                                        tutorial = str(lp["tutorial"])
                        if not tutorial:
                            continue
                        hm = has_three_headers(tutorial)
                        print(f"    stage={st.get('stage')} state_json 中 tutorial 三骨架标记: {hm}")
                        if not any(hm.values()):
                            print(f"    ❌ tutorial 全 False！前 300 字: {tutorial[:300].replace(chr(10),' / ')!r}")

            # 最近 content_blocks
            if "content_blocks" in cs and cs["content_blocks"]:
                try:
                    blk_rows = conn.execute(text(
                        "SELECT content_block_id, package_id, session_id, source_chunk_id,"
                        "       block_name, personalization_action, personalized_content,"
                        "       personalization_meta, created_at"
                        "  FROM content_blocks ORDER BY created_at DESC LIMIT 10"
                    )).mappings().fetchall()
                except Exception as exc:
                    print(f"\n  content_blocks 查询失败: {exc}")
                    blk_rows = []
                print(f"\n最近 content_blocks（实际个性化落库结果，共 {len(blk_rows)} 条）：")
                for b in blk_rows:
                    b = dict(b)
                    pc = b.get("personalized_content") or ""
                    meta_raw = b.get("personalization_meta") or "{}"
                    try:
                        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                    except Exception:
                        meta = {}
                    hm = has_three_headers(pc)
                    print(f"  BLK {str(b.get('content_block_id',''))[:16]}…  "
                          f"block={str(b.get('block_name',''))[:40]!r} "
                          f"path={meta.get('path')!r} fail_reason={meta.get('llm_fail_reason')!r}")
                    print(f"    → 三板块骨架标记: {hm}")
                    snapshot["blocks_for_sessions"].append({
                        "content_block_id": b.get("content_block_id"),
                        "block_name": b.get("block_name"),
                        "path": meta.get("path"),
                        "markers": hm,
                        "personalized_content": pc,
                    })
            else:
                print("\n  content_blocks 表不存在 → BLOCK_PIPELINE 的块结果要么走了 session_states.state_json，")
                print("    要么走了 tmp/sessionX_learning_package.json 文件。")

            # Linux 候选块
            if "knowledge_chunks" in cs and cs["knowledge_chunks"]:
                table = "knowledge_chunks"
            elif "source_chunks" in cs and cs["source_chunks"]:
                table = "source_chunks"
            else:
                table = ""
            if table:
                try:
                    sc_rows = conn.execute(text(
                        f"SELECT chunk_id, heading_path, content FROM {table}"
                        f" WHERE content LIKE :p1 OR heading_path LIKE :p2 OR content LIKE :p3"
                        f" LIMIT 5"
                    ), {"p1": "%Linux%", "p2": "%Linux%", "p3": "%ls -l%"}).mappings().fetchall()
                except Exception as exc:
                    print(f"\n  {table} 查询失败: {exc}")
                    sc_rows = []
                print(f"\nDB 中 Linux 候选块（{table}）: {len(sc_rows)} 条")
                for sc in sc_rows:
                    sc = dict(sc)
                    content = sc.get("content") or ""
                    print(f"  CHUNK {sc.get('chunk_id')}  heading_path={str(sc.get('heading_path',''))[:60]!r}  chars={len(content)}")
                if sc_rows:
                    sc = dict(sc_rows[0])
                    snapshot["linux_blocks"] = [{
                        "chunk_id": sc.get("chunk_id"),
                        "heading_path": sc.get("heading_path") or "Linux 示例块",
                        "content": sc.get("content") or "",
                    }]
    finally:
        db.close()

    # =============== Step 0b: 查 tmp/*.json 临时文件（用户的 learning_package 很可能在这里）===============
    print("\n" + "=" * 72)
    print("[Step 0b] 查 tmp/ 下的 session*_profile.json / *_learning_package.json")
    print("=" * 72)
    tmp_dir = os.path.join(ROOT, "tmp")
    if os.path.isdir(tmp_dir):
        for pat, label in (("session*_profile.json", "PROFILE"),
                           ("session*_learning_package.json", "LEARNING_PACKAGE")):
            import glob
            files = sorted(glob.glob(os.path.join(tmp_dir, pat)))
            print(f"\n  {label} 文件数：{len(files)}")
            for fp in files[:4]:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                except Exception as exc:
                    print(f"    {os.path.basename(fp)}: 读失败 {exc}")
                    continue
                base = os.path.basename(fp)
                if label == "PROFILE":
                    style = str(obj.get("learning_style") if isinstance(obj, dict) else "N/A") or "N/A"
                    grade = str(obj.get("grade") if isinstance(obj, dict) else "")
                    skill = obj.get("skill_tags") if isinstance(obj, dict) else []
                    print(f"    {base}: learning_style={style!r} grade={grade!r} skill_tags={skill}")
                else:
                    # LEARNING PACKAGE — 直接抓 tutorial/content_blocks
                    if not isinstance(obj, dict):
                        continue
                    tut = str(obj.get("tutorial") or "")
                    blocks = obj.get("content_blocks") or obj.get("blocks") or []
                    hm_tut = has_three_headers(tut)
                    blk_hm_any = False
                    for blk in blocks:
                        if isinstance(blk, dict):
                            pc = str(blk.get("personalized_content") or blk.get("content") or "")
                            if any(has_three_headers(pc).values()):
                                blk_hm_any = True
                                break
                    print(f"    {base}: tutorial 三骨架={hm_tut}  "
                          f"blocks数={len(blocks)}  块内任一含骨架={'✅' if blk_hm_any else '❌'}")
                    if not any(hm_tut.values()) and not blk_hm_any:
                        print(f"      ❌ 完全无三骨架 → 用户前端看到的「相同内容」很可能就是这个文件。")
                        print(f"      tutorial 前 280 字: {tut[:280].replace(chr(10),' / ')!r}")
    else:
        print(f"  tmp 目录不存在：{tmp_dir}")

    return snapshot


async def step1_run_hybrid_directly(linux_chunk: dict | None) -> dict:
    print()
    print("=" * 72)
    print("[Step 1] 直接 new HybridContentAgent 跑同一 source_chunk × 两种画像")
    print("=" * 72)
    from agents.deterministic import HybridContentAgent
    agent = HybridContentAgent(database_url=settings.DATABASE_URL)

    if linux_chunk:
        source_chunk = {"chunk_id": linux_chunk.get("chunk_id"),
                        "heading_path": linux_chunk.get("heading_path"),
                        "content": linux_chunk.get("content")}
    else:
        source_chunk = {"chunk_id": "DIAG-SAMPLE-1",
                        "heading_path": "Linux 文件权限基础",
                        "content": SOURCE_TEXT_SAMPLE}
    heading = source_chunk.get("heading_path") or "Linux 示例块"
    stage = StageInfo(stage_id="DIAG-STAGE-1", stage_num=1,
                      block_id="DIAG-BLOCK-1", block_name=heading,
                      subtopic_name=heading, subtopic_id="diag-sub",
                      subtopic_ids=["diag-sub"], subtopic_names=[heading],
                      block_ids=["DIAG-BLOCK-1"],
                      gap_skill="Linux 基础操作", prerequisites=[])

    results = {}
    for style in (LEARNING_STYLE_PRACTICE, LEARNING_STYLE_THEORY):
        print(f"\n── style={style!r} ──")
        profile = UserProfile(session_id="DIAG-SESSION-1",
                              user_id="diag_user_001", learning_style=style,
                              grade="本科二年级",
                              skill_tags=(["Python", "基础算法"] if style == LEARNING_STYLE_THEORY else []))
        practice = PracticeProfile(hardware="Windows 11 + WSL2 Ubuntu 22.04 + Python 3.11",
                                   device_type="laptop", gpu_model="", vram_gb=0)
        block = await agent.generate_block(
            stage=stage, source_chunk=source_chunk, profile=profile, practice=practice,
            block_index=0, session_id="DIAG-SESSION-1", learning_evidence=None,
        )
        meta = getattr(block, "personalization_meta", None)
        path = getattr(meta, "path", None)
        fail = getattr(meta, "llm_fail_reason", None)
        pc = block.personalized_content or ""
        print(f"  personalized_content chars = {len(pc)}")
        print(f"  path={path!r}  fail_reason={fail!r}")
        print(f"  顶部 headings = {heading_markers(pc)}")
        hm = has_three_headers(pc)
        print(f"  三板块骨架标记 = {hm}")
        results[style] = {
            "personalized_content": pc, "path": path, "fail_reason": fail, "markers": hm,
        }

    a = results[LEARNING_STYLE_PRACTICE]["personalized_content"]
    b = results[LEARNING_STYLE_THEORY]["personalized_content"]
    sim = char_sim(a, b)
    pm_old = results[LEARNING_STYLE_PRACTICE]["markers"]
    tm_old = results[LEARNING_STYLE_THEORY]["markers"]
    # 用户明确要求：不再强制 3 板块外壳（立即动手/概念与设计动机 + 表格 + 尾检查），只要
    # 块级有表达层个性化即可，所以：
    #   - 旧外壳标记：practice/theory 都应该是 False（除了理论版追加思考题是自然表达，不算外壳）
    #   - 新表达层标记：practice 有『先别急/概念摘要一句话版/我什么时候需要它/操作步骤/验证方式』
    #     ≥ 2 个就算 practice OK；theory 有『设计动机/为什么/概念对比/思考题』≥ 2 个就算 theory OK。
    def _exp_markers(text: str, style: str) -> Dict[str, bool]:
        s = text or ""
        if style == LEARNING_STYLE_PRACTICE:
            return {
                "实操口吻(先别急/一句话版/我什么时候需要它)": any(w in s for w in ("先别急", "一句话版", "我什么时候需要", "概念摘要")),
                "操作步骤+本机动手/验证方式": any(w in s for w in ("验证方式", "执行完", "你的 Windows", "WSL", "打开你的", "Win 键", "下一步动手", "接下来动手")),
                "代码块或命令行出现": (s.count("```") >= 2 or "`" in s and any(k in s for k in ("$", "wsl", "sudo", "apt", "cd ", "ls ", "Shell"))),
            }
        return {
            "理论展开(设计动机/关键原因/大部分服务器/生产环境)": any(w in s for w in ("设计动机", "关键原因", "绝大多数服务器", "生产环境", "底层管家", "trade-off", "架构")),
            "思考题或概念对比出现": ("思考题" in s or any(w in s for w in ("vs ", "对比", "而不是", "A 与 B", "和...区别"))),
            "术语解释更充分（解释/含义/是什么 + 不是/为什么）": any(w in s for w in ("缩写", "意思是", "为什么偏偏", "为什么学编程到后面", "LTS", "长期支持")),
        }
    pm = _exp_markers(a, LEARNING_STYLE_PRACTICE)
    tm = _exp_markers(b, LEARNING_STYLE_THEORY)
    practice_ok = sum(1 for v in pm.values() if v) >= 2 and not (pm_old["P_立即动手"] and pm_old["P_动手验证"])  # 表达层≥2 + 外壳不再强出现
    theory_ok   = sum(1 for v in tm.values() if v) >= 2 and not (tm_old["T_概念与设计"] and tm_old["T_思考提示"])  # 表达层≥2 + 外壳不再强出现
    print(f"\n── practice vs theory 对比结论 ──")
    print(f"  字符相似度 = {sim:.4f}  （目标 ≤ 0.70；≥ 0.90 = 假个性化）")
    print(f"  practice 旧外壳标记（应基本为 False）= {pm_old}")
    print(f"  theory   旧外壳标记（应基本为 False）= {tm_old}")
    print(f"  practice 新表达层标记 ≥ 2/3？ {'✅' if practice_ok else '❌'}  {pm}")
    print(f"  theory   新表达层标记 ≥ 2/3？ {'✅' if theory_ok else '❌'}  {tm}")

    # 用 DB 实际 content_blocks 做对比：如果 DB 里最近两块 markers 全 False，而脚本生成有差异
    # → 则根因 = 用户 DB 中是旧生成结果 / 前端展示的是 tutorial（整篇）而不是 content_blocks。
    results["__meta__"] = {
        "char_similarity": sim, "practice_ok": practice_ok, "theory_ok": theory_ok,
    }
    return results


def step2_diff(results: dict) -> None:
    meta = results["__meta__"]
    if meta["char_similarity"] <= 0.70 and meta["practice_ok"] and meta["theory_ok"]:
        return
    print()
    print("=" * 72)
    print("[Step 2] personalized_content unified diff（前 120 行差异）")
    print("=" * 72)
    a = results[LEARNING_STYLE_PRACTICE]["personalized_content"].splitlines(keepends=True)
    b = results[LEARNING_STYLE_THEORY]["personalized_content"].splitlines(keepends=True)
    diff = list(difflib.unified_diff(a, b, fromfile=f"practice lines={len(a)}",
                                     tofile=f"theory lines={len(b)}", n=0))
    print("".join(diff[:120]))


def main() -> int:
    print()
    print("=" * 72)
    print("DIAGNOSE: practice vs theory『生成完全相同』根因排查")
    print("ROOT=", ROOT)
    print("DB=", settings.DATABASE_URL)
    print(f"EXECUTION_MODE={settings.EXECUTION_MODE}  BLOCK_PIPELINE_ENABLED={settings.BLOCK_PIPELINE_ENABLED}")
    print(f"LLM_PROVIDER={settings.LLM_PROVIDER}  FALLBACK={settings.LLM_FALLBACK_PROVIDER}")
    print("=" * 72)
    snap = step0_db_snapshot()
    linux = (snap["linux_blocks"][0] if snap["linux_blocks"] else None)
    results = asyncio.run(step1_run_hybrid_directly(linux))
    step2_diff(results)
    meta = results["__meta__"]

    print()
    print("=" * 72)
    print("诊断结论：")
    print(f"  代码端 HybridContentAgent 字符相似度 = {meta['char_similarity']:.4f}  "
          f"（目标 ≤ 0.70；≥ 0.90 = 代码端仍未生效）")
    print(f"  代码端 practice 新表达层个性化 ≥ 2/3 = {'✅' if meta['practice_ok'] else '❌'}")
    print(f"  代码端 theory   新表达层个性化 ≥ 2/3 = {'✅' if meta['theory_ok'] else '❌'}")

    db_blocks = snap.get("blocks_for_sessions", [])
    if db_blocks:
        ok_count = sum(1 for b in db_blocks if any(b["markers"].values()))
        print(f"  DB 最近 content_blocks 新骨架落地数：{ok_count} / {len(db_blocks)}  "
              f"（全 0 = 用户看到的是旧学习包/个性化代码没走）")
    print()
    if meta["char_similarity"] <= 0.70 and meta["practice_ok"] and meta["theory_ok"]:
        print("【分支 A】代码端没问题：HybridContentAgent 的表达层个性化确实生效，且强制 3 外壳不再出现。")
        if db_blocks and all(not any(b["markers"].values()) for b in db_blocks):
            print("  → 用户当前 DB 中所有学习包都是修改代码之前就生成的旧数据！")
            print("    解决方法：前端重新生成学习包（start_stage / 重新开始学习），或者清 DB 后新建会话。")
        else:
            print("  → 如果前端仍显示一模一样 = 请检查 GET 学习包接口是否仍返回被 backfill_stage_tutorial 覆盖")
            print("    后的旧 tutorial 正文；本次 Fix-A（Guard1 + 放宽 ## 标题匹配）已解决此问题。")
    else:
        print("【分支 B】代码端仍未差异化：需要继续在 HybridContentAgent.generate_block() 里定位具体 bug。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
