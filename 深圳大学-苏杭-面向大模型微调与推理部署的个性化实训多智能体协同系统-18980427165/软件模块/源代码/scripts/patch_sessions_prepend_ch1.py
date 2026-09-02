# -*- coding: utf-8 -*-
"""对已有的 Linux 学习目标（session）原地补齐第一章序章 content_blocks。

不删除旧学习目标、不跑 LLM，不触发重生成：
  - 从 DB 读 LOCAL-KBS-01-04-S01 / S02 两个新增 chunk 的内容；
  - 按 existing content_blocks 的同样 schema（dict）构造 2 个序章块；
  - 理论版按 theory 口吻、实操版按 practice 口吻，对原文稍作"表达层"修改
    （不调用 LLM，纯 Python 规则改写，保证理论/实操差异）；
  - 插入到 state.learning_package.content_blocks 最前面；
  - 重写 block_index；
  - 保存回 session_states，用户刷新前端立刻看到第一章。
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text  # noqa: E402

engine = create_engine(f"sqlite:///{DB_PATH}")


def _theory_rewrite(src_text: str, heading_path: str) -> str:
    """理论口吻改写：开头用"设计动机/定位"，追加思考题。
    src_text 是 seed 内容，用轻量规则替换，不调 LLM。
    """
    leaf = heading_path.split(">")[-1].strip()
    # 去掉开头的 "##/###"（外层会补标题）
    t = re.sub(r"\A\s*#{1,3}\s*[^\n]*\n+", "", src_text)
    # 把"你可以把它理解成 / 为什么我们这门课 / 一个最朴素的直觉"这类实操口语 → 理论化
    t = t.replace("你可以把它理解成：", "从工程定位上讲，它承担的角色是：")
    t = t.replace(
        "为什么我们这门课把它放在\"命令行操作\"的学习目标里？",
        "**为什么在大模型开发栈里，它被列为第一个要掌握的运维观测工具？**",
    )
    t = t.replace(
        "这是 AI 开发 / 运维场景里的\"通行问候语\"。",
        "它是分布式训练 / 推理服务排障流程中的「事实入口」——任何资源问题的第一手观测，几乎都来自它。",
    )
    t = t.replace("一个最朴素的直觉：任何涉及",
                  "**核心观测场景（何时调用）**：在以下场景中，它是必须优先被触发的观测命令：")
    t = t.replace("显卡是不是被占满了 / ", "• 显存占用分析（是否 OOM 前兆）——")
    t = t.replace("训练为什么这么慢 / ", "• 训练吞吐诊断（GPU 利用率是否异常低）——")
    t = t.replace("为什么报 OOM / ", "• 内存异常定位——")
    t = t.replace("有没有真的在用 GPU 跑", "• 模型是否正确挂载到 GPU 设备——")
    t = t.replace("的问题，第一步都是敲一下它。", "；上述场景的标准第一步都是调用本工具。")

    # 针对 1.2 准入条件 chunk → 理论化：把"情况A/B/C"改成"输出语义分类"
    if "情况 A（正常）" in t:
        t = t.replace("会出现三种情况：", "按输出语义可以把结果分成三类：")
        t = t.replace("情况 A（正常）", "**A. 正常工作态**")
        t = t.replace("情况 B（驱动未装 / 显卡不识别）", "**B. 驱动/内核侧故障**")
        t = t.replace("情况 C（无 GPU 环境）", "**C. 不具备 GPU 资源的环境**")
        # 追加 1-2 条思考题（答案在原文）
        t += "\n\n**思考题**\n\n"
        t += "1. 为什么 `nvidia-smi` 能做到\"零配置\"，不需要额外安装任何包？（提示：它随哪个组件一起安装？）\n"
        t += "2. 假设你在 WSL2 下执行 `nvidia-smi` 提示找不到，但 Windows 侧能跑通。请结合「准入条件」说明根因大概在哪一层？\n"
        t += "\n**概念对比**：`nvidia-smi` vs Windows 任务管理器 GPU 面板——触发介质不同（命令行 vs 图形界面），但信息粒度本质一致；前者支持脚本化与日志导出，后者适合人工浏览。\n"
    else:
        # 1.1 的结尾追加思考题和概念对比
        t += "\n\n**思考题**\n\n"
        t += "1. `nvidia-smi` 的全称是什么？它要解决的核心问题用原文中的哪两个字可以概括？\n"
        t += "2. 为什么说它是\"零配置\"工具，而第三方 GPU 监控工具通常做不到？\n"
        t += "\n**概念对比**：任务管理器（CPU 观测） vs nvidia-smi（GPU 观测）——二者的「系统级快照」角色相同，只是观测的硬件子系统不同；前者随操作系统自带，后者随 GPU 驱动自带。\n"
    return t.strip()


def _practice_rewrite(src_text: str, heading_path: str) -> str:
    """实操口吻改写：保留"先做再解释"风格，命令紧跟验证方式。"""
    t = re.sub(r"\A\s*#{1,3}\s*[^\n]*\n+", "", src_text)
    # 1.1：中文定义 → 开门见山，命令先行
    if "1.1 nvidia-smi 是什么" in heading_path:
        lead = (
            "**一句话记住它**：`nvidia-smi` 就是 NVIDIA 官方给你在终端里看 GPU 状态的命令。\n\n"
            "**先敲再解释**：不管你现在懂不懂，先打开终端执行下面这行（后面再解释输出）：\n\n"
            "```bash\nnvidia-smi\n```\n\n"
            "**验证方式**：终端里出现一张大表格、表头包含 \"NVIDIA-SMI / Driver Version / CUDA Version\"，就 OK。\n\n"
        )
        # 把原定义里的"你可以把它理解成…"压缩一下当补充
        compressed = re.sub(
            r"你可以把它理解成：.*?通行问候语\"。",
            "简单说：GPU 版本的\"任务管理器\"，随驱动装好自带——训练慢 / 显存爆 / 报错 OOM，第一步都是截图发这个输出。",
            t, count=1, flags=re.S,
        )
        t = lead + compressed + (
            "\n\n**马上做的小练习**：\n"
            "1. 终端跑 nvidia-smi；\n"
            "2. 在输出里找到这三个字段：Driver Version / CUDA Version / Memory-Usage，"
            "用一句话记下你自己机器的数值。\n"
        )
    # 1.2：准入条件 → 操作清单 + 决策树
    elif "1.2 使用前的快速检查" in heading_path:
        lead = (
            "**先做 30 秒准入自查**，否则后面所有命令你都白敲：\n\n"
            "✅ 前提 1：机器必须是 **NVIDIA 显卡**（AMD / 核显 / Mac 不支持）\n"
            "✅ 前提 2：显卡 **驱动已装好**（不是 CUDA Toolkit，是驱动本身）\n"
            "✅ 前提 3：云服务器 / WSL2 / 虚拟机环境要做好 GPU 透传\n\n"
        )
        # 情况 A/B/C → 做成 3 条命令与判定输出
        t = re.sub(
            r"\*\*验证步骤（①）\*\*：打开终端，敲：.*?\*\*会出现三种情况\*\*：",
            "**验证步骤（就一条命令）**：\n\n```bash\nnvidia-smi\n```\n\n**看输出对照下表决策**：",
            t, count=1, flags=re.S,
        )
        t = t.replace("情况 A（正常）", "✅ **情况 A · 正常**：直接进入下一节开始学")
        t = t.replace("情况 B（驱动未装 / 显卡不识别）", "❌ **情况 B · 驱动挂了**：先退回去装驱动，装完回来")
        t = t.replace("情况 C（无 GPU 环境）", "⚠️  **情况 C · 无 GPU**：后面内容先当文档读，等有 GPU 再跑")
        t = lead + t.rstrip() + (
            "\n\n**WSL2 用户额外检查（1 条）**：\n"
            "```powershell\n# Windows PowerShell 里也跑一次，两边都能通才正常\nnvidia-smi\n```\n"
        )
    return t.strip()


def _clone_template(first_cb: dict, new_chunk_id: str, heading_path: str, content: str,
                    new_block_idx: int, session_id: str, stage: dict) -> dict:
    """按现有 content_blocks 的 schema 构造一个新 dict。"""
    import copy
    tmpl = copy.deepcopy(first_cb)
    # 核心字段覆盖
    tmpl["content_block_id"] = f"cb_{session_id[:8]}_{new_chunk_id}"
    tmpl["source_chunk_id"] = new_chunk_id
    tmpl["heading_path"] = heading_path
    tmpl["block_index"] = new_block_idx
    tmpl["personalized_content"] = content
    tmpl["content_version"] = 1
    # teaching_tasks：保持类型（list of dict），放一个符合画像的轻量动作
    ts = tmpl.get("teaching_tasks") or []
    if isinstance(ts, list) and ts:
        ts[0]["action"] = ts[0].get("action") or "anchor"
        ts[0]["markdown"] = ts[0].get("markdown") or "完成序章快速认知准入"
    tmpl["teaching_tasks"] = ts or [{"action": "anchor", "reason": "序章定位与准入", "markdown": "快速建立工具直觉"}]
    # personalization_meta / meta：保留
    return tmpl


def main():
    with engine.connect() as conn:
        # 1. 取序章 chunk 原文
        s01 = conn.execute(text("""
            SELECT chunk_id, heading_path, content FROM knowledge_chunks WHERE chunk_id='LOCAL-KBS-01-04-S01'
        """)).first()
        s02 = conn.execute(text("""
            SELECT chunk_id, heading_path, content FROM knowledge_chunks WHERE chunk_id='LOCAL-KBS-01-04-S02'
        """)).first()
        if not s01 or not s02:
            print("[ERR] 序章 chunks 未插入 DB，先跑 patch_kbs0104_ch1.py"); return 2

        # 2. 遍历 Linux sessions
        sids = [r[0] for r in conn.execute(
            text("SELECT session_id FROM sessions WHERE LOWER(goal) LIKE '%linux%'")
        ).fetchall()]
        if not sids:
            print("[SKIP] 无 Linux 学习目标"); return 0

        backups = {}
        updated = 0
        for sid in sids:
            row = conn.execute(text("SELECT state_json FROM session_states WHERE session_id=:sid"), {"sid": sid}).first()
            state = json.loads(row[0])
            lp = state.get("learning_package") or {}
            cbs = lp.get("content_blocks") or []
            if not isinstance(cbs, list) or not cbs:
                print(f"[{sid[:12]}] content_blocks 空，跳过"); continue

            # 判画像：theory or practice
            ls = ((state.get("profile") or {}).get("learning_style") or "practice").lower()
            rewrite_fn = _theory_rewrite if ls.startswith("theory") else _practice_rewrite

            # 如果已经有第一块是 "1.1 nvidia-smi 是什么"，说明已经补过，跳过
            first_leaf = ""
            if isinstance(cbs[0], dict):
                hp0 = cbs[0].get("heading_path") or ""
                first_leaf = (hp0.split(">")[-1].strip() if ">" in hp0 else hp0)
            if "1.1 nvidia-smi 是什么" in first_leaf:
                print(f"[{sid[:12]}] 序章已存在，跳过（style={ls})"); continue

            # 构造两个新块
            stage = (state.get("stage_info") or {})
            p0 = rewrite_fn(s01[2], s01[1])
            p1 = rewrite_fn(s02[2], s02[1])
            cb0 = _clone_template(cbs[0], s01[0], s01[1], p0, -2, sid, stage)
            cb1 = _clone_template(cbs[0], s02[0], s02[1], p1, -1, sid, stage)
            # 原有块 block_index 依次 +2，保持顺序稳定
            for i, b in enumerate(cbs):
                if isinstance(b, dict):
                    b["block_index"] = (b.get("block_index") or i) + 2
                else:
                    try: b.block_index = (getattr(b, "block_index") or i) + 2
                    except Exception: pass
            new_cbs = [cb0, cb1] + list(cbs)
            lp["content_blocks"] = new_cbs
            state["learning_package"] = lp

            # 备份 & 保存
            backups[sid] = {"old_state_json": row[0], "old_cb_count": len(cbs), "new_cb_count": len(new_cbs)}
            conn.execute(text("""
                UPDATE session_states SET state_json = :sj, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = :sid
            """), {"sj": json.dumps(state, ensure_ascii=False), "sid": sid})
            updated += 1
            print(f"[{sid[:12]}] style={ls}  已补序章 2 块（cb: {len(cbs)} → {len(new_cbs)}）")

        conn.commit()

    # 写出备份
    if backups:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bk = PROJECT_ROOT / "tmp" / f"backup_prepatch_ch1_{ts}.json"
        bk.parent.mkdir(exist_ok=True)
        with open(bk, "w", encoding="utf-8") as f:
            json.dump(backups, f, ensure_ascii=False, indent=2)
        print(f"备份：{bk}")
    print(f"原地补序章完成：共 {updated} 个 session。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
