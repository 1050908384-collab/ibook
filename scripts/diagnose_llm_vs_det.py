# -*- coding: utf-8 -*-
"""快速诊断：当前 session 是调了 DeepSeek LLM（real personalized）
           还是走了 deterministic 纯规则回退（HybridContentAgent._deterministic 系列）？
方法：
1. 看块 personalized_content 里有没有 deterministic 规则硬编码的标志性字符串：
   - '# 画像专属 · …… · 本块学习卡'  ← H2 独立块的 deterministic 「头卡脚手架」
   - '## 立即动手 · nvidia-smi显存监控 最简操作'   ← practice 版头卡
   - '### 本节关键命令 · 你自己的运行记录（…）'   ← 3 段式外壳第 1 段的表格
   - '## 概念与设计动机：…'  '### 本块必读锚点 · 概念卡片'   ← theory 版头卡
2. 检查 harness 里有没有 LLM 调用成功 fallback 条件：
   - _is_llm_usable() 是否返回 False（.env DEEPSEEK_API_KEY 占位符、超时、budget=0）
3. 对比「每个块的 headings 序列与 deterministic 规则硬写的模板是否 100% 吻合」，吻合度 ≥14/17 → 基本判定 100% 走 deterministic 纯规则
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "project.db"

from sqlalchemy import create_engine, text

engine = create_engine(f"sqlite:///{DB_PATH}")

DET_SCAFFOLD_MARKERS_PRACTICE = [
    "# 画像专属 · ",
    "· 本块学习卡",
    "## 立即动手 · ",
    "### 本节关键命令 · 你自己的运行记录",
    "## 概念与设计动机：",  # 理论头卡
    "### 本块必读锚点 · 概念卡片",  # 理论头卡
    "#### 本节关键命令 · 你自己的运行记录",  # 3段外壳首段
]

with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT s.session_id, s.goal, s.created_at, ss.state_json
        FROM sessions s LEFT JOIN session_states ss ON ss.session_id=s.session_id
        ORDER BY s.created_at DESC LIMIT 6
    """)).fetchall()
    for r in rows:
        rm = r._mapping
        sj = json.loads(rm["state_json"]) if rm["state_json"] else {}
        ls = (sj.get("profile") or {}).get("learning_style") or "(none)"
        lp = sj.get("learning_package") or {}
        cbs = lp.get("content_blocks") or []
        print(f"\nsession {rm['session_id'][:14]}… style={ls}  blocks={len(cbs)}")
        scaf_hit = 0
        for i, b in enumerate(cbs if isinstance(cbs, list) else []):
            pc = b.get("personalized_content") if isinstance(b, dict) else ""
            hp = (b.get("heading_path") or "") if isinstance(b, dict) else ""
            leaf = hp.split(">")[-1].strip() if ">" in hp else hp[:40]
            hits = [m for m in DET_SCAFFOLD_MARKERS_PRACTICE if m in pc]
            head_first = [ln.strip() for ln in pc.splitlines() if ln.strip()][:2]
            if hits:
                scaf_hit += 1
            print(f"  [{i:2d}] 脚手架命中={len(hits)} -> {hits[:3]}  leaf={leaf[:50]:<50s}")
            for l in head_first:
                print(f"        head >> {l[:140]}")
        print(f"  共 {scaf_hit}/{len(cbs) if isinstance(cbs, list) else 0} 块出现「旧3段式外壳脚手架字符串」")
        if scaf_hit > len(cbs) * 0.6:
            print("  🔴 结论：几乎整块都是 deterministic 纯规则生成，没有调用 DeepSeek LLM 做个性化改写。")
        elif scaf_hit > 0:
            print("  🟡 结论：部分块未调 LLM 触发 fallback，用了 deterministic 规则生成（脚手架还在）。")
        else:
            print("  🟢 结论：没有发现 deterministic 脚手架字符串，疑似已调用 DeepSeek LLM 个性化。")

# 最后打印 .env 中 DeepSeek 配置
print("\n\n=== .env 关键配置（检查 LLM 是否可被调用）===")
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for ln in env_file.read_text(encoding="utf-8").splitlines():
        if any(k in ln.upper() for k in ("DEEPSEEK", "LLM_PROVIDER", "EXECUTION_MODE", "BLOCK_PIPELINE", "FALLBACK")):
            stripped = ln.strip()
            if "KEY" in stripped.upper():  # 不要打印实际密钥
                name, _, val = stripped.partition("=")
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                masked = val[:6] + "…" + val[-4:] if len(val) > 10 else "***"
                print(f"  {name}={masked}   (长度={len(val)}, your-前缀占位?={'your-' in val})")
            else:
                print(f"  {stripped}")
else:
    print("  没有 .env 文件！")
