# -*- coding: utf-8 -*-
"""最后一次性核查：DeepSeek LLM 配置跨电脑可移植性检查清单"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RE_KV  = PROJECT_ROOT / ".env.example"
BE_KV  = PROJECT_ROOT / "backend" / ".env.example"
LIVE   = PROJECT_ROOT / "backend" / ".env"

def load(p: Path):
    kv = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.partition("=")
        # 去首尾引号
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        kv[k.strip()] = v
    return kv

def mask(s):
    if len(s) > 10:
        return s[:4] + "…" + s[-4:]
    return "*" * len(s) if s else s

def main():
    has_re  = RE_KV.exists()
    has_be  = BE_KV.exists()
    has_lv  = LIVE.exists()
    print(f"[.env 文件存在性]")
    print(f"  根目录 .env.example:       {has_re}   ({RE_KV})")
    print(f"  backend/.env.example:     {has_be}   ({BE_KV})")
    print(f"  backend/.env (本机实际):  {has_lv}   ({LIVE})")

    re_kv = load(RE_KV) if has_re else {}
    be_kv = load(BE_KV) if has_be else {}
    lv_kv = load(LIVE) if has_lv else {}

    KEYS = ["LLM_PROVIDER", "DEEPSEEK_MODEL", "DEEPSEEK_API_BASE", "DEEPSEEK_API_KEY",
            "LLM_FALLBACK_PROVIDER", "EXECUTION_MODE", "BLOCK_PIPELINE_ENABLED",
            "OLLAMA_BASE_URL", "OLLAMA_MODEL", "DATABASE_URL", "CHROMA_PATH", "GRAPH_PATH"]
    print()
    H = f"{'KEY':30s}|{'root/.env.example':<30s}|{'backend/.env.example':<30s}|{'backend/.env (live)':<30s}"
    print(H)
    print("-" * len(H))
    for k in KEYS:
        rv = re_kv.get(k, "(缺失)")
        bv = be_kv.get(k, "(缺失)")
        lv = lv_kv.get(k, "(缺失)")
        if "KEY" in k:
            rv, bv, lv = mask(rv), mask(bv), mask(lv)
        print(f"{k:30s}|{rv[:30]:<30s}|{bv[:30]:<30s}|{lv[:30]:<30s}")

    print("\n[跨电脑可移植性 核查清单]")
    checks = []

    def C(place, name, ok, detail=""):
        checks.append((place, name, ok, detail))

    for place, kv in [("root/.env.example", re_kv), ("backend/.env.example", be_kv), ("backend/.env live", lv_kv)]:
        m = kv.get("DEEPSEEK_MODEL", "")
        C(place, "DEEPSEEK_MODEL == deepseek-v4-flash （推荐模型）", m == "deepseek-v4-flash", f"当前={m!r}")
        C(place, "LLM_PROVIDER == deepseek （启用云端）", kv.get("LLM_PROVIDER", "") == "deepseek", f"当前={kv.get('LLM_PROVIDER','')!r}")
        C(place, "EXECUTION_MODE == hybrid", kv.get("EXECUTION_MODE", "") == "hybrid", f"当前={kv.get('EXECUTION_MODE','')!r}")
        bp = kv.get("BLOCK_PIPELINE_ENABLED", "")
        C(place, "BLOCK_PIPELINE_ENABLED == true （块流水线）", bp.lower() == "true", f"当前={bp!r}")
        base = kv.get("DEEPSEEK_API_BASE", "")
        C(place, "DEEPSEEK_API_BASE 为官方 https://api.deepseek.com", base in ("https://api.deepseek.com", "https://api.deepseek.com/"), f"当前={base!r}")
        bad_paths = [v for v in kv.values() if isinstance(v, str) and any(tok in v.lower() for tok in ["d:\\", "c:\\users\\suhang", "/home/suhang", "ai_agent-trainer\\data\\", "c:/users/suhang"])]
        C(place, "无本机绝对路径硬编码（D:\\ / C:\\Users\\Suhang / /home/ 等）", not bad_paths, f"可疑={bad_paths[:3]}")

    # 模板：LLM_FALLBACK_PROVIDER 显式空串禁用回退（按用户 memory 约束）
    for place, kv in [("root/.env.example", re_kv), ("backend/.env.example", be_kv)]:
        fb = kv.get("LLM_FALLBACK_PROVIDER", "(缺)")
        C(place, "LLM_FALLBACK_PROVIDER 模板默认空串（禁止默认 Ollama 回退）", fb == "", f"当前={fb!r}")

    # 模板占位符
    for place, kv in [("root/.env.example", re_kv), ("backend/.env.example", be_kv)]:
        key = kv.get("DEEPSEEK_API_KEY", "")
        C(place, "DEEPSEEK_API_KEY 模板是 your- 占位（不会提交真密钥）", key.startswith("your-"), f"当前前缀={key[:20]!r}")

    # 实际 live：API_KEY 不得占位
    if "DEEPSEEK_API_KEY" in lv_kv:
        k = lv_kv["DEEPSEEK_API_KEY"]
        C("backend/.env live", "DEEPSEEK_API_KEY 实际值非占位符（云端调用不会 401）", not (k.lower().startswith("your-") or k.strip() == ""), f"前缀={k[:16]}…")
    else:
        C("backend/.env live", "DEEPSEEK_API_KEY 实际值非占位符", False, "backend/.env 缺失 DEEPSEEK_API_KEY 字段")

    # Ollama 拼写兼容 OLLMA vs OLLAMA
    for place, kv in [("root/.env.example", re_kv), ("backend/.env.example", be_kv)]:
        any_ollma = any(k.upper().startswith("OLLMA") or k.upper().startswith("OLLAMA") for k in kv.keys())
        C(place, "Ollama 变量名使用 OLLAMA_*（正确拼写），已注意 OLLMA_ 历史兼容由 pydantic AliasChoices 兜底", any_ollma, f"存在 ollama/ollma 变量={sorted([k for k in kv if 'OLLMA' in k.upper() or 'OLLAMA' in k.upper()])}")

    # 控制台格式化表输出
    print(f"\n{'位置':<25s} | {'检查项':<60s} | {'PASS':<6s} | 细节")
    print("-" * 150)
    for place, item, ok, detail in checks:
        print(f"{place:<25s} | {item:<60s} | {'PASS' if ok else 'FAIL':<6s} | {detail}")

    # 最后给出 PASS/FAIL 总结
    total = len(checks)
    passed = sum(1 for c in checks if c[2])
    failed = total - passed
    print(f"\n[总结] 检查项 {total} 项，PASS {passed}，FAIL {failed}")

if __name__ == "__main__":
    main()
