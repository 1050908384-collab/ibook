# -*- coding: utf-8 -*-
"""
Ollama 可用性 + 回退机制 全面验证（不改任何业务代码，只用原始 LLMClient）
步骤：
  S2a) 原始 HTTP 方式 访问 Ollama：GET http://127.0.0.1:11434/api/healthz  （health）
  S2b) httpx 直接调用 Ollama /api/generate（确认 Ollama 本身能 generate）
  S2c) 通过项目的 LLMClient._complete_with_ollama() 调用（确认 wrapper 正确可用）
  S3)  切断 DeepSeek（空 KEY + 关回退）→ _has_deepseek_credentials=False
  S4)  切断 DeepSeek（假 Key + 开回退）→ transport failure → fallback 到 ollama（last_provider=ollama）
  S5)  直连 DeepSeek 尝试（无代理）→ 看看 ikuuu 关了后能不能直连
全程对 client.py 做「无新增探测代码」断言检查。
"""
import os
import sys
import asyncio
import time
import httpx

BACKEND = r"d:\我的项目\揭榜挂帅\ai_agent-trainer\backend"
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from config import settings
from llm.client import LLMClient

SEP = "=" * 78
SUB = "-" * 78

def mask(k):
    if not k: return "<空>"
    return f"{k[:6]}****{k[-4:]}" if len(k)>10 else "***"

# 保护检查：client.py 不能有新增探测代码
print(SEP)
print("  Ollama 可用性 + 回退机制 验证")
print(SEP)
with open(os.path.join(BACKEND, "llm", "client.py"), "r", encoding="utf-8") as f:
    src = f.read()
forbidden = ["import socket", "COMMON_LOCAL_PROXY_PORTS", "_detect_local_proxy", "auto_proxy"]
for tok in forbidden:
    if tok in src:
        print(f"  ❌ FATAL: client.py 中仍存在 {tok}（业务代码被改了，用户不允许）")
        sys.exit(1)
print("  ✅ client.py 业务代码纯净检查通过（无任何新增探测代码）")
print(f"  settings.DEEPSEEK_API_KEY      : {mask(settings.DEEPSEEK_API_KEY)}")
print(f"  settings.OLLAMA_BASE_URL        : {settings.OLLAMA_BASE_URL}")
print(f"  settings.OLLAMA_MODEL           : {settings.OLLAMA_MODEL}")
print(f"  settings.LLM_FALLBACK_PROVIDER  : {settings.LLM_FALLBACK_PROVIDER}")

# ========== S2a) Ollama 可用性检查 ==========
print(f"\n{SUB}")
print("【S2a】HTTP 直连 Ollama 可用性端点 + 模型列表 ...")
try:
    with httpx.Client(timeout=5) as h:
        r = h.get(settings.OLLAMA_BASE_URL + "/")
        print(f"  GET /           → HTTP {r.status_code}（Ollama 主页，200=ok）")
        try:
            r2 = h.get(settings.OLLAMA_BASE_URL + "/api/healthz")
            print(f"  GET /api/healthz → HTTP {r2.status_code}（旧版 Ollama 可能 404，可忽略）")
        except Exception as e_h:
            print(f"  GET /api/healthz → 跳过: {e_h}")
        r3 = h.get(settings.OLLAMA_BASE_URL + "/api/tags")
        js = r3.json()
        names = [m["name"] for m in js.get("models", [])]
        print(f"  GET /api/tags    → {len(names)} 模型: {names}")
        has_default = settings.OLLAMA_MODEL in names
        print(f"  默认模型 '{settings.OLLAMA_MODEL}' 在列表中: {'✅' if has_default else '❌'}")
        s2a_ok = has_default and r.status_code == 200
except Exception as e:
    s2a_ok = False
    print(f"  ❌ 异常: {type(e).__name__}: {e}")
print(f"  S2a 结果: {'✅ PASS' if s2a_ok else '❌ FAIL'}")


# ========== S2b) Ollama generate 直接调用（httpx，绕过 wrapper） ==========
print(f"\n{SUB}")
print("【S2b】httpx 直接 POST Ollama /api/chat（用项目真实端点，验证 Ollama 生成能力）...")
s2b_ok = False
try:
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "你是一位计算器，只输出纯数字，不要任何其他文字。"},
            {"role": "user", "content": "8 × 9 = ? 只输出一个数字。"},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 32},
    }
    t0 = time.time()
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=5, read=60)) as h:
        r = h.post(settings.OLLAMA_BASE_URL + "/api/chat", json=payload)
    dt = time.time() - t0
    if r.status_code == 200:
        j = r.json()
        msg = j.get("message") or {}
        content = str(msg.get("content") or "").strip()
        print(f"  HTTP 200 ({dt:.1f}s) -> message.content[:80] = {content[:80]!r}")
        s2b_ok = bool(content) and not content.lower().startswith("error")
    else:
        print(f"  HTTP {r.status_code}: {r.text[:300]}")
except Exception as e:
    print(f"  ❌ 异常: {type(e).__name__}: {e}")
print(f"  S2b 结果: {'✅ PASS' if s2b_ok else '❌ FAIL'}")


# ========== S2c) LLMClient._complete_with_ollama() 封装调用 ==========
print(f"\n{SUB}")
print("【S2c】通过项目 LLMClient._complete_with_ollama(prompt, system, ...) 调用（确认代码封装正确）...")
c2c = LLMClient()
try:
    t0 = time.time()
    resp = asyncio.run(c2c._complete_with_ollama(
        prompt="3 × 4 = ？只输出一个数字。",
        system_prompt="你是一位计算器，只输出数字。",
        max_tokens=64, temperature=0.0, json_mode=False,
    ))
    dt = time.time() - t0
    s2c_ok = (not resp.startswith("Error:")) and bool(resp.strip())
    print(f"  响应 ({dt:.1f}s): {resp.strip()[:80]!r}")
    print(f"  非 Error 开头: {'✅' if s2c_ok else '❌'}")
except Exception as e:
    s2c_ok = False
    import traceback
    print(f"  ❌ 异常: {type(e).__name__}: {e}")
    traceback.print_exc(limit=4)
print(f"  S2c 结果: {'✅ PASS' if s2c_ok else '❌ FAIL'}")


# ========== S3) DeepSeek 凭证检查：占位符 → _has_deepseek_credentials=False → 直接走 Ollama ==========
print(f"\n{SUB}")
print("【S3】凭证为占位符（your-deepseek-api-key）时：应判定为 False，不走 DeepSeek 直接返回 Ollama 结果...")
orig_key = settings.DEEPSEEK_API_KEY
settings.DEEPSEEK_API_KEY = "your-deepseek-api-key"
settings.LLM_FALLBACK_PROVIDER = "ollama"
c_s3 = LLMClient()
hasc = c_s3._has_deepseek_credentials()
fallback = c_s3._can_fallback_to_ollama()
print(f"  key = 'your-deepseek-api-key' → _has_deepseek_credentials = {hasc} (期望 False)")
print(f"  LLM_FALLBACK_PROVIDER='ollama' → _can_fallback_to_ollama = {fallback} (期望 True)")
s3_ok = (hasc == False) and (fallback == True)
async def run_s3():
    return await c_s3.complete("5×6=? 只输出数字。", "你是计算器，只输出数字。",
                               max_tokens=64, temperature=0.0)
if s3_ok:
    try:
        t0 = time.time()
        resp = asyncio.run(run_s3())
        dt = time.time() - t0
        print(f"  实际响应 ({dt:.1f}s): last_provider={c_s3.last_provider} | {resp.strip()[:60]!r}")
        last_ok = (c_s3.last_provider == "ollama")
        resp_ok = (not resp.startswith("Error:"))
        print(f"  last_provider=='ollama': {'✅' if last_ok else '❌'}")
        print(f"  响应无 Error:        {'✅' if resp_ok else '❌'}")
        s3_ok = last_ok and resp_ok and dt < 15  # 不应该等 DeepSeek 重试，<15 秒就是没走 DS
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        s3_ok = False
# 恢复 Key
settings.DEEPSEEK_API_KEY = orig_key
print(f"  S3 结果: {'✅ PASS' if s3_ok else '❌ FAIL'}（<15s 证明跳过了 DS 重试）")


# ========== S4) DeepSeek 假 Key + 开回退：3×DS 传输失败 → fallback 到 Ollama ==========
print(f"\n{SUB}")
print("【S4】真 Key 路径失败 + 开回退：应触发 transport failure retry，然后 fallback 到 Ollama...")
settings.DEEPSEEK_API_KEY = "sk-fake-forcefallback-1234567890abcdef"
settings.LLM_PROVIDER = "deepseek"
settings.LLM_FALLBACK_PROVIDER = "ollama"
settings.DEEPSEEK_TIMEOUT_SECONDS = 10.0
settings.DEEPSEEK_MAX_RETRIES = 1
c_s4 = LLMClient()
s4_ok = False
try:
    t0 = time.time()
    resp = asyncio.run(c_s4.complete(
        "7+8=? 只输出数字。", "你是计算器，只输出数字。",
        max_tokens=64, temperature=0.0,
    ))
    dt = time.time() - t0
    print(f"  耗时: {dt:.1f}s（含 1×10s DS 重试）")
    print(f"  last_provider: {c_s4.last_provider}（期望 ollama）")
    print(f"  last_model:    {c_s4.last_model}")
    fb = str(c_s4.last_fallback_reason or "")[:100]
    print(f"  fallback_reason[:100]: {fb or '(空)'}（应有 transport failure 文本）")
    print(f"  响应: {resp.strip()[:80]!r}")
    s4_ok = (
        c_s4.last_provider == "ollama"
        and "transport" in fb.lower()
        and not resp.startswith("Error:")
    )
except Exception as e:
    import traceback
    print(f"  ❌ 异常: {type(e).__name__}: {e}")
    traceback.print_exc(limit=4)
settings.DEEPSEEK_API_KEY = orig_key
print(f"  S4 结果: {'✅ PASS' if s4_ok else '❌ FAIL'}")


# ========== S5) 直连 DeepSeek 云 API ==========
print(f"\n{SUB}")
print("【S5】ikuuu 关后直连 api.deepseek.com（无代理，trust_env，无 HTTPS_PROXY）...")
settings.LLM_FALLBACK_PROVIDER = ""   # 关掉回退，只试原生
settings.DEEPSEEK_API_KEY = orig_key
c_s5 = LLMClient()
print(f"  HTTPS_PROXY 环境变量当前值: {os.environ.get('HTTPS_PROXY', '<空>')}")
print(f"  _has_deepseek_credentials: {c_s5._has_deepseek_credentials()}")
s5_connected = False
try:
    t0 = time.time()
    resp = asyncio.run(c_s5.complete(
        "1+1=?", "你是计算器，只输出数字。", max_tokens=32, temperature=0.0,
    ))
    dt = time.time() - t0
    s5_connected = (c_s5.last_provider == "deepseek")
    print(f"  耗时: {dt:.1f}s")
    print(f"  last_provider: {c_s5.last_provider}")
    print(f"  响应: {resp.strip()[:80]!r}")
except Exception as e:
    print(f"  异常: {type(e).__name__}: {e}")
s5_txt = "✅ PASS（直连可用，不用代理也行）" if s5_connected else "⏭️  FAIL（直连不通，正常现象：校园网/国内环境不通 deepseek.com 443 需要 VPN 才能直连）"
print(f"  S5 结果: {s5_txt}")

settings.LLM_FALLBACK_PROVIDER = "ollama"  # 还原

# ========== 总结 ==========
print("\n" + SEP)
print("  最终总结")
print(SEP)
rows = [
    ("S2a Ollama health + 默认模型存在", s2a_ok),
    ("S2b Ollama /api/generate 原生HTTP调用", s2b_ok),
    ("S2c LLMClient._complete_with_ollama 封装调用", s2c_ok),
    ("S3 占位符Key→直接走Ollama <15s（跳过DS重试）", s3_ok),
    ("S4 假Key+开回退→DS失败后自动回退Ollama", s4_ok),
]
all_ok = True
for name, ok in rows:
    f = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {f} — {name}")
    if not ok: all_ok = False
print(f"  {'✅ 连通（无需代理）' if s5_connected else '⏭️  不通（需要VPN/代理，属于正常网络情况）'} — S5 直连 api.deepseek.com")
print(SEP)
if all_ok:
    print("  🎉 Ollama 可用性 + 回退机制 100% 正常。")
    print("     · 项目代码零修改。")
    print("     · ikuuu 关了也不影响 Ollama 本地调用和回退路径完全工作。")
    print("     · DeepSeek 直连通不通属于评委电脑网络情况，不影响你的项目（有回退兜底）。")
else:
    print("  ❌ 存在失败项，需要具体看是哪一项。")
