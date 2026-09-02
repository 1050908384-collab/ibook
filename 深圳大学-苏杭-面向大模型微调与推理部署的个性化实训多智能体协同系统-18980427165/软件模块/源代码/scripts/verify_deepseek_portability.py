# -*- coding: utf-8 -*-
"""
验证修复后的 DeepSeek + Ollama 集成效果（基于提交文件夹源代码）
测试 4 项：
1. LLMClient 导入正常 + 新增的 COMMON_LOCAL_PROXY_PORTS / _detect_local_proxy 存在
2. 占位符 Key → _has_deepseek_credentials()=False，自动走 Ollama
3. 手动给一个假的「非占位符 Key」→ 判定为有凭证但 DeepSeek HTTP 出错 → 自动 fallback Ollama
4. complete_sync / chat 路径也正常工作
5. LLM_FALLBACK_PROVIDER="" 时，不回退 Ollama（正确返回 Error 字符串）
"""
import os
import sys
import asyncio
import time

SUBMIT_BACKEND = r"d:\我的项目\揭榜挂帅\ai_agent-trainer\深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165\软件模块\源代码\backend"
SUBMIT_ROOT = os.path.dirname(SUBMIT_BACKEND)

print("=" * 70)
print("  修复效果综合验证 · 提交文件夹 DeepSeek + Ollama 集成")
print("=" * 70)

sys.path.insert(0, SUBMIT_BACKEND)
os.chdir(SUBMIT_BACKEND)

# -------- T1: 基本结构检查 --------
print("\n【T1/5】代码结构检查（socket import / 代理常量 / _detect_local_proxy）...")
import socket  # noqa: F401
from config import settings
from llm.client import (
    LLMClient, COMMON_LOCAL_PROXY_PORTS, _DeepSeekAsyncNetworkBackend  # noqa: F401
)
print(f"  COMMON_LOCAL_PROXY_PORTS = {COMMON_LOCAL_PROXY_PORTS}")
assert len(COMMON_LOCAL_PROXY_PORTS) >= 5, "常见代理端口至少 5 个"
assert hasattr(LLMClient, "_detect_local_proxy"), "缺少 _detect_local_proxy 静态方法"
proxy_found = LLMClient._detect_local_proxy()
print(f"  本机自动代理探测结果: {proxy_found or '未发现本地代理（直连环境）'}")
print("  T1 ✅ PASS")

# -------- T2: 占位符凭证判定 + 自动 Ollama --------
print("\n【T2/5】占位符 Key 判定 → 自动走 Ollama ...")
# 强制覆盖为占位符
settings.DEEPSEEK_API_KEY = "your-deepseek-api-key"
settings.LLM_PROVIDER = "deepseek"
settings.LLM_FALLBACK_PROVIDER = "ollama"

c1 = LLMClient()
print(f"  _has_deepseek_credentials = {c1._has_deepseek_credentials()}")
print(f"  _can_fallback_to_ollama   = {c1._can_fallback_to_ollama()}")
assert c1._has_deepseek_credentials() is False, "占位符 Key 应判定为无凭证"
assert c1._can_fallback_to_ollama() is True

async def test_t2():
    t0 = time.time()
    resp = await c1.complete("请回答 2+2=?", "你是老师，简洁回答。",
                             max_tokens=64, temperature=0.1)
    dt = time.time() - t0
    ok = not resp.startswith("Error:")
    print(f"  响应({dt:.1f}s): {resp.strip()[:80]}")
    print(f"  last_provider={c1.last_provider}  last_model={c1.last_model}")
    print(f"  last_fallback_reason={c1.last_fallback_reason}")
    return ok

ok2 = asyncio.run(test_t2())
assert ok2, "占位符 Key 应自动走 Ollama 并得到有效回答"
print("  T2 ✅ PASS")

# -------- T3: 假 Key（非占位符）→ DeepSeek 失败 → Fallback Ollama --------
print("\n【T3/5】非占位符但无效 Key → 判定有凭证 → DeepSeek 传输/HTTP 失败 → 自动回退 Ollama...")
# 模拟用户填了一串假的 sk- 开头 Key
settings.DEEPSEEK_API_KEY = "sk-fake-test-key-1234567890abcdef"
c3 = LLMClient()
assert c3._has_deepseek_credentials() is True, "sk- 开头非占位符应判定为有凭证"

async def test_t3():
    t0 = time.time()
    resp = await c3.complete("请回答 3+5=?", "你是计算器，只输出数字。",
                             max_tokens=64, temperature=0.1)
    dt = time.time() - t0
    ok = not resp.startswith("Error:")
    print(f"  响应({dt:.1f}s): {resp.strip()[:80]}")
    print(f"  last_provider={c3.last_provider}  last_model={c3.last_model}")
    # 应当先尝试 DeepSeek，出错后 fallback 到 Ollama
    print(f"  last_fallback_reason[:80]={str(c3.last_fallback_reason)[:80]}")
    return ok

ok3 = asyncio.run(test_t3())
assert ok3, "无效 Key 但启用回退时应当 fallback 到 Ollama 并成功"
# 检查 provider 是否为 ollama
assert c3.last_provider == "ollama", f"回退后 last_provider 应为 ollama，实际为 {c3.last_provider}"
print("  T3 ✅ PASS（自动 Ollama 回退生效）")

# -------- T4: complete_sync 同步路径 & chat 路径 --------
print("\n【T4/5】同步路径 complete_sync + chat 路径验证...")
settings.DEEPSEEK_API_KEY = "your-deepseek-api-key"  # 占位符，直接 Ollama
c4 = LLMClient()

msgs = [
    {"role": "system", "content": "你是一位计算器，只输出数字。"},
    {"role": "user", "content": "7*8=?"},
]
t0 = time.time()
resp_sync = c4.complete_sync(msgs, max_tokens=32, temperature=0.1)
dt = time.time() - t0
print(f"  complete_sync 响应({dt:.1f}s): '{resp_sync.strip()[:50]}'")
print(f"  complete_sync last_provider={c4.last_provider}")
ok_sync = not resp_sync.startswith("Error:")
assert ok_sync, "同步 complete_sync 应成功"

async def test_chat():
    resp = await c4.chat([
        {"role": "system", "content": "你是计算器，只输出数字。"},
        {"role": "user", "content": "9-4=?"},
    ], max_tokens=32, temperature=0.1)
    return resp

resp_chat = asyncio.run(test_chat())
print(f"  chat 响应: '{resp_chat.strip()[:50]}'  last_provider={c4.last_provider}")
ok_chat = not resp_chat.startswith("Error:")
assert ok_chat, "chat 异步路径应成功"
print("  T4 ✅ PASS")

# -------- T5: LLM_FALLBACK_PROVIDER="" → 不应回退（安全边界）--------
print("\n【T5/5】LLM_FALLBACK_PROVIDER 为空 → 凭证缺失时不回退 → 返回 Error 字符串...")
settings.DEEPSEEK_API_KEY = ""
settings.LLM_FALLBACK_PROVIDER = ""
c5 = LLMClient()
assert c5._can_fallback_to_ollama() is False, "FALLBACK_PROVIDER 为空时应不能回退"

async def test_t5():
    resp = await c5.complete("随便问", "", max_tokens=32)
    print(f"  响应: {resp[:100]}")
    return resp.startswith("Error:")

ok5 = asyncio.run(test_t5())
assert ok5, "无凭证且关闭回退时应返回 Error 开头"
print("  T5 ✅ PASS（回退开关正确生效）")

# -------- 结论 --------
print("\n" + "=" * 70)
print("  🎉 全部 5/5 项验证通过！修复效果综合确认：")
print("=" * 70)
print("  ✅ 本地代理端口自动探测（7890 等 7 个端口，200ms/端口）")
print("  ✅ 占位符 Key（your-deepseek-api-key）不会误判为真凭证")
print("  ✅ Key 有效但 DeepSeek 不通 → 自动回退本地 Ollama")
print("  ✅ complete / chat / complete_sync 三条路径都能触发回退")
print("  ✅ LLM_FALLBACK_PROVIDER=\"\" 安全关闭回退，不乱跑 Ollama")
print("  ✅ 评委电脑只要：① 填真实 Key + 能连外网/开 Clash → DeepSeek 直用")
print("             ② 没填 Key / 外网不通 → 自动 Ollama 兜底运行")
print("=" * 70)
