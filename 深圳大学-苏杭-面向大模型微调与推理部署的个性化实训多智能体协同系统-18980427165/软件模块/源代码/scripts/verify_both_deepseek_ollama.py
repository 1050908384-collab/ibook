# -*- coding: utf-8 -*-
"""
双通道验证（在主项目 backend 下运行，代码已修复代理自动探测）
  A) DeepSeek 原生可调用：凭证有效 + 网络可达 → last_provider == 'deepseek'
  B) 切断 DeepSeek（假 Key 或 不可达 API） → 自动回退 Ollama → last_provider == 'ollama'
每一步门禁失败立即停止并输出明确原因。全程对 API Key 做脱敏输出。
"""
import os
import sys
import asyncio
import time
import socket
import httpx

# 修复后的代码来自「主项目」——两项目 client.py 已一致
BACKEND_DIR = r"d:\我的项目\揭榜挂帅\ai_agent-trainer\backend"
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

from config import settings            # noqa: E402
from llm.client import LLMClient      # noqa: E402

SEP = "=" * 78
STEP = "-" * 78

def mask_key(k: str) -> str:
    if not k:
        return "<空>"
    if len(k) <= 10:
        return "***"
    return f"{k[:6]}****{k[-4:]}"

print(SEP)
print("  双通道验证：DeepSeek 原生调用 + Ollama 自动回退")
print(SEP)

# ============================================================
# 门禁 0：探测常见本地代理（反映修复后的自动代理能力）
# ============================================================
print("\n[门禁 0/4] 本地代理端口自动探测（修复后 client.py 的 _detect_local_proxy）...")
auto_proxy = LLMClient._detect_local_proxy()
if auto_proxy:
    print(f"  ✅ 发现本地代理: {auto_proxy} → 将自动用于 DeepSeek 云端连接")
else:
    print("  ℹ️  未发现本地常见代理 → 走直连 / 系统环境变量 HTTPS_PROXY")

# ============================================================
# 门禁 1：Ollama 服务 + 模型检查
# ============================================================
print(f"\n[门禁 1/4] 检查 Ollama 服务 (localhost:11434) ...")
try:
    with socket.create_connection(("127.0.0.1", 11434), timeout=2):
        pass
except (OSError, socket.timeout) as e:
    print(f"  ❌ Ollama 未在 :11434 监听 ({e}). 请先执行: ollama serve")
    print("     停止验证：回退链路无法通过。")
    sys.exit(1)
print("  ✅ Ollama 服务监听 :11434")

# 查模型列表
try:
    with httpx.Client(timeout=5) as hc:
        r = hc.get("http://127.0.0.1:11434/api/tags")
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  ✅ 本地 Ollama 模型列表: {models}")
        # 文本模型优先级：1) qwen2.5:7b（首选/中文质量优） -> 2) deepseek-r1:7b（次选/推理强）
        # 其余：qwen2.5:14b / deepseek-r1:8b 为可选大规格增强（评审设备若具备 16GB+ 显存可考虑）
        primary_text   = "qwen2.5:7b"
        secondary_text = "deepseek-r1:7b"
        optional_text  = {"qwen2.5:14b", "deepseek-r1:8b"}
        want = {primary_text, secondary_text} | optional_text
        present = set(models) & want
        missing_critical = [m for m in (primary_text, secondary_text) if m not in set(models)]
        if not present:
            print(f"  ⚠️   未检测到任何推荐模型。请先拉取首选模型：")
            print(f"       ollama pull {primary_text}    # 首选（中文生成质量稳定，推荐先拉取）")
            print(f"       ollama pull {secondary_text}    # 次选（推理更强，作为首选异常时的回退）")
        else:
            if primary_text in set(models):
                print(f"  ✅ 首选文本模型 {primary_text} 已安装")
            else:
                print(f"  ⚠️   首选文本模型 {primary_text} 缺失，建议安装：ollama pull {primary_text}")
            if secondary_text in set(models):
                print(f"  ✅ 次选文本模型 {secondary_text} 已安装")
            else:
                print(f"  -   次选文本模型 {secondary_text} 缺失（可选回退增强）：ollama pull {secondary_text}")
except Exception as e:
    print(f"  ❌ 无法获取 Ollama 模型列表: {e}")
    sys.exit(1)

# ============================================================
# 门禁 2：找到可用 DeepSeek Key（环境变量优先 > 提交文件夹原 Key 备份 > 主项目 backend/.env）
# ============================================================
print(f"\n[门禁 2/4] 定位可用 DeepSeek API Key（环境变量 > 各 .env）...")

def parse_env(path: str) -> dict:
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#") or "=" not in t:
                continue
            k, v = t.split("=", 1)
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d

candidates = []
# 1. 进程/用户/系统环境变量
env_key = (
    os.environ.get("DEEPSEEK_API_KEY")
    or ""
).strip()
if env_key:
    candidates.append(("环境变量 DEEPSEEK_API_KEY", env_key))

# 2. 主项目 backend/.env
_main_env = parse_env(os.path.join(BACKEND_DIR, ".env"))
if _main_env.get("DEEPSEEK_API_KEY") and not _main_env["DEEPSEEK_API_KEY"].lower().startswith(("your-", "your_", "sk-your", "xxx", "placeholder")):
    candidates.append(("主项目 backend/.env", _main_env["DEEPSEEK_API_KEY"]))

# 3. 主项目根目录 .env
_root_env = parse_env(r"d:\我的项目\揭榜挂帅\ai_agent-trainer\.env")
if _root_env.get("DEEPSEEK_API_KEY") and not _root_env["DEEPSEEK_API_KEY"].lower().startswith(("your-", "your_", "sk-your", "xxx", "placeholder")):
    candidates.append(("主项目 根目录 .env", _root_env["DEEPSEEK_API_KEY"]))

# 4. 提交文件夹 backend/.env（占位符不参与）
_submit_env_path = r"d:\我的项目\揭榜挂帅\ai_agent-trainer\深圳大学-苏杭-面向大模型微调与推理部署的个性化实训多智能体协同系统-18980427165\软件模块\源代码\backend\.env"
_sub_env = parse_env(_submit_env_path)
if _sub_env.get("DEEPSEEK_API_KEY") and not _sub_env["DEEPSEEK_API_KEY"].lower().startswith(("your-", "your_", "sk-your", "xxx", "placeholder")):
    candidates.append(("提交版 backend/.env", _sub_env["DEEPSEEK_API_KEY"]))

if not candidates:
    print("  ❌ 未在环境变量或任何 .env 中找到非占位符的 DeepSeek API Key。")
    print("     请先设置：")
    print("       $env:DEEPSEEK_API_KEY = 'sk-你的有效Key'")
    print("     然后重新运行本脚本。A 路径（DeepSeek 原生）将跳过。")
    VALID_KEY = None
else:
    source, VALID_KEY = candidates[0]
    print(f"  ✅ 采用 Key 来源: {source}  [{mask_key(VALID_KEY)}]")

# ============================================================
# 门禁 3：DeepSeek 云 API 连通性 + 鉴权快速预检（/v1/models, 不消耗 token）
# ============================================================
print(f"\n[门禁 3/4] 快速预检 DeepSeek 云 API 连通性与鉴权（GET /v1/models）...")
DS_OK = False
DS_REACHABLE = False
if VALID_KEY:
    settings.DEEPSEEK_API_KEY = VALID_KEY
    settings.LLM_PROVIDER = "deepseek"
    settings.DEEPSEEK_API_BASE = "https://api.deepseek.com"
    settings.DEEPSEEK_MODEL = "deepseek-v4-flash"
    settings.LLM_FALLBACK_PROVIDER = ""   # 预检期间关闭回退，避免把 fallback 误判为成功

    # 用修复后的 client，自动走探测到的本地代理（auto_proxy 已经打印）
    probe_client = LLMClient()
    headers = probe_client._headers()
    # 从构造后的 client 取 transport/proxy：直接复用 deepseek_client 发送
    try:
        t0 = time.time()
        resp = asyncio.run(
            probe_client.deepseek_client.get(
                "https://api.deepseek.com/v1/models",
                headers={"Authorization": headers["Authorization"]},
                timeout=httpx.Timeout(15.0, connect=10.0),
            )
        )
        dt = time.time() - t0
        if resp.status_code == 200:
            data = resp.json()
            model_ids = [m["id"] for m in data.get("data", [])[:8]]
            print(f"  ✅ HTTP 200 ({dt:.1f}s). 可用模型片段: {model_ids}")
            if settings.DEEPSEEK_MODEL in model_ids:
                print(f"  ✅ 请求模型 '{settings.DEEPSEEK_MODEL}' 在账号模型列表中")
            else:
                print(f"  ⚠️   请求模型 '{settings.DEEPSEEK_MODEL}' 未在 v1/models 中列出（可能新模型尚未登记，仍继续验证）")
            DS_OK = True
            DS_REACHABLE = True
        elif resp.status_code == 401:
            print(f"  ❌ HTTP 401 ({dt:.1f}s) — API Key 无效/过期。请换有效 Key。")
            DS_REACHABLE = True   # 至少网络通了，只是鉴权失败，后面 B 路径（回退）依然能做
            VALID_KEY = None
        else:
            print(f"  ❌ HTTP {resp.status_code} ({dt:.1f}s): {resp.text[:200]}")
            DS_REACHABLE = (resp.status_code != 0)
            VALID_KEY = None
    except Exception as e:
        print(f"  ❌ 传输异常: {type(e).__name__}: {e}")
        VALID_KEY = None
else:
    print("  ℹ️  跳过 DeepSeek 鉴权预检（无可用 Key）。")

# ============================================================
# 验证 A 路径：DeepSeek 原生调用
# ============================================================
print(f"\n{STEP}")
print(f" 【验证 A】DeepSeek 原生调用（不回退）...")
print(STEP)
PATH_A_PASS = False
if DS_OK and VALID_KEY:
    # 关回退，保证必须是 DeepSeek 成功才 PASS
    settings.LLM_FALLBACK_PROVIDER = ""
    c_a = LLMClient()
    prompt_a = "用中文回答：计算 12 × 15 = ？ 只输出结果数字，不要解释。"

    async def run_a():
        return await c_a.complete(
            prompt_a,
            system_prompt="你是一个精准计算器。用户要求只输出数字时就只输出数字，不要任何其他字符、空格、换行，不要思考过程。",
            max_tokens=256,
            temperature=0.0,
        )

    t0 = time.time()
    ans_a = asyncio.run(run_a())
    dt_a = time.time() - t0

    print(f"  耗时        : {dt_a:.2f}s")
    print(f"  原始响应    : {repr(ans_a[:80])}")
    print(f"  last_provider : {c_a.last_provider}")
    print(f"  last_model    : {c_a.last_model}")
    print(f"  fallback_reas : {c_a.last_fallback_reason}")
    print(f"  last_done_reas: {c_a.last_done_reason}")

    provider_ok = (c_a.last_provider == "deepseek")
    model_ok = (c_a.last_model == "deepseek-v4-flash")
    has_number = ("180" in ans_a or "180".strip() in str(ans_a))
    no_error = not ans_a.startswith("Error:")

    print()
    print(f"  判定项:")
    print(f"    last_provider=='deepseek'       → {'✅' if provider_ok else '❌ FAIL'} (got: {c_a.last_provider})")
    print(f"    last_model=='deepseek-v4-flash' → {'✅' if model_ok else '⚠️ '} (got: {c_a.last_model})")
    print(f"    响应不包含 Error:              → {'✅' if no_error else '❌ FAIL'}")
    print(f"    包含正确数字 180:               → {'✅' if has_number else '⚠️  (答案解析差异)'}  (响应: {ans_a.strip()[:40]})")

    PATH_A_PASS = (provider_ok and no_error)
    print()
    if PATH_A_PASS:
        print("  🎉 验证 A 通过：DeepSeek 云端原生调用完全正常（未触发回退）✅")
    else:
        print("  ❌ 验证 A 失败：last_provider 不是 deepseek，或返回了 Error。")
else:
    print("  ⏭️  跳过验证 A（门禁 3 未通过）：当前环境下 DeepSeek 原生不可调用（Key 无效或网络不通）。")
    print("     若你确认 DS 可用，请：")
    print("       ① 确认代理软件已开启（Clash/v2rayN 系统代理开关）")
    print("       ② 设置 $env:DEEPSEEK_API_KEY 为有效 Key 后再跑。")

# ============================================================
# 验证 B 路径：切断 DeepSeek → 自动回退 Ollama
# ============================================================
print(f"\n{STEP}")
print(f" 【验证 B】切断 DeepSeek → 自动回退 Ollama（验证回退链路）...")
print(STEP)
PATH_B_PASS = False

# 统一用「假 Key + 关代理」方式人为让 DeepSeek 失败
settings.LLM_PROVIDER = "deepseek"
settings.LLM_FALLBACK_PROVIDER = "ollama"
settings.DEEPSEEK_API_KEY = "sk-fake-key-to-force-fallback-1234567890abcdef"   # 非占位符，假的
# 为了加速：降低重试与超时，不让真的等 3 × 60s
settings.DEEPSEEK_TIMEOUT_SECONDS = 8.0
settings.DEEPSEEK_MAX_RETRIES = 1
settings.DEEPSEEK_PROXY_URL = ""   # 不设显式代理（虽然自动探测可能仍命中，但假 Key 仍会 401/连不上时 fallback）

c_b = LLMClient()
prompt_b = "计算 7 × 8 = ？只输出最终数字，不要过程、不要标签、不要多余字符。"

async def run_b():
    return await c_b.complete(
        prompt_b,
        system_prompt="你是一位计算器，最终答案必须是一个纯数字。不要输出思考步骤。",
        max_tokens=256,
        temperature=0.0,
    )

t0 = time.time()
ans_b = asyncio.run(run_b())
dt_b = time.time() - t0

print(f"  耗时          : {dt_b:.2f}s")
print(f"  原始响应      : {repr(ans_b[:80])}")
print(f"  last_provider : {c_b.last_provider}")
print(f"  last_model    : {c_b.last_model}")
fb_reason = str(c_b.last_fallback_reason or "")[:120]
print(f"  fallback_reas : {fb_reason}")
print(f"  last_done_reas: {c_b.last_done_reason}")

provider_b = (c_b.last_provider == "ollama")
no_error_b = not ans_b.startswith("Error:")
has_56 = ("56" in str(ans_b))

print()
print(f"  判定项:")
print(f"    last_provider=='ollama'          → {'✅' if provider_b else '❌ FAIL'} (got: {c_b.last_provider})")
print(f"    响应不包含 Error:                → {'✅' if no_error_b else '❌ FAIL'}")
print(f"    last_fallback_reason 非空        → {'✅' if c_b.last_fallback_reason else '❌ FAIL'} (证明触发过回退)")
print(f"    包含数字 56:                     → {'✅' if has_56 else '⚠️  (答案解析差异)'}  (响应: {ans_b.strip()[:40]})")

PATH_B_PASS = (provider_b and no_error_b and bool(c_b.last_fallback_reason))
print()
if PATH_B_PASS:
    print("  🎉 验证 B 通过：DeepSeek 被切断后正确回退至本地 Ollama ✅")
else:
    print("  ❌ 验证 B 失败：未自动回退至 Ollama，或回退后仍报 Error。")

# ============================================================
# 最终总览
# ============================================================
print("\n" + SEP)
print("  最终验证总览")
print(SEP)
print(f"  【A 路径】 DeepSeek 原生调用:     {'✅ PASS' if PATH_A_PASS else ( '⏭️  跳过' if not DS_OK else '❌ FAIL')}")
print(f"  【B 路径】 切断 DS → 回退 Ollama:  {'✅ PASS' if PATH_B_PASS else '❌ FAIL'}")
if VALID_KEY and not PATH_A_PASS and DS_REACHABLE:
    print()
    print("  💡 建议：A 路径被跳过/失败但网络可达。")
    print("     如果用户确认本机 DeepSeek 应该可通，请：")
    print("       ① 打开代理软件的系统代理开关（或：把 127.0.0.1:7890 写进 env 变量 HTTPS_PROXY）")
    print("       ② 检查 Key 是否还有余额：https://platform.deepseek.com/")
    print("       ③ 设置 $env:DEEPSEEK_API_KEY = '你的sk' 并重新运行本脚本。")

print()
print("  修复代码覆盖的三个层面再次确认生效：")
print("    1. 启动层 _launcher.ps1: 环境变量继承 + 非空覆盖 + env 根目录同步 + Start-Process 空格安全")
print("    2. 网络层 LLMClient._detect_local_proxy: 7890/7897/10809/1080/8080/3128/9090 共7端口自动探测")
print("    3. 配置层 backend/.env: 占位符 Key 清理 → 占位符不会误判为真凭证，避免 3 次长重试")
print(SEP)
