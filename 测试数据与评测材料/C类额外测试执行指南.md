# 额外测试（C 类）执行指南 —— 自动化脚本 + 手动验证双路径

> 适用：已完成 A（环境配置）+ B（50 组主评测执行），
> 现在需完成：关键回退可靠性验证、消融测试（如需展示）、4 组完整案例。

---

## 一、一键全部跑（推荐先试单条）

```bash
# 第 0 步：语法预检（所有脚本已经过 py_compile）
python scripts\run_fallback_tests.py      --dry
python scripts\run_ablation_tests.py      --dry

# 第 1 步：先各取 1 条做冒烟
python scripts\run_fallback_tests.py      --types api_key_empty
python scripts\run_ablation_tests.py      --cases case_002 --configs Full-System

# 第 2 步：全部（顺序，预计耗时 6~12 小时）
python scripts\run_fallback_tests.py
python scripts\run_ablation_tests.py

# 第 3 步：抽取 4 组完整案例（依赖 50 组主评测已完成）
python scripts\extract_complete_cases.py
```

---

## 二、自动化脚本清单

| 脚本 | 路径 | 说明 | 输出目录/文件 |
|---|---|---|---|
| 回退可靠性 12 条 | [run_fallback_tests.py](file:///d:/code/ai_agent-trainer/scripts/run_fallback_tests.py) | 受控故障注入（env patch + httpx MockTransport） | `03_回退可靠性测试/fallback_cases.csv`、`fallback_traces.jsonl` |
| 消融 15×4=60 条 | [run_ablation_tests.py](file:///d:/code/ai_agent-trainer/scripts/run_ablation_tests.py) | Direct-LLM / RAG-only / No-Agent4 / Full-System 四种配置 | `04_多Agent消融测试/ablation_cases.csv` + `ablation_outputs/*.json` |
| 完整案例抽取 A/B/C/D | [extract_complete_cases.py](file:///d:/code/ai_agent-trainer/scripts/extract_complete_cases.py) | 从主评测 raw_outputs 抽取 + D受控故障注入 | `01_四组完整案例/case_01~case_04` |
| 汇总指标 | [calculate_real_metrics.py](file:///d:/code/ai_agent-trainer/scripts/calculate_real_metrics.py) | 读取真实 raw_outputs 自动计算画像/覆盖/策略/谬误率 | `06_指标统计/` 5 个文件 |

---

## 三、12 类回退故障：哪些自动注入，哪些需手动

`run_fallback_tests.py` 覆盖的自动注入方法与手动验证替代路径如下：

| 编号 | 故障类型 | 自动注入法 | ⚠️ 是否需要人工二次验证 | 手动操作步骤（替代/验证） |
|---|---|---|---|---|
| F01 | api_key_empty | 环境变量覆盖 `DEEPSEEK_API_KEY=""` | 建议 | ①把 `backend/.env` 中 `DEEPSEEK_API_KEY` 改成空或 `your-deepseek-api-key`；②启动后端；③前端提交画像输入；④观察日志出现 `Falling back to local Ollama`；⑤学习包仍正常生成 |
| F02 | http_401_auth_fail | `MockTransport` 返回 401 | **必须手动确认** | ①把 `DEEPSEEK_API_KEY` 改成**已作废 key**（`sk-invalid-xxx`）；②运行 case_001；③在 `llm_calls.jsonl` 中确认 401 → 回退 Ollama；④如果手头没有失效 key，就认定 MockTransport 结果有效 |
| F03 | http_403_denied | `MockTransport` 返回 403 | 建议 | 改 `.env` 的 `DEEPSEEK_API_BASE` 指向一个会返 403 的反向代理（若没有就相信 Mock） |
| F04 | http_404_invalid_model | `MockTransport` 返回 404 | 建议 | 改 `.env` 的 `DEEPSEEK_MODEL=deepseek-model-that-not-exist-999b`；真实请求 DeepSeek 会实际返 404 |
| F05 | http_429_rate_limit | `MockTransport` 返回 429 | 可选 | 不好手动复现，直接相信 Mock；脚本里会重试 2 次后转 Ollama |
| F06 | connect_timeout | `MockTransport` 抛 `ConnectTimeout` | **必须手动确认** | ①改 `.env`：`DEEPSEEK_API_BASE=http://10.255.255.1:65534`（不可达地址）；②再跑 1 个 case；③观察日志出现 `transport failure` → 回退 Ollama；④最后把 `DEEPSEEK_API_BASE` 还原 |
| F07 | read_timeout | `MockTransport` 抛 `ReadTimeout` | 建议 | 真实环境用代理 + sleep 模拟；或相信 Mock |
| F08 | http_500_server | `MockTransport` 返回 500/502/503 | 可选 | 相信 Mock |
| F09 | empty_content_body | `MockTransport` 返回 200 仅含 `reasoning_content`，无 `content` | 建议 | Mock 已经完全等价真实；脚本会走“简短重试指令”分支 |
| F10 | ollama_model_unavailable | `OLLAMA_MODEL=nonexistent-model:999b` | **必须手动确认** | ①`ollama list` 确认你没有 `nonexistent-model:999b`；②把 `.env` 改成 `LLM_PROVIDER=ollama` + `OLLAMA_MODEL=nonexistent-model:999b` + `DEEPSEEK_API_KEY=""`；③运行 case；④记录输出错误类型 |
| F11 | ollama_down | `OLLAMA_BASE_URL=http://127.0.0.1:1`（非法端口） | **必须手动确认** | ①`ollama stop`（或结束进程）；②再运行 case_001；③确认失败行为 + 状态连续性；④之后 `ollama serve` 启动回正常 |
| F12 | all_models_unavailable | 以上两者叠加 | 必须手动叠加 F11 | 按 F06 + F11 手动都做一遍；**正确结果**：系统明确返回失败 + 提示人工处理，不伪造成功 |

### 🔴 手动验证的硬性要求（比赛验收必查）

《测试数据建议方法》§9 第 3 条：

> 每条记录必须真实走回退代码，记录故障注入方式、实际接管模型、重试次数、流程最终状态和状态连续性。

所以即便用了 MockTransport，也请**至少对 F02/F06/F10/F11/F12 做 1 次真实手动验证**，
在 `fallback_traces.jsonl` 末尾追加一条 `inject_method: "manual_real_network"` 的记录。
脚本已在 CSV 中预留 `inject_method` 字段，手动结果只需按同样列名追加即可。

---

## 四、4 种消融配置：自动已实现，手动校验点

`run_ablation_tests.py` 通过以下方式保证四种配置**真实生效**，不是“打标签糊弄”：

| 配置 | 生效机制 | 手动验证点（抽查 1 个 case） |
|---|---|---|
| Direct-LLM | 直接实例化 `LLMClient`，Harness/Agent 永不加载 | 打开 `ablation_outputs/{case}_Direct-LLM.json`，确认 `used_rag=false, used_review=false, used_planning=false`，且 `llm_calls` 只有 1 条 |
| RAG-only | 只实例化 `HybridContentAgent`，内部会走 KB 查询 + 个性化，完全绕过 Agent1/4/6 | JSON 中 `used_rag=true, used_review=false, used_planning=false` |
| No-Agent4 | `Harness` 创建后 monkey-patch `_review_learning_package_pipeline`，把审核流程替换为“直接 approve + 空 history” | JSON 中 `review_iterations` 恒为 0，`review_passed` 恒为 true，`package_status=approved` |
| Full-System | 标准 Harness，不做任何 patch | JSON 中 `review_iterations ≥ 1` 或 `used_review=true`，`next_strategy` 为真实 Agent6 输出 |

**手动抽查操作**：运行完后任选 1 个 case，比对 4 份 JSON 的 `review_iterations`、`llm_calls.len`、`package_chars` 这三项：
- Direct-LLM 的 `llm_calls` 应最少（1 条）
- No-Agent4 的 `review_iterations` 必须为 0，Full-System 通常 ≥ 1
- 四者的 `package_chars` 差异应明显

---

## 五、4 组完整案例：自动抽取 + 手动截图填充

`extract_complete_cases.py` 自动从主评测结果拷贝 8 + 2 标准文件（详见脚本），但有两部分必须**手动**：

### 6.1 前端界面截图（必须手动）

每个案例目录下有 `11_screenshots/`，请跑真实前端（`frontend`）依次截图：

| 顺序 | 截图内容 | 触发操作 |
|---|---|---|
| 1 | 画像录入页 | `npm run dev` 进入 Onboarding 页 |
| 2 | 学习计划确认页（Agent1 输出） | 填写画像后点“生成学习计划” |
| 3 | 学习包 + 审核修订侧边栏 | 启动阶段后，浏览学习包正文 + 右侧 Agent4 issue 列表 |
| 4 | 测评题答题 | 做题界面 |
| 5 | 阶段结果页（Agent6 输出） | 提交答案后的“准确率/下一步策略” |
| 6（仅 D） | 控制台显示回退日志 | 浏览器 DevTools Console 或后端终端 |

### 6.2 四案例 D 的 README 必须写明（脚本已自动写）

```text
source_main_case_id: case_018
run_type: controlled_fallback_replay
```

**严禁写成“云端故障自然发生”**，脚本 README 已明确标注“受控故障注入测试”。

---

## 六、全部完成后的运行清单（核对表）

```
☐ 01_四组完整案例/ case_01, case_02, case_03, case_04 四个目录齐全，每目录 9 文件 + 11_screenshots + README
☐ 03_回退可靠性测试/ fallback_cases.csv 正好 12 行；fallback_traces.jsonl 有 12 条证据
☐ 04_多Agent消融测试/ ablation_cases.csv 正好 60 行；ablation_outputs/ 60 个 JSON
☐ 06_指标统计/ 5 个文件齐全（personalization/question/claim/decision/metrics_summary）
☐ 07_测试报告/ 测试与量化评估报告.md 基于 metrics_summary.json 真实数据填写（无预设）
☐ 对 F02/F06/F10/F11/F12 五关键故障有真实手动验证，fallback_traces.jsonl 追加了 manual_real_network 记录
☐ 抽查 1 组消融：4 份 JSON 的 review_iterations/llm_calls_len/package_chars 差异明显
☐ 4 案例截图齐全，D 案例 README 明确写 controlled_fallback_replay + 来源映射
```
