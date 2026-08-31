# 完整展示案例 D：自动回退（受控故障注入）

source_main_case_id：case_044（主评测 case_044 用户画像作为受控回放的基础）
run_type：controlled_fallback_replay
控制清单：controlled_run_manifest.json（case_id=case_04_fallback，fault_type=connect_timeout，inject_method=mock_transport_timeout）
重跑时间：2026-08-31（与 8/30 主批次独立，基于修复后后端）

## 说明
本案例为**受控故障注入测试**，不是"自然发生的云端故障"。

## 注入方式
- fault_type：`connect_timeout`
- inject_method：`mock_transport_timeout`
- 故障描述：连接超时（人为让 DeepSeek 云端链路的 transport 层不可达）

## 预期
- 云端 DeepSeek 调用失败后，系统自动回退到本地 Ollama
- 回退到真实可用的 Ollama 模型后流程仍完成；若云端和本地模型均不可用，必须明确失败并提示人工处理
- 状态连续性：画像、阶段规划不中断（不因回退而重置）

## 实际结果
- 实际接管模型：deepseek-r1:7b → qwen2.5:7b（两次 Ollama 本地模型接管）
- 回退触发：True（fallback_triggered=True，详见 `fallback_trace.jsonl`）
- 流程最终状态：success（完整端到端闭环）
- 状态连续性：True（画像与阶段规划正常生成）
- 受控主流程部分（深端 44 基础画像非故障路径）：阶段1 KB-01 环境与命令行操作，8 题，正确率 100%，策略 advance

详见 `controlled_run_manifest.json`、`fallback_trace.jsonl`、`run_manifest.json` 和 `run_summary.json`。
