# 完整展示案例 D：自动回退

source_main_case_id: case_044
run_type: controlled_fallback_replay

## 说明
本案例为**受控故障注入测试**，不是"自然发生的云端故障"。

## 注入方式
- fault_type: `connect_timeout`
- inject_method: `mock_transport_timeout`
- 故障描述: 连接超时

## 预期
- 云端 DeepSeek 调用失败后，系统自动回退到本地 Ollama
- 回退到真实可用的 Ollama 模型后流程仍完成；若云端和本地模型均不可用，必须明确失败并提示人工处理
- 状态保持连续性（画像、阶段规划不落空）

## 实际结果
- 实际接管模型: ['deepseek-r1:7b', 'qwen2.5:7b']
- 回退触发: True
- 流程最终状态: success
- 状态连续性: True

详见 `controlled_run_manifest.json` 和 `fallback_trace.jsonl`。
