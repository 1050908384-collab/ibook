# 完整展示案例 D：自动回退（受控故障注入）

来源主评测场景：case_044
验证定位：提交材料重点用户画像验证（受控故障注入可靠性）
测试类型：受控故障注入测试
控制清单：`controlled_run_manifest.json`（`case_id=case_04_fallback`，`fault_type=connect_timeout`，`inject_method=mock_transport_timeout`）

## 说明
本案例用于验证系统在云端模型链路不可用时的自动回退、流程完成状态和状态连续性，不作为自然发生故障的案例陈述。

## 注入方式
- `fault_type`：`connect_timeout`
- `inject_method`：`mock_transport_timeout`
- 故障描述：使 DeepSeek 云端链路的传输层不可达。

## 预期
- 云端 DeepSeek 调用失败后，系统自动切换至本地 Ollama。
- 回退到可用模型后流程仍完成；若云端和本地模型均不可用，系统明确返回失败并提示人工处理。
- 学情画像和阶段规划保持连续，不因回退重置。

## 实际结果
- 实际接管模型：`deepseek-r1:7b`、`qwen2.5:7b`
- 回退触发：`True`，详见 `fallback_trace.jsonl`
- 流程最终状态：`success`
- 状态连续性：`True`
- 受控主流程：阶段1 为 KB-01“环境与命令行操作”，8题，正确率100%，策略 `advance`

目录内 `run_manifest.json` 和 `run_summary.json` 是来源主评测 case_044 的基线运行记录；回退结论仅以 `controlled_run_manifest.json` 和 `fallback_trace.jsonl` 为准。
