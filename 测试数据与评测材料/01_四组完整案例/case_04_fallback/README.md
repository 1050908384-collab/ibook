# 案例D：云端故障自动回退（case_04_fallback）

## 一页说明
- **学习者**：熟悉Python/PyTorch/数据处理，首次接触PEFT（与B/C不同画像）
- **目标**：LoRA微调 | **硬件**：RTX 3060 12GB
- **核心验证**：DeepSeek超时后Ollama自动接管，会话/内容块/版本/审核Trace连续

## 回退链（真实调用证据）
1. block_s1_01 诊断：DeepSeek ✓
2. block_s1_02 生成：DeepSeek ✓
3. block_s1_03 生成：DeepSeek 超时（retry 2）→ Ollama（deepseek-r1:7b）接管 ✓
4. 审核 block_s1_03：DeepSeek 超时 → Ollama 接管 ✓（审核Trace ID 保持 trace_case_04_fallback_s1）
5. 阶段决策：Ollama 完成 ✓

## 状态连续性
- session_id 保留 | 内容块编号 01→04 连续 | 内容版本 v1 连续 | 审核Trace ID 保留 | 无重复内容

## 模型调用
- 5 次调用：2 次 DeepSeek 成功，3 次 Ollama 回退成功，0 次失败
- 记录见 05_llm_calls.jsonl（含 routing_reason / fallback_reason / continuity_evidence）
