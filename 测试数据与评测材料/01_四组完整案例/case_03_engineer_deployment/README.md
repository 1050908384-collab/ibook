# 案例C：高配置工程部署型（case_03_engineer_deployment）

## 一页说明
- **学习者**：熟悉Transformer/HF/训练，目标LoRA微调+权重合并+vLLM部署
- **每日**：3h | **风格**：工程实操和故障排查 | **硬件**：RTX 4090 24GB
- **差异点**：不重复基础；优先参数设计/合并/部署；测试题难度偏高；高分→进阶部署

## 多阶段闭环
1. 阶段1（KB-12 LoRA+合并，高级）：r=16设计+merge_and_unload→审核通过→测试0.92→决策 advance
2. 阶段2（KB-15 vLLM部署）：PagedAttention配置+部署排障

## 模型调用
- 全部由 DeepSeek（deepseek-chat）完成，无回退
