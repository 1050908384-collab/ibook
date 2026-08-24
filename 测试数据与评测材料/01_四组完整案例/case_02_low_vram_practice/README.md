# 案例B：低显存实操型（case_02_low_vram_practice）

## 一页说明
- **学习者**：熟悉Python/PyTorch/Transformer，目标QLoRA微调
- **每日**：2h | **风格**：边学边做 | **硬件**：RTX 4060 8GB
- **差异点**：跳过已掌握基础；8GB显存→QLoRA 4-bit小模型方案；高分→进阶

## 多阶段闭环
1. 阶段1（KB-12 LoRA/QLoRA，中级）：8GB显存QLoRA方案→审核通过→测试0.88（高分）→决策 advance
2. 阶段2（KB-13 PyTorch进阶）：梯度累积/AMP配合QLoRA

## 模型调用
- 全部由 DeepSeek（deepseek-chat）完成，无回退
