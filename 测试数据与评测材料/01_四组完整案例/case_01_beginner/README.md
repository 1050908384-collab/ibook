# 案例A：大模型方向初学者（case_01_beginner）

## 一页说明
- **学习者**：掌握基础Python，无深度学习/Transformer经验
- **目标**：LoRA微调 | **每日**：1h | **风格**：理论+简单实操 | **硬件**：无独立GPU/8GB
- **差异点**：低显存或无GPU → 给出云端/替代实训方案；低正确率 → 降维解释

## 多阶段闭环
1. 阶段1（KB-04 PyTorch，初级）：生成→审核发现 gradient_flow_error→生成修订→审核复核通过→测试0.52（低分）→决策 simplify
2. 阶段2（KB-08 Transformer，中级）：依据新画像重新规划，拆分内容颗粒

## 审核修订闭环（验收标准要求）
专业知识与教学质量审核 Agent 发现 `gradient_flow_error` → 个性化学习资源生成 Agent 修订 zero_grad 顺序 → 复核通过（第2轮）

## 模型调用
- 全部 LLM 调用由 DeepSeek（deepseek-chat）完成，无回退
- 记录见 05_llm_calls.jsonl

## 截图
- 见 11_screenshots/（前端诊断页、学习包页、审核Trace页、阶段决策页）
