# 完整展示案例 B：低显存 QLoRA 实操

来源：真实主评测场景 case_016（新 50 组主评测，2026-08-30 批次）
run_id：case_016-1788081274
运行时间：2026-08-30T09:14:34 ~ 2026-08-30T09:14:59 (UTC)
执行路径：deepseek_only

## 用户画像
- 目标：QLoRA微调（熟悉 PyTorch/Transformer，想在 8GB 卡上微调）
- 基础：深度学习框架PyTorch基础、Transformer架构（已掌握 KB-04、KB-08）
- 硬件：RTX 4060 8GB（低显存约束，学习包需按 8GB 给出 QLoRA 量化配置）
- 风格：边学边做 / 每日 2.0 小时

## 阶段闭环结果
- 阶段1：KB-02（Python）补齐前置，8 题，正确率 87.5%，策略 advance

## 文件说明
- `input.json` - 评测案例输入（用户原话）
- `01_learner_profile.json` - 真实 Agent1 学情画像
- `02_diagnostic_output.json` - 诊断结果
- `03_stage_planning_output.json` - 阶段规划
- `06_learning_package.md` - 真实 Agent3 生成 + Agent4 审核后的学习包
- `07_review_trace.json` - 审核修订闭环（含 iterations、must_fix issue）
- `08_assessment_questions.json` - 阶段测评题
- `09_learner_answers.json` - 受控答题结果
- `10_stage_decision.json` - 真实 Agent6 决策（accuracy、next_strategy）
- `run_manifest.json` / `run_summary.json` - 运行清单与结果摘要
- `generation_events.jsonl` / `llm_calls.jsonl` - 生成事件与真实 LLM 调用记录
- `11_screenshots/` - 前端界面截图

本案例所有数据文件从 `02_50组个性化主评测/raw_outputs/case_016/` 中抽取。
