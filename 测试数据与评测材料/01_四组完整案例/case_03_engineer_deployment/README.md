# 完整展示案例 C：工程师全链路部署

来源：真实主评测场景 case_026（新 50 组主评测，2026-08-30 批次）
run_id：case_026-1788082370
运行时间：2026-08-30T09:32:51 ~ 2026-08-30T09:33:17 (UTC)
执行路径：deepseek_only

## 用户画像
- 目标：LoRA微调+权重合并+vLLM部署（全链路工程目标）
- 基础：深度学习框架PyTorch基础、Transformer架构、HuggingFace生态工具链、PyTorch进阶训练技巧（已掌握 KB-04、KB-08、KB-11、KB-13）
- 硬件：RTX 4090 24GB
- 风格：工程实操和故障排查 / 每日 3.0 小时

## 阶段闭环结果
- 阶段1：KB-02（Python）补齐前置，8 题，正确率 62.5%，策略 reinforce（与预期策略一致）

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

本案例所有数据文件从 `02_50组个性化主评测/raw_outputs/case_026/` 中抽取。
