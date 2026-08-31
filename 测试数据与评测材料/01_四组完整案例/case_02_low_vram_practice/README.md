# 完整展示案例 B：低显存 QLoRA 实操

来源：真实主评测场景 case_016（新 50 组主评测，2026-08-31 后端修复后复跑批次）
run_id：run_20260831_105150
运行时间：2026-08-31T02:51:54 ~ 2026-08-31T02:52:34 (UTC)
执行路径：deepseek_only
LLM 调用：64 次（无回退）
复跑说明：本案例为展示材料配套重跑，后端代码与 8/30 50 组主批次相比做了多项链路修复；其余 47 组主评测仍保持 8/30 批次。

## 用户画像
- 目标：QLoRA微调（熟悉 Python/PyTorch/Transformer，想在 8GB 卡上做 QLoRA）
- 基础：深度学习框架PyTorch基础、Transformer架构（Agent1 识别为已掌握 KB-04、KB-08）
- 硬件：RTX 4060 8GB（低显存约束，学习包需按 8GB 显存给出 QLoRA 量化与 batch_size 建议）
- 风格：边学边做 / 每日 2.0 小时

## 阶段闭环结果
- 阶段1：KB-02（Python 补齐前置），8 题，正确率 100.0%，策略 advance

## 文件说明
- `input.json` - 评测案例输入（用户原话 + 受控答题脚本）
- `01_learner_profile.json` - 真实 Agent1 学情画像
- `02_diagnostic_output.json` - 诊断结果
- `03_stage_planning_output.json` - 阶段规划
- `06_learning_package.md` - 真实 Agent3 生成 + Agent4 审核后的学习包
- `07_review_trace.json` - 审核修订闭环（含 iterations、must_fix issue）
- `08_assessment_questions.json` - 阶段测评题
- `09_learner_answers.json` - 受控答题结果（按 answer_script_stage_1 的 high 档答题）
- `10_stage_decision.json` - 真实 Agent6 决策（accuracy、next_strategy）
- `run_manifest.json` / `run_summary.json` - 运行清单与结果摘要
- `generation_events.jsonl` / `llm_calls.jsonl` - 生成事件与真实 LLM 调用记录
- `11_screenshots/` - 前端界面截图（仅说明 README，无 PNG）

本案例所有数据文件与 `02_50组个性化主评测/raw_outputs/case_016/` 同步，为同一 8/31 批次复跑产物。
