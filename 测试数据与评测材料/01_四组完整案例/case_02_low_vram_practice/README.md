# 完整展示案例 B：低显存 QLoRA 实操

来源：50组主评测场景 case_016
验证定位：提交材料重点用户画像验证（低显存QLoRA实操）
执行路径：deepseek_only
LLM 调用：66 次

## 用户画像
- 目标：QLoRA微调，已有 Python、PyTorch、Transformer 基础，希望在8GB 显存设备上完成实操。
- 基础：深度学习框架 PyTorch 基础、Transformer 架构（Agent1 识别为已掌握 KB-04、KB-08）
- 硬件：RTX 4060 8GB；学习包按8GB 显存给出 QLoRA量化和 `batch_size` 建议。
- 风格：边学边做 / 每日2.0 小时

## 阶段闭环结果
- 阶段1：KB-02（Python 补齐前置），8题，正确率87.5%，策略 `advance`

## 文件说明
- `input.json`：评测案例输入和受控答题脚本
- `01_learner_profile.json`：Agent1 学情画像
- `02_diagnostic_output.json`：诊断结果
- `03_stage_planning_output.json`：阶段规划
- `06_learning_package.md`：Agent3 生成并经 Agent4 审核的学习包
- `07_review_trace.json`：审核修订闭环
- `08_assessment_questions.json`：阶段测评题
- `09_learner_answers.json`：受控答题结果
- `10_stage_decision.json`：Agent6 阶段策略决策
- `run_manifest.json`、`run_summary.json`：运行清单和结果摘要
- `generation_events.jsonl`、`llm_calls.jsonl`：生成事件和模型调用记录
- `11_screenshots/`：前端界面截图

本目录为 case_016 的完整证据展示，覆盖低显存硬件约束下的画像诊断、前置补齐、个性化学习包、阶段测评和策略决策。