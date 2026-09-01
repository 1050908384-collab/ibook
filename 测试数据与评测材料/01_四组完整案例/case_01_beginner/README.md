# 完整展示案例 A：双阶段初学者歧义输入

来源：50组主评测场景 case_046
验证定位：提交材料重点用户画像验证（初学者双阶段）
执行路径：deepseek_only
LLM 调用：108 次

## 用户画像
- 目标：LoRA微调。用户输入写“LoRa”，存在与物联网 LoRa 的歧义，系统按学习路径上下文判定为低秩适配 LoRA。
- 基础：Python编程基础（Agent1 识别为已掌握 KB-02）
- 硬件：RTX 3060 12GB
- 风格：理论型 / 每日 1.0 小时

## 阶段闭环结果
- 阶段1：KB-04（PyTorch 深度学习框架），8题，正确率87.5%，策略 `advance`
- 阶段2：正确率50.0%，策略 `simplify`；系统对未达掌握阈值的内容进一步降难度拆分。

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
- `stage_2_extra/`：阶段2 完整证据
- `11_screenshots/`：前端界面截图

本目录为 case_046 的完整证据展示，覆盖画像诊断、阶段规划、学习包生成与审核、测评答题及策略决策的双阶段闭环。