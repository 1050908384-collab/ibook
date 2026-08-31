# 完整展示案例 A：双阶段初学者歧义输入

来源：真实主评测场景 case_046（新 50 组主评测，2026-08-31 后端修复后复跑批次）
run_id：run_20260831_104803
运行时间：2026-08-31T02:48:13 ~ 2026-08-31T02:49:55 (UTC)
执行路径：deepseek_only
LLM 调用：108 次（无回退）
复跑说明：本案例为展示材料配套重跑，后端代码与 8/30 50 组主批次相比做了多项链路修复；其余 47 组主评测仍保持 8/30 批次。

## 用户画像
- 目标：LoRA微调（用户输入写"LoRa"，存在与物联网 LoRa 的歧义，系统按学习路径上下文判定为低秩适配 LoRA）
- 基础：Python编程基础（Agent1 识别为已掌握 KB-02）
- 硬件：RTX 3060 12GB
- 风格：理论型 / 每日 1.0 小时

## 阶段闭环结果
- 阶段1：KB-04（PyTorch 深度学习框架），8 题，正确率 87.5%，策略 advance
- 阶段2：正确率 50.0%，策略 simplify（多阶段案例，run_stage_2=True；对未达掌握阈值的内容进一步降难度拆分）

## 文件说明
- `input.json` - 评测案例输入（用户原话 + 受控答题脚本）
- `01_learner_profile.json` - 真实 Agent1 学情画像
- `02_diagnostic_output.json` - 诊断结果
- `03_stage_planning_output.json` - 阶段规划
- `06_learning_package.md` - 真实 Agent3 生成 + Agent4 审核后的学习包（阶段1）
- `07_review_trace.json` - 审核修订闭环（含 iterations、must_fix issue）
- `08_assessment_questions.json` - 阶段测评题（阶段1）
- `09_learner_answers.json` - 受控答题结果（阶段1，按 answer_script_stage_1 的 medium 档答题）
- `10_stage_decision.json` - 真实 Agent6 决策（accuracy、next_strategy）
- `run_manifest.json` / `run_summary.json` - 运行清单与结果摘要
- `generation_events.jsonl` / `llm_calls.jsonl` - 生成事件与真实 LLM 调用记录
- `stage_2_extra/` - 阶段2 全套证据（画像快照、规划、学习包、测评、决策）
- `11_screenshots/` - 前端界面截图

本案例所有数据文件与 `02_50组个性化主评测/raw_outputs/case_046/` 同步，为同一 8/31 批次复跑产物。
