# 50组个性化学习包人工审核结论

## 审核范围

- 数据源：`测试数据与评测材料/02_50组个性化主评测/raw_outputs/case_001` 至 `case_050`。
- 对照材料：`evaluation_cases.csv`、`learner_profile.json`、`stage_planning_output.json`、`learning_package.md`、`review_trace.json`，并按审核手册核对题目与终止状态。
- 记录数：50 个案例 × 5 项 = 250 条。
- 判定值：仅使用 `pass`、`fail`、`error`；每条均写入可定位证据。

## 结果

| 条件 | pass | fail | error |
| --- | ---: | ---: | ---: |
| target_path | 16 | 34 | 0 |
| difficulty | 22 | 28 | 0 |
| time_budget | 28 | 22 | 0 |
| hardware | 0 | 50 | 0 |
| learning_style | 4 | 46 | 0 |
| 合计 | 70 | 180 | 0 |

画像-资源难度适配准确率 = `70 / 250 = 28.0%`，低于目标 `>=85%`，差距 57 个百分点。

## 终止/空阶段专项

以下 22 个案例的 `stage_planning_output.json` 为 `subtopic_ids=[]`、`estimated_hours=0`、标题“已完成全部知识点”，但画像仍有未掌握知识点，且继续生成了学习包和 8 道题：

`case_023`、`024`、`025`、`026`、`032`、`033`、`035`、`036`、`037`、`038`、`039`、`040`、`041`、`042`、`043`、`044`、`045`、`046`、`047`、`048`、`049`、`050`。

这些案例在 `terminal_stage_review.csv` 中均为 `fail`；按本次要求仍保留在 250 条普通阶段记录中，因此五项均按实际证据判为 `fail`，没有使用 `not_applicable`。

## 主要问题

1. 阶段规划经常偏离学习目标或前置路径，例如 LoRA 目标被安排到 Python、PyTorch、HuggingFace 或 Transformer 阶段。
2. 多数案例的学习包要求 CUDA/GPU、模型下载或本地推理，但画像硬件不足、未提供可执行替代方案，因此硬件项 50/50 失败。
3. 学习风格字段与内容组织不一致：理论型、混合型学习者仍大量收到以命令和实操为主的包；实操型学习者也未稳定获得可运行、可验证的任务链。
4. Agent4 的 `review_passed=true` 不能替代画像适配人工判定；审核轨迹证明“审核过”，不证明五项适配成立。

## 交付文件

- `tmp/finalized_personalization.csv`：250 条完整逐条件审核记录。
- `测试数据与评测材料/06_指标统计/terminal_stage_review.csv`：22 条终止状态专项记录（原表已存在，未改动原始输出）。

本轮只完成用户优先要求的 85% 画像-资源适配审核；150 条技术事实核验表尚未据此结论填充。
