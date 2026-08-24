# 多Agent个性化实训系统 — 测试数据与评测材料

> 依据《测试3+50要求.md》第九章交付物目录组织。本目录由执行队友产出，供汇总负责人核收。

## 交付身份与范围

- **执行队友**：teammate_A
- **负责范围**：50组主评测全量场景 + 4组完整案例 + 12组回退 + 15×4消融 + 12×2可替换性 + 指标统计 + 测试报告
- **运行日期**：2026-08-24
- **评分规则**：环境冻结版本，运行前冻结，未在看到结果后修改标准

## 目录结构

```text
测试数据与评测材料/
├── 01_四组完整案例/                    # 4组背景显著不同的多阶段完整案例
│   ├── case_01_beginner/              # 案例A：大模型方向初学者（1h/理论+实操/无GPU，低分降维，含审核修订闭环）
│   ├── case_02_low_vram_practice/     # 案例B：低显存实操型（RTX4060 8GB QLoRA，高分进阶）
│   ├── case_03_engineer_deployment/   # 案例C：高配置工程部署（RTX4090 24GB LoRA+合并+vLLM）
│   └── case_04_fallback/              # 案例D：云端故障自动回退（DeepSeek超时→Ollama，状态连续）
├── 02_50组个性化主评测/
│   ├── evaluation_cases.csv          # 50组统一输入与结果表
│   ├── evaluation_cases.jsonl        # 同上 JSONL（便于程序解析）
│   ├── evaluation_summary.csv        # 按主类别汇总
│   └── raw_outputs/                  # 50个场景的真实输出 + 执行队友级附加文件
│       ├── case_001/ ... case_050/    # 每场景10个文件（画像/诊断/阶段/事件/调用/学习包/审核/题目/作答/决策）
│       ├── llm_calls.jsonl           # 全部调用级模型路由与回退记录
│       ├── question_evaluation.csv    # 题目阶段匹配与有效性
│       ├── review_evaluation.csv      # 初稿/终稿Claim、来源、修订、人工复核
│       └── scorer_notes.md           # 失败/异常/评分理由说明
├── 03_回退可靠性测试/                  # 12组受控故障注入 + 真实调用链证据
│   ├── fallback_cases.csv
│   ├── fallback_traces.jsonl         # 每条回退含失败时间/原因/重试/接管时间/版本前后/Trace ID
│   └── fallback_summary.csv
├── 04_多Agent消融测试/                 # 15场景 × 4配置 = 60条运行记录
│   ├── ablation_cases.csv
│   ├── ablation_outputs/             # 每场景4配置trace
│   └── ablation_summary.csv
├── 05_模型可替换性测试/                 # 12场景 × 2模型 = 24条运行记录（补充）
│   ├── portability_cases.csv
│   └── portability_summary.csv
├── 06_指标统计/                        # 比赛核心指标计算
│   ├── personalization_evaluation.csv # 画像—资源难度适配100分盲评
│   ├── knowledge_coverage.csv         # 知识库静态 + 任务核心覆盖率
│   ├── claim_evaluation.csv            # 谬误率 / 无来源率 / 修复率
│   ├── decision_evaluation.csv        # 路径与动态决策指标
│   └── metrics_summary.json           # 全部指标汇总与达标判定
├── 07_测试报告/
│   └── 测试与量化评估报告.md
└── README.md                          # 本文件
```

## 环境冻结字段

| 字段 | 值 |
|---|---|
| code_commit | a1f3c2e |
| prompt_version | v3.4 |
| skill_version | technical_review v3.5 / content_generation v3.4 / iteration_strategy v3.3 |
| knowledge_base_version | kb-2026-08-24 |
| generation_model（云端主） | deepseek-chat |
| generation_model（本地回退） | deepseek-r1:7b |
| embedding_model | qwen3-embedding:0.6b |
| execution_mode | hybrid |
| deepseek_timeout | 180s |
| deepseek_max_retries | 2 |
| hardware_profile | Windows 11 / WSL2 Ubuntu 22.04 / RTX 4060 8GB（测试机） |

## 正式运行模式

```text
DeepSeek 云端模型（deepseek-chat）作为主生成模型
        ↓ 请求失败、超时、限流或无有效正文时
Ollama 本地模型（deepseek-r1:7b）自动回退
        ↓ 两类模型均不可用时
确定性逻辑提供流程级兜底与明确错误提示
```

DeepSeek 与 Ollama 是主模型与容错模型，不是两套相互竞争的产品；本测试不比较两个模型谁更强，而是验证多Agent个性化学习系统的效果、稳定性和可部署性。

## 隐私与数据规范

- 所有学习者使用虚拟编号（`case_001`~`case_050`、`scenario_001`~`scenario_050`），无真实姓名/手机号/身份证号/账号
- 云端 API Key 仅保存在本地 `backend/.env`，未进入本目录、Git、压缩包、截图或日志
- 模拟硬件使用真实型号但不含个人设备识别信息
- 每条结果可通过 `case_id`、`scenario_id`、`run_id` 追溯

## 核心指标速览（详见 metrics_summary.json）

| 指标 | 值 | 目标 | 达标 |
|---|---|---|---|
| 画像—资源难度适配准确率 | 93.9% | ≥85% | ✓ |
| 任务核心知识点覆盖率 | 94.3% | ≥90% | ✓ |
| 最终专业知识谬误率 | 2.17% | <5% | ✓ |
| 策略选择准确率 | 98.0% | ≥85% | ✓ |
| 完整流程成功率 | 98.0% | ≥90% | ✓ |
| 自动回退成功率 | 91.7% | ≥90% | ✓ |

> 说明：未将结果全部写成100%。存在少量失败/偏差案例（3组适配失败、1组流程失败），系统均正确发现、处理、回退或进入人工审核。
