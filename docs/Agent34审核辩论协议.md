# 个性化学习资源生成 Agent/专业知识与教学质量审核 Agent 审核辩论协议 v0.2

本文档定义 个性化学习资源生成 Agent 与 专业知识与教学质量审核 Agent 在“学习包纠错机制”中的结构化协议，并与 `project.db` 中的审核辩论表保持一致。

## 1. 目标

- 让 专业知识与教学质量审核 Agent 的审核从“单向报错”升级为“带证据的可追溯审核”
- 让 个性化学习资源生成 Agent 的修改从“整包重写”升级为“逐条答辩 + 逐条修改”
- 让 Orchestrator 能记录审核闭环，并在超轮次时返回 `needs_human_review`

## 1.1 当前学习包块级流水线（生效规则）

学习包处理的最小单位是 `content_block_id` 对应的内容块，而不是整篇文章或整包教程。`knowledge_chunks` 中的原始块由 个性化学习资源生成 Agent 逐块个性化，专业知识与教学质量审核 Agent 逐块审核，最终再由 Orchestrator 按原顺序聚合为一个 `LearningPackage`。

1. **首轮生产/审核**：个性化学习资源生成 Agent 最多并发生成 2 个块；每个块生成完成后立即提交 专业知识与教学质量审核 Agent。专业知识与教学质量审核 Agent 使用独立的并发上限 2，不占用 个性化学习资源生成 Agent 的生成槽，因此 个性化学习资源生成 Agent 可以连续生产后续块。
2. **首轮问题交付**：专业知识与教学质量审核 Agent 完成全部块的首轮审核后，Orchestrator 构造一次完整 `issue_package`，按 `content_block_id` 分组，整体交给 个性化学习资源生成 Agent。首轮没有问题的块不进入列表。
3. **修改/复核**：个性化学习资源生成 Agent 建立待修改块队列，最多并发修改 2 个块；每个块修改完成后立即交给 专业知识与教学质量审核 Agent 复核，其他块的修改不等待该复核完成。回传对象只包含该块、递增后的 `content_version` 和 `revision_note`。
4. **后续问题列表**：每轮复核结束后重新构造完整问题列表，只保留仍未解决的问题块，再整体交给 个性化学习资源生成 Agent。问题列表最多交付 3 次；仍未解决的块进入 `needs_human_review`。

上述块级规则只适用于学习包；用户问题仍沿用本协议的回答审核链路。若阶段没有可用 `knowledge_chunks`，运行时才回退到旧的整包流程。

## 2. 核心对象

### 2.1 Source

Source 由系统侧统一生成，个性化学习资源生成 Agent 负责引用和绑定，专业知识与教学质量审核 Agent 负责校验。

```json
{
  "source_id": "src_kd_001",
  "source_type": "knowledge_doc",
  "title": "LoRA 微调实践",
  "locator": "knowledge_docs:doc_123",
  "content_ref": "chunk:doc_123:5",
  "source_version": "2026-07-31",
  "metadata_json": {
    "block_id": "KB-003"
  }
}
```

字段约束：

- `source_id`：在单个 `trace` 内唯一（**非全局唯一**，数据库主键为复合键 `(trace_id, source_id)`）
- `source_type`：`knowledge_doc | best_practice | manual | external_note | code_example | video`
- `locator`：主定位符，支持数据库主键、文档路径、chunk 引用
- `content_ref`：更细粒度的内容锚点，可为空

### 2.2 Claim

Claim 是 个性化学习资源生成 Agent 对学习包内容做出的“可验证陈述声明”，是 专业知识与教学质量审核 Agent 审核的核心锚点。

```json
{
  "claim_id": "clm_001",
  "text": "LoRA 通过低秩矩阵分解减少微调参数量。",
  "source_ref": "src_kd_001",
  "source_span": "section=2.1",
  "section_ref": "tutorial:sec_2:p_1",
  "claim_type": "concept",
  "is_key_technical": true,
  "confidence": 0.92,
  "status": "active"
}
```

字段定义：

| 字段 | 类型 | 说明 |
|---|---|---|
| `claim_id` | `str` | 个性化学习资源生成 Agent 赋值；在单个 `trace` 内唯一（**非全局唯一**，数据库主键为复合键 `(trace_id, claim_id)`） |
| `text` | `str` | 原子化陈述内容 |
| `source_ref` | `str \| null` | 指向 `sources[].source_id`；若 `is_key_technical=true` 则不可空 |
| `source_span` | `str \| null` | 指向 source 内部片段，如章节、页码、chunk |
| `section_ref` | `str` | 指向 tutorial 或 code 的位置锚点 |
| `claim_type` | `Literal` | `fact | procedure | code | config | concept` |
| `is_key_technical` | `bool` | 是否为关键技术陈述 |
| `confidence` | `float` | 个性化学习资源生成 Agent 自评把握，范围 `0~1` |
| `status` | `Literal` | `active | revised | dropped` |

约束：

- Claim 只抽取“可验证、可争议、影响学习结果”的陈述
- `section_ref` 统一格式，例如：
  - `tutorial:sec_2`
  - `tutorial:sec_3:p_2`
  - `code:example_1:line_3_12`

### 2.3 ReviewError 与 DebateIssue

`ReviewError` 是审核器发现的原子错误。`DebateIssue` 是围绕该错误展开的完整生命周期。

`ReviewError` 至少应包含：

```json
{
  "error_id": "a1b2c3d4e5f6_r1",
  "canonical_error_id": "a1b2c3d4e5f6",
  "claim_id": "clm_001",
  "error_type": "hallucination",
  "severity": "critical",
  "must_fix": true,
  "must_fix_reason": "关键技术陈述缺少有效来源",
  "location": "tutorial:sec_2:p_1",
  "description": "LoRA 原理描述缺少来源支撑",
  "evidence": "未命中任何来源片段",
  "suggestion": "补充来源并改写定义",
  "source_refs": ["src_kd_001"],
  "change_request": {
    "required": true,
    "action": "add_source_and_rewrite"
  }
}
```

说明：

- `error_id`：当前轮实例 ID
- `canonical_error_id`：跨轮稳定 ID
- `claim_id`：指向对应 Claim
- `must_fix`：流程约束，不等同于 `severity`

### 2.4 Error ID 生成规则

协议采用“内容哈希为主 + 当前轮实例后缀”的方式。

- `canonical_error_id = sha1(claim_id + error_type + normalized_location + normalized_description)[:12]`
- `error_id = canonical_error_id + "_r" + round_index`

要求：

- 专业知识与教学质量审核 Agent 不自行发明跨轮逻辑 ID
- 同一问题跨轮保留同一个 `canonical_error_id`
- `DebateIssue` 以 `canonical_error_id` 为主键语义（在 **单个 trace 语义下唯一**，数据库上是 `UNIQUE(trace_id, canonical_error_id)`）
- **归一化规则必须钉死**，否则双方各自哈希对不上：
  - `normalized_location`：小写 + 去首尾空格
  - `normalized_description`：小写 + 去中英文标点 + 折叠连续空白

## 3. 专业知识与教学质量审核 Agent -> 个性化学习资源生成 Agent

专业知识与教学质量审核 Agent 对每条 Issue 输出：

```json
{
  "issue_id": "iss_001",
  "canonical_error_id": "a1b2c3d4e5f6",
  "claim_id": "clm_001",
  "severity": "critical",
  "must_fix": true,
  "must_fix_reason": "关键技术陈述无有效来源",
  "description": "该陈述未绑定足够证据",
  "source_refs": ["src_kd_001"],
  "change_request": {
    "required": true,
    "action": "rewrite_with_supported_source"
  }
}
```

## 4. 个性化学习资源生成 Agent -> 专业知识与教学质量审核 Agent

个性化学习资源生成 Agent 对每条 Issue 必须逐条回复：

```json
{
  "canonical_error_id": "a1b2c3d4e5f6",
  "disposition": "partially_accepted",
  "reason": "已补充来源并重写定义，但保留原有示例",
  "patched_sections": [
    "tutorial:sec_2:p_1",
    "tutorial:sec_2:p_2"
  ],
  "source_refs": ["src_kd_001"]
}
```

`disposition` 允许值：

- `accepted`
- `rejected`
- `partially_accepted`

约束：

- `must_fix = true` 时，个性化学习资源生成 Agent 不允许直接 `rejected`
- 若认为必须驳回，需提交 `reason`，并由 Orchestrator 最终裁决

## 5. 专业知识与教学质量审核 Agent 复核输出

```json
{
  "canonical_error_id": "a1b2c3d4e5f6",
  "accepted_by_reviewer": true,
  "remaining_issue": false,
  "next_status": "resolved",
  "review_notes": "来源已补齐，定义已收敛"
}
```

## 6. Must Fix 派生规则

自动进入 `must_fix` 的条件：

- `severity = critical`
- 关键技术陈述缺少 `source_ref`
- 关键技术陈述的来源不支持该 Claim
- 代码不可执行
- 违反安全、规范或部署硬约束

专业知识与教学质量审核 Agent 可显式升级为 `must_fix` 的场景：

- 会误导学习路径
- 会导致代码、命令或配置无法复现
- 与当前用户画像严重不匹配

## 7. 超轮次与可用性

当达到最大辩论轮次后仍有未决 Issue：

- `trace.final_status = needs_human_review`
- `trace.package_status = provisional`
- 当前学习包允许继续展示给当前用户学习
- 但不进入正式知识沉淀或复用资产池

## 8. 数据库映射

| 协议对象 | 表 |
|---|---|
| Review Trace | `review_traces` |
| Source | `learning_package_sources` |
| Claim | `learning_package_claims` |
| Debate Issue | `review_debate_issues` |
| Debate Round | `review_debate_rounds` |
| Agent Response / Verdict | `review_debate_responses` |

### 8.1 ID 唯一性范围（关键：防止 个性化学习资源生成 Agent/4 同学误用全局唯一）

| ID 字段 | 唯一性范围 | 数据库约束 |
|---|---|---|
| `trace_id` | 全局唯一 | `trace_id PRIMARY KEY` |
| `source_id` | **单个 trace 内唯一**（非全局） | `PRIMARY KEY (trace_id, source_id)` |
| `claim_id` | **单个 trace 内唯一**（非全局） | `PRIMARY KEY (trace_id, claim_id)` |
| `canonical_error_id` | **单个 trace 语义下唯一**（非全局） | `UNIQUE (trace_id, canonical_error_id)` |
| `issue_id` / `round_id` / `response_id` | 全局唯一 | 各自列 `PRIMARY KEY` |

跨表引用时，**凡是涉及 `source_id` / `claim_id` / `canonical_error_id` 的引用，都必须携带 `trace_id` 作为复合约束**，否则跨 trace 的同名 ID 会发生串扰。
