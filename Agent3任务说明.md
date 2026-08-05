# Agent3 任务说明（作者侧产出与答辩）

> 本文档只描述 **Agent3 同学** 需要做的事。Agent4 的改动你不用管，Harness/DB/状态机由项目负责人处理。
> 最后更新：2026-07-31

---

## 一、开工前必读

| 文档 | 位置 | 重点看 |
|---|---|---|
| 审核辩论协议 | [`docs/Agent34审核辩论协议.md`](docs/Agent34审核辩论协议.md) | 2.1 Source、2.2 Claim、第 4 节 Agent3→Agent4、第 6 节 must_fix、第 7 节超轮次 |
| 数据库框架 | [`数据库框架.md`](数据库框架.md) 第九节 | `learning_package_sources`、`learning_package_claims` 两张表的字段和约束 |

---

## 二、你的职责（3 件事）

1. **首轮学习包必须带 `sources` + `claims`**：不能只返回 tutorial/code/questions
2. **对 Agent4 的每条 issue 逐条回复** `accepted / rejected / partially_accepted`
3. **修改时只改 `accepted` 的部分**，保留修改痕迹（claim 的 status 变更）

---

## 三、改动清单

### 1. 扩展 `backend/models/stage.py` 的 `LearningPackage`

**文件**：[`backend/models/stage.py`](backend/models/stage.py#L21)

当前定义：
```python
class LearningPackage(BaseModel):
    session_id: str
    stage_id: str
    block_id: str
    block_name: str
    tutorial: str = ""
    code_examples: List[dict] = Field(default_factory=list)
    questions: List[TestQuestion] = Field(default_factory=list)
    format: str = "markdown"
    version: int = 1
```

需要增加 3 个字段：
```python
    review_trace_id: Optional[str] = None                    # 关联 review_traces.trace_id
    sources: List[Source] = Field(default_factory=list)      # 来源清单
    claims: List[Claim] = Field(default_factory=list)        # 可验证陈述声明
```

### 2. 新增 `Source` 和 `Claim` 模型

建议在 `backend/models/stage.py` 同文件内定义（或新建 `backend/models/claim.py`）。

**Source 字段**（对齐协议 2.1 节 + 数据库 `learning_package_sources` 表）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `source_id` | `str` | ✅ | 在单个 trace 内唯一（**非全局**） |
| `source_type` | `Literal` | ✅ | `knowledge_doc \| best_practice \| manual \| external_note \| code_example \| video` |
| `title` | `str` | ❌ | 来源标题 |
| `locator` | `str` | ❌ | 主定位符（doc_id / chunk_id / 路径） |
| `content_ref` | `str` | ❌ | 细粒度内容锚点 |
| `source_version` | `str` | ❌ | 版本标识 |
| `metadata_json` | `dict` | ❌ | 附加元数据 |

**Claim 字段**（对齐协议 2.2 节 + 数据库 `learning_package_claims` 表）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `claim_id` | `str` | ✅ | 在单个 trace 内唯一（**非全局**） |
| `text` | `str` | ✅ | 原子化可验证陈述（不能是整段话） |
| `source_ref` | `str \| null` | ❌ | 指向 `sources[].source_id`；`is_key_technical=true` 时**不可空** |
| `source_span` | `str \| null` | ❌ | source 内部片段定位（章节、页码、chunk） |
| `section_ref` | `str` | ✅ | 指向 tutorial/code 位置，格式统一：`tutorial:sec_2` / `code:example_1:line_3_12` |
| `claim_type` | `Literal` | ✅ | `fact \| procedure \| code \| config \| concept` |
| `is_key_technical` | `bool` | ❌ | 是否关键技术陈述，默认 `false` |
| `confidence` | `float` | ❌ | Agent3 自评把握，范围 0~1，默认 0.5 |
| `status` | `Literal` | ❌ | `active \| revised \| dropped`，默认 `active` |

**强制约束**：
- `is_key_technical=true` 时 `source_ref` 不可空
- `confidence` 必须在 0~1 之间
- `text` 必须是"可验证陈述"，不能是整段话

### 3. 改 `backend/agents/agent_03_content_gen.py` 的 `generate_learning_package`

**文件**：[`backend/agents/agent_03_content_gen.py`](backend/agents/agent_03_content_gen.py#L26)

当前只返回 tutorial/code_examples/questions。需要改为：
- 生成 tutorial 后，**主动抽取 claims**（从内容里提取"可验证陈述"，绑定到 `section_ref`）
- 每个 claim 绑定 `source_ref`（指向 sources 里的某条）
- 返回值增加 `sources` 和 `claims` 字段

返回结构示例：
```python
return {
    "tutorial": tutorial,
    "code_examples": code_examples,
    "questions": questions,
    "format": "markdown",
    "sources": [...],   # 新增
    "claims": [...],    # 新增
}
```

### 4. 新增 `reply_to_review` 方法

**文件**：`backend/agents/agent_03_content_gen.py`（新增方法）

签名建议：
```python
async def reply_to_review(
    self,
    session_id: str,
    issues: List[Dict],          # Agent4 出具的 issues
    package: Dict,                # 当前学习包
    debate_history: List[Dict] = None   # 前几轮辩论记录（可选）
) -> List[Dict]:
    """对 Agent4 的每条 issue 逐条回复。"""
```

返回 `List[AgentReply]`，每条含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `canonical_error_id` | `str` | 对应的 issue ID |
| `disposition` | `Literal` | `accepted \| rejected \| partially_accepted` |
| `reason` | `str` | 修改说明或拒绝理由（**rejected 时必须提供**） |
| `source_refs` | `List[str]` | 拒改时的反证来源（**rejected 时必须提供**） |
| `patched_sections` | `List[str]` | accepted 时给出改后片段位置，如 `tutorial:sec_2:p_1` |

**边界规则（协议强制，不能违反）**：
- `must_fix=true` 的 issue：**只能 `accepted` 或 `partially_accepted`，不能 `rejected`**
- 若认为 `must_fix` 误判，只能提交 `challenge_reason`，由 Orchestrator 裁决，**不能自行跳过**
- `rejected` 必须带 `source_refs` 反证，否则无效

### 5. 改 `modify_package`

**文件**：[`backend/agents/agent_03_content_gen.py`](backend/agents/agent_03_content_gen.py#L44)

当前直接改 tutorial。需要改为：
- 只处理 `accepted` / `partially_accepted` 的 issue
- 修改时更新对应 claim 的 `status`：
  - 内容改写 → `status="revised"`
  - 内容删除 → `status="dropped"`
  - 新增的 claim → `status="active"`
- 保留 `review_trace_id` 不变（同一 trace 内迭代）
- `version` 递增

---

## 四、交互时序（你只看自己这侧）

```
1. Harness 调用你的 generate_learning_package()
   → 你返回带 sources + claims 的学习包

2. 【debate 子循环，最多 2 轮】
   Harness 调用你的 reply_to_review(issues)
   → 你对每条 issue 返回 accepted/rejected/partially_accepted + reason
   → Agent4 做二次复核（persist/withdraw/downgrade）
   → Orchestrator 裁决

3. Harness 调用你的 modify_package(accepted_issues)
   → 你只改 accepted 的部分，更新 claims status
```

---

## 五、禁止修改的文件

| 文件 | 原因 |
|---|---|
| `backend/agents/agent_04_tech_review.py` | Agent4 负责 |
| `backend/models/review.py` | Agent4 负责 |
| `backend/orchestrator/harness.py` | 项目负责人负责 |
| `backend/db/sqlite_client.py` | 项目负责人负责 |
| `backend/state/` | 项目负责人负责 |
| `backend/models/contracts.py` | 项目负责人负责（需要新 IO 模型向他提需求） |

---

## 六、对齐要点（开工前必须确认）

1. **`source_id` / `claim_id` 是 trace 内唯一**，不是全局唯一。跨表引用必须带 `trace_id`
2. **`must_fix=true` 不能 reject**，只能 accept 或 partially_accept
3. **claims 必须是"可验证陈述"**，不能是整段话；`is_key_technical=true` 时必须有 `source_ref`
4. **`section_ref` 格式统一**：`tutorial:sec_2` / `tutorial:sec_3:p_2` / `code:example_1:line_3_12`
