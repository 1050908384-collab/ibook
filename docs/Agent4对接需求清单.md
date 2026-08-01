# Agent4 审核辩论机制 — 对接需求清单

> 面向项目负责人 / Harness / Orchestrator / DB 侧。
> Agent4 侧（`backend/models/review.py`、`backend/agents/agent_04_tech_review.py`）已完成协议实现并自测通过（33/33）。
> 本清单列出让「审核 → 答辩 → 复核 → 裁决 → 落库」闭环真正跑起来所需的**跨模块改动**，均在 Agent4 的禁改范围之外。
> 依据：`docs/Agent34审核辩论协议.md`、`数据库框架.md 第九节`、`Agent4任务说明.md`。

---

## 0. 当前状态与核心缺口

| 项 | 状态 |
|---|---|
| `ReviewError` 扩展 6+1 字段 | ✅ 已完成（Agent4 侧） |
| `DebateIssue / DebateResponse / DebateRound` 模型 | ✅ 已完成，字段/枚举严格对齐 DDL |
| `canonical_error_id` 哈希 + 归一化（协议 2.4） | ✅ 已完成，跨轮稳定 |
| `must_fix` 派生规则（协议第 6 节） | ✅ 已完成 |
| `review()` 强制补全 source_refs/change_request/must_fix/ID | ✅ 已完成 |
| `rebut()` 复核裁决 persist/withdraw/downgrade | ✅ 已完成 |
| **运行时链路接入** | ❌ **未接入**（见下方缺口） |

**核心缺口**：Harness 实际调用的是 `deterministic.py` 里的 `HybridTechnicalReviewAgent.run(request)`（[harness.py:414](backend/orchestrator/harness.py)），
而我扩展的是 `agent_04_tech_review.py` 的 `TechnicalReviewAgent.review()` / `rebut()`。两者未打通，且 `ReviewInput` 未携带 `sources`/`claims`，辩论子循环也未在 Harness 中实现。

---

## 1. IO 模型（`backend/models/contracts.py`）— 需项目负责人改

> 任务说明第五节：`contracts.py` 由项目负责人负责，需新 IO 模型向他提需求。

### 1.1 `ReviewInput` 增加来源与轮次字段

当前 [`ReviewInput`](backend/models/contracts.py) 只有 `content: str`。需补充：

```python
class ReviewInput(BaseModel):
    session_id: str
    target_type: Literal["learning_package", "answer"]
    content: str
    stage: StageInfo
    profile: UserProfile
    question: Optional[str] = None
    code_examples: List[dict] = Field(default_factory=list)
    # === 新增（审核辩论协议 2.1 / 2.2）===
    trace_id: Optional[str] = None                       # 本次审核生命周期 ID
    sources: List[dict] = Field(default_factory=list)    # 学习包来源清单（Agent3 产出）
    claims: List[dict] = Field(default_factory=list)     # 可验证陈述声明（Agent3 产出）
    round_index: int = 1                                 # 当前审核轮次，用于 error_id 后缀
```

- `sources[]` 字段：`source_id / source_type / title / locator / content_ref / source_version / metadata_json`（协议 2.1）
- `claims[]` 字段：`claim_id / text / source_ref / source_span / section_ref / claim_type / is_key_technical / confidence / status`（协议 2.2）

### 1.2 新增 rebut 的 IO 模型（可选，或直接用 dict）

```python
class RebutInput(BaseModel):
    session_id: str
    trace_id: str
    issues: List[dict]           # Agent4 上一轮 issues（含 canonical_error_id / must_fix / severity ...）
    author_replies: List[dict]   # Agent3 逐条回复（canonical_error_id / disposition / reason / patched_sections / source_refs）

class RebutOutput(BaseModel):
    verdicts: Dict[str, str]     # {canonical_error_id: "persist" | "withdraw" | "downgrade"}
```

---

## 2. Harness / Orchestrator（`backend/orchestrator/harness.py`）— 需项目负责人改

### 2.1 把 sources/claims/round_index 透传给 Agent4

`_review()` 目前调 `self.agent4.run(request)`。需让实际执行体调用带证据的入口，并透传新字段：

```python
technical = await self.agent4.review(
    session_id=request.session_id,
    content={...},                 # tutorial / code_examples / answer 分支
    profile=request.profile,
    sources=request.sources,       # 新增
    claims=request.claims,         # 新增
    round_index=request.round_index,  # 新增
)
```

> 注意：`review()` 已做**向后兼容**（sources/claims 默认空、round_index 默认 1），不传也不会报错，只是退化为「无来源可校验」。

### 2.2 实现 debate 子循环（最多 2 轮，协议 / 任务说明第四节）

```
1. review() 产出 issues[]（带 canonical_error_id / source_refs / must_fix / change_request）
2. 【debate 子循环，最多 2 轮】
   - 调 Agent3 → 每条 issue 返回 author_reply(disposition: accepted|rejected|partially_accepted)
   - 调 Agent4.rebut(issues, author_replies) → 每条返回 persist|withdraw|downgrade
   - Orchestrator 依裁决更新 issue.current_status / final_disposition
3. 超轮次仍有未决 must_fix → 见第 4 节兜底
```

### 2.3 决定 `HybridTechnicalReviewAgent` vs `TechnicalReviewAgent` 的归属

二选一（需拍板）：
- **方案 A**：Harness 的 `self.agent4` 改指向 `agent_04_tech_review.TechnicalReviewAgent`（已含 review+rebut+辩论字段）。
- **方案 B**：把 `review()`/`rebut()` 的辩论增强逻辑同步进 `deterministic.HybridTechnicalReviewAgent`（`deterministic.py` 在 agents 目录，但当前由我按任务说明只改了 `agent_04_tech_review.py`）。

> 推荐方案 A：Agent4 侧协议实现集中在 `agent_04_tech_review.py`，避免逻辑双写。

---

## 3. 数据库落库（`backend/db/sqlite_client.py` + 写入方）— 需项目负责人改

6 张表 DDL 已存在（第九节）。需补**写入/读取方法**（Agent4 只产出对象，不落库）：

| 表 | 写入时机 | 写入方 |
|---|---|---|
| `review_traces` | 审核开始 | Orchestrator |
| `learning_package_sources` | Agent3 产出来源后 | Orchestrator（Agent3 侧数据） |
| `learning_package_claims` | Agent3 产出 claims 后 | Orchestrator（Agent3 侧数据） |
| `review_debate_issues` | Agent4 `review()` 产出 issues 后 | Orchestrator |
| `review_debate_rounds` | 每开一轮答辩 | Orchestrator |
| `review_debate_responses` | 每条 author_reply / reviewer_verdict / orchestrator_ruling | Orchestrator |

**落库映射注意（关键约束）**：
1. **JSON 字段序列化**：`source_refs` / `change_request` / `patched_sections` 在 DB 中为 `TEXT`，写入前需 `json.dumps`，读取后 `json.loads`。
2. **canonical_error_id 去重**：`review_debate_issues` 有 `UNIQUE(trace_id, canonical_error_id)`。Agent4 保证同一问题跨轮 canonical 稳定；**Orchestrator 需按 (trace_id, canonical_error_id) upsert**，不要为同一 canonical 插多条 issue（同一轮内若出现重复 canonical，`error_id` 已带 `_N` 后缀区分实例，但 issue 仍应合并为一条）。
3. **复合外键**：涉及 `claim_id` / `source_ref` 的写入必须带 `trace_id`（协议 8.1，防跨 trace 串扰）。
4. **CHECK 约束**：`learning_package_claims` 强制 `is_key_technical=1` 时 `source_ref` 非空——Agent3 侧数据需满足。
5. **reviewer_verdict 落库**：Agent4 `rebut()` 返回的 `persist/withdraw/downgrade` 应写入 `review_debate_responses`，`actor='agent4'`、`response_type='reviewer_verdict'`、`disposition=` 裁决值。
6. **downgrade 落库**：`downgrade` 裁决需更新 issue 的 severity（可用 Agent4 导出的 `downgrade_severity()` 工具函数：critical→major→minor）与/或 `must_fix=false`。

---

## 4. 超轮次兜底（协议第 7 节）— 需 Orchestrator 实现

达到最大辩论轮次仍有未决 `must_fix` 时：
- `review_traces.final_status = 'needs_human_review'`
- `review_traces.package_status = 'provisional'`
- 未决 issue 的 `current_status = 'needs_human_review'`
- 学习包**允许降级展示给当前用户**，但**不进入正式知识沉淀 / 复用资产池**
- Agent4 不做额外处理（任务说明第四节：Orchestrator 兜底）

---

## 5. Agent3 侧（`agent_03_content_gen.py` / `stage.py`）— Agent3 同学负责

Agent4 依赖 Agent3 产出以下结构（Agent4 不生成，只消费/校验）：
1. **sources[]**：来源清单（协议 2.1 字段）
2. **claims[]**：可验证陈述声明（协议 2.2 字段，含 `is_key_technical` / `source_ref` / `section_ref`）
3. **author_replies[]**：对 Agent4 每条 issue 的逐条回复
   - `disposition ∈ {accepted, rejected, partially_accepted}`
   - `reason`（"没修改"时必须给理由）、`patched_sections`、`source_refs`
   - 约束：`must_fix=true` 的 issue **不允许直接 rejected**（协议第 4 节）

> 用户原始诉求映射：
> - 「Agent3 给 Agent4：学习包 + 来源 + 我的回复（改好了/没改+为什么）」= `content` + `sources`/`claims` + `author_replies`
> - 「Agent4 给 Agent3：错误个数 + 来源」= `review()` 返回 `errors[]`（每条带 `source_refs`）+ summary 报错误个数/must_fix 数

---

## 6. 建议的接口契约（Agent4 已就绪，供对接方对齐）

```python
# 审核（Agent4 → Agent3）：产出带证据 issues
await agent4.review(
    session_id: str,
    content: dict,                    # {"tutorial"|"code_examples"|"answer": ...}
    profile: dict | UserProfile = None,
    sources: list[dict] = None,       # 命中则填 source_refs，未命中=空(=must_fix触发条件)
    claims: list[dict] = None,        # 用于关联 claim_id / is_key_technical
    round_index: int = 1,             # error_id = canonical + "_r" + round_index
) -> TechnicalReviewResult            # .errors[] 每条含 canonical_error_id/error_id/source_refs/
                                      #   change_request/must_fix/must_fix_reason/claim_id

# 复核（Agent4 对 Agent3 回复二次裁决）
await agent4.rebut(
    session_id: str,
    issues: list[dict],               # Agent4 上一轮 issues
    author_replies: list[dict],       # Agent3 逐条回复
) -> dict[str, str]                   # {canonical_error_id: "persist"|"withdraw"|"downgrade"}

# 工具（供 Orchestrator 落 downgrade 时用）
agent_04_tech_review.downgrade_severity("critical") -> "major"
```

---

## 7. 验收清单（Agent4 侧已自测通过项，供回归对照）

- [x] `canonical_error_id` 同串稳定、跨轮一致；大小写/首尾空格/中英文标点被归一化中和
- [x] `error_id = canonical + "_r" + round_index`；同轮重复 canonical 实例加 `_N` 后缀
- [x] `must_fix` 四条派生：critical / 事实-来源-代码-安全类 error_type / 关键技术无源 / change_request.required
- [x] 每条 error 强制带 `source_refs`（命中或空）+ `change_request`（must_fix 时 required=true）
- [x] `rebut`：must_fix+rejected→persist；accepted+整改→withdraw；partially_accepted→downgrade；rejected+反证→withdraw；非法/缺失→persist（保守）
- [x] 空输入 / 非法 disposition / 单条异常均有兜底，不中断流程
