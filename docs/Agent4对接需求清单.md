# 专业知识与教学质量审核 Agent 审核辩论机制 — 核心对接需求清单

> 依据：`docs/个性化学习资源生成 专业知识与教学质量审核 Agent审核辩论协议.md`、`数据库框架.md` 第九节、`专业知识与教学质量审核 Agent任务说明.md`。
> 前提：审核辩论过程由 Orchestrator 按 `trace_id` 写入 `project.db` 的 `review_traces`、`review_debate_rounds`、`review_debate_issues`、`review_debate_responses` 及来源/Claim 表；个性化学习资源生成 Agent 不负责直接写库。
>
> 学习包采用块级流水线：首轮审核完成后一次性交付完整问题列表；个性化学习资源生成 Agent 最多并发修改 2 个块，每块完成即交 专业知识与教学质量审核 Agent 复核；问题列表最多交付 3 次。
>
> **运行时执行体选型 = 方案 B（原地嫁接）**：Harness 实际调用的是 `deterministic.HybridTechnicalReviewAgent`，
> 它含一步 **LLM 独立语义审核**（重算参数量、辨析 LoRA 机制、注入领域基线），这是抓“读起来通顺但机制/参数讲错”的核心手段。
> 若改指向 `agent_04_tech_review.TechnicalReviewAgent`（方案 A）会**丢掉该 LLM 审核能力**，故不采用。
> 方案 B 保留 Hybrid 的完整审核链，只在其产出 errors 后**委托** `agent_04_tech_review` 的已就绪函数补协议字段，并加 `rebut`。
>
> 专业知识与教学质量审核 Agent 侧（`review.py` + `agent_04_tech_review.py`）已严格符合三份文档，无需再改。
> **上述 3 个项目负责人文件已经完成对接：`contracts.py`、`deterministic.py`、`harness.py`。本清单用于运行验收和接口约束，不再表示待开发任务。**

---

## 1. `backend/models/contracts.py`（项目负责人）

**功用**：`ReviewInput` 是 Orchestrator 递给 专业知识与教学质量审核 Agent 的输入信封。现在装不下 `sources`/`claims`，来源无法从 个性化学习资源生成 Agent 传到 专业知识与教学质量审核 Agent。

### 1.1 `ReviewInput` 增加字段

```python
class ReviewInput(BaseModel):
    session_id: str
    target_type: Literal["learning_package", "answer"]
    content: str
    stage: StageInfo
    profile: UserProfile
    question: Optional[str] = None
    code_examples: List[dict] = Field(default_factory=list)
    # === 新增 ===
    trace_id: Optional[str] = None                       # 本次审核生命周期 ID
    sources: List[dict] = Field(default_factory=list)    # 个性化学习资源生成 Agent 产出的来源清单（协议 2.1）
    claims: List[dict] = Field(default_factory=list)     # 个性化学习资源生成 Agent 产出的可验证陈述（协议 2.2）
    round_index: int = 1                                 # 当前审核轮次，用于 error_id 后缀 _r{n}
```

### 1.2 新增 rebut 的 IO 模型

```python
from typing import Dict   # 顶部补充

class RebutInput(BaseModel):
    session_id: str
    trace_id: str
    issues: List[dict]           # 专业知识与教学质量审核 Agent 上一轮 issues（canonical_error_id/must_fix/severity...）
    author_replies: List[dict]   # 个性化学习资源生成 Agent 逐条回复（canonical_error_id/disposition/reason/patched_sections/source_refs）

class RebutOutput(BaseModel):
    verdicts: Dict[str, str]     # {canonical_error_id: "persist" | "withdraw" | "downgrade"}
```

---

## 2. `backend/agents/deterministic.py`（项目负责人）

**功用**：`HybridTechnicalReviewAgent.run()` 是运行时真正被调用的审核体，保留其 LLM + 工具链能力不动，仅嫁接协议能力。

### 2.1 `run()` 末尾嫁接协议字段补全

`run()` 产出 `errors` 后、返回前，委托 `agent_04_tech_review` 的现成函数补全（不复制实现，只调用）：

```python
from agents.agent_04_tech_review import TechnicalReviewAgent
# run() 内，拿到 errors 后：
errors = TechnicalReviewAgent()._enrich_debate_fields(
    errors,
    sources=request.sources,      # 来自 1.1 扩展后的 ReviewInput
    claims=request.claims,
    round_index=request.round_index,
    raw_errors=raw_errors,        # 若可得，用于取回工具原始 change_request/claim_id 提示
)
```

补全后每条 error 携带：`canonical_error_id` / `error_id` / `source_refs` / `change_request` / `must_fix` / `must_fix_reason` / `claim_id`（协议 2.3 / 2.4 / 第 6 节）。

### 2.2 新增 `rebut()` 入口（委托 专业知识与教学质量审核 Agent 实现）

```python
async def rebut(self, session_id, issues, author_replies):
    from agents.agent_04_tech_review import TechnicalReviewAgent
    return await TechnicalReviewAgent().rebut(session_id, issues, author_replies)
```

> 说明：`DeterministicTechnicalReviewAgent`（纯工具链模式）如需同等能力，比照 2.1 / 2.2 同样处理。

---

## 3. `backend/orchestrator/harness.py`（项目负责人 / Orchestrator）

**功用**：Harness 是总调度。现在审核不过是「直接让 个性化学习资源生成 Agent 改」的单向流程，需改成完整辩论循环，并把 package 的 sources/claims 透传进去。

### 3.1 构造 `ReviewInput` 时透传 sources/claims（约 L332）

当前只传 `content=current_package.tutorial`，丢掉了 package 的 sources/claims。需补：

```python
reviews = await self._review(ReviewInput(
    session_id=session_id,
    target_type="learning_package",
    content=current_package.tutorial,
    stage=stage,
    profile=profile,
    code_examples=current_package.code_examples,
    trace_id=trace_id,                       # 新增：循环外创建一次，全程复用
    sources=current_package.sources,         # 新增
    claims=current_package.claims,           # 新增
    round_index=iteration,                   # 新增：复用现有 iteration
))
```

### 3.2 用「答辩 → 复核」替换现有单向 revise 循环（协议第 4~5 节，约 L347~L362）

当前逻辑：审核不过 → 直接 `agent3.revise_learning_package`。改为：

```
1. reviews = 审核结果（issues 已带 canonical_error_id/must_fix/source_refs/change_request）
2. 若 passed → break
3. author_replies = await agent3.reply_to_review(session_id, issues, package)   # 个性化学习资源生成 Agent 已实现
4. verdicts = await agent4.rebut(session_id, issues, author_replies)            # 见第 2.2 节
5. 依 verdicts 更新每条 issue 状态：
     persist   → 保持未决（must_fix 未消除）
     withdraw  → 视为已接受/撤销
     downgrade → 用 agent_04_tech_review.downgrade_severity() 降级，必要时 must_fix=false
6. 调用 agent3.modify_package(session_id, package, errors, accepted_issues)     # 个性化学习资源生成 Agent 已实现
7. 未达上限且仍有未决 → 下一轮；达上限见 3.3
```

### 3.3 超轮次兜底（协议第 7 节）

达到最大辩论轮次仍有**未决 must_fix**：
- `review_traces.final_status = needs_human_review`
- `review_traces.package_status = provisional`
- 学习包允许降级展示给当前用户，但不进入正式知识沉淀。

---

## 4. 供对接使用的既有接口（专业知识与教学质量审核 Agent 侧已就绪，无需改）

```python
await TechnicalReviewAgent().review(
    session_id, content, profile=None,
    sources=None, claims=None, round_index=1,
) -> TechnicalReviewResult   # errors[] 含 canonical_error_id/error_id/source_refs/
                             #   change_request/must_fix/must_fix_reason/claim_id

await TechnicalReviewAgent().rebut(
    session_id, issues, author_replies,
) -> dict[str, str]          # {canonical_error_id: "persist"|"withdraw"|"downgrade"}

# 供 Hybrid 嫁接协议字段（不复制实现，直接调用）
TechnicalReviewAgent()._enrich_debate_fields(errors, sources=, claims=, round_index=, raw_errors=)
agent_04_tech_review.downgrade_severity("critical") -> "major"
```
