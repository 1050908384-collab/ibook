# Agent4 任务说明（审核侧判定与复核）

> 本文档只描述 **Agent4 同学** 需要做的事。Agent3 的改动你不用管，Harness/DB/状态机由项目负责人处理。
> 最后更新：2026-07-31

---

## 一、开工前必读

| 文档 | 位置 | 重点看 |
|---|---|---|
| 审核辩论协议 | [`docs/Agent34审核辩论协议.md`](docs/Agent34审核辩论协议.md) | 2.3 ReviewError、2.4 Error ID 规则、第 3 节 Agent4→Agent3、第 5 节复核、第 6 节 must_fix、第 7 节超轮次 |
| 数据库框架 | [`数据库框架.md`](数据库框架.md) 第九节 | `review_debate_issues`、`review_debate_rounds`、`review_debate_responses` 三张表的字段和约束 |

---

## 二、你的职责（4 件事）

1. **审核产出必须带 `source_refs` + `change_request` + `must_fix`**：不能只报 error_type + description
2. **`canonical_error_id` 按协议哈希规则生成**（跨轮稳定，不能自己发明 ID）
3. **对 Agent3 的回复做二次复核裁决**：`persist / withdraw / downgrade`
4. **must_fix 按派生规则计算**，不能只靠 `severity=critical`

---

## 三、改动清单

### 1. 扩展 `backend/models/review.py` 的 `ReviewError`

**文件**：[`backend/models/review.py`](backend/models/review.py#L5)

当前定义只有 6 个字段：
```python
class ReviewError(BaseModel):
    error_type: str
    severity: Literal["critical", "major", "minor"] = "major"
    location: Optional[str] = None
    description: str
    evidence: Optional[str] = None
    suggestion: Optional[str] = None
```

需要增加 6 个字段：
```python
    error_id: Optional[str] = None                          # 当前轮实例 ID
    canonical_error_id: Optional[str] = None                # 跨轮稳定 ID
    source_refs: List[str] = Field(default_factory=list)    # 证据来源（指向 sources 的 source_id）
    change_request: Optional[Dict] = None                   # {"required": bool, "action": str}
    must_fix: bool = False
    must_fix_reason: Optional[str] = None
    claim_id: Optional[str] = None                          # 关联的 claim
```

### 2. 新增 DebateIssue / DebateResponse / DebateRound 模型

**文件**：`backend/models/review.py`（新增）

定义见协议文档 2.3 节，关键模型：

**DebateIssue**（争议单，围绕一个 canonical_error_id 的生命周期）：
- `issue_id`、`trace_id`、`canonical_error_id`、`error_id`
- `claim_id`、`error_type`、`severity`、`must_fix`、`must_fix_reason`
- `description`、`location`、`evidence`、`suggestion`
- `source_refs`、`change_request`
- `current_status`：`open | accepted | rejected | partially_accepted | deferred | overruled | withdrawn | needs_human_review`
- `final_disposition`、`opened_in_round_num`、`resolved_in_round_num`

**DebateResponse**（每轮里对某 issue 的回复）：
- `response_id`、`round_id`、`issue_id`
- `actor`：`agent3 | agent4 | orchestrator`
- `response_type`：`author_reply | reviewer_verdict | orchestrator_ruling`
- `disposition`：`accepted | rejected | partially_accepted | persist | withdraw | downgrade | overruled | deferred`
- `rationale`、`source_refs`、`patched_sections`

**DebateRound**（答辩复核轮次）：
- `round_id`、`trace_id`、`round_num`、`review_iteration`
- `round_status`：`open | closed`
- `orchestrator_verdict`：`pending | all_resolved | partial_resolved | deadlocked | needs_human_review`
- `verdict_summary`

### 3. 改 `backend/agents/agent_04_tech_review.py` 的 `review()`

**文件**：[`backend/agents/agent_04_tech_review.py`](backend/agents/agent_04_tech_review.py#L144)

当前 `_normalize_errors` 后直接返回。需要改为：
- 每个 error 强制填 `source_refs`（指向 sources 里的某条，或标注"未命中来源"）
- 每个 error 强制填 `change_request`（`{"required": bool, "action": str}`）
- 生成 `canonical_error_id`（哈希规则见下方第 4 点）
- 生成 `error_id`（`canonical_error_id + "_r" + round_index`）
- 按 must_fix 派生规则（见下方第 5 点）计算 `must_fix`

### 4. Error ID 生成规则（必须严格按协议，否则跨轮追踪断裂）

**文件**：协议 2.4 节

```python
canonical_error_id = sha1(claim_id + error_type + normalized_location + normalized_description)[:12]
error_id = canonical_error_id + "_r" + round_index
```

**归一化规则（必须钉死，不能自己发明）**：
- `normalized_location`：小写 + 去首尾空格
- `normalized_description`：小写 + 去中英文标点 + 折叠连续空白

**要求**：
- Agent4 不自行发明跨轮逻辑 ID
- 同一问题跨轮保留同一个 `canonical_error_id`
- `canonical_error_id` 在单个 trace 语义下唯一

### 5. 新增 `rebut` 方法（复核裁决）

**文件**：`backend/agents/agent_04_tech_review.py`（新增方法）

签名建议：
```python
async def rebut(
    self,
    session_id: str,
    issues: List[Dict],           # 你出具的 issues
    author_replies: List[Dict]    # Agent3 的逐条回复
) -> Dict[str, str]:
    """对 Agent3 的每条回复做二次裁决。"""
```

返回 `{canonical_error_id: "persist" | "withdraw" | "downgrade"}`：

| 裁决 | 含义 | 何时用 |
|---|---|---|
| `persist` | 坚持 original issue | Agent3 的反证不成立 |
| `withdraw` | 撤销该 issue | Agent3 反证成立，自己认错 |
| `downgrade` | 降级 | 如 critical → major，或 must_fix → false |

### 6. must_fix 派生规则（配置化，不能只靠 severity）

**自动进入 must_fix 的条件**（满足任一）：
- `severity = critical`
- `error_type` 属于事实错误、来源缺失、代码不可执行、违反安全/规范
- `source_refs` 为空但该 claim 是 `is_key_technical=true`
- `change_request.required = true`

**Agent4 可显式升级为 must_fix 的场景**：
- 会误导学习路径
- 会导致代码、命令或配置无法复现
- 与当前用户画像严重不匹配

**关键**：`must_fix` 是流程约束级别，`severity` 是风险级别，两者相关但不等价。

---

## 四、交互时序（你只看自己这侧）

```
1. Harness 调用你的 review(学习包 + sources + claims)
   → 你返回 issues[]（带 canonical_error_id, source_refs, must_fix, change_request）

2. 【debate 子循环，最多 2 轮】
   Agent3 对每条 issue 返回 accepted/rejected/partially_accepted
   Harness 调用你的 rebut(issues, author_replies)
   → 你对每条返回 persist/withdraw/downgrade
   → Orchestrator 根据你的裁决做最终判定

3. 超轮次（达到最大轮次仍有未决 must_fix）
   → Orchestrator 置 final_status=needs_human_review
   → 学习包降级为 provisional，不进入正式知识沉淀
   → 你不用做额外处理，Orchestrator 兜底
```

---

## 五、禁止修改的文件

| 文件 | 原因 |
|---|---|
| `backend/agents/agent_03_content_gen.py` | Agent3 负责 |
| `backend/models/stage.py` | Agent3 负责 |
| `backend/orchestrator/harness.py` | 项目负责人负责 |
| `backend/db/sqlite_client.py` | 项目负责人负责 |
| `backend/state/` | 项目负责人负责 |
| `backend/models/contracts.py` | 项目负责人负责（需要新 IO 模型向他提需求） |

---

## 六、对齐要点（开工前必须确认）

1. **`canonical_error_id` 归一化规则**：协议 2.4 节已钉死，必须严格按规则实现，否则跨轮追踪断裂
2. **`source_id` / `claim_id` / `canonical_error_id` 是 trace 内唯一**，不是全局唯一
3. **`must_fix` ≠ `severity`**：按派生规则算，不能只看 critical
4. **每个 error 必须带 `source_refs`**：要么指向 sources 里的某条，要么标注"未命中来源"（这本身就是 must_fix 的触发条件）
5. **每个 error 必须带 `change_request`**：明确告诉 Agent3 要改什么
