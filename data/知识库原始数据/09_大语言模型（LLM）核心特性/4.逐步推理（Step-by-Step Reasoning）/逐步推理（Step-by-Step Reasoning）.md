# 逐步推理（Step-by-Step Reasoning）：让大模型"一步一步想"

> **参考文章链接**
> 
> ① 底稿：《大模型思维链推理与 Chain-of-Thought 优化：从直接回答到逐步推理，LLM 推理能力的激发》
> 　　网址：https://blog.csdn.net/weixin_43272162/article/details/161917119
> 
> ② 参考文章：《Chain-of-Thought Prompting(CoT)大模型思维链概述》
> 　　网址：https://blog.csdn.net/qq_73472828/article/details/160045208

---

**适合人群**：已掌握指令遵循概念，希望理解如何让大模型从"直接猜答案"进化为"逐步推理"的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着理解。完成本章后，你将能够使用 Zero-shot CoT、Few-shot CoT、Self-Consistency 等技术显著提升模型在数学、逻辑、因果分析等复杂任务上的表现，为后续"幻觉问题"等 LLM 核心特性打下坚实基础。

**预计学习时长**：约 0.5 小时

---

## 第一章：为什么需要逐步推理——直接回答的瓶颈

### 1.1 一个直观的对比

假设你问模型一道数学题：

> **问题**：Roger 有 5 个网球，他又买了 2 罐网球，每罐 3 个。他现在有多少个网球？

**直接回答（标准提示）**：
```
A: The answer is 11.
```
然后换一道类似的题：
```
Q: 食堂有 23 个苹果，用了 20 个做午餐，又买了 6 个，现在有多少？
A: The answer is 27.   ← ❌ 错误！正确答案是 9
```

**逐步推理（CoT 提示）**：
```
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 
   5 + 6 = 11. The answer is 11.
```
然后换一道类似的题：
```
Q: 食堂有 23 个苹果...
A: The cafeteria had 23 apples originally. They used 20 to make lunch. 
   So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. 
   The answer is 9.   ← ✅ 正确！
```

**为什么这样写**：标准提示让模型直接输出答案，中间推理过程被压缩在隐层表示中，模型容易在长链推理中"走偏"。CoT 通过要求模型"逐步思考"，将推理过程**显式化**，让模型先输出中间步骤，再给出最终答案，准确率可提升 30-50%。

### 1.2 根本问题：单次前向传播的压缩

大模型在直接回答时，需要在**单次前向传播**中同时完成"推理"与"输出"：

```
输入问题 → [隐层压缩推理] → 直接输出答案
              ↑
        中间步骤被"压缩"在这里，不可见、不可控
```

当问题需要多步推理时（如数学证明、逻辑推导、因果分析），这种压缩会导致：
- 中间步骤出错无法被发现
- 长链推理中信息逐渐"迷失"
- 错误在后续步骤中**累积传播**

**为什么这样写**：CoT 的本质是**将隐式推理变为显式推理**。让模型把思考过程"说出来"，既方便人类检查，也让模型自身能基于已生成的中间步骤继续推导。

---

## 第二章：Chain-of-Thought（CoT）——核心机制

### 2.1 CoT 的映射方式

区别于传统 Prompt 从输入直接到输出的映射：

```
传统:  input ──→ output
CoT:  input ──→ reasoning chain ──→ output
```

**为什么这样写**：增加中间的 reasoning chain（推理链），让模型有机会在每一步验证自己的逻辑，降低一步错、步步错的概率。

### 2.2 一个完整的 CoT Prompt 结构

```
┌─────────────────────────────────────────┐
│  指令（Instruction）                     │
│  "请逐步思考以下问题..."                  │
├─────────────────────────────────────────┤
│  逻辑依据（Rationale）                   │
│  中间推理步骤、解决方案、外部知识          │
├─────────────────────────────────────────┤
│  示例（Exemplars）                       │
│  问题 + 推理过程 + 答案                   │
└─────────────────────────────────────────┘
```

**为什么这样写**：指令告诉模型"要做什么"，逻辑依据是模型实际生成的内容，示例（Few-shot）则给模型展示"应该怎么做"。

---

## 第三章：CoT 的四种变体——从简单到复杂

### 3.1 Zero-shot CoT：一句话唤醒推理

**核心思想**：不给任何示例，只在指令末尾加一句魔法咒语。

```python
# 标准 Zero-shot（容易出错）
prompt = """
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。
然后我去买了5个苹果并吃了1个。我还剩下多少苹果？
"""

# Zero-shot CoT（加一句"逐步思考"）
prompt_cot = """
我去市场买了10个苹果。我给了邻居2个苹果和修理工2个苹果。
然后我去买了5个苹果并吃了1个。我还剩下多少苹果？
让我们逐步思考。
"""
```

**输出对比**：

| 方式 | 输出 | 结果 |
|------|------|------|
| 标准 Zero-shot | "11个苹果" | ❌ 错误 |
| Zero-shot CoT | "首先，您从10个苹果开始...最后，您还剩下10个苹果。" | ✅ 正确 |

**为什么这样写**："Let's think step by step"（让我们逐步思考）这句话看似简单，但它**激活了模型在预训练阶段学到的"逐步推理"模式**。模型在训练数据中见过大量"先分析、后结论"的文本，这个提示就像一个开关，唤醒了这种能力。

> 注意：这是**足够大的语言模型**才会出现的涌现能力，小模型可能无效。

### 3.2 Few-shot CoT：用示例教会模型推理

**核心思想**：在提示中提供包含完整推理过程的示例，让模型"照猫画虎"。

```python
few_shot_cot_prompt = """
请参考以下示例的推理方式，解答新的问题。

示例1:
问题: 一个水池有两个进水管，A管5小时注满，B管3小时注满。两管同时开，几小时注满？
步骤1: A管速率 = 1/5 池/小时
步骤2: B管速率 = 1/3 池/小时
步骤3: 合并速率 = 1/5 + 1/3 = 8/15 池/小时
步骤4: 注满时间 = 1 / (8/15) = 15/8 = 1.875 小时
最终答案: 1.875小时

示例2:
问题: 一件商品先涨价20%，再降价20%，最终价格比原价高还是低？
步骤1: 设原价为 P
步骤2: 涨价20%后 = 1.2P
步骤3: 降价20%后 = 1.2P × 0.8 = 0.96P
步骤4: 0.96P < P，所以最终价格比原价低4%
最终答案: 比原价低4%

现在请解答：
问题: {question}
"""
```

**为什么这样写**：Few-shot CoT 比 Zero-shot CoT 更稳定，因为示例展示了**具体的推理格式和风格**。模型不仅知道要"逐步思考"，还知道"每一步应该怎么写"。

### 3.3 Self-Consistency：多条推理链投票

**核心思想**：生成多条独立的推理链，对最终答案投票，选择最一致的。

```python
import asyncio
from collections import Counter

async def self_consistency_cot(question, num_samples=5):
    """
    Self-Consistency CoT
    并行生成多条推理链，投票选择最一致的答案
    """
    # 使用较高温度增加多样性
    responses = []
    for _ in range(num_samples):
        prompt = f"""
        请逐步思考以下问题，展示完整推理过程：
        
        问题：{question}
        
        请按以下格式回答：
        步骤1: [推理步骤]
        ...
        最终答案: [答案]
        """
        # 模拟调用 LLM（实际使用 API）
        response = await call_llm(prompt, temperature=0.7)
        responses.append(response)
    
    # 提取每条推理链的答案
    answers = [extract_final_answer(r) for r in responses]
    
    # 投票：选择出现次数最多的答案
    answer_counts = Counter(answers)
    best_answer, count = answer_counts.most_common(1)[0]
    confidence = count / num_samples
    
    return {
        'answer': best_answer,
        'confidence': confidence,
        'all_answers': answers
    }

# 示例输出
# question: "16个球，一半是高尔夫球，一半高尔夫球是蓝色的，有几个蓝色高尔夫球？"
# responses: ["4", "4", "8", "4", "4"]
# best_answer: "4", confidence: 0.8
```

**为什么这样写**：
- `temperature=0.7` 增加生成多样性，让不同推理链可能探索不同路径
- 投票机制能**抵消单条推理链中偶然的错误**
- `confidence` 表示答案的可靠程度，可用于决策（如 confidence < 0.6 时拒绝回答）

### 3.4 Auto-CoT：自动构建示例

**核心思想**：自动从问题集合中生成高质量的 Few-shot 示例，省去人工编写。

**两阶段流程**：

```
阶段1: 问题聚类
        ↓
  将所有问题按语义相似度分成若干簇

阶段2: 演示抽样
        ↓
  从每个簇中选一个代表性问题
  用 Zero-shot-CoT 生成其推理链
  组合成 Few-shot 示例
```

**为什么这样写**：手工编写 Few-shot 示例费时费力，且可能不够多样化。Auto-CoT 通过聚类保证示例覆盖不同类型的问题，通过 Zero-shot-CoT 自动生成推理链，实现"自我教学"。

---

## 第四章：CoT 推理引擎的完整实现

以下是一个可直接运行的 CoT 推理引擎：

```python
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from collections import Counter

@dataclass
class ReasoningStep:
    step_number: int
    thought: str
    action: Optional[str] = None

@dataclass
class CoTResult:
    question: str
    reasoning_chain: List[ReasoningStep]
    answer: str
    confidence: float
    method: str  # zero_shot, few_shot, self_consistency

class CoTEngine:
    """Chain-of-Thought 推理引擎"""
    
    ZERO_SHOT_TEMPLATE = """请逐步思考以下问题，展示完整的推理过程：

问题：{question}

请按以下格式回答：
步骤1: [推理步骤]
步骤2: [推理步骤]
...
最终答案: [答案]"""

    FEW_SHOT_TEMPLATE = """请参考以下示例的推理方式，解答新的问题。

示例1:
问题: 一个水池有两个进水管，A管5小时注满，B管3小时注满。两管同时开，几小时注满？
步骤1: A管速率 = 1/5 池/小时
步骤2: B管速率 = 1/3 池/小时
步骤3: 合并速率 = 1/5 + 1/3 = 8/15 池/小时
步骤4: 注满时间 = 15/8 = 1.875 小时
最终答案: 1.875小时

现在请解答：
问题: {question}"""

    async def zero_shot_cot(self, question: str) -> CoTResult:
        """Zero-shot CoT"""
        prompt = self.ZERO_SHOT_TEMPLATE.format(question=question)
        response = await self._call_llm(prompt, temperature=0.3)
        
        steps = self._parse_reasoning_steps(response)
        answer = self._extract_final_answer(response)
        
        return CoTResult(
            question=question,
            reasoning_chain=steps,
            answer=answer,
            confidence=0.6,
            method='zero_shot',
        )

    async def few_shot_cot(self, question: str) -> CoTResult:
        """Few-shot CoT"""
        prompt = self.FEW_SHOT_TEMPLATE.format(question=question)
        response = await self._call_llm(prompt, temperature=0.3)
        
        steps = self._parse_reasoning_steps(response)
        answer = self._extract_final_answer(response)
        
        return CoTResult(
            question=question,
            reasoning_chain=steps,
            answer=answer,
            confidence=0.75,
            method='few_shot',
        )

    async def self_consistency(
        self, question: str, num_samples: int = 5
    ) -> CoTResult:
        """Self-Consistency CoT"""
        tasks = []
        for _ in range(num_samples):
            prompt = self.ZERO_SHOT_TEMPLATE.format(question=question)
            tasks.append(self._call_llm(prompt, temperature=0.7))
        
        responses = await asyncio.gather(*tasks)
        
        answers = []
        chains = []
        for response in responses:
            steps = self._parse_reasoning_steps(response)
            answer = self._extract_final_answer(response)
            answers.append(answer)
            chains.append(steps)
        
        answer_counts = Counter(answers)
        best_answer, count = answer_counts.most_common(1)[0]
        confidence = count / num_samples
        
        best_chain_idx = answers.index(best_answer)
        
        return CoTResult(
            question=question,
            reasoning_chain=chains[best_chain_idx],
            answer=best_answer,
            confidence=confidence,
            method='self_consistency',
        )

    def _parse_reasoning_steps(self, response: str) -> List[ReasoningStep]:
        """解析推理步骤"""
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('步骤') or line.startswith('Step'):
                step_num = len(steps) + 1
                thought = line.split(':', 1)[-1].strip() if ':' in line else line
                steps.append(ReasoningStep(step_number=step_num, thought=thought))
        return steps

    def _extract_final_answer(self, response: str) -> str:
        """提取最终答案"""
        for line in response.strip().split('\n'):
            if '最终答案' in line or 'Final Answer' in line:
                return line.split(':', 1)[-1].strip()
        return response.strip().split('\n')[-1]

    async def _call_llm(self, prompt: str, temperature: float) -> str:
        """模拟 LLM 调用（实际使用时替换为真实 API）"""
        # 这里应该调用 OpenAI API 或本地模型
        # 示例: return await openai.ChatCompletion.acreate(...)
        return f"模拟响应: {prompt[:50]}..."
```

---

## 第五章：CoT 的边界与权衡

### 5.1 推理链长度与准确率

| 推理步数 | 效果 | 建议 |
|----------|------|------|
| 1-5 步 | CoT 效果显著 | 直接使用 Zero-shot 或 Few-shot CoT |
| 5-10 步 | 效果尚可，错误开始累积 | 使用 Self-Consistency 提升稳定性 |
| 10+ 步 | 错误累积严重，模型可能"迷失" | **拆分为子问题**，每个子问题独立推理 |

**为什么这样写**：推理链越长，中间步骤出错的概率越高（**错误累积效应**）。对于超过 10 步的推理，建议拆分为多个子问题，每个子问题独立推理后组合结果。

### 5.2 计算成本权衡

| 方法 | LLM 调用次数 | 准确率 | 适用场景 |
|------|-------------|--------|----------|
| Zero-shot CoT | 1 | 中等 | 简单推理、成本敏感 |
| Few-shot CoT | 1 | 较高 | 特定领域、有高质量示例 |
| Self-Consistency | 5-10 | 最高 | 高准确率需求、可接受成本 |

**为什么这样写**：Self-Consistency 需要 N 次 LLM 调用，推理成本线性增长。在成本敏感场景下，可先用 Zero-shot CoT 尝试，仅在置信度低时升级为 Self-Consistency。

### 5.3 CoT 不是万能的

| 任务类型 | CoT 效果 | 原因 |
|----------|----------|------|
| 数学推理 | ✅ 显著提升 | 需要多步计算 |
| 逻辑推导 | ✅ 显著提升 | 需要逐步验证 |
| 因果分析 | ✅ 显著提升 | 需要中间推理 |
| 纯知识检索 | ❌ 无提升 | 无需推理，直接回忆即可 |
| 创意写作 | ❌ 可能有害 | 限制创造力，增加 Token 消耗 |

**为什么这样写**：CoT 对需要**推理**的任务效果显著，但对纯知识检索任务无提升，反而增加 Token 消耗。应在需要多步推理的场景**选择性使用**。

---

## 第六章：常见陷阱与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 所有任务都用 CoT | 纯知识检索任务不需要，浪费 Token |
| 推理链过长不拆分 | 超过 10 步拆分为子问题 |
| Few-shot 示例质量差 | 使用经过验证的高质量示例，错误示例会误导模型 |
| 忽视 Self-Consistency 的成本 | 成本敏感场景先用 Zero-shot，低置信度再升级 |
| 小模型强行用 CoT | CoT 是大模型的涌现能力，小模型效果有限 |
| 不解析推理链直接取答案 | 提取并展示推理步骤，方便检查和调试 |
| 温度设置不当 | Zero-shot 用低温度（0.3），Self-Consistency 用高温度（0.7） |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **CoT 本质** | 将隐式推理变为显式推理，input → reasoning chain → output |
| **Zero-shot CoT** | 加一句"Let's think step by step"，唤醒模型推理能力 |
| **Few-shot CoT** | 提供包含推理过程的示例，让模型照猫画虎 |
| **Self-Consistency** | 生成多条推理链投票，提升准确率和可靠性 |
| **Auto-CoT** | 自动聚类+生成示例，省去人工编写 |
| **错误累积** | 推理链越长，中间出错概率越高，需拆分长链 |
| **成本权衡** | Self-Consistency N 次调用，根据场景选择策略 |
| **适用范围** | 仅对需要多步推理的任务有效，知识检索无需 CoT |

**养成的好习惯**：
1. 复杂推理任务**默认使用 CoT**，简单任务不用
2. 生产环境优先用 **Few-shot CoT + Self-Consistency**
3. 长推理链**必须拆分**，避免错误累积
4. 始终**提取并记录推理步骤**，方便调试和审计
5. 根据**成本-准确率权衡**选择 CoT 变体

**下一步预告**：掌握了逐步推理后，你将进入 **幻觉问题（Hallucination）** 的学习——理解大语言模型为什么会"一本正经地胡说八道"、幻觉的分类与成因、以及当前主流的检测与缓解策略。这是使用大语言模型时必须正视的核心风险。继续加油！