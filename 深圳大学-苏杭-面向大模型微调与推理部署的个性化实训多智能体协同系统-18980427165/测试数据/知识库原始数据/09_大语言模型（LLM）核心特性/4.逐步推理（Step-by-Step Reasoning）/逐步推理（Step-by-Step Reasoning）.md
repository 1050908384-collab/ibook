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
**环境准备（仅用于本章推理/API示例）**：Python 3.10+，本地推理需 `transformers`、`torch`；若走 GPT 类 API，只需 `openai` SDK。

```bash
# 方案 A：本地跑开源模型做推理（需 GPU，最低建议 8GB 显存起步跑 7B 量化模型；否则会非常慢或 OOM）
pip install transformers torch accelerate

# 方案 B：仅走 API 调用示例（无需本地 GPU，只要能联网访问 API 即可）
pip install openai
```

> **硬件与适配人群提示**：
> - **纯概念学习者**：本章前半部分（1~3章、5章）全是原理与示例写法，**完全不需要任何环境、不需要跑代码**，直接阅读即可掌握 CoT 的核心用法。
> - **本地无 GPU 的学习者**：第四章代码实战建议跳过本地 `from_pretrained` 推理，改用 `openai`/兼容 API 的在线服务（或仅阅读代码理解流程）；强行用 CPU 跑 7B 模型推理速度会非常慢（单条可能几十秒到几分钟）。
> - **有 NVIDIA GPU（≥8GB 显存）**：可把示例中的模型换成 7B/14B 级的量化模型（4-bit/8-bit GPTQ、AWQ、bitsandbytes NF4），或直接使用更小的模型（如 1.5B/0.5B）；24GB 及以上显存可直接 FP16/BF16 跑 7B。

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

**为什么这样写**：标准提示让模型直接输出答案，中间推理过程被压缩在隐层表示中，模型容易在长链推理中"走偏"。CoT 通过要求模型"逐步思考"，把一段可读的推理文本显式写出来，让模型先输出中间步骤、再给出最终答案，**在很多推理类任务上通常能提升准确率**。

> ⚠️ **两点需要澄清，避免过度解读**：
> 1. **"提升 30~50%"不是普适规律**：CoT 带来的增益**高度依赖模型规模、任务类型、提示写法与评测方式**，从"几乎无提升"到"显著提升"都可能出现，甚至在某些任务（纯知识检索、创意写作）上无益或有害（见第五章 5.3）。请勿把某个具体百分比当作固定收益写进结论；需要数字时应引用一手论文（如 Wei et al., *Chain-of-Thought Prompting*, 2022，arXiv:2201.11903）在**特定模型+特定数据集**上的结果。
> 2. **"写出来的推理链" ≠ "模型内部真实的推理过程"**：CoT 输出的是一段**看起来像推理的文本**，它是否忠实反映了模型内部的真实计算，学界仍有争议——已有研究发现模型给出的解释可能**不忠实（unfaithful）**，即答案正确但中间步骤是事后编造的，或步骤有误却"蒙对"答案。因此不要把 CoT 文本当作"模型思路的透明记录"，它更像是一种**能改善输出、但需独立校验**的提示技巧。

### 1.2 根本问题：生成 token 数的差异带来的"工作空间"差异

> ⚠️ 澄清：**自回归语言模型无论是否用 CoT，都是逐 token 生成、每个 token 做一次前向传播**。直接回答并不是"一次前向就把完整答案算出来"。
>
> 两者的真正区别是：
> - **直接回答**：让模型只输出"最终答案"那一小段 token，中间推理过程必须被压缩在**少数几步前向的隐层状态**中；
> - **CoT**：先让模型生成若干行"推理步骤"（额外的 token），这些 token 会进入后续位置的可见上下文，相当于给了模型**更大的、可读的显式工作空间**，后续答案 token 可以基于已写好的步骤来生成。

```
直接回答：
输入问题 → 输出答案 token1 → 输出答案 token2 → ...（整个过程中，中间推理被压缩在答案前的很少几步隐层中，不可见、不可控）

CoT：
输入问题 → 输出步骤1 token → 输出步骤2 token → ... → 输出答案 token
            ↑ 步骤1 已写入上下文，后续 token 都可见    ↑ 答案基于已显式写出的步骤生成
```

当问题需要多步推理时（如数学证明、逻辑推导、因果分析），如果不给模型留出写中间步骤的空间，就会导致：
- 中间步骤出错无法被发现
- 长链推理中信息逐渐"迷失"
- 错误在后续步骤中**累积传播**

**为什么这样写**：CoT 的本质是**通过生成更多中间 token，把隐式推理变为显式、可被上下文复用的推理**。让模型把思考过程"说出来"，既方便人类检查，也让模型自身能基于已生成的中间步骤继续推导。

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
- 这里的 `confidence` 实际是**多数答案的投票占比（一致性/agreement）**，**并不是经过校准的概率置信度**：投票很一致（如 0.8）只说明多条链倾向同一答案，**不等于该答案有 80% 概率正确**——当模型对某个错误答案"稳定地一致"时，投票占比会很高但答案仍是错的。因此可把它作为"是否需要复核/升级策略"的**启发式信号**（如占比过低时触发复核或换用更强方法），而**不应把它当作真实正确率或直接据此对外承诺可靠性**。真正需要可靠置信度时，应做专门的校准评估（如可靠性图/ECE）或引入外部验证。

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

## 第四章：CoT 推理引擎的代码骨架

> ⚠️ **这是"框架/骨架代码"，不是开箱即用的推理引擎**：下面这段代码把 CoT 的**调度逻辑**（三种变体、解析步骤、投票）写完整了，但最关键的 `_call_llm()` **只是一个返回假字符串的占位实现**（见方法内注释）。也就是说，直接运行它只会拿到 `"[占位输出] ..."` 这样的模拟文本，**并不会产生真正的推理结果**。要真正跑起来，你必须把 `_call_llm()` 替换成对**真实模型**的调用（如 OpenAI/兼容 API，或本地 `transformers`/vLLM 推理），并确保输出格式能被 `_parse_reasoning_steps` / `_extract_final_answer` 正确解析。

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
        
        # ⚠️ 这里的 confidence 不是经过校准的真实概率置信度，
        #    只是单条生成时的一个占位启发式信号，不要用于拒答或可靠性承诺。
        #    如果需要可靠置信度/拒答策略，请用 self_consistency 的投票一致性，
        #    或额外做温度缩放等校准、或引入外部验证器。
        agreement_placeholder = None

        return CoTResult(
            question=question,
            reasoning_chain=steps,
            answer=answer,
            confidence=agreement_placeholder,   # 注意：None 表示"单次生成无法估计一致性"，不是0
            method='zero_shot',
        )

    async def few_shot_cot(self, question: str) -> CoTResult:
        """Few-shot CoT"""
        prompt = self.FEW_SHOT_TEMPLATE.format(question=question)
        response = await self._call_llm(prompt, temperature=0.3)
        
        steps = self._parse_reasoning_steps(response)
        answer = self._extract_final_answer(response)
        
        # ⚠️ 同 zero_shot：单条生成没有可靠的置信度估计，
        #    不要给固定阈值（如 0.75），更不要据此做生产级拒答。
        agreement_placeholder = None

        return CoTResult(
            question=question,
            reasoning_chain=steps,
            answer=answer,
            confidence=agreement_placeholder,   # None 表示"单次生成无法估计一致性"，不是0
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

        # ⚠️ vote_ratio = count / num_samples 是【投票一致性（agreement）】，
        #    不是校准后的概率置信度：
        #  - 一致性低 → 大概率不可靠（较强信号），可以触发复核/换策略
        #  - 一致性高 → 只能说明"多条答案互相一致"，≠ 答案一定正确
        #    （模型可能稳定地一致投票给同一个错误答案）
        #    因此不能直接把 0.8 / 0.9 这类高一致性作为拒答阈值的另一面，
        #    生产使用时仍需结合外部验证或人工抽检。
        vote_ratio = count / num_samples

        best_chain_idx = answers.index(best_answer)
        
        return CoTResult(
            question=question,
            reasoning_chain=chains[best_chain_idx],
            answer=best_answer,
            confidence=vote_ratio,        # 仅代表【投票一致性/agreement】
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
        """⚠️ 占位实现——必须替换为真实模型调用后本引擎才能真正工作。

        当前直接返回一段假字符串，仅用于让上面的调度/解析逻辑能跑通流程，
        它【不会】产生任何真实推理，因此本类目前不是"可用的推理引擎"。

        真实实现示例（择一，取消注释并按需改造）：

        # ① OpenAI / 兼容 API（异步）：
        # from openai import AsyncOpenAI
        # client = AsyncOpenAI()
        # resp = await client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=temperature,
        # )
        # return resp.choices[0].message.content

        # ② 本地 transformers（需在线程池中跑同步 generate，避免阻塞事件循环）：
        # import asyncio, torch
        # def _sync_gen():
        #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        #     out = model.generate(**inputs, max_new_tokens=512,
        #                          do_sample=temperature > 0, temperature=temperature)
        #     return tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
        #                             skip_special_tokens=True)
        # return await asyncio.to_thread(_sync_gen)
        """
        # 占位：返回假字符串，切勿在生产中依赖此返回值
        return f"[占位输出] 步骤1: (未接入真实模型) 最终答案: N/A  | prompt 摘要: {prompt[:30]}..."
```

---

## 第五章：CoT 的边界与权衡

### 5.1 推理链长度与准确率

> ⚠️ 下表是**经验性参考**，不是精确规则。具体"多少步开始明显掉点""是否值得拆分"高度依赖任务类型、模型能力与提示设计；表中的步数档位（1-5 / 5-10 / 10+）只是**粗略直觉**，请结合自身任务实测，不要当成硬性阈值。

| 推理步数（粗略档位） | 常见现象 | 可尝试的做法 |
|----------|------|------|
| 较短（约 1-5 步） | CoT 通常收益明显 | 直接使用 Zero-shot 或 Few-shot CoT |
| 中等（约 5-10 步） | 收益仍在，但错误可能开始累积 | 可用 Self-Consistency 提升稳定性 |
| 较长（约 10 步以上） | 错误累积风险上升，模型更易"迷失" | **考虑拆分为子问题**分别推理再组合（视任务而定，并非必须） |

**为什么这样写**：推理链越长，中间步骤出错并向后传播的概率通常越高（**错误累积效应**）。因此**较长推理时可以考虑**把问题拆成子问题分别求解、再组合——这是一种常用手段，但不同任务效果不一，需要具体验证，而非"超过 N 步就必须拆分"。

### 5.2 计算成本权衡

| 方法 | LLM 调用次数 | 准确率 | 适用场景 |
|------|-------------|--------|----------|
| Zero-shot CoT | 1 | 中等 | 简单推理、成本敏感 |
| Few-shot CoT | 1 | 较高 | 特定领域、有高质量示例 |
| Self-Consistency | 5-10 | 最高 | 高准确率需求、可接受成本 |

**为什么这样写**：Self-Consistency 需要 N 次 LLM 调用，推理成本线性增长。在成本敏感场景下，可先用 Zero-shot CoT 尝试，仅在**投票一致性偏低**（作为需要复核的启发式信号，而非校准概率）时再升级为 Self-Consistency。

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
| 所有任务都用 CoT | 纯知识检索任务通常不需要，会浪费 Token |
| 推理链过长不管理 | 较长推理可考虑拆分为子问题（视任务而定） |
| Few-shot 示例质量差 | 使用经过验证的高质量示例，错误示例会误导模型 |
| 忽视 Self-Consistency 的成本 | 成本敏感场景先用 Zero-shot，投票一致性低时再升级 |
| 假设 CoT 对任意规模都有效 | CoT 的收益总体随模型能力增强，小模型上可能收益有限（依任务而定，非绝对） |
| 把长思维链当作可信证据直接采信 | **审计"最终答案 + 可验证的简要理由 + 检索证据 + 工具调用记录 + 测试结果"**，而不是把长 CoT 文本当事实依据（推理文本未必忠实反映模型内部计算） |
| 温度设置不当 | Zero-shot 常用较低温度、Self-Consistency 常用较高温度以增加多样性（具体值需按任务调） |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **CoT 本质** | 将隐式推理变为显式推理，input → reasoning chain → output |
| **Zero-shot CoT** | 加一句"Let's think step by step"，唤醒模型推理能力 |
| **Few-shot CoT** | 提供包含推理过程的示例，让模型照猫画虎 |
| **Self-Consistency** | 生成多条推理链投票；投票占比是一致性信号，非校准置信度 |
| **Auto-CoT** | 自动聚类+生成示例，省去人工编写 |
| **错误累积** | 推理链越长，中间出错概率越高，较长链可考虑拆分 |
| **成本权衡** | Self-Consistency N 次调用，根据场景选择策略 |
| **适用范围** | 仅对需要多步推理的任务有效，知识检索无需 CoT |

**养成的好习惯**：
1. 复杂推理任务**优先尝试 CoT**，简单/纯检索任务不用
2. 需要更高稳定性时，可用 **Few-shot CoT + Self-Consistency**（按成本预算权衡，非硬性规定）
3. 较长推理链**考虑拆分**为子问题以缓解错误累积（视任务效果而定）
4. **审计"最终答案、可验证的简要理由、检索证据、工具调用记录、测试结果"**，不要把长思维链本身当作可信的事实依据（推理文本未必忠实，也可能泄漏不必要信息）
5. 根据**成本-准确率权衡**选择 CoT 变体

**下一步预告**：掌握了逐步推理后，你将进入 **幻觉问题（Hallucination）** 的学习——理解大语言模型为什么会"一本正经地胡说八道"、幻觉的分类与成因、以及当前主流的检测与缓解策略。这是使用大语言模型时必须正视的核心风险。继续加油！