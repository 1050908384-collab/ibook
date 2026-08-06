# 基于人类反馈的强化学习（RLHF）：整体流程与目标

> **参考文章链接**
> 
> ① 底稿：《RLHF/PPO 算法原理与工程实践全景指南——从PPO、DPO 到GRPO》
> 　　网址：https://blog.csdn.net/RichardsZ_/article/details/155745813
> 
> ② 参考文章1：《【强化学习】1、通俗理解 RLHF/PPO/DPO/GRPO》
> 　　网址：https://blog.csdn.net/jiaoyangwm/article/details/145832284
> 
> ③ 参考文章2：《一文详解大模型强化学习（RLHF）算法：PPO、DPO、GRPO、ORPO、KTO、GSPO》
> 　　网址：https://blog.csdn.net/zhangzhentiyes/article/details/151154629

---

**适合人群**：已完成预训练（Pretrain）与有监督微调（SFT）阶段学习，希望理解如何将"会办事"的指令模型进一步打磨为"做对事"的对齐模型的进阶学习者。每一步都拆解到"是什么、为什么、怎么做"，你只需要跟着理解。完成本章后，你将能够清晰描述 RLHF 的完整三阶段流程，掌握奖励模型训练、PPO/DPO/GRPO 三种强化学习优化算法的原理与代码实现，并能在实际项目中做出合理的算法选型。

**预计学习时长**：约 2 小时  
**环境准备**：Python 3.10+，安装 `transformers`、`trl`、`peft`、`datasets`、`accelerate`

```bash
pip install transformers trl peft datasets accelerate
```

---

## 第一章：RLHF 的定位——从"会办事"到"做对事"

### 1.1 三阶段训练中的 RLHF

大语言模型从"出生"到"上岗"，通常经历三个阶段：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  预训练 PT   │ → │  有监督微调 SFT │ → │   对齐 RLHF  │
│ Pre-training│    │ Supervised  │    │ Reinforcement│
│             │    │ Fine-Tuning │    │ from Human   │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                  ↓                  ↓
学习通用语言能力      学习指令遵循能力      对齐人类价值观
（读遍互联网文本）    （学习问答对话格式）   （安全、有用、诚实）
```

**RLHF 的核心定位**：
- **承上**：利用 SFT 模型学到的指令遵循能力
- **核心作用**：让模型不仅"会办事"，还要"做对事"——生成安全、有用、诚实且符合人类偏好的内容
- **关键挑战**：人类的偏好是复杂且主观的，很难用唯一的"正确答案"来概括

### 1.2 为什么 SFT 还不够？

SFT 用大量高质量的"指令-回答"对来教模型模仿人类的说话方式，但它有两个根本性局限：

| 局限 | 说明 | 示例 |
|------|------|------|
| **无法覆盖所有偏好** | 同一指令有无数优秀回答，SFT 只能覆盖一小部分 | "请写一首关于秋天的诗"——SFT 数据只有一种风格 |
| **无法区分好坏** | SFT 只教模型"模仿"，不教模型"判断" | 模型可能学到冗长但空洞的回答模式 |
| **可能放大有害内容** | 预训练语料中的偏见、谣言在 SFT 中可能被保留 | 模型对敏感问题给出不安全的回答 |

**RLHF 的核心思想**：不再直接告诉模型"正确答案是什么"，而是让模型生成多个回答，由人类评判好坏，再利用这些偏好反馈来"奖励"或"惩罚"模型，引导其生成更符合人类偏好的内容。

### 1.3 RLHF 与 SFT 的本质区别

| 维度 | 有监督微调（SFT） | 基于人类反馈的强化学习（RLHF） |
|------|------------------|------------------------------|
| **目标** | 学习指令-回答的映射关系 | 优化人类偏好的最大化 |
| **数据** | 指令-回答对（单条标准答案） | 成对偏好数据（A > B）或奖励分数 |
| **反馈信号** | 交叉熵损失（与标准答案的差异） | 奖励模型分数 + KL 散度约束 |
| **训练方式** | 监督学习 | 强化学习 |
| **输出能力** | 遵循指令、对话 | 对齐价值观、安全、有用、诚实 |

**为什么这样写**：SFT 让模型"识字"和"懂事"，RLHF 让模型"明辨是非"。没有 RLHF，模型可能生成语法正确但内容有害或不符合偏好的回答。

---

## 第二章：奖励模型训练——培养自动裁判

### 2.1 为什么需要奖励模型？

强化学习需要**数值化的奖励信号**。但人类反馈通常是"回答 A 比回答 B 好"这样的**相对比较**，不是绝对分数。奖励模型（Reward Model, RM）就是把这个相对偏好转化为绝对分数的桥梁。

**类比**：就像奥运会裁判——人类观众能看出"选手 A 比选手 B 跳得好"，但裁判需要给出一个具体的分数（如 9.5 分），才能决定奖牌归属。

### 2.2 数据收集：成对偏好数据

对于同一个指令 $x$，让 SFT 模型生成多个不同的回答，然后由人类标注员进行**成对排序**：

```json
{
  "prompt": "请解释什么是'强化学习'",
  "chosen": "强化学习是让智能体通过试错从环境反馈中学习，不断优化策略以获得更多奖励的技术",
  "rejected": "强化学习是一种机器学习"
}
```

**为什么这样写**：
- 人类更擅长做**相对比较**（A 比 B 好）而不是**绝对评分**（给 A 打 90 分）
- 成对比较的数据一致性更高，不同标注者的分歧更小

### 2.3 奖励模型的架构

奖励模型通常与 SFT 模型**同规模**，但修改了最后的输出层：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

# 加载为序列分类模型，num_labels=1 表示输出标量分数
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,  # 输出一个标量：奖励分数
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"奖励模型参数量: {sum(p.numel() for p in reward_model.parameters()) / 1e9:.2f}B")
```

**为什么这样写**：
- `num_labels=1`：将词表映射层替换为单值回归层，输出标量奖励分数
- 使用与 SFT 模型相同的基座，确保语义理解能力一致
- 奖励模型通常**冻结大部分层**，只训练最后的回归层，防止过拟合

### 2.4 奖励模型的训练

```python
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

# 加载偏好数据集（HuggingFace 标准格式）
# 每条数据包含：prompt, chosen, rejected
dataset = load_dataset("json", data_files="preference_data.jsonl", split="train")

# 配置训练参数
training_args = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,  # 奖励模型学习率要小，防止破坏语义理解
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

# 初始化训练器
trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
trainer.train()

# 保存奖励模型
trainer.save_model("./reward_model")
```

**为什么这样写**：
- `RewardTrainer`（来自 `trl` 库）自动处理成对偏好数据的损失计算
- 学习率 `1e-5` 比 SFT 更小，因为奖励模型只需要"微调"打分能力，不需要改变语义理解
- 通常只训练 **1 个 epoch**，防止奖励模型对训练数据过拟合

### 2.5 奖励模型的训练目标

假设对于指令 $x$，人类偏好 $y_w$（winner）优于 $y_l$（loser），训练目标是最大化以下概率：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

**直观理解**：
- 如果 $r_\theta(x, y_w) = 0.9$，$r_\theta(x, y_l) = 0.3$
- 那么 $\sigma(0.9 - 0.3) = \sigma(0.6) \approx 0.65$，损失就小
- 如果分数差很小甚至反转，损失就大，模型会被"惩罚"

**为什么这样写**：这个公式本质上是在说——**好回答的分数必须显著高于差回答的分数**。奖励模型不需要知道"绝对的好是多少分"，只需要学会"相对的好"。

---

## 第三章：PPO 强化学习优化——经典路径

### 3.1 PPO 的核心直觉：温和引导，防止"一步走太大"

PPO（Proximal Policy Optimization）是 OpenAI 在 2017 年提出的强化学习算法，长期以来是 RLHF 第三阶段的**绝对主力**。

**生活化比喻**：想象你教一个孩子玩游戏。如果每次尝试后你都给予非常剧烈的反馈（极高赞扬或严厉批评），孩子可能会困惑甚至放弃。更好的方法是：**在他当前行为基础上，温和地引导他向更好方向改进，每次只做小小调整**。

PPO 通过一个**裁剪机制（Clipping）**，将策略更新限制在一个"信任区域"内，防止模型性能急剧下降。

### 3.2 PPO 的四模型架构

PPO 需要同时维护**四个模型**：

| 模型 | 角色 | 是否可训练 | 作用 |
|------|------|-----------|------|
| **Actor（策略模型）** | 生成回答 | ✅ 可训练 | 根据指令生成文本，是被优化的对象 |
| **Reference Model（参考模型）** | 基准对照 | ❌ 冻结 | SFT 模型的副本，计算 KL 散度防止跑偏 |
| **Reward Model（奖励模型）** | 打分裁判 | ❌ 冻结 | 给生成的回答打分 |
| **Critic（价值模型）** | 价值评估 | ✅ 可训练 | 预测每个状态的期望回报，用于计算优势 |

```
┌─────────────┐      Response      ┌─────────────────┐
│   Actor     │ ─────────────────→ │  Reference Model │
│  (可训练)   │                    │    (冻结)        │
└─────────────┘                    └─────────────────┘
       │                                    │
       │                                    ↓
       │                              KL 散度约束
       │                                    │
       ↓                                    ↓
┌─────────────┐                    ┌─────────────────┐
│  Reward     │ ←──────────────── │   最终奖励计算   │
│   Model     │      Raw Score    │  R_total = R_raw │
│  (冻结)     │                   │    - β·KL       │
└─────────────┘                   └─────────────────┘
                                            │
                                            ↓
                                     ┌─────────────┐
                                     │   Critic    │
                                     │  (可训练)   │
                                     │  预测价值 V  │
                                     └─────────────┘
                                            │
                                            ↓
                                     ┌─────────────┐
                                     │  优势计算    │
                                     │  Advantage  │
                                     │  = R_total  │
                                     │    - V      │
                                     └─────────────┘
```

### 3.3 PPO 的损失函数

$$\mathcal{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：新策略与旧策略的概率比
- $\hat{A}_t$：优势函数（Advantage），表示这个动作比平均水平好多少
- $\epsilon$：裁剪阈值（通常 0.1 或 0.2）

**解读**：
- 当 $\hat{A}_t > 0$（好回答）：我们希望增大 $r_t$，但 `clip` 将其上限限制在 $1+\epsilon$，防止过度优化
- 当 $\hat{A}_t < 0$（差回答）：我们希望减小 $r_t$，但 `clip` 将其下限限制在 $1-\epsilon$，防止过度惩罚

### 3.4 PPO 训练代码

> ⚠️ **下面是"教学用伪代码"，用于说明 PPO 的数据流，并非可直接运行的 TRL 代码。**
> 真实的 `trl` API 在不同版本间差异较大（`PPOTrainer` 的初始化参数、`generate/step` 的签名、奖励模型的调用方式都不同）。**尤其注意：奖励模型不能直接吃原始字符串，必须先分词成张量再前向；`reward_model(...)` 返回的是 logits，需要取出标量分数。** 生产中请以你所用 TRL 版本的官方文档/示例为准。

```python
# ===== 教学伪代码：仅示意流程，不保证按现有 TRL 版本可运行 =====
from trl import PPOTrainer, PPOConfig
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch

# 加载 SFT 模型作为 Actor
actor_model = AutoModelForCausalLM.from_pretrained(
    "./sft_model", torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

# 加载奖励模型（冻结）。注意：num_labels=1 时输出一个标量分数（logit）
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model", num_labels=1, torch_dtype="auto", device_map="auto"
)
reward_model.eval()

@torch.no_grad()
def score_with_reward_model(texts):
    """奖励模型必须【先分词成张量】再前向，返回值取 logits 作为标量奖励。"""
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(reward_model.device)
    out = reward_model(**enc)            # 前向：得到 SequenceClassifierOutput
    scores = out.logits.squeeze(-1)     # [batch, 1] -> [batch]，每条一个分数
    return [s.detach().float().cpu() for s in scores]

ppo_config = PPOConfig(
    learning_rate=1e-5, batch_size=4, mini_batch_size=1,
    gradient_accumulation_steps=4, cliprange=0.2, kl_coef=0.05,
)

ppo_trainer = PPOTrainer(
    config=ppo_config, model=actor_model,
    tokenizer=tokenizer, dataset=train_dataset,  # 仅含 prompt
)

for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        query_tensors = batch["input_ids"]                 # 已分词的 prompt 张量
        # 1. Actor 生成回答（返回的是 token 张量，不是字符串）
        response_tensors = ppo_trainer.generate(query_tensors)

        # 2. 解码成文本 → 交给奖励模型【重新分词】后打分
        texts = [
            tokenizer.decode(torch.cat([q, r]), skip_special_tokens=True)
            for q, r in zip(query_tensors, response_tensors)
        ]
        rewards = score_with_reward_model(texts)           # List[标量张量]

        # 3. 与 Reference Model 的 KL 惩罚由 trl 内部处理
        # 4. PPO 更新（step 需要 query/response 的【张量】与奖励列表）
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
```

**为什么这样写**：
- **奖励模型的输入必须是分词后的张量**，而不是把 `["文本", ...]` 直接传进模型；输出是 `logits`，要 `squeeze` 成每条一个标量分数。原先 `reward_model(texts)` 直接喂字符串、把返回值当奖励，是无法运行的。
- `PPOTrainer.generate/step` 接收的是 **token 张量**（query/response），不是纯字符串；奖励需与之对齐。
- **API 会随版本变化**：本段仅示意"生成→打分→更新"的数据流，落地请对照所用 `trl` 版本的官方示例。

### 3.5 PPO 的优缺点

| 优点 | 缺点 |
|------|------|
| 训练稳定，是久经考验的工业标准 | 需要同时维护 4 个模型，显存消耗巨大 |
| 适用于各种复杂奖励函数 | 超参数调优困难（$\epsilon$、$\beta$、学习率等） |
| 通用性强，对话/摘要/代码都能用 | 长文本推理中 Critic 价值估计不准 |

**适用场景**：中等规模模型（7B-70B）的通用文本生成优化，如对话系统、文本摘要。

---

## 第四章：DPO 直接偏好优化——轻量路径

### 4.1 DPO 的核心直觉：跳过奖励模型

DPO（Direct Preference Optimization）由斯坦福大学于 2023 年提出，它的核心洞见是：

> **我们最终的目标就是让模型知道"A 比 B 好"，为什么非要先训练一个奖励模型，再回头去学这个偏好？**

**生活化比喻**：
- **传统 RLHF**：先学打分（这道菜 80 分），再根据分数改进
- **DPO**：直接记住"宫保鸡丁比鱼香肉丝更受欢迎"

DPO 通过精妙的数学变换，证明了可以直接从偏好数据推导出策略更新的损失函数，**将复杂的强化学习问题转化为了标准的分类问题**。

### 4.2 DPO 的损失函数

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

**解读**：
- $\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$：当前模型相对于参考模型，生成回答 $y$ 的概率增加了多少（**隐式奖励**）
- 核心：最大化 **隐式奖励(Winner) - 隐式奖励(Loser)**
- 最外层的 $-\log\sigma$ 结构，使目标变成了**标准的二元交叉熵损失**

**精妙之处**：不需要 Reward Model，不需要 Critic，不需要复杂的采样循环——**只需要 2 个模型**（策略模型 + 参考模型）！

### 4.3 DPO 训练代码

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 加载 SFT 模型作为初始策略
model = AutoModelForCausalLM.from_pretrained(
    "./sft_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

# 加载参考模型（SFT 模型的冻结副本）
ref_model = AutoModelForCausalLM.from_pretrained(
    "./sft_model",
    torch_dtype="auto",
    device_map="auto"
)

# 加载偏好数据集
# 格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("json", data_files="dpo_data.jsonl", split="train")

# DPO 配置
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,  # DPO 学习率通常比 SFT 小
    beta=0.1,  # KL 散度惩罚系数，对应公式中的 β
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

# 初始化 DPO 训练器
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 参考模型
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
)

# 开始训练
dpo_trainer.train()

# 保存模型
dpo_trainer.save_model("./dpo_model")
```

**为什么这样写**：
- `DPOTrainer` 自动处理偏好对的损失计算
- `beta=0.1`：控制 KL 散度惩罚，防止模型偏离参考模型太远
- 学习率 `5e-7` 比 SFT 小一个量级，因为 DPO 直接优化策略，更新幅度需要更谨慎

### 4.4 DPO 的优缺点

| 优点 | 缺点 |
|------|------|
| 流程极简，类似监督学习，实现成本低 | 对偏好数据质量极其敏感 |
| 计算成本比 PPO 低 30%~50% | 复杂多维度奖励难以适配 |
| 训练稳定，不易崩溃 | 长文本推理效果较弱 |

**适用场景**：资源有限、需要快速对齐人类偏好的场景，如开源模型的轻量微调、短文本对话。

---

## 第五章：GRPO 组别相对优化——推理专用路径

### 5.1 GRPO 的核心直觉：不需要"绝对老师"，用"同学互评"

GRPO（Group Relative Policy Optimization）由 DeepSeek 团队提出，旨在结合 PPO 的在线探索优势和 DPO 的计算效率。

**生活化比喻**：
- **PPO**：一对一辅导，专门的老师（Critic）给每个回答打分
- **DPO**：对着标准答案自学，没有实时指导
- **GRPO**：**专家会诊**——让模型自己生成一组回答，组内互相比较

面对一个难题，不需要权威老师告诉模型"这个方案得 85 分"。只需要让模型生成一组（如 8 个）不同的方案，放在一起比较："方案 A 比方案 B 好，方案 C 最差。"

### 5.2 GRPO 的关键机制：Group Relative Advantage

GRPO 最显著的特点是**摒弃了价值模型（Critic）**：

1. **组采样**：对指令 $q$，策略模型采样生成 $G$ 个回答 $\{o_1, o_2, ..., o_G\}$
2. **奖励计算**：用奖励函数（或规则验证）给每个回答打分，得到 $\{r_1, r_2, ..., r_G\}$
3. **优势估计**：用组内统计特性计算优势（无需 Critic！）：

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

4. **策略更新**：使用类似 PPO 的 Clipped Objective 进行更新

**组内平均分自然充当了基线（Baseline）**。高于平均分的回答优势为正，模型会强化其生成概率；反之则抑制。

### 5.3 GRPO 训练代码

> ⚠️ **教学伪代码**：TRL 的 `GRPOTrainer`/`GRPOConfig` 参数随版本变化，请以你所用版本官方文档为准。

```python
# ===== 教学伪代码：示意 GRPO 用法，不保证按现有 TRL 版本可运行 =====
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 SFT 模型
model = AutoModelForCausalLM.from_pretrained(
    "./sft_model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

# 定义奖励函数（数学/代码任务中常用规则验证）
def reward_func(completions, **kwargs):
    """
    奖励函数：根据回答的正确性打分
    completions: 模型生成的一组回答
    """
    rewards = []
    for completion in completions:
        # 示例：检查回答中是否包含正确答案"42"
        if "42" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 加载指令数据集（只有 prompt）
dataset = load_dataset("json", data_files="grpo_prompts.jsonl", split="train")

# GRPO 配置
training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,
    num_generations=8,  # 每组采样 8 个回答（G=8）
    beta=0.04,  # KL 散度系数
    logging_steps=10,
    bf16=True,  # 优先 bfloat16（无需 GradScaler）
)

# 初始化 GRPO 训练器
grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_func,  # 自定义奖励函数
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
grpo_trainer.train()

# 保存模型
grpo_trainer.save_model("./grpo_model")
```

**为什么这样写**：
- `num_generations=8`：每组采样 8 个回答，这是 GRPO 的核心超参数
- `reward_funcs`：GRPO 支持自定义奖励函数，特别适合数学/代码等可验证任务
- 不需要 Critic 模型，因此相比 PPO 可省下价值网络的显存与算力（具体节省比例取决于模型与实现）

### 5.4 GRPO 的优缺点

| 优点 | 缺点 |
|------|------|
| 去掉 Critic，显存/算力低于 PPO | 每个输入要生成多个（如 5~10 个）回答，采样开销更大 |
| 无需训练价值模型 | 全对/全错样本组导致梯度消失 |
| 长推理效果显著（数学/代码） | 对奖励噪声敏感 |

**适用场景**：长链推理任务（数学解题、代码调试）、大规模模型（70B+）的微调。

---

## 第六章：三种路径的全面对比与选型

### 6.1 核心维度对比表

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| **核心思想** | 在信任区域内小步更新 | 直接用偏好数据优化 | 组内相对比较估计优势 |
| **所需模型** | 4 个（Actor/Critic/RM/Reference） | 2 个（策略+参考） | 无固定"三模型"：一定有策略模型；奖励可用**独立 RM 或规则/可验证函数**（后者无需 RM）；参考模型是否需要、KL 是否启用**取决于具体算法与实现/配置** |
| **计算成本** | **最高** | **最低** | **中等** |
| **数据需求** | 需要奖励模型的绝对分数 | 需要成对偏好数据 | 需要一个打分来源：**RM 分数或规则/可验证奖励** |
| **训练稳定性** | 高（但调参复杂） | 高 | 中（全对/全错问题） |
| **长推理能力** | 弱（Critic 价值估计不准） | 弱 | **强** |
| **适用场景** | 通用对话、文本生成 | 快速对齐、资源有限 | 数学推理、代码生成 |

### 6.2 选型参考（非硬性规则）

> ⚠️ 下面是一张**经验性参考图**，用来帮你快速缩小候选范围，**不是"必须这样选"的通用铁律**。真实选型还取决于数据质量、算力预算、奖励可得性、稳定性要求与团队工程经验；同一场景往往有多个可行算法，需通过小规模实验比较后再定。图中的规模阈值（如 7B/70B）也只是粗略档位，不应理解为精确分界。

```
（以下为经验倾向，可交叉、可例外，需结合实测）
看模型规模与架构（阈值仅为粗略参考）
├── MoE 模型（很大规模）→ 可考虑 GSPO 等针对 MoE 更稳的做法
├── 稠密大模型 → GRPO（偏长推理）或 PPO（偏通用）常见，但非唯一
└── 中小规模稠密模型 → 往下看数据资源

看数据资源
├── 主要是单标签/好坏信号 → 可考虑 KTO
├── 有成对偏好数据 → 常用 DPO（实现简单、成本低）
├── 既有偏好又有监督数据 → 可考虑 ORPO
└── 有 RM 或可训练 RM / 有可验证的规则奖励 → PPO 或 GRPO 均可

看任务类型
├── 长链推理（数学/代码，答案可验证）→ GRPO 常有优势
├── 短文本对话/通用生成 → PPO / DPO / ORPO 都可尝试
└── 低成本快速对齐 → DPO / KTO 常见
```

### 6.3 实践建议

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 小团队，短文本对话 | DPO | 成本低，实现简单 |
| 中团队，长推理任务 | GRPO | 省去 Critic，长文本效果好 |
| 大团队，大规模 MoE | GSPO | 解决训练崩溃，释放千亿参数潜力 |
| 已有高质量奖励模型 | PPO | 通用性强，工业标准 |

---

## 第七章：常见陷阱与避坑清单

| 坑点 | 现象 | 正确做法 |
|------|------|----------|
| **Reward Hacking** | 训练曲线 reward 高，但实测效果差 | 设计端到端的 rule-based 奖励；加强 KL 约束；人工抽检 |
| **Reward 崩盘** | reward 突然断崖式下降 | 减小学习率；启用 Reject Sampling；检查 KL 散度是否爆炸 |
| **熵坍缩** | 输出几乎无多样性，回答雷同 | 调大 Temperature；长 CoT 预热；选择初始策略熵高的模型 |
| **全对/全错（GRPO）** | 一组样本全对或全错，优势≈0 | 引入动态采样，丢弃全对/全错组；使用 DAPO 改进 |
| **灾难性遗忘** | 模型忘记通用知识，只会对齐任务 | 保留 10%~20% 通用 SFT 数据；控制 KL 散度 |
| **奖励模型过拟合** | 奖励模型在训练集上表现好，泛化差 | 早停（Early Stopping）；只训练 1 个 epoch；冻结底层 |
| **学习率过大** | 模型输出混乱，KL 散度爆炸 | PPO 用 1e-5，DPO 用 5e-7，GRPO 用 1e-6，务必 warmup |

---

## 本章小结

| 知识点 | 核心要点 |
|--------|----------|
| **RLHF 定位** | SFT 之后，让模型从"会办事"进化为"做对事" |
| **三阶段流程** | SFT → 奖励模型训练 → 强化学习优化 |
| **奖励模型** | 将人类相对偏好转化为绝对分数，Bradley-Terry 损失 |
| **PPO** | 四模型协同，裁剪机制保证稳定，适合通用场景 |
| **DPO** | 跳过奖励模型，直接偏好优化，成本低、实现简单 |
| **GRPO** | 组内相对比较，去掉 Critic，适合长推理任务 |
| **KL 散度** | 防止模型为了刷分而偏离原始语言能力 |
| **Reward Hacking** | 模型钻奖励函数漏洞，需 rule-based 奖励 + 人工抽检 |
| **选型原则** | 通用对话选 PPO/DPO，长推理选 GRPO，MoE 选 GSPO |

---

## 养成的好习惯

1. **奖励函数设计是第一生产力**——宁可花 50% 时间设计奖励函数，也不要让模型在错误的奖励信号上训练
2. **从小学习率开始**——RL 阶段的学习率通常比 SFT 小 1~2 个量级
3. **每轮保存检查点**——RL 训练不稳定，随时可能崩溃，保留多个检查点便于回滚
4. **部署前做人工抽检**——自动化指标无法覆盖所有问题，特别是安全性和价值观对齐
5. **监控 KL 散度**——KL 散度是防止模型"跑偏"的生命线，超过阈值立即停训检查
6. **混合通用数据**——RLHF 训练中保留 10%~20% 通用 SFT 数据，防止灾难性遗忘

---

## 下一步预告

掌握了 RLHF 后，你已经完整理解了大语言模型从"预训练 → SFT → 对齐"的三阶段训练流程。接下来，你将进入 **HuggingFace 生态工具链** 的学习——深入掌握 `transformers` 库的 `AutoModel/AutoTokenizer` 加载机制、`datasets` 库的数据处理（map/filter/shuffle），以及 `Trainer` 训练器的使用及 `TrainingArguments` 参数配置。这是将理论转化为工程实践的必备工具箱，也是后续"参数高效微调（LoRA 与 QLoRA）"和"分布式训练（DeepSpeed）"的技术基础。继续加油！