# 预训练（Pretrain）：大语言模型的"筑基"阶段

> **参考文章链接**
> 
> ① 底稿：《LLM之幻觉（一）：大语言模型幻觉解决方案综述》
> 　　网址：https://blog.csdn.net/weixin_51762789/article/details/162148537
> 
> ② 参考文章1：《【LLM】3：从零开始训练大语言模型（预训练PT、微调SFT、RLHF）》
> 　　网址：https://blog.csdn.net/zzZ_CMing/article/details/143591945
> 
> ③ 参考文章2：《一文搞懂大模型预训练PT与有监督微调SFT》
> 　　网址：https://blog.csdn.net/Python_cocola/article/details/141070679

---

**适合人群**：已掌握 Transformer 架构与注意力机制，希望理解大语言模型"出生"过程——预训练阶段的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着理解。完成本章后，你将掌握预训练的完整流程、数据构造方法、训练目标与评估指标，为后续"有监督微调（SFT）""RLHF"打下坚实基础。

**预计学习时长**：约 1.5 小时  
**环境准备（仅用于本章教学示例/小规模验证代码）**：Python 3.10+，`transformers`、`torch`、`datasets`、`accelerate`、`tensorboard`。

```bash
# 教学示例环境；真实大规模预训练需要完整的 Megatron-LM / Megatron-DeepSpeed / Megatron-Core 等分布式框架，不在此处覆盖
pip install transformers torch datasets accelerate tensorboard evaluate
```

> **⚠️ 硬件门槛与适配人群提示（本章最重要的提示）**：
> - **真实"从零预训练一个现代 LLM（≥1B 参数、≥100B token 语料）"的硬件要求极高，普通学习者几乎不可能满足**：通常需要几十到上百张 A100/H100 级别的 GPU 组成集群，通过 3D 并行（数据/张量/流水线）+ ZeRO/FSDP + 优化器融合才能在数周内跑完；单机多卡或消费级 GPU 只能做**非常小的 Toy Demo**。
> - **99% 的学习者（强烈推荐）**：把本章定位在"理解概念与流程"——重点搞清楚数据清洗/去重、分桶、next-token loss、PPL 评估、checkpoint 恢复这些环节的目的；**不要尝试自己从零做真实预训练**，直接下载开源基座（如 Qwen2.5-7B、Llama 3、Mistral 等）进入 SFT/RLHF 阶段即可，这是业界绝大多数团队的真实路径。
> - **想动手验证代码流程的学习者**：可以把模型尺寸压到 ≤125M（例如 GPT2 small 级别）、数据量压到几十 MB、MAX_LENGTH 压到 256~512，用单卡消费级 GPU（≥8GB 显存）跑几个 step 验证 loss 在下降、PPL 在合理区间——**目的是验证流水线能跑通，而不是得到一个能用的模型**。
> - **有 ≥8×A100/H100 集群资源的团队**：可以在此基础上参考 Megatron-LM、Megatron-DeepSpeed、DeepSpeed Ulysses 的官方示例，把并行、混合精度、梯度检查点、FlashAttention2 等真正生产级开关打开，再把模型规模和语料量拉上去。

---

## 第一章：预训练是什么——让模型"读遍天下书"

### 1.1 三阶段训练流程概览

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

**预训练（Pre-training）** 是第一阶段，也是**最耗时、最烧钱**的阶段。它的目标是让模型通过海量无标注文本，学习语言的通用规律、世界知识和基本推理能力。

### 1.2 预训练的核心任务

预训练的任务极其简单：**给定前面的词，预测下一个词**。

```
输入: "今天天气"
预测: "很好"

输入: "今天天气很好"
预测: "，"

输入: "今天天气很好，"
预测: "适合"
...
```

**为什么这样写**：这个任务被称为**自回归语言建模（Autoregressive Language Modeling）**。虽然任务简单，但当训练数据达到 TB 级别、模型参数达到百亿甚至千亿时，模型被迫学习到：
- 语法结构（主谓宾搭配）
- 语义关系（"猫"和"狗"是同类）
- 世界知识（"爱因斯坦"是物理学家）
- 基本推理（数学计算、逻辑推断）

---

## 第二章：预训练数据——"吃什么"决定"长成什么样"

### 2.1 数据来源

预训练数据主要来自公开的大规模文本语料：

| 数据集 | 内容 | 特点 |
|--------|------|------|
| **Common Crawl** | 互联网网页抓取 | 规模最大，但噪声多、质量参差不齐 |
| **Wikipedia** | 百科词条 | 质量高、知识密集、结构清晰 |
| **BooksCorpus** | 图书文本 | 叙事性强、语言流畅 |
| **GitHub** | 代码仓库 | 让模型学会编程 |
| **ArXiv** | 学术论文 | 让模型学会科学推理 |

**为什么这样写**：数据多样性决定了模型的"知识面"。只读新闻的模型不懂诗歌，只读代码的模型不懂情感。

### 2.2 数据清洗——垃圾进，垃圾出

原始数据充满噪声，必须经过严格清洗：

```python
# 数据清洗的典型流程
def clean_text(text):
    # 1. 去除 HTML 标签
    text = remove_html_tags(text)
    
    # 2. 去除无意义字符（过多换行、乱码）
    text = remove_garbage_chars(text)
    
    # 3. 去重（删除重复句子、段落）
    text = deduplicate(text)
    
    # 4. 过滤低质量内容（太短、无意义）
    if len(text) < 100 or is_gibberish(text):
        return None
    
    # 5. 敏感内容过滤（可选）
    text = filter_sensitive(text)
    
    return text
```

**为什么这样写**：研究表明，**数据质量比数据数量更重要**。用 100B 的高质量数据训练，效果可能优于 1T 的低质量数据。

### 2.3 分词与词汇表构建

模型不认识文字，只认识数字。分词（Tokenization）将文本转换为 token 序列。

```python
from transformers import AutoTokenizer

# 以 GPT-2 的 BPE 分词器为例
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "LLMs are cool."
tokens = tokenizer.tokenize(text)
print(tokens)   # 会被切成若干子词，如 'LL'、'Ms' 等（GPT-2 用 'Ġ' 标记词前空格）

token_ids = tokenizer.encode(text)
print(token_ids)  # 每个子词对应一个整数 ID
# 注意：具体切分结果与 ID 取决于分词器版本，请以实际运行输出为准，切勿硬记
```

**为什么这样写**：
- **BPE（Byte-Pair Encoding）**：从字符/字节开始，逐步合并高频组合，平衡词汇表大小和表达能力。除 BPE 外，业界还常用 **Unigram（SentencePiece 常见）**、**WordPiece（BERT 系列）**、以及 **Byte-level BPE（GPT-2/LLaMA 系列，直接以单字节为最底层原子）**。
- **词表大小没有通用的"必须 3 万~5 万"**：实际取值高度依赖**目标语言、tokenizer 算法、模型规模和预训练语料**——
  - 英语为主的 Byte-level BPE 常见词表 32K~128K 档（如 GPT-2 约 50K）；
  - 中文/多语言模型词表通常更大（100K 甚至 150K 档也不罕见），因为要覆盖多语言字符、汉字与特殊标记。
  选型原则是"在**分词后总 token 数更少**（压缩率更高）与**embedding/输出层参数量可控**之间做取舍，不是固定阈值。
- **现代子词分词一般不会产生真正意义上的 OOV（整词未登录就直接崩掉的情况）**：
  - BPE/Unigram/WordPiece 会把生僻词、新造词拆成多个更小的子词甚至单字符/单字节来表示，而不是直接标 `[UNK]`；
  - 只有**词表最底层不是 byte/字符级**、且强行用 `[UNK]` 兜底的老派 word-level 分词器，才会出现"大量 OOV 损害效果"的问题。
  （拆分例 "Chat"+"G"+"PT" 只是手写示意，实际拆分方式以所用分词器的真实合并规则为准。）

### 2.4 数据格式：移位构造训练样本

预训练的核心是"预测下一个 token"。为了高效训练，使用**移位构造（Shifted Construction）**：

```python
import torch
import torch.nn.functional as F

# 原始文本经过分词后的 token ID 序列
# "你知道什么是预训练吗？" → [2, 5, 6, 1, 8, 7, 6]
# 假设最大序列长度为 10，后面补 0（padding）
sequence = [2, 5, 6, 1, 8, 7, 6, 0, 0, 0]

# 移位构造：输入 X 和 目标 targets 错位一位
X =      torch.tensor([2, 5, 6, 1, 8, 7, 6, 0, 0, 0])   # 输入
targets = torch.tensor([5, 6, 1, 8, 7, 6, 0, 0, 0, 0])  # 目标（X 左移一位）

# 这样，模型在位置 0 看到 2，要预测 5
# 在位置 1 看到 2,5，要预测 6
# ...一次性计算整个序列的所有位置的损失
```

**为什么这样写**：
- 如果不移位，需要把序列拆成多个 `(前缀, 下一个词)` 对，效率极低
- 移位构造允许**一次性把整个序列喂给模型**，通过因果注意力掩码（Causal Mask）确保每个位置只能看到前面的词，然后同时计算所有位置的损失

---

## 第三章：模型架构与初始化

### 3.1 架构选择：Decoder-Only 是主流

当前大语言模型主要采用三种架构：

| 架构类型 | 代表模型 | 特点 | 适用场景 |
|----------|----------|------|----------|
| **Encoder-Only** | BERT | 双向注意力，理解能力强 | 文本分类、NER |
| **Encoder-Decoder** | T5、BART | 兼顾理解和生成 | 翻译、摘要 |
| **Decoder-Only** | GPT、Llama、Qwen | 自回归生成，涌现能力强 | 通用大模型 |

**为什么这样写**：GPT 系列证明，Decoder-Only 架构在**大规模预训练**下展现出最强的涌现能力（如 ICL、CoT）。当前主流大模型（GPT-4、Llama、Qwen）均采用此架构。

### 3.2 超参数设计

```python
# 以 GPT-3 175B 为例的关键超参数
config = {
    "vocab_size": 50257,      # 词汇表大小
    "n_layers": 96,           # Transformer 层数
    "d_model": 12288,         # 模型维度（隐藏层大小）
    "n_heads": 96,            # 注意力头数
    "d_head": 128,            # 每个头的维度 (12288 / 96)
    "context_length": 2048,   # 最大上下文长度
    "dropout": 0.0,           # 预训练通常不用 dropout
}
```

**设计原则**：
- **参数量随层数与维度增长**：总参数量主要由"层数 × 维度² "量级决定（GPT-3 175B 即由 96 层、d_model=12288 等配置堆叠而成），并非层数与维度的简单相乘
- **头数要能整除维度**：`d_model % n_heads == 0`
- **去除偏置项**：去除 dense 层和 layer norm 的偏置项，提升训练稳定性（Llama 系列标准做法）

### 3.3 位置编码

Transformer 本身没有位置概念，需要额外注入位置信息：

| 位置编码 | 代表模型 | 特点 |
|----------|----------|------|
| **绝对位置编码** | GPT-2 | 每个位置有固定嵌入向量 |
| **RoPE（旋转位置编码）** | Llama、Qwen | 通过旋转矩阵注入相对位置信息，是目前的主流选择 |
| **ALiBi** | BLOOM | 基于距离的注意力偏置，训练更稳定 |

**为什么这样写**：RoPE 是当前最主流的选择，它让模型不仅能理解"第几个词"，还能理解"两个词相距多远"。不过要注意：**原始 RoPE 超出训练长度后并不必然稳定**，"外推能力强"是一种常见但被夸大的说法；要真正把上下文扩展到远超训练长度，通常还需配合**位置插值、NTK-aware/YaRN 等长上下文扩展方法**，而非仅靠 RoPE 本身。

---

## 第四章：训练前准备——工欲善其事，必先利其器

### 4.1 硬件配置

预训练大模型需要**海量计算资源**。要特别注意：**"全参数训练"所需显存远不止参数本身**，还要同时容纳梯度、优化器状态（AdamW 的一阶/二阶矩）以及前向激活值。以混合精度 + AdamW 全参训练为例，仅"参数 + 梯度 + 优化器状态"就约需 **16~20 字节/参数**（≈16GB/1B 参数起步），再加上与 batch、序列长度相关的激活值，实际峰值往往更高：

| 模型规模 | 仅参数(FP16 推理下限) | 全参训练显存(参数+梯度+AdamW 状态，约 16~20B/参数) | 推荐配置 |
|----------|----------|----------|----------|
| 7B | ~14GB | **~112~140GB+（再加激活）** | 多卡（如 4~8× A100 40/80GB）+ ZeRO/并行 |
| 13B | ~26GB | **~210~260GB+（再加激活）** | 多卡 8× A100 80GB + ZeRO-3 |
| 70B | ~140GB | **~1.1~1.4TB+（再加激活）** | 数十张 A100/H100 + 张量/流水线并行 |
| 175B (GPT-3) | ~350GB | **~2.8~3.5TB+（再加激活）** | 数百张 A100 + 3D 并行 |

**为什么这样写**：常见的"每 1B 参数 ×2GB"只是**FP16 存一份参数**的量，用于粗估**推理**显存。**全参数训练**要额外存：梯度（约 2 字节/参数）、AdamW 的一阶矩和二阶矩（fp32 各 4 字节）、以及为数值稳定保留的 fp32 主权重（4 字节），合计约 16 字节/参数起，加上激活后更多。因此单张卡几乎无法全参训练 7B 及以上模型，必须靠 ZeRO 分片、张量/流水线并行、梯度检查点、优化器 offload、或改用参数高效微调（LoRA/QLoRA）来降低显存。上表数字是数量级估算，实际取决于并行策略、序列长度、batch 与实现，请以实测为准。

### 4.2 分布式训练框架

```python
# DeepSpeed 配置示例（ZeRO-3）
ds_config = {
    "train_batch_size": 512,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 2000
        }
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO-3：参数、梯度、优化器状态都分片
        "offload_optimizer": {
            "device": "cpu"  # 将优化器状态卸载到 CPU，节省显存
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}
```

**为什么这样写**：
- **ZeRO-3**：将参数、梯度、优化器状态分片到多张 GPU，让大模型训练成为可能
- **CPU Offload**：将不常用的优化器状态放到 CPU 内存，进一步节省显存
- **混合精度训练（FP16）**：用半精度计算加速训练，同时用损失缩放防止梯度下溢

### 4.3 优化器与学习率调度

```python
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# 优化器
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,           # 最大学习率
    betas=(0.9, 0.95), # 一阶/二阶矩估计的衰减率
    eps=1e-8,
    weight_decay=0.1   # L2 正则化，防止过拟合
)

# 学习率调度：先 warmup 再余弦衰减
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,      # 前 2000 步线性升温
    num_training_steps=100000, # 总训练步数
    num_cycles=0.5              # 余弦衰减半个周期
)
```

**为什么这样写**：
- **Warmup**：训练初期用小学习率，防止模型参数剧烈震荡
- **余弦衰减**：学习率从高到低平滑下降，帮助模型收敛到更优的局部最小值
- **AdamW**：Adam + 权重衰减解耦，是大模型训练的标准选择

---

## 第五章：正式训练——预测下一个词

### 5.1 训练目标：交叉熵损失

预训练的目标函数是**最大化下一个 token 的预测概率**，等价于**最小化交叉熵损失**：

```python
import torch
import torch.nn as nn

# 假设
batch_size = 4
seq_len = 512
vocab_size = 50000
d_model = 4096

# 模拟数据
X = torch.randint(0, vocab_size, (batch_size, seq_len))      # 输入 [4, 512]
targets = torch.randint(0, vocab_size, (batch_size, seq_len)) # 目标 [4, 512]

# 模型前向传播（简化示意）
embedding = nn.Embedding(vocab_size, d_model)
# 注意：Decoder-Only 因果语言模型只有【带因果掩码的自注意力】，没有 encoder-decoder 的交叉注意力。
# nn.TransformerDecoderLayer 会引入对 memory 的 cross-attention（且默认不对 memory 做因果遮挡），
# 用它并传入 (embeds, embeds) 会让每个位置“看到完整的 memory 序列”，这不是 GPT 式因果 LM。
# 正确做法：用 nn.TransformerEncoderLayer + 因果掩码来模拟 Decoder-Only 的自回归自注意力。
block = nn.TransformerEncoderLayer(d_model, nhead=16, batch_first=True)
output_proj = nn.Linear(d_model, vocab_size, bias=False)

# 1. Embedding
embeds = embedding(X)  # [4, 512, 4096]

# 2. 自注意力（仅因果自注意力，无 cross-attention）
# 上三角为 True 的位置会被屏蔽，确保位置 i 只能看到 <= i 的 token
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
hidden = block(embeds, src_mask=causal_mask)  # [4, 512, 4096]

# 3. 输出投影到词汇表
logits = output_proj(hidden)  # [4, 512, 50000]

# 4. 计算交叉熵损失
# 将 logits 展平为 [batch*seq, vocab]，targets 展平为 [batch*seq]
loss = nn.functional.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=0  # 忽略 padding token（假设 0 是 pad_id）
)

print(f"Loss: {loss.item():.4f}")
```

**为什么这样写**：
- `ignore_index=0`：padding 位置不是真实数据，不应参与损失计算
- `view(-1, vocab_size)`：将批次和序列维度合并，满足 `cross_entropy` 的输入要求
- 每个位置的损失独立计算，最后取平均

### 5.2 训练循环完整代码

```python
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # 前向传播
        logits = model(input_ids)  # [B, L, V]
        
        # 计算损失
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # 每 100 步打印日志
        if batch_idx % 100 == 0:
            print(f"Step {batch_idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return total_loss / len(dataloader)
```

**为什么这样写**：
- **梯度裁剪（clip_grad_norm_）**：将梯度 L2 范数限制在 1.0 以内，防止梯度爆炸导致训练崩溃
- **梯度累积（本示例未实现）**：上面的循环是**每个 batch 就 `step()` 一次**的最简形式，并没有做梯度累积。若显存不足以放大 batch，可改为"每累积 N 个 batch 再 `optimizer.step()` 一次"（其间用 `loss / N` 归一化并暂不清零梯度），以等效更大的 batch size
- **定期打印日志**：监控训练是否稳定，loss 是否下降

### 5.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

# ⚠️ 关键区别：GradScaler（损失缩放）是为 float16 准备的。
#   - float16 的动态范围很窄，小梯度容易下溢为 0，所以需要 GradScaler 放大损失/梯度。
#   - bfloat16 的指数位与 float32 相同（动态范围一样大），几乎不会梯度下溢，
#     因此【用 bf16 训练时通常不需要 GradScaler】，直接 backward + step 即可。

# 情况一：使用 bfloat16（推荐用于大模型）——不需要 GradScaler
def train_step_bf16(model, input_ids, target_ids, optimizer):
    with autocast(dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0
        )
    loss.backward()                                   # 无需缩放
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

# 情况二：确需使用 float16 时——才用 GradScaler 防止梯度下溢
scaler = GradScaler()

def train_step_fp16(model, input_ids, target_ids, optimizer):
    with autocast(dtype=torch.float16):
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**为什么这样写**：
- **bfloat16**：比 float16 有更大的动态范围（指数位与 fp32 相同），训练更稳定，是大模型的推荐精度；**使用 bf16 时一般无需 GradScaler**。
- **GradScaler**：只有在用 **float16** 时才需要——它通过缩放损失把过小的梯度"抬"到可表示范围，避免下溢为 0。给 bf16 套 GradScaler 属于常见误用。

---

## 第六章：评估与验证——模型学得怎么样

### 6.1 核心评估指标：困惑度（Perplexity）

**困惑度（PPL）** 是语言模型的标准评估指标，衡量模型对测试集的"困惑"程度：

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid w_{<i})\right)$$

```python
def evaluate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
                reduction='sum'  # 不取平均，累加总损失
            )
            
            # 计算非 padding 的 token 数
            valid_tokens = (target_ids != 0).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return ppl.item()

# 解读
# PPL = exp(交叉熵)，是把模型的平均对数损失指数化之后得到的量
# 可以理解为"平均有效候选词数"或"分支因子"，但不能按字面读作：
# "每一步真的在 N 个等概率词之间选"——候选词的概率通常是高度偏斜的，
# 绝大多数词的概率接近 0，只是几何平均等价的有效数量恰好是 N。
# PPL 越低越好；强大的语言模型在常见评测语料上的 PPL 通常能降到较低的两位数甚至更低
# （具体数值高度依赖模型、语料、分词与评测口径，请以对应论文/官方报告为准）
```

**为什么这样写**：PPL = exp(H(p, q))，是模型预测分布 q 相对于真实下一词分布 p 的交叉熵 H 的指数化形式。它给出的是"等价的均匀候选词数量"：**只有当模型真的把概率均匀分配给 N 个词时，PPL 才等于 N**；而实际语言模型的概率分布通常高度集中（Top-K 词占了大部分概率质量），所以**不能按字面**把 PPL=100 解释为"每一步都在 100 个等概率词之间做选择"。PPL 的主要用途是**相对比较**：同一语料/分词/模型尺寸下，PPL 越低说明模型建模越准。

### 6.2 训练崩溃的挽救

大模型训练可能因数据异常而崩溃。挽救策略：

```python
# 检查点选择标准（教学示意）
def is_good_checkpoint(checkpoint, grad_norm_threshold=10.0):
    """
    判断一个检查点是否适合断点续训。
    ⚠️ 注意：
      1) BF16 训练通常不使用 GradScaler，loss_scale 字段可能不存在或恒为 1.0，
         因此**不能**用 "loss_scale <= 0" 作为 BF16 场景的判据；
      2) FP16 下 loss_scale 变小是 GradScaler 动态调整的正常行为（遇到溢出会
         自动下调 scale），并不等价于"检查点损坏"，把它作为恢复判据不可靠；
      3) 真正可靠的判据应优先看：验证集 loss/PPL 是否可接受、
         优化器状态键是否完整、模型 state_dict 键是否与当前结构对齐。
    """
    # 1. 验证 loss/ ppl 在合理范围（示例：非空、非无穷、非 NaN）
    loss = checkpoint.get('loss')
    if loss is None or not torch.isfinite(torch.tensor(loss)):
        return False

    # 2. 模型/优化器/调度器状态字典齐全（断点续训的最低要求）
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
        if key not in checkpoint or checkpoint[key] is None:
            return False

    # 3. 梯度范数若被记录了，可以作为"是否发生梯度爆炸"的辅助参考
    #    注意：若某步未记录 grad_norm，不能因此直接判定检查点坏。
    grad_norm = checkpoint.get('grad_norm')
    if grad_norm is not None and (grad_norm > grad_norm_threshold or not torch.isfinite(torch.tensor(grad_norm))):
        return False

    return True

# 训练崩溃后，找到最后一个好检查点，跳过崩溃数据段继续训练
```

---

## 第七章：后处理与保存

### 7.1 模型保存

```python
# 保存为 HuggingFace 格式
model.save_pretrained("./pretrained_model")
tokenizer.save_pretrained("./pretrained_model")

# 保存检查点（包含优化器状态，支持断点续训）
# ⚠️ BF16 训练通常不创建 GradScaler（变量 scaler 可能根本不存在），
#    grad_norm 也并非每步都会提前算好；保存前需要先判断/计算，
#    否则会直接 NameError。
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'loss': loss,
}

# 只有当 FP16 + 真的用了 GradScaler 时才保存 loss_scale
if 'scaler' in globals() and scaler is not None:
    checkpoint['loss_scale'] = scaler.get_scale()

# 如果 grad_norm 还没计算，这里做一次防御性取值（可选记录）
if 'grad_norm' not in locals() and 'grad_norm' not in globals():
    # 仅在需要时计算；若不需要该字段可直接跳过
    with torch.no_grad():
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), 2
        ).item() if any(p.grad is not None for p in model.parameters()) else None
if grad_norm is not None:
    checkpoint['grad_norm'] = grad_norm

torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
```

### 7.2 预训练后的模型能做什么

预训练完成后，模型具备：
- ✅ 语法正确的文本生成
- ✅ 基本的上下文理解
- ✅ 一定的世界知识
- ✅ 简单的推理能力

但还**不能**：
- ❌ 遵循指令（需要 SFT）
- ❌ 对话（需要 SFT）
- ❌ 拒绝有害请求（需要 RLHF）
- ❌ 对齐人类价值观（需要 RLHF）

---

## 本章小结与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 数据不清洗直接训练 | 严格去重、去噪、过滤低质量内容 |
| 用 float16 训练大模型 | 优先使用 bfloat16，更稳定 |
| 学习率太大导致训练崩溃 | 先 warmup 再衰减，最大学习率 5e-5 ~ 1e-4 |
| 不用梯度裁剪 | 设置 `clip_grad_norm_=1.0`，防止梯度爆炸 |
| 忽略 padding 的损失计算 | 用 `ignore_index=pad_token_id` 排除 padding |
| 不保存检查点 | 定期保存，支持断点续训 |
| 训练崩溃后从头开始 | 找到最后一个好检查点，跳过崩溃数据段 |
| 词汇表太小 | 3万~5万是常见范围，太小导致 OOV 过多 |

| 知识点 | 核心要点 |
|--------|----------|
| **预训练目标** | 自回归语言建模：预测下一个 token |
| **数据构造** | 移位构造：输入 X 和目标 targets 错位一位 |
| **损失函数** | 交叉熵损失，忽略 padding |
| **架构选择** | Decoder-Only 是当前主流（GPT、Llama、Qwen） |
| **位置编码** | RoPE 最常用，支持长文本外推 |
| **优化器** | AdamW，权重衰减 0.1 |
| **学习率调度** | Warmup + 余弦衰减 |
| **混合精度** | 优先 bfloat16（无需 GradScaler）；用 float16 时才配 GradScaler |
| **分布式训练** | DeepSpeed ZeRO-3 + CPU Offload |
| **评估指标** | 困惑度 PPL，越低越好 |

**养成的好习惯**：
1. **数据质量 > 数据数量**，投入 50% 精力在数据清洗上
2. 训练时**实时监控 loss 曲线**，异常波动立即排查
3. **定期保存检查点**，并记录 loss scale 和 grad norm
4. 用 **bfloat16** 而非 float16 训练大模型
5. 预训练完成后**立即做下游任务测试**，验证模型质量

**下一步预告**：掌握了预训练后，你将进入 **有监督微调（SFT）** 的学习——理解如何用"指令-回答"数据对让预训练模型学会遵循指令、进行对话，从"会接话"进化为"会办事"。这是大语言模型从"基座"走向"助手"的关键一步。继续加油！