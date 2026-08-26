# 学习包

## 块 1

```markdown
## 一、环境准备与模型加载

### 1.1 安装依赖

为了运行 GPT-2 模型并进行前向传播实验，我们需要安装必要的依赖库。以下是具体的安装命令：

```bash
# 创建虚拟环境（推荐）
conda create -n llm python=3.10 -y
conda activate llm

#

## 块 2

### 1.2 加载 GPT-2 模型和 Tokenizer

```python
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 加载带语言建模头的 GPT-2（用于生成任务）
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 查看模型配置，验证前面学过的理论参数
config = model.config
print(f"词汇表大小: {config.vocab_size}")
print(f"隐藏层维度: {config.n_embd}")
print(f"层数: {config.n_layer}")
print(f"注意力头数: {config.n_head}")
print(f"上下文长度: {config.n_positions}")
print(f"FFN 中间层维度: {config.n_inner or 4 * config.n_embd}")
```

> 上述配置参数会随 transformers 版本和模型版本变化，实际以运行结果为准。
>
> **知识验证**：GPT-2 是标准的 Decoder-Only 架构。12 层、768 维、12 头，与我们前面学习的 "Transformer Decoder 堆叠" 基本一致。注意：原始 GPT-2 使用 **LayerNorm**（而非 RMSNorm）和 **GELU**（而非 SwiGLU），这是历史版本与现代 LLaMA 系模型的区别。（权威①）

（个性化提示：学习风格：hands-on）

## 块 3

### 1.3 准备输入文本

```python
text = "Transformer architecture is"
inputs = tokenizer(text, return_tensors="pt")

print("input_ids:", inputs["input_ids"])
print("tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
```

> 输出结果会随 transformers 版本和词表版本变化，实际以运行结果为准。
>
> **注意**：GPT-2 使用 BPE 分词，`Ġ` 表示该 token 前有一个空格。`Transformer` 被拆成了 `Trans` + `former`，这是子词切分的典型表现，呼应了前面"子词切分与分词器"章节的内容。

## 二、执行前向传播并观察输出

（个性化提示：学习风格：hands-on）

## 块 4

### 2.1 基础前向传播

```python
import torch

with torch.no_grad():
    outputs = model(**inputs)

print("输出类型:", type(outputs))
print("logits 形状:", outputs.logits.shape)
```

> 输出结果会随 transformers 版本和模型版本变化，实际以运行结果为准。

**维度解读**：（权威① + ①）
- `[1]`：batch_size = 1
- `[4]`：输入序列长度（4 个 token）
- `[50257]`：词汇表大小，每个位置对每个词都有一个分数

`logits[0, -1, :]` 就是模型对**下一个 token** 的预测分布。我们可以用它生成下一个词：

```python
next_token_logits = outputs.logits[0, -1, :]  # 最后一个位置的预测
next_token_id = torch.argmax(next_token_logits).item()
next_token = tokenizer.decode(next_token_id)

print(f"预测的下一个 token: '{next_token}'")
```

（个性化提示：学习风格：hands-on）

## 块 5

### 2.2 观察 Hidden States（隐藏层状态）

前面我们学习了每一层 Transformer Block 的结构。现在让我们看看真实的中间结果：

```python
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states

print(f"Hidden States 层数: {len(hidden_states)}")
print(f"输入 Embedding 层输出形状: {hidden_states[0].shape}")
print(f"第 1 层 Transformer 输出形状: {hidden_states[1].shape}")
print(f"第 12 层 Transformer 输出形状: {hidden_states[12].shape}")
```

> 输出结果会随 transformers 版本和模型版本变化，实际以运行结果为准。
>
> **关键观察**：`hidden_states` 包含 13 个张量——第 0 个是 Embedding 层输出，第 1-12 个是 12 层 Transformer Decoder 的输出。每一层的输出形状通常保持 `[1, 4, 768]`，这正是**残差连接**保证的：无论经过多少层，序列长度和维度保持不变，信息可以逐层叠加而不失真。（权威①）

（个性化提示：学习风格：hands-on）

## 块 6

```markdown
# 观察 Attention Weights

让我们更深入地理解注意力机制的工作原理。在 GPT-2 模型中，注意力机制通过权重矩阵来表示不同 token 之间的关联性。通过观察这些权重，我们可以更好地理解模型的决策过程。

## 实验步骤

### 1. 获取模型输出
运行以下代码，加载模型并输入测试文本：

```python
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions
```

### 2. 可视化注意力权重
使用热力图可视化第一层的第一个注意力头的权重分布：

```python
import matplotlib.pyplot as plt

# 取第 1 层第 1 个头的注意力权重
attn = attentions[0][0, 0].cpu().numpy()  # [4, 4]

plt.figure(figsize=(6,

## 块 7

### 3.1 不使用 KV Cache：重复计算

```python
# 模拟生成 3 个新 token，每一步都重新输入完整序列
generated = inputs["input_ids"]
print("=== 不使用 KV Cache ===")

for i in range(3):
    with torch.no_grad():
        outputs = model(generated)  # 每次输入完整序列
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    print(f"Step {i+1}: 输入长度 {generated.size(1)}, 输出 token: {tokenizer.decode(next_token[0])}")
```

**观察**：每一步的输入长度都在增加（5, 6, 7...），模型每次都要重新计算整个序列的 Attention。

（个性化提示：学习风格：hands-on）

## 块 8

```markdown
# 使用 KV 缓存复用历史：理解 Transformer 架构的缓存机制

## 3.2 使用 KV 缓存：复用历史

### 3.2.1 什么是 KV 缓存？
在 Transformer 模型中，`past_key_values` 是一个重要的缓存机制，用于存储注意力机制中生成的键（Key）和值（Value）。通过使用缓存，模型可以避免在每一步都重新计算这些值，从而提高生成速度。

### 3.2.2 使用 KV 缓存的步骤

#### 步骤 1：初始化缓存
在模型的首次调用中，我们需要初始化 `past_key_values`。这可以通过调用

## 块 9

### 3.3 KV Cache 的内存占用估算

基于 GPT-2 的配置，我们可以手动计算 KV Cache 的最大内存占用：

```python
# GPT-2 配置
batch_size = 1
seq_len = 1024  # 最大上下文
n_layers = 12
n_heads = 12
head_dim = 64  # 768 / 12
bytes_per_param = 4  # FP32，默认加载

# KV Cache 总内存（K + V）
kv_cache_memory = batch_size * seq_len

## 块 10

```markdown
## 三、验证 KV Cache 机制

为了验证 KV Cache 机制的内存占用估算，我们需要深入理解其内部工作原理。通过实验，我们可以观察到 KV Cache 在生成过程中如何管理中间结果，从而优化模型的性能。

### ① 自动管理 KV Cache

在模型生成过程中，`generate()` 方法会自动管理 KV Cache。具体来说，它会缓存每一步生成的键值对（key-value），以便在后续步骤中快速访问。这种机制可以帮助减少重复计算，提高生成速度。

### ② 支持多种解码策略

`generate()` 方法支持多种解码策略，包括贪心搜索、Beam Search、Top-K/Top-P 采样等。这些策略决定了生成文本的多样性和质量。通过选择不同的策略，我们可以验证 KV Cache 在不同场景下的表现。

### ③ 处理特殊 token

在实际生成过程中，模型会自动处理特殊 token（如 EOS 和 PAD）。这些处理操作可以帮助我们避免生成过程中的异常情况，确保生成文本的质量。

### ④ 长度控制

`generate()` 方法支持控制生成文本的长度，例如设置 `max_new_tokens`

## 块 11

- ✅ **模型结构真实存在**：GPT-2 的 12 层 Decoder、768 维隐藏层、12 头 Attention 与理论基本一致
- ✅ **残差连接保持维度**：每一层的 hidden states 形状不变，信息逐层叠加
- ✅ **Causal Mask 生效**：Attention 权重热力图的上三角通常被正确屏蔽
- ✅ **KV Cache 工作正常**：Prefill 阶段输入完整 prompt 获得初始缓存，Decoding 阶段每步只输入 1 个新 token，K/V 缓存形状与公式推导基本一致
- ✅ **能够独立加载真实模型并执行前向传播**
- ✅ **理解 GPT-2 与现代模型在组件选型上的差异**：LayerNorm vs RMSNorm、GELU vs SwiGLU、可学习位置编码 vs RoPE

> **下一步预告**：在下一板块 **"大语言模型（LLM）核心特性"** 中，我们将跳出架构细节，从更高维度理解现代 LLM 的能力——什么是**涌现能力**？为什么大模型能进行**上下文学习（In-Context Learning）**？**指令遵循**和**逐步推理**是如何实现的？以及我们必须正视的**幻觉问题**。现在，试着修改上面的代码，将 GPT-2 换成 `gpt2-medium`（3.45 亿参数），观察层数、维度是否变化，并估算其最大 KV Cache 内存占用！

（个性化提示：学习风格：hands-on）

## 测评题目

### Q-KB08-ST0807-001
模型前向传播得到的 logits 通常在经过________后可转换为词表上的概率分布。

### Q-KB08-ST0807-003
前向实验中 `input_ids` 通常表示？

### Q-KB08-ST0804-001
Transformer 中的 FFN 通常对序列中的每个 token 位置________地应用同一组参数。

### Q-KB08-ST0807-002
推理时将模型切换到评估模式，应调用 `model.________()`。

### Q-KB08-ST0802-002
GPT、Llama、Qwen 等自回归语言模型通常使用________架构。

### Q-KB08-ST0800-003
加载 GPT-2 等预训练模型进行前向实验，最直接的目的是什么？

### Q-KB0801-004
在Transformer解码器（Decoder）中，为了防止模型在生成时“偷看”到未来的词，通常会使用因果掩码（Causal Mask）。（ ）

### Q-KB08-ST0800-001
用 `AutoModel.from_pretrained()` 加载模型时，模型权重通常来自名称对应的预训练________。
