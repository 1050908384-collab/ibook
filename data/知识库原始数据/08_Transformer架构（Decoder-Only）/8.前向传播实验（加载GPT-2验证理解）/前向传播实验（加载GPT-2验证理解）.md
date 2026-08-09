# 前向传播实验（加载 GPT-2 验证理解）

权威文章：《GPT2 — transformers documentation》  
　　网址：https://huggingface.co/docs/transformers/model_doc/gpt2

权威文章：《Using Transformers — Hugging Face NLP Course》  
　　网址：https://huggingface.co/learn/nlp-course/chapter2/2

① 底稿：《Hugging Face Transformers快速入门（加载与推理）-原理源码解析》  
　　网址：https://blog.csdn.net/l35633/article/details/161060456

② 参考文章：《手写Mini-GPT：从零实现因果语言模型的前向与反向传播》  
　　网址：https://blog.csdn.net/weixin_29775479/article/details/161847337

> **适合人群**：已完成 Decoder-Only 架构全部理论章节（RMSNorm、FFN/SwiGLU、RoPE、KV Cache），希望用真实模型验证理解的初学者。本节时长约 0.5 小时，目标是能够使用 Hugging Face Transformers 加载 GPT-2，执行前向传播，观察并验证模型内部结构与前面所学知识的对应关系，同时理解 GPT-2 与现代模型（如 `meta-llama/Llama-2-7b`、`Qwen/Qwen2.5-0.5B-Instruct`）在组件选型上的差异。（①）
>
> **前置说明**：前面我们拆解了 Decoder-Only 模型的每一个组件——RMSNorm、残差连接、SwiGLU、RoPE、KV Cache。本节我们将加载一个真实的 **GPT-2** 模型（约 1.37 亿参数），执行前向传播，验证 Decoder-Only 架构、因果注意力、残差连接、FFN 和 KV Cache 等核心概念。同时，我们也会对比 GPT-2 与现代 `meta-llama/Llama-2-7b`、`Qwen/Qwen2.5-0.5B-Instruct` 等模型在归一化、激活函数和位置编码上的差异——GPT-2 使用 **LayerNorm**（而非 RMSNorm）、**GELU**（而非 SwiGLU）和**可学习绝对位置编码**（而非 RoPE）。所有代码请在已安装 `torch` 和 `transformers` 的 Python 环境中运行。

## 一、环境准备与模型加载

### 1.1 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n llm python=3.10 -y
conda activate llm

# 安装核心依赖
pip install torch transformers
```

> **本地运行配置要求**：以下代码在 CPU 上即可运行（GPT-2 仅 137M 参数），内存建议 ≥ 4GB。如有 GPU，推理速度通常更快，但不是必需。

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

### 2.3 观察 Attention Weights（注意力权重）

让我们看看 Attention 到底在"关注"什么：

```python
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions

print(f"Attention 层数: {len(attentions)}")
print(f"第 1 层 Attention 权重形状: {attentions[0].shape}")
```

> 输出结果会随 transformers 版本和模型版本变化，实际以运行结果为准。

**维度解读**：（权威①）
- `[1]`：batch_size
- `[12]`：12 个注意力头
- `[4, 4]`：输入序列长度 4，每个 token 对其他 4 个 token 的注意力权重

由于 GPT-2 是 Decoder-Only，使用了 **Causal Mask**，每个 token 只能看到自己和之前的 token。我们可以通过热力图验证这一点：

```python
import matplotlib.pyplot as plt

# 取第 1 层第 1 个头的注意力权重
attn = attentions[0][0, 0].cpu().numpy()  # [4, 4]

plt.figure(figsize=(6, 5))
plt.imshow(attn, cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xticks(range(4), tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), rotation=45)
plt.yticks(range(4), tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
plt.title("Layer 1, Head 0: Attention Weights")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.tight_layout()
plt.show()
```

> **观察要点**：热力图的**上三角通常接近 0**（被 causal mask 屏蔽），下三角有非零值。这验证了我们前面学习的"Decoder-Only 模型使用下三角掩码防止看到未来信息"。（权威①）

## 三、验证 KV Cache 机制

前面我们学习了 KV Cache 的理论，现在用 GPT-2 实际验证。

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

### 3.2 使用 KV Cache：复用历史

```python
# 使用 past_key_values 缓存
generated = inputs["input_ids"]
past_kv = None  # 初始无缓存
print("=== 使用 KV Cache ===")

# 第 1 步：输入完整 prompt，获得初始 past_key_values
with torch.no_grad():
    outputs = model(generated, use_cache=True)
past_kv = outputs.past_key_values
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
generated = torch.cat([generated, next_token], dim=1)
print(f"Prefill: 输入长度 {inputs['input_ids'].size(1)}, "
      f"KV Cache 层数 {len(past_kv)}, K 形状 {past_kv[0][0].shape}, "
      f"输出 token: {tokenizer.decode(next_token[0])}")

# 第 2 步起：每次只输入最新生成的 1 个 token，并传入上一步的缓存
for i in range(2):
    with torch.no_grad():
        # 只输入最新生成的 1 个 token，同时传入 past_key_values
        outputs = model(generated[:, -1:], past_key_values=past_kv, use_cache=True)
    
    past_kv = outputs.past_key_values  # 更新缓存
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    
    # 观察 KV Cache 结构
    print(f"Step {i+1}: 输入长度 1, KV Cache 层数 {len(past_kv)}, "
          f"K 形状 {past_kv[0][0].shape}, 输出 token: {tokenizer.decode(next_token[0])}")
```

> **关键验证**：
> 1. **Prefill 阶段输入完整 prompt**：第一次调用 `model(generated, use_cache=True)` 输入完整序列，获得完整的初始 `past_key_values`
> 2. **Decoding 阶段输入长度始终为 1**：后续每步只输入最新 token，而非完整序列
> 3. **KV Cache 有 12 层**：与模型层数一致，每层缓存独立的 K 和 V
> 4. **K 形状 `[1, 12, 5, 64]`**：`1` 是 batch，`12` 是头数，`5` 是累计序列长度（4 个输入 + 1 个新生成），`64` 是 head_dim（768/12）。这与我们前面推导的 KV Cache 公式基本一致！（①）

### 3.3 KV Cache 的内存占用估算

基于 GPT-2 的配置，我们可以手动验证内存公式：

```python
# GPT-2 配置
batch_size = 1
seq_len = 1024  # 最大上下文
n_layers = 12
n_heads = 12
head_dim = 64  # 768 / 12
bytes_per_param = 4  # FP32，默认加载

# KV Cache 总内存（K + V）
kv_cache_memory = batch_size * seq_len * n_layers * n_heads * head_dim * 2 * bytes_per_param
print(f"GPT-2 KV Cache 最大内存: {kv_cache_memory / 1024 / 1024:.2f} MB")
```

> 输出结果会随运行环境变化，实际以运行结果为准。
>
> **对比**：GPT-2 模型参数约 137M × 4 bytes ≈ 548 MB。KV Cache 在最大上下文时约占 72 MB，远小于模型参数。但对于现代 7B 模型，KV Cache 可达数十 GB，这正是我们上一节学习的瓶颈所在。（①）

## 四、模型结构拆解：与理论知识的对应

让我们深入观察 GPT-2 的内部结构，找到前面学过的每一个组件：

```python
# 查看模型的 Transformer Block 结构
print(model.transformer.h[0])  # 第 1 层 Decoder Block
```

> 输出结果会随 transformers 版本和模型版本变化，实际以运行结果为准。

**与理论知识的对应**：（权威① + ②）

| 组件 | GPT-2 中的名称 | 前面学过的对应知识 | 现代模型（如 Llama/Qwen）的差异 |
|------|---------------|------------------|------------------------------|
| `ln_1` | LayerNorm | 预归一化（Pre-LN）：Attention 之前做归一化 | 现代模型通常使用 **RMSNorm** |
| `attn` | GPT2Attention | Multi-Head Self-Attention + Causal Mask | 现代模型通常使用 **RoPE** 位置编码 |
| `ln_2` | LayerNorm | FFN 之前的归一化 | 现代模型通常使用 **RMSNorm** |
| `mlp` | GPT2MLP | FFN：Conv1D 实现 `xW + b`，激活函数为 GELU | 现代模型通常使用 **SwiGLU** |
| 残差连接 | 未显式显示 | 在 `forward` 中实现：`x + Attention(LN(x))` | 结构基本一致 |

> **注意**：GPT-2 使用的是 **LayerNorm** 而非 RMSNorm，**GELU** 而非 SwiGLU，**可学习位置编码** 而非 RoPE。这些都是 GPT-2（2019）与 LLaMA（2023）时代的差异。现代 Decoder-Only 模型在结构上基本遵循相同的范式，但在归一化、激活函数和位置编码上做了优化升级。（权威①）

## 五、手动生成 vs model.generate()

前面我们手动写了逐 token 生成循环。实际上，Transformers 库提供了 `model.generate()` 方法，它内部做了大量优化：

```python
# 使用 model.generate() 生成
inputs = tokenizer("Transformer architecture is", return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,  # 贪心搜索
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"生成结果: {generated_text}")
```

**`generate()` 内部做了什么？**（①）

1. **自动管理 KV Cache**：无需手动传递 `past_key_values`
2. **支持多种解码策略**：贪心搜索、Beam Search、Top-K/Top-P 采样
3. **处理特殊 token**：自动添加/移除 EOS、PAD
4. **长度控制**：`max_new_tokens`、`min_new_tokens` 等

我们前面手动实现的循环，本质上就是 `generate()` 的简化版。理解手动实现，有助于你在 `generate()` 行为异常时进行调试。

## 六、常见问题与解决（①+②）

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `from_pretrained` 下载慢或失败 | 网络连接 Hugging Face 不稳定 | 设置镜像源：`export HF_ENDPOINT=https://hf-mirror.com`（①） |
| `RuntimeError: expected scalar type Float but found Half` | 模型权重与输入数据类型不匹配 | 确保 `inputs` 和 `model` 在同一设备且同一 dtype（①） |
| `generate()` 输出重复内容 | 未设置 `pad_token_id` 或重复惩罚不足 | 设置 `pad_token_id=tokenizer.eos_token_id`，调整 `repetition_penalty`（①） |
| `past_key_values` 形状不匹配 | 手动传入缓存时输入长度不为 1 | Decoding 阶段每次只应输入 1 个新 token（①） |
| 观察到的 Attention 上三角不为零 | 未使用 Causal Mask 或可视化错误 | 确认使用的是 Decoder 模型；检查可视化代码是否取对了维度（权威①） |
| 显存溢出（OOM） | 输入序列过长或 batch 过大 | GPT-2 最大上下文 1024，超长文本需截断；减小 batch_size（①） |

## 七、总结与后续学习路线（权威①+②）

恭喜你完成了 Transformer 架构（Decoder-Only）板块的全部内容！通过本次实验，你应当已经验证了：

- ✅ **模型结构真实存在**：GPT-2 的 12 层 Decoder、768 维隐藏层、12 头 Attention 与理论基本一致
- ✅ **残差连接保持维度**：每一层的 hidden states 形状不变，信息逐层叠加
- ✅ **Causal Mask 生效**：Attention 权重热力图的上三角通常被正确屏蔽
- ✅ **KV Cache 工作正常**：Prefill 阶段输入完整 prompt 获得初始缓存，Decoding 阶段每步只输入 1 个新 token，K/V 缓存形状与公式推导基本一致
- ✅ **能够独立加载真实模型并执行前向传播**
- ✅ **理解 GPT-2 与现代模型在组件选型上的差异**：LayerNorm vs RMSNorm、GELU vs SwiGLU、可学习位置编码 vs RoPE

> **下一步预告**：在下一板块 **"大语言模型（LLM）核心特性"** 中，我们将跳出架构细节，从更高维度理解现代 LLM 的能力——什么是**涌现能力**？为什么大模型能进行**上下文学习（In-Context Learning）**？**指令遵循**和**逐步推理**是如何实现的？以及我们必须正视的**幻觉问题**。现在，试着修改上面的代码，将 GPT-2 换成 `gpt2-medium`（3.45 亿参数），观察层数、维度是否变化，并估算其最大 KV Cache 内存占用！