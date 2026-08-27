# 学习包

## 块 1

**本块学习检查**（先做这个，再继续）

在动手安装前，先用 3 分钟快速回顾：用 `AutoTokenizer` 和 `from_pretrained()` 加载一个分词器，并解释这两行代码各自的作用。如果你能说清“加载分词器”和“加载模型”的区别，再进入下面的安装步骤。若卡壳，回到第七板块复习后再来。

---

## 一、安装与验证 transformers

### 1.1 快速安装

如果你已完成第七板块的学习，`transformers` 库应该已经安装。如果尚未安装，在已有的 `pytorch` conda 环境中执行：

```bash
conda activate pytorch
pip install transformers
```

> 若需要同时安装 PyTorch 后端依赖，可使用 `pip install 'transformers[torch]'`。

**运行环境说明**：`transformers` 库本身体积通常在数 MB 至十余 MB 之间，安装过程需要联网。CPU 即可执行，内存建议 ≥ 4GB。

---

**安装后验证**（不引入新事实，仅用原文信息）：在 Python 中执行 `import transformers` 并打印版本号。若成功，说明环境就绪

## 块 2

### 1.2 验证安装

装好后，直接跑一段代码确认环境没问题。用 `pipeline` 是最快的验证方式，它会自动加载一个小型情感分析模型：

```python
from transformers import pipeline
result = pipeline('sentiment-analysis')('we love you')
print(result)
```

预期会输出类似 `[{'label': 'POSITIVE', 'score': 0.99...}]` 的结果，具体数值取决于你安装的 transformers 和模型版本。看到这个输出，就说明库装好了、能联网拉模型、前向传播链路通了。

**动手试一下**：把 `'we love you'` 换成 `'I hate this'`，观察输出标签和分数怎么变。这能帮你直观感受模型的行为。

### 1.3 常见问题

如果验证时报错，对照下表排查：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `Failed building wheel for tokenizers` | 缺少 Rust 编译工具链 | 确保已安装 Rust 编译工具链（rustc/cargo），或使用预编译的 tokenizers 版本（如 `pip install tokenizers --no-binary :all:` 前先安装 Rust） |
| `urllib3.exceptions.SSLError` | `urllib3` 版本或系统证书/代理问题 | 升级 urllib3 到最新版本（`pip install --upgrade urllib3`），或检查系统证书、代理设置 |
| 无法连接 HuggingFace Hub | 网络限制 | 参考第四节"离线配置"或设置镜像源 |

---

## 二、加载 GPT-2 模型

GPT-2 是 OpenAI 的因果语言模型，有多个尺寸。本实验用最小的 `openai-community/gpt2`（124M 参数），CPU + 4GB 内存就能跑，适合快速上手

## 块 3

### 2.1 使用 pipeline 快速体验

你的目标是 vLLM 部署，但先别急着上重型工具——用 `pipeline` 跑通 GPT-2，能帮你快速验证 HuggingFace 生态的基本流程，这是后续理解 vLLM 推理入口的基石。

`pipeline` 是 transformers 提供的高级 API，核心价值在于**一行代码封装了加载、预处理、推理、后处理全流程**。对于你这种 hands-on 风格，直接看代码：

```python
from transformers import pipeline

# 创建文本生成 pipeline，device=-1 表示使用 CPU
generator = pipeline("text-generation", model="openai-community/gpt2", device=-1)

result = generator("Hello, I'm a language model", max_new_tokens=20)
print(result)
```

**运行前须知**：
- 首次运行会自动从 HuggingFace Hub 下载 GPT-2 模型文件（约 500MB，fp32 精度）

## 块 4

## 2.2 手动加载 Tokenizer 和 Model

既然你的目标是 vLLM 部署，理解底层加载机制是必要的地基。这里我们不走 `pipeline` 捷径，而是用底层 API 手动加载，方便你后续观察模型内部细节。

**第一步：加载分词器和模型**

```python
from transformers import AutoTokenizer, GPT2LMHeadModel

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# 加载模型
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
```

**第二步：处理 pad_token 坑**

GPT-2 默认没有 `pad_token`，不设置的话 batch 处理会报错。手动把它设为 `eos_token`：

```python
tokenizer.pad_token = tokenizer.eos_token
```

**第三步：打印配置，验证加载成功**

```python
print(f"词表大小: {model.config.vocab_size}")
print(f"最大序列长度: {model.config.n_positions}")
print(f"隐藏层维度: {model.config.n_embd}")
print(f"层数: {model.config.n_layer}")
print(f"注意力头数: {model.config.n_head}")
```

**这些参数意味着什么？** 它们对应 Transformer 架构的核心超参数：`n_embd` 是隐藏层维度，`n_layer` 是 Transformer Block 堆叠层数，`n_head` 是多头注意力头数。后续课程会逐一拆解这些组件的内部机制。

**本块学习检查**：

## 块 5

## 3.1 完整前向传播流程——动手跑通 GPT-2

你的目标是 vLLM 部署，但部署前必须亲手摸清模型推理时内部发生了什么。下面这段代码就是“解剖”GPT-2 的起点，跟着敲一遍，重点看输出形状。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器（模型许可证：MIT License，商用无限制）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 注意：GPT-2 模型 FP16 推理需要至少 4GB 显存
# 准备输入
text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")

print(f"input_ids 形状: {inputs['input_ids'].shape}")
```

先看 `input_ids` 的形状——它告诉你模型把这句话切成了几个 token。记住这个数字，后面所有输出都和它挂钩。

```python
# 执行前向传播
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,   # 返回每层的隐藏状态
        output_attentions=True       # 返回每层的注意力权重
    )
```

`torch.no_grad()` 表示只推理、不计算梯度，部署时也是这个模式。两个 `output_*` 参数是观察内部的关键开关。

## 块 6

## 3.2 观察下一词预测

GPT-2 的核心能力是预测下一个词。我们通过取模型输出中**最后一个位置**的 logits，就能看到它对下一个词的预测：

```python
# 取最后一个位置的 logits，预测下一个词
next_token_logits = outputs.logits[0, -1, :]
next_token_id = torch.argmax(next_token_logits).item()
next_token = tokenizer.decode(next_token_id)

print(f"模型预测的下一个词: '{next_token}'")
```

`torch.argmax` 选出概率最高的 token ID，再用 `tokenizer.decode` 转回文本。这就是 GPT-2 生成文本的基本单元——每次只预测一个词，然后把预测结果拼回输入，继续预测下一个。

## 3.3 观察因果注意力掩码

GPT-2 是 **Decoder-Only** 模型，使用**因果（causal）注意力**——每个 token 只能看到自己和之前的 token，不能看到未来的 token。这是它与 BERT 等 Encoder 模型的关键区别。

我们可以通过注意力权重矩阵来验证这一点：

```python
# 取第一层的注意力权重
# 形状: [batch=1, n_head=12, seq_len, seq_len]
attn = outputs.attentions[0]

# 查看第一个注意力头的权重矩阵
attn_matrix = attn[0, 0].numpy()

# 打印上三角是否接近 0（因果掩码的效果）
import numpy as np
upper_triangular = np.triu(attn_matrix, k=1)
print(f"上三角区域最大权重: {upper_triangular.max():.6f}")
```

**验证要点**：如果因果掩码实现正确，上三角区域（未来token对应的位置）的注意力权重应接近0。

## 块 7

**动手做：用 labels 算损失（GPT-2 训练视角）**

既然你要部署 vLLM，理解“损失”怎么算，能帮你更清楚模型在训练和推理时的区别。这里我们只看一个关键点：**训练时，`labels` 就是 `input_ids` 本身**。

为什么？因为语言模型的任务是预测下一个词。你给它 `input_ids`，它内部会自动把标签**左移一位**：预测第 `i` 个位置的词时，目标其实是第 `i+1` 个词。这个偏移，`GPT2LMHeadModel` 已经帮你处理好了，**你不需要手动做任何偏移**。

直接看代码，一行搞定：

```python
import torch

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])

print(f"Loss: {outputs.l

## 块 8

你的目标是快速上手 vLLM 部署，而 HuggingFace 模型缓存是绕不开的第一步。这里直接给你最实用的路径：

**模型默认缓存位置**（下载后自动存放）：
- Linux/macOS：`~/.cache/huggingface/hub`
- Windows：`C:\Users\你的用户名\.cache\huggingface\hub`

**为什么要改缓存目录？** 因为默认路径常在系统盘，空间有限，且重装系统会丢失。部署 vLLM 时模型动辄几个 GB，建议一开始就指到你的大容量磁盘。

**修改方法**（二选一，推荐第一种）：
```bash
# 方法 1：设置 HF_HOME（推荐，统一管理所有 HF 相关

## 块 9

## 4.2 离线运行：内网环境下的模型加载

你的目标是 vLLM 部署，但部署前的模型准备和验证往往发生在受限网络环境中。这里直接给你一套可操作的离线方案。

**第一步：设置离线环境变量**

在防火墙或内网环境中，先执行以下命令，告诉 transformers 和 datasets 库“不要联网”：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**第二步：从本地路径加载模型**

设置好环境变量后，直接指向本地缓存目录加载模型。建议使用 try-except 捕获加载过程中的异常（如文件缺失、网络误触等），便于快速定位问题：

```python
try:
    tokenizer = AutoTokenizer.from_pretrained("/local/path/to/gpt2")
    model = GPT2LMHeadModel.from_pretrained("/local/path/to/gpt2")
except OSError as e:
    print(f"模型加载失败，请检查本地路径及文件完整性: {e}")
    raise
```

**关键操作：预先下载模型**

离线前，先在有网环境中用 `from_pretrained()` 下载模型，再用 `save_pretrained("./local/path")` 保存到指定目录。这样离线时就能直接加载本地文件。

> **模型许可证说明**：本教程以 GPT-2 为例，其采用 MIT 许可证，可自由商用。若你后续替换为其他模型（如 Llama 系列），请务必确认对应许可证条款（例如 Llama 2 Community License 对商用月活超 7 亿用户有额外授权要求），并在部署前完成合规审查。

---

## 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `OSError: Can't load tokenizer` | 本地路径缺少必要的配置文件 | 确保本地目录包含完整的分词器文件 |
| `RuntimeError: Expected all tensors on same device` | 模型和数据不在同一设备 | 确保 `model.to(device)` 和数据 `.to(device)` 一致 |
| GPT-2 padding 报错 | 默认没有 `pad_token` | 手动设置 `tokenizer.pad_token = tokenizer.eos_token` |
| 下载速度极慢 | 网络问题 | 使用 `HF_HOME` 指定缓存目录，或配置镜像源 |
| 显存不足（OOM） | 加载了过大的模型 | 换用 `openai-community/gpt2`，或尝试启用 `load_in_8bit=True` |

---

## 总结与下一步

你现在应该掌握了：

- ✅ transformers 库的安装验证方法
- ✅ 使用 `pipeline` 和 `AutoModel` 两种方式加载 GPT-2
- ✅

## 块 10

## 三、Encoder 的层次结构（①）：Multi-Head Self-Attention

### 3.1 核心思想：为什么需要“多头”？

上一节你观察 GPT-2 的因果掩码时，其实已经接触过 Self-Attention 的基本计算：每个 token 都会“关注”序列中的其他 token，并据此更新自己的表示。

但**单个注意力头**有一个局限：它只能捕捉一种“关注模式”。比如，一个头可能特别关注相邻词，另一个头可能关注句法上的主语-谓语关系。如果只用单头，模型被迫在多种模式中“折中”，表达能力受限。

**Multi-Head Self-Attention（多头自注意力）** 的解决办法很直接：**并行运行多个独立的注意力头**，每个头学习不同的关注模式，最后把结果拼接起来再做一次线性变换。

### 3.2 数据流拆解（以 Encoder 层为例）

假设输入序列有 $n$ 个 token，每个 token 的向量维度是 $d_{model}$。一个 Encoder 层内的 Multi-Head Self-Attention 按以下步骤流动：

1. **线性投影**：对输入 $\mathbf{X}$（形状 $n \times d_{model}$），分别用三组可学习的权重矩阵 $W^Q, W^K, W^V$ 计算出 **Query、Key、Value** 矩阵。每组权重形状为 $d_{model} \times d_k$（其中 $d_k = d_{model} / h$，$h$ 是头数）。
2. **拆分多头**：把 Q、K、V 沿特征维度切成 $h$ 份，每份维度 $d_k$。这样每个头独立处理一份子空间。
3. **并行注意力**：对每个头 $i$，计算：
   $$\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$
   注意：**Encoder 的 Self-Attention 没有掩码**，因为 Encoder 是双向的，每个 token 可以看到完整输入序列的所有 token。
4. **拼接与输出投影**：把 $h$

## 块 11

## 3.1 Multi-Head Self-Attention：让每个词“看到”整个句子

在 Encoder 的每一层里，第一个子层就是 **Multi-Head Self-Attention**。它的核心任务是：**让输入序列中的每个位置，都能聚合整个序列的上下文信息**。

### 三个关键点

**1. Q/K/V 都来自前一层**

这里的 Q（Query）、K（Key）、V（Value）三个矩阵，**全部来自 Encoder 前一层的输出**。也就是说，每个位置既“提问”，也“被检索”，还“提供信息”——这就是“Self”（自注意力）的含义：序列自己和自己做注意力计算。

**2. 无因果 Mask（双向可见）**

Encoder 的 Self-Attention **没有因果 mask**。这意味着序列中所有位置互相可见：第 1 个词可以看到第 100 个词，第 100 个词也可以看到第 1 个词。这是 Encoder 与 Decoder 的关键区别之一（Decoder 的 Self-Attention 带因果 mask，只能看左侧）。

> 补充：在批处理变长序列时，通常仍会有 padding mask 来处理长度不一致的问题，但这与“因果 mask”是两回事。

**3. 作用：聚合全局上下文**

通过 Self-Attention，每个位置的输出向量都“融合”了全序列的信息。比如在“苹果很好吃，我咬了一口”中，“苹果”这个词的表示会包含“咬”这个动作的信息——这就是上下文聚合。

### 结合你的 vLLM 部署目标

你学习 Transformer 是为了 vLLM 部署。这里有一个直接关联：**vLLM 的 PagedAttention 优化，本质上就是在高效管理 Self-Attention 计算中的 KV Cache**。理解“Q/K/V 都来自前一层”这个数据流，是理解 KV Cache 为什么存在、为什么需要被缓存的前提——因为 K 和 V 在生成时会被反复

## 块 12

### 3.2 位置级前馈网络（FFN）

你的目标是部署 vLLM，所以这里直接看关键点：FFN 是每个位置独立做的两次线性变换，中间夹一个 ReLU 激活，公式为：

**FFN(x) = max(0, xW₁ + b₁)W₂ + b₂**

维度先放大再缩回（通常为 d_model→4×d_model→d_model）。放大到4倍是为了让模型有更多参数去捕捉特征，缩回是为了和残差连接对齐。

### 3.3 残差连接 + 层归一化

每个子层输出都做：**LayerNorm(x + Sublayer(x))**。残差让梯度能直接流过，归一化稳定训练。

**Encoder 完整数据流**：
1. x₁ = Embedding + 位置编码
2. x₂ = LayerNorm(x₁ + 多头注意力(x₁))
3. x₃ = LayerNorm(x₂ + FFN(x₂))
4

## 块 13

### 4.1 Masked Multi-Head Self-Attention

你的目标是 vLLM 部署，核心是搞懂 Decoder 如何“一步步”生成。这里的关键就是**因果掩码**。

**Q/K/V 来源**：全部来自 Decoder 前一层的输出，和 Encoder 无关。

**Mask 作用**：**因果掩码**，防止当前位置看到“未来”的词。这保证了自回归——生成第 3 个词时，只能看第 1、2 个词。

**Mask 机制**：计算 attention score 时，把未来位置的 score 设为 $-\infty$：

$$\text{score}_{ij} = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d_k}} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Softmax 后，$-\infty$ 处权重变为 0

## 块 14

**4.2 Multi-Head Cross-Attention：让Decoder“看见”输入**

Cross-Attention是Encoder和Decoder之间的桥梁。它的核心是**Q来源**和**K/V来源**不同：

- **Q**：来自Decoder的**前一层**输出。
- **K和V**：来自**Encoder的输出**（记为$\mathbf{z}$）。

**关键点1：无因果Mask**
Cross-Attention**没有因果mask**，意味着Decoder可以同时看到完整的输入序列（批处理时通常仍有padding mask，用于忽略填充部分）。

**关键点2：每层都连接**
Encoder输出$\mathbf{z}$会被送到**每一层**Decoder的Cross-Attention（共6次），**不是**只送给第一层。

## 块 15

## Decoder 的 Feed-Forward Network：让每个位置“独立思考”

你已经掌握了 Decoder 中的两种注意力机制——Masked Self-Attention 负责“看自己已经生成的词”，Cross-Attention 负责“看 Encoder 编码好的输入”。现在来到 Decoder 每一层的最后一步：**Feed-Forward Network（FFN）**。

好消息是：**Decoder 的 FFN 与 Encoder 的 FFN 结构相同，但参数不共享**，不需要额外学习新概念。它的作用是对每个位置的信息做一次独立的非线性变换，让模型有更强的表达能力。

### 完整数据流中的位置

把 FFN 放到 Decoder 的完整流程里看，它处于每层的最后一步（采用 Pre-LN 结构，每个子层后均有残差连接，实际实现中通常还包含 Dropout）：

1. **y₁** = Embedding(输出) + 位置编码
2. **y₂** = y₁ + Masked Self-Attention(LayerNorm(y₁), 带因果mask)
3. **y₃** = y₂ + Cross-Attention(LayerNorm(y₂), K=z, V=z)  ← z 是 Encoder 输出
4. **y₄** = y₃ + **FFN(LayerNorm(y₃))**  ← 就是这里

以上步骤**重复 6 次**（即 Decoder 有 6 层），最后经过 Linear 层得到 logits，再经过 Softmax 得到概率分布。

### 三种 Attention 的对比：核心区别只在 Q/K/V 来源

三种 Attention 的公式完全一样：`Attention(Q, K, V) = softmax(QKᵀ/√d_k)V`。区别只在于**谁提供 Q、K、V**：

| 类型 | 位置 | Q来源 | K来源 | V来源 | Mask | 作用 |
|---|---|---|---|---|---|---|
| Encoder Self-Attention | Encoder每层 | 前一子层输出 | 前一子层输出 | 前一子层输出 | 无因果mask | 编码输入序列上下文 |
| Decoder Masked Self-Attention | Decoder每层 | 前一子层输出 | 前一子层输出 | 前一子层输出 | **因果mask** | 自回归生成，防止“看到未来” |
| Decoder Cross-Attention | Decoder每层 | Decoder前一子层输出 | **Encoder输出** | **Encoder输出** |

## 块 16

**动手看 Decoder 的 Feed-Forward Network**

你的目标是 vLLM 部署，核心是搞懂模型结构。这里我们用 T5-small 直接“解剖”它。

# 模型许可证：T5 采用 Apache 2.0 许可证，可商用。

**运行这段代码，观察输出：**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

print(f"Encoder 层数: {model.config.num_layers}")
print(f"Decoder 层数: {model.config.num_decoder_layers}")
print(model)
```

# 注意：T5-small 模型 FP16 精度下约需 1GB 显存。

**看什么：**

1. 打印出的 `model` 结构里，找到 `decoder` 部分，展开 `block`。
2. 每个 Decoder 层里，你会看到 `T5LayerFF` —— 这就是 **Feed-Forward Network（前馈网络）**。它位于自注意力之后，对每个位置的表示做独立变换。
3.

## 块 17

## 6.1 观察 Encoder-Decoder 的前向传播

既然你的目标是 vLLM 部署，理解前向传播的形状变化是后续部署时排查问题的基本功。我们直接动手跑代码。

**操作步骤**：复制以下代码到你的 Python 环境（纯 CPU 即可，内存 ≥ 4GB），逐行运行并观察输出。

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载模型和分词器（以 T5-small 为例）
# 注意：若替换为更大模型（如 Llama-2-7b），FP16 需至少 14GB 显存
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

# 输入文本
input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors="pt")

# 构造 Decoder 输入（翻译任务中，Decoder 的输入是目标语言的前缀）
decoder_input_ids = tokenizer("Das Haus", return_tensors="pt").input_ids

# 前向传播
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        decoder_input_ids=decoder_input_ids,
        output_attentions=True,
        output_hidden_states=True
    )

# 观察各输出形状
print(f"Encoder 最后一层隐藏状态形状: {outputs.encoder_last_hidden_state.shape}")
print(f"Decoder 最后一层隐藏状态形状: {outputs.decoder_hidden_states[-1].shape}")
print(f"Cross-Attention 层数: {len(outputs.cross_attentions)}")
print(f"每层 Cross-Attention 形状: {outputs.cross_attentions[0].shape}")
```

**关键观察**：`cross_attentions[0].shape` 的形状通常为 `[batch_size, num_heads, decoder_len, encoder_len]`，其中最后两个维度说明 Decoder 的每个位置都在关注 Encoder 的所有位置。这正是 Cross-Attention 的作用。注意，当 `output_attentions=True` 时，`cross_attentions` 可能为 `None`（取决于模型是否支持），建议运行前打印 `outputs` 的属性以确认。上述输出会随 transformers 版本和模型版本变化，实际以运行结果为准。

**动手验证**：运行后，记录你得到的四个形状值。重点对比 `encoder_last_hidden_state` 和 `decoder_hidden_states[-1]` 的维度差异——前者只含输入序列长度，后者还包含 decoder 前缀长度。这个差异在 vLLM 部署时会影响 KV Cache 的分配策略。

---

## 七、架构演化：为什么后续模型"拆分"了 Encoder-Decoder？

理解了完整的 Encoder-Decoder 结构后，一个自然的问题是：为什么 GPT-2、Llama

## 块 18

**目标：vLLM部署，先搞懂为什么主流模型都是Decoder-only**

你学vLLM，核心是部署GPT这类生成模型。所以看架构演化，重点就是搞清楚：为什么后来大家都“砍掉”了Encoder，只留Decoder？

**7.1 Encoder-only：BERT（2018）**

BERT只用Encoder，去掉Decoder。它靠**Masked Language Model（MLM）**训练——盖住词猜原词，所以能看**双向上下文**。适合**理解任务**（分类、问答、NER），但**不适合生成**。

**7.2 Decoder-only：GPT（2018-2024）**

GPT反过来，只用Decoder，去掉Encoder和Cross-Attention。训练方式是**Causal Language Model（Next Token Prediction）**，只能往前看。

对你部署vLLM，它的优势很关键：

- **架构更简单**，参数量全在Decoder，容易扩展到超

## 块 19

## 7.3 Encoder-Decoder 仍有广泛应用（①）

你已经知道通用语言模型主流是 Decoder-only，但别急着把 Encoder-Decoder 扔进垃圾桶——在**特定任务**里它依然是主力，而且特定条件下效果可能更好。直接看例子：

- **机器翻译**：`google-t5/t5-small`、`facebook/mbart-large-50`
- **语音识别**：`openai/whisper-base`
- **多模态**：`OFA`、`UniT`

这些模型都是 Encoder-Decoder 架构。为什么？因为翻译、语音识别这类任务，**输入和输出是两种不同形态的东西**——Encoder 先把源语言"读懂"，Decoder 再逐词生成目标语言，分工清晰。

> 后续"重点理解 Decoder-Only 架构"会讲：GPT 去掉 Encoder 后为什么还能工作？核心答案是——**把输入和输出拼接在一起**，用统一的 Causal Mask 处理。先记住这个结论，后面细拆。

## 八、常见问题与解决（①）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 混淆三种 Attention 的 Q/K/V 来源 | 没有系统对比 | 记住核心区别：Self-Attention 的 Q/K/V 同源，Cross-Attention 的 K/V 来自 Encoder |
| 不理解为什么 Decoder 需要 Mask | 忽略了自回归性质 | 生成任务中，模型不能"偷看"未来的正确标签 |
| 以为 Encoder 输出只送到 Decoder 第一层 | 对数据流理解不深入 | Encoder 输出送到**每一层** Decoder 的 Cross-Attention |
| Cross-Attention 的 K 和 V 能否只用其一 | 对 Attention 机制理解不深入 | 在标准实现中缺一不可，但存在共享或简化的变体 |

**动手验证**：拿 T5 跑一次前向传播，打印 `cross_attentions` 矩阵，观察 Decoder 生成不同 token 时分别关注了输入序列的哪些位置。这一步能帮你把"Cross-Attention 的 K/V 来自 Encoder"从文字变成直觉。

## 九、总结与后续学习路线（①）

到这里，你已经掌握了：

- ✅ Encoder-Decoder 架构的本质：Encoder 双向编码，Decoder 单向自回归生成
- ✅ Encoder 的层次结构：Self-Attention + FFN，残差连接 + LayerNorm
- ✅ Decoder 的层次结构：Masked Self-Attention

## 块 20

## 为什么 GPT / Llama / Qwen 可以去掉 Encoder？

你已经在上一节掌握了完整的 Encoder-Decoder 结构，知道 Decoder 依赖 Masked Self-Attention 和 Cross-Attention。现在问题来了：**GPT、Llama、Qwen 这些大模型为什么只保留 Decoder？去掉 Encoder 后，模型还怎么“理解”输入？**

### 核心答案：把输入和输出拼成同一个序列

Decoder-Only 架构的关键在于**统一的因果语言建模目标**。它不区分“输入”和“输出”，而是把 prompt 和要生成的内容组织成**同一个 token 序列**。

在这个序列里：

- **prompt 内部**：仍然遵守从左到右的因果可见性——每个 token 只能看到它前面的 token，这和 Encoder-Decoder 中 Decoder 的 Masked Self-Attention 规则一致。
- **输出 token**：能访问它之前的**全部 prompt**。也就是说，模型生成下一个词时，已经“看”完了你给它的所有上下文。

所以，Encoder 的职责（编码输入）被“吸收”进了这个统一的序列里——prompt 本身就是序列的开头部分，Decoder 在自回归生成时天然就能访问到它。

### 为什么这样设计？——为了扩展性

Decoder-Only 通过**统一目标和统一结构**获得了较好的**扩展性**。所有任务（对话、写作、代码生成）都变成“给定前面的 token，预测下一个 token”，这让模型可以在海量文本上规模化训练，架构简单、工程实现也更高效。

### 但 Encoder-Decoder 并没有被淘汰

注意：**不能仅根据输入输出是否为同一种模态来判断架构优劣**。Encoder-Decoder 在**翻译、摘要**等条件生成任务中仍有独特优势：

- **独立编码输入**：Encoder 可以一次性双向编码整个输入，捕捉全局语义。
- **计算复用**：输入只需编码一次，生成多个输出时不必重复处理。
- **任务归纳偏置**：结构本身自带“输入-输出分离”的归纳偏置，更适合严格的条件生成。

### 一句话总结

架构选择取决于**训练目标、任务类型和工程约束**。Decoder-Only 靠“统一序列 + 因果可见性”去掉了 Encoder，换来扩展性；Encoder-Decoder 靠“独立编码”保住条件生成任务的效率。没有绝对优劣，只有适不适合。

---

**本块学习检查**：请用一句话向自己解释——在 Decoder-Only 架构中，模型生成第一个输出 token 时，它能“看到”哪些内容？它和 Encoder-Decoder 中 Decoder 生成第一个 token 时的

## 块 21

### 1.2 统一框架：Causal Language Modeling（①）

你的目标是部署 vLLM，而 vLLM 的核心推理对象正是 Decoder-Only 模型。理解这里的“统一框架”是后续配置 KV Cache、优化吞吐的基础。

**核心思想**：Decoder-Only 模型只做一件事——**因果语言建模（CLM）**，也叫“预测下一个 token”。给定前 $t$ 个 token，预测第 $t+1$ 个。数学上：

$$P(x_1, ..., x_n) = \prod_{t=1}^{n} P(x_t | x_1, ..., x_{t-1})$$

**为什么这对 vLLM 部署是好事？** 看四个优势：

| 优势 | 对你的意义 |
|---|---|
| **架构简化** | 没有 Encoder 和 Cross-Attention，部署时只需关注 Decoder 的权重，参数量更集中，模型加载和推理路径更短 |
| **数据要求低** | 不需要成对的输入-输出数据，任何纯文本都能训练，这意味着你部署的模型（如 Qwen）可以处理任意文本续写任务 |
| **统一接口** | 问答、摘要、代码等

## 块 22

**GPT-1：Decoder-Only 的开山之作**

你的目标是 vLLM 部署，核心是让模型高效跑起来。理解 GPT-1 的架构取舍，能帮你搞懂为什么这类模型在部署时如此“吃香”。

2018 年的 GPT-1 没搞复杂模块，就做了三个关键决定：

1. **只留 Decoder**：砍掉 Encoder 和 Cross-Attention，结构更简洁。
2. **训练目标**：直接预测下一个 Token（Next Token Prediction）。
3. **两阶段**：先在海量无标注文本上预训练，再针对具体任务微调。

**动手验证**：部署时你只需加载一个 Decoder 模型，输入 prompt，模型自动续写——

## 块 23

### GPT-2：规模即能力（①）

2019年，GPT-2没改核心架构，只把**规模**推到新高度：1.5B参数，40GB训练数据。结果出现惊人现象——**zero-shot能力**：不微调参数，只靠设计不同的输入prompt，模型就能执行翻译、摘要、问答等任务。这说明Decoder-Only的"续写"能力本身就蕴含通用性。

**对你部署vLLM的启示**：GPT-2证明，模型能力主要来自规模而非架构改动。你在用vLLM部署时，选模型大小直接影响显存和吞吐——规模越大，能力越强，但推理资源需求也越高。注意：部署开源模型需遵守其许可证，例如Llama 2 Community License（商用月活>7亿需Meta授权）。

**动手验证**：用vLLM加载一个小型Decoder-Only模型（如GPT-2），输入一个翻译prompt（如"Translate to French: Hello"），观察不微调时模型能否直接输出

## 块 24

### 3.1 Llama 不是推翻，而是"做减法 + 做强化"

你已了解 GPT 的 Decoder-Only 路线，现在看 Llama 如何在此路线上优化——它没有推翻架构，而是对三个关键组件做了"做减法 + 做强化"：

| 改进项 | 原始 Transformer | Llama 的做法 | 作用 |
|---|---|---|---|
| **归一化** | Post-LayerNorm（子层后归一化） | **Pre-RMSNorm**（子层输入前归一化） | 训练更稳定，简化计算，提升训练稳定性 |
| **激活函数** | ReLU | **SwiGLU** | 表达能力更强，门控机制过滤信息 |
| **位置编码** | 绝对正弦位置编码 | **RoPE（旋转位置编码

## 块 25

## Llama 三代演进：从“能训”到“好用”再到“全能”

你学 vLLM 部署，核心是理解模型架构如何影响推理。Llama 三代演进正好展示了 Decoder-Only 架构的实用化路径。

**Llama1**：开源高性价比基础模型，上下文较短，以 base 模型为主。它的意义在于证明“公开数据也能训出强模型”——这为后续开源生态铺路。

**Llama2**：定位更成熟的对话基座，支持 4K 上下文，Chat 对齐更成熟。关键变化是“开始强调实用性”——模型从实验室走向产品。

**Llama3**：定位多语言、代码、工具综合能力平台，部分版本（如 Llama

## 块 26

**本块学习检查**：读完 Qwen3 架构亮点后，请用一句话回答：MoE 模型相比 Dense 模型，在“总参数量”和“每 token 实际计算量”上分别有什么特点？

---

### 4.1 Qwen3 的架构亮点（②）

Qwen3（阿里巴巴，2025）在 Decoder-Only 主线上做了三处关键创新，直接关系到你后续用 vLLM 部署时的显存规划和推理速度调优。

**（1）Dense + MoE 双轨并存**

- **Dense 模型**：每层 FFN 只有一套权重，每个 token 前向时全部参与计算（如 Qwen3-8B、14B、32B）。
- **MoE 模型**：FFN 换成多个独立专家，门控网络每 token 只激活少数专家（如 Qwen3-30B-A3B：总参 30B，每 token 激活 3B；235B-A22B：总参 235B，每 token 激活 22B）。

> **MoE 优势**：模型仓库能装下超大总参数量，但生成每个 token 的实际计算量只和激活的支路相当，兼顾容量与效率。部署时注意：**显存占用看总参，推理速度看激活参**。

**（2）Thinking / Non-Thinking 统一**

Qwen3 把“深度推理”和“快速响应”统一进**同一套权重**，通过 chat template 中的开关（如 `/think`、`/no_think`）控制。你还可以设置 **thinking budget**（思考预算），在效果与延时之间灵活平衡——部署时这是调节响应速度的实用旋钮。

**（3）关键架构细节演进**

| 项目 | Qwen1 | Qwen2 | Qwen3 |
|---|---|---|---|
| 注意力 | MHA 为主

## 块 27

## 4.2 Qwen3 的分阶段预训练

Qwen3 的预训练不是"一把梭"，而是分阶段推进，这个设计直接关系到你部署 vLLM 时看到的模型能力分布：

**第一阶段：广度优先**
使用大规模高质量数据，在短序列上进行广谱学习，让模型吸收语言与世界知识。相当于先给模型"喂饱"常识底座。

**第二阶段：深度强化**
使用高质量数据，在短序列上强化推理相关分布，让模型在逻辑任务上更"聪明"。

**第三阶段：长度扩展**
使用适量数据，将序列长度拉长，目的是让优化目标直接覆盖万级 token 的因果依赖——这对你部署长上下文推理场景至关重要。

**核心思想：先广度、再深度、最后长度。**

> 部署提示：三阶段顺序意味着模型在长序列上的表现是"后训练"出来的，vLLM 部署时若遇到长文本性能问题，优先检查推理阶段的上下文窗口配置，而非模型本身能力缺失。

---

## 五、三条主线的架构对比总结

| 维度 | GPT 系列 | Llama 系列 | Qwen 系列 |
|---|---|---|---|
| **核心路线** | Decoder-only + 规模化 | Decoder-only + 组件优化标准化 | Decoder-only + MoE + 多模态能力 |
| **归一化** | LayerNorm → 后续改进 | **Pre-RMSNorm** | **Pre-RMSNorm** |
| **位置编码** | 绝对位置编码 | **RoPE** | **RoPE** |
| **激活函数** | GeLU / 其他 | **SwiGLU** | **SwiGLU** |
| **注意力优化** | 标准 MHA | 后续引入 GQA | **全系 GQA + QK-Norm** |
| **长上下文** | 逐步扩展 | 逐步扩展 | **分阶段预

## 块 28

## 6.1 观察 GPT-2 的“原始”Decoder-Only 结构

既然你的目标是 vLLM 部署，先亲手“解剖”一个最经典的 Decoder-Only 模型，能帮你建立对后续部署对象的直观认知。GPT-2 就是最好的起点——它结构简单、资源占用低，非常适合快速上手。

**动手运行以下代码**（建议在 Jupyter 或脚本中逐行执行）：

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
config = model.config

print(f"层数: {config.n_layer}")
print(f"隐藏维度: {config.n_embd}")
print(f"注意力头数: {config.n_head}")
print(f"FFN 中间维: {config.n_inner or config.n_embd * 4}")
print(f"位置编码: 绝对位置编码 (n_positions={config.n_positions})")
print(f"激活函数: {config.activation_function}")
```

**预期输出**：

```
层数: 12
隐藏维度: 768
注意力头数: 12
FFN 中间维: 3072
位置编码: 绝对位置编码 (n_positions=1024)
激活函数: gelu_new
```

**关键观察**：GPT-2 采用绝对位置编码、近似 GELU（gelu_new）激活函数。

**部署说明**：GPT-2 采用 MIT 许可证，可自由商用。加载模型时 # 注意：GPT-2 显存需求较低，但部署更大模型

## 块 29

## 6.2 观察 Qwen2.5 的现代 Decoder-Only 结构

既然你的目标是 vLLM 部署，理解模型配置是第一步——部署时你需要根据这些参数决定显存分配和推理优化策略。下面直接用代码观察一个真实模型。

### 运行代码

```python
from transformers import AutoConfig

# 使用 Qwen2.5-0.5B（小模型，无需申请权限，适合教学观察）
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")

print(f"层数: {config.num_hidden_layers}")
print(f"隐藏维度: {config.hidden_size}")
print(f"Q 头数: {config.num_attention_heads}")
print(f"KV 头数: {config.num_key_value_heads}")  # GQA 的关键标志
print(f"FFN 中间维: {config.intermediate_size}")
print(f"位置编码: RoPE (rope_theta={config.rope_theta})")
print(f"归一化: RMSNorm (rms_norm_eps={config.rms_norm_eps})")
print(f"QKV bias: {getattr(config, 'attention_bias', False)}")
```

**输出**（以实际配置为准）：

```
层数: 24
隐藏维度: 896
Q 头数: 14
KV 头数: 2          ← GQA：14 个 Q 头共享 2 组 KV
FFN 中间维: 4864
位置编码: RoPE (rope_theta=1000000.0)  ← ABF：基频从 1e4 提升到 1e6
归一化: RMSNorm (rms_norm_eps=1e-06)
QKV bias: False     ← Attention 投影层不使用 bias
```

### 三个关键观察

1. **GQA（分组查询注意力）**：`num_key_value_heads=2` 远小于 `num_attention_heads=14`，意味着 14 个 Q 头共享 2 组 KV

## 块 30

## 6.3 对比三种 Attention 的 KV Cache 体积

KV Cache 是推理时缓存历史 token 的 K/V 张量，避免重复计算。它的显存占用直接决定你能跑多长的序列、多大的 batch——这正是 vLLM 部署时最关心的资源瓶颈之一。

我们用下面的函数来算：

```python
import torch

def kv_cache_size(batch, seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes=2):
    """计算 KV Cache 显存占用（字节）"""
    # 每层有 K 和 V 两个张量
    # 形状: [batch, num_kv_heads, seq_len, head_dim]
    per_layer = 2 * batch * num_kv_heads * seq_len * head_dim * dtype_bytes
    return num_layers * per_layer
```

公式拆解：每层需要存 K 和 V 两个张量（所以乘 2），每个张量形状是 `[batch, num_kv_heads, seq_len, head_dim]`，再乘上 `dtype_bytes`（默认 FP16，2 字节）。最后乘以层数。

**动手跑一下**，对比三种情况：

```python
# GPT-2（标准 MHA，num_kv_heads = num_attention_heads = 12）
gpt2_kv = kv_cache_size(1, 2048, 12, 12, 64)
print(f"GPT-2 KV Cache: {gpt2_kv / 1024 / 1024:.2f} MB")

# Qwen2.5-0.5B（GQA，num_kv_heads = 2, head_dim = 128）
qwen_kv = kv_cache_size(1, 2048, 24, 2, 128)
print(f"Qwen2.5-0.5B KV Cache: {qwen_kv / 1024 / 1024:.2f} MB")

# 对比：如果 Qwen 也用 MHA（假设 14 个 KV 头）
qwen_mha_kv = kv_cache_size(1, 2048, 24, 14, 128)
print(f"Qwen2.5-0.5B (假设 MHA) KV Cache:

## 块 31

你的目标是快速上手 vLLM 部署，而 KV Cache 体积直接决定显存占用和吞吐量。这里我们用 `transformers` 库实际对比 GPT-2 和 Qwen2.5 的配置，直观感受 GQA 带来的节省。

**动手观察：**

```python
from transformers import AutoConfig

for model_id in ["gpt2", "Qwen/Qwen2.5-0.5B"]:
    config = AutoConfig.from_pretrained(model_id)
    print(f"\n{model_id}:")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {getattr(config, 'num_key_value_heads', None)}")
```

**关键对比：**

- **GPT-2**：`num_key_value_heads` 不存在，默认等于 `num_attention_heads`（实际为 12），即每个注意力头都有独立的 K/V，KV Cache 体积 = 2 × 层数 × 12 × 序列长度 × 头维度。
- **Qwen2.5**：`num_key_value_heads` 远小于 `num_attention_heads`（GQA），多个查询头共享一组 K/V。以 Qwen2.5-0.5B 为例，`num_attention_heads` 为 14，`num_key_value

## 块 32

## 二、从 LayerNorm 到 RMSNorm

### 2.1 LayerNorm 的原始设计

在动手部署 vLLM 之前，你得先搞清楚模型内部的一个关键组件——LayerNorm。它是原始 Transformer 用来稳定深层网络训练的基石之一。

**为什么需要它？**

GPT-3 有 96 层，Llama、Qwen 也有几十层。网络一深，反向传播时梯度逐层相乘，数值容易失控（梯度消失/爆炸），而且深层网络比浅层更难训练。原始 Transformer 的解法是：**残差连接 + 层归一化**。

**LayerNorm 在做什么？**

简单说，它对每个样本的所有特征维度，算出一个均值 μ 和一个方差 σ²，然后做标准化，把数据拉回到一个稳定的分布上。公式如下：

```
LN(x_i) = γ · (x_i - μ) / √(σ² + ε) + β
```

拆开看：

- **μ**：该样本所有特征维度的均值
- **σ²**：该样本所有特征维度的方差
- **ε**：一个极小的数，防止分母为 0
- **γ 和 β**：可学习的缩放和偏移参数，让模型决定标准化后的数据“长什么样”

**关键点**：LayerNorm 是对**每个样本自己**做归一化，不依赖 batch 里的其他样本。这一点和 BatchNorm 不同，也是它适合 Transformer 这类变长序列输入的原因。

**原始 Transformer 怎么用？**

每个子层（比如注意力层、前馈层）的输出，先和输入做残差相加，再做 LayerNorm：

```
LayerNorm(x + Sublayer(x))
```

这个结构叫 **Post-LN**——先加残差，再归一化。

**但现代 LLM（比如 Llama）不这么干了**，它们改用：

```
x + Sublayer(RMSNorm(x))
```

每个子层的输入先经过 RMSNorm，然后进入子层，子层输出与原始输入相加（残差连接）。这叫 **Pre-LN**。

**为什么换？** 这正是下一节要拆解

## 块 33

你的目标是快速上手 vLLM 部署，而 vLLM 底层大量使用了 RMSNorm。理解它从 LayerNorm 简化而来的动机，能帮你更清楚地看懂推理时的计算流程。

LayerNorm 的原始设计是：先算均值 μ 和方差 σ²，再用它们把数据标准化，最后用可学习的 γ（缩放）和 β（平移）做变换。公式如下：

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \quad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$$

RMSNorm 论文提出了一个关键假设：LayerNorm 之所以有效，主要靠的是**重缩放不变性**，而不是**重中心化不变性**。换句话说，减去均值（re-centering）那一步对最终效果贡献不大。

基于这个假设，RMSNorm 直接砍掉了均值计算和偏置 β，只保留 RMS 统计量：

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}$$

$$\text{RMSNorm}(x_i) = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i$$

数值稳定形式

## 块 34

**为什么可以去掉均值和偏置？——从 vLLM 部署视角看**

你要部署 vLLM，核心是让模型跑得快、省显存。RMSNorm 就是为此服务的。

看 Pre-LN 结构：`x + Sublayer(Norm(x))`。Norm 先把输入均值归零，但 Attention 或 FFN 这些子层一过，线性变换加激活函数，输出均值早就不为零了。再和原始 x 相加，最终结果均值主要由 x 决定，**零均值根本保不住**。

所以两个结论：

1. **偏置 β 冗余**：它和后续线性层的 bias 功能重叠，纯属浪费参数。
2. **去均值冗余**：残差连接注定破坏零均值，你费劲算均值、强行减掉，下一轮又被重置，白算。

RMSNorm 论文实验也证实

## 块 35

## 2.4 计算效率对比：为什么 RMSNorm 更快？

假设隐藏维度为 $d$，归一化层对每个 token 的每个隐藏维度都要计算，频率极高。对比两者的操作步骤：

| 步骤 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 求均值 / 平方 | $d$ 次加法算 $\mu$ | $d$ 次乘法算 $x_i^2$ |
| 中心化 / 求和 | $d$ 次减法 $(x_i - \mu)$ | $d$ 次加法算 $\sum x_i^2$ |
| 平方 / 开方 | $d$ 次乘法算方差 | 开方 + 除法 |
| 归一

## 块 36

## Post-Norm：原始 Transformer 的选择

原始 Transformer 的每个子层（如自注意力或前馈网络）采用这样的结构：

**Output = LayerNorm(x + Sublayer(x))**

拆开看：输入 `x` 先进入子层计算，得到的结果与原始 `x` 相加（残差连接），最后再做 LayerNorm 归一化。也就是说，**残差分支在归一化之前**。

这个设计的代价是：训练深层网络时稳定性较差，对学习率很敏感——学习率稍大就容易震荡甚至发散。但它的优势在于模型表达能力理论上更强。

---

**给你的实操提示**：如果你在 vLLM 部署中遇到加载模型后推理结果不稳定、或微调时 loss 剧烈波动的情况，可以先检查模型用的是 Post-Norm 还是 Pre-Norm 架构——Post-Norm 模型对学习率和初始化更"挑剔"，这不是部署代码的问题，而是架构本身的特性。

# 模型许可证：Llama 2 Community License（商用

## 块 37

**本块学习检查**：读完下面内容后，请用一句话回答：残差连接为什么能缓解梯度消失？答案就在原文里，找出来即可。

---

### 4.1 梯度高速公路

没有残差连接时，梯度要逐层反向传播。假设网络有100层，每层梯度只衰减到0.99，最终梯度只剩 $0.99^{100} \approx 0.366$——梯度几乎消失了。残差连接相当于给梯度修了一条“高速公路”，让它能直接抄近路回传，从而缓解梯度消失。

### 4.2 学习残差而非完整映射

传统网络要直接学映射 $F(x)$。残差网络只学残差 $H(x) = F(x) - x$。当变换接近恒等映射时，残差趋近于0，这比学一个任意复杂映射容易得多。

### 4.3 残差连接与激活值控制

残差连接的核心作用就是改善信息和梯度传播。但要注意：未经缩放的残差不断相加，激活方差会随网络深度增长。现代模型通常用 **Pre-Norm/RMSNorm**、合理的权重初始化或残差缩放（如 GPT-3 的 $1/\sqrt{N}$ 缩放）来控制。

### 4.4 标准 Decoder-Only 层的残

## 块 38

**RMSNorm 性能优化：`torch.rsqrt`**

RMSNorm 的核心计算是除以 $\sqrt{\text{variance} + \epsilon}$。PyTorch 的 `torch.rsqrt` 直接算 $1/\sqrt{x}$，在 CUDA 上通常映射为 `rsqrt.rn.f32`（精确）或 `rsqrt.approx.f32`（近似，取决于实现和设置），但无论哪种，其性能通常优于先开方再除法。

**对比两种写法**：

```python
# ❌ 慢：先开方，再除法（两次高延迟操作）
denominator = torch.sqrt(variance + eps)
output = x / denominator

# ✅ 快：rsqrt + 乘法（一条专用指令 + 一次快速乘法）
inv_rms = torch.rsqrt(variance + eps)
output = x * inv_rms
```

**为什么对你重要**：vLLM 部署时，

## 块 39

**目标：vLLM部署**，你最终要把模型高效跑起来。混合精度是部署时的常态，而RMSNorm里的类型转换是保证数值稳定的关键一步。

> **模型许可证**：Llama 2 Community License（商用月活>7亿需Meta授权）。

## 为什么通常转FP32？

大模型训练和推理常用BF16/FP16省显存，但RMSNorm的**统计量累加通常建议在FP32下进行**，原因有三：

1. **防溢出**：长序列（如32K）的平方和累加，BF16/FP16很容易超范围变成`Inf`
2. **防精度丢失**：低精度下累加会“大数吃小数”，方差算不准
3. **保rsqrt稳定**：FP32的倒数平方根精度更高，避免梯度异常

## 标准流程

**输入(BF16) → 转FP32算统计量 → 归一化 → 转回BF16 → 乘γ**

## 代码实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype  # 记录原始精度
        
        x_fp32 = x.float()  # 【关键】提升到FP32
        
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        x_normed = x_fp32 * inv_rms
        
        x_normed = x_normed.to(input_dtype)  # 【关键】转回原始精度
        return x_normed * self.weight.to(input_dtype)  # 【关键】权重同步转换
```

## 块 40

## 5.4 与 PyTorch 原生 LayerNorm 对比

你的目标是 vLLM 部署，而 vLLM 在底层推理时对算子和参数的效率极其敏感。理解 RMSNorm 与 LayerNorm 的差异，能帮你判断部署时模型配置中的归一化层选择。

直接看代码对比（`hidden_dim=512`）：

```python
import torch
import torch.nn as nn

hidden_dim = 512

# 原生 LayerNorm（含均值、方差、gamma、beta）
ln = nn.LayerNorm(hidden_dim, elementwise_affine=True)

# 自定义 RMSNorm（仅 RMS + gamma）
rms_norm = RMSNorm(hidden_dim)

# 模拟输入：[batch=2, seq_len=10, hidden_dim=512]
x = torch.randn(2, 10, hidden_dim)

out_ln = ln(x)
out_rms = rms_norm(x)

print(f"LayerNorm 输出形状: {out_ln.shape}")
print(f"RMSNorm 输出形状:  {out_rms.shape}")

# 观察 gamma 参数量差异
print(f"LayerNorm 参数: {sum(p.numel() for p in ln.parameters())}")   # 1024 (gamma + beta)
print(f"RMSNorm 参数:   {sum(p.numel() for p in rms_norm.parameters())}")  # 512 (仅 gamma)
```

**观察**：RMSNorm 比 LayerNorm 少了一半的可学习参数（去掉了 β）。在 vLLM 部署场景中，参数减少意味着权重文件更小，但显存和计算优势需结合具体实现（如融合算子）评估，不能仅凭参数量直接推断。此外，LayerNorm 的 beta 参数在 Pre-LN 残差结构中可能被残差连接抵消，但并非所有场景都冗余。

## 常见问题与解决

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| "去掉均值后效果不会变差吗？" | 忽略了 Pre-LN + 残差的结构特性 | Pre-LN 中残差连接会重置均值，去均值在特定条件下可能是冗余计算；RMSNorm 原论文（Zhang & Sennrich, 2019）在多种任务上报告了与 LayerNorm 相近的效果，但具体差异取决于模型架构和任务类型，不能一概而论 |
| "Post-LN 和 Pre-LN 怎么选？" | 混淆了架构演进时序 | 现代 LLM 训练通常优先选 Pre-LN，通常更稳定；Post-LN 是原始 Transformer 的选择 |
| "代码里为什么要 `.float()`？" | 不理解混合精度训练原理 | 统计量累加在 FP32 下更安全，防止 BF16/FP16 溢出和"大数吃小数" |
| "`rsqrt` 和 `1/sqrt` 有什么区别？" | 不了解硬件指令差异 | `rsqrt` 是单条硬件指令，延迟更低，且将除法转为乘法；但其精度可能略低于 `1/sqrt`，且并非所有硬件都支持该指令，需根据部署平台权衡 |
| "训练深层模型 loss 为 NaN" | 梯度爆炸或统计量溢出 | 检查是否使用 Pre-LN + RMSNorm，确认 `eps` 设置合理，添加梯度裁剪 |
| "RMSNorm 的

## 测评题目

### Q-KB08-ST0800-001
用 `AutoModel.from_pretrained()` 加载模型时，模型权重通常来自名称对应的预训练________。

### Q-KB08-ST0800-002
不计算梯度地执行前向推理，常使用 PyTorch 的 `torch.________()` 上下文。

### Q-KB0801-003
以下哪个模型是典型的 **Encoder-only** 架构？

### Q-KB0801-001
Transformer模型中，用于处理机器翻译等序列到序列（Seq2Seq）任务的完整架构是？

### Q-KB08-ST0802-001
Decoder-Only 模型在生成第 t 个 token 时，只能关注当前位置及其________位置。

### Q-KB08-ST0802-003
Decoder-Only 注意力掩码的核心作用是？

### Q-KB08-ST0803-003
残差连接最直接的训练收益是？

### Q-KB08-ST0803-002
RMSNorm 使用均方根统计进行归一化，通常不减去输入的________。
