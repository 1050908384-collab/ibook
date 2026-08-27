# 学习包

## 块 1

**本块学习检查**（先做，再继续）

在动手安装前，先用一个最小示例回顾前置知识：用 `AutoTokenizer` 和 `from_pretrained()` 加载一个分词器，并对一句话完成编码。确认你能解释 `from_pretrained()` 的作用（从预训练仓库加载配置与权重），以及 `AutoTokenizer` 如何自动匹配模型对应的分词器。能说清这两点，再进入下面的安装与验证。

---

## 一、安装与验证 transformers

### 1.1 快速安装

你的目标是 vLLM 部署，但环境准备是第一步。如果你已完成第七板块，`transformers` 库应该已经装好。若未安装，在已有的 `pytorch` conda 环境中执行：

```bash
conda activate pytorch
pip install transformers
```

若需要同时安装 PyTorch 后端依赖，可使用：

```bash
pip install 'transformers[torch]'
```

> **本地运行配置**：`transformers` 库体积通常在数 MB 至十余 MB，安装需联网。CPU 即可运行，内存建议 ≥ 4GB。

### 1.2 验证安装并加载 GPT-2

安装完成后，用一段最小代码验证库可用，并加载小型模型 GPT-2 做一次前向传播。这一步是为了让你直观看到 Transformer 解码器的输出形态，为后续理解 Decoder-Only 架构、RMSNorm、SwiGLU、RoPE 等概念建立感性认识。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载分词器和模型（首次运行会联网下载）
# 模型许可证：GPT-2 采用 MIT 许可证，可自由使用
# 注意：GPT-2 模型 FP32 推理约需 1.5GB 显存
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 编码输入
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

# 前向传播，输出 logits
outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

# 观察输出结构
print(outputs.logits.shape)          # 词汇表概率分布
print(len(outputs.hidden_states))    # 各层隐藏状态
print(len(outputs.attentions))       # 各层注意力权重
```

**操作验证任务**：运行上述代码，记录 `logits` 的维度，并数一数 `hidden_states` 和 `attentions` 各包含多少层。这个数字对应 GPT-2 的层数，记住它，后续讲解架构时会用到。

> 注意：首次运行需联网下载模型权重（约 500-600MB），请确保网络畅通

## 块 2

### 验证安装：跑通你的第一个 HuggingFace 实验

安装完成后，别急着往下走——先花 30 秒验证环境是否真的可用。打开你的 Python 环境，逐行运行以下代码：

```python
from transformers import pipeline
result = pipeline('sentiment-analysis')('we love you')
print(result)
```

这段代码会加载一个预训练的情感分析模型，对 `'we love you'` 这句话做判断，并打印结果。预期输出类似这样（实际内容会随 transformers 和模型版本略有不同，分数也可能有差异）：

```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

看到 `POSITIVE` 和接近 1 的分数，说明 transformers 安装成功、模型下载和推理链路都通了。如果报错，参考下面的排查表。

---

### 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `Failed building wheel for tokenizers` | 缺少 Rust 编译工具链 | 用 `conda install -c conda-forge tokenizers` 安装预编译版本，或尝试安装特定版本 |
| `urllib3.exceptions.SSLError` | `urllib3` 版本过高 | 执行 `pip install 'urllib3<2.0'` 或更新 `certifi`，并检查网络代理设置 |
| 无法连接 HuggingFace Hub | 网络限制 | 设置镜像源：`export HF_ENDPOINT=https://hf-mirror.com`，或参考离线配置方法 |

---

### 下一步：选对 GPT-2 模型

GPT-2 是 OpenAI 发布的因果语言模型，有多个尺寸。你的目标是 vLLM 部署，但先用小模型跑通流程更稳妥。本实验选用 **`openai-community/gpt2`**（124M）。# 模型许可证：MIT License # 注意：加载该模型进行推理时

## 块 3

### 2.1 使用 pipeline 快速体验

`pipeline` 是 transformers 的高级 API，让你**一行代码**就能跑通 GPT-2。既然你的目标是 vLLM 部署，这一步先建立“模型能跑”的直觉，后续对比 vLLM 的加速优势才有参照。

**操作步骤**（hands-on 直接复制运行）：

```python
from transformers import pipeline

# 创建文本生成 pipeline，device=-1 表示使用 CPU
# 使用 safetensors 格式加载权重，更安全且加载更快
generator = pipeline("text-generation", model="openai-community/gpt2", device=-1, use_safetensors=True)

result = generator("Hello, I'm a language model", max_new_tokens=20)
print(result)
```

**运行前须知**：
- 首次运行会自动从 HuggingFace Hub 下载 GPT-2 模型文件（约 500MB，实际以版本为准），**必须联网**。
- 下载后缓存到本地，后续运行不再重复下载。
- CPU 即可

## 块 4

### 手动加载 GPT-2：看清内部结构

既然你的目标是 vLLM 部署，理解模型加载的底层细节是第一步。这里我们不用 `pipeline` 这种高级封装，而是直接调用底层 API，把 Tokenizer 和 Model 分开加载，这样能更清楚地看到 GPT-2 的“骨架”。

**操作步骤：**

```python
from transformers import AutoTokenizer, GPT2LMHeadModel

# 模型许可证：MIT License（GPT-2 为开源模型，可自由商用）
# 注意：GPT-2 模型 FP16 推理需要至少 2GB 显存
# 1. 加载分词器（负责把文本转成数字 ID）
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# 2. 加载模型（GPT-2 是 Decoder-Only 架构）
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# 3. 关键一步：GPT-2 默认没有 pad_token，不设置的话 batch 处理会报错
tokenizer.pad_token = tokenizer.eos_token

# 4. 打印模型配置，观察核心超参数
print(f"词表大小: {model.config.vocab_size}")
print(f"最大序列长度: {model.config.max_position_embeddings}")
print(f"隐藏层维度: {model.config.n_embd}")
print(f"层数: {model.config.n_layer}")
print(f"注意力头数: {model.config.n_head}")
```

**这些参数意味着什么？**

- `n_embd`（隐藏层维度）：每个 token 被映射成多长的向量。
- `n_layer`（层数）：Transformer Block 堆叠了多少

## 块 5

## 3.1 完整前向传播流程

你的目标是 vLLM 部署，但部署前必须看清模型“内部怎么想”。这一节我们直接跑通 GPT-2 的前向传播，观察三个关键输出。

**第一步：准备输入**

```python
inputs = tokenizer(text, return_tensors="pt")
```

`return_tensors="pt"` 让 tokenizer 返回 PyTorch 张量。`input_ids` 的形状是 `(1, token数量)`——1 是 batch size，token 数量由你的文本长度决定。

**第二步：执行前向传播**

```python
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
```

- `torch.no_grad()`：推理时不需要计算梯度，省显存、提速度——部署时这是常态。
- `output_hidden_states=True`：让模型返回每一层的隐藏状态。
- `output_attentions=True`：让模型返回每一层的注意力权重。

**第三步：观察输出**

```python
print(f"logits 形状: {outputs.logits.shape}")
print(f"hidden_states 数量: {len(outputs.hidden_states)}")
print(f"attentions 数量: {len(outputs.attentions)}")
```

## 块 6

## 3.2 观察下一词预测

既然你的目标是 vLLM 部署，理解模型“如何预测下一个词”是部署时排查输出异常的基础。这里我们用 GPT-2 做一次最直接的前向传播观察。

**操作：** 取模型输出中最后一个位置的 logits（即模型对“下一个词”的打分），用 `argmax` 取出得分最高的 token ID，再解码成文字：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('gpt2', output_logits=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

next_token_logits = outputs.logits[0, -1, :]
next_token_id = torch.argmax(next_token_logits).item()
next_token = tokenizer.decode(next_token_id)

print(f"模型预测的下一个词: '{next_token}'")
```

这就是 GPT-2 生成文本的基本单元——每步只预测一个词，然后把它拼回输入，继续预测下一个。

## 3.3 观察因果注意力掩码

GPT-2 是 Decoder-Only 模型，使用**因果注意力**：每个 token 只能看到自己和之前的 token，不能偷看未来。这一点直接影响 vLLM 部署时的 KV Cache 机制（后续你会遇到）。

**验证方法：** 取出第一层注意力权重，检查上三角区域是否接近 0：

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('gpt2', output_attentions=True)
token

## 块 7

**本块学习检查**：运行下方代码，观察 Loss 数值是否小于 10。若大于 10，检查是否误传了 `labels` 参数。

---

**目标**：用一行代码让 GPT-2 输出损失值，为后续 vLLM 部署中的性能评估打基础。

**核心逻辑**：训练时，模型要预测“下一个词”。所以 `labels` 就是 `input_ids` 本身，但 transformers 会自动把标签**左移一位**——预测第 i 个词时，目标其实是第 i+1 个词。你**不需要手动偏移**。

**动手做**（直接复制运行）：

```python
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])

print(f"Loss: {outputs.loss.item():.4f}")
```

**观察点**：
- `outputs.l

## 块 8

**本块学习检查**：运行 `echo $HF_HOME`，确认输出是你刚设置的缓存路径；若为空，说明环境变量未生效，检查命令是否在同一个终端执行。

你的目标是 vLLM 部署，但模型下载是第一步。HuggingFace 默认把模型缓存到 `~/.cache/huggingface/hub`（Linux/macOS）或 `C:\Users\username\.cache\huggingface\hub`（Windows）。部署时磁盘空间紧张，建议手动改路径：

```bash
export HF_HOME="/path/to/your/cache"
```

`HF_HOME` 是推荐方式，一次设置全局生效；`TRANSFORMERS_CACHE` 也可用，但仅影响 transformers 库的缓存路径，而

## 块 9

### 离线运行：为 vLLM 部署打好本地基础

你的目标是 vLLM 部署，而 vLLM 通常运行在受控的服务器或内网环境中。因此，**离线运行**不是可选项，而是你即将面对的常态。这一节帮你搞定它。

#### 核心操作：两步走

**第一步：设置环境变量，告诉工具“别联网”**

在防火墙或内网环境中，执行以下命令：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

这两个变量分别让 `transformers` 库和 `datasets` 库进入离线模式，不再尝试访问 HuggingFace 中心。

**第二步：直接指向本地路径加载模型**

离线模式下，不再使用模型名称，而是使用本地目录路径。加载时建议用 `try-except` 捕获异常，以便在文件缺失或路径错误时快速定位问题：

```python
try:
    tokenizer = AutoTokenizer.from_pretrained("/local/path/to/gpt2")
    model = GPT2LMHeadModel.from_pretrained("/local/path/to/gpt2")
except OSError as e:
    print(f"加载失败，请检查本地路径是否正确：{e}")
```

> **模型许可证**：若你后续部署 Llama 2 系列模型，需遵守 Llama 2 Community License；当商用且月活用户超过 7 亿时，需另行获得 Meta 授权。部署前请确认所用模型的许可证条款。

#### 关键准备：预先下载模型

离线环境没有网络，所以模型必须**提前在有网环境下载好**。流程是：

1. 在有网机器上，用 `from_pretrained()` 正常下载模型。
2. 用 `save_pretrained("./local/path")` 将模型和分词器保存到指定目录。
3. 将整个目录拷贝到离线环境。

> **动手验证**：如果你当前有网络，可以立即用 `save_pretrained` 保存一个小模型（如 `openai-community/gpt2`）到本地目录，然后设置 `TRANSFORMERS_OFFLINE=1`，再用本地路径加载它，确认离线流程跑通。

#### 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `OSError: Can't load tokenizer` | 本地路径缺少必要的配置文件 | 确保本地目录包含完整的分词器文件 |
| `RuntimeError: Expected all tensors on same device` | 模型和数据不在同一设备 | 确保 `model.to(device)` 和数据 `.to(device)` 一致 |
| GPT-2 padding 报错 | 默认没有 `pad_token` |

## 块 10

## 三、Encoder 的层次结构（①）—— Multi-Head Self-Attention

你之前用 GPT-2 做解剖实验时，看到注意力矩阵是下三角——那是 Decoder 的因果掩码。现在我们把镜头切到 Encoder 这边，它的 Self-Attention **没有掩码**，因为 Encoder 的任务是"看懂整个句子"，而不是"预测下一个词"。

### 3.1 为什么叫"多头"？

想象你读一句话时，不会只盯着一个角度：你会同时注意主谓关系、指代关系、修饰关系……"多头"就是让模型**并行地从多个子空间**去计算注意力。

具体来说，原始论文把注意力计算拆成 $h$ 个"头"（论文里 $h=8$）。每个头有自己独立的 $W^Q, W^K, W^V$ 权重矩阵，把输入映射到不同的表示空间，各自做一次 Scaled Dot-Product Attention，得到 $h$ 个输出。最后把这 $h$ 个输出**拼接**起来，再过一层线性变换，得到最终的 Multi-Head Attention 输出。

### 3.2 数据流长什么样？

在 Encoder 的每一层里，数据是这样流动的：

1. 输入序列 $(x_1, ..., x_n)$ 先经过 Embedding + 位置编码，得到初始表示。
2. 这个表示**同时作为 Query、Key、Value** 送入 Multi-Head Self-Attention——这就是"Self"的含义：序列自己跟自己算注意力。
3. 注意力输出经过残差连接（加上输入）和 LayerNorm，得到本子层的输出。
4. 再送入一个前馈网络（FFN），同样经过残差连接和 LayerNorm，完成这一层的处理。

**关键点**：Encoder 的 Self-Attention 是**双向**的——计算第 $i$ 个位置时，可以看到序列

## 块 11

## 3.1 Multi-Head Self-Attention —— 为 vLLM 部署打基础

先看整体图：在原始 Transformer 论文中，Encoder 有 6 层，每层都由 **Multi-Head Self-Attention** 和 **Feed Forward** 两个子层组成，每个子层后面都跟着 **Add & Norm**（残差连接 + 层归一化）。

**这一节只讲第一个子层：Multi-Head Self-Attention。**

### 核心要点

**1. Q/K/V 从哪来？**

在 Encoder 的 Self-Attention 里，**Q（查询）、K（键）、V（值）都来自前一层的输出**。也就是说，每个位置既当“提问者”，又当“被查询者”，还当“信息提供者”。

**2. 有没有 Mask？**

**没有因果 mask**。Encoder 是**双向**的——序列中所有位置互相可见。每个 token 都能“看到”整个输入序列的任意位置。

> 注意：批处理时通常仍有 padding/attention mask，那是为了处理变长序列，跟“因果 mask”是两回事。

**3. 作用是什么？**

让每个位置**聚合整个输入序列的上下文信息**。比如在翻译任务中，编码“bank”这个词时，Self-Attention 会让它同时参考“river”或“money”等上下文词，从而正确理解语义。

### 为什么这对 vLLM 部署重要？

vLLM 主要面向 **Decoder-Only** 架构（如 GPT 系列），但理解 Encoder 的 Self-Attention 是理解 Decoder 的 **Masked Self-Attention** 和 **Cross-Attention** 的前提。三者关系：

| 注意力类型 | Q来源 | K/V来源 | Mask |
|---|---|---|---|
| Encoder Self-Attn | 前一层 | 前一层 | 无（双向） |
| Decoder Masked Self-Attn |

## 块 12

**目标：vLLM部署，动手实操型**

FFN是Encoder里每个位置都独立做的一次“特征加工”，公式就一行：

`FFN(x) = max(0, xW1 + b1)W2 + b2`

**拆解动作**：
1. 第一层：`xW1 + b1`，把512维向量升到2048维（`d_model=512 → d_ff=2048`）。
2. 中间过ReLU：`max(0, ...)`，把负值清零，引入非线性。
3. 第二层：`...W2 + b2`，把2048维压回512维，恢复原始维度。

**为什么这样设计**：先拓宽再压缩，让每个位置能独立“思考”更丰富的特征组合，再整合回标准尺寸，方便后续层继续处理。

**残差连接 + LayerNorm**：FFN输出后不是直接用，而是先加回输入（`x + FFN(x)`），再做归一化（`LayerNorm`），防止

## 块 13

### 4.1 Masked Multi-Head Self-Attention：让 Decoder 只看过去

你的目标是 vLLM 部署，核心是理解 Decoder 如何一步步生成。这里的关键就是**因果掩码**。

**它做什么？**
Decoder 生成第 i 个词时，只能看它**之前**的词。Q、K、V 都来自 Decoder 前一层输出，但计算注意力分数时，把**未来位置**的分数设为 `-∞`：

- 当 `j ≤ i`（当前位置或之前）：正常计算 `q_i · k_j / √d_k`
- 当 `j > i`（未来位置）：分数设为 `-∞`

经过 Softmax 后，`-∞` 的权重变成 0，模型就“看不见”未来了。

**直观理解（4 个位置）**：
```
位置:    1    2    3    4
  1   [ ✓    ✗    ✗    ✗ ]  →

## 块 14

**Multi-Head Cross-Attention：让Decoder“看见”输入**

你的目标是vLLM部署，核心是让模型跑起来。理解Cross-Attention，就是理解模型如何“读取”输入。

**它做什么？**
- **Q（查询）**：来自Decoder的**前一层**输出，代表“我目前生成到哪了”。
- **K/V（键/值）**：都来自**Encoder的输出**（记作 **z**），代表“原始输入里有什么信息”。
- **无因果mask**：Decoder能看到**完整输入序列**（批处理时通常仍有padding mask，用于忽略填充位）。
- **作用**：它是Encoder和Decoder之间的“**桥梁**”，让Decoder能关注输入序列。

**关键

## 块 15

## 四、Decoder 的层次结构（①）——4.3 Feed-Forward Network

### 核心要点

Decoder 中的 **FFN（前馈网络）** 与 Encoder 中的 FFN **结构完全相同**，没有额外差异。你不需要为 Decoder 单独学习一套新的 FFN 规则——它做的就是同样的事：对每个位置的表示做一次非线性变换，增强模型的表达能力。

### Decoder 完整数据流（①）

把 FFN 放回 Decoder 的整体流程中看，它位于**第二个多头注意力（Cross-Attention）之后**，是整个 Decoder 层的最后一步。完整流程如下：

$$
\begin{aligned}
y_1 &= \text{Embedding}(output) + \text{PositionalEncoding} \\
y_2 &= \text{LayerNorm}(y_1 + \text{MultiHeadAttention}(Q=y_1, K=y_1, V=y_1, \text{mask}=\text{causal})) \\
y_3 &= \text{LayerNorm}(y_2 + \text{MultiHeadAttention}(Q=y_2, K=\mathbf{z}, V=\mathbf{z})) \\
y_4 &= \text{LayerNorm}(y_3 + \text{FFN}(y_3)) \\
&\text{...重复6次...} \\
\text{logits} &= \text{Linear}(y_{final}) \\
\text{probs} &= \text{Softmax}(\text{logits})
\end{aligned}
$$

**关键观察**：FFN 的输入是 `y_3`（Cross-Attention 的输出 + 残差 + LayerNorm），输出 `y_4` 再经过残差连接和 LayerNorm，就完成了 Decoder 一层的全部工作。**整个 Decoder 层重复 6 次**，最后经过 Linear + Softmax 得到输出概率。

### 三种 Attention 的系统对比（①）

三种 Attention 的公式都是 $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$，**区别只在 Q/K/V 的来源不同**，导致作用完全不同：

| Attention 类型 | 所在位置 | Q 来源 | K 来源 | V 来源 | Mask | 作用 |
|---|---|---|---|---|---|---|
| **Encoder Self-Attention** | Encoder 每层 | Encoder 前一层 | Encoder 前一层 | Encoder 前一层 | 无因果 mask（批处理时通常有 padding mask） | 编码输入序列的上下文 |
| **Decoder Masked Self-Attention** | Decoder 每层 | Decoder 前一层 | Decoder 前一层 | Decoder 前一层 | 因果 mask | 自回归生成，防止"看到未来" |
| **Decoder Cross-Attention** | Decoder 每层 | Decoder 前一层 | Encoder 输出 | Encoder 输出 | 无因果 mask | 从编码后的输入中提取信息 |

## 块 16

**目标**：用代码看清 T5 的 Decoder 里 FFN 长什么样，为 vLLM 部署时理解层结构打底。

**动手跑**（CPU 即可，内存 ≥4GB，首次需联网下载模型，实际大小约 240MB 以上，可能因版本而异）：
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 模型许可证：Apache 2.0
# 注意：T5-small FP32 推理约需 1GB 内存，请确保内存充足
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

print(f"Encoder 层数: {model.config.num_layers}")
print(f"Decoder 层数: {model.config.num_decoder_layers}")
print(model)
```

**看什么**：运行后，在打印出的结构里找到 `decoder` 部分，每个 Decoder 层（`T5Block`）内部都包含一个 `T5LayerFF`——这就是 FFN。它夹在自注意力和交叉注意力之后，负责对每个位置的表示做逐点非线性变换

## 块 17

## 6.1 观察 Encoder-Decoder 的前向传播

既然你的目标是 vLLM 部署，理解 Encoder-Decoder 的前向传播是打好基础的关键一步。我们直接动手，用代码观察模型内部发生了什么。

**准备输入**：我们给模型一句英文，让它翻译成德语。`input_text` 就是源语言句子，`tokenizer` 会把它转成模型能读的数字 ID。

**构造 Decoder 输入**：注意这里——翻译任务中，Decoder 不是空着开始的。在训练时，Decoder 接收目标语言的前缀（teacher forcing），我们给它 `"Das Haus"`（德语“房子”的开头），模型会接着预测后面的词。在推理时，Decoder 初始输入为起始符（如 `<bos>`），然后逐步自回归生成。

**前向传播**：用 `torch.no_grad()` 关闭梯度计算（推理时不需要），然后调用模型。注意我们传入了 `output_attentions=True` 和 `output_hidden_states=True`，这样才能拿到中间结果来观察。

**观察输出**：代码打印了四个关键信息：
- **Encoder 最后一层隐藏状态**：形状是 `[batch_size, encoder_len, hidden_dim]`，这是 Encoder 对源句子的完整理解。
- **Decoder 最后一层隐藏状态**：形状类似，但长度是 `decoder_len`，代表 Decoder 当前已生成部分的理解。
- **Cross-Attention 层数**：有多少层 Transformer，就有多少层 Cross-Attention。
- **每层 Cross-Attention 形状**：最后两个维度是 `[decoder_len, encoder_len]`——这表示 Decoder 的**每一个**位置，都在计算对 Encoder **所有**位置的注意力权重。这就是 Cross-Attention 的核心：Decoder 生成每个词时，都会“回看”源句子的全部信息。

> **关键观察**：`cross_attentions[0].shape` 的最后两个维度通常是 `[decoder_len, encoder_len]`，说明 Decoder 的每个位置都在关注 Encoder 的所有位置。这正是 Cross-Attention 的作用。上述输出会随 transformers 版本和模型版本变化，实际以运行结果为准。

> **本地运行配置要求**：此代码纯 CPU 运行即可，内存 ≥ 4GB。

---

## 七、架构演化：为什么后续模型“拆分”了 Encoder-Decoder？

## 块 18

**目标对齐**：你要用 vLLM 部署模型，核心是搞清“该选哪种架构”。本块直接对比 BERT 和 GPT，帮你建立判断依据。

**7.1 Encoder-only：BERT（2018）**
- **架构**：只保留 Encoder，去掉 Decoder。
- **训练方式**：Masked Language Model（MLM）——随机遮住词，让模型根据上下文猜。
- **优势**：能同时看左右两侧上下文（双向），适合理解类任务：分类、问答、命名实体识别（NER）。
- **劣势**：不适合生成任务（因为它不擅长从左到右逐词产出）。

**7.2 Decoder-only：GPT（2018-2024）**
- **架构**：只保留 Decoder，同时去掉 Encoder 和 Cross-Attention。
- **训练方式**：Causal Language Model（Next Token Prediction）——预测下一个词。

## 块 19

### 7.3 Encoder-Decoder 仍有广泛应用（①）

你可能会问：既然主流是 Decoder-only，那 Encoder-Decoder 是不是该淘汰了？**不是**。在特定任务里，它依然能打，甚至效果更好。结合你部署 vLLM 的目标，这些模型你迟早会碰到：

- **机器翻译**：`Helsinki-NLP/opus-mt-en-zh`（专门用于翻译的模型）
- **语音识别**：`facebook/wav2vec2-base`（专门用于语音识别的模型）
- **视觉语言**：`openai/clip-vit-base-patch32`（用于图像与文本的跨模态理解）

这些模型在 HuggingFace 生态里都能直接加载。不过要注意，vLLM 对模型的支持情况因模型而异，部署前建议查阅 vLLM 官方文档确认兼容性。另外，不同模型的许可证类型不同，商用前需逐一确认授权条款；显存需求也因模型参数量和精度而异，部署时请预留足够显存。

**为什么 GPT 去掉 Encoder 还能工作？** 答案很简洁：**GPT 使用自回归生成，通过因果掩码确保每个位置只能看到之前的位置，从而无需 Encoder**。这个机制在下一章会深入拆解，现在先记住这个结论。

---

### 八、常见问题与解决（①）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 混淆三种 Attention 的 Q/K/V 来源 | 没有系统对比 | 记住核心区别：Self-Attention 的 Q/K/V 同源，Cross-Attention 的 K/V 来自 Encoder |
| 不理解为什么 Decoder 需要 Mask | 忽略了自回归性质 | 生成任务中，模型不能"偷看"未来的正确标签 |
| 以为 Encoder 输出只送到 Decoder 第一层 | 对数据流理解不深入 | Encoder 输出送到**每一层** Decoder 的 Cross-Attention |
| Cross-Attention 的 K 和 V 能否只用其一 | 对 Attention 机制理解不深入 | K 用于计算相似度，V 用于加权求和，缺一不可 |

---

### 九、总结与后续学习路线（①）

到这里，你已经掌握了 Transformer Encoder-Decoder 的核心：

- ✅ Enc

## 块 20

好的，我们直接切入正题。你的目标是搞定 vLLM 部署，那理解模型架构就是打地基。这一节，我们只解决一个问题：**为什么 GPT、Llama、Qwen 这些模型，可以不要 Encoder，只靠 Decoder 就能工作？**

想象一下，你不再需要把“翻译任务”的源语言和目标语言分开处理。Decoder-Only 的做法更简单粗暴：**把所有的输入和输出，全部拼成一个长长的 token 序列**。比如，你的 prompt 是“中国的首都是”，模型要生成的“北京”就接在后面，变成“中国的首都是北京”。

这里的关键在于“因果可见性”。在这个长序列里，**从左到右**，每个 token 都只能“看到”它前面的 token。对于 prompt 部分，模型能看到完整的 prompt；对于要生成的部分，模型能看到它之前的所有 prompt 和已经生成的 token。这就是它“理解”输入的方式——不是像 Encoder 那样双向地看，而是像人读书一样，逐字往后读，边读边预测下一个字。

这种设计的最大好处是**统一**。训练目标和模型结构都统一了，这让模型在扩大规模（比如从 7B 到 70B 参数）时，扩展性更好，工程实现也更简单。这也是 GPT、Llama、Qwen 这些主流大模型普遍采用它的原因。

但是，这并不意味着 Encoder-Decoder 就过时了。在**翻译、摘要**这类任务里，Encoder 可以先完整地“读”完整个源文本，再让 Decoder 去生成。这种“先理解，后生成”的方式，在需要精确控制输入信息的场景下，依然有它的优势。

所以，架构选择不是看输入输出是不是同一种语言，而是看你的**训练目标、任务类型和工程约束**。对于通用的大规模语言建模，Decoder-Only 是当前更高效的选择。

**本块学习检查**

为了确保你

## 块 21

## 1.2 统一框架：Causal Language Modeling（①）

你部署 vLLM 时，核心要抓的就是这个“统一框架”——**因果语言建模（CLM）**，也叫 Next Token Prediction。

**一句话理解**：给定序列中前 $t$ 个 token，预测第 $t+1$ 个 token。数学表达为：

$$P(x_1, x_2, ..., x_n) = \prod_{t=1}^{n} P(x_t | x_1, ..., x_{t-1})$$

**为什么这个设计对 vLLM 部署至关重要？** 看四个优势：

| 优势 | 说明 | 对 vLLM 的意义 |
|---|---|---|
| **架构简化** | 没有 Encoder 和 Cross-Attention，参数量全在 Decoder，易扩展到超大规模 | 模型结构单一，部署时无需处理双塔逻辑 |
| **数据要求低** | 不需要成对的输入-输出数据，任何纯文本都可训练 | 预训练语料丰富，模型能力通用 |
| **统一接口** | 问答、摘要、代码等任务可转化为“续写”问题，通过 prompt 切换 |

## 块 22

**本块学习检查**：读完下面内容后，请用一句话回答：GPT-1 相比传统 Transformer，在架构上去掉了哪两个部分？

---

2018年，OpenAI发布GPT-1。它的核心创新不在某个模块，而在**架构取舍**：

- **只留Decoder**：砍掉Encoder和Cross-Attention
- **训练目标**：预测下一个词（因果语言建模）
- **两阶段范式**：先在海量无标注文本上预训练，再针对具体任务微调

这证明了：一个Decoder-Only模型，微调后能适配分类、问答等多种任务。对你做vLLM部署而言，理解“Decoder-Only”是后续推理优化的地基——GPT-1正是这个范式的起点

## 块 23

**本块学习检查**：读完本节后，请合上资料，用一句话回答：GPT-2 展示的“zero-shot 能力”指的是什么？如果你能说出“不微调参数，只靠改 prompt 就能做翻译、摘要等任务”，说明你抓住了核心。

---

## 2.2 GPT-2：规模即能力

2019 年，GPT-2 没改核心架构，只把**规模**推高：1.5B 参数、40GB 训练数据。结果出现一个关键现象——**zero-shot 能力**：不微调参数，只换不同的输入 prompt，模型就能执行翻译、摘要、问答等任务。这说明 Decoder-Only 的“续写”本身就藏着通用性。

## 2.3 GPT-3：Prompt 即接口

2020 年，GPT-3 把参数扩到 175B，把 **few-shot / in-context learning** 推向主流：

- **Few-shot**：在 prompt 里给几个示例，模型就模仿示例的格式和风格
- **In-context learning**：模型从 prompt 上下文

## 块 24

**Llama 的"做减法 + 做强化"——为部署而生的架构选择**

你学 vLLM 部署，先抓住一个关键认知：**Llama 没有推翻 GPT 的 Decoder-Only 路线，而是对内部组件做了三处精准改造**。这三处改动直接影响推理性能和部署体验，值得你现在就建立印象。

| 改进项 | 原始 Transformer | Llama 的做法 | 作用 |
|---|---|---|---|
| **归一化** | Post-LayerNorm（子层后） | **Pre-RMSNorm**（子层输入前） | 训练更稳定，简化计算，提升训练稳定性 |
| **激活函数** | ReLU | **SwiGLU** | 表达能力更强，门控机制过滤信息 |
| **位置编码** | 绝对正弦 | **

## 块 25

**Llama 三代演进：从“能训”到“好用”再到“全能”**

你的目标是 vLLM 部署，所以看 Llama 演进时，重点盯住**架构和推理效率**的变化——这直接决定你部署时的显存占用和吞吐量。

**Llama1**：证明了公开数据也能训出强模型。经典 Decoder-Only 架构，上下文较短，以 base 模型为主。部署时中规中矩，适合入门。

**Llama2**：把开源大模型往可用产品推进。上下文常见 4K，Chat 对齐更成熟。部署时开始强调实用性，是当时部署的主流选择。

**Llama3**：把多语言、代码、工具和长上下文能力进一步拉上去。部分版本上下文 128K+，工具使用和推理能力更

## 块 26

## 4.1 Qwen3 的架构亮点（②）

既然你的目标是 **vLLM 部署**，那读 Qwen3 架构时，重点盯住“哪些设计会影响推理时的显存和速度”。下面按这个思路拆解。

### （1）Dense 和 MoE：先分清你要部署哪一类

Qwen3 同时提供两种架构：

- **Dense（稠密）模型**：每一层 FFN 只有一套权重，每个 token 前向时全部参与计算。代表：Qwen3-8B、14B、32B。
- **MoE（混合专家）模型**：FFN 换成多个独立专家，通过门控网络，每个 token 只激活少数专家。代表：Qwen3-30B-A3B（总参 30B，每 token 激活约 3.3B）、235B-A22B（总参 235B，每 token 激活约 22B）。

**部署含义**：MoE 模型“仓库里装得下大参数，但每个 token 实际计算量只和激活支路相当”。也就是说，**显存要按总参数量预留，算力消耗按激活参数量估算**。用 vLLM 部署时，这两者分别对应模型加载显存和推理吞吐。

### （2）Thinking / Non-Thinking：同一个模型，两种模式

Qwen3 把“深度推理”和“快速响应”统一进**同一套权重**，通过 chat template 中的开关（如 `/think`、

## 块 27

## Qwen3 的三阶段预训练：先广度、再深度、最后长度

Qwen3 的预训练不是“一把梭”，而是分阶段推进，这个设计直接服务于不同的能力目标。作为要上手 vLLM 部署的你，理解这个演进逻辑，能帮你预判模型在不同任务下的行为差异。

**第一阶段：General（广度）**  
用大规模 tokens，序列长度 4,096，进行广谱学习。这个阶段让模型掌握语言与世界知识，相当于“通识教育”。

**第二阶段：Reasoning（深度）**  
用高质量数据（STEM、代码、推理比例升高），序列长度仍为 4,096。这个阶段强化推理相关分布，相当于“专业训练”。

**第三阶段：Long Context（长度）**  
用数百 B tokens，序列长度拉到 32,768。这个阶段让优化目标直接覆盖万级 token 的因果依赖——对长文档、长对话场景至关重要。

**核心思想：先广度、再深度、最后长度。**

> **部署视角**：你在 vLLM 里配置 Qwen3 时，长上下文阶段意味着 KV Cache 占用会显著增加。部署长上下文任务前，务必确认显存预算和 vLLM 的 `--max-model-len` 参数设置。

---

### 三条主线对比速查

| 维度 | GPT 系列 | Llama 系列 | Qwen 系列 |
|---|---|---|---|
| **核心路线** | Decoder-only + 规模化 | Decoder-only + 组件优化标准化 | Decoder-only，部分模型采用 MoE，并有专门的多模态版本 |
| **归一化** | LayerNorm → 改进 | **Pre-RMSNorm** | **Pre-RMSNorm** |
| **位置编码** | 绝对位置编码 | **RoPE** | **RoPE** |
| **激活函数** | GeLU / 其他 | **Swi

## 块 28

## 6.1 观察 GPT-2 的“原始”Decoder-Only 结构

既然你的目标是 vLLM 部署，理解 GPT-2 的原始结构是第一步——它是后续所有优化（如 KV Cache、PagedAttention）的“地基”。我们直接上手看代码。

**运行这段代码，观察输出：**

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
config = model.config

print(f"层数: {config.n_layer}")
print(f"隐藏维度: {config.n_embd}")
print(f"注意力头数: {config.n_head}")
print(f"FFN 中间维: {config.n_embd * 4}")
print(f"位置编码: 绝对位置编码 (n_positions={config.n_positions})")
print(f"激活函数: {config.activation_function}")
```

**预期输出：**

```
层数: 12
隐藏维度: 768
注意力头数: 12
FFN 中间维: 3072
位置编码: 绝对位置编码 (n_positions=1024)
激活函数: gelu
```

**关键观察：** GPT-2 用的是**绝对位置编码**、**GeLU 激活**和**标准 LayerNorm**，这就是“经典”Decoder-Only

**模型许可证：** GPT-2 采用 MIT 许可证，可自由商用，无月活限制。

**显存要求：** 本模型 FP16

## 块 29

## 6.2 观察 Qwen2.5 的现代 Decoder-Only 结构

既然你的目标是 **vLLM 部署**，理解模型结构是部署前的关键一步——不同架构直接决定了推理时的显存占用和加速策略。我们直接动手，用代码“解剖”一个真实的现代 Decoder-Only 模型。

### 运行代码，查看配置

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
print(f"QKV bias: {config.attention_bias}")
```

**输出**（以实际配置为准）：

```
层数: 24
隐藏维度: 896
Q 头数: 14
KV 头数: 2          ← GQA：14 个 Q 头共享 2 组 KV
FFN 中间维: 4864
位置编码: RoPE (rope_theta=1000000.0)  ← Qwen2.5 默认值
归一化: RMSNorm (rms_norm_eps=1e-06)
QKV bias: False     ← Attention 投影层不使用 bias
```

### 三个关键观察（部署时直接影响显存与速度）

1. **GQA（分组查询注意力）**：`num_key_value_heads(2) < num_attention_heads(14)` 就是 GQA 的标志。14 个 Q 头共享 2 组 KV，这意味着 KV cache

## 块 30

## 6.3 动手算一算：三种 Attention 的 KV Cache 体积

既然你的目标是 vLLM 部署，那 **KV Cache 的显存占用**就是你上线前必须心里有数的一件事——它直接决定了你能塞下多长的序列、多大的 batch。下面这段代码，就是用来量化对比不同注意力机制下 KV Cache 体积的。

```python
import torch

def kv_cache_size(batch, seq_len, num_layers, num_kv_heads, head_dim, dtype_bytes=2):
    """计算 KV Cache 显存占用（字节）"""
    # 每层有 K 和 V 两个张量
    # 形状: [batch, num_kv_heads, seq_len, head_dim]
    per_layer = 2 * batch * num_kv_heads * seq_len * head_dim * dtype_bytes
    return num_layers * per_layer

# GPT-2（标准 MHA，num_kv_heads = num_attention_heads = 12）
gpt2_kv = kv_cache_size(1, 2048, 12, 12, 64)
print(f"GPT-2 KV Cache: {gpt2_kv / 1024 / 1024:.2f} MB")

# Qwen2.5-0.5B（GQA，num_kv_heads = 2, head_dim = 128）
qwen_kv = kv_cache_size(1, 2048, 24, 2, 128)
print(f"Qwen2.5-0.5B KV Cache: {qwen_kv / 1024 / 1024:.2f} MB")

# 对比：如果 Qwen 也用 MHA（假设 14 个 KV 头）
qwen_mha_kv = kv_cache_size(1, 2048, 24, 14, 128)
print(f"Qwen2.5-0.5B (假设 MHA) KV Cache: {qwen_mha_kv / 1024 / 1024:.2f} MB")
```

**输出**：
```
GPT-2 KV Cache: 6.00 MB
Qwen2.5-0.5B KV Cache: 2.34 MB
Qwen2.5-0.5B (假设 MHA) KV Cache

## 块 31

## 6.3 对比三种 Attention 的 KV Cache 体积

你已经知道 Decoder-Only 架构用统一因果语言建模目标，把输入输出组织成同一序列。现在直接动手，用 `transformers` 库观察 GPT-2 与 Qwen2.5 的配置差异，理解 GQA 如何节省 KV Cache。

**三种 Attention 的 KV Cache 差异**：

- **MHA（多头注意力）**：每个头都存独立的 K 和 V，KV Cache 体积最大
- **MQA（多查询注意力）**：所有头共享一组 K 和 V，KV Cache 最小但质量可能下降
- **GQA（分组查询注意力）**：将头分组，组内共享 K 和 V，是 MHA 和 MQA 的折中

**实操验证**：加载 `Qwen2.5` 的 `config.json`，你会看到 `num_attention_heads` 和 `num_key_value_heads` 两个参数。当 `num_key_value_heads` 小于 `num_attention_heads`

## 测评题目

### Q-KB08-ST0800-001
用 `AutoModel.from_pretrained()` 加载模型时，模型权重通常来自名称对应的预训练________。

### Q-KB08-ST0800-003
加载 GPT-2 等预训练模型进行前向实验，最直接的目的是什么？

### Q-KB0801-002
在Transformer中，**编码器**（Encoder）通常用于生成上下文相关的文本表示（如BERT），而**解码器**（Decoder）通常用于自回归的文本生成任务（如GPT系列）。

### Q-KB0801-003
以下哪个模型是典型的 **Encoder-only** 架构？

### Q-KB08-ST0802-002
GPT、Llama、Qwen 等自回归语言模型通常使用________架构。

### Q-KB08-ST0802-003
Decoder-Only 注意力掩码的核心作用是？

### Q-KB08-ST0804-002
SwiGLU 是一种带门控机制的________函数变体。

### Q-KB0805-001
为什么Transformer需要引入位置编码（Positional Encoding）？
