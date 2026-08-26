# 学习包

## 块 1

**本块学习检查**  
在开始前，请先完成一个最小回顾：在已激活的 `pytorch` conda 环境中，确认 `transformers` 库是否已安装（若未安装，执行 `pip install transformers`）。然后运行以下代码，加载 GPT-2 并执行一次前向传播：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs, output_attentions=True)
print(outputs.logits.shape)
```

能解释 `from_pretrained()` 和 `AutoTokenizer` 的用法，并看到 logits 输出形状，再继续本块。

---

## 一、安装与验证 transformers

### 1.1 快速安装

如果你已完成第七板块的学习，`transformers` 库应该已经安装。如果尚未安装，在已有的 `pytorch` conda 环境中执行：

```bash
conda activate pytorch
pip install transformers
```

若需要同时安装 PyTorch 后端依赖，可使用 `pip install 'transformers[torch]'`。

**本地运行配置要求**：`transformers` 库本身体积通常在数 MB 至十余 MB 之间，安装过程需要联网。CPU 即可执行，内存建议 ≥ 4GB。

**模型许可证**：本块加载的 GPT-2 模型采用 MIT 许可证，可自由使用。

**显存要求**：GPT-2 模型（约 124M 参数）在 FP32 下推理约需 1GB 显存，CPU 运行则无显存要求。

安装完成后，建议立即验证：运行上述最小示例，确认模型能成功加载并输出 logits。这一步能确保环境就绪，为后续观察 GPT-2 的 hidden_states 做准备。

## 块 2

### 1.2 动手验证：让 transformers 跑起来

安装完成后，我们立刻用一个小实验来确认环境没问题。打开你的 Python 环境，输入下面这段代码：

```python
from transformers import pipeline
result = pipeline('sentiment-analysis')('we love you')
print(result)
```

这段代码会加载一个预训练的情感分析模型，然后判断 "we love you" 这句话的情感倾向。运行后，你会看到类似这样的输出（具体内容取决于你的 transformers 和模型版本）：

```python
[{'label': 'POSITIVE', 'score': 0.9998...}]
```

看到 `POSITIVE` 和接近 1 的分数，就说明安装成功了！如果没看到，别急，先看下面的常见问题。

### 1.3 常见问题速查表

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `Failed building wheel for tokenizers` | 缺少 Rust 编译工具链 | 用 `conda install -c huggingface transformers` 安装，或尝试安装特定版本 |
| `urllib3.exceptions.SSLError` | `urllib3` 版本过高 | 执行 `pip install urllib3==1.25.11` |
| 无法连接 HuggingFace Hub | 网络限制 | 参考第四节"离线配置"或设置镜像源 |

### 二、加载 GPT-2 模型

验证通过后，我们来加载一个小型模型做实验。GPT-2 是 OpenAI 发布的因果语言模型，有多个尺寸：

| 模型 checkpoint | 参数量 | 大小（约） | 运行要求 |
|---|---|---|---|
| `gpt2` | 124M | ~500MB | CPU / 4GB 内存通常即可 |
| `openai-community/gpt2-medium` | 355M | ~1.5GB | # 注意：加载模型时请确保显存充足，例如 Llama-

## 块 3

### 2.1 使用 pipeline 快速体验

`pipeline` 是 transformers 库的高级 API，让你用一行代码就能加载 GPT-2 并体验文本生成。这对你理解 Transformer 的 Decoder-Only 架构很有帮助——模型接收输入后，逐个预测下一个 token，直到达到设定的长度。

**操作步骤：**

```python
from transformers import pipeline

# 创建文本生成 pipeline，device=-1 表示使用 CPU
# 模型许可证：MIT License
# 注意：GPT-2 模型 FP16 推理需要至少 4GB 显存（CPU 模式无显存要求）
generator = pipeline("text-generation", model="openai-community/gpt2", device=-1)

result = generator("Hello, I'm a language model", max_new_tokens=20)
print(result)
```

**运行前须知：**
- 首次运行会自动从 HuggingFace Hub 下载 GPT-2 模型文件（约 550MB），**需要联网**。
- 下载后缓存到本地，后续运行无需重复下载。
- 内存建议 ≥ 4GB，CPU

## 块 4

**个人化教学正文**

你正处在理解 Transformer 架构的路上，这次我们用“动手拆解”的方式，通过底层 API 手动加载 GPT-2，亲眼看看它的核心配置。这比一键加载更能看清内部结构。

**加载分词器与模型**

```python
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# 模型许可证：MIT License（GPT-2 开源许可）
# 注意：GPT-2 模型 FP16 推理约需 2GB 显存，训练或长序列需更多
# GPT-2 默认没有 pad_token，手动设为 eos_token，避免 batch 处理报错
tokenizer.pad_token = tokenizer.eos_token
```

**查看模型配置**

```python
print(f"词表大小: {model.config.vocab_size}")
print(f"最大序列长度: {model.config.n_positions}")
print(f"隐藏层维度: {model.config.n_embd}")
print(f"层数: {model.config.n_layer}")
print(f"注意力头数: {model.config.n_head}")
```

**这些数字对应 Transformer 的核心超参数**：`n_embd` 是隐藏层维度，`n_layer` 是 Transformer Block 的堆叠层数，`n_head` 是多头注意力中的头数。后续课程会逐一拆解这些组件的内部机制。

---

**本块学习检查**

运行上述代码后，记录你看到的 `n_layer` 和 `n_head` 数值，并思考：这两个参数分别对应 Transformer 架构中的哪两个核心组件？

## 块 5

**动手跑通 GPT-2 前向传播**

你已有 PyTorch 基础，这里直接上手。我们分三步观察 GPT-2 内部输出。

**第一步：准备输入**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型许可证：GPT-2 采用 MIT 许可证，可自由商用
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")

print(f"input_ids 形状: {inputs['input_ids'].shape}")
```

`tokenizer` 把文本转成数字 ID，`return_tensors="pt"` 表示返回 PyTorch 张量。打印出的形状是 `[1, 6]`——1 个句子，6 个 token。

**第二步：执行前向传播**

```python
# 注意：GPT-2 模型 FP16 推理至少需要 2GB 显存
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,   # 返回每层的隐藏状态
        output_attentions=True       # 返回每层的注意力权重
    )
```

`torch.no_grad()` 关闭梯度计算，省内存。两个 `output_*` 参数让模型额外返回中间结果。

**第三步：观察输出**

```python
print(f"\nlogits 形状: {outputs.logits.shape}")
print(f"hidden

## 块 6

我们直接上手，用代码“看”GPT-2 是怎么预测下一个词的。

**第一步：预测下一个词**

先加载模型和分词器，并让模型处理一段输入文本。注意，为了让模型返回注意力权重，需要在调用时加上 `output_attentions=True`。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型许可证：GPT-2 采用 MIT License，可自由使用（商用需注明版权与许可声明）。
# 注意：GPT-2 模型 FP16 推理约需 3GB 显存，请确保环境满足要求。
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

inputs = tokenizer('Hello', return_tensors='pt')
outputs = model(**inputs, output_attentions=True)
```

模型处理完你输入的文本后，`outputs.logits` 里存着每个位置对所有候选词的打分。我们只关心**最后一个位置**——因为我们要预测的是“下一个”词。

```python
# 取最后一个位置的 logits，预测下一个词
next_token_logits = outputs.logits[0, -1, :]
next_token_id = torch.argmax(next_token_logits).item()
next_token = tokenizer.decode(next_token_id)

print(f"模型预测的下一个词: '{next_token}'")
```

`argmax` 就是挑分数最高的那个词，`decode` 把词 ID 变回文字。

**第二步：验证“看不见未来”**

GPT-2 是 Decoder-Only 模型，它有个铁律：**每个 token 只能看自己和它前面的 token**，绝不能偷看后面的。这个机制叫“因果注意力”（causal attention）。我们直接检查注意力权重矩阵，眼见为实：

```python
# 取第一层的注意力权重
# 形状: [batch=1, n_head=12, seq_len, seq_len]
attn = outputs.attentions[0]

# 查看第一个注意力头的权重矩阵
attn

## 块 7

## 3.4 用 labels 算损失：让模型自己当“考官”

你已经知道 GPT-2 是预测“下一个词”的。训练时，**labels 就是 input_ids 本身**——因为模型要预测的是“当前位置的下一个词”，所以目标就是输入序列向后移一位。好消息是，**transformers 内部自动处理这个偏移**，你完全不用手动操作。

直接看代码，训练模式下这样写：

```python
# 训练模式：传入 labels 会自动计算语言建模损失
outputs = model(**inputs, labels=inputs["input_ids"])

print(f"Loss: {outputs.loss.item():.4f}")
```

**关键点**：`GPT2LMHeadModel` 内部已经把 labels 做了**左移一位**处理——预测第 i 个词时，目标其实是第 i+1 个

## 块 8

### 模型缓存目录：你的模型都藏在这里

每次从 HuggingFace Hub 下载模型，它默认会存到电脑的缓存文件夹里。不同系统位置不一样：

- **Linux/macOS**：`~/.cache/huggingface/hub`
- **Windows**：`C:\Users\你的用户名\.cache\huggingface\hub`

想改存放位置？用环境变量指定新路径，推荐第一种：

```bash
# 方法 1：设置 HF_HOME（推荐）
export HF_HOME="/path/to/your/cache"

# 方法 2：设置 TRANSFORMERS_CACHE
export TRANSFORMERS_CACHE="/path/to/your/cache"
```

**动手试试**：打开终端，输入 `echo $HF_HOME`

## 块 9

## 4.2 离线运行：没有网络也能跑 GPT-2

你在防火墙或内网环境里，没法直接访问 Hugging Face 下载模型，这时需要开启离线模式。做法很简单，设置两个环境变量：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

第一个告诉 transformers 库不要尝试联网，第二个告诉 datasets 库同样离线。设置好后，加载模型时直接指向本地缓存目录：

```python
tokenizer = AutoTokenizer.from_pretrained("/local/path/to/gpt2", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("/local/path/to/gpt2", local_files_only=True)
```

注意这里的路径是你自己机器上的实际路径，不是模型名称。**关键一步**：你需要提前在有网环境中把模型下载好并保存到本地。方法是先用 `from_pretrained()` 正常下载，再用 `save_pretrained("./local/path")` 把模型和分词器保存到指定目录，之后就能离线加载了。在 `from_pretrained()` 中设置 `local_files_only=True` 可确保只从本地加载，不触发任何网络请求。

**模型许可证说明**：GPT-2 采用 MIT 许可证，允许自由使用、修改和商用，无用户规模限制。

## 常见问题速查

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `OSError: Can't load tokenizer` | 本地路径缺少必要的配置文件 | 确保本地目录包含完整的分词器文件 |
| `RuntimeError: Expected all tensors on same device` | 模型和数据不在同一设备 | 确保 `model.to(device)` 和数据 `.to(device)` 一致 |
| GPT-2 padding 报错 | 默认没有 `pad_token` | 手动设置 `tokenizer.pad_token = tokenizer.eos_token` |
| 下载速度极慢 | 网络问题 | 配置镜像源（如 `HF_ENDPOINT=https://hf-mirror.com`） |
| 显存不足（OOM） | 加载了过大的模型 | 换用 `openai-community/gpt2`，或尝试 `load_in_8bit=True` |

## 总结

到这里，你已经掌握了 transformers 库的安装验证、用 `pipeline` 和 `AutoModel` 两种方式加载 GPT-2、执行完整前向传播、理解 `logits`、`hidden_states`、`attentions` 的含义、观察因果注意力掩码效果、用 `labels` 计算语言建模损失，以及模型缓存和离线运行方法。

**本块学习检查**：请你在本地完成一次完整的离线加载流程——先在有网环境下载 GPT-2 并保存到本地目录，然后设置 `TRANSFORMERS_OFFLINE` 和 `HF_DATASETS_OFFLINE` 环境变量，最后用 `local_files_only=True` 从本地路径加载模型和分词

## 块 10

## 三、Encoder 的层次结构（①）——Multi-Head Self-Attention

### 3.1 多头自注意力：让模型“多角度”看句子

你已经通过 GPT-2 实验见过 Self-Attention 的核心：每个 token 都会“关注”序列中其他 token，并据此更新自己的表示。但**单个注意力**有一个局限——它只能从一种“视角”去计算 token 之间的关系。

**Multi-Head Self-Attention（多头自注意力）** 解决了这个问题。它的思路很直观：**与其只做一次注意力计算，不如并行做多次**，每次用不同的“头”去关注输入序列的不同方面，最后再把所有头的结果拼起来。

打个比方：读一句话时，一个“头”可能关注语法关系（谁修饰谁），另一个“头”可能关注指代关系（“它”指什么），还有一个“头”可能关注位置远近。多个头各司其职，模型就能捕捉更丰富的语义信息。

在 Encoder 的每一层中，Multi-Head Self-Attention 是第一个子层。它的输入是上一层的输出（第一层时就是输入序列的嵌入表示）。由于 Encoder 是**双向**的，每个 token 在计算注意力时，可以同时看到**完整的输入序列**——包括它自己、它前面的 token 和它后面的 token。

**关键点回顾**：
- Encoder 的 Self-Attention 是**双向**的（无掩码），与 Decoder 中带因果掩码的 Self-Attention 形成对比。
- 多头机制让模型从多个子空间并行捕捉不同维度的依赖关系。

---

**本块学习检查**（动手验证，约 3 分钟）：

请用一句话回答：Encoder 的 Multi-Head Self-Attention 与你在 GPT-2

## 块 11

## 三、Encoder 的层次结构（①）——Multi-Head Self-Attention

先看整体图：输入经过 Embedding 和位置编码后，进入 Encoder。Encoder 有 **6 层**，每一层内部结构相同。本块聚焦第一层里的第一个子层：**Multi-Head Self-Attention**。

### 3.1 Multi-Head Self-Attention 是什么

一句话：**让每个位置聚合整个输入序列的上下文信息**。

比如输入是“猫 喜欢 鱼”，处理“喜欢”这个词时，Self-Attention 会让它同时“看到”前面的“猫”和后面的“鱼”，从而理解“谁喜欢谁”。这就是“聚合上下文”。

### 三个关键点

1. **Q/K/V 来源**：都来自 Encoder **前一层**的输出。注意：第一层 Encoder 的“前一层”就是输入 Embedding + 位置编码的结果。Q、K、V 不是外部传入的，而是同一个输入通过不同变换得到的。

2. **Mask**：**无因果 mask**。这意味着**双向**——序列中所有位置互相可见，每个词都能看到完整句子。但批处理时通常仍有 padding/attention mask 来处理变长序列（这是工程细节，不影响核心理解）。

3. **作用**：每个位置都能聚合整个输入序列的信息，而不是只看局部窗口。

### 动手验证（hands-on 小任务）

拿一句话“我 爱 学习 Transformer”，在纸上画出 5 个位置。问自己：

- 处理第 3 个词“学习”时，它能“看到”哪些词？
- 如果加了因果 mask，它还能看到“Transformer”吗？

**答案**：不加 mask 时能看到全部 5 个词；加因果 mask 后只能看到前 3 个（“我 爱 学习”）。这正是 Encoder 和 Decoder 的关键区别——Decoder 的 Self-Attention 带因果 mask，而 Encoder 不带。

### 记住这个结构

每个子层（包括 Self-Attention）后面都跟着 **Add & Norm**——残差连接（Add）

## 块 12

### 3.2 Position-wise Feed-Forward Network (FFN)

这一层是**逐位置**的两层全连接网络，对序列中每个位置的向量独立做相同变换，中间用 ReLU 激活：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

维度先**放大**再**还原**：$d_{\text{model}}=512 \to d_{ff}=2048 \to d_{\text{model}}=512$。放大到 2048 让网络有更多参数学习复杂特征，再压回 512 以匹配后续层。

### 3.3 残差连接 + Layer Normalization

每个子层（注意力或 FFN）输出后都接：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

**残差连接**（$x + \text{Sublayer}(x)$）让梯度能直接流过，缓解深层训练困难；**LayerNorm** 稳定每层输出分布。

**Encoder 完整数据流**：

$$
\begin{aligned}
x_1 &= \text{Embedding}(input) + \text{PositionalEncoding} \\
x_2 &= \text{LayerNorm}(x_1 + \text{MultiHeadAttention}(

## 块 13

**目标：让 Decoder 只看“过去”，不看“未来”**

想象你在写句子，每写一个词，只能参考已经写好的词，不能偷看后面的词。Masked Multi-Head Self-Attention 就是干这个的。

**Q/K/V 从哪来？** 都来自 Decoder 前一层的输出。也就是说，Decoder 用自己的历史输出，去查询和匹配自己。

**Mask 是什么？** 因果掩码（Causal Mask）。在计算注意力分数时，把“未来位置”的分数直接设为 **-∞**。公式里，当 `j > i`（后面的位置）时，分数就是负无穷。

**效果呢？** 经过 Softmax 后，负无穷对应的权重会变成 **0**。这样，每个位置只能关注到自己和之前的位置，保持了

## 块 14

**动手理解 Cross-Attention：Decoder 的“输入桥梁”**

你现在是初学者，目标是快速掌握 Transformer 架构。我们直接切入重点，用“谁在看谁”的视角来理解。

**核心三问（记住这个框架）**

1. **Q（查询）从哪来？** 来自 Decoder 自己的**前一层**输出。它代表“我（Decoder）现在想找什么信息”。
2. **K 和 V（键和值）从哪来？** 都来自 **Encoder 的输出**（记作 $\mathbf{z}$）。它代表“输入序列提供了什么信息”。
3. **能用 Mask 吗？** 这里**没有**因果 mask——Decoder 能看到**完整**的输入句子，因为

## 块 15

## Decoder 的 Feed-Forward Network（FFN）

### 一、FFN 是什么？

Decoder 里的 **FFN（前馈网络）** 和 Encoder 里的 FFN **结构相同，但参数独立**。你可以把它理解成每个位置上的“独立思考层”——它不关心位置之间的关系，而是对每个位置的表示做一次非线性变换，增强模型的表达能力。

在 Decoder 的数据流中，FFN 出现在第二个注意力（Cross-Attention）之后：

```
y4 = LayerNorm(y3 + FFN(y3))
```

也就是说，Decoder 每一层都由三个子层组成：**Masked Self-Attention → Cross-Attention → FFN**，每个子层后面都跟着残差连接和 LayerNorm。

---

### 二、Decoder 完整数据流（一次前向传播）

假设你已经有了：
- `output`：Decoder 的输入（已嵌入的 token 序列）
- `z`：Encoder 的最终输出（作为 Cross-Attention 的 K 和 V）

那么 Decoder 的完整计算过程如下：

```
x = Embedding(output) + PositionalEncoding
y1 = LayerNorm(x + MultiHeadAttention(Q=x, K=x, V=x, mask=causal))
y2 = LayerNorm(y1 + MultiHeadAttention(Q=y1, K=z, V=z))
y3 = LayerNorm(y2 + FFN(y2))
...重复N次（如6层）...
logits = Linear(y_final)
probs = Softmax(logits)
```

**注意**：这个流程会重复 **N 次**（如 base 模型为 6 层），每一层都做同样的三个子层操作。FFN 内部通常包含两个线性层和一个激活函数。

---

### 三、三种 Attention 对比（面试常考）

三种 Attention 的公式都是同一个：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**区别只在于 Q/K/V 的来源不同**，作用也因此完全不同：

| Attention 类型 | 位置 | Q 来源 | K 来源 | V 来源 | Mask | 作用 |
|---|---|---|---|---|---|---|
| **Encoder Self-Attention** | Encoder 每层 | 输入嵌入或前一层输出 | 输入嵌入或前一层输出 | 输入嵌入或前一层输出 | 无因果 mask | 编码输入序列的上下文 |
| **Decoder Masked Self-Attention** | Decoder 每层 | 输入嵌入或前一层输出 | 输入嵌入或前一层输出 | 输入嵌入或前一层输出 | **因果 mask** | 自回归生成，防止“看到未来” |
| **Decoder Cross-Attention** | Decoder 每层 | Decoder 前

## 块 16

## Decoder 里的“小工坊”：Feed-Forward Network

前面我们看了 Decoder 的层数怎么打印，现在聚焦每一层里的一个关键部件——**前馈网络（FFN）**。你可以把它想成每个 Decoder 层自带的一个“小工坊”：注意力机制负责“收集信息”，而 FFN 负责“加工信息”。

在刚才打印的模型结构里，你会在每个 Decoder 层内部看到类似 `T5Block` 的结构，其中就包含 `T5LayerFF` 这样的子模块——这就是 FFN 的实体。它的作用是对注意力层输出的每个位置向量，独立地做两次线性变换加一次激活函数，相当于对每个词的表征做一次“深加工”。

**动手验证一下**：运行下面的代码，看看 FFN 到底长什么样：

```python
# 取第一个 Decoder 层，看它的子模块
decoder_layer = model.decoder.block[0]
print(decoder_layer)
```

你会看到 `T5LayerFF`

## 块 17

## 6.1 亲手跑一次：观察 Encoder-Decoder 的前向传播

既然你偏好动手实践，我们直接通过代码来“看见”Transformer 内部发生了什么。下面这段代码演示了如何让一个训练好的翻译模型（Encoder-Decoder 结构）处理一句英文，并观察它内部各部分的输出形状。

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载模型和分词器（以 t5-small 为例）
# 注意：t5-small 约 60M 参数，FP32 推理约需 1GB 显存，CPU 运行需内存 ≥ 4GB
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

**动手验证任务**：运行这段代码，把打印出的四个形状记录下来。重点看最后一行——`cross_attentions[0].shape` 的最后两个维度。

**关键观察**：`cross_attentions[0].shape` 的最后两个维度通常是 `[decoder_len, encoder_len]`，说明 Decoder 的每个位置都在关注 Encoder 的所有位置。这正是 Cross-Attention 的作用。上述输出会随 transformers 版本和模型版本变化，实际以运行结果为准。

> **本地运行配置要求**：此代码纯 CPU 运行即可，内存 ≥ 4GB。

---

## 七、架构演化：为什么后续模型“拆分”了 Encoder-Decoder？

理解了完整的 Encoder-Decoder 结构后，一个自然的问题是：为什么 GPT-2、Llama、Qwen 这些大模型**只保留了 Decoder**？

这个问题留给你思考——结合你刚观察到的 Cross-Attention 形状，想一想

## 块 18

**动手理解：为什么 BERT 只用 Encoder？**

你已知道 Transformer 原本是 Encoder-Decoder 结构。现在看它的第一个“拆分”方向：**只用 Encoder**，也就是 BERT（2018）。

**核心思路**：把 Decoder 整个去掉，只保留 Encoder 部分。

**它怎么学习？** 用 **Masked Language Model（MLM）**——训练时随机遮住一些词，让模型根据上下文猜被遮住的词。因为能看到左右两侧的词，所以是**双向上下文**。

**适合做什么？** 理解类任务：分类、问答、命名实体识别（NER）。

**不适合做什么？** 生成任务（比如写文章、对话），因为它没有 Decoder 来逐词产出。

**动手验证（不引入新知识）**：对比一下，如果让你设计一个“只做情感分类”的模型，用 Encoder-only 还是 Decoder-only

## 块 19

### 7.3 Encoder-Decoder 仍在这些任务里发光发热

你可能已经知道，现在像 GPT 这样的通用大模型主流是 Decoder-only（只用解码器）。但别急着下结论说 Encoder-Decoder 就过时了——它在**特定任务**里依然是主力选手，而且在某些条件下效果可能更好。

来看几个你以后会经常碰到的例子：

- **机器翻译**：比如 `google-t5/t5-small`、`facebook/mbart-large-50`，这类模型天生就是干"把一种语言换成另一种语言"的活。
- **语音识别**：比如 `openai/whisper-base`，它要把声音波形转成文字，输入和输出是两种完全不同的形态。
- **多模态**：比如 `OFA` 这类模型，它要同时理解图片和文字，再生成回答。

这些任务的共同点是：**输入和输出的"形态"或"结构"差异很大**。Encoder 先把输入"看懂"（双向编码），Decoder 再一步步"说出来"（自回归生成），这种分工在结构不对称的任务里特别顺手。

> 那为什么 GPT 去掉 Encoder 还能干活？下一章会揭晓答案——核心就一句话：**把输入和输出拼接在一起，用统一的 Causal Mask（因果掩码）处理**。先记住这个结论，后面我们拆开揉碎讲。

---

### 常见问题速查表

学到这里，你可能会踩几个坑，先给你打预防针：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| 混淆三种 Attention 的 Q/K/V 来源 | 没有系统对比 | 记住核心区别：Self-Attention 的 Q/K/V **同源**，Cross-Attention 的 K/V **来自 Encoder** |
| 不理解为什么 Decoder 需要 Mask | 忽略了自回归性质 | 生成任务中，模型不能"偷看"未来的正确标签 |
| 以为 Encoder 输出只送到 Decoder 第一层 | 对数据流理解不深入 | Encoder 输出送到**每一层** Decoder 的 Cross-Attention |
| Cross-Attention 的 K 和 V 能否只用其一 | 对 Attention 机制理解不深入 | K 用于计算相似度，V 用于加权求和，**缺一不可** |

---

### 总结：你已掌握的核心要点

到这里，Transformer 的 Encoder-Decoder 结构你已经通关了，现在你应该能：

- ✅ 说清 Encoder

## 测评题目

### Q-KB08-ST0800-002
不计算梯度地执行前向推理，常使用 PyTorch 的 `torch.________()` 上下文。

### Q-KB08-ST0800-001
用 `AutoModel.from_pretrained()` 加载模型时，模型权重通常来自名称对应的预训练________。

### Q-KB0801-005
请简述 Encoder-Decoder 架构中，交叉注意力（Cross-Attention）层的 Query、Key、Value 分别来自哪里？

### Q-KB0801-003
以下哪个模型是典型的 **Encoder-only** 架构？

### Q-KB0806-005
请简述 PagedAttention（vLLM的核心技术）是如何解决KV Cache显存碎片化问题的。

### Q-KB08-ST0800-003
加载 GPT-2 等预训练模型进行前向实验，最直接的目的是什么？

### Q-KB0806-004
FlashAttention 的核心思想是通过分块（Tiling）和重计算（Recomputation）来减少HBM（显存）的读写次数，从而加速并降低显存占用。（ ）

### Q-KB08-ST0807-002
推理时将模型切换到评估模式，应调用 `model.________()`。
