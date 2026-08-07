# 注意力机制核心概念：Query、Key、Value 详解

> **参考文章链接**
> 
> ① 底稿：《Transformer 核心机制详解（QKV 与注意力机制）》
> 　　网址：https://blog.csdn.net/qq_43604387/article/details/156722747
> 
> ② 参考文章1：《〖AI大模型入门指南〗Transformer架构中的QKV--自注意力机制详解》
> 　　网址：https://blog.csdn.net/superherowupan/article/details/148839069

---

**适合人群**：已掌握 Python 编程、NLP 基础概念和文本数值化表示，希望深入理解 Transformer 核心机制的学习者。每一步都拆解到"写什么、为什么这样写"，你只需要跟着敲。完成本章后，你将彻底理解 Query、Key、Value 的本质，掌握注意力分数的计算流程，为后续"Self-Attention 与 Multi-Head Attention 的公式推导"打下坚实基础。

**预计学习时长**：约 1 小时  
**环境准备**：Python 3.10+，安装 `numpy`、`torch`

```bash
pip install numpy torch
```

---

## 第一章：为什么需要注意力机制

### 1.1 从 RNN 的痛点说起

在 Transformer 出现之前，循环神经网络（RNN/LSTM）是处理序列数据的主流。但 RNN 有一个致命缺陷：**必须按顺序处理**。

```
RNN 处理句子 "我喜欢自然语言处理"：

我 → 喜 → 欢 → 自 → 然 → 语 → 言 → 处 → 理
 ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
必须逐个计算，后面的词必须等前面的词算完
```

**问题**：
- **慢**：无法并行，长句子处理速度极慢
- **遗忘**：距离越远，信息衰减越严重（"处理"很难记住"我"的信息）

### 1.2 注意力机制的直觉

注意力机制的灵感来自**人类的阅读理解**。

当你读到这句话时：

> "你昨天给我的苹果很好吃，你的苹果股票涨了没？"

你的大脑会自动判断：
- 第一个"苹果" → 与"吃"相关 → 水果
- 第二个"苹果" → 与"股票"相关 → 公司

**注意力机制让模型也能做同样的事**：在处理每个词时，自动"看"向句子中所有其他词，判断哪些词与当前词最相关，然后加权汇总信息。

**为什么这样写**：这种"全局视野"让模型能直接捕捉任意两个词之间的关系，无论它们相距多远。

---

## 第二章：QKV 的本质——三个"角色"

### 2.1 通俗比喻：图书馆找书

想象你走进一座图书馆，要找一本关于"深度学习"的书：

| 角色 | 比喻 | 作用 |
|------|------|------|
| **Query（查询）** | 你脑海中的问题 | "我想找一本关于深度学习的书" |
| **Key（键）** | 书架上的标签 | "机器学习""深度学习""Python编程" |
| **Value（值）** | 书架上的书本身 | 书的具体内容 |

**过程**：
1. 你带着 **Query**（问题）走进图书馆
2. 你把 Query 和每个书架的 **Key**（标签）对比，找到最匹配的
3. 从匹配的书架取出 **Value**（书的内容），带回阅读

**为什么这样写**：Query 是"我要什么"，Key 是"我有什么标签"，Value 是"实际内容"。三者分离，让信息检索变得灵活高效。

### 2.2 在 Transformer 中的对应

在 Transformer 中，输入序列的每个词都会生成自己的 Q、K、V：

```
句子: "猫 喜欢 鱼"

词"喜欢"的视角:
  Query(喜欢):  "我是谁？我需要什么信息？"
  Key(喜欢):    "我的标签是什么？"
  Value(喜欢):  "我携带了什么信息？"

注意力计算:
  Query(喜欢) 与 Key(猫) 匹配 → 相关度高 → 多取 Value(猫) 的信息
  Query(喜欢) 与 Key(鱼) 匹配 → 相关度高 → 多取 Value(鱼) 的信息
  Query(喜欢) 与 Key(喜欢) 匹配 → 相关度高 → 多取 Value(喜欢) 的信息
```

**为什么这样写**：每个词既是"提问者"（Query），又是"被检索对象"（Key 和 Value）。这就是**自注意力（Self-Attention）**——自己问自己，自己检索自己。

---

## 第三章：QKV 的生成——线性变换

### 3.1 输入是什么

Transformer 的输入是经过**词嵌入（Word Embedding）**和**位置编码（Positional Encoding）**后的向量。

```python
import torch
import torch.nn as nn

# 假设参数
batch_size = 2      # 一次处理2个句子
seq_len = 10        # 每个句子10个词
d_model = 512       # 模型维度（每个词的向量长度）

# 输入 X: [batch_size, seq_len, d_model]
X = torch.randn(batch_size, seq_len, d_model)
print(f"输入 X 形状: {X.shape}")   # torch.Size([2, 10, 512])
```

**为什么这样写**：`d_model = 512` 是 Transformer 论文中的标准设置。每个词被表示为一个 512 维的向量。

### 3.2 用三个线性层生成 Q、K、V

```python
# 定义三个独立的线性投影层
W_Q = nn.Linear(d_model, d_model)   # Query 投影
W_K = nn.Linear(d_model, d_model)   # Key 投影
W_V = nn.Linear(d_model, d_model)   # Value 投影

# 生成 Q、K、V
Q = W_Q(X)   # [2, 10, 512]
K = W_K(X)   # [2, 10, 512]
V = W_V(X)   # [2, 10, 512]

print(f"Q 形状: {Q.shape}")   # torch.Size([2, 10, 512])
print(f"K 形状: {K.shape}")   # torch.Size([2, 10, 512])
print(f"V 形状: {V.shape}")   # torch.Size([2, 10, 512])
```

**为什么这样写**：
- 三个**独立的**权重矩阵 $W^Q$、$W^K$、$W^V$ 让模型学会从同一个输入中提取三种不同"视角"的信息
- 如果不做投影，直接用 $X$ 当 Q/K/V，模型就失去了学习不同表示的能力
- 投影后维度不变（512 → 512），是为了后续多头拆分的方便

---

## 第四章：注意力分数计算——从 Q 和 K 到权重

### 4.1 核心公式

注意力机制的数学表达：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

拆解为四步：
1. **点积**：$Q \times K^T$ → 得到原始相似度分数
2. **缩放**：除以 $\sqrt{d_k}$ → 防止数值过大
3. **归一化**：Softmax → 转为概率分布（权重和为1）
4. **加权求和**：权重 × V → 得到最终输出

### 4.2 第一步：计算注意力分数矩阵

```python
# Q: [batch_size, seq_len, d_model] = [2, 10, 512]
# K: [batch_size, seq_len, d_model] = [2, 10, 512]

# 转置 K 的最后两个维度，使内积维度匹配
# K.transpose(-2, -1): [2, 512, 10]

scores = torch.matmul(Q, K.transpose(-2, -1))
print(f"注意力分数矩阵形状: {scores.shape}")   # torch.Size([2, 10, 10])
```

**为什么这样写**：
- `scores[i][j]` 表示**第 i 个词对第 j 个词的"关注程度"**
- 结果是一个 $10 \times 10$ 的矩阵，因为句子有 10 个词，每个词都要"看"其他 10 个词
- 对角线上的值表示"自己对自己的关注度"

### 4.3 第二步：缩放（Scale）

```python
import math

d_k = d_model   # 这里 d_k = 512（单头情况）

scaled_scores = scores / math.sqrt(d_k)
print(f"缩放后分数范围: [{scaled_scores.min():.2f}, {scaled_scores.max():.2f}]")
```

**为什么这样写**：在常见的零均值、单位方差假设下，点积的**方差随 d_k 线性增长**。不除以 √d_k 会导致 Softmax 的输入数值过大，进入**梯度饱和区**（梯度趋近于 0，无法学习）。除以 √d_k 能将分数的尺度稳定在方差约为 1 的范围，保持梯度健康。

**直观理解（统计视角）**：
- 假设 Q、K 各分量独立、均值为 0、方差为 1，则点积 `Q·K = Σ qᵢkᵢ` 的**期望为 0**，**方差为 d_k**
- 对 d_k = 512，标准差约为 √512 ≈ 22.6，意味着点积结果的数值**量级**大约在 ±22.6 附近波动
- 除以 √d_k ≈ 22.6 后，分数的标准差回到约 1，Softmax 工作在 0 附近的高梯度区域

### 4.4 第三步：Softmax 归一化

```python
attn_weights = torch.softmax(scaled_scores, dim=-1)
print(f"注意力权重形状: {attn_weights.shape}")   # torch.Size([2, 10, 10])
print(f"每行权重之和: {attn_weights[0].sum(dim=-1)}")   # 全为 1.0
```

**为什么这样写**：
- `dim=-1` 表示对每一行做 Softmax，确保**每个词对所有词的权重之和为 1**
- 这样权重就是合法的概率分布
- 某一行中值越大的位置，表示当前词越"关注"那个位置的词

### 4.5 第四步：加权求和

```python
# attn_weights: [2, 10, 10]
# V: [2, 10, 512]

output = torch.matmul(attn_weights, V)
print(f"注意力输出形状: {output.shape}")   # torch.Size([2, 10, 512])
```

**为什么这样写**：
- 每个词的输出，是**所有词的 Value 向量的加权平均**
- 权重由 Query 和 Key 的匹配程度决定
- 输出形状与输入相同（$[2, 10, 512]$），可以堆叠多层

---

## 第五章：完整自注意力实现

把以上步骤封装成一个完整的自注意力模块：

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 三个线性投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
    def forward(self, X):
        """
        X: [batch_size, seq_len, d_model]
        """
        # 1. 生成 Q、K、V
        Q = self.W_Q(X)   # [B, L, D]
        K = self.W_K(X)   # [B, L, D]
        V = self.W_V(X)   # [B, L, D]
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))   # [B, L, L]
        
        # 3. 缩放
        d_k = Q.size(-1)
        scaled_scores = scores / math.sqrt(d_k)
        
        # 4. Softmax
        attn_weights = torch.softmax(scaled_scores, dim=-1)   # [B, L, L]
        
        # 5. 加权求和
        output = torch.matmul(attn_weights, V)   # [B, L, D]
        
        return output, attn_weights


# 测试
d_model = 512
seq_len = 10
batch_size = 2

X = torch.randn(batch_size, seq_len, d_model)

attention = SelfAttention(d_model)
output, weights = attention(X)

print(f"输入形状:  {X.shape}")        # [2, 10, 512]
print(f"输出形状:  {output.shape}")   # [2, 10, 512]
print(f"权重形状:  {weights.shape}")  # [2, 10, 10]
```

**为什么这样写**：
- 输入输出形状相同，这是 Transformer 可以**堆叠多层**的基础
- 权重矩阵 `[2, 10, 10]` 可以可视化，观察模型"在看哪里"

---

## 第六章：不对称注意力——Q 和 KV 可以长度不同

### 6.1 关键洞察

注意力机制有一个非常灵活的特性：**Q 的序列长度和 K/V 的序列长度可以不同**。

```
Q: [batch_size, q_len, d_model]     → 100 个查询
K: [batch_size, kv_len, d_model]    → 500 个键
V: [batch_size, kv_len, d_model]    → 500 个值

scores = Q @ K^T  → [batch_size, 100, 500]
output = softmax(scores) @ V → [batch_size, 100, d_model]
```

**为什么这样写**：输出维度永远跟 **Q 的序列长度**一致。这意味着：
- 你可以用 1 个 Query 去查 1000 条记忆（检索增强）
- 也可以用 100 个 Query 同时查 10 条记忆（批处理）

### 6.2 三种典型场景

| 场景 | Q 长度 | KV 长度 | 说明 |
|------|--------|---------|------|
| **Encoder 自注意力** | $L$ | $L$ | 对称，自己查自己 |
| **Decoder 交叉注意力** | $L_{out}$ | $L_{in}$ | 解码器查编码器输出 |
| **自回归生成（KV Cache）** | 1 | 越来越长 | 每步只算新 token 的 Q，复用历史 KV |

---

## 第七章：从代码中提取 KV Cache

在 Hugging Face 的 Transformers 库中，KV Cache 的使用方式：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# 第一次前向传播，同时返回 KV Cache
outputs = model(**inputs, use_cache=True)

# past_key_values 就是 KV Cache
# 结构: tuple of length = num_layers
# 每层: (key, value)，形状 [batch, num_heads, seq_len, head_dim]
past_key_values = outputs.past_key_values

# 继续生成下一个 token 时，直接传入 past_key_values
# 只需计算新 token 的 Q，历史 KV 直接复用
next_token_id = outputs.logits[:, -1:].argmax(dim=-1)
outputs2 = model(next_token_id, past_key_values=past_key_values, use_cache=True)
```

**为什么这样写**：自回归生成时，每次只新增 1 个 token。如果没有 KV Cache，每生成一个 token 都要**重新计算所有历史 token 的 K 和 V**（相当于反复编码整段上下文），总体代价很高。有了 KV Cache，每步只需：① 计算新 token 自己的 Q/K/V，② 把新 K/V 追加到缓存里，③ 用新 Q 与**全部缓存 K/V** 做一次注意力。总体核心节省来自**避免了历史 K/V 的重复投影计算**，生成越长优势越明显。

---

## 第八章：维度变化完整速查表

| 步骤 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 | $X$ | `[B, L, D]` | 原始 token 嵌入 |
| 线性投影 | $Q, K, V$ | `[B, L, D]` | 三个独立投影 |
| 分头（多头时） | $Q, K, V$ | `[B, H, L, D/H]` | $H$=头数，$D/H$=每头维度 |
| 点积 | $QK^T$ | `[B, H, L, L]` | 注意力原始分数 |
| 缩放 + Softmax | $\text{attn\_weights}$ | `[B, H, L, L]` | 归一化权重 |
| 加权求和 | $\text{attn\_weights} \times V$ | `[B, H, L, D/H]` | 每头输出 |
| 拼接 + 投影 | $\text{output}$ | `[B, L, D]` | 多头合并后最终输出 |

---

## 本章小结与避坑清单

| 坑点 | 正确做法 |
|------|----------|
| 忘记缩放（不除以 √d_k） | **强烈推荐加 Scale**；不加容易导致 Softmax 输入过大、训练初期梯度消失或不稳定 |
| Q 和 K 的维度不匹配 | 确保 Q 的最后一个维度 = K 的倒数第二个维度（即 d_k 要对齐） |
| Softmax 的维度搞错 | `dim=-1` 对每行做，确保每行和为 1 |
| 混淆 Q/K/V 的作用 | Q=查询（我要什么），K=键（标签/我有什么），V=值（实际内容），三者职责不同 |
| 以为 Q 和 KV 必须等长 | 它们可以不同，输出序列长度始终等于 Q 的长度 |
| 自己实现时不初始化权重 | 用 `nn.Linear` 自动处理权重初始化 |

| 知识点 | 核心要点 |
|--------|----------|
| **Query** | 当前位置的"问题"，主动去匹配 |
| **Key** | 每个位置的"标签"，被动被匹配 |
| **Value** | 每个位置的"内容"，被加权取出 |
| **点积** | QKᵀ，衡量查询与标签的相似度 |
| **Scale** | 除以 √d_k，防止方差随 d_k 增大导致 Softmax 饱和 |
| **Softmax** | 将分数转为概率权重 |
| **加权求和** | 权重 × V，得到融合上下文信息的输出 |
| **KV Cache** | 自回归生成时缓存历史 K/V，避免重复投影计算 |

**养成的好习惯**：
1. 实现注意力时，先在纸上/注释里写清楚 Q/K/V 的维度
2. **默认加 Scale**，这是 Transformer 论文中的标准设置；除非有明确理由，不要移除
3. 自回归生成场景一定启用 KV Cache，否则生成速度会非常慢
4. 理解"不对称注意力"：Q 和 KV 可以不同长

**下一步预告**：掌握了 QKV 的核心概念后，你将进入 **Self-Attention 与 Multi-Head Attention 的计算原理与公式**——理解如何将单头注意力拆分为多个并行头、掌握 Multi-Head Attention 的完整公式推导、以及 Masked Self-Attention 在 Decoder 中的作用。这是理解 Transformer 架构的最后一道门槛。继续加油！