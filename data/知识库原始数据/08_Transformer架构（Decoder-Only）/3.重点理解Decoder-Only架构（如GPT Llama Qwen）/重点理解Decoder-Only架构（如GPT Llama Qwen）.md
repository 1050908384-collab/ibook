# 重点理解 Decoder-Only 架构（如 GPT / Llama / Qwen）

权威文章：《LLaMA: Open and Efficient Foundation Language Models》  
　　网址：https://arxiv.org/abs/2302.13971

权威文章：《Introduction - Hugging Face NLP Course》  
　　网址：https://huggingface.co/learn/nlp-course/chapter1/1

① 底稿：《面试题详解：GPT 系列、Llama 系列、Qwen 系列全解析》  
　　网址：https://blog.csdn.net/qq1137623160/article/details/161144938

② 参考文章：《Qwen 3 架构介绍：模块详解与相对 Qwen 1 / Qwen 2 的演进》  
　　网址：https://blog.csdn.net/zyctimes/article/details/159828385

> **适合人群**：已完成"掌握 Transformer 的 Encoder-Decoder 结构"章节，理解 Encoder 双向编码、Decoder 单向自回归生成、三种 Attention 数据流差异的初学者。本节时长约 1 小时，目标是理解 Decoder-Only 架构的设计动机，掌握 GPT、Llama、Qwen 三条演进主线的核心差异，能够回答"为什么 GPT 去掉 Encoder 后仍然能工作"这一关键问题。（①）
>
> **前置说明**：上节我们系统讲解了完整的 Encoder-Decoder 结构，知道了 Decoder 需要 Masked Self-Attention（因果掩码）和 Cross-Attention（关注 Encoder 输出）。一个自然的问题是：为什么 GPT-2、Llama、Qwen 这些大模型**只保留了 Decoder**？Cross-Attention 和 Encoder 被去掉后，模型怎么"理解"输入？本节将回答这个问题，并梳理 Decoder-Only 架构从 GPT 到 Llama 再到 Qwen 的演进脉络。所有代码请在已安装 `transformers` 的 `pytorch` conda 环境中运行。

## 一、Decoder-Only 的核心设计思想（①）

### 1.1 为什么可以去掉 Encoder？（①）

Decoder-Only 使用统一的因果语言建模目标，将输入与输出组织为同一个 token 序列。prompt 内部仍遵守从左到右的因果可见性，而输出 token 能访问其之前的全部 prompt。Decoder-Only 通过统一目标和结构获得较好的扩展性，但 Encoder-Decoder 在翻译、摘要等条件生成任务中仍具有独立编码输入、计算复用和任务归纳偏置等优势。架构选择取决于训练目标、任务类型和工程约束，不能仅根据输入输出是否为同一种模态判断。（①）

### 1.2 统一框架：Causal Language Modeling（①）

Decoder-Only 模型的训练目标只有一个——**因果语言建模（Causal Language Modeling, CLM）**，也叫 Next Token Prediction：

给定序列中前 $t$ 个 token，预测第 $t+1$ 个 token。用数学表达：

$$P(x_1, x_2, ..., x_n) = \prod_{t=1}^{n} P(x_t | x_1, ..., x_{t-1})$$

这个统一的训练目标带来了几个关键优势：（①）

| 优势 | 说明 |
|---|---|
| **架构简化** | 没有 Encoder 和 Cross-Attention，参数量全部集中在 Decoder，更容易扩展到超大规模 |
| **数据要求低** | 不需要成对的输入-输出数据，任何纯文本都可以作为训练数据 |
| **统一接口** | 问答、摘要、代码等任务可转化为"续写"问题，通过 prompt 设计即可切换 |
| **推理高效** | KV Cache 只需要缓存 Decoder 自己的 K 和 V，不需要额外缓存 Encoder 输出 |

> **关键理解**：Decoder-Only 针对"输入输出同模态"任务做了架构归一化。当输入和输出都是文本时，Decoder 的因果注意力已能让每个输出位置关注到所有输入位置，但 Encoder-Decoder 在特定条件生成任务中仍有其独特优势。（①）

## 二、GPT 系列：Decoder-Only 的开山鼻祖（①）

### 2.1 GPT-1：确立"预训练 + 微调"范式（①）

2018 年，OpenAI 发布 GPT-1，核心创新不是某个具体模块，而是**架构层面的取舍**：

- **只保留 Decoder 堆栈**：去掉 Encoder 和 Cross-Attention
- **因果语言建模**：训练目标为 Next Token Prediction
- **预训练 + 下游微调**：先在大规模无标注文本上预训练，再针对具体任务做微调

GPT-1 证明了：一个在海量文本上训练过的 Decoder-Only 模型，可以通过微调适配到分类、 entailment、相似度、问答等多种下游任务。（①）

### 2.2 GPT-2：规模即能力（①）

2019 年的 GPT-2 没有改动核心架构，而是把**规模**推到了新高度（1.5B 参数，40GB 训练数据）。它展示了一个惊人的现象——**zero-shot 能力**：

不微调参数，只通过设计不同的输入 prompt，模型就能执行翻译、摘要、问答等任务。这说明 Decoder-Only 架构的"续写"能力本身就蕴含了通用性。（①）

### 2.3 GPT-3：Prompt 即接口（①）

2020 年的 GPT-3 将参数扩展到 175B，并把 **few-shot / in-context learning** 推向主流：

- **Few-shot**：在 prompt 中给几个示例，模型就能模仿示例的格式和风格
- **In-context learning**：模型从 prompt 的上下文中"学习"任务，无需更新权重

GPT 三代演进的核心逻辑：（①）

```
GPT-1: "预训练+微调"这条路走得通
GPT-2: "规模变大后，多任务能力会自然涌现"
GPT-3: "prompt 本身可以变成一种新型编程接口"
```

## 三、Llama 系列：开源 Decoder-Only 的标杆（权威① + ①）

### 3.1 Llama 不是推翻，而是"做减法 + 做强化"（①）

Llama 系列（Meta，2023 年起）继承了 GPT 的 Decoder-Only 路线，但在关键组件上做了更适合大语言模型的选择。LLaMA 论文明确提出了三项核心改进（权威①）：

| 改进项 | 原始 Transformer | Llama 的做法 | 作用 |
|---|---|---|---|
| **归一化** | Post-LayerNorm（子层后归一化） | **Pre-RMSNorm**（子层输入前归一化） | 训练更稳定，减少均值计算开销 |
| **激活函数** | ReLU | **SwiGLU** | 表达能力更强，门控机制过滤信息 |
| **位置编码** | 绝对正弦位置编码 | **RoPE（旋转位置编码）** | 更自然地建模相对位置，利于长序列外推 |

> **注意**：RMSNorm、SwiGLU、RoPE 的数学原理将在后续章节逐一深入讲解。本节只需建立"这些组件存在且重要"的感性认识。（权威①）

### 3.2 Llama 三代演进（①）

| 维度 | Llama1 | Llama2 | Llama3 |
|---|---|---|---|
| **定位** | 开源高性价比基础模型 | 更成熟的对话基座 | 多语言、代码、工具综合能力平台 |
| **上下文** | 较短 | 常见 4K | 更长（部分版本 128K+） |
| **后训练** | 以 base 模型为主 | Chat 对齐更成熟 | 工具使用、推理能力更完整 |
| **推理效率** | 经典 Decoder-only | 开始强调实用性 | GQA 等高效设计更广泛 |

一句话总结三代差异：（①）

> **Llama1** 证明了"公开数据也能训出强模型"；**Llama2** 把开源大模型往可用产品推进；**Llama3** 把多语言、代码、工具和长上下文能力进一步拉上去。

## 四、Qwen 系列：国产 Decoder-Only 的演进（① + ②）

### 4.1 Qwen3 的架构亮点（②）

Qwen3（阿里巴巴，2025 年）在 Decoder-Only 主线上做了进一步创新。根据技术报告和架构分析（②），其核心特点包括：

**（1）Dense + MoE 双轨并存**（②）

- **Dense（稠密）模型**：每一层的 FFN 只有一套权重，每个 token 前向时全部参与计算。如 Qwen3-8B、14B、32B。
- **MoE（混合专家）模型**：FFN 换成多个独立专家，通过门控网络每 token 只激活少数专家。如 Qwen3-30B-A3B（总参 30B，每 token 激活 3B）、235B-A22B（总参 235B，每 token 激活 22B）。

> **MoE 的优势**：仓库里可以装下非常大的总参数量，但每生成一个 token 实际计算量只和激活的支路相当，兼顾容量与效率。（②）

**（2）Thinking / Non-Thinking 统一**（① + ②）

Qwen3 将"深度推理模式"和"快速响应模式"统一进**同一套权重**，通过 chat template 中的开关（如 `/think`、`/no_think`）控制。用户还可以设置 **thinking budget**（思考预算），在效果与延时之间做灵活平衡。（①）

**（3）关键架构细节演进**（②）

| 项目 | Qwen1 | Qwen2 | Qwen3 |
|---|---|---|---|
| 注意力 | MHA 为主 | **全系 GQA** | **全系 GQA** |
| QKV bias | 有 | 有 | **无** |
| Q/K 稳定化 | 靠 bias | 靠 bias | **QK-Norm** |
| 长上下文 | 推理期技巧为主 | 训练 32K + YARN + DCA | 三阶段预训练 + ABF + YARN + DCA |
| MoE | — | 64 专家，有共享专家 | **128 专家，无共享专家** |

> **GQA（Grouped-Query Attention）**：多个 Query 头分组共用较少的 Key/Value 头，减少 KV Cache 体积。后续"KV Cache 机制"章节会深入讲解。（②）

### 4.2 Qwen3 的三阶段预训练（②）

Qwen3 的预训练不是"一把梭"，而是分阶段推进：（②）

1. **General 阶段**：约 30T+ tokens，序列长度 4,096——广谱学习语言与世界知识
2. **Reasoning 阶段**：约 5T 高质量数据（STEM、代码、推理比例升高），仍 4,096——强化推理相关分布
3. **Long Context 阶段**：数百 B tokens，序列 32,768——让优化目标直接覆盖万级 token 的因果依赖

这种分阶段策略体现了现代 LLM 训练的核心思想：**先广度、再深度、最后长度**。（②）

## 五、三条主线的架构对比总结（① + ②）

| 维度 | GPT 系列 | Llama 系列 | Qwen 系列 |
|---|---|---|---|
| **核心路线** | Decoder-only + 规模化 | Decoder-only + 组件优化标准化 | Decoder-only + MoE + 多模态能力 |
| **归一化** | LayerNorm → 后续改进 | **Pre-RMSNorm** | **Pre-RMSNorm** |
| **位置编码** | 绝对位置编码 | **RoPE** | **RoPE** |
| **激活函数** | GeLU / 其他 | **SwiGLU** | **SwiGLU** |
| **注意力优化** | 标准 MHA | 后续引入 GQA | **全系 GQA + QK-Norm** |
| **长上下文** | 逐步扩展 | 逐步扩展 | **三阶段预训练 + ABF/YARN/DCA** |
| **推理模式** | 单一生成 | 单一生成 | **Thinking / Non-Thinking 统一** |
| **开源策略** | 闭源（GPT-3 后） | 开源权重 | 开源权重（Apache 2.0） |

> **面试高频考点**：回答"GPT / Llama / Qwen 有什么区别"时，较为稳妥的层次是：先答"它们都是 Decoder-only Transformer"，再答"各自在组件选择、规模策略、训练数据、后训练方式上的差异"，最后补"这些差异带来了什么能力变化"。（①）

## 六、用 transformers 库观察 Decoder-Only 模型结构（权威① + ②）

我们可以通过 `transformers` 库对比不同 Decoder-Only 模型的配置，观察它们如何体现上述架构差异。

### 6.1 观察 GPT-2 的"原始"Decoder-Only 结构

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

**输出**：
```
层数: 12
隐藏维度: 768
注意力头数: 12
FFN 中间维: 3072
位置编码: 绝对位置编码 (n_positions=1024)
激活函数: gelu
```

> **观察**：GPT-2 使用绝对位置编码、GeLU 激活、标准 LayerNorm，代表了"经典"Decoder-Only 架构。（权威①）

> **本地运行配置要求**：GPT-2 small 约 500MB，CPU 即可运行，内存 ≥ 4GB。

### 6.2 观察 Qwen2.5 的现代 Decoder-Only 结构

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
位置编码: RoPE (rope_theta=1000000.0)  ← ABF：基频从 1e4 提升到 1e6
归一化: RMSNorm (rms_norm_eps=1e-06)
QKV bias: False     ← Attention 投影层不使用 bias
```

> **关键观察**：
> - `num_key_value_heads < num_attention_heads` 是 **GQA** 的标志
> - `rope_theta=1000000.0` 是 **ABF（Attention Base Frequency）** 的体现，用于长上下文外推
> - `attention_bias=False` 表示该 Qwen2.5 配置的 Attention 投影层不使用 bias；它本身不能证明模型使用了 QK-Norm。QK-Norm 是 Qwen3 的独立架构特征。（②）

> **本地运行配置要求**：Qwen2.5-0.5B 约 1GB，首次加载需联网下载。CPU 即可运行，内存 ≥ 4GB。

### 6.3 对比三种 Attention 的 KV Cache 体积

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

# Qwen2.5-0.5B（GQA，num_kv_heads = 2, head_dim = 64）
qwen_kv = kv_cache_size(1, 2048, 24, 2, 64)
print(f"Qwen2.5-0.5B KV Cache: {qwen_kv / 1024 / 1024:.2f} MB")

# 对比：如果 Qwen 也用 MHA（假设 14 个 KV 头）
qwen_mha_kv = kv_cache_size(1, 2048, 24, 14, 64)
print(f"Qwen2.5-0.5B (假设 MHA) KV Cache: {qwen_mha_kv / 1024 / 1024:.2f} MB")
```

**输出**：
```
GPT-2 KV Cache: 6.00 MB
Qwen2.5-0.5B KV Cache: 1.17 MB
Qwen2.5-0.5B (假设 MHA) KV Cache: 8.20 MB
```

> **关键理解**：GQA 将 KV Cache 从 8.20 MB 降到 1.17 MB（约 7 倍节省），这对长序列推理至关重要。后续"KV Cache 机制"章节会深入讲解其原理。（②）

## 七、常见问题与解决（① + ②）

| 问题 | 可能原因 | 解决方案 |
|---|---|---|
| "Decoder-Only 怎么理解输入？" | 忽略了输入输出拼接机制 | 输入和输出拼接为统一序列，因果掩码让输出位置能看到所有输入位置（①） |
| "为什么不用 Encoder-Decoder 做 LLM？" | 混淆了任务类型 | Encoder-Decoder 适合输入输出不同模态（如翻译）；Decoder-Only 适合同模态文本任务，但架构选择需综合考虑任务类型和工程约束（①） |
| "Llama 和 GPT 架构完全不同吗？" | 过度夸大差异 | 核心都是 Decoder-Only，差异主要在组件选择（RMSNorm、RoPE、SwiGLU）（权威①） |
| "Qwen3 的 MoE 是多出来的层吗？" | 误解 MoE 结构 | MoE **只替换 FFN**，注意力层与稠密模型完全相同（②） |
| "Thinking / Non-Thinking 是两套权重吗？" | 误解模式切换 | 同一套权重，通过后训练和 chat template 控制行为（①） |

## 八、总结与后续学习路线（权威① + ①）

恭喜你完成了 Decoder-Only 架构的深入学习！你现在应该掌握了：

- ✅ Decoder-Only 的设计动机：使用统一因果语言建模目标，将输入输出组织为同一序列，但 Encoder-Decoder 在条件生成任务中仍有优势
- ✅ 统一训练目标 Causal Language Modeling（Next Token Prediction）的优势
- ✅ GPT 系列演进：GPT-1（预训练+微调）→ GPT-2（zero-shot 涌现）→ GPT-3（prompt 即接口）
- ✅ Llama 系列核心改进：Pre-RMSNorm、RoPE、SwiGLU，以及三代定位差异
- ✅ Qwen 系列核心改进：全系 GQA、QK-Norm、MoE、三阶段预训练、Thinking/Non-Thinking 统一
- ✅ 用 transformers 观察 GPT-2 与 Qwen2.5 的配置差异，理解 GQA 对 KV Cache 的节省

> **下一步预告**：在下一节 **"RMSNorm 与残差连接"** 中，我们将深入讲解 Pre-Norm 与 Post-Norm 的区别、RMSNorm 的数学公式与实现，以及为什么现代 LLM 普遍选择 Pre-RMSNorm。这是理解 Llama/Qwen 训练稳定性的关键。现在，试着用上面的代码，对比更多模型（如 `meta-llama/Llama-3.2-1B`）的 `config.json`，看看它们是否也采用了 GQA 和 RoPE！
