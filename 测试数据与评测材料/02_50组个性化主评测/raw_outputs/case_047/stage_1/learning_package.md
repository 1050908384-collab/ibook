# 学习包

## 块 1

好的，我们先从“为什么”开始。你之前已经跑通了 Transformers 环境，现在要学 LoRA，核心动机就一个：**省钱省显存**。

传统全参数微调，是把整个模型（比如70亿参数）的所有权重都拿来更新。想象一下，训练时不仅要保存模型参数，还要额外保存一份梯度，以及优化器状态（比如Adam的动量）。这三样东西加起来，显存需求会飙升到模型本身的好几倍。普通消费级GPU根本扛不住。

所以，参数高效微调（PEFT）的思路是：**不更新全部参数，只更新一小部分**。LoRA就是其中一种主流方法，它冻结原始模型，只训练注入的少量低秩矩阵，效果接近全量微调，但显存和算力开销大幅下降。

**本块学习检查**：请用自己的话，向同伴解释“为什么7B模型全参数微调在消费级GPU上跑不动”，并指出是哪三样东西占满了显存。

## 块 2

## 1.2 参数高效微调（PEFT）

PEFT = 参数高效微调。核心思路：**不动整个大模型，只训练少量新增参数**。

流程：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter。

三种方式对比：

| 方式 | 训练参数 | 显存需求 |
|------|---------|---------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**本块学习检查**：对照上表，用自己的话说清LoRA相比全参数微调省显存的原因（提示：训练参数规模）。

## 块 3

**LoRA是什么？——用“小补丁”微调大模型**

想象原模型权重是`W`，我们**不更新**它，而是新增一个变化量`ΔW`，最终效果是`W + ΔW`。

这个`ΔW`由两个小矩阵相乘得到：**`ΔW = B × A`**。若`W`形状是`d_out × d_in`，秩为`r`，则`A`约为`r × d_in`，`B`约为`d_out × r`，相乘后还原成`W`的形状。不同资料可能写作`A × B`，只是符号顺序不同，关键是形状能相乘并还原。

**我们只训练`A`和`B`这两个小矩阵**，原模型保持冻结。

**好处**：参数量少、训练快、显存占用低、Adapter文件小。训练完成后保存`adapter_model`和`adapter_config`，通常只有几十MB到几百MB。

---

**本块学习检查**：请用一句话说明：为什么

## 块 4

## 3.1 LoRA 的问题：模型还是太大

你已经知道 LoRA 能大幅减少**需要训练**的参数。但这里有个关键点：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个 7B（70亿参数）的模型，如果用 FP16（半精度）格式加载，大约需要 **14GB 显存**。也就是说，**还没开始训练，显存就已经被占掉一大半了**。如果你的显卡显存不够，LoRA 再省训练参数也跑不起来。

> **动手感受一下**：打开你的 GPU 监控（比如 `nvidia-smi`），加载一个 7B 模型，观察显存占用。你会发现，即使只做推理（不训练），显存也已经被吃掉很多——这就是 LoRA 要解决的"入场门槛"。

---

## 3.2 QLoRA 的思路：把"地基"压缩

QLoRA 的全称是 **Quantized LoRA**，即"量化版 LoRA"。它的核心思路是：**先把 FP16 的基础模型压缩成 4bit 精度，再在上面挂 LoRA 参数进行训练**。

流程可以这样理解：

```text
FP16 模型（占 14GB）
    ↓ 4bit 量化
4bit 基础模型（显存大幅下降）
    +
LoRA 参数（可训练）
    ↓
开始训练
```

**关键点**：量化的是**基础模型**，不是 LoRA 参数。LoRA 参数仍然

## 块 5

## 电脑配置批注：你的LoRA/QLoRA训练环境要什么？

先给你一个总览：LoRA和QLoRA都能在本地训练，但QLoRA的核心优势是**用更小的显存跑大模型**。下面按“先LoRA、再QLoRA”的顺序帮你理清配置要求。

### LoRA最低配置

- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA GPU
- **显存**：最低8GB，推荐12GB+

> 动手提示：如果你手头只有8GB显存，LoRA也能跑，但建议从短序列、小batch开始试。

### QLoRA要求

**GPU**：本教程实训范围为NVIDIA CUDA GPU。注意——QLoRA并非技术上只能用NVIDIA，但其他后端请以bitsandbytes官方说明为准，本文命令按NVIDIA CUDA给出。

**显存**：QLoRA通过4-bit量化基础模型来降低显存占用，通常**低于同规模LoRA全精度加载**。以下是7B级别模型的**经验参考值**（不是硬保证）：

- 参考下限：约8~12GB（7B级别，短序列、小batch）
- 推荐：16GB+

### 模型许可证

本教程引用的基础模型采用 **Llama 2 Community License**：商用月活>7亿需Meta授权。

###

## 块 6

### 5.1 Python版本检查

在开始LoRA微调之前，先确认你的Python环境。打开终端，运行：

```bash
python --version
```

只要输出是 **Python 3.10或更高版本**，就满足要求。如果版本过低，建议先升级Python再继续。

---

### 5.2 PyTorch检查（关键步骤）

LoRA/QLoRA依赖PyTorch，而且`bitsandbytes`的量化功能要求安装**与你的CUDA版本匹配的GPU版PyTorch**。

> ⚠️ **常见坑**：直接执行`pip install torch`通常装到CPU版，QLoRA会无法使用。

**正确做法**：前往 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)，选择你的CUDA版本对应的安装命令。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证PyTorch是否能看到GPU**，在Python中运行：

```python
import torch

print(torch.__version__)        # 查看PyTorch版本
print(torch.cuda.is_available()) # 是否可用CUDA
print(torch.version.cuda)        # torch编译用的CUDA版本
```

如果`torch.cuda.is_available()`输出`True`，说明PyTorch侧能识别GPU。

> 📌 **注意**：这里返回`True`只代表PyTorch能用GPU，**不代表**`bitsandbytes`的4-bit量化路径可用。后者还需要额外检查，后续章节会专门验证。

---

### 5.3 查看GPU信息

在终端运行：

```bash
nvidia-smi
```

输出会包含：
- **GPU型号**（如RTX 4090）
- **CUDA版本**（如CUDA 12.x）
- **显存大小**（如24GB）

记下这些信息，后续安装`bitsandbytes`时需要确认匹配。

---

### 6.1 PEFT是什么？

PEFT是Hugging Face官方的**参数高效微调库**，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter等方法。你的目标是LoRA微调，所以PEFT是核心工具。

---

### 6.2 安装PEFT

```bash
pip install peft
```

验证安装：

```python
import peft
print(peft.__version__)
```

**设备设置建议**：

## 块 7

**本块学习检查**：读完本块后，请合上资料，用自己的话说出 `bitsandbytes` 在本教程中的三个核心用途，并回答：为什么本教程必须安装它？

---

### 7.1 bitsandbytes作用

`bitsandbytes` 是一个Python库，在本教程中承担三项关键任务：

1. **8bit量化**：将模型权重压缩到8位精度，大幅减少显存占用。
2. **4bit量化**：进一步压缩到4位精度，是QLoRA的核心基础。
3. **QLoRA训练**：在4bit量化模型上直接进行LoRA微调，实现高效训练。

**为什么必须安装？** 因为本教程采用基于 `bitsandbytes`

## 块 8

**动手前先确认环境：为什么必须查CUDA？**

你的目标是跑LoRA微调，而bitsandbytes是量化工具，它依赖CUDA来调用GPU。如果CUDA版本不匹配，后果很直接：CUDA不可用、量化失败、加载模型报错。所以，这一步是“防坑”关键。

**两步检查，30秒搞定**

1. **看GPU驱动支持的CUDA版本**  
   终端运行：
   ```bash
   nvidia-smi
   ```
   在输出右上角找到 `CUDA Version`，记下这个数字。

2. **确认bitsandbytes能识别你的GPU**  
   终端运行：
   ```bash
   python -m bitsandbytes
   ```
   如果正常，

## 块 9

**本块学习检查**：请用一句话说明，为什么在Windows上遇到bitsandbytes加载CUDA动态库失败时，优先推荐改用WSL2？

---

**9.1 推荐使用WSL2（动手实践路线）**

按以下顺序搭建环境，能避开大部分坑：

```text
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

**为什么？** 核心原因就一条：**Linux环境兼容性更好**。WSL2让你在Windows里跑一个真实的Linux子系统，后续装CUDA、跑bitsandbytes时，行为更接近标准Linux环境。

---

**9.2 Windows原生可能遇到的问题（先看再动手）**

如果你坚持在Windows原生环境装，最常撞见的是：

**问题1：bitsandbytes无法加载CUDA动态库**

这不是单一原因造成的，常见因素包括：

- 你装的`bitsandbytes`

## 块 10

## 十三、QLoRA环境验证实验

测试bitsandbytes是否正常工作：

```python
import torch
import bitsandbytes as bnb

print("CUDA:", torch.cuda.is_available())

# 测试4-bit量化是否可用
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("QLoRA 4-bit配置:", quant_config)
```

运行：

```bash
python test_lora_env.py
```

**成功标准**：看到`CUDA: True`且能打印出QLoRA 4-bit配置信息，说明✅ PyTorch正常、✅ bitsandbytes正常、✅ QLoRA配置可用。

## 块 11

## 九、Windows用户特别注意事项：推荐使用WSL2

在Windows上运行LoRA/QLoRA环境时，你可能会遇到兼容性问题。官方推荐的解决方案是使用**WSL2**（Windows Subsystem for Linux 2）。

### 为什么推荐WSL2？

在Windows原生环境下，`bitsandbytes`等库可能因为CUDA兼容性或系统差异而报错。WSL2提供了一个更接近Linux的环境，能有效规避这些兼容性问题，让你更顺畅地完成环境配置。

### 常见问题速查

| 问题 | 原因 | 解决 |
| ---- | ---- | ---- |
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch安装CPU版本 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或使用更小模型 |
| Windows报错 | 兼容性问题 | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

### 本块学习检查

对照上表，检查你的当前环境：确认PyTorch是否为CUDA版本（非CPU版），并记录你的CUDA版本号。若遇到Windows报错，请直接切换到WSL2环境继续后续操作。

## 块 12

## 一、为什么需要LoRA？（①②③）

### 1.1 大模型微调的问题

现代大语言模型（如LLaMA、Qwen、Baichuan、Mistral）通常拥有数十亿甚至上百亿参数。以7B模型为例，其总参数量约70亿。

如果采用传统方式微调——把预训练模型拿过来，更新全部参数，再保存新模型——你会遇到三个现实问题：

- **巨大的GPU显存**：训练时显存需求包括模型参数、梯度、优化器状态（如Adam的动量项），实际显存需求远大于模型大小；
- **大量训练时间**：更新所有参数意味着每步计算量巨大；
- **巨大的存储空间**：每个微调版本都要保存一份完整模型。

### 1.2 全参数微调的问题

假设模型有100亿参数。训练时你不仅需要保存模型参数W，还需要保存：

- **梯度**：∂W，用于反向传播；
- **优化器状态**：如Adam的动量项。

实际显存需求远大于模型本身大小。这就是为什么全参数微调对大模型来说代价高昂。

---

## 二、LoRA核心思想（①②）

LoRA的核心思路是：**冻结原始权重W，只学习一个小的增量**。

具体来说，预训练模型的权重W在微调时保持冻结（不动），我们只学习一个增量ΔW，最终

## 块 13

### 2.1 不直接修改原模型参数

**传统微调**是直接改原权重：`W → W'`，也就是把大模型原来的参数`W`整个更新掉。

**LoRA的做法**完全不同：

- 把原模型的`W`**冻结**住，训练时不动它；
- 额外新增一个参数`ΔW`；
- 最终模型用的是：`W' = W + ΔW`。

简单说：**原模型参数`W`保持不变，只训练新增的`ΔW`**。

---

### 2.2 为什么只训练ΔW？

大模型已经通过海量数据学会了语言规律、世界知识、推理能力。微调通常只是**对已有能力做方向调整**，不必重新学全部参数，只学任务相关的变化量往往就够了。

> 注意：这是教学直觉，帮助理解LoRA为何可行，**不是严格结论**。任务与预训练能力差距大、数据充足时，全参数微调仍可能更好。

**核心思路**：只学习`任务相关变化量`。

---

## 块 14

## 3.1 矩阵基础概念

Transformer里大量参数都是矩阵。比如注意力层：`Q = X · Wq`，其中 `Wq` 就是一个权重矩阵。假设 `Wq` 是 `4096 × 4096`，参数数量就是 `4096 × 4096 ≈ 1677万`。

## 3.2 直接训练矩阵的问题

如果直接修改 `ΔW`，仍然需要 `4096 × 4096` 个参数。LoRA的思路是：**不要直接学习完整矩阵，而是拆成两个小矩阵**。

## 四、LoRA低秩分解数学原理（重点）

LoRA核心公式：原始 `W` 保持冻结，新增 `ΔW` 由 `B × A` 组成。

---

**本块学习检查**：请你口述回答——`Wq` 是 `4096 × 4096` 时，直接训练 `ΔW` 需要多少个参数？LoRA用什么方式避免直接学习这个大矩阵？

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设原矩阵是 **4096 × 4096**，直接训练意味着要更新全部参数，数量为：

**4096 × 4096 = 16,777,216**（约1677万）。

LoRA的思路是**不直接动这个大矩阵**，而是冻结原权重W，将增量ΔW拆成两个小矩阵的乘积。设 **rank r = 8**，拆成：

- **A**：形状 **8 × 4096**，参数 = 8 × 4096 = **32,768**
- **B**：形状 **4096 × 8**，参数 = 4096 × 8 = **32,

## 块 16

**5.1 神经网络参数变化具有低维特征**

想象一下，当模型学习新任务时，它的参数会发生变化，这个变化记作 **ΔW**。LoRA 论文的核心观点是：**ΔW 并不需要覆盖所有方向**。

这就像在一个 4096 维的空间里，真正影响任务的“有效方向”其实很少，它们集中在一个**低维子空间**中——也就是**远小于原维度的少数方向**。

> 注意：这里说的是“权重更新往往具有较低的内在维度”这一**假设**。具体有效维度是多少，没有固定值，会随模型、任务和数据变化。我们只需定性理解，不必记成“几十维”这样的确定数字。

因此，我们可以用一个**低秩矩阵 BA** 来近似表示 **ΔW

## 块 17

## 5.2 类比理解：为什么LoRA用少量参数就能微调大模型？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。但如果这张图片的主要变化只是**亮度、对比度**和**少量结构变化**，你根本不需要存下全部1600万个像素——用**少量参数**就能描述这些变化。

LoRA的思路完全一样：**用少量参数（低秩矩阵A和B）描述模型权重的变化量（增量ΔW），原权重W保持不变**，而不是重新训练整个模型。需要特别说明的是：LoRA**冻结原始权重W**，仅训练低秩矩阵A和B，通过增量ΔW=(α/r)BA更新模型，**并非对W本身进行分解或变换**。

---

## LoRA在Transformer中加在哪里？

Transformer主干结构是：

```text
Input → Embedding → Attention → FFN → Output
```

其中Attention里有 **Q、K、V、O** 四个投影矩阵。LoRA早期实践**常见**的做法是只加到 `q_proj` 和 `v_proj` 上：

```python
target_modules=["q_proj", "v_proj"]
```

意思是：只训练 `Wq` 和 `Wv` 的变化量。

> ⚠️ 注意：这只挑 `q_proj`/`v_proj` 是LoRA原论文的**常见选择**，**不是固定标准**。现代实践经常把LoRA加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选哪些模块会影响可训练参数量、显存和效果，要结合模型结构（先确认实际模块名）和验证集表现来定。

下面是一个**覆盖更完整线性层**的配置示例（以Llama类模型常见模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力投影
        "gate_proj", "up_proj", "down_proj",       # FFN（MLP）线性层
    ],
)
```

> 上例是**显式列出模块名**的写法；若当前PEFT版本支持，也可用 `target_modules="all-linear"` 自动匹配所有线性层（不含输出lm_head等，具体以版本文档为准）。覆盖模块越多，可训练参数、

## 块 18

**r 是什么？**
`r`（rank）控制 LoRA 低秩矩阵的维度。代码里写 `r=8`，意思是把权重更新量压缩成一个 8 维的低秩空间。

**r 越大越好吗？**
不是。`r` 大，表达能力更强，但参数更多、显存占用更高。`r` 小则相反。这是个权衡。

**新手起步参考（仅作起点，非定死规则）**
- 简单任务：`r` 约 4–8
- 普通微调：`r` 约 8–32
- 复杂任务：`r` 约 64+

合适的 `r` 和模型规模、目标模块、数据量、任务难度都有关，别死记数字。

**更可靠的做法：用验证集选 `r`**
1. 从数据里留出验证集（如 8:2 或 9:1），所有比较都在它上面看指标。
2. 固定 `target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数，只改 `r`，做受控对比。
3. 从小值起步，如 `r ∈ {4, 8, 16, 32}`，逐个训练并在验证集上记录 loss/accuracy/F1、显存和耗时。
4. 看边际收益：若 `r` 增大后验证集提升很小，或开始过拟合（训练指标升但验证指标降），说明再加大不划算，取“性价比拐点”。
5. 结合成本决策：效果接近时

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个超参数，它直接控制 LoRA 对模型权重的**更新强度**。简单说，它决定了你微调时“改模型的力度”有多大。

**代码示例：**
```python
lora_alpha=16
```

**核心机制：**
LoRA 冻结原始权重 W，仅训练低秩矩阵 A 和 B。其更新量为 ΔW = (α/r)BA，缩放比例 α/r 应用于这个增量，而非直接缩放原始权重。该缩放比例决定了 LoRA 参数对原始模型的影响程度。

**直观理解：**
- `alpha` 越大 → 缩放比例越大 → LoRA 对模型的修改越“猛烈”
- `alpha` 较小 → 缩放比例较小 → 更新更温和，模型变化更细微

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 的作用是**防止LoRA参数过拟合**。它通过在训练过程中随机“丢弃”一部分 LoRA 参数（将其置零），迫使模型不过度依赖某些特定参数，从而提升泛化能力。`0.05` 表示每次训练时有 5% 的 LoRA 参数被随机丢弃。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 |

## 块 20

## 二、Rank（r）是什么？

### 2.1 Rank表示什么？

**一句话理解**：Rank（中文叫“矩阵的秩”）在LoRA里，就是**低秩矩阵中间维度的大小**。它决定了你要用多大的“压缩通道”去模拟权重更新。

**用数字感受一下**：

假设原始权重矩阵是：

```text
W: 4096 × 4096
```

这是一个巨大的矩阵（约1600万参数）。如果直接微调它，成本极高。

现在设置：

```python
r=8
```

LoRA会把这个大更新拆成两个小矩阵：

```text
A: 4096 × 8
B: 8 × 4096
```

**关键点来了**：这两个矩阵相乘，结果仍然是：

```text
ΔW = (α/r) × B × A
```

恢复成：

```text
4096 × 4096
```

其中α/r是缩放因子，用于控制更新幅度。

**但实际需要训练的参数**，从原来的1600万，变成了：

```text
r×(d_in+d_out) = 8×4096 + 8×4096 = 65536
```

**少了约244倍**（1600万 ÷ 65536 ≈ 244）。

---

**动手验证一下**（帮助你建立直觉）：

在PyTorch里跑一下这段代码，观察输出形状：

```python
import torch

r = 8
W_dim = 4096

A = torch.randn(W_dim, r)   # 形状: 4096×8
B = torch.randn(r, W_dim)   #

## 块 21

**Rank控制什么？——用“变化方向”来理解**

把Rank想成一个“旋钮”，它决定LoRA能学习**多少种变化方向**。

**低Rank（如r=4）**：模型只能学较少的变化模式。好处是参数少、显存低、训练快；代价是表达能力有限。

**高Rank（如r=64）**：LoRA拥有更多自由度，能学更复杂任务、适应能力更强。代价是参数更多、显存增加、训练成本提高。

**动手验证**：在你本地跑通LoRA微调时，分别用`r=4`和`r=64`训练同一个任务，对比训练速度和最终效果差异。

## 块 22

**动手算一算：不用LoRA要多少参数？**

假设模型某层权重是 **4096×4096** 的矩阵。

直接训练它，参数量就是：

```text
4096 × 4096 = 16,777,216
```

约 **1677万** 个参数。

**关键感受**：这只是**一层**的参数量。完整微调整个模型，要更新的参数远不止这些，训练开销巨大。

**对比一下**：如果改用LoRA，同样处理4096维的输入和输出，参数量公式是：

```text
参数量 = r × (输入维度 + 输出维度)
```

- 当 **r=8** 时：`8 × (4096+4096)

## 块 23

### 4.1 Rank太小的问题

如果你把 `r` 设得很小，比如 `r=2`，会发生什么？

**直接后果**：模型的学习能力不足，容易欠拟合，新任务的知识没法充分表达出来。

**你会看到的现象**：
- 训练时 loss 下降得很慢；
- 最终效果提升有限。

**怎么理解**：`r` 就像给模型配的“记忆容量”。容量太小，它记不住新任务的关键模式，自然学不好。

---

### 4.2 Rank太大的问题

反过来，如果你把 `r` 设得很大，比如 `r=256`，又会怎样？

**直接后果**：
- 参数数量增加；
- 显存占用增加；
- 训练速度下降。

**更关键的是**：LoRA 的“参数高效”优势会被削弱。

**用数字看清楚**：假设权重矩阵是 `4096 × 4096`（完整参数量 16,777,216），LoRA 的参数量是 `r × (4096 + 4096)`。

| rank r | LoRA 参数量 | 占完整矩阵比例 |
| ---: | ---: | ---: |
| 8 | 65,536 | ≈ 0.39% |
| 64 | 524,288 | ≈ 3.13% |
| 256 | 2,097,152 | ≈ 12.5% |

**注意**：即使 `r=256`，LoRA 参数也只占完整矩阵的约 1/8，远没到“接近完整矩阵”的程度。它的问题不是“逼近满秩”，而是相对 `r=8` 来说，参数占比明显变大，显存和训练成本都升高了，所以“参数高效”的优势被削弱了。

---

**本块学习检查**：动手算一下，如果

## 块 24

### 4.3 常见rank选择：从哪开始？

你是动手实践型学习者，目标是把LoRA跑起来。下面这张表是**经验起点**，不是标准答案：

| 任务类型 | rank（起点参考） |
| ------ | ------ |
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**关键提醒**：别直接照搬。实际项目中，`r` 要根据数据量、模型大小、序列长度、batch大小、目标模块选择、GPU显存以及验证集表现来调。同一个 `r`，在不同模型或序列长度下，显存和效果差异可能很大。

---

### 五、Alpha（lora_alpha）是什么？

LoRA 更新权重时

## 块 25

## 6.1 控制LoRA影响强度——动手调一调

你已经知道LoRA通过低秩矩阵来微调模型，那怎么控制它“影响多大”呢？关键就在 `alpha` 和 `rank` 的比值。

**先看一个具体例子：**

假设你设置：

```python
r=8      # rank
alpha=16 # alpha
```

缩放系数就是：

```text
16/8 = 2
```

这意味着：LoRA 的低秩增量 ΔW 会被**放大2倍**，即 ΔW = (alpha/r) * B @ A，从而增强LoRA对模型的影响。

**再试另一个组合：**

```python
alpha=8
```

那么：

```text
8/8 = 1
```

增量保持**原始比例**，不放大也不缩小。

**如果反过来：**

```python
alpha=4
```

那么：

```text
4/8 = 0.5
```

LoRA 的影响就被**减弱**了，低秩增量只有原来的一半。

---

## 七、Alpha与Rank的关系（重点）

记住这个核心公式：

```text
alpha/r
```

这个比值就是 LoRA 低秩增量的**关键缩放系数**。`alpha` 越大（相对 `rank`），LoRA 影响越强；`alpha` 越小，影响越弱。

**但注意**：`alpha/r` 不是“唯一决定实际强度的开关”。实际落到权重上的

## 块 26

**配置1 vs 配置2：缩放比例的秘密**

先看两组配置：

- 配置1：`r=8`，`alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16`，`alpha=32` → 比例 = 32 ÷ 16 = **2**

发现了吗？虽然 rank 从 8 变成了 16，但 `alpha / r` 的缩放比例始终是 **2**，没变。

**为什么要保持比例稳定？**

调整 rank 时，常见做法是**同步调整 alpha**，让 `α/r` 大致稳定。这只是**经验做法**，不是硬性规则。`alpha = 2 × r` 也只是常见

## 块 27

**本块学习检查**：请对照下方“方案1”配置，在代码中找出 `r` 和 `alpha` 分别控制什么，并说出它们各自的作用。

---

### 方案1：小模型实验（推荐你先试这个）

配置：

```python
r=8
alpha=16
```

特点：参数少、训练快，适合测试。因为你刚开始学LoRA，先用小配置跑通流程，再逐步加大。

---

### 对应代码示例

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

三个关键参数的作用：

- **r**：控制LoRA容量（即“记忆空间”大小）
- **lora_alpha**：控制LoRA更新幅度（即“每次调整的力度”）
- **target_modules**：控制LoRA插入位置（即“改模型的哪些部分”）

---

###

## 块 28

## 误区1：rank越大一定越好？

**错误。**

很多初学者容易陷入“参数越多越强”的直觉，但在LoRA里，**rank不是越大越好**。

关键判断标准是**任务复杂度**：

- 任务简单时，大rank会带来两个实际代价：
  - **浪费显存**：更大的rank意味着更多可训练参数，占用更多显存；
  - **增加训练成本**：计算量和显存开销同步上升，训练更慢。

**实操建议**：先从小rank（如r=8）开始，观察验证集表现。如果模型表达能力不足（欠拟合），再逐步增大rank。不要一上来就堆大rank。

---

## 误区2：只调rank，不调alpha

很多人改rank时忘了同步调整alpha，导致比例失衡。

**示例**：

原配置：
```python
r=8
alpha=16
```

修改后（只改rank）：
```python
r=64
alpha=16
```

此时缩放比例变为：
```text
16/64 = 0.25
```

对比原来的 `16/8 = 2`，缩放比例从2骤降到0.25，**LoRA对模型更新的影响被大幅削弱**，微调效果可能反而变差。

**核心原则**：`alpha` 和 `rank` 是配套的，改rank时必须重新考虑alpha，保持合理的 `α/r` 比例。

---

## 误区3：alpha越大越好？

**错误。**

alpha过大的本质是 `α/r` 缩放过大，会导致：

- **更新过强**：每次参数更新的幅度过大；
- **训练不稳定**：loss可能出现震荡甚至发散。

**重要提醒**：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡/发散还常来自：

- 学习率过高
- 数据质量差
- 有效batch size过小
- 精度（fp16/bf16）与数值稳定性问题
- 缺少梯度裁剪

排查不稳定时，应**综合排查**以上因素，不要只盯着

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。在 LoRA 微调流程中，`LoraConfig` 是核心配置对象，它决定了 LoRA 的插入位置、规模、训练强度和正则化方式。

**LoRA 训练流程：**

```
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

`LoraConfig` 负责告诉 PEFT 四件事：

1. **LoRA放在哪里** —— 通过 `target_modules` 指定要插入 LoRA 的模块（如注意力层的 `q_proj`、`v_proj`）
2. **LoRA规模多大** —— 通过 `r`（秩）控制低秩分解的维度，`r` 越大，可学习的参数量越多，模型容量越大
3. **LoRA训练强度如何** —— 通过 `lora_alpha` 控制权重更新时的缩放比例，`lora_alpha` 越大，LoRA 对原始权重的修改幅度越强
4. **是否使用dropout** —— 通过 `lora_dropout` 在训练时随机丢弃部分 LoRA 参数，防止过拟合

---

**本块学习检查

## 块 30

## 1.2 一个简单示例——动手敲一遍

你是个喜欢动手的人，那就直接看代码。下面这段就是用 PEFT 库配置 LoRA 的最简写法：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,                # LoRA矩阵大小
    lora_alpha=16,      # 缩放强度
    target_modules=["q_proj", "v_proj"],  # 插入哪些层
    lora_dropout=0.05   # 随机失活比例
)
```

**先别急着往下读，把这段代码自己敲一遍，然后回答：**
- 如果我想让 LoRA 作用在更多层上，应该改哪个参数？
- `r=8` 和 `lora_alpha=16`，你觉得哪个控制"矩阵大小"？

---

## 二、LoRA配置整体流程——参数怎么用起来

光有 `config` 还不够，它要配合模型一起用。流程是：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

完整代码：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些部分动手脚”**。

在 Transformer 模型里，Attention 结构通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

**关键点**：LoRA 不会默认修改所有层，你必须自己指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思就是：**只对 Query 投影层和 Value 投影层应用低秩增量更新**。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开理解两个角色：

- **Q（Query）**：决定模型**关注什么信息**
- **V（Value）**：决定模型**传递什么信息**

所以，通过低秩增量ΔW=(α/r)BA调整Q和V投影层的权重，从而改变模型行为——这就是为什么它们是最常见的 LoRA 注入起点。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名可能不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 `target_modules`。

---

## 偷懒技巧：`target_modules="all-linear"`

如果你不想手动枚举模块名，较新版本的 PEFT（一般 ≥ 0.7）支持一个便捷写法：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your/base-model")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",   # 自动匹配（几乎）所有 nn.Linear
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比
```

**用 `"all-linear"` 的注意点**：

- 它会自动把 LoRA 加到模型中（几乎）所有线性层（通常**不含** `lm_head` 等输出层）
- 覆盖模块变多 → **可训练参数和显存占用会增加**
- 如果版本太旧不支持，就退回**显式列出模块名**的写法
- 用 `print_trainable_parameters()` 快速验证注入是否符合预期

**怎么选？** 在**参数量/显存**和**效果**之间权衡，最终以验证集结果为准。

## 块 32

**动手试一下：只调Q和V**

先看最简单的配置。在`LoraConfig`里，把`target_modules`设成只包含`q_proj`和`v_proj`：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这样做的直接好处是**参数少**，所以**训练快**。对于刚开始做LoRA微调、想快速跑通流程的场景，这是一个常见的轻量起点。注意，它不限于某个特定模型规模，具体效果还是要看模型结构和实际任务表现。

**本块学习检查**：打开你的训练脚本，把`target_modules`改成上面这个列表，跑一个极小的训练步数（比如几十步），观察显存占用和每步耗时。然后对比一下，如果换成下面这个“全Attention”配置，显存和耗时有什么变化：

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "

## 块 33

## LLaMA系列：target_modules 怎么配？

LLaMA 系列模型（如 Llama 2、Llama 3）的注意力层中，Q、K、V投影是分开的独立模块，分别叫 `q_proj`、`k_proj`、`v_proj`。常见的 `target_modules` 写法包括 `["q_proj", "v_proj"]` 或 `["q_proj", "k_proj", "v_proj", "o_proj"]`，具体选择取决于任务需求。

**为什么只选这两个？** 在 LoRA 微调中，我们不需要对每一层都注入低秩矩阵。实践表明，只对 Q 和 V 投影做低秩适配，在某些任务上能取得不错的效果，同时减少可训练参数量。但具体选择需根据任务验证，有时更新更多层（如K、O）效果更佳。

---

## 对比：GPT-2 的 `c_attn` 与 `fan_in_fan_out`

作为对照，GPT-2 的注意力实现和 LLaMA 不同：它把 Q/K/V 三个投影**合并**在一个叫 `c_attn` 的模块里，使用的是 Transformers 中的 `Conv1D`（不是普通的 `nn.Linear`）。

关键区别在于**权重布局**：

- `nn.Linear` 的权重形状是 `(out_features, in_features)`
- `Conv1D` 的权重布局与 `nn.Linear` 不同，因此需要设置 `fan_in_fan_out=True` 以正确注入 LoRA

可运行示例（GPT-2）：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    l

## 块 34

## BERT的target_modules配置

BERT模型和LLaMA不同，**不能直接照搬**`["q_proj", "v_proj"]`这类配置——虽然BERT确实常配这两个，但不同模型的模块命名差异很大。

**正确做法**：先查看你实际加载的模型结构，再决定配什么。

```python
for name, _ in model.named_modules():
    print(name)
```

运行后会看到类似`bert.encoder.layer.0.attention.self.query`这样的名字，末尾的`query`/`value`就是你要找的。然后配置：

```python
target_modules=["query", "value"]
```

**匹配机制**：`target_modules`按模块名做**后缀匹配**——写`"query"`会命中所有以`query`结尾的层，各层各头都会生效。也可以传完整名字精确定位，或传正则表达式。

**注意**：不同Transformers/PEFT版本、不同模型实现，命名可能变化。务必以你实际加载模型的`named_modules()`输出为准。

---

**本块学习检查**：加载一个BERT模型，运行上面的代码，把输出中`attention`相关的模块名抄下来，确认`query`和`value`是否存在。

## 块 35

**r是什么？——用一个小旋钮理解LoRA的“容量”**

想象你要微调一个4096×4096的大矩阵（原矩阵）。直接改它，参数太多、显存爆炸。LoRA的做法是：不直接改原矩阵，而是训练两个小矩阵A和B，让它们的乘积近似原矩阵的变化量，即 **ΔW = (α/r)BA**，其中α为缩放系数，r为秩，二者共同控制更新幅度。

这里的 **r（rank，秩）**，就是控制A、B“中间维度”大小的旋钮。

- 设 **r=8**，则A的尺寸是 **4096×8**，B是 **8×4096**。  
- r越大，A

## 块 36

### 7.1 alpha作用

LoRA最终更新不是 `BA`，而是 `(alpha/r)BA`。

**alpha控制LoRA更新幅度。**

例：`r=8`，`lora_alpha=16`，缩放为 `16/8=2`，表示LoRA影响增强。

### 7.2 alpha常见设置

常见起点：`alpha = 2 × r`（仅为经验值，非通用规则，最终以验证集表现为准）。

例：`r=8, alpha=16` 或 `r=16, alpha=32`。

---

**动手验证**：修改`lora_alpha`为`r`的1倍和4倍，对比微调后模型在验证集上的表现差异，观察更新幅度的影响。

## 块 37

**8.1 dropout作用：防止过拟合**

训练时，`lora_dropout` 会随机将LoRA低秩矩阵中的部分元素置零，以减少**过拟合**（模型死记训练数据，泛化变差）。

配置示例：
```python
lora_dropout=0.05
```
含义：5%的LoRA矩阵元素（或激活）被随机置零。

---

**8.2 如何选dropout？**

经验起点（非固定，需结合验证集是否过拟合调整）：

| 数据规模 | dropout起点 |
|---------|------------|
| 数据少   | 约0.05~0.1 |
| 数据多   | 约0~0.05   |

若

## 块 38

## 9.1 基础配置：动手搭一个LoRA

这份配置适合在7B模型上做测试，也是你上手LoRA微调的第一块积木。我们直接看代码，然后逐行拆解。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**每个参数在干什么？**

| 参数 | 值 | 作用 | 你的理解角度 |
|------|-----|------|-------------|
| `r` | 8 | LoRA容量 | 秩的大小，决定注入的低秩矩阵能“记住”多少新知识。8是7B模型测试的常用起点 |
| `lora_alpha` | 16 | 更新缩放 | 控制微调时权重更新的幅度。alpha/r = 2，即更新强度是容量的2倍 |
| `target_modules` | `["q_proj", "v_proj"]` | 插入位置 | 只改造注意力层的Q和V投影矩阵，这是最经典的轻量改法 |
| `lora_dropout` |

## 块 39

## 9.1 基础配置：一次跑通 LoRA 微调的最小完整流程

你现在的目标是 LoRA 微调，而且喜欢动手实操。这一节我们就直接进入代码，把“加载模型 → 配置 LoRA → 训练 → 保存 → 重载验证”这条完整链路跑通。代码里每一步都有注释，建议你边看边在本地环境跟着敲。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：Apache 2.0（可商用，无额外授权限制）
# 注意：bert-base-uncased FP16 推理约需 1.3GB 显存，训练时按 batch size 相应增加
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里用的是 `bert-base-uncased` 做二分类任务。注意：分类头是**新初始化**的，如果不管它，训练时它不会更新，最终结果就是乱的。所以后面要用 `modules_to_save` 把它纳入训练。

### 第二步：配置 LoRA 参数

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",                 # 序列分类任务
    target_modules=["query", "value"],   # BERT 注意力模块名
    modules_to_save=["classifier"],       # 关键：把新分类头也纳入训练并保存
)
```

逐个解释你关心的参数：

- `r=8`：LoRA 的低秩维度。越小参数越少，但表达能力也越弱；8 是常用起点。
- `lora_alpha=16`：缩放系数。实际生效的缩放比例是 `alpha / r`，这里就是 2 倍。
- `lora_dropout=0.05`：防止过拟合的 dropout 比例。
- `target_modules=["query", "value"]`：指定在 BERT 的哪些模块上挂 LoRA。这里只改了注意力里的 query 和 value 矩阵。
- `modules_to_save=["classifier"]`：**这一行是本块的重点**。它让分类头参与训练，并且保存时和 Adapter 一起存下来。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

执行后你会看到输出里同时包含 LoRA 参数和 classifier 参数，说明两者都在训练范围内。

### 第四步：保存与重载验证

训练完成后（代码里省略了训练过程），保存并验证：

```python
# 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")

# 重新加载并验证分类头确实被带回（而不是随机初始化）
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证：重载后的分类头权重应与保存前一致
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
```

LoRA 的原理是冻结原始权重，仅训练低秩矩阵 A 和 B，通过 A×B 形成增量更新加到原始权重上，而不是对原始权重做分解。

## 块 40

## 十、普通文本生成配置（LoRA 微调起步）

你现在的目标是掌握 LoRA 微调，而且喜欢动手实践。那我们从最直接的配置开始——**普通文本生成**任务，这是理解 LoRA 参数的最佳切入点。

### 推荐配置

```python
r=8
alpha=16
dropout=0.05
```

### 参数怎么理解？

- **`r`（秩）**：决定 LoRA 引入的可训练参数量。`r=8` 意味着每个被适配的权重矩阵被分解为两个低秩矩阵，秩为 8。值越小，参数量越少，训练越快，但表达能力受限。
- **`alpha`（缩放因子）**：控制 LoRA 更新对原模型权重的影响强度。`alpha=16` 配合 `r=8`，缩放比例为 `alpha/r = 2`，即 LoRA 的更新会被放大 2 倍。
- **`dropout`**：在 LoRA 层中随机丢弃一部分神经元，防止过拟合。`0.05` 是较低的值，适合数据量适中的场景。

> **动手验证**：如果你在 CPU 上运行，只配置 `LoraConfig` 并打印出来看参数值，内存 4GB 以上就够。如果加载真实大模型执行 `get_peft_model()`，则需要 NVIDIA CUDA GPU，显存最低 8GB，推荐 16GB+。

### 不同任务的参数对比

| 任务 | r | alpha | dropout |
|------|---|---|---------|
| 普通文本生成 | 8 | 16 | 0.05 |
| 指令微调 | 16 | 32 | 0.05 |
| 领域知识增强 | 32 | 64 | （未指定） |

规律很清晰：**任务越复杂、越需要模型学习新知识，`r` 和 `alpha` 就越大**。指令微调需要模型学会遵循指令，领域知识增强需要注入大量新知识，所以参数随之增大。

### 扩展边界（知道存在即可）

`LoraConfig` 还支持一些扩展能力，本文不展开，但你要知道它们存在，避免以为 LoRA 只有上面几个参数：

- `use_rslora=True`：

## 块 41

**本块学习检查**：对照下面的配置，你能说出每个参数在“改哪里、容量多大、更新多强、防过拟合”中分别扮演什么角色吗？

```python
LoraConfig(
    r=8,              # LoRA容量：8维低秩矩阵
    lora_alpha=16,    # 更新强度：16/8=2倍缩放
    target_modules=["q_proj", "v_proj"],  # 只改注意力层的Q和V投影
    lora_dropout=0.05 # 5%随机丢弃，缓解过拟合
)
```

**动手验证**：把 `r` 改成4，`lora_alpha` 保持16，α/r 缩放比

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经知道，LoRA只训练一小部分参数——原模型参数被冻结，只有LoRA参数在更新。但这里有个关键点：**基础模型本身仍然要完整加载到显存里**。

我们算一笔账。以7B模型为例，如果用FP16（每个参数占2字节）加载：

```
7B × 2 bytes ≈ 14GB显存
```

这只是模型权重本身。实际训练时，显存里还要同时放：

- **激活值**：前向传播过程中每层的中间输出；
- **梯度**：反向传播时计算出的参数更新量；
- **优化器状态**：如Adam的动量项和方差项。

把这些都加起来，实际显存需求远高于14GB。消费级显卡（比如24GB显存）即使能勉强放下模型，也几乎不可能完成微调。

**QLoRA要解决的核心问题就是：能不能在不牺牲太多效果的前提下，把模型加载的显存开销大幅降下来？**

答案是：把模型从FP16降到4-bit精度。这样7B模型的权重占用就从约14GB降到约3.5GB，省出来的空间留给

## 块 43

**QLoRA核心思想（动手版）**

QLoRA = **Quantized LoRA**，一句话：**先把基础模型压成4-bit，再在上面挂LoRA训练**。

**流程（跟着做一遍）**
1. FP16基础模型 → 2. 4-bit量化 → 3. 加载量化模型 → 4. 插入LoRA → 5. 训练LoRA参数

**和LoRA的区别**
- LoRA：FP16模型 + LoRA
- QLoRA：4-bit模型 + LoRA

**关键结构**
- 基础模型：4-bit权重，**冻结**（不训练）
- LoRA：FP16/BF16参数，**训练**
- 数据流：输入 →

## 块 44

## 3.1 模型存储方式：从“重量”看显存

模型参数在硬盘/显存里占多大地方，取决于每个参数用多少字节（Byte）存。核心对照表：

| 类型 | 每参数占用 |
|---|---|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下（7B模型，即70亿参数）：**

- **FP16**：7B × 2 ≈ **14GB**
- **4-bit**：7B × 0.5 ≈ **3.5GB**

直观感受：换成4-bit，权重体积直接**减少约75%**。

> ⚠️ **别高兴太早**：3.5GB只是“理论下界”。实际运行显存会明显更高，因为还要算上量化常数/缩放因子、分组元数据（NF4按组量化，每组都有额外参数）、未量化的模块（如norm、embedding、lm_head、LoRA参数），以及训练/推理时的计算缓存和激活值。**真实占用以实际运行为准**，别拿这个数去配显卡。

## 3.2

## 块 45

**NF4是什么？——用4位存下“正态分布”的权重**

NF4全称NormalFloat4，中文叫“正态分布4位浮点”。但别被名字骗了，它不是IEEE 754那种通用浮点格式，而是QLoRA论文为**神经网络权重**专门设计的**4-bit量化码本**。

**核心思路：**
- 权重通常近似正态分布，0附近数值密集。
- NF4预先定好16个（2⁴）取值点，基于标准正态分布的分位数量化（quantile quantization）计算得到，0附近划分更细，远处更粗。
- 量化时：把每组权重归一化，映射到最接近的码本值，只存4-bit索引。
-

## 块 46

### 4.2 为什么适合模型权重？

大模型训练后的权重近似**正态分布**，大量参数集中在0附近。普通INT4平均划分区间（如[-8,-7,...,7]），无法精细表示中心密集区域。

**NF4**根据标准正态分布的分位数设计非均匀量化级别，使得量化级别在权重分布密集的区域（如0附近）更精细，从而在4-bit精度下更好地保持模型性能。

### 4.3 NF4量化过程

原始FP16权重（如0.023, -0.145, 0.876）经**归一化 → 映射到NF4 → 保存4-bit值**。推理时再反量化回近似FP16。

---

## 五、双重量化

## 块 47

**5.1 为什么需要双重量化？**

4-bit量化时，每组权重除了存4-bit数据，还要额外保存`scale`和`zero point`这两个量化参数。这些参数同样占用显存，模型越大，这笔开销越不可忽视。

**5.2 双重量化思想**

普通量化流程：FP16权重 → 4-bit权重 + scale参数。

双重量化在此基础上，对scale参数再做一次量化：FP16权重 → NF4量化 → 保存4-bit权重；同时scale参数 → 再次量化 → 减少scale存储。本质就是“量化量化参数”。

**5.3 双重量化优势**

核心收益是减少额外量化参数的显存占用。对大模型而言，参数基数巨大，哪怕每个参数只省一点点，累计起来也能释放大量显存。

---

**

## 块 48

**为什么需要分页优化器？——从“显存突然不够”说起**

你在用QLoRA做LoRA微调时，模型本身已经通过4-bit量化瘦身了，但训练时显存占用**不只是模型**。还有两样东西也在吃显存：

- **梯度**：反向传播算出来的更新方向。
- **优化器状态**：以Adam为例，它要为每个参数保存**一阶动量**和**二阶动量**，这两份状态本身就很占空间。

所以，即使QLoRA把模型压小了，训练过程中**仍可能显存突然不足**，直接报OOM（Out of Memory）中断训练。这就是分页优化器要解决的问题。

**分页优化器的核心思想**

它借鉴了操作系统里的“虚拟内存”概念，借助CUDA虚拟内存管理（如bitsandbytes库的paged optimizers），机制如下：

1. 正常训练时，优化器状态放在**GPU显存**里。

## 块 49

## 6.3 类比理解：分页优化器到底在干嘛？

把显存想象成你的**桌面**，把CPU内存想象成**柜子**。

**普通优化器**的做法是：把所有东西（模型参数、梯度、优化器状态）全堆在桌面上。桌面满了，就**无法继续**——这就是你常遇到的显存溢出（OOM）报错。

**分页优化器**的做法是：桌面上只放**常用文件**（当前正在计算的数据），**暂不用的文件**放进柜子保存，**需要时再取出**。这样桌面永远有空间，训练就能继续跑下去。

---

## 6.4 代码里怎么启用？

好消息：**不需要你手写换页逻辑**，bitsandbytes 库已经帮你实现了。你只需要在 `TrainingArguments` 里设置 `optim` 参数：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,     # 配合 QLoRA 进一步降低显存峰值
    # 关键：使用 bitsandbytes 的分页优化器
    optim="paged_adamw_8bit",        # 或 "paged_adamw_32bit"（精度更高、显存更大）
)
```

**两个选项怎么选？**
- `paged_adamw_8bit`：优化器状态用 8-bit 存储，**更省显存**
- `paged_adamw_32bit`：精度更高，但**占用更多显存**

**注意**：不同 Transformers / bitsandbytes 版本支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报"未知 optim"错误，去查你当前版本的 `TrainingArguments` 文档，换用支持的名称即可。

**什么时候用？** 分页优化器主要缓解**峰值OOM**——就是训练中途突然爆显存的情况。通常配合 `gradient_checkpointing=True`

## 块 50

## 7.1 4-bit量化：给模型“瘦身”

你的目标是LoRA微调，但显存是硬约束。QLoRA的第一步，就是把模型权重从FP16（16位）压缩到NF4（4位），减少约87.5%的存储空间（从16位降到4位，即1−4/16=75%的位宽缩减，对应存储空间减少87.5%）。你可以把它理解为：把一张高清照片压缩成更小的文件，虽然细节有损，但整体内容还在。

```
FP16 → NF4 4-bit
```

**动手理解**：想象你有一个装满FP16数字的数组，NF4量化就是为每个数字找一个最近的4位“代号”，用代号代替原数存储。加载时再通过“查表”还原近似值。

---

## 7.2 双重量化：连“尺子”也压缩

量化需要一把“尺子”（scale）来告诉系统每个数字的缩放比例。这把尺子本身也占显存。双重量化就是对量化步长（scale）进行二次量化，以减少量化参数的存储开销。

```
scale → 再次量化
```

**动手理解

## 块 51

## 4-bit 加载：把模型“压缩”进显存

你现在要做 LoRA 微调，但模型太大、显存不够？`load_in_4bit=True` 就是关键开关——它让模型以 **4-bit 精度加载**，大幅降低显存占用，这是 QLoRA 的基石。

### 三个配套参数，一次配齐

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 开启 4-bit 加载
    bnb_4bit_quant_type="nf4",        # 用 NF4 量化（QLoRA 推荐）
    bnb_4bit_use_double_quant=True,   # 开启双重量化，进一步省显存
    bnb_4bit_compute_dtype=compute_dtype,  # 计算精度，见下方
)
```

### 最容易踩的坑：存储精度 ≠ 计算精度

这是新手最容易混淆的点：

- **存储 dtype**：权重以 4-bit（NF4）**存**在显存里，这是省显存的关键。
- **计算 dtype**：权重真正参与矩阵运算时，会**反量化**成更高精度来计算——这个精度由 `bnb_4bit_compute_dtype` 决定。

**不设置会怎样？** 不设置时默认可能为 float32，但推荐显式指定为 bfloat16 或 float16 以获得更好的性能与稳定性。

### 选 BF16 还是 FP16？别写死，让代码判断

`torch.bfloat16` 需要较新 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系及更早）**不支持**，得用 `torch.float16`。

**正确做法：自动判断**

```python
import torch

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

**为什么优先 BF16？** 两者都是 16-bit，但 BF16 动态范围更大

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个**能直接跑的最小骨架**，让你看到 QLoRA 训练的完整链路。

### 整体流程（5步）

1. **配置 4-bit 量化** → 2. **加载模型** → 3. **prepare 准备 k-bit 训练** → 4. **注入 LoRA** → 5. **训练**

---

### 第1步：4-bit 量化配置

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

- `load_in_4bit=True`：启用 4-bit 加载
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化类型
- `bnb_4bit_use_double_quant=True`：开启双重量化（进一步省显存）
- `bnb_4bit_compute_dtype`：按 GPU 能力自动选 bfloat16 或 float16

### 第2步：以 4-bit 加载基础模型

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except OSError as e:
    raise RuntimeError(f"模型加载失败，请检查网络连接或模型名称是否正确：{e}") from e
```

注意：`quantization_config` 传入第1步的 `bnb_config`，`device_map="auto"` 自动分配设备。模型许可证请参考所选模型的官方说明（如 Llama 2 Community License，商用月活>7亿需Meta授权）。

### 第3步：prepare 准备 k-bit 训练（关键一步）

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步处理 norm/输出层精度，并开启梯度检查点所需的输入梯度。`use_gradient_checkpointing=True` 与后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 是**对应关系**。

### 第4步：注入 LoRA

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

- `r=8`：低秩矩阵的秩
- `lora_alpha=16`：缩放因子
- `target_modules`：按实际模型模块名调整，也可用 `"all-linear"`
- `print_trainable_parameters()`：打印可训练参数量，确认只有 LoRA 参数在训练

### 第5步：训练（分页优化器 + 梯度检查点）

```python
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,      # 与第3步呼应
    optim="paged_adamw_8bit",         # 分页优化器
)
```

关键点：
- `gradient_checkpointing=True` 与第3步的 `use_gradient_checkpointing=True` 配套使用
- `optim="paged_adamw_8bit"` 使用分页优化器（依赖 bitsandbytes，名称随版本可能不同）

数据部分用极简示例（真实场景替换为足够规模的数据），Causal LM 的 `labels` 直接复制 `input_ids`：

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

最后用 `Trainer` 组装并训练（需 transformers ≥ 4.46 以支持 `processing_class` 参数，旧版本请使用 `tokenizer=tokenizer`）：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,   # 旧版本用 tokenizer=tokenizer
)
trainer.train()
```

---

### 核心关系总结

| 组件 | 位置 | 作用 |
|------|------|------|
| `BitsAndBytesConfig` | 第1步 | 定义

## 块 53

## 九、把 QLoRA 的四个关键步骤串起来（可跑通骨架）

你已经理解了 4-bit 量化、NF4、双重量化和分页优化器的原理，现在我们把它们**真正组装成一个能跑的骨架**。下面这 5 步缺一不可，顺序也别乱：

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么被加载**

这一步是"地基"。它告诉 HuggingFace：基础模型要用 **4-bit 方式加载**，具体包括：
- 用 **NF4** 做量化格式；
- 开启**双重量化**（进一步省显存）；
- 设定**计算 dtype**（比如 BF16/FP16，用于实际运算时的精度）。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的必要一步**

这一步**不能跳过**。它的作用是让量化后的模型"准备好被训练"：
- 冻结模型参数，将 layernorm 等层转为 fp32 以稳定训练；
- 启用梯度检查点以节省显存。

> ⚠️ **常见坑**：跳过这一步，轻则训练不收敛，重则直接报错。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

这一步把**可训练的 LoRA 适配器**注入到**冻结的 4-bit 主干模型**上。主干不动，只训练 LoRA 那一点点参数。

**第 4 步：梯度检查点**

在 `TrainingArguments` 中设置 `gradient_checkpointing=True` 以启用梯度检查点。`prepare_model_for_kbit_training` 默认会启用梯度检查点，但为确保生效，可在 `TrainingArguments` 中显式设置。

**第 5 步：`optim="paged_adamw_8bit"` —— 分页优化器**

设置这个优化器，利用分页机制**缓解峰值显存**，防止训练中途显存爆掉。

---

**一句话总结这个骨架**：

```
BitsAndBytesConfig（4-bit 加载）
→ prepare_model_for_kbit_training（适配训练）
→ get_peft_model（注入 LoRA）
→ gradient_checkpointing（省显存）
→ paged_adamw_8bit（防爆显存）
```

> 📌 **注意**：上面是"能说明关系"的最小骨架。真实训练时，你需要换成足够规模的数据、合理的超参，并且**按你实际使用的库版本**核对 `optim` 的名称、`prepare_model_for_kbit_training` 的参数，以及 `processing_class`/`tokenizer` 的写法。

---

## 十、你的电脑能跑吗？

**如果只学原理**：不需要 GPU，看代码就行。

**如果实际跑 QLoRA**，参考下面的经验值（以 7B 模型为例）：

| 显存 | 适合

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会误以为LoRA训练完，得到一个“新模型”。**不是的**。

LoRA的训练过程是这样的：

- **基础模型**（Base Model）：全程**冻结**，参数一动不动；
- **LoRA Adapter**（适配器）：只训练这一小部分参数。

训练结束后，你得到的结构是：

```
基础模型 + LoRA Adapter
```

举个例子：你用Llama-7B做LoRA训练，保存时通常只会得到两个文件：

```
adapter_model.safetensors   ← LoRA权重
adapter_config.json         ← 配置信息
```

**而不是**一个完整的Llama-7B模型文件。

需要明确的是，LoRA保存的并不是完整模型，而是低秩增量矩阵A和B，即ΔW=(α/r)BA。基础模型保持不变，合并时需将增量写回原权重。

---

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（可能只有几十MB，而完整模型有几个GB甚至更大）；
- 保存快；
- 可以多个任务共用一个基础模型。

比如同一个基础模型，可以搭配：

```
客服Adapter
医疗Adapter
法律Adapter
```

换任务？换个Adapter就行，基础模型不用动。

**缺点：**

部署推理时，需要**基础模型 + Adapter**两部分同时存在，缺一不可。这就引出了下一个问题——为什么要合并。

---

## 二、为什么需要合并LoRA权重？

因为Adapter不能独立使用。每次推理都要同时加载基础模型和Adapter，不仅麻烦，而且部署时容易出错。

**合并**就是把Adapter的权重“写回”基础模型，生成一个**完整的、独立的模型**。合并之后，你只需要一个模型文件，不需要再额外

## 块 55

**本块学习检查**：看完后，请合上资料，画出“未合并”和“合并后”两条推理路径的流程图。

---

**推理时的问题**

不合并权重时，模型推理需要同时加载两部分：**基础模型**和**LoRA适配器**。输入数据要依次经过它们，流程是：输入 → 基础模型 + LoRA适配器 → 输出。

但部署时，我们通常希望只有一个完整的模型文件。理想流程是：输入 → 完整模型 → 输出。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**为什么能合并？**

Lo

## 块 56

## 3.1 merge_and_unload() 是做什么的？

你已经训练好了LoRA，现在要把它“装回”基础模型里。`merge_and_unload()` 就是干这件事的：

```python
merge_and_unload()
```

**一句话理解**：把LoRA的增量矩阵 `(α/r)BA` 加到基础模型的权重 `W` 上，得到新的权重 `W'`，然后脱掉PEFT的“包装”，让模型变回一个普通的Transformer模型。

```
基础模型 W
    +
LoRA增量 (α/r)BA
    ↓
完整模型 W'（W + (α/r)BA）
```

---

## 3.2 合并前后，模型发生了什么变化？

**合并前**，你的模型是一个 `PeftModel`，里面有两层结构：

```
PeftModel
├── Base Model（基础模型）
└── LoRA Adapter（LoRA适配器）
```

**合并后**，`PeftModel` 这层“壳”被去掉了，LoRA的权重已经融进基础模型里：

```
普通Transformer模型（包含LoRA更新后的权重）
```

---

## 四、LoRA模型合并的完整流程

整体走一遍，心里有个地图：

```
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

---

## 五、加载LoRA Adapter（第3、4步）

假设你已经有：

```python
base_model_path = "model_path"   # 基础模型路径
adapter_path = "adapter_path"    # LoRA适配器路径
```

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 把LoRA适配器挂到基础模型上
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：

```
Base Model + LoRA Adapter
```

---

## 六、执行合并（第5步）

一行代码搞定：

```python
merged_model = model.merge_and_unload()
```

这一步完成的是：

```
W + (α/r)BA → W'
```

**关键点**：执行完这行后，`merged_model` 不再是 `PeftModel`，而是一个**普通的Transformer模型**。这意味着它已经“独立”了，不再依赖PEFT库。

---

## 七、导出合并后的模型（第6步）

保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后的目录大概长这样（示意）：

```
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> **注意**：这只是示意。实际文件会随模型/分词器类型变化，比如可能还有 `generation_config.json`、`special_tokens_map.json`，
> **模型许可证**：请确认所用基础模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）。
> **显存要求**：加载模型前请确认显存充足（

## 块 57

## 三、merge_and_unload是什么？（重点）

### 3.1 方法作用

简单说，`merge_and_unload` 就是把 LoRA 训练时加的“小补丁”（Adapter）**焊死**到原模型里，得到一个完整的、独立的模型文件。LoRA冻结原始权重W，仅训练低秩矩阵A和B，合并时将增量ΔW=(α/r)BA加到W上，得到完整模型。合并后，你不再需要 PEFT 库来加载模型，直接用 Transformers 就能跑。

**合并前 vs 合并后，一张表看懂：**

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | Base+Adapter（略大） | 与原模型相当 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

**什么时候该合并？** 这是个权衡，不是非黑即白。如果你需要**多个 Adapter 动态切换**（比如一个模型服务多个任务），或者你的推理框架原生支持 LoRA，那就不合并。如果你追求**部署简单**、**存储体积不是问题**，那就合并。QLoRA 场景下还要考虑精度接受度。

---

## 合并后如何加载推理？

合并后**不需要** PEFT，直接：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")
inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

流程就是：用户输入 → 完整模型 → 生成结果。

---

## 合并后必查清单（避免“能加载但行为不对”）

导出后务必核查这四项：

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `token

## 块 58

**本块学习检查**（动手验证，约5分钟）

**场景1：上线部署（追求简单）**

生产环境只需一个固定Adapter时，合并更方便：

```text
服务器 → 直接加载模型 → 提供API
```

合并后无需额外依赖PEFT、不做运行时Adapter加载，部署链路最短。

**注意**：若推理框架支持直接加载LoRA（如vLLM多LoRA服务），或需热切换多个Adapter，不合并反而更灵活。

**模型许可证**：Llama 2 Community License（商用月活>7亿需Meta授权）。

**显存要求**：Llama-2-7b FP16 需要至少14GB显存。

**场景2：分享模型**

想让对方开箱即用、无需理解Adapter机制，发布合并后的完整模型（`merged_model`），用户无需额外下载Adapter。

**权衡**：合并后体积明显更大（等于完整基础

## 块 59

### 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个专用“小插件”（Adapter A、B、C）。合并后这些插件就焊死了，没法单独切换。**保留Adapter更灵活**——想用哪个任务就加载哪个，互不干扰。

**场景2：持续训练**

如果你还打算继续调整LoRA，先别合并。合并相当于把当前状态“定格”，后续再改就得重新来，**保留Adapter**方便你随时接着调。

> **动手小验证**：打开你的训练脚本，检查是否同时加载了多个Adapter，或计划后续继续训练。若命中任一情况，请保持Adapter不合并。

---

### 十一、QLoRA模型合并注意事项（重点）

QL

## 块 60

## 11.1 直接合并可能遇到的情况

如果你直接在 4-bit 量化状态下执行 `merge_and_unload()`，可能会遇到两类问题：

- **精度损失**：4-bit 权重需要先反量化再合并，这个过程与全程使用高精度计算并不完全等价，合并后的模型质量可能略有下降。
- **权重类型不匹配或版本不支持**：某些 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并，或者权重类型对不上，导致操作失败。

> ⚠️ 注意：以上只是**可能**出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你所用库的**版本**、量化配置，以及是否按下面的流程先反量化。请以你实际环境的文档和运行结果为准。

## 11.2 常见做法：先反量化到高精度再合并

一个稳妥的思路是：**在量化状态下直接合并是可行的，PEFT 会自动处理反量化**。若遇到版本兼容问题，可尝试先反量化到高精度再合并。这样合并发生在高精度权重上，绕开了 4-bit 带来的类型和精度问题。

流程如下：

```
以高精度（FP16/BF16）重新加载基础模型
        ↓
加载同一份 LoRA Adapter
        ↓
merge_and_unload()
        ↓
保存完整模型
```

下面是完整的可执行代码，对应上面每一步：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 1) 关键：以高精度（FP16/BF16）重新加载基础模型，而不是再走 4-bit 量化
#    合并要发生在高精度权重上，所以这里不传 quantization_config
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=compute_dtype,
    device_map="auto",        # 显存不足时可改为 device_map={"": "cpu"} 在 CPU 上合并
)

# 2) 挂载训练得到的同一份 Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3) 合并；safe_merge=True 会在合并时做数值检查（如检测 NaN）
merged_model = model.merge_and_unload(safe_merge=True)

# 4) 保存完整模型与 tokenizer（tokenizer 从基础模型或训练目录加载均可，保持与训练一致）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)
```

**动手验证一下**：把你自己的 `base_model_path` 和 `adapter_path` 填进去，跑一遍上面的代码，观察合并过程是否顺利、保存的模型能否正常加载。如果遇到报错，先检查版本兼容性，再确认是否真的走了高精度加载路径。

> 📄 **模型许可证**：部署引用模型时请注意其许可证类型。例如，Llama 2 采用 Llama 2 Community License，商用月活用户超过 7 亿需获得 Meta 授权。请以你所用模型的实际许可证为准。

## 块 61

### 直接合并QLoRA模型：三个关键提醒

你正在做LoRA微调，目标是把训练好的QLoRA模型合并导出。这里有三件必须知道的事：

**1. 版本决定能不能直接合并**
不同PEFT版本对“量化模型能否直接合并”支持不同。对于QLoRA模型，PEFT的`merge_and_unload`通常需要先反量化（dequantize），具体支持情况取决于PEFT版本。若需合并，可考虑加载量化模型后调用`merge_and_unload`，或使用其他工具如GPTQ等。动手前先查你所用版本的`merge_and_unload`文档，并注意模型许可证（如Llama 2 Community License，商用月活>7亿需Meta

## 块 62

## 合并校验与一致性验证

`merge_and_unload` 可以加一个保险：合并后自动做数值检查（比如检测有没有产生 NaN）。

```python
merged_model = model.merge_and_unload()
# 自行检查合并后权重是否包含 NaN
```

合并完别急着用，**建议对比合并前后模型在同一批输入上的输出**，确认没引入明显偏差。流程是：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits / 生成文本是否基本一致

最小验证代码思路：用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本是否一致。

**本

## 块 63

## 11.3 合并时开启校验并做一致性验证（建议）

合并完模型，心里没底？这一步就是给你吃“定心丸”的。QLoRA 合并涉及低精度量化与高精度权重的融合，结果是否可靠，不能靠感觉，要用数据说话。核心思路：**拿一份“未合并”的模型当参照，和“已合并”的模型对比输出，差异越小越安全。**

### 为什么必须重新加载一份“未合并”模型？

这是最容易踩的坑。`merge_and_unload()` 是**原地操作**——它直接修改并返回底层模型，合并后原来的 `PeftModel` 对象就不再是“未合并”状态了。如果你拿它当“合并前”的基准，对比就失效了。

**正确做法**：单独重新加载一份全新的、未合并的 `PeftModel` 作为对比基准：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

注意：这里用的是 `base_model_path` 和 `adapter_path`，不是 §11.2 里合并后的对象。

### 校验一：对比 logits 最大绝对差

logits 是模型输出的原始分数，最能反映权重变化。差异越小，说明合并越保真：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果**：理想情况是 0，但 QLoRA 因为量化精度损失，通常是一个**较小的近似值**。如果这个值异常大（比如超过 1 甚至更多），说明合并过程可能有问题，需要回头检查。

### 校验二：对比贪心生成文本

数值差异再小，也不如实际生成结果直观。用**贪心解码**（`do_sample=False`）去掉随机性，保证两次生成完全可复现：

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后:", text_after)
print("生成文本是否一致:", text_before == text_after)
```

## 块 64

## 合并后验证：为什么必须另存一份“未合并”的对照？

你可能会想：合并完直接对比不就行了？问题在于 `merge_and_unload()` 返回合并后的新模型，原模型保持不变；若需释放显存，可删除原模型引用。也就是说，合并后你再拿这个 `model` 当“合并前”的参照，两边其实是**同一份已合并的权重**，`max_abs_diff` 永远等于 0，验证就完全失效了。

**正确做法**：在合并之前，保存原始基础模型（如 `base_model = model.base_model`）或重新加载原始模型作为对照。这样你手里才有真正的“合并前”和“合并后”两份独立模型。

**显存不够怎么办？** 分两次跑：
1. 先用未合并的 PeftModel 跑一遍推理，把 logits 或生成文本**存到磁盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比两次结果。

**结果预期（重点）**：
- **QLoRA 场景**：合并前后通常只是**近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 不同。这是正常的——你要确认的是**没有明显退化或 NaN**，而不是追求逐位相同。
- **LoRA（非量化）场景**：一致性通常更好，`max_abs_diff` 可以非常小。
- 是否可接受，最终以你的**任务指标和

## 块 65

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

你正在学习LoRA微调，目标是把训练好的Adapter合并回基础模型。这一步做不好，前面所有训练都白费。所以，合并后**必须验证结果是否正确**。

---

#### 合并前必查：你是不是真的在操作PeftModel？

`merge_and_unload` 报“不存在”时，九成是下面两种情况：

1. **忘了挂载Adapter**——没有用 `PeftModel.from_pretrained` 把LoRA权重挂到基础模型上；
2. **模型已经被合并过**——当前对象已经是普通模型，不再是PeftModel。

**排查动作**：确认加载流程是“基础模型 → PeftModel.from_pretrained挂载Adapter → 再调merge_and_unload”。顺序错了，后面全错。

---

#### 合并后必查：结果对不对？

合并完成后，别急着保存。先做**一致性验证**——用同一段输入，分别跑合并前后的模型，对比输出是否一致。不一致就说明合并过程出了问题。

---

#### 常见问题速查表

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| merge_and_unload不存在 | 当前对象不是PeftModel | 确认加载方式，先得到PeftModel再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型 | 无需处理；在意体积可保留Adapter不合并 |
| 显存不足 | 高精度完整模型占用高；FP32未offload | 改用CPU合并或offload、用FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存tokenizer | 重新save_pretrained并核查tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 4-bit状态直接合并；版本不支持；量化未正确反量化 | 先反量化再合并；核对版本；必要时CPU合并；开safe_merge=True并做一致性验证 |

---

#### 关键操作：先反量化再合并

QLoRA合并失败的头号原因，是**在4-bit量化状态下直接合并**。解决办法不是手动逐层还原权重，而是：

**重新加载一份不带量化配置的基础模型**——用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`，**不传** `quantization_config` 或 `load_in_4bit`，得到高精度权重，再挂载**同一份Adapter**后合并。

这样合并发生在高精度权重上，天然避开4-bit状态下的类型/精度问题。

**版本提示**：不同PEFT/bitsandbytes/Transformers版本对“量化模型直接merge_and_unload”的支持不同——有的报错，有的警告，有的能

## 块 66

## 路线A：合并后部署完整模型

**一句话理解**：把 LoRA 训练好的“补丁”直接缝进基础模型里，得到一个完整的、独立的模型文件，之后所有推理框架都把它当成普通模型来用。

### 操作流程

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

### 这条路线的优缺点

**优点（为什么选它）：**

- **部署链路简单**——推理框架不需要知道 LoRA 是什么，按普通模型加载即可，省心；
- **推理时没有 Adapter 切换成本**——模型已经是“缝合”好的，推理过程没有任何额外开销。

**代价（什么时候不选它）：**

- 每个 Adapter 都要保存/分发一份**完整模型**。如果你有多个任务、多个 Adapter，就要存多份完整模型，**存储成本高**。

---

## 路线B：保留 Adapter，用支持 LoRA 的推理框架动态服务

**一句话理解**：基础模型只加载一份，多个 LoRA Adapter 像“插件”一样在运行时挂上去，按请求切换用哪个。

### 操作流程

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

### 这条路线的优缺点

**优点：**

- **多任务/多租户时只需一份基础模型 + 多个小体积 Adapter**，大幅节省显存与存储；
- **支持热切换**——不用为每个任务各存一份完整模型，随时切换。

**代价：**

- 依赖推理框架对 LoRA 的支持与配置，链路比“直接加载合并模型”略复杂；
- 不同框架/版本对 LoRA 的支持范围与限制不同，需要额外学习配置。

---

## vLLM 启动 LoRA 服务的示意形态（版本敏感，仅示意）

以 vLLM 为例，启动 OpenAI 兼容服务时，需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，请求里用 Adapter 名字指定用哪个。命令行大致形如：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
    --max-loras 2 \
    --max-lora-rank 16
```

**参数含义：**

- `--enable-lora`：开启多 LoRA 服务；
- `--lora-modules 名字=路径`：声明可用 Adapter。**注意：这里的路径

## 块 67

## 14.3 怎么选？——两条路线，看你的场景

先记住一句话：**合并不是唯一出路**。这节就帮你建立这个认知，具体框架用法第15板块再展开。

**路线A：合并（merge_and_unload）**
适合：**单一Adapter + 追求最简部署**。一个模型配一个LoRA，合并后就是一个完整模型，直接部署，省心。

**路线B：保留Adapter动态服务**
适合：**同一基础模型要服务多个Adapter，且推理框架原生支持LoRA**。相当于一个底座，随时切换不同Adapter，灵活但依赖框架支持。

**特别提醒（QLoRA用户）**：如果你用的是QLoRA，精度敏感，**无论走哪条路线**，都建议按§11.3做合并前后/上线前的一致性验证，别跳过。

---

## 十五、本章总结

你已经掌握：
- LoRA Adapter保存方式
- 为什么需要权重合并
- merge_and_unload作用
- LoRA数学合并过程
- 完整模型导出流程
- 合并模型独立部署方式
- LoRA与QL

## 测评题目

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1205-003
合并 LoRA 权重后最直接的用途是？

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1202-003
在其他条件相同时，增大 LoRA rank 最可能带来什么？

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？
