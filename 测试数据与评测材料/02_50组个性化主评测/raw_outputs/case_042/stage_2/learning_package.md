# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么工作的：

```text
输入数据
    ↓
整个模型参数更新
    ↓
保存完整模型
```

以7B参数模型为例（70亿参数），训练时你需要同时保存三样东西：

- **参数**：模型本身的权重
- **梯度**：用于更新参数的计算结果
- **优化器状态**：控制更新步长的额外信息

这三样叠加在一起，显存需求会非常高。普通消费级GPU根本扛不住。

---

**为什么这对你很重要？**

你的目标是LoRA微调，而LoRA正是为了解决上面这个痛点而生的。理解“全参数微调为什么难”，你才能明白LoRA到底省了什么、怎么省的。

**快速自查**：7B模型训练时，除了参数本身，还需要保存哪两类东西？

> 答案：梯度和优化器状态。这三者叠加导致显存需求极高，普通消费级GPU难以完成训练。

## 块 2

**1.2 参数高效微调（PEFT）**

PEFT，即参数高效微调。核心思想：**不修改整个大模型，只训练少量新增参数**。

流程：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter。

| 方式 | 训练参数 | 显存需求 |
|------|---------|---------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**本块学习检查**：对照上表，说出LoRA相比全参数微调在训练参数和显存需求上的区别。

## 块 3

## 2.1 LoRA是什么？（①）

LoRA（Low-Rank Adaptation，低秩适配）是一种参数高效微调方法。核心思路是：**冻结原模型权重，只训练新增的小矩阵**。

原模型权重记为 `W`，训练时**不更新**它。我们新增一个变化量 `ΔW`，最终模型变为：

```
W + ΔW
```

其中 `ΔW = (α/r) × B × A`。这里 `A` 和 `B` 是两个低秩小矩阵：若原权重 `W` 形状为 `d_out × d_in`、秩为 `r`，则 `A` 形状为 `d_in × r`，`B` 形状为 `r

## 块 4

### 3.1 LoRA的问题：显存瓶颈

你已经知道LoRA能大幅减少**训练参数**，但有个关键点别忽略：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个7B参数的模型，如果用FP16精度加载，大约需要 **14GB显存**。这意味着，即使LoRA只训练一小部分参数，你的GPU也得先装下整个“大块头”模型，显存压力依然很大。

---

### 3.2 QLoRA的解决思路：给模型“瘦身”

QLoRA（Quantized LoRA，量化版LoRA）就是来解决这个问题的。它的核心思路是：**先把模型“压缩”再训练**。

流程可以理解为：

```text
FP16模型 → 4bit量化 → 显存占用大幅下降 → 加载LoRA进行训练
```

训练时的结构是：

```text
4bit基础模型 + LoRA参数 → 一起训练
```

简单说，就是把基础模型从FP16“瘦身”成4bit，显存占用立刻降下来，给LoRA训练腾出空间。

---

### 3.3 关键依赖：bitsandbytes

要实现QLoRA，你需要这几个库配合：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中，`bitsandbytes` 是负责“瘦身”的关键，它管三件事：

- **低精度量化**
- **4bit加载**
- **8bit加载**

> **重要提醒（关于硬件支持）**：本教程的示例和验证**统一以NVIDIA CUDA GPU为准**，这是项目支持范围，也是当前最成熟、资料最全的路径。但请注意，QLoRA/4-bit量化

## 块 5

## 电脑配置批注：你的LoRA/QLoRA训练环境需要什么？

为了完成LoRA微调，先确认电脑配置是否达标。下面分两档说明。

### LoRA最低配置（硬性门槛）

- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA GPU
- **显存**：最低8GB，推荐12GB+

### QLoRA要求（更省显存的选择）

QLoRA的核心优势是**用更小显存微调大模型**——通过4-bit量化基础模型，相比LoRA的全精度加载可降低显存占用，但运行时仍需加载量化后的基础模型和LoRA参数。

**经验参考值（7B级别模型）**：
- 参考下限：约8~12GB（短序列、小batch）
- 推荐：16GB+

> **重要提醒**：以上数值不是固定门槛，而是经验起点。实际显存占用受以下条件影响：
> - **模型规模**：7B级别；13B/70B显存需求显著上升
> - **序列长度**：短序列（≤512）占用低，越长激活值显存增长越明显
> - **batch大小**：小batch（1~2）占用低，越大占用越高
> - **梯度检查点**

## 块 6

## 5.1 Python版本检查

在开始LoRA微调之前，先确认你的Python环境。打开终端，运行：

```bash
python --version
```

如果输出显示 **Python 3.10或更高版本**，就满足要求了。如果版本偏低，建议先升级Python再继续。

---

## 5.2 PyTorch检查（关键一步）

> ⚠️ **为什么这一步很重要？**  
> LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的量化功能要求安装**与你本机CUDA匹配的GPU版PyTorch**。如果直接`pip install torch`，通常会装上CPU版，这样QLoRA就跑不起来了。

**正确做法**：去 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 选择对应你CUDA版本的安装命令。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证PyTorch是否能看到GPU**，运行以下Python代码：

```python
import torch

print(torch.__version__)        # 查看PyTorch版本
print(torch.cuda.is_available()) # 是否可用CUDA
print(torch.version.cuda)        # torch编译用的CUDA版本
```

如果`torch.cuda.is_available()`输出`True`，说明PyTorch能识别GPU。

> 📌 **注意**：`True`只代表PyTorch侧能用GPU，**不代表bitsandbytes的4-bit量化一定能用**。后者还需要bitsandbytes正确匹配你的CUDA环境，具体检查会在后续章节进行。

---

## 5.3 查看GPU信息

在终端运行：

```bash
nvidia-smi
```

输出会包含：
- **GPU型号**（如RTX 4090）
- **CUDA版本**（如CUDA 12.x）
- **显存大小**（如24GB）

记下这些信息，后面配置bitsandbytes时会用到。

---

## 六、安装PEFT库

### 6.1 PEFT是什么？

PEFT是Hugging Face官方的**参数高效微调库**，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter等方法。你要做的LoRA微调，就是通过它来实现的。

### 6.2 安装PEFT

```bash
pip install peft
```

## 块 7

**bitsandbytes 是干嘛的？**

一句话：它是让模型“瘦身”的工具，专门做 **8bit** 和 **4bit 量化**，也是跑 **QLoRA 训练**的必备依赖。

**为什么你必须装？**

因为本教程用的是 **QLoRA 4-bit 量化方案**，这个方案建立在 bitsandbytes 之上，所以它是前置依赖。注意：QLoRA 的核心思想（4-bit 量化基础模型 + LoRA）并不绑定某个特定库，但本教程统一用 bitsandbytes + NVIDIA CUDA 演示。如果你用别的量化后端，以对应文档为准。

**安装命令（基础版）：**

```bash
pip install bitsandbytes
```

---

**本块学习检查**：在终端执行上述安装命令，确认安装成功（无报错）。若已安装，可

## 块 8

**为什么先查CUDA？**  
bitsandbytes依赖CUDA调用GPU。版本不匹配会导致CUDA不可用、量化失败或加载报错。

**两步检查**  
1. 运行`nvidia-smi`，查看`CUDA Version`。  
2. 运行`python -m bitsandbytes`，正常显示CUDA和GPU信息。

**检查**  
记录版本号，确认第二条无报错。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。  
# 注意：Llama-2-7b FP16需至少14GB显存。

## 块 9

**本块学习检查**：安装完环境后，运行 `import bitsandbytes`，若报错“无法加载CUDA动态库”，请记录错误信息，并对照本块内容判断最可能的原因。

---

**Windows用户必看：用WSL2避开环境坑**

你的目标是跑通LoRA微调，但如果在Windows原生环境装bitsandbytes，很可能卡在第一步。推荐路径是：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

原因很简单：Linux环境兼容性更好，能少踩很多坑。

**Windows原生可能遇到的问题**

最常见的是：**bitsandbytes无法加载CUDA动态库**。这不是单一原因，通常受多方面影响，排查时建议按以下顺序检查：

1. **CUDA运行时库未安装或路径未配置**：系统找不到CUDA动态库文件，需检查CUDA是否安装，并确认`LD_L

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配

在Windows上安装LoRA环境时，最常见的坑就是**CUDA版本对不上**。别慌，先跑一条命令看看你的机器到底什么状态：

```bash
nvidia-smi
```

这条命令会告诉你三件关键事：

- **NVIDIA驱动**是否正常安装
- **CUDA版本**（驱动支持的版本）
- 你的GPU型号

记住：**驱动支持的CUDA版本 ≠ PyTorch实际用的CUDA版本**。你只需要确认驱动版本够新，然后安装匹配的PyTorch即可。

---

## 十、安装完整LoRA环境

假设你已经装好了基础库：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**关键前置**：确保已安装**匹配本机CUDA的GPU版PyTorch**，否则后面 `import torch` 或QLoRA 4-bit会直接失败。以CUDA 12.1为例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**显存提示**：LoRA微调建议至少8GB显存，QLoRA可低至4GB；模型规模越大，所需显存越多，请先用 `nvidia-smi` 确认显存容量。

然后装上LoRA核心库：

```bash
pip install peft bitsandbytes
```

一步到位，直接复制这条完整命令：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

执行：

```bash
accelerate config
```

按提示选择，单GPU场景示例：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

完成后会生成 `default_config.yaml`。

---

## 十二、LoRA环境验证实验

新建 `test_lora_env.py`，粘贴以下代码：

```python
import torch
from peft import LoraConfig

try:
    print("CUDA:", torch.cuda.is_available())
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )
    print(config)
except Exception as e:
    print("环境验证失败，请检查PyTorch和PEFT安装：", e)
```

运行：

```bash
python test_lora_env.py
```

看到输出说明：

- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes是否就绪：

```python
import torch
import bitsandbytes

try:
    print(torch.cuda.is_available())
    print(bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes加载失败，请检查安装：", e)
```

如果输出 `True`，说明GPU环境正常，可以放心跑QLoRA了

## 块 11

## 九、Windows用户特别注意事项：推荐使用WSL2

在Windows上配置LoRA/QLoRA环境时，你可能会遇到兼容性问题。最直接的解决方案是使用WSL2（Windows Subsystem for Linux 2），它能提供更接近Linux的环境，避免许多Windows特有的报错。

**常见问题速查表**（遇到报错时对照检查）：

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或用更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

**动手试一试**：如果你在Windows上遇到bitsandbytes加载失败，先检查你的CUDA版本是否与当前环境匹配——这是最常见的坑。

---

## 完整LoRA/QLoRA学习路线

**当前环境依赖链**（按顺序安装）：

```
Transformers → Datasets → Accelerate → PEFT → bitsandbytes
```

**模型许可证**：Llama 2 Community License（商用月活>7亿需Meta授权）。

**

## 块 12

## 一、为什么需要LoRA？（①②③）

### 1.1 大模型微调的问题

现代大语言模型——比如LLaMA、Qwen、Baichuan、Mistral——通常拥有数十亿甚至上百亿参数。以7B模型为例，就是70亿个参数。

如果采用传统方式微调：把预训练模型拿过来，更新全部参数，再保存为新模型。这个过程需要：

- **巨大的GPU显存**：你不仅要存模型本身，还要存训练过程中的中间数据；
- **大量训练时间**：每更新一个参数都要计算梯度，70亿个参数的计算量可想而知；
- **巨大的存储空间**：每次微调都要保存一份完整的新模型副本。

### 1.2 全参数微调的问题

假设模型有100亿参数，训练时你实际要保存的不只是模型参数W，还包括：

- **梯度**∂W：每个参数更新方向的数值；
- **优化器状态**（如Adam moment）：优化器维护的额外变量。

这三样加起来，实际显存需求远大于模型本身的大小。这就是为什么全参数微调对硬件要求极高，也是LoRA要解决的核心痛点。

---

## 二、LoRA核心思想（①②）

LoRA的核心思路很直接：**既然更新全部参数太贵，那能不能只更新一小部分？**

答案藏在"低秩分解"这个概念里。LoRA认为，模型微调时参数的更新量ΔW虽然理论上和W一样大，但它的"有效信息"其实可以用一个很小的矩阵来表达——这就是"低秩"的含义。

具体做法是：**把ΔW分解成两个小矩阵的乘积，ΔW = BA**。其中B和A的维度远小于W，训练时只更新B和A，而不是整个W。

这样做的直接好处是：训练参数量大幅减少

## 块 13

## 动手理解：LoRA 到底改了什么？

先看传统微调。假设模型原始权重是 `W`，传统做法是直接训练它，把它变成 `W'`——也就是**直接修改** `W` 本身。

LoRA 的思路完全不同：**保持 `W` 不动（冻结）**，另外增加一个新参数 `ΔW`。最终模型实际使用的权重是：

```text
W' = W + ΔW
```

- `W`：原模型参数（冻结，不训练）
- `ΔW`：LoRA 新增参数（只训练这个）

---

**为什么只训练 `ΔW` 就够了？**

大模型已经通过海量数据学会了语言规律、世界知识和推理能力。微调通常只是**对已有能力做方向调整**，不需要重新学全部参数，只学“任务相关变化量”往往就够用。

> ⚠️ 注意：这是帮助理解的直觉，不是严格结论。能否用 LoRA 替代全参数微调，取决于任务与预训练能力的差距、数据规模、目标模块与秩

## 块 14

你正在学LoRA微调，核心思想就是“别直接动大矩阵，拆成小矩阵来学”。咱们先看矩阵长什么样。

Transformer里到处都是矩阵。比如注意力层算Q，就是输入X乘一个权重矩阵Wq：

```text
Q = X · Wq
```

这里Wq就是个权重矩阵。假设它形状是4096×4096，那参数数量就是：

```text
4096 × 4096 ≈ 1678万
```

问题来了：如果微调时要改这个矩阵，直接学一个完整的ΔW，那也得是4096×4096，照样1678万个参数要训练，成本太高。

LoRA的思路很直接：**冻结原始权重W，只训练两个小矩阵B和A来形成增量更新**。

原始权重W保持冻结不动，新增的ΔW由两个小矩阵B和A相乘得到：

```text
ΔW = (α/r

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设你有一个大矩阵 **W**，尺寸是 **4096 × 4096**。如果直接训练它，需要更新的参数数量是：

**4096 × 4096 = 16,777,216**（约1677万）

这就像要记住一整本电话簿，非常吃力。

现在用 **LoRA** 来“瘦身”。设置 **rank r = 8**，把大矩阵拆成两个小矩阵：

- **A**：尺寸 **4096 × 8**，参数数量 = **32,768**
- **B**：尺寸 **8 × 4096**，参数数量 = **32,768**

两者相加，

## 块 16

**5.1 神经网络参数变化具有低维特征**

LoRA的核心依据是一个假设：模型适应新任务时，参数变化ΔW并不需要覆盖所有方向。真正影响任务的有效变化，集中在远小于原维度的少数方向上——即一个**低维子空间**。

举个例子：一个4096维的空间里，真正起作用的“有效方向”可能远少于4096个。但请注意，这里没有固定的“有效维度”数值，它会随模型、任务、数据变化，我们只需定性理解。

因此，低秩矩阵BA可以近似这个ΔW——用少数方向逼近完整变化。

**本块学习检查**：用自己的话解释“低维子空间”是什么意思，并说明为什么BA能近似ΔW。

## 块 17

## 5.2 类比理解：为什么LoRA用少量参数就能微调大模型？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。但如果这张图片的主要变化只是**亮度、对比度**，以及**少量结构变化**，你根本不需要保存全部1600万个像素——用少量参数就能描述这些变化。

**LoRA的核心思想完全一样**：大模型有海量参数，但微调时真正需要改变的“方向”其实很有限。LoRA就是**用少量参数来描述模型的变化方向**，而不是去动整个模型。

---

## 六、LoRA加在Transformer的哪里？

先回顾Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中 **Attention** 部分包含四个投影矩阵：

```text
Q（查询）、K（键）、V（值）、O（输出）
```

**LoRA加在哪里？** 原论文的**常见选择**是只加在 `q_proj` 和 `v_proj` 上：

```python
target_modules=["q_proj", "v_proj"]
```

意思是：只训练 `Wq` 和 `Wv` 的变化量，其他参数全部冻结。

> ⚠️ **重要提醒**：只挑 `q_proj`/`v_proj` 是原论文的**常见做法**，**不是固定标准**。现代实践中经常把LoRA加到注意力的全部投影（q/k/v/o），甚至FFN的线性层。

**更完整的配置示例**（以Llama类模型为例，不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

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
        "gate_proj", "up_proj", "down_proj",       # FFN线性层
    ],
)
```

> 如果PEFT版本支持，也可用 `target_modules="all-linear"` 自动匹配所有线性层。**覆盖模块越多，可训练参数、显存和Adapter体积越大**——是否值得，要结合验证集表现权衡。

---

## 七、LoRA训练完整流程

```text
加载预训练模型 → 冻结所有参数 → 插入LoRA

## 块 18

### 8.1 rank（r）——用“矩阵秩”控制微调的“参数量”

你正在学LoRA微调，核心就是**用一个小矩阵去模拟大模型的更新**。`r`（rank）是低秩矩阵的秩，它决定了低秩矩阵A和B的维度（A为r×d_in，B为d_out×r），从而控制微调参数量。

**一句话理解**：`r` 越大，低秩矩阵的参数量越多，模型表达能力越强，但容易过拟合；`r` 越小，参数量越少，但可能欠拟合。

**怎么选？先看任务难度（这只是起点，不是铁律）**：

| 任务类型 | rank 起点参考 |
|---------|-------------|
| 简单任务 | 4 ~ 8 |
| 普通微调 | 8 ~ 32 |
| 复杂任务 | 64+ |

**但别死记这个表**。`r` 的最佳值受模型规模、数据量、任务难度共同影响。更可靠的做法是**用验证集来“试”**：

1. **留出验证集**：比如 8:2 或 9:1 的数据划分

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制LoRA对模型参数的更新强度。代码中常见写法：

```python
lora_alpha=16
```

它通常与秩 `r` 配合使用，缩放比例为 `alpha/r`。简单理解：**这个值越大，LoRA对原模型的影响越强**；值越小，影响越温和。

> 动手体会：你可以把 `lora_alpha` 想象成“音量旋钮”——`r` 决定音质细节（矩阵维度），`alpha` 决定音量大小（更新幅度）。

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

作用：在训练LoRA参数时随机丢弃一部分更新，防止模型对训练数据“死记硬背”（过拟合）。0.05 表示每次有5%的LoRA参数被临时忽略。

---

## 九、LoRA vs 全参数微调：核心区别

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

**重要提醒**：上表是**相对趋势**，不是绝对数值。如果你把LoRA加到很多模块、用较大 `r`，参数量和Adapter体积也会明显上升。“更快

## 块 20

**本块学习检查**：读完本节后，请合上资料，在纸上画出 `r=8` 时 `A` 和 `B` 两个矩阵的尺寸，并标出它们相乘后恢复出的 `ΔW` 的尺寸。画完再对照下文验证。

---

### 2.1 Rank 表示什么？

**一句话理解**：Rank（秩）决定了 LoRA 在“压缩通道”里塞了多少信息。

**直观类比**：想象你要传递一张 4096×4096 的大图，rank 就是中间那条“窄桥”的宽度。桥越宽（r 越大），能同时通过的细节越多，但成本也越高。

**技术定义**：在 LoRA 中，rank 表示低秩矩阵的**中间维度大小**。

**具体看例子**：

原始权重矩阵 `W` 的尺寸是：

```text
4096 × 4096
```

如果你设置：

```python
r=8
```

那么原本要学习的 `ΔW`（也是 4096×4096）就被拆成两个小矩阵：

- 矩阵 `A`：尺寸为 `8 × 4096`
- 矩阵 `B`：尺寸为 `4096 × 8`

**关键点**：这两个小矩阵相乘：

```text
ΔW = B × A
```

`4096×8` 乘以 `8×4096`，结果正好恢复成原来的尺寸：

```text
4096 × 4096
```

**所以**：rank 就是中间那个“8”。它越小，需要训练的参数越少（省显存、速度快），但能捕捉的信息也越少；它越大，效果可能越好，但训练成本随之上升。

## 块 21

**Rank控制什么？——用“变化方向”来理解**

Rank决定LoRA能学多少种变化方向。把它想成模型的“自由度”：

- **低rank（如r=4）**：模型只能学较少变化模式。好处是参数少、显存低、训练快；代价是表达能力有限。
- **高rank（如r=64）**：LoRA拥有更多自由度，能学更复杂任务、适应能力更强；但参数更多、显存增加、训练成本提高。

**动手验证**：在代码中分别设置`r=4`和`r=64`，对比训练时的显存占用和速度，感受差异。

**一句话总结**：rank是“表达能力”和“资源消耗”之间的权衡

## 块 22

**本块学习检查**：动手算一遍——把 r=16 代入公式，写出参数量，再和完整训练（1677万）比大小。

---

### 3.1 不用LoRA：全量训练

假设权重矩阵是 **4096×4096**，参数量就是：

```text
4096 × 4096 = 16,777,216
```

约 **1677万** 参数。这就是“完整训练”的代价——每个参数都要更新。

---

### 3.2 用LoRA：只学一个小矩阵

LoRA 不直接改这个大矩阵，而是插入两个小矩阵，参数量公式：

```text
参数量 = r × (输入维度 + 输出维度)
```

当 r

## 块 23

### 4.1 Rank太小的问题：为什么你的模型“学不动”？

如果你在LoRA微调时把`r`设成`2`（比如`r=2`），最直接的后果就是**模型学不进去**。这就像给一个学生只发一张便签纸，让他记下整本教材的重点——纸太小，根本写不下。

具体表现有两个：
- **训练loss下降慢**：模型在训练时，误差减少得非常缓慢，像是“挤牙膏”。
- **效果提升有限**：即使训练了很久，最终模型在新任务上的表现也远达不到预期。

为什么会这样？因为`r`代表LoRA增量的“容量”。`r=2`时，这个增量矩阵能表达的信息非常有限，导致**学习能力不足**，模型无法充分表达新任务需要的知识，最终陷入**欠拟合**——模型既没学好旧知识，也没学会新任务。

---

### 动手验证：感受“小rank”的瓶颈

如果你正在跑微调实验，可以做一个快速对比：
1. 先用`r=2`跑几个epoch，记录训练loss的变化曲线。
2. 再把`r`改成`64`（或`8`），用相同数据和epoch数再跑一次。
3. 对比两条loss曲线：你会发现`r=2`那条下降得又慢又平，而`r=64`那条明显更陡、降得更低。

这个实验不需要额外知识，只是观察现象，能帮你直观理解“容量不足”的含义。

---

**小结**：`r`太小 = 模型“记笔记的纸”太小，知识装不下，训练慢、效果差。下次看到loss降不下去，先检查一下你的`r`是不是设得太小了。

## 块 24

### 4.3 常见rank选择

选 `r` 别纠结，先按任务类型找个起点：

| 任务类型 | rank（起点参考） |
|------|------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**注意**：这只是经验起点，不是自动调参标准。实际项目要结合数据量、模型大小、序列长度、batch大小（含梯度累积）、`target_modules`选择、GPU显存、验证集表现来调。同一个 `r` 在不同模型规模、序列长度/batch、目标模块下，显存和效果差别可能很大。

---

### 五、Alpha（lora_alpha）是什么？

LoRA 除了 `ΔW = BA`，

## 块 25

### 6.1 控制LoRA影响强度

假设你设了 `r=8`，然后 `alpha=16`，缩放就是 `16/8=2`，意思是LoRA更新会被**放大2倍**。

如果 `alpha=8`，缩放是 `8/8=1`，保持原始比例，不放大也不缩小。

如果 `alpha=4`，缩放是 `4/8=0.5`，LoRA影响就**减弱**了。

**核心公式**：LoRA加入模型的更新量会被 `α/r` 缩放，所以 `α/r` 是控制影响强度的关键缩放系数。

---

### 七、Alpha与Rank的关系（重点）

**但注意**：`α/r` 只是缩放系数，不是"实际强度"的唯一决定因素。实际落到权重上的更新量还取决于：

- **学习率**：直接决定每步对 `A`、`B` 的更新步长。把 `α/r` 调大又不下调学习率，容易放大更新导致震荡。很多实践里甚至通过调学习率而非频繁改 `α` 来控制强度。
- **初始化**：常见实现里 `A` 用小随机值、`B` 初始化为0，使训练开始时 `ΔW = BA = 0`（不扰动原模型），之后再逐步学习。
- **训练状态**：`ΔW` 是训练**逐步累积**出来的。同样的 `α/r`，训练

## 块 26

**动手试一下：两种配置，同一个缩放比例**

先看两组参数：

- 配置1：`r=8`，`alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16`，`alpha=32` → 比例 = 32 ÷ 16 = **2**

发现了吗？虽然 rank 从 8 变成了 16，但 `alpha / r` 的缩放比例**都是 2**，没变。

**为什么这很重要？**

缩放比例决定了 LoRA 对原模型权重的影响强度。比例相同，意味着即使你换了 rank，模型更新的“力度”大致保持一致——这能帮你更公平地对比不同 rank 的效果。

**实操建议（

## 块 27

**方案1：小模型实验（推荐你先试这个）**

配置：`r=8`，`alpha=16`。

特点：参数少、训练快，适合测试流程。因为你目标是LoRA微调且喜欢动手实践，先用这个配置跑通整个流程，建立手感。

**PEFT代码示例**

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**三个关键参数速记**

- `r`：控制LoRA容量（即可学习的参数规模）。
- `lora_alpha`：控制LoRA更新幅度（即每次调整的步长大小）。alpha 控制更新幅度，通常设为 r 的 1~2 倍，此处 alpha=16, r=8，缩放因子为2。

## 块 28

## 误区1：rank越大一定越好？

**错误。**

任务简单时，大rank反而可能：

- 浪费显存；
- 增加训练成本。

> 你的直觉可能是“参数越多越强”，但LoRA的目标是**高效**微调——用最少资源完成任务。rank是容量，不是越多越好，够用就行。

---

## 误区2：只调rank，不调alpha

举例：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

比例变成：

```text
16/64=0.25
```

这意味着LoRA对模型的影响**被削弱了**。rank和alpha要一起调，别只动一个。

---

## 误区3：alpha越大越好？

**错误。**

alpha过大（即 `α/r` 缩放过大）可能导致：

- 更新过强；
- 训练不稳定。

**注意**：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡/发散还常来自：

- 学习率过高
- 数据质量差
- 有效batch size过小
- 精度（fp16/bf16）与数值稳定性
- 缺少梯度裁剪

排查不稳定时，要综合这些因素，别只盯着alpha。

---

## 本块学习检查

**动手验证**：打开你的LoRA配置文件，检查当前 `r` 和 `alpha` 的取值，计算 `alpha/r` 的比例。如果比例小于0.5，说明LoRA影响偏弱；如果大于2，则可能更新过强。记录你的数值，并思考：如果任务很简单，你会怎么调整这对参数？

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。它的核心思路是：不微调整个大模型，只训练一小部分新增参数。

在 LoRA 训练流程中，`LoraConfig` 是承上启下的关键一环：

```text
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

**`LoraConfig` 负责告诉 PEFT 四件事：**

1. **LoRA放在哪里** —— 哪些层要被插入 LoRA 模块
2. **LoRA规模多大** —— 用多大的低秩矩阵（由 `r` 控制）
3. **LoRA训练强度如何** —— 更新幅度有多大（由 `lora_alpha` 控制）
4. **是否使用dropout** —— 训练时是否随机丢弃部分神经元来防止过拟合（由 `l

## 块 30

## 1.2 一个简单示例——先跑通，再理解

你正在学LoRA微调，目标很明确。下面这段代码就是LoRA最核心的配置，先把它复制运行起来：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**这4个参数分别管什么？**

| 参数 | 作用 | 直观理解 |
|------|------|----------|
| `target_modules` | 决定插入哪些层 | 告诉LoRA“改造模型的哪些部分” |
| `r` | 决定LoRA矩阵大小 | 数字越大，新增的可训练参数越多 |
| `lora_alpha` | 控制缩放强度 | 调节LoRA对原模型的影响力度 |
| `lora_dropout` | 控制随机失活 | 防止过拟合，训练时随机“丢弃”部分信息 |

---

## 完整流程：三步走

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

实际代码：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要改模型的哪些部分”**。

Transformer 模型的 Attention 部分，内部通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

**LoRA 不会默认修改所有层**，你需要手动指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思就是：**只给 Query 投影层和 Value 投影层加 LoRA**，其他层保持原样。

---

## 四、为什么选 q_proj 和 v_proj？

先看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开看两个角色的分工：

- **Q（Query）**：决定模型**关注什么信息**；
- **V（Value）**：决定模型**传递什么信息**。

所以，调整 Q 和 V，通常就能有效改变模型的行为——这就是它们成为“常见起点”的原因。

> ⚠️ **重要提醒**：`q_proj` / `v_proj` 只是**常见起点**，不是所有模型通用的“标准配置”。不同模型的模块命名可能不同，**配置前务必先查看模型实际的模块名**（用 `model.named_modules()`），再决定 `target_modules`。

---

### 偷懒技巧：`target_modules="all-linear"`

如果你不想一个个枚举模块名，较新版本的 PEFT（一般 ≥ 0.7，具体以你所装版本文档为准）支持直接写：

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
    target_modules="all-linear",   # 自动匹配所有nn.Linear层，包括输出层lm_head（若为nn.Linear），具体行为请参考PEFT文档
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**注意权衡**：`"all-linear"` 会自动把 LoRA 加到（几乎）所有线性层，省事，但覆盖模块变多会**增加可训练参数与显存占用**。如果你的 PEFT 版本太旧不支持这个字符串，就退回**显式列出模块名**的写法，或者升级 PEFT。用 `print_trainable_parameters()` 可以快速验证注入是否符合预期。

---

**本块学习检查**：打开你的模型，运行 `model.named_modules()` 查看实际模块名，确认你的模型

> **模型许可证**：部署文档引用模型时需注意许可证类型。例如：Llama 2

## 块 32

**方案1：只调整Q和V**

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**特点**：参数少，训练快。这是一个常见的轻量起点，适合快速验证LoRA微调流程。

**动手试一试**：先按这个方案配置，跑通一次微调，观察训练速度和显存占用。如果效果不够好，再考虑扩展。

**本块学习检查**：写出方案1的`target_modules`列表，并说出它的两个特点。

## 块 33

## LLaMA系列：target_modules怎么选？

LLaMA系列模型在LoRA微调时，最常用的`target_modules`设置是：

```python
[
"q_proj",
"v_proj"
]
```

也就是说，你只需要把注意力机制中的**查询投影（q_proj）**和**值投影（v_proj）**这两层加上LoRA低秩适配器。

**为什么只选这两个？**

这是实践中最常见、效果与效率平衡最好的选择。`q_proj`和`v_proj`是LLaMA注意力计算中的关键线性层，对它们做低秩适配，既能有效调整模型行为，又不会引入过多可训练参数，训练速度也更快。

**动手验证一下（hands-on）**

如果你手边有环境，可以跑这段代码确认`target_modules`生效：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

运行后观察输出：可训练参数占比很小，说明LoRA只注入了`q_proj`和`v_proj`两层。

---

**对比：GPT-2为什么不一样？**

GPT-2的写法完全不同，它用的是：

```python
[
"c_attn"
]
```

原因是GPT-2把Q/K/V三个投影**合并**在了一个叫`c_attn`的模块里，所以只需要指定这一个目标即可覆盖全部注意力投影。

**关键坑：`fan_in_fan_out=True`**

GPT-2的注意力层用的是`Conv1D`（这是transformers库自定义的层，并非标准PyTorch的`nn.Conv1d`），它的权重形状是`(in_features, out_features)`，和`nn.Linear`的`(out_features, in_features)`正好**转置**，且其计算逻辑与`nn.Linear`不同。所以在GPT-2上加LoRA时，必须设置：

```python
fan_in

## 块 34

**BERT的target_modules配置（动手版）**

先别急着抄别人的配置——BERT和LLaMA的模块名不一样，直接复制会报错或没效果。

**第一步：查看你的模型实际模块名**

运行下面代码，把输出打印出来：

```python
for name, _ in model.named_modules():
    print(name)
```

**第二步：找到注意力层**

在输出里找类似 `q_proj`、`v_proj` 的名字。BERT通常只对这两个模块做LoRA，所以配置写：

```python
target_modules=["q_proj", "v_proj"]
```

注意：这里写的是**精确匹配**，PEFT会精确匹配名为 `q_proj`、`v_proj` 的层，不需要写全路径。

**第三步：验证**

配置后打印模型，确认 `lora_A`、`lora_B` 只出现在 `q_proj` 和 `v_proj` 层。

**关键提醒**：不同版本模型命名可能不同，务必以你实际 `named_modules()` 的输出为准。

---

**本块学习检查**：运行 `named_modules()`，找出你当前BERT模型中所有包含 `proj` 的模块名，并判断哪些

## 块 35

**6.1 r是什么？——你的LoRA“瘦身”开关**

r就是rank（秩），它控制LoRA低秩矩阵的维度。LoRA的核心公式是：

```
ΔW = (α/r)BA
```

r决定的是A、B矩阵的中间维度。比如原矩阵是4096×4096，设置r=8后：

- A：4096×8
- B：8×4096

**6.2 r影响什么？**

r越大：
- 优势：学习能力更强，能表达更复杂任务
- 缺点：参数增加，显存占用上升

**经验起始范围**（非固定配方，需用验证集调优）：

| 场景 | r（起点参考） |
|------

## 块 36

**7.1 alpha作用：控制LoRA更新幅度**

LoRA的最终更新公式不是 `BA`，而是 `(alpha/r)BA`。这里的 `alpha` 就是控制更新幅度的关键。

**动手算一下：**
- 配置 `r=8`，`lora_alpha=16`
- 缩放系数 = `16/8 = 2`
- 含义：LoRA对模型的影响被**增强**了2倍。

**7.2 alpha常见设置（经验起点）**

一个常见的起点是：`alpha = 2 × r`。注意这只是经验值，不是硬性规则，最终以验证集表现定。

- 例1：`r=8`，`alpha=16`
- 例2：`r=16`，`alpha=32`

**本块

## 块 37

**8.1 dropout作用：防过拟合的“开关”**

训练时，LoRA会**随机关闭一部分神经连接**（比如设`lora_dropout=0.05`，就是让5%的路径临时失效）。这能防止模型“死记硬背”训练数据，从而**减少过拟合**。

**8.2 怎么选dropout？**

- **数据少**：起点约 `0.05~0.1`（多关一些，防过拟合）
- **数据多**：起点约 `0~0.05`（数据够多，可以少关甚至不关）

如果训练数据非常丰富，直接设`lora_dropout=0`也行。

> **经验起点

## 块 38

## 9.1 基础配置——快速上手

这段配置是 LoRA 微调的“最小可用模板”，特别适合你现在的场景：**7B 模型测试**。7B 模型参数量大，全量微调显存吃不消，用 LoRA 只训练一小部分参数，就能快速跑通流程。

直接看代码，每一行都对应一个关键决策：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量：秩为8，决定新增参数量
    lora_alpha=16,          # 更新缩放：16/8=2，放大更新幅度
    target_modules=[        # 插入位置：只改注意力层的q和v投影
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,      # 防止过拟合，训练时随机丢弃5%的LoRA参数
    bias="none",            # 不训练偏置项
    task_type="CAUSAL_LM"   # 任务类型：因果语言模型（如GPT类）
)
```

**四个核心参数速记：**

| 参数 | 值 | 作用 |

## 块 39

## 9.1 基础配置：把 LoRA 装到 BERT 上

你正在做 LoRA 微调，目标很明确：**用尽量少的可训练参数，把 BERT 适配到你的分类任务上**。这一节我们直接看代码，边看边理解每个参数在干什么。

### 第一步：加载基础模型

```python
# 模型许可证：BERT 使用 Apache 2.0 许可证，可商用
# 注意：加载模型时需确保显存充足，建议至少预留 2GB 显存
base_name = "bert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里加载的是带分类头的 BERT。注意：**分类头是新初始化的**，它没有经过预训练，所以必须跟着一起训练，否则分类结果就是随机的。

### 第二步：配置 LoRA

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

逐个看参数：

- **`r=8`**：LoRA 的低秩维度。r 越小，可训练参数越少，但表达能力也越弱。8 是一个常用起点。
- **`lora_alpha=16`**：缩放因子。实际生效的缩放比例是 `lora_alpha / r`，这里就是 16/8 = 2。它控制 LoRA 更新的强度。
- **`lora_dropout=0.05`**：防止过拟合的 dropout 比例，训练时随机丢弃一部分 LoRA 层的输出。
- **`bias="none"`**：不对偏置项做 LoRA，保持默认。
- **`task_type="SEQ_CLS"`**：告诉 PEFT 你正在做序列分类任务。
- **`target_modules=["query", "value"]`**：指定在 BERT 的哪些模块上插入 LoRA。这里选择的是注意力机制里的 query 和 value 投影层，这是 LoRA 最常用的注入位置。
- **`modules_to_save=["classifier"]`**：**这是最容易踩坑的地方**。它让分类头参与训练，并且随 LoRA 权重一起保存。如果不加这一行，分类头不会被保存，重载时会退回随机初始化，模型就废了。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

执行后你会看到：可训练参数 = LoRA 参数 + classifier 参数。这就是 LoRA 的核心优势——大部分 BERT 参数被冻结，只有少量新增参数在训练。LoRA 的原理是冻结原始权重 W，仅训练低秩矩阵 A 和 B，通过 A×B 形成对 W 的增量更新，而不是对 W 做分解。

### 第四步：保存与重载验证

```python
model.save_pretrained("./lora_ckpt")
```

保存的是 **Adapter + classifier**，不是完整的基础模型。所以重载时必须先加载同一个基础模型，再挂载 LoRA：

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
re

## 块 40

## 十、普通文本生成：LoRA参数怎么配？

你已经知道LoRA的核心参数了，现在直接看结论——做**普通文本生成**任务时，推荐这样配置：

```python
r=8
alpha=16
dropout=0.05
```

**怎么理解这三个数？**

- **`r=8`**：低秩矩阵的秩。`r`越小，可训练参数越少，模型越“轻”；`r=8`是文本生成任务的常用起点，兼顾效果与效率。
- **`alpha=16`**：缩放因子。实际生效的缩放比例是 `alpha / r`，这里就是 `16 / 8 = 2`。`alpha` 通常设为 `r` 的2倍，让微调时新注入的权重有合适的“力度”。
- **`dropout=0.05`**：随机丢弃5%的神经元，防止过拟合。文本生成任务数据量通常较大，`0.05` 是低丢弃率的稳妥选择。

**对比一下其他任务，帮你建立直觉：**

| 任务类型 | r | alpha | dropout |
|---------|---|---|---------|
| 普通文本生成 | 8 | 16 | 0.05 |
| 指令微调 | 16 | 32 | 0.05 |
| 领域知识增强 | 32 | 64 | （原文未给出） |

规律很明显：**任务越复杂、越需要模型学习新知识，`r` 和 `alpha` 就越大**。指令微调需要模型理解“指令→回答”的格式，领域知识增强要让模型记住专业内容，所以参数逐级上调。

> ⚠️ **注意**：本文只

## 块 41

**动手配置LoRA：普通文本生成**

直接看核心代码，这是给文本生成任务用的LoRA配置：

```python
LoraConfig(
    r=8,                # LoRA容量：8，控制新增参数的规模
    lora_alpha=16,      # 缩放因子：16，通过 α/r = 2 影响更新强度
    target_modules=["q_proj", "v_proj"],  # 改哪里：只改注意力层的q和v投影
    lora_dropout=0.05   # 防过拟合：5%的随机丢弃
)
```

**四个参数，逐个击破：**

- **target_modules**：决定改哪里。这里只选 `q_proj` 和 `

## 块 42

## 为什么LoRA还不够省显存？

你已经知道LoRA只训练一小部分参数，但注意：**基础模型仍然需要完整加载到显存里**。

以7B模型为例，如果用FP16（每个参数占2字节）加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这还没算上训练时的**激活值、梯度和优化器**，实际占用会更高。

所以，即使LoRA大幅减少了可训练参数，**模型本身的显存占用依然是瓶颈**。QLoRA的核心思路就是：把基础模型压得更小，让消费级GPU也能跑起来。

> **模型许可证**：本教程涉及的模型需遵守相应许可证，例如 Llama 2 Community License（商用月

## 块 43

**QLoRA核心思想：让大模型微调“瘦身”**

你已经在学LoRA了，QLoRA就是它的“省显存版”。核心一句话：**先把基础模型压成4-bit，再插LoRA训练**。

**流程对比**

- LoRA：FP16模型 + LoRA
- QLoRA：4-bit模型 + LoRA

**训练时谁动谁不动**

- 基础模型：4-bit权重，**冻结**（不训练）
- LoRA参数：FP16/BF16，**训练**

**关键点**：量化的是基础模型权重，**不是**LoRA参数。

**动手验证**：回想你跑LoRA时的显存占用，如果换成QLoRA，基础模型从FP16变4-bit，省下的显存主要用在哪部分？（提示：冻结

## 块 44

## 3.1 模型存储方式：从“重量”看量化

模型参数在显存里占多大地方，取决于每个参数用多少字节（Byte）存。看这张表：

| 类型 | 每个参数占用 |
|---|---|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

**动手算一下**：一个7B（70亿参数）模型，用FP16存，就是 `7B × 2 ≈ 14GB`；换成4-bit，就是 `7B × 0.5 ≈ 3.5GB`。显存直接省约75%。

**但注意**：这3.5GB只是**理论下界**——只算了量化后的权重本身。实际跑起来会明显更高，因为还要算上量化常数、分组元数据（NF4按组量化，每组都存额外参数）、没量化的模块（norm、embedding、lm_head、LoRA参数），以及训练时的激活值和缓存。所以把它当“最低值”参考，真实占用以运行为准。

## 3.2 为什么不用普通INT4？

普通INT4整数范围只有 `-8 ~ 7`

## 块 45

**4.1 NF4是什么？**

NF4，全称 NormalFloat4，中文叫“正态分布4位浮点”。这是 QLoRA 论文为神经网络权重专门设计的一种 4-bit 数据类型。

**注意**：别把它当成 IEEE 754 那种通用浮点格式。NF4 本质是一套**离散码本（codebook）**——预先定好 16 个（2⁴）量化取值点，这些点按标准正态分布的分位数排布，让 0 附近（权重密集区）划分更细。

**工作流程**：量化时，把每组权重归一化后映射到最接近的码本取值，只存 4-bit 索引；反量化时，按码本和缩放

## 块 46

**为什么NF4适合模型权重？**

大模型训练完的权重有个特点：**大量参数集中在0附近**，整体像个正态分布（中间高、两边低）。

普通INT4量化是**平均划分**的，比如[-8,-7,...,7]，每个区间一样宽。但权重都挤在0附近，这种平均分法**没法精细表示中心密集区域**，一量化就丢精度。

NF4不一样——它**基于标准正态分布的分位数进行量化**，通过经验累积分布函数估计分位数，使得量化区间在权重分布密集的区域（如0附近）更精细，从而减少量化误差。好处很直接：

- 0附近精度更高
- 更贴合真实权重分布
- 整体精度损失更低

## 块 47

**本块学习检查**：读完5.1后，请暂停并回答：4-bit量化除了保存量化后的权重，还需要额外保存哪两个参数？它们的作用是什么？

---

### 5.1 为什么需要双重量化？

你已经知道，把FP16权重压成4-bit能省显存。但别忘了——**量化不是白干的**，它需要额外保存一组“量化参数”来记录压缩方式。

具体来说，**每组权重**都需要保存两个东西：

- **scale**：缩放因子，告诉你怎么把4-bit值还原回原来的大小范围；
- **zero point**：零点偏移，用于对齐原始数据的零点位置。

**关键点来了**：这些额外参数本身也是数据，也要占显存。模型越大，分组越多，这些“附属开销”就越可观。所以，光

## 块 48

你正在用QLoRA微调模型，已经知道它能省显存。但训练时还有个“隐形杀手”：**显存峰值**。

模型本身占显存，但**梯度**和**优化器状态**（比如Adam的**一阶动量**和**二阶动量**）也要占地方。QLoRA虽然压缩了模型，可训练中仍可能突然**显存不足**，直接报错中断。

分页优化器就是来救场的。它借鉴了操作系统的**虚拟内存**思路，核心是借助 **CUDA 的 pinned memory（页锁定内存）和异步内存拷贝**：当显存出现峰值、快不够用时，自动把优化器状态**“分页”换出到CPU内存**，需要计算时再**换回GPU**。

注意：这不是“平时就把优化器放CPU省显存”，而是**应对偶发峰值**——平稳时

## 块 49

### 6.3 类比理解：分页优化器在干什么？

把显存想象成一张**桌面**，把CPU内存想象成**柜子**。

- **普通方式**：所有东西（模型参数、梯度、优化器状态）都堆在桌面上。桌面满了，就**无法继续**，直接报OOM（显存溢出）。
- **分页优化器**：桌面只放**常用文件**（当前计算需要的参数），**暂不用的文件**（部分优化器状态）自动挪到柜子里保存，**需要时再取出**。这样桌面永远有空间，训练就不会因为峰值显存不够而中断。

---

### 6.4 实际怎么启用？代码就一行

原理懂了，落地很简单。**不需要你手写换页逻辑**，bitsandbytes库已经实现好了，你只需要在 `TrainingArguments` 里通过 `optim` 参数选择即可：

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

**两个关键点：**

1. **选哪个？** `paged_adamw_8bit` 优化器状态用8-bit存储，更省显存；`paged_adamw_32bit` 精度更高但占用更多。QLoRA论文常用这两个。
2. **版本注意**：不同Transformers/bitsandbytes版本支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报"未知optim"错误，去查你当前版本的文档，换用支持的名称即可。

**使用场景**：分页优化器主要解决**峰值OOM**

## 块 50

### 7.1 4-bit量化：把模型“瘦身”

你现在目标是用LoRA微调，但显存是瓶颈对吧？QLoRA的第一招就是**4-bit量化**，直接把模型权重从FP16（16位）压缩到NF4（4位），体积缩小至原来的约1/4。想象一下，原来装不下的模型，现在能塞进显存了。不过实际显存节省还受反量化计算、优化器状态等因素影响。

### 7.2 双重量化：连“尺子”也压缩

量化需要“scale”（缩放因子）来还原数值，这些参数也占空间。双重量化就是把这些scale**再量化一次**，进一步减少额外开销。相当于你不仅压缩了书本，连书签也做成了迷你版。

### 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态很吃显存。分页优化器的思路是：**将优化器状态分页到CPU内存，通过统一内存管理，避免显存溢出**，从而降低训练时的峰值显存压力。

---

### 动手配置：bitsandbytes三步走

在你的LoRA微调代码里，加上这段配置即可：

```python
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 开启4-bit量化
    bnb_4bit_quant_type="nf4",      # 使用NF4类型（

## 块 51

## 4-bit 加载与 QLoRA 核心配置

你已经知道 LoRA 是给大模型“打补丁”的高效微调方法，而 QLoRA 的关键一步，就是先把模型“压缩”到 4-bit 再加载。下面我们直接看 bitsandbytes 里四个核心开关怎么配。

### 1. 开启 4-bit 加载

```python
load_in_4bit=True
```

这一行让模型以 4-bit 精度加载进显存，是省显存的第一步。

### 2. 选择量化类型：NF4

```python
bnb_4bit_quant_type="nf4"
```

NF4 是专门为 4-bit 设计的量化格式，QLoRA 默认推荐用它。

### 3. 开启双重量化

```python
bnb_4bit_use_double_quant=True
```

双重量化会再压缩一层量化参数，进一步省显存。建议开启。

### 4. 设置计算精度（最容易踩坑）

这里要区分两个概念：

- **存储 dtype**：权重以 4-bit（NF4）存进显存，用来省空间。
- **计算 dtype**：真正做矩阵运算时，权重会被“反量化”回更高精度来计算。这个精度由 `bnb_4bit_compute_dtype` 控制。

**不设置会怎样？** 默认可能为 `float32` 或由模型配置决定，建议显式指定以避免性能问题。

**选 BF16 还是 FP16？** 别写死，按你的 GPU 自动判断：

```python
import torch
from transformers import BitsAndBytesConfig

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

**为什么这样选？** BF16 需要较新的 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系）不支持 BF16，就得退回 FP16。BF16 动态范围大、不易溢出，训练更稳定；FP16

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面几节我们把 QLoRA 的零件拆开讲了：`BitsAndBytesConfig` 负责 4-bit 加载、分页优化器负责省显存、`bnb_4bit_compute_dtype` 决定计算精度。现在最关键的疑问是——**这些零件怎么拼成一台能跑的车？**

下面这个**最小可运行骨架**就是答案。它把从加载模型到启动训练的全流程串在一起，你只需要把 `model_name` 换成自己的模型、把数据换成自己的数据，就能跑通一个完整的 QLoRA 微调。

> **模型许可证说明**：使用 QLoRA 微调前，请确认所选基础模型的许可证类型。例如，若使用 Llama 2 系列模型，需遵守 Llama 2 Community License（商用且月活用户超 7 亿时需获得 Meta 授权）；其他模型请查阅其官方许可证。

### 代码骨架逐段拆解

**第 1 步：配置 4-bit 量化**

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

这段做了两件事：一是开启 4-bit 加载（`load_in_4bit=True`），并指定用 NF4 量化类型（`bnb_4bit_quant_type="nf4"`）和双重量化（`bnb_4bit_use_double_quant=True`）；二是根据你的 GPU 能力自动选择计算精度——支持 bfloat16 就用它，否则退回 float16。

**第 2 步：以 4-bit 加载基础模型**

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
    raise RuntimeError(f"模型加载失败，请检查模型路径或网络连接：{e}") from e
```

注意两点：一是把刚才的 `bnb_config` 传给 `quantization_config`，模型就会以 4-bit 精度加载；二是如果 tokenizer 没有 pad token，就用 eos token 顶上，否则训练会报错。此外，模型加载涉及磁盘读取和网络下载等 IO 操作，用 `try-except` 捕获 `OSError` 并给出明确提示，便于排查问题。

**第 3 步：为 k-bit 训练做准备（关键一步）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步**不能省**。它专门处理 4-bit 模型训练时的精度问题（比如 norm 层和输出层需要保持较高精度），同时开启梯度检查点所需的输入梯度。这里的 `use_gradient_checkpointing=True` 要和后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 呼应。

**第 4 步：注入 LoRA**

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

LoRA 的核心思路是：冻结原始模型参数，只训练低秩增量。`r=8` 是低秩矩阵的维度，`lora_alpha=16` 是缩放系数，`target_modules` 指定注入哪些模块（这里选了 `q_proj` 和 `v_proj`，你可以按实际模型调整，或用 `"all-linear"` 全注入）。`print_trainable_parameters()` 会打印可训练参数的数量，帮你确认 LoRA 注入成功。

**第 5 步：准备数据**

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

这是极简示例数据，真实场景要换成足够规模的数据。关键点是：Causal LM 的 `labels` 直接复制 `input_ids`，不需要额外标注。

**第 6 步：训练参数 + 启动训练**

```python
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

try:
    trainer.train()
except Exception as e:
    raise RuntimeError(f"训练启动失败，请检查数据、显存或参数配置：{e}") from e
```

这里有两个显存优化手段：`gradient_checkpointing=True` 用计算换显存（和前面第 3 步呼应）；`optim="paged_adamw_8bit"` 利用分页优化器减少显存峰值。

## 块 53

## 九、把 QLoRA 的“五步骨架”串起来

你现在已经分别了解了 4-bit 量化、NF4、双重量化和分页优化器，这一节我们把它们**串成一条能跑通的代码骨架**。这五步缺一不可，顺序也别乱：

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么被 4-bit 加载**

这一步是“入口配置”，告诉 HuggingFace 用 NF4 + 双重量化加载基础模型，同时指定计算精度（compute dtype）。它决定了模型**以什么格式进入显存**。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的必要一步**

这一步**不能跳过**。它会对量化模型做适配：冻结量化模型的参数、把部分层转成计算精度、开启输入梯度以支持梯度检查点等。跳过它，常见后果是**训练不收敛或直接报错**。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

这一步把**可训练的 LoRA 适配器**注入到冻结的 4-bit 主干模型上。主干是冻结的，只有 LoRA 参数在更新。

**第 4 步：`gradient_checkpointing=True` —— 两处可独立设置**

在 `TrainingArguments` 里设置 `gradient_checkpointing=True`，第 2 步里也可以设置 `use_gradient_checkpointing=True`。两处可以独立设置，并非必须对应，但建议同时开启以节省显存。

**第 5 步：`optim="paged_adamw_8bit"` —— 启用分页优化器**

这个设置让优化器在显存不足时把部分状态“分页”到 CPU 内存，缓解显存峰值。

> 上面这个骨架是“能说明关系”的最小版本。真实训练时，你需要换成足够规模的数据、合理的超参，并且**按你所用版本核对** `optim` 名称、`prepare_model_for_kbit_training` 的参数和 `processing_class`/`tokenizer` 写法。

---

## 十、跑 QLoRA 需要什么配置？

**只学原理**：不需要 GPU。

**实际运行 QLoRA**，最低参考：

- CPU：4 核+
- 内存：16GB+
- GPU：NVIDIA CUDA GPU
- 显存：7B 级别模型，经验参考约 **8~12GB+**（推荐 16GB 以上）

**注意**：下表是**带条件的经验参考**，不是保证值。隐含测试条件：7B 级别 Causal LM、NF4 4-bit + 双重量

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会误以为LoRA训练完，得到一个“升级版”的完整模型。**不是的。**

LoRA训练时，基础模型是**冻结**的（不参与训练），真正被训练的是一个额外挂上去的小模块——**LoRA Adapter**。

训练结束后，你得到的结构是：

```
基础模型（冻结，没变）
+
LoRA Adapter（训练后的增量参数）
```

保存时，通常只保存Adapter部分，也就是两个文件：

- `adapter_model.safetensors`（Adapter权重）
- `adapter_config.json`（配置信息）

**并没有保存完整模型。**

---

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（相比动辄几个GB的完整模型，Adapter可能只有几十到几百MB）；
- 保存快；
- 一个基础模型可以搭配多个Adapter，实现多任务复用。

比如同一个基础模型，可以分别搭配：

```
客服Adapter
医疗Adapter
法律Adapter
```

**缺点：**

部署推理时，需要**基础模型 + Adapter**两部分同时存在，缺一不可。

---

## 二、为什么需要合并LoRA权重？

既然Adapter这么方便，为什么还要合并？

因为部署场景下，同时加载两个部分会带来额外负担：

- 推理框架需要额外支持Adapter加载逻辑；
- 每次部署都要先加载基础模型，再挂载Adapter，流程更复杂。

**合并（merge）** 就是把Adapter的增量参数“写回”基础模型，得到一个**完整的、独立的模型**。合并后，部署时只需要加载一个模型文件，不再依赖Adapter。

> # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
> # 注意：Llama-2-7b FP16 需要至少14GB显存。

---

## 动手验证：检查你的训练产物

打开你LoRA训练的输出目录，确认是否包含以下两个文件

## 块 55

**本块学习检查**：请观察下方未合并的推理流程，并回答：部署时我们期望的流程是什么？为什么需要合并权重？

---

**2.1 推理时的问题**

未合并时，模型推理需要**两步加载**：

```text
输入 → Base Model + LoRA Adapter → 输出
```

即：既要加载基础模型，又要单独加载Adapter。

---

**部署时的期望**：只加载一个完整模型，直接推理：

```text
输入 → 完整模型 → 输出
```

因此，需要**合并权重**。

---

**2.2 合并的数学原理**

LoRA训练时，权重更新为：

```text
W' = W + ΔW，其中 ΔW = BA
```

**合并

## 块 56

## 3.1 merge_and_unload() 是做什么的？

你可以把 LoRA 微调想象成：在基础模型（Base Model）旁边贴了一张“补丁”（LoRA Adapter），训练时只更新这张补丁，不动原模型。

但最终你要的是一个**完整的、独立的模型**，而不是“模型+补丁”的组合。`merge_and_unload()` 就是干这件事的：

```python
merge_and_unload()
```

它做两件事：
1. **合并**：把 LoRA 的权重变化（数学上记为低秩增量 `ΔW`，即 `BA`）加到基础模型的权重（`W`）上，得到更新后的权重（`W'`）。
2. **卸载**：移除 PEFT 的包装层，让模型变回一个普通的 Transformer 模型。

用公式理解就是：

```text
W' = W + (α/r)BA
```

其中 `α` 是缩放系数，`r` 是低秩矩阵的秩。

---

## 3.2 合并前后，模型结构的变化

**合并前**，你的模型是一个 `PeftModel`，内部结构是：

```text
PeftModel
├── Base Model      ← 原始权重
└── LoRA Adapter    ← 训练得到的增量补丁
```

**合并后**，`merged_model` 不再是一个 `PeftModel`，而是一个**普通 Transformer 模型**，LoRA 的更新已经“融进”了基础模型的权重里：

```text
普通Transformer模型（包含LoRA更新后的权重）
```

---

## 四、完整流程：从训练到导出

整个合并导出流程按顺序走：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步拆解。

---

## 五、第1步：加载基础模型和 LoRA Adapter

假设你已经有训练好的 LoRA adapter，路径如下：

```python
base_model_path = "model_path"   # 基础模型路径
adapter_path = "adapter_path"    # LoRA adapter路径
```

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 把 LoRA adapter 加载到基础模型上
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：

```text
Base Model + LoRA Adapter
```

---

## 六、第2步：执行 merge_and_unload() 合并权重

一行代码完成合并：

```python
merged_model = model.merge_and_unload()
```

执行后：
- `merged_model` 已经是**普通 Transformer 模型**，不再是 `PeftModel`。
- LoRA 的权重增量（`ΔW`）已经写入了基础模型权重（`W`），得到 `W'`。

---

## 七、第3步：导出合并后的完整模型

## 块 57

## 三、merge_and_unload 是什么？（重点）

简单说，`merge_and_unload` 就是把 LoRA 训练出来的"低秩增量矩阵"（ΔW）**永久合并**到基础模型的原始权重 W 中，得到一个完整的、独立的模型文件。之后你不再需要 PEFT 库，直接用 Transformers 就能加载推理。

### 合并后怎么加载？

合并前加载需要 PEFT 包装，合并后只需要：

```python
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程变成：**用户输入 → 完整模型 → 生成结果**，中间不再有 Adapter 这层。

### 合并 vs 不合并，怎么选？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

### 什么时候该合并？

合并是**权衡**，不是必须。考虑这几点：

- 是否需要**多个 Adapter 动态切换**？需要就不合并。
- 部署框架是否**原生支持 LoRA**？不支持就合并。
- 对**存储/分发体积**的要求高不高？体积敏感就不合并。
- 对**精度**的接受度（尤其 QLoRA 场景）？能接受就合并。

### 合并后必查清单（避免"能加载但行为不对"）

导出后务必核查这四项：

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若训练时扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。
2. **generation_config.json**：如果依赖

## 块 58

### 10.1 推荐合并场景

你正在学LoRA微调，目标是能实际用起来。这里讲清楚**什么时候该把LoRA合并回基础模型**，帮你少走弯路。

**场景1：上线部署（只要一个Adapter，追求简单）**

想象你的模型要上服务器、提供API。如果只用一个固定的LoRA，合并后部署链路最省事：

```text
服务器 → 直接加载模型 → 提供API
```

合并后，服务器不需要额外装PEFT库，也不用在运行时加载Adapter，一步到位。

> **注意**：如果你的推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你需要**热切换多个Adapter**，那就不合并、以Adapter形式部署更灵活。这时是否合并，要看框架能力和业务需求。

```text
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权

## 块 59

## 10.2 不建议合并场景

**场景1：多个任务Adapter**

假设你有一个基础模型，上面挂了多个Adapter（比如Adapter A、B、C），每个负责不同任务。**保留Adapter更灵活**——想用哪个任务就加载哪个，合并后反而绑死了。

**场景2：持续训练**

如果你还想继续调整LoRA，**保留Adapter**。合并后就改不动了，得重新训练。

> **动手验证**：打开你的LoRA训练脚本，看看当前保存了几个Adapter？如果后续要换任务或继续调参，先别急着合并。

---

## 十一、QLoRA合并注意事项（重点）

QLoRA有个关键差异：

- **基础模型**：4-bit量化（NF4）存储

## 块 60

## 直接合并可能遇到的情况与稳妥做法

你在 4-bit 状态下直接合并 LoRA 权重时，可能会遇到两类问题：

- **精度损失**：4-bit 权重需要先反量化再合并，这个过程与全程用高精度计算并不完全等价，可能会引入微小误差；
- **类型或版本不兼容**：可能出现权重类型不匹配，或者你使用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接调用 `merge_and_unload`。

**注意**：以上只是“可能出现”的情况，并非每次都会报错。能否顺利合并、合并质量如何，取决于你的版本、量化配置，以及是否按下面的流程先反量化。请以你所用版本的实际运行结果为准。

---

### 常见做法：先反量化到高精度再合并

一个稳妥的思路是：**不在 4-bit 状态下直接合并**，而是先用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一份 Adapter 后合并。流程如下：

```
以高精度（FP16/BF16）重新加载基础模型
        ↓
加载同一份 LoRA Adapter
        ↓
merge_and_unload()
        ↓
保存完整模型
```

这样做的好处是：合并发生在高精度权重上，绕开了 4-bit 状态下的类型和精度问题。

---

### 可执行代码

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

---

**动手验证**：把你训练好的 QLoRA Adapter 按上面代码跑一遍，观察合并过程中是否出现报错或警告。如果显存不足，把 `device_map` 改为 `{"": "cpu"}` 在 CPU 上合并。合并完成后，用 `AutoModelForCausalLM.from_pretrained` 加载 `merged_out` 目录，简单生成一句话，确认模型能正常推理。

**模型许可证**：合并后保存的模型沿用基础模型的许可证。例如，若基础模型为 Llama 2，则适用 Llama 2 Community License（商用月活>7亿需Meta授权）。部署前请确认你所用基础模型的许可证条款。

## 块 61

### 直接合并QLoRA模型：三个关键点

你正在学LoRA微调，这里聚焦QLoRA合并时最容易踩的坑。记住核心思路：**想要干净合并，就别带着量化配置加载基础模型**。

**1. 版本差异（先查文档）**
不同PEFT版本对“量化模型能否直接合并”支持不同，有的要求先反量化。动手前，务必查你所用版本的`merge_and_unload`文档。上例关键操作是：加载基础模型时**不传**`quantization_config`或`load_in_4bit`，这样得到高精度权重再合并。

**2. 内存压力（提前规划）**
以高精度重新加载完整基础模型，会占用约等于

## 块 62

## 合并时开启校验并做一致性验证（建议）

`merge_and_unload` 可以加一个保险开关：`safe_merge=True`。它在合并时自动做数值检查（比如检测有没有出现 NaN），防止合并过程悄悄出错。

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完别急着用，建议**对比合并前后模型在同一批输入上的输出**，确认没引入明显偏差。流程很简单：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比两者的 logits 或生成文本是否基本一致

最小验证代码的思路：用固定 prompt，先比 logits 的最大绝对差，再比

## 块 63

## 合并后别急着用——先做一次“一致性体检”

你已经完成了 LoRA 合并（§11.2），现在模型已经是一个“合并后”的完整模型了。但问题来了：**合并过程会不会引入误差？** 尤其是 QLoRA 场景下，因为涉及量化与反量化，合并前后的输出可能会有细微差别。这一节就是教你如何用一段“对比脚本”给合并结果做体检。

### 核心思路：拿“合并前”和“合并后”做对比

关键点在于：`merge_and_unload()` 是**原地操作**——它会直接修改并返回底层模型，合并后原来的 `PeftModel` 对象就已经“回不去了”。所以你不能拿 §11.2 里那个已经被合并过的对象当“合并前”的基准。

正确做法是：**重新加载一份全新的、未合并的 `PeftModel`** 作为对比基准。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 变量沿用 §11.2：base_model_path、adapter_path、compute_dtype、tokenizer
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
prompt = "请用一句话介绍你自己。"        # 换成你任务里有代表性的固定输入

# 单独加载一份"未合并的 PeftModel"当作对比基准
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter

# 加载合并后的模型（§11.2 中 merge_and_unload() 的结果）
merged_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
merged_model = PeftModel.from_pretrained(merged_model, adapter_path).merge_and_unload()

inputs = tokenizer(prompt, return_tensors="pt")
```

注意：这里的 `merged_model` 是合并完成后的模型，`model_before` 是刚加载的未合并版本。

### 体检项目 1：比较 logits（模型输出的原始分数）

```python
# 1) 比较 logits：未合并的 PeftModel vs 合并后的 merged_model
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs.to(merged_model.device)).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果？** 这个数值越接近 0，说明合并前后的模型行为越一致。QLoRA 场景下，由于量化精度损失，这个值通常是一个**较小的近似值**，而不是精确的 0。如果这个值异常大，说明合并可能出了问题。

### 体检项目 2：比较生成文本

logits 是数值层面的对比，

## 块 64

## 合并后如何确认没“合坏”？——校验与一致性验证

你刚学完 §11.2 的合并操作，现在面临一个关键问题：**合并完怎么知道结果对不对？** 这一块就是教你做“事后检查”，别急着往下走。

### 为什么必须单独再加载一份模型？

`merge_and_unload()` 有个“副作用”：它会**原地**修改你原来的 `model`——把 LoRA Adapter 融进底层权重，同时把 PEFT 包装拆掉。注意：此行为主要适用于**非量化 LoRA**；在 **QLoRA 场景**下，`merge_and_unload()` 通常需要先反量化权重，且可能不支持原地合并，需重新加载模型，具体实现请参考 PEFT 文档确认。

这意味着：合并之后，`model` 本身已经“变了”。如果你拿它和合并前的自己比，两边其实是**同一份已合并的权重**，`max_abs_diff` 永远等于 0——验证完全失效。

**正确做法**：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter，作为对照基准。具体获取方式：用 `PeftModel.from_pretrained` 加载原始基础模型，再挂载合并前的 Adapter 权重，即可得到 `base_for_ref` 和 `model_before`。这样你比较的才是“合并前”和“合并后”的真实差异。

### 显存不够怎么办？

如果两个模型同时放不下，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits

## 块 65

## 十一、QLoRA模型合并注意事项——常见问题排查与部署衔接

### 13. 常见问题排查（重点）

你在合并时可能遇到下面这些坑，我们逐个看：

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 `PeftModel`（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到 `PeftModel` 再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积 Adapter | 无需处理；若在意体积可保留 Adapter 不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是 dtype 用了 FP32、未做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存 tokenizer；或特殊 token / generation_config 未一并导出 | 重新 `save_pretrained` 并核查 tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 在 4-bit 状态直接合并、PEFT/bitsandbytes 版本不支持、量化包装未正确反量化、内存/显存不足 | 先反量化到 FP16/BF16 再合并；核对版本；必要时用 CPU 合并；开 `safe_merge=True` 并做一致性验证 |

### 关键：什么是"先反量化到 FP16/BF16"？

这里说的"反量化"**不是**手动逐层还原 4-bit 权重，而是——**重新加载一份不带量化配置的基础模型**。

具体做法：用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`（**不传** `quantization_config`/`load_in_4bit`）得到高精度权重，再 `PeftModel.from_pretrained` 挂载**同一份 Adapter** 后合并。这样合并发生在高精度权重上，天然避开了"在 4-bit 状态下合并"的类型/精度问题。

**版本提示**：不同 PEFT / bitsandbytes / Transformers 版本对"量化模型能否直接调用 `merge_and_unload`"的支持不同——部分版本会直接报错或警告，部分较新版本对某些情形有一定支持但仍可能有精度损失。因此**"先反量化（重载高精度）再合并"是跨版本最稳妥的做法**；是否可以在量化状态下直接合并

## 块 66

## 路线A：合并后部署完整模型

### 一句话理解

**合并部署**就是把微调好的 LoRA 权重“焊死”进基础模型里，得到一个**完整的、独立的模型文件**，之后推理框架完全不知道 LoRA 的存在，把它当普通模型用就行。

### 操作流程

```text
merge_and_unload()   ← 把 LoRA 权重合并回基础模型

↓

保存完整模型          ← 得到一个完整的模型目录

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

### 这么做的好处

1. **部署链路简单**——推理框架不需要理解 LoRA 是什么，按普通模型加载即可，省心。
2. **推理时零额外开销**——没有 Adapter 切换成本，因为 LoRA 已经“消失”在模型里了。

### 需要付出的代价

- **存储成本高**：每个 Adapter 都要保存/分发一份**完整模型**。如果你有多个任务（多个 Adapter），就要存多份完整模型，非常占空间。

### 什么时候选路线A？

- 你只需要部署**一个**微调后的模型，不需要频繁切换任务。
- 你的推理框架不支持 LoRA 动态加载。
- 你追求**极简部署**，不想处理额外配置。

### 动手验证（30秒）

在合并保存后，检查你保存的模型目录里是否包含完整的模型权重文件

## 块 67

## 14.3 怎么选？记住两条线就行

你现在处于LoRA微调的学习中段，目标很明确：跑通流程、能部署。这里不需要背规则，只需要建立判断直觉。

**选择就两句话：**

- **只调了一个Adapter（LoRA权重），想最简单地上线** → 走**合并路线**。把Adapter融进基础模型，得到一个完整模型文件，部署时不用额外加载LoRA逻辑，最省事。
- **同一个基础模型要服务多个Adapter，且推理框架原生支持LoRA** → 走**保留Adapter动态服务**。模型文件不动，运行时按需切换Adapter，灵活但依赖框架支持。

**一个必须记住的坑：** 如果你用的是**QLoRA**（量化+LoRA），精度敏感，**无论选哪条路线**，合并前后都要做一致性验证（对应§11.3的内容）。别偷懒，这是上线前最后一道保险。

> 核心认知：**合并不是唯一部署方式**。具体框架用法和限制，下一板块（vLLM部署）再展开，现在先记住这个判断框架。

## 测评题目

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB12-ST1204-003
分页优化器的主要目标是？

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。
