# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题：为什么你的显卡“扛不住”？

先看传统大模型训练是怎么工作的：

```text
输入数据 → 整个模型参数更新 → 保存完整模型
```

以7B参数模型为例，就是**70亿个参数**。训练时，你需要同时保存三类东西：

- **参数**：模型本身的权重
- **梯度**：用来更新参数的中间结果
- **优化器状态**：控制学习过程的额外信息

这三样加起来，显存需求非常高。**普通消费级GPU根本跑不动**——这就是全参数微调最大的痛点。

---

**动手验证一下**：打开你的GPU监控（比如`nvidia-smi`），看看当前显存占用。想象一下：如果要把70亿参数的梯度也塞进去，你的显存够用吗？这个直观感受，就是理解LoRA价值的起点。

---

**小结**：全参数微调的问题是“全都要管”——参数、梯度、优化器状态一个不少，显存自然吃紧。接下来的LoRA，就是来解决这个问题的。

## 块 2

**PEFT：只训练“小补丁”，不碰大模型**

全参数微调要动整个大模型，显存需求“非常高”。PEFT（参数高效微调）的核心思想是：**不修改整个大模型，只训练少量新增参数**。

流程很简单：

1. 原始大模型 + LoRA适配器参数
2. 只训练LoRA参数
3. 保存小规模Adapter（适配器）

对比一下三种方式：

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**

## 块 3

## LoRA 是什么？（①）

想象一下：你有一个已经训练好的大模型，它的权重矩阵是 `W`。传统微调要更新整个 `W`，又慢又费显存。

LoRA（Low-Rank Adaptation，低秩适配）的思路是：**冻结原模型 `W` 不动**，在旁边新增一个增量 `ΔW`，让模型变成 `W + ΔW`。

关键来了：`ΔW` 不是随便一个大矩阵，而是拆成两个小矩阵的乘积：

```
ΔW = B × A
```

- 若 `W` 形状是 `d_out × d_in`，秩为 `r`
- 则 `A` 形状约 `r × d_in`，`B` 形状约 `d_out × r`
- `B × A` 相乘后，形状正好还原成和 `W` 一样的 `ΔW`

> 注意：不同资料可能写成 `A × B`，只是符号顺序不同，含义一致；关键是矩阵形状能相乘并还原成 `W`

## 块 4

## 3.1 LoRA 的问题：模型还是太大

你已经知道，LoRA 能大幅减少**需要训练**的参数。但这里有个关键点：**基础模型本身，仍然要完整加载到显存里**。

举个例子：一个 7B（70亿）参数的基础模型，如果用 FP16（半精度）格式加载，大约需要 **14GB 显存**。也就是说，哪怕 LoRA 训练的参数很少，你的显卡也得先装得下这个“大家伙”。

## 3.2 QLoRA 的解法：把模型“压缩”再训练

QLoRA（Quantized LoRA，量化版 LoRA）的思路很直接：**先把模型变小，再加载 LoRA 训练**。

流程如下：

```text
FP16模型
   ↓
4bit量化（压缩）
   ↓
显存占用大幅减少
   ↓
加载LoRA进行训练
```

训练时的结构是：

```text
4bit基础模型 + LoRA参数 → 训练
```

这样，显存压力就小多了。

## 3.3 QLoRA 依赖 bitsandbytes 库

要用 QLoRA，你需要这几个组件：

```text
transformers + peft + bitsandbytes + 支持的量化后端
```

其中，**bitsandbytes** 负责三件事：
- **低精度量化**（把模型精度降下来）
- **4bit 加载**
- **8bit 加载**

> **重要提醒（关于硬件）**：本教程的实训和验证**统一以 NVIDIA CUDA GPU 为准**。这是项目支持范围，也是当前最常见、资料最全的路径。但请注意：**“QLoRA 只能用 NVIDIA CUDA”并不是技术必然**——bitsandbytes 近年已在扩展对 CPU、AMD ROCm、Intel 等后端的支持，具体以你所用版本的官方说明为准。如果你的硬件不是 NVIDIA GPU，请务必查阅官方文档确认对应后端与安装方式，本文示例命令未必适用。
>
> **各后端支持程度差异

## 块 5

**个人化讲解：你的LoRA/QLoRA电脑配置清单**

你好！既然你的目标是快速上手LoRA微调，而且喜欢动手实践，那我们直接看“硬件门槛”。别担心，下面这些数字不是死规矩，是帮你少走弯路的“经验起点”。

**1. LoRA最低配置（先跑通再说）**
*   **CPU**：4核以上（多核能加快数据预处理）。
*   **内存**：16GB+（内存不够容易爆，这是常见坑）。
*   **磁盘**：50GB以上（模型文件+数据集+输出，很快就占满）。
*   **GPU**：推荐NVIDIA显卡，**显存最低8GB**，但**推荐12GB+**。显存越大，能塞的模型和batch就越大，训练越稳。

**2. QLoRA要求（显存不够时的“省显存”方案）**
*   **GPU**：本教程按 **NVIDIA CUDA GPU** 来写命令。技术上其他后端可能也行，但请以bitsandbytes官方说明为准，别自己瞎试。
*   **显存**：QLoRA的核心优势就是**用更小显存跑大模型**（通过4-bit量化）。对于常见的7B级别模型，**参考下限约8~12GB**（短序列、小batch），**推荐16GB+**。

> **⚠️ 重要提醒：别把“参考下限”当“保证”**
> 实际显存占用受这些条件影响，条件不同结果天差地别：
> *   **模型规模**：7B是基础，13B/70B显存需求会显著上升。
> *   **序列长度

## 块 6

### 5.1 Python版本检查

先确认你的Python版本是否达标。在终端执行：

```bash
python --version
```

输出应类似 `Python 3.8+`。如果版本低于3.8，建议先升级Python，否则后续安装PEFT和bitsandbytes可能遇到兼容性问题。推荐使用Python 3.10或更高版本。

---

### 5.2 PyTorch检查（关键步骤）

**为什么重要**：LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的4-bit量化要求安装**与你的CUDA版本匹配的GPU版PyTorch**。直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

**正确安装方式**：访问[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)，选择你的CUDA版本。例如NVIDIA CUDA 12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

请根据实际CUDA版本选择对应的cu版本（如cu118、cu121等）。

**验证安装**：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # 查看torch编译使用的CUDA版本
```

如果`torch.cuda.is_available()`输出`True`，说明PyTorch能看到GPU。

> **注意**：`True`只说明PyTorch侧能用GPU，**不代表bitsandbytes的4-bit量化路径可用**。后者还需检查bitsandbytes是否正确匹配本机CUDA，具体见后续章节。

---

### 5.3 查看GPU信息

终端执行：

```bash
nvidia-smi
```

输出应包含：GPU型号、CUDA版本、显存大小。例如：

```text
RTX 4090
CUDA 12.x
Memory 24GB
```

---

### 六、安装PEFT库

**PEFT是什么**：Hugging Face官方参数高效微调库，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter。

**安装**：

```bash
pip install peft
```

**验证**：

```python
import peft
print(peft.__version__)
```

成功输出版本号即安装完成。

---

### 七、安装bitsandbytes

**要求**：必须与你的CUDA环境匹配。安装命令：

```bash
pip install bitsandbytes
```

**验证**（确保4-bit量化路径可用）：

```python
import bitsandbytes as bnb
print(bnb.__version__)
```

## 块 7

**bitsandbytes 是干嘛的？**

一句话：它让你的 GPU 能用 4-bit 或 8-bit 精度跑模型，是 QLoRA 训练的核心依赖。

**为什么你必须装？**

本教程的 QLoRA 方案基于 4-bit 量化，而 `bitsandbytes` 就是实现这个量化的工具。所以，**必须安装**。

> 注意：QLoRA 本身不绑定某个量化库，但本教程统一用 `bitsandbytes` + NVIDIA CUDA 演示。如果你换别的后端，以对应文档为准。

**安装命令（一条搞定）：**

```bash
pip install bitsandbytes
```

装完别急着跑，下一步是检查 CUDA 兼容性（见 7.2 之后的内容）。

---

**本块学习检查**：在你自己的环境里执行上面的安装命令，

## 块 8

**为什么先查CUDA？**
你的目标是跑LoRA微调，而bitsandbytes是量化工具，它依赖CUDA来调用GPU。版本不匹配，轻则CUDA不可用，重则量化失败、模型加载报错。所以，动手前先确认环境。

**两步检查（动手做）**

1. **查CUDA版本**  
   终端运行：
   ```bash
   nvidia-smi
   ```
   看输出里的`CUDA Version`。

2. **查bitsandbytes**  
   运行：
   ```bash
   python -m bitsandbytes
   ```
   正常会显示CUDA版本和GPU信息。

**你的检查任务**  
现在执行上面两条命令，确认输出正常。若`n

## 块 9

**本块学习检查**：请用一句话说明，为什么在Windows上遇到bitsandbytes报错时，优先推荐改用WSL2？

---

**9.1 推荐使用WSL2**

在Windows上做LoRA微调，建议走这条路径：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

原因很简单：**Linux环境的兼容性更好**。

---

**9.2 Windows原生可能遇到的问题**

如果你坚持在Windows原生环境跑，可能会遇到：

**问题1：bitsandbytes无法加载CUDA动态库**

这个报错可能由多种原因引起，常见的有：

- 你安装的`bitsandbytes`版本，和本机的CUDA或GPU版PyTorch**不匹配**；
- 系统里**缺少或找不到**对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本的支持**不如Linux完善**（不同

## 块 10

## 九、Windows用户特别注意事项：推荐使用WSL2

在Windows上安装LoRA环境时，最常见的坑就是**CUDA版本不匹配**。这个问题会导致你明明装好了PyTorch，但一运行就报错。**推荐使用WSL2**（Windows Subsystem for Linux 2）来避免此类问题，因为它能提供更接近Linux的原生CUDA支持。

**先检查你的环境**，在终端里运行：

```bash
nvidia-smi
```

这个命令会告诉你三件关键信息：
- **NVIDIA驱动**是否正常安装
- **CUDA版本**（驱动支持的版本）
- 你的GPU型号

**为什么要检查？** 因为PyTorch的GPU版本必须和你的CUDA版本匹配。如果版本对不上，后续`import torch`或者QLoRA的4-bit量化就会直接失败。**建议至少8GB显存**，否则QLoRA训练可能因内存不足而中断。

---

## 十、安装完整LoRA环境

假设你已经有了基础环境（`transformers`、`datasets`、`accelerate`），现在来装LoRA需要的核心库。

**第一步：确保PyTorch是GPU版**

如果你还没装GPU版PyTorch，先按你的CUDA版本安装。比如CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**第二步：安装PEFT和bitsandbytes**

```bash
pip install peft bitsandbytes
```

**一步到位**的完整安装命令：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

运行配置命令：

```bash
accelerate config
```

按提示选择你的环境。比如单GPU的情况：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

配置完成后会生成一个`default_config.yaml`文件。

---

## 十二、验证LoRA环境

创建一个测试文件`test_lora_env.py`：

```python
import torch
from peft import LoraConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA:", torch.cuda.is_available())

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

print(config)
```

运行：

```bash
python test_lora_env.py
```

如果成功，说明：
- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、验证QLoRA环境

再测试bitsandbytes是否正常工作：

```python
import torch
import bitsandbytes as bnb

print("CUDA:", torch.cuda.is_available())
print("bitsandbytes:", bnb.__version__)

# 验证4-bit量化配置
quant_config = bnb.nn.Par

## 块 11

**本块学习检查**：对照下方“常见问题排查”表，检查你当前环境是否命中任一问题；若命中，按表中对应方案记录你的解决步骤。

---

### 九、Windows用户特别注意事项（新增）

#### 9.1 推荐使用WSL2

如果你在Windows上遇到报错，优先使用WSL2解决兼容性问题。

**常见问题排查速查表**

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

**动手验证**：运行你的训练脚本，若报错，先对照上表定位原因，再按“解决”列操作。重点确认：CUDA版本是否匹配、PyTorch是否为GPU版。

---

### 十、完整LoRA/QLoRA学习路线

**当前

## 块 12

## 一、为什么需要LoRA？

### 1.1 大模型微调的问题

你手上可能已经跑过Transformers和Trainer了，应该知道像LLaMA、Qwen、Baichuan、Mistral这些现代大语言模型，动辄就是**数十亿甚至上百亿参数**。拿7B模型来说，就是70亿个参数。

如果按传统方式微调——把预训练模型拿来，**更新全部参数**，再保存一个新模型——你会立刻撞上三堵墙：

- **GPU显存不够**：光是把模型加载进去就占掉一大块显存，训练时还要额外空间存梯度和优化器状态；
- **训练时间漫长**：每更新一个参数都要计算梯度，全量更新意味着每步都要算70亿次；
- **存储空间爆炸**：每个微调版本都要保存一份完整的70亿参数模型。

### 1.2 全参数微调到底有多贵？

假设模型有100亿参数。训练时你不仅要保存模型参数 **W**，还得保存：

- **梯度 ∂W**：反向传播算出来的每个参数的梯度；
- **优化器状态**：比如Adam的动量项（moment）。

这三份加起来，实际显存需求**远大于模型本身的大小**。这就是为什么全参数微调在消费级显卡上几乎跑不动。

---

## 二、LoRA核心思想

LoRA（Low-Rank Adaptation）的核心洞察是：**你不需要更新全部参数**。

它把微调过程从“更新整个W”变成“在W旁边加一个小补丁”。这个补丁用两个低秩

## 块 13

### 2.1 不直接修改原模型参数

**传统微调**：直接改原权重 `W`，训练后变成 `W'`。

**LoRA 的做法**：
1. **冻结**原模型参数 `W`（保持不动）；
2. **新增**一个小参数 `ΔW`；
3. 最终模型使用：`W' = W + ΔW`。

> 简单说：LoRA 不碰原模型，只在旁边加一个“补丁” `ΔW`。

---

**动手验证**：想象你有一个训练好的模型（`W`），试着在纸上写出 LoRA 更新权重的公式，并对比传统微调的公式差异。

## 块 14

**3.1 矩阵基础概念**  
Transformer里大量参数都是矩阵。比如注意力层中，`Q = X · Wq`，这里的`Wq`就是一个权重矩阵。假设`Wq`形状是`4096 × 4096`，那它的参数数量就是`4096 × 4096 ≈ 1677万`。

**3.2 直接训练矩阵的问题**  
如果直接微调，需要修改`ΔW`，它同样有`4096 × 4096`个参数，成本太高。

**LoRA的解法**  
LoRA不直接学习完整矩阵，而是把它拆成两个小矩阵`B × A`来组成`ΔW`，原始权重`W`保持冻结，仅训练低秩矩阵`A`和`B`，

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设我们要微调一个大模型，原始权重矩阵是 **4096 × 4096**。如果直接训练整个矩阵，参数量是：

**4096 × 4096 = 16,777,216**（约1677万）

这非常庞大，训练成本极高。

**LoRA的做法**：不直接训练这个大矩阵，而是把它拆成两个小矩阵的乘积。设置 **rank r = 8**，拆成：

- **A矩阵**：8 × 4096，参数量 = 32,768
- **B矩阵**：4096 × 8，参数量 = 32,768

**总参数量 = 65,

## 块 16

**5.1 为什么低秩分解有效？——先动手感受一下**

想象你正在微调一个模型，权重更新量是 `ΔW`。直觉上，这个更新可能涉及成千上万个方向。但 LoRA 论文提出：**模型适应新任务时，`ΔW` 并不需要覆盖所有方向**。

**动手验证**：打开一个调试脚本，打印一个 4096 维向量的实际有效分量数。你会发现，真正影响任务的“有效方向”往往远少于满维度，集中在一个**低维子空间**里（远小于原维度的少数方向）。

**核心假设**：LoRA 论文（Hu et al., 2021）假设权重更新具有较低的内在维度，这一假设在多个任务

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。但如果这张图片的主要变化只是**亮度**、**对比度**和**少量结构变化**，你其实不需要存下全部1600万个像素——用**少量参数**就能描述这些变化。

**LoRA 就是这个思路**：用少量参数来描述模型在微调时的“变化方向”，而不是重新训练和存储整个模型的全部参数。

---

## 六、LoRA 在 Transformer 中的位置

Transformer 的基本流程是：

```text
Input → Embedding → Attention → FFN → Output
```

其中 **Attention** 部分包含四个投影矩阵：**Q、K、V、O**。

LoRA 早期实践中**常见**的做法是只加到 **Q投影** 和 **V投影** 上，代码写法如下：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示只训练 `Wq` 和 `Wv` 的变化。

> ⚠️ **注意**：只选 `q_proj`/`v_proj` 是 LoRA 原论文的**常见选择**，**不是固定标准**。现代实践中也常把 LoRA 加到注意力的全部投影（q/k/v/o）和 FFN 的线性层上。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

下面是一个**覆盖更完整线性层**的配置示例（以 Llama 类模型常见模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

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

> 若当前 PEFT 版本支持，也可用 `target_modules="all-linear"` 自动匹配所有线性层。覆盖模块越多，

## 块 18

## 8.1 rank（r）：低秩矩阵的“宽度”

在LoRA中，`r` 就是低秩矩阵的维度，代码里通常写成 `r=8`。你可以把它理解为微调时“额外学习通道”的宽度——`r` 越大，这个通道越宽，能承载的信息就越多。

**`r` 调大时：**
- 优点：表达能力更强，模型更能适应复杂任务。
- 缺点：参数更多，显存占用也更高。

**`r` 调小时：**
- 参数少、省显存，但表达能力受限，可能学不够。

**经验起点参考（不是铁律）：**

| 任务 | rank（起点参考） |
|------|------------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

> 注意：合适的 `r` 跟模型规模、目标模块、数据量、任务难度都有关，上表只是搜索起点，不是查表定死。

**更可靠的做法：用验证集选 `r`**

与其背“配方”，不如

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

```python
lora_alpha=16
```

这个参数控制LoRA在训练时的**更新强度**。通常用 `alpha/r` 作为缩放比例——`r` 是秩，`alpha` 是你要设的值。直观理解：`alpha` 越大，LoRA 对原模型权重的改动越“猛”；`alpha` 越小，改动越“温和”。

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

作用：在训练时随机“丢弃”一部分 LoRA 参数，防止模型对训练数据记得太死（过拟合）。0.05 表示 5% 的 LoRA 参数会被随机置零。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> ⚠️ 注意：上表是**相对趋势**，不是绝对数值。LoRA 的参数量、显存、速度和文件大小会随 `target_modules` 选择、秩 `r`、模型层数与 `dtype` 变化——比如把 LoRA

## 块 20

## 二、Rank（r）是什么？

### 2.1 Rank表示什么？

**一句话理解**：Rank就是“矩阵的秩”，在LoRA里，它决定了低秩矩阵的“中间维度”有多大。

**为什么叫“低秩”？** 因为原始权重矩阵很大（比如4096×4096），而LoRA用两个很小的矩阵去近似它的变化量ΔW，这个“小”的程度就是由rank控制的。

**具体看个例子**：

假设原始权重W是4096×4096的矩阵。如果你设置`r=8`，那么：

- 矩阵A的尺寸是：**8 × 4096**
- 矩阵B的尺寸是：**4096 × 8**

两个矩阵相乘：B × A = (4096×8) × (8×4096) = **4096×4096**

看到了吗？最终ΔW的尺寸和原始W一模一样，但中间只用了“8”这个维度来传递信息。

**直观类比（动手想象一下）**：

把原始权重想象成一条4096车道的超级大马路。rank=8的意思是：你只开一条8车道的“捷径”来传递更新信息，而不是把整条马路都翻修一遍。rank越大，捷径越宽，能传递的信息越多，但修路成本（参数量、显存）也越高。

**动手验证一下**（不用写代码，用纸笔）：

1. 如果r=4，A和B的尺寸分别是什么？ΔW最终能恢复成多大？
2. 如果r=16呢？对比一下，哪个方案的参数量更大？

> 答案提示：A永远是 r×4096，B永远是4096×r，ΔW永远是4096×4096。r越大，A和B里的数字越多，参数量越大。

**记住这个核心结论**：rank决定了LoRA的“容量”——rank越大，模型能学到的变化越复杂，但训练成本和显存占用也越高。

## 块 21

**Rank控制什么？——用“变化方向”来理解**

把LoRA想象成一个“旋钮”，Rank（r）就是旋钮能拧出的**变化方向数量**。

**低rank（如 r=4）**：旋钮只有4个档位，模型只能学**较少的变化模式**。好处是参数少、显存低、训练快；代价是表达能力有限，复杂任务可能学不好。

**高rank（如 r=64）**：旋钮有64个档位，LoRA拥有**更多自由度**，能学习更复杂任务、适应能力更强。但参数更多、显存增加、训练成本提高。

**一句话总结**：Rank越高，LoRA能学的变化方向越多，但开销也越大。

## 块 22

**3.1 不使用LoRA：全量训练**
直接训练完整矩阵，参数量为：

```text
4096 × 4096 = 16,777,216
```

约 **1677万** 参数。这是“基准线”，所有LoRA效果都与此对比。

---

**3.2 使用LoRA：只学增量**
LoRA不碰原矩阵，只训练一个低秩分解，公式：

```text
参数量 = r × (输入维度 + 输出维度)
```

LoRA参数量仅计算低秩矩阵A和B，不包含原权重。实际更新量为ΔW=(α/r)BA，其中α为缩放系数。

- **r=8**：`8 × (4096+4096

## 块 23

### 4.1 Rank太小的问题

如果你把 `r` 设得很小，比如 `r=2`，会出什么问题？

**核心症状**：模型“学不进去”。

- **学习能力不足**：低秩矩阵能表达的信息有限，相当于给模型戴了副“近视镜”，看不清新任务里的复杂模式。
- **欠拟合**：训练时 `loss` 下降得很慢，怎么调都降不到理想水平。
- **新知识表达不出来**：即使训练结束，模型对新任务的理解也很肤浅，实际效果提升非常有限。

**直观表现**：
```text
训练loss下降慢
效果提升有限
```

**一句话总结**：`r` 太小 = 模型“记性差”，新知识装不下。

---

### 4.2 Rank太大的问题（对比看）

反过来，如果你把 `r` 设得很大，比如 `r=256`，又会怎样？

**核心症状**：LoRA 的“省钱”优势被削弱。

- **参数增加**：要训练的矩阵变大了。
- **显存增加**：显卡内存占用更高。
- **训练速度下降**：计算量变大，跑得更慢。

**为什么说“优势降低”？** 用数字说话（以 `4096 × 4096` 的权重矩阵为例）：

| rank r | LoRA参数量 | 完整矩阵参数量 | 占比

## 块 24

**Rank选择：从经验值开始，但别照搬**

你是动手实践型学习者，目标是把LoRA用起来。这里给你一张“起步参考表”，注意是**起点**，不是标准答案：

| 任务类型 | rank起点 |
|---------|---------|
| 简单分类 | 4~8 |
| 文本风格调整 | 8~16 |
| 指令微调 | 16~64 |
| 复杂领域任务 | 64以上 |

**为什么不能直接抄？** 因为实际效果受多个因素影响：数据量、模型大小、序列长度、batch大小（含梯度累积）、目标模块选择与数量、GPU显存，以及验证集表现。同一个`r`值，在不同模型或不同设置下，显存占用和效果可能天差地别。

**实操建议**：先选一个起点值跑通流程，再根据验证集表现调整，而不是一开始就追求“最优”。

---

**Alpha是什么？**

## 块 25

### 6.1 控制LoRA影响强度

咱们直接上手感受一下 `alpha` 和 `rank` 怎么配合。假设你设了 `r=8`，`alpha=16`，那缩放系数就是 `16/8=2`，意思是LoRA的更新会被**放大2倍**。

如果 `alpha=8`，缩放就是 `8/8=1`，保持原始比例，不放大也不缩小。

如果 `alpha=4`，缩放变成 `4/8=0.5`，LoRA的影响就**减弱一半**。

**核心公式**：LoRA加入模型的更新量 = `alpha/r` 这个缩放系数。所以调 `alpha` 就是直接调“强度”的旋钮。

---

### 七、Alpha与Rank的关系（重点）

记住：`alpha/r` 是**关键缩放系数**，但别误会成“唯一决定强度”的开关。实际落到权重上的更新量，还受这些因素影响：

- **学习率**：决定每步对 `A`、`B` 参数的更新步长。`alpha/r` 调大又不下调学习率，容易一起放大更新导致震荡。很多实战里，直接调学习率比频繁改 `alpha` 更常用。
- **初始化**：常见实现里 `A` 用小随机值、`B` 初始化为0，训练开始时 `ΔW = BA = 0`（不扰动原模型），之后再逐步学习。
- **训练状态**：`ΔW` 是训练**逐步累积**出来的。同样的 `

## 块 26

**动手试一下：配置1 vs 配置2**

先看两组代码：

```python
# 配置1
r=8
alpha=16

# 配置2
r=16
alpha=32
```

**关键观察**：虽然 rank 从 8 变成 16，但 `alpha / r` 的比例都是 **2**（16÷8=2，32÷16=2）。

**为什么这很重要？** 缩放比例相同，意味着 LoRA 对模型权重的整体影响强度保持一致。调整 rank 时，**同步调整 alpha** 是常见经验做法，目的是让 `α/r` 比例大致稳定——但这只是经验起点，不是硬性规则。

**实操建议**：你可以把 `alpha =

## 块 27

**动手试试：小模型实验（方案1）**

你现在的目标是快速跑通LoRA微调流程，所以从**方案1**开始最合适。

- **配置**：`r=8`，`alpha=16`
- **特点**：参数少、训练快，适合先做测试，验证流程没问题后再加大规模。

**为什么这样配？**

- `r`（Rank）控制**LoRA容量**——即新增参数的数量。`r=8`表示容量较小，但足够小模型实验。
- `alpha`（lora_alpha）控制**LoRA更新幅度**——即每次参数更新的步长。`alpha=16`是`r`的两倍，这是一个常见的经验搭配，让更新幅度与容量匹配。

**在代码里怎么用？**

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

## 块 28

## 误区1：rank越大一定越好？

很多初学者容易陷入“参数越大越强”的直觉，但**rank并非越大越好**。

**为什么？** 关键看任务复杂度：

- **任务简单时**，大rank不仅**浪费显存**，还会**增加训练成本**。
- 模型并不需要那么高的表达能力，多余的容量只会拖慢训练。

**实操建议**：从较小的rank（如8）开始，观察效果后再逐步调整，而不是一上来就追求大rank。

---

## 误区2：只调rank，不调alpha

这是最常见的配置错误。看个例子：

**原配置：**
```python
r=8
alpha=16
```

**修改后：**
```python
r=64
alpha=16
```

此时缩放比例变为：

```text
16/64 = 0.25
```

**问题在哪？** LoRA的更新强度由缩放因子 `α/r` 与低秩矩阵乘积 `BA` 共同决定。你把rank从8提到64，但alpha没变，导致缩放比例从 `16/8=2` 降到 `0.25`——**LoRA对模型的影响可能被削弱**，甚至可能等于白训。此外，`α` 和 `r` 的绝对大小也会影响训练动态，不能只看比值。

**正确做法**：调整rank时，alpha要同步调整，保持 `α/r` 在一个合理范围。

---

## 误区3：alpha越大越好？

同样错误。alpha过大会导致 `α/r` 缩放过大，引发：

- **更新过强**，模型参数被过度修改；
- **训练不稳定**，loss震荡甚至发散。

**注意**：`α/r` 过大只是训练不稳定的**可能原因

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。它的核心思想是：不微调整个大模型，而是只训练一小部分新增参数。

**LoraConfig 就是这份“微调说明书”**——它告诉 PEFT 库四件事：

| 它回答的问题 | 对应参数 |
|---|---|
| LoRA 放在模型的哪些层？ | `target_modules` |
| LoRA 的规模（容量）多大？ | `r` |
| LoRA 的训练强度多大？ | `lora_alpha` |
| 是否用 dropout 防过拟合？ | `lora_dropout` |

### LoRA 微调的整体流程

```text
加载基础模型
    ↓
创建 LoraConfig
    ↓
指定 LoRA 插入位置
    ↓
生成 LoRA Adapter
    ↓
训练 LoRA 参数
```

可以看到，**LoraConfig 是第二步**——它决定了后面所有步骤怎么执行。

### 一句话理解

> 基础模型是“原材料”，LoraConfig 是“加工图纸”，LoRA Adapter 是“加工出来的零件”。

你只需要写好这份“图纸”，PEFT 库就会自动在模型里插入 LoRA 层并完成训练。

---

**本块

## 块 30

## 1.2 一个简单示例——动手敲一遍

你正在学LoRA微调，咱们直接上手。下面这段代码就是LoRA最核心的配置，先把它复制到你的Python环境里跑一遍（不需要GPU，纯配置）：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05
)
```

跑完没报错，说明你已经成功创建了一个LoRA配置对象。现在看这张表，记住每个参数管什么：

| 参数 | 作用 |
| --- | --- |
| target_modules | 决定插入哪些层 |
| r | 决定LoRA矩阵大小 |
| lora_alpha | 控制缩放强度 |
| lora_dropout | 控制随机失活 |

**给你的速记口诀**：`target_modules`选层，`r`定大小，`alpha`管缩放，`dropout`防过拟合。

---

## 二、LoRA配置整体流程——配置怎么用起来

光有配置还不够，你得把它接到模型上。流程是这样的：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

实际操作代码：

```python
from peft import Lora

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些部位动手”**。

你不需要改整个模型，只需要挑几个关键“关节”装上 LoRA 就行。

### Transformer 的常见结构

大模型里的 Attention 部分，通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

你可以把 `q_proj`、`v_proj` 想象成模型里的“旋钮”——LoRA 就是去拧这些旋钮。

### 怎么指定？

用代码告诉 LoRA 你要改哪些层，比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思是：**只给 Query 投影层和 Value 投影层加 LoRA**，其他层不动。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开理解：

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

所以，调整 Q 和 V，就能直接改变模型的行为方式——这就是为什么它们常被选为 LoRA 的注入点。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型都通用的“标准配置”。不同模型的模块命名可能不同，**配置前一定要先查看模型实际的模块名**（用 `model.named_modules()`），再决定 `target_modules`。

---

### 偷懒技巧：`"all-linear"`

较新版本的 PEFT 支持一个便捷写法——让 LoRA 自动匹配（几乎）所有线性层：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
model = AutoModelForCausalLM.from_pretrained("your/base-model")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",   # 自动匹配所有线性层（一般不含输出层）
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量，确认 LoRA 已注入
```

**注意权衡**：覆盖的层越多，可训练参数和显存占用就越大。是否用 `"all-linear"`，要在**参数量/显存

## 块 32

**动手试一下：先只调Q和V**

既然你目标是快速上手LoRA微调，而且喜欢动手操作，那我们就从最轻量的方案开始。

**方案1：只调整Q和V**

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这个配置的意思是：微调时，只修改模型注意力机制里的“查询”（Q）和“键值”（V）两个投影层。

**为什么先选它？**

- **参数少**：只动两个层，需要训练的参数总量小。
- **训练快**：参数少，计算量小，训练速度自然快。
- **轻量起点**：这是一个常见的起步配置，适合先跑通流程、观察效果。

**你的操作建议**：先用这个配置跑一次微调，把流程走通。如果效果不理想，再考虑增加模块。

**动手验证一下**：你可以把上面的代码块复制到你的训练脚本里，替换掉原来的`target_modules

## 块 33

## LLaMA系列：target_modules怎么配？

既然你的目标是LoRA微调，而且喜欢动手实践，那我们直接看最常用的配置。LLaMA系列模型（比如Llama 2、Llama 3）里，最经典的`target_modules`写法是：

```python
[
"q_proj",
"v_proj"
]
```

**为什么是这两个？** 简单说，`q_proj`和`v_proj`分别对应注意力机制里的Query（查询）和Value（值）投影层。LoRA通过在**部分**层注入低秩矩阵来微调，而不是动全部参数——这就是它高效的原因。选Q和V是实践中最常见、效果也稳的组合。

> **动手验证一下**：你可以加载一个LLaMA模型（比如`meta-llama/Llama-2-7b-hf`），用上面的配置创建`LoraConfig`，然后打印可训练参数数量。你会发现可训练参数只占总参数的极小比例——这就是LoRA省显存的直观体现。

---

## 对比：GPT-2为什么不一样？

GPT-2的写法完全不同，常见配置是：

```python
[
"c_attn"
]
```

**关键区别在这里**：GPT-2的注意力层用的是Transformers库里的`Conv1D`，**不是**普通的`nn.Linear`。而且`Conv1D`把Q/K/V三个投影**合并**在了一个叫`c_attn`的模块里——所以只需要指定一个目标。

更坑的是权重布局：`Conv1D`的权重形状是`(in_features, out_features)`，和`nn.Linear`的`(out_features, in_features)`**正好转置**。如果不处理这个差异，LoRA注入的低秩矩阵就会放错位置，模型直接崩。

**解决办法**：在`LoraConfig`里加一个参数：

```python
fan_in_fan_out=True
```

告诉PEFT：“这个层的权重是转置布局，你注入矩阵的时候要按转置来。”

完整的可运行示例（GPT-2）：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,

## 块 34

## BERT的target_modules配置：动手查，别照抄

你正在学LoRA微调，目标很明确：让BERT跑起来。但这里有个关键点——**不能直接复制其他模型的配置**。BERT的注意力层里，LoRA通常只作用于 `query` 和 `value` 两个投影，所以 `target_modules` 一般写成：

```python
["query", "value"]
```

但问题来了：不同版本的BERT、不同库的实现，模块命名可能不一样。你如果直接照抄，很可能报错或没效果。

**正确做法是：先查再配。**

用下面这段代码，把模型的所有模块名打印出来：

```python
for name, _ in model.named_modules():
    print(name)
```

在输出里找 `q_proj` 和 `v_proj` 这样的名字（不同版本可能叫 `query`、`value` 或其他），确认存在后，再把这些名字填进 `target_modules`。

**匹配机制**：`target_modules` 里的字符串是按模块名做**后缀匹配**的。比如写 `"q_proj"`，所有以 `q_proj` 结尾的层都会命中（各层各头都算）。你也可以写更完整的名字精确定位，或用正则

## 块 35

**r是什么？——LoRA的“瘦身开关”**

想象你要更新一个4096×4096的大矩阵（约1678万参数）。直接全量更新太费资源，LoRA的做法是：**不碰原矩阵，只训练两个小矩阵A和B**，让它们的乘积近似等于要更新的变化量ΔW。

公式就是：**ΔW = (α/r) × B × A**，其中α为缩放系数，r为秩。

这里的**r**，就是这两个小矩阵的“中间维度”——它决定了A和B的尺寸。

**动手算一下：**
- 原矩阵：4096×4096
- 设 `r=8`
- 则 A 是 8×4096，B

## 块 36

## 7.1 alpha作用：控制LoRA更新幅度

LoRA最终更新不是 `BA`，而是 `(alpha/r)BA`。

**关键点**：alpha 控制 LoRA 更新幅度。

**动手算一下**：

配置 `r=8, lora_alpha=16` 时，缩放系数 = 16/8 = 2，表示 LoRA 影响增强。

---

## 7.2 alpha常见设置

常见起点：`alpha = 2 × r`（注意：这只是经验起点，不是通用规则，最终以验证集表现为准）。

示例：
- `r=8, alpha=16`
- `r=16, alpha=32`

---

**本块学习检查**：给定 `r=4, lora_alpha=8`，计算缩放系数并说明对Lo

## 块 37

**8.1 dropout在做什么？**

训练时，它随机关闭部分LoRA连接，防止模型“死记硬背”训练数据（过拟合）。

**怎么配？**
```python
lora_dropout=0.05
```
含义：5%的LoRA路径随机失活。

**选多少？**（经验起点，看验证集调）

| 数据规模 | 起点参考 |
| ---- | -------- |
| 数据少 | 约0.05~0.1 |
| 数据多 | 约0~0.05 |

数据非常丰富时，可直接设：
```python
lora_dropout=0
```

**动手试**：先设0.05跑通，若验证集表现好

## 块 38

**9.1 基础配置：动手搭一个LoRA配置**

你正在准备微调一个7B模型，目标是LoRA微调。这一节我们直接上手，看一个最常用的基础配置。先记住一个感觉：**这个配置就是LoRA的“出厂设置”，适合大多数7B模型测试**。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：秩的大小，8是常见起点
    lora_alpha=16,      # 更新缩放：控制微调强度，通常设为r的2倍
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的q和v投影
    lora_dropout=0.05,  # 防止过拟合：训练时随机丢弃5%的LoRA参数
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（如GPT类）
)
```

**四个核心参数，逐个拆解：**

| 参数 | 值 | 作用 | 怎么理解 |
|------|-----|------|---------|
| `r` | 8 | LoRA容量 | 秩越大，能学的新知识

## 块 39

## 九、完整LoraConfig案例（重点）

### 9.1 基础配置：一次跑通LoRA微调全流程

你现在的目标是**LoRA微调**，而且喜欢动手实践——那我们就直接进入一个能跑通的完整案例。这个案例用BERT做**二分类任务**（`num_labels=2`），把LoRA配置、训练、保存、重载验证全部串起来。

#### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：BERT模型采用Apache 2.0许可证，可商用
# 注意：bert-base-uncased FP16 推理需要至少2GB显存，训练建议4GB以上
# 加载带分类头的基础模型（分类头是新初始化的，需要一起训练/保存）
try:
    base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
except (OSError, ValueError) as e:
    print(f"模型加载失败，请检查网络连接或模型名称：{e}")
    raise
```

注意：这里的分类头是**随机初始化**的，如果不训练它，模型输出就是乱的。

#### 第二步：配置LoraConfig——核心参数逐个看

```python
lora_config = LoraConfig(
    r=8,                    # LoRA秩，决定低秩矩阵的维度；LoRA冻结原始权重W，仅训练低秩矩阵A和B，增量ΔW=(α/r)BA
    lora_alpha=16,          # 缩放因子，控制LoRA更新的强度
    lora_dropout=0.05,      # 防止过拟合的随机丢弃率
    bias="none",            # 不训练偏置项
    task_type="SEQ_CLS",    # 任务类型：序列分类
    target_modules=["query", "value"],  # 只微调BERT注意力中的query和value层
    modules_to_save=["classifier"],     # 关键：把新分类头也纳入训练并保存
)
```

**`modules_to_save`为什么关键？** 因为分类头是随机初始化的，如果只保存LoRA权重，重载时分类头会退回随机状态，结果全错。加上它，分类头会**参与训练并随Adapter一起保存**。

#### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 应看到 LoRA 参数 + classifier 一起可训练
```

#### 第四步：训练后保存

```python
# 训练过程略...
try:
    model.save_pretrained("./lora_ckpt")
except (OSError, ValueError) as e:
    print(f"模型保存失败，请检查磁盘空间或路径权限：{e}")
    raise
```

`save_pretrained`保存的是**Adapter + `modules_to_save`模块**，不是完整基础模型。

#### 第五步：重载并验证分类头

```python
try:
    reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")
except (OSError, ValueError) as e:
    print(f"模型重载失败，请检查路径或基础模型：{e}")
    raise

# 验证：重载后的分类头权重应与保存前一致
# 注意：base_model.model.classifier路径依赖基础模型结构，若结构不同可改用model.named_parameters()查找
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

**重载要点**：必须先加载**同一个**基础模型，再用`PeftModel.from_pretrained`挂载Adapter。

---

### 动手验证任务

跑完上面的代码

## 块 40

## 10.1 普通文本生成：从最简单的配置开始

你现在要学的是 LoRA 微调，咱们从最常用的场景——普通文本生成——入手。这个场景下的参数配置非常简洁，是理解 LoRA 的绝佳起点。

**推荐配置：**

```python
r=8
alpha=16
dropout=0.05
```

**逐个拆解，让你明白每个参数在干什么：**

- **`r=8`**：这是 LoRA 的“秩”。你可以把它理解为**低秩矩阵的秩，控制可训练参数的数量和更新表达能力**——r 越大，模型能学的新知识越多，但占用的显存和计算量也越大。普通文本生成任务，8 是一个性价比很高的起点。
- **`alpha=16`**：这是缩放系数。实际缩放为 `alpha/r`，影响强度由 `alpha` 和 `r` 共同决定。注意看，`alpha` 是 `r` 的两倍——这是一个常用的经验比例，让更新幅度既不过猛也不过弱。
- **`dropout=0.05`**：这是应用于低秩矩阵的 dropout 率，用于防止过拟合。5% 是一个很轻的 regularization，适合普通文本生成这种相对“温和”的任务。

**为什么这么配？** 普通文本生成任务（比如续写、闲聊）对模型原有能力的依赖较强，不需要大幅调整模型行为。所以 `r` 和 `alpha` 都取较小的值，让 LoRA 做“微调”而不是“大改”。

---

### 对比一下：其他任务怎么配？

为了帮你建立直觉，这里给出同一份资料中另外两种场景的推荐配置，你可以感受一下规律：

| 任务类型 | r | alpha | dropout | 特点 |
|---------|---|-------|---------|------|
| 普通文本生成 | 8 | 16 | 0.05 | 轻量调整 |
| 指令微调 | 16 | 32 | 0.05 | 中等力度 |
| 领域知识增强 | 32 | 64 | （未指定） | 大幅调整 |

**规律总结：** 任务越复杂、越需要模型学习新知识，`r` 和 `alpha` 就越大。`alpha` 始终是

## 块 41

**动手配置：普通文本生成的LoRA**

你正走向LoRA微调实战，先拆解这个核心配置。它像给模型装“小开关”，只调一部分参数。

```python
LoraConfig(
    r=8,              # LoRA容量：8，决定新增参数多少
    lora_alpha=16,    # 缩放系数：16，配合r使用
    target_modules=["q_proj", "v_proj"],  # 改哪里：注意力层的q和v投影
    lora_dropout=0.05 # 防过拟合：随机丢弃5%的LoRA参数
)
```

**四个旋钮，逐个上手：**

- **target_modules**：决定“改哪里”。这里选

## 块 42

## 为什么LoRA还不够省显存？

你已经掌握了LoRA的核心思想——冻结原模型，只训练一小部分低秩矩阵。但这里有个关键问题：**LoRA只是减少了“训练”的参数，并没有减少“加载”的负担**。

### 动手算一笔账

假设我们要微调一个7B模型，用FP16精度存储：

```
7B × 2 bytes ≈ 14GB显存
```

这14GB只是**基础模型本身**占用的空间。实际训练时，你还得额外腾出空间给：

- **激活值**：前向传播时每层的中间输出
- **梯度**：反向传播时计算的更新方向
- **优化器状态**：如Adam的动量项

把这些都加起来，实际显存需求会**远高于14GB**。

### 关键结论

> LoRA的训练参数虽然少，但**基础模型必须完整加载进显存**。模型有多大，这个“底座”就有多重。

这就是为什么即便有了LoRA，在消费级GPU上微调大模型依然困难

## 块 43

**本块学习检查**：读完下面内容后，请说出QLoRA与LoRA在“基础模型”上的关键区别，并指出QLoRA训练时哪部分参数是冻结的、哪部分是可训练的。

---

QLoRA = Quantized LoRA，核心就一句话：**先把基础模型压成4-bit，再在上面跑LoRA**。

对比一下流程就清楚了：

- LoRA：FP16模型 + LoRA
- QLoRA：4-bit模型 + LoRA

训练时，**基础模型（4-bit权重）是冻结的**，不更新；真正训练的是**LoRA参数（FP16/BF16）**。

所以整体结构是：输入 → 4-bit量化模型 → LoRA Adapter → 输出。

重点记住：量化只针对**基础模型权重**，**不是LoRA

## 块 44

## 3.1 模型存储方式：为什么量化能省显存？

先看一张表，记住一个核心概念：**参数精度越高，占的显存越多**。

| 类型 | 每个参数占用 |
|------|------------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

**动手算一下（7B模型）：**

- FP16：7B × 2 ≈ **14GB**
- 4-bit：7B × 0.5 ≈ **3.5GB**

显存直接减少约 **75%**。

> ⚠️ **注意**：这个3.5GB只是**理论下界**（只算了量化后的权重本身）。实际运行还要加上量化常数、分组元数据、未量化的模块（如norm、embedding、lm_head、LoRA参数）以及计算缓存和激活值。所以真实占用会明显更高

## 块 45

**NF4是什么？——用“量身定做的尺子”来理解**

NF4全称NormalFloat4，中文叫“正态分布4位浮点”。别被名字骗了，它**不是**IEEE那种通用4位浮点格式，而是QLoRA论文为神经网络权重“量身定做”的一套4-bit量化码本（codebook）。

**核心逻辑**：神经网络权重近似正态分布，0附近数值密集。NF4预先定好16个（2⁴）量化取值点，这些点按标准正态分布分位数排布——0附近分得更细，远离0分得粗。

**量化过程**：把每组权重归一化后，映射到最接近的码本取值，只存对应的4-bit索引；反量化时按码本和缩放因子还原。

**一句话总结**：

## 块 46

**4.2 为什么适合模型权重？**

大模型训练后的权重近似**正态分布**，大量参数集中在0附近。普通INT4用平均划分（如[-8,-7,...,7]），无法精细表示中心密集区域。

**NF4**则根据正态分布设计量化范围：**0附近精度更高**，更贴合权重分布，精度损失更低。

**4.3 NF4量化过程**

原始FP16权重（如0.023、-0.145、0.876）基于**分位数量化**，通过查找表映射到NF4的16个离散值，保存4-bit值。实际实现中通常先进行**absmax归一化**，再映射。推理时反量化回近似FP16。

---

**

## 块 47

## 5.1 为什么需要双重量化？

你已经知道4-bit量化能把FP16权重压成4-bit，但别忘了——**量化本身也要存“说明书”**。

每组权重都要额外保存两个参数：

- **scale**（缩放因子）
- **zero point**（零点偏移）

这些“量化参数”同样占显存。模型越大，分组越多，这些额外开销就越可观。

## 5.2 双重量化思想

普通量化只做一步：

> FP16权重 → 4-bit权重 + scale参数

双重量化多做一步：

> **把scale参数本身也量化一遍**

也就是：

- **第一次量化**：量化模型权重
- **第二次量化**：量化量化参数

结构变成：

> FP16

## 块 48

## 6.1 为什么需要分页优化器？

训练时显存占用不只是模型本身，还有**梯度**和**优化器状态**。以 Adam 为例，它需要保存**一阶动量**和**二阶动量**，这部分开销不小。

QLoRA 虽然大幅降低了模型本身的显存占用，但训练过程中仍可能遇到**显存突然不足**的情况——这就是所谓的"峰值 OOM"。

## 6.2 分页优化器思想

Paged Optimizer 借鉴了操作系统里的**虚拟内存**思路，核心机制是：

> 借助 NVIDIA 的 **Unified Memory（统一内存）**，当显存出现峰值、即将 OOM 时，自动把优化器状态"分页"换出到 CPU 内存；需要参与计算时再换回 GPU。

流程如下：

```text
GPU显存
  ↓
出现峰值（接近/超出显存上限）
  ↓
优化器状态按

## 块 49

**本块学习检查**：读完下面两个类比和代码片段后，请你在自己的训练脚本里找到 `TrainingArguments` 那一行，确认 `optim` 参数当前的值是什么。如果还没设置，就按下面的代码加上 `optim="paged_adamw_8bit"`，然后跑一次 `--help` 或查文档，确认这个取值在你当前版本里存在。

---

**6.3 类比理解**

把显存想象成你的电脑桌面。

**普通方式**：所有东西（模型参数、梯度、优化器状态）都堆在桌面上。桌面满了，就再也放不下任何东西，训练直接报错（OOM）。

**分页优化器**：就像你整理桌面——常用的文件（当前计算需要的参数）放在桌面上，暂时不用的文件（优化器状态）存到柜子里（CPU内存）。需要的时候再从柜子里取出来。这样桌面永远有空间，训练就不会因为显存爆掉而中断。

---

**6.4 如何在训练中实际启用分页优化器？**

原理懂了，代码上其实很简单。分页优化器是通过 `TrainingArguments` 的 `optim` 参数选择的，不需要你手动写换页逻辑，bitsandbytes 库已经帮你实现了：

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

几个要点：

- `optim="paged_adamw_8bit"` 和 `"paged_adamw_32bit"` 是 QLoRA 论文里常用的分页优化器取值（依赖 bitsandbytes）。前者优化器状态用 8-bit 存储、更省显存；后者精度更高但占用更多。

## 块 50

## 7.1 4-bit量化：把模型“瘦身”

你的目标是LoRA微调，但显存不够？QLoRA的核心思路就是：**先把大模型压缩，再微调**。

原始模型权重通常是FP16（16位浮点数），占用显存大。4-bit量化将其转换为NF4格式（一种基于分位数的4-bit数据类型），权重存储大小缩小约4倍：

```text
FP16 → NF4 4-bit
```

**动手理解**：想象一张高清照片（FP16）压缩成小尺寸缩略图（NF4）。细节略损，但整体轮廓还在——模型能力基本保留，显存却省了一大截。

---

## 7.2 双重量化：连“尺子”也压缩

量化时需要一个`scale`（缩放系数），相当于“尺子”，用来把压缩后的数值还原。这个尺子本身也占空间。

双重量化是对量化过程中产生的缩放因子（scale）再进行一次量化，从而减少缩放因子本身的存储开销：

```text
scale → 再次量化
```

**动手理解**：你不仅压缩了照片，连照片旁边的“说明标签”也一起压缩了。省一点是一点

## 块 51

## 4-bit 加载与 QLoRA 核心配置

你已经知道 LoRA 是给大模型“加小尾巴”来微调，而 QLoRA 的关键一步，就是先把模型“压缩”到 4-bit 再加载，这样显存才够用。下面我们拆开看四个配置项。

### 1. `load_in_4bit`：开启 4-bit 加载

这是总开关。设为 `True`，模型权重就以 4-bit 精度载入显存，大幅节省空间。不开启，后面三项都无从谈起。

### 2. `bnb_4bit_quant_type`：量化方式选 `"nf4"`

NF4 是一种专门为 4-bit 设计的量化格式，比普通 4-bit 更聪明——它把数值映射到更合理的分布上，减少精度损失。QLoRA 默认推荐它。

### 3. `bnb_4bit_use_double_quant`：开启双重量化

简单理解：量化过程本身也会产生一些“额外开销”（比如缩放系数），双重量化就是把这些开销再量化一次，进一步省显存。建议开启。

### 4. `bnb_4bit_compute_dtype`：计算精度（最容易踩坑）

这里要区分两个概念：

- **存储 dtype**：权重以 4-bit（NF4）存着，省显存。
- **计算 dtype**：真正做矩阵运算时，4-bit 会被**反量化**成更高精度来计算，这个精度就是 `bnb_4bit_compute_dtype` 控制的。

**为什么必须显式指定？** 不设置时默认通常是 `float32`，计算慢且显存开销大。QLoRA 配置里几乎总是显式指定。

**选 BF16 还是 FP16？关键看你的 GPU。** 不要写死 `bfloat16`——它需要较新 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系及更早）不支持，必须用 `float16`。

**推荐做法：自动判断**

```python
import torch
from transformers import BitsAndBytesConfig

# 根据 GPU 能力选择：支持 BF16 就用 bfloat16，

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个**能直接跑的 QLoRA 训练骨架**，让你看清各部分如何协作。

> **模型许可证**：本示例使用 Llama 2 系列模型，遵循 Llama 2 Community License（商用月活>7亿需Meta授权）。实际使用时请根据所选模型确认对应许可证。

### 整体流程（5步走）

1. **配置 4-bit 量化** → 2. **加载模型** → 3. **为 k-bit 训练做准备** → 4. **注入 LoRA** → 5. **训练**

下面逐段拆解代码，每一步对应什么作用。

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

- `load_in_4bit=True`：以 4-bit 精度加载模型，这是 QLoRA 的核心。
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化类型。
- `bnb_4bit_use_double_quant=True`：开启双重量化，进一步省显存。
- `bnb_4bit_compute_dtype`：**计算精度**。这里自动选择：GPU 支持 bfloat16 就用它，否则用 float16。这决定了实际运算时的精度。

> **动手验证**：如果你有 GPU，可以打印 `compute_dtype` 看看你的环境选的是哪个。

---

### 第2步：以 4-bit 加载基础模型

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

- 把第1步的 `bnb_config` 传给 `quantization_config`，模型就会以 4-bit 加载。
- `device_map="auto"`：让库自动分配模型到可用设备。
- 如果 tokenizer 没有 pad_token，用 eos_token 代替——这是 Causal LM 的常见操作。

---

### 第3步：关键——为 k-bit 训练做准备

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步**不能省**。它做了两件重要的事：

- 处理 norm 层和输出层的精度，让它们在训练中保持较高精度（而不是 4-bit）。
- 开启梯度检查点所需的输入梯度等设置。

`use_gradient_checkpointing=True` 与后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 是**对应**的——一个在模型侧准备，一个在训练参数侧开启。

---

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

- `r=8`：低秩矩阵的秩，控制新增参数量。
- `lora_alpha=16`：缩放因子，通常设为 r 的 2 倍。
- `target_modules=["q_proj", "v_proj"]`：只给注意力层的 q 和 v 投影注入 LoRA。**按实际模型模块名调整**，也可以用 `"all-linear"` 自动选择。
- `model.print_trainable_parameters()`：打印可训练参数量，确认只有 LoRA 参数在训练。

---

### 第5步：准备数据

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

- 这是**极简示例数据**，真实场景要换成足够规模的数据。
- Causal LM 的 `labels` 直接复制 `input_ids`——模型预测下一个 token，所以标签就是输入本身。
- `mlm=False`：表示这不是掩码语言建模，而是自回归语言建模。

---

### 第6步：训练参数——分页优化器 + 梯度检查点

```python
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    fp16=(compute_dtype == torch.float16),
    bf16=(compute_dtype == torch.bfloat16),
)
```

- `optim="paged_adamw_8bit"`：使用分页优化器，避免 4-bit 模型训练时的显存峰值。
- `gradient_checkpointing=True`：与第3步的 `use_gradient_checkpointing=True` 配合，共同开启梯度检查点。

## 块 53

## 把 QLoRA 的五个关键步骤串起来（可跑通骨架）

你现在要做的，是把前面学到的量化配置、模型准备、LoRA 注入、梯度检查点和优化器选择，按顺序组装成一个能跑的最小骨架。这五步缺一不可，顺序也不能乱。

**第一步：`BitsAndBytesConfig`——决定基础模型怎么加载**

这一步负责告诉 HuggingFace：加载模型时用 4-bit 量化，具体用 NF4 格式 + 双重量化，并指定计算精度（compute dtype）。它管的是"模型以什么形态进内存"。

**第二步：`prepare_model_for_kbit_training()`——QLoRA 的必要一步**

这一步是对量化后的模型做"训练适配"。它会做类似把部分层转成计算精度、开启输入梯度以支持梯度检查点等操作。**跳过它可能导致梯度计算错误或显存不足，建议按需调用**。记住：这不是可选项，是必选项。

**第三步：`get_peft_model(..., LoraConfig)`——注入可训练的 LoRA**

基础模型此时是冻结的 4-bit 主干，本身不可训练。这一步把 LoRA 适配器"插"进去，只有 LoRA 参数是可学习的。

**第四步：`gradient_checkpointing=True`——与第二步对应开启**

注意：TrainingArguments 里的 `gradient_checkpointing=True` 和第二步中 `prepare_model_for_kbit_training` 里的 `use_gradient_checkpointing=True` **建议同时开启以节省显存，但并非必须对应**。

**第五步：`optim="paged_adamw_8bit"`——启用分页优化器**

这一步用分页机制缓解峰值显存，是 QLoRA 省显存的最后一环。需安装 bitsandbytes 库，且优化器名称可能因版本而异。

> 上面这个骨架是"能说明关系"的最小版本。真实训练时，你需要换成足够规模的数据、合理的超参，并且按你实际使用的库版本核对 `optim` 的准确名称、`prepare_model_for_kbit_training` 的参数写法，以及 `processing_class`/`tokenizer` 的调用方式。

---

## 你的电脑能跑吗？

**如果只学原理：不需要 GPU。**

**如果要实际运行 QLoRA：**

- CPU：4 核以上
- 内存：16GB 以上
- GPU：NVIDIA CUDA GPU

**显存参考（以 7B 模型为例）：**

QLoRA 的设计目标就是在更小显存上微调大模型。下面是经验参考，不是保证值——实际下限随模型规模、序列长度、batch 大小、是否开梯度检查点等变化。

| 显存 | 适合场景（经验参考） |
|-----|-------------------|
| 8GB | 小模型 LoRA，或较短序列/小 batch 的 7B QLoRA（较紧张） |
| 12GB | 常见可跑 7B QLoRA（

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会误以为LoRA训练完，得到一个“升级版”的完整模型。**不是的**。

LoRA训练时，基础模型（比如Llama-7B）是**冻结**的——它一动不动。真正被训练的是额外加上的**LoRA Adapter**（适配器）。

最终的结构是：

```
基础模型（冻结） + LoRA Adapter（训练）
```

训练完成后，你保存的通常只有两个文件：

```
adapter_model.safetensors   ← LoRA的权重
adapter_config.json         ← LoRA的配置
```

**没有完整模型**。你手里拿到的只是一个“补丁”，不是“完整程序”。

---

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（只有几十到几百MB，而完整模型可能几个GB甚至更大）
- 保存快
- 一个基础模型可以搭配多个Adapter，各干各的活

比如同一个基础模型，可以搭配：

```
客服Adapter  → 做客服对话
医疗Adapter  → 做医疗问答
法律Adapter  → 做法律咨询
```

**缺点：**

部署推理时，需要**基础模型 + Adapter** 两个部分同时在场，缺一不可。这就带来了麻烦——你不能只带Adapter去跑推理。

---

## 二、为什么需要合并LoRA权重？

既然Adapter不能独立使用，那最省事的办法就是：**把Adapter“焊死”回基础模型里**，合成一个完整的模型文件。这样部署时只需要加载一个模型，不用再管Adapter了。

这就是接下来要讲的 `merge_and_unload()` 要做的事——把LoRA权重合并回基础模型，然后卸载Adapter，得到一个干净的完整模型。

---

**

## 块 55

**本块学习检查**：请观察下方“未合并”流程图，指出推理时需要加载哪两个部分？

---

**推理时的问题**

未合并时，模型推理需要同时加载两部分：

- **基础模型**（Base Model）
- **LoRA适配器**（LoRA Adapter）

流程为：输入 → 基础模型 + LoRA适配器 → 输出。

但部署时，我们通常希望只用一个“完整模型”：输入 → 完整模型 → 输出。因此，需要**合并权重**。

---

**合并的数学原理**

LoRA训练时，模型权重变为：

`W' = W + ΔW`，其中 `ΔW = BA`。

合并，就是把 `BA` 直接加进 `W`，得到 `

## 块 56

### 3.1 merge_and_unload() 是干什么的？

一句话：**把LoRA训练出来的“补丁”焊死进基础模型，然后拆掉LoRA的外壳。**

你训练LoRA时，模型其实是两层的：

```text
PeftModel
├── Base Model（原始模型，权重W被冻结）
└── LoRA Adapter（你训练出来的小补丁，即低秩矩阵A和B）
```

注意：LoRA训练时**冻结原始权重W**，仅训练低秩矩阵A和B。`merge_and_unload()` 做的就是：

```text
W' = W + (α/r)BA
```

即把训练好的低秩增量（缩放后）加到原始权重上，得到完整模型。

```text
Base Model + LoRA Adapter → 完整模型
```

执行完这一步，你手里的就不再是“带补丁的模型”，而是一个**普通的、完整的Transformer模型**——它的权重已经包含了LoRA带来的更新。

---

### 3.2 合并前后对比

| 阶段 | 模型结构 |
|------|----------|
| 合并前 | PeftModel（基础模型 + LoRA Adapter） |
| 合并后 | 普通Transformer模型（权重已更新） |

---

### 四、完整流程：从训练到导出

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

你现在要做的，就是走通“加载 → 合并 → 保存”这三步。

---

### 五、加载LoRA Adapter（第1步）

假设你有：

- 基础模型路径：`base_model_path`
- LoRA适配器路径：`adapter_path`

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

try:
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, adapter_path)
except (OSError, ValueError) as e:
    print(f"加载失败: {e}")
    raise
```

此时模型结构是：`Base Model + LoRA Adapter`。

---

### 六、合并权重（第2步）

执行一行代码：

```python
try:
    merged_model = model.merge_and_unload()
except (OSError, ValueError) as e:
    print(f"合并失败: {e}")
    raise
```

完成后，`merged_model` 就不再是 `PeftModel`，而是**普通Transformer模型**。

---

### 七、导出合并后的模型（第3步）

保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后的目录大致长这样（实际文件会因模型类型略有不同）：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

---

**本块学习检查**：请你在自己的代码环境中，完成以下操作验证——

1. 加载一个基础模型和对应的LoRA Adapter；
2. 调用 `merge_and_unload()` 合并权重；
3. 打印合并后模型的类型，确认它不再是 `PeftModel`。

**部署注意事项**：
- 模型许可证：请确认基础模型的许可证类型（例如：Llama 2 Community License，商用月活>7亿需Meta授权）。
- 显存要求：加载和合并模型时需注意显存占用

## 块 57

## 三、merge_and_unload是什么？（重点）

### 3.1 方法作用

合并后，你不再需要 `peft` 库来加载模型，直接用 `transformers` 就行：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

**合并 vs 不合并，一张表看懂：**

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base+Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要PEFT | 直接Transformers |

**什么时候该合并？** 这是权衡，不是必须。考虑这几点：是否需要多个Adapter动态切换、推理框架是否原生支持LoRA、存储/分发体积要求、对精度（尤其QLoRA）的接受度。

---

### 合并后必查清单（避免"能加载但行为不对"）

1. **Tokenizer完整且一致**：确认保存的tokenizer与训练时一致，特别是新增/修改过的特殊token（如`<|im_start|>`、pad/eos）都在`special_tokens_map.json`/`tokenizer_config.json`里；若扩过词表，确认embedding/lm_head尺寸匹配。

2. **generation_config.json**：如果依赖特定生成默认值（eos/pad token id、`max_new_tokens`、采样参数），确认它一并导出。

3. **dtype**：确认合并/保存精度符合预期（FP16/BF16/FP32），不一致会影响显存与数值表现。

4. **config.json完整性**：确认模型结构配置正确，且已**移除PEFT/adapter相关配置**——合并后应是普通Transformers模型，不应残留`adapter_config.json`。

---

**动手验证**：合并后，用上面的代码加载模型，然后检查`model.config`里有没有`adapter`相关字段，再跑一次推理看输出是否正常。

**模型许可证**：部署引用模型时需注意许可证类型，

## 块 58

## 10.1 推荐合并场景

你是要学LoRA微调的，这里直接告诉你：**什么时候该把LoRA合并回基础模型？** 两个典型场景，记住就行。

**场景1：上线部署——只要一个Adapter，追求简单**

比如生产环境里，服务器直接加载模型、提供API。如果你只需要一个固定的Adapter，不想在部署时额外装PEFT、不想做运行时加载，那合并更省事。

> ⚠️ 但注意：如果你的推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你要在同一个基础模型上热切换多个Adapter，那**不合并**、以Adapter形式部署反而更灵活。合不合并，看框架能力和业务需求。

> 📋 部署前确认：模型许可证（如Llama 2 Community License，商用月活>7亿需Meta授权）及显存要求（如Llama-2-7b FP16需至少14

## 块 59

### 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个Adapter（A、B、C），每个负责不同任务。**保留Adapter更灵活**——想用哪个任务就加载哪个，随时切换，不用重新训练。

**场景2：持续训练**

如果你打算继续调整LoRA，**保留Adapter**。合并后权重固化，再想微调就得从头来，不划算。

---

### 十一、QLoRA合并注意事项（重点）

QLoRA有个关键点：基础模型用**4-bit量化**存储（省内存），但LoRA参数是**FP16/BF16**（高精度）。训练时两者分开，合并时，先将4-bit量化权重反

## 块 60

## 直接合并可能遇到的情况

你现在已经知道 QLoRA 训练完会得到一个 Adapter（适配器）。但**合并**这一步，如果直接在 4-bit 量化状态下操作，可能会遇到两类问题：

1. **精度损失**——4-bit 权重需要先反量化再合并，这个过程和全程用高精度计算并不完全等价，结果会有细微偏差。
2. **类型不匹配或版本不支持**——权重类型对不上，或者你用的 PEFT / bitsandbytes / Transformers 版本不允许在量化状态下直接调用 `merge_and_unload`。

> 注意：这些是**可能**出现的情况，不是每次都会报错。能不能顺利合并、合并质量如何，取决于你的**版本**、量化配置，以及是否先做了反量化。最终以你所用版本的文档和实际运行结果为准。

## 常见做法：先反量化到高精度再合并

一个稳妥的思路是：**别在 4-bit 状态下直接合并**。先以高精度（FP16/BF16）重新加载一份基础模型，再挂载同一个 Adapter，然后合并。

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

这样合并发生在高精度权重上，就绕开了 4-bit 状态下的类型和精度问题。

## 可执行代码

下面这段代码就是上面流程的完整实现：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 模型许可证：请确认你使用的基础模型许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存，请根据实际模型规格评估显存需求

# 1) 关键：以高精度（FP16/BF16）重新加载基础模型，而不是再走 4-bit 量化
#    合并要发生在高精度权重上，所以这里不传 quantization_config
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float16

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=compute_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
except Exception as e:
    raise RuntimeError(f"加载基础模型失败: {e}")

# 2) 挂载训练得到的同一份 Adapter
try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
except Exception as e:
    raise RuntimeError(f"加载 Adapter 失败: {e}")

# 3) 合并；safe_merge=True 会在合并时做数值检查（如检测 NaN）
try:
    merged_model = model.merge_and_unload(safe_merge=True)
except Exception as e:
    raise RuntimeError(f"合并模型失败: {e}")

# 4) 保存完整模型与 tokenizer（tokenizer 从基础模型或训练目录加载均可，保持与训练一致）
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    merged_model.save_pretrained(merged_out)
    tokenizer.save_pretrained(merged_out)
except Exception as e:
    raise RuntimeError(f"保存模型失败: {e}")
```

**代码要点拆解：**

- **第 1 步是关键**：重新加载基础模型时**不传** `quantization_config`，

## 块 61

**QLoRA合并：三个关键提醒**

你训练完QLoRA模型，准备合并权重时，先记住这三点：

**1. 版本决定“能不能直接合并”**
不同PEFT版本对“量化模型能否直接合并”支持不同，有的版本要求先反量化。**最稳妥的做法**：加载基础模型时**不传**`quantization_config`或`load_in_4bit`参数，直接加载高精度权重再合并。这样绕开了版本差异。

**2. 内存要够用**
以高精度重新加载完整基础模型，会占用约等于一份FP16/BF16完整模型的内存/显存。显存不够时，把`device_map`设为`{"": "cpu"}`，在

## 块 62

## 合并时开启校验并做一致性验证

`merge_and_unload` 支持 `safe_merge=True`，合并时会做数值检查（比如检测是否产生 NaN），相当于给合并过程加一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，建议**对比合并前后模型在同一批输入上的输出**，确认合并没有引入明显偏差。流程如下：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits / 生成文本是否基本一致

最小验证代码思路：用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本是否一致。

**本块

## 块 63

## 合并后别急着用——先做一致性验证

你已经完成了 LoRA 权重合并（§11.2），现在模型看起来能用了。但这里有个关键问题：**合并后的模型和未合并的模型，输出真的一样吗？** 尤其是 QLoRA 场景下，因为涉及量化精度损失，合并前后往往会有细微差异。本块教你如何用一段代码，快速验证合并是否成功。

### 为什么需要单独加载一份“未合并”的模型？

先看代码里的注释，它强调了一个容易踩的坑：

> `merge_and_unload()` 是“原地”操作，会就地修改并返回底层模型，合并后原来的 `PeftModel` 已经不再是“未合并”状态。

也就是说，如果你在 §11.2 里已经执行过合并，那个对象就“脏”了。**不能拿它当“合并前”的基准**。所以代码里专门重新加载了一份：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(量化)+Adapter
```

这里的 `model_before` 才是真正的“合并前”状态：原始基础模型 + LoRA 适配器，权重还没融合。注意，QLoRA 场景下基础模型本身是量化后的（4bit/8bit），合并前后精度一致，差异源于量化误差，而非合并操作本身。

### 验证分两步：先比 logits，再比生成文本

**第一步：比较 logits（模型输出的原始分数）**

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

这段代码计算两个模型对同一输入产生的 logits 的最大绝对差。**数值越接近 0，说明合并前后模型行为越一致。** 注意注释里的提示：QLoRA 场景下通常不会是精确的 0，而是一个较小的近似值——这是量化带来的正常误差，不用慌。

**第二步：比较贪心生成的文本**

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

这里用 `do_sample=False` 去掉随机性，保证两次生成都是贪心解码，结果可复现、可对比。如果两个模型行为一致，生成的文本应该完全相同。

### 实操

## 块 64

## 合并时开启校验并做一致性验证（建议）

**为什么必须另加载一份模型？**

`merge_and_unload()` 返回一个合并后的新模型，原模型保持不变。如果你拿合并后的 `model` 去和"合并前"对比，两边其实是同一份已合并的权重，`max_abs_diff` 永远等于 0，验证就白做了。

正确做法：用 `base_for_ref` + `model_before` 重新加载一份**未合并**的 Adapter 作为对照，再和合并后的模型做对比。

**显存不够怎么办？**

分两次跑：
1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本**存到磁盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比两次结果。

**结果预期（重点）**

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 差异。你的目标是确认**没有明显退化或 NaN**，而不是要求逐位相同。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。
- 是否可接受，以你的**任务指标与验证集**为准。

---

## 电脑配置批注

本章需要**实际加载模型**，涉及：加载完整模型 → 合并权重 → 导出模型。

**最低

## 块 65

## 十一、合并时开启校验并做一致性验证（建议）

### 为什么需要校验？

QLoRA 合并最容易踩的坑是：**在 4-bit 量化状态下直接调用 `merge_and_unload`**。这可能导致合并失败或结果异常。原因可能不止一种：PEFT/bitsandbytes 版本不支持、量化包装未正确反量化、内存/显存不足等。

### 最稳妥的做法：先"反量化"再合并

为避免在 4-bit 量化状态下直接合并，可先对量化模型执行反量化（如通过 `merge_and_unload` 或 `dequantize` 方法），或重新加载一份不带量化配置的基础模型并挂载 Adapter 后再合并。具体操作：

```python
# 不传 quantization_config / load_in_4bit
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16  # 或 float16
)
# 再挂载同一份 Adapter
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
# 然后合并
merged_model = peft_model.merge_and_unload(safe_merge=True)
```

这样合并发生在高精度权重上，天然避开了"在 4-bit 状态下合并"的类型/精度问题。

> **版本提示**：不同 PEFT / bitsandbytes / Transformers 版本对"量化模型能否直接调用 `merge_and_unload`"支持不同——部分版本直接报错或警告，部分较新版本有一定支持但仍有精度损失。**"先反量化再合并"是跨版本最稳妥的做法**；是否可在量化状态下直接合并，以你所装版本的文档和实际运行结果为准。

> **模型许可证**：部署引用模型时需标注许可证类型，例如：Llama 2 Community License（商用月活>7亿需Meta授权），具体以模型发布页为准。

### 合并后必查清单

| 检查项 | 常见问题 | 排查方向 |
|--------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已合并成普通模型） | 确认加载方式，先得到 PeftModel 再调用 |
| 合并后模型变大 | **正常现象**：合并后是完整模型，不再是 Adapter | 无需处理；在意体积可保留 Adapter 不合并 |
| 显存不足 | 高精度完整模型占用高；dtype 用了 FP32、未做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载

## 块 66

## 路线A：合并后部署完整模型（动手做一遍）

你已经学会了用 `merge_and_unload()` 把 LoRA 权重合并回基础模型。现在我们要回答一个实际问题：**合并完之后，模型怎么上线服务？**

这条路线最直接，就三步：

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

**为什么这条路省心？** 因为合并后的模型在推理框架眼里就是一个“普通模型”——框架不需要知道 LoRA 是什么，你也不用额外配置任何 LoRA 相关的参数。加载方式和你加载一个没微调过的模型完全一样。

**代价是什么？** 每合并一个 Adapter，你就要保存和分发一份完整的模型文件。如果你有多个任务（比如 A/B/C 三个 Adapter），就得存三份完整模型，存储成本会明显上升。

> **模型许可证：** 部署前请确认基础模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需 Meta 授权），并遵守相应条款。

---

## 路线B：保留 Adapter，用支持 LoRA 的框架动态服务

如果你不想存三份完整模型，可以走另一条路：**只保留一份基础模型，把多个小体积的未合并 LoRA Adapter 挂在旁边，运行时按需切换**。

```text
基础模型（只加载一份）

+

运行时挂载多个未合并的 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

**这条路的好处：** 多任务时只需一份基础模型 + 几个小 Adapter，显存和存储都省；支持热切换，不用为每个任务单独存完整模型。

**代价：** 依赖推理框架对 LoRA 的支持，配置比“直接加载合并模型”复杂一些；不同框架/版本对 LoRA 的支持范围和限制不一样。

---

## vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

以 vLLM 为例，启动 OpenAI 兼容服务时，通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，请求里用 Adapter 名字指定用哪个。命令行大致长这样：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
    --max-loras 2 \
    --max-lora-rank 16
```

参数含义：

- `--enable-lora`：开启多 LoRA 服务
- `--lora-modules 名字=路径`：声明可用的 Adapter。**注意：这里的路径是训练保存下来的 Adapter 目录**（含 `adapter_model.safetensors

## 块 67

## 14.3 如何选择：合并还是保留？

你正在做 LoRA 微调，最终要把模型用起来。这里有个关键认知：**合并不是唯一部署方式**。具体怎么选，看你的实际场景：

**情况一：只有一个 Adapter，追求最简部署**
→ 选**路线A（合并）**。把 LoRA 权重合并进基础模型，得到一个完整模型，直接部署，省心。

**情况二：同一基础模型要服务多个 Adapter，且推理框架原生支持 LoRA**
→ 选**路线B（保留 Adapter 动态服务）**。不合并，保留多个 Adapter，按需动态切换，灵活高效。

**情况三：QLoRA 精度敏感**
→ 无论走哪条路线，**都必须做合并前后/上线前的一致性验证**（见 §11.3），确保精度无损。

> 注意：具体框架用法和限制，留到第 15 板块（vLLM 部署）再展开。现在你只需要记住：**部署方式不止合并一条路**，选择取决于你的场景需求。

---

**本块学习

## 测评题目

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？
