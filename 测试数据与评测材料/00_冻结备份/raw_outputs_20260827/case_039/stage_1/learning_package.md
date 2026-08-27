# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么做的：

```text
输入数据 → 整个模型参数更新 → 保存完整模型
```

以 7B 参数模型为例，就是 **70亿个参数**。训练时，你需要同时保存三类东西：

- **参数**本身
- **梯度**（用于反向传播）
- **优化器状态**（如 Adam 的动量项）

这三者叠加，显存需求非常高。**普通消费级 GPU 根本跑不动**。

---

**本块学习检查**（动手验证）：

打开终端，输入以下命令查看你的 GPU 显存大小：

```bash
nvidia-smi
```

观察 "Memory-Usage" 一栏，记下你的显存总量（如 8GB、16GB 等）。然后思考：如果训练 7B 模型需要保存参数+梯度+优化器状态，你的显存够用吗？大概率不够——这正是我们接下来要学习 LoRA 的原因。

## 块 2

**本块学习检查**：请用一句话向自己解释：为什么LoRA只训练少量参数，却能完成微调任务？

---

**1.2 参数高效微调（PEFT）**

PEFT，全称Parameter-Efficient Fine-Tuning，中文即“参数高效微调”。

它的核心思想是：**不修改整个大模型，只训练少量新增参数**。

想象一下，原始大模型就像一个巨大的基座，我们不动它，只在上面加一个“LoRA适配器”。训练时，只更新这个适配器的参数，完成后保存这个规模很小的Adapter即可。

对比一下三种方式的资源消耗：

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 |

## 块 3

**LoRA是什么？——用最少的参数做微调**

你现在的目标是LoRA微调，而且喜欢动手实践，那我们直接切入核心：LoRA（Low-Rank Adaptation，低秩适配）是一种“只训练一小部分参数”的技巧。

**原模型怎么处理？**
- 原始权重记为 `W`，微调时**不更新**它。
- 新增一个变化量 `ΔW`，最终模型变成：`W + ΔW`。

**关键：ΔW怎么来？**
- 公式：`ΔW = (α/r) × B × A`，其中 `α` 是缩放因子，`r` 是秩。
- 这里 `A` 和 `B` 是两个**低秩小矩阵**

## 块 4

## 3.1 LoRA的问题：省了训练，没省加载

你已经知道LoRA能大幅减少**训练**时的参数，但这里有个关键点：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个7B参数的模型，如果用FP16精度加载，大约需要 **14GB显存**。这意味着，即使LoRA只训练一小部分参数，你的GPU也得先装得下这个“大块头”才能开始干活。

---

## 3.2 QLoRA的解法：把模型“压缩”了再训练

QLoRA（Quantized LoRA）就是量化版的LoRA。它的核心思路是：

```text
FP16模型 → 4bit量化 → 减少显存占用 → 再加载LoRA进行训练
```

训练时的结构变成了：

```text
4bit基础模型 + LoRA参数 → 一起训练
```

简单说：**先把大模型压缩到4bit精度，省下显存，再挂上LoRA参数去微调**。

---

## 3.3 QLoRA依赖bitsandbytes

要跑QLoRA，你需要这几个组件：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中，`bitsandbytes` 负责三件事：

- 低精度量化
- 4bit加载
- 8bit加载

> **重要说明**：本教程统一以NVIDIA CUDA GPU为准，这是项目支持范围，也是当前资料最全的路径。但“QLoRA只能用NVIDIA”并不是技术必然——bitsandbytes近年已在扩展对CPU、AMD ROCm、Intel等后端的支持，具体以你所用版本的官方说明为准。**各后端支持程度差异很大**，NVIDIA CUDA是完整支持且验证最充分的路径；其他后端可能只是实验性/部分支持。请务必核对官方硬件支持矩阵：
> - bitsandbytes官方文档：https://huggingface.co/docs/bitsandbytes/main/en/installation
> - bitsandbytes仓库：https://github.com/bitsandbytes-foundation/bitsandbytes
> - PyTorch安装选择器：https://

## 块 5

好的，我们直接进入正题。你的目标是跑通 LoRA 微调，并且喜欢动手实践。这一块内容就是帮你搞清楚，你的电脑到底能不能跑得动，以及为什么。

先看 LoRA 的最低配置，你可以把它当成一个“及格线”：

*   **CPU**：4核以上
*   **内存**：16GB 以上
*   **磁盘**：50GB 以上可用空间
*   **GPU**：推荐使用 NVIDIA 显卡
*   **显存**：最低 8GB，推荐 12GB 以上

如果你的电脑满足这个“及格线”，那么恭喜你，跑通 LoRA 微调是很有希望的。

接下来是 QLoRA。它的核心优势就是**用更小的显存来微调大模型**。但要注意，它并不是没有要求的。

QLoRA 的显存需求，这里给一个 7B 级别模型的**经验参考值**：

*   **参考下限**：约 8~12GB（短序列、小 batch）
*   **推荐**：16GB 以上

**特别提醒**：这个“参考下限”不是硬性规定，它成立是有前提的。实际显存占用受很多因素影响，比如：

*   **模型规模**：7B 级别；模型越大（比如 13B、70B），显存需求会显著上升。
*   **序列长度**：序列越长，显存占用越高。
*   **batch 大小**：batch 越大，占用越高。
*   **是否开启梯度检查点**：开启后可以“用时间换显存”，降低峰值占用。
*   **量化配置**：比如

## 块 6

## 5.1 Python版本检查

先确认你的Python版本。在终端执行：

```bash
python --version
```

建议使用 **Python 3.10 或更高版本**。如果版本偏低，请先升级Python再继续后续步骤。

---

## 5.2 PyTorch检查（关键！）

> ⚠️ **为什么这一步重要？** LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的CUDA量化路径要求安装**与本机CUDA匹配的GPU版PyTorch**。直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

**安装GPU版PyTorch**：请以 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 为准。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证PyTorch能否看到GPU**，运行：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # 查看torch编译使用的CUDA版本
```

正常时`torch.cuda.is_available()`应返回`True`，表示PyTorch侧能用GPU。

> ⚠️ **注意**：返回`True`只说明PyTorch能用GPU，**并不足以证明bitsandbytes的4-bit量化路径可用**（后者还依赖bitsandbytes是否正确编译/匹配到本机CUDA）。4-bit能否真跑，以第八章、第十三章对`bitsandbytes`的检查为准。

---

## 5.3 查看GPU信息

终端执行：

```bash
nvidia-smi
```

输出应包含：**GPU型号**、**CUDA版本**、**显存大小**。例如：

```text
RTX 4090
CUDA 12.x
Memory 24GB
```

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是 **Hugging Face官方参数高效微调库**，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter。

### 6.2 安装PEFT

```bash
pip install peft
```

验证：

```python
import peft
print(peft.__version__)
```

能打印版本号即安装成功。

---

## 5.4 模型设备指定

在后续代码中，请使用以下方式将模型指定到设备

## 块 7

**bitsandbytes 是干嘛的？**

简单说，它是你这次 LoRA 微调路上的“翻译官”——负责把大模型“压缩”成 8bit 或 4bit 的格式，让显存不够大的 GPU 也能跑得动。QLoRA 训练，就是靠它实现的。

**所以，本教程的 QLoRA 方案里，它是前置依赖，必须安装。**

> 注意：这里“必须”指的是本教程选用的 4-bit 量化方案。QLoRA 本身不绑定某个库，但咱们统一用 `bitsandbytes` + NVIDIA CUDA 来演示。

**动手装一下：**

```bash
pip install bitsandbytes
```

装完先别急着跑，下一步咱们还要检查它和你的 CUDA 环境是否兼容——这是重点，跳过容易踩坑。

## 块 8

**本块学习检查**：运行 `nvidia-smi`，记下输出的 `CUDA Version`，再运行 `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`，对比两者显示的CUDA版本是否一致。

---

**为什么先查CUDA？**

bitsandbytes是QLoRA量化的底层依赖，它通过CUDA调用GPU计算。如果CUDA版本不匹配，后果很直接：CUDA不可用、量化失败、加载模型报错——任何一个都会让你的微调卡在起点。

**两步检查法（跟着做）**

1. **看GPU的CUDA版本**：终端执行 `nvidia-smi`，在输出右上角找到 `CUDA Version` 字段，

## 块 9

## 9.1 推荐使用WSL2

你的目标是跑通LoRA微调，那环境这关别在Windows上硬磕。推荐路径：

```text
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

核心原因就一句：**Linux环境兼容性更好**。

## 9.2 Windows原生可能问题

如果你坚持在Windows原生环境装，最可能撞上的坑是：

**bitsandbytes无法加载CUDA动态库。**

这不是单一原因，常见的有：

- 你装的`bitsandbytes`版本和本机CUDA / GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本的支持不如Linux完善（不同版本差异较大）。

所以别把锅全甩给"Windows不支持"——真正的问题是 **PyTorch、bitsandbytes版本与本机

## 块 10

## 九、安装完整LoRA环境

先别急着装库，先检查你的GPU环境是否“对得上号”。在终端运行：

```bash
nvidia-smi
```

确认三件事：
- **NVIDIA驱动**已安装且正常；
- **CUDA版本**（驱动支持的版本）；
- **PyTorch版本**（必须与CUDA匹配）。

**常见坑**：驱动新、CUDA旧，或PyTorch装成了CPU版——后面QLoRA的4-bit量化会直接报错。

---

## 十、安装完整LoRA环境

如果你已有基础环境（`transformers`、`datasets`、`accelerate`），直接补装：

```bash
pip install peft bitsandbytes
```

**关键前置**：确保已安装**匹配本机CUDA的GPU版PyTorch**。CUDA 12.1示例：

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**注意**：运行QLoRA微调需要至少8GB显存，请确保GPU满足要求。

一步到位完整安装：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

运行：

```bash
accelerate config
```

按提示选择。单GPU示例：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

生成配置文件 `default_config.yaml`。

---

## 十二、LoRA环境验证实验

创建 `test_lora_env.py`：

```python
import torch
from peft import LoraConfig

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

**成功标准**：输出 `CUDA: True` 且打印出LoRA配置。说明PyTorch、PEFT、LoRA配置全部正常。

---

## 十三、QLoRA环境验证实验

测试bitsandbytes：

```python
import torch
import bitsandbytes

print(torch.cuda.is_available())
print(bitsandbytes.__version__)
```

输出 `True` 即GPU环境正常，可以继续QLoRA微调。

---

**本块学习检查**：运行上述两个验证脚本，截图输出结果。若 `CUDA: False`，回头检查PyTorch版本是否为GPU版；若bitsandbytes报错，检查CUDA版本匹配。

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：Windows报错怎么办？**

在Windows上运行LoRA/QLoRA环境时，如果遇到兼容性问题导致报错，**推荐使用WSL2**（Windows Subsystem for Linux 2）来解决。

**为什么推荐WSL2？**

Windows原生环境对bitsandbytes等库的支持不够完善，容易出现兼容性报错。WSL2提供了一个更接近Linux的环境，能有效规避这类问题。

**操作验证任务**：如果你在Windows上遇到报错，先确认是否已安装WSL2；若未安装，请将其作为首选解决方案。

---

**补充排查速查表**（供遇到问题时对照）：

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

**本块学习检查**：请回答——在Windows上遇到兼容性报错时

## 块 12

## 一、为什么需要LoRA？——先别急着全量更新

你现在要学LoRA微调，那我们先搞清楚一个关键问题：**为什么大模型微调不需要更新全部参数？**

### 1.1 大模型微调的问题

现代大语言模型（比如LLaMA、Qwen、Baichuan、Mistral）通常拥有**数十亿到上百亿参数**。拿7B模型来说，就是70亿个参数。

如果采用传统方式微调——把预训练模型的所有参数都更新一遍，然后保存新模型——你需要面对三个现实问题：

- **巨大的GPU显存**：训练时不仅要存模型参数本身，还要存梯度和优化器状态；
- **大量训练时间**：参数越多，计算量越大；
- **巨大的存储空间**：每个微调任务都要保存一份完整的新模型。

### 1.2 全参数微调到底有多“贵”？

假设模型有100亿参数。训练时你需要同时保存：

- **模型参数**：记为 W
- **梯度**：记为 ∂W（用于反向传播更新参数）
- **优化器状态**：比如Adam的动量项

这三样加起来，**实际显存需求远大于模型本身的大小**。这就是为什么全参数微调大模型对硬件要求极高。

---

## 二、LoRA的核心思路：只学“变化量”

LoRA（Low-Rank Adaptation）的核心思想很直接：**既然全量更新太贵，那我冻结原始权重W，只学习参数的变化量，而且用低秩矩阵来表示这个变化量。**

关键概念

## 块 13

### 2.1 不直接修改原模型参数

**传统微调**：直接改原权重 `W`，训练后变成 `W'`。

**LoRA 做法**：把 `W` 冻结不动，额外加一个新参数 `ΔW`，最终模型用：

```text
W' = W + ΔW
```

- `W`：原模型参数（冻结）
- `ΔW`：LoRA 新增参数（只训练这个）

---

### 2.2 为什么只训练ΔW？

大模型已通过海量数据学会语言规律、世界知识、推理能力。微调只是**方向调整**，不必重新学全部参数，只学**任务相关变化量**即可。

> 注意：这是教学直觉，非严格结论。任务与预训练差距大或数据充足时，全参数微调可能更好。

---

**本块学习检查**：请用一句话说明 LoRA 中 `W` 和 `ΔW` 各自的作用，并指出哪个参数会被训练。

## 块 14

**本块学习检查**：请用一句话说明：为什么 `ΔW` 不直接学成 `4096 × 4096` 的完整矩阵，而要拆成两个小矩阵？

---

### 3.1 矩阵基础概念

Transformer 里大量参数都是**矩阵**。比如注意力层中：

```text
Q = X · Wq
```

这里 `Wq` 就是一个权重矩阵。假设它的形状是 `4096 × 4096`，那么它的参数数量就是：

```text
4096 × 4096 ≈ 1677万
```

也就是说，光这一个矩阵，就有约 1677 万个参数。

---

### 3.2 直接训练矩阵的问题

如果我们要微调，就需要修改一个 `ΔW`（权重的变化量）。如果直接学完整的 `ΔW`，它仍然需要 `4096 × 4096` 个参数——和原矩阵一样大，成本太高。

---

### 四、LoRA低秩分解数学原理（重点）

LoRA 的核心思路是：**不要直接学习完整矩阵**，而是把它**拆成两个小矩阵**。

原始权重 `W`

## 块 15

**本块学习检查**：请亲手算一遍——原矩阵是 4096×4096，直接训练要更新多少参数？如果改用 LoRA，设 rank r=8，拆成 A（8×4096）和 B（4096×8），A 和 B 各有多少参数？加起来是多少？对比完整矩阵的 1677 万，LoRA 的 6.5 万参数少了多少倍？

---

**4.1 矩阵尺寸变化**

先看原矩阵：`W` 是 **4096 × 4096**。直接训练它，要更新的参数就是：

`4096 × 4096 = 16,777,216`（约 1677 万）。

现在用 LoRA。

## 块 16

**5.1 神经网络参数变化具有低维特征**

LoRA的核心洞察来自一个假设：模型适应新任务时，参数变化（ΔW）并不需要覆盖所有方向。有效变化往往集中在一个**低维子空间**里。

想象一个4096维的空间，真正影响任务的“有效方向”可能远少于满维度，只集中在少数方向上。这就是“权重更新具有较低的内在维度”的含义。

**注意**：这里的“低维”是定性理解，没有通用固定值（比如“几十维”），具体维度随模型、任务、数据变化。

因此，我们可以用低秩矩阵 **BA** 来近似这个 ΔW——用更少的参数捕捉最关键的变化方向。

---

**本块学习检查**：用自己的话解释“为什么可以用低秩矩阵BA近似ΔW

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。

但仔细想想——如果这张图片的主要变化只是**亮度**、**对比度**，外加**少量结构变化**，你真的需要存1600万个数字吗？

不需要。用**少量参数**就能描述这些变化。

**LoRA的核心思想正是如此**：用少量参数（低秩矩阵A和B）来近似模型参数的更新增量，而不是直接修改和存储全部参数。

---

## 六、LoRA在Transformer中的位置

Transformer主要结构：

```
Input → Embedding → Attention → FFN → Output
```

其中Attention包含四个投影矩阵：**Q、K、V、O**。

LoRA早期实践中**常见**加入Q投影和V投影，例如：

```python
target_modules=["q_proj", "v_proj"]
```

表示只训练 `Wq` 和 `Wv` 的变化。

> ⚠️ **注意**：只挑 `q_proj`/`v_proj` 是LoRA原论文的**常见选择**，**不是固定标准**。现代实践常把LoRA加到全部投影（q/k/v/o）、FFN线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。

**实操示例**（以Llama类模型常见模块名为例，不同模型请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

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

> 覆盖模块越多，可训练参数、显存与Adapter体积越大——是否值得，需结合验证集表现权衡。

---

## 七、LoRA训练过程

完整流程：

```
加载预训练模型 → 冻结所有参数 → 插入LoRA矩阵 → 训练LoRA参数 → 保存Adapter
```

训练时：原模型 `W` **冻结**，LoRA的 `A`、`B` 矩阵**更新**。

最终模型 = 原模型权重 + LoRA低秩增量（ΔW=(α/r)BA）。

---

## 八、LoRA核心参数理解

你已经看到了 `r

## 块 18

## 8.1 rank（r）——低秩矩阵的“宽度”

`r` 是 LoRA 中最核心的超参数，它决定了低秩矩阵的维度。代码里通常这样设置：

```python
r=8
```

**一句话理解**：`r` 越大，模型能“记住”的细节越多，但代价是参数和显存占用也越大。

### 怎么选起始值？

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

注意：这只是**搜索起点**，不是固定规则。合适的 `r` 取决于模型规模、目标模块、数据量和任务难度。

### 比背“配方”更可靠的做法：用验证集选 `r`

1. **划出验证集**：从数据中留出 20%（或 10%）不参与训练，后续所有比较都在验证集上看指标。
2. **固定其它变量**：只改 `r`，保持 `target_modules`、学习率、`lora_alpha`（或 `alpha/r` 比例）、训练步数不变。
3. **小范围网格搜索**：从 `r ∈ {4, 8, 16, 32}` 起步，逐个训练，记录验证集上的 loss/accuracy/F1 以及显存和耗时。
4. **看边际收益**：如果 `r` 增大后验证集提升很小，甚至训练指标升但验证指标降（过拟合），说明再加大不划算，取

## 块 19

## 8.2 lora_alpha：控制LoRA的“更新力度”

你现在已经知道LoRA通过低秩矩阵 `ΔW = BA` 来微调模型。那 `lora_alpha` 是干什么的？简单说：**它控制LoRA参数对模型的更新强度**。

```python
lora_alpha=16
```

这个值越大，LoRA对原模型权重的“改动”越明显；越小则改动越温和。实际使用时，通常配合秩 `r` 一起看，用 **`alpha/r`** 作为最终的缩放比例。比如 `r=8`、`alpha=16` 时，缩放比例就是 `16/8=2`，相当于把低秩矩阵的贡献放大2倍。

**动手验证（不需要GPU）**：你可以把 `alpha` 分别设为 `4` 和 `32`，观察同一任务下模型输出的差异——`alpha` 越大，输出变化越明显。

---

## 8.3 lora_dropout：防止LoRA过拟合

```python
lora_dropout=0.05
```

`dropout` 是深度学习中常用的正则化手段。在LoRA里，它随机“丢弃”一部分LoRA参数（此处为5%），防止模型在训练数据上过度拟合，提升泛化能力。**训练时开启，推理时自动关闭**，无需你手动处理。

---

## 九、LoRA vs 全参数微调：一张表看懂

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|----------------|
| 训练参数

## 块 20

## 2.1 Rank 表示什么？

Rank，中文叫“矩阵的秩”。在 LoRA 里，它指的就是**低秩矩阵中间那个维度的大小**。

咱们直接看例子，比干讲概念好懂：

假设原始权重矩阵是 `4096 × 4096`（这规模在真实大模型里很常见）。

你设置 `r=8`，那 LoRA 就会拆出两个小矩阵：

- 矩阵 A：`8 × 4096`
- 矩阵 B：`4096 × 8`

中间那个 `8`，就是 rank 的值。

然后 `ΔW = B × A`，乘完结果正好是 `4096 × 4096`，跟原矩阵形状一致，可以直接加回去。

**一句话记住**：rank 决定了 LoRA 这个“补丁”的宽度。r 越大，补丁能装的信息越多，但训练和存储成本也越高。

**动手验证一下**：拿张纸，把 `r=16` 的情况写出来——A 和 B 分别是什么形状？乘完能恢复成 `4096 × 4096` 吗？写一遍就记住了。

## 块 21

**本块学习检查**：读完下面内容后，请用一句话回答：r=4和r=64，哪个能学更多变化方向？

Rank决定LoRA能学多少种变化方向。

**低rank（如r=4）**：只能学较少变化模式。优点是参数少、显存低、训练快；缺点是表达能力有限。

**高rank（如r=64）**：拥有更多自由度。优点是能学更复杂任务、适应能力更强；缺点是参数更多、显存增加、训练成本提高。

**动手验证**：假设某层是4096×4096，分别用r=4和r=64，感受一下参数量差异——r越大，要更新的参数就越多。

## 块 22

**3.1 不用LoRA，要训多少？**

一个4096×4096的矩阵，参数量是：

`4096 × 4096 = 16,777,216`，约**1677万**。

**3.2 用LoRA，参数骤减**

LoRA的参数量公式：`r × (输入维度 + 输出维度)`。

- **r=8**：`8 × (4096+4096) = 65,536`，约**6.5万**。
- **r=64**：`64 × 8192 = 524,288`，约**52万**。

**对比一下：**

| 方式 | 参数量 |
|------|--------|
|

## 块 23

## 4.1 Rank太小的问题

你可能会想：把 `r` 设小一点，比如 `r=2`，是不是更省资源？**但太小会“学不动”。**

**具体表现：**
- **学习能力不足**：模型能调整的参数太少，新任务的知识“装不下”；
- **欠拟合**：训练集上都没学好，更别说泛化；
- **新任务知识无法充分表达**：想学的东西被“小水管”限制住了。

**你在训练时能看到的现象：**
- 训练loss下降慢（学得慢）；
- 效果提升有限（学不好）。

> **动手验证**：如果你在跑LoRA微调，试着把 `r` 从常见值（如8或16）改小到 `r=2`，观察训练loss的下降速度。你会发现loss下降明显变慢，最终效果也上不去——这就是“欠拟合”的直观感受。

---

## 4.2 Rank太大的问题

那反过来，把 `r` 设大一点，比如 `r=256`，是不是更好？**也不是，代价很高。**

**可能带来的问题：**
- **参数增加**：要训练的量变大；
- **显存增加**：GPU显存占用更高；
- **训练速度下降**：每一步都更慢。

**核心矛盾**：LoRA的优势就是“参数高效”，`r` 太大

## 块 24

## 四、常见rank选择（动手版）

你是要做LoRA微调，先记住：**下表只是经验起点，不是标准答案**。

| 任务类型 | rank起点 |
|---------|---------|
| 简单分类 | 约4~8 |
| 文本风格调整 | 约8~16 |
| 指令微调 | 约16~64 |
| 复杂领域任务 | 约64以上 |

**怎么用这张表？** 从对应范围的低值开始试，然后看效果调。

**实际调参要看这7个因素**：数据量、模型大小、序列长度、batch大小（含梯度累积后的有效batch）、目标模块选择与数量、GPU显存、验证集表现。

⚠️ **别直接照搬**——同一个`r`，换模型规模、序列长度、batch或目标模块，显存和效果都可能差别很大。

---

## 五、Alpha是什么？

LoRA

## 块 25

**本块学习检查**：打开你的LoRA训练脚本，找到 `r` 和 `alpha` 的赋值处，把 `alpha` 改成 `r` 的一半（例如 `r=8` 时 `alpha=4`），观察训练日志中loss曲线的变化趋势，并记录你的观察。

---

### 6.1 控制LoRA影响强度

`alpha` 和 `rank` 的比值 `alpha/r` 是控制LoRA更新强度的**关键缩放系数**。我们通过一个具体例子来理解：

**例1：放大效果**
```python
r = 8
alpha = 16
```
缩放系数 = `16/8 = 2`，表示LoRA对模型的更新量被**放大2倍**。

**例2：保持原样**
```python
r = 8
alpha = 8
```
缩放系数 = `8/8 = 1`，LoRA更新保持**原始比例**。

**例3：减弱效果**
```python
r = 8
alpha = 4
```
缩放系数 = `4/8 = 0.5`，LoRA影响被**减弱**。

---

### 七、Alpha与Rank的关系（重点）

LoRA加入模型的更新量会被 `alpha/r` 缩放。但请注意：**`alpha/r` 只是缩放系数，不是唯一决定实际强度的开关**。实际落到权重上的更新量还取决于：

- **学习率**：直接决定每步对 `A`、`B` 的更新步长。学习率与 `alpha/r` 会**叠加**影响有效更新幅度——例如把 `alpha

## 块 26

**配置1与配置2的对比**

先看两组代码：

- 配置1：`r=8, alpha=16`
- 配置2：`r=16, alpha=32`

两组虽然rank不同，但**缩放比例**都是 `alpha / r = 2`，即比例相同。

**关键结论**

调整rank时，常见做法是**同步调整alpha**，以保持 `α/r` 比例大致稳定。注意：这只是**经验做法**，不是必须遵守的规则。`alpha = 2 × r` 只是常见的**经验起点**，并非通用最优规则。

**动手验证**

在代码中分别设置 `r=8, alpha=16` 和 `r=16, alpha=32`，观察

## 块 27

## 方案1：小模型实验（动手试试）

既然你是**hands-on**风格，咱们直接上手。先用最小配置跑通流程：

```python
r=8
alpha=16
```

**为什么先选这个？** 因为参数少、训练快，最适合测试。你的目标是LoRA微调，先用小模型验证整个流程能不能跑通，再考虑效果优化。

**核心理解：**
- `r=8` 控制**LoRA容量**——即注入的参数量大小
- `alpha=16` 控制**LoRA增量矩阵的缩放系数**——alpha与r共同决定最终增量的大小

**动手验证：** 用下面代码跑一次，观察配置结构：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora

## 块 28

## 误区1：rank越大一定越好？

**错。**

很多新手觉得rank设大点，模型能力就强。但任务简单时，大rank只会：

- 浪费显存；
- 增加训练成本。

**动手验证**：下次训练时，分别用 `r=8` 和 `r=64` 跑同一个简单任务，对比显存占用和训练速度，你会发现差距很明显。

---

## 误区2：只调rank，不调alpha

很多人改rank时忘了同步改alpha，比如：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

此时缩放比例变成：

```text
16/64=0.25
```

这意味着当r增大而alpha不变时，`alpha/r` 减小，LoRA的更新幅度变小——你虽然加了容量，但更新强度反而减弱了，效果可能还不如原来。

**记住**：改rank时，alpha要跟着调。当r增大时，应同步增大alpha，以保持 `alpha/r` 在合理范围（如2或4）。

---

## 误区3：alpha越大越好

**错。**

alpha过大（即 `α/r` 缩放过大）会导致：

- 更新过强；
- 训练不稳定。

**但注意**：`α/r` 过大只是训练不稳定的**可能原因之一**。训练震荡或发散还常常来自：

- 学习率过高；
- 数据质量差；
- 有效batch size过小；
- 精度（fp16/bf16）与数值稳定性问题；
- 缺少梯度裁剪。

排查不稳定时，要综合这些因素，别只盯着alpha

## 块 29

## 一、LoraConfig是什么？

### 1.1 PEFT中的LoraConfig作用

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。它的核心思想是：不训练全部模型参数，只训练一小部分新增参数，就能完成微调任务。

LoRA 的训练流程是这样的：

```text
加载基础模型
    ↓
创建LoraConfig
    ↓
指定LoRA插入位置
    ↓
生成LoRA Adapter
    ↓
训练LoRA参数
```

在这个流程中，**LoraConfig** 就是你的“配置总开关”。它负责告诉 PEFT 四件事：

| 问题 | 对应参数 | 通俗理解 |
|------|----------|----------|
| LoRA放在哪里？ | `target_modules` | 给模型的哪些层“加装”LoRA |
| LoRA规模多大？ | `r` | 低秩分解的维度，决定新增参数数量 |
| LoRA训练强度如何？ | `lora_alpha` | 控制LoRA对原模型权重的更新力度 |
| 是否使用dropout？ | `lora_dropout` | 训练时随机丢弃部分神经元，防止过拟合 |

**一句话总结**：`LoraConfig` 就是一份“施工图纸”，PEFT 按照它来决定在哪里改造模型、改造多大、改得多猛、以及如何防过拟合。

---

**本块学习检查**（动手验证）：

打开你的 Python 环境，执行以下代码，观察输出：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
print(config)
```

请回答

## 块 30

**本块学习检查**：读完下面这个最小配置示例后，请你在自己的代码环境里**原样运行一遍**，并观察输出——确认 `LoraConfig` 能成功创建，不报错。这是你掌握 LoRA 配置的第一步。

---

### 1.2 一个简单示例

先看一个最精简的 LoRA 配置长什么样：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

这段代码创建了一个 `LoraConfig` 对象，里面只有 4 个参数，但它们就是 LoRA 的核心。对照下表理解每个参数在干什么：

| 参数 | 作用 |
|---|---|
| `target_modules` | 决定把 LoRA 插入模型的哪些层（这里选了 `q_proj` 和 `v_proj`） |
| `r` | 决定 LoRA 矩阵的大小（值越小，新增参数越少） |
| `lora_alpha` | 控制缩放强度（影响 LoRA 对原模型输出的影响程度） |
| `lora_dropout` | 控制随机失活比例（防止过拟合） |

---

### 二、LoRA 配置整体流程

光有 `config` 还不够，它要配合模型一起使用。完整流程是：

```

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入低秩矩阵”**。

LoRA 不会默认修改模型的所有层。Transformer 的 Attention 结构通常包含四个投影层：

```text
Attention
├── q_proj   （Query 投影）
├── k_proj   （Key 投影）
├── v_proj   （Value 投影）
└── o_proj   （Output 投影）
```

你需要通过 `target_modules` 明确指定给哪些层加 LoRA。例如：

```python
target_modules=["q_proj", "v_proj"]
```

这表示：只给 **Query 投影层** 和 **Value 投影层** 增加 LoRA。

---

## 四、为什么通常选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

- **Q（Query）** 决定模型“**关注什么信息**”
- **V（Value）** 决定模型“**传递什么信息**”

所以调整 Q 和 V，通常就能有效改变模型的行为——这是 `q_proj`/`v_proj` 成为常见起点的原因。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 `target_modules`。

---

## 进阶：`target_modules="all-linear"` 自动匹配

较新版本的 PEFT（一般 ≥ 0.7）支持直接写 `"all-linear"`，让 PEFT 自动匹配模型中（几乎）所有线性层，省去手工枚举。可运行示例：

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
    target_modules="all-linear",   # 自动匹配（几乎）所有 nn.Linear；一般不含 lm_head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**注意权衡**：`"all-linear"` 覆盖的模块变多，**可训练参数和显存占用会增加**。是否用全线性层，要在**参数量/显存**与**效果**之间权衡，并以验证集结果为准。如果你的 PEFT 版本较旧不支持该字符串，请退回显式列出模块名的写法，或升级 PEFT。

---

## 本块学习检查

> 📋 **模型许可证说明**：

## 块 32

**动手试一下：只调Q和V（方案1）**

先别急着全改，我们从最轻量的起点开始。把 `target_modules` 配成下面这样：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示只微调注意力里的查询（Q）和值（V）投影层。这么做的好处很直接：

- **参数少**：要训练的量小，显存压力低。
- **训练快**：迭代速度更快，适合快速验证想法。

这是常见的轻量起点，但具体效果仍取决于模型结构和数据，不是所有场景都够用。

**本块学习检查**：打开你的LoRA配置文件，把 `target_modules` 改成只含 `"q_proj"` 和 `"v_proj"`，然后打印模型可训练参数量，对比修改前后的数值变化，确认参数确实变少了。

## 块 33

## LLaMA系列：target_modules怎么配？

你正在学LoRA微调，目标很明确：用最少参数把大模型调成你想要的样子。LLaMA系列是当前最常用的底座模型，它的`target_modules`配置非常标准，先记住这个最简写法：

```python
["q_proj", "v_proj"]
```

**为什么是这两个？** LLaMA的注意力机制里，Q（查询）、K（键）、V（值）三个投影是分开的独立模块，分别叫`q_proj`、`k_proj`、`v_proj`。实践中，只对Q和V做低秩适配，已经能在多数任务上取得不错效果，同时节省显存和训练时间。如果你后续想追求更高精度，可以再把`k_proj`和`o_proj`（输出投影）加进去，但那是后话——先跑通Q/V再说。

---

### 对比：GPT-2为什么不一样？

如果你换用GPT-2做实验，会发现它的`target_modules`写法完全不同：

```python
["c_attn"]
```

**核心区别在于权重布局。** GPT-2的注意力层用的是Transformers库里的`Conv1D`模块，而不是标准的`nn.Linear`。`Conv1D`把Q/K/V三个投影合并进了一个叫`c_attn`的模块里，而且它的权重形状是`(in_features, out_features)`——这跟`nn.Linear`的`(out_features, in_features)`正好是转置关系。

**这带来一个关键坑：** 在GPT-2这类层上加LoRA时，必须设置`fan_in_fan_out=True`，PEFT才能按转置布局正确注入低秩矩阵。否则矩阵维度对不上，训练直接报错或结果错乱。

一个可直接运行的GPT-2配置示例：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn

## 块 34

## BERT的target_modules配置

对于BERT这类模型，LoRA通常只作用于**注意力层**的投影矩阵。你可能会看到这样的配置：

```python
["query", "value"]
```

但**注意**：不同模型的模块命名方式不同，不能直接复制其他模型的配置。你需要先查看自己加载的模型实际有哪些模块：

```python
for name, _ in model.named_modules():
    print(name)
```

运行后会看到类似 `q_proj`、`v_proj` 这样的名字，这些就是你要传给 `target_modules` 的值。

**匹配机制**：`target_modules` 按模块名做**后缀匹配**——写 `"q_proj"` 会命中所有以 `q_proj` 结尾的层（各层各头都会覆盖）；也可以传完整名字精确定位，或传正则表达式。

**动手验证**：加载一个BERT模型，运行上面的代码，把输出中所有包含 `proj` 的模块名记下来，然后思考：如果只微调query和value，应该传什么？

> ⚠️ 不同Transformers/PEFT版本、不同模型实现，命名可能变化。请以你实际加载模型的 `named_modules()` 输出

## 块 35

### 6.1 r是什么？

r就是**rank（秩）**，它控制LoRA低秩矩阵的维度。

LoRA的核心公式是：

```text
ΔW = (α/r)BA
```

这里的r，决定了A和B的**中间维度**。

举个例子：假设原矩阵是`4096×4096`，当你设置`r=8`时：

- A的维度是`8×4096`（输入侧，即r×d_in）
- B的维度是`4096×8`（输出侧，即d_out×r）

这样，原本需要更新4096×4096个参数，现在只需训练两个小矩阵，大大减少了参数量。

---

### 6.2 r影响什么？

**

## 块 36

**lora_alpha 的作用：控制 LoRA 的更新幅度**

LoRA 的最终更新公式不是 `BA`，而是 `(alpha/r)BA`。这里的 `alpha` 就是缩放系数，它直接控制 LoRA 对模型权重的影响强度。

**动手算一算：**

配置 `r=8`，`lora_alpha=16`，缩放比例就是 `16/8=2`。这意味着 LoRA 的更新被放大了 2 倍，影响增强。

**常见起点设置**

一个常用的经验起点是 `alpha = 2 × r`。注意这只是起点，不是硬性规则，最终应以验证集表现来调整。

- `r=8`，`alpha=16`
- `r=16`，`alpha=32`

**本块学习检查**：

## 块 37

**8.1 dropout作用（动手理解）**

训练时，LoRA会**随机关闭部分神经连接**，就像临时“屏蔽”几条小路，防止模型死记硬背训练数据（即**过拟合**）。

配置示例：`lora_dropout=0.05`，意思是**5%的LoRA路径随机失活**。

**8.2 怎么选dropout？（经验起点，非固定）**

| 数据规模 | dropout参考值 |
|---------|-------------|
| 数据少   | 0.05~0.1    |
| 数据多   | 0~0.05      |

如果训练数据**非常丰富**，可直接设 `lora_dropout=0`。

**动手小

## 块 38

## 9.1 基础配置：快速上手LoRA

这一节我们直接看一个能跑的配置——它特别适合**7B模型测试**，也就是你刚开始做LoRA微调时的“标准起手式”。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量
    lora_alpha=16,          # 更新缩放
    target_modules=[        # 插入位置
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,      # 防止过拟合
    bias="none",
    task_type="CAUSAL_LM"   # 语言模型任务
)
```

**每个参数在干什么？**

| 参数 | 值 | 作用 |
|---|---|---|
| `r` | 8 | 决定LoRA的容量，值越大能学的东西越多，但参数也越多 |
| `lora_alpha` | 16 | 控制更新幅度，通常设为`r`的2倍 |
| `target_modules` | `q_proj`/`v_proj` | 指定在模型的哪些层插入LoRA，这里只改注意力层的Q和V |
| `lora_dropout` | 0.05 |

## 块 39

## 九、完整LoraConfig案例（重点）——基础配置

你现在的目标是LoRA微调，而且喜欢动手实操，所以这一节我们直接通过一个完整案例，把 `LoraConfig` 的关键参数串起来。案例用的是 `bert-base-uncased` 做序列分类（`SEQ_CLS`），比如情感二分类。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 加载带分类头的基础模型（分类头是新初始化的，需要一起训练/保存）
# 模型许可证：BERT 采用 Apache 2.0 许可证，可商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练需根据 batch size 适当增加
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

注意：这里分类头是**随机初始化**的，所以它必须参与训练，否则分类结果就是乱的。

### 第二步：配置LoraConfig

```python
lora_config = LoraConfig(
    r=8,                # LoRA秩，决定低秩矩阵的维度
    lora_alpha=16,      # 缩放系数，控制LoRA更新的强度
    lora_dropout=0.05,  # 防止过拟合
    bias="none",        # 不训练偏置项
    task_type="SEQ_CLS",# 序列分类任务
    target_modules=["query", "value"],  # BERT注意力模块名
    modules_to_save=["classifier"],     # 关键：把新分类头也纳入训练并保存
)
```

**重点理解 `modules_to_save`**：`save_pretrained` 默认只保存Adapter权重，分类头作为基础模型的一部分不会被保存，除非通过 `modules_to_save` 显式指定。如果不加这一项，分类头在保存/重载时不会随Adapter一起保存，导致重载后分类头退回**随机初始化**，分类结果错乱。加上它，分类头才会**参与训练并随Adapter一起保存**。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 应看到 LoRA 参数 + classifier 一起可训练
```

### 第四步：保存与重载验证

```python
# 训练完成后保存：Adapter权重 + modules_to_save指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")

# 重载时：先加载同一个基础模型（结构需与训练时一致），再用PeftModel挂载
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证分类头确实被带回（而不是随机初始化）
# 注意：重载后需确认 reloaded.base_model.model.classifier 与原始模型结构一致
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

### 关键要点

- `save_pretrained` 保存的是 **Adapter + `modules_to_save` 模块**，不是完整基础模型。
- 重载时必须先加载**同一个**基础模型（结构需一致），再挂载Adapter。
- 重载后需验证分类头权重是否一致，确保 `modules_to_save` 生效。
- 不同模型的分类头属性名可能不同（如 `classifier` / `score`），`base_model.model.<...>` 的访问路径也可能随版本/模型而变，请以实际结构为准。
- 实际项目里也可以用一批固定输入对比logits是否一致来验证。

### 本块学习检查

动手验证一下：把 `modules_to_save` 去掉后重新训练并保存，观察重载后分类头权重是否一致，记录结果差异。

## 块 40

## 10.1 普通文本生成：LoRA参数怎么配？

你已经知道LoRA的核心是`r`、`lora_alpha`、`lora_dropout`这几个参数，现在直接看结论——做普通文本生成（比如续写、生成故事），推荐这样配：

```python
r=8
alpha=16
dropout=0.05
```

**为什么这么配？** 普通文本生成任务相对“温和”，不需要模型记住太多新知识，所以`r`（低秩矩阵的秩）不用太大，8就够了。`alpha`是缩放系数，取16，`alpha/r`是缩放因子，用于调整低秩更新的幅度，实际更新强度还受学习率等因素影响。`dropout=0.05`是防止过拟合的，因为文本生成数据量通常不小，5%的随机丢弃足够。

---

### 对比一下其他任务，你会更清楚

| 任务类型 | r | alpha | dropout | 直觉 |
|---------|---|---|---------|------|
| 普通文本生成 | 8 | 16 | 0.05 | 任务简单，小r够用 |
| 指令微调 | 16 | 32 | 0.05 | 需要模型学会“听指令”，能力更强 |
| 领域知识增强 | 32 | 64 | 0.05 | 要注入大量新知识，r要更大 |

注意：领域知识增强的dropout推荐值为0.05。在知识注入任务中，dropout同样用于防止过拟合，但调整原则是：若数据量充足且模型容量够，可保持0.05；若出现过拟合迹象，可适当增大至0.1。

---

### 动手验证一下

**本块学习检查**：在代码里分别用`r=8, alpha=16`和`r=32, alpha=64`配置两个`LoraConfig`，打印出两个配置的`r`和`

## 块 41

## 10.1 普通文本生成：LoRA核心配置实战

你正从零掌握LoRA微调，目标很明确：用最少代码跑通文本生成。下面这份配置是起点，先动手跑起来，再逐行理解。

```python
LoraConfig(
    r=8,                # LoRA容量：秩为8，控制可学习参数规模
    lora_alpha=16,      # 缩放因子：α/r = 16/8 = 2，放大更新强度
    target_modules=["q_proj", "v_proj"],  # 只改注意力层的Q和V投影
    lora_dropout=0.05   # 训练时随机丢弃5%的LoRA参数，

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思想：冻结原模型，只训练一小部分新增参数。但这里有一个关键点需要你注意——**LoRA省的是“训练”的力气，不是“加载”的力气**。

用一张图来理解：

```text
原模型参数：冻结（仍然占满显存）
LoRA参数：训练（只占很小一部分）
```

**基础模型本身仍然需要完整加载到显存中。**

我们来算一笔账。假设你要微调一个7B（70亿参数）的模型：

- 如果用FP16精度存储，每个参数占2字节
- 7B × 2 bytes ≈ **14GB显存**

这14GB只是**把模型放进去**的代价。真正训练时，还需要额外空间存放：

- **激活值**：前向传播过程中的中间结果
- **梯度**：反向传播计算出的更新方向
- **优化器状态**：如Adam的动量项

把这些都加起来，实际显存需求会**远高于14GB**。

这就是为什么即使有了LoRA，普通消费级显卡（如8GB或12GB显存）依然跑不动7B模型的微调——**瓶颈

## 块 43

## 1.2 QLoRA核心思想

**一句话理解**：QLoRA = 把大模型“压缩”成4-bit精度，再在上面挂LoRA进行微调。

**为什么这么做？**

- 普通LoRA：FP16模型 + LoRA（模型占显存大）
- QLoRA：4-bit模型 + LoRA（模型占显存小，省下的空间留给LoRA训练）

**具体流程**：

1. 先把FP16基础模型做4-bit量化
2. 加载这个量化后的模型（权重冻结不动）
3. 在模型上插入LoRA适配器
4. 只训练LoRA参数（保持FP16/BF16精度）

**关键区别**：量化的是**基础

## 块 44

## 3.1 模型存储方式：为什么4-bit能省这么多显存？

先看一张表，记住每个参数占多少空间：

| 类型 | 每个参数占用 |
|------|------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下**：7B模型用FP16存，需要 `7B × 2 ≈ 14GB`；换成4-bit，只需 `7B × 0.5 ≈ 3.5GB`。显存直接减少约75%。

**注意**：3.5GB只是**理论下界**（只算量化后的权重）。实际占用会更高，因为还要存量化常数/缩放因子、分组元数据（NF4按张量归一化量化，每组都要额外参数）、未量化的模块（norm、embedding、lm_head、LoRA参数），以及训练时的激活

## 块 45

**NF4是什么？——给想动手做LoRA微调的你**

NF4全称NormalFloat4，中文叫“正态分布4位浮点”。注意，这只是通俗叫法，它**不是**IEEE 754那种通用浮点格式，而是一套**为神经网络权重“量身定制”的4-bit离散码本**。

QLoRA论文发现：神经网络权重通常近似正态分布，大量数值集中在0附近。NF4就利用这一点，预先设定**16个（2⁴）量化取值点**，这些点基于标准正态分布的量化函数（quantile function）生成，使得每个量化区间包含大致相等的概率质量，从而在权重密集区提供更高的精度。

**量化过程**：把每组权重归一化后，映射

## 块 46

**NF4为什么适合模型权重？**

大模型训练后的权重近似**正态分布**——大量参数集中在0附近。普通INT4是平均划分（如[-8,-7,...,7]），无法精细表示中心密集区域。

NF4则**根据正态分布设计量化范围**，让0附近精度更高，更贴合权重分布，从而降低精度损失。

**量化过程**：FP16权重（如0.023、-0.145、0.876）→ 归一化 → 映射到NF4 → 保存4-bit值。推理时反量化回近似FP16。

---

**本块学习检查**：请用一句话说明NF4与普通INT4在量化范围设计上的核心差异，并指出这一差异带来的直接好处。

## 块 47

**5.1 为什么需要双重量化？**

4-bit量化不只是存权重，还得为每组权重保存**量化参数**（scale和zero point）。这些额外参数同样占显存，大模型参数多，积少成多也很可观。

**5.2 双重量化思想**

普通量化：FP16权重 → 4-bit权重 + scale参数。

双重量化：先量化模型权重，再**把scale参数也量化一次**，减少scale的存储开销。

**5.3 优势**

减少额外量化参数和显存占用。大模型参数巨大，哪怕少量优化也能省下大量显存。

---

**本块学习检查**：请用自己的话复述“双重量化”与“普通量化”的区别，并指出它主要节省了哪部分显存。

## 块 48

**6.1 为什么需要分页优化器？**

训练时显存峰值不只来自模型，还有**梯度**和**优化器状态**。以Adam为例，它需要保存**一阶动量**和**二阶动量**，这两项会额外占用大量显存。

QLoRA虽然大幅降低了模型本身的显存占用，但训练过程中仍可能遇到**显存突然不足**（OOM）的情况——尤其是当梯度计算和优化器状态更新同时达到峰值时。

**6.1 分页优化器思想**

Paged Optimizer借鉴了操作系统的**虚拟内存**机制。核心思路是：借助NVIDIA的**统一内存（Unified Memory）**，在显存出现峰值、即将OOM时，把优化器状态自动“分页”换出到CPU内存，需要参与计算时再换回GPU。

流程如下：

```
GPU显存 → 出现峰值（接近/超出上限） → 优化器状态按页换出到CPU内存

## 块 49

## 6.3 类比理解：把显存想象成你的桌面

**普通优化器（非分页）**：所有优化器状态（动量、方差等）都放在“桌面”（显存）上。桌面满了，训练就报 OOM，无法继续。

**分页优化器**：桌面只放当前常用的文件（活跃参数状态）；暂时不用的文件放进柜子（CPU 内存）；需要时再取回桌面。这样桌面永远不会被塞满。

## 6.4 实际启用：一行代码搞定

分页优化器**不需要你手写换页逻辑**，bitsandbytes 库帮你实现。只需在 `TrainingArguments` 中设置 `optim` 参数：

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
    optim="paged_adamw_8bit",        # 关键：分页优化器
)
```

**两个可选值**：
- `paged_adamw_8bit`：优化器状态用 8-bit 存储，更省显存（QLoRA 论文常用）
- `paged_adamw_32bit`：精度更高，但显存占用更大

**注意**：`optim` 的可用取值随 Transformers / bitsandbytes 版本变化（还有 `adamw_bnb_8bit` 等）。若报“未知 optim”错误，查你当前版本的文档或 `--help`，换用支持的名称即可。

**使用策略**：分页优化器主要缓解**峰值 OOM**，通常搭配 `gradient_checkpointing=True`、小 `per_device_train_batch_size` + 梯度累积一起用，把显存压到最低。

---

**本块学习检查**：打开你的训练脚本，确认 `optim` 参数已设置为 `paged_adamw_8bit`，并检查 `gradient_checkpointing` 是否为 `

## 块 50

## 7.1 4-bit量化：把模型“瘦身”

**作用**：减少模型权重大小。

想象一下，原本用FP16（16位精度）存储的模型权重，现在改用NF4（4位精度）来存。位数少了，占用的显存自然就小了。

```
FP16（16位）  →  NF4（4位）
```

**直观理解**：就像把一张高清照片压缩成缩略图，文件变小了，但关键内容还在。NF4就是QLoRA用来压缩模型权重的“压缩算法”。

---

## 7.2 双重量化：连“压缩参数”也压缩

**作用**：减少量化参数占用。

第一次量化后，模型会生成一些“缩放因子”（scale）来帮助还原精度。这些缩放因子本身也占空间，怎么办？**再量化一次**。

```
scale（缩放因子）  →  再次量化
```

**直观理解**：压缩完照片后，发现压缩工具本身也挺重，那就把工具也折叠一下。双重量化就是“连压缩参数也压缩”，进一步省显存。

---

## 7.3 分页优化器：显存不够，内存来凑

**作用**：降低训练峰值显存。

训练时显存不够用？QLoRA会把一部分数据暂时

## 块 51

### load_in_4bit：让模型“瘦身”的关键开关

想象一下，你要把一个巨大的模型塞进显存有限的GPU里。`load_in_4bit=True` 就是那个“压缩打包”的开关——它让模型权重以4-bit精度加载，而不是默认的32-bit。这样，模型占用的显存能大幅下降，你才能在有限的硬件上跑起大模型微调。

### bnb_4bit_quant_type：用哪种“压缩算法”

设置 `"nf4"`，即使用NF4量化。这是一种专门为4-bit设计的量化方式，能在压缩的同时尽量保留模型原有的表达能力。简单说，它决定了“怎么压”才能让信息损失最小。

### bnb_4bit_use_double_quant：再压一层

开启双重量化。这相当于在第一次压缩的基础上，对量化过程中产生的额外参数再做一次量化，进一步节省显存。代价是略微增加一点计算量，但通常收益远大于开销。

### bnb_4bit_compute_dtype：计算时“解压”成什么精度

这是最容易踩坑的地方，需要你动手验证。

**核心概念**：权重以4-bit（NF4）**存储**，是为了省显存；但在实际做矩阵运算时，计算机会把它们“解压”回一个更高精度的格式来计算，这个计算精度就是由 `bnb_4bit_compute_dtype` 指定的。

**为什么必须显式设置**：如果不设置，默认通常是 `float32`，但 bitsandbytes 实际计算时可能使用 float16，显式设置可以确保计算精度符合预期。所以QLoRA配置里几乎总是显式指定它。

**怎么选：BF16还是FP16？** 不要写死 `torch.bfloat16`。BF16需要较新的GPU支持（NVIDIA Ampere架构及以后，如RTX 30/40系、A100、H100）。老GPU（如RTX 20系）不支持，这时必须用 `torch.float16`。**最稳妥的做法是让代码自动判断**：

```python
import torch
from transformers import BitsAndBytesConfig

# 根据GPU能力选择：支持BF16就用bfloat16，否则退回float16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

**两者的区别**：BF16动态范围更大、更不易溢出，训练通常更稳定

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面几节我们分别看了 `BitsAndBytesConfig`（怎么用 4-bit 加载模型）、分页优化器（`optim`）和 `bnb_4bit_compute_dtype`（计算精度）。你可能已经意识到这些是 QLoRA 的“零件”，但最常卡住的问题是：**这些零件到底怎么拼成一个能跑的训练脚本？**

下面给你一个**最小可运行骨架**，把完整流程串起来。模型和数据集替换成你自己的即可。

> **模型许可证说明**：本骨架示例引用的模型需遵守其原始许可证。例如，若使用 Llama 2 系列模型，需遵循 Llama 2 Community License（商用且月活用户大于 7 亿时需获得 Meta 授权）。请根据你实际选用的模型确认对应的许可证条款。

> **显存要求提示**：4-bit 量化加载 7B 参数模型需要至少 6GB 显存，请确保你的环境满足最低要求。

### 整体流程：5 步走

1. **配置 4-bit 量化**（`BitsAndBytesConfig`）
2. **以 4-bit 加载基础模型**
3. **为 k-bit 训练做准备**（`prepare_model_for_kbit_training`）
4. **注入 LoRA**（`get_peft_model`）
5. **配置训练参数并启动**（`Trainer`）

---

### 第 1 步：配置 4-bit 量化

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

- `load_in_4bit=True`：启用 4-bit 加载。
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化类型。
- `bnb_4bit_use_double_quant=True`：开启双重量化，进一步省显存。
- `bnb_4bit_compute_dtype`：计算时用的精度，优先 bfloat16，不支持则退回 float16。

> **注意**：`torch.cuda.is_bf16_supported()` 在 PyTorch 2.x 中已弃用。若你的 PyTorch 版本较新，可改用 `torch.cuda.get_device_capability()` 检查设备能力（如计算能力 ≥ 8.0 通常支持 bfloat16），或直接固定使用 `torch.float16` 以确保兼容性。

### 第 2 步：以 4-bit 加载基础模型

```python
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except OSError as e:
    print(f"模型或 tokenizer 加载失败：{e}")
    raise
```

注意：如果 tokenizer 没有 pad token，用 eos token 顶上，否则数据整理时会报错。上述 `try-except` 用于捕获 IO 异常（如模型文件下载失败、路径错误等），避免程序直接崩溃。

### 第 3 步：为 k-bit 训练做准备（关键一步）

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步处理了量化模型训练时的两个隐患：norm/输出层的精度问题，以及梯度检查点所需的输入梯度。**不要跳过这一步**，否则训练可能直接崩。

### 第 4 步：注入 LoRA

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

`target_modules` 按你实际模型的模块名调整，也可以用 `"all-linear"` 省事。`print_trainable_parameters()` 会打印可训练参数量，确认只有 LoRA 参数在训练。

### 第 5 步：训练参数 + 启动训练

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
trainer.train()
```

两个关键呼应点：

- `gradient_checkpointing=True` 与第 3 步的 `use_gradient_checkpointing=True` 是配套的，两者都要开。
- `optim="paged_adamw_8bit"` 是分页优化器，依赖 bitsandbytes，名称可能随版本略有不同。

---

### 本块学习检查

**任务**：把上面的骨架代码复制到你的环境里，把 `model_name` 换成你实际使用的模型名称，运行并确认训练循环能正常启动。

## 块 53

## 九、把 QLoRA 的零件组装起来（可跑通骨架）

你现在的目标是跑通 LoRA 微调，但用的是更省显存的 QLoRA 路线。下面这 5 步就是完整的“组装清单”，每一步都有明确职责，缺一不可：

**第 1 步：`BitsAndBytesConfig` —— 决定基础模型怎么“瘦身”加载**

这一步配置的是 4-bit 加载方式：用 NF4 量化 + 双重量化 + 计算 dtype。它决定了模型以什么精度被读进显存。

**第 2 步：`prepare_model_for_kbit_training()` —— 量化模型的“适配器”**

这是 QLoRA 的**必要一步**。量化后的模型不能直接拿去训练，这个函数会冻结模型参数、启用梯度检查点、将模型转换为半精度等，以准备模型进行 k-bit 训练。**跳过它，训练常常不收敛或直接报错**。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

把可训练的 LoRA 适配器注入到已经冻结的 4-bit 主干模型上。主干被冻结不更新，只有 LoRA 参数在训练。

**第 4 步：`gradient_checkpointing=True` —— 梯度检查点要开启**

在 `TrainingArguments` 中设置 `gradient_checkpointing=True`，同时确保 `prepare_model_for_kbit_training` 中 `use_gradient_checkpointing=True`，以启用梯度检查点。

**第 5 步：`optim="paged_adamw_8bit"` —— 分页优化器**

启用分页优化器可以缓解显存峰值压力，但 QLoRA 省显存的关键是 4-bit 量化。

> 上面这个骨架是“能说明关系”的最小版本。真实训练时，你需要换成足够规模的数据、合理的超参，并且按你实际使用的库版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数写法，以及 `processing_class`/`tokenizer` 的写法。

---

## 十、你的电脑能跑吗？

**如果只学原理：不需要 GPU。**

**如果要实际运行 QLoRA：**

- **CPU**：4 核以上
- **内存**：16GB 以上
- **GPU**

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会误以为LoRA训练完，就得到了一个“微调后的新模型”。**不是的。**

LoRA训练过程中，基础模型是**冻结**的（不更新参数），真正被训练的是额外添加的**LoRA Adapter**（适配器）。最终的结构是：

```text
基础模型 + LoRA Adapter
```

举个例子，你用Llama-7B做LoRA训练，训练完成后保存的**通常只有**：

```text
adapter_model.safetensors   （Adapter权重）
adapter_config.json          （Adapter配置）
```

**而不是**完整的Llama-7B模型文件。

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（通常只有几MB到几十MB）；
- 保存快；
- 可以多个任务共享一个基础模型。

例如同一个基础模型，可以搭配不同的Adapter：

```text
Base Model + 客服Adapter
Base Model + 医疗Adapter
Base Model + 法律Adapter
```

**缺点：**

部署推理时，需要**基础模型和Adapter两个部分同时存在**，缺一不可。

---

## 动手验证一下

打开你训练好的LoRA输出目录，确认里面是否只有`adapter_model.safetensors`和`adapter_config.json`，而不是完整的模型文件。这能帮你直观理解“LoRA不直接修改基础模型”这句话的含义。

## 块 55

## 推理时为什么要先合并LoRA权重？

你训练完LoRA后，模型在推理时其实有两条路可以走。先看**不合并**的情况：

```
输入 → Base Model + LoRA Adapter → 输出
```

这种方式需要你同时加载基础模型和Adapter两个东西，部署时比较麻烦——你得维护两份文件，还要保证加载逻辑正确。

而**合并权重**之后，情况就变成：

```
输入 → 完整模型 → 输出
```

只需要一个完整的模型文件，部署时直接加载它就行，省心很多。

### 合并到底做了什么？

LoRA训练时，模型权重的变化可以写成：

```
W' = W + ΔW
```

其中 `ΔW = BA`，也就是LoRA学到的

## 块 56

### 3.1 merge_and_unload() 是做什么的？

你训练完LoRA后，模型其实还是“基础模型 + 适配器”的组合。`merge_and_unload()` 就是把这层包装拆掉，把LoRA学到的权重直接写进基础模型里，得到一个完整的、独立的模型。

**简单理解：**

```text
基础模型 + LoRA适配器 → 完整模型
```

---

### 3.2 合并前后对比

**合并前：**

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后：**

```text
普通Transformer模型（包含LoRA更新后的权重）
```

---

### 四、完整流程（重点）

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

---

### 五、加载LoRA Adapter（动手操作）

假设你有基础模型路径 `base_model_path` 和适配器路径 `adapter_path`：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 加载LoRA适配器
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：`基础模型 + LoRA适配器`。

---

### 六、合并权重（重点）

执行一行代码：

```python
# 注意：Llama-2-7b FP16 需要至少14GB显存
merged_model = model.merge_and_unload()
```

完成后，`merged_model` 不再是 `PeftModel`，而是**普通Transformer模型**，权重已经包含LoRA的更新。

---

### 七、导出合并后的模型

保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

目录示意：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：实际文件可能更多（如 `generation_config.json`、分片权重等），具体取决于模型和分词器类型。

---

**本块学习检查**：请自己动手跑一遍上面的代码，确认 `merged_model` 的类型不再是 `PeftModel`，并检查导出目录里是否包含 `model.safetensors` 和 `config.json`。

## 块 57

### 3.1 merge_and_unload 方法作用：把“临时拼装”变成“正式成品”

你可以把 LoRA 微调想象成通过训练低秩增量 ΔW=(α/r)BA 来更新预训练模型（Base）的权重。训练时，增量和原权重是分开的，推理时可将增量合并到原权重中。

**merge_and_unload 的作用，就是把这个“合并”过程永久固化**：把 LoRA 增量真正合并进 Base 模型，得到一个独立的、完整的模型文件。合并后，这个模型就是一个普通的 Transformers 模型，不再依赖 PEFT 库。

---

### 合并后如何加载推理？

合并后，加载方式变得极其简单，**不再需要 PEFT**，直接用 Transformers 加载即可：

```python
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

推理流程也回归最简：**用户输入 → 完整模型 → 生成结果**。

---

### 合并 vs 不合并：一张表看懂区别

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + 增量 | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高（需 PEFT） | 简单（直接 Transformers） |
| 多个任务切换 | 方便（换增量即可） | 不方便（需重新合并） |
| 推理加载 | 需要 PEFT | 直接 Transformers |

---

### 什么时候应该合并？

**核心原则：合并是权衡，不是必须。** 取决于你的实际场景：

- **需要多个增量动态切换**（如多任务）→ **不合并**更方便
- **部署框架原生支持 LoRA** → 可以不合并
- **对存储/分发体积敏感** → 不合并更省空间
- **对精度有要求（尤其 QLoRA）** → 合并可能引入精度损失，需谨慎

---

### 合并后必查清单（避免“能加载但行为不对”）

合并导出

## 块 58

### 10.1 推荐合并场景

你正在学LoRA微调，最终目标是把模型用起来。这里讲两个**该合并**的典型场景，帮你快速判断。

---

**场景1：上线部署——只要一个固定Adapter，追求简单**

想象生产环境：服务器 → 直接加载模型 → 提供API。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

如果你只需要**一个固定的Adapter**，且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载），合并通常更方便。

> ⚠️ 注意：如果推理框架本身支持直接加载LoRA（如vLLM的多LoRA服务），或你需要**热切换多个Adapter**，那**不合并**反而更灵活。是否合并，看框架能力和业务需求。

---

**场景2：分享模型——让对方开箱即用**

## 块 59

## 10.2 不建议合并场景

**场景1：多个任务Adapter**

假设你有一个基础模型，上面挂了多个Adapter（比如Adapter A、B、C）。**保留Adapter更灵活**——你可以随时切换不同任务，而不需要重新训练。

**场景2：持续训练**

如果你打算继续调整LoRA，**保留Adapter**。合并后就没法再微调了，灵活性会大打折扣。

> **动手验证**：想象你训练了三个LoRA，分别用于情感分析、命名实体识别和文本分类。你会选择合并还是保留？为什么？——答案：保留，因为可以按需切换。

---

## 十一、QLoRA模型合并注意事项（重点）

QLoRA训练时，基础模型以**4-bit（NF4）**存储，

## 块 60

## 11.1 直接合并可能遇到的情况

如果你在 4-bit 量化状态下直接执行 `merge_and_unload`，**可能**会遇到两类问题：

- **精度损失**：4-bit 权重需要反量化后再合并，这个过程与全程用高精度计算并不完全等价，结果可能有细微偏差；
- **类型或版本不兼容**：权重类型不匹配，或者你使用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> ⚠️ 注意：上面只是**可能出现**的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你的**版本**、量化配置，以及是否按下面的流程先反量化。请以你所用版本的文档和实际运行结果为准。

---

## 11.2 常见做法：先反量化到高精度再合并

一个稳妥的做法是：**不要在 4-bit 状态下直接合并**，而是先用高精度（FP16/BF16）重新加载基础模型，再挂载同一个 Adapter 后合并。

流程如下：

```text
以高精度（FP16/BF16）重新加载基础模型
        ↓
加载同一份 LoRA Adapter
        ↓
merge_and_unload()
        ↓
保存完整模型
```

这样合并发生在高精度权重上，避开了 4-bit 状态下的类型和精度问题。

### 可执行代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 模型许可证：请确认基础模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权），并遵守相应条款
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

## 本块学习检查

**任务**：对照上面的代码，回答两个问题（不需要运行）：

1. 代码中哪一行是“避免在 4-bit 状态下合并”的关键？为什么这一行能避开精度问题？
2. 如果显存不足，代码中给出的备选方案是什么？

## 块 61

## 11.1 直接合并可能遇到的情况（QLoRA 合并注意事项）

**核心要点**：QLoRA 训练时模型是 4-bit 量化的，但合并 LoRA 权重时，**不要**再用量化配置加载基础模型。

**具体做法**：加载基础模型时，**不传** `quantization_config` 或 `load_in_4bit`，这样拿到的是 FP16/BF16 高精度权重，再执行 `merge_and_unload` 完成合并。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**三个必须知道的坑

## 块 62

## 合并时开启校验并做一致性验证（建议）

`merge_and_unload` 支持 `safe_merge=True` 参数，合并过程中会自动做数值检查（比如检测是否产生 NaN），相当于给合并操作加了一道保险。代码很简单：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，**强烈建议做一次一致性验证**：用同一批固定输入，分别让「合并前的 PeftModel」和「合并后的模型」做推理，对比两者的输出是否基本一致。这样能确认合并过程没有引入明显偏差。

**最小验证流程：**

1. 准备一小批固定输入（prompt）
2. 分别用合并前和合并后的模型推理
3. 对比 logits 的最大绝对差

## 块 63

## 合并后别急着用——先做一次“一致性体检”

你刚在 §11.2 里用 `merge_and_unload()` 把 LoRA 适配器合并进了基础模型。现在模型看起来“正常”了，但**合并过程有没有悄悄改变模型的输出？** 尤其是 QLoRA 场景下，因为权重是量化过的，合并后和合并前存在微小差异是正常的，但差异不能大到离谱。这一节就是教你用一段固定输入，给合并前后的模型做一次“体检”。

### 为什么不能直接拿合并前的对象对比？

代码注释里强调了一个关键点：`merge_and_unload()` 会**返回一个新的合并后的模型对象**，原 `PeftModel` 对象不会被修改，仍可保留用于对比。但为了确保对比基准的纯净性，正确做法是：**重新加载一份全新的、未合并的 `PeftModel`**，专门用来做对比。这就是代码里 `base_for_ref` 和 `model_before` 的作用——它们是独立加载的，没被合并操作污染过。

### 体检项目一：比较 logits（模型输出的原始分数）

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

这段代码计算合并前后模型对同一输入产生的 logits 的**最大绝对差**。这个值越接近 0，说明合并前后模型的行为越一致。QLoRA 场景下，由于量化误差，这个值通常是一个**较小的近似值**，而不是精确的 0——这属于正常现象。

### 体检项目二：比较生成文本

logits 数值接近还不够，更直观的验证是看模型**实际生成的话**是否一致：

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

注意这里设置了 `do_sample=False`——**去掉随机采样**，保证生成过程是确定性的，这样对比才有意义。如果开了采样，即使模型完全没变，两次生成也可能不同，那就没法判断了。

### 怎么解读结果？

- **logits 最大绝对差很小 + 生成文本一致**：合并质量很好，可以放心使用。
- **logits 差异较大但生成文本一致**：差异在可接受范围内，模型输出没受影响。
- **生成文本

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 64

## 合并后怎么确认没搞坏？——QLoRA 合并校验实操

你刚把 LoRA 权重合并进基础模型，心里肯定犯嘀咕：**这模型还能用吗？** 别急，这一步就是教你如何验证。

> **模型许可证**：本教程以 Llama 2 为例，适用 Llama 2 Community License（商用月活>7亿需Meta授权）。使用其他模型时请自行确认其许可证。

### 为什么不能拿合并后的模型自己对比自己？

`merge_and_unload()` 返回合并后的新模型，**原 model 对象不受影响**，PEFT 包装仍然存在。若需保留合并前模型，应提前保存副本。

**正确做法**：重新加载一份**未合并**的 PeftModel 作为对照（代码里叫 `base_for_ref` + `model_before`），用这份“干净”的模型和合并后的模型做对比。

### 显存不够怎么办？

如果两个模型同时放不下，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍推理，把 logits 或生成文本**存到硬盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比两次结果。

### 结果怎么看？——别追求“一模一样”

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个**较小的非零值**，生成文本也可能有个别 token 不同。**重点**是确认没有明显退化、没有 NaN，而不是逐位相同。
- **LoRA（非量化）场景**：一致性更高，`max_abs_diff` 可以非常小。

**是否可接受**：以你的任务指标和验证集为准。

---

## 电脑配置要求

这一步要**实际加载

## 块 65

## 十一、合并时开启校验并做一致性验证（建议）

**为什么建议你做这一步？**

你正在学 LoRA 微调，目标是最终能稳定部署模型。合并 LoRA 权重到基础模型，是“训练完 → 部署前”的关键一步。如果合并后模型输出异常，排查起来很费时间。开启校验并做一致性验证，就是用少量成本提前发现合并问题，避免带病上线。

### 核心操作：开启 `safe_merge=True`

在调用 `merge_and_unload` 时，加上参数 `safe_merge=True`。这个参数的作用是让合并过程更安全，降低因权重叠加顺序或类型问题导致的异常风险。这是官方建议的稳妥做法，尤其适合你目前“LoRA 尚未掌握”的阶段——少踩一个坑，就多一分信心。

### 一致性验证：合并前后对比

验证思路很简单：**用同一份输入，分别跑合并前的模型（Adapter 挂载状态）和合并后的模型，对比输出是否一致**。

- 如果两者输出基本一致，说明合并成功，权重正确融入了基础模型。
- 如果差异明显，说明合并过程可能出了问题，需要回头检查加载方式、dtype 或版本兼容性。

> 注意：这里说的“对比”是逻辑上的验证动作，不需要额外安装工具，用你已有的推理代码跑两遍即可。

### 特别提醒：QLoRA 合并前必须先“反量化”

如果你用的是 QLoRA（4-bit 量化微调），**不要直接在 4-bit 状态下合并**。正确做法是：

1. 重新加载一份**与训练时完全一致的基础模型权重**（即训练时所用的同一份基础模型文件），用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`，注意**不传** `quantization_config` 或 `load_in_4bit`。
2. 然后用 `PeftModel.from_pretrained` 挂载**同一份 Adapter**，并将模型统一转换为一致的 dtype（如 `bfloat16`），再执行合并。

这样合并发生在高精度权重上，天然避开了“在 4-bit 状态下合并”的类型/精度问题。不同 PEFT / bitsandbytes / Transform

## 块 66

## 路线A：合并后部署完整模型

你刚学完 `merge_and_unload()` 合并权重，现在看它怎么落地到部署环节。路线A的思路很直接：**合并完，就当普通模型用**。

### 操作流程

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

三步走完，部署链路非常简单——推理框架只认识"完整模型"，不需要知道 LoRA 是什么。

### 路线A的特点

**优点：**

- **部署链路简单**：推理框架把合并后的模型当作普通模型加载即可，无需理解 LoRA 机制，省去额外配置。
- **推理时不再有 Adapter 切换成本**：权重已经固化在模型里，推理过程中没有额外的 LoRA 计算开销。

**代价：**

- **每个 Adapter 都要保存/分发一份完整模型**。如果你有多个任务（比如 A/B/C 三个 Adapter），就要存三份完整模型，存储成本高。

### 对比：路线B（保留 Adapter 动态服务）

与路线A相对，路线B是**保留 Adapter 不合并**，用支持 LoRA 的推理框架（如 vLLM）在运行时动态挂载：

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

**路线B的优势：**

- 多任务/多租户时**只需一份基础模型 + 多个小体积 Adapter**，节省显存与存储。
- 支持**热切换**，无需为每个任务各存一份完整模型。

**路线B的代价：**

- 依赖框架对 LoRA 的支持与相应配置，链路比直接加载合并模型略复杂。
- 不同框架/版本对 LoRA 的支持范围与限制不同。

### 关键认知

- **路线A**：合并 → 普通模型加载，简单但多任务存储成本高。
- **路线B**：保留 Adapter → 框架动态挂载，省存储但依赖框架支持。

> ⚠️ **注意**：vLLM 的多 LoRA 走的是**路线B**——传入的是**未合并的 Adapter 目录**（含 `adapter_model.safetensors` / `adapter_config.json`），**不是**合并后的完整模型。合并后的模型属于路线A，按普通模型直接

## 块 67

**本块学习检查**：读完本节后，请合上资料，用一句话回答：合并部署和保留 Adapter 动态服务，分别适合什么场景？

---

### 14.3 如何选择：两条路线，看需求

你已经知道 LoRA 微调后有两种部署方式：**合并**（路线A）或**保留 Adapter**（路线B）。怎么选？记住三个经验参考，不是硬性规则：

1. **只训练了一个 Adapter，想部署最简单** → 选**路线A（合并）**。把 LoRA 权重合进基础模型，得到一个完整模型，直接部署，省心。

2. **同一个基础模型要服务多个 Adapter，且推理框架原生支持 LoRA** → 选**路线B（保留 Adapter 动态服务）**。不用反复合并，按需加载不同 Adapter，灵活切换。

3. **QLoRA 精度敏感** → 无论走哪条路，都建议做**合并前后/上线前的一致性验证**，确保精度没跑偏。

> 核心认知：**合并不是唯一部署方式**。具体框架用法

## 测评题目

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1205-003
合并 LoRA 权重后最直接的用途是？
