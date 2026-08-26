# 学习包

## 块 1

好的，我们来把“全参数微调的问题”这部分，用最适合你的方式讲清楚。

你学习LoRA的目标很明确，而且喜欢动手实践。理解“为什么需要LoRA”，是动手配置环境前最关键的一步。我们直接切入核心。

**一、为什么需要参数高效微调？（①②）**
**1.1 全参数微调的问题**

想象一下，传统的模型训练就像要把一整座冰山全部融化重塑。

```text
输入数据

↓

整个模型参数更新

↓

保存完整模型
```

这个过程对硬件的要求极其苛刻。以我们即将接触的7B（70亿）参数模型为例，训练时你的GPU不仅要存下这70亿个参数本身，还要额外保存：

*   **梯度**：用于计算如何更新参数。
*   **优化器状态**：控制更新步伐和方向。

这三样东西加起来，对显存的需求是灾难性的。普通的消费级显卡（比如我们常用的游戏卡）根本装不下，更别提训练了。这正是我们需要LoRA这类“参数高效微调”技术的根本原因——它只更新一小部分参数，而不是全部。

---

**本块学习检查**

为了确保你抓住了核心，请完成这个小任务：

**思考并回答**：全参数微调一个7B模型时，GPU除了要保存模型参数本身，还需要保存哪两类关键数据？这导致了什么直接后果？

## 块 2

**PEFT是什么？**

PEFT（参数高效微调）的核心思想是：**不修改整个大模型，只训练少量新增参数**。这比全参数微调省显存、省时间。

**流程速览**

原始大模型权重冻结，仅训练LoRA引入的低秩矩阵A和B，形成增量ΔW=(α/r)BA，最终保存LoRA适配器（含A、B矩阵）。

**三种方式对比**

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数（r(d_in+d_out)） | 较低 |
| QLoRA | LoRA参数（量化模型

## 块 3

**LoRA是什么？——用“小补丁”微调大模型**

想象你有一本很厚的书（原模型权重 `W`），微调时不想重写整本书，只贴一张“补丁”来修正内容。LoRA（低秩适配）就是这个思路：**原模型 `W` 保持不变**，新增一个更新量 `ΔW`，最终效果是 `W + ΔW`。

关键来了：`ΔW` 不是直接算出来的大矩阵，而是拆成两个小矩阵的乘积并乘以缩放因子：`ΔW = (α/r) × B × A`。如果 `W` 的形状是 `d_out × d_in`，那么 `A` 约为 `r × d_in`

## 块 4

## 3.1 LoRA的问题：模型还是太大

你已经知道LoRA能大幅减少**需要训练的参数**，但有个关键点别忽略：

**基础模型本身，还是得完整加载进显存。**

以7B模型为例，如果用FP16精度加载，大约需要：

```text
14GB显存
```

也就是说，哪怕LoRA只训练一小部分参数，你的GPU也得先装得下这个“大块头”基础模型。

---

## 3.2 QLoRA：给模型“瘦身”再训练

QLoRA（Quantized LoRA）就是来解决这个问题的——**量化版LoRA**。

核心思路一句话：

```text
FP16模型 → 4bit量化 → 减少显存 → 加载LoRA训练
```

训练时的结构变成了：

```text
4bit基础模型 + LoRA参数 → 训练
```

简单说：**先把基础模型“压缩”成4bit精度，省下大量显存，再在上面挂LoRA参数做微调。**

---

## 3.3 QLoRA依赖bitsandbytes

要跑QLoRA，你需要这几个组件：

```text
transformers + peft + bitsandbytes + 支持的量化后端
```

其中`bitsandbytes`负责：

- 低精度量化
- 4bit加载
- 8bit加载

> **关于后端的重要提醒**：本教程统一以**NVIDIA CUDA GPU**为准（这是项目支持范围，也是资料最全的路径）。但“QLoRA只能用NVIDIA”并非技术必然——bitsandbytes近年已在扩展对CPU、AMD ROCm、Intel等后端的支持。**各后端支持程度差异很大**：NVIDIA CUDA是完整支持且验证最充分的；其他后端可能只是实验性/部分支持，功能、性能、稳定性都有差距。如果你的硬件不是NVIDIA GPU，务必查阅官方文档确认对应后端与安装方式，本文示例命令未必适用。
>
> 官方参考：
> - bitsandbytes安装/后端支持：https://huggingface.co/docs/bitsandbytes/main/en/installation
> - bitsandbytes仓库：https://github.com/bitsandbytes-foundation/bitsandbytes
> - PyTorch安装选择器：https://pytorch.org/get-started/locally

## 块 5

## 本块学习检查

动手前先确认你的电脑能跑起来。按下面三步自查，全部通过再继续：

1. **CPU与内存**：打开任务管理器（Windows）或活动监视器（Mac），确认CPU至少4核、内存16GB以上。
2. **磁盘空间**：确认剩余空间≥50GB，模型和训练文件都很占地方。
3. **GPU**：在终端输入 `nvidia-smi`，能看到显卡信息且显存≥8GB即可。若显存不足8GB，建议先升级硬件或改用云GPU。

---

## 电脑配置要求详解

### LoRA最低配置

| 项目 | 要求 |
|------|------|
| CPU | 4核以上 |
| 内存 | 16GB+ |
| 磁盘 | 50GB以上 |
| GPU | NVIDIA GPU（推荐） |
| 显存 | 最低8GB，推荐12GB+ |

### QLoRA的显存要求

QLoRA的核心优势是**用更小的显存跑大模型**——它把基础模型量化到4-bit，显存占用通常比LoRA全精度加载更低。但具体能跑多大，取决于多个因素，**不是固定门槛**：

**7B级别模型的参考值：**
- 参考下限：约8~12GB（短序列、小batch）
- 推荐：16GB+

**为什么是“参考”而不是“硬保证”？** 因为实际显存占用受这些条件影响：

- **模型规模**：7B是基准，13B/70B显存需求显著上升
- **序列长度**：短序列（≤512）省显存，越长占用越高
- **batch大小**：小batch（1

## 块 6

**本块学习检查**：在终端运行 `python --version`，确认输出的 Python 版本号以 `3.10` 或更高数字开头（例如 `3.10.12` 或 `3.11.5`）。如果版本低于 3.10，请先升级 Python 环境再继续。

---

### 5.1 Python版本检查

动手第一步，先确认你的 Python 版本是否达标。在终端执行：

```bash
python --version
```

**目标版本**：`Python 3.10+`（即 3.10 或更高版本）。

> **为什么重要**：LoRA 微调依赖的现代库（如 PEFT、bitsandbytes）对 Python 版本有最低要求。版本过低会导致安装依赖时出现兼容性报错，白白浪费时间排查。

---

### 5.2 PyTorch检查（关键一步）

**核心要点**：LoRA/QLoRA 依赖 PyTorch，且 `bitsandbytes` 的 CUDA 量化路径要求安装**与本机 CUDA 匹配的 GPU 版 PyTorch**。

**常见坑**：直接 `pip install torch` 通常装到 CPU 版，QLoRA 会无法使用。

**正确做法**：以 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为准。例如 NVIDIA CUDA 12.1 环境，执行：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证安装**：运行以下 Python 代码：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # 查看 torch 编译使用的 CUDA 版本
```

**正常结果**：`torch.cuda.is_available()` 输出 `True`，表示 PyTorch 能看到 CUDA GPU。

> **注意**：`True` 只说明 **PyTorch 侧**能用 GPU，**并不足以证明 bitsandbytes 的 4-bit 量化路径可用**。后者是否真能跑 4-bit，以第八章、第十三章对 `bitsandbytes` 的检查为准。

---

### 5.3 查看GPU信息

在终端运行：

```bash
n

## 块 7

**bitsandbytes：量化与QLoRA的关键依赖**

bitsandbytes 是一个 Python 库，核心功能包括：

- **8bit量化**：将模型权重压缩到8位，大幅减少显存占用。
- **4bit量化**：进一步压缩到4位，显存占用更低。
- **QLoRA训练**：这是本教程的核心场景。

**为什么必须安装？**

本教程采用 **基于 bitsandbytes 的 QLoRA 4-bit 量化方案**，因此它是前置依赖。简单说：QLoRA 的核心思想是“4-bit 量化基础模型 + LoRA”，但具体实现上，本教程统一用 bitsandbytes + NVIDIA CUDA 来演示。如果你用其他量化库，请参考对应文档。

**安装命令**

```bash
pip install bitsandbytes
```

---

**本块学习

## 块 8

**动手前先搞懂：为什么必须检查CUDA？**

你的目标是跑LoRA微调，而bitsandbytes是QLoRA量化的核心工具。它就像一座桥，必须依赖CUDA才能让GPU工作。如果桥没搭对（版本不匹配），后果很直接：CUDA不可用、量化失败、加载模型报错——每一步都会卡住你。

**三步快速自检（跟着做就行）：**

1. **看CUDA版本**  
   终端运行`nvidia-smi`，在输出里找到`CUDA Version`，记下这个数字。

2. **测bitsandbytes**  
   运行`python -m bitsandbytes`。如果正常，它会显示CUDA版本和GPU信息——看到

## 块 9

**本块学习检查**：安装完成后，在 WSL2 的 Ubuntu 终端里运行 `python -c "import bitsandbytes"`，若没有报错，说明基础导入成功。

---

**9.1 推荐使用WSL2**

你的目标是跑通LoRA微调，建议按这条路径走：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

原因很简单：**Linux环境兼容性更好**。WSL2能给你一个接近Linux的环境，后续遇到环境问题会少很多。

---

**9.2 Windows原生可能问题**

如果你坚持在Windows原生环境装，可能会遇到：

**问题1：bitsandbytes无法加载CUDA动态库**

这不是单一原因造成的，常见的有：

- 你装的`bitsandbytes`版本，和本机CUDA或GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配怎么办？

你在Windows上装环境时，最常遇到的坑就是 **CUDA版本不匹配**。别慌，先跑一条命令看看你的机器到底什么状态：

```bash
nvidia-smi
```

这条命令会告诉你三件事：

- **NVIDIA驱动**是否正常安装；
- 驱动支持的 **CUDA版本**（注意：这是驱动支持的，不是PyTorch用的）；
- 你的显卡型号。

记住一个关键点：**驱动支持的CUDA版本 ≥ PyTorch需要的CUDA版本** 就能用。如果你发现版本对不上，优先检查PyTorch装的是不是匹配的GPU版（见下文5.2节）。

---

## 十、安装完整LoRA环境

假设你已经有了一个Python环境，先装基础库：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**关键前置**：确保你按5.2节安装了**匹配本机CUDA的GPU版PyTorch**，否则后面 `import torch` 或QLoRA的4-bit量化会直接失败。以CUDA 12.1为例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

然后装上LoRA的核心库：

```bash
pip install peft bitsandbytes
```

想一步到位，直接跑这条完整命令：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

装完库后，运行：

```bash
accelerate config
```

按提示选择你的环境。单GPU场景这样选：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

完成后会生成一个 `default_config.yaml` 配置文件，加速训练就靠它。

---

## 十二、LoRA环境验证实验

创建文件 `test_lora_env.py`，写入：

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

看到输出就说明：

- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

再测一下bitsandbytes是否正常。创建文件 `test_qlora_env.py`，写入：

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("4-bit量化配置:", quant_config)
```

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：Windows报错怎么办？**

在Windows上运行LoRA/QLoRA环境时，如果遇到报错，最常见的原因是**兼容性问题**。

**解决方案：使用WSL2**

WSL2（Windows Subsystem for Linux 2）是Windows下的Linux子系统，能有效解决兼容性问题。当你遇到Windows环境下的bitsandbytes加载失败或CUDA相关报错时，切换到WSL2通常是最直接的解决办法。

---

**动手检查：**

打开终端，确认你的环境是否正常。如果当前是Windows环境且遇到报错，可以记录报错信息，然后尝试在WSL2中重新运行同样的命令，对比结果。

---

**小提示：** 在后续的LoRA微调实战中，如果你在Windows上遇到任何环境问题，优先考虑使用WSL2来规避兼容性坑。

## 块 12

**个性化教学正文（LoRA微调 · 动手实践导向 · 快速节奏）**

---

## 一、大模型微调，为什么不能“全量更新”？

你准备微调一个7B模型（70亿参数）。如果走传统路线——更新全部参数再保存新模型——会遇到三个“拦路虎”：

1. **显存爆炸**：训练时不仅要存模型参数 `W`，还要存梯度 `∂W` 和优化器状态（如Adam的动量），实际显存需求远大于模型本身。
2. **训练极慢**：每一轮都要计算和更新几十亿参数的梯度。
3. **存储浪费**：每个微调任务都要保存一份完整的70亿参数副本。

**核心痛点**：我们其实只想要模型“学会新任务”，并不需要它“忘掉旧知识”。全量更新是杀鸡用牛刀。

---

## 二、LoRA的破局思路：只学“变化量”

LoRA的洞察是：**微调时，模型权重的变化量（ΔW）往往是低秩的**——它不需要在70亿维空间里自由移动，而是被压缩在少数几个关键方向上。

于是LoRA不直接更新原始权重 `W`，而是**冻结它**，在旁边训练一个很小的“补丁”：

```
ΔW = (α/r) ×

## 块 13

**动手理解：LoRA 不碰原模型**

你的目标是 LoRA 微调，我们先抓住最核心的直觉：**LoRA 不直接修改原模型参数**。

传统微调是直接改原权重 `W`，把它变成 `W'`。而 LoRA 的做法是：**把 `W` 冻结（不动）**，在旁边新增一个参数 `ΔW`，最终模型用的是：

`W' = W + ΔW`

也就是说，原模型 `W` 保持不变，我们只训练新增的 `ΔW`。

**为什么只训练 ΔW？** 因为大模型已经通过海量数据学会了语言规律、世界知识和推理能力。微调通常只是对已有能力做“方向调整”，所以不必重新学全部参数，只学任务相关的变化量往往就够了。

> 注意：这是一种教学直觉，并非严格结论。能否用 LoRA 替代全参数微调，取决于任务与预训练能力的差距、数据规模、目标模块与秩的设置等。

**本块学习检查**：请用一句话向自己解释：LoRA 微

## 块 14

你正在学习LoRA微调，目标很明确：搞懂大模型参数高效微调的核心思想。这一块，我们从最基础的矩阵概念讲起，为后面的低秩分解原理铺路。

**3.1 矩阵基础概念**

Transformer模型里，大量的参数都是以矩阵形式存在的。比如注意力层中，`Q = X · Wq` 这个公式（数学示意，非运行代码），其中 `Wq` 就是一个权重矩阵。

- 假设 `Wq` 的形状是 `4096 × 4096`，它的参数数量就是 `4096 × 4096 ≈ 1677万`。

**3.2 直接训练矩阵的问题**

如果你想微调模型，直接去修改这个完整的权重矩阵 `W`，那么你需要更新的参数量同样是 `4096 × 4096`，也就是1677万个。这在微调时计算量和存储开销都很大

## 块 15

**4.1 矩阵尺寸变化（动手算一遍）**

假设原矩阵 W 是 **4096 × 4096**。

如果直接训练它，参数量就是：
4096 × 4096 = **1677万**（16,777,216）

这个数字太大，训练起来很慢、很吃显存。

---

**LoRA 的做法：**

设 **rank r = 8**，把 W 拆成两个小矩阵：

- **A**：4096 × 8 → 参数量 = 4096 × 8 = **32,768**
- **B**：8 × 4096 → 参数量 = 8 × 4096 = **32,768**

两者相加：32,768 + 32

## 块 16

**本块学习检查**：读完本块后，请用自己的话回答：LoRA为什么能用低秩矩阵近似权重更新？答案需包含“低维子空间”和“内在维度”两个词。

---

**5.1 神经网络参数变化具有低维特征**

LoRA论文提出：模型适应新任务时，参数变化 ΔW 并不需要覆盖所有方向。实际上，有效变化集中在一个**低维子空间**里。

举个例子：一个4096维的空间，真正影响任务的“有效方向”可能远少于满维度，集中在一个远小于原维度的少数方向上。

> 注意：这是基于“权重更新往往具有较低的内在维度”这一假设。具体“有效维度”没有固定值，会随模型、任务、数据变化；这里只做定性理解，别

## 块 17

## 类比理解：为什么LoRA能用少量参数描述大模型的变化？

想象一张高清图片，尺寸是4000×4000像素。如果要把这张图片完整保存下来，需要存储1600万个数据点——这就像大模型的完整参数，体量巨大。

但关键洞察在于：如果这张图片的主要变化其实很有限——只是亮度调了调、对比度变了变、再加上少量结构上的小改动——那么，我们根本不需要保存全部1600万个像素来记录这些变化。用一小部分参数，就足以描述“图片发生了什么改变”。

LoRA的核心思想正是如此：**用少量参数来描述模型的变化方向**。

在微调大模型时，我们并不需要重新训练和存储所有参数。我们只需要找到“模型需要往哪个方向变化”，然后用一个低秩矩阵（少量参数）来捕捉这个变化方向。这就像只记录“亮度+10、对比度-5”这几个关键变化，而不是重新保存整张图片。

需要特别明确的是：**LoRA冻结原始权重W，仅训练低秩矩阵A和B，通过增量ΔW=(α/r)BA更新模型，而非直接分解或替换原权重**。也就是说，原始权重W保持不变，模型的实际更新量为ΔW，最终生效的权重为W+ΔW。

---

## LoRA在Transformer中的位置

Transformer的主要结构是：

```text
Input → Embedding → Attention → FFN → Output
```

其中Attention模块包含四个投影矩阵：**Q、K、V、O**。

LoRA在早期实践中，一个**常见**的做法是只把LoRA加到Q投影和V投影上。代码示例如下：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示：只训练Wq（Q的权重矩阵）的变化和Wv（V的权重矩阵）的变化。

> ⚠️ **重要说明**：只挑`q_proj`和`v_proj`是LoRA原论文中的一种常见选择，**并不是固定标准**。现代实践中，LoRA也经常被加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用`target_modules="all-linear"`覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，需要结合模型结构（先确认实际模块名）与验证集表现来定。

下面是一个覆盖更完整线性层的配置示例（以Llama类模型常见的模块名为例；不同模型模块名可能不同，请先用`print(model)`或`model.named_modules()`确认实际名称）：

```python
from peft import LoraConfig

# 示例：把LoRA同时加到注意力的q/k/v/o与FFN的gate/up/down线性层

## 块 18

## 8.1 rank（r）——从“调参”到“调明白”

你学 LoRA 是为了真正上手微调，那 `r` 就是你第一个要亲手摸透的旋钮。先看代码里它长什么样：

```python
r=8
```

**它代表什么？** 一句话：**低秩矩阵的维度**。你可以把它理解为“微调时给模型开的‘记忆通道’宽度”——通道越宽，能塞下的新知识越多，但占用的资源也越大。

**怎么权衡？** 记住这个核心矛盾：

- **r 越大** → 表达能力越强（能学更复杂的任务），但参数更多、显存占用更高。
- **r 越小** → 省显存、训练快，但表达能力受限。

**新手别纠结“标准答案”，用这张表当起点：**

| 任务类型 | rank（起点参考） |
| ---- | ---- |
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 32 |

注意，这只是**搜索起点**，不是定死的规则。`r` 的合适

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

在代码里你会看到类似这样的设置：

```python
lora_alpha=16
```

这个参数的作用是**控制LoRA更新强度**——也就是微调时，LoRA对模型权重的影响有多大。

实际使用中，`alpha/r` 会作为缩放比例。比如 `r=8`、`alpha=16` 时，缩放比例就是 `16/8=2`。这个比例越大，LoRA对模型的改动越明显；比例越小，改动越温和。

**动手理解**：你可以把 `alpha` 想象成“音量旋钮”，`r` 是“音源大小”，`alpha/r` 就是最终播放出来的音量。调大 `alpha`，LoRA 的“声音”就更响。

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 的作用是**防止LoRA参数过拟合**。训练时随机丢弃一部分LoRA参数（这里是5%），让模型不能过度依赖某几个参数，从而提升泛化能力。

---

## 九、LoRA与全参数微调区别

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> **注意**：上表是**相对趋势**，不是绝对数值。LoRA

## 块 20

## 2.1 Rank 表示什么？

Rank 这个词，中文意思是**矩阵的秩**。但在 LoRA 里，你可以先把它理解成一个“**中间通道的粗细**”——它决定了 LoRA 在改动模型时，内部“传递信息”的宽度。

我们直接看例子，这样最直观。

假设原始权重矩阵 `W` 的尺寸是 **4096 × 4096**（这是一个很大的方阵）。

如果设置 `r=8`，那么 LoRA 会拆出两个小矩阵：

- 矩阵 `A`：尺寸是 **8 × 4096**
- 矩阵 `B`：尺寸是 **4096 × 8**

注意看，这两个矩阵的“中间”维度都是 **8**——这个数字就是 rank 的值。

然后，LoRA 通过 `B × A` 把两个小矩阵相乘，最终恢复出一个 **4096 × 4096** 的矩阵，这就是 `ΔW`（权重的更新量）。

**一句话总结：rank 就是低秩矩阵中间那个“瓶颈”的宽度。** 它越小，中间能“挤过去”的信息就越少；它越大，能传递的信息就越多，但训练成本（显存和计算量）也会跟着涨。

---

**本块学习检查**（动手验证一下）：

打开 Python 环境，运行下面这段代码，观察输出结果：

```python
import torch

r = 8
A = torch.randn(r, 4096)   # 模拟矩阵 A
B = torch.randn(4096, r)   # 模拟矩阵 B

delta_W = B @ A            # 模拟 ΔW = B × A
print(delta_W.shape)       # 看看恢复出来的形状是多少？
```

如果你看到输出是 `torch.Size([4096, 4096])`，就说明你已经理解了 rank 如何控制中间维度、并最终还原出原始尺寸的权重更新量。

## 块 21

### 2.2 Rank控制什么？

**一句话理解**：Rank（r）决定LoRA能学习多少种变化方向。

**低rank（如 `r=4`）**：模型只能学较少变化模式。  
✅ 参数少、显存低、训练快  
❌ 表达能力有限

**高rank（如 `r=64`）**：LoRA自由度更高。  
✅ 能学更复杂任务、适应力强  
❌ 参数多、显存增、训练成本高

**实操对照**：你微调时，若任务简单（如风格迁移），选低r省资源；任务复杂（如多意图对话），选高r保效果。

**本块学习检查**：  
设某

## 块 22

**3.1 不使用LoRA：全参数训练**

先看完整训练要动多少参数。假设模型某层是4096×4096的矩阵，直接算：

```text
4096 × 4096 = 16777216
```

约**1677万参数**。这是你要全部更新、存储的量。

---

**3.2 使用LoRA：只学一个小矩阵**

LoRA不碰原矩阵，只训练一个低秩分解。公式：

```text
参数量 = r × (输入维度 + 输出维度)
```

- **r=8**：`8 × (4096+4096) = 65536`，约**6.5万**
- **r=64**：`64 × (409

## 块 23

## 4.1 Rank太小的问题

想象你让LoRA用极少的“记忆空间”去学新任务——比如设 `r=2`。这就像给一个学生只发两张便签纸去记整本书的笔记，结果自然是：

- **学习能力不足**：能记下的信息太少；
- **欠拟合**：模型根本没“吃透”训练数据，学不到位；
- **新任务知识无法充分表达**：想学的技能装不进这么小的容器里。

**你会看到的现象**：
- 训练时loss下降得很慢，半天不见进步；
- 微调完效果提升有限，甚至感觉跟没调差不多。

**怎么判断自己踩了这个坑？** 如果你训练了好几轮，loss曲线还是平缓得像条直线，而且验证集上的表现始终上不去，大概率就是rank设太小了。这时候别急着调学习率，先试试把 `r` 往上提一档，比如从2改成8，观察loss是否开始明显下降。

---

## 4.2 Rank太大的问题

反过来，如果你把rank设得很大，比如 `r=256`，又会遇到另一组麻烦：

- **参数增加**：要训练的东西变多了；
- **显存增加**：GPU内存吃紧，可能直接OOM；
- **训练速度下降**：每一步都更慢。

更关键的是——**LoRA的核心优势被削弱了**。

你可能会想：“rank大一点，不是更接近完整模型吗

## 块 24

### 4.3 常见rank选择：从哪开始？

你是动手派，目标明确——LoRA微调。记住，下面只是**经验起始范围**，不是自动调参标准，别照搬：

| 任务类型 | rank（起点参考） |
| ------ | ------ |
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**实际项目**要根据数据量、模型大小、序列长度、batch大小（含梯度累积后的有效batch）、目标模块（`target_modules`）选择与数量、GPU显存、验证集表现来调。同一个`r`在不同模型规模、序列长度/batch、目标模块下，显存和效果都可能差别很大。

---

### 五、Alpha（lora_alpha）是什么？

LoRA除了`Δ

## 块 25

### 6.1 控制LoRA影响强度——用`α/r`这个“音量旋钮”调强度

先看一个具体例子，帮你建立直觉。假设你设了：

```python
r=8
alpha=16
```

那么缩放系数就是：

```text
16/8=2
```

意思是：LoRA对模型的更新会被**放大2倍**。你可以把`α/r`想成一个“音量旋钮”——值大于1，更新力度增强；小于1，力度减弱。

再对比两个情况：

- 如果`alpha=8`，那么`8/8=1`，更新保持**原始比例**，不放大也不缩小。
- 如果`alpha=4`，那么`4/8=0.5`，LoRA的影响**减半**，更新力度变弱。

**核心公式**：LoRA加入模型的更新量会被`α/r`这个系数缩放，所以`α/r`是控制LoRA影响强度的**关键缩放系数**。

---

### 七、Alpha与Rank的关系（重点）

**一句话记住**：`α/r`是缩放系数，但不是“唯一决定实际强度的开关”。

为什么这么说？因为实际落到权重上的更新量，还取决于：

- **学习率**：它直接决定每步对`A`、`B`的更新步长。学习率和`α/r`会**叠加**影响有效更新幅度——比如你把`α/r`调大又不下调学习率，容易一起放大更新导致震荡。实践中很多人更习惯调学习率来控制强度，而不是频繁改`α`。
- **初始化**：常见实现里`A`用小随机值、`B`

## 块 26

**本块学习检查**：请动手改代码，把 `r` 从 8 改成 16，同时把 `alpha` 从 16 改成 32，然后算一下 `alpha / r` 的值，确认它是否还是 2。

---

先看**配置1**：`r=8`，`alpha=16`，比例是 `16 / 8 = 2`。

再看**配置2**：`r=16`，`alpha=32`，比例是 `32 / 16 = 2`。

发现了吗？虽然 `rank` 变了（从 8 变 16），但**缩放比例 `alpha / r` 始终保持为 2**。

这就是关键点：**调整 rank

## 块 27

**方案1：小模型实验（你的起点）**

先别急着上大参数。对于初次跑LoRA，用最小配置最稳妥：

```python
r=8
alpha=16
```

- **参数少**——模型改动小，不容易出错；
- **训练快**——适合快速验证流程是否跑通；
- **适合测试**——先看效果，再决定要不要加码。

**动手建议**：先用这个配置跑通一次完整训练，记录loss下降情况。如果效果不理想，再升级到方案2（`r=16, alpha=32`）或方案3（`r=64, alpha=128`）。

**记住两个核心概念**：
- `r` 控制 **LoRA容量**（能学多少）；
- `lora_alpha` 控制 **LoRA更新幅度**（每次学多猛）。

**本块学习检查**：把`r=8, alpha=16`写进`LoraConfig`，并设置`target_modules=["q_proj", "v_proj"]`，

## 块 28

## 误区1：rank越大一定越好？

**错。** 这是最常见的一个直觉误区——觉得“容量越大，能力越强”。但 LoRA 不是这么玩的。

**为什么错？** 如果你的任务本身很简单，比如只做特定领域的风格适配，那么一个大 rank 并不会带来额外收益，反而会：

- **浪费显存**：更大的 rank 意味着更多可训练参数，占用更多显存；
- **增加训练成本**：训练更慢，而效果并不会因此变得更好。

**记住一句话：rank 是容量，不是性能。** 容量够用就好，多了是负担。

---

## 误区2：只调 rank，不调 alpha

很多人改参数时只动 rank，alpha 保持原样。来看一个例子：

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
16/64 = 0.25
```

这意味着什么？**LoRA 的更新强度被稀释了。** 原来 alpha/r = 2，现在变成 0.25，同样的训练步数下，模型实际被调整的幅度大幅降低。你虽然增大了容量，但更新力度反而变小了——**两个参数是配套的，不能只动一个。**

---

## 误区3：alpha 越大越好？

**同样错。** alpha 的作用是通过 `α/r` 这个缩放比来控制 LoRA 更新的强度。如果 alpha 过大，即 `α/r` 过大，会导致：

- **更新过强**：每一步对权重的改动太大；
- **训练不稳定**：loss 震荡甚至发散。

**但注意**：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡还常常来自学习率过高、数据质量差、batch size 过小、精度设置（fp16/bf16）问题、缺少梯度裁剪等。排查不稳定时，要综合这些因素，别只盯着 alpha。

---

## 关于电脑配置

本章是**概念讲解**，不需要 GPU

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。在 LoRA 训练流程中，`LoraConfig` 是核心配置对象，负责告诉 PEFT 四件事：LoRA 放在哪里、规模多大、训练强度如何、是否使用 dropout。

**LoRA 训练流程**：

```text
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

你可以把 `LoraConfig` 理解为一份“施工图纸”——它不直接参与训练，但决定了 LoRA 适配器（Adapter）如何被构建和训练。

---

### 动手理解：先跑通一个最小配置

既然你喜欢动手，建议你现在打开 Python 环境，执行下面这段代码（不需要 GPU，CPU 即可）：

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

你会看到打印出的配置对象，里面包含了刚才设置的每个参数。先记住这个输出结果，后续几节我们会逐一拆解每个参数的含义，并让你修改它们来观察变化。

> **小提示**：如果你还没安装 PEFT，先执行 `pip install peft` 完成环境准备，再运行上面的代码。

---

### 本节要点回顾

- `LoraConfig` 是 PEFT 中创建 LoRA 配置的入口对象；
- 它控制 LoRA 的**插入位置**（target_modules）、**规模**（r）、**训练强度**（lora_alpha）和**正则化**（lora_dropout）；
- 创建配置后，后续会用它生成实际的 LoRA Adapter。

下一节，我们将深入 `target_modules`，看看如何

## 块 30

## 1.2 一个简单示例——动手跑通第一个LoRA配置

先别急着理解每个参数背后的原理，我们直接看代码。下面这段就是LoRA最核心的配置，你把它复制到自己的环境里跑一遍，感受一下：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,                # LoRA矩阵的大小
    lora_alpha=16,      # 缩放强度
    target_modules=["q_proj", "v_proj"],  # 插入哪些层
    lora_dropout=0.05   # 随机失活比例
)
```

跑完这段代码，你已经完成了LoRA配置的**第一步**。这四个参数就是LoRA的全部核心，记住它们的作用：

| 参数 | 作用 |
|---|---|
| target_modules | 决定在模型的哪些层插入LoRA |
| r | 决定LoRA矩阵的尺寸（越小越省显存） |
| lora_alpha | 控制LoRA对原模型的影响强度 |
| lora_dropout | 防止过拟合的随机失活比例 |

---

## 二、LoRA配置整体流程——三步走

配置只是第一步，完整的LoRA微调流程是这样的：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

把配置

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入”**。

LoRA 不会默认修改所有层，你需要手动指定。在 Transformer 模型中，最常见的注入对象是 Attention 模块里的四个投影层：

```text
Attention
├── q_proj
├── k_proj
├── v_proj
└── o_proj
```

比如你想给 Query 投影层和 Value 投影层加 LoRA，就写成：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

所以调整 Q 和 V，通常就能有效改变模型的行为——这就是它俩成为常见起点的原因。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名可能不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 `target_modules`。

---

## 偷懒技巧：`target_modules="all-linear"`

如果你不想一个个枚举模块名，PEFT 较新版本（一般 ≥ 0.7，具体以你装的版本文档为准）支持直接写：

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
    target_modules="all-linear",   # 自动匹配所有 nn.Linear 层，包括 lm_head（若为 nn.Linear）
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**注意权衡**：`"all-linear"` 覆盖的模块变多，可训练参数和显存占用也会增加。是否用全线性层，要在**参数量/显存**和**效果**之间取舍，最终以验证集结果为准。如果你的 PEFT 版本太旧不支持这个写法，就退回显式列出模块名，或者升级 PEFT。

---

## 动手验证（建议实操）

跑完上面的代码后，观察 `print_trainable_parameters()` 的输出，确认：
1. 可训练参数量是否明显小于全量微调？
2. 如果换成 `target_modules=["q_proj", "v_proj"]`，参数量有什么变化

> 📄 **模型许可证**：本文示例使用

## 块 32

### 方案1：只调整Q和V

这是最轻量的起点，适合你快速跑通LoRA微调流程。

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么选它？**  
- **参数少**：只训练注意力中的查询（Q）和值（V）投影层，显存占用低。  
- **训练快**：参数量小，迭代速度明显提升，适合1.5小时/天的碎片化学习节奏。  
- **常见起点**：很多任务用它做基线，再根据效果决定是否扩展。

**动手验证**：  
先按此配置跑一次微调，记录训练时间和验证集指标。如果效果不理想，再尝试方案2。

---

### 方案2：调整全部Attention

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

**特点**：  
- 参数

## 块 33

**LLaMA 系列：从零开始配 LoRA（动手版）**

你的目标是跑通 LoRA 微调，LLaMA 系列是最常见的起点。这里只需要记住一个核心动作：**告诉 PEFT 你要改哪些层**。

LLaMA 的注意力层里，查询（Q）和值（V）投影是分开的，名字分别叫 `q_proj` 和 `v_proj`。这是社区验证过的最稳妥组合，直接照抄即可：

```python
target_modules=["q_proj", "v_proj"]
```

**为什么只改这两个？** 简单理解：LoRA 是给模型“打补丁”，Q 和 V 是补丁收益最高的位置。你不需要动其他层，先跑通再说。

**动手验证（1 分钟）**：打开你的训练脚本，找到 `LoraConfig`，把 `target_modules` 改成上面这行。如果模型是 LLaMA 架构（如 `meta-llama/Llama-2-7b-hf`），这个配置直接可用。

```python
# 注意：Llama-2-7b FP16 需要至少14GB显存。
```

---

**对比：GPT-2 的坑（遇到再回看）**

如果你换用 GPT-2，`target_modules` 要写成 `["c_attn"]`——它把 Q/K/V 合并成了一个模块。但这里有个**关键陷阱**：GPT-2 用的是 `Conv1D` 层，权重形状是 `(in, out)`，和常规 `nn.Linear` 的 `(out, in)` 正好相反。

所以，给 GPT-2 加 LoRA 时必须额外加一个参数：

```python
fan_in_fan_out=True
```

否则低秩矩阵的初始化方向与 Conv1D 不匹配，可能导致训练效果不佳或收敛缓慢。完整示例：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,

## 块 34

## BERT的target_modules配置：动手查，别照抄

你正在学LoRA微调，目标很明确——把大模型调成自己想要的“形状”。但配置`target_modules`时，最容易踩的坑就是“照搬别人代码”。BERT和LLaMA的模块名不一样，直接复制会报错或根本没生效。

**核心思路：用代码查，而不是猜。**

BERT的注意力层包含`query`、`key`、`value`三个投影，LoRA通常只对`query`和`value`进行低秩适配，因此`target_modules`常设为：

```python
["query", "value"]
```

但注意：**不能直接复制所有模型的配置**。不同模型的内部命名差异很大。

**正确做法**：加载模型后，打印所有模块名，亲眼确认：

```python
# 注意：Llama-2-7b FP16 需要至少14GB显存
for name, _ in model.named_modules():
    print(name)
```

在输出中找类似`q_proj`、`v_proj`这样的名字（BERT里可能是`query`、`value`），然后照实际名字配置。

**匹配机制补充**：
- `target_modules`按模块名做**后缀匹配**——写`"q_proj"

## 块 35

**r是什么？——一个控制“压缩程度”的旋钮**

想象你有一个巨大的矩阵（比如4096×4096），直接训练它太费显存。LoRA的做法是：不直接改这个大矩阵，而是训练两个小矩阵A和B，让它们的乘积近似等于原矩阵的变化量（ΔW = BA）。

这里的 **r（rank）**，就是决定A和B“中间维度”大小的关键数字。r越小，A和B越“瘦”，参数越少；r越大，A和B越“胖”，参数越多。

**动手感受一下：**
- 原矩阵是4096×4096。
- 如果你设 `r=8`，那么：
  - A的形状是 `8×4096`
  - B的形状是 `4096×8`
-

## 块 36

**本块学习检查**：  
配置 `r=8, lora_alpha=16`，写出 LoRA 的最终更新公式，并说明缩放值是多少。

---

**7.1 alpha 作用**  
LoRA 的最终更新不是 `BA`，而是 `(alpha/r)BA`。  
所以 `alpha` 控制的是 **LoRA 更新幅度**。

**动手理解**：  
设 `r=8, lora_alpha=16`，则缩放为 `16/8=2`。  
这意味着 LoRA 的影响被**增强**了——更新幅度翻倍。

---

**7.2 alpha 常见设置**  
一个常见起点是 `alpha = 2 × r`（注意：这只是经验值，非硬性规则，最终以验证集表现定）。  
例如：

## 块 37

## 8.1 dropout作用：防止LoRA“死记硬背”

训练时，LoRA会**随机关闭一部分神经连接**。这就像做练习题时故意跳过几道，逼自己真正理解规律，而不是背答案——目的是**减少过拟合**（模型只记住训练数据，遇到新数据就懵）。

配置方法很简单：
```python
lora_dropout=0.05
```
这表示**5%的LoRA路径会被随机失活**。

## 8.2 怎么选dropout值？

下面给的是**经验起点**，不是死规则，最终要看验证集效果来调：

| 数据规模 | dropout（起点参考） |
|---------|------------------|
| 数据少   |

## 块 38

### 9.1 基础配置

这段配置适合在7B模型上做LoRA微调测试，我们直接看代码：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：秩的大小，决定新增参数量
    lora_alpha=16,      # 更新缩放：控制LoRA对原权重的影响强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的q和v投影
    lora_dropout=0.05,  # 防止过拟合：训练时随机丢弃5%的LoRA参数
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（如GPT类）
)
```

**参数速查表**

| 参数 | 值 | 作用 |
|---|---|---|
| r | 8 | LoRA容量 |
| alpha | 16 | 更新缩放 |
| target_modules | q/v | 插入位置 |
| dropout | 0.05 | 防止过拟合 |
| task_type | CAUSAL_LM | 语言模型任务 |

**动手试一下**：把`r`改成4，`lora_alpha`改成8，观察

## 块 39

## 九、完整LoraConfig案例（重点）——动手跑通一个分类任务

你现在的目标是LoRA微调，而且喜欢动手实践。下面这个例子把前面学的参数全部串起来，完成一个**带分类头的序列分类任务**。请跟着代码一步步敲，边敲边看注释。

### 9.1 基础配置：加载模型 + 设置LoRA

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：BERT 使用 Apache 2.0 许可证，可商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练需根据 batch size 适当增加
# 1) 加载带分类头的基础模型（分类头是新初始化的，需要一起训练/保存）
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",                 # 序列分类任务
    target_modules=["query", "value"],   # BERT 注意力模块名
    modules_to_save=["classifier"],       # 关键：把新分类头也纳入保存
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

**动手观察点**：运行 `print_trainable_parameters()`，你会看到可训练参数包含两部分——LoRA注入的低秩矩阵，以及 `classifier` 分类头。这就是 `modules_to_save` 的作用。

### 为什么必须加 `modules_to_save=["classifier"]`？

这是本块最容易踩的坑。BERT基础模型本身**没有分类头**，你加载时用 `num_labels=2` 新初始化了一个。如果不把它加入 `modules_to_save`：

- 训练时它**仍会参与更新**（PEFT默认训练所有需要梯度的模块）
- 保存时它**不会被保存**
- 重载时它**退回随机初始化** → 分类结果完全错乱

### 保存与重载验证

```python
# ...（此处进行训练，略）...

# 2) 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")

# 3) 重新加载并验证分类头确实被带回（而不是随机初始化）
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证：重载后的分类头权重应与保存前一致（说明 classifier 被 modules_to_save 正确保存/恢复）
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

**动手验证任务**：运行最后两行，如果输出 `True`，说明分类头被正确保存和恢复。你可以试着把 `modules_to_save` 删掉再跑一遍，对比输出——这就是"踩坑"的最佳方式。

### 三个必须记住的要点

1. **保存范围**：`save_pretrained` 保存的是 **Adapter + `modules_to_save` 模块**，不是完整基础模型

## 块 40

### 10.1 普通文本生成：LoRA参数怎么配？

你现在要微调一个普通文本生成模型（比如写故事、续写文本），这是LoRA最基础的场景。配置很简单，直接照抄这组参数就行：

```python
r=8
alpha=16
dropout=0.05
```

**这三个参数分别是什么意思？**

- **`r`（秩）**：决定LoRA引入的可训练参数规模。`r=8`表示低秩增量矩阵的秩为8，即训练两个低秩矩阵A（d×r）和B（r×d），其乘积BA构成对权重更新的低秩增量ΔW=(α/r)BA，而原始权重W保持冻结。这属于中等偏小的规模，适合普通文本生成这种“不需要大幅改变模型行为”的任务。
- **`alpha`（缩放系数）**：控制LoRA更新的强度。`alpha=16`是`r`的两倍，这是PEFT库常用的默认比例（`alpha = 2 * r`），能让更新幅度既明显又稳定。
- **`dropout`（丢弃率）**：防止过拟合。`0.05`表示在LoRA层随机丢弃5%的神经元，这个值比较保守，适合数据量不大时使用。

**为什么普通文本生成推荐这套参数？** 因为这类任务通常希望模型保持原有的语言能力，只做小幅调整。`r=8`不会引入太多额外参数，训练快、显存占用低；`alpha=16`保证更新有效但不激进；`dropout=0.05`提供基础的正则化保护。

---

### 动手验证一下

**本块学习检查**（不引入新知识，只验证你理解了）：

1. 打开你的Python环境，导入`LoraConfig`，用上面三个参数创建一个配置对象，打印出来看看`r`、`lora_alpha`、`lora_dropout`是否和预期一致。
2. 把`r`改成`16`、`alpha`改成`32`，对比两次配置输出，观察`alpha`和`r`的比例关系是否保持为2:1。

---

### 补充：你的电脑能跑吗？

如果只是**配置LoraConfig**（不加载模型），最低要求很低：**CPU即可，内存4GB以上**。

但如果要

## 块 41

**动手配置：普通文本生成的LoRA参数**

先看这段核心代码，它是你微调文本生成模型的起点：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**逐参数理解（按你的学习节奏，快速过）：**

- **`target_modules`**：决定改哪里。这里只改注意力层的 `q_proj` 和 `v_proj`，其他层保持不动。
- **`r=8`**：决定LoRA容量。数值越小，可学习的参数越少，模型越“轻”。
- **`lora_alpha=16`**

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思路：冻结原模型，只训练少量新增参数。但这里有个关键点——**LoRA只是减少了“训练”的参数，并没有减少“加载”的参数**。

用图来表示就是：

```text
原模型参数：冻结（但必须完整加载到显存）
LoRA参数：训练（只占很小一部分）
```

也就是说，无论LoRA多轻量，**基础模型本身仍然需要完整放进显存**。

我们来算一笔账。以7B模型为例，如果用FP16（每个参数占2字节）加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这还没完。实际训练时，显存里还要同时放：

- **激活值**：前向传播的中间结果
- **梯度**：反向传播计算的更新量
- **优化器状态**：如Adam的动量项

把这些都加起来，实际显存需求会**远高于14GB**。这就是为什么即便有LoRA，普通消费级显卡依然跑不动7B

## 块 43

**QLoRA核心思想：用4-bit模型跑LoRA**

你的目标是LoRA微调，QLoRA就是它的“省显存版”。核心区别一句话：

- **LoRA**：FP16模型 + LoRA
- **QLoRA**：4-bit模型 + LoRA

**流程**：先把FP16基础模型量化成4-bit，加载它，再插入LoRA，最后只训练LoRA参数。

**关键点**：
- 量化的是**基础模型权重**（冻结不动）
- LoRA参数保持FP16/BF16，**正常训练**

**动手验证**：打开你的HuggingFace环境，对比加载同一个模型时，FP16和4-bit两种方式占用的显存差异——这就是QL

## 块 44

## 3.1 模型存储方式

先记住这张表：**FP32每个参数占4字节，FP16占2字节，INT8占1字节，INT4只占0.5字节**。

以7B模型为例：
- 用FP16存：`7B × 2 ≈ 14GB`
- 用4-bit存：`7B × 0.5 ≈ 3.5GB`

显存直接减少约75%。但注意：**3.5GB只是理论下界**。实际占用会更高，因为还要算上量化常数、分组元数据（NF4按组量化，每组都要额外存参数）、未量化的模块（如部分norm、embedding、lm_head、LoRA参数），以及训练时的缓存和激活值。真实占用以实际运行为准。

## 3.2 为什么不用普通INT4？

传统INT4整数范围是`-8 ~ 7`。但神经网络权重通常**接近正态分布、小数多、分布中心接近0**——直接整数映射会严重损失精度。所以QLoRA提出了**NF4**。

---

## 四、NF4量化原理（重点）

NF4（NormalFloat4）是专门为近似正态分布的权重设计的4-bit量化格式。它

## 块 45

### 4.1 NF4是什么？

NF4（NormalFloat4）就是“正态分布4位浮点”。别被名字骗了——它不是IEEE那种通用浮点格式，而是QLoRA论文为**神经网络权重**专门设计的**4-bit量化码本**。

**核心思路**：权重值大多集中在0附近，近似正态分布。NF4预先定好16个（2⁴）量化点，这些点按标准正态分布的分位数排布——0附近点更密，代表权重密集区划分更细。量化时，把每组权重归一化后映射到最近的码本点，只存4-bit索引；反量化时用码本和缩放因子还原。

**一句话**：它是为“近似正态分布”量身定制的离散码本，不是通用浮点格式。

---

**动手验证

## 块 46

**为什么NF4适合模型权重？**

大模型训练完的权重，大致像**正态分布**——大量参数挤在0附近，越往两边越少。

普通INT4是**平均划分**，比如[-8,-7,...,7]，每段一样宽。但权重都堆在中心，这种均匀切法在0附近“分辨率”不够，精度损失大。

NF4则**按正态分布设计量化范围**：0附近分得更细、精度更高，两边分得粗。这样更贴合真实权重分布，损失更低。

**量化过程**：FP16权重（如0.023、-0.145、0.876）→ 归一化 → 映射到NF4 → 存4-bit值。推理时反量化回近似FP16。

## 块 47

**本块学习检查**：读完5.1-5.3后，请用自己的话回答：双重量化“第二次量化”的对象是什么？它能帮我们省下什么资源？

---

**5.1 为什么需要双重量化？**

做4-bit量化时，除了权重本身，每组权重还要额外保存两个参数：**scale**（缩放因子）和**zero point**（零点）。这些“量化参数”本身也占显存，不能忽略。

**5.2 双重量化思想**

普通量化：FP16权重 → 4-bit权重 + scale参数。

双重量化：在普通量化基础上，**把scale参数再量化一次**。即第一次量化模型权重，第二次量化量化参数本身。

**5.3

## 块 48

**为什么需要分页优化器？——别让显存“瞬间爆炸”**

训练时显存峰值不只来自模型，还有**梯度**和**优化器状态**。比如Adam优化器要额外保存**一阶动量**和**二阶动量**。

QLoRA虽然大幅降低了模型占用的显存，但训练过程中仍可能遇到**显存突然不足**（OOM）的情况。

**分页优化器的思路**：借鉴操作系统虚拟内存，借助CUDA的pinned memory（页锁定内存）和异步内存拷贝，在显存出现峰值、即将OOM时，把优化器状态自动“分页”换出到CPU内存，需要时再换回GPU。

流程：GPU显存 → 出现峰值（接近/超出上限） → 优化器状态按页换出到CPU内存 → 需要计算时再换回GPU。

**注意**：这不是“平时就把部分状态放CPU”，而是**应对显存峰值**——

## 块 49

## 6.3 类比理解：分页优化器在做什么？

把显存想象成你的**桌面**，把CPU内存想象成**柜子**。

**普通方式**：所有东西（优化器状态）都堆在桌面上。桌面满了，就没法继续干活了——这就是**峰值OOM**。

**分页优化器**：桌面只放**常用文件**（当前需要的优化器状态），暂时不用的放进**柜子**（CPU内存），需要时再取出来。这样桌面永远有空间，训练就不会因为显存爆掉而中断。

---

## 6.4 代码里怎么启用？

不用自己写“换页”逻辑，bitsandbytes库已经实现了。你只需要在 `TrainingArguments` 里通过 `optim` 参数指定：

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
    optim="paged_adamw_8bit",        # 关键：使用 bitsandbytes 的分页优化器
)
```

**两个常用取值**：
- `paged_adamw_8bit`：优化器状态用8-bit存储，更省显存
- `paged_adamw_32bit`：精度更高，但占用更多显存

**注意**：不同版本的Transformers/bitsandbytes支持的`optim`取值可能不同（还有`adamw_bnb_8bit`等）。如果报“未知optim”错误，去查你当前版本的文档，换用支持的名称即可。

**实战建议**：分页优化器主要解决**峰值OOM**，通常搭配`gradient_checkpointing=True`、小`per_device_train_batch_size`+梯度累积一起用，把显存峰值压下来。

---

**本块学习检查**：打开你正在用的Transformers版本，在Python里执行`from transformers import TrainingArguments; print(TrainingArguments.__doc__)`，找到`optim`

## 块 50

## 7.1 4-bit量化：给模型“瘦身”

你的目标是LoRA微调，但显存不够？QLoRA的核心思路就是：**先把大模型“压缩”了再微调**。

**量化**就是把模型权重从FP16（16位）变成NF4 4-bit格式。简单说，每个权重参数占用的空间变小了，整个模型文件也就更小，能塞进显存更小的GPU里。

**关键点**：NF4是QLoRA论文推荐、也是本教程采用的常见4-bit量化类型。bitsandbytes也支持`fp4`等其它类型，但NF4是默认选择。

---

## 7.2 双重量化：连“尺子”也压缩

量化需要一组`scale`参数来记录缩放比例。双重量化就是**把这组scale参数再做一次量化**，进一步减少额外开销。

**直观理解**：第一次量化是“压缩模型”，第二次量化是“压缩压缩模型时用的工具”。

---

## 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态也占显存。分页优化器让GPU显存不够用时，**自动把

## 块 51

## 4-bit量化配置：让你的LoRA微调跑得更省显存

你已经知道LoRA能大幅减少可训练参数，但模型本身加载进显存依然很“吃”内存。QLoRA的秘诀就是——**先把模型压成4-bit存起来，用的时候再临时“解压”计算**。下面我们逐个看配置项。

### 第一步：开启4-bit加载

```python
load_in_4bit=True
```

这行代码告诉bitsandbytes：把模型权重以4-bit精度加载进显存。相比16-bit加载，显存占用降至约1/4，这是QLoRA能在大模型上跑起来的关键。

### 第二步：选择量化类型——NF4

```python
bnb_4bit_quant_type="nf4"
```

NF4（NormalFloat4）是一种专门为神经网络权重设计的4-bit量化格式。它比普通的4-bit整数量化更聪明——会根据权重的分布来分配数值精度，让重要的数值区间保留更多细节。简单说：**同样占4-bit，NF4能保留更多有效信息**。

### 第三步：开启双重量化

```python
bnb_4bit_use_double_quant=True
```

双重量化是“省上加省”的技巧：不仅量化权重本身，连量化过程中产生的“缩放因子”也再做一次量化。这样能再节省约0.4 bits/parameter的额外内存开销。建议直接开启，白捡的优化。

### 第四步：设置计算精度（最容易踩坑的地方）

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

这里要区分两个概念：

- **存储dtype**：权重以4-bit（NF4）存着，省显存。
- **计算dtype**：实际做矩阵运算时，4-bit权重会被**反量化**成更高精度来算，这个精度就是`bnb_4bit_compute_dtype`。

**为什么不直接用4-bit算？** 因为目前GPU对4-bit运算支持不成熟，强行算会又慢又不稳定。所以策略是：**存的时候用4-bit省空间，算的时候临时“解压”成16-bit保证速度和精度**。

**关键提醒：不要无脑写死`torch.bfloat16`！**

BF16需要Ampere及以上架构，RTX 30系及以后、A100、H100等支持，RTX 20系及更早不支持。更稳妥的做法是**让代码

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（4-bit 怎么加载）、分页优化器（`optim`）和计算精度（`bnb_4bit_compute_dtype`）。现在把它们拼成一个**能直接跑的最小骨架**，让你看清 QLoRA 训练从加载到训练的整体流程。

### 完整骨架代码

```python
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "your/base-causal-lm"   # 替换为实际的 Causal LM
# 模型许可证：请确认所选基础模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：4-bit QLoRA训练建议至少8GB显存；若使用7B模型FP16加载则需至少14GB显存

# 1) 4-bit 量化配置（按 GPU 能力选择计算 dtype）
compute_dtype = torch.bfloat16 if torch.cuda.bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

# 2) 以 4-bit 加载基础模型
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
    raise RuntimeError(f"模型加载失败，请检查网络连接或模型路径：{e}") from e

# 3) 关键：为 k-bit 训练做准备（处理 norm/输出层精度、并开启梯度检查点所需的输入梯度等）
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # 与 TrainingArguments 的 gradient_checkpointing 对应
)

# 4) 注入 LoRA（只训练低秩增量参数）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],   # 按实际模型模块名调整，或用 "all-linear"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5) 极简数据（真实场景替换为足够规模的数据；Causal LM 的 labels 直接复制 input_ids）
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6) 训练参数：分页优化器 + 梯度检查点
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,        # 与第 3 步的 use_gradient_checkpointing 呼应
    optim="paged_adamw_8bit",           # 分页优化器（依赖 bitsandbytes；名称随版本可能不同）
)

# 注意：Trainer 初始化参数因 transformers 版本而异——新版本用 processing_class，旧版本用 tokenizer
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,          # 旧版本用 tokenizer=tokenizer
    )
    trainer.train()
except (OSError, RuntimeError) as e:
    raise RuntimeError(f"训练过程中出现异常，请检查显存或文件写入权限：{e}") from e
```

### 骨架的六个步骤，按执行顺序理解

**第 1 步：配置量化。** `BitsAndBytesConfig` 里四个参数各管一件事：`load_in_4bit=True` 开启 4-bit 加载；`bnb_4bit_quant_type="nf4"` 用 NF4 量化类型；`bnb_4bit_use_double_quant=True` 开启双重量化；`bnb_4bit_compute_dtype` 决定计算精度——优先用 bfloat16，不支持才退回 float16。

**第 2 步：加载模型和分词器。** 注意 `quantization_config=bnb_config` 把第 1 步的配置传进去，`device_map="auto"` 自动分配设备。如果分词器没有 pad token，就用 eos token 顶上——这是 Causal LM 的常见处理。

**第 3 步：为 k-bit 训练做准备。** `prepare_model_for_kbit_training` 是**连接量化模型和后续训练的关键桥梁**。它处理 norm 层和输出层的精度问题，同时开启梯度检查点所需的输入梯度。这里的 `use_gradient_checkpointing` 与第 6 步 `TrainingArguments` 中的 `gradient_checkpointing=True` 相互呼应，两者需保持一致。

**第 4 步：注入 LoRA。** `LoraConfig` 中 `r=8` 是低秩矩阵的秩，`lora_alpha=16` 是缩放系数，`lora_dropout=0.05` 是 dropout 率，`target_modules` 指定注入 LoRA 的模块。`get_peft_model` 把 LoRA 注入到量化后的模型上，`print_trainable_parameters()` 会打印可训练参数量，方便确认只有 LoRA 参数在训练。

**第 5 步：准备数据。** 这里用两句话构造极简数据集，`tok` 函数对文本做分词和截断，并把 `labels` 设为 `input_ids` 的副本——Causal LM 训练时标签就是输入本身。`DataCollatorForLanguageModeling` 负责在批处理时自动构建标签和注意力掩码。

**

## 块 53

## 九、把 QLoRA 的四个关键步骤串起来

你现在要做的不是记住零散函数，而是理解这四步**为什么缺一不可**。想象你在搭一条流水线：

**第1步：`BitsAndBytesConfig` —— 给模型“瘦身”的开关**
这一步决定基础模型如何以 4-bit 加载。你在这里指定 NF4 量化、双重量化、计算 dtype。没有它，模型就是正常精度，后面的省显存无从谈起。

**第2步：`prepare_model_for_kbit_training()` —— 让量化模型“能训练”的适配器**
这是 QLoRA 的**必要一步**。量化后的模型不能直接训练，这个函数会做适配：把部分层转成计算精度、开启输入梯度以支持梯度检查点等。**跳过它，常导致训练不收敛或直接报错**。这是最常见的坑，务必记住。

**第3步：`get_peft_model(..., LoraConfig)` —— 注入可训练的 LoRA**
把 LoRA 适配器注入到冻结的 4-bit 主干上。主干保持冻结，只有 LoRA 参数可训练。

**第4步：梯度检查点配合使用**
确保在 `TrainingArguments` 中设置 `gradient_checkpointing=True`，并确认 `prepare_model_for_kbit_training` 已启用梯度检查点（该函数默认会设置），两者配合使用以节省显存，但并非必须同时显式开启。

**第5步：`optim="paged_adamw_8bit"` —— 分页优化器**
启用分页优化器来缓解峰值显存。

> 上面是“能说明关系”的最小骨架。真实训练需要换成足够规模的数据、合理超参，并按你的版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数和 `processing_class`/`tokenizer` 写法。

---

## 十、你的电脑能跑吗？

**只学原理**：无需 GPU。

**实际运行 QLoRA**（最低配置）：
- CPU：4核+
- 内存：16GB+
- GPU：NVIDIA CUDA GPU
- 显存：7B 级别经验参考约 **8~12GB+**，推荐 **16GB以上

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会以为LoRA训练完，得到的是一个“升级版”的完整模型。**不是的。**

LoRA训练时，基础模型全程**冻结**（不动），只有新增的**Adapter**（适配器）参数在训练。最终的结构是：

```text
基础模型 + LoRA Adapter
```

举个例子：你用Llama-7B做LoRA微调，训练完成后保存的通常只有两个文件：

```text
adapter_model.safetensors   ← LoRA权重
adapter_config.json          ← 配置信息
```

**没有完整模型**。基础模型还是原来那个，LoRA只是“贴在旁边”的一层额外参数。

---

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（通常几MB到几十MB，远小于完整模型）
- 保存快
- 一个基础模型可以搭配多个Adapter，各干各的活

比如同一个基础模型，可以配：

```text
客服Adapter  → 做客服对话
医疗Adapter  → 做医疗问答
法律Adapter  → 做法律咨询
```

**缺点：**

部署推理时，必须**基础模型 + Adapter 两部分同时存在**，缺一不可。这就带来一个问题：每次部署都要加载两个东西，麻烦。

---

## 二、为什么需要合并LoRA权重？

既然Adapter必须搭配基础模型才能用，那能不能把两者“焊死”成一个完整模型？——这就是**合并（merge）**要做的事。

合并后，你得到的是一个**独立的完整模型**，不再依赖Adapter文件，直接加载就能推理。

---

## 动手验证一下

打开终端，检查你训练输出目录下是否有这两个文件：

```bash
ls your_output_dir/
```

你应该能看到 `adapter_model.safetensors` 和 `adapter_config.json`。如果只有这两个文件而没有完整模型

## 块 55

### 推理时的问题：为什么合并？

你现在用LoRA微调，但推理时模型是“分裂”的：

```text
输入 → Base Model + LoRA Adapter → 输出
```

这意味着每次运行都要**加载基础模型**，再**加载Adapter**，两步缺一不可。

但部署时你希望模型是“完整”的：

```text
输入 → 完整模型 → 输出
```

所以需要**合并权重**。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

### 合并的数学原理

训练时，LoRA学到的变化是：

```text
W

## 块 56

## 3.1 merge_and_unload 是做什么的？

你现在已经训练好了一个 LoRA 模型，但训练完的模型还带着"包装"——它由**基础模型**和**LoRA适配器**两部分组成。`merge_and_unload()` 就是 PEFT 库提供的一个方法，作用非常直接：

> **把 LoRA 权重合并进基础模型，然后移除 PEFT 包装。**

用一句话理解：

```text
基础模型 + LoRA适配器 → 完整模型
```

合并前，你的模型是 `PeftModel`，结构像这样：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

合并后，它变成了一个**普通的 Transformer 模型**，里面已经包含了 LoRA 更新后的权重，不再有适配器这一层。

---

## 四、完整流程：从训练到导出

整个 LoRA 合并导出流程如下：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步走。

---

## 五、加载 LoRA Adapter

假设你有：

- 基础模型路径：`base_model_path`
- LoRA 适配器路径：`adapter_path`

加载代码如下：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 把 LoRA 适配器加载到模型上
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构为：

```text
Base Model + LoRA Adapter
```

---

## 六、执行合并

一行代码完成合并：

```python
merged_model = model.merge_and_unload()
```

合并的数学本质是：原权重 `W` 加上 LoRA 训练得到的低秩增量 `ΔW=(α/r)BA`，得到新权重 `W'`：

```text
W + (α/r)BA → W'
```

合并完成后，`merged_model` **不再是** `PeftModel`，而是一个**普通 Transformer 模型**。

---

## 七、导出合并后的模型

合并完成后，保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后的目录示意：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：上面是**示意结构**，实际文件会随模型/分词器类型变化。比如可能还有 `generation_config.json`、`special_tokens_map.json`；大模型权重可能分片保存（`model.safetensors.index.json` + 多个分片）；有些分词器用的是 `vocab.json`/`merges.txt` 或 `tokenizer.model`。

---

## 本块学习检查

请你在自己的环境中完成以下操作验证：

1.

## 块 57

## 三、merge_and_unload 是什么？（重点）

**一句话理解**：`merge_and_unload` 就是把 LoRA 训练时加在底座模型上的低秩增量矩阵（Adapter）**合并**进底座模型权重中，得到一个包含微调效果的完整、独立的模型文件。合并后，你不再需要 PEFT 库来加载它，直接用 Transformers 就能跑。

> **模型许可证**：Llama 2 Community License（商用月活>7亿需Meta授权）。

---

### 合并后必须检查的 4 件事

合并导出不是点一下按钮就完事，**“能加载”不等于“行为正确”**。请按下面清单逐项核对：

1. **Tokenizer 完整且一致**  
   确认保存的 tokenizer 和训练时用的是同一个，尤其是训练中**新增或改过的特殊 token**（比如自定义的 `<|im_start|>`、pad/eos 等），必须都在 `special_tokens_map.json` 或 `tokenizer_config.json` 里。如果训练时扩过词表，还要确认 embedding 和 lm_head 的尺寸跟新词表对得上。

2. **generation_config.json 已导出**  
   如果你依赖特定的生成默认值（比如 eos/pad token id、`max_new_tokens`、采样参数），确认这个文件一并保存了。否则推理时可能回退到不合适的默认行为，输出结果会“怪怪的”。

3. **dtype 精度符合预期**  
   确认合并/保存时的精度是 FP16、BF16 还是 FP32。精度不一致会影响显存占用和数值表现——比如你训练时用的 BF16，导出却变成 FP32，显存直接翻倍。以 Llama-2-7b 为例，FP16 约需 14GB 显存，FP32 约需 28GB。

4. **config.json 干净完整**  
   确认模型结构配置（层数、hidden size、任务头等）正确，并且**已经移除了 PEFT/adapter 相关配置**。合并后应该是普通 Transformers 模型，不应该再残留 `adapter_config.json` 这类文件。

---

### 合并后怎么加载推理？

**不需要**再写 `peft` 相关代码，直接：

```python
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from

## 块 58

### 10.1 推荐合并场景：什么时候该把LoRA“焊死”进模型？

你正在学LoRA微调，目标很明确：跑通流程、做出能用的模型。这里讲两个**最该合并**的场景，帮你判断“要不要merge”。

---

**场景1：上线部署——只要一个固定Adapter，追求简单**

想象生产环境：服务器 → 直接加载模型 → 提供API。如果只需要**一个固定Adapter**，且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载），合并通常更方便。

> ⚠️ 注意：如果推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或你需要在同一基础模型上**热切换多个Adapter**，那么不合并、以Adapter形式部署反而更灵活。此时是否合并要看框架能力与业务需求。
> # 模型许可证：Llama 2 Community License

## 块 59

### 10.2 不建议合并场景（动手实践视角）

如果你是**动手实践型**学习者，记住一个核心判断：**“还要继续用，就别合并。”**

**场景1：多个任务Adapter**  
假设你有一个基础模型，上面挂了多个Adapter（比如Adapter A、B、C），分别对应不同任务。  
- **为什么不合并？** 合并后你就把多个任务“焊死”成一个了，想单独切换任务就难了。  
- **动手建议：** 保留Adapter，训练时按需加载对应Adapter，灵活切换。

**场景2：持续训练**  
如果你打算继续调整LoRA参数，**保留Adapter**。  
- 合并相当于“定型”，后续再想微调就得重新来，不划算。

---

### 十一

## 块 60

## 直接合并可能遇到的情况

如果你在 4-bit 量化状态下直接执行 `merge_and_unload`，可能会碰到两类问题：

1. **精度损失**——4-bit 权重需要先反量化再合并，这个过程和全程用高精度计算并不完全等价，结果会有细微偏差。
2. **兼容性问题**——可能出现权重类型不匹配，或者你当前使用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> ⚠️ 注意：这些只是**可能**出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你的库版本、量化配置，以及是否先做了反量化。请以你所用版本的实际运行结果为准。

---

## 常见做法：先反量化到高精度再合并

更稳妥的做法是：**别在 4-bit 状态下直接合并**，而是先用 FP16/BF16 高精度重新加载一份基础模型，再挂载同一个 Adapter 后合并。

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

这样做的好处是：合并发生在高精度权重上，绕开了 4-bit 状态下的精度和类型问题。

---

## 可执行代码

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

**代码要点回顾：**

- 第 1 步是核心：**不传 `quantization_config`**，用 `torch_dtype` 指定 FP16/BF16，这样基础模型就以高精度加载了。
- `device_map="auto"` 自动分配设备；如果显存不够，改成 `{"": "cpu"}` 在 CPU 上合并。
- `safe_merge=True` 会在合并时检查数值

---

## 模型许可证说明

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

## 块 61

**QLoRA合并前必看：三个关键点**

你马上要实操LoRA微调了，这里直接给你划重点，全是合并时容易踩的坑。

**1. 版本决定“能不能直接合并”**
不同PEFT版本对“量化模型能否直接合并”支持不一样，有的要求先反量化。**务必先查你所用版本的`merge_and_unload`文档**。关键操作是：加载基础模型时**不要传**`quantization_config`或`load_in_4bit`，这样拿到高精度权重再合并。

**2. 内存不够？换CPU跑**
以高精度重新加载完整模型，显存占用约等于一份FP16/BF16完整模型。显存不足时，把`

## 块 62

## 11.3 合并时开启校验并做一致性验证（建议）

`merge_and_unload` 支持 `safe_merge=True`，合并时会做数值检查（例如检测是否产生 NaN），相当于给合并操作加一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，**务必对比合并前后模型在同一批输入上的输出**，确认合并没有引入明显偏差。流程如下：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits / 生成文本是否基本一致

最小验证代码思路：用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本

## 块 63

## 11.3 合并时开启校验并做一致性验证（建议）

合并完模型，别急着收工。QLoRA 的合并本质上是把低精度的适配器权重“折算”回基础模型，这个过程会引入数值上的微小偏差。为了确认合并结果没“跑偏”，我们需要做一次**前后对比校验**。

### 为什么必须单独加载一份“未合并”的模型？

这是本块最容易踩的坑。`merge_and_unload()` 是一个**原地操作**——它直接修改了当前这个 `PeftModel` 对象，合并完成后，这个对象就不再是“未合并”状态了。

所以，你不能拿 §11.2 里那个已经合并过的 `merged_model` 当作“合并前”的基准。正确做法是：**重新加载一份全新的、未合并的 `PeftModel`**，专门用来做对比。

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
# 单独加载一份“未合并”的模型作为对比基准
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

### 校验一：比较 logits（数值层面）

用同一个固定输入（比如 `prompt = "请用一句话介绍你自己。"`），分别让“合并前”和“合并后”的模型跑一次前向传播，比较输出的 logits 差异：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果？** 这个值越接近 0，说明合并前后的数值越一致。由于 QLoRA 本身涉及低精度量化，这里通常不会精确等于 0，而是一个**较小的近似值**——只要没有数量级上的异常，就说明合并是可靠的。

### 校验二：比较生成文本（行为层面）

数值接近还不够，我们更关心模型“说出来的话”是否一致。用**贪心生成**（`do_sample=False`）去掉随机性，这样两次生成的结果完全可复现、可对比：

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后:", text_after)
```

## 块 64

## 合并后，为什么还要“多此一举”再加载一份模型？

你可能会想：`merge_and_unload()` 不是已经把 LoRA 合进模型了吗？直接拿合并后的模型和合并前的对比不就行了？

**关键坑在这里**：`merge_and_unload()` 返回一个**新的合并后模型**，原模型对象保持不变。但如果你只保留返回值，原模型变量就会被覆盖或丢失，你手里只剩合并后的结果。如果你拿它当“合并前”的对照，两边其实是**同一份已合并权重**，算出来的 `max_abs_diff` 恒为 0，验证完全失效。

**正确做法**：在合并前，先用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照，再执行合并。这样你手里同时有“合并前”和“合并后”两份模型，才能做真正的对比。

---

### 显存不够怎么办？

如果同时放不下两个模型，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本**存下来**，然后**释放显存**；
2. 再加载合并后的模型，跑同样的输入，对比结果。

---

### 结果预期：别追求“逐位相同”

- **QLoRA 场景**：合并前后通常只是**近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 差异。重点不是“完全一样”，而是**确认没有明显退化或出现 NaN**。
- **LoRA（非量化）场景

## 块 65

## 合并后出问题了？这份排查清单请收好

你在 LoRA/QLoRA 微调路上已经走到合并这一步，非常棒！但合并时可能会遇到几个“坑”。别担心，这很正常。下面是一份**实战排查清单**，按“症状→原因→怎么办”的顺序帮你快速定位问题。

### 常见问题与排查方向

| 你遇到的症状 | 可能的原因（不止一种） | 优先排查方向 |
| :--- | :--- | :--- |
| **`merge_and_unload` 不存在** | 你当前操作的对象不是 `PeftModel`。可能是忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或者模型已经被合并/加载成了普通模型。 | 确认加载方式。**先**用 `PeftModel.from_pretrained` 得到 `PeftModel`，**再**调用合并方法。 |
| **合并后模型变大** | 这是**正常现象**。合并后是完整模型，不再是体积小的 Adapter。 | **无需处理**。如果在意体积，可以保留 Adapter 不合并。 |
| **显存不足（OOM）** | 一次性加载高精度完整模型占用高；也可能是用了 FP32、没做 offload/分片。 | 改用 **CPU 合并**或 offload；用 **FP16/BF16** 精度；或**分片加载**。 |
| **Tokenizer/配置缺失** | 只保存了模型，没保存 tokenizer；或特殊 token / `generation_config` 没一并导出。 | 重新 `save_pretrained`，并核查 tokenizer、`generation_config`、config 是否都保存了。 |
| **QLoRA合并失败或结果异常** | 在 4-bit 状态直接合并、PEFT/bitsandbytes 版本不支持、量化包装未正确反量化、内存/显存不足。 | **先反量化到 FP16/BF16 再合并**（具体做法见下）；核对版本；必要时用 CPU 合并；开启 `safe_merge=True` 并做一致性验证。 |

### 关键操作：什么是“先反量化到 FP16/BF16”？

这里的“反量化”**不是**手动去逐层还原 4-bit 权重，而是——**重新加载一份不带量化配置的基础模型**。

具体来说，就是：
1.  用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float

## 块 66

## 路线A：合并后部署完整模型（动手实操版）

你已经完成了 LoRA 微调，现在到了“把模型用起来”的环节。这里先讲**路线A**——把 LoRA 合并进基础模型，得到一个“普通模型”，然后像用任何预训练模型一样去部署它。

### 核心流程（三步走）

```
merge_and_unload()   ← 将LoRA适配器权重合并到基础模型并卸载适配器，返回合并后的模型
        ↓
保存完整模型          ← 保存合并后的结果
        ↓
推理框架按普通模型加载 ← 比如 vLLM 直接加载 merged_model
```

### 为什么选这条路？

- **部署链路简单**：推理框架把合并后的模型当**普通模型**加载，它完全不知道 LoRA 的存在，不需要任何特殊配置。
- **推理时零切换成本**：没有 Adapter 切换的开销，推理速度更稳定。
- **代价要清楚**：每个 Adapter 都要保存/分发一份**完整模型**。如果你有多个任务（多个 Adapter），存储成本会成倍增加。

### 什么时候用路线A？

- 你只需要部署**一个**微调模型，不打算频繁切换任务。
- 你的推理环境不支持 LoRA 动态加载。
- 你追求部署简单、稳定，存储空间不是瓶颈。

### 动手验证（建议实操）

1. 在微调脚本末尾调用 `merge_and_unload()`，确认返回的是合并后的模型对象。注意该操作会**卸载LoRA适配器**，合并后的模型权重已包含LoRA增量。
2.

## 块 67

### 14.3 如何选择：合并还是保留？

你正在学LoRA微调，目标是让模型能用起来。这里有个关键认知：**合并不是唯一部署方式**。

按你的情况（动手实操、节奏快），直接看决策点：

- **只有一个Adapter，想最省事** → 选**合并**（路线A）。训练完直接merge_and_unload()，导出完整模型，部署简单。
- **同一个基础模型要挂多个Adapter，且推理框架原生支持LoRA** → 选**保留Adapter动态服务**（路线B）。不合并，运行时按需加载，灵活切换。
- **用QLoRA且对精度敏感** → 无论选哪条，都建议做**合并前后/上线前的一致性验证**（见§11.3），确保结果没偏差。

> 注意：以上是经验参考，非硬性规则，以实际部署环境为准。具体框架用法留到第15板块（vLLM部署）再展开。

---

### 本块学习检查

**动手验证**：假设你训练了一个LoRA Adapter，基础模型是同一个。请

## 测评题目

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1204-003
分页优化器的主要目标是？

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。
