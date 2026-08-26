# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么做的：

```text
输入数据
    ↓
整个模型参数更新
    ↓
保存完整模型
```

以 7B 参数模型为例，就是 **70亿个参数**。训练时，你需要同时保存三样东西：

* **参数**：模型本身的权重
* **梯度**：用于更新参数的计算结果
* **优化器状态**：控制学习过程的额外信息

这三样加起来，显存需求**非常高**。普通消费级 GPU 根本扛不住。

---

**动手想一想**：假设你的 GPU 显存是 8GB，你觉得能装下 70 亿参数的模型加梯度加优化器状态吗？大概率不行——这正是我们要引入参数高效微调的原因。

**下一步**：接下来我们会看到 LoRA 如何用“只训练一小部分参数”的方式，把显存需求降下来，让普通 GPU 也能跑微调。

## 块 2

**本块学习检查**：读完下表后，请说出LoRA相比全参数微调，在“训练参数”和“显存需求”上的区别。

PEFT（参数高效微调）的核心是：**不修改整个大模型，只训练少量新增参数**。

流程上，原始大模型加上LoRA适配器参数，只训练LoRA参数，最后保存小规模Adapter。

对比三种方式：

- **全参数微调**：训练全部参数，显存需求非常高
- **LoRA**：只训练少量参数，显存需求较低
- **QLoRA**：量化模型+LoRA，显存需求更低

结合你的LoRA微调目标，记住：**LoRA用少量参数换低显存

## 块 3

## 2.1 LoRA是什么？

**LoRA** = **Low-Rank Adaptation**（低秩适配），是一种“不动原模型、只加小补丁”的微调方法。

**核心思路：**

- 原模型的权重矩阵 `W` **不更新**；
- 在旁边新增一个增量 `ΔW`，最终模型变成：`W + ΔW`；
- 其中 `ΔW = B × A`，`A` 和 `B` 是两个**低秩小矩阵**。

**为什么可行？**

假设原权重 `W` 形状是 `d_out × d_in`，那么：

- `A` 形状为 `d_in × r`
- `B` 形状为 `r × d

## 块 4

## 3.1 LoRA的问题：省了训练，省不了加载

你已经知道 LoRA 能大幅减少**训练**时的参数，但这里有个关键点：**基础模型本身仍然要完整加载到显存里**。

以 7B 模型为例，如果用 FP16（半精度）加载，大约需要 **14GB 显存**。也就是说，LoRA 只是让“训练”更轻量，但“加载模型”这一步依然很占显存。

## 3.2 QLoRA 的解决思路：先把模型“压缩”再训练

QLoRA 的全称是 **Quantized LoRA**，即“量化版 LoRA”。

核心流程一句话：

> **FP16 模型 → 4bit 量化 → 减少显存占用 → 再加载 LoRA 进行训练**

结构上就是：

```text
4bit 基础模型 + LoRA 参数 → 训练
```

这样，基础模型以 4bit 精度驻留显存，占用的空间大幅下降，给 LoRA 训练留出了余量。

## 3.3 QLoRA 依赖 bitsandbytes 库

要跑 QLoRA，你需要这几个组件：

```text
transformers + peft + bitsandbytes + 支持的量化后端
```

其中 **bitsandbytes** 负责三件事：

- 低精度量化
- 4bit 加载
- 8bit 加载

> **关于硬件后端的重要说明**：本教程的实训与验证**统一以 NVIDIA CUDA GPU 为准**，这是项目支持范围，也是当前资料最全、验证最充分的

## 块 5

### 电脑配置批注：为你的LoRA微调备好“行装”

既然你的目标是动手跑通LoRA微调，那我们先来对照一下电脑配置，确保“弹药充足”。别担心，门槛没有想象中高。

**LoRA最低配置（能跑起来）：**
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上空闲空间
- **GPU**：推荐NVIDIA显卡，**显存最低8GB**

**推荐配置（跑得更顺畅）：**
- **显存**：12GB以上

**QLoRA的特别说明：**
QLoRA的设计初衷就是**在更小的显存上微调大模型**。它通过4-bit量化基础模型，显存占用通常比LoRA全精度加载更低。对于常见的7B级别模型，经验参考如下：
- **参考下限**：约8~12GB显存（短序列、小batch）
- **推荐**：16GB以上

> **重要提醒**：上面的数值是**经验起点**，不是硬性门槛。实际显存占用受多个因素影响：
> - **模型规模**：7B级别；13B/70B等更大模型需求显著上升
> - **序列长度**：短序列（≤512）占用低；越长占用越高
> - **batch大小**：

## 块 6

## 5.1 先确认你的 Python 版本

动手前，先花10秒确认环境。打开终端，运行：

```bash
python --version
```

看到 `Python 3.10` 或 `3.11` 就没问题。如果版本偏低，建议先升级，否则后面装库容易踩坑。注意，Python 3.12 及以上版本可能存在部分库兼容性问题，建议优先使用 3.10 或 3.11。

---

## 5.2 检查 PyTorch 是否就绪（关键一步）

LoRA/QLoRA 依赖 PyTorch，而且 **bitsandbytes 的量化功能要求安装和本机 CUDA 匹配的 GPU 版 PyTorch**。直接 `pip install torch` 通常装到 CPU 版，QLoRA 会直接跑不起来。

去 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选对应版本。比如 NVIDIA CUDA 12.1 就执行：

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

装完验证一下：

```python
import torch

print(torch.__version__)

print(
    torch.cuda.is_available()
)

print(torch.version.cuda)   # 查看 torch 编译使用的 CUDA 版本
```

看到 `True`，说明 PyTorch 能识别 GPU。

> ⚠️ 注意：`True` 只代表 PyTorch 这边能用 GPU，**不代表 bitsandbytes 的 4-bit 量化一定能跑**。后者还要看 bitsandbytes 是否和本机 CUDA 匹配，后面章节会专门验证。

---

## 5.3 查看 GPU 信息

终端执行：

```bash
nvidia-smi
```

输出里能看到：

- GPU 型号（如 RTX 4090）
- CUDA 版本（如 12.x）
- 显存大小（如 24GB）

记下这三项，后面配置 bitsand

## 块 7

**本块学习检查**：读完下面内容后，请用自己的话说一句：bitsandbytes 在 QLoRA 里到底负责什么？

---

### 7.1 bitsandbytes 是干嘛的？

一句话：**它负责把模型“压缩”到 8bit 或 4bit，QLoRA 训练必须靠它。**

具体来说，bitsandbytes 用于三件事：

- **8bit 量化**：把模型精度降低，省显存；
- **4bit 量化**：比 8bit 更省，QLoRA 用的就是它；
- **QLoRA 训练**：整个训练过程依赖这个库跑起来。

> ⚠️ 注意：这里说“必须安装”，是指**本教程的 QLoRA 方案**必须用它。QLoRA 本身不一定绑死某个量化

## 块 8

**本块学习检查**：先跑 `nvidia-smi`，记下输出的 `CUDA Version`；再跑 `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`，确认它能显示版本号。两者都正常，再继续下一步。

---

**为什么先查CUDA？**  
bitsandbytes依赖CUDA来驱动GPU计算。版本不匹配，后面量化、加载模型都会报错，甚至CUDA直接不可用。所以，动手前先花1分钟确认环境。

**两步检查法（照着做就行）**  
1. 终端运行 `nvidia-smi`，看输出里的 `CUDA Version` 字段。  
2. 终端运行 `python -c "import

## 块 9

## 9.1 推荐使用WSL2

在Windows上跑LoRA微调，最省心的路径是：

```
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

**为什么推荐？** 因为Linux环境兼容性更好，能少踩很多坑。

---

## 9.2 Windows原生可能遇到的问题

如果你坚持用Windows原生环境，可能会遇到：

**问题1：bitsandbytes无法加载CUDA动态库**

这通常不是单一原因，而是多个因素叠加：

- 你装的`bitsandbytes`版本，和本机CUDA/GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本的支持不如Linux完善（不同版本差异大）。

**核心结论：** 这不是简单的"Windows不支持"，而是**PyTorch、bitsandbytes版本、本机CUDA

## 块 10

好的，收到你的要求。作为你的个性化学习助手，我会根据你“动手实践”的学习风格和“快速上手”的节奏，为你提炼这份环境配置指南的核心步骤，并直接聚焦于“如何做”和“如何验证成功”。

---

### 九、Windows用户特别注意事项（新增）

#### 问题2：CUDA版本不匹配

如果你在后续步骤中遇到报错，很可能是CUDA版本问题。请先运行 `nvidia-smi` 检查你的NVIDIA驱动和CUDA版本，确保它们与你要安装的PyTorch版本兼容。

---

### 十、安装完整LoRA环境

在确保已安装匹配本机CUDA的GPU版PyTorch后（例如CUDA 12.1的安装命令如下），我们开始安装LoRA所需的库。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**核心安装命令：**

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

这条命令会一次性安装所有必需的组件。

---

### 十一、配置Accelerate

运行 `accelerate config` 命令，根据你的环境进行配置。对于单GPU环境，可以按以下示例选择：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

这会生成一个 `default_config.yaml` 配置文件，用于加速训练。

---

### 十二、LoRA环境验证实验

创建一个名为 `test_lora_env.py` 的文件，并写入以下代码，来验证PyTorch和PEFT库是否正常工作。

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
except ImportError as e:
    print("导入失败，请检查PyTorch或PEFT库是否正确安装。")
except RuntimeError as e:
    print("运行时错误，请检查CUDA环境。")
```

运行 `python test_lora_env.py`。如果成功，你会看到 `CUDA: True` 和LoRA配置信息，说明环境就绪。

---

### 十三、QLoRA环境验证实验

最后，测试bitsandbytes库是否可用。运行以下Python代码：

```python
import torch

try:
    import bitsandbytes
    print(torch.cuda.is_available())
    print(bitsandbytes.__version__)
except ImportError:
    print("bitsandbytes导入失败，请先安装bitsandbytes库。")
```

如果输出 `True`，则说明GPU环境和bitsandbytes库都正常，可以支持QLoRA的4

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**为什么推荐WSL2？**

在Windows上运行LoRA/QLoRA环境时，最常见的坑就是**兼容性问题**——比如bitsandbytes加载失败、CUDA无法识别等。这些问题往往不是你的操作错误，而是Windows原生环境与深度学习工具链的“水土不服”。

**解决方案：使用WSL2**

WSL2（Windows Subsystem for Linux 2）能让你在Windows上运行一个完整的Linux环境，而深度学习工具链（PyTorch、CUDA、bitsandbytes等）在Linux下兼容性最好。遇到Windows报错时，直接切换到WSL2环境，大部分问题迎刃而解。

**快速排查表**

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| **Windows报错** | **兼容性问题** | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

**本

## 块 12

## 一、为什么需要LoRA？（①②③）

### 1.1 大模型微调的问题

现代大语言模型（如LLaMA、Qwen、Baichuan、Mistral）通常拥有数十亿甚至上百亿参数。以7B模型为例，就是70亿参数。

如果采用传统方式微调——预训练模型 → 更新全部参数 → 保存新模型——你需要：

- **巨大的GPU显存**：训练时不仅要保存模型参数W，还要保存梯度∂W和优化器状态（如Adam moment），实际显存需求远大于模型本身大小；
- **大量训练时间**：更新全部参数意味着每次迭代都要计算和传播所有参数的梯度；
- **巨大的存储空间**：每个微调版本都要保存一份完整的新模型。

假设模型有100亿参数，全参数微调时显存需求会膨胀到模型大小的数倍——这就是为什么我们需要更高效的方法。

---

### 1.2 LoRA的核心思想

LoRA（Low-Rank Adaptation）的核心洞察是：**微调时不需要更新全部参数**。

LoRA冻结原始权重W，仅训练低秩矩阵A和B，通过增量ΔW=(α/r)BA来近似微调过程中的变化，其中r是秩，α是缩放系数。这样你只需要训练B和A这两个小矩阵，而不是整个大模型。

**rank（秩）决定了训练参数量**：rank越小，可训练参数越少，训练越快、显存占用越低；rank越大，表达能力越强，但训练成本也越高。

LoRA通过这种方式

## 块 13

## 2.1 不直接修改原模型参数

**传统微调的做法**：直接改动原始权重 W，训练后变成 W'，也就是“改原模型”。

**LoRA 的做法**：把 W 冻结住，不去动它。另外增加一个新参数 ΔW，最终模型用的是：

```
W' = W + ΔW
```

- **W**：原模型参数（冻结，不训练）
- **ΔW**：LoRA 新增的参数（只训练这个）

## 2.2 为什么只训练ΔW？

大模型已经通过海量数据学会了语言规律、世界知识、推理能力。微调本质上只是“对已有能力做方向调整”，所以**不必重新学习全部参数**，只学任务相关的变化量就够了。

> 注意：这是帮助理解的直觉假设，并非严格结论。任务与预训练能力差距大、或数据充足时，全参数微调仍可能更好。

**动手验证**：想象你有一个训练好的模型，现在要让它学会“写诗”。按 LoRA 思路，你会冻结原模型，只训练

## 块 14

**本块学习检查**：看完后，请合上资料，自己写出：一个 `4096 × 4096` 的矩阵有多少个参数？LoRA 把 `ΔW` 拆成了哪两个小矩阵的乘积？

---

### 3.1 矩阵基础概念

Transformer 里大量参数都是**矩阵**。比如注意力层：

```text
Q = X · Wq
```

这里 `Wq` 就是一个权重矩阵。假设它是 `4096 × 4096`，那它的参数数量就是：

```text
4096 × 4096 ≈ 1677万
```

一个矩阵就有这么多参数，非常庞大。

### 3.2 直接训练矩阵的问题

如果我们要微调，直接去修改 `ΔW`，那 `ΔW` 同样需要 `4096 × 4096` 个参数，成本太高。

**LoRA 的核心思路**：不直接学习完整矩阵，而是把它**拆成两个小矩阵**。

原始权重 `W` 保持冻结不动，新增的 `ΔW` 由 `B × A` 组成。这样，要学的参数就从一个大矩阵变成了两个

## 块 15

**本块学习检查**：动手算一遍——把 W 从 4096×4096 换成 1024×1024，r 仍为 8，写出 A、B 的尺寸和总参数量，再对比完整矩阵参数，感受下降幅度。

---

先看原矩阵：W 是 4096×4096，直接训练要更新 **1677万** 个参数（4096×4096）。

LoRA 的思路是拆成两个小矩阵：设 **r=8**，得到 A（8×4096）和 B（4096×8）。A 有 32768 个参数，B 也有 32768 个，合计 **约6.55万**。

对比一下：完整

## 块 16

**5.1 神经网络参数变化具有低维特征**

LoRA论文的核心观察是：模型适应新任务时，参数变化 **ΔW** 不需要覆盖所有方向。

想象一个4096维的空间，真正影响任务的“有效方向”可能远少于满维度，集中在一个**低维子空间**里——即远小于原维度的少数方向。

> 注意：这是“权重更新具有较低内在维度”的**假设**，具体有效维度没有通用固定值，随模型、任务、数据变化。这里只做定性理解，别记成“几十维”这种确定数字。

所以，用低秩矩阵 **BA** 就能近似 **ΔW**。

---

**本块学习检查**：用自己的话解释“低维子空间”是什么意思？提示：对比“409

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。但如果这张图片的主要变化只是**亮度、对比度**和**少量结构变化**，其实用**少量参数**就能描述清楚。

LoRA 的核心思想正是如此：**用少量参数描述模型变化方向**。预训练模型已经学到了大量通用知识，微调时我们不需要重新学习全部参数，只需要捕捉“模型需要改变的方向”——这个方向往往可以用低秩矩阵来近似表达。

---

## 六、LoRA 在 Transformer 中的位置

Transformer 的主要结构是：

```text
Input → Embedding → Attention → FFN → Output
```

其中 Attention 包含 **Q、K、V、O** 四个投影矩阵。

LoRA 在早期实践中**常见**只加到 `q_proj` 和 `v_proj` 上，例如：

```python
target_modules=["q_proj", "v_proj"]
```

这表示只训练 `Wq` 和 `Wv` 的变化量。

> ⚠️ **注意**：只挑 `q_proj`/`v_proj` 是 LoRA 原论文的**常见选择**，**不是固定标准**。现代实践经常把 LoRA 加到全部投影（q/k/v/o）和 FFN 线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构确认实际模块名，再根据验证集表现调整。

**更完整的配置示例**（以 Llama 类模型常见模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认）：

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
        "gate_proj", "up_proj", "down_proj",       # FFN 线性层
    ],
)
```

> 覆盖模块越多，可训练参数、显存与 Adapter 体积也越大——是否值得，应结合验证集表现权衡。

---

## 块 18

## 8.1 rank（r）——低秩矩阵的“宽度”旋钮

先看代码里这一行：

```python
r=8
```

这个 `r` 就是**低秩矩阵的维度**。你可以把它理解为：在微调时，我们不去动整个大模型（那太贵了），而是插入一个“小通道”，`r` 就是这个通道的宽度。

### 调大 `r` 会发生什么？

**好处**：模型的表达能力更强——通道更宽，能学到的任务相关模式更多。

**代价**：参数更多、显存占用增加。你训练时吃的显存会变大，保存的 Adapter 文件也会变大。

### 那 `r` 到底该设多少？

没有“标准答案”，但有一个**经验起始范围**（注意：只是起点，不是铁律）：

| 任务类型 | rank 起点参考 |
| ---- | ---- |
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

**警告**：`r≥64` 容易过拟合，建议从 `r=8~32` 开始

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

你可能会问：设置了秩 `r` 之后，LoRA 参数在训练时更新得多猛？`lora_alpha` 就是干这个的。

```python
lora_alpha=16
```

它的作用很直接：**控制 LoRA 更新强度**。数值越大，每次更新时对模型的影响越强；数值越小，更新越温和。

实际使用时，通常用 `alpha / r` 作为缩放比例。比如 `r=8`、`alpha=16` 时，缩放比例就是 `16/8=2`。这个比例会作用在 LoRA 参数上，决定最终注入到模型中的更新幅度。

**动手建议**：先记住这个公式 `alpha/r`，后面实战调参时，你会反复用到它来平衡更新强度与稳定性。

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 的作用是**防止 LoRA 参数过拟合**。训练时随机丢弃一部分 LoRA 参数（这里是 5%），让模型不依赖某几个特定参数，从而更泛化。

**动手建议**：如果你发现训练集表现好但验证集差，可以尝试把 `lora_dropout` 从 0 调到 0.05 或 0.1。

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言）

## 块 20

## 2.1 Rank表示什么？

**一句话理解**：Rank（r）就是“矩阵的秩”，在LoRA中，它决定了你用来近似原权重变化的那两个小矩阵的“中间维度”有多大。

**为什么需要它？**

还记得上一节的公式吗？LoRA不直接更新原始权重 `W`，而是加上一个变化量 `ΔW`，且 `ΔW = B × A`。这里的 `A` 和 `B` 是两个低秩矩阵，而 `r` 就是控制它们形状的关键数字。

**用例子看最直观**：

假设原始权重矩阵是：

```
W: 4096 × 4096
```

这是一个巨大的矩阵。如果直接微调它，成本极高。LoRA的思路是：用两个小矩阵的乘积来近似这个变化。

如果你设置：

```python
r=8
```

那么两个低秩矩阵的形状就是：

```
A: 8 × 4096
B: 4096 × 8
```

注意看：`A` 的列数等于 `r`（8），`B` 的行数也等于 `r`（8）。这两个矩阵相乘：

```
ΔW = B × A
```

结果恰好恢复成：

```
4096 × 4096
```

**核心要点**：

- `r` 就是中间那个“瓶颈”的宽度，它决定了LoRA参数的总量。
- `r` 越大，`A` 和 `B` 能承载的信息越多，理论上效果可能更好，但训练和存储成本也随之上升。
- `r` 越小，参数越少、速度越快，但表达能力受限。

**动手验证一下**：你可以把 `r` 分别设为 4、16、32，算一算 `A` 和 `B` 的维度变化，感受一下 `r` 如何直接影响参数量。

## 块 21

**Rank控制什么？**

Rank是LoRA中低秩矩阵的秩。LoRA的核心机制是：冻结原权重W，训练低秩增量ΔW=(α/r)BA，其中r控制低秩近似的秩，影响参数量和表达能力。你正学LoRA微调，记住这个核心：**r=4**（低rank）→ 参数量少、显存低、训练快，代价是表达能力有限；**r=64**（高rank）→ 能学复杂任务、适应性强，但参数多、显存涨、训练成本高。

**动手验证（不引入新知识）**：在你的微调脚本里，分别设`r=4`和`r=64`，对比训练时显存

## 块 22

**本块学习检查**：不看表格，自己算一遍：r=32时LoRA参数量是多少？对比完整训练1677万，你发现了什么？

---

**3.1 不用LoRA：全量训练**

微调一个4096×4096的权重矩阵，要更新**全部**参数：

- 4096 × 4096 = **16,777,216**（约1677万）

**3.2 用LoRA：只训练两个小矩阵**

LoRA把一个大矩阵拆成两个小矩阵，参数量公式：

- **参数量 = r × d_in + r × d_out = r × (d_in + d_out)**

**r=8 时：**
8 × (4096 + 409

## 块 23

### 4.1 Rank太小的问题——为什么你的LoRA“学不动”？

假设你在代码里把LoRA的秩设为 `r=2`，这就像给模型配了一支“极细的画笔”——它能画，但笔触太细，画不出复杂的内容。

**可能出现的后果：**
- **学习能力不足**：模型能学习的低秩增量矩阵的维度太小，导致表达能力受限；
- **欠拟合**：训练集上的表现都不够好，更别说泛化到新数据；
- **新任务知识无法充分表达**：你想让模型学会的新能力，根本没有足够的“空间”被存储下来。

**你会在训练时看到这些“症状”：**
```text
训练loss下降慢
效果提升有限
```

**怎么自查？** 如果你跑LoRA训练时，发现loss曲线像“平躺”一样降不下去，或者微调后模型输出和基座模型差别不大，先检查一下是不是rank设得太小了。

---

### 4.2 Rank太大的问题——为什么“用力过猛”反而亏？

反过来，如果你把rank设成 `r=256`，又会遇到另一组麻烦：

- **LoRA增量参数增加**：要训练和存储的增量参数量变多；
- **显存增加**：GPU显存占用上升，可能直接OOM；
- **训练速度下降**：计算量变大，每个step更慢。

更关键的是：**LoRA的优势被削弱了**。

**用数字看清楚（避免直觉

## 块 24

**本块学习检查**：根据下表，为“情感分类”任务选一个起始rank值，并说明为什么不能直接照搬这个值。

---

### 4.3 常见rank选择

下表是**经验起始范围**，不是自动调参标准：

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

实际项目需根据数据量、模型大小、序列长度、batch大小（含梯度累积后的有效batch）、目标模块选择与数量、GPU显存、验证集表现等调整。**不要照搬**——同一个r在不同模型规模、序列长度/batch、目标模块下，显存和效果差别很大。

---

## 五、Alpha（lora_alpha）是什么？

### 5.1 Alpha作用

LoRA除了 `ΔW = BA`，还会缩放：

```text
W' = W + (α/r)BA
```

其中α是lora_alpha，r是rank。

因此

## 块 25

### 6.1 控制LoRA影响强度

咱们用代码来感受一下这个“缩放”是怎么工作的。假设你设了 `r=8` 和 `alpha=16`，那么缩放系数就是 `16/8=2`。这意味着，缩放系数为2，即 `ΔW = 2 * BA`，其中 `BA` 是训练得到的低秩矩阵乘积，实际更新量还取决于训练过程。

如果 `alpha=8`，缩放系数就是 `8/8=1`，更新保持原始比例，不放大也不缩小。

如果 `alpha=4`，缩放系数就是 `4/8=0.5`，LoRA 的影响就**减弱**了。

---

### 七、Alpha与Rank的关系（重点）

记住这个核心公式：**LoRA 加入模型的更新量会被 `α/r` 缩放**。所以 `α/r` 是控制 LoRA 更新“强度”的关键缩放系数。

**但注意**：`α/r` 只是一个“缩放系数”，不是“唯一开关”。实际落到权重上的更新量，还取决于很多其他因素：

- **学习率**：它直接决定每步更新 `A`、`B`

## 块 26

## 七、Alpha与Rank的关系（重点）

### 配置1
```python
r=8
alpha=16
```
比例：`2`

### 配置2
```python
r=16
alpha=32
```
比例：`2`

**关键点**：虽然两个配置的rank不同（8 vs 16），但缩放比例相同（都是2）。

**实操建议**：调整rank时，常见做法是**同步调整alpha**，保持 `α/r` 比例大致稳定。这只是经验做法，不是硬性规则。`alpha = 2 × r` 只是常见起点，实际应结合验证集表现调整，也可以保持alpha固定、单独调r。

**动手验证**：把配置1改成 `r=8

## 块 27

### 方案1：小模型实验（推荐你从这开始）

**配置：**
```python
r=8
alpha=16
```

**特点：**
- 参数少，显存占用低；
- 训练速度快，适合快速验证想法；
- 最适合你目前“动手试”的学习节奏——先跑通流程，再调优。

---

### 动手小任务（基于方案1）

按上面的配置跑一次微调，记录训练时间和损失值变化。跑完后，把 `r` 改成 `16`、`alpha` 改成 `32`，对比两次训练的速度和效果差异。这个对比能帮你直观感受 `r` 和 `alpha` 对模型的影响。

---

### 代码参考（PEFT中的配置）

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,          # LoRA容量：决定

## 块 28

## 误区1：rank越大一定越好？

**错误。** 这是新手最容易踩的坑。你可能会想：“rank越大，模型能学的东西越多，效果肯定越好？”但事实并非如此。

**为什么？** 当你的任务本身比较简单时，大rank只会带来两个坏处：
- **浪费显存**——你的GPU在干无用功；
- **增加训练成本**——训练更慢，却未必更准。

**实操建议**：先从小rank（如8）开始，观察效果。如果模型表现不够好，再逐步增大。不要一上来就追求大rank。

---

## 误区2：只调rank，不调alpha

很多人改参数时只动rank，alpha保持不变。举个例子：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

这时缩放比例变成：

```text
16/64 = 0.25
```

**问题来了**：alpha没变，但rank变大了，`α/r` 从原来的2降到了0.25，缩放因子减小，但实际更新力度还取决于低秩矩阵BA的乘积，需综合考虑。

**正确做法**：调整rank时，alpha要跟着调整，保持 `α/r` 在一个合理范围（通常接近1~2）。

---

## 误区3：alpha越大越好？

**错误。** alpha过大（即 `α/r` 缩放过大）会导致：
- **更新过强**——模型参数被大幅改动；
- **训练不稳定**——loss可能震荡甚至发散。

**注意**：`α/r` 过大只是训练不稳定的**可能原因之一**。如果训练发散，还要排查学习率是否过高、数据质量、batch size是否过小、精度设置（fp16

## 块 29

**本块学习检查**：读完本节后，请合上资料，用自己的话回答：LoraConfig在LoRA训练流程中处于哪一步？它主要“告诉”PEFT哪四类信息？

---

## 一、LoraConfig是什么？

### 1.1 PEFT中的LoraConfig作用

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的**参数高效微调库**。它的核心思想是：不微调全部模型参数，只训练一小部分新增参数。

LoRA训练流程如下：

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

可以看到，**创建LoraConfig**是流程中的第二步，也是承上启下的关键一步——基础模型加载好后，由它来决定LoRA怎么“接”到模型上。

**LoraConfig** 是一个配置对象，它的职责是告诉PEFT四件事：

1. **LoRA放在哪里**——即插入位置；
2. **LoRA规模多大**——即参数量；
3. **LoRA训练强度如何**——即更新幅度；
4. **是否使用dropout**——即正则化设置。

简单说，**LoraConfig就是LoRA微调的“施工图纸”**。你写好图纸，PEFT照着图纸把LoRA适配器（Adapter）搭到模型上，然后开始训练。

---

**操作验证**：打开你的Python环境，

## 块 30

## 动手跑通你的第一个LoRA配置

你正在学LoRA微调，目标很明确：用最少的代码让大模型“听话”。下面这段代码就是LoRA的“启动开关”，先看它长什么样：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,                # LoRA矩阵的大小
    lora_alpha=16,      # 缩放强度
    target_modules=["q_proj", "v_proj"],  # 插入哪些层
    lora_dropout=0.05   # 随机失活比例
)
```

**这4个参数分别管什么事？**

| 参数 | 作用 |
|------|------|
| `target_modules` | 决定在模型的哪些层插入LoRA |
| `r` | 决定LoRA矩阵的尺寸，r越大可学习参数越多 |
| `lora_alpha` | 控制LoRA对原模型的影响强度 |
| `lora_dropout` | 训练时随机丢弃部分神经元，防止过拟合 |

---

### 从配置到模型：完整流程

光有配置还不够，你需要把它“装”到模型上。流程是这样的：

```
model → LoraConfig → get_peft_model() → LoRA模型
```

实际操作代码：

```python
from peft

## 块 31

## 3.1 target_modules 是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入低秩矩阵”**。

Transformer 模型里，Attention 结构通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

关键点：**LoRA 不会默认修改所有层**，你需要手动指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思就是：只给 **Query 投影层** 和 **Value 投影层** 加上 LoRA。

---

## 为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开看两个角色的分工：

- **Q（Query）**：决定模型**“关注什么信息”**
- **V（Value）**：决定模型**“传递什么信息”**

所以，调整 Q 和 V，通常就能有效改变模型的行为——这就是为什么它们是最常见的 LoRA 注入起点。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名可能不同，**配置前务必先用 `model.named_modules()` 查看你所用模型的实际模块名**，再决定 `target_modules` 写什么。

---

## 偷懒技巧：`target_modules="all-linear"`

如果你不想手工枚举模块名，较新版本的 PEFT（一般 ≥ 0.7，具体以你装的版本文档为准）支持直接写：

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
    target_modules="all-linear",   # 自动匹配（几乎）所有 nn.Linear；一般不含输出层 lm_head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**用它的代价**：覆盖的模块变多 → 可训练参数和显存占用都会增加。所以要在**参数量/显存**和**效果**之间权衡，最终以验证集结果为准。

如果你的 PEFT 版本太旧不支持这个写法，就退回**显式列出模块名**的方式，或者升级 PEFT。

---

## 🛠 本块操作验证

打开你的 Python 环境，加载你准备微调的模型（

## 块 32

### 方案1：只调整Q和V（你的轻量起点）

既然你的目标是快速上手LoRA微调，而且喜欢动手实践，那这个方案最适合你起步。

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么先选它？**
- **参数少**：只改动注意力机制里的查询（Q）和值（V）两部分，改动范围小。
- **训练快**：参数少意味着计算量小，跑起来更省时间，适合你每天1.5小时的练习节奏。
- **常见轻量起点**：很多任务用这个配置就能看到效果，不用一上来就追求复杂。

**动手小验证**：你可以先跑一个简单任务，对比只调Q/V和后面方案2、3的效果差异，感受“参数少”带来的速度优势。

---

### 方案2：调整全部Attention（更强表达）

```python
target_modules=[
    "q_proj",
    "k_proj",

## 块 33

## LLaMA系列：`target_modules`怎么设？

你在做LoRA微调，最关心的就是“到底该改哪些层”。对LLaMA系列，最常用的配置是：

```python
[
"q_proj",
"v_proj"
]
```

**为什么是这两个？** 简单说：`q_proj`是查询投影，`v_proj`是值投影，它们是注意力机制的核心部分。LoRA冻结原始权重W，仅训练低秩矩阵A和B，形成增量ΔW=(α/r)BA，其中r控制低秩维度（越小参数量越少），lora_alpha控制增量缩放比例。只改这两个投影就能高效适配下游任务，同时大幅减少可训练参数量。

**实操建议**：如果你是新手，先用这个组合跑通流程。它覆盖了大多数LLaMA微调场景，效果稳定且计算开销可控。

---

## 对比：GPT-2的差异（帮你理解原理）

GPT-2的写法不同，可能用：

```python
[
"c_attn"
]
```

**关键点**：GPT-2的注意力用`Conv1D`（不是普通`nn.Linear`），它把Q/K/V三个投影合并到一个`c_attn`模块里。GPT-2的`c_attn`使用`Conv1D`，其权重形状为`(out_features, in_features)`，与`nn.Linear`相同，但PEFT需要设置`fan_in_fan_out=True`以匹配其前向计算方式。

**所以**：在GPT-2上加LoRA时，需要设置`fan_in_fan_out=True`，PEFT才能按转置布局正确注入低秩矩阵。示例：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
# 注意：Llama-2-7b FP16 需要至少14GB显存

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_L

## 块 34

## BERT 的 target_modules 配置

BERT 和 LLaMA 的模块命名不同，**不能直接复制**别人的配置。BERT 通常只需要设置：

```python
["query", "value"]
```

但不同模型实现可能不同，所以**必须自己查看**实际加载的模型结构：

```python
for name, _ in model.named_modules():
    print(name)
```

运行后，在输出中找到类似 `q_proj`、`v_proj` 这样的名字，再填入 `target_modules`。

**匹配机制**：`target_modules` 按模块名做**后缀匹配**——写 `"q_proj"` 会命中所有以 `q_proj` 结尾的层；也可以写完整名字精确定位，或传正则表达式。

**注意**：不同 Transformers/PEFT 版本、不同模型实现，命名可能变化。请以你实际运行 `named_modules()` 的输出为准。

---

**动手验证**：加载一个 BERT 模型，运行上面的代码，把输出中前 10 个模块名抄下来，确认 `query` 和 `value`

## 块 35

## 6.1 r是什么？

r就是**rank（秩）**，它控制LoRA低秩矩阵的维度。

核心公式：**ΔW = (α/r)BA**

r决定的是A、B两个矩阵的**中间维度**。

**直观例子**：原矩阵是4096×4096，设置r=8后——

- A：8×4096
- B：4096×8

这样就把一个大矩阵拆成了两个小矩阵，大幅减少训练参数。

---

## 6.2 r影响什么？

**r越大**：
- ✅ 学习能力更强，能表达更复杂任务
- ❌ 参数增加，显存占用增加

**经验起始范围**（非固定配方，需用验证集调

## 块 36

## 7.1 alpha 作用——控制 LoRA 的“力度”

LoRA 微调时，模型权重的更新量不是直接用 `BA`，而是乘以一个缩放系数：

```
更新量 = (alpha / r) × BA
```

所以 **alpha 控制的是 LoRA 更新的幅度**——alpha 越大，LoRA 对模型的影响越强。

**动手算一下：**

```python
r = 8
lora_alpha = 16
```

缩放系数 = `16 / 8 = 2`，意味着 LoRA 的更新被放大了 2 倍，影响增强。

## 7.2 alpha 常见设置

一个常见的经验起点是：

```
alpha = 2 × r
```

例如：

## 块 37

**8.1 dropout在做什么？**

训练时，LoRA会**随机关掉一部分神经连接**。目的很简单：**减少过拟合**（模型死记训练数据，换新数据就失灵）。

配置写法：
```python
lora_dropout=0.05
```
意思是：**5%的LoRA路径会随机失活**，强迫模型不依赖某几条固定路径，学得更稳。

**8.2 怎么选数值？**

经验起点（不是死规则，要看验证集表现调整）：

| 数据规模 | dropout起点 |
|---------|------------|
| 数据少   | 0.05~0.1   |
| 数据多   | 0~0.05     |

如果你训练数据较少

## 块 38

## 9.1 基础配置：7B模型快速上手

这段配置是LoRA微调的“标准起手式”，特别适合7B规模模型测试。我们直接看代码，然后逐参数拆解。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量
    lora_alpha=16,          # 更新缩放
    target_modules=["q_proj", "v_proj"],  # 插入位置
    lora_dropout=0.05,      # 防止过拟合
    bias="none",
    task_type="CAUSAL_LM"   # 语言模型任务
)
```

**四个核心参数，逐个击破：**

- **`r=8`**：决定LoRA的“容量”。值越小，可学习的参数越少，训练越快，但模型表达能力受限；值越大，效果可能更好，但显存和计算开销增加。8是平衡之选。
- **`lora_alpha=16`**：控制“更新缩放”。它和`r`配合，实际缩放比例为`alpha/r = 16/8 = 2`。这个比例影响微调时权重更新的幅度。
- **`target_modules=["q_proj", "v_proj"]`

## 块 39

## 9.1 基础配置：跑通一个完整的LoRA微调流程

你现在的目标是**LoRA微调**，而且喜欢动手实践。这一节我们就直接上手，把代码跑通，亲眼看到LoRA是怎么工作的。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：BERT 采用 Apache 2.0 许可证，可自由商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，微调建议 8GB 以上
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里用的是BERT做**序列分类**任务（比如情感二分类）。注意：`num_labels=2`意味着分类头是**新初始化**的，它还没学过任何东西，所以后面必须让它参与训练。

### 第二步：配置LoraConfig——四个核心参数

```python
lora_config = LoraConfig(
    r=8,                # LoRA矩阵的秩r，决定低秩矩阵的维度，r越小，可训练参数越少（参数量为r×(d_in+d_out)），但表达能力可能下降
    lora_alpha=16,      # 缩放系数，控制LoRA更新的幅度
    lora_dropout=0.05,  # 防止过拟合的dropout比例
    bias="none",        # 不训练偏置项
    task_type="SEQ_CLS",# 任务类型：序列分类
    target_modules=["query", "value"],  # 只改BERT注意力中的query和value层
    modules_to_save=["classifier"],      # 关键！把新分类头也纳入训练并保存
)
```

**重点理解`modules_to_save`**：BERT原本的分类头被替换成了新的随机初始化分类头。如果不加这一行，LoRA只会训练query/value的低秩矩阵，分类头**不会被训练**，模型自然无法正确分类。加上它，分类头才会参与训练。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 应看到 LoRA 参数 + classifier 一起可训练
```

运行这行代码，你会看到可训练参数包含两部分：LoRA引入的低秩矩阵参数 + 分类头参数。这就是LoRA"参数高效"的体现——大部分原始BERT参数被冻结，只训练一小部分。

### 第四步：保存与重载——验证分类头没丢

训练完成后（代码中省略），保存并重新加载：

```python
# 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")

# 重新加载并验证分类头确实被带回（而不是随机初始化）
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证：重载后的分类头权重应与保存前一致
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

**为什么必须验证？** 如果忘记加`modules_to_save=["classifier"]`，重载时分类头会退回基础模型的**随机初始化**状态，分类结果完全错乱。`tor

## 块 40

**10.1 普通文本生成（动手配置）**

先别急着背参数，我们直接看代码。对于**普通文本生成**任务（比如续写、生成故事），`LoraConfig` 推荐这样配：

```python
r=8
alpha=16
dropout=0.05
```

**逐个拆解，你马上就能上手：**

- **`r=8`**：这是 LoRA 的“秩”，即低秩矩阵的秩，决定增量矩阵的维度。`r` 越小，微调时参与更新的参数越少，训练越快、显存占用越低；`r` 越大，模型能学到的“新东西”越多，但开销也越大。普通文本生成任务，`8` 是个不错的起点。
- **`alpha=16`**：这是**缩放因子**，控制增量更新的幅度。`alpha` 通常设置为 `r` 的 1~4 倍，常见取 2 倍，这里的 `16` 就是 `8` 的两倍，但需根据任务调整。
- **`dropout=0.05`**：这是**随机失活率**。在训练时随机将 LoRA 低秩矩阵的部分元素置零，防止模型死记硬背训练数据（也就是防止**过拟合**）。`0.05` 是个很轻的 regularization，适合大多数情况。

**为什么这么配？** 普通文本生成对“改变模型风格”的要求不高，更看重稳定。所以用较小的 `r` 和 `alpha`，让模型在原有能力基础上做**微调**，而不是“大改”。

---

**动手小任务（不引入新知识，帮你巩固）：**

打开你的 Python 环境，新建一个 `LoraConfig`，把上面三个参数填进去。然后打印出这个配置对象，观察一下输出。确认你能在输出中找到 `r=8`、`lora_alpha=16`、`lora_dropout=0.05` 这三个值。如果都能找到，说明你已经掌握了最基本的配置写法。

---

**关于电脑配置，给你个底：**

- **只写配置代码**（不加载模型）：**CPU 就

## 块 41

## 10.1 普通文本生成：LoRA核心配置

你正在学习LoRA微调，目标是掌握最常用的配置。下面这段代码就是普通文本生成任务的标准起点：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**逐个拆解，动手理解：**

- **`target_modules`（改哪里）**：指定应用LoRA的模块，这里选了 `q_proj` 和 `v_proj`（注意力机制中的查询和值投影）。LoRA冻结这些模块的原始权重，仅训练低秩矩阵A和B形成增量更新。
- **

## 块 42

## 一、为什么需要QLoRA？

你已经掌握了LoRA的核心思路：冻结原模型参数，只训练少量低秩矩阵。但这里有个关键问题——**LoRA并没有减少“加载模型”的显存开销**。

### 动手算一算：7B模型的显存账

假设我们要微调一个70亿参数的模型，用FP16精度存储：

```
7B × 2 bytes ≈ 14GB显存
```

这14GB只是**把模型放进显存**的“入场费”。实际训练时，显存里还躺着：

- **激活值**：前向传播的中间结果
- **梯度**：反向传播计算出的更新方向
- **优化器状态**：如Adam的动量项

把这些都加起来，实际显存需求会**远高于14GB**。这就是为什么即使有了LoRA，消费级显卡（比如8GB或16GB显存）依然跑不动大模型微调。

### 核心矛盾

LoRA解决了“训练多少参数”的问题，但没解决“加载多大模型”的问题。**模型本体依然以完整精度躺在显存里**，这是显存压力的主要来源。

### 本块学习检查

打开你的训练脚本，找到模型加载那行代码。观察：模型加载时

## 块 43

**QLoRA核心思想（动手版）**

你的目标是LoRA微调，但显存有限。QLoRA就是解决办法：**先把基础模型压缩成4-bit，再插入LoRA训练**。

**对比一下：**
- LoRA：FP16模型 + LoRA
- QLoRA：4-bit模型 + LoRA

**训练时谁动谁不动？**
- 基础模型：4-bit权重，**冻结**
- LoRA：FP16/BF16参数，**训练**

**关键点：** 量化只针对基础模型权重，**不是** LoRA参数。

**流程：**
FP16模型 → 4-bit量化 → 加载量化模型 → 插入LoRA → 训练LoRA参数

**模型许可证

## 块 44

## 三、为什么需要4-bit量化？——先看模型怎么“存”

咱们先搞清楚一件事：模型文件在硬盘上、显存里，都是按“每个参数占多少字节”来算的。不同精度，占的地方差很多：

| 类型 | 每个参数占用 |
|------|------------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下**：7B模型（70亿参数），用FP16存，就是 `7B × 2 ≈ 14GB`；换成4-bit，`7B × 0.5 ≈ 3.5GB`——显存直接省了约75%。

**但注意**：这个3.5GB只是**理论下界**，只算了量化后的权重本身。实际跑起来会更高，因为还要存量化常数/缩放因子、没量化的模块（norm、embed

## 块 45

**NF4是什么？**

NF4（NormalFloat4，正态分布4位浮点）是QLoRA论文为神经网络权重设计的4-bit量化数据类型。注意：它不是IEEE 754那种通用浮点格式，而是一套**针对近似正态分布权重的离散码本**。

**核心机制**：预先设定16个（2⁴）量化取值点，这些点按标准正态分布的分位数排布——权重密集的0附近划分更细。量化时，将每组权重归一化后映射到最近的码本值，保存4-bit索引；反量化时按码本与缩放因子还原。

**一句话理解**：它是为“权重集中在0附近”这一特点量身定制的查表量化方案，而非通用浮点表示。

**动手验证**：想象一个权重分布直方图——中间高、

## 块 46

**4.2 为什么适合模型权重？**

大模型训练后的权重近似**正态分布**，大量参数集中在0附近。普通INT4均匀覆盖[-8,7]共16个整数，但无法精细表示中心密集区域。NF4基于分位数量化，使量化值在标准正态分布下期望平方误差最小，从而在0附近精度更高，更贴合权重分布，精度损失更低。

**4.3 NF4量化过程**

FP16权重（如0.023、-0.145、0.876）→ 归一化 → 映射到NF4 → 保存4-bit值。推理时反量化回近似FP16。

**动手验证**：用你的LoRA脚本，打印微调前后权重直方图

## 块 47

**动手理解：为什么量化参数也要“瘦身”？**

你正在学LoRA微调，目标是让大模型跑起来。QLoRA用4-bit量化把权重压缩，但**量化本身也要存“说明书”**——每组权重需要保存`scale`（缩放因子）和`zero point`（零点）。这些额外参数同样占显存，模型越大，累积越可观。

**双重量化：给“说明书”再压缩一次**

普通量化流程是：FP16权重 → 4-bit权重 + scale参数。

双重量化的思路很直接：**既然scale参数也占地方，那就把它也量化一遍**。第一次量化模型权重，第二次量化量化参数本身。

结构上就是：

- FP16权重 → NF4量化 → 保存4-bit权重
- 同时，scale参数 →

## 块 48

**本块学习检查**：读完本节后，请用自己的话回答：分页优化器解决的是“显存不够用”还是“显存突然不够用”？它靠什么机制实现？

---

训练大模型时，显存不只是模型本身在占用。梯度、优化器状态（比如 Adam 的一阶动量和二阶动量）都会挤占显存。QLoRA 虽然把模型压小了，但训练中仍可能遇到显存突然不足（OOM）的情况。

分页优化器的思路，是借用操作系统虚拟内存的做法。它依靠 NVIDIA 的统一内存（Unified Memory），在显存出现峰值、快要 OOM 时，把优化器状态按“页”自动换出到 CPU 内存；等需要参与计算时，再换回 GPU。

注意：这不是“平时就把优化器状态放在 CPU 里省显存”。它的重点是应对**峰值**——平稳时状态尽量留在 GPU，只在显存不足的瞬间临时换出，避免偶发峰值直接中断训练。

## 块 49

## 6.3 类比理解：分页优化器在做什么？

把显存想象成你的**桌面**。

**普通优化器**的做法是：把所有东西都摊在桌面上。桌面一旦堆满，就只能报错——这就是显存溢出（OOM）。

**分页优化器**的做法是：桌面只放**常用文件**，暂时用不到的**存进柜子**，等需要时再取出来。这样桌面永远有空间，训练就不会因为显存爆掉而中断。

---

## 6.4 实际训练中怎么启用？

你**不需要手写换页逻辑**，`bitsandbytes` 库已经帮你实现了。只需在 `TrainingArguments` 里设置 `optim` 参数：

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
    optim="paged_adamw_8bit",        # 关键：使用分页优化器
)
```

**两个常用取值：**
- `paged_adamw_8bit`：优化器状态用 8-bit 存储，更省显存
- `paged_adamw_32bit`：精度更高，但占用更多显存

**注意：** 不同版本的 Transformers / bitsandbytes 支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报"未知 optim"错误，查一下你当前版本的文档，换用支持的名称即可。

**使用技巧：** 分页优化器主要解决**峰值 OOM**，建议搭配 `gradient_checkpointing=True` + 小 `per_device_train_batch_size` + 梯度累积一起用，显存控制效果最好。

---

## 动手验证

打开你的训练脚本，检查两件事：
1. `optim` 参数是否设置为 `paged_adamw_8bit` 或 `paged_adamw_32bit`？
2

## 块 50

### 7.1 4-bit量化：给模型“瘦身”

你正在学LoRA微调，核心目标是在有限显存里跑起更大的模型。QLoRA的第一招就是**4-bit量化**——把模型权重从FP16（16位）压缩到NF4（4位），体积直接缩小约4倍。想象一下，原本占4个格子的行李，现在塞进1个格子，显存压力瞬间减轻。

### 7.2 双重量化：连“尺子”也压缩

量化时需要一把“尺子”（scale）来把数值映射回原范围。这把尺子本身也占空间。**双重量化**就是把这把尺子再量化一次，进一步减少额外开销。相当于行李箱压缩后，连标签也换成更小的。

### 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态很吃显存。**分页优化器**让GPU显存和CPU内存之间自动“换页”——显存紧张时，把暂时

## 块 51

## 4-bit 量化配置：让大模型“瘦身”跑起来

你在做 LoRA 微调，目标是把大模型跑起来。QLoRA 的核心思路是：**权重用 4-bit 存，算的时候再变回高精度**。这样显存占用大幅下降，但计算精度不受太大影响。

下面拆解 `BitsAndBytesConfig` 里的四个关键参数，按“存什么、怎么存、怎么算”的逻辑串起来。

### 1. `load_in_4bit=True` —— 开启 4-bit 加载

这是总开关。打开后，模型权重不再以 32-bit 或 16-bit 加载进显存，而是直接以 4-bit 格式存放。**省显存的第一步**。

### 2. `bnb_4bit_quant_type="nf4"` —— 用 NF4 量化

NF4（NormalFloat4）是一种 4-bit 量化格式，专门为神经网络权重分布设计。你不需要理解它的数学细节，只需知道：**NF4 是 QLoRA 推荐的量化方式**，在 4-bit 精度下尽量保留权重信息。

### 3. `bnb_4bit_use_double_quant=True` —— 双重量化

开启后，连量化过程中产生的“缩放因子”也会被量化一次，进一步压缩显存。**一句话：省上加省**。

### 4. `bnb_4bit_compute_dtype` —— 计算精度（最容易踩坑）

这里要区分两个概念：

- **存储 dtype**：权重以 4-bit NF4 格式存在显存里，省空间。
- **计算 dtype**：真正做矩阵运算时，权重会被**反量化**回更高精度再算。这个计算精度由 `bnb_4bit_compute_dtype` 决定。

**不设置会怎样？** 默认通常是 `float32`，计算慢、显存开销高。所以 QLoRA 配置里**必须显式指定**。

**选 BF16

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么用 4-bit 加载模型）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们**拼成一个能跑的最小骨架**，让你直观看到 QLoRA 训练从加载到训练的完整链路。下面代码中模型和数据路径请替换为你自己的。

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
# 模型许可证：请确认所选模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：以 Llama-2-7b 为例，FP16 加载至少需要14GB显存；4-bit 量化后显存需求显著降低

# 1) 4-bit 量化配置（按 GPU 能力选择计算 dtype）
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

# 2) 以 4-bit 加载基础模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 3) 关键：为 k-bit 训练做准备（处理 norm/输出层精度、并开启梯度检查点所需的输入梯度等）
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # 与 TrainingArguments 的 gradient_checkpointing 对应
)

# 4) 注入 LoRA（冻结原权重，仅训练低秩矩阵A和B形成增量更新）
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,          # 旧版本用 tokenizer=tokenizer
)

trainer.train()
```

### 骨架的四个关键衔接点

**① 量化配置 → 模型加载**：`BitsAndBytesConfig` 里 `load_in_4bit=True` 配合 `nf4` 量化类型和 `double_quant`，让模型以 4-bit 精度加载。`compute_dtype` 决定计算时用的精度，优先 bfloat16，不支持则退回 float16。

**② 4-bit 模型 → k-bit 训练准备**：`prepare_model_for_kbit_training` 是**必须的一步**。它处理两件事：一是让 norm 层和输出层保持较高精度（避免低精度影响训练稳定性），二是开启梯度检查点所需的输入梯度保存。这里 `use_gradient_checkpointing=True` 要和后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 对应。

**③ 准备后的模型 → LoRA 注入**：`get_peft_model` 把 LoRA 适配器加到目标模块（这里是 `q_proj` 和 `v_proj`）。`r=8` 是低秩维度，`lora_alpha=16` 是缩放系数

## 块 53

## 把 QLoRA 的五个关键步骤串起来（可跑通骨架）

你之前已经了解了 NF4 量化、双重量化和分页优化器的原理，现在我们把它们**真正组装成一个能跑的 QLoRA 训练骨架**。记住：这几步缺一不可，顺序也不能乱。

### 五步串联，缺一不可

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么被加载**

这一步是"地基"。它决定了基础模型以 **4-bit 方式加载**，具体包含三件事：用 **NF4** 做量化、开启**双重量化**、设定**计算 dtype**（比如 BF16/FP16）。换句话说，模型加载成什么样，全看这个配置。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的必要一步**

这是最容易跳过的坑。它的作用是**对量化模型做适配**：冻结模型参数、启用梯度检查点、强制输入梯度等。**跳过它，常常导致训练不收敛或直接报错**。所以这一步不是可选项，是必选项。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

这一步把**可训练的 LoRA** 注入到**冻结的 4-bit 主干**里。主干保持 4-bit 不动，只有 LoRA 参数参与训练——这正是 QLoRA 省显存的核心。

**第 4 步：梯度检查点**

在 `TrainingArguments` 里设置 `gradient_checkpointing=True` 即可。若使用旧版本 PEFT，可保留 `prepare_model_for_kbit_training` 中的 `use_gradient_checkpointing=True`，但需注意版本兼容性。

**第 5 步：`optim="paged_adamw_8bit"` —— 分页优化器**

设置这个优化器，启用**分页机制**来缓解峰值显存。当显存不够时，优化器状态可以被"分页"到 CPU 内存，避免 OOM。

> ⚠️ **重要提醒**：上面这个骨架是"能说明关系"的最小示例。真实训练时，你需要换成**足够规模的数据**、合理的超参数，并且**按你所用版本核对** `optim` 名称、`prepare_model_for_kbit_training` 的参数、以及 `processing_class`/`tokenizer` 的写法。版本不同，写法可能有差异。

---

### 你的电脑能跑吗？—— 显存参考

**只学原理**：完全不需要 GPU。

**实际运行 QLoRA**，最低配置参考：

- **CPU**：4 核以上
- **内存**：16GB 以上
- **GPU**

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者以为LoRA训练完会得到一个全新的模型文件，其实不是这样。

训练过程中，**基础模型是被冻结的**（不参与训练），真正被训练的是**LoRA Adapter**（适配器）。最终的结构是：

```
基础模型 + LoRA Adapter
```

举个例子：你用Llama-7B做LoRA微调，训练完成后保存的通常只有两个文件：

- `adapter_model.safetensors`（Adapter权重）
- `adapter_config.json`（Adapter配置）

**而不是**完整的Llama-7B模型文件。

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（相比动辄几GB的完整模型，Adapter通常只有几十到几百MB）
- 保存快
- 可以多个任务共享一个基础模型

比如同一个基础模型，可以搭配：

- 客服Adapter
- 医疗Adapter
- 法律Adapter

**缺点：**

部署推理时，需要**基础模型 + Adapter**两部分同时存在，缺一不可。

---

## 动手验证一下

检查你训练输出目录，确认是否只包含 `adapter_model.safetensors` 和 `adapter_config.json` 这两个文件，而不是完整的模型文件。

## 块 55

**动手理解：为什么不能直接拿LoRA模型去部署？**

你训练完LoRA后，模型其实是“两部分”组成的：

1. **基础模型**（Base Model）
2. **LoRA适配器**（Adapter）

推理时，数据要**依次经过**这两部分才能得到输出：

```text
输入 → 基础模型 + LoRA适配器 → 输出
```

这意味着，每次使用前你都得**分别加载**这两个东西，非常麻烦。

---

**部署时的痛点**

部署场景下，你真正想要的是**一个完整的模型**，直接输入就能出结果：

```text
输入 → 完整模型 → 输出
```

所以，必须把LoRA的权重**合并**回基础模型里。

---

**合并的数学

## 块 56

## 3.1 merge_and_unload() 是干什么的？

你现在已经训练好了LoRA，但模型还是“基础模型 + 额外适配器”的组合状态。`merge_and_unload()` 就是**把LoRA学到的权重真正写进基础模型里，然后拆掉外面的包装**。

用公式看就是：

```text
基础模型 + LoRA适配器 → 完整模型
```

执行后，你得到的就不再是 `PeftModel`，而是一个**普通Transformer模型**，它的权重已经包含了LoRA更新后的结果。

---

## 合并完整流程（先看全局）

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面按步骤走。

---

## 加载LoRA Adapter

假设你有基础模型路径和Adapter路径：

```python
base_model_path = "model_path"
adapter_path = "adapter_path"
```

加载基础模型，再挂上Adapter：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

> 注意：请确认你所使用模型的许可证类型。例如，Llama 2 采用 Llama 2 Community License，商用且月活用户超过7亿时需获得Meta授权。加载模型前请自行核实并遵守相应许可条款。
> 注意：加载模型需满足显存要求。例如，Llama-2-7b 以 FP16 精度加载至少需要14GB显存，请根据实际模型大小和精度评估你的硬件条件。

此时模型结构是：

```text
Base Model + LoRA Adapter
```

---

## 执行合并（核心一步）

```python
merged_model = model.merge_and_unload()
```

这一步完成的是：

```text
W' = W + (α/r)BA
```

`W` 是原来的权重，`BA` 是LoRA学到的增量，`α/r` 是LoRA的缩放系数，合并时需按此缩放增量，合并后得到更新后的 `W'`。

此时 `merged_model` **不再是PeftModel**，而是普通Transformer模型。

---

## 导出合并后的模型

保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后目录大致长这样（示意）：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：实际文件会随模型/分词器类型变化。比如可能有 `generation_config.json`、`special_tokens_map.json`；大模型权重可能分片（`model.safetensors.index.json` + 多个分片）；有些分词器是 `vocab.json`/`merges.txt` 或 `tokenizer.model`。

---

## 动手验证

合并完成后，你可以检查一下 `merged_model` 的类型，确认它不再是 `PeftModel`，而是一个普通模型对象。这是验证合并

## 块 57

## 三、merge_and_unload 是什么？（重点）

简单说，**merge_and_unload 就是把 LoRA 训练出的"小补丁"（Adapter）永久焊接到原模型上**，得到一个完整的、独立的模型文件。之后推理时，你不再需要 PEFT 库，直接用 HuggingFace 的 Transformers 就能加载。

### 合并后怎么加载？

合并完成后，加载方式变得非常干净：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

用户输入 → 完整模型 → 生成结果，就这么简单。

### 合并 vs 不合并，怎么选？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

**核心权衡**：合并后部署简单，但如果你需要**多个任务动态切换 Adapter**，或者你的推理框架原生支持 LoRA，那就不必合并。QLoRA 场景下还要考虑精度接受度。

### 合并后必查清单（避免"能加载但行为不对"）

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中新增/修改过的特殊 token（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。
2. **generation_config.json**：如果依赖特定生成默认值（eos/pad token id、`max_new_tokens`、采样参数等），确认它一并导出，否则推理时可能回退到不合适的默认行为。
3. **dtype**：确认合并/保存时的精度符合预期（FP16/BF16 还是 FP32），dtype 不一致会影响显存占用与数值表现。
4. **config.json 完整性**：确认模型结构配置正确，且已**移除 PEFT/adapter 相关配置**（合并后应是普通 Transformers 模型，不应再残留 `adapter_config.json`）。

### 动手验证

合并后，用 `print(model

## 块 58

**本块学习检查**：读完下面两个场景后，请用自己的话回答：在“上线部署”和“分享模型”两种情况下，合并 LoRA 各有什么好处？什么情况下不合并反而更灵活？

---

**10.1 推荐合并场景**

你正在学 LoRA 微调，最终目标是把模型用起来。什么时候该把 LoRA 权重合并回基础模型？记住两个典型场景：

**场景1：上线部署（追求简单）**

想象生产环境：服务器 → 直接加载模型 → 提供API。如果你只需要**一个固定的 Adapter**，并且希望部署链路尽量简单——不额外依赖 PEFT、不做运行时加载——那么合并通常更方便。

> 但注意：如果你的推理框架本身支持直接加载 LoRA（比如 vLLM 的多 LoRA 服务），或者你需要在同一基础模型上**热切换多个 Adapter**，那么不合并、以 Adapter 形式部署反而

## 块 59

## 10.2 不建议合并场景

**场景1：多个任务Adapter**

基础模型上挂了多个Adapter（A、B、C），各自对应不同任务。**保留Adapter更灵活**——想切换任务时直接换Adapter就行，不用重新加载模型。

**场景2：持续训练**

如果你还要继续调LoRA，**保留Adapter**。合并后如需继续微调，需重新加载基础模型并重新训练LoRA适配器。

---

## 十一、QLoRA模型合并注意事项（重点）

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

QLoRA训练时，基础

## 块 60

## 动手实操：QLoRA 模型合并，先避开这个坑

你正在学 LoRA 微调，走到合并模型这一步了。这里有个关键提醒：**别直接在 4-bit 量化状态下合并**，否则可能踩坑。

### 直接合并会遇到什么？

如果你在 4-bit 状态下直接调用 `merge_and_unload()`，可能出现两种情况：

1. **精度损失**：在4-bit量化状态下直接调用`merge_and_unload()`，PEFT会自动处理反量化并合并，但合并后的权重会保留量化误差，可能影响模型质量。若需更高精度，可先以高精度加载模型再合并。
2. **报错风险**：权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> 注意：上面说的是“可能出现”，不是每次都必然报错。能不能顺利合并、合并质量如何，取决于你的**版本**、量化配置，以及是否先做了反量化。以你所用版本的文档和实际运行结果为准。

### 稳妥做法：先反量化到高精度再合并

推荐流程很简单——**别在 4-bit 下硬刚，先换成高精度再合并**：

```
用 FP16/BF16 重新加载基础模型
        ↓
加载同一份 LoRA Adapter
        ↓
merge_and_unload()
        ↓
保存完整模型
```

这样合并发生在高精度权重上，绕开了 4-bit 状态下的类型和精度问题。

### 可直接运行的代码

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

### 本块学习检查

对照代码确认你理解了三个关键点：

1. **为什么第 1 步不传 `quantization_config`？** —— 因为合并必须发生在高精度权重上，而不是 4-bit 量化权重上。
2. **`safe_merge=True` 的作用是什么？** —— 合并时做数值检查，比如检测 NaN。
3. **模型许可证**：部署引用模型时需标注许可证类型，例如：# 模型许可证：Llama 2 Community License（商用

## 块 61

**动手前先看三点（QLoRA合并）**

你是要跑LoRA微调，这里直接关系到你最后能不能拿到好模型。记住三个关键点：

1. **版本决定能不能直接合并**：不同PEFT版本对“量化模型能否直接合并”支持不一样，有的要求先反量化。**最稳的做法**：加载基础模型时**不要传** `quantization_config` 或 `load_in_4bit`，用高精度权重再合并。动手前先查你所用版本的 `merge_and_unload` 文档。

2. **内存要够**：高精度重新加载完整模型，显存占用约等于一份FP16/BF16完整模型。显存不够？把 `device_map` 改成 `

## 块 62

**动手实操：合并后一定要“验一验”**

`merge_and_unload` 可以加一个保险参数 `safe_merge=True`，合并时自动做数值检查（比如检测有没有出现 NaN），防止模型“悄悄坏掉”：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完别急着用，**先验证一致性**：拿同一小批固定输入，分别用合并前的 PeftModel 和合并后的模型推理，对比输出是否基本一致。

**最小验证流程**：固定 prompt → 两个模型分别推理 → 对比 logits 最大绝对差，再对比贪心生成文本是否一致。

**本块学习检查**：跑一遍上面的验证代码，记录 logits 最大绝对差和生成文本

## 块 63

## 合并后别急着走：先做一次“前后对比”验证

你已经完成了 QLoRA 模型的合并（§11.2 的 `merge_and_unload()`），现在进入最后一道建议工序：**验证合并前后模型行为是否一致**。这一步能帮你发现合并过程中可能引入的数值偏差，避免“模型能跑但输出已悄悄变化”的隐患。

### 为什么需要单独加载一份“合并前”的模型？

关键点：`merge_and_unload()` 是**原地操作**——它直接修改并返回底层模型，合并后原来的 `PeftModel` 对象已经不再是“未合并”状态。所以你不能拿 §11.2 里那个被合并过的对象当“合并前”的基准。

正确做法是：**重新加载一份全新的、未合并的 `PeftModel`**，作为对比基准。代码中通过 `base_for_ref` 重新加载基础模型，再挂上 adapter，得到 `model_before`。

### 验证分两步：数值层面 + 文本层面

**第一步：比较 logits（数值层面）**

用同一个固定输入（示例中的 `prompt`），分别让 `model_before`（未合并）和 `merged_model`（已合并）前向传播，计算两者 logits 的**最大绝对差**：

```python
max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

- 结果**越接近 0 越好**。
- 注意：QLoRA 场景下，由于量化精度损失，这个值通常是**较小的近似值**，不是精确的 0。只要数量级很小（比如 1e-3 以下），就说明合并质量可接受。

**第二步：比较生成文本（行为层面）**

数值接近不代表生成结果一致，所以还要做一次**贪心生成对比**（`do_sample=False` 去掉随机性，保证可复现）：

```python
gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后:", text_after)
print("生成文本是否一致:", text_before == text_after)
```

如果 `text_before == text_after` 为 `True`，说明合并前后模型在贪心解码下行为一致，可以放心使用合并后的模型。

### 实操建议

1. **选一个有代表性的固定输入**：不要用“你好”这种太短的，选一个能触发模型完整推理链的 prompt（比如你任务里的典型问题）。
2. **先跑数值对比，再跑文本对比**：数值差异过大时，先排查合并配置（如 dtype 是否一致），不必急着看文本。
3. **记住 QLoRA 的预期**：合并后模型用于部署时，请注意：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权；# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 64

## 合并后如何确认没“合坏”？——QLoRA 合并校验实操

### 为什么必须“再加载一份”？

`merge_and_unload()` 返回合并后的新模型，原始模型保持不变。若需对比，可直接使用原始模型与合并后的模型进行一致性验证。

**正确做法**：在合并前保存原始模型的副本作为对照，再和合并后的模型做对比。

### 显存不够怎么办？

如果同时放不下两个模型，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本**存下来**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

### 结果预期：别追求“一模一样”

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 不同。  
  **重点**：确认没有明显退化、没有 NaN，而不是要求逐位相同。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。

是否可接受？**以你的任务指标和验证集为准**。

---

## 电脑配置要求（本章节需要实际加载模型）

**最低配置**：
- CPU：4核+
- 内存：16GB+
- 磁盘：预留模型大小 **2倍** 空间

**GPU 经验参考（

## 块 65

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

你是动手实践型学习者，目标是把LoRA微调跑通并部署。这一节我们聚焦一个关键动作：**合并时别“闭眼”操作，要开校验、做验证**。

#### 为什么必须做校验？

QLoRA合并失败或结果异常，常见原因不止一种：在4-bit状态直接合并、PEFT/bitsandbytes版本不支持、量化包装未正确反量化、内存/显存不足。这些问题不会提前报错，可能等你部署上线才发现模型输出是乱的。所以，**合并时开启`safe_merge=True`，并做一致性验证**，是建议的保险动作。

#### 核心操作：先“反量化”再合并

这里有个容易误解的点——“反量化”**不是**手动逐层还原4-bit权重，而是：

**重新加载一份不带量化配置的基础模型。**

具体做法：
1. 用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)` 加载高精度权重——关键：**不传** `quantization_config` 或 `load_in_4bit`。
2. 再用 `PeftModel.from_pretrained` 挂载**同一份Adapter**。
3. 然后执行合并。

这样合并发生在高精度权重上，天然避开了“在4-bit状态下合并”的类型/精度问题。

#### 版本兼容性提示

不同PEFT / bitsandbytes / Transformers版本对“量化模型能否直接调用`merge_and_unload`”的支持不同：
- 部分版本直接报错或给出警告；
- 部分较新版本对某些情形有一定支持，但仍可能有精度损失。

因此，**“先反量化（重载高精度）再合并”是跨版本最稳妥的做法**。是否可以在量化状态下直接合并，请以你所装版本的`merge_and_unload`文档与实际运行结果为准。

#### 常见问题速查

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload`不存在 | 当前对象不是PeftModel（可能忘了用`PeftModel.from_pretrained`挂载Adapter，或已合并/加载成普通模型） | 确认加载方式，先得到PeftModel再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积Adapter | 无需处理；若在意体积可保留Adapter不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是dtype用了FP32、未做offload/分片 | 改用CPU合并或offload、用FP16/BF16、分片加载 |

## 块 66

### 路线A：合并后部署完整模型

既然你的目标是跑通 LoRA 微调并最终落地部署，我们先看最直接的一条路：**把训练好的 LoRA 合并回基础模型，然后当普通模型用**。

流程很简单，三步：

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

**这么做的好处**：

- 部署链路最简单。推理框架（比如 vLLM）看到的就是一个普通模型，它完全不需要知道 LoRA 是什么，直接加载就行。
- 推理时没有 Adapter 切换的开销，性能更稳定。

**代价**：

- 每合并一个 LoRA，就要保存和分发一份完整的模型文件。如果你有多个任务（多个 Adapter），存储成本会直线上升。

**模型许可证**：合并后的模型继承基础模型的许可证。若基础模型为 Llama 2，则适用 Llama 2 Community License（商用且月活用户>7亿时需Meta授权）。

---

### 路线B：保留 Adapter，用支持 LoRA 的框架动态服务

如果你有多个任务，或者要服务多个用户，更省资源的是**保留 Adapter 不合并**，用支持 LoRA 的推理框架（如 vLLM）在运行时动态切换。

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

**这么做的好处**：

- 基础模型只存一份，多个 Adapter 都是小体积文件，显存和存储都省。
- 支持热切换，不用为每个任务单独存一份完整模型。

**代价**：

- 依赖框架对 LoRA 的支持，配置比直接加载合并模型复杂一些。
- 不同框架、不同版本对 LoRA 的支持范围和限制不一样，需要查文档确认。

---

#### vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

vLLM 启动 OpenAI 兼容服务时，通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**。命令行大致长这样：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA/ adapterB=/path/to/adapterB/ \
    --max-loras 2 \
    --max-lora-rank 16
```

参数含义：

- `--enable-lora`：开启多 LoRA 服务。
- `--lora-modules 名字=路径`：声明可用的 Adapter。**注意**：这里的路径是训练保存下来的 Adapter 目录（含 `

## 块 67

## 14.3 怎么选？两条路线对照

你现在面临一个实际选择：训练完 LoRA 后，是**合并**进基础模型，还是**保留 Adapter** 动态加载？记住一句话：**合并不是唯一部署方式**。

**什么时候选合并（路线A）？**
- 你只训练了**一个 Adapter**，想要最简部署——一个模型文件搞定，不用额外加载步骤。

**什么时候选保留 Adapter（路线B）？**
- 你打算在**同一个基础模型**上服务**多个 Adapter**，而且你的推理框架（比如后面要学的 vLLM）**原生支持 LoRA**。这样切换任务时不用加载多个完整模型，省显存、更灵活。

**特别提醒（QLoRA 用户）**：如果你用的是 QLoRA，精度比较敏感。**无论走哪条路线**，上线前都建议做一次合并前后的一致性验证，确保结果没跑

## 测评题目

### Q-KB12-ST1204-003
分页优化器的主要目标是？

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB12-ST1202-003
在其他条件相同时，增大 LoRA rank 最可能带来什么？

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。
