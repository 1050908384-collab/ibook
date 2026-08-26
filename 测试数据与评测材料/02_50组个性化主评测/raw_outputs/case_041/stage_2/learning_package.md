# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练的方式：

```
输入数据 → 整个模型参数更新 → 保存完整模型
```

以7B参数模型为例——也就是70亿个参数。训练时，你需要同时保存三类东西：

- **参数本身**：70亿个数值
- **梯度**：每个参数对应的更新方向，又是70亿个数值
- **优化器状态**：Adam等优化器维护的动量信息，通常是参数量的2倍

三者加起来，显存需求非常高。普通消费级GPU根本扛不住。

**这就是为什么需要参数高效微调**——与其动整个模型，不如只训练一小部分参数，大幅降低显存门槛。LoRA就是这类方法的代表，也是你本次微调实验要用的核心技术。

---

## 块 2

**PEFT：只训练“小配件”，不碰大模型**

全参数微调要动整个大模型，显存需求非常高。PEFT（参数高效微调）的核心思想是：**不修改整个大模型，只训练少量新增参数**。

流程上，你在原始大模型上叠加一个 **LoRA适配器参数**，然后**只训练LoRA参数**，最后**保存小规模Adapter**（适配器）即可。

三种方式对比：

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

## 块 3

**LoRA是什么？（①）**

LoRA，全称 Low-Rank Adaptation（低秩适配），是一种参数高效微调方法。它的核心思路是：**冻结原模型权重，只训练新增的小矩阵**。

原模型权重记为 `W`，训练时**不更新**。我们新增一个增量 `ΔW`，最终模型变为：

```text
W + ΔW
```

其中 `ΔW = B × A`。这里 `A` 和 `B` 是两个低秩小矩阵：若 `W` 形状为 `d_out × d_in`、秩为 `r`，则 `A` 形状为 `d_in × r`，`B` 形状为 `r × d_out`。两者相乘后，

## 块 4

## 3.1 LoRA的问题：省了参数，没省显存

你已经知道LoRA能大幅减少**训练参数**，但这里有个关键点要拎清：**基础模型本身仍然要完整加载进显存**。

以7B模型为例，如果用FP16精度加载，大约需要 **14GB显存**。也就是说，就算LoRA只训练少量参数，这14GB的“底料”一分也省不掉。显存不够，LoRA照样跑不起来。

## 3.2 QLoRA的解法：把“底料”压缩

QLoRA（Quantized LoRA）的思路很直接：**既然基础模型省不掉，那就把它压缩了再加载**。

流程是：

```text
FP16模型 → 4bit量化 → 显存占用下降 → 加载LoRA进行训练
```

训练时的结构变成：

```text
4bit基础模型 + LoRA参数 → 训练
```

这样既保留了LoRA参数高效的优势，又把基础模型的显存开销砍了下来。

## 3.3 依赖bitsandbytes：量化由它负责

QLoRA的落地需要三个库配合：

- `transformers`：模型加载与训练框架
- `peft`：提供LoRA实现
- `bitsandbytes`：负责低精度量化，支持4bit和8bit加载

**bitsandbytes是量化的核心执行者**，没有它，4bit加载就无从谈起。

> **硬件注意（重要）**：本教程统一以 **NVIDIA CUDA GPU** 为准，这是项目支持范围，也是当前最成熟、资料最全的路径。但“QLoRA只能用NVIDIA”并非技术必然——bitsandbytes近年已在扩展CPU、AMD ROCm、Intel等后端支持。**各后端支持程度差异很大**：NVIDIA CUDA是完整支持且验证最充分；其他后端可能只是实验性/部分支持，功能、性能、稳定性都有差距，某些量化特性甚至尚未落地。如果你的硬件不是NVIDIA GPU，务必查阅官方硬件支持矩阵确认后端与安装方式，本文示例命令未必适用。

官方参考：
- bitsandbytes安装/后端支持文档：https://huggingface

## 块 5

### 电脑配置批注：为你的LoRA/QLoRA实训备好“行头”

既然你的目标是动手跑通LoRA微调，咱们直接看硬件门槛。别被术语吓到，这里只讲“跑得动”和“跑得舒服”的区别。

**LoRA最低配置（能跑）：**
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上空闲
- **GPU**：推荐NVIDIA显卡，显存**最低8GB**，**推荐12GB+**

**QLoRA要求（更省显存）：**
QLoRA的核心价值就是**用更小的显存微调大模型**（通过4-bit量化基础模型实现）。但请注意，本教程的命令按**NVIDIA CUDA GPU**给出（其他后端请以bitsandbytes官方说明为准）。

- **显存参考**：对于常见的**7B级别模型**，在**短序列（≤512）、小batch（1~2）** 条件下，经验下限约**8~12GB**；**推荐16GB+**。

> **关键提醒**：上面的数值是**经验起点，不是硬性门槛**。实际显存占用受以下因素影响，条件不同结论就不同：
> - **模型规模**：7B是基准，13B/70B显存需求显著上升。
> - **序列长度**：越长，激活值显存增长越明显。
> - **batch大小**：越大占用越高。
> - **梯度检查点**：开启可用时间换显存，降低峰值。
> - **量化配置**：4-bit（NF4）是否双重量化、`bnb_4bit_compute_dtype`等。
> - **LoRA配置**：目标模块数量/rank

## 块 6

### 5.1 检查Python版本

先确认你的Python环境。在终端运行：

```bash
python --version
```

建议使用 **Python 3.10 或更高版本**。如果版本过低，后续安装依赖时可能会遇到兼容性问题。

---

### 5.2 检查PyTorch与GPU支持

LoRA/QLoRA 依赖 PyTorch，而且 `bitsandbytes` 的 CUDA 量化路径要求安装**与本机 CUDA 匹配的 GPU 版 PyTorch**。直接 `pip install torch` 通常装到 CPU 版，QLoRA 会无法使用。请以 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为准，例如 NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

安装后，运行以下代码验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # 查看 torch 编译使用的 CUDA 版本
```

如果输出 `True`，说明 PyTorch 能看到 CUDA GPU。

> **注意**：`torch.cuda.is_available()` 返回 `True` 只说明 **PyTorch 侧**能用 GPU，**并不足以证明 bitsandbytes 的 4-bit 量化路径可用**（后者还依赖 bitsandbytes 是否正确编译/匹配到本机 CUDA）。是否真能跑 4-bit，请以第八章、第十三章对 `bitsandbytes` 的检查为准。

---

### 5.3 查看GPU信息

在终端运行：

```bash
nvidia-smi
```

输出应包含：

* GPU型号（如 RTX 4090）
* CUDA版本（如 CUDA 12.x）
* 显存大小（如 24GB）

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT 是 Hugging Face 官方的参数高效微调库，支持 LoRA、Prefix Tuning、Prompt Tuning、Adapter 等方法。

### 6.2 安装PEFT

```bash
pip install peft
```

验证安装：

```python
import peft
print(peft.__version__)
```

能打印出版本号即安装成功。

---

## 七

## 块 7

**bitsandbytes 是干嘛的？**

一句话：它是实现 **QLoRA 训练**的“地基”，负责把模型压缩成 **8bit** 或 **4bit** 精度，从而让你在有限显存下跑起大模型微调。

**为什么必须装？**

本教程的 QLoRA 方案是 **4-bit 量化基础模型 + LoRA**，而 `bitsandbytes` 正是这个方案的前置依赖。注意：QLoRA 本身不强制绑定某个量化库，但本教程统一用 `bitsandbytes` + NVIDIA CUDA 演示，所以**必须安装**。

**安装命令**

```bash
pip install bitsandbytes
```

装完后，下一步是检查它和你的 CUDA 环境是否兼容（见下一节）。

---

**本块学习检查**：在终端执行 `pip

## 块 8

**本块学习检查**：运行 `nvidia-smi`，记下输出的 `CUDA Version`，再运行 `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`，对比两者显示的 CUDA 版本是否一致。

---

**8.1 为什么先查CUDA？**

你的目标是跑LoRA微调，而bitsandbytes是量化工具，它依赖CUDA来调用GPU。版本不匹配，后面就会遇到“CUDA不可用”“量化失败”或“加载模型报错”。所以，动手前先花1分钟确认环境。

**8.2 查CUDA版本**

打开终端，运行：

```bash
nvidia-smi
```

在输出里找 `CUDA Version

## 块 9

**本块学习检查**：请按“Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes”的链路，在WSL2中完成环境搭建，并运行一条导入bitsandbytes的Python命令验证CUDA动态库可加载。

---

**9.1 推荐使用WSL2**

在Windows上做LoRA微调，推荐走这条路径：

```text
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

原因很直接：Linux环境兼容性更好，能少踩很多坑。

**9.2 Windows原生可能遇到的问题**

如果你坚持在Windows原生环境跑，最典型的问题是：**bitsandbytes无法加载CUDA动态库**。

常见原因不止一个，可能包括：

- 所装的`bitsandbytes`版本与本机CUDA或GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配

在Windows上配置LoRA环境时，最常见的坑就是**CUDA版本不匹配**。别慌，先运行下面这条命令，看看你的机器到底什么状态：

```bash
nvidia-smi
```

这条命令会告诉你三件事：

- **NVIDIA驱动**是否正常安装
- **CUDA版本**（驱动支持的版本）
- 显卡型号

记住：**驱动支持的CUDA版本 ≠ PyTorch实际使用的CUDA版本**。你只需要确保PyTorch的CUDA版本不超过驱动支持的版本即可。

---

## 十、安装完整LoRA环境

如果你已经装好了基础库（transformers、datasets、accelerate），那现在只需要补上LoRA的核心组件：

```bash
pip install peft bitsandbytes
```

**关键前置**：确保你安装的是**匹配本机CUDA的GPU版PyTorch**，否则后面`import torch`或QLoRA 4-bit量化会直接报错。CUDA 12.1的安装示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

一次性装齐所有依赖：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

运行配置向导：

```bash
accelerate config
```

单GPU场景下，按提示选择：

```text
Compute environment: This machine
Distributed: No
Mixed precision: fp16
```

完成后会生成`default_config.yaml`配置文件，Accelerate会自动读取它。

---

## 十二、LoRA环境验证实验

创建`test_lora_env.py`，写入：

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

看到输出说明：

- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes是否正常工作：

```python
import torch
import bitsandbytes

print(torch.cuda.is_available())
print(bitsandbytes.__version__)
```

如果输出`True`，说明GPU环境正常，QLoRA的4-bit量化可以放心使用。

## 块 11

## Windows用户特别注意事项：推荐使用WSL2

如果你在Windows上遇到报错，通常是因为兼容性问题。最直接的解决方案是**使用WSL2**（Windows Subsystem for Linux 2）。

**为什么推荐WSL2？**  
Windows原生环境对bitsandbytes等库的支持不够稳定，而WSL2提供了一个完整的Linux环境，能避免大部分兼容性坑。

**操作建议：**  
1. 安装WSL2后，在WSL2内部完成全部环境配置。  
2. 后续所有命令（安装PEFT、bitsandbytes、检查CUDA）都在WSL2终端中执行。

**快速自查：**  
如果你在Windows上遇到任何环境报错，先问自己：**“我是不是在用WSL2？”** 如果不是，优先切换到WSL2再继续。

---

**本块学习检查**  
请确认：你的训练环境是否运行在WSL2中？如果不是，记录当前遇到的报错信息，并计划迁移到WSL2。

## 块 12

## 一、为什么需要LoRA？

你已经跑通了Transformers和PEFT环境，现在我们要回答一个关键问题：**微调大模型，为什么不能直接更新全部参数？**

以7B模型（70亿参数）为例，传统全参数微调的流程是：

```text
预训练模型 → 更新全部参数 → 保存新模型
```

这条路会撞上三堵墙：

1. **显存墙**：训练时不仅要存模型参数W，还要存梯度∂W和优化器状态（如Adam的动量项）。实际显存需求远大于模型本身的大小——100亿参数的模型，显存占用可能翻好几倍。
2. **时间墙**：更新全部参数意味着每次反向传播都要计算海量梯度，训练时间被拉长。
3. **存储墙**：每个任务微调完都要保存一份完整的新模型，7B模型就是70亿个参数，存几个任务就占满硬盘。

所以核心矛盾是：**我们真的需要动所有参数吗？**

---

## 二、LoRA的破局思路

LoRA（低秩适配）给出的答案是：**不需要**。

它基于一个观察：大模型在适配新任务时，参数的变化量（ΔW）其实“没那么复杂”——这个变化量可以用低秩矩阵来近似表达。

具体做法是：**冻结原始权重W不动**，只

## 块 13

**本块核心：LoRA 不碰原模型，只学“变化量”**

传统微调是直接改原权重 `W`，训练后变成 `W'`。  
LoRA 的思路完全不同：**冻结** `W`，新增一个参数 `ΔW`，最终模型使用 `W' = W + ΔW`。

也就是说，你训练的不是原模型，而是那个“补丁” `ΔW`。

**为什么只训练 ΔW？**  
大模型已经通过海量数据学会了语言规律、世界知识和推理能力。微调通常只是对已有能力做方向调整，所以不必重新学全部参数，只学**任务相关变化量**往往就够了。

> 注意：这是教学直觉，不是严格结论。任务与预训练能力差距大、数据充足时，全参数微调仍可能更好。

---

**本块学习检查**  
用一句话向自己解释：LoRA 微调时，模型实际使用的前向公式是什么？`W` 和 `ΔW` 各自扮演什么角色？

## 块 14

### 3.1 矩阵基础概念

Transformer里大量参数都是矩阵。比如注意力层中，`Q = X · Wq`，这里的`Wq`就是一个权重矩阵。

假设`Wq`形状是`4096 × 4096`，那它的参数数量就是：

`4096 × 4096 ≈ 1677万`

---

### 3.2 直接训练矩阵的问题

如果我们要微调，直接修改`ΔW`，那仍然需要`4096 × 4096`个参数，成本太高。

**LoRA的核心思路**：不直接学习完整矩阵，而是把它拆成两个小矩阵。

---

## 四、LoRA低秩分解数学原理（重点）

LoRA核心公式：

- 原始权重`W`保持冻结（不更新）
- 新增的`ΔW`由`B × A`组成

也就是说，我们只训练两个小矩阵`B`和`A`，而不是整个大矩阵`W`。

---

**本块学习检查**：请口述——为什么`4096 × 4096`的矩阵直接训练代价大？LoRA用哪两个小矩阵替代了

## 块 15

### 4.1 矩阵尺寸变化：从1677万到6.5万

**核心对比**  
假设原矩阵 `W` 是 **4096 × 4096**。  
- 直接训练全部参数：`4096 × 4096 = 16,777,216`（约1677万）。  

**LoRA 做法**  
冻结原权重 `W`，设置 `rank r = 8`，仅训练两个低秩矩阵作为增量更新：  
- `A`：`4096 × 8` → 参数 `4096 × 8 = 32,768`  
- `B`：`8 × 4096` → 参数 `8 × 4096 = 32,768

## 块 16

**本块学习检查**：读完下面内容后，请合上资料，用自己的话回答：为什么可以用低秩矩阵BA近似参数变化ΔW？提示：从“维度”和“有效方向”两个词入手。

---

想象你要在一个4096维的空间里调整模型参数。直觉上，你可能觉得每个方向都得动一动，但LoRA论文提出了一个关键观察：**模型适应新任务时，参数变化ΔW并不需要覆盖所有方向**。

真正影响任务的“有效变化”，其实集中在一个**低维子空间**里——也就是远小于原维度的少数方向。好比一个4096维的空间，真正起作用的可能只是其中很少的几个方向。

> 注意：这里说的是“低内在维度”这一**假设**，具体有效维度是多少没有固定值，会随

## 块 17

## 动手理解：为什么LoRA用“少量参数”就能微调大模型？

你正在学LoRA微调，目标很明确：**用最少的学习成本，让大模型适配你的任务**。这一节我们用类比帮你建立直觉，再看LoRA实际加在Transformer的哪里。

### 一、先建立直觉：一张图的压缩存储

想象一张高清图片，尺寸是 **4000×4000像素**。如果完整保存，需要 **1600万个数据**（4000×4000=16,000,000）。

但关键来了：如果这张图片的主要变化，只是**亮度**、**对比度**，以及**少量结构变化**——比如只是调亮一点、对比度拉高一点、某个局部区域稍微改动——你根本不需要重新保存1600万个像素。

你只需要用**少量参数**，描述“和原图相比，变化了什么”即可。

**LoRA的核心思想完全一样：**

> 用少量参数，描述模型权重的变化方向。

具体来说，LoRA**冻结原始权重W**（相当于原图，保持不变），只训练两个低秩矩阵A和B，通过增量 **ΔW = (α/r)BA** 来更新模型。也就是说，LoRA学习的是权重的**变化量**，而不是直接修改或分解原始权重。

### 二、LoRA在Transformer中的实际位置

先回顾Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中Attention模块内部，包含四个投影矩阵：

```text
Q（查询）  K（键）  V（值）  O（输出）
```

**LoRA加在哪里？** 在LoRA原论文的早期实践中，一个**常见选择**是只加到Q投影和V投影上。代码里这样写：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示：只训练 `Wq` 的变化和 `Wv` 的变化。

**但请注意**：只挑 `q_proj`/`v_proj` 是LoRA原论文里的一种常见选择，**并不是固定标准**。现代实践中，LoRA经常被加到：

- 注意力的全部投影（q/k/v/o）
- FFN（前馈网络）的线性层
- 甚至用 `target_modules="all-linear"` 覆盖所有线性层

选择哪些模块会影响**可训练参数量、显存占用和效果**，需要结合模型结构（先确认实际模块名）和验证集表现来权衡。

**一个覆盖更完整线性

## 块 18

**r 是什么？一句话版**

`r` 就是 LoRA 里低秩矩阵的“维度”。代码里写 `r=8`，意思是把权重更新量压缩成一个 8 维的低秩空间来学习。

**r 调大/调小，影响什么？**

- **调大**：表达能力更强（能学更复杂的模式），但参数更多、显存占用更高。
- **调小**：省显存、Adapter 更小，但表达能力受限。

**经验起点（不是铁律）**

| 任务类型 | rank 起点参考 |
|---------|-------------|
| 简单任务 | 4–8 |
| 普通微调 | 8–32 |
| 复杂任务 | 64+ |

注意：合适的 `r` 跟模型规模、目标模块、数据量、任务难度都有关。上表只是搜索起点，不是查表定死。

**别背配方，用验证集选 r（实操流程）**

1. **划验证集**：从数据里留出 8:2 或 9:1 不参与训练，后续比较都看它。
2. **固定其它变量**：`target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数全固定，只改 `r`。
3. **小范围网格搜索**：从 `r ∈ {4, 8, 16, 32}` 起步，逐个训练，记录验证集指标（loss/accuracy/F1）和显存、耗时。
4. **看边际收益**：如果 `r` 增大，验证集

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制LoRA参数对模型的更新强度。代码中写作：

```python
lora_alpha=16
```

它的核心用法是与秩 `r` 配合，形成缩放比例：

```text
alpha / r
```

这个比例决定了LoRA注入的增量矩阵 `ΔW = (α/r)BA` 在最终更新时被放大或缩小的程度。`alpha` 越大，LoRA对原模型权重的改动越明显；`alpha` 越小，改动越温和。

**实践直觉**：可以把 `alpha` 理解为“音量旋钮”，`r` 是“声道数量”。同样的 `r` 下，调大 `alpha` 就是调高LoRA对模型行为的影响力。

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 在训练时随机“丢弃”一部分LoRA参数（置零），防止LoRA层对训练数据过度拟合，提升泛化能力。数值表示丢弃比例，`0.05` 即5%。

---

## 九、LoRA与全参数微调区别

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> **

## 块 20

### 2.1 Rank表示什么？——用“中间通道”理解它

Rank（中文：矩阵的秩）在LoRA里，可以直观理解为**低秩矩阵的“中间维度大小”**。

想象原始权重矩阵是 `4096 × 4096` 的大方块。直接更新它，计算量巨大。LoRA的做法是：不碰这个大方块，而是用两个小矩阵去“模拟”它的变化量 ΔW。

设 `r=8`，那么：

- 矩阵 A 的形状是 `8 × 4096`
- 矩阵 B 的形状是 `4096 × 8`

B 和 A 相乘后，`ΔW = B × A`，形状恢复为 `4096 × 4096`。

**关键点**：r 就是这两个小矩阵“共享”的那个中间数字（这里是8）。它决定了“压缩通道”的宽度。

**动手验证（hands-on）**：在纸上画一个 `4×4` 的矩阵，设 `r=2`，写出 A（`2×4`）和 B（`4×2`）的形状，并确认 `B × A` 能变回 `4×4`。这能帮你直观感受“中间维度”的含义。

**一句话总结**：r 越大，中间通道越宽，ΔW 能表达的信息越多（效果可能更好），但训练参数和显存开销也越高。

## 块 21

**Rank控制什么？——用“变化方向”来理解**

Rank（r）决定LoRA能学习多少种变化方向。你可以把它想成模型的“自由度旋钮”。

**低rank（如 r=4）**
- 模型只能学较少的变化模式
- 好处：参数少、显存低、训练快
- 代价：表达能力有限

**高rank（如 r=64）**
- LoRA拥有更多自由度
- 好处：能学更复杂任务、适应力更强
- 代价：参数多、显存增加、训练成本提高

**直观感受一下参数量差异**

假设某层权重是 4096×4096（约1678万参数）。LoRA通过低秩分解大幅压缩这个数字

## 块 22

**3.1 不使用LoRA**

完整训练一个4096×4096的权重矩阵，参数量是：

```text
4096 × 4096 = 16,777,216
```

约**1677万**参数。这就是全参数微调的成本。

---

**3.2 使用LoRA**

LoRA只训练两个小矩阵，公式：

```text
参数量 = r × (输入维度 + 输出维度)
```

- **r=8**：`8 × (4096+4096) = 65,536`（约6.5万）
- **r=64**：`64 × (4096+4096) = 524,288`（约52万）

---

**对比**

| 方式

## 块 23

## 4.1 Rank太小的问题

假设你设了 `r=2`，这会带来什么后果？

**核心问题：学习能力不足。**

- 模型容易**欠拟合**——训练数据里的模式没学透；
- 新任务的知识**无法充分表达**——LoRA 能调整的“旋钮”太少，装不下新任务需要的变化。

**直观表现：**

```text
训练loss下降慢
效果提升有限
```

**动手验证（hands-on）：** 如果你在跑 LoRA 微调，可以故意把 `r` 设成 2 跑几个 epoch，对比 loss 曲线——你会看到它下降得明显比稍大 rank（比如 8）更慢、更平。这就是“学习能力不足”的直接证据。

---

## 4.2 Rank太大的问题

反过来，如果你设 `r=256` 呢？

**代价：**

- 参数增加；
- 显存增加；
- 训练速度下降。

**本质：LoRA 的参数高效优势被削弱了。**

注意一个常见误解：**不要以为 `r=256` 就“接近完整矩阵”了**。以文中 `4096 × 4096` 的权重为例，完整矩阵的秩是 4096，`r=256` 仍然远小于它。

用参数效率公式看真实比例：

- 完整矩阵参数量：`d

## 块 24

**Rank选择速查（起点参考）**

| 任务类型 | rank起点 |
|---------|---------|
| 简单分类 | 4~8 |
| 文本风格调整 | 8~16 |
| 指令微调 | 16~64 |
| 复杂领域任务 | 64以上 |

**注意**：这只是经验起始范围，不是自动调参标准。实际项目需根据数据量、模型大小、序列长度、batch大小、目标模块选择、GPU显存及验证集表现来调整。同一个`r`在不同模型规模或序列长度下，显存和效果差别可能很大。

**Alpha（lora_alpha）是什么？**

LoRA更新公式为：`ΔW = BA`，但实际应用时会缩放：

`W' = W + (α/r)BA`

其中α即`lora_alpha`，r即rank。所以真正加入模型的变化量是`(α/r)BA`，而非直接`BA`。

**动手验证**：若设r=8、α=16，则缩放系数为2，实际更新幅度是BA的两倍。你可以尝试调整α值，观察模型输出变化，加深理解。

## 块 25

### 6.1 控制LoRA影响强度

我们直接看代码和数字，理解 `alpha` 如何“拧动”LoRA的强度。

**基准设定**：假设 `r=8`，`alpha=16`。

**缩放计算**：`alpha / r = 16 / 8 = 2`。

**效果**：LoRA 对权重的更新被**放大2倍**。

---

**对比场景**：

- 若 `alpha=8`，则 `8 / 8 = 1`，更新保持**原始比例**。
- 若 `alpha=4`，则 `4 / 8 = 0.5`，LoRA 影响**减弱**。

---

### 七、Alpha与Rank的关系（重点）

核心公式：**更新量被 `alpha / r` 缩放**。这个比值是影响LoRA更新“强度”的**关键缩放系数**。

**重要提醒**：`alpha/r` 不是“唯一开关”。实际落到权重上的更新量，还取决于训练得到的 `A`、`B` 参数值、初始化方式、学习率、优化器、训练步数与数据，以及不同LoRA变体的处理方式（例如RS-Lo

## 块 26

**动手验证：配置1 vs 配置2**

先看两组配置：

- 配置1：`r=8, alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16, alpha=32` → 比例 = 32 ÷ 16 = **2**

**关键发现**：虽然rank不同（8 vs 16），但缩放比例相同（都是2）。

**实操要点**：调整rank时，**同步调整alpha**保持比例稳定是常见做法——这只是经验法则，不是硬性规则。`alpha = 2 × r`（如8配16、16配32）是常见起点，但**不是最优解**。实际训练中应结合验证集表现调整

## 块 27

## 方案1：小模型实验——先跑通，再调优

你现在目标是LoRA微调，且偏好动手实践。那这个方案就是你的**起点**：用最小配置把流程跑通。

```python
r=8
alpha=16
```

**为什么先选它？** 三个字：少、快、测。参数少，显存占用低；训练快，迭代周期短；适合测试，能快速验证你的数据管道和代码有没有问题。

**动手建议**：先用这个配置跑一个极小的子集（比如几百条样本），确认loss在下降、输出有变化，再考虑升级配置。别一上来就追求效果，先确保“能跑”。

---

## 九、PEFT中的参数配置示例

看实际代码，把概念落到具体位置：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8

## 块 28

### 误区1：rank越大一定越好？

**错误。** 别急着把rank拉满。如果任务本身简单，大rank反而会**浪费显存**、**增加训练成本**。rank不是越大越强，而是**够用就好**。

---

### 误区2：只调rank，不调alpha

很多人改参数时只动rank，忘了alpha要跟着调。比如：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

算一下比例：`16/64 = 0.25`。alpha没变，rank变大，导致`α/r`从2缩到0.25，**LoRA的实际影响反而降低了**。改rank时，记得同步检查alpha。

---

### 误区3：alpha越大越好？

**错误。** alpha过大意味着`α/r`缩放过大，可能导致**更新过强**、**训练不稳定**。

> 注意：`α/r`过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡/发散还常来自学习率过高、数据质量差、batch size过小、精度问题、缺少梯度裁剪等。排查时综合看，别只盯着alpha。

---

### 实训条件说明

本章是**数学原理+参数含义+配置策略**，不需要GPU。最低配置：任意现代CPU + 4GB内存即可。

如果后续实际训练LoRA，参考显存（带条件经验值，非固定门槛）：LoRA约8GB+，QLoRA因4-bit量化通常更低，7B级别约8GB起。实际取决于模型规模、序列长度、batch、精度等，以实际运行为准。

---

### 总结

你已经掌握：Rank是什么、如何影响参数量与表达能力、Alpha作用、`α/r`缩放公式、两者关系、PEFT配置方式、常见选择策略。

核心公式：

```text
W' = W + (α/r)BA
```

一句话理解：**r决定LoRA容量，α通过α/r缩放影响更新强度**。抓住这两点

## 块 29

## 一、LoraConfig是什么？

你已经理解了LoRA的低秩分解原理，现在要进入实战配置阶段。在PEFT（Hugging Face的参数高效微调库）中，`LoraConfig`就是你的“总控台”。

### 1.1 PEFT中的LoraConfig作用

先看LoRA训练的整体流程：

```text
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

`LoraConfig`在这个流程中扮演“配置中心”的角色，它告诉PEFT四件事：

| 配置项 | 回答的问题 |
|--------|-----------|
| LoRA放在哪里 | 哪些模块被注入LoRA |
| LoRA规模多大 | 低秩矩阵的秩r是多少 |
| LoRA训练强度如何 | 更新幅度由lora_alpha控制 |
| 是否使用dropout | 防止过拟合的lora_dropout |

**动手验证**：打开你的Python环境，输入以下代码，观察输出：

```python
from peft import LoraConfig

config = LoraConfig()
print(config)
```

你会看到所有参数的默认值——这就是PEFT给LoRA设定的“出厂设置”。接下来几节，我们逐个调整这些参数，让配置贴合你的微调任务。

> **学习提示**：如果你能说出“低秩分解”中“秩”的含义，说明你已经准备好学习`r

## 块 30

## 1.2 一个简单示例——动手跑通LoRA核心配置

你正在学LoRA微调，目标是快速上手。下面这段代码就是LoRA最核心的配置，直接看、直接敲：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

四个参数，各管一件事：

| 参数 | 作用 |
|---|---|
| `target_modules` | 决定插入哪些层 |
| `r` | 决定LoRA矩阵大小 |
| `lora_alpha` | 控制缩放强度 |
| `lora_dropout` | 控制随机失活 |

**动手验证**：把上面代码复制到你的Python环境跑一遍，确认`config`能正常创建。然后试着把`r`改成4或16，观察配置对象的变化——这只是配置，还没碰模型。

---

## 二、LoRA配置整体流程——配置怎么用起来

光有`LoraConfig`还不够，它要配合`get_peft_model()`才能作用到模型上。流程关系：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

完整代码：

```python
from peft import LoraConfig, get_

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入低秩矩阵”**。

Transformer 模型里，Attention 结构通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA **不会默认修改所有层**——你必须明确告诉它改谁。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思就是：只给 **Query 投影层**和 **Value 投影层** 加上 LoRA。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开看两个关键角色：

- **Q（Query）**：决定模型**关注什么信息**；
- **V（Value）**：决定模型**传递什么信息**。

所以，调整 Q 和 V，通常就能有效改变模型的行为——这是 `q_proj` 和 `v_proj` 成为常见起点的原因。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名可能不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 `target_modules`。

---

### 偷懒方案：`target_modules="all-linear"`

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
    target_modules="all-linear",   # 自动匹配（几乎）所有 nn.Linear；一般不含 lm_head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**注意权衡**：`"all-linear"` 覆盖的模块变多，可训练参数和显存占用也会增加。用不用全线性层，要在**参数量/显存**和**效果**之间取舍，最终以验证集结果为准。如果你的 PEFT 版本太旧不支持这个字符串，就退回显式列模块名的写法，或者升级 PEFT。

---

### 快速验证技巧

配置完后，用 `model.print_trainable_parameters()` 确认 LoRA 已注入。

> 📄 **模型许可证**：使用上述代码

## 块 32

### 方案1：只调整Q和V（轻量起点）

如果你刚开始做LoRA微调，或者只想快速验证效果，可以只调整注意力机制中的Q（查询）和V（值）投影层：

```python
target_modules=["q_proj", "v_proj"]
```

**这样做的好处是：**
- **参数少**：只训练Q和V的增量矩阵，显存占用小；
- **训练快**：参数量减少，迭代速度更快；
- **常见起点**：很多轻量级微调任务都从这一步开始，先看效果再决定是否扩大范围。

> **动手试一试**：先用这个配置跑一个小任务（比如文本分类），记录训练时间和显存占用，作为后续对比的基线。

---

### 方案2：调整全部Attention（更强表达）

如果只调Q和V效果不够，可以扩展到完整的注意力四个投影层：

```python

## 块 33

## LLaMA 与 GPT-2 的 target_modules 差异

**LLaMA 系列**的常见配置是只对 `q_proj` 和 `v_proj` 注入 LoRA：

```python
target_modules=["q_proj", "v_proj"]
```

**GPT-2** 则不同，它把 Q/K/V 三个投影合并到了一个模块里，所以你可能只需要：

```python
target_modules=["c_attn"]
```

### 关键坑：GPT-2 的 `fan_in_fan_out=True`

GPT-2 的注意力层用的是 `Conv1D`，不是普通的 `nn.Linear`。`Conv1D` 的权重形状与 `nn.Linear` 相同，均为 `(out_features, in_features)`，但 PEFT 需要设置 `fan_in_fan_out=True` 以正确注入 LoRA。

**完整可运行示例（GPT-2）：**

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
    target_modules=["c_attn"],   # GPT-2 的 Q/K/V 合并投影
    fan_in_fan_out=True,         # Conv1D 需置 True 以正确注入
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**版本差异提示**：较新版本的 PEFT 对 GPT-2 这类已知模型能自动推断 `fan_in_fan_out`，不显式设置也可能正常工作。但为了跨版本稳妥，遇到 `Conv1D`（如 `c_attn`/`c_proj`）时**建议显式写明**。具体以你所用版本文档为准。

**显存参考**：若将上述示例中的模型替换为 Llama-2-7b，加载时需注意显存要求。例如：# 注意：

## 块 34

## BERT 的 target_modules 配置

你正在学 LoRA 微调，目标很明确。但注意：**BERT 的 LoRA 配置不能照抄 LLaMA 那套**。LLaMA 用 `q_proj`/`v_proj`，BERT 则不同。

**动手查一下**：加载你的 BERT 模型后，运行：

```python
for name, _ in model.named_modules():
    print(name)
```

你会看到类似 `q_proj`、`v_proj` 这样的名字——这就是你要传给 `target_modules` 的值。BERT 通常配置为：

```python
target_modules=["query", "value"]
```

**关键机制**：`target_modules` 按模块名做**后缀匹配**。写 `"query"` 会命中所有以 `query` 结尾的层（各层各头都会覆盖）。你也可以传完整模块名做精确定位，或传正则表达式。

**两个坑**：
1. **GPT-2 别照搬**：GPT-2 的注意力用 `Conv1D`（非 `nn.Linear

## 块 35

**6.1 r是什么？**

r就是rank（秩），它控制LoRA低秩矩阵的维度。核心公式是 `ΔW = (α/r)BA`，其中α是缩放系数，r是秩，决定A和B的中间维度大小。

**直观理解（动手算一下）：**

假设原矩阵是 `4096×4096`，设置 `r=8`，那么：
- A的维度是 `8×4096`
- B的维度是 `4096×8`

这样，训练参数量从4096×4096减少到8×4096 + 4096×8，大幅减少训练时的显存和优化器状态占用（原权重W仍被存储）。

**6.2 r影响什么？

## 块 36

## 7.1 alpha作用

LoRA的最终更新不是 `BA`，而是 `(alpha/r)BA`。

**alpha 控制的是 LoRA 更新的幅度**——它像一个音量旋钮，决定微调对模型的影响强度。

**动手算一下：**

```python
r = 8
lora_alpha = 16
```

缩放系数 = `16 / 8 = 2`

这意味着 LoRA 的影响被**增强**了（更新幅度翻倍）。

---

## 7.2 alpha常见设置

一个常见的经验起点是：

```text
alpha = 2 × r
```

> 注意：这只是经验起点，不是通用规则。也可以保持 alpha 固定而单独调 r，或用其它比例，最终以验证集表现定。

**示例：**

```python
r = 8

## 块 37

**lora_dropout 是什么？**

训练时，它会让一部分 LoRA 连接随机“失活”（暂时不参与计算）。目的很简单：**防止过拟合**——让模型别太依赖某几条特定路径，泛化更好。

**怎么配？**

```python
lora_dropout=0.05
```

这表示 5% 的 LoRA 路径会被随机关闭。

**选多少合适？**

经验起点（不是死规则，要结合验证集效果调）：

- 数据少：约 0.05~0.1
- 数据多：约 0~0.05

如果你的训练数据非常充足，直接设 `lora_dropout=0` 也行。

**动手试一下**

## 块 38

## 9.1 基础配置——快速上手LoRA

这段配置是LoRA微调的“最小可用模板”，特别适合7B模型的测试场景。你正在做LoRA微调，直接照这个结构来就行。

**核心代码：**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：低秩矩阵的秩，越大容量越大
    lora_alpha=16,      # 更新缩放：控制LoRA对原权重的影响强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的Q和V投影
    lora_dropout=0.05,  # 防止过拟合，训练时随机丢弃5%的LoRA参数
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（如GPT类）
)
```

**参数速查表：**

| 参数 | 值 | 作用 |
|------|-----|------|
| r | 8 | LoRA容量 |
| alpha | 16 | 更新缩放 |
| target_modules | q/v | 插入位置 |
| dropout | 0.05 | 防止过拟合 |
|

## 块 39

## 9.1 基础配置：一个完整的 LoRA 配置长什么样

你已经掌握了 `target_modules`、`r`、`lora_alpha`、`lora_dropout` 这几个核心参数的含义，现在我们直接看一个**能跑通全流程**的完整案例。这个案例会带你走完“配置 → 加载 → 训练 → 保存 → 重载验证”的完整闭环，重点解决一个实战中极易踩坑的问题：**新加的分类头怎么保住**。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：BERT 采用 Apache License 2.0，可自由商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练需根据 batch size 适当增加
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

注意：这里用的是 `AutoModelForSequenceClassification`，它会额外带一个分类头。这个分类头是**随机初始化**的，如果不训练它，模型输出就是乱的。

### 第二步：配置 LoraConfig

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",                 # 序列分类任务
    target_modules=["query", "value"],   # BERT 注意力模块名（小写，与HuggingFace实际模块名一致）
    modules_to_save=["classifier"],       # 关键：把新分类头也纳入训练并保存
)
```

逐项拆解：

- `r=8`：低秩矩阵的秩，控制 LoRA 参数量。
- `lora_alpha=16`：缩放因子，实际缩放比例是 `alpha / r = 2`。
- `lora_dropout=0.05`：防止过拟合。
- `bias="none"`：不训练 bias 参数。
- `task_type="SEQ_CLS"`：告诉 PEFT 这是序列分类任务。
- `target_modules=["query", "value"]`：只微调 BERT 注意力中的 query 和 value 投影层。HuggingFace BERT 的注意力模块名即为小写的 `query` 和 `value`，此处配置正确。
- **`modules_to_save=["classifier"]`**：这一行是重点。它让分类头**参与训练**，并且**随 Adapter 一起保存**。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

`print_trainable_parameters()` 会输出可训练参数量。你应该能看到 LoRA 参数和 classifier 的权重都在可训练列表里。

### 第四步：保存（Adapter + 分类头）

```python
# ...（此处进行训练，略）...

model.save_pretrained("./lora_ckpt")
```

`save_pretrained` 保存的是 **Adapter 权重 + `modules_to_save` 指定的模块**，**不是**完整的基础模型。所以保存的文件夹里只有 LoRA 适配器和分类头的权重。

### 第五步：重载并验证分类头被正确恢复

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证分类头权重是否与保存前一致
```

## 块 40

## 10.1 普通文本生成：LoRA参数怎么设？

你已经学完了`r`、`lora_alpha`、`lora_dropout`各自的含义，现在来看它们在真实任务中怎么搭配。对于**普通文本生成**（比如续写、开放式生成），推荐配置是：

```python
r=8
alpha=16
dropout=0.05
```

**怎么理解这组数？**

- `r=8`：低秩矩阵的秩设为8，这是比较轻量的设置。普通文本生成任务对模型原有能力的改动需求不大，用小一点的`r`就够，还能省显存、加快训练。
- `alpha=16`：缩放系数设为16，此时`alpha/r = 16/8 = 2`，即LoRA更新部分会被放大2倍，让微调对模型输出有足够的影响力。
- `dropout=0.05`：5%的随机丢弃率，防止过拟合，同时不至于让训练不稳定。

**动手验证一下：** 如果你手头有文本生成模型，可以分别用`r=8`和`r=32`跑几步训练，观察loss下降速度和显存占用差异——你会直观感受到`r`对资源的影响。

---

### 对比：其他任务的推荐值

为了帮你建立整体感觉，这里列出原文中另外两类任务的推荐配置（后续章节会展开）：

| 任务类型 | r | alpha | dropout |
|---------|---|---|---------|
| 普通文本生成 | 8 | 16 | 0.05 |
| 指令微调 | 16 | 32 | 0.05 |
| 领域知识增强 | 32 | 64 | （原文未指定dropout） |

规律很明显：**任务越复杂、对模型改动需求越大，`r`和`alpha`就越大**。指令微调需要模型学会遵循指令，领域知识增强需要注入大量新知识，所以需要更高的秩来容纳更多可学习参数。但需注意，`r`通常建议在2~32之间，过大会导致过拟合；`alpha/r`比例一般设为1~2，过大可能导致训练不稳定。另外，`r=32`已接近推荐上限，若数据量不足或任务简单，建议从`r=16`开始尝试，避免过拟合。

---

### 关于扩展参数：知道存在即可

`Lora

## 块 41

**普通文本生成：LoRA四参数速配**

你已掌握LoRA基础，目标是用它微调文本生成模型。看这组核心配置：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**动手理解四件事**：

1. **target_modules**：决定改哪里——只改注意力层的`q_proj`和`v_proj`，不动其他权重。
2. **r**：决定LoRA容量——值越大，可学习的参数越多，但并非越大越好。
3. **lora_alpha**：通过`α/r`（即16/

## 块 42

好的，我们直接切入正题。你已经掌握了LoRA的核心思想，现在我们来解决它的“最后一公里”问题：显存。

**1.1 为什么LoRA还不够“省”？**

你已经知道，LoRA冻结了原模型，只训练一小部分参数。这在概念上很美好，但实际操作中，你依然要面对一个“庞然大物”。

**关键点在于：基础模型本身，必须完整地加载到显存里。**

让我们来算一笔账。假设我们要微调一个7B（70亿）参数的大模型。如果用FP16（半精度，每个参数占2字节）来加载，那么仅仅是模型权重一项，就需要：

**7B × 2 bytes ≈ 14GB 显存**

这14GB还只是“入场券”。一旦开始训练，显存里还需要同时存放**激活值**（前向计算中的中间结果）、**梯度**（用于更新参数）和**优化器状态**（如Adam的动量项）。这些加起来，实际显存需求会远高于14GB。

所以，结论很清晰：**LoRA降低了训练的门槛，但“加载完整模型

## 块 43

## 1.2 QLoRA核心思想

QLoRA = Quantized LoRA，核心就一句话：**把基础模型压成4-bit，再挂LoRA去训练**。

**流程对比，一眼看懂区别：**

- LoRA：`FP16模型 + LoRA`
- QLoRA：`4-bit模型 + LoRA`

**QLoRA训练时的分工：**

```text
基础模型：4-bit权重，冻结不动
LoRA参数：FP16/BF16，负责训练
```

**执行流程：**

```text
FP16基础模型 → 4-bit量化 → 加载量化模型 → 插入LoRA → 训练LoRA参数
# 注意：Llama-2-7b FP16 需要

## 块 44

## 3.1 模型存储方式：从“重量”看量化

模型参数占用的存储空间，取决于每个参数用多少字节表示。常见精度对应关系如下：

| 类型 | 每个参数占用 |
|------|------------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

**动手算一算**：一个7B（70亿参数）模型，用FP16存储需要 `7B × 2 ≈ 14GB`；改用4-bit后只需 `7B × 0.5 ≈ 3.5GB`，显存减少约75%。

> ⚠️ 注意：这3.5GB只是**理想权重位数的理论下界**。实际占用会更高，因为还要算上量化常数/缩放因子、分组元数据（NF4按组量化，每组都要存额外参数）、未量化的模块（如部分norm、embedding、lm_head、LoRA参数），以及训练/推理时的计算缓存、激活值和运行时开销。真实占用以实际运行为准。

## 3.2 为什么不用普通INT4？

传统INT4整数范围是 `-8 ~ 7`。

## 块 45

**NF4是什么？——给正在动手做LoRA微调的你**

NF4，全称NormalFloat4，中文叫“正态分布4位浮点”。这是QLoRA论文里专门为**神经网络权重**设计的一种4-bit数据类型。

注意，这里的“浮点”是通俗叫法，**不是**IEEE 754那种通用4位浮点格式。NF4本质是一套**码本（codebook）**：预先定好16个（2⁴）量化取值点，这些点按标准正态分布的分位数函数（quantile function）计算得到，使得每个量化区间内的概率质量相等，0附近（权重最密集处）划分更细。

**怎么用？** 量化时，把每组权重归一化后

## 块 46

**本块学习检查**：读完4.2和4.3后，请用一句话回答：NF4相比普通INT4，为什么能减少精度损失？

---

**4.2 为什么适合模型权重？**

大模型训练后的权重分布，近似**正态分布**：大量参数集中在0附近。

普通INT4是**平均划分**量化范围（如`[-8,-7,...,7]`），对中心密集区域表示能力差。

NF4则**根据正态分布设计量化范围**，让0附近精度更高，更贴合权重分布，从而降低精度损失。

---

**4.3 NF4量化过程**

原始FP16权重（如`0.023`、`-0.145`、`0.876`）经过：**归一化 →

## 块 47

**5.1 为什么需要双重量化？**

你已经知道，4-bit量化要把FP16权重压成4-bit，但别忘了，**量化参数本身也占显存**。每组权重都要保存一个`scale`（缩放因子）和一个`zero point`（零点），这些额外开销在大模型里累积起来很可观。

**5.2 双重量化思想**

普通量化只做一步：FP16权重 → 4-bit权重 + scale参数。

双重量化多做一步：**把scale参数本身也再量化一次**。第一次量化模型权重，第二次量化量化参数。这样最终保存的是：4-bit权重 + 被压缩后的scale。

**5.3 优势**

直接减少额外量化参数的存储，从而降低显存占用

## 块 48

**为什么需要分页优化器？**

训练时显存峰值不只来自模型本身，还有**梯度**和**优化器状态**。比如Adam优化器需要保存**一阶动量**和**二阶动量**，这两项在训练中会持续占用显存。

QLoRA虽然通过4-bit量化大幅降低了模型本身的显存占用，但训练过程中仍可能遇到**显存突然不足**的情况——尤其是当batch size较大或序列较长时，优化器状态会在某个瞬间冲高，直接导致OOM（Out of Memory），训练中断。

**分页优化器怎么解决？**

它的思路借鉴了操作系统的**虚拟内存**机制，核心是借助NVIDIA的**统一内存（Unified Memory）**：

1. 正常训练时，优化器状态留在GPU显存里；
2. 当显存出现峰值、即将OOM时，系统自动把优化器状态**按页换出**到CPU内存；
3. 等需要参与计算时，再**换回GPU**。

**注意**：这不是“平时就把优化器状态常驻CPU”来省显存。分页优化器的重点是**应对偶发峰值**——平稳阶段状态仍

## 块 49

## 6.3 类比理解

把显存想象成你的桌面：

- **普通方式**：所有东西都堆在桌面上。桌面一满，就什么都放不下了——这就是**峰值OOM**。
- **分页优化器**：常用文件放桌面，暂不用的存柜子里，需要时再取出来。显存不够时，优化器状态会自动“换页”到CPU内存，用完再换回GPU。

你不需要手写任何换页逻辑，bitsandbytes 库已经帮你实现了。

## 6.4 如何在训练中实际启用分页优化器？

在 `TrainingArguments` 里通过 `optim` 参数一行搞定：

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
    optim="paged_adamw_8bit",        # 或 "paged_adamw_32bit"（精度更高、显存更大）
)
```

**关键点**：

- `optim="paged_adamw_8bit"`：优化器状态用8-bit存储，更省显存。
- `optim="paged_adamw_32bit"`：精度更高，但占用更多显存。
- **注意**：不同版本的 Transformers / bitsandbytes 支持的取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报“未知 optim”错误，查一下你当前版本的文档，换个名称即可。
- 分页优化器主要缓解**峰值OOM**，通常配合 `gradient_checkpointing=True`、小 `per_device_train_batch_size` + 梯度累积一起使用。

---

**本块学习检查**：打开你的训练脚本，确认 `optim` 参数当前取值。如果报错，尝试换成 `paged_adamw_32bit` 或 `adamw_bnb_8bit`，记录下你版本中哪个取值可用。

## 块 50

## 7.1 4-bit量化：把模型“瘦身”

你的目标是LoRA微调，但显存不够？QLoRA的核心思路就是：**先把大模型压缩，再微调**。

**量化做了什么？**

原始模型权重用FP16（16位）存储，QLoRA把它压成NF4格式（4位）。直观理解：每个权重从占16个格子变成占4个格子，**模型体积直接缩小约4倍**。

```
FP16（16位） → NF4（4位）
```

**为什么用NF4？**

NF4是QLoRA论文推荐的4-bit量化类型，也是本教程采用的方案。注意：bitsandbytes也支持`fp4`等其它4-bit类型，NF4只是**常见选择**，不是唯一选项。

**量化后怎么训练？**

关键点：量化只压缩**基础模型**，你训练的还是LoRA参数。也就是说——模型变小了，但训练方式不变，这就是QLoRA“省显存还能微调”的秘诀。

**对比一下LoRA和QLoRA：**

| 项目 | LoRA | QLoRA |
|------|------|-------|
| 基础模型 | FP16/BF16 | 4-bit（NF4） |
| 训练参数 | LoRA | LoRA |
| 量化 | 无 | 有（4-bit） |
| 显存 | 较低 | **通常更低** |
| 依赖 | PEFT | PEFT + bitsandbytes |

**适合谁？** LoRA适合中高显存GPU；

## 块 51

## load_in_4bit：让大模型“瘦身”到4-bit

你在做LoRA微调时，最头疼的往往是显存不够。`load_in_4bit=True`就是帮你解决这个问题的开关——开启后，模型权重会以4-bit精度加载，显存占用大幅下降。

## bnb_4bit_quant_type：用哪种4-bit？

设置`"nf4"`，即使用NF4量化。这是bitsandbytes专门为4-bit设计的一种量化方式，能在极低精度下尽量保留模型表达能力。

## bnb_4bit_use_double_quant：再省一点

开启双重量化。它会对量化过程中的“量化常数”再做一次量化，相当于把压缩后的模型再压缩一层，进一步节省显存。

## bnb_4bit_compute_dtype：存储和计算要分开看

这是最容易混淆的地方。记住一个关键区别：

- **存储dtype**：权重以4-bit（NF4）存着，目的是省显存。
- **计算dtype**：模型真正做矩阵运算时，会把4-bit权重**反量化**成更高精度来算，这个精度就是`bnb_4bit_compute_dtype`。

**为什么不能一直用4-bit算？** 因为4-bit精度太低，直接做矩阵乘法会导致数值误差过大，训练不稳定。所以QLoRA的策略是：**存的时候用4-bit省空间，算的时候用高精度保质量**。

**默认值是什么？** 不显式设置时，计算dtype通常为float16（若GPU支持）或float32，具体取决于模型和库的默认行为。所以QLoRA配置里**务必显式指定**。

## 按GPU能力选BF16还是FP16（不要写死）

`torch.bfloat16`需要GPU支持（一般NVIDIA Ampere架构及以后，如RTX 30/40系、A100、H100）。较老GPU（如部分RTX 20系及更早）**不支持BF16**，此时应改用`torch.float16`。

**推荐做法：自动判断**

```python
import torch
from transformers import BitsAndBytesConfig

# 根据GPU能力选择计算dtype：支持BF16就用bfloat16，否则退回float16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

**BF16 vs FP16怎么选？**

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面几节我们把 `BitsAndBytesConfig`（4-bit 加载）、分页优化器、计算精度**分开**讲了。现在把它们拼成一个能跑的 QLoRA 训练骨架——这是你从"看懂配置"到"跑通训练"的关键一跃。

### 整体流程：6 步串起 QLoRA

```
① 配置 4-bit 量化 → ② 加载量化模型 → ③ 准备 k-bit 训练 → ④ 注入 LoRA → ⑤ 准备数据 → ⑥ 配置训练参数并训练
```

下面逐段拆解，每一步对应代码中的哪个部分、为什么必须这么做。

---

### 第 1 步：4-bit 量化配置（按 GPU 能力选 dtype）

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

- `load_in_4bit=True`：开启 4-bit 加载，这是 QLoRA 的根基。
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化类型（4-bit NormalFloat）。
- `bnb_4bit_use_double_quant=True`：开启双重量化，进一步压缩显存。
- `bnb_4bit_compute_dtype`：**计算精度**。优先用 `bfloat16`（如果 GPU 支持），否则退回 `float16`。这个 dtype 决定反量化后实际做矩阵运算时的精度。

> **为什么先定 compute_dtype？** 它影响后续所有计算的数值精度，所以放在配置第一步。

---

### 第 2 步：以 4-bit 加载基础模型

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
    print(f"模型加载失败: {e}")
    raise
```

- `quantization_config=bnb_config`：把第 1 步的量化配置传给模型加载。
- `device_map="auto"`：让框架自动分配设备。
- **补 pad_token**：很多 Causal LM 的 tokenizer 没有 pad_token，训练时 batch 需要 padding，所以用 `eos_token` 顶上。
- **异常捕获**：`from_pretrained` 涉及 IO 操作，用 `try-except` 捕获 `OSError` 并记录日志，避免未捕获异常导致程序崩溃。
- **模型许可证**：使用模型前请确认其许可证类型。例如，若使用 Llama 2 系列模型，需遵守 Llama 2 Community License（商用且月活用户大于 7 亿时需 Meta 授权）。

---

### 第 3 步：关键——为 k-bit 训练做准备

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步**不能省**。它做两件事：

1. **处理 norm/输出层的精度**：4-bit 量化后，某些层（如 LayerNorm、LM head）需要保持较高精度才能稳定训练，这个函数会自动处理。
2. **开启梯度检查点所需的输入梯度**：梯度检查点（gradient checkpointing）需要保存中间激活值，这个函数会为 k-bit 模型正确设置相关机制。

`use_gradient_checkpointing=True` 与后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 是**呼应关系**——一个在模型侧准备，一个在训练侧启用。

---

### 第 4 步：注入 LoRA（只训练低秩增量参数）

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
- `target_modules=["q_proj", "v_proj"]`：只对注意力层的 Q 和 V 投影注入 LoRA。**按实际模型模块名调整**，也可以用 `"all-linear"` 对所有线性层注入。
- `get_peft_model` 把 LoRA 适配器包到量化模型上。**LoRA 冻结原始权重 W，仅训练低秩矩阵 A 和 B，增量 ΔW=(α/r)BA**，基础模型保持 4-bit 冻结。
- `print_trainable_parameters()` 打印可训练参数量，确认注入成功。

---

### 第 5 步：极简数据（Causal LM 的 labels 直接复制 input_ids）

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModel
```

## 块 53

## 九、把 QLoRA 的四个关键步骤串起来（可跑通骨架）

你的目标是 LoRA 微调，而且喜欢动手实践。下面这四步就是 QLoRA 的**最小可运行骨架**——每一步都有明确职责，缺一不可：

**第 1 步：`BitsAndBytesConfig` —— 决定基础模型怎么被 4-bit 加载**

这一步是"地基"。它决定了模型加载时的三个关键参数：是否启用 4-bit 加载（`load_in_4bit=True`）、量化类型（NF4，对应 `bnb_4bit_quant_type`）、是否启用双重量化、以及计算时用的 dtype。注意，`load_in_4bit` 是启用 4-bit 加载的总开关，NF4 是量化类型，`compute_dtype` 是计算精度，三者是不同维度。换句话说，模型以什么精度"住"进显存，全由它说了算。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的必要适配**

这是最容易跳过、但**跳过容易出问题**的一步。量化后的模型不能直接拿去训练，这个函数会做必要的适配：把部分层转成计算精度、开启输入梯度以支持梯度检查点等。**跳过它，可能导致显存不足、梯度计算错误或训练失败。**

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

此时主干模型是冻结的 4-bit 权重。这一步把**可训练的 LoRA 适配器**注入进去，让模型在保持主干冻结的前提下，只训练少量新增参数。

**第 4 步：两个开关要对应开启**

- `TrainingArguments` 里的 `gradient_checkpointing=True`
- 第 2 步里的 `use_gradient_checkpointing=True`

建议**同时开启**以确保梯度检查点生效，但并非必须，可根据库版本选择其一。

**额外一步：`optim="paged_adamw_8bit"`**

在 `TrainingArguments` 中设置这个优化器，启用**分页优化器**来缓解峰值显存压力。

> **动手提示**：上面是"能说明关系"的最小骨架。真实训练时，你需要替换成足够规模的数据、合理的超参，并且按你实际使用的库版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数写法，以及 `processing_class`/`tokenizer` 的调用方式。

---

## 十、跑 QLoRA 需要什么配置？

**只学原理**：不需要 GPU。

**实际运行**，最低要求：

| 硬件 | 最低配置 |
|------|----------|
| CPU | 4核+ |
| 内存 | 16GB+ |
| GPU | NVIDIA CUDA GPU |
|

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

先看一个关键认知：**LoRA训练时，基础模型是被冻结的**。

训练过程中，真正被更新的是额外添加的LoRA参数（也叫Adapter）。最终的结构是：

```
基础模型（冻结） + LoRA Adapter（训练）
```

举个例子：你用Llama-7B做微调，训练完成后保存的通常**不是**完整模型，而是两个小文件：

```
adapter_model.safetensors   ← LoRA权重
adapter_config.json          ← 配置信息
```

### 1.2 Adapter有什么特点？

**优点：**
- 文件小，保存快
- 一个基础模型可以搭配多个Adapter，按需切换

比如同一个基础模型，可以分别搭配：

```
客服Adapter / 医疗Adapter / 法律Adapter
```

**缺点：**
部署推理时，需要**基础模型 + Adapter**两部分同时存在，缺一不可。

---

## 二、为什么需要合并LoRA权重？

既然Adapter有"部署时需两部分同时存在"的缺点，那有没有办法把两者合成一个完整模型？这就是**merge（合并）**要解决的问题——把训练好的Adapter权重合并回基础模型，得到一个可直接部署的完整模型。

---

## 本块学习检查

**动手验证一下你的理解：**

1. 打开你训练完成的LoRA输出目录，确认里面是否只有`adapter_model.safetensors`和`adapter_config.json`，而不是完整模型文件。
2. 用一句话回答：

## 块 55

### 推理时的问题：为什么要合并？

你现在做LoRA微调，训练完拿到的是**基础模型 + 一个轻量的Adapter**。推理时，系统得同时加载这两样东西，流程是：

```text
输入 → Base Model + LoRA Adapter → 输出
```

这在实验环境没问题，但到了**部署**阶段就麻烦了——你希望交付的是一个开箱即用的完整模型，而不是“模型+配件”的组合。所以，需要把Adapter“焊死”进基础模型里。

### 合并的数学原理

LoRA训练时，实际权重变化是：

```text
W' = W + (α/r)BA
```

其中 `ΔW = (α/r)BA`，B和A是低秩矩阵

## 块 56

### 3.1 merge_and_unload 是做什么的？

PEFT 库提供了一个方法：

```python
merge_and_unload()
```

它的作用很直接：**把 LoRA 的权重合并进基础模型，然后移除 PEFT 的包装**。

你可以这样理解：训练 LoRA 时，模型是“基础模型 + LoRA 适配器”的组合体。执行 `merge_and_unload()` 之后，适配器里的权重被“写回”基础模型，得到一个完整的、独立的模型。

用公式表示就是：

```text
Base Model + LoRA Adapter → 完整模型
```

---

### 3.2 合并前后，模型结构发生了什么变化？

**合并前**，模型是一个 `PeftModel`，内部结构是：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后**，`PeftModel` 的包装被移除，变成了一个**普通的 Transformer 模型**，但它的权重已经包含了 LoRA 更新后的结果：

```text
普通Transformer模型（包含LoRA更新后的权重）
```

---

### 四、LoRA 模型合并的完整流程

整个流程可以概括为：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步来看。

---

### 五、第一步：加载 LoRA Adapter

假设你已经有了基础模型和训练好的 adapter，路径分别记为：

```python
base_model_path = "model_path"
adapter_path = "adapter_path"
```

加载的代码是：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

此时，模型结构就是：

```text
Base Model + LoRA Adapter
```

---

### 六、第二步：使用 merge_and_unload 合并权重

执行下面这一行代码：

```python
merged_model = model.merge_and_unload()
```

这一步完成的操作是：把 LoRA 的权重（记为 `BA`）加到基础模型的权重（`W`）上，得到新的权重 `W'`：

```text
W' = W + (α/r)BA
```

其中，`α` 是缩放系数，`r` 是秩，`BA` 是低秩矩阵的乘积。

执行完后，`merged_model` 已经**不再是 `PeftModel`**，而是一个**普通的 Transformer 模型**。

---

### 七、第三步：导出合并后的模型

合并完成后，保存模型和 tokenizer：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后的目录结构大致如下（示意）：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> **注意**：上面的目录是**示意**。实际文件会随模型/分词器类型变化，例如可能还有 `generation_config.json`、`special_tokens_map

## 块 57

**本块学习检查**：读完下面内容后，请用一句话回答：合并后的模型加载时，还需要导入 `peft` 库吗？

---

## 3.1 merge_and_unload 到底做了什么？

简单说，`merge_and_unload` 把 LoRA 的“小补丁”**焊死**进原模型，得到一个**完整模型**。之后你不再需要“Base + Adapter”这套组合，直接拿着一个普通模型就能用。

## 合并后怎么加载推理？

合并前，加载模型要借助 `peft` 库；合并后，**不需要** `peft`，直接用 Transformers 就行：

```python
from transformers import AutoModelForCausalLM

# 注意：Llama-2-7b FP16 需要至少14GB显存
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程变成：用户输入 → 完整模型 → 生成结果。

## 合并 vs 不合并，差别在哪？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

## 什么时候该合并？

合并是**权衡**，不是必须。主要看这几点：

- 要不要**多个 Adapter 动态切换**？（要切换就别合并）
- 部署的**推理框架**是否原生支持 LoRA？
- 对**存储/分发体积**的要求（合并后文件更大）
- 对**精度**的接受度（尤其 QLoRA 场景）

## 合并导出后，务必核查这 4 项

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若训练时扩过

## 块 58

### 10.1 推荐合并场景：什么时候该把LoRA“焊死”进模型？

你已经掌握了LoRA微调的核心流程，现在到了关键抉择点：训练完的LoRA权重，是合并进基础模型，还是保持独立？这取决于你的**使用场景**。下面两个场景帮你快速判断。

#### 场景1：上线部署——追求简单，只要一个固定Adapter

想象一下，你要把模型部署到生产环境的服务器上，对外提供API服务。此时，你的需求是**链路越简单越好**。

- **推荐做法**：如果只需要**一个固定的Adapter**，且希望部署时**不额外依赖PEFT库**、**不做运行时Adapter加载**，那么**合并**是更优解。合并后，服务器直接加载完整模型即可，省去了加载LoRA的额外步骤和潜在故障点。
- **例外情况**：如果你的推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你需要在同一个基础模型

## 块 59

### 不建议合并的两个场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个专用Adapter（比如A、B、C分别处理不同任务）。这时候**保留Adapter更灵活**——你可以按需切换，不用重新训练。如果合并了，就固化成一个模型，失去灵活性。

**场景2：持续训练**

如果你打算**继续调整LoRA**（比如加数据、调超参），那就先别合并。保留Adapter，下次训练直接接着来；合并了反而要重新加载基础模型，麻烦。

---

### QLoRA合并注意事项（重点）

QLoRA有个特殊点：**基础模型是4-bit（NF4）存储的，而LoRA参数是FP16/BF16高精度**。

## 块 60

## 11.1 直接合并可能遇到的情况

你在QLoRA训练完成后，可能会想直接调用`merge_and_unload()`把LoRA权重合并回基础模型。这里要提醒你：**在4-bit量化状态下直接合并，是有风险的**。

具体可能出现两类问题：

- **精度损失**：4-bit权重需要先反量化再合并，这个过程与全程使用高精度计算并不完全等价，会带来数值上的偏差；
- **兼容性问题**：可能出现权重类型不匹配，或者你使用的PEFT / bitsandbytes / Transformers版本不支持在量化状态下直接合并。

需要说明的是：**这些只是“可能”出现的情况，并非每次都会报错**。能否顺利合并、合并质量如何，取决于你的库版本、量化配置，以及是否先做了反量化。请以你所用版本的实际运行结果为准。

---

## 11.2 常见做法：先反量化到高精度再合并

为了避免上述问题，更稳妥的做法是：**不在4-bit状态下直接合并**，而是先用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一个Adapter后合并。

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

这样合并发生在高精度权重上，绕开了4-bit状态下的类型和精度问题。

下面是完整可执行的代码：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 模型许可证：请根据你所使用的基础模型，确认并遵守其许可证条款（例如 Llama 2 Community License，商用月活>7亿需Meta授权）

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

- 第1步是核心：**不传`quantization_config`**，用`torch_dtype`指定FP16/BF16加载基础模型；
- 第3步的`safe_merge=True`会在合并时做数值检查，帮你提前发现NaN等问题；
- 如果显存

## 块 61

### QLoRA合并：三个关键提醒

你已掌握`merge_and_unload`的基本用法，但QLoRA（4-bit量化训练）的合并有额外坑点。直接合并时，请盯住这三点：

**1. 版本决定“能不能直接合并”**
不同PEFT版本对“量化模型能否直接合并”支持不同，部分版本要求先反量化。动手前，务必查阅你所用版本的`merge_and_unload`文档。核心操作是：**加载基础模型时不再传`quantization_config`或`load_in_4bit`**，这样得到的是高精度权重，再执行合并。

**2. 内存代价：一份完整FP16模型**
以高精度重新加载基础模型，会占用约

## 块 62

**动手做：合并时加一道保险**

`merge_and_unload` 支持 `safe_merge=True`，合并时自动检查数值（比如有没有出现 NaN）。这相当于给合并过程上了道保险，建议开启：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

**合并后必做：一致性验证**

合并没有报错不代表结果正确。请用同一批固定输入，对比合并前后的输出：

1. 准备一小批固定 prompt
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits 或生成文本是否基本一致

**最小验证代码**（固定 prompt，先比 logits 最大绝对差，再比贪心生成文本）：

## 块 63

## 11.3 合并时开启校验并做一致性验证（建议）

合并 LoRA 权重后，最怕的一件事就是：**模型看起来合并成功了，但实际行为悄悄变了**。这一节教你一个“合并前后对照实验”，用同一个固定输入，分别跑合并前和合并后的模型，对比输出是否一致。这是 hands-on 排查问题最直接的手段。

### 为什么不能拿合并后的对象当“合并前”？

先记住一个关键陷阱：`merge_and_unload()` 返回合并后的模型，**原 `PeftModel` 对象仍保留未合并状态**，可用于对比。也就是说，一旦执行了合并，原来的 `PeftModel` 并没有被修改，它仍然是未合并状态。

所以，如果你想对比“合并前 vs 合并后”，可以直接使用原来的 `PeftModel` 对象作为合并前对照组，无需重新加载。

### 操作步骤：加载一个“对照基准”

```python
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)
```

这里 `model_before` 就是你的“合并前”对照组：底层是原始 Base 模型（高精度），上面叠加了 LoRA Adapter。而 `merged_model` 是 §11.2 里合并完成后的模型。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

### 校验一：对比 logits（数值层面）

用同一个固定 prompt（比如 `"请用一句话介绍你自己。"`，换成你任务里有代表性的输入），分别让两个模型前向传播，比较输出 logits 的最大绝对差：

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

**怎么看结果？** 这个值越接近 0 越好。在 QLoRA 场景下，由于量化精度损失，通常不会完全等于 0，而是一个较小的近似值。如果这个值异常大，说明合并过程可能出了问题。

### 校验二：对比生成文本（行为层面）

数值接近不代表生成结果一致，所以还要做一次贪心生成对比。关键是 `do_sample=False`，去掉随机采样带来的不确定性，让两次生成完全可复现、可对比：

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

## 十一、合并时开启校验并做一致性验证（建议）

### 为什么必须单独再加载一份模型？

`merge_and_unload()` 会**原地**修改你的 `model`——它把 LoRA Adapter 合进底层权重，并移除 PEFT 包装。这意味着合并完成后，`model` 本身就已经是"合并后"的状态了。

如果你拿它当"合并前"的对照，两边其实是同一份已合并权重，`max_abs_diff` 会**恒为 0**——验证直接失效。

正确做法：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照基准。

### 显存不够怎么办？

如果无法同时放下两个模型，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits/文本存下来，**释放显存**；
2. 再加载合并后的模型，对比之前保存的结果。

### 结果预期：别追求逐位相同

- **QLoRA 场景**：合并前后通常只是**近似一致**，`max_abs_diff` 一般是较小的非零值，生成文本也可能有个别 token 差异。重点确认**没有明显退化或 NaN**，而不是逐位相同。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。

是否可接受，以你的任务指标与验证集为准。

---

## 十二、电脑配置批注

本章节需要

## 块 65

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

合并QLoRA模型时，强烈建议开启校验机制。核心操作是调用 `merge_and_unload` 时设置 `safe_merge=True`，并在合并完成后做一致性验证，确保合并结果正确。

**为什么必须做这一步？**

QLoRA在4-bit量化状态下直接合并容易出问题。最稳妥的做法是**先反量化再合并**——这里的“反量化”不是手动逐层还原4-bit权重，而是**重新加载一份不带量化配置的基础模型**：

```python
# 不传 quantization_config / load_in_4bit
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16  # 或 float16
)
# 再挂载同一份 Adapter 后合并
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload(safe_merge=True)
```

这样合并发生在高精度权重上，天然避开“在4-bit状态下合并”的类型/精度问题。

**版本兼容性提示**：不同PEFT / bitsandbytes / Transformers版本对“量化模型能否直接调用 `merge_and_unload`”支持不同——部分版本直接报错或警告，部分较新版本有一定支持但可能有精度损失。因此“先反量化再合并”是**跨版本最稳妥的做法**。是否能在量化状态下直接合并，以你所装版本的文档和实际运行为准。

**模型许可证说明**：部署引用模型时需确认并标注许可证类型，例如：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

### 常见问题排查速查表

| 问题 | 常见原因 | 排查方向 |
|------|---------|---------|
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（忘了用 `PeftModel.from_pretrained` 挂载，或已合并成普通模型） | 确认加载方式，先得到 PeftModel 再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积Adapter | 无需处理；若在意体积可保留Adapter不合并 |
| 显存不足 | 高精度完整模型占用高；dtype用了FP32、未offload/分片 | 改用CPU合并或offload、用FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存tokenizer；特殊token/generation_config未导出 | 重新 `save_pretrained` 并核查tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 4-bit状态直接合并、PEFT/bitsandbytes版本不支持、量化包装未正确反量化、内存不足 | 先反量化到FP16/BF16再合并 |

## 块 66

## 路线A：合并后部署完整模型

**核心思路**：把 LoRA 训练好的“增量”直接融回基础模型，得到一个**完整模型**，然后像部署普通模型一样去部署它。

**操作链路**：

```
merge_and_unload()  →  保存完整模型  →  推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

**这条路线的特点**：

- **部署链路简单**：推理框架把合并后的模型当作**普通模型**加载即可，框架完全不需要理解 LoRA 是什么；
- **推理时零 Adapter 切换成本**：因为 Adapter 已经“焊死”在模型里了，推理过程中不存在切换 Adapter 的开销；
- **代价是存储成本高**：每合并一个 Adapter，就要保存/分发一份**完整模型**。如果你有多个任务（多个 Adapter），就得存多份完整模型，存储开销随任务数线性增长。

> **模型许可证**：部署前请确认基础模型的许可证类型。例如，Llama 2 采用 Llama 2 Community License（商用月活>7亿需Meta授权）。

---

## 路线B：保留 Adapter，用支持 LoRA 的推理框架动态服务

**核心思路**：不合并，保留训练出来的小体积 Adapter 文件，让推理框架（如 **vLLM**）在**同一份基础模型**上运行时加载/切换**多个 LoRA Adapter**（Multi-LoRA serving）。

**操作链路**：

```
基础模型（只加载一份）
        +
运行时挂载多个 LoRA Adapter（A / B / C）
        ↓
按请求指定使用哪个 Adapter
```

**这条路线的特点**：

- **多任务/多租户时省显存省存储**：只需一份基础模型 + 多个小体积 Adapter，不用为每个任务各存一份完整模型；
- **支持热切换**：运行时动态切换 Adapter，无需重启服务；
- **代价是链路更复杂**：依赖框架对 LoRA 的支持与配置，不同框架/版本对 LoRA 的支持范围和限制各不相同。

---

## vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

以 vLLM 为例，启动 OpenAI 兼容服务时通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，请求里再用 Adapter 名字指定用哪个。命令行大致形如：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path

## 块 67

**本块学习检查**：读完下面两条选择建议后，暂停一下，对照你自己的LoRA微调场景，回答：你的部署目标是“单一模型最简上线”还是“一个底座服务多个任务”？答案决定你走哪条路。

---

**14.3 如何选择（实战决策版）**

选路线前先问自己两个问题，答案直接对应两条路：

- **问题1：你手里有几个Adapter？**
  - 只有1个，且想部署最简单 → **走合并路线A**。把LoRA权重融进基础模型，得到一个完整模型文件，部署时不用额外加载Adapter，省事。
  
- **问题2：你的推理框架支持动态加载LoRA吗？**
  - 如果同一个基础模型要挂多个Adapter（比如一个底座同时服务客服、翻译两个任务），且你的推理框架原生支持LoRA → **走保留Adapter路线B**。这样不用每个任务都合并一份完整模型，省存储

## 测评题目

### Q-KB12-ST1202-003
在其他条件相同时，增大 LoRA rank 最可能带来什么？

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1204-003
分页优化器的主要目标是？

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）
