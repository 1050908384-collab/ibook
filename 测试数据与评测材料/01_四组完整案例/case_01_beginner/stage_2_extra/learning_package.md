# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么跑的：

```text
输入数据 → 整个模型参数更新 → 保存完整模型
```

以 7B 参数模型为例，就是 **70 亿个参数**。训练时你要同时保存三类东西：

- **参数**本身
- **梯度**（用于反向传播）
- **优化器状态**（如 Adam 的动量项）

这三样加起来，显存需求非常高。普通消费级 GPU 根本扛不住——不是慢，是直接放不下。

**一句话总结**：全参数微调 = 显存杀手，消费级硬件玩不转。

---

**本块学习检查**

请用自己的话回答：全参数微调为什么在消费级 GPU 上跑不动？至少说出两个原因（提示：想想训练时要保存哪些东西）。

## 块 2

**1.2 参数高效微调（PEFT）**

你只想微调LoRA，不用动整个大模型——这就是PEFT的核心：**只训练少量新增参数**。

流程很简单：

```
原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter
```

对比一下开销：

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**动手验证**：对照表格，你的目标是LoRA微调，记住它的两个关键词：**少量参数**、**较低显

## 块 3

**本块学习检查**：读完本块后，请合上资料，用一句话向自己解释“LoRA到底在做什么”，并对照原文检查是否说清了“冻结原模型、只训练两个小矩阵”这个核心。

---

**LoRA是什么？——用“小补丁”微调大模型**

想象原模型是一张已经画好的大图（权重 `W`）。LoRA（低秩适配）的做法是：**不修改原图**，而是在旁边加一张小透明补丁（`ΔW`），最终效果就是“原图 + 补丁”。

这个补丁怎么来？它由两个很小的低秩矩阵相乘得到：`ΔW = (α/r) × B × A`，其中 `r` 为

## 块 4

## 3.1 LoRA的问题：模型还是太大

你已经知道LoRA能大幅减少**训练**的参数，但有个关键点别忽略：**基础模型本身仍然要完整加载到显存里**。

以7B模型为例，如果用FP16精度加载，大约需要 **14GB显存**。也就是说，即使LoRA省了训练参数，模型加载这一关依然卡着显存。

---

## 3.2 QLoRA：给模型“瘦身”再训练

QLoRA（Quantized LoRA）就是**量化版LoRA**，思路很直接：

**FP16模型 → 4bit量化 → 显存占用下降 → 再加载LoRA进行训练**

训练时的结构是：

**4bit基础模型 + LoRA参数 → 训练**

这样既保留了LoRA参数高效的优势，又通过量化把基础模型“压缩”到更小，显存压力自然就降下来了。

---

## 3.3 QLoRA依赖bitsandbytes

要跑QLoRA，你需要这几个组件：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中 `bitsandbytes` 负责三件事：

- 低精度量化
- 4bit加载
- 8bit加载

> **关于后端支持的重要提醒**：本教程统一以NVIDIA CUDA GPU为准，这是项目支持范围，也是当前最常见、资料最全的路径。但“QLoRA只能用NVIDIA CUDA”并非技术必然——bitsandbytes近年已在扩展对CPU、AMD ROCm、Intel等后端的支持，具体以你所用版本的官方说明为准。如果你的硬件不是NVIDIA GPU，请查阅官方文档确认后端与安装方式，本文示例命令未必适用

## 块 5

**你的目标：跑通 LoRA 微调。** 下面直接告诉你，你的电脑需要什么。

### 一、LoRA 最低配置（先看这个）
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：NVIDIA 显卡（推荐）
- **显存**：最低 8GB，推荐 12GB+

> **动手检查**：打开终端，输入 `nvidia-smi`，看右上角显存大小。如果显示 8GB 以上，就能跑 LoRA。

---

### 二、QLoRA 要求（显存不够时的选择）
QLoRA 的设计目标就是**用更小显存微调大模型**（通过 4-bit 量化基础模型，显存占用通常比 LoRA 全精度加载更低）。

**注意**：本教程按 NVIDIA CUDA GPU 给出命令。其他后端请以 bitsandbytes 官方说明为准。

**7B 级别模型的显存经验参考**（不是硬性最低配置）：
- 参考下限：约 8~12GB（短序列、小 batch）
- 推荐：16GB+

---

### 三、别把数值当死门槛
同一个模型的实际显存占用，受以下**测试条件**影响，条件不同结论就不同：

- **模型规模**：7B 级别；13B/70B 显存需求显著上升
- **序列长度**：短序列（≤512）；越长激活值显存增长越明显
- **batch 大小**：小 batch（1~2）；越大占用越高
- **梯度检查点**：开启可用时间换显存，降低峰值

## 块 6

### 5.1 Python版本检查

先确认你的Python版本，终端里跑：

```bash
python --version
```

建议使用 **Python 3.10+**。版本太低的话，后面装PEFT和bitsandbytes容易踩坑。

---

### 5.2 PyTorch检查（重点）

LoRA/QLoRA依赖PyTorch，而且**bitsandbytes的CUDA量化路径要求安装与本机CUDA匹配的GPU版PyTorch**。直接`pip install torch`通常装到CPU版，QLoRA会无法使用。请以[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)为准，例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

装完跑这段代码验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # 查看torch编译使用的CUDA版本
```

看到`True`，说明PyTorch能看到CUDA GPU。

> **注意**：`torch.cuda.is_available()`返回`True`只说明PyTorch侧能用GPU，**并不足以证明bitsandbytes的4-bit量化路径可用**（后者还依赖bitsandbytes是否正确编译/匹配到本机CUDA）。是否真能跑4-bit，请以第八章、第十三章对bitsandbytes的检查为准。

---

### 5.3 查看GPU信息

终端跑：

```bash
nvidia-smi
```

输出包含GPU型号、CUDA版本、显存大小，例如：

```text
RTX 4090
CUDA 12.x
Memory 24GB
```

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是Hugging Face官方的**参数高效微调库**，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter。

### 6.2 安装PEFT

```bash
pip install peft
```

验证：

```python
import peft
print(peft.__version__)
```

能打印版本号，说明安装成功。

---

## 七、安装bitsandbytes（重点）

> 本块只确认环境准备到位。bitsandbytes的安装与4-bit量化验证，按后续章节指引操作。

## 块 7

**bitsandbytes 是干嘛的？**

简单说，它就是让模型“瘦身”的工具，支持**8bit量化**和**4bit量化**，而QLoRA训练正是靠它实现的。

**为什么必须装？**

本教程用的是 `bitsandbytes` 的 **QLoRA 4-bit 量化方案**，所以它是前置依赖。不过要注意：QLoRA 的核心思想（4-bit量化基础模型 + LoRA）并不绑定某个库，只是本教程统一用它 + NVIDIA CUDA 演示。如果你用其他量化后端，请以对应文档为准。

**安装命令**

```bash
pip install bitsandbytes
```

装完后，记得做 **CUDA兼容检查**（见下一节），确保它能正常调用你的GPU。

---

**本块学习检查**：确认你的环境已安装 `bitsandbytes`，

## 块 8

**为什么先查CUDA？**
你的目标是跑通LoRA微调，而bitsandbytes是量化工具，它依赖CUDA来调用GPU。版本不匹配，轻则CUDA不可用，重则量化失败、模型加载报错。所以，动手前先花一分钟确认环境。

**三步检查（跟着做）**

1. **看GPU驱动**：终端运行`nvidia-smi`，在输出末尾找到`CUDA Version`，记下这个数字。
2. **看bitsandbytes状态**：运行`python -c "import bitsandbytes as bnb; print(bnb.cuda.is_available())"`，正常会输出`True`。
3. **对比**：两者版本号应一致或兼容

## 块 9

**本块学习检查**：在终端运行 `python -c "import bitsandbytes"`，观察是否报错。若报错提示CUDA相关，记录错误信息，并对照下文原因排查。

---

## 9.1 推荐使用WSL2

你的目标是用LoRA微调模型，环境搭建别在Windows上硬磕。推荐走这条路径：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

原因很简单：Linux环境兼容性更好，能少踩很多坑。

---

## 9.2 Windows原生可能问题

如果你坚持用Windows原生，最可能遇到**问题1：bitsandbytes无法加载CUDA动态库**。

这不是单一原因，常见的有：

- 你装的`bitsandbytes`版本，和本机CUDA或GPU版PyTorch不匹配；
- 缺少或

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配

在Windows上配置LoRA环境时，最常见的坑就是**CUDA版本不匹配**。别慌，先跑一条命令看看你的机器到底什么状态：

```bash
nvidia-smi
```

这条命令会告诉你三件事：
- **NVIDIA驱动**是否正常安装
- 驱动支持的**CUDA版本**（注意：这是驱动支持的版本，不是已安装的）
- 你的GPU型号

记住一个关键点：**PyTorch的CUDA版本必须和驱动兼容**，否则后面`import torch`或QLoRA 4-bit量化会直接报错。

---

## 十、安装完整LoRA环境

假设你已经有了基础环境（`transformers`、`datasets`、`accelerate`），现在只需要两步：

**第一步**：确保PyTorch是匹配你CUDA的GPU版本。CUDA 12.1的安装示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ 如果跳过这步，后面QLoRA的4-bit量化会失败。

**第二步**：安装LoRA核心库：

```bash
pip install peft bitsandbytes
```

想一步到位？直接跑这条完整命令：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

装完库之后，需要告诉Accelerate你的硬件环境。执行：

```bash
accelerate config
```

按提示选择，单GPU场景就选：

```text
Compute environment: This machine
Distributed: No
Mixed precision: fp16
```

它会自动生成`default_config.yaml`配置文件，搞定。

---

## 十二、LoRA环境验证实验

现在动手验证环境是否OK。创建`test_lora_env.py`：

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

最后测一下bitsandbytes是否能用：

```python
import torch
import bitsandbytes

print(torch.cuda.is_available())
print(bitsandbytes.cuda_setup())
```

运行：

```bash
python test_qlora_env.py
```

看到输出就说明：
- ✅ PyTorch正常
- ✅ bitsandbytes正常
- ✅ CUDA可用

## 块 11

**Windows用户特别注意事项：用WSL2解决兼容性问题**

你在Windows上跑LoRA/QLoRA环境时，如果遇到报错，最常见的原因就是兼容性问题。解决办法很直接：**使用WSL2**。

WSL2能帮你绕开Windows原生环境下的各种坑，让bitsandbytes、CUDA这些组件更稳定地工作。记住这个对应关系：**Windows报错 → 用WSL2**。

**常见问题速查表**（遇到问题先对照这个）：

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型太大 | 降低batch或用更小模型 |
| Windows报错 | 兼容性问题 | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

**你现在的位置**：环境准备已全部完成——PEFT装好、bitsandbytes就绪、CUDA兼容性已检查、QLoRA的GPU环境确认过、Windows方案也了解了。

**下一步**：进入LoRA微调实战，用PEFT + Trainer训练大语言模型。

**动手验证**：打开终端，运行

## 块 12

## 一、为什么需要LoRA？（①②③）

### 1.1 大模型微调的问题

现代大语言模型（如LLaMA、Qwen、Baichuan、Mistral）通常拥有数十亿甚至上百亿参数。以7B模型为例，就是70亿参数。

如果采用传统方式微调——从预训练模型出发，更新全部参数，再保存新模型——你需要面对三个现实问题：

- **巨大的GPU显存**：训练时不仅要保存模型参数W，还要保存梯度∂W和优化器状态（如Adam moment）。假设模型有100亿参数，实际显存需求会远大于模型本身的大小。
- **大量训练时间**：更新全部参数意味着每轮训练都要计算和回传海量梯度。
- **巨大的存储空间**：每个微调版本都要保存一份完整的新模型。

**核心痛点**：全参数微调的成本太高，而大模型的大部分参数在微调时其实并不需要大幅改动。

---

## 二、LoRA核心思想（①②）

LoRA（Low-Rank Adaptation）的思路是：**不更新全部参数，而是只学习一个很小的增量**。

具体来说：

- 预训练模型的权重W保持冻结（不更新）；
- 引入一个低秩分解的增量ΔW，用两个小矩阵B和A的乘积来表示：**ΔW = BA**；
- 训练时只更新B和A，参数量远小于W。

**为什么用低秩矩阵？** 因为大模型在微调时，真正需要的权重变化往往集中在一个低维子空间里，不需要完整的参数量来表达。rank（秩）决定了B和A的大小——**rank越大，可学习的参数量越多，但训练成本也越高**。

**LoRA如何插入

## 块 13

**本块学习检查**：读完下面内容后，请暂停并回答：LoRA 微调时，原模型权重 W 是被修改了，还是被冻结了？最终模型用的是哪个公式？

---

**2.1 不直接修改原模型参数**

传统微调是直接改原权重：`W → W'`（直接修改 W）。

LoRA 的做法不同：**冻结 W**（原模型参数保持不变），另外新增一个参数 `ΔW`。最终模型用的是：

`W' = W + ΔW`

其中 W 是原模型参数，ΔW 是 LoRA 新增参数。

---

**2.2 为什么只训练 ΔW？**

大模型已经通过海量数据学会了语言规律、世界知识和推理能力。微调通常只是对已有能力做“方向调整”，所以不必重新学全部参数，只学**任务相关的变化量**（ΔW）往往就够了。

> 注意：这是教学直觉，不是严格结论。能否用 LoRA 替代全参数微调，取决于任务与预训练能力的差距、数据规模、目标模块

## 块 14

**3.1 矩阵基础概念（动手版）**

Transformer里，大量参数是矩阵。以注意力层为例，Q由输入X乘以权重矩阵Wq得到：

```text
Q = X · Wq
```

> 形状示意：设X为(序列长度, d_model)，Wq为(d_model, d_model)，则Q为(序列长度, d_model)。实际还有batch维、多头拆分，这里只为说明“Q由输入乘以权重矩阵得到”。

假设Wq是：

```text
4096 × 4096
```

参数数量就是：

```text
4096 × 4096 ≈ 1677万
```

**3.2 直接训练的问题**

如果直接修改ΔW，仍需4096×4096个参数。LoRA的思路是：**不直接学完整矩阵，而是拆成两个小矩阵**。

---

**本块学习检查**：请手写一遍“Wq形状为4096×4096时，参数数量如何计算”，并说出LoRA的核心思路（一句话即可）。

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设你有一个大矩阵 W，尺寸是 **4096 × 4096**。如果直接训练它，要更新的参数数量是：

> 4096 × 4096 = **16,777,216**（约1677万）

现在用 LoRA，设置 **rank r = 8**。我们冻结原权重 W，只训练两个低秩矩阵：

- **A**：尺寸 **8 × 4096**，参数 = 8 × 4096 = **32,768**
- **B**：尺寸 **4096 × 8**，参数 = 4096 × 8 = **32,768**

LoRA 的总参数量为

## 块 16

**本块学习检查**：读完请自问——“为什么能用低秩矩阵BA近似ΔW？”用一句话回答，再继续。

---

**5.1 神经网络参数变化具有低维特征**

LoRA论文的核心观察是：模型适应新任务时，参数变化ΔW**不需要覆盖所有方向**。

想象一个4096维的空间——理论上参数可以朝任意方向调整。但实际有效的改变，往往集中在**远少于原维度的少数方向**上，也就是一个**低维子空间**。

> 注意：这是“权重更新具有较低内在维度”的**假设**。具体有效维度是多少，没有通用固定值，随模型、任务、数据变化。定性理解即可，别记成“几十维”这种确定数字。

**为什么这很重要？** 既然Δ

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。但如果这张图片的主要变化只是**亮度、对比度**和**少量结构变化**，其实只需要**少量参数**就能描述这些变化。

**LoRA的核心思想正是如此：用少量参数描述模型变化的方向。**

预训练模型已经学到了大量通用知识，微调时我们不需要重新学习全部内容，只需要捕捉“变化的方向”——这个方向本身是低秩的，所以可以用更少的参数来表达。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构是：**Input → Embedding → Attention → FFN → Output**。

其中Attention模块包含 **Q、K、V、O** 四个投影矩阵。

LoRA在早期实践中**常见**加入 **Q投影** 和 **V投影**，例如：

```python
target_modules=["q_proj", "v_proj"]
```

表示只对 `Wq` 和 `Wv` 应用LoRA，即冻结 `Wq` 和 `Wv`，训练对应的低秩矩阵A和B，形成增量 `ΔWq` 和 `ΔWv`。

> **注意**：只挑 `q_proj`/`v_proj` 是LoRA原论文的**常见选择**，**不是固定标准**。现代实践常把LoRA加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

**一个覆盖更完整线性层的配置示例**（以Llama类模型常见模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

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

> 上例是**显式列出模块名**的写法；若当前PEFT版本支持，也可用 `target_modules="all-linear"` 自动匹配所有线性

## 块 18

### 8.1 rank（r）——从“调参”到“调优”

你正在学LoRA微调，`r`（rank）是第一个要拿下的核心参数。别把它当“玄学”，它就是一个**低秩矩阵的维度**，控制着微调时新增参数的“粗细”。

**直观理解**：`r` 越大，模型能“记住”的新知识就越多，表达能力越强；但代价是参数更多、显存占用更高。`r` 越小，越省资源，但可能“学不够”。

**起点参考**（不是死规则，是搜索起点）：
- 简单任务：`r ≈ 4-8`
- 普通微调：`r ≈ 8-32`
- 复杂任务：`r ≈ 64+`

**关键**：`r` 没有“标准答案”，它和模型规模、目标模块、数据量、任务难度都相关。别背表，用**验证集**来选。

**实操流程（比查表可靠）**：
1. **划验证集**：从数据中留出20%（或10%），不参与训练，专门

## 块 19

## 8.2 lora_alpha：控制LoRA的“更新力度”

`lora_alpha` 是一个数值参数，比如 `lora_alpha=16`。它的作用是**控制LoRA参数在训练时的更新强度**——可以理解为“每次学习时，LoRA矩阵对原模型的影响有多大”。

实际使用中，`alpha` 通常会和秩 `r` 配合，用 **`alpha/r`** 作为最终的缩放比例。比如 `r=8`、`alpha=16` 时，缩放比例就是 2。这个比例越大，LoRA 对模型的改动越明显；比例越小，改动越温和。

## 8.3 lora_dropout：防止过拟合的小开关

`lora_dropout` 也是一个数值参数，比如 `lora_dropout=0.05`。它的作用是**防止LoRA参数过拟合**——在训练时随机“丢弃”一部分LoRA参数的计算，让模型不要过度依赖某几个参数，从而提升泛化能力。

---

## 九、LoRA vs 全参数微调：一张表看懂

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> ⚠️ 注意：上表是**相对趋势**，不是绝对数值。LoRA 的可训练参数量、显

## 块 20

## 2.1 Rank表示什么？

Rank，中文叫**矩阵的秩**。在LoRA里，你可以把它直接理解为**低秩矩阵中间那个“通道数”**——它决定了你要用多大的“中间尺寸”去近似原来的大矩阵。

我们直接看例子，这样最直观。

假设原始权重矩阵是：

```text
W: 4096 × 4096
```

这个矩阵很大，直接训练它成本太高。LoRA的思路是：不直接改W，而是训练两个小矩阵A和B，让它们的乘积去模拟W的变化量ΔW。

如果你设置：

```python
r=8
```

那么这两个小矩阵的尺寸分别是：

```text
A: 8 × 4096
B: 4096 × 8
```

注意看，**中间的维度就是8**，这个8就是rank的值。

最后，LoRA的更新量是：

```text
ΔW = B × A
```

把B和A乘回去，得到的结果仍然是：

```text
4096 × 4096
```

也就是说，**不管r取多少，最终ΔW的尺寸都和原始W一样**，能正常加到W上。区别只在于：r越大，中间维度越宽，能“装”的信息越多，但训练的参数也越多、成本越高；r越小，参数越少、越省资源，但能表达的变化也越有限。

**一句话总结**：rank就是LoRA两个小矩阵之间的“瓶颈宽度”，它控制着你在“效果”和“成本”之间的平衡点。

---

**本块学习检查**：如果原始权重是 `1024 × 1024`，你设置 `r=4`，请写出A和B的尺寸，以及ΔW乘回去后的尺寸。

## 块 21

### 2.2 Rank控制什么？

**一句话理解**：Rank决定LoRA能学习多少种变化方向——就像调色盘上的颜色种类，颜色越多，能画出的风格越丰富。

**低rank（如`r=4`）**：模型只能学习较少的变化模式。  
✅ 参数少、显存低、训练快  
❌ 表达能力有限，复杂任务可能“学不动”

**高rank（如`r=64`）**：LoRA拥有更多自由度。  
✅ 能学习更复杂任务、适应能力更强  
❌ 参数更多、显存增加、训练成本提高

---

**动手验证**（结合你的LoRA微调目标）：  
在代码中分别设置`r=4`

## 块 22

## 3.1 不使用LoRA：全参数训练的代价

先看最直接的对比——**不微调LoRA时，你要动多少参数**。

假设模型某层权重是 `4096 × 4096` 的矩阵，直接算：

```text
4096 × 4096 = 16,777,216
```

约 **1677万参数**。这意味着，全量微调时，这1677万个数值都要更新、都要存梯度。

---

## 3.2 使用LoRA：参数瞬间缩水

LoRA的思路是**不碰原矩阵**，只训练两个小矩阵。参数量公式：

```text
参数量 = r × (输入维度 + 输出维度)
```

**当 r=

## 块 23

### 4.1 Rank太小的问题

想象你让LoRA用 `r=2` 去学一个新任务——这就像给一个学生只发两张便签纸记笔记，空间太小，根本写不下重点。

**后果**：
- **学习能力不足**：模型“记不住”新知识；
- **欠拟合**：训练集都没学好，更别说泛化；
- **新任务知识无法充分表达**：想表达但“纸”不够。

**你会看到的现象**：
```text
训练loss下降慢
效果提升有限
```

---

**动手验证（hands-on）**：如果你在微调时发现loss曲线“磨蹭”着不降，先检查 `r` 是不是设成了2这类小值。试着调大一档（比如 `r=8`），观察loss下降速度是否明显变快——这是最直接的“Rank太小”信号。

---

### 4.2 Rank太大的问题

反过来，如果你图省事直接上 `r=256`：

**代价**：
- 参数增加；
- 显存增加；
- 训练速度下降。

**核心矛盾**：LoRA的“参数高效”优势被削弱了。

> **别被直觉骗了**：别以为 `r=256` 就“接近完整矩阵”。以文中 `4096×4096` 的权重为例，完整秩是4096，`r=256` 还差得远呢。它只是相对 `r=8` 来说，参数占比明显变大，成本升高，所以“高效优势下降”。

**用数字看清比例**（`d_in = d_out = 4096`）：

| rank r | LoRA参数量 | 完整矩阵 | 占比 |
| ---: | ---: | ---: | ---: |
| 8 | 65,536 | 16,777,216 |

## 块 24

## 4.3 常见rank选择

你是要动手做LoRA微调的人，先记住：下面只是**经验起点**，不是标准答案。

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**实际怎么选？** 看这几个因素：数据量、模型大小、序列长度、batch大小（含梯度累积后的有效batch）、目标模块的选择与数量、GPU显存，以及验证集/基准任务上的表现。

**关键提醒**：同一个`r`，在不同模型规模、不同序列长度/batch、不同目标模块下，显存和效果可能差别很大。别把上表当自动调参标准直接套用。

## 五、Alpha（lora_alpha）是什么？

### 5.1 Alpha作用

LoRA除了 `ΔW = BA`，还会做缩放：

```text
W' = W + (α/r)BA
```

- α：即`lora_

## 块 25

### 6.1 控制LoRA影响强度——动手调一调

我们直接看代码。假设你设了：

```python
r=8
alpha=16
```

缩放系数就是 `16/8=2`，意思是：LoRA 对模型的更新会被**放大2倍**。

如果改成：

```python
alpha=8
```

那么 `8/8=1`，更新保持**原始比例**，不放大也不缩小。

再试试：

```python
alpha=4
```

`4/8=0.5`，LoRA 的影响就**减半**了。

**核心公式**：LoRA 加入模型的更新量会被 `α/r` 这个系数缩放。

> ⚠️ 注意：`α/r` 只是**缩放系数**，不是“实际强度”的唯一决定者。实际落到权重上的更新量，还取决于训练得到的 `A`、`B` 参数值、初始化方式、学习率、优化器、训练步数与数据，以及不同 LoRA 变体的处理方式（例如 RS-LoRA 用 `α/√r`）。相同 `α/r` 只代表缩放比例一致，并不保证更新强度完全相同。

**还有哪些因素影响实际更新强度？**

- **学习率**：直接决定每步对 `A`、`B` 的更新步长。学习率与 `α/r` 会**叠加**影响有效更新幅度——比如把 `α/r` 调大又不下调学习率，容易一起放大更新导致震荡。很多实践中甚至通过调学习率而非

## 块 26

**动手试一下：同步调整的“经验起点”**

你正在学LoRA微调，我们直接看两组配置，感受一下`alpha`和`r`的搭配规律。

**配置1**：`r=8`，`alpha=16` → 比例 = 16 ÷ 8 = **2**
**配置2**：`r=16`，`alpha=32` → 比例 = 32 ÷ 16 = **2**

发现了吗？虽然`rank`变了（8→16），但`alpha/r`的缩放比例**都是2**，保持不变。

**核心要点**：调整`rank`时，常见做法是**同步调整`alpha`**，让`α/r`比例大致稳定。注意，这只是

## 块 27

## 方案1：小模型实验——从这儿开始动手

你的目标是快速跑通LoRA微调，所以**方案1**就是为你准备的起点。

**配置：**
```python
r=8
alpha=16
```

**为什么先选它？**
- **参数少**：只训练少量新增参数，显存压力小；
- **训练快**：迭代速度快，适合你每天2小时的节奏；
- **适合测试**：先验证流程能否跑通，再追求效果。

**动手验证（建议今天做）：**
用这个配置跑一次最小规模的微调，记录训练时间和loss下降情况。跑通后，再对比方案2（`r=16, alpha=32`）的效果差异，你会直观感受到Rank对模型表达能力的影响。

**提醒：** 方案1的效果可能不够好，这正常——它的定位是“验证流程”，不是“追求最优”。先跑通，再调参。

## 块 28

**本块学习检查**：读完本节后，请暂停并回答：为什么“rank越大越好”是错的？如果你把 `r` 从 8 调到 64，但 `alpha` 保持 16，LoRA 的更新强度会发生什么变化？

---

### 误区1：rank越大一定越好？

**不对。** 如果任务本身简单，过大的 `rank` 只会带来两个坏处：

- **浪费显存**；
- **增加训练成本**。

所以，`rank` 不是越大越好，够用就行。

---

### 误区2：只调rank，不调alpha

很多人只改 `r`，却忘了 `alpha` 要跟着动。看个例子：

改之前：

```python
r=8
alpha=16
```

改之后（只动 rank）：

```python
r=64
alpha=16
```

此时缩放因子 `alpha/r` 从 2 降至 0.25。这意味着 LoRA 的更新强度可能被削弱，但实际更新强度还受学习率等因素影响，不能仅凭 `alpha/r` 判断最终效果。

---

### 误区3：alpha越大越好？

**也不对。** 如果 `alpha` 过大，也就是 `α/r` 这个缩放比例太大，会导致：

- **更新过强**；
- **训练不稳定**。

不过要提醒你：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡或发散，还常常来自学习率太高、数据质量差、batch size 太小、精度设置（fp16/bf16）问题、缺少梯度裁剪等。排查不稳定时，要综合看这些因素，别只盯着 alpha。

---

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。你的目标是 LoRA 微调，而 `LoraConfig` 就是整个流程的“总开关”。

**LoRA 训练流程**：

```text
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

`LoraConfig` 负责告诉 PEFT 四件事：

- **LoRA 放在哪里**（target_modules）
- **LoRA 规模多大**（r）
- **LoRA 训练强度如何**（lora_alpha）
- **是否使用 dropout**（lora_dropout）

---

### 动手理解：先跑通一个最小配置

打开你的 Python 环境，输入以下代码（这是后续所有参数讨论的骨架）：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
print(config)
```

**观察**：`print` 输出的内容，就是 `LoraConfig` 帮你生成的完整配置字典。你不需要手动写任何配置文件——PEFT 已经替你封装好了。

---

### 本块学习检查

回答两个问题（不用写代码，口头或笔记即可）：

1. 在 LoRA 训练流程中，`LoraConfig` 位于哪两个步骤之间？
2. 上面代码里，`r=8` 和 `lora_alpha=16` 分别对应 `LoraConfig` 负责的哪一类信息（位置 / 规模 / 强度 / dropout）？

> 提示

## 块 30

```markdown
## 1.2 一个简单示例：动手配置你的第一个LoRA

你正在学LoRA微调，咱们直接上手。下面这段代码就是LoRA最核心的配置，先看它长什么样：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**这四个参数，就是LoRA的“四梁八柱”**，逐个拆开看：

| 参数 | 作用 | 直观理解 |
|------|------|----------|
| `target_modules` | 决定插入哪些层 | 告诉LoRA“改造模型的哪些部位” |
| `r` | 决定LoRA矩阵大小 | 数值越小，新增参数越少，模型越“轻” |
| `lora_alpha` | 控制缩放强度 | 调节LoRA对原模型的影响力度 |
| `lora_dropout` | 控制随机失活 | 防止过拟合，让训练更稳 |

---

## 二、LoRA配置整体流程：三步走

配置不是孤立的，它要嵌入整个微调流程。看这个流程关系：

```text
model → LoraConfig

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些地方‘打补丁’”**。

Transformer 模型的 Attention 部分，内部通常由四个投影层组成：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA 不会默认修改所有层，你需要用 `target_modules` 明确指定。例如：

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

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

所以调整 Q 和 V，通常就能有效改变模型行为——这就是它俩成为常见选择的原因。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名可能不同，**配置前务必先查看模型实际模块名**（用 `model.named_modules()`），再决定 `target_modules`。

---

## 进阶：用 "all-linear" 一键匹配

较新版本的 PEFT 支持 `target_modules="all-linear"`，让 PEFT 自动匹配模型中（几乎）所有线性层，省去手工枚举。可运行示例：

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
model = AutoModelForCausalLM.from_pretrained("your/base-model")  # 请替换为实际模型路径

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",   # 自动匹配所有 nn.Linear 层，包括 lm_head（除非模型配置中排除）
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**权衡点**：覆盖模块变多 → 可训练参数和显存占用增加。是否用全线性层，要在**参数量/显存**与**效果**之间权衡，以验证集结果为准。若你的 PEFT 版本较旧不支持该字符串，请退回显式列出模块名，或升级 PEFT。

---

## 本块学习检查

**动手验证**：运行

## 块 32

### 方案1：只调整Q和V（你的轻量起点）

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么先试这个？** 因为你目标是快速跑通LoRA微调，且喜欢动手实践。这个方案**参数少、训练快**，是常见的轻量起点——注意，它不限于某个模型规模，具体效果要看模型结构和实际表现。

**动手建议**：先用这个配置跑一个小实验，观察训练速度和loss变化，感受“轻量”的含义。

---

**本块学习检查**：对照上面的代码，说出`target_modules`列表里包含哪两个模块？它们分别对应注意力机制的哪两个部分？

## 块 33

## LLaMA 与 GPT-2 的 target_modules 设置

### LLaMA 系列：最常见的配置

对于 LLaMA 系列模型，最常用的 `target_modules` 是：

```python
["q_proj", "v_proj"]
```

这意味着 LoRA 只注入到注意力机制中的 **Q（查询）** 和 **V（值）** 投影层。这是社区验证过的默认选择，兼顾效果与效率。

### GPT-2：注意权重布局差异

GPT-2 的配置则不同，通常使用：

```python
["c_attn"]
```

**为什么不一样？** 因为 GPT-2 的注意力层用的是 `Conv1D`（而非普通 `nn.Linear`），它将 Q/K/V 三个投影合并放在一个 `c_attn` 模块里。更关键的是，`Conv1D` 的权重形状是 `(in_features, out_features)`，**与 `nn.Linear` 的 `(out_features, in_features)` 正好相反**。

**你必须做的设置**：在这类层上使用 LoRA 时，需要设置 `fan_in_fan_out=True`，PEFT 才能按转置布局正确注入低秩矩阵。

可运行示例（GPT-2）：

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
    task_type="CAUSAL_LM",
    target_modules=["c_attn"],   # GPT-2 的 Q/K/V 合并投影
    fan_in_fan_out=True,         # Conv1D 权重是 (in, out) 布局，需置 True
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**版本差异提示**：较新版本的 PEFT 对 GPT-2 等已知模型可能自动推断并处理 `fan_in_fan_out`，不显式设置也可能正常工作。但为了跨

## 块 34

## 五、BERT的target_modules配置

BERT的LoRA配置**不能照搬**其他模型的写法。以LLaMA系为例，它们常用`["q_proj", "v_proj"]`，但BERT内部的模块名不同。

**动手验证**：加载你的BERT模型后，运行以下代码查看实际模块名：

```python
for name, _ in model.named_modules():
    print(name)
```

你会看到类似`query`、`value`这样的名字（不同版本可能略有差异），然后据此配置：

```python
target_modules = ["query", "value"]
```

**匹配机制**：`target_modules`按模块名做**后缀匹配**——写`"q_proj"`会命中所有以`q_proj`结尾的层（各层各头都会生效），也可以传完整名字精确定位，或使用正则。

**重要提醒**：
- **GPT-2（Conv1D）**：GPT-2的注意力用`Conv1D`而非`nn.Linear`，模块名是`c_attn`（q/k/v合并），命名和结构都不同于BERT/LLaMA，不能照搬。
- **版本差异**：不同Transformers/PEFT版本模块命名可能变化，`"all-linear"`等便捷写法依赖较新版本。

## 块 35

### 6.1 r是什么？

`r`，即 **rank**（秩），是 LoRA 中控制低秩矩阵维度的核心参数。它决定了公式 `ΔW = BA` 中 A、B 两个矩阵的中间维度大小。

**直观理解**：想象你要微调一个 `4096×4096` 的权重矩阵。直接更新它需要海量参数，而 LoRA 的思路是“绕个弯”——用两个小矩阵的乘积来近似这个更新量。

- 设置 `r=8` 时：
  - 矩阵 A 的维度是 `4096×8`
  - 矩阵 B 的维度是 `8×4096`

这样，你实际训练的参数从 1600

## 块 36

### 7.1 alpha作用：控制LoRA更新幅度

LoRA的最终更新公式不是简单的 `BA`，而是：

```text
(alpha/r)BA
```

这里的 `alpha` 就是用来控制LoRA更新幅度的关键参数。

**动手算一算：**

假设你配置：

```python
r=8
lora_alpha=16
```

那么缩放因子就是：

```text
16/8=2
```

这意味着LoRA的影响被**增强**了（更新幅度放大2倍）。

---

### 7.2 alpha常见设置：经验起点

一个常见的**起点**是：

```text
alpha = 2 × r
```

> 注意：这只是经验起点，不是通用规则。你也可以保持alpha固定而单独调r，或用其它比例——最终以验证集表现来定。

**

## 块 37

**本块学习检查**：读完本块后，请用一句话回答：dropout值设得越大，模型越容易欠拟合还是过拟合？（答案：越容易欠拟合）

---

**dropout在LoRA里干什么？**

训练时，它随机“关掉”一部分LoRA连接，防止模型死记硬背训练数据（即过拟合）。

**怎么配？**

```python
lora_dropout=0.05
```

意思是：5%的LoRA路径随机失活。

**选多少？**

经验起点（不是死规则，要看验证集效果调）：

- 数据少：约0.05~0.1
- 数据多：约0~0.05

如果训练数据非常丰富，可以直接

## 块 38

好的，我们直接进入正题。这份配置是LoRA微调的“最小可行方案”，特别适合你手头7B模型的测试场景。

**核心配置解读**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：决定新增参数的“宽度”
    lora_alpha=16,      # 增量缩放系数：控制低秩增量ΔW的缩放比例，与r共同决定LoRA更新的幅度
    target_modules=[    # 插入位置：只改造模型的q和v投影层
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,  # 防止过拟合，训练时随机丢弃部分更新
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 指定任务为因果语言模型
)
```

**参数速查表**

| 参数 | 值 | 作用 |
|------|-----|------|
| r | 8 | LoRA容量 |
| alpha | 16 | 增量缩放系数 |
| target_modules | q/v | 插入

## 块 39

## 9.1 基础配置：跑通一个完整的LoRA微调流程

你正在学习LoRA微调，目标很明确：用最少的代码把BERT改造为可训练的序列分类模型，并确保**训练完的东西能完整保存、正确加载**。这一节我们直接上手，跟着代码走一遍。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：BERT 采用 Apache 2.0 许可证，可自由商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练时需根据 batch size 适当增加
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

注意：`num_labels=2` 表示二分类任务，此时BERT顶部的分类头是**随机初始化**的，它不在预训练权重里，所以必须参与训练，否则分类结果就是乱的。

### 第二步：配置LoRA参数

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

逐项理解：

- `r=8`：LoRA低秩矩阵的秩，秩越大参数量越多，表达能力越强。
- `lora_alpha=16`：缩放系数，实际缩放比例为 `lora_alpha / r`，这里即 16/8=2。
- `lora_dropout=0.05`：防止过拟合的Dropout比例。
- `bias="none"`：不训练偏置项。
- `task_type="SEQ_CLS"`：声明任务是序列分类。
- `target_modules=["query", "value"]`：只改造BERT注意力中的query和value两个模块。
- `modules_to_save=["classifier"]`：**这是本块的重中之重**。它告诉PEFT：分类头也要参与训练，并且要随Adapter一起保存。如果不加这一项，重载模型时分类头会退回随机初始化，训练就白费了。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

这一步把基础模型包装成PEFT模型，打印出的可训练参数应同时包含LoRA注入的权重和分类头权重。

### 第四步：训练并保存

```python
# ...（此处进行训练，略）...

# 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")
```

注意：`save_pretrained` 保存的是 **Adapter + `modules_to_save` 模块**，不是完整的基础模型。所以加载时必须先加载同一个基础模型，再挂载Adapter。

### 第五步：重新加载并验证

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证
```

## 块 40

## 十、普通文本生成：LoRA参数怎么配？

你已经掌握了 `r`、`lora_alpha`、`lora_dropout` 各自的作用，现在来看一个最直接的场景——**普通文本生成**（比如续写、生成故事）。这类任务对模型能力的改动需求不大，参数可以设得比较保守。

**推荐配置：**

```python
r=8
alpha=16
dropout=0.05
```

**怎么理解这组数？**

- `r=8`：低秩矩阵的秩设为 8，意味着用较小的参数量去适配任务。普通文本生成和预训练任务差距不大，8 足够。
- `alpha=16`：缩放系数是 r 的两倍，即 `α/r = 2`。这个比例保证微调时权重更新的幅度适中，不会因为缩放过大导致训练不稳定。
- `dropout=0.05`：5% 的随机失活，防止过拟合。文本生成任务数据量通常不小，轻度 dropout 即可。

**对比一下其他任务**（帮你建立直觉）：

| 任务类型 | r | alpha | dropout | 说明 |
|---------|---|-------|---------|------|
| 普通文本生成 | 8 | 16 | 0.05 | 任务与预训练差距小，低秩即可 |
| 指令微调 | 16 | 32 | 0.05 | 需要更强的适配能力，秩翻倍 |
| 领域知识增强 | 32 | 64 | （未指定） | 知识注入需求大，秩进一步加大 |

规律很简单：**任务和预训练差距越大，r 和 alpha 就越大**。alpha 始终是 r 的两倍，保持 `α/r = 2` 的稳定缩放比。

---

## 十一、你的电脑跑得动吗？

分两种情况：

**只写 LoraConfig 配置代码**（不加载模型）：CPU 就行，内存 4GB+ 足够。

**要执行 `get_peft_model()` 并加载真实大模型**：

- 最低要求：NVIDIA CUDA GPU，显存

## 块 41

**动手配置：普通文本生成的LoRA参数**

你正学习LoRA微调，下面这段配置是文本生成任务的常用起点，直接看代码和解释：

```python
LoraConfig(
    r=8,                # LoRA容量：值越大，可学习参数越多
    lora_alpha=16,      # 缩放系数：实际缩放为 alpha/r = 16/8 = 2，用于控制低秩矩阵更新的幅度
    target_modules=["q_proj", "v_proj"],  # 改哪里：只改注意力层的q和v投影
    lora_dropout=0.05   # 防过拟合：训练时随机丢弃5%的LoRA参数
)
```

**四个参数速记

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思路——冻结原模型，只训练一小部分参数。但这里有个关键点：**LoRA只是减少了“训练”的参数，并没有减少“加载”的负担**。

用一张图理解：

```text
原模型参数：冻结（但必须完整加载到显存）
LoRA参数：训练（只占很小一部分）
```

**问题出在哪？**

以7B模型为例，如果用FP16精度加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这还只是**模型权重**本身。实际训练时，你还需要额外空间存放：

- **激活值**：前向传播的中间结果
- **梯度**：反向传播的计算结果
- **优化器状态**：如Adam的动量项

把这些都算上，显存需求会**远高于14GB**。这就是为什么即使有了LoRA，普通消费级显卡依然跑不动大模型——**瓶颈不在训练参数，而在基础模型的加载**。

---

**动手验证一下**：打开你的Python环境，用`transformers`加载一个1B左右的模型（

## 块 43

**QLoRA核心思想：让大模型微调“瘦身”**

你正在学LoRA微调，QLoRA就是它的“省显存版”。核心一句话：**把基础模型压缩成4-bit，再挂上LoRA训练**。

**流程速览**（记住这个顺序）：
FP16基础模型 → 4-bit量化 → 加载量化模型 → 插入LoRA → 训练LoRA参数

**和普通LoRA的区别**：
- LoRA：FP16模型 + LoRA
- QLoRA：4-bit模型 + LoRA

**关键结构**：
- 基础模型：4-bit权重，**冻结**（不训练）
- LoRA：FP16/BF16参数，**训练**

**

## 块 44

## 3.1 模型存储方式：从“斤两”看显存

模型参数在显存里占多大地方，取决于精度。精度越高，每个参数越“重”：

| 类型 | 每个参数占用 |
|---|---|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下**：7B模型用FP16存，需要 `7B × 2 ≈ 14GB` 显存；换成4-bit，只需 `7B × 0.5 ≈ 3.5GB`，直接省下约75%。

> ⚠️ 注意：这个3.5GB是**理论下界**。实际跑起来还会多出量化常数、分组元数据、未量化的模块（如norm、embedding）以及训练缓存等开销。把它当“最低消费”，别当最终账单。

## 3.2 为什么不用普通INT4？

普通INT4整数范围只有 `-8 ~ 7`。但神经网络权重通常是**接近正态分布**、**小数多**、**中心靠近0**的。直接整数映射，精度损失严重。

所以QLoRA

## 块 45

**NF4是什么？**

NF4（NormalFloat4），中文叫“正态分布4位浮点”，是QLoRA论文里针对神经网络权重**正态分布**特点设计的4-bit数据类型。

注意：别把它当成IEEE 754那种通用4位浮点格式。NF4本质是**一套为“近似正态分布”权重定制的离散码本（codebook）**——预先定好16个（2⁴）量化取值点，按标准正态分布的分位数排布，让0附近（权重密集区）划分更细。量化时把每组权重归一化，映射到最近的码本值，存4-bit索引；反量化时按码本和缩放因子还原。

**动手验证**：想象权重大多集中在0附近，

## 块 46

**NF4为什么适合模型权重？**

大模型训练完的权重长什么样？大量参数挤在0附近，整体近似正态分布。普通INT4把范围平均切成[-8,-7,...,7]这样的格子，中心密集区反而分得粗，表达不精细。

NF4的思路是：顺着正态分布的形状来设计量化格子——0附近分得更细，两端分得粗。这样量化误差更小，精度损失更低。

**NF4量化过程（三步）**

1. 原始FP16权重（如0.023、-0.145、0.876）
2. 归一化 → 映射到NF4 → 保存4-bit值
3. 推理时反量化：NF4 → 近似FP16

**动手验证**：拿三个

## 块 47

**动手理解：为什么量化参数也要省？**

你在做LoRA微调时，显存是硬约束。4-bit量化把权重从FP16压到4-bit，但**每组权重还得额外存一个参数**：`scale`。这些“额外参数”同样占显存，积少成多。

**双重量化 = 给“量化参数”再量化一次**

普通量化流程：FP16权重 → 4-bit权重 + 保存scale。

双重量化在此基础上，**把scale参数本身也做一次量化**。即：第一次量化模型权重，第二次量化量化参数。最终结构：FP16权重 → NF4量化 → 保存4-bit权重；scale参数 → 再次量化 → 减少scale存储。

**好处**

## 块 48

## 6.1 为什么需要分页优化器？

训练时显存峰值不只是模型本身，还包括**梯度**和**优化器状态**。比如 Adam 优化器需要额外保存**一阶动量**和**二阶动量**，这两项会显著占用显存。

QLoRA 虽然大幅降低了模型本身的显存占用，但训练过程中仍可能遇到**显存突然不足（OOM）** 的情况——尤其是当梯度计算和优化器状态更新同时达到峰值时。

## 6.2 分页优化器思想

Paged Optimizer 借鉴了操作系统里的**虚拟内存**思路，核心机制是：

> 借助 NVIDIA 的 **Unified Memory（统一内存）**，在显存出现峰值、即将 OOM 时，把优化器状态自动“分页”换出到 CPU 内存，需要时再换回 GPU。

流程如下：

```text
GPU显存
   ↓
出现

## 块 49

## 6.3 类比理解：分页优化器在做什么？

把显存想象成一张**桌面**，把优化器状态（训练中需要保存的动量、方差等中间数据）想象成**文件**。

**普通方式**：所有文件都摊在桌面上。桌面一旦放满，就无法继续工作——这就是显存溢出（OOM）。

**分页优化器**：桌面只放**当前正在用的常用文件**；暂时用不到的文件存进**柜子**；等需要时再取出来放回桌面。这样桌面永远有空间，训练就不会因为显存爆掉而中断。

需要说明的是，这里的“柜子”指的是**CPU内存**。分页优化器利用CPU内存作为后备存储，当GPU显存不足时将优化器状态换出到CPU内存，需要时再换回，从而避免OOM。

---

## 6.4 实际怎么启用？

不需要自己写“换页”逻辑，`bitsandbytes` 库已经实现了。使用前需先安装该库并确保CUDA环境正确：

```bash
pip install bitsandbytes
```

然后在 `TrainingArguments` 里通过 `optim` 参数指定：

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

**两个取值怎么选？**
- `paged_adamw_8bit`：优化器状态用 8-bit 存储，更省显存。
- `paged_adamw_32bit`：精度更高，但占用更多显存。

**注意**：不同版本的 Transformers / bitsandbytes 支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit`

## 块 50

## 7.1 4-bit量化：把模型“瘦身”

你的目标是LoRA微调，而微调大模型最怕显存不够。QLoRA的第一招就是**4-bit量化**，把模型权重从FP16（16位）压缩到NF4（4位），体积压缩为原来的1/4（减少约75%）。

**直观理解**：FP16像是用16个开关记录一个数，NF4只用4个开关，省下的空间自然多。

## 7.2 双重量化：连“尺子”也压缩

量化需要“scale”（缩放系数）来还原数值。QLoRA的第二招是**双重量化**——对量化常数（scale）进行二次量化，以减少其存储开销。

**一句话**：模型瘦身了，连“说明书”也压缩了。

## 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态很吃显存。第三招是**分页优化器**：使用统一内存技术，将优化器状态分页，在GPU显存不足时自动换页

## 块 51

## 4-bit 量化配置：让大模型“瘦身”跑起来

你正在做 LoRA 微调，核心痛点就是显存不够。QLoRA 的 `load_in_4bit` 就是来解决这个问题的——它把模型权重以 **4-bit** 精度加载进显存，而不是常规的 32-bit 或 16-bit。直观理解：相比 32-bit 存储，显存占用可降至约 1/8；相比 16-bit 存储，可降至约 1/4。省下的空间就能塞进更大的模型或更长的序列。

### 三个关键开关

配置里你会看到四个参数，逐个拆解：

1. **`load_in_4bit=True`** — 总开关，开启 4-bit 加载模式。
2. **`bnb_4bit_quant_type="nf4"`** — 指定量化方式为 **NF4**（一种专门为 4-bit 设计的归一化浮点格式，比普通 4-bit 整数更精准）。
3. **`bnb_4bit_use_double_quant=True`** — 开启**双重量化**，对量化参数再做一次量化，进一步压缩显存。
4. **`bnb_4bit_compute_dtype`** — 这个最容易被忽略，下面重点讲。

### 存储精度 vs 计算精度：别搞混

这是新手最容易踩的坑。权重虽然以 4-bit **存储**在显存里（省空间），但真正参与矩阵运算时，会被**反量化**回一个更高精度的格式来算（保证训练稳定）。这个“算的时候用什么精度”，就是 `bnb_4bit_compute_dtype` 控制的。

- 常用 `torch.bfloat16`（BF16）或 `torch.float16`（FP16）。
- 若不设置，默认通常为 float32，但建议显式指定为 bfloat16 或 float16 以提升性能。

### 按 GPU 能力选 BF

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（4-bit 加载）、分页优化器、计算精度这些“零件”。现在把它们拼成一个**能直接跑的最小骨架**——你只需要替换模型名和数据，就能看到 QLoRA 训练全流程。

### 为什么需要这个骨架？

单独看每个配置都懂，但拼在一起容易懵。这个骨架的价值在于：**它展示了从“加载模型”到“开始训练”的完整顺序**，每一步都有明确目的。跟着代码走一遍，你就知道 QLoRA 训练到底长什么样。

### 骨架代码（可跑通）

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
# 模型许可证：请根据所选模型确认并标注许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：加载 7B 级模型时，FP16 精度下需至少 14GB 显存；4-bit 量化可显著降低显存占用

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

# 4) 注入 LoRA（冻结原权重，仅训练低秩矩阵 A 和 B 形成增量更新）
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

### 关键步骤拆解（按执行顺序）

**第 1 步：量化配置**——决定模型以 4-bit 加载。`nf4` 是量化类型，`double_quant` 开启双重量化，`compute_dtype` 根据 GPU 能力自动选 bf16 或 fp16。

**第 2 步：加载模型**——把 `bnb_config` 传给 `quantization_config`，模型就以 4-bit 精度加载。`device_map="auto"` 自动分配设备。

**第 3 步：prepare_model_for_kbit_training**——这是**最容易漏掉的一步**。它专门为 k-bit 训练做准备：处理 norm 层和输出层的精度问题，同时开启梯度检查点所需的输入梯度。注意 `use_gradient_checkpointing=True` 要和后面 TrainingArguments 里的 `gradient_checkpointing=True` 对应。

**第 4 步：注入 LoRA**——冻结原始权重，仅训练低秩矩阵 A 和 B，二者相乘形成对权重的增量更新。`r` 控制秩的大小，`target_modules` 指定注入的模块。

**第 5 步：准备数据**——Causal LM 的 labels 直接复制 input_ids，`DataCollatorForLanguageModeling` 负责动态填充和批处理。

**第 6 步：训练参数**——`paged_adamw_8bit` 是分页优化器，在显存不足时自动将优化器状态换出到 CPU 内存；`gradient_checkpointing` 开启梯度检查点以节省显存。

## 块 53

## 九、把 QLoRA 的四个关键步骤串起来（可跑通骨架）

你现在已经理解了 NF4 量化、双重量化和分页优化器的原理，这一节我们把它们**真正接上线**。下面这 5 步是一个**缺一不可**的链条，每一步都在解决一个具体问题：

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么“瘦身”加载**
这一步是配置“开关”，告诉系统基础模型要以 **4-bit 方式加载**，具体包括：用 **NF4** 量化格式、开启**双重量化**、以及设定**计算 dtype**（BF16/FP16）。没有它，模型就是正常精度加载，后面的省显存都无从谈起。

**第 2 步：`prepare_model_for_kbit_training()` —— 建议调用的模型适配**
建议调用该函数以对量化模型进行适配（如启用梯度检查点、将部分层转为计算精度），但并非强制要求。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA 适配器**
这一步把**可训练的 LoRA 参数**注入到已经冻结的 4-bit 主干模型上。主干保持 4-bit 不动，只有 LoRA 适配器在训练中更新。

**第 4 步：梯度检查点开启其一即可**
建议在 `TrainingArguments` 中设置 `gradient_checkpointing=True`，或在 `prepare_model_for_kbit_training` 中开启 `use_gradient_checkpointing=True`，两者选其一即可，无需同时开启。

**第 5 步：`optim="paged_adamw_8bit"` —— 分页优化器降低显存峰值**
使用分页优化器（如 `paged_adamw_8bit`）可将优化器状态分页到 CPU 内存，降低显存峰值，减少 OOM 风险。

> 以上是“能说明关系”的最小骨架。真实训练中，你需要换成足够规模的数据、合理的超参，并且**按你实际使用的版本**核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数以及 `processing_class`/`tokenizer` 的写法。

---

## 十、你的电脑能跑吗？（配置参考）

**只学原理**：不需要 GPU，纯看代码和概念即可。

**实际运行 QLo

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

先明确一个关键点：**LoRA训练时，基础模型是被冻结的**，不参与参数更新。真正被训练的是额外添加的LoRA参数（Adapter）。

训练结束后，你保存的通常**不是完整模型**，而是两个小文件：

- `adapter_model.safetensors`（LoRA权重）
- `adapter_config.json`（配置信息）

**打个比方**：基础模型像一台电脑主机，LoRA Adapter像外接显卡。训练只更新显卡，主机保持原样。

### 1.2 Adapter有什么特点？

**优点**：
- 文件小，保存快
- 一个基础模型可以搭配多个Adapter，按需切换

比如同一个基础模型，可以分别搭配客服、医疗、法律三个Adapter，互不干扰。

**缺点**：
- 部署时必须**同时加载**基础模型和Adapter，两者缺一不可

---

## 二、为什么需要合并LoRA权重？

既然Adapter有"部署时需要两部分同时存在"的缺点，那有没有办法把两者合成一个完整模型？这就是`merge_and_unload()`要解决的问题——它能把LoRA权重合并回基础模型，得到一个独立的完整模型，方便后续部署和推理。

> # 模型许可证：Llama 2 Community License（商用月活>7亿

## 块 55

### 2.1 推理时的问题

不合并LoRA时，模型是“双件套”：

- **输入** → 基础模型 + LoRA适配器 → **输出**

这意味着每次推理都要：
1. 加载基础模型；
2. 再加载Adapter。

但部署时，你通常只想要一个“完整模型”：

- **输入** → 完整模型 → **输出**

所以，**必须把权重合并**。

---

### 2.2 合并的数学原理

LoRA训练时，权重变化为：

- W' = W + ΔW，其中 ΔW = BA

**合并**，就是把BA加进W，直接得到W'。

合并后，原来的“W + Adapter”就变成了一个整体

## 块 56

## 3.1 merge_and_unload() 是做什么的？

PEFT库提供了一个方法：

```python
merge_and_unload()
```

**一句话理解**：把LoRA训练得到的增量权重“焊死”回基础模型里，然后拆掉PEFT的外包装。

用公式看就是：

```text
基础模型 + LoRA适配器 → 完整模型
```

---

## 3.2 合并前后，模型发生了什么变化？

**合并前**，你的模型是一个`PeftModel`，结构上是：

```text
PeftModel
├── Base Model（基础模型）
└── LoRA Adapter（LoRA适配器）
```

**合并后**，它变成了一个**普通Transformer模型**——LoRA的增量权重已经被融进基础模型的参数里，不再有单独的Adapter层。

**合并的数学本质**：LoRA的增量权重是低秩矩阵A和B的乘积（ΔW = BA），合并操作是权重相加而非替换，即：

```text
W' = W + (α/r)BA
```

其中A和B是低秩矩阵，α是缩放系数，r是秩。

---

## 四、LoRA模型合并的完整流程

整体走一遍，你就知道现在处在哪一步了：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

---

## 五、加载LoRA Adapter（动手前先准备）

假设你已经有训练好的基础模型和Adapter：

```python
base_model_path = "model_path"   # 基础模型路径
adapter_path = "adapter_path"    # LoRA适配器路径
```

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 第一步：加载基础模型
try:
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto')
except (OSError, ValueError) as e:
    print(f"加载基础模型失败: {e}")
    raise

# 第二步：把LoRA适配器挂上去
try:
    model = PeftModel.from_pretrained(model, adapter_path)
except (OSError, ValueError) as e:
    print(f"加载LoRA适配器失败: {e}")
    raise
```

> **注意**：`device_map='auto'`确保模型自动分配到可用设备（如GPU）。若使用Llama-2-7b FP16，需至少14GB显存。

此时模型结构是：`基础模型 + LoRA Adapter`，也就是我们前面说的`PeftModel`。

---

## 六、执行合并：merge_and_unload()

一行代码搞定：

```python
merged_model = model.merge_and_unload()
```

这一步做的事，用数学符号表示：

```text
W（原始权重） + BA（LoRA学习到的增量） → W'（更新后的权重）
```

执行完后，`merged_model` **不再是**`PeftModel`，而是一个**普通Transformer模型**——可以直接当普通模型用了。

---

## 七、导出合并后的模型

合并完不保存等于白干。保存模型和分词器：

```python
try:
    merged_model.save_pretrained("./merged_model")
    tokenizer.save_pretrained("./merged_model")
except (OSError, ValueError) as e:
    print(f"保存模型失败: {e}")
    raise
```

>

## 块 57

## 三、merge_and_unload是什么？（重点）

**一句话理解**：`merge_and_unload` 就是把 LoRA 训练时加在底座模型上的“小补丁”（Adapter）**永久焊死**进底座模型，得到一个完整的、独立的模型。之后推理时，不再需要 PEFT 库，直接用 HuggingFace 的 Transformers 就能加载。

**为什么需要它？**

- **不合并**：模型 = 底座 + Adapter，加载推理时必须用 PEFT 库，部署时多一层依赖。
- **合并后**：模型 = 完整模型，直接 `AutoModelForCausalLM.from_pretrained` 加载，部署简单。

**合并后加载推理（核心代码）**

```python
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程：用户输入 → 完整模型 → 生成结果。

**合并 vs 不合并（快速对比）**

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

**什么时候应该合并？**

合并与否是**权衡**，不是“上线就一定要合并”。取决于：

- 是否需要**多个 Adapter 动态切换**（需要则别合并）
- 部署框架是否**原生支持 LoRA**
- 对**存储/分发体积**的要求
- 对**精度**的接受度（尤其 QLoRA 场景）

**合并后必查清单（避免“能加载但行为不对”）**

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若训练时扩过词表，还要确认 embedding/lm_head 尺寸与词

## 块 58

### 10.1 推荐合并场景

你正在学LoRA微调，目标是最终把模型用起来。什么时候该把LoRA权重合并回基础模型？记住两个最典型的场景。

**场景1：上线部署（单一Adapter、追求简单）**

想象生产环境：服务器 → 直接加载模型 → 提供API。当只需要**一个固定的Adapter**、且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载）时，合并通常更方便。

> 注意：如果推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或你需要在同一基础模型上**热切换多个Adapter**，那么不合并、以Adapter形式部署反而更灵活；此时是否合并要看框架能力与业务需求。

> 部署提示：请确认模型许可证类型（如Llama 2 Community License，商用月活>7亿需Meta授权）；并注意显存要求（如Llama-2-7b FP16需至少14GB显存）。

**场景2

## 块 59

**本块学习检查**：读完本节后，请暂停并回答——如果你正在用LoRA同时做“情感分类”和“命名实体识别”两个任务，你会选择合并还是保留Adapter？为什么？

---

### 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个Adapter（如A、B、C），每个负责不同任务。**保留Adapter更灵活**——想用哪个任务就切换哪个，合并反而会“焊死”能力。

**场景2：持续训练**

如果你还打算继续调整LoRA参数，**保留Adapter**。合并后权重就固定了，后续微调得从头再来。

---

### 十一、QLoRA合并注意事项（重点）

QLoRA有个特殊点：

## 块 60

## 11.1 直接合并可能遇到的情况

你在用 QLoRA 训练后，最想做的事可能就是直接把 LoRA 权重合并回基础模型。但要注意：**直接在 4-bit 量化状态下合并，可能出问题**。

具体来说，可能遇到两种情况：

- **精度损失**：4-bit 权重需要先反量化再合并，这个过程与全程用高精度计算并不完全等价，结果会有细微偏差；
- **权重类型不匹配或版本不支持**：某些 PEFT / bitsandbytes / Transformers 版本不允许在量化状态下直接调用 `merge_and_unload`。

> ⚠️ 注意：上面说的是**可能**出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你用的**版本**、量化配置，以及是否先做了反量化。**以你所用版本的实际运行结果为准。**

---

## 11.2 常见做法：先反量化到高精度再合并

既然直接合并有风险，更稳妥的做法是：**不在 4-bit 状态下合并，而是先用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一个 Adapter 后合并。**

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

这样合并发生在高精度权重上，就避开了 4-bit 状态下的类型和精度问题。

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
#    先检查 CUDA 可用性，CPU-only 环境回退到 float32
if torch.cuda.is_available():
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    compute_dtype = torch.float32
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=compute_dtype,
        device_map="auto",        # 显存不足时可改为 device_map={"": "cpu"} 在 CPU 上合并
    )
except OSError as e:
    raise RuntimeError(f"基础模型加载失败，请检查路径是否正确: {base_model_path}") from e

# 2) 挂载训练得到的同一份 Adapter
try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
except OSError as e:
    raise RuntimeError(f"Adapter 加载失败，请检查路径是否正确: {adapter_path}") from e

# 3) 合并；safe_merge=True 会在合并时做数值检查（如检测 NaN）
merged_model = model.merge_and_unload(safe_merge=True)

# 4) 保存完整模型与 tokenizer（tokenizer 从基础模型或训练目录加载均可，保持与训练一致）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)
```

> 📋 **模型许可证**：合并后的模型继承基础模型的许可证。例如，若基础模型为 Llama 2，则适用 Llama 2 Community License（商用月活>7亿需Meta授权）。部署前请确认所用基础模型的许可证类型及合规要求。

## 块 61

### 直接合并QLoRA模型：三个关键提醒

你已完成LoRA微调，现在要把QLoRA模型合并导出。直接合并时，注意这三点：

**1. 版本差异**  
不同PEFT版本对“量化模型能否直接合并”支持不同。部分版本要求先反量化。务必查阅所用版本的`merge_and_unload`文档。关键操作：加载基础模型时**不要传**`quantization_config`或`load_in_4bit`，这样得到高精度权重再合并。

**2. 内存开销**  
以高精度重新加载完整基础模型，会占用约等于一份FP16/BF16完整模型的内存/显存。显存不足时，把`device_map`改为`{"":

## 块 62

## 合并校验：给模型合并加一道保险

`merge_and_unload` 有个 `safe_merge=True` 参数，合并时会自动做数值检查（比如检测有没有出现 NaN），相当于一道保险。代码就一行：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

**为什么还要做一致性验证？**

合并本身是数学运算，但浮点数计算可能引入微小偏差。所以合并完成后，建议对比合并前后模型在**同一批输入**上的输出，确认没出大问题。

**验证流程（三步）：**

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比两者的 logits /

## 块 63

## 合并后别急着用——先做一次“一致性体检”

你刚在 §11.2 里完成了 QLoRA 模型的合并，现在 `merged_model` 已经拿到手了。但先别急着部署，QLoRA 场景下合并本身带有近似性，直接上线有风险。这一节教你一个“体检”方法：把合并前后的模型输出做对比，用数据确认合并质量。

### 为什么要单独加载一份“合并前”的模型？

代码注释里强调了一个关键陷阱：`merge_and_unload()` 返回一个新的合并后的模型，原始 `PeftModel` 对象保持不变，仍可用于推理或进一步操作。因此，如果需要对比合并前后的模型，可以直接使用原始对象作为合并前的基准，无需重新加载。

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

注意：`base_model_path`、`adapter_path`、`compute_dtype` 这些变量都沿用 §11.2 里的定义，不用重新设置。

### 第一项检查：logits 最大绝对差

用同一个固定输入（`prompt`），分别让 `model_before` 和 `merged_model` 跑一次前向传播，比较输出的 logits 差异：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果？** 这个值越接近 0，说明合并前后的模型行为越一致。QLoRA 因为涉及量化，通常不会完全等于 0，而是得到一个“较小的近似值”——这是正常现象，不用追求完美一致。

### 第二项检查：贪心生成文本是否一致

logits 差异是数值层面的，但你可能更关心实际输出。用 `do_sample=False` 去掉随机性，让两个模型各自生成一段文本，直接对比：

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)

## 块 64

## 合并后如何确认没“合坏”？——校验与一致性验证

`merge_and_unload()` 默认返回合并后的新模型，原始模型保持不变。若需原地合并，可显式设置 `merge=True`。

**为什么建议用返回的新模型做对比？** 合并完成后，用返回的新模型与合并前保存的原始模型进行对比，`max_abs_diff` 能真实反映合并带来的权重变化。

**正确做法：** 在合并前，额外加载一份**未合并**的模型作为对照（比如 `base_for_ref` + `model_before`），用它来和合并后的模型做对比。

---

### 显存不够怎么办？

如果两个模型同时放不下，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍推理，把 logits 或生成文本**存下来**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

---

### 结果预期：别追求“逐位相同”

- **QLoRA 场景**：合并前后通常是**近似一致**——`max_abs_diff` 一般是个较小的非零值，生成文本可能有个别 token 差异。重点不是“完全一样”，而是**确认没有明显退化或出现 NaN**。
- **LoRA（非量化）场景**：一致性通常更高，`max_abs_diff` 可以非常小。

**是否可接受？** 以你的任务指标和验证集结果为准，而不是看数值是否为零。

---

### 硬件配置参考（不是

## 块 65

## 合并后出问题？先看这张排查表

你在 LoRA 微调的路上已经走完了大半程，现在到了最容易踩坑的收尾阶段。合并模型时遇到报错或异常结果，别慌——绝大多数问题都有固定套路可解。下面这张表，就是给你准备的“故障速查手册”。

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 `PeftModel`（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到 `PeftModel` 再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积 Adapter | 无需处理；若在意体积可保留 Adapter 不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是 dtype 用了 FP32、未做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存 tokenizer；或特殊 token / generation_config 未一并导出 | 重新 `save_pretrained` 并核查 tokenizer、generation_config、config |
| QLoRA 合并失败或结果异常 | 在 4-bit 状态直接合并、PEFT/bitsandbytes 版本不支持、量化包装未正确反量化、内存/显存不足 | 先反量化到 FP16/BF16 再合并；核对版本；必要时用 CPU 合并；开 `safe_merge=True` 并做一致性验证 |

## 重点拆解：QLoRA 合并失败的“反量化”到底怎么做？

这是最容易误解的一步。**“反量化”不是让你手动逐层还原 4-bit 权重**——那既繁琐又容易出错。正确的做法是：

**重新加载一份不带量化配置的基础模型。**

具体来说，用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`，注意**不传** `quantization_config` 或 `load_in_4bit`，这样得到的就是高精度权重。然后用 `PeftModel.from_pretrained` 挂载**同一份 Adapter** 再合并。

这样做的核心价值：合并发生在高精度权重上，天然避开了“在 4-bit 状态下合并”的类型/精度问题。

**版本兼容性提醒**：不同 PEFT / bitsandbytes / Transformers 版本对

## 块 66

## 路线A：合并后部署完整模型

你已经完成了 LoRA 微调，现在面临一个选择：**怎么把模型部署上线？** 本节先讲最直接的一条路——**合并后部署**。

### 核心流程

```
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

三步走完，你的模型就是一个**普通模型**了。

### 这条路的特点

**优点：**
- **部署链路简单**——推理框架把合并后的模型当普通模型加载，**完全不需要理解 LoRA** 是什么；
- **推理时没有 Adapter 切换成本**——模型已经是完整的，不用在多个 LoRA 之间切换。

**代价：**
- 每个 Adapter 都要保存/分发一份**完整模型**。如果你有多个任务、多个 Adapter，存储成本会很高。

### 什么时候选路线A？

- 你只做**单任务**，不需要频繁切换不同 LoRA；
- 你希望部署环境**最简单**，不想依赖推理框架对 LoRA 的特殊支持。

---

## 路线B：保留 Adapter，动态服务（对比了解）

与路线A相对，路线B是**保留 LoRA Adapter**，用支持 LoRA 的推理框架（如 **vLLM**）在**同一份基础模型**上运行时加载/切换多个 Adapter。

```
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

**特点：**
- 多任务/多租户时**只需一份基础模型 + 多个小体积 Adapter**，节省显存与存储；
- 支持**热切换**，不用为每个任务各存一份完整模型；
- 代价：依赖框架对 LoRA 的支持与配置，链路比路线A略复杂；不同框架/版本对 LoRA 的支持范围和限制不同。

> **vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）**：
>
> ```text
> vllm serve <base_model_path> \
>     --enable-lora \
>     --lora-modules adapterA=/path/to/lora_adapter_a adapterB=/path/to/lora_adapter_b \
>     --max-loras 2 \
>     --max-lora-rank 16
> ```
>
> 其中 `--enable-lora` 开启多 LoRA 服务，`--lora-modules 名字=路径` 声明可用 Adapter（**路径必须指向未合并的 LoRA Adapter 目录**，即含 `

## 块 67

## 14.3 如何选择：合并还是保留？

你已经学完 LoRA 微调，现在到了部署前的关键岔路口。记住一句话：**合并不是唯一部署方式**。选择哪条路，看你的实际场景：

**路线A（合并）**：如果你只训练了一个 Adapter，追求部署最简单，直接合并进基础模型，得到一个完整模型，独立运行，不依赖额外框架。

**路线B（保留Adapter动态服务）**：如果你要在同一个基础模型上服务多个 Adapter，且你的推理框架原生支持 LoRA，那就保留 Adapter，运行时动态加载切换，省去反复合并的麻烦。

**特别提醒（QLoRA）**：如果你用的是 QLoRA，精度敏感，无论走哪条路线，都建议做合并前后/上线前的一致性验证，确保输出一致。

> 具体框架用法留到第15板块（vLLM部署）再展开，现在只需建立这个认知。

---

**本块学习检查**：对照你的项目，回答——你的场景是单一 Adapter 还是多 Adapter？推理框架是否原生支持 LoRA？据此判断你该走

## 测评题目

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）
