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

以7B参数模型为例，训练时你需要同时保存三类东西：

- **参数**：模型本身的70亿个权重
- **梯度**：每个权重对应的更新方向
- **优化器状态**：Adam等优化器维护的动量信息

这三者叠加，显存需求会飙升到远超模型本身大小的程度。结果就是——**普通消费级GPU根本跑不动**。

这就是为什么需要参数高效微调：不更新全部参数，只更新一小部分，大幅降低显存和算力门槛。

---

**快速自查**：你能说出全参数微调时，显存被哪三样东西占满吗？如果答不上来，回头再看一遍上面的列表。

## 块 2

**PEFT：参数高效微调**

核心思想：**不修改整个大模型，只训练少量新增参数**。

流程：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter

| 方式 | 训练参数 | 显存需求 |
|------|---------|---------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**本块学习检查**：对照上表，说出LoRA相比全参数微调在训练参数和显存需求上的差异。

## 块 3

**LoRA是什么？——用“小补丁”微调大模型**

想象你有一个训练好的大模型，它的核心权重是 `W`。传统微调要更新整个 `W`，又慢又费显存。LoRA（低秩适配）的思路是：**`W` 保持不动**，在旁边加一个“补丁” `ΔW`，最终模型变成 `W + ΔW`。

这个补丁怎么来？它由两个小矩阵相乘并乘以缩放因子得到：**`ΔW = (α/r) × B × A`**。假设 `W` 的形状是 `d_out × d_in`，那么 `A` 约为 `r × d_in`，`B` 约为 `d

## 块 4

## 3.1 LoRA 的问题与 QLoRA 的解决思路

你已经知道 LoRA 能大幅减少**训练**的参数，但有个关键点别忽略：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个 7B 参数的模型，如果用 FP16 精度加载，大约需要 **14GB 显存**。也就是说，就算 LoRA 只训练一小部分参数，你的 GPU 也得先装得下整个基础模型才行。

QLoRA（Quantized LoRA，量化版 LoRA）的思路很直接：**先把模型变小，再加载 LoRA 训练**。

流程是这样的：

```text
FP16模型 → 4bit量化 → 显存占用大幅下降 → 加载LoRA开始训练
```

训练时的结构就是：

```text
4bit基础模型 + LoRA参数 → 训练
```

核心就一句话：**用 4bit 精度存基础模型，省下显存，LoRA 参数照常训练**。

要跑 QLoRA，你需要这几个组件：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中 `bitsandbytes` 负责三件事：**低精度量化、4bit 加载、8bit 加载**。

> ⚠️ **重要说明**：本教程的示例和验证统一以 **NVIDIA CUDA GPU** 为准，这是项目支持范围，也是目前资料最全、验证最充分的路径。但“QLoRA 只能用 NVIDIA”并不是技术必然——bitsandbytes 近年已在扩展对 CPU、AMD ROCm、Intel 等后端的支持。**各后端支持程度差异很大**，NVIDIA CUDA 是完整支持，其他后端可能是实验性或部分支持，功能、性能和稳定性都可能打折扣。如果你的硬件不是 NVIDIA

## 块 5

personalization_action: explain  
personalization_reason: 学习者目标为LoRA微调，偏好hands-on，需快速上手；原文配置信息较散，按“先看结论、再看条件、最后自查”重排，突出可操作性与显存弹性理解。  
fact_source_chunk_ids: ["KB-12-04"]  
teaching_tasks: []  

### 电脑配置批注：跑LoRA/QLoRA，你的机器够吗？

先给你结论，再解释为什么。你的目标是**LoRA微调**，所以重点看第一张表；QLoRA是“显存不够时的备选方案”，第二张表帮你判断自己的卡能不能用。

#### 一、LoRA最低配置（先看这张）

| 项目 | 要求 |
|------|------|
| CPU | 4核以上 |
| 内存 | 16GB+ |
| 磁盘 | 50GB以上 |
| GPU | **推荐NVIDIA GPU** |
| 显存 | 最低8GB，**推荐12GB+** |

> **动手自查**：打开任务管理器（Windows）或`nvidia-smi`（Linux），核对你的CPU核数、内存大小、GPU型号和显存。如果显存≥12GB，直接按LoRA路线走；如果只有8GB，也能跑，但batch要调小。

#### 二、QLoRA要求：显存不够时的“省显存方案”

QLoRA通过**4-bit量化**压缩基础模型，所以显存占用通常低于LoRA全精度加载。但注意：**它只是降低门槛，不是无门槛**。

**经验参考值（7B级别模型）**：
- 参考下限：**约8~12GB**（短序列、小batch）
- 推荐：**16GB+**

**为什么是“参考”而不是“保证”？** 因为实际显存占用受这些条件影响，条件不同结论就不同：
- **模型规模**：7B是基准，13B/70B显存需求显著上升
- **序列长度**：短序列（≤512）省显存，越长激活值占用越大
- **batch大小**：1~2是小

## 块 6

**5.1 Python版本检查（动手第一步）**

在终端里先确认你的Python版本，这是所有后续操作的地基：

```bash
python --version
```

建议输出为 **Python 3.10+**。如果版本低于3.10，建议先升级Python再继续，否则后面装PEFT和bitsandbytes时容易遇到兼容性报错。

---

**5.2 PyTorch检查（关键一步，别跳过）**

LoRA/QLoRA依赖PyTorch，而且`bitsandbytes`的CUDA量化路径要求安装**和你本机CUDA匹配的GPU版PyTorch**。直接`pip install torch`通常装到CPU版，QLoRA会直接无法使用。

请用[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)选对应命令。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

装完跑这段验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # torch编译用的CUDA版本
```

看到`True`表示PyTorch能识别GPU。但注意：`True`只说明**PyTorch侧**能用GPU，**不代表bitsandbytes的4-bit量化路径可用**——后者还要看bitsandbytes是否匹配你的CUDA。这个留到后面章节专门验证。

---

**5.3 查看GPU信息**

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

**六、安装PEFT库**

**6.1 PEFT是什么？**

PEFT是Hugging Face官方的参数高效微调库，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter。你的LoRA微调目标就靠它。

**6.2 安装并验证**

```bash
pip install peft
```

```python
import peft
print(peft.__version__)
```

能打印版本号即安装成功。

---

**七、安装bitsandbytes**

这是QLoRA 4-bit量化的核心依赖，安装时确保和你的CUDA版本匹配。具体检查方法在后续章节展开。

## 块 7

**bitsandbytes 是干嘛的？**

一句话：它是让模型“瘦身”跑起来的工具，负责**8bit/4bit量化**，也是**QLoRA训练**的必备依赖。

**为什么你必须装？**

本教程的QLoRA方案，就是用 `bitsandbytes` 把基础模型压到4bit，再配合LoRA微调。所以它是前置依赖，**必须安装**。

> 注意：QLoRA本身不绑定某个量化库，但本教程统一用 `bitsandbytes` + NVIDIA CUDA演示。如果你用别的量化后端，以对应文档为准。

**动手装（基础安装）：**

```bash
pip install bitsandbytes
```

装完别急着跑，下一步做CUDA兼容检查（重点），确保它能调用你的GPU。

## 块 8

**动手前先查兼容性**

你目标是跑LoRA微调，而bitsandbytes是量化工具，它依赖CUDA来调用GPU。如果版本不匹配，后果很直接：CUDA不可用、量化失败、模型加载报错——任何一个都会卡住你的实验。

**两步检查，30秒搞定**

1. **看CUDA版本**：终端运行`nvidia-smi`，在输出里找到`CUDA Version`字段，记下数字。
2. **验证bitsandbytes**：运行`python -c "import bitsandbytes as bnb; print(bnb.__version__)"`，正常会显示版本号，说明环境通了。

**本块学习检查**：运行上述两条命令，确认输出的CUDA

## 块 9

**本块学习检查**：完成阅读后，请用一句话说明：为什么在Windows上遇到bitsandbytes报错时，优先改用WSL2而不是直接排查版本问题？

---

**9.1 先走推荐路线：WSL2**

在Windows上做LoRA微调，环境路径建议按这个顺序搭：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

核心原因就一条：**Linux环境兼容性更好**。WSL2能给你一个接近Linux的运行环境，省掉很多Windows原生下的坑。

---

**Windows原生可能遇到的问题**

如果你坚持在Windows原生环境跑，最典型的问题是：

**bitsandbytes无法加载CUDA动态库。**

这不是单一原因造成的，常见的有：

- 你装的`bitsandbytes`版本和本机CUDA / GPU版PyTorch不匹配；
- 系统缺少或找不到对应的CUDA动态库。

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配

在Windows上安装LoRA环境时，最常见的坑就是**CUDA版本对不上**。别慌，先跑一条命令看清你的机器：

```bash
nvidia-smi
```

这条命令会告诉你三件事：
- **NVIDIA驱动**是否正常安装
- 驱动支持的**CUDA版本**（注意：这是驱动支持的，不是已安装的）
- 显卡型号

记住：**驱动支持的CUDA版本 ≥ PyTorch要求的CUDA版本**即可，不需要完全相等。

---

## 十、安装完整LoRA环境

假设你已经有基础环境（`transformers`、`datasets`、`accelerate`），现在补上LoRA的核心组件：

```bash
pip install peft bitsandbytes
```

**关键前置**：确保已安装**匹配本机CUDA的GPU版PyTorch**，否则后面`import torch`或QLoRA 4-bit会直接失败。CUDA 12.1示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

一步到位装全：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

```bash
accelerate config
```

按提示选择，单GPU场景示例：

```text
Compute environment: This machine
Distributed: No
Mixed precision: fp16
```

生成配置文件`default_config.yaml`。

---

## 十二、LoRA环境验证实验

创建`test_lora_env.py`：

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

看到配置输出即说明：✅ PyTorch正常 ✅ PEFT正常 ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

```python
import torch
import bitsandbytes

print(torch.cuda.is_available())
print(bitsandbytes.__version__)
```

输出`True`说明GPU环境正常，可以开始QLoRA微调了。

---

**本块学习检查**：运行`nvidia-smi`，记录你的CUDA版本号，然后运行`python -c "import torch; print(torch.__version__)"`查看PyTorch版本。对比两者：若PyTorch版本中的CUDA号（如`cu121`）小于或等于驱动支持的CUDA版本号，则环境匹配，可继续后续操作。

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：Windows报错怎么办？**

在Windows上运行LoRA/QLoRA环境时，如果遇到兼容性问题导致报错，推荐直接使用**WSL2**（Windows Subsystem for Linux 2）来解决。

**为什么？**

Windows原生环境对bitsandbytes等工具的兼容性较差，容易出现各种报错。WSL2提供了一个更接近Linux的环境，能有效规避这些兼容性问题。

---

### 常见问题速查表

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版本 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| **Windows报错** | **兼容性问题** | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

### 本块学习检查

**操作验证**：如果你正在使用Windows，请确认你的系统是否支持WSL2（Windows 10版本2004及以上或Windows 11）。若支持，记录

## 块 12

## 一、为什么需要LoRA？

### 1.1 大模型微调的问题

现代大语言模型（如LLaMA、Qwen、Baichuan、Mistral）通常拥有数十亿甚至上百亿参数。以7B模型为例，就是70亿参数。

如果采用传统方式微调——预训练模型 → 更新全部参数 → 保存新模型——你需要：

- **巨大的GPU显存**：训练时不仅要存模型参数本身，还要存梯度和优化器状态（如Adam的动量项），实际显存需求远大于模型大小；
- **大量训练时间**：更新全部参数意味着每次反向传播都要计算所有层的梯度；
- **巨大的存储空间**：每个微调任务都要保存一份完整的模型副本。

### 1.2 全参数微调的具体代价

假设模型有100亿参数，训练时你需要同时保存：

- 模型参数 **W**
- 梯度 **∂W**
- 优化器状态（Adam moment）

这三者叠加，显存需求通常是模型大小的数倍。对于个人开发者或中小团队，这几乎不可行。

---

## 二、LoRA的核心思想

LoRA（Low-Rank Adaptation）的核心思想是冻结预训练权重W，仅训练两个低秩矩阵A和

## 块 13

## 2.1 不直接修改原模型参数

**传统微调的做法**：直接改动原始权重 W，训练后变成 W′，也就是“直接修改 W”。

**LoRA 的做法**：把 W 冻结住，不动它。额外增加一个新参数 ΔW，最终模型使用的是：

```
W′ = W + ΔW
```

- **W**：原模型参数（冻结，不训练）
- **ΔW**：LoRA 新增参数（只训练这个）

---

## 2.2 为什么只训练 ΔW？

大模型已经通过海量数据学会了语言规律、世界知识、推理能力。微调本质上只是**对已有能力做方向调整**，不需要重新学全部参数，只学“任务相关的变化量”往往就够了。

> ⚠️ 注意：这是帮助理解的直觉，不是严格结论。任务与预训练能力差距大、数据充足时，全参数微调仍可能更好。

---

**动手验证**：打开一个 HuggingFace 模型，查看其线性层权重，确认它是可训练的；再对比 LoRA

## 块 14

**目标：LoRA微调 | 风格：动手实践 | 节奏：快**

---

### 3.1 矩阵基础概念

Transformer里的参数，**几乎全是矩阵**。比如注意力层的核心计算：

```text
Q = X · Wq
```

- `X`：输入，形状 `(序列长度, d_model)`
- `Wq`：权重矩阵，形状 `(d_model, d_model)`，这里按同维投影举例

**动手验证**：假设 `d_model = 4096`，那么 `Wq` 就是 `4096 × 4096` 的矩阵，参数量约 **1678万**。你可以自己算一下：`4096 × 4096 = ?`（答案：16,777,216）。

---

### 3.2 直接训练矩阵的问题

微调时，如果直接修改权重，需要学习一个完整的 `Δ

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

想象你有一个 **4096×4096** 的大矩阵 W，直接训练它意味着要更新 **1677万** 个参数——这就是完整微调的代价。

LoRA 的做法是：设 **rank r=8**，把更新量拆成两个小矩阵：
- **A**：4096×8，参数 **32768**
- **B**：8×4096，参数 **32768**

两者相加仅 **65536** 个参数，比完整矩阵少了 **99.6%**。

| 方式 | 参数数量 |
|------|---------|
| 完整矩阵 | 1677万 |
| LoRA(r=8)

## 块 16

**5.1 神经网络参数变化具有低维特征**

微调时，模型参数的变化量 ΔW 并不需要覆盖所有方向。真正影响新任务的有效变化，往往集中在远小于原始维度的少数方向上——即一个**低维子空间**。

以4096维空间为例：看似需要调整4096个方向，但实际起作用的“有效方向”可能远少于这个数。这就是LoRA的核心理据——**权重更新具有较低的内在维度**。

> 注意：这里的“低维”是定性理解，具体有效维度没有通用固定值，会随模型、任务、数据变化，别记成“几十维”之类的确定数字。

因此，用两个低秩矩阵的乘积 **BA** 就能近似替代完整的 ΔW，大幅减少可训练参数量

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张 **4000×4000像素** 的高清图片，完整保存需要 **1600万个数据**。但如果这张图片的主要变化只是**亮度、对比度**和**少量结构变化**，你完全可以用**少量参数**来描述这些变化——不需要保存每一个像素。

LoRA 的核心思想正是如此：**用少量参数描述模型变化方向**。

在微调大模型时，我们不需要重新训练全部参数，只需要找到模型“变化的方向”，用低秩矩阵去捕捉这个方向即可。这就是低秩分解有效的直观原因。

---

## 六、LoRA在Transformer中的位置

Transformer 的基本结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中 **Attention** 包含四个投影矩阵：

```text
Q（查询）、K（键）、V（值）、O（输出）
```

LoRA 在早期实践中**常见**加入 `Q投影` 和 `V投影`，例如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

表示只训练 `Wq` 和 `Wv` 的变化。

> **注意**：只挑 `q_proj`/`v_proj` 是 LoRA 原论文中的**常见选择**，**不是固定标准**。现代实践中也常把 LoRA 加到注意力的全部投影（q/k/v/o）、FFN 的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

**一个覆盖更完整线性层的配置示例**（以 Llama 类模型常见的模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

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

> 上例是**显式列出模块名**的写法；若当前 PEFT 版本支持，也可用 `target_modules="all-linear

## 块 18

### 8.1 rank（r）——用“动手试”代替“背参数”

你正在学 LoRA 微调，目标很明确：用最少显存把模型调好。那 `r` 就是你要亲手调的第一个旋钮。

**它是什么？**
`r` 代表低秩矩阵的维度。代码里写 `r=8`，意思就是让 LoRA 用 8 维的低秩矩阵去近似权重更新。

**调大它，会怎样？**
- 优点：表达能力更强，模型能学更复杂的模式。
- 缺点：参数变多，显存占用增加。

**从哪起步？**
下表只是经验起点，不是铁律：

| 任务 | rank（起点参考） |
|------|----------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

注意：合适的 `r` 和模型规模、目标模块、数据量、任务难度都有关。表只是给你一个“先试哪几个数”的参考，最终要靠验证集说话。

**比查表更可靠的做法——基于验证集选 `r`：**

1. **划出验证集**：从数据里留出 20%（或 10%）不参与训练，后面所有比较都在它上面看指标。
2. **固定其它变量**：把 `target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数都定死，只改 `r`，做受控对比。
3. **小范围网格搜索

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制LoRA对模型参数的**更新强度**。代码中常见写法：

```python
lora_alpha=16
```

实际起作用的是 **alpha/r 的比值**，这个比值作为缩放比例，决定低秩矩阵的更新幅度。

**理解要点**：`r` 决定低秩矩阵的“宽度”（信息容量），`alpha` 决定这个更新的“力度”。两者配合使用，`alpha/r` 越大，LoRA对原模型的影响越强。

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 在训练时随机丢弃一部分LoRA参数，防止LoRA参数过拟合。值越大，丢弃比例越高，训练更保守。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> ⚠️ 注意：上表是**相对趋势**。LoRA的参数量、显存、速度和Adapter大小会随 `target_modules` 选择、秩 `r`、模型层数与 `dtype` 变化。例如把LoRA加到很多模块并用较大 `r` 时，参数量和Adapter体积都会明显上升。“更快/更省显

## 块 20

## 二、Rank（r）是什么？

### 2.1 Rank表示什么？

**一句话理解**：Rank（中文叫“矩阵的秩”）在LoRA里，就是那个“中间维度”的大小——它决定了你要用多“瘦”的低秩矩阵去逼近完整的权重更新。

**直观感受一下**：

假设原始权重矩阵 `W` 的尺寸是 `4096 × 4096`（也就是有约1600万个参数要更新）。

如果不用LoRA，你得更新全部1600万个参数。但用LoRA，你只训练两个小矩阵：

- 设置 `r=8` 时：
  - 矩阵 `A` 的尺寸是 `4096 × 8`
  - 矩阵 `B` 的尺寸是 `8 × 4096`

这两个矩阵相乘：

```
ΔW = (α/r) × B × A
```

结果恰好恢复成 `4096 × 4096` 的形状——和原始 `W` 一模一样。

**关键点来了**：

- 你**没有**直接训练 `4096 × 4096` 的大矩阵，而是训练了两个小矩阵（总共 `8×4096 + 8×4096 =

## 块 21

**Rank控制什么？——动手理解**

想象你在调一个旋钮，Rank（r）决定LoRA能学习多少种“变化方向”。r越小，变化越少；r越大，变化越丰富。

**低rank（如 r=4）**：模型只能学较少变化模式。好处是参数少、显存低、训练快；坏处是表达能力有限，复杂任务可能不够用。

**高rank（如 r=64）**：LoRA自由度更高，能学更复杂任务、适应力更强。代价是参数多、显存增加、训练成本提高。

**动手验证**：在你的微调脚本里，把 `r` 从4改成64，对比训练时的显存占用和速度变化。你会直观感受到

## 块 22

**动手算一算：不用LoRA要训练多少参数？**

假设模型某层权重是 4096×4096 的矩阵，直接全量训练，参数量就是：

4096 × 4096 = 16,777,216，约 **1677万**。

这个数字就是“完整训练”的基准。记住它，后面对比用。

**对比一下：用了LoRA之后**

LoRA的参数量公式很简单：**r × (输入维度 + 输出维度)**。

- 当 r=8 时：8 × (4096+4096) = **65,536**，约6.5万
- 当 r=64 时：64 × 8192 = **524,288**，约

## 块 23

### Rank太小：模型“学不动”了

如果你把 `r` 设得很小，比如 `r=2`，LoRA 的可训练参数就会很少。这会导致：

- **学习能力不足**：可训练参数太少，模型表达能力受限，可能学不到足够的信息。
- **欠拟合**：模型根本没学会训练数据里的规律，表现自然不好。
- **新任务知识无法充分表达**：你想让模型学会的新技能，它可能“装不下”。

**典型表现**：训练时 loss（损失值）下降得很慢，训练完效果提升也有限。不过，`r=2` 并非绝对过小，实际效果需结合任务和数据判断。

---

### 用数字感受一下“太小”和“太大”

为了让你有直观感受，我们算一笔账。假设权重矩阵是 `4096 × 4096`（完整参数量约 1677 万），不同 `r` 值下 LoRA 的参数量占比：

| rank r | LoRA 参数量 | 占完整矩阵比例 |
| ---: | ---: | ---: |
| 8 | 65,536 | ≈ 0.39% |
| 64 | 524,288 | ≈ 3.13% |
| 256 | 2,097,152 | ≈ 12.5% |

## 块 24

## 4.3 常见rank选择：从哪开始？

你正在做LoRA微调，最关心的肯定是：**rank到底选多少？** 下面是一张经验起点表，注意是“起点”，不是“标准答案”：

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**但别急着照搬。** 实际项目中，你需要根据以下因素动态调整：

- 数据量
- 模型大小
- 序列长度
- batch大小（含梯度累积后的有效batch）
- 目标模块（`target_modules`）的选择与数量
- GPU显存
- 验证集/基准任务上的表现

**关键认知：** 同一个`r

## 块 25

## 6.1 控制LoRA影响强度——用Alpha旋钮调音量

你已经知道Rank（r）决定LoRA的“内部容量”，那Alpha（α）是干嘛的？**一句话：Alpha是控制LoRA影响强度的旋钮**。

**核心公式就一个：**

```
缩放系数 = alpha / rank
```

**动手算三组，马上有感觉：**

**场景1：放大影响**
```python
r = 8
alpha = 16
```
缩放系数 = 16 / 8 = **2**，LoRA更新被放大2倍。

**场景2：保持原样**
```python
alpha = 8
```
缩放系数 = 8 / 8 = **1**，保持原始比例。

**场景3：减弱影响**
```python
alpha = 4
```
缩放系数 = 4 / 8 = **0.5**，LoRA影响减半。

---

## 七、Alpha与Rank的关系（重点）

LoRA加入模型的更新量会被 `α/r` 缩放，所以 **`α/r` 是关键缩放系数**。

**但注意——别踩坑：** 不能说“实际强度仅由 `α/r` 决定”。它只是缩放系数，实际落到权重上的更新量还取决于：

- **学习率**：直接决定每步对A、B的更新步长。`α/r` 调大又不下调学习率，容易一起放大导致震荡。很多实践里甚至直接调学习率来控制强度，而不是频繁改α。
- **初始化**：常见实现中A用小随机值、B初始化为0，训练开始时 `ΔW =

## 块 26

**本块学习检查**：请你在终端里分别跑两个配置（r=8/alpha=16 和 r=16/alpha=32），打印出各自的 `alpha/r` 值，确认它们都等于 2。然后只改 rank 为 16、alpha 保持 16，再打印一次比例，观察差异。

---

**配置1**：`r=8, alpha=16` → 比例 = 2  
**配置2**：`r=16, alpha=32` → 比例 = 2  

注意：两个配置的 rank 不同，但缩放比例相同。

**核心要点**：调整 rank 时，常见做法是同步调整 alpha，让 `α/r` 比例大致稳定。这只是**经验

## 块 27

**动手实验：先从小模型试起**

你正在学习LoRA微调，目标明确，喜欢动手。那就直接开干——**先用最小配置跑通流程**。

**方案1：小模型实验**

```python
r=8
alpha=16
```

这个组合**参数少、训练快**，特别适合你用来测试整个微调流程是否顺畅。别小看它，跑通一次比看十遍理论都管用。

**PEFT代码长这样：**

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**三个参数各管一件事：**
- `r`：控制**LoRA容量**（模型能学多少新东西）
- `lora_alpha`：控制**LoRA更新的缩放

## 块 28

## 误区1：rank越大一定越好？

**错误。**

任务简单时，大rank反而可能：
- 浪费显存；
- 增加训练成本。

所以，rank不是越大越好，而是**够用就好**。

---

## 误区2：只调rank，不调alpha

举例：

```python
# 原配置
r=8
alpha=16

# 改成
r=64
alpha=16
```

此时比例变为：

```text
16/64 = 0.25
```

这意味着LoRA对模型的影响被**削弱**了。  
所以：**改了rank，alpha也要跟着调整**，否则效果会打折扣。

---

## 误区3：alpha越大越好？

**错误。**

alpha过大（即 `α/r` 缩放过大）可能导致：
- 更新过强；
- 训练不稳定。

> 注意：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡/发散还常来自学习率过高、数据质量差、batch size过小、精度（fp16/bf16）与数值稳定性、缺少梯度裁剪等。排查不稳定时应综合这些因素，不要只盯着alpha。

---

## 本块学习检查

**动手验证**：  
找一组你之前用过的LoRA配置，检查它的 `r` 和 `alpha` 比例。  
如果 `alpha/r` 小于0.5或大于4，写下你的调整方案，并说明理由（基于本块三个误区）。

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。在 LoRA 训练流程中：

```
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

`LoraConfig` 是连接基础模型与 LoRA 训练的核心配置对象，它负责告诉 PEFT 四件事：

1. **LoRA放在哪里**——通过 `target_modules` 指定要插入 LoRA 的模块；
2. **LoRA规模多大**——通过 `r` 控制低秩分解的秩；
3. **LoRA训练强度如何**——通过 `lora_alpha` 调节更新权重；
4. **是否使用dropout**——通过 `lora_dropout` 设置随机丢弃比例。

简单说，**LoraConfig 就是 LoRA 微调的“总开关”**，你通过它告诉 PEFT 库“在哪些层、用多大容量、以多强力度、加多少正则”来训练。

---

**本块学习检查**（动手验证，约3分钟）：

打开你的 Python 环境，执行以下代码，观察输出：

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

**任务**：观察打印出的配置信息，对照上面四个问题，指出你刚刚设置的每个参数分别

## 块 30

## 一个简单示例：动手搭出你的第一个LoRA配置

先别急着背参数，我们直接看一段代码——这是LoRA配置的“最小骨架”，你以后每次微调都会用到它：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,                # LoRA矩阵的大小
    lora_alpha=16,      # 缩放强度
    target_modules=[    # 插入到哪些层
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05   # 随机失活比例
)
```

**这4个参数就是LoRA的核心，先记住它们各自管什么：**

| 参数 | 作用 |
|------|------|
| `target_modules` | 决定在模型的哪些层插入LoRA |
| `r` | 决定LoRA矩阵的尺寸（越小越省显存） |
| `lora_alpha` | 控制LoRA更新的缩放强度 |
| `lora_dropout` | 训练时随机丢弃部分神经元，防过拟合 |

---

## 完整流程：从配置到模型

光有配置还不够，你需要把它“装”到模型上。整体流程是：

```
model → LoraConfig → get_peft_model() → LoRA模型
```

实际代码长这样：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入低秩矩阵”**。

Transformer 的 Attention 部分，常见结构长这样：

```text
Attention
├── q_proj
├── k_proj
├── v_proj
└── o_proj
```

LoRA **不会默认修改所有层**，你必须手动指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思就是：只给 **Query 投影层**和 **Value 投影层**加 LoRA。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

所以，调整 Q 和 V，通常就能有效改变模型行为——这是最常见的起点配置。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型通用的“标准配置”。不同模型的模块命名可能不同，**配置前先查看模型实际模块名**（用 `model.named_modules()`），再决定 `target_modules`。

---

### 偷懒方案：`target_modules="all-linear"`

较新版本的 PEFT（一般 ≥ 0.7，以你装的版本文档为准）支持直接写：

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
    target_modules="all-linear",   # 自动匹配（几乎）所有 nn.Linear；一般不含 lm_head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**这个写法的特点**：
- 自动把 LoRA 加到（几乎）所有线性层，不用手工枚举模块名
- 但覆盖模块变多 → **可训练参数和显存占用会增加**
- 如果版本太旧不支持，就退回显式列模块名的写法，或升级 PEFT
- 用 `print_trainable_parameters()` 快速验证注入是否符合预期

**怎么选？** 在“参数量/显存”和“效果”之间权衡，最终以验证集结果为准。

---

### 动手验证（建议实操）

跑一下上面的 `all-linear` 示例代码，观察 `print_trainable_parameters()` 输出的可训练

## 块 32

### 方案1：只调整Q和V（轻量起点）

```python
target_modules=["q_proj", "v_proj"]
```

**特点**：参数少、训练快，是常见的轻量起点。具体效果需结合模型结构验证。

---

### 动手验证（基于上述代码）

1. 运行上方代码，观察`target_modules`列表内容。
2. 对比方案2（加入`k_proj`、`o_proj`）和方案3（再加入`gate_proj`、`up_proj`、`down_proj`）的代码差异。
3. 思考：若追求“最快训练速度”，你会选哪个方案？为什么？

## 块 33

## LLaMA系列：target_modules怎么配

你正在学LoRA微调，目标很明确——用最少的显存和算力把大模型调成你想要的样子。那第一步就是告诉PEFT：**该往模型的哪些层“插”低秩矩阵**。这个参数就是`target_modules`。

### LLaMA系列：最简配置

对于LLaMA系列模型（包括Llama 1/2/3等），最常见的配置是：

```python
target_modules = ["q_proj", "v_proj"]
```

**为什么是这两个？** 注意力机制里，Q（查询）和V（值）投影是最敏感、最值得微调的部分。只改这两个，就能以最小的参数量获得不错的微调效果。这也是社区实践中最稳妥的起点。

> **操作建议**：如果你不确定从哪开始，先照抄这个配置跑通流程。之后可以尝试加上`k_proj`、`o_proj`，对比效果差异——但那是后话，先把基础跑通。

### 对比：GPT-2为什么不一样？

如果你换到GPT-2，会发现`target_modules`变成了：

```python
target_modules = ["c_attn"]
```

原因在于：**GPT-2把Q/K/V三个投影合并到了一个模块里**，这个模块叫`c_attn`。所以一个`c_attn`就覆盖了LLaMA里`q_proj`、`k_proj`、`v_proj`三个模块的功能。

### 关键陷阱：`fan_in_fan_out=True`

这是最容易踩坑的地方。GPT-2的注意力层用的是`Conv1D`（不是普通的`nn.Linear`），它的权重形状是`(in_features, out_features)`——**和`nn.Linear`正好转置**。

如果不告诉PEFT这个差异，LoRA注入的低秩矩阵就会放错位置，模型直接跑偏。解决办法就是显式声明：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

## 块 34

### BERT 的 target_modules 配置：别照搬，先查模块名

你正在做 LoRA 微调，目标很明确。但配置 `target_modules` 时，**千万别直接复制其他模型的配置**（比如 LLaMA 的 `["q_proj", "v_proj"]`），因为不同模型的内部模块命名完全不同。

**动手查一下**：用下面代码打印你加载的 BERT 模型的所有模块名：

```python
for name, _ in model.named_modules():
    print(name)
```

运行后，你会看到类似 `bert.encoder.layer.0.attention.self.query` 这样的名字。**找到以 `query` 和 `value` 结尾的层**，再把它们写进 `target_modules`。

**匹配机制要点**：
- `target_modules` 按**后缀匹配**：写 `"query"` 会命中所有以它结尾的层（各层各头都中）。
- 也可以写完整模块名做精确定位，或传正则表达式。
- **注意**：不同 Transformers/PEFT 版本、不同模型实现，命名可能变化。**务必以你实际 `named_modules()` 的输出为准**。

**本块学习检查**：运行上面的代码，把 BERT

## 块 35

**r是什么？——LoRA的“瘦身”核心**

r就是rank（秩），它控制LoRA低秩矩阵的维度。简单说，LoRA通过公式 `ΔW = (α/r)BA` 来近似权重更新，其中α为缩放系数，r为秩。LoRA冻结原始权重W，仅训练低秩矩阵A和B。

**直观理解（动手算一下）**

假设原权重矩阵是 `4096×4096`，设置 `r=8`：

- A的维度：`8×4096`
- B的维度：`4096×8`

原权重矩阵W被冻结，不参与训练。原本需要更新的参数量为4096×4096，现在仅需训练LoRA的低

## 块 36

### 7.1 alpha作用：控制LoRA的“力度”

LoRA的最终更新公式为 ΔW = (alpha/r) * B * A，其中A和B为低秩矩阵，最终权重更新为 W' = W + ΔW。

这里的 `alpha` 直接控制**LoRA更新幅度**——数值越大，微调时对模型权重的改动越强。

**动手验证（hands-on）**：  
配置 `r=8, lora_alpha=16`，则缩放系数为 `16/8=2`，表示LoRA影响被增强。你可以试着把 `alpha` 改成 `4`（缩放系数变为 `0.5`），观察训练时loss下降速度的变化——alpha越大，初期loss

## 块 37

**dropout在LoRA里做什么？**

训练时，它会**随机关闭**一部分LoRA神经连接。目的很直接：**减少过拟合**——防止模型死记训练数据，失去泛化能力。

**怎么配？**

```python
lora_dropout=0.05
```

这表示**5%的LoRA路径会随机失活**，剩下的95%正常参与计算。

**选多少合适？**（经验起点，非硬性规则，最终看验证集是否过拟合来调）

| 数据规模 | dropout参考值 |
|---------|-------------|
| 数据少 | 约0.05~0.1 |
| 数据多 | 约0~0.05 |

如果训练数据**非常丰富

## 块 38

## 9.1 基础配置：快速上手 LoRA

这个配置是**7B模型测试**的经典起点，参数少、见效快，非常适合你快速跑通 LoRA 微调流程。

### 核心配置代码

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量：秩的大小，决定新增参数量
    lora_alpha=16,          # 更新缩放：控制微调幅度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的Q和V矩阵
    lora_dropout=0.05,      # 防止过拟合
    bias="none",            # 不训练偏置项
    task_type="CAUSAL_LM"   # 任务类型：因果语言模型（如GPT类）
)
```

### 参数速查表

| 参数 | 值 | 作用 |
|------|-----|------|
| r | 8 | LoRA容量（秩） |
| alpha | 16 | 更新缩放 |
| target_modules | q/v | 插入位置 |
| dropout | 0.05 | 防止过拟合 |
| task_type | CAUSAL_LM | 语言模型任务 |

### 关键补充：`modules

## 块 39

## 9.1 基础配置：一次跑通 LoRA 微调的最小闭环

你现在的目标是 **LoRA 微调**，而且喜欢动手实操。这一节我们不绕弯子，直接看一个能跑通的完整案例——用 BERT 做二分类，同时把 LoRA 和新初始化的分类头一起训练、保存、再加载验证。全程跟着代码走一遍，你会对 `target_modules`、`r`、`lora_alpha`、`lora_dropout` 这几个参数产生直观的手感。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

#

## 块 40

```
## 10.1 普通文本生成：LoRA参数推荐

对于普通文本生成任务，PEFT库中`LoraConfig`的推荐配置如下：

```python
r=8
alpha=16
dropout=0.05
```

**参数解读：**
- `r=

## 块 41

## 普通文本生成：LoRA配置速览

你正迈向LoRA微调实战，先拆解这份核心配置：

```python
LoraConfig(
    r=8,              # LoRA容量：越低省显存，越高拟合强
    lora_alpha=16,    # 缩放系数：实际更新强度 = alpha / r = 2
    target_modules=["q_proj", "v_proj"],  # 只改注意力层的Q、V投影
    lora_dropout=0.05 # 防过拟合，5%随机丢弃
)
```

**四参数速记**：
- `target_modules`：决定改哪里（本例仅Q/V）
- `r`：决定LoRA容量（

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思路：冻结原模型，只训练少量低秩矩阵。但这里有个关键点——**LoRA只是减少了“训练”的参数，并没有减少“加载”的负担**。

看这张图：

```text
原模型参数：冻结（但必须完整加载到显存）
LoRA参数：训练（新增的少量参数）
```

问题就出在“冻结”这两个字上。即使模型参数不参与训练，它依然要占着显存。

我们来算一笔账。以7B模型为例，如果用FP16精度加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是“把模型放进去”的起步价。实际训练时，你还需要额外空间存放：

- **激活值**：前向传播过程中的中间结果
- **梯度**：反向传播时计算出来的更新方向
- **优化器状态**：如Adam的动量项和方差项

所以，**实际显存需求远高于14GB**。这就是为什么很多人用LoRA微调7B模型时，显卡还是爆显存——LoRA帮你省了“训练参数”的显存，但“加载完整模型”这块开销，一分没少。

**一句话总结**：LoRA解决的是“训练多少参数”的问题，而QLoRA要解决的是“加载

## 块 43

**QLoRA核心思想：量化基座 + LoRA训练**

你的目标是LoRA微调，但直接微调大模型显存不够。QLoRA的思路很直接：**先把基础模型压缩到4-bit，再在其上训练LoRA**。

流程对比：
- **LoRA**：FP16原模型 + LoRA
- **QLoRA**：4-bit量化模型 + LoRA

训练时，**4-bit权重被冻结**，只有**LoRA参数以FP16/BF16精度训练**。注意：量化的是**基础模型权重**，不是LoRA参数。

**动手验证**：在你的环境中，加载一个4-bit模型后，检查模型参数的`dtype`——你会看到基础层是4-bit类型

## 块 44

**3.1 模型存储方式：从14GB到3.5GB**

模型参数按精度不同，占用的存储空间差异巨大。FP32每个参数占4字节，FP16/BF16占2字节，INT8占1字节，而INT4仅占0.5字节。

以7B模型为例，若用FP16存储，需要 `7B × 2 ≈ 14GB`；若用4-bit量化，只需 `7B × 0.5 ≈ 3.5GB`，显存直接减少约75%。

**注意**：这3.5GB只是**理论下界**，仅统计量化后的权重本身。实际显存会更高，因为还要算上量化常数/缩放因子、NF4分组元数据、未量化的模块（如norm、embedding、lm_head、LoRA参数），以及训练时的计算缓存和激活值。真实占用以实际运行为准。

---

**3.2 为什么不用普通INT4？**

普通INT4整数范围是 `-8 ~ 7`。但神经网络权重通常接近正态分布、小数多、中心接近0，直接整数映射会严重损失精度。因此QLoRA提出了**NF4**（NormalFloat4）来解决这个问题。

---

**本块学习

## 块 45

**NF4是什么？——为你的LoRA微调准备的“省内存利器”**

你要做LoRA微调，显存是硬约束。NF4（NormalFloat4，正态分布4位浮点）就是QLoRA论文为神经网络权重“量身定制”的4-bit量化方案。

**关键点**：别被名字骗了，它不是IEEE 754那种通用4位浮点格式，而是一套**离散码本**。它预先定好16个（2⁴）取值点，这些点按标准正态分布的分位数排布——权重集中在0附近，所以0附近划分更细，精度更高。

**工作流程**：量化时，把每组权重归一化，映射到最接近的码本值，只存4-bit索引；反量化时，按码本和缩放因子还原

## 块 46

## 4.2 为什么NF4适合模型权重？

大模型训练完的权重有个特点：**近似正态分布**——大量参数挤在0附近，越远离0越稀疏。

普通INT4量化通常基于数值范围进行线性映射（如对称量化），在数值分布不均匀时，中心密集区域的分辨率可能不足。

NF4则**根据正态分布来设计量化范围**：0附近分配更多档位，精度更高；远离0的区域档位稀疏。这样更贴合真实权重分布，整体精度损失更低。

## 4.3 NF4量化过程

原始FP16权重（如0.023、-0.145、0.876）经过三步：

1. **归一化** — 把数值范围压缩到标准

## 块 47

**5.1 为什么需要双重量化？**

4-bit量化时，除了压缩后的权重，你还得额外保存一组**量化参数**——每组权重都对应一个`scale`（缩放因子）和`zero point`（零点）。这些"附带的"参数同样占用显存，模型越大，这笔开销越不可忽视。

**5.2 双重量化思想**

普通量化流程是：FP16权重 → 4-bit权重 + scale参数。

双重量化在此基础上**再进一步**：把scale参数本身也量化一次。即第一次量化模型权重，第二次量化量化参数。最终结构为：FP16权重 → NF4量化 → 保存4-bit权重；同时scale参数 → 再次量化 → 减少scale存储。

**5

## 块 48

## 6.1 为什么需要分页优化器？

训练时显存峰值不只来自模型本身，还有**梯度**和**优化器状态**。以Adam为例，它需要保存**一阶动量**和**二阶动量**，这两项会额外占用大量显存。

QLoRA虽然通过4-bit量化大幅降低了模型本身的显存占用，但训练过程中仍可能遇到**显存突然不足（OOM）**的情况——问题往往就出在优化器状态上。

## 6.2 分页优化器思想

Paged Optimizer借鉴了操作系统**虚拟内存**的思路，核心机制是：

> 将优化器状态分页存储，当显存出现峰值、即将OOM时，自动把优化器状态**按页换出**到CPU内存，需要参与计算时再**换回GPU**。

流程如下：

```
GPU显存
  ↓
出现峰值（接近/超出显存上限）
  ↓
优化器

## 块 49

### 6.3 类比理解：为什么需要分页优化器？

想象你的GPU显存是一张**桌面**，训练时所有数据都堆在上面。普通优化器就像把所有东西都放桌面——一旦放满，训练直接报错（OOM）。

分页优化器则像**文件柜**：桌面只放常用文件（当前计算需要的参数和优化器状态），暂不用的优化器状态存进柜子（CPU内存），需要时再取回。这样桌面永远有空间，训练不会因显存爆掉而中断。

---

### 6.4 实际启用：一行代码搞定

分页优化器不用你手动管理“换页”，`bitsandbytes` 库已经实现好了。你只需在 `TrainingArguments` 里设置 `optim` 参数：

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

**关键点：**
- `optim="paged_adamw_8bit"`：优化器状态用8-bit存储，更省显存。
- `optim="paged_adamw_32bit"`：精度更高，但占用更多显存。
- **注意**：使用 `paged_adamw_8bit` 需要安装 `bitsandbytes` 库，并确保 `transformers` 版本支持该优化器名称（如 `transformers>=4.30`）。不同版本支持的 `optim` 取值可能不同（如 `adamw_bnb_8bit`），若报错“未知optim”，查一下你装的版本文档，换用支持的名称即可。
- 分页优化器主要解决**峰值显存溢出**，建议搭配 `gradient_checkpointing=True`、小 `batch

## 块 50

### 7.1 4-bit量化：把模型“瘦身”

**核心作用**：把模型权重从FP16（16位）压缩到NF4（4位），直接减小模型体积。  
**直观理解**：就像把高清照片转成压缩格式，占空间小了，但画面依然可辨认。

---

### 7.2 双重量化：连“尺子”也压缩

**核心作用**：量化时需要记录缩放因子（scale），这个参数本身也占空间。双重量化就是把这个缩放因子**再量化一次**，进一步减少额外开销。  
**一句话**：省完权重，再省量化的“附件”。

---

### 7.3 分页优化器：显存不够，内存来凑

**核心作用**：训练时优化器状态很吃显存。分页优化器让GPU显存和CPU内存**自动交换数据**，峰值显存压力大大降低。  
**适用场景**：显存紧张时，用速度换空间。

---

## 八、QLoRA vs LoRA：关键差异

| 项目 | LoRA | QLoRA |
|------|------|-------|
| 基础模型 | FP16/BF16 | 4-bit（本教程用NF4） |
| 训练参数 | LoRA | LoRA |
| 量化 | 无 | 4-bit（NF4常见） |
| 显存占用 | 较低 | 通常更低 |
| 依赖库 | PEFT | PEFT + bitsand

## 块 51

### 4-bit量化加载实战：从配置到代码

你正在为LoRA微调做准备，核心目标是**用更少的显存跑更大的模型**。这一块我们聚焦`load_in_4bit`相关的四个关键参数，直接看它们怎么配、为什么这么配。

#### 1. 开启4-bit加载：`load_in_4bit=True`
这是总开关。设为`True`后，模型权重会以4-bit精度加载进显存，这是QLoRA省显存的第一步。

#### 2. 量化类型：`bnb_4bit_quant_type="

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个**能跑的 QLoRA 训练骨架**——这是你从“看懂片段”到“跑通流程”的关键一步。

### 整体流程：6 步串起来

整个骨架按顺序做 6 件事，每一件都有明确目的：

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

- `load_in_4bit=True`：启用 4-bit 加载
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化类型
- `bnb_4bit_use_double_quant=True`：开启双重量化
- `bnb_4bit_compute_dtype`：按 GPU 能力选 `bfloat16` 或 `float16`，**先查 GPU 是否支持 bf16，再决定用哪个**

**第 2 步：以 4-bit 加载基础模型**

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

注意 `device_map="auto"` 让模型自动分配到可用设备。如果 tokenizer 没有 pad token，就用 eos token 顶上——这是 Causal LM 的常见处理。

**第 3 步：为 k-bit 训练做准备（关键一步）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步处理 norm/输出层的精度问题，并开启梯度检查点所需的输入梯度。**`use_gradient_checkpointing=True`**：与第 6 步的 `gradient_checkpointing=True` 对应起来，两处都要开才能生效。

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

只训练低秩增量参数，原始模型权重冻结。`target_modules` 按实际模型模块名调整，也可以用 `"all-linear"`。`print_trainable_parameters()` 会打印可训练参数量，帮你确认 LoRA 注入成功。

**第 5 步：准备极简数据**

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

Causal LM 的 `labels` 直接复制 `input_ids`。真实场景替换为足够规模的数据。

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
    gradient_checkpointing=True

## 块 53

## 九、把 QLoRA 配置串起来：一个可跑通的最小骨架

你已经有 LoRA 的基础，现在我们把 QLoRA 的各个零件组装起来。记住一个核心思想：**这五步缺一不可**，每一步都在解决一个特定的问题。

### 五步骨架（按顺序执行）

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么被加载**

这一步是"地基"。它告诉系统：基础模型要用 **4-bit 方式加载**，具体用 **NF4 量化** + **双重量化**，并指定计算时用的 dtype（如 BF16/FP16）。没有它，模型就是普通精度加载，省显存无从谈起。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的必要一步**

这是最容易跳过、也最容易出问题的一步。它的作用是**对量化模型做适配**：把部分层转成计算精度、开启输入梯度以支持梯度检查点等。**跳过它，常见后果是训练不收敛或直接报错**。把它理解为"让量化模型准备好被训练"的开关。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

这一步把**可训练的 LoRA 适配器**注入到**冻结的 4-bit 主干模型**上。主干是冻结的（不更新），只有 LoRA 的低秩增量矩阵 A 和 B 在训练，原始权重 W 保持冻结。这就是 QLoRA 省显存的核心逻辑：模型主体不动，只训练一小部分增量参数。

**第 4 步：`gradient_checkpointing=True` —— 与第 2 步配合开启**

注意：建议同时开启 TrainingArguments 里的 `gradient_checkpointing=True` 和第 2 步的 `use_gradient_checkpointing=True` 以确保梯度检查点生效，但二者并非强制对应，可独立设置。

**第 5 步：`optim="paged_adamw_8bit"` —— 分页优化器**

这一步启用分页优化器，作用是**缓解峰值显存**。分页优化器将优化器状态存储在 CPU 内存中，通过统一内存管理按需分页到 GPU，以缓解显存压力。

### 重要提醒

上面是**能说明关系的最小骨架**。真实训练时，你需要：
- 替换为足够规模的数据集
- 设置合理的超参数
- 按你所用版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数、`processing_class`/`tokenizer` 的写法

---

## 十、你的电脑能跑吗？

**如果只学原理：不需要 GPU。**

**如果要实际运行 QLoRA：**

| 硬件 | 最低要求 |
|------|----------|
| CPU | 4核+ |
| 内存 | 16GB+ |
| GPU | NVIDIA CUDA GPU |

**模型许可证**：部署文档引用的模型需标注许可证类型，例如：# 模型许可证：Llama 2 Community License（商用月活>

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多同学训练完LoRA后，习惯性地去找一个“完整的新模型文件”——但LoRA的工作方式完全不同。

**训练过程中发生了什么？**

- **基础模型**（比如Llama-7B）：全程**冻结**，权重一动不动；
- **LoRA Adapter**：这才是真正被**训练**的部分，它是一个轻量级的参数模块。

训练结束后，你得到的结构是：

```text
基础模型（冻结） + LoRA Adapter（已训练）
```

**保存下来的是什么？**

通常只有两个文件：

```text
adapter_model.safetensors   ← LoRA的权重
adapter_config.json         ← LoRA的配置信息
```

注意：**没有**完整模型文件。你保存的只是那个“补丁”，不是“整件衣服”。

---

### 1.2 Adapter有什么特点？

**优点：**

- 文件小——只有几MB到几十MB，而基础模型动辄几个GB；
- 保存快——训练完秒存；
- 灵活——同一个基础模型可以搭配多个不同的Adapter。

举个例子，同一个基础模型可以同时搭配：

```text
客服Adapter     → 用于客服场景
医疗Adapter     → 用于医疗场景
法律Adapter     → 用于法律场景
```

切换任务时，只需换Adapter，基础模型不用动。

**缺点：**

部署推理时，**基础模型和Adapter两个部分必须同时存在**，缺一不可。这给部署带来了额外负担——你没法只带一个Adapter文件就去跑推理。

---

### 动手验证一下

训练完LoRA后，去你的输出目录看看，确认是否只有`adapter_model.safetensors`和`adapter_config.json`这两个文件，

## 块 55

### 合并LoRA权重：从“双组件”到“单模型”

你在推理时，如果**不合并**LoRA权重，流程是这样的：

```text
输入 → 基础模型 + LoRA适配器 → 输出
```

这意味着，每次使用模型，你都得**同时加载**基础模型和适配器，两个都得管。

但对于部署（比如上线一个服务），你更希望模型是**一个整体**：

```text
输入 → 完整模型 → 输出
```

所以，**需要合并权重**。

---

### 合并的数学原理

LoRA训练时，模型权重变成了：

```text
W' = W + ΔW
```

其中，`ΔW = BA`（就是LoRA学到的“增量”）。

#

## 块 56

## 3.1 方法作用：`merge_and_unload()` 是什么？

你训练完 LoRA 后，模型其实还是“两件套”：一个基础模型（Base Model）加上一个 LoRA 适配器（Adapter）。但最终部署时，你通常只想要一个完整的模型文件，而不是带着两个部件跑。

`merge_and_unload()` 就是干这件事的——它把 LoRA 的权重**合并**进基础模型，然后**移除** PEFT 的包装层。

用公式理解：

```text
Base Model + LoRA Adapter → 完整模型
```

---

## 3.2 合并前后的变化

**合并前**，你的模型结构是：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后**，它变成了：

```text
普通Transformer模型（包含LoRA更新后的权重）
```

也就是说，合并后你拿到的就是一个“普通”模型，不再有 LoRA 的痕迹，但它的权重已经包含了 LoRA 微调带来的变化。

---

## 四、LoRA 模型合并完整流程

整体流程如下：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

---

## 五、加载 LoRA Adapter

假设你有基础模型路径 `base_model_path` 和 Adapter 路径 `adapter_path`，加载代码如下：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构为：

```text
Base Model + LoRA Adapter
```

---

## 六、使用 merge_and_unload 合并权重

执行一行代码：

```python
merged_model = model.merge_and_unload()
```

这一步完成的操作是：

```text
W' = W + (α/r)BA
```

其中 `W` 是基础模型权重，`BA` 是 LoRA 的增量，`α` 是缩放系数，`r` 是秩，`W'` 是合并后的新权重。

此时 `merged_model` 已经**不再是** `PeftModel`，而是**普通 Transformer 模型**。

---

## 七、导出合并后的模型

合并完成后，保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后的目录结构示意：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：上面是示意结构，实际文件会随模型/分词器类型变化。例如可能有 `generation_config.json`、`special_tokens_map.json`，大模型可能还有 `model.safetensors.index.json` 和分片权重；有些模型的 tokenizer 是 `vocab.json`/`merges.txt` 或 `tokenizer.model`。

---

**部署说明：**

- **模型许可证**：部署文档引用模型需标注许可证类型。例如：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
- **显存要求**：模型加载需标注显存要求。例如：# 注意

## 块 57

## 三、merge_and_unload 是什么？（重点）

**一句话理解**：`merge_and_unload` 就是把“基础模型 + LoRA适配器”合并成一个完整模型，然后“卸载”掉 LoRA 适配器。合并后，你手里就是一个**普通 Transformers 模型**，不再需要 PEFT 库

## 块 58

### 10.1 推荐合并场景

你正在做LoRA微调，最终目标是把模型用起来。什么时候该把LoRA权重合并回基础模型？记住两个核心场景：

**场景1：上线部署——只要一个固定Adapter，追求简单**

想象生产环境的标准链路：服务器 → 直接加载模型 → 提供API。如果你只需要一个固定的Adapter，且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载），合并通常更方便。

但注意：如果你的推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或者你需要在同一基础模型上**热切换多个Adapter**，那么不合并、以Adapter形式部署反而更灵活。此时是否合并要看框架能力与业务需求。

**场景2：分享模型——希望对方开箱即用**

如果你希望对方拿到就能用、无需理解Adapter机制，可以发布合并后的完整模型（merged_model），用户无需额外下载Adapter。

权衡一下：合并后的完整模型体积**明显更大**（等于一份完整基础模型），分发/存储成本更高

## 块 59

### 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个LoRA适配器（比如Adapter A、B、C），分别对应不同任务。这时**保留Adapter更灵活**——想用哪个任务就加载哪个，不用反复合并、切换。

**场景2：持续训练**

如果你打算继续调整LoRA参数，也建议保留Adapter。合并后参数固化，后续微调就麻烦了。

---

### 十一、QLoRA模型合并注意事项（重点）

QLoRA有个特殊点：基础模型以**4-bit量化**（NF4）存储，而LoRA参数是**FP16/BF16**高精度。把LoRA更新量直接加回4-bit权重，

## 块 60

## 直接合并的坑，以及绕开它的稳妥路径

你在做 LoRA 微调，最终目标是把训练好的 Adapter 合并回基础模型，得到一个可以直接部署的完整模型。这块内容解决的是 QLoRA 场景下的一个关键问题：**在 4-bit 量化状态下直接合并，可能出问题**。

### 直接合并会遇到什么

如果你在 4-bit 状态下直接调用 `merge_and_unload()`，可能遇到两类情况：

- **精度损失**：4-bit 权重需要先反量化再合并，反量化过程会引入舍入误差，与全程用高精度计算并不完全等价，合并结果会有精度上的折损。实际影响取决于你的任务对精度的敏感度，通常在小规模评测中差异不明显，但在某些任务上可能影响生成质量。
- **类型或版本问题**：可能出现权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

注意，这不是“必然报错”，而是“可能出现”。能否顺利合并、合并质量如何，取决于你的版本、量化配置，以及是否先做了反量化。最终以你所用版本文档和实际运行结果为准。

### QLoRA 与普通 LoRA 合并的区别

QLoRA 在训练时基础模型是 4-bit 量化加载的，而普通 LoRA 训练时基础模型通常是 FP16/BF16。因此：

- **普通 LoRA**：基础模型本身就是高精度，直接加载后合并即可。
- **QLoRA**：训练时基础模型是 4-bit 量化状态，合并前必须先反量化，即用高精度重新加载基础模型，再挂载 Adapter 合并。

### 稳妥做法：先反

## 块 61

**QLoRA合并：先别急着merge，看这三点**

你目标是LoRA微调，且喜欢动手实操，那这块直接关系到你合并模型时会不会“翻车”。核心就一句话：**QLoRA（4-bit量化）模型能不能直接合并，取决于PEFT版本**。

**1. 版本决定行为**
不同PEFT版本对“量化模型能否直接合并”支持不同，有的要求或推荐先反量化。**动手前先查你所用版本的`merge_and_unload`文档**。上例关键操作是：**加载基础模型时不传`quantization_config`或`load_in_4bit`**，这样拿到的是高精度权重，再合并。

**2. 内存代价**
以高精度重新加载完整基础模型，会

## 块 62

**合并时开启校验并做一致性验证（建议）**

`merge_and_unload` 支持 `safe_merge=True`，合并时会做数值检查（如检测 NaN），作为保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并后，建议对比合并前后模型在同一批输入上的输出，确认无明显偏差：

- 准备一小批固定输入（prompt）
- 分别用「合并前的 PeftModel」和「合并后的模型」推理
- 对比 logits / 生成文本是否基本一致

**最小验证代码**：用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本是否一致。

**动手验证**：跑通上述代码，记录 logits 最大绝对差；

## 块 63

### 合并后别急着用：先做一次“前后对照”校验

你刚在 §11.2 里用 `merge_and_unload()` 把 LoRA 适配器合并进了基础模型，得到了 `merged_model`。现在先别急着部署，花两分钟做一次一致性验证——这能帮你确认合并过程没有引入意外偏差，尤其是 QLoRA 这种低精度场景下，合并结果和原始模型在行为上可能只有“近似”而非“完全一致”。

#### 为什么必须单独加载一份“合并前”的模型？

关键点：`merge_and_unload()` 是**原地操作**。它会在你调用它的那个对象上直接修改，并返回合并后的底层模型。也就是说，§11.2 里那个 `PeftModel` 对象一旦被合并，就不再是“未合并”状态了。

**所以你不能拿它当“合并前”的对比基准。** 你需要重新从磁盘加载一份全新的、未合并的 `PeftModel`，专门用来做对照。

```python
# 重新加载一份“未合并”的模型作为对比基准
import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度) + Adapter
# 注意：Llama-2-7b FP16 需要至少14GB显存。
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
```

这里 `base_model_path`、`adapter_path`、`compute_dtype` 都沿用你在 §11.2 里定义好的变量。

#### 校验步骤 1：比较 logits（模型输出分数）

用同一个固定输入（比如 `prompt = "请用一句话介绍你自己。"`，你可以换成自己任务里有代表性的输入），分别让“合并前”和“合并后”的模型跑一次前向，比较输出的 logits 差异。

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

这个差值越小，说明合并前后的模型在“理解”这个输入上越一致。QLoRA 因为量化精度损失，通常不会精确到 0，但应该是一个很小的近似值。

#### 校验步骤 2：比较生成文本

数值差异是一回事，更直观的是看模型实际“说”了什么。用贪心生成（`do_sample=False` 去掉随机性），让两个模型各自生成一段文本，然后对比。

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_b =

## 块 64

## 合并时开启校验并做一致性验证（建议）

**为什么必须单独再加载一份模型？**

`merge_and_unload()` 返回合并后的新模型，原模型保持不变；如需原地修改，需设置相应参数。合并后，原来的 `model` 已经变成了"合并版"，不能再拿它当"合并前"的对照。如果硬拿它对比，两边其实是同一份已合并权重，`max_abs_diff` 会恒为 0，验证就完全失效了。

正确做法是：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照，这样才能真正比较合并前后的差异。

**显存不够怎么办？**

如果两个模型放不下，就**分两次跑**：
1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本存下来，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

**结果预期（重点）**

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 差异。**重点是确认没有明显退化或 NaN**，而不是要求逐位相同。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。
- 是否可接受，以你的任务指标与验证集为准。

---

### 关于电脑配置的实操建议

本章节需要**实际加载模型**，所以硬件有基本要求：

**最低配置**：CPU 4核+，内存 16GB+，磁盘预留模型大小 2 倍空间。

**GPU 显存（经验参考，非硬性门槛）**：7B 模型约 16GB 显存起（宽松些约 24

## 块 65

## 合并后翻车？这份排查清单请收好

你正在做 LoRA 微调，目标是把训练好的 Adapter 合并回基础模型。这一步最容易出幺蛾子，下面直接给你一份“问题→原因→解法”的速查表，全是实战中踩过的坑。

### 常见问题与排查方向

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 `PeftModel`（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已合并成普通模型） | 确认加载方式，先得到 `PeftModel` 再调用 |
| 合并后模型大小变化 | 正常现象：合并后是完整模型，不再是体积小的 Adapter | 无需处理；若在意体积可保留 Adapter 不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是 dtype 用了 FP32、未使用 CPU 合并或低精度 dtype | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存 tokenizer；或特殊 token / generation_config 未一并导出 | 重新 `save_pretrained` 并核查 tokenizer、generation_config、config |
| QLoRA 合并失败或结果异常 | 在 4-bit 状态直接合并、PEFT/bitsandbytes 版本不支持、未重新加载高精度模型、内存/显存不足 | 先反量化到 FP16/BF16 再合并；核对版本；必要时用 CPU 合并；开 `safe_merge=True` 并做一致性验证 |

### 重点：什么是“先反量化再合并”

这里说的“反量化”**不是**手动逐层还原 4-bit 权重，而是——**重新加载一份不带量化配置的基础模型**。

具体做法：用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`（**不传** `quantization_config`/`load_in_4bit`）得到高精度权重，再 `PeftModel.from_pretrained` 挂载**同一份 Adapter** 后合并。这样合并发生在高精度权重上，天然避开“在 4-bit 状态下合并”的类型/精度问题。

**版本提示**：不同 PEFT / bitsandbytes / Transformers 版本对“量化模型能否直接调用 `merge_and_unload`”支持不同——部分版本直接报错或警告，部分较新版本有一定支持但仍可能有精度损失。因此**“先反量化（重载高精度）再合并”是跨版本最稳妥的做法

## 块 66

## 路线B：保留 Adapter，用支持 LoRA 的框架动态服务

与路线A不同，路线B不合并权重，而是让推理框架在**同一份基础模型**上运行时挂载多个 LoRA Adapter（即 Multi-LoRA serving）。vLLM 是支持这一能力的典型框架。

**核心流程**：

```
基础模型（只加载一份）
    +
运行时挂载多个 LoRA Adapter（A / B / C）
    ↓
按请求指定使用哪个 Adapter
```

**路线B的优势**：

- **多任务/多租户场景下节省资源**：只需一份基础模型 + 多个小体积 Adapter，而不是每任务一份完整模型。显存和存储成本都大幅降低。
- **支持热切换**：运行时可以动态切换 Adapter，无需为每个任务单独保存完整模型。

**路线B的代价**：

- 依赖推理框架对 LoRA 的原生支持，需要额外配置，链路比路线A略复杂。
- 不同框架/版本对 LoRA 的支持范围和限制不同，需要查阅对应文档。

---

### vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

以 vLLM 为例，启动 OpenAI 兼容服务时通常需要**显式开启 LoRA 支持**并**

## 块 67

## 14.3 如何选择：合并还是保留？

学到这里，你可能会问：既然合并能导出完整模型，那是不是每次都该合并？**不一定**。合并不是唯一部署方式，关键看你的实际场景。

**三个判断标准（经验参考，非硬性规则）：**

1. **只训练了一个 Adapter，追求最简部署** → 选**路线A（合并）**。一个模型文件搞定，部署链路最简单。

2. **同一基础模型要服务多个 Adapter，且推理框架原生支持 LoRA** → 选**路线B（保留 Adapter 动态服务）**。这样不用为每个 Adapter 存一份完整模型，省存储、切换灵活。

3. **QLoRA 精度敏感** → 无论走哪条路线，都建议按 §11.3 做**合并前后/上线前的一致性验证**，确保精度没被合并过程损伤。

> 注意：具体框架用法和限制，留到第 15 板块（vLLM 部署）展开。这里只需建立认知：**合并不是唯一出路**。

---

## 十五、总结

## 测评题目

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）
