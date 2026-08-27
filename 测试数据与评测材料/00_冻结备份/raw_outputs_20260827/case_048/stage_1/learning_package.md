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

以7B参数模型为例，就是**70亿个参数**。训练时，你需要同时保存：

- **参数**本身
- **梯度**（用于更新参数）
- **优化器状态**（控制更新步长）

这三样东西加在一起，显存需求非常高。普通消费级GPU根本扛不住。

---

**动手感受一下**：打开你的GPU监控（如 `nvidia-smi`），看看当前显存占用。想象一下，如果要把70亿参数的模型、梯度、优化器状态全部塞进去，你的显存够用吗？

这正是参数高效微调要解决的问题——**只更新少量参数，而不是全部**。接下来的LoRA和QLoRA，就是为此而生的工具。

## 块 2

**PEFT：参数高效微调**

核心思想：**不修改整个大模型，只训练少量新增参数。**

流程：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter

| 方式 | 训练参数 | 显存需求 |
|------|---------|---------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**动手理解**：对比表格中三种方式的“训练参数”和“显存需求”，LoRA只训练少量参数，显存需求较低——这就是你实现LoRA微调的核心优势。

## 块 3

## LoRA 是什么？——用“小补丁”微调大模型

想象一下：你不需要重写整本教科书，只需在关键页面上贴几张便利贴，就能让内容“适配”新场景。LoRA（低秩适配）就是这个思路。

**核心逻辑**：原模型权重 `W` 保持**冻结不动**，我们额外增加一个更新量 `ΔW`，最终效果相当于 `W + ΔW`。

**关键设计**：`ΔW` 被拆成两个小矩阵的乘积，并带有缩放因子：`ΔW = (α/r) × B × A`。如果原权重 `W` 形状是 `d_out × d_in`，那么 `A` 的维度为 `

## 块 4

## 3.1 LoRA 的问题：模型还是太大

你已经知道 LoRA 能大幅减少**需要训练的参数**，但有个关键点别忽略：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个 7B（70亿）参数的模型，如果用 FP16（半精度）格式加载，大约需要 **14GB 显存**。也就是说，哪怕 LoRA 只训练一小部分参数，这 14GB 的“底子”是省不掉的。

> **动手感受一下**：如果你手头有 GPU，可以用 `nvidia-smi` 看看当前显存占用。想象一下，14GB 只是“把模型放进去”的起步价，还没开始训练呢。

---

## 3.2 QLoRA 的解法：先把模型“压缩”再训练

QLoRA 全称是 **Quantized LoRA**，也就是“量化版 LoRA”。它的核心思路很简单：

**先把 FP16 模型做 4bit 量化 → 显存占用大幅下降 → 再加载 LoRA 参数进行训练**

结构上就是：

```text
4bit基础模型 + LoRA参数 → 一起训练
```

这样，你既享受了 LoRA 只训练少量参数的好处，又通过量化把“基础模型”这块大头给压缩了。

---

## 3.3 QLoRA 依赖 bitsandbytes 库

要跑 QLoRA，你需要这几样东西：

- `transformers`
- `peft`
- `bitsandbytes`
- 一个**受支持的量化后端**（本教程以 **NVIDIA CUDA GPU** 为准）

其中 `bitsandbytes` 负责三件事：**低精度量化、4bit 加载、8bit 加载**。

> ⚠️ **重要提醒**：本教程的示例命令和验证流程，**统一按 NVIDIA CUDA GPU 来写**。但这不代表 QLoRA 只能在 NVIDIA

## 块 5

## 电脑配置批注：你的LoRA微调“装备清单”

既然你的目标是动手跑通LoRA微调，那咱们先把“装备”备齐。别担心，门槛没有想象中高。

**LoRA最低配置（能跑起来）**
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA GPU，显存**最低8GB**，**推荐12GB+**

**QLoRA要求（更省显存的选择）**
QLoRA的核心思路就是“花小钱办大事”——通过4-bit量化技术压缩基础模型，让显存占用比LoRA全精度加载更低。但注意，它依然需要加载量化后的模型、LoRA参数、激活值和batch数据。

- **GPU**：本教程实训范围是**NVIDIA CUDA GPU**（命令按此给出；其他后端请以bitsandbytes官方说明为准）。
- **显存参考**（以7B级别模型为例，**经验值非硬性保证**）：
  - 参考下限：**约8~12GB**（短序列、小batch）
  - 推荐：**16GB+**

> **重要提醒**：别把上面的数字当死门槛。实际显存占用受这些条件影响：
> - **模型规模**：7B是基准，13B/70B会显著增加需求；

## 块 6

好的，我们直接开始。你的目标是跑通 LoRA 微调，而且喜欢动手操作。这个块就是帮你把“地基”打牢，确保后面代码能顺利跑起来。我们快速过一遍，重点看那些会“卡住”你的地方。

**第一步：确认Python版本**

打开你的终端（命令行），输入下面这行命令，然后回车：

```bash
python --version
```

只要输出的版本号是 **Python 3.10 或更高**，就满足要求了。如果不是，建议先升级Python版本再继续。

**第二步：检查PyTorch和GPU（最容易出错的地方）**

这一步非常关键。LoRA/QLoRA 依赖 PyTorch，而且 `bitsandbytes`（后面会装）要求 PyTorch 必须是**和你的显卡驱动（CUDA）匹配的GPU版**。

> **重点坑**：直接 `pip install torch` 通常会装成CPU版，这样QLoRA的4-bit量化功能是用不了的。**千万别这么干**。

正确做法是去 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 找到适合你机器的安装命令。比如，如果你的CUDA是12.1，命令会是：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

装完后，在Python环境里运行下面这段代码来验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
```

如果输出中 `torch.cuda.is_available()` 的结果是 **`True`**，就说明PyTorch能看到你的GPU了。

> **注意**：`True` 只代表PyTorch这边没问题，还不能保证 `bitsandbytes` 的4-bit路径一定能用。那个要等后面章节专门检查。

**第三步：看一眼你的GPU**

在终端运行：

```bash
nvidia-smi
```

这个命令会显示你的GPU型号、CUDA版本和显存大小，比如“RTX 4090, CUDA 12.x, Memory 24GB”。心里有个数就行。

**第四步：安装PEFT库**

PEFT是Hugging Face官方的参数高效微调库，我们用的LoRA就是它支持的。安装很简单：

```bash
pip install peft
```

装完验证一下：

```python
import peft
print(peft.__version__)
```

## 块 7

**bitsandbytes 是干嘛的？**

简单说，它就是让模型“瘦身”的工具，帮你把大模型压缩到 8bit 或 4bit，从而在显存有限的 GPU 上跑起来。这也是 QLoRA 训练的关键依赖。

**为什么必须装？**

本教程的 QLoRA 方案就是基于 bitsandbytes 做 4-bit 量化。所以，**必须安装**。注意：QLoRA 本身不强制绑定这个库，但咱们这条路统一用它 + CUDA。

**动手装：**

```bash
pip install bitsandbytes
```

装完别急着用，下一步还要检查 CUDA 兼容性（见第八节）。

---

**本块学习检查**：在终端执行安装命令，确认无报错后，运行 `pip show bitsandbytes` 查看版本号，记下来

## 块 8

**先检查，再训练——为什么这一步不能跳？**

你的目标是跑通LoRA微调，而bitsandbytes是QLoRA量化依赖的底层库。它通过CUDA调用GPU计算。如果CUDA版本不匹配，后果很直接：**CUDA不可用、量化失败、加载模型报错**。所以，动手训练前，先花1分钟确认环境。

**操作验证（跟着做）**

1. 打开终端，运行：
   ```bash
   nvidia-smi
   ```
   在输出末尾找到`CUDA Version`，记下这个数字。

2. 再运行：
   ```bash
   python -c "import bitsandbytes as bnb; print(bnb.__version__)"
   ```

#

## 块 9

**本块学习检查**：安装完成后，在 WSL2 的 Ubuntu 终端里运行 `python -c "import bitsandbytes"`，若没有报错，说明环境基本就绪。

---

在 Windows 上做 LoRA 微调，最省心的路径是：**先装 WSL2（Ubuntu），再在 Ubuntu 里配置 CUDA 环境和 bitsandbytes**。这样能绕开 Windows 原生环境下的兼容性坑。

如果你坚持用 Windows 原生环境，最常撞见的问题是：**bitsandbytes 无法加载 CUDA 动态库**。这通常不是单一原因，而是三个因素共同作用的结果：

1. **版本不匹配**：你装的 bitsandbytes 版本，和你本机的 CUDA 或 GPU 版 PyTorch 对不上；
2. **缺少运行库**：系统里找不到对应的 CUDA 运行库文件；

## 块 10

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

#### 问题2：CUDA版本不匹配

在Windows上跑LoRA，最常遇到的坑就是CUDA版本对不上。别慌，先跑一条命令看看你的机器到底什么状态：

```bash
nvidia-smi
```

这条命令会告诉你两件关键信息：

- **NVIDIA驱动**：驱动装没装、版本是多少
- **CUDA版本**：驱动支持的CUDA版本号

要查看当前环境里PyTorch用的是哪个CUDA编译的，可以用：

```bash
python -c "import torch; print(torch.__version__)"
```

**为什么这很重要？** 因为PyTorch的CUDA版本必须和驱动支持的CUDA版本匹配，否则后面`import torch`会报错，QLoRA的4-bit量化更是直接跑不起来。

---

## 十、安装完整LoRA环境

假设你已经有了一个Python环境，先装基础依赖：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**⚠️ 前置检查**：确保你已经按5.2节安装了**匹配本机CUDA的GPU版PyTorch**。如果你本机CUDA是12.1，用这个命令装：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后装上LoRA的核心库：

```bash
pip install peft bitsandbytes
```

**一条命令搞定全部**：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

装完依赖后，运行：

```bash
accelerate config
```

它会问你几个问题，按你的实际环境回答。比如单GPU机器：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

配置完成后会生成一个`default_config.yaml`文件，Accelerate会自动读取它。

---

## 十二、LoRA环境验证实验

创建一个测试文件`test_lora_env.py`：

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
print("LoRA config created successfully")
```

运行：

```bash
python test_lora_env.py
```

**看到什么算成功？**

- ✅ PyTorch正常（CUDA返回True）
- ✅ PEFT正常（能导入LoraConfig）
- ✅ LoRA配置创建成功

## 块 11

**本块学习检查**：对照下方“常见问题排查”表，检查你的环境是否已避开这些坑——特别是CUDA版本和PyTorch版本是否匹配。

---

### 一、先看这张“避坑表”

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或用更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

**怎么用**：你现在是Windows环境，最容易踩的是第4行——遇到报错别硬扛，直接切WSL2。其他行是通用检查点，装完环境后逐条对照。

---

### 二、你的完整路线图

**当前环境（已就绪）**：
```
Transformers → Datasets → Accelerate → PEFT → bitsandbytes → LoRA → QLoRA
```

**后续实战（下一步）**：

## 块 12

**本块学习检查**：读完本节后，请暂停并回答：全参数微调一个100亿参数的模型，除了模型权重本身，还需要额外保存哪两类东西？为什么这会带来显存压力？

---

## 一、为什么需要LoRA？

### 1.1 大模型微调的问题

现代大语言模型（如LLaMA、Qwen、Baichuan、Mistral）通常拥有数十亿甚至上百亿参数。以7B模型为例，就是70亿个参数。

如果采用传统方式微调——把预训练模型的所有参数全部更新，然后保存整个新模型——会面临三个巨大挑战：

- **巨大的GPU显存**：模型本身就要占用大量显存；
- **大量训练时间**：更新全部参数意味着海量计算；
- **巨大的存储空间**：每个微调版本都要保存一份完整模型。

### 1.2 全参数微调的问题

假设模型有100亿参数。训练时，你不仅需要保存模型参数 **W**，还需要额外保存：

- **梯度 ∂W**：用于反向传播更新参数；
- **优化器状态**（如Adam的动量项）：用于优化器计算。

这意味着实际显存需求**远大于模型本身的大小**。这就是为什么全参数微调大模型在资源上非常吃力——你真正在“扛”的，不只是模型，

## 块 13

## 2.1 不直接修改原模型参数

**传统微调是怎么做的？**

想象你有一个训练好的大模型，它的原始权重叫 `W`。传统微调的做法是**直接修改**这些权重：

```text
W → W'
```

也就是说，训练过程会改变 `W` 本身，得到一个新的 `W'`。

**LoRA 的做法完全不同：**

1. **保持** `W` 不动——把它**冻结**起来；
2. **新增**一个参数 `ΔW`（读作“delta W”）；
3. 最终模型使用的是：

```text
W' = W + ΔW
```

其中：
- `W`：原模型参数（冻结，不训练）；
- `ΔW`：LoRA 新增的参数（只训练这个）。

**一句话理解：** 传统微调是“改旧参数”，LoRA 是“加新参数”。

---

## 2.2 为什么只训练 ΔW？

大模型已经通过海量数据学到了语言规律、世界知识和推理能力。微调通常只是**对已有能力

## 块 14

## 3.1 矩阵基础概念

Transformer 里大量参数都是矩阵。以注意力层为例，计算查询向量 Q 的数学表达式是：

```text
Q = X · Wq
```

这里 `X` 是输入（形状为 `(序列长度, d_model)`），`Wq` 是权重矩阵（形状为 `(d_model, d_model)`，按同维投影举例）。两者相乘得到 Q，形状也是 `(序列长度, d_model)`。实际实现中还有 batch 维、多头拆分等，这里只为说明“Q 由输入乘以权重矩阵得到”。

假设 `Wq` 是 `4096 × 4096` 的矩阵，它的参数数量就是：

```text
4096 × 4096 ≈ 1678万
```

---

## 3.2 直接训练矩阵的问题

如果微调时要修改权重，直接学习

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设原始矩阵是 **4096 × 4096**。如果直接训练，你需要更新 **4096×4096 = 1677万** 个参数——这非常庞大。

LoRA 的做法是：设置 **rank r=8**，把大矩阵拆成两个小矩阵的乘积：
- **A**：尺寸 **8 × 4096**，参数 **32768**
- **B**：尺寸 **4096 × 8**，参数 **32768**

两者相加，LoRA 总参数仅 **65536**（约6.5万）。

**对比一下：**

| 方式 | 参数数量 |
|------|---------|
|

## 块 16

**5.1 神经网络参数变化具有低维特征**

微调大模型时，我们真正要改的是参数变化量 ΔW。但 LoRA 论文发现：**ΔW 并不需要覆盖所有方向**。

打个比方：假设模型参数处在一个 4096 维的空间里，真正影响新任务表现的“有效方向”其实很少，它们集中在一个**低维子空间**中——远小于原始维度的少数方向。

> 注意：这是 LoRA 的一个**假设**——“权重更新具有较低的内在维度”。具体有效维度是多少没有固定值，会随模型、任务、数据变化。这里只做定性理解，别记成“几十维”这种确定数字。

因为有效变化集中，所以我们就能用一个**低秩矩阵 BA** 去近似 ΔW，

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张高清图片，尺寸是 **4000×4000像素**。如果要把这张图片的每一个像素都完整保存下来，你需要存储 **1600万个数据**——这就像大模型里完整的权重矩阵，参数数量巨大。

但关键洞察是：这张图片的**主要变化**往往只是少数几个方面，比如**亮度**、**对比度**，或者**少量结构变化**。这些变化完全可以用**少量参数**来描述，而不需要重新保存整张1600万像素的图。

LoRA的核心思想正是如此：**用少量参数来描述模型权重的变化方向**，而不是直接修改和存储整个巨大的权重矩阵。你只需要训练一个很小的"变化描述器"，就能实现和全量微调相近的效果。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构是：

```text
Input → Embedding → Attention → FFN → Output
```

其中**Attention**模块包含四个投影矩阵：**Q、K、V、O**。

LoRA在早期实践中**常见**的做法是只把LoRA加到**Q投影**和**V投影**上，代码写法如下：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示：只训练 `Wq` 的变化和 `Wv` 的变化。

> **重要说明**：只挑 `q_proj`/`v_proj` 是 LoRA 原论文里的**常见选择**，**不是固定标准**。现代实践中也经常把 LoRA 加到注意力的全部投影（q/k/v/o）、FFN 的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

下面给出一个**覆盖更完整线性层**的配置示例（以 Llama 类模型常见的模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

```python
from peft import LoraConfig

# 示例：把 LoRA 同时加到注意力的 q/k/v/o 与 FFN 的 gate/up/down 线性层
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

> 上例是**显式

## 块 18

## 8.1 rank（r）——低秩矩阵的“宽度”旋钮

先看代码里的那个参数：

```python
r=8
```

它代表**低秩矩阵的维度**。你可以把它想象成在原始权重矩阵旁边“加装”的一条窄通道——`r` 就是这条通道的宽度。

**`r` 越大：**
- 优点：表达能力更强，能学更复杂的模式；
- 缺点：参数更多，显存占用也更高。

**`r` 越小：**
- 参数少、省显存，但表达能力受限。

### 起点参考（不是铁律）

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

注意：合适的 `r` 和模型规模、目标模块、数据量、任务难度都有关。上表只是**搜索起点**，不是查表定死。

### 比背“配方”更可靠的做法：基于验证集选 `r`

既然你是动手型学习者，直接给你一套可操作的流程：

1. **划出验证集**：从数据里留出 8:2 或 9:1 不参与训练，后续所有比较都在它上面看指标。
2. **固定其它变量**：`target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数都不动，只改 `r`。
3. **小范围网格搜索**：从 `r ∈ {4

## 块 19

## 8.2 lora_alpha：控制LoRA的“更新力度”

`lora_alpha` 是一个数值参数，例如 `lora_alpha=16`。它的作用是**控制LoRA更新强度**——简单说，就是决定LoRA在微调时对模型参数的“改动幅度”有多大。

实际使用中，通常用 `alpha/r` 作为缩放比例。比如 `r=8`、`alpha=16` 时，缩放比例就是 `16/8=2`。这个比例越大，LoRA对模型的更新越“猛”；比例越小，更新越“温和”。

## 8.3 lora_dropout：防止过拟合

`lora_dropout` 也是一个数值参数，例如 `lora_dropout=0.05`。它的作用是**防止LoRA参数过拟合**——在训练时随机“丢弃”一部分LoRA参数，让模型不过度依赖某些特定参数，从而提升泛化能力。

---

## 九、LoRA vs 全参数微调：一张表看懂

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

**注意**：上表是**相对趋势**，不是绝对数值。LoRA的可训练参数量、显存、训练速度和Adapter文件大小会随 `target_modules` 选择、秩 `r`、模型层数与 `dtype` 变化——比如把LoRA加到很多模块并用较大 `r` 时，参数量和Adapter体积都会明显上升。“更快/更省显存/MB级”通常成立，但不宜当作恒定保证。

---

## 十、LoRA vs QLoRA：差在哪？

**LoRA** 的组成：

```text
FP16模型 + LoRA参数
```

**QLo

## 块 20

## 二、Rank（r）是什么？

### 2.1 Rank表示什么？

**一句话理解**：Rank就是“矩阵的秩”，在LoRA里，它决定了低秩矩阵中间那个“通道”的宽度。

**为什么需要它？**

还记得LoRA的核心思路吗？我们不直接更新原始权重矩阵 `W`，而是训练一个增量 `ΔW`，最终：

```
W' = W + ΔW
```

而这个 `ΔW` 是由两个小矩阵相乘得到的：

```
ΔW = (α/r) × B × A
```

其中，`A` 的维度为 `d×r`，`B` 的维度为 `r×d`，`α` 是缩放因子，`r` 是秩。

**Rank就是这两个小矩阵的“中间维度”大小。**

**举个具体例子**（动手算一遍就懂了）：

假设原始权重矩阵是：

```
W: 4096 × 4096
```

这是一个巨大的矩阵（约1600万参数）。如果直接微调它，成本太高。

现在设置：

```python
r=8
```

那么两个低秩矩阵的维度就变成了：

```
A: 4096 × 8
B: 8 × 4096
```

注意看：A的列数等于r（8），B的行数也等于r（8）。它们相乘：

```
ΔW = (α/r) × B × A
```

结果恢复成：

```
409

## 块 21

## 2.2 Rank控制什么？

一句话记住：**Rank决定LoRA能学多少种变化方向**。

**低rank（如 r=4）**：模型只能学较少变化模式。好处是参数少、显存低、训练快；代价是表达能力有限。

**高rank（如 r=64）**：LoRA自由度更高。能学更复杂任务、适应力更强；但参数更多、显存增加、训练成本提高。

**动手验证**：在代码里把 `r` 从4改成64，对比训练时显存占用和速度变化，感受"自由度"与"成本"的取舍。

**本块学习检查**：用自己的话说，为什么低rank训练快但表达能力有限？

## 块 22

**本块学习检查**：算一下，如果不用LoRA，微调一个4096×4096的权重矩阵，要训练多少参数？

---

**动手算一算：**

完整训练时，这个矩阵的参数就是 `4096 × 4096 = 1677万`。这就是“全参数微调”的代价——每个参数都要更新。

**对比一下LoRA：**

LoRA只训练两个小矩阵，参数量 = `r × (输入维度 + 输出维度)`。

- **r=8**：`8 × (4096+4096) = 65536`，约6.55万
- **r=64**：`64 × 8192 = 524288`，约52.4万

## 块 23

### 4.1 Rank太小的问题

想象你让LoRA用很小的“内存条”去记新知识，比如设 `r=2`。结果就是：

- **学习能力不足**：能装下的信息太少，新任务的关键模式根本塞不进去。
- **欠拟合**：模型“学不进去”，训练时loss下降很慢，效果提升也有限。

**怎么发现？** 看训练曲线：如果loss半天降不下来，或者微调后效果和原来差不多，很可能就是rank设太小了。

---

### 4.2 Rank太大的问题

反过来，如果你“用力过猛”，把 `r=256` 设得很大，会带来：

- **参数增加**：要训练的东西变多了。
- **显存增加**：GPU内存占用更高。
- **训练速度下降**：每一步都更慢。

更关键的是：**LoRA的优势被削弱了**。LoRA本来就是为了用少量参数达到接近全量微调的效果，rank太大时，参数高效的优势就变小了。

**用数字看清楚**：假设权重矩阵是 `4096×4096`（完整参数约1677万），不同rank下LoRA的参数量占比：

| rank | LoRA参数量 | 占比 |
|---|---:|---:|
| 8 | 65,536 | ≈0.39% |
| 64 | 524,288 | ≈3.13% |
| 256 | 2,097,152 | ≈12.5% |

注意：即使 `r=256`，LoRA参数也只占完整矩阵的**约1/8**，远没到“接近完整矩阵”的程度。说它“优势下降”，是相对 `r=8` 的0.39%而言，占比大了很多，成本高了，但**不是**因为它逼近了满秩。

---

**本块学习

## 块 24

## 四、4.3 常见rank选择：从哪开始？

你是动手型学习者，目标是把LoRA用起来。先记住：**下表只是经验起点，不是标准答案**。

| 任务类型 | rank起点参考 |
|---------|------------|
| 简单分类 | 约4~8 |
| 文本风格调整 | 约8~16 |
| 指令微调 | 约16~64 |
| 复杂领域任务 | 约64以上 |

**怎么用这张表？** 选任务对应范围的下限起步，跑通后再调。比如指令微调，先设`r=16`。

**实际项目中，还要看这些因素调整：**
- 数据量（数据少，rank别太大）
- 模型大小（大模型可承受更高rank）
- 序列长度、batch大小（含梯度累积后的有效batch）
- 目标模块（`target_modules`）的选择与

## 块 25

### 6.1 控制LoRA影响强度

咱们直接上手感受一下。假设你设了 `r=8`，然后 `alpha=16`，那缩放系数就是 `16/8=2`。意思是：LoRA 对模型的更新会被**放大2倍**。

如果 `alpha=8`，那 `8/8=1`，更新保持原始比例，不加不减。

如果 `alpha=4`，那 `4/8=0.5`，LoRA 的影响就**减弱**了。

**核心公式**：LoRA 加入模型的更新量会被 `α/r` 这个系数缩放。所以 `α/r` 就是控制 LoRA 影响强度的**关键缩放系数**。

---

### 七、Alpha与Rank的关系（重点）

**注意**：别把 `α/r` 当成“唯一决定实际强度的开关”。它只是一个**缩放系数**。实际落到权重上的更新量，还取决于：

- **学习率**：直接决定每步对 A、B 的更新步长。调大 `α/r` 又不下调学习率，容易一起放大更新导致震荡。很多实践里，你甚至可以直接调学习率来控制强度，而不是频繁改 `alpha`。
- **初始化**：常见实现里 A 用小随机值、B 初始化为 0，所以训练开始时 `ΔW = BA = 0`（不扰动原模型）。初始化方式影响早期训练动态。
- **训练状态**：`ΔW` 是训练**逐步累积**出来的。同样的 `

## 块 26

**本块学习检查**：动手验证“缩放比例”的含义——分别用 `r=8, alpha=16` 和 `r=16, alpha=32` 计算 `alpha / r`，两个结果是否相同？写下来。

---

**配置1**：`r=8, alpha=16`，比例 = 2。

**配置2**：`r=16, alpha=32`，比例 = 2。

注意：两个配置的 **rank 不同**（8 vs 16），但**缩放比例相同**（都是 2）。

**核心结论**：调整 rank 时，常见做法是**同步调整 alpha**，让 `alpha / r` 大致稳定。这只是**经验做法**，不是硬性规则

## 块 27

**本块学习检查**：按下面步骤动手跑一次，确认你理解了 r 和 alpha 的搭配逻辑。

**方案1：小模型实验**
- 配置：`r=8`，`alpha=16`
- 特点：参数少、训练快，适合先做测试。这是你入门 LoRA 微调的首选起点。

**动手验证（基于上文配置）**
1. 在代码中设置 `r=8, lora_alpha=16`，跑一个最小训练循环。
2. 观察训练速度：对比默认配置，确认“训练快”这一特点。
3. 记录 loss 下降趋势，判断模型是否正常收敛。

**关键点回顾**
- `r` 控制 LoRA 容量（即可学习参数的规模）。
- `lora_alpha` 控制 LoRA 更新幅度（即每次更新的步长大小）。
- 两者搭配时，`alpha` 通常设为 `r` 的 2 倍，如方案1中的 8 和 16。

完成上述实验后

## 块 28

好的，我们直接进入正题。你正在为LoRA微调打基础，这部分内容非常关键，尤其是关于Rank和Alpha的常见误区，能帮你少走很多弯路。

咱们直接看最常见的三个坑：

**误区1：Rank越大一定越好？**
**错。** 这就像用大炮打蚊子。如果你的任务本身很简单，把Rank设得很大，不仅浪费显存，还会白白增加训练成本。记住，Rank是模型的“容量”，但不是越大越聪明。

**误区2：只调Rank，不调Alpha？**
这会导致比例失衡。举个例子，你原来设置 `r=8, alpha=16`，效果不错。后来你把Rank改成64，但Alpha还是16，会发生什么？

计算一下缩放比例：`16/64 = 0.25`。这个比例直接决定了LoRA对模型更新的影响强度。比例变小了，LoRA的影响力就变弱了，相当于你换了更大的“容器”，但只往里面倒了一点点水。

**误区3：Alpha越大越好？**
**错。** Alpha本身不是越大越好，关键看 `α/r` 这个缩放比例。如果这个比例过大，会导致模型更新过强，训练过程就会不稳定，甚至发散。

> **重要提醒**：`α/r` 过大只是训练不稳定的**可能原因之一**。如果训练震荡，也要检查学习率是否过高、数据质量、batch size是否过小、精度设置（fp16/bf16）以及是否缺少梯度裁剪。别只盯着Alpha。

---

**关于你的电脑配置**

好消息是，学习本章节的数学原理和参数含义，**完全不需要GPU**。任意现代CPU和4GB以上内存就足够了。

如果你之后要进行实际的LoRA训练，才需要NVIDIA CUDA GPU。显存方面，LoRA训练经验参考约8GB+，QLoRA因为用了4-bit

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。它的核心思想是：不微调整个大模型，只训练一小部分新增参数。

而 `LoraConfig` 就是你在 LoRA 微调中要创建的第一个配置对象——它负责告诉 PEFT 四件事：

1. **LoRA 放在哪里**（target_modules）
2. **LoRA 规模多大**（r）
3. **LoRA 训练强度如何**（lora_alpha）
4. **是否使用 dropout**（lora_dropout）

你可以把 `LoraConfig` 理解成一份"施工图纸"：模型是房子，LoRA 是在墙上加装的小型加固结构，而这份图纸决定了加固点在哪、用多少材料、加固力度多大。

### LoRA 训练流程

```
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

注意流程中的第二步：**创建 LoraConfig** 发生在训练之前，它是整个 LoRA 微调的起点。后面的 target_modules、r、lora_alpha、lora_dropout 这四个参数，都是在创建 `LoraConfig` 时填写的。

### 一个直观类比

想象你在给一件大衣缝补丁：

- **target_modules**：补丁缝在哪个位置（袖子、领口还是下摆）
- **r**：补丁的尺寸大小（决定能容纳多少"修补信息"）
- **lora_alpha**：补丁的粘合强度（决定修补效果有多明显）
- **lora_dropout**：缝补时随机跳过一些针脚（防止补丁太"死板"）

接下来，我们就逐一拆解这四个参数的具体含义和配置

## 块 30

## 1.2 一个简单示例：动手搭一个LoRA配置

你正在学LoRA微调，目标很明确——用最少的时间跑通流程。下面这段代码就是LoRA最核心的配置，先把它复制到你的环境里跑一遍：

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

跑完没报错，说明配置对象创建成功。现在看这张表，记住每个参数管什么：

| 参数 | 作用 |
| --- | --- |
| target_modules | 决定插入哪些层 |
| r | 决定LoRA矩阵大小 |
| lora_alpha | 控制缩放强度 |
| lora_dropout | 控制随机失活 |

**动手验证一下**：把 `r` 改成4，再跑一次，观察配置对象能正常创建——这就是你亲手调整LoRA参数的第一步。

---

## 二、LoRA配置整体流程

光有配置还不够，你得知道它怎么接进模型。整体流程是：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

对应代码如下：

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些部分动手脚”**。

你正在做 LoRA 微调，核心思路是：冻结原始权重矩阵 W，只往某些关键位置插入一小批可训练的低秩矩阵。那插入到哪里？就由 `target_modules` 来指定。

### Transformer 里的常见结构

大模型内部通常有 Attention（注意力）模块，它由几个子层组成：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

**LoRA 不会默认修改所有层**，你需要显式告诉它选哪些。

### 怎么指定？

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码表示：只给 **Query 投影层** 和 **Value 投影层** 加上 LoRA。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开理解：

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

所以调整 Q 和 V，通常就能有效改变模型的行为——这就是为什么它们常被选为 LoRA 的注入点。

> ⚠️ **重要提醒**：`q_proj` / `v_proj` 只是**常见起点**，不是所有模型都通用的“标准配置”。不同模型的模块命名可能不同，**配置前应先查看模型实际模块名**（用 `model.named_modules()`），再决定 `target_modules`。

---

### 偷懒技巧：`target_modules="all-linear"`

较新版本的 PEFT 支持直接写 `"all-linear"`，让 PEFT 自动匹配模型中（几乎）所有线性层，省去手工枚举。示例：

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

**注意权衡**：覆盖模块变多 → 可训练参数和显存占用也会增加。是否用全线性层，要在**参数量/

## 块 32

## 方案1：只调整Q和V——轻量起步

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**特点：**
- 参数少，训练快
- 是常见的轻量起点（具体效果仍需看模型结构与任务表现）

---

### 动手验证（建议实操）

1. 用上述配置跑一次微调，记录训练耗时和显存占用。
2. 对比只调Q/V与全量微调的效果差异，观察哪些任务上表现接近。

---

**小结**：方案1适合快速验证LoRA流程，先跑通再逐步扩展。后续可尝试加入更多模块（如k_proj、o_proj或FFN层）来增强表达能力，但需权衡显存消耗。

## 块 33

## LLaMA系列：target_modules怎么配？

对于LLaMA系列模型，最常见的`target_modules`配置是：

```python
[
"q_proj",
"v_proj"
]
```

这意味着LoRA只作用于注意力机制中的**查询（Q）投影**和**值（V）投影**这两个线性层。这是社区验证过的默认稳妥选择——只改这两个层，既能有效微调，又不会引入过多额外参数。

> **动手理解**：LLaMA的注意力层包含`q_proj`、`k_proj`、`v_proj`、`o_proj`四个投影。上面只选了Q和V，K和O保持冻结。你可以思考：为什么只动Q和V就够用？——因为LoRA的核心思想是**用低秩矩阵近似权重更新**。

---

## 对比：GPT-2的target_modules长什么样？

GPT-2的配置可能长这样：

```python
[
"c_attn"
]
```

**关键区别**：GPT-2的注意力不是像LLaMA那样把Q/K/V拆成三个独立层，而是把三者**合并**在一个叫`c_attn`的模块里。

---

## ⚠️ GPT-2的坑：`Conv1D`与`fan_in_fan_out=True`

这是最容易踩坑的地方，务必注意：

GPT-2的注意力层用的是Transformers里的`Conv1D`（**不是**普通的`nn.Linear`）。`Conv1D`的权重形状是`(in_features, out_features)`——和`nn.Linear`的`(out_features, in_features)`**正好转置**。

**后果**：在这类层上加LoRA时，必须设置`fan_in_fan_out=True`，PEFT才能按转置布局正确注入低秩矩阵。不设的话，矩阵维度对不上，模型会报错或训练异常。

**可运行示例（GPT-2）**：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 注意：Llama-2-7b FP16 需要至少14GB显存。
model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = Lora

## 块 34

**BERT 的 target_modules 配置：别照搬，先查模块名**

你之前可能见过 LLaMA 系模型用 `["q_proj", "v_proj"]`，但 **BERT 不一样**——它通常只配 `["query", "value"]`。原因很简单：不同模型的内部模块命名不同，**不能直接复制**。但具体模块名需以实际模型为准，不同库或版本可能有所不同。

**动手查一下（30秒）**

加载你的 BERT 模型后，运行：

```python
# 注意：加载模型时请确保显存充足，例如 Llama-2-7b FP16 需要至少14GB显存
for name, _ in model.named_modules():
    print(name)
```

输出里找 `query` 和 `value`（或类似名字），**以你实际看到的为准**，再填进 `target_modules`。

**匹配机制速记**

- `target_modules` 支持**精确匹配或正则匹配**，例如设置 `"q_proj"` 会匹配所有名为 `"q_proj"` 的模块

## 块 35

## 6.1 r是什么？

r就是rank（秩），它控制LoRA低秩矩阵的维度。

LoRA的核心公式是：

```text
ΔW = (α/r)BA
```

这里的r，决定了A和B的中间维度大小，α/r是缩放因子，用于控制更新幅度。

**举个例子**：假设原矩阵是4096×4096，设置r=8，那么：

- A的维度是：4096×8
- B的维度是：8×4096

也就是说，r=8时，我们用两个小矩阵（总共4096×8 + 8×4096个参数）来近似表达原来4096×4096的更新量，参数量大幅减少。

---

## 6.2 r

## 块 36

## 7.1 alpha 作用：控制 LoRA 的“力度”

LoRA 最终更新时，用的不是 `BA`，而是 `(alpha/r)BA`。

**一句话理解**：`alpha` 控制 LoRA 更新幅度——值越大，微调时对模型权重的改动越“猛”。

**动手算一下**（跟着写）：

```python
r = 8
lora_alpha = 16
```

缩放系数 = `16 / 8 = 2`，表示 LoRA 影响被增强到 2 倍。

---

## 7.2 alpha 常见设置：从 2×r 起步

常见经验起点：`alpha = 2 × r`

> 注意：这只是起点，不是铁律。有人固定 alpha 只调 r，也有人用其他比例

## 块 37

### 8.1 dropout作用

训练时，LoRA会**随机关闭部分神经连接**，防止模型“死记硬背”训练数据，从而减少**过拟合**（即模型在训练集上表现好、但在新数据上表现差）。

配置示例：

```python
lora_dropout=0.05
```

含义：训练中**5%的LoRA路径随机失活**。

---

### 8.2 如何选择dropout？

以下为**经验起点**（非固定规则，需结合验证集是否过拟合来调整）：

| 数据规模 | dropout（起点参考） |
|---------|------------------|
| 数据少   | 约 0.05~0.1      |
|

## 块 38

### 9.1 基础配置：动手搭一个LoRA

这段配置适合在**7B模型**上做测试，是LoRA微调的“最小可用”起点。你不需要动整个模型，只插入两个小模块就能开始。

**直接看代码**（这是你要动手敲的部分）：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：秩为8，控制可学习参数的量
    lora_alpha=16,      # 更新缩放：微调时权重更新的强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的Q和V投影
    lora_dropout=0.05,  # 防止过拟合：训练时随机丢弃5%的LoRA参数
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（如GPT类）
)
```

**四个核心参数速记**：

| 参数 | 值 | 作用 |
|---|---|---|
| r | 8 | LoRA容量（秩） |
| alpha | 16 | 更新缩放 |
| target_modules | q/v | 插入位置 |
| dropout | 0.05 | 防止过

## 块 39

## 9.1 基础配置：把 LoRA 装进 BERT 分类任务

你现在的目标是**用 LoRA 微调一个模型**，而且喜欢动手边做边学。这一节我们直接看一个能跑通的完整例子，把 `target_modules`、`r`、`lora_alpha`、`lora_dropout` 这几个参数一次讲透。

### 第一步：加载基础模型

```python
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里加载的是带分类头的 BERT。注意：**分类头是新初始化的**，它未被 LoRA 冻结，需要训练。

### 第二步：配置 LoraConfig

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # 低秩增量矩阵的秩，越小参数越少
    lora_alpha=16,      # 缩放系数，与秩r共同决定LoRA更新的缩放比例（α/r）
    lora_dropout=0.05,  # 防止过拟合的 dropout 比例
    bias="none",        # 不训练偏置项
    task_type="SEQ_CLS",# 任务类型：序列分类
    target_modules=["query", "value"],  # 只改 BERT 注意力里的 query 和 value 层
    modules_to_save=["classifier"],     # 关键！把分类头也纳入训练并保存
)
```

**重点理解 `target_modules`**：它告诉 LoRA 去修改哪些层。这里只选了 `query` 和 `value`，意味着 BERT 的注意力机制中只有这两个子层会被注入低秩矩阵，其余层全部冻结。这是 LoRA 省显存的核心。

**重点理解 `modules_to_save`**：这是最容易踩坑的地方。如果不加这一行，分类头可能不会被保存，导致重载时分类头权重丢失。加上它，分类头才会随 Adapter 一起保存。

### 第三步：包装模型并查看可训练参数

```python
from peft import get_peft_model

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

运行后你会看到可训练参数包含两部分：LoRA 注入的矩阵 + 分类头。这正是 `modules_to_save` 的效果。

### 第四步：保存与重载验证

训练完成后（代码中省略），保存并验证分类头确实被带回来了：

```python
import torch
from peft import PeftModel

# 保存：Adapter + modules_to_save 指定的分类头
model.save_pretrained("./lora_ckpt")

# 重载：先加载同一个基础模型，再挂载 Adapter
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证分类头权重是否一致
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

如果输出 `True`，说明分类头被成功保存并重载。

# 模型许可证：BERT（bert-base-uncased）采用 Apache License 2.0，可自由商用及修改。

## 块 40

好的，我们直接进入正题。你正在学习 LoRA 微调，目标很明确。这一块内容非常实用，就是告诉你**不同任务该抄哪组参数**。我们直接看代码。

## 10.1 普通文本生成：最省心的起点

如果你只是想跑通一个文本生成任务，或者做一个简单的续写，直接用这组最经典的配置：

```python
r=8
alpha=16
dropout=0.05
```

**怎么理解？** 记住一个关键比例：`alpha / r = 16 / 8 = 2`。这个 2 就是 LoRA 的缩放系数，用于控制低秩增量 ΔW 的幅度，影响微调时权重更新的步长。这个组合是经过大量实践验证的“安全牌”，效果稳定，不容易出错。

## 10.2 指令微调：让模型学会“听话”

如果你要让模型遵循指令（比如让它“总结这段话”或“翻译成英文”），任务更复杂，需要更强的学习能力。推荐：

```python
r=16
alpha=32
dropout=0.05
```

**怎么理解？** 把 `r` 翻倍到 16，意味着让 LoRA 学习更多、更复杂的特征。`alpha` 也跟着翻倍到 32，**保持 `alpha / r = 2` 这个比例不变**。`dropout` 保持 0.05，用来防止过拟合。

## 10.3 领域知识增强：注入专业“内功”

如果你想让模型学习特定领域的知识（比如法律、医学、代码），这属于深度定制，需要更大的容量：

```python
r=32
alpha=64
```

**怎么理解？** `r` 和 `alpha` 再次翻倍，比例依然是 2。更大的 `r` 意味着更强的学习能力，但代价是**显存占用更高、训练更慢**。注意，这里没有写

## 块 41

**本块学习检查**：对照下面的配置，说出每个参数控制什么。

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**四个旋钮，一次看懂：**

- **target_modules**：决定改哪里——只动注意力层的 `q_proj` 和 `v_proj`。
- **r**：LoRA容量，设为8，值越大可学习参数越多。
- **lora_alpha**：通过 α/r 缩放更新强度，16/8=2，即放大2倍。
- **lora_dropout**：0.05，随机丢弃

## 块 42

## 为什么LoRA还不够省显存？

你已经知道LoRA只训练一小部分参数，但有个关键点容易被忽略：**基础模型本身还是要完整加载到显存里**。

打个比方：LoRA像给一本厚书做“重点笔记”，但书本身还得摆在桌上——你不能只带笔记去考试。

具体算一笔账。一个7B（70亿参数）模型，如果用FP16精度存储：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是**模型权重**的占用。实际训练时还要加上：

- **激活值**：前向传播时每层的中间结果
- **梯度**：反向传播时计算的更新方向
- **优化器状态**：如Adam的动量项

这些加起来，实际显存需求会**远高于14GB**。所以即便LoRA大幅减少了“可训练参数”，消费级显卡依然扛不住——瓶颈不在训练参数，而在“加载完整模型”这件事本身。

## 一句话总结

QLoRA要解决的核心问题就是：**能不能把“那本厚书”也压缩一下再放上桌？** 这正是后面4-bit量化要做的事。

---

**本块学习检查**（动手验证你的理解）：

打开

## 块 43

**QLoRA核心思想：用4-bit模型跑LoRA**

你的目标是LoRA微调，但直接在大模型上跑LoRA，显存压力大。QLoRA的思路很简单：**先把基础模型“压缩”成4-bit，再在上面插LoRA训练**。

**流程对比：**

- **LoRA**：FP16模型 + LoRA（模型大，显存占用高）
- **QLoRA**：4-bit模型 + LoRA（模型小，显存占用低）

**训练时谁在动？**

- **基础模型**：4-bit权重，**冻结**（不更新）
- **LoRA参数**：FP16/BF16，**训练**（只更新这部分）

**关键点：** 量化的是**基础模型权重**，**不是**LoRA参数。LoRA保持

## 块 44

## 3.1 模型存储方式：为什么4-bit能省这么多显存？

模型参数在电脑里占用的空间，取决于用多少字节存一个数。简单记：

- **FP32**：4字节
- **FP16/BF16**：2字节
- **INT8**：1字节
- **INT4**：0.5字节

以7B（70亿参数）模型为例：

- 用FP16存：`7B × 2 ≈ 14GB`
- 用4-bit存：`7B × 0.5 ≈ 3.5GB`

显存直接减少约75%。

> ⚠️ 注意：这3.5GB只是**理论下界**——只算了量化后的权重本身。实际运行还要加上量化常数、分组元数据、未量化的模块（如norm、embedding、LoRA参数）以及计算缓存等，真实占用会明显更高。

##

## 块 45

**NF4是什么？——用“查表”理解4-bit量化**

NF4全称NormalFloat4，中文叫“正态分布4位浮点”。别被名字骗了，它不是IEEE那种通用浮点格式，而是QLoRA论文为**神经网络权重近似正态分布**这个特点，专门设计的一套**4-bit量化码本**。

**核心机制（三步走）：**

1. **预置码本**：提前定好16个（2⁴）取值点，这些点按标准正态分布的累积分布函数（CDF）等距分位点排布，使得权重密集的0附近划分更细，稀疏的尾部更粗。
2. **量化（存索引）**：把每组权重归一化后，映射

## 块 46

## 4.2 为什么NF4适合模型权重？

训练好的大模型权重有个特点：**近似正态分布**——大量参数集中在0附近，越往两端越少。

普通INT4量化是**平均划分**区间（比如[-8,-7,...,7]），每个区间一样宽。但权重集中在0附近，平均划分导致中心区域精度不足，损失大。

**NF4的思路**：根据正态分布的形状来设计量化范围——**0附近精度更高**，两端精度低一些。这样更贴合权重分布，精度损失更低。

## 4.3 NF4量化过程（三步）

原始FP16权重（如0.023、-0.145、0.876）经过：

1. **归一化**：将权重

## 块 47

**本块学习检查**：读完本节后，请用一句话回答：双重量化“第二次量化”的对象是什么？

---

**5.1 为什么需要双重量化？**

4-bit量化时，每组权重除了存4-bit数据，还得额外保存两个量化参数：`scale`和`zero point`。这些参数同样占用显存，模型越大，这笔开销越可观。

**5.2 双重量化思想**

普通量化流程：FP16权重 → 4-bit权重 + scale参数。

双重量化在此基础上，**把scale参数本身也做一次量化**。即：第一次量化模型权重，第二次量化量化参数（scale）。

结构简化为：FP16权重 → NF4量化 → 保存4-bit权重；同时scale参数 → 再次量化 → 减少scale存储。

**5.3

## 块 48

## 6.1 为什么需要分页优化器？

训练时显存峰值来自三部分：**模型本身**、**梯度**、**优化器状态**。以Adam优化器为例，它需要额外保存**一阶动量**和**二阶动量**两份状态。

QLoRA虽然大幅降低了模型占用的显存，但训练过程中仍可能遇到**显存突然不足（OOM）**的情况——问题往往出在优化器状态上。

## 6.2 分页优化器思想

Paged Optimizer借鉴了**操作系统虚拟内存**的思路，核心机制是：

> 借助 **NVIDIA 统一内存（Unified Memory）**，当显存出现峰值、即将OOM时，自动把优化器状态**按页换出到CPU内存**，需要参与计算时再**换回GPU**。

流程如下：

```text
GPU显存
   ↓
出现峰值（接近/超出显存上限）
   ↓

## 块 49

## 6.3 类比理解：分页优化器在做什么？

把显存想象成一张**桌面**，把CPU内存想象成**柜子**。

**普通优化器**的做法是：把所有东西（模型参数、梯度、优化器状态）都堆在桌面上。桌面一旦放满，训练就报错（OOM），无法继续。

**分页优化器**的做法是：桌面上只放**常用文件**（当前正在计算的数据），**暂不用的文件**放进柜子（CPU内存）保存，**需要时再取出**。这样桌面永远有空间，训练不会因为显存爆掉而中断。

---

## 6.4 代码里怎么启用？

不用手写换页逻辑，`bitsandbytes` 已经实现好了。只需在 `TrainingArguments` 里设置 `optim` 参数：

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

- `optim="paged_adamw_8bit"`：优化器状态用 8-bit 存储，更省显存。
- `optim="paged_adamw_32bit"`：精度更高，但占用更多显存。
- 两者都是 QLoRA 论文中使用的分页优化器取值，该机制借鉴了vLLM的PagedAttention思想，由bitsandbytes库实现。
- **注意**：不同 Transformers / bitsandbytes 版本支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报"未知 optim"错误，去查你当前版本的文档，换用支持的名称即可。
- 分页优化器主要解决**峰值 OOM**，通常配合 `grad

## 块 50

## 7.1 4-bit量化：给模型“瘦身”

QLoRA 的第一招，是把模型权重从 FP16（16位）压缩成 NF4（4位）。你可以把它想成：原来用 16 位精度存储每个数值，现在用 4 位精度，存储空间大幅减小，但精度有所损失。这一步直接**减少模型权重大小**，是省显存的基础。

## 7.2 双重量化：连“尺子”也压缩

量化时需要一把“尺子”（scale）来记录缩放比例。QLoRA 的妙处在于：**这把尺子本身也被再次量化**。于是，连量化参数占用的空间也省下来了。两层压缩叠加，显存自然更省。

## 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态很吃显存。分页优化器的思路是：**GPU 显存不够时，自动把部分数据交换到 CPU 内存**，

## 块 51

好的，这是为你准备的个性化教学正文。

## 动手配置：让模型“瘦身”加载

你已经了解了LoRA的基本思路，现在我们进入实操环节，看看在代码里如何具体配置QLoRA的4-bit量化。这一块是纯“动手”内容，跟着步骤走就行。

### 第一步：总开关——`load_in_4bit`

这个参数就像电源开关，设置为 `True`，模型才会以4-bit的“压缩”格式加载进显存，这是节省显存的第一步。

### 第二步：量化“算法”——`bnb_4bit_quant_type`

设置成 `"nf4"`。你可以把它理解为一种特定的“压缩算法”，它决定了权重是如何被压缩到4-bit的。QLoRA论文里推荐的就是这个。

### 第三步：再省一点——`bnb_4bit_use_double_quant`

把它设为 `True`，开启“双重量化”。这相当于对“压缩算法”本身再压缩一次，能进一步减少显存占用，是QLoRA的一个关键优化点。

### 第四步：关键区分——`bnb_4bit_compute_dtype`

这是最容易混淆的地方，请务必注意：

*   **存储 vs 计算**：权重是用4-bit（NF4）**存储**的，这是为了省显存。但在真正做矩阵运算时，会临时“解压”回一个更高精度的格式来计算，这个计算时用的精度就是由 `bnb_4bit_compute_dtype` 决定的。
*   **为什么重要**：若不设置，`compute_dtype` 默认与模型加载时的 `torch_dtype` 一致，通常为 `float16` 或 `bfloat16`，但显式设置可确保计算精度符合预期，避免因默认值不确定而影响训练效果。
*   **怎么选**：通常用 `torch.bfloat16`（BF16）或 `torch.float16`（FP16）。**不要写死**，要根据你的GPU能力来选择。BF16需要较新的GPU（如NVIDIA RTX 30/40系、A100等），老一点的GPU（如RTX 20系）不支持，就得用FP16。

**最稳妥的写法**是让代码自动判断：

```python
import torch
from transformers import BitsAndBytesConfig

# 根据GPU能力选择计算dtype：支持BF16就用bfloat16，否则退回float16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个**能直接跑的最小骨架**，让你看到 QLoRA 训练的完整链路。

### 整体流程（5 步）

1. **配置 4-bit 量化** → 2. **加载模型** → 3. **prepare 准备训练** → 4. **注入 LoRA** → 5. **训练**

### 逐步拆解

**第 1 步：4-bit 量化配置**

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

**第 2 步：以 4-bit 加载基础模型**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except OSError as e:
    print(f"模型加载失败，请检查模型路径或网络连接: {e}")
    raise
model = model.to(device)
```

注意：tokenizer 的 pad_token 若为空，要设为 eos_token（否则训练可能报错）。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）

**第 3 步：prepare_model_for_kbit_training（关键！）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步做两件事：处理 norm/输出层的精度问题，并开启梯度检查点所需的输入梯度。**没有这一步，4-bit 模型无法正常训练。**

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

冻结原始模型权重，仅训练低秩矩阵 A 和 B 形成增量更新（r=8 表示秩为 8）。`target_modules` 按你的模型实际模块名调整，或用 `"all-linear"` 省事。

**第 5 步：训练**

```python
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,   # 与第 3 步呼应
    optim="paged_adamw_8bit",      # 分页优化器
)
```

两个关键点：
- `gradient_checkpointing=True` 必须与第 3 步的 `use_gradient_checkpointing=True` 同时开启
- `optim="paged_adamw_8bit"` 使用分页优化器，避免显存峰值溢出

### 数据准备（极简示例）

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()  # Causal LM 的 labels 直接复制 input_ids
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
```

真实场景请替换为足够规模的数据。

### 完整骨架的运行顺序

```
量化配置 → 加载模型 → prepare → LoRA注入 → 数据准备 → TrainingArguments → Trainer.train()
```

### 常见坑位提醒

- **pad_token 未设置**：训练时可能报错，记得检查
- **prepare 步骤跳过**：4-bit 模型直接训练会出问题
- **两个 gradient_checkpointing 不匹配**：第 3 步和第 6 步必须同时为 True
- **optim 名称随 bitsandbytes 版本可能不同**：如果报错，检查当前版本支持的优化器名称

---

**本块学习检查**：把上面骨架复制到你的环境，替换 `model_name` 为你本地的任意 Causal LM（如 `"gpt2"`），跑通后观察 `model.print_trainable_parameters()` 输出的可训练参数量占比。记录你实际使用的 `target_modules` 和可训练参数量，确认 Lo

## 块 53

## 把 QLoRA 的“五步骨架”串起来

你现在要做的不是背概念，而是把前面学的 4-bit 量化、双重量化、分页优化器这些零件，**组装成一段能跑的代码骨架**。下面这五步就是完整的装配顺序，缺一步都可能翻车。

### 五步装配顺序（缺一不可）

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么“瘦身”加载**

这一步是给基础模型设定 4-bit 加载规则，核心三要素：NF4 量化类型、双重量化开关、计算 dtype。它决定了模型以什么精度住进显存。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的“必要一步”**

这是最容易跳过的坑。它专门为量化模型做训练适配：冻结模型参数、将非量化层转为 fp32、启用输入梯度等。**跳过它，常见后果是训练不收敛或直接报错**。记住：它不是可选项，是必选项。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

这一步把可训练的 LoRA 适配器“插”进冻结的 4-bit 主干模型上。主干不动，只训练 LoRA 那一点点参数。

**第 4 步：梯度检查点要“两头都开”**

`TrainingArguments` 里的 `gradient_checkpointing=True`，同时确保模型配置中 `use_gradient_checkpointing=True`（或通过 `model.gradient_checkpointing_enable()` 开启）。**必须对应开启**。只开一边，配置就对不上。

**第 5 步：`optim="paged_adamw_8bit"` —— 开分页优化器**

这一步启用分页 8 位优化器，作用是缓解峰值显存——通过 8 位量化减少优化器状态显存占用，并利用分页机制在显存不足时将优化器状态临时换出到 CPU 内存。

### 给你的实操提醒

上面这个骨架是“能说明关系”的最小版本。真实训练时你要注意三点：

1. **换数据**：用足够规模的真实数据集，不是演示用的小样本；
2. **调超参**：学习率、训练轮数等要按任务重新设；
3. **核对版本**：`optim` 的名称、`prepare_model_for_kbit_training` 的参数、`processing_class`/`tokenizer` 的写法，都要按你实际用的库版本确认。

---

## 你的电脑能不能跑？（配置

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会以为LoRA训练完，得到一个“升级版”的完整模型。**不是的。**

LoRA训练过程中，基础模型是被**冻结**的（不参与训练），真正被训练的是一个额外添加的**Adapter**（适配器）。训练完成后，你的模型结构其实是：

```
基础模型（冻结，没变）
+
LoRA Adapter（训练后的增量知识）
```

以Llama-7B为例，训练完保存的通常只有两个文件：

- `adapter_model.safetensors`（Adapter权重）
- `adapter_config.json`（Adapter配置）

**而不是**完整的Llama-7B模型文件。

---

### 1.2 Adapter有什么特点？

**优点：**

- 文件小（相比动辄几GB的完整模型，Adapter可能只有几十MB）
- 保存快
- 一个基础模型可以搭配多个Adapter，实现多任务复用

比如同一个基础模型，你可以搭配：

```
客服Adapter  →  做客服对话
医疗Adapter  →  做医疗问答
法律Adapter  →  做法律咨询
```

**缺点：**

部署推理时，需要**基础模型 + Adapter**两部分同时存在，不能只用其中一个。

---

## 二、为什么需要合并LoRA权重？

既然Adapter有“轻量、灵活”的优点，为什么还要合并？

因为部署场景下，同时维护“基础模型 + Adapter”两个部分很麻烦：

- 推理框架要额外加载Adapter逻辑
- 每次都要确保两个文件版本匹配
- 不方便直接分发或部署

**合并（merge）**就是把Adapter的增量知识“写回”基础模型，得到一个**完整的、独立的模型**。合并后，你只需要一个模型文件，不再依赖Adapter。

---

## 本块学习检查

请用一句话回答：训练完成后，你

## 块 55

**本块学习检查**：读完下面内容后，请合上资料，在纸上画出“未合并”和“合并后”两种推理流程图，并标出各自的输入和输出。

---

**2.1 推理时的问题**

未合并时，模型推理需要两步：先加载基础模型（Base Model），再加载LoRA适配器（Adapter），两者叠加才能得到输出。

但对于部署场景，我们通常希望模型是一个完整的整体——输入直接进入“完整模型”就能得到输出，不需要额外挂载适配器。因此，**需要合并权重**。

> # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
> # 注意：Llama-2-7b FP16 需要

## 块 56

## 3.1 merge_and_unload() 是做什么的？

你现在已经训练好了LoRA，但训练完的模型还是一个“套着壳子”的 `PeftModel`——里面既有基础模型，又额外挂着LoRA适配器。这样用起来不方便，部署或推理时还得带着两套东西。

`merge_and_unload()` 就是干这件事的：**把LoRA权重合并进基础模型，然后移除PEFT的包装**。

用公式看就是：

```text
基础模型 + LoRA适配器 → 完整模型
```

合并前，模型结构是：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

合并后，变成：

```text
普通Transformer模型（包含LoRA更新后的权重）
```

也就是说，合并后你拿到的就是一个**普通的完整模型**，不再有LoRA的痕迹。

---

## 完整流程：从训练到导出

整个合并导出的流程是：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面一步步来。

---

## 加载LoRA Adapter

假设你有基础模型路径 `base_model_path` 和训练好的Adapter路径 `adapter_path`，先加载基础模型，再挂上Adapter：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：基础模型 + LoRA适配器。

---

## 执行合并

一行代码搞定：

```python
merged_model = model.merge_and_unload()
```

这行代码做的事，用公式表示就是：

```text
W + (α/r)BA → W'
```

`W` 是原来的权重，`BA` 是低秩矩阵 A 和 B 的乘积，`α` 为缩放系数，`r` 为秩，合并后得到更新后的权重 `W'`。

此时 `merged_model` 已经**不是** `PeftModel`，而是**普通Transformer模型**。

---

## 导出合并后的模型

合并完成后，保存模型和分词器：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后的目录大致长这样：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：这是**示意结构**。实际文件会随模型/分词器类型变化，比如可能还有 `generation_config.json`、`special_tokens_map.json`，有些模型的tokenizer是 `vocab.json`/`merges.txt` 或 `tokenizer.model`，大模型还可能有分片权重文件。

---

## 本块学习检查

**动手验证一下**：用你之前训练好的LoRA模型，执行以下操作：

1. 加载基础模型和Adapter
2. 调用 `merge_and_unload()` 合并
3. 导出合并后的模型并确认其不再是 `PeftModel`

> 部署提示：若使用 Llama 2 系列模型，请遵守其社区许可证（商用月活>7亿需Meta授权）；加载时注意显存占用，例如：Llama-2-7b FP16 需要至少14GB显存。

## 块 57

## 三、merge_and_unload是什么？（重点）

**一句话理解**：`merge_and_unload` 将 LoRA 训练得到的低秩增量矩阵合并到原始权重中，得到完整的模型权重，之后推理不再需要加载 LoRA 适配器。

**为什么需要它？** 训练时模型是“底座 + 补丁”的组合体，推理时每次都要额外加载补丁逻辑，麻烦且容易出错。合并后，补丁和底座融为一体，直接用 HuggingFace 标准方式加载即可。

---

## 合并后必查清单（避免“能加载但行为不对”）

合并导出后，**别急着欢呼**，先核查这四项：

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致。如果你训练时**新增或修改过特殊 token**（比如自定义的 `<|im_start|>`、pad/eos），确认它们都在 `special_tokens_map.json` / `tokenizer_config.json` 里。如果训练时扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。

2. **generation_config.json**：如果你依赖特定的生成默认值（如 eos/pad token id、`max_new_tokens`、采样参数），确认它一并导出了。否则推理时可能回退到不合适的默认行为。

3. **dtype（精度）**：确认合并/保存时的精度符合预期（FP16/BF16 还是 FP32）。dtype 不一致会影响显存占用和数值表现。

4. **config.json 完整性**：确认模型结构配置（层数、hidden size、任务头等）正确，且**已移除 PEFT/adapter 相关配置**。合并后应是普通 Transformers 模型，不应再残留 `adapter_config.json`。

---

## 合并后如何加载推理？

合并后模型通常可直接用 Transformers 加载，无需 PEFT 库，但需确保模型结构完整。

```python
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained("./merged_model")
except OSError as e:
    print(f"模型加载失败: {e}")
    raise
```

# 注意：合并后模型大小与原始模型相同，加载时需相应显存。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

流程：

## 块 58

### 10.1 推荐合并场景

你正在学LoRA微调，目标很明确：把模型用起来。什么时候该把LoRA权重合并回基础模型？记住两个核心场景。

**场景1：上线部署——只要一个Adapter，追求简单**

想象生产环境：服务器 → 直接加载模型 → 提供API。如果你只需要一个固定的Adapter，且希望部署链路尽量简单——不额外依赖PEFT库、不做运行时Adapter加载——合并通常更方便。

但注意：如果你的推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你需要在同一基础模型上热切换多个Adapter，那么不合并、以Adapter形式部署反而更灵活。这时候是否合并，要看框架能力和业务需求。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**场景2：分享模型——让对方开箱即用**

如果你希望对方拿到就能用、

## 块 59

**本块学习检查**：读完下面内容后，请合上资料，口头说出“不建议合并”的两个场景各是什么，并解释为什么保留Adapter更灵活。

---

### 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象一个基础模型上挂了多个Adapter（如A、B、C），每个负责不同任务。这时**不要合并**——保留它们，你随时可以切换使用，非常灵活。一旦合并，就固定死了，失去这种灵活性。

**场景2：持续训练**

如果你还打算继续调整LoRA，也**先别合并**。保留Adapter，方便后续继续微调。

---

### 十一、QLoRA合并注意事项（重点）

QLoRA有个特殊点：基础模型是**4-bit量化存储**的

## 块 60

## 11.1 直接合并可能遇到的情况

你可能会想：训练完 QLoRA 后，直接调用 `merge_and_unload()` 把权重合并了不就行了？**可以，但要注意**——直接在 4-bit 状态下合并，可能出现两类问题：

1. **精度损失**：4-bit 权重需要先反量化再合并，这个过程与全程用高精度计算并不完全等价，合并后的模型质量可能打折扣。
2. **类型或版本不兼容**：可能出现权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> ⚠️ 注意：上面说的是**可能出现**的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你的**版本**、量化配置，以及是否按下面的流程先反量化。**以你所用版本的文档和实际运行结果为准。**

---

## 11.2 常见做法：先反量化到高精度再合并

既然直接合并有风险，那稳妥的做法是什么？核心思路就一句话：

> **不在 4-bit 状态下合并，而是先用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一份 Adapter 后合并。**

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

这样合并发生在高精度权重上，绕开了 4-bit 状态下的类型和精度问题。

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

> 📄 **模型许可证**：使用基础模型前，请确认其许可证类型。例如，Llama 2 采用 Llama 2 Community License，商用月活用户超过 7 亿时需获得 Meta 授权。合并后的模型继承基础模型的许可证约束，请按对应条款使用。

### 代码要点拆解

| 步骤 | 关键点 | 为什么 |
|------|--------|--------|
| 加载基础模型 | **不传** `quantization_config

## 块 61

### 11.1 直接合并可能遇到的情况（QLoRA 合并前必读）

你正在做 LoRA 微调，目标是把训练好的 adapter 合并回基础模型。但 QLoRA 用的是 4-bit 量化基础模型，直接合并会遇到三个关键点：

**1. 版本决定能不能直接合并**
PEFT 的 `merge_and_unload` 不支持 4-bit 量化模型，需先反量化或加载非量化权重（如不传 `quantization_config` 或 `load_in_4bit`）再合并。具体支持情况请查阅所用 PEFT 版本的官方文档。

**2. 内存开销要心里有数**
以高精度重新加载完整基础模型，会占用约等于一份

## 块 62

## 合并时开启校验并做一致性验证

合并模型时，给 `merge_and_unload` 加上 `safe_merge=True` 参数，它会在合并过程中自动做数值检查（比如检测有没有产生 NaN），相当于给合并操作上了一道保险：

```python
# 注意：Llama-2-7b FP16 需要至少14GB显存
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，**强烈建议**对比合并前后模型在同一批输入上的输出，确认合并没有引入明显偏差。流程如下：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits 或生成文本是否基本一致

最小验证代码：

```python
import torch
from transformers import AutoModelForC
# 模型许可证：Ll

## 块 63

## 11.3 合并时开启校验并做一致性验证（建议）

合并完模型，别急着收工。QLoRA 的合并本质上是把低秩适配器的权重“加回”基座模型，这个过程涉及精度转换，可能会出现细微偏差。所以，**建议你合并后做一次“前后对比”**，确认合并没有破坏模型行为。

### 为什么必须单独加载一份“合并前”的模型？

`merge_and_unload()` 返回合并后的基础模型，原 `PeftModel` 对象在调用后仍保留适配器权重，并未被修改或销毁。因此，原对象仍可作为“合并前”的基准，无需重新加载。

不过，为了确保对比的公平性，建议你重新加载一份全新的、未合并的 `PeftModel`，专门用来做对比。加载时需注意使用与量化配置一致的 `BitsAndBytesConfig`（如 `load_in_4bit=True`），以保证对比的是量化+合并前后的差异，而非量化本身带来的差异。

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)  # 与训练时量化配置一致
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
    quantization_config=quant_config,
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(量化)+Adapter
```

### 校验步骤一：比较 logits

用同一个固定输入（比如 `prompt = "请用一句话介绍你自己。"`），分别让“合并前”和“合并后”的模型前向传播，计算 logits 的最大绝对差：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(model_before.device)

with torch.no_grad():
    logits_before = model_before(**inputs).logits
    logits_after = merged_model(**inputs.to(merged_model.device)).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

**怎么看结果？** 差值越接近 0 越好。QLoRA 场景下，由于量化精度损失，通常是一个较小的近似值，而不是严格为 0。只要数量级很小，就说明合并基本保真。

### 校验步骤二：比较生成文本

logits 差异是数值层面的，更直观的是看**生成结果是否一致**。用 `do_sample=False` 去掉随机性，保证对比公平：

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs.to(merged_model.device), max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后:", text_after)
print("生成文本是否一致:", text_before == text_after)
```

**重点观察**：两段文本是否完全一致。如果一致，说明合并对模型行为没有可感知的影响；如果不一致，检查差值大小和差异

### 部署注意事项

- **模型许可证**：部署文档引用模型时需标注许可证类型，例如：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
- **显存要求

## 块 64

## 十一、合并时开启校验并做一致性验证（建议）

### 为什么必须单独再加载一份模型？

`merge_and_unload()` 会**返回**一个合并后的新模型——它把 LoRA Adapter 合进底层权重，并返回不再带有 PEFT 包装的模型。**原始 `model` 对象本身不会被修改**。

**关键陷阱**：如果你拿这个合并后的模型去和“合并前”对比，而“合并前”的模型实际已被你误用为合并后的返回值覆盖，两边其实是同一份已合并的权重，`max_abs_diff` 会**恒为 0**——验证完全失效。

**正确做法**：在调用 `merge_and_unload()` 之前，先保存一份未合并的模型引用作为对照基准，再和合并后的模型做对比。具体代码示例：

```python
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, adapter_path)
model_before = model  # 保存未合并的模型引用
model_merged = model.merge_and_unload()  # 返回新模型，原 model 不变
# 对比 model_before 和 model_merged 的输出
```

### 显存不够怎么办？

如果无法同时放下两个模型，采用**分两次跑**策略：

1. 先用未合并的 PeftModel 跑一遍推理，用 `torch.save` 将 logits 或生成文本**保存到磁盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比两次输出。

**确保两次输入一致**：设置相同的随机种子和采样参数（如 `temperature`、`top_p`），并确保输入 token 完全一致。

### 结果预期：别追求“逐位相同”

-

## 块 65

## 十一、QLoRA模型合并注意事项（重点）——合并时开启校验并做一致性验证（建议）

> **模型许可证**：本部署文档引用的基础模型为 Llama 2，适用 **Llama 2 Community License**（商用且月活用户>7亿时需获得Meta授权）。请在使用前确认您的应用场景符合许可证条款。

合并QLoRA模型时，最稳妥的做法是**先"反量化"再合并**。这里说的"反量化"不是手动逐层还原4-bit权重，而是**重新加载一份不带量化配置的基础模型**，具体做法是：

```python
# 不传 quantization_config / load_in_4bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16/float16
)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
```

这样合并发生在高精度权重上，天然避开了"在4-bit状态下合并"的类型/精度问题。

**为什么建议这么做？** 不同PEFT / bitsandbytes / Transformers版本对"量化模型能否直接调用`merge_and_unload`"的支持不同——部分版本会直接报错或警告，部分较新版本对某些情形有一定支持但仍可能有精度损失。**"先反量化（重载高精度）再合并"是跨版本最稳妥的做法**。是否可以在量化状态下直接合并，请以你所装版本的`merge_and_unload`文档与实际运行结果为准。

---

### 常见问题排查速查表

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload`不存在 | 当前对象不是PeftModel（可能忘了用`PeftModel.from_pretrained`挂载Adapter，或已合并成普通模型） | 确认加载方式，先得到PeftModel再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是Adapter小体积 | 无需处理；若在意体积可保留Adapter不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是dtype用了FP32、未做offload/分片 | 改用CPU合并或offload、用FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存tokenizer；或特殊token / generation_config未一并导出 | 重新`save_pretrained`并核查tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 在4-bit状态直接合并、PEFT/bitsandbytes版本不支持、量化包装未正确反量化、内存/显存不足 | 先反量化到FP16/BF16再合并；核对版本；必要时用CPU合并；开`safe_merge=True`并做一致性验证 |

---

### 动手验证（建议）

按上面的代码流程，用你自己的模型和Adapter跑一遍合并，然后

## 块 66

## 路线A：合并后部署完整模型

你已经学会了用 `merge_and_unload()` 把 LoRA 权重合并回基础模型。现在来看合并之后怎么部署——这是最直接的路线。

### 操作流程

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

三步走：合并 → 保存 → 当普通模型加载。就这么简单。

### 这条路线的特点

**优点：**

- **部署链路简单**：推理框架把合并后的模型当作普通模型加载，完全不需要理解 LoRA 是什么。你不需要在部署环节解释"这个模型是微调过的"，框架只看到一个完整模型。
- **推理时零切换成本**：因为 LoRA 已经融进模型权重里了，推理时不存在"切换 Adapter"这个动作，也就没有相关开销。

**代价：**

- **存储成本高**：每合并一个 Adapter，就要保存和分发一份完整的模型文件。如果你有多个任务（多个 Adapter），就得存多份完整模型。假设基础模型 7GB，3 个 Adapter 就是 3 份 7GB，存储和分发成本直线上升。

### 什么时候选路线A？

简单说：**任务少、追求部署简单**时选路线A。如果你只做一个任务的微调，合并后直接部署是最省心的方式——不需要额外配置，任何推理框架都能加载。

---

## 动手验证一下

如果你已经完成了合并，可以检查一下合并后的模型目录：确认里面是完整的模型文件（而不是 `adapter_model.safetensors` 这种小体积的 Adapter 文件）。这个区别很关键——合并后的模型是"完整模型"，而 Adapter 只是"补丁"。

---

**小提示**：路线A和路线B（保留 Adapter 动态服务）的核心区别在于——路线A

## 块 67

## 14.3 怎么选？——两条路线的决策指南

先记住一句话：**合并不是唯一出路**。你完全可以根据场景灵活选择。

**什么时候选“合并”（路线A）？**
如果你只训练了**一个** LoRA 适配器，并且希望部署时最简单、不引入额外依赖——那就合并。合并后就是一个完整的独立模型，直接拿去用。

**什么时候选“保留Adapter”（路线B）？**
如果你打算用**同一个基础模型**服务**多个不同的 LoRA 适配器**（比如一个模型同时服务客服、写作、翻译等多个场景），而且你的推理框架**原生支持 LoRA**——那就别合并，保留 Adapter 做动态切换，省去重复加载基础模型的成本。

**一个特殊提醒（QLoRA用户必看）：**
如果你用的是 QLoRA（量化版），精度比较敏感。**无论走哪条路线**，上线前都建议做一次一致性验证——对比合并前后（或上线前后）的输出是否一致，

## 测评题目

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1204-003
分页优化器的主要目标是？

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。
