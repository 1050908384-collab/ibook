# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么跑的：

```
输入数据 → 整个模型参数更新 → 保存完整模型
```

以 7B（70亿）参数模型为例，训练时你需要同时保存三类东西：

- **参数**：模型本身的权重
- **梯度**：用于反向传播的中间结果
- **优化器状态**：如 Adam 的动量项

这三者叠加，显存需求会高得离谱。普通消费级 GPU 根本扛不住——不是显存溢出，就是训练慢到怀疑人生。

---

**为什么这对你很重要？**

你的目标是 QLoRA 微调，而 QLoRA 正是为了解决“全参数微调太吃显存”这个痛点而生的。先理解这个痛点，后面用 LoRA / QLoRA 时你才能明白每一步配置在解决什么问题。

---

**动手验证一下（可选

## 块 2

**个人化教学正文（Markdown）**

你的目标是QLoRA微调，且偏好动手实操。这里直接切入核心：**PEFT（参数高效微调）** 的精髓就是“不修大模型，只训小参数”。

想象你有一个巨大的预训练模型，全参数微调要动它所有参数，显存需求“非常高”。PEFT的思路是：保留原始大模型不动，在旁边挂一个**LoRA适配器**（新增的少量参数）。训练时，**只训练LoRA参数**，最后**保存这个小规模Adapter**即可。

对比三种方式，显存需求从高到低：全参数微调（非常高） > LoRA（较低） > QLoRA（更低）。QLoRA通过4

## 块 3

### 2.1 LoRA是什么？（①）

LoRA（Low-Rank Adaptation，低秩适配）是一种参数高效微调方法，核心思想是**冻结原模型权重，只训练新增的少量参数**。

**原理拆解：**
- 原模型权重记为 `W`，在微调时**不更新**。
- 新增一个增量 `ΔW`，最终模型变为 `W + ΔW`。
- 关键来了：`ΔW` 被分解为两个低秩小矩阵的乘积：`ΔW = B × A`。
  - 若 `W` 形状为 `d_out × d_in`，则 `A` 形状约为 `r × d_in`，`B` 形状约为 `d_out × r`。
  - 相乘后 `B × A` 与 `W` 形状一致。注意：不同资料可能写成 `A × B`，只是符号顺序不同，关键是形状能相乘并还原。

**训练时只更新 `A` 和 `B` 两个小矩阵**

## 块 4

## 3.1 LoRA的问题：模型还是太大

你已经知道LoRA能大幅减少**训练参数**，但有个关键点别忽略：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个7B参数的模型，如果用FP16精度加载，大约需要 **14GB显存**。也就是说，即使LoRA只训练一小部分参数，你的GPU也得先装得下整个基础模型才行。

这就像你要装修一间大房子——虽然只改动几面墙（LoRA参数），但整栋楼（基础模型）得先立在那儿。

---

## 3.2 QLoRA的解法：先把模型“压缩”再训练

QLoRA全称 **Quantized LoRA**，即“量化版LoRA”。核心思路很简单：

```text
FP16模型 → 4bit量化 → 显存占用大幅下降 → 再加载LoRA进行训练
```

训练时的结构变成了：

```text
4bit基础模型 + LoRA参数 → 训练
```

这样，基础模型变小了，显存压力自然就缓解了。

---

## 3.3 关键依赖：bitsandbytes

要跑QLoRA，你需要这几个组件：

```text
transformers + peft + bitsandbytes + 支持的量化后端
```

其中 **bitsandbytes** 负责三件事：

- 低精度量化
- 4bit加载
- 8bit加载

> **关于硬件后端的重要说明**：本教程统一以 **NVIDIA CUDA GPU** 为准，这是当前支持最完整、资料最全的路径。但请注意，“QLoRA只能用NVIDIA”并非技术必然——bitsandbytes正在扩展对CPU、AMD ROCm、Intel等后端的支持。如果你的硬件不是NVIDIA GPU，务必查阅官方文档确认对应后端与安装方式，本文示例命令未必适用。
>
> 各后端支持程度差异很大：NVIDIA CUDA是完整支持且经过最广泛验证的路径；AMD ROCm、Intel等后端在不同版本中可能只是实验性/部分支持，功能、性能和稳定性都可能与CUDA有差距。官方支持矩阵见：
> - bitsandbytes安装/后端支持文档：https://huggingface.co/docs/bitsandbytes/main/en

## 块 5

## 电脑配置批注：跑得动吗？

你的目标是QLoRA微调，核心诉求是“在更小的显存上微调大模型”。先看底线，再看推荐。

### LoRA最低配置（先打个底）

- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA GPU
- **显存**：最低8GB，推荐12GB+

### QLoRA要求（你的主战场）

本教程实训范围限定为 **NVIDIA CUDA GPU**。注意：QLoRA并非技术上只能用于NVIDIA，但其他后端请以bitsandbytes官方说明为准，本教程命令均按NVIDIA CUDA给出。

**显存参考**（7B级别模型的**经验值**，非硬性保证）：

- **参考下限**：约8~12GB（短序列、小batch）
- **推荐**：16GB+

### 别把数值当死门槛

QLoRA通过4-bit量化基础模型来降低显存占用，通常低于同规模LoRA的全精度加载。但实际占用受以下条件影响，条件不同结论就不同：

- **模型规模**：7B级别；13B/70B显存需求显著上升
- **序列长度**：短序列（≤512）；越长激活值显存增长越明显
- **batch大小**：小batch（1~2）；越大占用越高
- **梯度检查点**：开启可用时间换显存，降低峰值占用
- **量化配置**：4-bit（NF4）、是否双重量化、`bnb_4bit_compute_dtype`等
- **LoRA目标模块数量/rank**：可训练参数越多，优化器状态与激活越多

把上表当**经验起点**，以你实际运行情况为准。

### 为什么QLoRA还是需要显存？

因为运行仍需加载：

- 量化后的基础模型
- LoRA参数
- 激活值
- batch数据

### 本块学习检查

对照你的实际环境，逐项确认：

1. 你的GPU是否为NVIDIA CUDA GPU？显存多大？

## 块 6

## 5.1 先确认你的Python版本

在动手装QLoRA之前，先花10秒确认Python版本。打开终端，输入：

```bash
python --version
```

**目标版本：Python 3.10+**

如果低于3.10，建议先升级Python再继续，否则后面装PEFT和bitsandbytes容易踩坑。

---

## 5.2 PyTorch检查（关键一步）

**为什么重要：** LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的CUDA量化路径要求安装**与本机CUDA匹配的GPU版PyTorch**。

**常见坑：** 直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

**正确做法：** 去[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)选对应CUDA版本。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证是否装对：**

```python
import torch

print(torch.__version__)

print(
    torch.cuda.is_available()
)

print(torch.version.cuda)   # 查看 torch 编译使用的 CUDA 版本
```

**正常输出：** `True`，表示PyTorch能看到CUDA GPU。

> **注意：** `torch.cuda.is_available()`返回`True`只说明PyTorch侧能用GPU，**不足以证明bitsandbytes的4-bit量化路径可用**。后者还依赖bitsandbytes是否正确编译/匹配到本机CUDA。能否真跑4-bit，以第八章、第十三章对`bitsandbytes`的检查为准。

---

## 5.3 查看GPU信息

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

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是Hugging Face官方参数高效微调库，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter。

### 6.2 安装PEFT

```bash
pip install peft
```

**验证：**

```python
import peft

print(peft.__version__)
```

### 6.3 加载模型（含显存与异常处理）

> **模型许可证：** 本教程涉及的Llama 2模型采用Llama

## 块 7

### 7.1 bitsandbytes作用（动手前必读）

你要做QLoRA微调，**bitsandbytes是绕不开的依赖**。它负责两件核心事：

1. **量化**：把模型压到8bit或4bit，大幅降低显存占用。
2. **支撑QLoRA训练**：本教程的QLoRA方案基于它实现4-bit量化。

**注意**：这里说“必须安装”，是针对本教程选用的`bitsandbytes` + CUDA路线。QLoRA思想本身不绑定特定库，但既然你目标明确，就按这条路径走。

---

### 7.2 安装（一步到位）

```bash
pip install bitsandbytes
```

装完先别急着跑，下一步做CUDA兼容检查，确保你的GPU能正常调用。

---

**本块学习

## 块 8

**动手前先看这里：为什么必须查CUDA？**

你要用QLoRA做微调，核心依赖是bitsandbytes，它靠CUDA驱动GPU计算。版本不匹配，后果很直接：CUDA不可用、量化失败、加载模型报错。所以，先花1分钟排查环境。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**第一步：查CUDA版本**

终端运行：

```bash
nvidia-smi
```

看输出里的`CUDA Version`，记下这个数字。

**第二步：查bitsandbytes是否兼容**

运行：

```bash

## 块 9

### 9.1 推荐使用WSL2

**推荐路径**：Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes

**为什么推荐**：Linux环境兼容性更好。如果你目标是QLoRA微调，建议直接走这条路，省去后续环境折腾。

---

### 9.2 Windows原生可能问题

**问题1**：bitsandbytes无法加载CUDA动态库。

**可能原因**（不止一种）：
- 所装`bitsandbytes`版本与本机CUDA / GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本的支持不如Linux完善（不同版本差异较大）。

**关键理解**：这不是单一的"Windows不支持"问题，而是受**PyTorch、bitsandbytes版本与本机CUDA/硬件**共同影响。

**行动建议**：

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配

在Windows上配置QLoRA环境时，最常见的坑就是**CUDA版本不匹配**。这个问题会导致你后续安装的PyTorch无法调用GPU，进而让整个QLoRA微调流程卡死。

**排查方法**：打开终端，运行以下命令：

```bash
nvidia-smi
```

这个命令会显示三样关键信息：

- **NVIDIA驱动**：确认驱动已正确安装
- **CUDA版本**：显示你的驱动支持的CUDA版本号
- **PyTorch版本**：需另行通过 `python -c "import torch; print(torch.__version__)"` 查看

**核心原则**：你安装的PyTorch必须与你机器的CUDA版本匹配，否则`import torch`或QLoRA的4-bit量化会直接失败。

---

## 十、安装完整LoRA环境

假设你已有基础环境（`transformers`、`datasets`、`accelerate`），现在需要安装LoRA相关的两个核心库：

```bash
pip install peft bitsandbytes
```

**重要前置**：确保你已安装**匹配本机CUDA的GPU版PyTorch**。以CUDA 12.1为例：

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

如果想一步到位，直接运行完整安装命令：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

安装完成后，需要初始化Accelerate配置：

```bash
accelerate config
```

根据你的环境回答几个问题。以单GPU为例：

```text
Compute environment:
This machine

Distributed:
No distributed training

Mixed precision:
fp16
```

根据你的 accelerate 版本，选项可能略有不同。这会生成一个`default_config.yaml`配置文件，Accelerate会用它来管理训练时的资源分配。

---

## 十二、LoRA环境验证实验

创建一个测试文件`test_lora_env.py`：

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

print("CUDA:", torch.cuda.is_available())

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, config)

print(config)
```

运行：

```bash
python test_lora_env.py
```

**成功标准**：如果输出显示`CUDA: True`且能打印出LoRA配置信息，说明：

- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA

## 块 11

## 九、Windows用户特别注意事项（新增）——9.1 推荐使用WSL2

**问题2：Windows报错怎么办？**

在Windows上运行bitsandbytes或QLoRA相关代码时，如果遇到兼容性报错，**推荐使用WSL2**（Windows Subsystem for Linux 2）来解决。

**为什么？**

Windows原生环境对bitsandbytes等库的支持不够完善，容易出现兼容性问题。WSL2提供了一个更接近Linux的环境，能有效规避这些报错。

**排查思路速查表**

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch安装的是CPU版本 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或使用更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

**实操建议（动手验证）**

1. 如果你在Windows上遇到报错，先对照上表判断原因。
2. 若是兼容性问题，安装并切换到WSL2环境，重新运行之前的安装步骤。
3. 记录你遇到的报错信息和解决方式，方便后续排查。

## 块 12

## 一、为什么需要LoRA？——先动手感受一下问题

你现在的目标是 **QLoRA微调**，但在碰代码之前，先花2分钟做一个直观实验，感受“全参数微调”为什么走不通。

**动手验证（约1分钟）**：打开Python，运行下面这段代码，看看一个7B模型全参数微调时，显存大概要多少：

```python
# 7B模型 = 70亿参数
# 每个参数用float32（4字节）存储
model_params = 7e9
model_memory = model_params * 4 / (1024**3)  # 转成GB
print(f"模型本身: {model_memory:.1f} GB")

# 训练时还需要：梯度 + 优化器状态（Adam需要梯度、一阶动量和二阶动量，共3份模型参数大小的内存）
gradient_memory = model_memory
optimizer_memory = model_memory * 3  # 梯度 + Adam的一阶动量和二阶动量
total = model_memory + gradient_memory + optimizer_memory
print(f"全参数微调总显存需求: {total:.1f} GB")
```

你会看到：**光一个7B模型，全参数微调就需要超过100GB显存**。这就是大模型微调的第一个大问题。

---

### 1.1 大模型微调的问题

现代大语言模型（如 **LLaMA、Qwen、Baichuan、Mistral**）通常拥有**数十亿到上百亿参数**。以7B模型为例，就是**70亿参数**。

如果采用传统方式微调：

```text
预训练模型 → 更新全部参数 → 保存新模型
```

你需要：

- **巨大的GPU显存**：不仅存模型，还要存梯度和优化器状态
- **大量训练时间**：每个参数都要计算和更新

## 块 13

**本块核心：LoRA 不碰原模型，只加一个小补丁。**

传统微调是直接改原权重 `W`，训练完变成 `W'`。LoRA 的思路完全不同：**把 `W` 冻住，不动它**，另外新增一个参数 `ΔW`，最终模型用的是 `W' = W + ΔW`。

- `W`：原模型参数（冻结）
- `ΔW`：LoRA 新增参数（只训练这个）

**为什么只训练 ΔW 就够了？** 大模型已经通过海量数据学会了语言规律、世界知识、推理能力，微调通常只是对已有能力做方向调整，不必重新学全部参数，只学任务相关的变化量往往就能取得不错效果。

> 注意：这是教学直觉/假设，不是严格结论。能否用 LoRA 替代全参数微调，取决于任务与预训练能力的差距、数据规模、目标模块与秩的设置等；差距大的任务或数据充足时，全参数微调仍可能更好。

**动手验证（hands-on）：**

## 块 14

**本块学习检查**：  
先别看下文，自己算一遍：如果 `Wq` 是 `4096 × 4096`，它的参数数量是多少？算完再往下读，对照检查。

---

### 3.1 矩阵基础概念

Transformer 里大量参数都是矩阵。以注意力层为例（数学示意，非代码）：

```text
Q = X · Wq
```

这里 `Wq` 就是一个权重矩阵。假设它的形状是：

```text
4096 × 4096
```

那么它的参数数量就是：

```text
4096 × 4096 ≈ 1677万
```

---

### 3.2 直接训练矩阵的问题

如果我们要微调，就需要修改 `ΔW`。如果直接学完整的 `ΔW`，它同样需要 `4096 × 4096` 个参数——这代价太高。

LoRA 的核心思路是：**不直接学习完整矩阵，而是把它拆成两个小矩阵**。原始 `W` 保持冻结，新增的 `ΔW` 由 `B × A` 组成。这样要学的参数就少多了。

## 块 15

**本块学习检查**：动手算一遍——把W设为2048×2048，r=4，写出A和B的尺寸，并算出总参数量，对比完整矩阵参数少了多少倍。

---

### 4.1 矩阵尺寸变化

假设原矩阵W是**2048×2048**。直接训练它，要更新**419万**个参数（2048×2048）。

用LoRA，设**rank r=4**，把W的更新拆成两个小矩阵：

- **A**：4×2048，参数8192
- **B**：2048×4，参数8192

两者相加，LoRA总参数仅**16384**。

| 方式 | 参数数量 |

## 块 16

### 5.1 为什么低秩分解有效？——抓住“关键方向”

你要微调大模型，直觉上会以为要调整上亿个参数，像在4096维空间里每个方向都动一点。但LoRA论文发现一个关键事实：**模型适应新任务时，参数变化（ΔW）并不需要覆盖所有方向**。

真正起作用的“有效变化”，其实集中在**一个低维子空间**里——也就是远少于原维度的少数几个方向。你可以把它想象成：4096维的空间里，真正影响任务成败的“主路”可能只有几条，其余方向都是“支路”，动了也白动。

所以，LoRA用两个低秩矩阵（B和A）相乘来近似这个ΔW。因为有效变化本来就“瘦”，

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张高清图片：**4000×4000像素**，完整保存需要**1600万个数据**。

但如果这张图片的主要变化只是**亮度**、**对比度**和**少量结构变化**，那么用少量参数就能描述它。

**LoRA的核心思想正是如此**——用少量参数描述模型的变化方向，而不是重新训练全部参数。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中Attention包含**Q、K、V、O**四个投影矩阵。

**LoRA的常见做法**：早期实践中通常只加到Q投影和V投影上，例如：

```python
target_modules=["q_proj", "v_proj"]
```

这表示只训练`Wq`和`Wv`的变化。

> ⚠️ **注意**：只挑`q_proj`/`v_proj`是LoRA原论文的**常见选择**，**不是固定标准**。现代实践经常把LoRA加到全部投影（q/k/v/o）和FFN的线性层，甚至用`target_modules="all-linear"`覆盖所有线性层。

**更完整的配置示例**（以Llama类模型为例，不同模型模块名可能不同，请先用`print(model)`或`model.named_modules()`确认实际名称）：

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

> 覆盖模块越多，可训练参数、显存与Adapter体积越大——是否值得，应结合验证集表现权衡。

---

## 七、LoRA训练过程

完整流程：

```text
加载预训练模型 → 冻结所有参数 → 插入LoRA矩阵 → 训练LoRA参数 → 保存Adapter
```

训练时：
- **原模型W**：冻结，不更新
- **LoRA的A和B**：更新

最终模型 = **原模型 + LoRA Adapter**

---

## 八、核心参数速览

| 参数 | 作用 |
|------|------|
| `r` | 低秩矩阵的秩，决定参数量 |
| `lora_alpha` | 缩放因子，影响更新幅度 |
| `lora_dropout` | 防止过拟合 |
| `target_modules` | 指定插入LoRA的模块 |

---

**本块学习检查**：请用`model.named_modules()`查看

## 块 18

## 8.1 rank（r）——从“改多少参数”到“改多准”

你正在为 QLoRA 微调做准备，那 `r` 就是你最先要拍板的超参数之一。先看代码里它长什么样：

```python
r=8
```

**它代表什么？** 低秩矩阵的维度。简单说，`r` 决定了你用多大的“压缩通道”去近似完整的权重更新。

**它越大怎样？**

- **优点**：表达能力更强——模型能学到的变化更丰富；
- **缺点**：参数更多、显存占用增加——训练更重、Adapter 文件更大。

**常见起点参考**（注意：这是**经验起始范围**，不是铁律）：

| 任务 | rank（起点参考） |
| ---- | ---- |
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

但对你这种 hands-on 风格，**别背表，跑实验**。合适的 `r` 和模型规模、目标模块、数据量、任务难度都相关，最终要靠验证集说话。

**基于验证集选 `r

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

你之前已经理解了 `ΔW = BA` 这个核心公式，现在要问的是：**这个更新到底该多大？** `lora_alpha` 就是干这个的。

```python
lora_alpha=16
```

这个参数控制LoRA的**更新强度**。实际生效时，通常用 `alpha/r` 作为缩放比例——也就是说，`alpha` 和秩 `r` 是配合使用的。`alpha` 越大，LoRA对原模型权重的改动越激进；`alpha` 越小，改动越保守。

> **动手理解**：如果你设 `r=8, alpha=16`，缩放比例就是 `16/8=2`。试试把 `alpha` 改成 `32`，比例变 `4`，更新强度翻倍——微调出来的模型行为会更“偏离”原模型。

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

和神经网络里常见的 dropout 一样，作用是在训练时随机丢弃一部分LoRA参数，防止LoRA参数过拟合。`0.05` 表示5%的丢弃率，是常见默认值。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（

## 块 20

## 2.1 Rank表示什么？

Rank，中文叫**矩阵的秩**。在LoRA里，它直接决定了低秩矩阵的“中间维度”大小——也就是你打算用多“窄”的通道去逼近原本巨大的权重更新。

**动手算一遍，比看十遍都清楚：**

假设原始权重矩阵 `W` 的尺寸是 **4096 × 4096**（约1670万参数）。如果设置 `r=8`，那么：

- 矩阵 `A` 的尺寸是 **8 × 4096**
- 矩阵 `B` 的尺寸是 **4096 × 8**

两个矩阵相乘 `ΔW = B × A`，结果恢复成 **4096 × 4096**——和原矩阵形状完全一致。

**关键点来了：** 你实际训练的参数不是1670万个，而是 `8×4096 + 4096×8 = 65536` 个，只有原来的 **0.4%**。

**直觉理解：** 把 `r` 想象成“信息管道的直径”。`r=8` 就是让信息通过8条管道流动，`r=64` 就是64条管道。管道越多，能传递的“细节”越丰富，但你要训练的参数也越多，显存和算力开销随之上升。

**对你当前QLoRA微调目标的直接含义：** 选 `r` 就是在“效果”和“成本”之间做权衡。任务越复杂、领域差异越大，通常需要更大的 `r` 来容纳足够的信息；反之，小 `r` 更省资源，适合简单适配。

**本块学习检查：** 如果原始权重是 `2048 × 2048`，设置 `r=4`，请写出 `A` 和 `B` 的尺寸，以及训练参数量占原参数量的百分比。

## 块 21

**Rank控制什么？——用“自由度”来理解**

Rank决定LoRA能学多少种变化方向，就像给模型配几个“旋钮”。

**低rank（如r=4）**：旋钮少，只能学有限模式。好处是参数少、显存低、训练快；代价是表达能力有限。

**高rank（如r=64）**：旋钮多，自由度大，能学复杂任务、适应力强。但参数多、显存涨、训练成本高。

**动手验证**：在PyTorch中分别设`r=4`和`r=64`，对比某层`4096×4096`权重矩阵的LoRA参数量——低rank版本参数量明显更小，

## 块 22

**本块学习检查**：算一下，若把4096维全量训练换成LoRA r=32，参数量是多少？对比完整训练的1677万，差了多少倍？

---

**3.1 先看“不动手”的代价**

完整训练一个4096×4096的矩阵，参数量是：

`4096 × 4096 = 16,777,216`，约**1677万**。

---

**3.2 再看LoRA的“轻量方案”**

LoRA不碰完整矩阵，只训练两个小矩阵，公式：

`参数量 = r × (输入维度 + 输出维度)`

- **r=8**：`8 × (4096+4096) = 65,

## 块 23

**本块学习检查**：读完本节后，请用一句话回答：`r=2` 时，训练loss下降慢的根本原因是什么？

---

### 4.1 Rank太小的问题

假设你设置 `r=2`，这意味着LoRA只用极少的参数去“记住”新任务的知识。结果就是：

- **学习能力不足**：模型能调整的空间太小，学不动；
- **欠拟合**：训练集上的规律都没抓住，更别说泛化；
- **新任务知识无法充分表达**：想学的东西“装不下”。

**直观表现**：
```text
训练loss下降慢
效果提升有限
```

---

**动手验证（hands-on）**：如果你在跑QLoRA微调，试着把 `r` 从 `8` 改成 `2`，对比前几个epoch的loss曲线。你会发现loss下降明显变缓——这就是“学习能力不足”的直接证据。

---

**核心结论**：`r` 太小，模型“没力气”学；`r` 太大，又会牺牲LoRA的轻量优势。下一节我们看 `r=256` 的另一个极端。

## 块 24

## 4.3 常见rank选择（实操起点）

你是要做QLoRA微调，那rank就是你要敲的第一个关键参数。下面是经验起始范围，**注意只是起点，不是自动调参标准**：

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**动手建议**：先按任务类型选一个起点值跑通流程，再根据实际效果调整。别指望上表直接套用——同一个`r`在不同模型规模、序列长度、batch大小、目标模块下，显存和效果差别可能很大。

## 五、Alpha（lora_alpha）是什么？

LoRA不只是`ΔW = BA`，还会做缩放：

```
W'

## 块 25

### 6.1 控制LoRA影响强度——动手调一调

你正在为QLoRA微调做准备，这里我们用最直观的方式理解`alpha`怎么控制LoRA的“劲儿”。

**核心公式就一个：`alpha / rank`**

假设你设了`rank=8`，然后：

- **`alpha=16`** → 缩放系数 = `16/8 = 2` → LoRA更新量被**放大2倍**，模型改动更猛。
- **`alpha=8`** → 缩放系数 = `8/8 = 1` → 保持原始比例，不增不减。
- **`alpha=4`** → 缩放系数 = `4/8 = 0.5` → LoRA影响**减半**，改动更温和。

**关键认知（别踩坑）：**

`alpha/r` 只是一个**缩放系数**，不是“实际强度”的唯一开关。真正落到权重上的更新量，还取决于：

1. **学习率**：学习率大 + `alpha/r`大 = 更新容易震荡。很多实战里，调学习率比频繁改`alpha`更有效。
2. **初始化**：常见实现中`A`用小随机值、`B`初始化为0，所以训练开始时`ΔW=0`，不扰动原模型。
3. **训练状态**：`ΔW`是逐步累积的。同样的`alpha/r`，训练早期和收敛后的实际更新量完全不同。

所以，把`alpha/r`理解为“可调的旋钮”，调强度时**要和学习率一起看**，最终以验证集表现定。

---

## 块 26

## 七、Alpha与Rank的关系（重点）——配置1

先看两组配置：

```python
# 配置1
r=8
alpha=16

# 配置2
r=16
alpha=32
```

两组配置的 **rank 不同**（8 vs 16），但 `alpha / r` 的**缩放比例相同**，都是 **2**。

**核心要点**：调整 rank 时，常见做法是**同步调整 alpha**，让 `α/r` 比例保持稳定。这只是**经验做法**，不是硬性规则。比如 `alpha = 2 × r`（r=8/alpha=16、r=16/alpha=32）只是常见的**经验起点**，并非通用最优。

## 块 27

**本块学习检查**：用 `r=8, alpha=16` 跑一次小模型微调，观察训练速度与参数量，对比默认配置的差异。

---

### 方案1：小模型实验（动手首选）

配置直接照抄：

```python
r=8
alpha=16
```

**为什么适合你？** 你的目标是QLoRA微调，且偏好hands-on快速验证。这个组合参数最少、训练最快，适合先跑通流程、确认代码无误，再逐步加大规模。

**操作建议**：先用这个配置跑通一个最小数据集，记录训练时间和显存占用，作为后续方案的基准线。

---

### 方案2：普通指令微调（进阶）

```python
r=16
alpha=32
```

当你确认小模型流程没问题，想提升效果时切换。它在效果和成本之间取平衡。

---

### 方案3：复杂任务（备用）

```python
r=64
alpha=128
```

仅当任务确实复杂、且你有足够算力时使用。表达能力强，

## 块 28

### 误区1：rank越大一定越好？

**错误。** 任务简单时，盲目调大rank不仅浪费显存，还会增加训练成本。所以，rank不是越大越好，够用就行。

### 误区2：只调rank，不调alpha

比如你把`r`从8改成64，但`alpha`还是16：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

此时缩放比例变成`16/64=0.25`，LoRA对模型的影响反而被削弱了。**改rank时，记得同步调整alpha**，否则效果可能不升反降。

### 误区3：alpha越大越好

**错误。** alpha过大（即`α/r`缩放过大）会导致更新过强、训练不稳定。但注意：`α/r`过大只是不稳定的**可能原因之一**，不是唯一原因。训练震荡还常来自学习率过高、数据质量差、batch size过小、精度问题或缺少梯度裁剪。排查时请综合检查，别只盯着alpha。

---

### 本块学习检查

**动手验证一下：** 打开你的代码，把`r`从8改成64，`alpha`保持不变，算一下新的`α/r`值。然后回答：这个改动会让LoRA更新变强还是变弱？如果想让更新强度和原来一样，`alpha`应该改成多少？

---

### 电脑配置说明

本章只讲原理和参数策略，**不需要GPU**。任意现代CPU + 4GB内存即可。如果你之后要实际跑LoRA训练，参考经验值：LoRA约8GB+显存，QLoRA因4-bit量化通常更低，7B

## 块 29

**本块学习检查**：阅读完本块后，请用一句话向自己解释“LoraConfig在LoRA训练流程中处于哪一步、负责做什么”，并对照下方流程图口述一遍顺序。

---

## 一、LoraConfig是什么？

### 1.1 PEFT中的LoraConfig作用

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。你正在为 QLoRA 微调打基础，而 LoRA 是 QLoRA 的前置核心，所以先吃透这里。

**LoRA训练流程**（请按顺序记忆）：

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

**LoraConfig 是流程中的“总指挥”**，它负责告诉 PEFT 四件事：

1. **LoRA放在哪里** —— 决定哪些层被插入低秩矩阵；
2. **LoRA规模多大** —— 决定低秩分解的维度（即容量）；
3. **LoRA训练强度如何** —— 决定参数更新的缩放比例；
4. **是否使用dropout** —— 决定训练时是否随机丢弃部分神经元以防过拟合。

---

**动手验证**：打开你的 Python 环境，输入以下代码（仅查看类型，不运行训练）：

```python
from peft import LoraConfig

## 块 30

## 一个简单示例，动手跑起来

先看一段最精简的LoRA配置代码，这是你后续做QLoRA微调的基础骨架：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,                # LoRA矩阵的秩，决定新增参数的大小
    lora_alpha=16,      # 缩放强度，控制LoRA对原模型的影响程度
    target_modules=["q_proj", "v_proj"],  # 在哪些层插入LoRA
    lora_dropout=0.05   # 训练时随机失活比例，防止过拟合
)
```

**四个参数各管一件事：**

| 参数 | 作用 |
|---|---|
| `target_modules` | 决定插入哪些层（这里选了注意力层的q和v投影） |
| `r` | 决定LoRA矩阵大小（秩越大，可学习参数越多） |
| `lora_alpha` | 控制缩放强度（影响微调幅度） |
| `lora_dropout` | 控制随机失活（正则化手段） |

## 配置如何生效：完整流程

光有`LoraConfig`还不够，需要把它和原模型绑定：

```python
from peft import LoraConfig, get_peft_model

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入低秩适配器”**。

Transformer 模型的 Attention 部分，常见结构是这样的：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

**关键点**：LoRA 不会默认修改所有层，你必须显式指定要注入哪些模块。

例如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的含义是：只给 **Query 投影层** 和 **Value 投影层** 增加 LoRA 适配器。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

所以，调整 Q 和 V 的投影层，通常就能有效改变模型的行为——这就是 q_proj 和 v_proj 成为常见选择的原因。

---

### 重要提醒

`q_proj`/`v_proj` 只是**常见起点**，不是所有模型通用的“标准配置”。不同模型的模块命名可能不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 target_modules。

另外，较新版本的 PEFT 支持 `target_modules="all-linear"`，让 PEFT 自动匹配（几乎）所有线性层，省去手工枚举。但覆盖模块变多会增加**可训练参数与显存**，是否使用需要在**参数量/显存**与**效果**之间权衡，以验证集结果为准。

### 可运行示例（PEFT ≥ 0.7 支持）

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
    target_modules="all-linear",   # 自动匹配（几乎）所有 nn.Linear；一般不含输出层 lm_head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

> 注意：若你的 PEFT 版本较旧、不支持 `"all-linear"` 字符串，请**显式列出模块名**，或升级 PEFT。用 `print_trainable_parameters` 确认注入结果。

## 块 32

**方案1：只调整Q和V（轻量起点）**

如果你刚开始做QLoRA微调，想快速跑通流程，推荐先试这个配置：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么选它？** 只调整注意力机制中的查询（Q）和值（V）投影层，参数少、训练快，是常见的轻量起点。具体效果仍需结合模型结构验证，但作为第一步足够。

**对比：调整全部Attention**

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj"
]
```

把键（K）和输出（O）投影也加入，参数更多，表达能力更强，适合追求更高精度的场景。

**进阶：加入FFN层**

部分任务还会把前馈网络（FFN）的投影

## 块 33

## LLaMA系列：target_modules怎么选？

在LLaMA系列模型上做QLoRA微调，`target_modules`最常见的配置是：

```python
["q_proj", "v_proj"]
```

**为什么是这两个？** 因为LLaMA的注意力机制中，Q（查询）和V（值）投影是独立的线性层（`q_proj`和`v_proj`）。只对这两个层注入LoRA低秩矩阵，就能以较小参数量捕捉微调所需的核心变换，这也是社区验证过的默认高效方案。

> **动手验证**：在你自己的LLaMA模型上，分别用`["q_proj", "v_proj"]`和`["q_proj", "k_proj", "v_proj", "o_proj"]`配置LoraConfig，调用`model.print_trainable_parameters()`对比可训练参数量差异。你会发现前者参数量更少，但通常已足够下游任务微调。

---

## 对比：GPT-2的target_modules

与LLaMA不同，GPT-2的target_modules通常写为：

```python
["c_attn"]
```

**关键区别在于权重布局**：GPT-2的注意力用的是Transformers里的`Conv1D`（不是普通`nn.Linear`），它把Q/K/V三个投影**合并**在一个`c_attn`模块里。`Conv1D`的权重形状为`(out_features, in_features)`，与`nn.Linear`相同。但在GPT-2中，权重矩阵以转置形式存储，因此需要设置`fan_in_fan_out=True`以正确应用LoRA。

```python
fan_in_fan_out=True
```

PEFT才能按转置布局正确注入低秩矩阵。可运行示例：

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
    fan_in_fan_out=True
)
```

> **注意**：若在本地部署Llama-2-7b模型，# 注意：Llama-2-7b FP16 需要

## 块 34

**BERT的target_modules怎么配？**

先别急着抄LLaMA的配置。BERT的注意力层包含`q_proj`、`k_proj`、`v_proj`、`o_proj`四个投影矩阵，实践中常选择`q_proj`和`v_proj`进行LoRA适配，但也可根据任务需求调整。

**动手验证（hands-on）**：加载你的BERT模型，运行：

```python
for name, _ in model.named_modules():
    print(name)
```

在输出里找`q_proj`和`v_proj`——这就是你要填进`target_modules`的名字。

**匹配机制**：`target_modules`按模块名做**后缀匹配**。写`"q_proj"`会命中所有以`q_proj`结尾的层（各层各头都中）。想更精确，可以传完整模块名或正则。

**两个坑**：
1. **GPT-2**用的是`Conv1D`（非`nn.Linear`），模块名是`c_attn`（q/k/v合在一起），不能照搬`q_proj`/`v_proj`。
2. **版本差异**：不同Transformers/PEFT

## 块 35

## 6.1 r是什么？

r就是**rank（秩）**，它控制LoRA低秩矩阵的维度。

LoRA的核心公式是：

```text
ΔW = (α/r)BA
```

这里的r，决定了A和B的**中间维度**大小。

**举个例子**：假设原权重矩阵是 `4096×4096`，你设置 `r=8`，那么：

- A的形状是 `4096×8`
- B的形状是 `8×4096`

也就是说，你并没有直接训练那个巨大的4096×4096矩阵，而是用两个小得多的矩阵（A和B）的乘积来**近似**它的更新量ΔW。r就是这两个小矩阵的“厚度”。

---

## 6.

## 块 36

### 7.1 alpha作用（动手验证）

LoRA的最终更新公式是：

```text
ΔW = (alpha/r)BA
```

其中B是d×r矩阵，A是r×d矩阵，ΔW是加到原始权重W上的增量。

**动手试一下**：配置 `r=8`、`lora_alpha=16`，缩放因子为 `16/8=2`，这意味着ΔW会被放大2倍，但实际影响还需结合训练过程。

---

### 7.2 alpha常见设置（起点参考）

一个常见的经验起点是：

```text
alpha = 2 × r
```

> 注意：这只是起点，并非固定规则，实际需根据任务调整。你可以固定alpha只调r，

## 块 37

**8.1 dropout作用**

训练时，LoRA会以概率p随机将低秩矩阵中的部分元素置零（即对增量矩阵ΔW施加随机掩码），目的是减少对训练数据的过拟合，提升泛化能力。

配置 `lora_dropout=0.05`，表示低秩矩阵中每个元素有5%的概率在训练时被随机置零。

**8.2 如何选值？**

经验起点（非固定规则，需结合验证集调）：

| 数据规模 | dropout起点 |
|---------|------------|
| 数据少   | 0.05~0.1   |
| 数据多   | 0~0.05     |

若训练数据**非常丰富**，可直接设 `

## 块 38

## 9.1 基础配置：快速上手 LoRA

你正在为 QLoRA 微调打基础，目标模型是 **7B 级别**。这个配置是 LoRA 的“标准起手式”，先跑通，再调优。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量：秩的大小，8是轻量级起点
    lora_alpha=16,          # 更新缩放：控制LoRA对原模型的影响强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的Q和V投影
    lora_dropout=0.05,      # 防止过拟合：训练时随机丢弃5%的LoRA参数
    bias="none",            # 不训练偏置项，保持轻量
    task_type="CAUSAL_LM"   # 任务类型：因果语言模型（如GPT系列）
)
```

**四个核心参数速记：**

| 参数 | 值 | 作用 | 直观理解 |
|------|-----|------|---------|
| `r` | 8 | LoRA容量 | 秩越大，可学习的“新

## 块 39

## 9.1 基础配置：让分类头跟着 LoRA 一起训练

你的目标是 QLoRA 微调，并且喜欢动手实践。这一节我们直接看一个完整可运行的例子，重点解决一个实战中必踩的坑：**新初始化的分类头如果不特殊处理，保存后再加载会“失忆”**。

### 核心代码与逐行解读

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 1) 加载带分类头的基础模型（分类头是新初始化的，需要一起训练/保存）
# 模型许可证：BERT 使用 Apache 2.0 License，商用需保留版权声明
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练需根据 batch size 适当增加
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",                 # 序列分类任务
    target_modules=["query", "value"],   # BERT 注意力模块名
    modules_to_save=["classifier"],       # 关键：把新分类头也纳入训练并保存
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

**动手观察点**：运行 `print_trainable_parameters()`，你会看到可训练参数包含两部分——LoRA 注入的低秩矩阵，以及 `classifier` 的权重。这验证了 `modules_to_save` 确实把分类头“拉进”了训练集合。

### 保存与重载：验证分类头没丢

```python
# 2) 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")

# 3) 重新加载并验证分类头确实被带回（而不是随机初始化）
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证：重载后的分类头权重应与保存前一致
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

**为什么必须加 `modules_to_save=["classifier"]`？**  
因为 `save_pretrained` 只保存 Adapter 权重和 `modules_to_save` 指定的模块。如果不加这一行，重载时分类头会退回基础模型的**随机初始化**状态——你训练好的分类能力全部丢失，结果自然错乱。

**验证手段**：用 `torch.allclose` 对比保存前/重载后的分类头权重。实际项目中，你也可以用一批固定输入对比 logits 是否一致，效果相同。

### 三个必须记住的要点

1. **保存范围**：`save_pretrained` 保存的是 **Adapter + `modules_to_save` 模块**，不是完整基础模型。重载时需先加载**同一个**基础模型（这里是 `bert-base-uncased`），再用 `PeftModel.from_pretrained` 挂载。
2. **属性名可能不同**：不同版本或模型结构中，分类头的属性名可能不同（如 `classifier` 或 `score`），需根据实际模型确认。
3. **LoRA 原理**：LoRA 冻结原始权重 W，训练低秩矩阵 A 和 B，增量 ΔW=(α/r)BA，而非分解原始权重。

## 块 40

## 10.1 普通文本生成：LoRA参数配置

既然你的目标是QLoRA微调，我们先从最基础的普通文本生成场景入手，把参数手感建立起来。

### 推荐配置

```python
r=8
alpha=16
dropout=0.05
```

### 参数怎么理解？

- **`r=8`**：这是LoRA的低秩维度。可以把它理解为“用多细的笔触去微调模型”——`r`越小，新增的可训练参数越少，显存占用越低，但表达能力也越弱。普通文本生成任务对领域专精要求不高，`r=8`是个稳妥的起点。
- **`alpha=16`**：这是缩放因子（`lora_alpha`），用于控制LoRA更新的强度。实际更新时，增量ΔW = (`lora_alpha` / r) * BA，其中`lora_alpha`/r是缩放系数，即 `16 / 8 = 2`。
- **`dropout=0.05`**：防止过拟合。5%的随机丢弃率，在数据量适中时既能起到正则化作用，又不会明显拖慢收敛。

### 和QLoRA的关系

QLoRA的核心区别在于：基础模型被4-bit量化后再挂LoRA。也就是说，**上面这些LoRA参数

## 块 41

**目标：QLoRA微调，动手实操**

先看这份核心配置，它就是普通文本生成任务的LoRA起点：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**逐个拆解，按你的学习节奏来：**

- **`target_modules`（改哪里）**：指定要注入LoRA的层。这里选`q_proj`和`v_proj`，即注意力机制中的查询和值投影。这是最常用的选择，改动小、效果好。
- **`r`（容量）**：设为8，代表LoRA矩阵的秩。数值越大

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思路：冻结原模型，只训练少量低秩矩阵。但这里有个关键点——**LoRA只是减少了“训练”的参数，并没有减少“加载”的参数**。

用图来表示：

```text
原模型参数：冻结（仍然占显存）
LoRA参数：训练（额外占显存）
```

**基础模型本身，依然要完整地放进显存里。**

我们来算一笔账。假设你要微调一个7B（70亿参数）模型，用FP16精度存储：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是**模型权重**的占用。实际训练时，显存里还要同时放：

- **激活值**：前向传播的中间结果；
- **梯度**：反向传播计算出的更新量；
- **优化器状态**：如Adam的动量项。

把这些都加起来，实际显存需求会**远高于14GB**。这就是为什么即便有了LoRA，很多消费级显卡依然跑不动7B模型的微调——瓶颈不在训练

## 块 43

### 1.2 QLoRA核心思想

**一句话理解**：QLoRA = 把大模型“压缩”到4-bit，再在上面挂LoRA小翅膀训练。

**动手对比一下**（你偏实操，直接看差异）：

| 方法 | 基础模型 | LoRA参数 |
|------|----------|----------|
| LoRA | FP16（全精度） | FP16，训练 |
| QLoRA | **4-bit（量化）** | FP16，训练 |

**关键流程**（按顺序走一遍）：
1. 把FP16基础模型 **4-bit量化**（压缩权重）
2. 加载这个量化后的模型
3. 插入LoRA适配器
4. 只训练LoRA

## 块 44

**3.1 模型存储方式：从14GB到3.5GB**

模型参数占用的显存，直接由“每个参数需要多少字节”决定。看这张表：

| 类型 | 每个参数占用 |
|------|------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

以7B模型为例，用FP16存储需要 `7B × 2 ≈ 14GB`；换成4-bit后只需 `7B × 0.5 ≈ 3.5GB`，显存直接减少约75%。

**注意**：这里的3.5GB只是**理论下界**——只算了量化后的权重本身。实际运行中，你还要加上量化常数/缩放因子、分组元数据（NF4按组量化，每组都有额外参数）、未量化的模块（如norm、embedding、lm_head、LoRA参数），以及

## 块 45

**NF4是什么？**

NF4（NormalFloat4）是 QLoRA 论文提出的一种 4-bit 量化数据类型，专门针对神经网络权重的分布特点设计。

**关键理解：它不是通用浮点格式**

别被名字误导——NF4 不是 IEEE 754 那种能表示任意实数的 4 位浮点格式。它本质上是**一套离散码本（codebook）**，为“近似正态分布”的权重量身定制。

**它怎么工作？**

1. 预先确定 16 个（2⁴）量化取值点，这些点根据标准正态分布的分位数（quantiles）确定，而非等间隔排布
2. 权重密集的 0 附近划分更细，精度更高

## 块 46

**为什么NF4适合模型权重？**

大模型训练完的权重有个特点：**近似正态分布**——大量参数挤在0附近，越往两边越少。

普通INT4是**平均划分**的，比如[-8,-7,...,7]，每个区间一样宽。但权重都堆在中心，平均划分根本没法精细表示0附近的密集区域，精度浪费在很少用到的两端。

NF4的思路是：**按正态分布设计量化范围**。0附近分得更细、精度更高，两端分得粗一些。这样量化误差更小，更贴合真实权重分布。

**NF4量化流程（一句话版）**：
FP16权重 → 归一化 → 映射到NF4 → 存4-bit值；推理时再反量化回近似FP

## 块 47

**5.1 为什么需要双重量化？**

4-bit量化不只是把权重压成4-bit，还得为**每组权重**额外保存两个量化参数：`scale`和`zero point`。这些参数同样占显存，模型越大，这笔开销越不可忽视。

**5.2 双重量化思想**

普通量化：FP16权重 → 4-bit权重 + scale参数。

双重量化：在此基础上，**再对scale参数做一次量化**。第一次量化模型权重，第二次量化量化参数本身，从而减少scale的存储开销。

**5.3 双重量化优势**

核心收益是减少额外量化参数和显存占用。大模型参数海量，哪怕每层只省一点，累计下来也能释放大量显存，为更大模型或更长序列腾出空间。

---

**

## 块 48

**本块学习检查**：读完 6.1 和 6.2 后，请合上资料，用自己的话回答：QLoRA 已经降低了模型显存，为什么训练时还可能 OOM？分页优化器是在什么“瞬间”出手救场的？

---

**6.1 为什么需要分页优化器？**

训练时显存峰值不只来自模型，还有**梯度**和**优化器状态**。比如 Adam 优化器就要额外保存**一阶动量**和**二阶动量**。QLoRA 虽然大幅降低了模型本身的显存占用，但训练过程中仍可能遇到**显存突然不足**的情况。

**6.2 分页优化器思想**

Paged Optimizer 借鉴了操作系统的**虚拟内存**思路。核心机制是：借助 NVIDIA 的 **Unified Memory**，当显存出现峰值、即将 OOM 时，自动把优化器状态**按页换出到 CPU 内存**，需要参与计算时再**换回 GPU**。

**关键理解**：别把它当成“平时就把优化器状态常驻 CPU”。它的重点是**应对显存峰值**——在显存不足的瞬间临时换

## 块 49

## 6.3 类比理解：把显存当成你的桌面

**普通优化器**的做法是：所有东西都堆在桌面上（显存里）。桌面满了，就什么都放不下了——这就是**峰值OOM（显存溢出）**。

**分页优化器**的做法更像真实办公：
- 桌面上只放**常用文件**（当前活跃的优化器状态）
- **暂不用的文件**自动存进柜子（CPU内存）
- 需要时再取回桌面

你不需要手动管理"换页"，bitsandbytes 库在后台自动完成。这套机制专门用来**缓解峰值OOM**——不是减少总用量，而是把高峰期的压力"借"给内存。

---

## 6.4 实际启用：只需改一个参数

在 `TrainingArguments` 里设置 `optim` 即可，不需要写任何换页逻辑：

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

**两个常用取值：**
- `paged_adamw_8bit`：优化器状态用8-bit存储，更省显存
- `paged_adamw_32bit`：精度更高，但占用更多显存

**注意：** 不同版本的 Transformers / bitsandbytes 支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报"未知 optim"错误，查一下你当前版本的文档，换用支持的名称即可。

**实战组合拳**（QLoRA标配）：
- `optim="paged_adamw_8bit"` 缓解峰值OOM
- `gradient_checkpointing=True` 降低激活值显存
- 小 `per_device_train_batch_size` + `gradient_accumulation_steps`

## 块 50

## 7.1 4-bit量化：把模型“瘦身”到四分之一

**核心作用**：减少模型权重大小。

想象一下，原本每个权重用FP16（16位）存储，现在换成NF4（4位）存储，存储空间直接缩到原来的1/4。这就是QLoRA能让你在消费级GPU上跑大模型的第一个关键手段。

```text
FP16（16位） → NF4（4位）
```

**动手理解**：你可以把模型权重想象成一本厚厚的书，FP16是每页写满16行字，NF4是每页只写4行——书变薄了，但内容（精度）也打了折扣。不过NF4并非简单截断，而是基于分位数的量化格式，能在4位下保留更多关键信息。QLoRA的聪明之处在于：**只量化基础模型，不量化LoRA参数**，所以训练时微调的部分依然保持高精度。

---

## 7.2 双重量化：连“尺子”也压缩

**核心作用**：减少量化参数占用。

4-bit量化不是随便砍掉位数就完事，它需要一个“缩放因子”（scale）来

## 块 51

## load_in_4bit 与 QLoRA 核心配置

你已经知道 QLoRA 的目标是用更少的显存微调大模型，这一块我们直接动手配置。核心就四个参数，逐个过。

### 1. load_in_4bit：开启 4-bit 加载

这是总开关，设为 `True` 后，模型权重会以 4-bit 精度加载进显存，而不是默认的 32-bit。这是显存节省的第一步。

### 2. bnb_4bit_quant_type：量化方式选 NF4

设为 `"nf4"`。NF4 是 bitsandbytes 提供的一种 4-bit 量化格式，专门为适配大模型权重分布而设计，精度损失相对更小。

### 3. bnb_4bit_use_double_quant：开启双重量化

设为 `True`。双重量化会对量化过程中产生的额外参数（如缩放因子）再做一次量化，进一步压缩显存占用。简单理解：**先量化权重，再量化“量化的参数”**，省上加省。

### 4. bnb_4bit_compute_dtype：计算精度，最容易踩坑

这里要区分两个概念：

- **存储 dtype**：权重以 4-bit（NF4）存在显存里，省显存。
- **计算 dtype**：实际做矩阵运算时，4-bit 权重会被**反量化**成更高精度来算，这个精度就是 `bnb_4bit_compute_dtype` 控制的。

**不设置时，默认计算精度可能为 float32，导致计算速度较慢且显存占用较高，因此建议显式设置 compute_dtype 为 bfloat16 或 float16。**

**关键点：不要写死 `bfloat16`**。BF16 需要较新的 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系）不支持，强行用会报错。正确做法是自动判断：

```python
import torch
from transformers import BitsAndBytesConfig

# 根据 GPU 能力选择：支持 BF16 就用，否则退回 FP16
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
else:
    compute_dtype = torch.float16

bnb

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面几节我们分别看了 `BitsAndBytesConfig`（怎么用 4-bit 加载模型）、分页优化器（`optim`）和 `bnb_4bit_compute_dtype`（计算精度）。你可能已经理解了每一块是干什么的，但最困惑的往往是：**这些零件拼在一起，到底怎么组成一个能跑的 QLoRA 训练脚本？**

这一节直接给你一个**最小可运行骨架**。它把前面所有概念串成一条完整的流水线，你只需要替换模型名和数据就能跑起来。我们按代码顺序拆解每一步的职责。

### 第一步：配置 4-bit 量化（决定模型怎么被压缩）

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

这里做了三个决定：
- **`load_in_4bit=True`**：告诉 transformers 用 4-bit 精度加载模型，这是 QLoRA 的“Q”。
- **`bnb_4bit_quant_type="nf4"`**：使用 NF4（4-bit NormalFloat）量化类型，这是专门为权重分布设计的量化格式。
- **`bnb_4bit_use_double_quant=True`**：开启双重量化，进一步压缩量化常数本身的开销。
- **`bnb_4bit_compute_dtype`**：计算时用的精度。优先用 bfloat16（如果 GPU 支持），否则退回 float16。这决定了实际矩阵运算的精度，和存储精度（4-bit）是两回事。

### 第二步：以 4-bit 加载基础模型

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

注意两点：
- 加载时传入 `quantization_config=bnb_config`，模型才会按 4-bit 方式加载。
- 如果 tokenizer 没有 pad_token，用 eos_token 顶上——这是训练时 padding 的常见坑。
- `device_map="auto"` 让模型自动分配到可用设备。

### 第三步：为 k-bit 训练做准备（关键衔接）

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步**不能省**。它做了两件重要的事：
1. 把某些层的精度调整好（比如 LayerNorm 和输出层保持较高精度），避免 4-bit 量化影响这些敏感部分。
2. 开启梯度检查点所需的输入梯度设置——为后面第 6 步的 `gradient_checkpointing=True` 铺路。

### 第四步：注入 LoRA（只训练低秩增量）

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

核心思想：**冻结原模型，只训练新增的低秩矩阵**。`r=8` 是低秩的维度，`lora_alpha=16` 是缩放系数，`target_modules` 指定注入到哪些模块（这里选了 q_proj 和 v_proj，按你的模型结构调整，也可以用 `"all-linear"` 省事）。`print_trainable_parameters()` 会打印可训练参数的数量，帮你确认 LoRA 注入成功。

### 第五步：准备极简数据

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

Causal LM 的 labels 直接复制 input_ids（自回归任务，预测下一个 token）。`mlm=False` 表示不是掩码语言模型，而是标准因果语言建模。

### 第六步：训练参数——分页优化器 + 梯度检查点

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
)
trainer.train()
```

## 块 53

## 九、把 QLoRA 配置串起来：一个可跑通的最小骨架

你的目标是QLoRA微调，而且喜欢动手实践——那这一节就是为你准备的。前面我们拆解了4-bit量化、NF4、双重量化和分页优化器，现在把它们**串成一条完整的链路**。记住：下面这5步，**缺一不可**。

### 9.1 五步串联：从量化配置到训练启动

**第1步：`BitsAndBytesConfig` —— 决定模型怎么被加载**

这是入口。它告诉系统：用NF4做4-bit量化、开启双重量化、指定计算精度（compute dtype）。这一步决定了基础模型以什么形态进入显存。

**第2步：`prepare_model_for_kbit_training()` —— 最容易跳过、也最容易出事的一步**

这是QLoRA的**必要步骤**。它会对量化后的模型做适配：冻结所有参数、把非量化层转成计算精度、启用输入梯度等。梯度检查点需在`TrainingArguments`中单独设置。**跳过它，常见的后果是训练不收敛或直接报错**。所以，别省。

**第3步：`get_peft_model(..., LoraConfig)` —— 注入可训练的LoRA**

模型已经以4-bit冻结状态加载好了，这一步把LoRA适配器“插”进去。主干是冻结的，只有LoRA参数是可训练的。

**第4步：梯度检查点设置**

梯度检查点需在`TrainingArguments`中设置`gradient_checkpointing=True`。`prepare_model_for_kbit_training`会自动启用梯度检查点，无需额外设置。

**第5步：`optim="paged_adamw_8bit"` —— 分页的8位优化器**

这个设置启用分页的8位AdamW优化器，优化器状态使用分页内存管理，当显存不够时会被临时换出到CPU内存，降低峰值显存压力。

> **注意**：上面是一个“能说明关系”的最小骨架。真实训练中，你需要换成足够规模的数据、合理的超参，并且**按你实际使用的库版本**核对`optim`名称、`prepare_model_for_kbit_training`的参数写法。

---

## 十、你的电脑能跑吗？——显存配置参考

如果你**只学原理**：不需要GPU，纯CPU就能理解上述逻辑。

如果你**要实际跑QLoRA**，最低配置如下：

- **CPU**：4核+
- **内存**：16GB+
- **GPU**：NVIDIA CUDA GPU

**显存参考**（以7B模型、NF4 4-bit + 双重量化、BF16/FP16计算精度、短序列≤512、小batch=1+梯度累积、开启梯度检查点、LoRA只

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

先明确一个关键点：**LoRA训练过程中，基础模型是被冻结的**，只有Adapter（LoRA参数）在训练。

```
基础模型（冻结）  +  LoRA Adapter（训练）  →  最终结构
```

以Llama-7B为例，训练完成后，你保存的**不是完整模型**，通常只有两个文件：

- `adapter_model.safetensors`（Adapter权重）
- `adapter_config.json`（Adapter配置）

### 1.2 Adapter有什么特点？

**优点：**

- 文件小，保存快
- 一个基础模型可以搭配多个任务Adapter

比如同一个基础模型，可以分别搭配：

```
客服Adapter  /  医疗Adapter  /  法律Adapter
```

**缺点：**

部署时必须同时存在：

```
基础模型 + Adapter
```

两者缺一不可。

---

## 二、为什么需要合并LoRA权重？

结合你的QLoRA微调目标，这一步很关键：合并就是把Adapter的增量权重写回基础模型，得到一个**独立的完整模型**，之后部署推理就不需要再额外加载Adapter了。

---

## 本块学习检查

**动手验证一下你的理解：**

打开你训练输出目录，确认是否只包含`adapter_model.safetensors`和`adapter_config.json`这两个文件，而不是完整的模型文件。如果看到了这两个文件，说明你的训练产物就是Adapter形式——这正是需要

## 块 55

**本块学习检查**：读完下面内容后，请用自己的话解释“为什么合并后推理时只需加载一个完整模型”。

---

**2.1 推理时的问题**

未合并时，模型推理需要同时加载两部分：**基础模型**和**LoRA适配器**，流程是“输入 → 基础模型 + LoRA适配器 → 输出”。

但对部署来说，更希望只有一个**完整模型**：输入 → 完整模型 → 输出。所以需要把权重合并。

---

**2.2 合并的数学原理**

LoRA训练时，实际权重变化是：

`W' = W + ΔW`

其中 `ΔW = BA`（B和A是LoRA训练出的两个小矩阵）。

**合并**就是把 Lo

## 块 56

## 3.1 merge_and_unload() 是做什么的？

你在做QLoRA微调时，训练完的模型其实是一个**套着壳子的模型**——PEFT在基础模型外面包了一层`PeftModel`，里面既有`Base Model`，又挂着`LoRA Adapter`。这个状态不能直接拿去部署或推理，因为推理时每次都要额外计算LoRA的增量。

`merge_and_unload()`就是用来**拆壳**的：

```python
merge_and_unload()
```

它的作用一句话概括：**把LoRA权重合并到基础模型里，然后移除PEFT包装**。

用图来表示就是：

```text
Base Model
+
LoRA Adapter
↓
完整模型
```

执行完之后，你拿到的不再是`PeftModel`，而是一个**普通的Transformer模型**，它的权重已经包含了LoRA更新后的结果。

---

## 3.2 合并前后对比

**合并前**，模型结构是这样的：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后**，变成了：

```text
普通Transformer模型
包含LoRA更新后的权重
```

也就是说，LoRA的增量已经写进了原始权重（`W`）里，变成新的权重（`W'`）：

```text
W + (α/r)BA → W'
```

其中，α为缩放系数，r为秩，B为d×r矩阵，A为r×d矩阵。

---

## 四、完整流程：从训练到导出

整个合并导出流程如下：

```text
训练LoRA
↓
保存Adapter
↓
加载基础模型
↓
加载Adapter
↓
merge_and_unload()
↓
保存完整模型
```

下面我们一步步走。

---

## 五、加载LoRA Adapter

假设你已经有训练好的Adapter和基础模型路径：

```python
base_model_path = "model_path"
adapter_path = "adapter_path"
```

先加载基础模型，再挂上Adapter：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构为：

```text
Base Model
+
LoRA Adapter
```

---

## 六、执行合并

一行代码完成合并：

```python
merged_model = model.merge_and_unload()
```

此时`merged_model`**不再是`PeftModel`**，而是普通的Transformer模型，LoRA的增量已经融进权重里了。

---

## 七、导出合并后的模型

把合并好的模型和分词器保存到文件夹：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后目录大致长这样：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：这是**示意结构**，实际文件会随模型/分词器类型变化。比如可能还有`generation_config

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 57

## 3.1 merge_and_unload 是什么？

一句话理解：**将 LoRA 训练得到的增量权重合并到基础模型权重中，生成一个独立的完整模型**。

### 为什么需要它？

你训练时用的是 `Base + Adapter` 的组合体（基础模型上挂一个轻量 LoRA 小插件）。合并就是把这个小插件**永久写入**基础模型的权重里，生成一个全新的、完整的模型文件。

### 合并后，加载方式彻底简化

**合并前**（不合并）加载推理，必须依赖 `peft` 库来加载适配器。

**合并后**，不再需要 `peft`，直接用 HuggingFace 标准接口加载：

```python
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

推理流程也变成一条直线：

```text
用户输入 → 完整模型 → 生成结果
```

### 合并 vs 不合并：一张表看懂

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

### 什么时候该合并？

这是**权衡**，不是非黑即白。问自己三个问题：

1. **需要多个 Adapter 动态切换吗？** 需要 → 不合并；固定一个任务 → 合并。
2. **部署框架原生支持 LoRA 吗？** 不支持 → 合并；支持 → 可不合并。
3. **对存储/分发体积和精度（尤其 QLoRA）敏感吗？** 敏感 → 谨慎合并。

### 合并后必查清单（避免“能加载但行为不对”）

导出后务必核查以下四项：

- **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中

## 块 58

## 10.1 推荐合并场景

既然你的目标是QLoRA微调，那合并这个动作就像“打包发货”——不是所有情况都要打包，但下面两种场景，打包最省事。

**场景1：上线部署（单一Adapter、追求简单）**

想象生产环境链路：服务器 → 直接加载模型 → 提供API。当你只需要**一个固定Adapter**，且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载）时，合并通常更方便。

> 注意：如果推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或你需要在同一基础模型上**热切换多个Adapter**，那么不合并、以Adapter形式部署反而更灵活。是否合并要看框架能力与业务需求。

> # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
> # 注意：Llama-2-7b FP16 需要至少14GB显存。

**场景2：分享模型**

## 块 59

## 10.2 不建议合并场景

**场景1：多个任务Adapter**

一个基础模型上挂了多个Adapter（A、B、C），分别对应不同任务。**保留Adapter更灵活**——想切换任务时直接换Adapter即可，合并后就固定死了。

**场景2：持续训练**

如果你还打算继续调整LoRA，先别合并。合并后权重固化，后续微调就得重来。

> **动手验证**：你正在做QLoRA微调，模型上挂了2个Adapter。请判断：① 你只想用其中一个Adapter推理；② 你计划下周继续训练。哪种情况需要合并？哪种不需要？——答案：①不合并（保留灵活性），②不合并（持续训练）。

## 十一、QLoRA合并注意事项（

## 块 60

### 直接合并的坑与稳妥做法（QLoRA微调必看）

你正在做QLoRA微调，目标是把训练好的LoRA适配器合并回基础模型。这里有个关键点：**别在4-bit量化状态下直接合并**。

#### 直接合并可能遇到的情况

如果你直接在4-bit状态下执行`merge_and_unload`，可能会碰到两个问题：

1. **精度损失**：4-bit权重需要先反量化再合并，这个过程和全程用高精度计算不完全等价，结果可能有细微偏差。
2. **类型不匹配或版本不支持**：某些PEFT/bitsandbytes/Transformers版本不允许在量化状态下直接合并，或者权重类型对不上。

注意：**这不是必然报错**。能否成功取决于你的库版本和量化配置。但为了稳妥，推荐下面这个做法。

#### 常见做法：先反量化到高精度再合并

核心思路：**绕开4-bit状态**，用高精度（FP16/BF16）重新加载基础模型，再挂载同一个Adapter合并。

流程如下：

```
以高精度（FP16/BF16）重新加载基础模型
→ 加载同一份LoRA Adapter
→ merge_and_unload()
→ 保存完整模型
```

这样合并发生在高精度权重上，避开了4-bit带来的类型和精度问题。

#### 可执行代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练QLoRA时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存的LoRA Adapter目录
merged_out = "./merged_model"

# 1) 关键：以高精度重新加载基础模型，不传quantization_config
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=compute_dtype,
    device_map="auto",        # 显存不足时改为 device_map={"": "cpu"}
)

# 2) 挂载训练得到的同一份Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3) 合并；safe_merge=True会做数值检查（如检测NaN）
merged_model = model.merge_and_unload(safe_merge=True)

# 4) 保存完整模型与tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)
```

**动手验证**：按上面代码跑一遍，观察`merge_and_unload`是否成功，并检查`merged_model`的权重类型是否为FP16/BF16。如果遇到报错，记录错误信息并检查你的PEFT/bitsandbytes版本。

**模型许可证**：使用前请确认基础模型的许可证类型（如Llama 2 Community License，商用月活>7亿需Meta授权），并遵守相应条款。

## 块 61

**QLoRA合并前的三个关键提醒**

你正走在QLoRA微调的最后一步——合并权重。这一步有三个“坑”必须提前知道：

**1. 版本决定做法**
不同PEFT版本对“量化模型能否直接合并”支持不同。有的版本要求先反量化，有的可以直接合并。**务必先查你所用版本的`merge_and_unload`文档**。最稳妥的做法是：加载基础模型时**不传**`quantization_config`或`load_in_4bit`，用高精度权重完成合并。

**2. 内存代价**
以高精度重新加载完整模型，会占用约等于一份FP16/BF16完整模型的内存/显存。显存不够时，把`device_map`设为

## 块 62

**本块学习检查**：运行下方最小验证代码，记录 `max_abs_diff` 是否远小于 1（如 <1e-4），并确认两段生成文本完全一致。

---

`safe_merge=True` 是合并时的“保险丝”：它会在合并过程中做数值检查（比如检测 NaN），一旦发现异常立即报错，避免带病模型流入后续步骤。

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并后别急着用——先做一次“前后对照”：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型与tokenizer（以实际路径为准）
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 63

## 合并后别急着收工：给模型做一次“一致性体检”

你刚在 §11.2 里用 `merge_and_unload()` 把 QLoRA 适配器合并进了基础模型。现在的问题是：合并后的模型，和合并前的“基础模型+适配器”组合，行为上还一样吗？

理论上应该几乎一样，但 QLoRA 的 4-bit 量化会引入微小误差，所以**必须实测验证**。下面这段代码就是你的“体检报告”。

### 为什么需要单独加载一份“合并前”的模型？

代码注释里强调了一个关键陷阱：`merge_and_unload()` 是**原地操作**——它会直接修改你手里的 `PeftModel` 对象。一旦执行，原来的对象就不再是“未合并”状态了。

所以你不能拿 §11.2 里那个已经合并过的对象当“合并前”的基准。正确做法是：**重新从磁盘加载一份全新的基础模型，再挂上适配器**，得到 `model_before`，这才是货真价实的“合并前”状态。

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

### 第一项检查：logits 数值差异

把同一个固定 prompt 分别喂给合并前和合并后的模型，比较它们输出的 logits（模型对下一个词的预测分数）：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果？** 理想情况是 0，但 QLoRA 场景下通常是一个较小的近似值。只要这个差值足够小，说明合并没有显著改变模型的预测行为。

### 第二项检查：生成文本是否一致

数值差异小还不够，更直观的验证是看模型实际“说”了什么。用贪心生成（`do_sample=False`）去掉随机性，让两次生成完全可复现：

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

### 为什么需要“再加载一份”？

`merge_and_unload()` 会**原地**修改模型——它把 LoRA Adapter 合进底层权重，并移除 PEFT 包装。这意味着合并后，原来的 `model` 已经变成了“合并版”，不能再拿它当“合并前”的对照。

如果硬要用它对比，两边其实是同一份已合并权重，`max_abs_diff` 会恒为 0，验证就完全失效了。

**正确做法**：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照，再和合并后的模型做对比。

### QLoRA 特有的量化合并风险

QLoRA 合并时需额外注意：合并过程涉及**反量化**（将 4-bit 量化权重还原为更高精度），合并后权重可能**偏离原始量化权重**，带来额外的数值精度损失。建议使用 bitsandbytes 的合并功能，或合并后做数值验证，确认偏差在可接受范围。

### 显存不够怎么办？

如果无法同时放下两个模型，就**分两次跑**：

1. 先用未合并的 `PeftModel` 跑一遍，把 logits 或生成文本存下来，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

### 结果预期：别追求“逐位相同”

- **QLoRA 场景**：合并前后**通常只是近似一致**。

## 块 65

## 十一、合并时开启校验并做一致性验证（建议）

你正在走QLoRA微调这条路，前面已经掌握了`merge_and_unload`的基本用法。这一块是**实战中容易翻车的地方**，尤其当你用的是4-bit量化模型时。核心就一句话：**别在4-bit状态下直接合并，先重载高精度权重再合并，并开启校验。**

### 为什么不能直接在4-bit状态下合并？

QLoRA为了省显存，把基础模型压成了4-bit。但`merge_and_unload`是把LoRA权重**加回**基础模型权重里，这个过程需要高精度计算。在4-bit状态下直接合并，轻则报错，重则静默产生精度损失——模型看起来合并成功了，但输出结果不对。

### 正确做法：先"反量化"再合并

这里的"反量化"**不是**手动逐层还原4-bit权重，而是**重新加载一份不带量化配置的基础模型**：

```python
# 第一步：用高精度加载基础模型（关键：不传quantization_config/load_in_4bit）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16  # 或 torch.float16
)

# 第二步：挂载同一个Adapter
model = PeftModel.from_pretrained(base_model, adapter_path)

# 第三步：合并并卸载
model = model.merge_and_unload(safe_merge=True)
```

这样合并发生在高精度权重上，天然避开了4-bit状态下的类型/精度问题。

### 版本兼容性提醒

不同PEFT / bitsandbytes / Transformers版本对"量化模型能否直接调用`merge_and_unload`"支持不同——部分版本直接报错或警告，部分较新版本有一定支持但仍可能有精度损失。**"先反量化再合并"是跨版本最稳妥的做法**。至于你的版本能否在量化状态下直接合并，以你实际装的`merge_and_unload`文档和运行结果为准。

### 合并后必查清单

合并完别急着走，按下面几项快速排查：

| 问题 | 原因 | 处理 |
|------|------|------|
| `merge_and_unload`不存在 | 当前对象不是`PeftModel`（可能忘了用`PeftModel.from_pretrained`挂载Adapter，或已合并成普通模型） | 确认加载方式，先得到PeftModel再调用 |
| 合并后模型变大 | **正常现象**：合并后是完整模型，不再是Adapter小体积 | 无需处理；在意体积就保留Adapter不合并 |
| 显存不足 | 高精度完整模型占用高；或用了FP32、未offload/分片 | 改用CPU合并或offload、用FP16/BF16、分片加载 |
| Token

## 块 66

## 路线A：合并后部署完整模型

你已经掌握了 `merge_and_unload()` 合并权重的方法，现在来看部署阶段的第一条路：**合并后部署**。

**核心流程**：

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

**这条路的特点**：

- **部署链路简单**——推理框架把合并后的模型当作**普通模型**加载，它完全不知道 LoRA 的存在，不需要任何额外配置；
- **推理时无Adapter切换开销**——因为 Adapter 已经融进权重里了，推理时无需切换Adapter，但模型推理计算量仍取决于模型规模；
- **代价是存储成本高**——每有一个 Adapter（比如针对不同任务微调的版本），都要保存和分发一份**完整模型**。如果你有多个任务，存储和分发成本会线性增长。

**什么时候选这条路？** 如果你的场景是"一个模型服务一个任务"，且部署环境希望尽量简单（比如不想依赖框架对 LoRA 的支持），路线A 是最省心的选择。

---

## 路线B：保留 Adapter，动态服务

另一条路是**不合并**，保留 Adapter 本身，用支持 LoRA 的推理框架（如 **vLLM**）在运行时动态挂载。

**核心流程**：

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

**这条路的特点**：

- **多任务/多租户时只需一份基础模型 + 多个小体积 Adapter**——Adapter 通常只有几十到几百 MB，相比完整模型（几 GB 到几十 GB）小得多，显存和存储都大幅节省；
- **支持热切换**——不用为每个任务各存一份完整模型，请求来了动态切换即可；
- **代价是链路略复杂**——依赖推理框架对 LoRA 的支持，不同框架/版本的支持范围和限制不同，需要额外配置。

**关键认知**：路线B 传入的是**未合并的 Adapter 目录**（即含 `adapter_model.safetensors` / `adapter_config.json` 的目录），**不是**合并后的完整模型。合并后的模型属于路线A，按普通模型直接加载。

---

## vLLM 启动 LoRA 服务（示意）

以 vLLM 为例，启动 OpenAI 兼容服务时通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，命令行大致形如：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
    --max-loras 2 \
    --max

## 块 67

## 14.3 怎么选？两条路线，一个判断标准

你现在面临一个实际选择：微调完的模型，是**合并**成完整模型部署，还是**保留Adapter**动态加载？这没有绝对对错，取决于你的场景。

**路线A：合并部署**——如果你只有一个Adapter，且追求最简部署，直接合并。好处是部署简单，一个模型文件搞定，不用额外加载逻辑。

**路线B：保留Adapter动态服务**——如果你打算用同一个基础模型服务多个Adapter（比如不同任务的LoRA），且推理框架原生支持LoRA加载，那就别合并。这样切换任务只需换Adapter，不用维护多个完整模型副本。

**特别提醒（QLoRA精度敏感）**：无论选哪条路，合并前后一定要做一致性验证（见§11.3）。QLoRA对精度损失更敏感，别省这一步。

> 核心认知：**合并不是唯一部署方式**。具体框架用法留到第15板块（v

## 测评题目

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。
