# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题——为什么你的显卡“扛不住”

先看传统大模型训练的全过程：

```text
输入数据 → 整个模型参数更新 → 保存完整模型
```

以 7B 参数模型为例，70亿参数在训练时，你需要同时保存三样东西：

- **参数**：模型本身的权重
- **梯度**：反向传播计算的更新方向
- **优化器状态**：如 Adam 的动量信息

这三者叠加，显存需求会飙升到参数量的数倍。**普通消费级 GPU 根本装不下**——这就是你学习 LoRA 的直接动机：不是“想不想省”，而是“不省就跑不动”。

---

**动手验证一下**（不引入新知识，只是观察现象）：

打开终端，运行 `nvidia-smi` 查看你的 GPU 显存总量。

## 块 2

### 1.2 参数高效微调（PEFT）

PEFT，即参数高效微调，核心思想是：**不修改整个大模型，只训练少量新增参数**。

**流程**：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter。

**为什么适合你**：相比全参数微调（训练全部参数、显存需求非常高），LoRA只需训练少量参数，显存需求较低；QLoRA则通过量化模型+LoRA，显存需求更低。

**动手验证**：对比三种方式的训练参数规模与显存需求，确认LoRA在资源受限下的优势。

## 块 3

**LoRA是什么？——用“小补丁”微调大模型**

想象原模型权重是`W`，微调时我们不碰它，而是加一个更新量`ΔW`，得到`W + ΔW`。这个`ΔW`由两个低秩小矩阵相乘并缩放得到：`ΔW = (α/r) × B × A`，其中`r`是低秩维度，`α`是缩放因子。

- 若`W`形状为`d_out × d_in`，则`A`约为`r × d_in`，`B`约为`d_out × r`，相乘后形状与`W`一致。
- 不同资料可能写`A × B`，只是符号顺序不同，关键是形状能还原成`W`

## 块 4

## 3.1 LoRA的问题：模型还是太大

你已经知道LoRA能大幅减少**训练参数**，但这里有个关键点：**基础模型本身仍然需要完整加载到显存里**。

以7B模型为例，如果用FP16精度加载，大约需要 **14GB显存**。也就是说，LoRA只是帮你省了训练时的优化器状态和梯度，但模型本体这块“大头”还是占着显存。

> **模型许可证**：本文示例涉及的模型需遵守其对应许可证（如Llama 2 Community License，商用月活>7亿需Meta授权），使用前请自行确认。

---

## 3.2 QLoRA：把模型“压缩”了再训练

QLoRA（Quantized LoRA，量化版LoRA）的思路很直接：

**先将基础模型权重量化为4bit，减少显存占用，再加载LoRA参数进行训练。**

结构上就是：

```text
4bit基础模型 + LoRA参数 → 训练
```

这样你就能在更小的显存上跑更大的模型，同时保留LoRA参数高效微调的优势。

---

## 3.3 依赖bitsandbytes：量化靠它

QLoRA需要以下组件配合：

```text
transformers + peft + bitsandbytes + 支持的量化后端
```

其中，**bitsandbytes** 负责4bit和8bit量化加载，支持低精度计算。

> **关于后端支持（重要）**：本教程统一以 **NVIDIA CUDA GPU** 为准，这是项目支持范围，也是当前资料最全、验证最充分的路径。但请注意，QLoRA/4-bit量化**并非只能用在NVIDIA上**——bitsandbytes近年已在扩展对CPU、AMD ROCm、Intel等后端的支持。如果你的硬件不是NVIDIA GPU，务必查阅官方文档确认对应后端和安装方式，本文示例命令未必适用。
>
> **各后端支持程度差异很大**：截至本文，NVIDIA CUDA是**完整支持

## 块 5

## 电脑配置批注：动手前的“家底”盘点

你要跑 LoRA / QLoRA 本地训练，先别急着敲命令，花两分钟对照下面清单，确认你的电脑“带得动”。

### LoRA 最低配置（先看这个）

- **CPU**：4 核以上
- **内存**：16GB+
- **磁盘**：50GB 以上（模型和数据集都要占地方）
- **GPU**：推荐 NVIDIA GPU
- **显存**：最低 8GB，推荐 12GB+

### QLoRA 要求（你的进阶目标）

QLoRA 的核心思路是**用更小显存跑大模型**——它把基础模型量化到 4-bit，显存占用通常比 LoRA 全精度加载低不少。但注意，下面数值只是**经验参考**，不是硬性门槛：

- **参考下限**：约 12GB（7B 级别模型，短序列、小 batch）；8GB 仅适用于更小模型（如 1B~3B）或极端短序列、小 batch 场景
- **推荐**：16GB+

**为什么不是固定值？** 实际显存占用受这些条件影响：

- **模型规模**：7B 级别；换 13B/70B 显存需求会显著上升
- **序列长度**：短序列（≤512）占用低；越长激活值显存增长越明显
- **batch 大小**：小 batch（1~2）占用低；越大越高
- **梯度检查点**：开启可“用时间换显存”，降低峰值
- **量化配置**

## 块 6

## 5.1 Python版本检查

先确认你的Python版本是否达标。在终端执行：

```bash
python --version
```

建议使用 **Python 3.10+**。如果版本过低，后续安装PEFT和bitsandbytes时可能遇到兼容性问题。

---

## 5.2 PyTorch检查（关键步骤）

> ⚠️ **重点提示**：LoRA/QLoRA依赖PyTorch，且bitsandbytes的CUDA量化路径要求安装**与本机CUDA匹配的GPU版PyTorch**。直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

请以 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 为准。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

安装后运行以下代码验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())   # 应为 True
print(torch.version.cuda)          # 查看torch编译使用的CUDA版本
```

**正常输出**：`True` 表示PyTorch能看到CUDA GPU。

> ⚠️ **注意**：`torch.cuda.is_available()`返回`True`只说明**PyTorch侧**能用GPU，**不足以证明bitsandbytes的4-bit量化路径可用**（后者还依赖bitsandbytes是否正确编译/匹配到本机CUDA）。4-bit能否真正运行，以第八章、第十三章对bitsandbytes的检查为准。

---

## 5.3 查看GPU信息

终端执行：

```bash
nvidia-smi
```

输出应包含：
- GPU型号（如RTX 4090）
- CUDA版本（如CUDA 12.x）
- 显存大小（如Memory 24GB）

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是Hugging Face官方的参数高效微调库，支持：
- **LoRA**（你的学习目标）
- Prefix Tuning
- Prompt Tuning
- Adapter

### 6.2 安装PEFT

```bash
pip install peft
```

验证安装：

```python
import peft
print(peft.__version__)
```

成功打印版本号即安装完成。后续加载模型时，请使用统一设备写法：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

> 📄 **模型许可证**：本教程涉及的模型（如Llama 2）遵循相应开源许可证

## 块 7

**7.1 bitsandbytes 在做什么？**

简单说，`bitsandbytes` 是 QLoRA 训练的关键依赖，负责两件事：**8bit 量化** 和 **4bit 量化**。

**为什么你必须要装它？**

因为本教程的 QLoRA 方案，就是基于 `bitsandbytes` 做 4-bit 量化。所以它是前置依赖，**必须安装**。

> 注意：QLoRA 的核心思想（4-bit 量化基础模型 + LoRA）并不绑定某个特定库。只是本教程统一用 `bitsandbytes` + NVIDIA CUDA 演示。如果你用别的量化后端，以对应文档为准。

**安装命令（直接执行）：**

```bash
pip install bitsandbytes
```

---

**本块学习检查**：在终端执行上述安装命令，确认安装

## 块 8

**动手前先查环境：为什么必须看CUDA？**

bitsandbytes是QLoRA量化的底层依赖，它通过CUDA调用GPU计算。如果CUDA版本不匹配，后果很直接：CUDA不可用、量化失败、加载模型报错。你正在学LoRA微调，这一步跳过，后面全是坑。

**两步检查，30秒搞定**

1. **看GPU驱动支持的CUDA版本**  
   终端运行：
   ```bash
   nvidia-smi
   ```
   找到输出中的`CUDA Version`，记下这个数字。

2. **确认bitsandbytes能识别你的环境**  
   运行：
   ```bash
   python -m bitsandbytes
   ```
   正常输出会显示CU

## 块 9

**本块学习检查**：在终端运行 `python -c "import bitsandbytes"`，观察是否报错。若报错提示 CUDA 动态库加载失败，请记录错误信息，并对照下文排查原因。

---

**9.1 推荐使用WSL2（动手实践优先）**

你的目标是快速跑通 LoRA 微调，建议直接采用以下环境路径：

```text
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

**为什么？** 核心原因一句话：**Linux 环境兼容性更好**。WSL2 能让你在 Windows 上获得接近 Linux 的体验，大幅减少环境配置的坑。

---

**9.2 Windows 原生可能遇到的问题**

如果你坚持用 Windows 原生环境，最典型的问题是：**bitsandbytes 无法加载 CUDA 动态库**。

这不是单一原因造成的，常见因素包括：

-

## 块 10

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：CUDA版本不匹配**

在Windows上配置GPU环境时，最常见的坑就是CUDA版本对不上。别慌，按下面三步排查：

**第一步：查看驱动和CUDA版本**

打开终端，运行：

```bash
nvidia-smi
```

这个命令会告诉你三件事：
- **NVIDIA驱动**是否正常安装
- 驱动支持的**CUDA版本**（看右上角）
- GPU型号和显存

**第二步：确认PyTorch版本匹配**

记住一个核心原则：**确保NVIDIA驱动版本不低于PyTorch所需的CUDA版本对应的最低驱动版本**。如果驱动版本过低，就会报错。

---

## 十、安装完整LoRA环境

假设你已经有了基础环境（transformers、datasets、accelerate），现在补上LoRA需要的组件。

**关键前置**：确保已安装**匹配本机CUDA的GPU版PyTorch**，否则后面`import torch`或QLoRA 4-bit会失败。建议至少8GB显存，QLoRA 4-bit量化可降低显存需求。

以CUDA 12.1为例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

然后安装LoRA核心库：

```bash
pip install peft bitsandbytes
```

**一步到位**（推荐）：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

运行配置向导：

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

完成后会生成`default_config.yaml`。

---

## 十二、LoRA环境验证实验

创建`test_lora_env.py`：

```python
import torch
from peft import LoraConfig

try:
    print("CUDA:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch检查失败，请确认GPU版PyTorch已正确安装。")

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

看到`CUDA: True`且配置正常打印，说明：
- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes：

```python
import torch

try:
    import bitsandbytes
    print(torch.cuda.is_available())
except ImportError:
    print("bitsandbytes未安装，请先安装。")
```

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

如果你在Windows上遇到兼容性问题，**推荐使用WSL2**。这是官方建议的解决方案，能有效规避Windows原生环境下的各种报错。

---

**常见问题速查表**

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

**动手检查**：确认你的CUDA版本与bitsandbytes要求一致。若在Windows下报错，优先切换到WSL2环境再继续。

---

## 十、完整LoRA/QLoRA学习路线

**当前环境栈**（按依赖顺序）：

```
Transformers → Datasets → Accelerate → PEFT → bitsandbytes
```

**后续实战流程**：

```
加载基础模型 → 准备训练

## 块 12

## 一、为什么需要LoRA？——先动手感受一下问题

你已经有PyTorch和HuggingFace基础了，咱们直接进入正题。先想一个场景：你手上有一个7B模型（比如LLaMA、Qwen、Baichuan或Mistral），想让它学会某个新任务。

**传统做法是这样的：**
预训练模型 → 更新全部参数 → 保存新模型

听起来简单，但实际跑起来你会发现三个“拦路虎”：

1. **GPU显存爆掉**：7B模型有70亿参数，训练时不仅要存模型参数W，还要存梯度∂W和优化器状态（比如Adam的moment），实际显存需求远大于模型本身大小。假设100亿参数的模型，显存需求会大到离谱。
2. **训练时间漫长**：更新全部参数意味着每次反向传播都要计算所有层的梯度，训练周期以周甚至月为单位。
3. **存储空间巨大**：每个微调任务都要保存一份完整的模型副本，7B模型就是几十GB，多个任务直接存不下。

**核心矛盾：** 你只是想让它学会一个新技能，但全参数微调逼你为“全部知识”买单。

---

## 二、LoRA的破局思路——只学“增量”

LoRA的核心洞察是：**大模型微调时，不需要更新全部参数，只需要学习一个“低秩的增量”**。

用数学语言说：预训练模型的权重是W，微调后的权重是W'，那么：

**W' = W + ΔW**

## 块 13

**本块学习检查**：请用一句话说明 LoRA 微调时，模型最终使用的权重公式是什么，并指出哪个参数是被冻结的。

---

**2.1 不直接修改原模型参数**

传统微调直接修改原始权重 `W`，训练后变成 `W'`。

LoRA 的做法完全不同：**保持 `W` 冻结**，不动它。然后额外增加一个新参数 `ΔW`。最终模型使用的权重是：

`W' = W + ΔW`

- `W`：原模型参数（冻结，不训练）
- `ΔW`：LoRA 新增参数（只训练这个）

**2.2 为什么只训练 ΔW？**

大模型已经通过海量数据学会了语言规律、世界知识和推理能力。微调通常只是对已有能力做**方向调整**，不需要重新学全部参数，只学**任务相关的变化量**就能见效。

> 注意：这是教学直觉，不是严格结论。能否替代全参数微调取决于任务差距、数据规模、目标模块与秩的设置；差距大或数据充足

## 块 14

你正在学LoRA，目标很明确：搞懂它怎么省参数。咱们直接看核心。

Transformer里，大量参数是矩阵。比如注意力层，`Q = X · Wq`，这里的`Wq`就是一个权重矩阵。假设它形状是`4096 × 4096`，那参数量就是`4096 × 4096`，约1677万。

问题来了：如果微调时要修改这个矩阵，直接学一个`ΔW`，那`ΔW`也得是`4096 × 4096`，同样是1677万参数。这跟全量微调没区别，不高效。

LoRA的思路很直接：**不直接学完整矩阵**，而是把它拆成两个低秩小矩阵。

具体来说，原始权重`W`保持冻结不动，新增的`ΔW`由`B ×

## 块 15

**4.1 矩阵尺寸变化（动手算一遍）**

假设原矩阵 W 是 **4096 × 4096**（即 d_in = d_out = 4096）。直接训练，参数量就是：

`4096 × 4096 = 16,777,216`（约1677万）。

现在用 LoRA，设 **rank r = 8**。不直接动 W，而是拆成两个小矩阵：

- **A**：`8 × 4096` → 参数量 `8 × 4096 = 32,768`
- **B**：`4096 × 8` → 参数量 `4096 × 8 = 32,768`

两者相加：`32,768 + 32

## 块 16

## 5.1 神经网络参数变化具有低维特征

LoRA 的核心洞察来自一个关键假设：模型适应新任务时，参数变化 **ΔW** 并不需要覆盖所有方向。

**直觉理解**：想象一个 4096 维的空间，真正影响任务的"有效方向"可能远少于满维度——它们集中在一个**低维子空间**里，即远小于原维度的少数方向。

> ⚠️ 注意：这里的"低维"是**定性**概念，没有通用固定值（如"几十维"），具体维度随模型、任务、数据而变化，不要记成确定数字。

**结论**：既然 ΔW 的有效变化集中在一个低维子空间，那么用低秩矩阵 **BA** 来近似 ΔW 就是合理的

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张高清图片，尺寸是 **4000×4000像素**。如果要把这张图片**完整保存**下来，需要存储 **1600万个数据**。

但问题来了——如果这张图片的主要变化其实很有限，只是：

- **亮度**调整；
- **对比度**调整；
- **少量结构变化**。

那么，我们完全可以用**少量参数**来描述这些变化，而不必保存全部1600万个像素。

**LoRA的核心思想与此类似**：在微调大模型时，我们不需要更新全部参数，而是**冻结原始权重W，仅训练低秩矩阵A和B，形成增量ΔW=(α/r)BA**，用少量参数来描述模型权重的变化方向。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构如下：

```text
Input → Embedding → Attention → FFN → Output
```

其中，**Attention**模块包含四个投影矩阵：

```text
Q（查询）、K（键）、V（值）、O（输出）
```

在LoRA的**早期实践**中，常见的做法是将LoRA加到：

```text
Q投影 和 V投影
```

代码示例：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示：**只训练 Wq 的变化和 Wv 的变化**。

> ⚠️ **重要说明**：只挑 `q_proj`/`v_proj` 是LoRA原论文中的一种**常见选择**，并**不是固定标准**。现代实践中也经常把LoRA加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

为了让说明与实操不脱节，这里给出一个**覆盖更完整线性层**的配置示例（以Llama类模型常见的模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

```python
from peft import LoraConfig

# 示例：把LoRA同时加到注意力的q/k/v/o与FFN的gate/up/down线性层
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v

## 块 18

### 8.1 rank（r）——用“旋钮”思维理解它

参数里那个 `r=8`，就是**低秩矩阵的维度**。你可以把它想象成一个“信息容量旋钮”：旋钮拧得越大，微调时能塞进的新知识就越多，但代价也越明显。

**r 越大：**
- 优点：表达能力更强，模型能学得更复杂的变化；
- 缺点：参数更多、显存占用更高。

**r 越小：**
- 优点：省显存、Adapter 更小、训练更快；
- 缺点：表达能力受限，复杂任务可能学不动。

**经验起点参考**（注意：只是起点，不是定死的规则）：

| 任务类型 | rank 起点参考 |
|---------|-------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

> 实际选多少，跟模型规模、目标模块、数据量、任务难度都有关。上表只帮你“起步”，最终值要靠验证集表现和资源成本一起定。

---

**比背“配方”更可靠的做法：基于验证集选 r**

既然你是 hands-on 风格，直接给你一套可操作的流程：

1. **划出验证集**：从数据里留出 20%（或 10%）不参与训练，后面所有比较都在这上面看指标。
2. **固定其它变量**：`target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数都保持不变，只改 `r`

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制 LoRA 参数对原模型权重的更新强度。

```python
lora_alpha=16
```

**核心公式**：`alpha / r` 作为缩放比例。

- `r` 是低秩矩阵的秩（rank），决定矩阵分解的维度
- `alpha` 越大，LoRA 对模型权重的修改幅度越大

**直观理解**：`alpha/r` 相当于一个“音量旋钮”——`alpha` 相对 `r` 越大，LoRA 的“声音”越响，对原模型的影响越明显。

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

在训练 LoRA 参数时随机丢弃一部分神经元，防止 LoRA 参数过拟合训练数据。常见取值为 `0.05`（5%）。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> ⚠️ **注意**：上表是**相对趋势**，不是绝对数值。LoRA 的可

## 块 20

**本块学习检查**：读完本节后，请暂停并回答——若原权重是 4096×4096，设置 r=8，那么 A 和 B 的维度分别是多少？最终 ΔW 恢复成什么形状？答不上来就回头重读一遍。

---

### 2.1 Rank 到底在表示什么？

Rank，中文叫**矩阵的秩**。但在 LoRA 里，你不需要去翻线性代数的定义，只需要记住一个更直观的说法：

> **rank 就是低秩矩阵中间那个“瓶颈”的宽度。**

什么意思？直接看例子。

假设原始权重矩阵 W 是：

```text
4096 × 4096
```

这是一个巨大的方阵。LoRA 不直接动它，而是拆成两个小矩阵去近似它的变化量 ΔW。

如果你设置：

```python
r=8
```

那么这两个小矩阵的形状分别是：

```text
A: 8 × 4096
B: 4096 × 8
```

注意看：A 的行数是 8，B 的列数也是 8。这个 **8 就是 rank**，也就是中间那个“窄通道”的尺寸。

最后把 B 和 A 乘回来：

```text
ΔW = B × A
```

得到：

```text
4096 × 4096
```

形状和原始 W 完全一致，但参数量从 4096×4096（约 1677 万）降到了 8×4096 + 4096×8（约 6.5 万），少了 99.6%。

**一句话总结**：rank 决定了 LoRA 能“记住”多少信息——rank 越大，中间通道越宽，能表达的变化越丰富，但训练和存储成本也越高。

## 块 21

**Rank控制什么？——用你的话理解**

Rank就是LoRA能学的“变化方向”数量。

- **低rank（r=4）**：变化模式少。好处是参数少、显存低、训练快；坏处是表达能力有限。
- **高rank（r=64）**：自由度更多。能学复杂任务、适应力强；但参数多、显存涨、训练成本高。

**动手验证**：在代码里把`r`从4改成64，对比训练前后模型输出差异。你观察到的变化，就是Rank控制的“表达能力”差异。

## 块 22

**本块学习检查**：动手算一遍——把r=16代入公式，写出参数量并对比完整训练。

---

**3.1 不用LoRA时**，微调要更新整个权重矩阵：4096×4096 = 1677万参数，全量更新。

**3.2 用LoRA后**，只训练低秩矩阵，参数量 = r × (输入维度 + 输出维度)。

- r=8：8×(4096+4096) = 6.5万
- r=64：64×8192 = 52万

对比：完整训练1677万，LoRA即使r=64也仅52万，**远低于全参数**。所以rank增大只是让可训练参数

## 块 23

### 4.1 Rank太小的问题

如果你把 `r` 设得很小，比如 `r=2`，会出什么问题？

**核心症状**：模型“记不住”新任务。

- **学习能力不足**：低秩矩阵能表达的信息有限，就像用一张很小的便签纸记笔记，写不下太多东西。
- **欠拟合**：模型在训练集上就学不好，loss 降不下去。
- **新任务知识无法充分表达**：微调后模型对新任务的效果提升很有限。

**直观表现**：
```text
训练loss下降慢
效果提升有限
```

**动手验证（hands-on）**：你可以分别用 `r=2` 和 `r=8` 跑同一个微调任务，对比前几个 epoch 的 loss 曲线。`r=2` 的曲线会明显更平缓，下降更慢。

---

### 4.2 Rank太大的问题

那如果把 `r` 设得很大呢？比如 `r=256`。

**核心症状**：LoRA 的“省钱”优势被削弱。

- **参数增加**：要训练的矩阵变大了。
- **显存增加**：更大的矩阵占用更多显存。
- **训练速度下降**：计算量变大，训练变慢。

**关键澄清**：`r=256` 并不意味着“接近完整矩阵”。以文中 `4096 × 4096` 的权重为例，完整矩阵的秩是 4096，`r=256` 远小于它。它的问题只是**相对** `r=8` 来说，参数占比明显变大，成本升高，所以“参数高效”的优势下降了。

**用数字看清楚**（`d_in = d_out = 4096`）：

| rank r | LoRA 参数量 | 完整矩阵参数量 | 占比 |
| ---

## 块 24

**4.3 常见rank选择（实操起点）**

你正在学LoRA微调，且偏好动手。记住：下表是**经验起始范围**，不是自动调参标准。

| 任务类型 | rank（起点参考） |
|------|------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**动手建议**：从表格起点开始，然后根据以下实际因素调整：数据量、模型大小、序列长度、batch大小（含梯度累积后的有效batch）、目标模块（`target_modules`）的选择与数量、GPU显存、验证集/基准任务表现。

**关键提醒**：不要照搬表格。同一个`r`在不同模型规模、序列长度/batch、目标模块下的显存与效果差别很大。

---

**五、

## 块 25

### 6.1 控制LoRA影响强度

我们直接上手感受一下 `alpha` 怎么控制 LoRA 的“力气”。假设你已经设好了 `r=8`，这是 LoRA 矩阵的“宽度”。

**场景一：放大影响**
如果你把 `alpha` 设为 16，那么缩放系数就是 `16/8=2`。这意味着 LoRA 的更新量会被**放大2倍**。模型微调时，LoRA 对原模型参数的改动会更有力。

**场景二：保持原样**
如果你把 `alpha` 也设为 8，缩放系数就是 `8/8=1`。LoRA 更新保持原始比例，不多不少。

**场景三：减弱影响**
如果你把 `alpha` 设为 4，缩放系数就是 `4/8=0.5`。LoRA 的影响会被**减半**，改动更温和。

---

### 七、Alpha与Rank的关系（重点）

记住这个核心公式：**LoRA 的更新量会被 `lora_alpha / r` 缩放**。这个 `lora_alpha / r` 就是你调节 LoRA 影响强度的“旋钮”。

**但要注意**：`lora_alpha / r` 是**缩放系数

## 块 26

**动手验证：Rank与Alpha的搭配**

先看两组配置：

- 配置1：`r=8, alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16, alpha=32` → 比例 = 32 ÷ 16 = **2**

**关键点**：虽然rank不同，但缩放比例相同。

**实操建议**：调整rank时，**同步调整alpha**，保持 `α/r` 比例大致稳定。这是**经验做法**，非硬性规则。`alpha = 2 × r` 只是常见**经验起点**，不是最优解。

**你的任务**：实际微调时，先按 `alpha = 2 × r`

## 块 27

**方案1：小模型实验（动手首选）**

既然你偏好动手实践，这个方案最适合快速跑通流程。配置很简单：

```python
r=8
alpha=16
```

**特点**：参数少、训练快，非常适合先做小规模测试，验证你的LoRA流程是否跑得通。

**为什么适合你**：你每天只有2小时，且学习节奏快。先用这个方案把环境、代码、数据链路全部打通，建立信心，再逐步加大规模。

---

**动手验证任务**：

用上述配置跑一个最小训练脚本（比如只训练几十步），观察loss是否下降。这一步能确认你的LoRA配置和代码无误。跑通后，再尝试方案2（`r=16, alpha=32`）对比效果差异。

---

**关键提醒**：`r`控制LoRA容量（即参数量），`alpha`控制更新幅度。方案1中`alpha`是`r`的两倍，这是常见搭配，但并非固定规则。先按这个配置跑通，后续再根据效果调整

## 块 28

## 误区1：rank越大一定越好？

**错误。**

很多初学者会陷入“参数越多，效果越好”的直觉陷阱，但LoRA的rank并非越大越优。当任务本身比较简单时，过大的rank只会带来两个问题：

- **浪费显存**：更大的rank意味着更多的可训练参数，占用更多显存资源；
- **增加训练成本**：参数增多，计算量和训练时间也随之上升。

**核心认知**：rank决定的是LoRA的“容量”——即它能表达多少新知识。任务简单时，小容量就够用，强行扩容反而得不偿失。

---

## 误区2：只调rank，不调alpha

这是一个非常隐蔽的坑。来看一个典型错误：

```python
# 原配置
r=8
alpha=16

# 错误修改
r=64
alpha=16
```

改动后，`α/r` 的缩放比例变成了：

```text
16/64 = 0.25
```

这意味着LoRA对模型权重的更新强度被压缩到了原来的四分之一。**你虽然扩大了容量，但更新力度反而被削弱了**，最终效果可能不升反降。

**正确做法**：调整rank时，alpha要同步调整，保持 `α/r` 比例在合理范围。

---

## 误区3：alpha越大越好？

**错误。**

alpha过大会导致 `α/r` 缩放过大，可能引发：

- **更新过强**：每次参数更新的步幅过大，容易越过最优解；
- **训练不稳定**：损失震荡甚至发散。

> **重要提醒**：`α/r` 过大只是训练不稳定的**可能原因之一**。训练震荡还常常来自学习率过高、数据质量差、batch size过小、精度设置（fp16/bf16）与数值稳定性、缺少梯度裁剪等。排查不稳定问题时，应综合考量这些因素，不要只盯着alpha。

---

## 配置建议

本章节内容以**数学原理、参数含义、配置策略**为主，不需要GPU。最低配置：任意现代CPU + 4GB内存即可。

若后续进行

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。在 LoRA 微调流程中，`LoraConfig` 是核心配置对象，它告诉 PEFT 四件事：**LoRA 放在哪里、规模多大、训练强度如何、是否使用 dropout**。

LoRA 训练流程如下：

```text
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

其中，`LoraConfig` 就是流程第二步的“总指挥”，你通过它来声明后续所有 LoRA 行为。

---

### 动手理解：先跑通最小配置

既然你是 hands-on 型学习者，建议立刻在代码里创建一个最简配置：

```python
from peft import LoraConfig

config = LoraConfig()
print(config)
```

观察输出，你会看到所有参数的默认值。接下来我们逐个拆解这些参数分别控制什么。

---

### 四个核心参数速览

| 参数 | 作用 | 类比 |
|------|------|------|
| `target_modules` | 指定 LoRA 插入哪些层 | 决定“改哪里” |
| `r` | 低秩矩阵的秩，控制 LoRA 容量 | 决定“规模多大” |
| `lora_alpha` | 缩放因子，控制更新强度 | 决定“改多猛” |
| `lora_dropout` | 丢弃率，防止过拟合 | 决定“稳不稳” |

---

### 学习检查

在继续之前，请确认你能回答：`Lora

## 块 30

## 一、先看一个最小配置

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

这四行就是LoRA的核心。对照下表，逐个理解：

| 参数 | 作用 |
|------|------|
| `target_modules` | 决定插入哪些层 |
| `r` | 决定LoRA矩阵大小 |
| `lora_alpha` | 控制缩放强度 |
| `lora_dropout` | 控制随机失活 |

## 二、完整流程：从配置到模型

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

实际代码：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, config)
```

执行后：**原模型参数冻结**，**新增LoRA参数训练**。

## 三、重点：target_modules

这个参数决定LoRA插在哪。示例中只选了`q_proj`和`v_proj`——即注意力层的query和value投影。选

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA 你要在模型的哪些模块上添加低秩适配（即训练低秩矩阵A和B，以增量方式更新权重）**。

Transformer 的 Attention 结构通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA **不会默认修改所有层**，你必须显式指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思就是：**只给 Query 投影层和 Value 投影层加 LoRA**。

---

## 四、为什么通常选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开看两个角色的分工：

- **Q（Query）和 K（Key）**：共同决定模型**关注什么信息**（注意力权重）；
- **V（Value）**：决定模型**传递什么信息**（被加权求和的内容）。

所以，调整 Q 和 V，通常就能有效改变模型的行为——这就是它们成为“常见起点”的原因。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型通用的“标准配置”。不同模型的模块命名可能不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 `target_modules`。

---

## 偷懒技巧：`target_modules="all-linear"`

如果你不想手工枚举模块名，较新版本的 PEFT（一般 ≥ 0.7，具体以你装的版本文档为准）支持直接写：

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

**用 `all-linear` 的代价**：覆盖模块变多 → 可训练参数和显存占用都会增加。所以要在**参数量/显存**和**效果**之间权衡，最终以验证集结果为准。

如果你的 PEFT 版本太旧不支持这个字符串，就退回**显式列出模块名**的写法，或者升级 PEFT。用 `print_trainable_parameters()` 可以快速验证注入是否符合预期。

## 块 32

### 方案1：只调整Q和V（你的轻量起点）

如果你刚上手LoRA微调，想快速跑通流程，这个方案最合适。代码就两行：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么选它？** 三个字：**少、快、稳**。  
- **参数少**：只训练注意力机制里的查询（Q）和值（V）投影层，其他权重全部冻结，显存占用小。  
- **训练快**：可训练参数量少，反向传播计算量低，迭代速度明显提升。  
- **常见起点**：这是业界常用的轻量配置，尤其适合先验证“LoRA在你的任务上是否有效”，再决定是否增加模块。

**动手验证（2分钟）**：  
用你现有的模型，先按此配置跑一个极小的训练步（比如10步），观察显存占用和每步耗时。记住这两个数字——

## 块 33

## LLaMA系列：target_modules怎么设？

你正在做LoRA微调，核心思路是：**只训练一小部分参数，其余冻结**。`target_modules`就是告诉PEFT“你要动哪些层”。

对于LLaMA系列模型，最常用、最稳妥的写法是：

```python
target_modules = ["q_proj", "v_proj"]
```

**为什么是这两个？**

- `q_proj`：查询投影层（Query）
- `v_proj`：值投影层（Value）

在LLaMA的注意力机制中，Q和V是两个独立的线性层。实践表明，只微调这两个层就能在多数任务上取得不错的效果，同时训练参数量可控、显存占用友好。

**动手验证一下**（如果你手头有LLaMA模型）：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-llama-model")

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

观察输出的可训练参数量——你会看到它只占全模型参数的很小比例，这就是LoRA的威力。

---

## 对比：GPT-2为什么不一样？

GPT-2的写法可能让你困惑：

```python
target_modules = ["c_attn"]
```

**关键区别**：GPT-2的`c_attn`是一个合并的QKV投影层，权重形状为`(in_features, 3*out_features)`，同时计算Q、K、V，而不是像LLaMA那样拆成三个独立层。所以你要target的是`c_attn`这一个整体。

**还有一个坑：`fan_in_fan_out=True`**

GPT-2用的是`Conv1D`（不是普通的`nn.Linear`），它的权重形状是`(in_features, out_features)`——和`nn.Linear`的`(out_features, in_features)`**正好转置**。由于Conv1D权重形状与nn.Linear转置，需设置`fan_in_fan_out=True`以正确注入LoRA。

所以GPT-2的完整

## 块 34

**动手配置BERT的LoRA：先查模块名，别照搬**

你已经在学LoRA微调，目标很明确。这里有个关键点：**BERT的target_modules不能直接复制LLaMA那套配置**。

BERT的注意力层中，常见的做法是对 `query` 和 `value` 投影应用LoRA，但具体选择应依据模型结构和任务需求，可通过实验确定。不同模型实现不一样，**千万别猜**。

**实操步骤：**

1. 加载你的BERT模型后，先跑这段代码看真实模块名：

```python
for name, _ in model.named_modules():
    print(name)
```

2. 在输出里找 `q_proj` 和 `v_proj`（或类似名字），确认它们存在。

3. 再配置：

```python
target_modules=["q_proj", "v_proj"]
```

**匹配机制提醒**：`target_modules` 是按**后缀匹配**的。写 `"q_proj"` 会命中所有以 `q_proj` 结尾的层，不用一个个列。想精确控制也可以写全名或正则。

**注意**：不同模型命名

## 块 35

**r是什么？——LoRA的“瘦身开关”**

r（rank）控制LoRA低秩矩阵的维度。它决定了更新量ΔW = BA中，A和B的中间维度大小。

**直观理解：**
原矩阵是4096×4096，设置r=8后，A为8×4096，B为4096×8。r越小，中间“瓶颈”越窄，参数越少。

**r的影响：**
- **r越大**：学习能力更强，能表达复杂任务，但参数和显存占用增加。
- **r越小**：更省资源，但表达能力受限。

**经验起点（非固定配方，需用验证集调整）：**
- 简单任务：r≈4-8
- 通用指令微调：r≈8-

## 块 36

## 7.1 alpha作用：控制LoRA更新幅度

LoRA的最终更新公式不是简单的 `BA`，而是：

```text
(alpha/r)BA
```

这里的 `alpha` 控制的就是 **LoRA更新幅度**。

**动手验证**：配置 `r=8`、`lora_alpha=16`，缩放系数为 `16/8=2`，意味着LoRA对模型的影响被增强到2倍。你可以试着改大 `alpha`，观察微调后模型输出的变化幅度是否变大。

## 7.2 alpha常见设置

一个常见的经验起点是：

```text
alpha = 2 × r
```

> 注意：这只是起点，不是通用规则。也可以固定 `alpha` 单独调 `r`，或用其他比例，最终以验证集表现定。

**

## 块 37

**8.1 dropout作用**  
训练时，LoRA会对低秩矩阵A和B的权重进行随机置零（dropout），例如`lora_dropout=0.05`表示有5%的权重被随机置零，目的是减少**过拟合**。配置示例：`lora_dropout=0.05`。

**8.2 如何选择dropout？**  
经验起点（非固定规则，需结合验证集调参）：  
- 数据少：约 `0.05~0.1`  
- 数据多：约 `0~0.05`  

若训练数据非常丰富，可设 `lora_dropout=0`。

**动手验证**：在你的LoRA训练脚本

## 块 38

**9.1 基础配置（动手版）**

这个配置适合在 7B 模型上快速测试 LoRA。先看代码，再逐行拆解：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：秩的大小，决定注入矩阵的维度
    lora_alpha=16,      # 更新缩放：控制新参数对原模型的影响强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的Q和V投影
    lora_dropout=0.05,  # 防止过拟合：训练时随机丢弃5%的LoRA参数
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（如GPT类）
)
```

**参数速查表**

| 参数 | 值 | 作用 |
|---|---|---|
| r | 8 | LoRA容量 |
| alpha |

## 块 39

## 9.1 基础配置：一次跑通 LoRA + 分类头保存

你正在学 LoRA 微调，目标很明确：用最少的参数把 BERT 调成自己的分类器。这一节直接给你一个能跑通的完整案例，重点看 `modules_to_save` 这个参数——它是新手最容易踩坑的地方。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里加载的是带分类头的 BERT。注意：分类头是新初始化的，**必须跟着 LoRA 一起训练和保存**，否则推理时输出就是乱的。

### 第二步：配置 LoraConfig

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
```

逐个看参数：

- `r=8`：LoRA 矩阵的秩，控制新增参数量
- `lora_alpha=16`：缩放系数，实际缩放比例是 `alpha / r = 2`
- `lora_dropout=0.05`：防止过拟合
- `target_modules=["query", "value"]`：只微调 BERT 注意力里的 Q 和 V 矩阵
- `modules_to_save=["classifier"]`：**关键参数**，把分类头也纳入训练并保存

### 第三步：包装模型并检查参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

运行后会看到可训练参数 = LoRA 参数 + classifier 参数。如果只看到 LoRA 参数，说明 `modules_to_save` 没生效。

### 第四步：保存

```python
model.save_pretrained("./lora_ckpt")
```

这里保存的是 **Adapter 权重 + classifier 权重**，不是完整模型。所以重载时必须先加载同一个基础模型，再挂载 LoRA。

### 第五步：重载并验证

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

如果输出 `True`，说明分类头被正确保存和恢复了。

### 为什么必须加 `modules_to_save`

不加它，重载时分类头会退回**随机初始化**状态——你训练好的分类能力全部丢失，模型输出完全错乱。这是最常见的坑，务必记住。

### 两个实操提醒

1. **验证方式**：除了对比权重，也可以用一批固定输入对比保存前后的 logits 是否一致，效果相同。
2. **属性名差异**：不同模型的分类头属性名可能不同（`classifier` / `score`），访问路径也可能随版本变化，以实际结构为准。

### 模型许可证与显存说明

- **模型许可证**：BERT 采用 Apache 2.0 许可证，可商用，无月活限制。
- **显存要求**：BERT-base 共 1.1 亿参数，FP16 推理约需 4GB 显存；LoRA 微调建议至少 8GB 显存。

## 块 40

## 10.1 普通文本生成：LoRA参数怎么配？

你正在学习LoRA微调，目标是让大模型在**普通文本生成**任务上工作（比如续写、开放式生成）。这类任务对模型原有能力的改动最小，所以参数设置也最“轻”。

**直接给结论，普通文本生成推荐配置：**

```python
r=8
alpha=16
dropout=0.05
```

**这三个参数怎么理解？**

- **`r`（秩）**：决定LoRA训练的低秩矩阵A和B的维度。`r=8`意味着用较小的维度来构造增量更新矩阵ΔW=(α/r)BA，改动幅度小，适合普通文本生成这种不需要大幅改变模型行为的任务。
- **`alpha`（缩放系数）**：用于缩放LoRA增量更新的强度。最终增量ΔW=(alpha/r)*BA，alpha/r是缩放因子，这里alpha=16，r=8，alpha/r=2，是一个常用且稳妥的起点。
- **`dropout`（丢弃率）**：防止过拟合。`0.05`是一个很轻的随机丢弃比例，因为普通文本生成通常希望模型保持原有能力，不需要太强的正则化。

**对比一下其他任务，帮你建立直觉：**

| 任务类型 | r | alpha | dropout |
|---------|---|---|---------|
| 普通文本生成 | 8 | 16 | 0.05 |
| 指令微调 | 16 |

## 块 41

**普通文本生成：LoRA配置四要素**

你已进入LoRA微调实战前的最后一块拼图。针对普通文本生成任务，核心配置如下：

```python
LoraConfig(
    r=8,                # LoRA容量
    lora_alpha=16,      # 缩放系数
    target_modules=["q_proj", "v_proj"],  # 改哪里
    lora_dropout=0.05   # 缓解过拟合
)
```

**四个参数，四个问题：**

- **target_modules**：决定改哪里——只修改注意力层的q和v投影。
- **r**：决定LoRA容量，即低秩矩阵的秩。
- **lora_alpha**：缩放系数，

## 块 42

## 一、为什么需要QLoRA？——先看清LoRA的“隐藏成本”

你已经掌握了LoRA的核心思路：冻结原模型，只训练少量低秩矩阵。听起来很省资源，对吧？但这里有个关键点容易被忽略——

**基础模型本身，仍然要完整加载到显存里。**

我们算一笔账。以7B模型为例，如果用FP16精度存储：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是“把模型放进去”的静态开销。实际训练时，你还需要额外空间存放：

* 激活值（前向传播的中间结果）
* 梯度（反向传播计算得到）
* 优化器状态（如Adam的动量项）

这些加起来，实际显存需求会**远高于14GB**。换句话说，LoRA只是让“可训练的参数”变少了，但“模型本身占用的空间”一分没省。

**这就是QLoRA要解决的第一个问题**：能不能

## 块 43

**QLoRA核心思想：让大模型微调“轻装上阵”**

你的目标是用LoRA微调模型，但直接加载完整FP16模型可能显存不够。QLoRA就是为此而生——它先将基础模型权重以4-bit精度存储（NF4量化），减少显存占用，但参数量不变，再在上面挂LoRA训练。

**核心流程**（对比记忆）：

- LoRA：FP16模型 + LoRA
- QLoRA：4-bit量化模型 + LoRA

**关键区别**：QLoRA量化的是**基础模型权重**（冻结不动），而LoRA参数保持FP16/BF16精度并参与训练。推理时输入经过量化基座模型和LoRA适配器共同计算。

## 块 44

## 3.1 模型存储方式：从“重”到“轻”

先看一张表，记住每个参数占用的空间：

| 类型 | 每个参数占用 |
|------|------------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下**：7B模型用FP16存储，需要 `7B × 2 ≈ 14GB`；改用4-bit后，只需 `7B × 0.5 ≈ 3.5GB`。显存直接减少约75%。

**注意**：这个3.5GB只是**理论下界**——只算了量化后的权重本身。实际运行时还要加上量化常数、分组元数据、未量化的模块（norm、embedding、lm_head、LoRA参数）以及计算缓存和激活值。所以真实占用会明显更高，别被这个数字误导。

## 3.2 为什么不用普通INT4？

普通INT4整数范围只有 `-8 ~ 7`，而神经网络权重通常：
- 接近正态分布
- 小数多
- 分布中心接近0

直接整数映射会**严重损失精度**。所以

## 块 45

**NF4是什么？——给正在做LoRA微调的你**

NF4（NormalFloat4）直译是“正态分布4位浮点”，但别被名字骗了——它不是IEEE 754那种通用4位浮点格式，而是QLoRA论文为**神经网络权重“近似正态分布”**量身定做的一套**4-bit量化码本**。

**核心机制（3步）：**

1. **预置码本**：固定16个（2⁴）量化取值点，按标准正态分布分位数排布，0附近（权重密集区）划分更细。
2. **量化**：每组权重归一化后，映射到最接近的码本值，只存4-bit索引。
3. **反量化**：用码本+缩放因子还原近似权重。

**一句话理解**

## 块 46

好的，我们直接进入正题。既然你的目标是LoRA微调，并且喜欢动手实践，那我们就用最直观的方式来理解NF4量化。

**4.2 为什么NF4适合模型权重？**

想象一下，大模型训练完的权重，大部分数值都挤在0附近，像一个中间高、两边低的山丘（正态分布）。

如果用普通的INT4量化，它把数值范围平均切成16份（比如-8到7），这就好比用一把刻度均匀的尺子去量一座山。结果就是，山脚和山顶的细节被粗糙地忽略，而最需要精度的“山腰”（0附近）反而分到的刻度太少，精度损失大。

NF4的聪明之处在于，它不搞“平均主义”。

## 块 47

**本块学习检查**  
先看一个具体场景：4-bit量化后，每组权重除了存4-bit数值，还要额外保存两个参数——`scale`和`zero point`。这些参数本身也占显存，模型越大，累积开销越可观。  

双重量化的思路很直接：**第一次量化模型权重**（FP16→NF4），得到4-bit权重和scale参数；**第二次再量化scale参数本身**，减少它的存储开销。  

你可以这样验证理解：在纸上画两行流程——上行是“FP16权重→NF4量化→4-bit权重”，下行是“scale参数→再次量化→减少存储”。对比普通量化（只做上行）和双重量化（上下都做），显存节省来自哪里？答案就是下行：省掉了scale参数的冗余存储。

## 块 48

## 6.1 为什么需要分页优化器？

训练时显存峰值不只来自模型本身，**梯度**和**优化器状态**同样占显存。比如Adam优化器要保存**一阶动量**和**二阶动量**两份状态。

QLoRA虽然大幅降低了模型显存占用，但训练过程中仍可能遇到**显存突然不足（OOM）**的情况——尤其是显存出现瞬时峰值时。

## 6.2 分页优化器思想

Paged Optimizer借鉴了操作系统**虚拟内存**的思路，核心机制是：

> 借助 **NVIDIA 统一内存（Unified Memory）**，当显存出现峰值、即将OOM时，自动把优化器状态**按页换出到CPU内存**，需要参与计算时再**换回GPU**。

流程如下：

```text
GPU显存 → 出现峰值（接近/超出上限） → 优化器状态按页换出到CPU内存 → 需要时再换回GPU
```

**关键理解**：不要把它想成"平时就把部分优化器状态常驻CPU、需要时交换"。分页优化器的重点是**应对显存峰值**——在显

## 块 49

## 6.3 类比理解：为什么需要分页优化器？

把显存想象成一张**桌面**，把CPU内存想象成**柜子**。

**普通优化器**的做法是：把所有东西（模型参数、梯度、优化器状态）全堆在桌面上。桌面一旦放满，程序就会报错（OOM，显存溢出），训练直接中断。

**分页优化器**的做法是：只把**常用的文件**放在桌面（当前计算需要的张量），把**暂时不用的文件**存进柜子（CPU内存），等需要时再取出来。这样桌面永远不会被塞满，训练就能继续跑下去。

---

## 6.4 代码实操：如何启用分页优化器？

原理懂了，代码上其实**不需要你写任何换页逻辑**——bitsandbytes 库已经帮你实现了。你只需要在 `TrainingArguments` 里通过 `optim` 参数指定优化器类型：

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

1. **`optim` 的两个取值**：`paged_adamw_8bit` 用 8-bit 存储优化器状态，更省显存；`paged_adamw_32bit` 精度更高但占用更多。QLoRA论文中使用的是paged_adamw_32bit。

2. **版本兼容性提醒**：不同 Transformers / bitsandbytes 版本支持的 `optim` 取值可能不同（比如还有 `adamw_bnb_8bit`）。如果报"未知 optim"错误，查一下你当前版本的 `Training

## 块 50

## 7.1 4-bit量化：把模型“瘦身”

QLoRA的第一步，是把基础模型从FP16（16位浮点）压缩成NF4（4位）格式。你可以把它想象成把一张高清照片压缩成更小的文件——权重变小了，显存占用自然就降下来了。

```
FP16 → NF4 4-bit
```

**关键点**：NF4是QLoRA论文推荐的量化类型，也是本教程采用的方案。虽然bitsandbytes库也支持`fp4`等其他4-bit类型，但NF4是论文首选。

---

## 7.2 双重量化：连“尺子”也压缩

量化过程中需要`scale`（缩放因子）来还原数值。双重量化就是把这个`scale`也做一次量化，进一步减少额外参数的占用。

```
scale → 再次量化
```

**效果**：省下来的显存虽然不多，但积少成多，对大模型很关键。

---

## 7.3 分页优化器：显存不够，CPU来凑

训练时优化器状态很吃显存。分页优化器的思路是：GPU显存不够时，自动把部分数据交换到CPU内存，需要

## 块 51

## load_in_4bit 与 QLoRA 核心配置

你已经掌握了 LoRA 的基本思想，现在进入 QLoRA 的关键环节：**如何用 4-bit 精度把大模型塞进显存**。这里你只需要理解四个配置项，它们共同决定了“省显存”和“算得快”的平衡。

### 1. load_in_4bit：开启 4-bit 加载

这是总开关。设为 `True` 后，模型权重会以 4-bit 精度加载到显存中，而不是默认的 32-bit。这一步直接让显存占用暴降约 8 倍。

### 2. bnb_4bit_quant_type：量化方式选 "nf4"

`"nf4"` 是 NF4（4-bit NormalFloat）量化，专为神经网络权重分布设计，比普通 4-bit 量化精度损失更小。在 QLoRA 中，这就是默认且推荐的选项。

### 3. bnb_4bit_use_double_quant：双重量化

开启后，量化过程中产生的额外缩放参数也会被二次量化，进一步省显存。建议开启，几乎无副作用。

### 4. bnb_4bit_compute_dtype：计算精度（最容易踩坑）

这是你最容易出错的地方，务必理解**存储精度 ≠ 计算精度**：

- **存储**：权重以 4-bit（NF4）存着，目的是省显存。
- **计算**：权重真正参与矩阵运算时，会被**反量化**成更高精度来算，这个计算精度就是 `bnb_4bit_compute_dtype` 指定的。

**不设置会怎样？** 默认通常是 `float32`，计算慢且显存开销大。所以必须显式指定。

**选 BF16 还是 FP16？关键看你的 GPU：**

- `torch.bfloat16`：需要较新 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。BF16 动态范围大，不易溢出，训练更稳定，**新 GPU 上优先选它**。
- `tor

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个**能直接跑的 QLoRA 训练骨架**——你只需要替换模型名和数据，就能看到完整的训练流程。

### 整体流程速览

整个骨架分 6 步，逻辑顺序是：**配置量化 → 加载模型 → 准备 k-bit 训练 → 注入 LoRA → 准备数据 → 配置训练参数并启动**。下面逐步拆解。

---

### 第 1 步：4-bit 量化配置

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

这里做了三件事：
- `load_in_4bit=True`：启用 4-bit 加载；
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化类型；
- `bnb_4bit_use_double_quant=True`：开启双重量化；
- `bnb_4bit_compute_dtype`：**计算精度**，根据 GPU 能力自动选择 bfloat16 或 float16。

> 注意：`compute_dtype` 是**计算时**用的精度，不是存储精度——存储是 4-bit，计算时反量化到这个 dtype。

---

### 第 2 步：以 4-bit 加载基础模型

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

关键点：
- 把第 1 步的 `bnb_config` 传给 `quantization_config`；
- `device_map="auto"` 让模型自动分配到可用设备；
- 如果 tokenizer 没有 pad token，用 eos token 代替——这是训练时的常见坑。

---

### 第 3 步：为 k-bit 训练做准备（关键一步）

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
model.gradient_checkpointing_enable()
```

这一步**不能省略**。它做了两件重要的事：
- 处理 norm 层和输出层的精度问题（这些层在低精度下容易不稳定）；
- 开启梯度检查点所需的输入梯度支持。

`use_gradient_checkpointing=True` 要和后面 `TrainingArguments` 里的 `gradient_checkpointing=True` **呼应**——两处都要开。`model.gradient_checkpointing_enable()` 确保梯度检查点真正生效。

---

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

LoRA 的核心思路：**冻结原模型，只训练低秩增量参数**。`r=8` 是秩，`lora_alpha=16` 是缩放系数，`target_modules` 指定注入哪些模块（`q_proj`、`v_proj` 是注意力中的 Q 和 V 投影）。

`model.print_trainable_parameters()` 会打印可训练参数量——你应该看到可训练参数占比非常小，这就是 LoRA 高效的原因。

---

### 第 5 步：准备数据

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tok, remove_columns=raw.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

Causal LM 的 `labels` 直接复制 `input_ids`——这是语言建模的标准做法。真实场景替换成你自己的数据集即可。

---

### 第 6 步：训练参数 + 启动训练

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

## 九、把 QLoRA 的“五步骨架”串起来（可跑通）

你已经知道 LoRA 是往冻结模型里插小适配器，QLoRA 则是先把模型压成 4-bit 再插。但光有零件不够——**这五步必须按顺序配合，缺一步就崩**。下面就是“能说明关系”的最小骨架，每一步解决一个具体问题：

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么被“压”**
这一步配置基础模型的 4-bit 加载方式：用 NF4 量化、开双重量化、设定计算 dtype。它回答的是“模型以什么形态进内存”。

**第 2 步：`prepare_model_for_kbit_training()` —— 量化模型的“适配层”**
这是 QLoRA 的**推荐步骤**。量化后的模型通常需要适配才能高效训练，这一步会启用梯度检查点、设置输入梯度等，以支持量化模型的训练。**跳过它，可能影响训练效果或导致报错。**

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入可训练的 LoRA**
模型已经 4-bit 冻结了，这一步把 LoRA 适配器“插”进去。从此，只有 LoRA 参数在更新，主干保持冻结。

**第 4 步：梯度检查点要“两边对应开”**
建议在 `TrainingArguments` 里开启 `gradient_checkpointing=True` 以节省显存。只开一边，显存优化效果会打折扣或报错。

**第 5 步：`optim="paged_adamw_8bit"` —— 分页优化器**
这一步启用分页优化器，使用分页优化器将优化器状态分页到 CPU 内存，以处理超出 GPU 内存的情况，避免 OOM。

> ⚠️ 上面是“能说明关系”的最小骨架。真实训练要换成足够规模的数据、合理超参，并按你所用版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数和 `processing_class`/`tokenizer` 写法。

---

## 十、你的电脑能跑吗？

**只学原理：不需要 GPU。**

**实际跑 QLoRA：**
- CPU：4 核+
- 内存：16GB+
- GPU：NVIDIA CUDA GPU

**显存参考（7B 级别，经验值

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 先搞懂：LoRA不是直接改基础模型

你训练LoRA时，实际发生的是这样的分工：

- **基础模型**：全程冻结，一动不动（比如Llama-7B）
- **LoRA Adapter**：这才是真正被训练的部分

训练结束后，你得到的结构是：

```
基础模型 + LoRA Adapter
```

**关键点来了**：保存时，通常只保存Adapter，而不是完整模型。你会在输出文件夹里看到：

```
adapter_model.safetensors
adapter_config.json
```

就这两个文件，没有基础模型。

### 1.2 Adapter的特点：小而灵活

**优点**：
- 文件小，保存快
- 一个基础模型可以搭配多个Adapter，各干各的活

比如同一个基础模型，你可以挂上：

```
客服Adapter  → 做客服
医疗Adapter  → 做医疗问答
法律Adapter  → 做法律咨询
```

**缺点**：部署时必须同时准备两部分：

```
基础模型 + Adapter
```

缺一不可，这就给部署带来了麻烦——你总不能每次推理都手动加载两个东西吧？

---

## 动手验证一下

打开你的训练输出目录，确认里面是不是只有`adapter_model.safetensors`和`adapter_config.json`，没有完整模型文件。这一步能帮你直观理解“LoRA保存的是Adapter”这件事。

---

**小结**：LoRA训练后保存的是轻量级Adapter，不是完整模型。部署时需要基础模型和Adapter同时在场——这就是为什么下一步我们要学`merge_and_unload()`来

## 块 55

**动手做：先看“未合并”的推理流程**

你微调完LoRA后，如果直接拿去推理，模型内部其实是这样的：

- **输入** → 先经过**基础模型**（Base Model）计算 → 再额外叠加**LoRA适配器**（Adapter）的结果 → 才得到**输出**。

这意味着，每次推理你都得**同时加载两样东西**：基础模型 + Adapter。麻烦不说，部署时还容易出错。

**你的目标：把两样变一样**

部署时，你真正想要的是：

- **输入** → 经过一个**完整的模型** → **输出**

所以，我们需要把LoRA的权重**合并**进基础模型里，让它们变成一个整体。

**合并的数学

## 块 56

## 3.1 merge_and_unload() 是干什么的？

你已经知道 LoRA 微调的核心思路：**不直接改基础模型**，而是在旁边加一个轻量的小适配器（Adapter）来学习新知识。那么问题来了——训练完之后，这个“基础模型 + 适配器”的组合体怎么变成**一个**能直接用的完整模型？

答案就是 `merge_and_unload()`。

### 一句话理解

它做两件事：

1. **合并**：把 LoRA 适配器中学到的权重变化，**加回**基础模型的原始权重里。
2. **卸载**：把 PEFT 的包装层拆掉，让模型变回一个**普通模型**。

用公式看更直观：

```text
基础模型 + LoRA适配器 → 完整模型
```

### 合并前后对比

**合并前**，你的模型是一个 `PeftModel`，内部结构是：

```text
PeftModel
├── Base Model（基础模型）
└── LoRA Adapter（适配器）
```

**合并后**，`merged_model` 变成了一个**普通 Transformer 模型**，LoRA 的更新已经“焊死”进权重里了：

```text
普通Transformer模型（包含LoRA更新后的权重）
```

---

## 完整流程：从训练到导出

整个合并导出的链路是这样的：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步走。

---

## 加载基础模型和 Adapter

假设你已经有训练好的 LoRA 适配器，现在要把它合并回基础模型。

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
base_model_path = "model_path"   # 你的基础模型路径
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 加载 LoRA 适配器
adapter_path = "adapter_path"    # 你的适配器路径
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：

```text
Base Model + LoRA Adapter
```

---

## 执行合并

一行代码搞定：

```python
merged_model = model.merge_and_unload()
```

这行代码背后的数学本质是：假设 LoRA 学到的低秩矩阵是 `BA`，基础模型原始权重是 `W`，那么合并后的新权重就是：

```text
W' = W + (α/r)BA
```

其中 `α/r` 是 LoRA 的缩放系数，合并时需按此缩放增量。

合并完成后，`merged_model` **不再是** `PeftModel`，而是一个**普通 Transformer 模型**。

---

## 导出合并后的模型

最后一步，把合并好的模型保存到磁盘：

```python
# 保存模型权重
merged_model.save_pret

## 块 57

## 3.1 merge_and_unload 方法作用——合并后必查清单

合并完模型，别急着欢呼。**“能加载”不等于“行为正确”**。下面四项检查，帮你避开“模型能跑但输出不对”的坑。

### 检查一：Tokenizer 完整且一致

保存的 tokenizer 必须和训练时**完全一致**。重点看：

- 训练中**新增或修改过的特殊 token**（比如自定义的 `<|im_start|>`、pad/eos 等），是否都在 `special_tokens_map.json` / `tokenizer_config.json` 里
- 如果训练时**扩过词表**，还要确认 embedding 和 lm_head 的尺寸与词表匹配

> 漏了这一步，推理时可能出现 token 对不上、生成乱码的问题。

### 检查二：generation_config.json

如果你依赖特定的生成默认值（比如 eos/pad token id、`max_new_tokens`、采样参数），确认它**一并导出**了。否则推理时可能回退到不合适的默认行为，生成结果和你预期不符。

### 检查三：dtype 精度

确认合并/保存时的精度符合预期——是 FP16/BF16 还是 FP32。dtype 不一致会影响：

- 显存占用
- 数值表现

### 检查四：config.json 完整性

确认模型结构配置（层数、hidden size、任务头等）正确，且**已移除 PEFT/adapter 相关配置**。合并后模型权重已包含 LoRA 增量，保存时应使用 `save_pretrained` 保存完整模型，并确认 `adapter_config.json` 已移除或不再需要。

---

### 合并后如何加载推理？

合并后，**不再需要** PEFT 库。直接这样加载：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程很简单：

```text
用户输入 → 完整模型 → 生成结果
```

---

### LoRA 合并 vs 不合并，怎么选？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter

## 块 58

**10.1 推荐合并场景（动手版）**

你学LoRA是为了微调，那什么时候该把LoRA“焊死”进模型？两个场景最典型：

**场景1：上线部署（单一Adapter、追求简单）**

想象生产环境：服务器 → 直接加载模型 → 提供API。如果只用**一个固定Adapter**，合并后部署链路最简——不额外依赖PEFT、不做运行时Adapter加载。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存

但注意：如果推理框架本身支持直接加载LoRA（如vLLM多LoRA服务），或你需要**热切换多个Adapter**，那**不合并**反而更灵活。是否合并，看框架能力和业务需求。

**场景2：分享模型**

想让对方**开箱即用**、不用理解Adapter机制？发布合并后的完整模型（merged

## 块 59

### 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个Adapter（比如A、B、C），每个负责不同任务。**保留Adapter更灵活**——想用哪个任务就切换哪个，合并反而会“焊死”它们。

**场景2：持续训练**

如果你打算继续调整LoRA，**保留Adapter**。合并后权重就固定了，再想微调就得重新来，不划算。

**实操建议**：你正在学LoRA微调，记住这个判断标准——**只要未来可能切换任务或继续训练，就别合并**。

---

### 十一、QLoRA合并注意事项（重点）

QLoRA有个关键差异：基础模型是**4-bit

## 块 60

## 11.1 直接合并可能遇到的情况

你已经走到QLoRA微调的最后一公里了。先记住一个关键认知：**4-bit状态下直接合并，不是总能成功的**。

具体来说，直接在4-bit状态下执行 `merge_and_unload`，可能出现两类问题：

- **精度损失**：4-bit权重需要先反量化再合并，这个过程与全程使用高精度计算并不完全等价，合并后的模型质量会打折扣；
- **类型或版本不兼容**：权重类型不匹配，或者你当前使用的PEFT / bitsandbytes / Transformers版本不支持在量化状态下直接合并。

> 注意：以上是**可能**出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你所用库的版本、量化配置，以及是否按下面的流程先做反量化。**以你所用版本的官方文档和实际运行结果为准。**

## 11.2 常见做法：先反量化到高精度再合并

既然直接合并有风险，稳妥的做法就是：**绕开4-bit状态**。

核心思路一句话：**用高精度重新加载基础模型，再挂载同一个Adapter，然后合并。**

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

这样合并发生在高精度权重上，从根源上避开了4-bit状态下的类型和精度问题。

下面是完整可执行代码，注意看注释里的关键点：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 模型许可证：请根据你使用的基础模型确认并遵守对应许可证（如 Llama 2 Community License，商用月活>7亿需Meta授权）

# 1) 关键：以高精度（FP16/BF16）重新加载基础模型，而不是再走 4-bit 量化
#    合并要发生在高精度权重上，所以这里不传 quantization_config
compute_dtype = torch.float16  # 默认使用 FP16；如需 BF16，请确认所用 PyTorch 版本支持相应 API
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

**代码要点拆解**（按你动手实操的顺序）：

1. **第1步是整个流程的灵魂**：重新加载基础模型时**不传** `quantization_config`，用 `torch_dtype` 指定FP16或BF16。代码里默认使用FP16，以保证兼容性；若你的环境支持BF16且库版本允许，可自行调整。

## 块 61

### 直接合并QLoRA模型：三个关键提醒

你正在用LoRA微调，目标是把训练好的Adapter合并回基础模型。对于QLoRA（4-bit量化训练），直接合并时有三个坑必须避开：

**1. 版本差异：先查文档**
不同PEFT版本对“量化模型能否直接合并”支持不同。有的版本要求先反量化，有的直接支持。**务必查阅你所用版本的`merge_and_unload`文档**。关键操作是：使用与训练时相同的量化配置加载基础模型，然后调用`merge_and_unload`进行合并。若需高精度合并，可先反量化模型再合并，但需注意显存开销。具体操作请参考PEFT官方文档。

**2.

## 块 62

## 合并时开启校验并做一致性验证（建议）

`merge_and_unload` 支持 `safe_merge=True` 参数，合并时会自动做数值检查（比如检测是否产生 NaN），相当于给合并操作加了一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，强烈建议做一次**前后对比验证**：用同一批固定输入，分别让「合并前的 PeftModel」和「合并后的模型」推理，对比两者的 logits 或生成文本是否基本一致，确认合并没有引入明显偏差。

**最小验证代码思路**：准备固定 prompt → 两个模型分别推理 → 先比较 logits 的最大绝对差 → 再比较贪心生成的文本是否一致。

**本块学习检查

## 块 63

好的，我们直接进入正题。你正在学习 LoRA 微调，并且已经走到了合并权重这一步。这一节的核心不是“怎么合并”，而是“怎么确认合并没搞砸”。对于 QLoRA 这种在低精度下操作的情况，合并后模型的表现可能会有细微变化，所以验证是必须的，不是可选项。

我们直接看代码，我会带你一步步拆解它的逻辑。

**核心思想：拿“合并前”和“合并后”的模型做对比。**

这就像你修改了一份重要文件，保存前总得检查一下改动有没有生效、有没有改错地方。这里我们对比的就是模型对同一个问题的反应。

**第一步：准备一个“合并前”的对照组**

代码里有个关键注释，我帮你划重点：

> `merge_and_unload()` 是"原地"操作，会就地修改并返回底层模型，合并后原来的 `PeftModel` 已经不再是"未合并"状态。

这意味着，你在上一节（§11.2）合并完的那个 `merged_model`，它本身已经“回不去”了。你不能拿它当“合并前”的基准。所以，代码里**重新加载了一份全新的、未合并的模型**作为对比基准：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
# 注意：Llama-2-7b FP16 需要至少14GB显存
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

这里 `model_before` 就是你的“对照组”，它代表的是“合并前”的原始状态。

**第二步：对比逻辑输出（Logits）**

模型内部对下一个词的预测分数叫 logits。如果合并操作是完美的，那么“合并前”和“合并后”的模型对同一个输入，算出来的 logits 应该几乎一模一样。

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

这段代码计算了两个模型输出 logits 的**最大绝对差**。这个数字越小，说明两个模型在“思考”层面越接近。在 QLoRA 场景下，由于精度损失，这个值通常不会精确为 0，而是一个“较小的近似值”，这是正常的。

**第三步：对比实际生成文本**

数字对比还不够直观，我们直接让两个模型“开口说话”，看看它们生成的内容是否一致。

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
```

## 块 64

## 十一、合并时开启校验并做一致性验证（建议）

### 为什么必须单独再加载一份模型？

`merge_and_unload()` 会**原地**修改模型——它把 LoRA Adapter 直接融进底层权重，并移除 PEFT 包装。这意味着合并后，原来的 `model` 变量指向的已经是“合并完成”的权重了。

如果你拿它和“合并前”的自己对比，两边其实是同一份数据，`max_abs_diff` 永远是 0，验证完全失效。

**正确做法**：在合并前，用 `base_for_ref` + `model_before` 重新加载一份**未合并**的 Adapter 作为对照基准，再和合并后的模型做对比。

### 显存不够怎么办？

如果两张模型放不下，就**分两次跑**：

1. 先用未合并的 `PeftModel` 跑一遍推理，把 logits 或生成文本**存到磁盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

### 结果预期：别追求“逐位相同”

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 不同。重点确认：**没有明显退化、没有 NaN**。
- **LoRA（非量化）场景**：一致性更高，`max_abs_diff` 可以非常小。

是否可接受，以你的**任务指标和验证集**为准，而不是纠结于

## 块 65

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

在动手合并之前，先记住一个关键原则：**QLoRA合并最容易翻车的地方，不是合并本身，而是“在4-bit状态下直接合并”**。下面直接给你一套能跑通的思路。

> **模型许可证**：本部署文档引用的基础模型需遵守其许可证条款。例如，若使用 Llama 2 系列模型，适用 **Llama 2 Community License**（商用月活用户超过7亿需获得Meta授权）。请根据实际使用的模型确认对应许可证。

#### 第一步：先重新加载高精度模型，再合并

这里的“重新加载高精度模型”不是让你手动去逐层还原4-bit权重——那太麻烦也没必要。正确做法是：

**重新加载一份不带量化配置的基础模型**，也就是用：

```python
AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16  # 或 float16
    # 注意：Llama-2-7b FP16 需要至少14GB显存
)
```

注意：**不要传** `quantization_config`，**不要传** `load_in_4bit`。这样你拿到的是高精度权重。然后再用 `PeftModel.from_pretrained` 挂载**同一份Adapter**，最后执行合并。

这样合并发生在高精度权重上，天然避开了“在4-bit状态下合并”的类型/精度问题。

#### 第二步：版本兼容性提醒

不同版本的 PEFT / bitsandbytes / Transformers，对“量化模型能否直接调用 `merge_and_unload`”的支持不一样：

- 部分版本会直接报错或给出警告；
- 部分较新版本对某些情形有一定支持，但**仍可能有精度损失**。

所以结论很明确：**“重新加载高精度模型再合并”是跨版本最稳妥的做法**。至于你的版本能不能在量化状态下直接合并，以你实际装的 `merge_and_unload` 文档和运行结果为准，别赌。

#### 第三步：合并后必查清单

合并完别急着走，对照下面几个高频坑快速自查：

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 `PeftModel`（可能忘了用 `PeftModel.from_pretrained` 挂载Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到

## 块 66

## 路线A：合并后部署——把微调成果“焊死”成完整模型

你已经掌握了 LoRA 微调的核心流程，现在到了关键的分岔口：**微调完的模型怎么上线服务？** 这一节先讲最直接的一条路——**合并后部署**。

### 核心操作：merge_and_unload()

一句话理解：`merge_and_unload()` 就是把 LoRA 的权重**物理写回**基础模型，生成一个**全新的、完整的模型文件**。之后这个模型跟 LoRA 再无关系，就是一个普普通通的模型。

```
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

### 这条路的特点

**优点（对你这种追求快速上手的风格很友好）：**

- **部署链路极简**：推理框架（比如 vLLM）把这个模型当成**普通模型**加载就行，**完全不需要理解 LoRA 是什么**。你不需要在推理端配置任何 LoRA 相关的参数。
- **推理零额外开销**：因为权重已经合并，推理时**不存在 Adapter 切换的成本**，性能最直接。

**代价（必须心里有数）：**

- **存储成本高**：每合并一个 Adapter，就要保存/分发一份**完整模型**。如果你有多个任务（比如 A/B/C 三个 Adapter），就得存三份完整模型，磁盘和分发成本直线上升。

### 什么时候选这条路？

- 你的任务**单一且固定**，不需要频繁切换不同微调版本；
- 你的推理环境**越简单越好**，不想引入任何 LoRA 相关的配置和依赖；
- 你不在乎多存几份完整模型的磁盘开销。

### 动手验证（2分钟）

在本地跑通以下流程，确认你对“合并”的直觉是对的：

1. 加载你微调好的 PeftModel；
2. 调用 `merge_and_unload()`，然后 `save_pretrained()` 保存；
3. 用 `AutoModel.from_pretrained()` **直接加载**保存出来的模型（注意：不是 `PeftModel.from_pretrained`）；
4. 跑一次推理，确认输出正常。

**观察点**：加载方式从“PeftModel 包装”变成了“普通 AutoModel”，

## 块 67

## 14.3 如何选择：合并还是保留 Adapter？

先记住一个核心认知：**合并不是唯一部署方式**。具体框架用法留到第15板块（vLLM部署）再展开，这里只帮你建立判断标准。

**三条经验参考（非硬性规则，以实际部署为准）：**

1. **单一Adapter、追求最简部署** → 倾向路线A（合并）。如果你只训练了一个LoRA，且希望部署时越简单越好，直接合并进基础模型。

2. **同一基础模型服务多个Adapter、且推理框架原生支持LoRA** → 倾向路线B（保留Adapter动态服务）。如果你要在一个模型上切换多个LoRA，且推理框架支持，就别合并，保留Adapter按需加载。

3. **QLoRA精度敏感** → 无论选哪条路线，都建议按§11.3做合并前后/上线前的一致性验证。QLoRA对精度变化更敏感，验证是必须步骤。

**你的情况**：当前学习目标是LoRA微调，处于"学习中"状态（进度85.7%）。建议先动手跑通路线

## 测评题目

### Q-KB12-ST1202-003
在其他条件相同时，增大 LoRA rank 最可能带来什么？

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。
