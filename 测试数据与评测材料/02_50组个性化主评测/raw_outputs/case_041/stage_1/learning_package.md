# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练长什么样：

```text
输入数据 → 整个模型参数更新 → 保存完整模型
```

以 **7B参数模型**（70亿参数）为例，训练时你需要同时保存三样东西：

- **参数**：模型本身的权重
- **梯度**：用于反向传播更新权重
- **优化器状态**：如Adam的动量项

这三者叠加，显存需求会**成倍飙升**。普通消费级GPU根本扛不住——这就是为什么我们需要参数高效微调（PEFT）。

---

**本块学习检查**

> **动手算一算**：假设7B模型参数占14GB显存（FP16），梯度在FP

## 块 2

**本块学习检查**：读完下表后，请用自己的话说说“全参数微调”和“LoRA”在训练参数数量上的区别，并指出QLoRA额外做了什么。

---

PEFT（参数高效微调）的核心思想是：**不修改整个大模型，只训练少量新增参数**。

流程：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter。

三种方式对比：

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**动手验证

## 块 3

**LoRA是什么？（①）**

LoRA，全称Low-Rank Adaptation（低秩适配），是一种“只训练小部分参数”的微调方法。

**核心思路：**

原模型的权重矩阵 `W` **不更新**。我们新增一个增量 `ΔW`，让实际生效的权重变成：

```text
W + ΔW
```

其中，`ΔW` 由两个低秩小矩阵相乘得到：

```text
ΔW = B × A
```

**符号约定：** 本课程统一用 `ΔW = B × A`。若原权重 `W` 形状为 `d_out × d_in`，则 `A` 形状约为 `r × d_in`，`B` 形状约为 `d_out × r`，相乘后还原成与 `W` 同形状的 `ΔW`。其他资料可能写作 `A × B`，只是顺序不同，含义一致。

**训练时：** 只训练 `A` 和 `B` 两个小矩阵，原模型 `W`

## 块 4

## 3.1 LoRA 的痛点：模型还是太大

你已经知道 LoRA 能大幅减少**训练参数**，但有个关键问题：**基础模型本身仍然要完整加载到显存里**。

以 7B 模型为例，如果用 FP16（半精度）加载，大约需要 **14GB 显存**。也就是说，即使 LoRA 只训练一小部分参数，你也得先把整个大模型塞进显存，才能开始微调。显存门槛依然很高。

---

## 3.2 QLoRA 的思路：把模型“压缩”了再训练

QLoRA 全称 **Quantized LoRA**，即“量化版 LoRA”。核心思路很简单：

```
FP16 模型 → 4bit 量化 → 显存占用大幅下降 → 加载 LoRA 进行训练
```

训练时的结构是：

```
4bit 基础模型 + LoRA 参数 → 一起训练
```

关键区别在于：**基础模型不再用 FP16 完整加载，而是压缩成 4bit 精度**。这样显存占用显著降低，让 LoRA 微调在更小的显卡上也能跑起来。

---

## 3.3 QLoRA 依赖 bitsandbytes 库

要跑 QLoRA，你需要装好这几样：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中 `bitsandbytes` 负责三件事：

- **低精度量化**
- **4bit 加载**
- **8bit 加载**

> ⚠️ **重要提醒**：

## 块 5

## 电脑配置批注：你的LoRA/QLoRA训练环境要什么？

既然目标是跑通LoRA微调，先对照你的电脑配置，心里有个数。下面是本地训练的最低门槛和推荐值。

### LoRA最低配置

- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA GPU
- **显存**：最低8GB，推荐12GB+

### QLoRA要求

**GPU**：本教程实训范围限定为NVIDIA CUDA GPU。QLoRA技术上不限于NVIDIA，但其他后端请以bitsandbytes官方说明为准，本文命令按NVIDIA CUDA给出。

**显存**：QLoRA的核心设计目标就是**在更小显存上微调大模型**——通过4-bit量化基础模型，显存占用通常低于同规模LoRA的全精度加载。但具体下限取决于多项条件，下面数值仅为**7B级别模型的经验参考**，不是硬保证：

- **参考下限**：约6~12GB（7B级别，短序列、小batch）
- **推荐**：16GB+

### 别把数值当固定门槛

实际显存占用受以下**测试条件**影响，条件不同结论就不同：

- **模型规模**：7B级别；13B/70B显存需求显著上升
- **序列长度**：短序列（如≤512）；越长激活值显存增长越明显

## 块 6

## 5.1 Python版本检查

在开始LoRA微调之前，先确认你的Python环境。打开终端，运行：

```bash
python --version
```

如果输出显示 **Python 3.10或更高版本**，就满足要求了。如果版本偏低，建议先升级Python环境再继续。

---

## 5.2 PyTorch检查（关键步骤）

LoRA/QLoRA依赖PyTorch，而且`bitsandbytes`的量化功能要求安装**与你本机CUDA匹配的GPU版PyTorch**。

> ⚠️ **常见坑**：直接`pip install torch`通常会装到CPU版，这样QLoRA会无法使用。请务必使用[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)来安装。例如NVIDIA CUDA 12.1对应：
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

安装后，用以下代码验证：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())   # 关键：应为True
print(torch.version.cuda)          # 查看torch编译用的CUDA版本
```

如果`torch.cuda.is_available()`输出`True`，说明PyTorch能识别GPU。

> **注意**：这里返回`True`只说明PyTorch侧能用GPU，**不代表bitsandbytes的4-bit量化一定能跑**。后者还需要bitsandbytes正确匹配你的CUDA版本，具体检查会在后续章节进行。

---

## 5.3 查看GPU信息

在终端运行：

```bash
nvidia-smi
```

输出会包含：
- **GPU型号**（如RTX 4090）
- **CUDA版本**（如CUDA 12.x）
- **显存大小**（如24GB）

记下这些信息，后续安装bitsandbytes时需要确认CUDA版本匹配。

---

## 六、安装PEFT库

### 6.1 PEFT是什么？
PEFT是Hugging Face官方的**参数高效微调库**，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter等多种方法。

### 6.2 安装PEFT

```bash
pip install peft
```

验证安装：

```python
import peft
print(peft.__version__)
```

能正常输出版本号即表示安装成功。

## 块 7

**7.1 bitsandbytes 是干嘛的？**

一句话：它是实现 QLoRA 的关键工具，负责把模型“压缩”到 8bit 或 4bit 精度，从而让你在有限显存下跑起大模型微调。

- **核心用途**：8bit 量化、4bit 量化、QLoRA 训练。
- **你的场景**：本教程采用 **bitsandbytes + 4bit 量化** 的 QLoRA 方案，所以它是**必须安装的前置依赖**。

> 注意：QLoRA 的思想（4bit 基础模型 + LoRA）并不强制绑定某个库，但本教程统一用 bitsandbytes + NVIDIA CUDA 演示。若你换用

## 块 8

**为什么必须检查CUDA？**

bitsandbytes是QLoRA量化的核心工具，它依赖CUDA来调用GPU计算。如果CUDA版本不匹配，你会遇到三类典型问题：CUDA不可用、量化失败、加载模型报错。所以，动手前先花1分钟确认环境。

**两步检查法（跟着做）**

1. **查CUDA版本**：终端运行`nvidia-smi`，看输出中的`CUDA Version`字段。
2. **查bitsandbytes**：运行`python -m bitsandbytes`，正常会显示CUDA版本和GPU信息。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama

## 块 9

**本块学习检查**：在终端运行 `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`，若报错提示找不到CUDA动态库，说明你正踩中问题1，请按下方建议排查。

---

你的目标是跑通LoRA微调，而QLoRA依赖bitsandbytes做量化。在Windows上，最省心的路径是：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

原因很简单：Linux环境兼容性更好，能避开Windows原生下的大部分坑。

如果你坚持用Windows原生，最可能撞见**问题1：bitsandbytes无法加载CUDA动态库**。这不是单一原因，常见的有三种：

1. 你装的bitsandbytes版本和本机CUDA或GPU版PyTorch不匹配；
2. 系统缺少或找不到对应的CUDA运行库；
3. Windows原生对CUDA动态库的支持

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配

在Windows上安装LoRA环境时，最常见的坑就是**CUDA版本不匹配**。别慌，先运行下面这条命令，看看你的机器到底什么情况：

```bash
nvidia-smi
```

这条命令会告诉你三件关键信息：

- **NVIDIA驱动**是否正常安装；
- 驱动支持的**CUDA版本**；
- 你的GPU型号。

记住：**PyTorch的CUDA版本必须不高于驱动支持的CUDA版本**，否则后面`import torch`或跑QLoRA 4-bit时直接报错。

---

## 十、安装完整LoRA环境

假设你已经有了基础环境（`transformers`、`datasets`、`accelerate`），现在要装LoRA相关的核心库。

**关键前置**：确保你已经按5.2节安装了**匹配本机CUDA的GPU版PyTorch**。比如CUDA 12.1的安装命令是：

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

然后安装LoRA两大件：

```bash
pip install peft bitsandbytes
```

一条命令搞定全部依赖：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

运行配置向导：

```bash
accelerate config
```

按你的实际环境回答。单GPU的典型配置：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

完成后会生成`default_config.yaml`，Accelerate就能正确调用你的GPU了。

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

如果成功，说明：

- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes是否就绪：

```python
import torch
import bitsandbytes

print(torch.cuda.is_available())
print(bitsandbytes.__version__)
assert bitsandbytes.cuda_setup.cuda_available
```

如果输出`True`，说明GPU环境正常，可以放心跑QLoRA

## 块 11

**本块学习检查**（动手验证，约5分钟）

在Windows上跑LoRA/QLoRA，最常踩的坑就是兼容性。记住一句话：**别在原生Windows硬刚，直接用WSL2**。

**为什么？** 表格里写得很清楚——`Windows报错`的根源就是`兼容性问题`，官方给的解法就是`使用WSL2`。这不是可选项，是推荐路径。

**你的操作清单**（按顺序过一遍）：

1. **装环境前先查CUDA**：如果`bitsandbytes加载失败`，别瞎猜，先查`CUDA版本`是否匹配。不匹配就对齐版本。
2. **确认PyTorch是CUDA版**：如果报`CUDA unavailable`，十有八九装成了CPU版PyTorch。直接`重新安装CUDA版PyTorch`，别在CPU版上浪费时间。
3. **显存不够怎么办**：模型太大跑不动，就`降低batch`或`换更小模型`，这是最直接的解法。
4. **Windows报错**：别折腾原生环境，直接`使用WSL2`。

**为什么这套流程对你有效？** 你的目标是快速上手LoRA微调，每天2小时。与其在Windows兼容性上耗时间，不如一步到位用WSL

## 块 12

## 一、为什么需要LoRA？——先动手感受一个矛盾

假设你现在拿到一个7B模型（比如LLaMA、Qwen、Baichuan或Mistral），想让它学会一个新任务。传统做法是：把整个模型的所有参数都更新一遍，然后保存一份全新的模型。

**请你现在做一个快速估算**：7B = 70亿参数。如果每个参数用16位浮点数存储，仅模型权重就需要约14GB显存。但训练时你还要额外保存梯度（∂W）和优化器状态（如Adam的动量项），实际显存需求会**远大于模型本身的大小**，可能超过28GB。这就是全参数微调的第一个痛点——硬件门槛极高。

第二个痛点是**存储成本**：每微调一个任务，你就要保存一份完整的7B模型副本。微调10个任务，就是10份70亿参数的模型。

第三个痛点是**时间成本**：更新全部参数意味着每一轮训练都要计算和回传所有参数的梯度，训练周期被显著拉长。

---

## 二、LoRA的核心思想——用“小改动”代替“大重写”

LoRA的出发点是一个关键观察：**预训练模型已经学到了通用的语言能力，微调时不需要大动干戈地重写所有参数**。

它采用的做法是：**冻结原始模型参数W不动**，只在旁边增加一个小的“补丁”模块。这个补丁用两个低秩矩阵的乘积再乘以一个缩放因子来表示，即：

**ΔW = (α/r) × B × A**

其中：
- **A** 是一个低秩矩阵

## 块 13

**本块学习检查**：读完本块后，请合上资料，用一句话回答：LoRA 微调时，原模型权重 W 是被修改了，还是被冻结了？最终模型用的是哪个公式？

---

**2.1 核心：不碰原模型**

传统微调是直接改原权重：`W → W'`。

LoRA 的做法完全不同：

1. **冻结**原权重 `W`，一个字都不改；
2. **新增**一个参数 `ΔW`；
3. 最终模型用的是：`W' = W + (α/r)BA`，其中 `ΔW = (α/r)BA`，A和B为低秩矩阵。

也就是说，`W` 保持不变，我们只训练 `ΔW`。

---

**2.

## 块 14

## 3.1 矩阵基础概念

Transformer里大量参数都是矩阵。以注意力层为例，`Q = X · Wq`，其中`Wq`就是一个权重矩阵。假设`Wq`形状为`4096 × 4096`，参数数量就是`4096 × 4096 ≈ 1677万`。

## 3.2 直接训练矩阵的问题

如果直接修改`ΔW`，仍然需要`4096 × 4096`个参数。LoRA的思路是：**不直接学习完整矩阵，而是拆成两个小矩阵**。

## 四、LoRA低秩分解数学原理

LoRA核心公式：原始权重`W`保持冻结，新增的`ΔW`由`B × A`组成。这样就把一个大矩阵的更新拆解为两个小矩阵的乘积，大幅减少需要训练的参数数量。

---

**本块学习检查**：请回答——`Wq`是`4096 × 4096`矩阵时，直接训练`ΔW`需要多少参数？LoRA用`B × A`替代`ΔW`的核心思路是什么？

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设原矩阵是 **4096 × 4096**。直接训练它，参数量是：

`4096 × 4096 = 16,777,216`（约1677万）。

用LoRA，设 **rank r = 8**，把这个大矩阵拆成两个小矩阵：

- **A**：`8 × 4096`，参数量 = 32,768
- **B**：`4096 × 8`，参数量 = 32,768

A和B的参数量之和为 **65,536**（约6.5万）。

**对比一下：**

| 方式 | 参数数量 |
|

## 块 16

**5.1 为什么低秩分解有效？**

微调大模型时，我们其实是在调整权重矩阵，产生一个变化量 ΔW。LoRA 的核心洞察是：**这个 ΔW 不需要在所有方向上都有分量**。

想象一个 4096 维的空间，真正影响任务效果的方向，往往只集中在一个**远小于原维度的低维子空间**里。换句话说，权重更新的“有效内在维度”很低。

> 注意：这是 LoRA 的一个**假设**，具体“有效维度”是多少没有固定值，会随模型、任务、数据变化。这里只需定性理解，别记成“几十维”这种具体数字。

正因为 ΔW 集中在低维子空间，我们才能用两个低秩矩阵 **BA** 去近似它，从而大幅

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张高清图片，尺寸是 **4000×4000像素**。如果要把每个像素都完整保存下来，需要存储 **1600万个数据**——这就像大模型里完整的权重矩阵，参数数量巨大。

但关键洞察是：这张图片的**主要变化**往往很有限。比如只是**亮度**调了调、**对比度**改了改、或者有**少量结构变化**。这些变化完全可以用**少量参数**来描述，而不需要重新保存整张1600万像素的图。

LoRA的核心思想正是如此：**冻结原始权重W，仅训练一个低秩的增量矩阵ΔW=(α/r)BA，其中A和B为低秩矩阵，从而用少量参数描述模型权重的变化方向**。在微调时，我们不需要更新整个巨大的权重矩阵，只需要学习一个低秩的变化量，这个变化量用少量参数就能表达。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构是：

```
Input → Embedding → Attention → FFN → Output
```

其中Attention模块包含 **Q、K、V、O** 四个投影矩阵。LoRA早期实践中**常见**的做法是只加到：

- `q_proj`（Q投影）
- `v_proj`（V投影）

```python
target_modules=["q_proj", "v_proj"]
```

这表示只训练Wq和Wv对应的低秩增量矩阵A和B，原始权重Wq和Wv被冻结。

> ⚠️ **重要说明**：只挑 `q_proj`/`v_proj` 是LoRA原论文的**常见选择**，**不是固定标准**。现代实践经常把LoRA加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

**实操示例**——覆盖更完整线性层的配置（以Llama类模型为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认）：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "

## 块 18

## 8.1 rank（r）：低秩矩阵的“宽度”旋钮

在 LoRA 里，`r` 代表低秩矩阵的维度。你可以把它想象成“旁路”的宽度：`r` 越大，旁路能承载的信息越多，模型表达能力越强；但代价是参数更多、显存占用更高。

**怎么选起点？** 下表是经验起始范围，不是铁律：

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

**但更可靠的做法是：用验证集来选 `r`，而不是背配方。**

**操作流程（5步）：**

1. **划验证集**：从数据中留出 8:2 或 9:1 的验证集，后续比较都看它的指标。
2. **固定其它变量**：锁定 `target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数，只改 `r`。
3. **小范围搜索

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

在代码中你会看到这样的参数：

```python
lora_alpha=16
```

这个参数的作用是**控制LoRA更新强度**。简单理解：它决定了微调时对原模型权重的“改动幅度”有多大。

实际使用中，通常用 **`alpha / r`** 作为缩放比例。比如 `r=8`、`alpha=16` 时，缩放比例为2，意味着LoRA的更新会被放大2倍。

**动手验证**：你可以试着把 `alpha` 从16改成32，观察训练时损失下降的速度变化——`alpha` 越大，参数更新越激进，但过大可能导致训练不稳定。

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

这个参数的作用是**防止LoRA参数过拟合**。它和普通神经网络中的dropout原理一致：训练时随机“丢弃”一部分LoRA参数，迫使模型学习更鲁棒的特征。

**动手验证**：在数据量较小的微调任务中，把 `lora_dropout` 从0调到0.1，对比验证集表现——通常能看到过拟合现象有所缓解。

---

## 九、LoRA与全参数微调的区别

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> **注意**：上表是**相对趋势**，不是绝对数值

## 块 20

**本块学习检查**：读完本节后，请你在纸上画出 `r=8` 时 `A` 和 `B` 两个矩阵的尺寸，并标出它们相乘后恢复出的 `ΔW` 的维度。画完再继续往下读。

---

## 二、Rank（r）是什么？

### 2.1 Rank 表示什么？

Rank 的中文意思是**矩阵的秩**。但在 LoRA 里，你可以更直观地把它理解为：**低秩矩阵中间那个“瓶颈”的维度大小**。

我们用一个具体例子来看。假设原始权重矩阵 `W` 的尺寸是：

```text
4096 × 4096
```

如果设置 `r=8`，那么 LoRA 引入的两个低秩矩阵就变成了：

- 矩阵 `A` 的尺寸：`8 × 4096`
- 矩阵 `B` 的尺寸：`4096 × 8`

注意看：`A` 和 `B` 的中间维度都是 8——这个 8 就是 rank 的值。

最后，`ΔW = B × A`，也就是：

```text
(4096 × 8) × (8 × 4096) = 4096 × 4096
```

你看，`ΔW` 的尺寸被恢复成了和原始 `W` 一样的大小。

**一句话总结**：rank 决定了 LoRA 在“压缩”和“恢复”之间那个中间通道有多宽。rank 越大，这个通道越宽，能承载的信息越多，但需要训练的参数也越多、成本越高。

## 块 21

**Rank控制什么？——用“自由度”来理解**

Rank（r）决定了LoRA能学习多少种“变化方向”。你可以把它想象成模型可用的“旋钮数量”：旋钮越多，能调出的花样就越多。

**低rank（如 r=4）**：模型只能学习较少的变化模式。好处是参数少、显存低、训练快；坏处是表达能力有限，复杂任务可能学不好。

**高rank（如 r=64）**：LoRA拥有更多自由度，能学习更复杂任务、适应能力更强。代价是参数更多、显存增加、训练成本提高。

**动手验证**：如果你在微调一个简单分类任务，先用 `r=4` 跑一轮，再改用 `

## 块 22

**3.1 不使用LoRA：全量训练的成本**

假设模型某层权重矩阵为4096×4096，完整训练该单层权重矩阵需更新：

`4096 × 4096 = 16,777,216`，约**1677万参数**（此为单层权重矩阵的参数量，非整个模型总参数量）。

---

**对比**

全量训练需更新单层全部1677万参数；而LoRA不直接更新这个大矩阵，而是训练两个小矩阵，参数量公式：

`参数量 = r × (输入维度 + 输出维度)`

- **r=8**：`8 × (4096+4096) = 65,536`（约6.5万）
- **

## 块 23

### 4.1 Rank太小的问题

假设你把 `r` 设成 `2`，会发生什么？

**核心问题**：模型能学的“新东西”太少。

- **学习能力不足**：LoRA 的增量矩阵太小，装不下新任务需要的知识。
- **欠拟合**：模型在训练集上就学不好，更别提泛化。
- **新任务知识无法充分表达**：想学的东西“塞不进去”。

**典型表现**（你会在训练日志里看到）：

```text
训练loss下降慢
效果提升有限
```

**直观理解**：这就像用一张很小的便签纸去记一整本笔记——不是不想记，是纸不够大。

---

### 4.2 Rank太大的问题

反过来，如果你把 `r` 设成 `256` 呢？

**核心问题**：参数变多，成本变高，LoRA 的优势被削弱。

- **参数增加**：要训练和存储的参数量变大。
- **显存增加**：GPU 显存占用上升。
- **训练速度下降**：计算量变大，跑得更慢。

**关键点**：LoRA 的核心优势是“参数高效”。`r` 越大，这个优势越小。

> **用数字看清楚（避免误解）**：
> 以一个 `4096 × 4096` 的权重矩阵为例：
> - 完整矩阵参数量：`4096 × 4096 = 16,777,216`
> - LoRA 增量参数量：`r × (4096 + 4096)`
>
> | rank r | LoRA 参数量 | 占完整矩阵比例 |
> | ---: | ---: | ---: |
> | 8 | 65,536 | ≈ 0.39% |
> | 64 | 524,288 | ≈

## 块 24

**4.3 常见rank选择（动手起点）**

先记住：下表只是**经验起始范围**，不是自动调参标准。

| 任务类型 | rank起点参考 |
|---------|------------|
| 简单分类 | 4~8 |
| 文本风格调整 | 8~16 |
| 指令微调 | 16~64 |
| 复杂领域任务 | 64以上 |

实际调参要看：数据量、模型大小、序列长度、batch大小（含梯度累积）、目标模块选择与数量、GPU显存、验证集表现。**同一个r在不同模型规模、序列长度/batch、目标模块下，显存和效果差别很大**，别直接照搬。

**五、Alpha（lora_alpha）是什么？**

LoRA不仅做 `ΔW = BA`，还会缩放：

```
W' = W + (α/r)BA
```

- α：lora_alpha
- r：rank

所以真正加入

## 块 25

### 6.1 控制LoRA影响强度（动手版）

我们直接通过改数字来感受 `alpha` 的作用。假设你已经设好 `r=8`：

- **放大效果**：把 `alpha` 设为 `16`，缩放系数就是 `16/8=2`。这意味着LoRA的更新量会被**放大2倍**，模型改动更明显。
- **保持原样**：把 `alpha` 设为 `8`，缩放系数是 `8/8=1`，更新量保持原始比例，不加不减。
- **减弱效果**：把 `alpha` 设为 `4`，缩放系数是 `4/8=0.5`，LoRA的影响就**减半**了。

**核心公式**：LoRA加入模型的更新量会被 `alpha/r` 这个系数缩放。所以，`alpha/r` 是控制LoRA影响强度的**关键缩放系数**。

**但注意**：`alpha/r` 不是唯一决定实际强度的开关。实际落到权重上的更新量还受这些因素影响：
- **学习率**：决定每步对 `A`、`B` 参数的更新步长。`alpha/r` 调大时如果学习率不变，容易放大更新导致震荡。实践中常通过调学习率来控制强度，而不是频繁改 `alpha`。
- **初始化**：常见实现里 `A` 用小随机值、`B` 初始化为0，训练开始时 `ΔW=BA=0`，不扰动原模型。初始化方式影响早期训练动态。
- **训练状态**：`ΔW` 是逐步累积的。同样的

## 块 26

**本块学习检查**：  
对照配置1（r=8, alpha=16）和配置2（r=16, alpha=32），两者缩放比例均为2。请动手验证：将alpha改为8（r保持8），比例变为1，观察与alpha=16时的差异——这能帮你直观感受“同步调整”的意义。

---

**正文**：  
先看两个配置：`r=8, alpha=16` 和 `r=16, alpha=32`。虽然rank不同，但 `alpha/r` 都是2，缩放比例相同。  

调整rank时，**常见做法是同步调整alpha**，保持比例大致稳定——这只是经验做法，非硬性规则。`alpha = 2 × r`

## 块 27

### 方案1：小模型实验（你的起点）

既然你目标是快速上手LoRA微调，且偏好动手实践，我们从最轻量的配置开始：

```python
r=8
alpha=16
```

**为什么先选它？** 参数少、训练快，特别适合用来跑通流程、验证想法。先用小模型把整个微调链路走一遍，比一开始就上大参数更高效。

---

### 实际代码对照

在PEFT中，这个方案对应如下配置：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

拆解来看：
- **r**：控制LoRA容量（即低秩矩阵的维度）
- **lora_alpha**：控制LoRA更新幅度（缩放系数）
- **target_modules**：控制LoRA插入位置（这里选的是注意力层的q和v投影）

---

###

## 块 28

## 误区1：rank越大一定越好？

**错误。**

很多初学者觉得rank设大点，模型能力就强。但任务简单时，大rank只会：

- 浪费显存
- 增加训练成本

**实操建议**：先从小rank（如8）开始，任务效果不够再往上加。

---

## 误区2：只调rank，不调alpha

很多人改rank时忘了同步调alpha。看这个例子：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

比例变成 `16/64=0.25`，LoRA的更新强度反而被削弱了。

**关键**：`α/r` 的比值决定LoRA对原模型的影响力度。改rank时，alpha要跟着调整，保持合理比例。

---

## 误区3：alpha越大越好？

**错误。**

alpha过大（即 `α/r` 缩放过大）会导致：

- 更新过强
- 训练不稳定

**注意**：`α/r` 过大只是训练不稳定的**可能原因之一**。训练震荡还常来自学习率过高、数据质量差、batch size过小、精度问题、缺少梯度裁剪等。排查不稳定时，要综合检查这些因素，别只盯着alpha。

---

## 核心公式

```
W' = W + (α/r)BA
```

**一句话理解**：

- `r` 决定LoRA容量（能学多少）
- `α` 通过 `α/r` 缩放影响更新强度（学多猛）

---

## 本块学习检查

**动手验证**：打开你的LoRA配置文件，分别计算 `r=8, alpha=16` 和 `r=64, alpha=16` 两种情况下的 `α/r` 比值，写下你的发现。

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。在 LoRA 微调流程中，`LoraConfig` 是核心配置对象，它告诉 PEFT 四件事：**LoRA 放在哪里、规模多大、训练强度如何、是否使用 dropout**。

LoRA 训练流程如下：

```
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

其中，`LoraConfig` 就是流程第二步的“总指挥”。你不需要手动修改模型内部结构，只需通过这个配置对象声明需求，PEFT 会自动完成 LoRA 的插入与训练准备。

---

### 动手理解：先跑通一个最小配置

打开你的 Python 环境，运行以下代码（不需要 GPU，CPU 即可）：

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

你会看到配置对象被打印出来。现在对照上面的“四件事”逐一观察：

- `target_modules` 回答“放在哪里”——这里指定了 `q_proj` 和 `v_proj` 两个模块；
- `r` 回答“规模多大”——值为 8，代表低秩分解的秩；
- `lora_alpha` 回答“训练强度如何”——值为 16，控制更新强度；
- `lora_dropout` 回答“是否使用 dropout”——值为 0.1，表示 10

## 块 30

## 1.2 一个简单示例——先跑通，再理解

你目标明确：**LoRA微调**。咱们直接上手，看一段最精简的配置代码：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

这段代码只做一件事：**告诉PEFT库“你想怎么微调”**。四个参数就是LoRA的核心开关：

| 参数 | 作用 | 直观理解 |
|------|------|----------|
| `target_modules` | 决定插入哪些层 | 改哪几块“积木” |
| `r` | 决定LoRA矩阵大小 | 每块积木改多细 |
| `lora_alpha` | 控制缩放强度 | 改动幅度多大 |
| `lora_dropout` | 控制随机失活 | 防过拟合的随机丢弃 |

**动手验证**：把上面代码复制到Python环境跑一遍，打印`config`看看输出——你会看到每个参数都被记录下来了。

---

## 二、LoRA配置整体流程——三步走

配置不是孤立的，完整流程是：

```text
model → Lora

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入低秩矩阵”**。

### 为什么需要手动指定？

Transformer 的 Attention 结构通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA **不会默认修改所有层**，你必须显式告诉它改哪些。例如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的含义是：**只给 Query 投影层和 Value 投影层增加 LoRA**。

---

### 为什么通常选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开看两个关键角色：

- **Q（Query）**：决定模型“**关注什么信息**”
- **V（Value）**：决定模型“**传递什么信息**”

选择 q_proj 和 v_proj 是常见实践，但并非唯一选择，实际效果需实验验证。

---

### 两个必须注意的点

**1. 先查模块名，再配置**

`q_proj` / `v_proj` 只是常见命名，**不是所有模型都通用**。不同模型的模块命名可能不同，配置前务必先用 `model.named_modules()` 查看实际模块名，再决定 `target_modules`。

**2. 偷懒写法：`"all-linear"`**

较新版本的 PEFT 支持直接写：

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

`"all-linear"` 会自动把 LoRA 加到模型中（几乎）所有线性层，省去手工枚举。但注意：**覆盖模块变多 → 可训练参数和显存占用增加**，需要在“参数量/显存”和“效果”之间权衡，最终以验证集结果为准。

如果你的 PEFT 版本较旧不支持该字符串，就退回**显式列出模块名**的写法。

## 块 32

### 方案1：只调整Q和V（轻量起步）

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么先试这个？** 因为参数少，训练快。这是最常见的轻量起点，适合你快速跑通LoRA微调流程、验证效果。具体效果仍要看模型结构与任务表现，但作为第一步，它成本最低。

**动手建议**：先用这个配置跑一次小规模训练，观察loss下降速度和最终效果，再决定是否升级到方案2或3。

---

### 对比：方案2与方案3（何时升级）

**方案2：调整全部Attention**，加上`k_proj`和`o_proj`：

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj"
]
```

参数更多，表达能力更强

## 块 33

## LLaMA 系列：target_modules 怎么设？

LLaMA 系列模型（以及大多数现代开源大模型）的注意力层中，Q、K、V 的投影是**分开的独立模块**，分别叫 `q_proj`、`v_proj`（还有 `k_proj`、`o_proj` 等，但 LoRA 最常作用于前两者）。

**最常见的配置就是：**

```python
target_modules = ["q_proj", "v_proj"]
```

**为什么只选这两个？** 直观理解：Q（查询）和 V（值）投影是注意力机制中信息“入口”和“出口”的关键通道，对它们做低秩适配，能以最少参数撬动最大的行为改变。K（键）投影和 O（输出）投影也可以加，但实践中先试 `q_proj` + `v_proj` 是性价比最高的起点。

---

## 对比：GPT-2 为什么写法不一样？

GPT-2 的注意力层**不是**分开的 `q_proj`/`v_proj`，而是把 Q/K/V 三个投影**合并**在一个叫 `c_attn` 的模块里。所以它的 `target_modules` 要写成：

```python
target_modules = ["c_attn"]
```

### ⚠️ 关键坑：`fan_in_fan_out=True`

GPT-2 用的不是普通 `nn.Linear`，而是 Transformers 里的 `Conv1D`。它的权重形状是 `(in_features, out_features)`——**和 `nn.Linear` 的 `(out_features, in_features)` 正好转置**。

如果不告诉 PEFT 这个差异，低秩矩阵注入的位置就会错。解决办法是显式设置：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
# 注意：Llama-2-7b FP16 需要至少14GB显存。

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL

## 块 34

## BERT 的 target_modules 配置：动手查，别照抄

你正在为 LoRA 微调配置 `target_modules`，目标很明确：**让 LoRA 只作用于注意力层的 query 和 value 投影**。但 BERT 和 LLaMA 这类模型的模块命名**不一样**，所以**不能直接复制**别人的配置。

**核心操作：用代码查你实际加载的模型**

```python
# 注意：Llama-2-7b FP16 需要至少14GB显存
for name, _ in model.named_modules():
    print(name)
```

运行后，你会看到类似 `bert.encoder.layer.0.attention.self.query` 这样的输出。找到以 **`query`** 和 **`value`** 结尾的模块名，然后这样配置：

```python
target_modules = ["query", "value"]
```

**关键机制（务必理解）**：

- `target_modules` 里的字符串是**按模块名做匹配**的，PEFT 通常支持**后缀匹配**。写 `"query"` 会命中所有以 `query` 结尾的层（各层各头都会生效）。
- 你也可以传更完整的模块名做精确定位，或使用正则表达式。
- **版本差异**：不同 Transformers/PEFT 版本、不同模型实现，

## 块 35

**r是什么？——LoRA的“瘦身”旋钮**

r就是rank（秩），它控制LoRA低秩矩阵的维度。LoRA的核心公式是：

```text
ΔW = (α/r)BA
```

这里的r，决定了A和B矩阵的中间维度大小，α为缩放系数，且原权重W被冻结。

**直观理解：**

假设原权重矩阵是 `4096×4096`，如果设置 `r=8`，那么：

- A矩阵是 `8×4096`
- B矩阵是 `4096×8`

这样，原本需要训练 `4096×4096` 个参数，现在只需训练 `8×4096 + 4096×8` 个参数，大幅减少

## 块 36

## 7.1 alpha作用：控制LoRA的“力度”

LoRA最终更新时，**不是**直接使用 `BA`，而是使用：

```text
(alpha/r)BA
```

这里的 `alpha/r` 就是**缩放系数**，它直接控制 **LoRA更新幅度**——即微调时对原模型权重的“改动力度”。

**动手算一算（hands-on）：**

假设你配置：

```python
r=8
lora_alpha=16
```

那么缩放系数为：

```text
16/8 = 2
```

这意味着LoRA的影响被**增强**了2倍——改动幅度更大，模型适应新任务的速度可能更快，但也可能更易过拟合。

---

### 7.2 alpha常见设置：一个经验起点

一个常见的**起点**是：

```

## 块 37

## 8.1 dropout作用

训练时，LoRA会对低秩矩阵（A或B）的元素以概率p随机置零，防止模型过度依赖某些特征，减少过拟合。

配置示例：

```python
lora_dropout=0.05
```

含义：**5%的LoRA路径随机失活**，让模型不能过度依赖某几条路径。

## 8.2 如何选择dropout？

这是**经验起点**，不是固定规则，要结合验证集是否过拟合来调：

| 数据规模 | dropout（起点参考） |
|---------|-------------------|
| 数据少   | 约 0.05~0.1       |
| 数据多   | 约 0~0.

## 块 38

## 9.1 基础配置：动手搭一个LoRA

这段配置适合**7B模型测试**，也是你上手LoRA微调的第一个完整示例。直接看代码：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**逐个参数拆解，边看边想：**

| 参数 | 值 | 作用 | 你的理解检查 |
|------|-----|------|-------------|
| `r` | 8 | LoRA容量（低秩矩阵的秩） | 秩越大，可学习参数越多，但微调成本越高 |
| `lora_alpha` | 16 | 更新缩放（缩放系数） | 控制LoRA更新的强度，通常设为r的2倍 |
| `target_modules` | `["q_proj", "v_proj"]` | 插入位置（注入哪些层） | 只改注意力层的Q和V投影，其他层不动 |
| `lora_dropout` | 0.05 | 防止过拟合 | 训练时随机丢弃5%的LoRA参数 |
| `task_type

## 块 39

## 9.1 基础配置：把分类头也“锁进”LoRA训练

你正在做 LoRA 微调，目标是让 BERT 学会二分类。下面这段代码是完整流程的“地基”，请跟着动手敲一遍，重点观察 **`modules_to_save`** 这个参数——它是本块最容易出错的地方。

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 1) 加载带分类头的基础模型（分类头是新初始化的，需要一起训练/保存）
# 模型许可证：BERT 采用 Apache License 2.0，可商用，无月活限制
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

# ...（此处进行训练，略）...

# 2) 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
try:
    model.save_pretrained("./lora_ckpt")
except OSError as e:
    print(f"保存失败: {e}")
    raise

# 3) 重新加载并验证分类头确实被带回（而不是随机初始化）
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
try:
    reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")
except OSError as e:
    print(f"加载失败: {e}")
    raise

# 验证：重载后的分类头权重应与保存前一致（说明 classifier 被 modules_to_save 正确保存/恢复）
w_before = model.classifier.weight.detach().cpu()
w_after = reloaded.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

### 为什么 `modules_to_save` 是“关键”？

- 你的 BERT 分类头是**新初始化的**。LoRA 默认冻结原有权重，仅训练低秩矩阵 A 和 B，形成增量 ΔW=(α/r)BA，只作用于 `target_modules` 指定的注意力层（这里是 `query` 和 `value`）。如果不加 `modules_to_save=["classifier"]`，分类头**不会参与训练**，保存时也不会被带走。
- 后果：重载模型时，分类头会退回基础模型的**随机初始化**状态，分类结果直接错乱。
- 加上它之后，分类头**参与训练并随 Adapter 一起保存**。

### 三个必须记住的要点

1. **保存内容**：`save_pretrained` 保存的是 **Adapter + `modules_to_save` 模块**，不是完整基础模型。重载时需先加载**同一个**基础模型（`bert-base-uncased`），再用 `PeftModel.from_pretrained` 挂载。
2. **验证方法**：上面用 `torch.allclose` 对比“保存前 / 重载后”的分类头权重来验证它确实被保存/恢复。实际项目里也可以用一批固定输入对比 logits 是否一致。
3. **属性名差异**：`get_peft_model` 返回的模型对象中，分类头可直接通过 `model.classifier` 访问，无需经过 `base_model.model`

## 块 40

## 10.1 普通文本生成配置

对于普通文本生成任务（如续写、故事生成等），推荐使用以下LoRA配置：

```python
r=8
alpha=16
dropout=0.05
```

**参数解读：**

- **r=8**：低秩矩阵的秩。普通文本生成任务相对简单，较小的r足以捕捉任务所需的适配信息，同时保持参数效率。
- **alpha=16**：缩放因子。alpha是r的2倍，这个比例（α/r=2）是实践中常用的默认缩放，既能保证微调效果，又不会因缩放过大导致训练不稳定。
- **dropout=0.05**：LoRA层的丢弃率。5%的丢弃率提供了适度的正则化，防止过拟合，同时不会因丢弃过多而影响模型学习能力。

**对比其他任务：** 指令微调推荐r=16、alpha=32（需要更强适配能力）；领域知识增强推荐r=32、alpha=64（需要更大容量吸收领域知识）。普通文本生成选择较小的r，是因为生成任务与预训练目标接近，不需要大幅调整模型行为。

---

## 动手验证：配置并检查

打开你的Python环境，运行以下代码确认配置生效：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
print(config)
```

观察输出中`r`、`lora_alpha`、`lora_dropout`字段是否与你设置的参数一致。这是确认配置正确的最快方式。

---

## 扩展边界提醒

本文聚焦经典LoRA核心参数。`LoraConfig`还支持`use_rslora`、`use_dora`、`rank_pattern`等扩展能力，这些行为随PEFT版本变化，使用前务必查阅对应版本官方文档。

---

## 硬件要求

- **仅运行配置代码**：CPU即可，内存4GB+。
- **加载真实模型训练**：最低NVIDIA CUDA GPU、显存8GB+；推荐16GB+。若显存不足，可考虑QLoRA（4-bit量化），7B级别模型常见参考约8GB起。
- **模型许可证**：若使用Llama 2等模型，需遵守其许可证（如Llama 2 Community License，商用月活>7亿需Meta授权）。

---

## 常见问题速查

| 问题 | 排查方向 |
|------|

## 块 41

### 核心配置：普通文本生成的LoRA设置

直接上手，先看这段配置：

```python
LoraConfig(
    r=8,               # LoRA容量：决定低秩矩阵的维度
    lora_alpha=16,     # 缩放因子：通过 α/r = 2 影响更新强度
    target_modules=["q_proj", "v_proj"],  # 决定改哪里：只微调注意力层的Q和V投影
    lora_dropout=0.05  # 缓解过拟合，影响泛化
)
```

**四个参数速记**：
- `target_modules`：改哪里——指定要注入LoRA的模块
- `r`：容量——秩越高，可学习参数

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思路：冻结原模型，只训练少量低秩矩阵。但这里有个关键点——**LoRA只是减少了“训练”的参数，并没有减少“加载”的负担**。

用一句话概括：**基础模型仍然需要完整加载到显存中**。

我们来算一笔账。以7B模型为例，如果用FP16精度加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是模型权重本身。实际训练时，你还需要额外空间存放：

- **激活值**：前向传播过程中的中间结果；
- **梯度**：反向传播时计算出的更新方向；
- **优化器状态**：如Adam的动量项和

## 块 43

**QLoRA核心思想：把大模型“瘦身”后微调**

你的目标是LoRA微调，但直接微调大模型太吃显存。QLoRA的思路很直接：**先把基础模型压缩成4-bit，再在上面跑LoRA**。

**流程对比**

- **LoRA**：FP16模型 + LoRA（模型大，显存压力大）
- **QLoRA**：4-bit模型 + LoRA（模型小，显存友好）

**执行步骤**：FP16基础模型 → 4-bit量化 → 加载量化模型 → 插入LoRA → 训练LoRA参数。

**关键区别**：量化只针对**基础模型权重**（冻结不动），而**LoRA参数**保持

## 块 44

## 3.1 模型存储方式：从“重量”看量化

先看一张“体重表”——每个参数占多少字节：

| 类型 | 每个参数占用 |
|---|---|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下**：一个7B（70亿参数）模型，用FP16存，需要 `7B × 2 ≈ 14GB`；换成4-bit，只需 `7B × 0.5 ≈ 3.5GB`，显存直接**减少约75%**。

> ⚠️ **注意**：这3.5GB是**理论下界**，只算了量化后的权重本身。实际跑起来会更高，因为还要存量化常数、分组元数据（NF4按组量化）、未量化的模块（norm、embedding

## 块 45

**NF4是什么？——给“动手派”的直观解释**

NF4全称NormalFloat4，中文叫“正态分布4位浮点”。但别被名字骗了，它**不是**IEEE 754那种通用浮点格式，而是QLoRA论文为神经网络权重“量身定制”的4-bit量化码本。

**核心逻辑（三步走）：**

1. **预置码本**：因为神经网络权重近似正态分布（0附近密集），NF4预先定好16个（2⁴）量化取值点，这些点按标准正态分布的分位数排布——**0附近分得更细**，保证权重密集区精度高。
2. **量化映射**：把每组权重归一化后，映射到码本里最接近的那个值，只存对应的4-bit索引（省内存的关键

## 块 46

### 为什么NF4适合模型权重？

大模型训练完的权重，有个关键特征：**近似正态分布**——大量参数挤在0附近，越往两边越少。

普通INT4量化，是把范围**平均划分**（比如 -8 到 7 等分）。这导致中心密集区（0附近）分到的档位太少，精度不够。

NF4的思路是：**按正态分布来设计量化范围**。0附近分配更多档位，精度更高；两边稀疏区域少分点。结果就是——**更贴合权重真实分布，精度损失更低**。

### NF4量化过程（速览）

原始FP16权重（如 0.023, -0.145, 0.876）→ 归一化 → 映射到NF

## 块 47

**本块学习检查**：读完5.1-5.3后，请用一句话向自己解释：双重量化到底“双”在哪两次量化上？能说清就算过关。

---

**5.1 为什么需要双重量化？**

做4-bit量化时，不光要存压缩后的权重，还得额外存一组“量化参数”——每组权重都要配一个`scale`（缩放系数）和一个`zero point`（零点偏移）。这些参数本身也占显存，模型越大，这笔开销越不可忽视。

**5.2 双重量化思想**

普通量化是：FP16权重 → 4-bit权重 + 存scale参数。

双重量化则再进一步：把存下来的scale参数**也做一次量化**

## 块 48

**6.1 为什么需要分页优化器？**

训练时显存峰值不只来自模型，还有**梯度**和**优化器状态**。比如Adam要保存**一阶动量**和**二阶动量**。QLoRA虽降低了模型显存，但训练中仍可能**显存突然不足**。

**6.2 分页优化器思想**

Paged Optimizer借鉴操作系统虚拟内存，核心是：借助NVIDIA统一内存（Unified Memory），在显存出现峰值、即将OOM时，把优化器状态自动"分页"换出到CPU内存，需要时再换回GPU。

流程：GPU显存 → 出现峰值 → 优化器状态按页换出到CPU内存 → 需要计算时换回GPU。

注意：这不是"平时就把优化器状态常驻CPU"。重点是**应对显存峰值**：在显存不足的瞬间临时换出，避免偶发峰值直接OOM中断训练；平稳时状态仍留在GPU。它解决的是"峰值OOM"，而非常态化省显存。

**本块学习检查**：用自己的话复述"分页优化器"与"把优化器放CPU"的本质区别

## 块 49

### 类比理解：分页优化器在做什么？

把显存想象成一张**桌面**，模型训练时的临时数据（优化器状态）就是桌上的文件。

- **普通方式**：所有文件都摊在桌面上。桌面一旦堆满，你就没法再放新文件了——对应训练时显存爆掉（OOM），直接报错中断。
- **分页优化器**：桌面只放**正在用的常用文件**，暂时不用的文件自动挪到旁边的**柜子**里，需要时再取出来。这样桌面永远有空间放新文件，训练就能继续跑下去。

这个“桌面↔柜子”的换页动作，由 bitsandbytes 库自动完成，你不需要手写任何换页逻辑。

---

### 如何在代码里启用分页优化器？

在 `TrainingArguments` 里通过 `optim` 参数指定即可（关键就一行）：

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
- `"paged_adamw_8bit"`：优化器状态用 8-bit 存储，更省显存（QLoRA 论文常用）。
- `"paged_adamw_32bit"`：精度更高，但显存占用更大。

**注意**：不同 Transformers / bitsandbytes 版本支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报“未知 optim”错误，去查你当前版本的 `TrainingArguments` 文档，换用支持的名称即可。

**使用建议**：分页优化器主要解决**峰值显存溢出**问题，通常搭配 `gradient_checkpointing=True`、小 `per_device_train_batch_size` + 梯度累积一起

## 块 50

### 7.1 4-bit量化：把模型“瘦身”

你的目标是LoRA微调，但显存是硬约束。QLoRA的第一步就是给模型“瘦身”：把原本用FP16（16位浮点数）存储的权重，压缩成NF4（4位浮点数）格式。**核心作用**：直接减少模型权重的体积，让模型能塞进显存更小的GPU。

想象一下：FP16是“大箱子”，NF4是“压缩包”。模型还是那个模型，但占用的空间小多了。

---

### 7.2 双重量化：连“尺子”也压缩

量化过程中，每个权重块需要一个“缩放因子”（scale）来还原数值。这个scale本身也占空间。**双重量化**就是把这个scale也做一次量化，进一步减少额外参数的占用。**核心作用**：把量化参数本身也“瘦身”，省下更多显存。

---

### 7.3 分页优化器：显存不够，内存来凑

训练时，优化器状态（如Adam的动量）非常吃显存。**分页优化器**的做法是：当GPU显存不够时，自动把部分优化器状态“换页”到CPU内存，需要时再换回来。**核心作用**：降低训练时的峰值显存，避免“显存爆炸”。

---

### 动手验证：配置一个QLoRA的4-bit量化

下面这段代码就是QLoRA在`bits

## 块 51

## load_in_4bit：让模型“瘦身”加载

你已经知道 LoRA 是参数高效微调的核心，而 QLoRA 更进一步——它先把大模型的权重压缩到 4-bit 再加载，这样显存占用大幅下降，普通消费级 GPU 也能跑大模型。

**load_in_4bit=True** 就是开启这个“4-bit 加载”的开关。打开它，模型权重不再以完整的 16/32-bit 精度驻留显存，而是以 4-bit 格式存放。

## 三个配套参数，一个都不能少

### 1. bnb_4bit_quant_type="nf4"

这是**量化方式**。NF4（NormalFloat4）是专门为神经网络权重分布设计的 4-bit 量化格式，比普通的 4-bit 整数量化精度损失更小。QLoRA 论文推荐的就是它。

### 2. bnb_4bit_use_double_quant=True

开启**双重量化**。简单理解：量化后的权重本身也需要一些“辅助参数”来记录缩放信息，双重量化把这些辅助参数也量化一遍，再省一笔显存。开就对了。

### 3. bnb_4bit_compute_dtype：最容易踩坑的地方

这里要分清两个概念：

- **存储 dtype**：权重以 4-bit（NF4）存进显存，这是“省显存”的关键。
- **计算 dtype**：权重参与矩阵运算时，会先**反量化**回更高精度再算，这个精度由 `bnb_4bit_compute_dtype` 指定。

**不设置会怎样？** 不设置时，计算 dtype 可能默认为 float32，但实际可能根据硬件自动调整，建议显式指定以避免性能问题。

**选 BF16 还是 FP16？别写死！**

`torch.bfloat16` 需要较新的 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100/H100）。老 GPU（如 RTX 20 系）不支持，硬用会报错。正确做法是**自动判断**：

```python
import torch
from transformers import BitsAndBytesConfig

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个能跑的 QLoRA 训练骨架，让你直观看到各部分如何协作。

### 整体流程一览

整个骨架分 6 步，顺序很重要：

1. **配置 4-bit 量化** — 告诉模型加载时用 NF4 量化 + 双重量化
2. **以 4-bit 加载基础模型** — 真正把模型压缩到内存里
3. **为 k-bit 训练做准备** — 处理 norm/输出层精度，并开启梯度检查点所需的输入梯度
4. **注入 LoRA** — 只训练低秩增量参数，冻结其余部分
5. **准备极简数据** — Causal LM 的 labels 直接复制 input_ids
6. **设置训练参数并启动** — 分页优化器 + 梯度检查点

> **模型许可证**：本骨架以 Llama 2 为例，适用 Llama 2 Community License（商用月活>7亿需Meta授权）。使用其他模型时请自行确认其许可证。

### 逐步拆解

**第 1 步：量化配置**

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

这里根据 GPU 能力自动选择计算精度：支持 bfloat16 就用它，否则退回 float16。`nf4` 是量化类型，`double_quant=True` 开启双重量化。

**第 2 步：加载模型**

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

`device_map="auto"` 让模型自动分布到可用设备上。注意 tokenizer 的 pad_token 如果为空，要设为 eos_token。

**第 3 步：k-bit 训练准备（关键）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步容易漏，但它很关键：它处理 norm/输出层的精度问题，并开启梯度检查点所需的输入梯度。`use_gradient_checkpointing=True` 要和后面 TrainingArguments 里的 `gradient_checkpointing=True` 呼应。

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

`r=8` 是低秩维度，`lora_alpha=16` 是缩放因子。`target_modules` 指定注入哪些模块，这里用 `q_proj` 和 `v_proj`（按实际模型调整，也可用 `"all-linear"`）。`print_trainable_parameters()` 会打印可训练参数量，帮你确认 LoRA 注入成功。

**第 5 步：极简数据**

```python
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tok(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
```

Causal LM 的 labels 直接复制 input_ids，不需要额外标注。真实场景替换成足够规模的数据即可。

**第 6 步：训练参数**

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
```

两个关键点：

- `optim="paged_adamw_8bit"` — 分页优化器，依赖 bitsandbytes，名称随版本可能不同
- `gradient_checkpointing=True` — 与第 3 步的 `use_gradient_checkpointing=True` 呼应，两者缺一不可

### 骨架的完整逻辑

这个骨架展示的核心思想是：**量化负责省内存，LoRA 负责只训练少量参数，梯度检查点进一步省显存，分页优化器处理优化器状态的显存压力**。四者配合，让 4-bit 微调在有限显存下跑起来。

### 本块学习检查

请完成以下验证任务（不引入新知识，仅基于本骨架）：

1. **参数对应关系**：指出第 3 步 `prepare_model_for_kbit_training` 中的 `use_gradient_checkpointing` 参数与第 6 步 `TrainingArguments` 中 `gradient_checkpointing=True` 的对应关系，并说明两者缺一不可的原因。
2. **量化配置核对**：回顾第 1 步的 `BitsAndBytesConfig`，确认 `load_in_4bit`、`bnb_4bit_quant_type`、`bnb_4bit_use_double_quant` 三个

## 块 53

## 九、把 QLoRA 配置串起来：一个可跑通的最小骨架

你已经理解了 4-bit 量化、NF4 和双重量化各自的作用，现在我们把它们组装成一个完整的训练流程。下面这 5 步是**缺一不可**的，顺序也不能乱：

**第一步：`BitsAndBytesConfig` —— 决定模型怎么加载**
这一步配置基础模型以 4-bit 方式加载，包含三个关键设定：NF4 量化类型、双重量化开关、计算 dtype（BF16/FP16）。它决定了模型"以什么精度被读进显存"。

**第二步：`prepare_model_for_kbit_training()` —— QLoRA 的必要适配**
这是最容易跳过的一步，但**跳过它常导致训练不收敛或报错**。它会为量化模型做适配：把部分层转成计算精度、开启输入梯度以支持梯度检查点等。记住：这是 QLoRA 的**必要一步**。

**第三步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**
把可训练的 LoRA 适配器注入到冻结的 4-bit 主干模型上。主干被冻结，只有 LoRA 参数在更新。

**第四步：梯度检查点对应开启**
在 `TrainingArguments` 中设置 `gradient_checkpointing=True`，并确保模型支持梯度检查点。

**第五步：`optim="paged_adamw"` —— 分页优化器**
启用分页优化器来缓解峰值显存，这是 QLoRA 省显存的最后一环。注意：较新版本 bitsandbytes 推荐使用 `paged_adamw` 或 `adamw_bnb_8bit`，旧版本中为 `paged_adamw_8bit`，请按你所用的版本核对名称。

> ⚠️ 上面是"能说明关系"的最小骨架。真实训练需要替换为足够规模的数据、合理的超参，并按你所用版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数与 `processing_class`/`tokenizer` 写法。

---

## 十、你的电脑能跑吗？

**只学原理**：不需要 GPU。

**实际运行 QLoRA**，最低配置参考：

| 硬件 | 最低要求 |
|------|----------|
| CPU | 4核+ |
| 内存 | 16GB+ |
| GPU | NVIDIA CUDA GPU |

**显存需求**（7B 级别模型的经验参考，实际以运行为准）：

| 显存 | 适合场景 |
|------|----------|
| 8GB | 小模型 LoRA，或较短序列/小 batch 的 7

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 关键认知：LoRA不是直接修改基础模型

很多初学者以为LoRA训练完会得到一个“升级版”的完整模型，**这是最常见的误解**。

实际过程是这样的：

- **基础模型**：全程**冻结**，一动不动（比如Llama-7B）
- **LoRA Adapter**：这才是真正被**训练**的部分

最终的结构是：

```
基础模型（冻结） + LoRA Adapter（训练后）
```

训练完成后，你保存的**通常只有**：

```
adapter_model.safetensors   ← LoRA权重文件
adapter_config.json         ← 配置信息
```

**没有完整模型**。你手里的是一个“补丁”，不是“完整程序”。

---

### 1.2 Adapter的特点：小而灵活

**优点：**
- 文件小，保存快
- 一个基础模型可以搭配多个Adapter，互不干扰

比如同一个Llama-7B，你可以配：

```
客服Adapter  → 做客服
医疗Adapter  → 做医疗问答
法律Adapter  → 做法律咨询
```

**缺点（也是痛点）：**
部署推理时，必须**基础模型 + Adapter 同时存在**，缺一不可。这给部署带来额外负担——你没法只带一个文件走。

---

## 二、为什么需要合并LoRA权重？

既然Adapter和基础模型是分离的，那部署时就要同时加载两份东西，麻烦且容易出错。

**合并（merge）** 就是把Adapter的权重“写回”基础模型，让两者融为一体。合并后，你得到的是一个**完整的、独立的模型**，不再依赖Adapter文件。

这正是 `merge_and_unload()` 要解决的问题——它负责把LoRA权重合并进基础模型，然后**卸载Adapter**，让模型恢复成“普通完整模型”的状态，方便后续导出

## 块 55

**动手理解：为什么推理时要合并LoRA？**

你训练完LoRA后，模型其实是“两部分”：

- 输入 → **基础模型** + **LoRA适配器** → 输出

推理时，你需要同时加载基础模型和Adapter，这很麻烦。部署时，你更希望只有一个“完整模型”：

- 输入 → **完整模型** → 输出

所以，需要把LoRA的权重合并进基础模型。

**合并的数学原理**

训练时，模型权重变成了：

- W' = W + ΔW

其中 ΔW = BA（这就是LoRA学到的增量）。

合并，就是把BA加到W上，得到W'。合并后，原来的“W + Adapter”就变成了一个整体W'。

## 块 56

## 3.1 `merge_and_unload()` 是什么？

你训练完LoRA后，模型其实还是“两件套”：**基础模型**（Base Model）加上**LoRA适配器**（LoRA Adapter）。但部署时，你通常只需要一个完整的模型文件。`merge_and_unload()` 就是干这个的。

PEFT库提供了这个方法，调用方式很简单：

```python
merge_and_unload()
```

它的核心作用就一句话：**把LoRA权重合并到基础模型里，然后移除PEFT的包装**。

用图来理解就是：

```text
基础模型 + LoRA适配器  →  完整模型
```

---

## 3.2 合并前后，模型结构发生了什么变化？

**合并前**，你的模型是一个 `PeftModel`，内部结构是：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后**，它变成了一个**普通的Transformer模型**，LoRA的更新已经写进权重里了：

```text
普通Transformer模型（包含LoRA更新后的权重）
```

---

## 四、LoRA模型合并的完整流程

整体流程是五步：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步来。

---

## 五、加载LoRA Adapter

假设你已经有了基础模型路径和Adapter路径：

```python
base_model_path = "model_path"
adapter_path = "adapter_path"
```

加载基础模型，然后挂上Adapter：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：

```text
Base Model + LoRA Adapter
```

---

## 六、使用 `merge_and_unload()` 合并权重

执行合并：

```python
merged_model = model.merge_and_unload()
```

这一步在数学上做的事是：把原始权重 `W` 和LoRA的缩放增量 `(α/r)BA` 相加，得到新的权重 `W'`：

```text
W + (α/r)BA → W'
```

合并完成后，`merged_model` **不再是** `PeftModel`，而是一个**普通的Transformer模型**。

---

## 七、导出合并后的模型

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

> **注意**：上面是示意结构。实际文件会随模型/分词器类型变化，比如可能还有 `generation_config.json`、`special_tokens_map.json`，大模型可能是分片权重（`model.safetensors.index.json

> **模型许可证**：请确认所用基础模型的许可证类型，例如：Llama 2 Community License（商用月活>7亿需Meta授权）。

> **显存要求**：加载模型时请留意显存占用，例如：Llama-2-7b FP16 需要至少14GB显存。

## 块 57

## 3.1 merge_and_unload 方法作用：合并后必查清单

合并完 LoRA 权重后，别急着欢呼——模型能加载不代表行为正确。按下面四项逐一核查，避免“能跑但结果不对”的坑：

**① Tokenizer 完整且一致**
确认保存的 tokenizer 与训练时完全一致。重点检查训练中**新增或修改过的特殊 token**（比如自定义的 `<|im_start|>`、pad/eos 等）是否都在 `special_tokens_map.json` / `tokenizer_config.json` 里。如果训练时扩过词表，还要确认 embedding 和 lm_head 的尺寸与词表匹配，并确认 tokenizer 的 `vocab_size` 与模型 embedding 尺寸一致。

**② generation_config.json**
如果你依赖特定的生成默认值（如 eos/pad token id、`max_new_tokens`、采样参数），务必确认它一并导出，并确认 `eos_token_id` 和 `pad_token_id` 设置正确。否则推理时可能回退到不合适的默认行为，生成结果会“跑偏”。

**③ dtype 精度**
确认合并/保存时的精度符合预期——FP16、BF16 还是 FP32。dtype 不一致会直接影响显存占用和数值表现，尤其是 QLoRA 场景下更敏感。

**④ config.json 完整性**
确认模型结构配置（层数、hidden size、任务头等）正确，且**已移除 PEFT/adapter 相关配置**。合并后应该是普通 Transformers 模型，不应再残留 `adapter_config.json`。

---

### 合并后如何加载推理？

合并后可使用 Transformers 直接加载，但若使用 PEFT 库训练，推荐使用 `PeftModel.from_pretrained` 加载以确保兼容性：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程简化为：用户输入 → 完整模型 → 生成结果。

---

### 合并 vs 不合并：怎么选？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 |

## 块 58

## 10.1 推荐合并场景

你学LoRA微调，最终目标是把模型用起来。什么时候该把LoRA权重“合并”回基础模型？看两个典型场景：

**场景1：上线部署——只要一个Adapter，追求简单**

生产环境链路通常是：

```text
服务器 → 直接加载模型 → 提供API
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```

如果你只需要一个固定的Adapter，且希望部署链路尽量简单——不额外依赖PEFT库、不做运行时Adapter加载——那么合并通常更方便。

> 注意：如果推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或你需要在同一基础模型上热切换多个Adapter，那么不合并、以Adapter形式部署反而更灵活。是否合并要看框架能力与业务需求。

**场景2：分享模型——让对方开箱即用**

如果希望对方

## 块 59

### 不建议合并的两个场景

**场景1：多个任务Adapter**

假设你有一个基础模型，上面挂了多个Adapter（如Adapter A、B、C），每个Adapter对应不同任务。此时**保留Adapter更灵活**——你可以随时切换或组合它们，而合并后这种灵活性就没了。

**场景2：持续训练**

如果你打算继续调整LoRA，也请保留Adapter。合并会固化当前权重，后续再改就得重新训练，得不偿失。

---

### 动手验证

检查你手头的训练脚本：如果用了多个Adapter或计划继续训练，**不要执行合并**，直接保存Adapter即可。

---

### QLoRA合并注意事项（重点）

QLoRA训练时，基础模型以**4-bit（NF4）**存储，而LoRA参数是

## 块 60

## 直接合并的坑，以及更稳的做法

你训练完 QLoRA 后，**不要急着在 4-bit 状态下直接合并**。这一步很容易出问题，咱们先看清风险，再走稳妥路线。

### 直接合并可能遇到的情况

如果你直接在 4-bit 状态下执行 `merge_and_unload`，PEFT 会自动处理反量化过程，但可能遇到两类问题：

1. **精度损失**：4-bit 权重需要先反量化再合并，这个过程和全程用高精度计算并不完全等价，结果会有细微偏差。
2. **类型或版本不兼容**：可能出现权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> ⚠️ 注意：上面说的是**可能**出现的情况，不是每次都会报错。能不能顺利合并、合并质量如何，取决于你的**版本**、量化配置。部分版本支持在 4-bit 状态下直接合并，但并非所有版本都支持。最终以你所用版本文档和实际运行结果为准。

### 推荐做法：先反量化到高精度再合并

更稳妥的路线是：**绕开 4-bit 状态**，用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一份 Adapter 后合并。这样合并发生在高精度权重上，直接避开类型和精度问题。这是推荐做法，但并非唯一可行方案——如果你的版本支持，在 4-bit 状态下直接合并也可以，只是风险更高。

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

### 动手验证

跑完上面的代码后，检查 `merged_out` 目录下是否生成了模型权重文件和 tokenizer 文件。然后试着用 `AutoModelForCausalLM.from_pretrained(merged_out)` 加载合并后的模型。

> 📄 模型许可证：

## 块 61

**QLoRA合并：先“还原”再合并**

你训练QLoRA时用的是4-bit量化基础模型，但**合并时不能直接拿它来合**。关键一步是：**加载基础模型时不再传`quantization_config`或`load_in_4bit`**，这样得到的是高精度权重（FP16/BF16），再执行`merge_and_unload`。

> **模型许可证**：Llama 2 Community License（商用月活>7亿需Meta授权）。

**注意三点：**

1. **版本差异**：不同PEFT版本对“量化模型能否直接合并”支持不同，有的要求先反量化。务必查你所用版本的`merge_and_unload`文档。
2. **内存开销**：高精度

## 块 62

**合并时开启校验并做一致性验证（建议）**

在 `merge_and_unload` 时加上 `safe_merge=True`，合并过程中会自动做数值检查（比如检测 NaN），相当于给合并操作加一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，强烈建议做一次**前后一致性验证**：用同一小批固定 prompt，分别让「合并前的 PeftModel」和「合并后的模型」推理，对比两者的 logits 最大绝对差，以及贪心生成的文本是否一致。如果差异很小，说明合并没有引入明显偏差。

**最小验证代码思路**：固定 prompt → 两个模型分别推理 → 比较 logits 最大绝对差 → 比较生成文本。

**本块学习

## 块 63

**本块学习检查**：运行下方代码，记录 `max_abs_diff` 的值，并对比合并前后的生成文本是否一致。若差异过大（如生成文本不同），说明合并过程可能存在问题，需检查量化配置或适配器路径。

---

### 为什么需要“校验”？

合并 LoRA 权重时，你是在**数学上把适配器“折叠”进基础模型**。QLoRA 场景下，基础模型是量化过的（如 4-bit），合并过程涉及精度转换，结果可能与“未合并但逻辑上等价”的模型存在微小偏差。校验的目的就是量化这个偏差，确认它不会影响实际使用。

### 核心陷阱：`merge_and_unload()` 是“原地”操作

代码注释强调了一个关键点：`merge_and_unload()` 会**就地修改**并返回底层模型。也就是说，一旦执行合并，原来的 `PeftModel` 对象就不再是“未合并”状态了。

**因此，你不能拿 §11.2 里合并后的对象去当“合并前”的基准**——那已经是被修改过的了。必须**重新加载一份全新的、未合并的 `PeftModel`** 作为对比基准。

### 校验步骤拆解

整个过程分三步，核心思想是“同一输入，对比输出”。

**第一步：准备“合并前”的参照模型**

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)
```

这里重新加载了基础模型和适配器，得到 `model_before`。它的状态是：`Base(高精度) + Adapter`，即**未合并**的原始状态。

**第二步：对比 logits（模型输出的原始分数）**

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

- `logits` 是模型预测下一个词的概率分布（未归一化）。
- 计算两个模型对**同一 prompt** 输出的 logits 的**最大绝对差**。
- **理想情况**：差值为 0，说明合并完全无损。
- **QLoRA 现实**：由于量化精度损失，通常是一个**较小的近似值**（比如 0.01 或 0.1 量级）。只要生成文本一致，这个数值小到不影响使用即可。

**第三步：对比生成文本（更直观的验证）**

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print(text_before)
print(text

## 块 64

## 合并时开启校验并做一致性验证（建议）

**为什么必须单独再加载一份模型？**

`merge_and_unload()` 会**原地**修改你之前那个 `model`——它把 LoRA Adapter 直接合进底层权重，然后移除 PEFT 包装。这意味着合并完成后，这个 `model` 已经**不再是合并前的样子**了。

如果你拿它和合并前对比，两边其实是同一份已合并的权重，`max_abs_diff` 会**恒等于 0**——验证完全失去意义。

正确做法是：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照，再和合并后的模型做对比。

**显存不够怎么办？**

如果两个模型同时放不下，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本**存到磁盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

**结果预期——别期待逐位相同**

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 差异。重点确认**没有明显退化或 NaN**，而不是追求逐位相同。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。

是否可接受，以你的**任务指标与验证集**为准。

---

## 电脑配置批注

本章节需要**实际加载模型**，涉及：加载完整模型 → 合并权重 → 导出模型。

**模型许可证：** 请确认所用模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需 Meta 授权），确保合规使用。

**最低配置

## 块 65

## 合并后翻车？这份排查清单帮你稳过QLoRA合并

你已经走到了QLoRA微调的最后一公里——合并权重并导出模型。这个环节最容易出各种“玄学”问题，但别慌，绝大多数坑都有固定解法。下面直接进入实战排查。

### 常见问题速查表

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到 PeftModel 再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积 Adapter | 无需处理；若在意体积可保留 Adapter 不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是 dtype 用了 FP32、未做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存 tokenizer；或特殊 token / generation_config 未一并导出 | 重新 `save_pretrained` 并核查 tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 在 4-bit 状态直接合并、PEFT/bitsandbytes 版本不支持、量化包装未正确反量化、内存/显存不足 | 先反量化到 FP16/BF16 再合并；核对版本；必要时用 CPU 合并；开 `safe_merge=True` 并做一致性验证 |

### 重点：什么是“先反量化再合并”？

这里有个关键误解要澄清：**“反量化”不是手动逐层还原 4-bit 权重**，而是——**重新加载一份不带量化配置的基础模型**。

具体操作是：用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`（**不传** `quantization_config`/`load_in_4bit`）得到高精度权重，再 `PeftModel.from_pretrained` 挂载**同一份 Adapter** 后合并。这样合并发生在高精度权重上，天然避开了“在 4-bit 状态下合并”的类型/精度问题。

**版本提示**：不同 PEFT / bitsandbytes / Transformers 版本对“量化模型能否直接调用 `merge_and_unload`”的支持不同——部分版本直接报错或警告，部分较新版本有一定支持但仍可能有精度损失。

## 块 66

## 路线A：合并后部署完整模型

你已经掌握了 `merge_and_unload()` 合并权重的方法，现在来看部署阶段的第一条路线——**合并后部署**。

### 核心流程

```
merge_and_unload() → 保存完整模型 → 推理框架按普通模型加载
```

这条路线最直观：把 LoRA 权重合并进基础模型后，得到一个**完整的模型文件**。部署时，推理框架（比如 vLLM）直接加载这个合并后的模型即可，**完全不需要知道 LoRA 是什么**。

### 优点

- **部署链路简单**：推理框架当作普通模型处理，无需额外配置
- **推理零额外开销**：运行时不再有 Adapter 切换成本，性能更稳定

### 代价

- **存储成本高**：每个 Adapter 都要保存和分发一份完整模型
- **多任务不友好**：如果你有多个任务（多个 Adapter），就得存多份完整模型，非常占空间

### 什么时候选路线A？

简单说：**任务少、追求部署简单**时选路线A。如果你的场景是"一个模型只服务一个任务"，合并后直接部署是最省心的方式。

> **模型许可证**：部署前请确认基础模型的许可证类型。例如，Llama 2 采用 Community License，商用月活超 7 亿需获得 Meta 授权。

---

## 路线B：保留 Adapter，动态服务

与路线A相对，路线B的思路是：**基础模型只加载一份，多个 LoRA Adapter 按需挂载**。

### 核心流程

```
基础模型（只加载一份） + 运行时挂载多个 LoRA Adapter（A/B/C）
→ 按请求指定使用哪个 Adapter
```

### 优点

- **省显存省存储**：多任务时只需一份基础模型 + 多个小体积 Adapter
- **支持热切换**：运行时动态切换 Adapter，无需为每个任务各存一份完整模型

### 代价

- **依赖框架支持**：需要推理框架原生支持 LoRA 动态加载
- **配置略复杂**：链路比"直接加载合并模型"多几步
- **版本敏感**：不同框架/版本对 LoRA 的支持范围和限制不同

### vLLM 的 LoRA 服务形态（示意）

以 vLLM 为例，启动 OpenAI 兼容服务时，需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**：

```bash
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB

## 块 67

**14.3 如何选择：你的部署路线图**

学到这里，你已经知道LoRA模型有两种“出场方式”：合并成完整模型，或保留Adapter动态加载。怎么选？记住这三个经验法则（非硬性规则，以实际部署为准）：

1. **追求最简部署** → 如果你只有一个Adapter，且希望部署流程最简单，选**合并**（路线A）。
2. **多Adapter动态服务** → 如果同一基础模型要服务多个Adapter，且你的推理框架原生支持LoRA，选**保留Adapter**（路线B）。
3. **QLoRA精度敏感** → 无论选哪条路，都建议按§11.3做合并前后/上线前的一致性验证。

**本块学习检查**：合并不是唯一部署方式——请用自己的话回答：什么场景下你会选择“保留Adapter”而不是“合并”？

---

**十五、总结**

本章你已掌握：✅ Adapter保存方式 ✅ 权重合并原因 ✅ merge_and_unload作用 ✅ LoRA数学合并过程 ✅ 完整模型导出流程 ✅ 合并模型独立部署方式 ✅ LoRA与QLoRA合并注意事项

## 测评题目

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB12-ST1205-003
合并 LoRA 权重后最直接的用途是？

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）
