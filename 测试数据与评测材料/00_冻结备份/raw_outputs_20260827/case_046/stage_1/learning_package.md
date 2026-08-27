# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

传统大模型训练流程是这样的：

```text
输入数据 → 整个模型参数更新 → 保存完整模型
```

以7B参数模型为例，它包含**70亿个参数**。训练时，你需要同时保存三样东西：

- **参数**：模型本身的权重
- **梯度**：用于更新参数的计算结果
- **优化器状态**：优化算法维护的额外信息

这三者叠加，显存需求会非常高。普通消费级GPU根本扛不住。

---

**你的动手小实验**：打开你的GPU监控工具（如`nvidia-smi`），看看当前显存占用。想象一下：如果要把70亿参数、对应的梯度和优化器状态全部塞进显存，你的显卡够用吗？这个直观感受，就是全参数微调最大的痛点。

**下一步**：既然全参数微调这么“吃”显存，那有没有办法只更新一小部分参数、同时保持效果？这就是接下来要讲的**PEFT（参数高效微调）**要解决的问题。

## 块 2

**本块学习检查**：读完下表后，请用自己的话说说“全参数微调”和“LoRA”在训练参数数量上的区别。

PEFT（参数高效微调）的核心是**不修改整个大模型，只训练少量新增参数**。流程是：原始大模型加上LoRA适配器参数，只训练LoRA参数，最后保存小规模Adapter。

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**动手验证**：对比上表三行的“训练参数”和“显存

## 块 3

## 2.1 LoRA是什么？

**一句话理解**：LoRA（Low-Rank Adaptation，低秩适配）是一种“只训练小部分参数”的微调技巧。

**核心思路**：

- 原始模型的权重 `W` **冻结不更新**；
- 在旁边新增一个增量 `ΔW`，最终效果相当于 `W + ΔW`；
- 这个 `ΔW` 被分解为两个低秩小矩阵的乘积：`ΔW = B × A`。

**形状约定**（本课程统一采用）：

- 若原权重 `W` 形状为 `d_out × d_in`，秩为 `r`；
- 则 `A` 形状约为 `r × d_in`，`B` 形状约为 `d_out × r`；
- 相乘后 `B × A` 得到与 `W` 同形状的 `ΔW`。

> 注意：不同资料可能写成 `A × B`，只是符号顺序不同，关键是矩阵形状能相乘并还原成 `W` 的形状。

## 块 4

## 3.1 LoRA的问题：省了训练，却没省加载

你已经知道 LoRA 能大幅减少**训练**时的参数，但这里有个关键点：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个 7B 参数的模型，如果用 FP16（半精度）加载，大约需要 **14GB 显存**。也就是说，无论 LoRA 让训练多“轻量”，这 14GB 的“底子”是省不掉的——显存瓶颈依然存在。

## 3.2 QLoRA 的解法：把模型“压缩”了再训练

QLoRA（Quantized LoRA，量化版 LoRA）的思路很直接：既然基础模型太大，那就先把它**变小**再加载。

核心流程是：

1. 原本 FP16 的模型 → **4bit 量化**（大幅压缩）
2. 量化后显存占用显著下降
3. 在压缩后的模型上加载 LoRA 参数进行训练

训练时的结构就是：

**4bit 基础模型 + LoRA 参数 → 训练**

这样既保留了 LoRA 的高效训练，又解决了基础模型加载占显存的问题。

## 3.3 QLoRA 依赖 bitsandbytes：量化操作的“执行者”

要实现 QLoRA，你需要这几个组件配合：

- `transformers`（模型框架）
- `peft`（LoRA 实现）
- `bitsandbytes`（负责量化）
- 支持的量化后端（本教程以 **NVIDIA CU

## 块 5

## 电脑配置批注：跑 LoRA / QLoRA 需要什么？

本节帮你确认本地电脑能否跑通 LoRA 微调。先看最低门槛，再看 QLoRA 的显存参考。

### LoRA 最低配置（硬门槛）

- **CPU**：4 核以上
- **内存**：16GB+
- **磁盘**：50GB 以上
- **GPU**：推荐 NVIDIA GPU
- **显存**：最低 8GB，推荐 12GB+

### QLoRA 要求（显存参考）

QLoRA 的目标就是**用更小的显存微调大模型**——它把基础模型量化到 4-bit，显存占用通常低于同规模 LoRA 的全精度加载。

**注意**：本教程按 NVIDIA CUDA GPU 给出命令。QLoRA 并非只能用 NVIDIA，但其他后端请以 bitsandbytes 官方说明为准。

**7B 级别模型的显存参考**（经验值，非硬保证）：

- 参考下限：约 12GB（短序列、小 batch）
- 推荐：16GB+

**说明**：8GB 显存仅适用于极端优化场景（如开启梯度检查点、极短序列、batch=1），多数情况下无法正常运行。

### 别把数值当死门槛

实际显存占用受这些条件影响，条件不同结论就不同：

- **模型规模**：7B 级别；13B/70B 显存需求显著上升
- **序列长度**：短序列（≤512）占用低；越长激活值显存增长越明显
- **batch 大小**：小 batch（1~2）占用低；

## 块 6

## 5.1 先确认你的Python版本

在终端里运行：

```bash
python --version
```

看到 **Python 3.10 或更高版本**就符合要求。如果版本偏低，建议先升级Python再继续，否则后面装库容易出兼容问题。

---

## 5.2 检查PyTorch是否装对了（关键一步）

> 这一步最容易踩坑：LoRA/QLoRA依赖PyTorch，而 `bitsandbytes` 的量化功能要求安装**和你本机CUDA匹配的GPU版PyTorch**。直接 `pip install torch` 通常装到的是CPU版，QLoRA会直接无法使用。

**正确做法**：去 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 选你对应的CUDA版本。比如NVIDIA CUDA 12.1就执行：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证是否装对**，运行：

```python
import torch

print(torch.__version__)

print(
    torch.cuda.is_available()
)

print(torch.version.cuda)   # 查看 torch 编译使用的 CUDA 版本
```

看到 `True` 就说明PyTorch能识别到你的GPU。

> ⚠️ 注意：`torch.cuda.is_available()` 返回 `True` 只说明 **PyTorch侧**能用GPU，**不代表bitsandbytes的4-bit量化一定能跑**（那还依赖bitsandbytes是否正确匹配你的CUDA）。真正的4-bit验证留到后面第八章、第十三章再做。

---

## 5.3 看一眼你的GPU

终端运行：

```bash
nvidia-smi
```

输出里会包含：

- GPU型号（如 `RTX 4090`）
- CUDA版本（如 `CUDA 12.x`）
- 显存大小（如 `Memory 24GB`）

记住这三个信息，后面配置bitsandbytes时要用。

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是 **Hugging Face官方**的参数高效微调库，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter等多种方法。你要做的LoRA微调就是靠它。

### 6.2 安装PEFT

```bash
pip install peft

## 块 7

**bitsandbytes 是干嘛的？**

一句话：它是让模型“瘦身”后还能高效训练的工具。具体干三件事：

- **8bit量化**：把模型精度从32位降到8位，省显存。
- **4bit量化**：进一步降到4位，更省。
- **QLoRA训练**：用4bit量化后的模型跑LoRA微调，这就是你要做的。

**对你来说**：你的目标是LoRA微调，而本教程用的是QLoRA方案，所以**必须装**。

> 注意：这里“必须”是针对本教程的4bit量化方案。QLoRA本身不绑定某个量化库，但本教程统一用bitsandbytes + NVIDIA CUDA演示。你跟着装就行。

**安装命令**：

```bash
pip install bitsandbytes
```

装完先别

## 块 8

**本块学习检查**：运行 `nvidia-smi`，找到 `CUDA Version` 并记下数字。然后运行 `python -c "import bitsandbytes as bnb; print(bnb.cuda_version)"`，确认它显示的 CUDA 版本与前者一致。

---

**8.1 为什么检查CUDA？**

你要用 LoRA 微调，QLoRA 依赖 bitsandbytes 做量化。它的工作链路是：

```text
CUDA → GPU计算
```

如果 CUDA 版本不匹配，你会遇到：CUDA 不可用、量化失败、加载模型报错。所以先检查，省得后面踩坑。

**8.2 查看CUDA版本**

打开终端，运行：

```

## 块 9

**动手前的关键一步：Windows 用户请优先用 WSL2**

你的目标是跑通 LoRA 微调，而环境配置往往是第一个坑。如果你是 Windows 用户，**强烈建议直接使用 WSL2（Windows Subsystem for Linux 2）**，而不是在 Windows 原生环境下折腾。

推荐的路径是：

```text
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

**为什么？** 核心原因是：**Linux 环境的兼容性更好**。特别是对于 `bitsandbytes` 这个库（QLoRA 量化依赖它），在 Linux 下运行更稳定。

---

**如果坚持用 Windows 原生，你可能会遇到：**

**问题1：`bitsandbytes` 无法加载 CUDA 动态库**

这个报错很常见，但原因可能不止一种：

- 你安装的 `bitsandbytes` 版本，

## 块 10

## 九、Windows用户特别注意：CUDA版本不匹配怎么办？

在Windows上配置LoRA环境时，最常见的坑就是**CUDA版本对不上**。别慌，先按下面三步自查：

**第一步：查看你的显卡驱动支持什么**

打开命令行（CMD或PowerShell），输入：

```bash
nvidia-smi
```

这个命令会告诉你三件关键事：

- **NVIDIA驱动**：是否已安装、版本号是多少
- **CUDA版本**：你的驱动最高支持哪个CUDA版本
- **显卡型号**：确认你有NVIDIA显卡（没有的话后面都白搭）

**第二步：确认PyTorch版本匹配**

记住一个核心原则：**PyTorch的CUDA版本必须 ≤ 你驱动支持的CUDA版本**。如果PyTorch要求的CUDA比驱动高，就会报错。

---

## 十、安装完整LoRA环境（实操步骤）

假设你已经有Python基础环境，按顺序执行：

**第1步：安装基础库**

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**第2步：安装GPU版PyTorch（关键！）**

> ⚠️ **前置条件**：必须安装**匹配本机CUDA的GPU版PyTorch**，否则后面`import torch`或QLoRA 4-bit会直接失败。

以CUDA 12.1为例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**第3步：安装LoRA核心库**

```bash
pip install peft bitsandbytes
```

**第4步：一条命令搞定全部**

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate（加速配置）

运行：

```bash
accelerate config
```

按提示选择，单GPU场景这样选：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

完成后会生成一个`default_config.yaml`配置文件，后面训练时会自动读取。

---

## 十二、验证LoRA环境（动手测试）

创建一个测试文件`test_lora_env.py`，写入：

```python
import torch

from peft import LoraConfig


print(
    "CUDA:",
    torch.cuda.is_available()
)


config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05
)


print(config)
```

运行

## 块 11

## 九、Windows用户特别注意事项：推荐使用WSL2

在Windows上跑LoRA/QLoRA，最常遇到的坑就是**兼容性问题**——比如bitsandbytes加载失败、CUDA识别不到。这些问题在Windows原生环境里往往很难缠。

**最省心的解法：直接用WSL2**（Windows Subsystem for Linux 2）。它相当于在Windows里跑一个轻量Linux环境，PyTorch、CUDA、bitsandbytes这些工具在Linux下的兼容性远好于Windows原生环境，能帮你绕开大部分报错。

---

### 常见问题速查表

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版本 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

### 你的完整学习路线

**当前环境（已就绪）：**

```
Transformers → Datasets → Accelerate → PEFT → bitsandbytes → LoRA → QLoRA
```

**后续实战（下一步）：**

## 块 12

## 一、为什么需要LoRA？——先看全量微调的“代价”

你现在的目标是LoRA微调，那我们先搞清楚：**传统微调到底贵在哪？** 这样你才能理解LoRA为什么“省”。

现代大模型（比如LLaMA、Qwen、Baichuan、Mistral）动辄**数十亿甚至上百亿参数**。拿一个7B模型来说，就是**70亿参数**。

如果走传统路线：

```text
预训练模型 → 更新全部参数 → 保存新模型
```

你需要付出三笔“巨款”：

1. **巨大的GPU显存**——模型本身已经很大，训练时还要额外空间；
2. **大量训练时间**——每个参数都要算梯度、更新；
3. **巨大的存储空间**——每个微调版本都要完整保存一份新模型。

**关键点来了**：假设模型有100亿参数，训练时你不仅要保存模型参数W，还要保存**梯度∂W**和**优化器状态（如Adam moment）**。实际显存需求**远大于模型本身大小**——这就是全量微调“烧钱”的根本原因。

---

## 二、LoRA的核心思想：不更新全部参数

既然全量更新太贵，LoRA的思路就是：**冻结原始权重W，只训练一个很小的“补丁”**。

这个补丁就是**ΔW = BA**。

- **W**：原始模型的权重矩阵（冻结不动）；
- **ΔW**：我们要训练的“增量补丁”；
- **B和A**：两个小矩阵，它们的乘积就是ΔW。

**为什么用低

## 块 13

**2.1 不直接修改原模型参数**

传统微调是直接改原始权重：`W → W'`。

LoRA 的做法完全不同：**冻结**原始权重 `W`，不动它。然后额外增加一个新参数 `ΔW`。最终模型用的是：

`W' = W + ΔW`

也就是说，`W` 是原模型参数，`ΔW` 是 LoRA 新增参数。你只训练 `ΔW`，原模型保持原样。

**2.2 为什么只训练ΔW？**

大模型已经通过海量数据学会了语言规律、世界知识、推理能力。微调通常只是“对已有能力做方向调整”，所以不必重新学全部参数，只学任务相关的变化量往往就够了。

> 注意：这是教学直觉，不是严格结论。能否用 LoRA 替代全参数微调，取决于任务与预训练能力的差距、数据规模、目标模块与秩的设置等。差距大或数据足时，全参数微调仍可能更好。

**动手验证**：打开你的 PyTorch 环境，

## 块 14

**本块学习检查**：读完下面内容后，请合上资料，在白纸上画出“一个矩阵拆成两个小矩阵”的示意图，并标出每个矩阵的维度。

---

## 3.1 矩阵基础概念

Transformer里大量参数都是**矩阵**。以注意力层为例（数学示意，非代码）：

```text
Q = X · Wq
```

这里 `X` 是输入，`Wq` 是**权重矩阵**。假设 `Wq` 的形状是 `4096 × 4096`，它的参数数量就是：

```text
4096 × 4096 ≈ 1677万
```

一个矩阵就有近1700万参数，模型里这样的矩阵还有很多。

## 3.2 直接训练矩阵的问题

微调时，如果要修改权重，通常需要学习一个变化量 `ΔW`。如果 `ΔW` 也是 `4096 × 4096`，那就要再训练1677万个参数——成本太高。

## LoRA的核心思路

LoRA提出：**不要直接学习完整矩阵**，而是把它**拆成两个小矩阵**。原始权重 `W` 保持

## 块 15

**4.1 矩阵尺寸变化：LoRA如何“瘦身”？**

先看原始大矩阵 W，尺寸是 **4096 × 4096**。如果直接训练它，要更新的参数数量就是：

**4096 × 4096 = 16,777,216**（约1677万）。

现在用 LoRA，设置 **rank r = 8**。我们不直接动 W，而是把它拆成两个小矩阵 A 和 B：

- **A** 的尺寸是 **8 × 4096**，参数数量 = 8 × 4096 = **32,768**
- **B** 的尺寸是 **4096 × 8**，参数数量 = 4096 × 8 = **32,768**

## 块 16

**5.1 神经网络参数变化具有低维特征**

想象一下，你要微调一个大模型。LoRA论文的核心洞察是：模型适应新任务时，参数变化 **ΔW** 并不需要覆盖所有方向。

**关键直觉**：真正影响任务的有效变化，其实集中在一个**低维子空间**里。

举个例子：一个4096维的空间，但真正起作用的"有效方向"可能远少于满维度——只是少数几个方向。这就是LoRA的假设：**权重更新往往具有较低的内在维度**。

> ⚠️ 注意：具体"有效维度"是多少，没有通用固定值，会随模型、任务、数据变化。这里只做**定性**理解，别记成"几十维"这种确定数字。

**所以**：低秩矩阵 **

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张**4000×4000像素**的高清图片，完整保存需要**1600万个数据**。但如果图片的主要变化只是**亮度、对比度、少量结构变化**，其实用**少量参数**就能描述这些变化。

LoRA的核心思想正是如此：**用少量参数描述模型变化方向**。预训练模型已经学到了丰富的知识，微调时我们不需要重新学习全部内容，只需要捕捉“变化的方向”——这个方向往往是低秩的，可以用少量参数高效表示。

> **澄清**：LoRA**不是**对原始权重矩阵做SVD分解来降低参数量，而是**冻结原始权重**，仅训练两个低秩增量矩阵A和B，用它们的乘积`BA`来近似微调过程中的权重变化量`ΔW`。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中Attention包含**Q、K、V、O**四个投影矩阵。

LoRA在早期实践中**常见**加入Q投影和V投影，例如：

```python
target_modules=["q_proj", "v_proj"]
```

表示只训练`Wq`和`Wv`的变化。

> **注意**：只挑`q_proj`/`v_proj`是LoRA原论文的**常见选择**，**不是固定标准**。现代实践也常把LoRA加到全部投影（q/k/v/o）、FFN的线性层，甚至用`target_modules="all-linear"`覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

**更完整的配置示例**（以Llama类模型常见模块名为例；不同模型模块名可能不同，请先用`print(model)`或`model.named_modules()`确认实际名称）：

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

> 上例是**显式列出模块名**的写法；若PEFT版本支持，也可用`target_modules="all-linear"`自动匹配所有线性层。覆盖模块越多，可训练参数、显存与Adapter体积也越大——是否值得，应结合验证集表现

## 块 18

## 8.1 rank（r）——LoRA的“旋钮”

先记住这个参数长什么样：

```python
r=8
```

**它代表什么？**

`r` 是低秩矩阵的维度。你可以把它理解为 LoRA 在微调时“开了一条多宽的小路”——`r` 越大，这条小路越宽，能承载的信息越多。

**调大 `r` 的代价与收益**

- **优点**：表达能力更强，模型能学到更复杂的任务模式。
- **缺点**：参数更多、显存占用增加。

**经验起点参考**（注意：只是起点，不是铁律）

| 任务类型 | rank（起点参考） |
| ---- | ---- |
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+ |

**怎么选？别背配方，用验证集说话**

`r` 的合适取值和模型规模、目标模块、数据量、任务难度都有关。与其猜，不如跑一个受控实验：

1. **划出验证集**：从数据中留出 10%~20% 不参与训练，后续所有比较都看它在验证集上的表现。
2. **固定其它变量**：只改 `r`，把 `target_modules`、学习率、`lora_alpha`（或 `alpha/r` 比例）、训练步数全部固定。
3. **小范围搜索**：从 `r ∈ {4, 8, 16, 32}` 开始，逐个训练，记录验证集上的 loss/accuracy

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制LoRA参数对模型的影响力度。代码中常写作：

```python
lora_alpha=16
```

它的核心作用就是**控制LoRA更新强度**。实际使用时，通常用 `alpha/r` 作为缩放比例——其中 `r` 是秩（rank）。比如 `r=8`、`alpha=16` 时，缩放比例为 `16/8=2`，意味着LoRA的更新会被放大2倍。

**动手理解**：你可以把 `alpha` 想象成“音量旋钮”——`alpha` 越大，LoRA对模型行为的改变越明显；`alpha` 越小，改变越温和。调整这个值，就是在调节微调的“力度”。

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 的作用是**防止LoRA参数过拟合**。它会在训练过程中随机“丢弃”一部分LoRA参数（比例为5%），迫使模型学习更鲁棒的特征，避免过度依赖某些特定参数。

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

> **注意**：上表是**相对趋势**，不是绝对数值。LoRA的可训练参数量、显存、训练速度和Adapter文件大小会随 `target_modules` 选择、秩 `r`、模型层数与 `dtype` 变化——例如把LoRA加到很多模块并用较大 `r` 时，参数量和Adapter体积都会明显上升。“更快/更省显存/MB级”通常成立，但不宜

## 块 20

## 2.1 Rank 表示什么？

**一句话理解**：Rank（秩）决定了 LoRA 在微调时“用多细的笔触”去学习新知识。

**拆开来看**：

想象原模型权重是一个巨大的矩阵，比如 `4096 × 4096`。直接更新它，计算量太大。LoRA 的做法是：不碰这个大矩阵，而是训练两个小矩阵 A 和 B，让它们的乘积 `ΔW = B × A` 来模拟对大矩阵的修改。

**Rank 就是这两个小矩阵的“中间连接尺寸”**。

具体来说，如果你设置 `r=8`：

- 矩阵 A 的形状是 `8 × 4096`
- 矩阵 B 的形状是 `4096 × 8`
- 两者相乘 `B × A`，结果正好恢复成 `4096 × 4096`

**这意味着什么？**

- **r 越小**（比如 4 或 8）：中间通道越窄，需要训练的参数越少，速度越快，但能“记住”的新知识容量也越小。
- **r 越大**（比如 64 或 128）：中间通道越宽，表达能力越强，效果可能更好，但训练成本和显存占用也随之上升。

**动手验证一下**（不需要写代码，用纸笔即可）：

假设原权重是 `4 × 4`，你设置 `r=2`。请写出 A 和 B 的形状，并确认 `B × A` 的结果维度。

> 答案自查：A 是 `2 × 4`，B 是 `4 × 2`，相乘后得到 `4 × 4`，与原始权重维度一致。

这就是 Rank 的核心作用——**在“效果”和“成本”之间做权衡**。

## 块 21

**Rank控制什么？**

Rank决定LoRA能学多少种变化方向，就像给模型“可调的旋钮”数量。

**低rank（如r=4）**：变化模式少，参数少、显存低、训练快，但表达能力有限。

**高rank（如r=64）**：自由度更高，能学复杂任务、适应力强，但参数多、显存增加、训练成本高。

**动手验证**：在PyTorch中分别设`r=4`和`r=64`，对比同一层（如4096×4096）的可训练参数量，观察差异。

## 块 22

**本块学习检查**：请手算并填写下表，验证LoRA如何压缩参数量。

| 方式 | 计算过程 | 参数量 |
| --- | --- | --- |
| 完整训练 | 4096×4096 | 1677万 |
| LoRA r=8 | 8×(4096+4096) | 6.5万 |
| LoRA r=16 | 16×(4096+4096) | 13.1万 |

**核心结论**：即使把rank从8提高到16，参数量从6.5万涨到13.1万，仍远低于全量训练的1677万。这就是LoRA“高效”的本质——只训练低秩矩阵，而非整个权重矩阵。

## 块 23

**本块学习检查**：请用 `r=2` 跑一个最小LoRA训练，观察训练loss下降速度，并记录与预期效果的差距。

---

### 4.1 Rank太小的问题

**核心直觉**：Rank（秩）决定LoRA能“记住”多少新知识。`r` 太小，就像用一张小纸条记长篇笔记——写不下，自然学不好。

**具体表现**（用 `r=2` 时）：

- **学习能力不足**：模型“脑子”容量太小，新任务的关键模式装不进去。
- **欠拟合**：训练集都学不透，更别说泛化到新数据。
- **新任务知识无法充分表达**：想学的东西被“压缩”得变了形，表达不出来。

**训练中的典型信号**：

```text
训练loss下降慢        ← 学得慢，半天降不下去
效果提升有限          ← 最终效果也一般，提升不明显
```

**怎么判断是不是这个病**：如果你发现loss曲线像“蜗牛爬”，且调高学习率也没用，先检查 `r` 是不是设太小了。`r=2` 通常只适合极简单的任务或基线测试。

---

**动手验证**（hands-on）：

1. 用 `r=2` 微调一个分类小模型，记录训练10个epoch的loss曲线。
2. 把 `r` 改成 `8`，同样训练10个epoch，对比loss下降速度。
3. 结论：如果 `r=8` 明显更快收敛，说明 `r=2` 确实“饿着”模型了。

## 块 24

**4.3 常见rank选择（动手版）**

你正在学LoRA微调，目标是快速上手。记住：下面只是**起点参考**，不是标准答案。

| 任务类型 | rank（起点） |
|---------|------------|
| 简单分类 | 4~8 |
| 文本风格调整 | 8~16 |
| 指令微调 | 16~64 |
| 复杂领域任务 | 64以上 |

**动手建议**：从表格中间值开始，跑一次验证集看效果，再上下调整。

**为什么不能照搬？** 同一个`r`，换模型大小、序列长度、batch、目标模块，显存和效果差别很大。你还要看数据量、GPU显存。

---

**五、Alpha（lora_alpha）是什么？**

LoRA不只是`ΔW = BA`，还会缩放：

```
W' = W + (α/r)BA
```

`α`就是

## 块 25

### 6.1 控制LoRA影响强度

咱们直接上手看数字。假设你设了 `r=8`、`alpha=16`，那缩放系数就是 `16/8=2`。意思是：LoRA 对权重的更新会被**放大2倍**。

如果 `alpha=8`，`8/8=1`，更新保持原始比例，不放大也不缩小。

如果 `alpha=4`，`4/8=0.5`，LoRA 的影响就**减弱一半**。

---

### 七、Alpha与Rank的关系（重点）

记住这个核心公式：**LoRA 加入模型的更新量会被 `alpha/r` 缩放**。这个比值就是控制 LoRA 影响强度的**关键缩放系数**。

**但注意**：别以为实际强度只由 `alpha/r` 决定。它只是个**缩放系数**，真正落到增量上的更新量还受这些因素影响：

- **学习率**：直接决定每步更新多大。`alpha/r` 调大时如果不下调学习率，容易一起放大导致训练震荡。很多实战里，大家更习惯调学习率来控制强度，而不是频繁改 `alpha`。
- **初始化**：常见实现里 `A` 用小随机值、`B` 初始化为0，所以训练开始时

## 块 26

**配置1与配置2的对比**

先看两组代码：

- 配置1：`r=8`，`alpha=16`，比例 = 16 ÷ 8 = **2**
- 配置2：`r=16`，`alpha=32`，比例 = 32 ÷ 16 = **2**

虽然两组rank不同（8和16），但`alpha/r`的缩放比例都是2，完全一致。

**核心要点**

调整rank时，常见做法是**同步调整alpha**，让`α/r`比例大致稳定。这只是经验做法，不是硬性规则。例如`alpha = 2 × r`（r=8配alpha=16，r=16配alpha=32）是常见起点，并非最优

## 块 27

**动手实验：从小模型开始（方案1）**

既然你喜欢边做边学，我们直接上手。先跑通一个最小配置：

```python
r=8
lora_alpha=16
```

**为什么先选这个？** 参数少、训练快，非常适合快速验证你的LoRA流程是否跑通。别急着追求效果，先让代码转起来。

**关键理解：**
- `r`（Rank）控制**LoRA容量**——记住：r越小，能学的新知识越少，但训练越快。
- `alpha`是LoRA的数学符号，表示**更新幅度**——在代码中，对应参数名为`lora_alpha`，它决定每次参数更新的步长。

**动手验证任务：** 用上面的配置跑一次微调，观察训练速度。然后只把`r`改成16、`lora_alpha`

## 块 28

### 误区1：rank越大一定越好？错！

很多新手容易觉得“rank设大点，模型能力肯定更强”。但**任务简单时**，大rank反而带来两个问题：

- **浪费显存**：更大的矩阵占用更多内存。
- **增加训练成本**：计算量上升，训练更慢。

**实操建议**：先从小rank（如4或8）开始，观察效果，不够再调大。

---

### 误区2：只调rank，不调alpha

这是最常见的配置错误。看个例子：

```python
# 原配置
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

**后果**：缩放因子从2（16/8）降到0.25，LoRA更新幅度被大幅削弱。虽然rank增大本身增加了参数量，但缩放因子的减小会抵消这部分效果，整体影响可能不变或变小。

**正确做法**：调rank时，**同步调整alpha**，保持`α/r`比例在合理范围（比如1~2之间）。

---

### 误区3：alpha越大越好？错！

alpha过大，意味着`α/r`缩放过大，会导致：

- **更新过强**：模型参数被“推”得太猛。
- **训练不稳定**：loss可能震荡甚至发散。

> **注意**：`α/r`过大只是训练不稳定的**可能原因之一**，不是唯一原因。学习率过高、数据质量差、batch size过小、精度问题（fp16/

## 块 29

**本块学习检查**：读完本块后，请暂停并回答：LoraConfig在LoRA训练流程中处于哪一步？它主要“告诉”PEFT哪四件事？

---

## 一、LoraConfig是什么？

### 1.1 PEFT中的LoraConfig作用

**PEFT**（Parameter-Efficient Fine-Tuning，参数高效微调）是 Hugging Face 提供的库，让你不用训练全部模型参数，也能完成微调。

**LoRA训练流程**（共5步）：

1. 加载基础模型
2. **创建LoraConfig** ← 本块主角
3. 指定LoRA插入位置
4. 生成LoRA Adapter
5. 训练LoRA参数

**LoraConfig** 就是第2步的“配置单”，它负责告诉PEFT四件事：

| 问题 | 对应配置项 |
|------|-----------|
| LoRA放在哪里？ | target_modules |
| LoRA规模多大？ | r |
| LoRA训练强度如何？ | lora_alpha |
| 是否使用dropout？ | lora_dropout |

**一句话理解**：LoraConfig 是 LoRA 微调的“总开关”，你通过它告诉 PEFT 在模型的哪些层、用多大规模、以多强力度、是否加防过拟合手段来插入 LoRA 训练。

---

**

## 块 30

## 1.2 一个简单示例：动手搭一个LoRA配置

你正在学LoRA微调，那就直接上手看代码。下面这段就是最核心的配置，先读一遍，感受一下结构：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**这4个参数就是LoRA的灵魂**，逐个拆开看：

| 参数 | 作用 | 直观理解 |
|------|------|----------|
| `target_modules` | 决定插入哪些层 | 告诉LoRA“改造模型的哪些部位” |
| `r` | 决定LoRA矩阵大小 | 数值越大，新增参数越多，学习能力越强 |
| `lora_alpha` | 控制缩放强度 | 调节LoRA对原模型的影响力度 |
| `lora_dropout` | 控制随机失活 | 防止过拟合，训练时随机丢弃部分信息 |

---

### 完整流程：配置怎么用起来

光有配置还不够，要把它接到模型上。整体关系如下：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

对应代码：

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,

## 块 31

### 3.1 target_modules是什么？

一句话理解：**`target_modules` 就是告诉 LoRA：“你要往模型的哪些模块里‘打补丁’。”**

在 Transformer 模型里，注意力机制（Attention）内部通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA 默认**不会**动所有层。你需要手动指定，比如：

```python
target_modules=["q_proj", "v_proj"]
```

这表示：只给 **Query 投影层** 和 **Value 投影层** 增加 LoRA。

---

### 四、为什么选 q_proj 和 v_proj？

看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

- **Q（Query）** 决定：模型**关注什么信息**。
- **V（Value）** 决定：模型**传递什么信息**。

所以，调整 Q 和 V，通常就能有效改变模型行为——这是最常见的起点。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型的“标准配置”。不同模型的模块命名不同，**配置前务必先用 `model.named_modules()` 查看实际模块名**，再决定 `target_modules`。

---

### 偷懒技巧：`target_modules="all-linear"`

如果你不想手工枚举模块名，较新版本的 PEFT（一般 ≥ 0.7）支持直接写：

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
    target_modules="all-linear",   # 自动匹配所有 nn.Linear 层，包括输出层 lm_head（若为 nn.Linear），但通常建议排除 lm_head 以避免过拟合
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**注意权衡**：`"all-linear"` 覆盖的模块变多，**可训练参数和显存占用也会增加**。是否用全线性层，要在**参数量/显存**和**效果**之间取舍，最终以验证集结果为准。如果你的 PEFT 版本太旧不支持这个字符串，就退回显式列出模块名的写法。

---

### 🛠️ 动手验证任务

运行上面的代码，观察 `print_trainable_parameters()` 的输出：
1. 记录可训练参数的总数和占比。
2. 对比：如果把 `

## 块 32

### 方案1：只调整Q和V（你的轻量起点）

既然你的目标是快速上手LoRA微调，且偏好动手实践，这个方案最适合作为第一步。

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么先选它？** 因为只调整注意力机制中的查询（Q）和值（V）投影层，带来的直接好处是：

- **参数少**：只训练一小部分权重，模型改动小。
- **训练快**：计算量小，迭代速度快，适合你每天1小时的节奏，能快速看到效果。

这是一个常见的轻量级起点，不限定于某个模型规模，具体效果仍需结合你的模型结构验证。

---

**动手验证（不引入新知识）**：在你已有的微调脚本中，将`target_modules`设为上述列表，观察训练日志中可训练参数量相比全量微调减少了多少。记录这个数字，作为后续对比的基线。

## 块 33

## LLaMA系列：`target_modules` 怎么配？

你是初学者，目标是跑通LoRA微调，而且喜欢动手试。这一块我们只讲一件事：**LLaMA模型里，`target_modules` 该写什么。**

直接给结论，LLaMA系列最常见的配置是：

```python
[
"q_proj",
"v_proj"
]
```

**为什么是这两个？** 简单理解：`q_proj` 和 `v_proj` 是LLaMA注意力机制里负责计算查询（Q）和值（V）的投影层。LoRA通过在**部分**层上注入低秩矩阵来微调，而实践中最常用、效果也稳的组合就是同时选Q和V。你不需要深究每个投影的数学细节，先记住这个“标配”即可。

> **动手验证一下**：在代码里把 `target_modules` 分别设为 `["q_proj"]`、`["v_proj"]`、`["q_proj", "v_proj"]`，用 `model.print_trainable_parameters()` 看可训练参数量变化。你会发现第三种组合参数量最多，这也是它最常用的原因之一。

---

## 对比：GPT-2 为什么不一样？

如果你换用GPT-2，`target_modules` 可能要写成：

```python
[
"c_attn"
]
```

**关键区别在这里（重要，但别慌）：**

GPT-2的注意力用的是Transformers里的 `Conv1D`（不是普通的 `nn.Linear`），它把Q/K/V三个投影**合并**在了一个 `c_attn` 模块里。而且 `Conv1D` 的权重形状是 `(in_features, out_features)`——**和 `nn.Linear` 的 `(out_features, in_features)` 正好转置**。

所以，在GPT-2这类层上加LoRA时，需要设置：

```python
fan_in_fan_out=True
```

这样PEFT才能按转置布局正确注入低秩矩阵。**不设这个，LoRA可能静默出错或效果不对。**

**可运行的GPT-2完整示例：**

```python
from peft import LoraConfig, get_peft_model
from

## 块 34

**动手查一查，别照抄配置**

BERT 这类模型的 LoRA 配置不能直接复制 LLaMA 的。BERT 的 `target_modules` 通常写 `["q_proj", "v_proj"]`，但不同模型命名不同，照搬会报错或没效果。

**操作步骤（跟着做一遍）**

1. 加载你的 BERT 模型后，运行：
```python
# 注意：加载模型时请确保显存充足，例如 BERT-base 约需 2GB 显存
for name, _ in model.named_modules():
    print(name)
```
2. 在输出里找注意力层的名字，你会看到类似 `q_proj`、`v_proj` 这样的模块名。
3. 把找到的名字填进 `target_modules`，例如：
```python
target_modules=["q_proj", "v_proj"]
```

**关键机制**

- `target_modules` 是**按模块名后缀匹配**的。写 `"q_proj"` 会命中所有以 `q_proj` 结尾的层（各层各头

## 块 35

## 6.1 r是什么？

r就是**rank（秩）**，它控制LoRA低秩矩阵的维度。

核心公式是：

```text
ΔW = (α/r)BA
```

r决定的是A、B两个矩阵的**中间维度**。

**举个例子**：假设原矩阵是4096×4096，设置`r=8`，那么：

- A的形状是：4096×8
- B的形状是：8×4096

这样，你只需要训练这两个小矩阵，而不是整个4096×4096的大矩阵。

---

## 6.2 r影响什么？

**r越大**：
- 优势：学习能力更强，能表达更复杂的任务
- 缺点：参数增加，显存占用也

## 块 36

## 7.1 alpha作用：控制LoRA的“音量”

LoRA最终更新时，**不是**直接用 `BA`，而是用：

```text
(alpha/r)BA
```

也就是说，`alpha` 控制的是 **LoRA更新幅度**——相当于一个“音量旋钮”。

**动手算一算：**

配置：

```python
r=8
lora_alpha=16
```

缩放系数 = `16/8 = 2`

这意味着 LoRA 的影响被**增强**了（原始更新乘以2）。

---

## 7.2 alpha常见设置

一个常见的**起点**是：

```text
alpha = 2 × r
```

> 注意：这只是经验起点，不是通用规则。也可以固定 alpha 只调 r，或用其他比例。**最终以验证集表现定**。

## 块 37

**8.1 dropout作用**

训练时，LoRA会**随机关闭部分神经连接**，目的是**减少过拟合**（模型死记训练数据，泛化变差）。

配置示例：
```python
lora_dropout=0.05
```
含义：5%的LoRA路径随机失活。

---

**8.2 如何选值？**（经验起点，非固定规则）

| 数据规模 | dropout起点 |
|---------|------------|
| 数据少   | 0.05~0.1   |
| 数据多   | 0~0.05     |

若训练数据非常丰富，可直接设：
```python
lora_dropout=0
```

---

**本块学习检查**：

## 块 38

## 9.1 基础配置——动手搭一个LoRA

这一节我们直接上手，用代码把LoRA配置写出来。下面这个配置是**7B模型测试**的常用起点，你先照着敲一遍，感受每个参数的位置：

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

**逐个拆解，记住这5个关键参数：**

| 参数 | 值 | 作用 | 你的理解 |
|------|-----|------|---------|
| `r` | 8 | LoRA容量 | 秩的大小，决定可学习参数的量 |
| `lora_alpha` | 16 | 更新缩放 | 控制LoRA对原模型的影响强度

## 块 39

## 9.1 基础配置：跑通一个完整的LoRA微调流程

你现在的目标是LoRA微调，而且喜欢动手实操。这一节我们就直接上手，把代码跑通，同时搞清楚四个关键参数（`r`、`lora_alpha`、`lora_dropout`、`target_modules`）到底在干什么。

### 第一步：加载基础模型

```python
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里加载的是带分类头的BERT模型。注意：**分类头是新初始化的**，里面的权重是随机的，所以它必须参与训练，否则模型没法做分类任务。

# 模型许可证：BERT模型采用Apache 2.0许可证，可自由商用。
# 注意：bert-base-uncased FP32 推理需要约1.2GB显存，训练时需额外预留优化器与梯度显存。

### 第二步：配置LoraConfig

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                  # LoRA矩阵的秩，越小参数越少
    lora_alpha=16,        # 缩放系数，控制LoRA更新的强度
    lora_dropout=0.05,    # 防止过拟合，训练时随机丢弃部分LoRA参数
    bias="none",          # 不训练偏置项
    task_type="SEQ_CLS",  # 任务类型：序列分类
    target_modules=["query", "value"],  # 只微调BERT注意力中的query和value层
    modules_to_save=["classifier"],      # 关键！让分类头也参与训练并保存
)
```

**重点理解两个参数：**

- `target_modules=["query", "value"]`：只对BERT注意力机制中的query和value层做LoRA微调，其他层冻结不动。这是LoRA省参数的核心——只改一小部分权重。
- `modules_to_save=["classifier"]`：这个参数很关键。因为分类头是随机初始化的，如果不加这个参数，分类头就不会被训练和保存，模型重载后分类结果会完全错乱。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

运行后你会看到，可训练的参数包括LoRA参数和classifier分类头，而BERT主体大部分参数都是冻结的。

### 第四步：保存模型

```python
model.save_pretrained("./lora_ckpt")
```

注意：这里保存的**不是完整模型**，而是Adapter（LoRA权重）+ `modules_to_save`指定的分类头。BERT主体参数不会被保存。

### 第五步：重载并验证

```python
from peft import PeftModel

reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")
```

重载时必须先加载**同一个**基础模型（`bert-base-uncased`），再挂载Adapter。

**验证分类头是否被正确保存：**

```python
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

如果输出`True`，说明分类头被正确保存和恢复了。如果输出`False`，说明分类头回到了随机初始化。

## 块 40

## 10.1 普通文本生成：从LoRA微调你的第一个文本模型

你现在的目标是**LoRA微调**，而且偏好动手实践。这一节我们直接看“普通文本生成”任务下，`LoraConfig` 里最核心的三个参数该怎么设。

### 直接可用的配置

对于普通文本生成（比如让模型续写、写故事），推荐这样设置：

```python
r=8
alpha=16
dropout=0.05
```

### 这三个参数分别是什么？

- **`r`（秩）**：决定 LoRA 引入的低秩矩阵的维度。`r=8` 意味着我们用较小的参数量去近似权重更新，这是 LoRA 高效的关键——你不需要微调全部参数，只训练这部分低秩矩阵。
- **`alpha`（缩放因子）**：控制 LoRA 更新的缩放比例。这里 `alpha=16`，是 `r` 的两倍。它的作用是调节 LoRA 分支对原模型的影响强度。
- **`dropout`（丢弃率）**：防止过拟合。`0.05` 是一个较小的值，意味着在训练时随机丢弃 5% 的 LoRA 参数，让模型更稳健。

### 为什么普通文本生成用这个组合？

普通文本生成通常任务相对直接，`r=8` 已经足够捕捉任务所需的适应能力，参数少、训练快。`alpha=16` 配合 `r=8`，缩放比例适中（`alpha/r = 2`），既不会让更新太激进，也不会太微弱。`dropout=0.05` 在数据量不是特别大的情况下，提供了基本的正则化保护。

### 动手验证一下

你可以直接打开 Python 环境，运行下面这段代码，观察配置是否生效：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"  # 普通文本生成对应因果语言模型
)
print(config)
```

如果输出中显示 `r: 8`、`lora_alpha: 16`、`lora_dropout: 0.05`，说明配置正确。

### 一个提醒

注意：这里只给出了 `r`、`alpha`、`dropout` 三个参数。完整的 `LoraConfig` 还需要 `target_modules`（指定要适配哪些层）等参数，这些在后续章节会讲到。现在先记住这个“普通

## 块 41

## 普通文本生成：LoRA配置逐项拆解

你已掌握PyTorch和HuggingFace基础，目标是用LoRA微调模型。下面这份配置是文本生成任务的典型起点，我们逐项拆开看。

```python
LoraConfig(
    r=8,              # LoRA容量：秩越大，可学习的参数越多
    lora_alpha=16,    # 缩放因子：实际更新强度由 α/r 决定，此处 α/r = 2
    target_modules=["q_proj", "v_proj"],  # 只改造注意力层的Q和V投影
    lora_dropout=0.05 # 训练时对LoRA层的输出应用dropout，丢弃比例为5%，

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经知道LoRA的核心思路：**冻结原模型，只训练一小部分额外参数**。这确实大幅减少了需要更新的参数量，但有一个关键点容易被忽略——

> **基础模型本身，仍然要完整加载到显存里。**

我们来算一笔账。假设你要微调一个7B（70亿参数）模型，以FP16精度加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是**模型权重**的占用。实际训练时，显存里还要同时放：

- **激活值**：前向传播过程中每层的中间结果；
- **梯度**：反向传播时计算出来的更新方向；
- **优化器状态**：比如Adam优化器维护的动量等额外变量。

把这些都加起来，实际显存需求会**远高于14GB**。这就是为什么

## 块 43

**QLoRA核心思想（动手版）**

你已有PyTorch和HuggingFace基础，目标是用LoRA微调。先记住一句话：**QLoRA = 量化 + LoRA**。

**它和普通LoRA的区别？**

- 普通LoRA：加载FP16模型，插入LoRA训练。
- QLoRA：先把模型压成4-bit，再插入LoRA训练。

**流程（按顺序操作）：**

1. 加载FP16基础模型
2. 量化为4-bit
3. 加载量化后的模型
4. 插入LoRA适配器
5. 只训练LoRA参数

**关键点（别搞混）：**

- **量化对象**：基础模型的权重（冻结

## 块 44

## 3.1 模型存储方式：从14GB到3.5GB

先记住这张表——它解释了为什么量化能省显存：

| 类型 | 每个参数占用 |
|------|------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一遍**（7B模型）：

- FP16：7B × 2 ≈ **14GB**
- 4-bit：7B × 0.5 ≈ **3.5GB**

显存直接减少约75%。

> ⚠️ 注意：这3.5GB是**理论下界**，只算了量化后的权重本身。实际占用更高，因为还要算上量化常数/缩放因子、分组元数据（NF4按组量化，每组都存额外参数）、未量化的模块（norm、embedding、lm_head、LoRA参数），以及训练/推理时的激活值和缓存。**真实占用以实际运行为准**。

## 3.2 为什么不用普通INT4？

普通INT4整数范围只有 **-8 ~ 7**。但神经网络权重通常：

- 接近正态分布
- 小数多

## 块 45

**NF4是什么？——给想动手做LoRA的你**

NF4，全称NormalFloat4，中文叫“正态分布4位浮点”。别被名字骗了，它不是IEEE 754那种通用浮点格式，而是QLoRA论文为**神经网络权重**专门设计的一套4-bit量化方案。

**核心思路**：权重值大致呈正态分布——0附近最密集，两端稀疏。NF4使用分位数量化，基于标准正态分布的分位数将归一化权重映射到16个量化级别，量化点由输入分布决定。量化前使用绝对最大值归一化，反量化时乘以缩放因子还原。

**一句话记忆**：NF4不是“能表示任意数的浮点”，而是“为近似正态分布量身定

## 块 46

**4.2 为什么适合模型权重？**

大模型训练完的权重，分布近似**正态分布**——大量参数挤在0附近。普通INT4是平均划分（如[-8,-7,...,7]），对中心密集区表达很差。NF4则按正态分布设计量化范围，**0附近精度更高**，更贴合权重分布，精度损失更低。

**4.3 NF4量化过程**

FP16权重（如0.023、-0.145、0.876）→ 归一化 → 映射到NF4 → 保存4-bit值。推理时反量化回近似FP16。

---

**本块学习检查**：动手验证——取三个FP16权重（一个接近0，两个远离0），写出NF4量化后哪个精度

## 块 47

**本块学习检查**（动手验证）：

打开你的代码，找到QLoRA配置里`quantization_config`的部分。先只开启4-bit量化（`load_in_4bit=True`），打印模型显存占用；再开启双重量化（`double_quant=True`），再次打印。对比两次数值，你会发现第二次的显存更小——这就是双重量化省下的空间。

---

**正文开始**

你已经知道，4-bit量化会把FP16权重压成4-bit来省显存。但你可能忽略了一个细节：**量化本身也要存参数**。

具体来说，每组权重都需要保存两个东西：`scale`（缩放因子）和`zero point`（零点）。这些参数虽然单个不大，但模型有成千上万组权重，加起来就是一笔不小的显存开销。

双重量化的思路

## 块 48

### 6.1 为什么需要分页优化器？

训练大模型时，显存不只是被模型本身占满。梯度、优化器状态（比如Adam里的一阶动量和二阶动量）都会额外吃显存。QLoRA虽然把模型显存降下来了，但训练过程中仍可能突然显存不足，直接OOM中断。

### 6.2 分页优化器思想

Paged Optimizer借鉴了操作系统虚拟内存的思路。核心是：借助NVIDIA的统一内存（Unified Memory），在显存出现峰值、即将OOM时，把优化器状态自动“分页”换出到CPU内存，需要时再换回GPU。

流程：GPU显存 → 出现峰值 → 优化器状态按页换出到CPU内存 → 需要计算时再换回GPU。

注意：这不是平时就把优化器状态常驻CPU，而是应对显存峰值的机制——平稳时状态尽量留在GPU，只在峰值瞬间临时换出，避免偶发OOM中断训练。

**本块学习检查**：用自己的话复述“分页优化器解决什么问题”和“它如何工作”，对照原文检查是否准确。

## 块 49

## 6.3 类比理解

把显存想象成你的**桌面**。

**普通方式**：所有东西（模型参数、梯度、优化器状态）都堆在桌面上。桌面满了，就**无法继续**，直接报错（OOM）。

**分页优化器**：桌面只放**常用文件**（当前计算需要的参数）；**暂不用的文件**（历史优化器状态）自动存进**柜子**（CPU内存）；**需要时再取出**。这样桌面永远有空间，训练不中断。

---

## 6.4 如何在训练中实际启用分页优化器？

不用手写换页逻辑，`bitsandbytes` 已实现。只需在 `TrainingArguments` 里设置 `optim` 参数：

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
    optim="paged_adamw_8bit",        # 或 "paged_adamw_32bit"
)
```

**关键点**：

- `"paged_adamw_8bit"`：优化器状态用 8-bit 存储，**更省显存**。
- `"paged_adamw_32bit"`：精度更高，但**占用更多显存**。
- 参数名随版本变化（还有 `adamw_bnb_8bit` 等），以你装的版本文档为准；报"未知 optim"就换当前版本支持的名称。
- 分页优化器主要解决**峰值OOM**，通常配合 `gradient_checkpointing=True` + 小 batch + 梯度累积一起用。

---

### 本块学习检查

**动手验证**：打开你的训练脚本，找到 `TrainingArguments`，把 `optim` 改为 `"paged_adamw_8bit"`，运行训练。观察显存占用是否比之前更低、是否不再中途OOM。如果报错

## 块 50

**7.1 4-bit量化：给模型“瘦身”**

你的目标是LoRA微调，但显存有限？QLoRA的第一步就是解决模型太大装不下的问题。

**核心思路**：把模型权重从FP16（16位精度）压缩到NF4（4位精度）。精度降低，体积直接缩小约4倍，显存占用大幅下降。

**直观理解**：FP16像用高精度秤称重，NF4像用粗略刻度尺量——牺牲一点精度，换来了能装进显存的机会。

---

**7.2 双重量化：连“刻度尺”也压缩**

量化时需要一个`scale`（缩放系数）来还原数值。这个系数本身也占空间。

**双重量化**：把这个`scale`再量化一次，进一步减少参数占用。相当于把“刻度尺”也做得更小巧。

---

**7.3 分页优化器：显存不够，内存来凑**

训练时优化器状态也吃显存。分页优化器让GPU和CPU内存之间可以交换数据——显存紧张时，部分数据暂时放到CPU内存，需要时再取回。

**效果**：降低训练时的

## 块 51

### 4-bit 加载配置：一次说清四个参数

QLoRA 的核心思路是：**用 4-bit 精度存权重，用更高精度算矩阵**。这样既省显存，又不牺牲训练效果。下面四个参数就是实现这一思路的关键开关。

#### 1. `load_in_4bit=True`：开启 4-bit 加载
这是总开关。设为 `True` 后，模型权重会以 4-bit 格式加载进显存，而不是默认的 32-bit。这一步直接让显存占用大幅下降。

#### 2. `bnb_4bit_quant_type="nf4"`：选择量化方式
`nf4` 是一种专门为 4-bit 设计的量化类型，能更好地保留权重分布的信息。在 QLoRA 中，这就是默认且推荐的选择。

#### 3. `bnb_4bit_use_double_quant=True`：开启双重量化
这个开关会**对量化过程中的额外开销（如缩放因子）再做一次量化**，进一步节省显存。开启后，显存占用会更低，而精度损失几乎可以忽略。

#### 4. `bnb_4bit_compute_dtype`：指定计算精度（最容易踩坑）
这里要区分两个概念：
- **存储 dtype**：权重在显存里以 4-bit（NF4）保存，这是为了省空间。
- **计算 dtype**：权重参与矩阵运算时，会被**反量化**回更高精度（如 16-bit）来计算，这个精度就是由 `bnb_4bit_compute_dtype` 决定的。

**关键点**：如果不显式设置，默认通常为 `None`，此时计算精度会回退到模型加载时指定的 `torch_dtype`（如 `float16` 或 `bfloat16`），但为了确保性能与稳定性，建议显式指定。

**怎么选？别写死，让代码帮你判断**：
```python
import torch
from transformers import BitsAndBytesConfig

# 根据 GPU 能力自动选择：支持 BF16 就用 bfloat16，否则退回 float16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载模型）、分页优化器（`optim`）和 `bnb_4bit_compute_dtype`（计算精度）。现在把它们**拼成一个能跑的 QLoRA 训练骨架**——这是你从“看懂配置”到“跑通代码”的关键一步。

整个流程分 6 步，顺序不能乱：

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

这里做了三件事：开启 4-bit 加载、用 NF4 量化类型、开启双重量化。`compute_dtype` 根据你的 GPU 是否支持 bfloat16 自动选择——支持就用 bfloat16，否则退回 float16。

**第 2 步：以 4-bit 方式加载基础模型**

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
except Exception as e:
    raise RuntimeError(f"模型加载失败，请检查网络连接或模型名称是否正确：{e}")
```

注意：如果 tokenizer 没有 pad_token，就用 eos_token 顶上（否则训练时 padding 会报错）。`device_map="auto"` 让模型自动分配到可用设备上。模型加载涉及网络和 IO 操作，用 `try-except` 捕获异常并给出友好提示。

**第 3 步：为 k-bit 训练做准备（关键一步）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步会冻结模型的所有参数（后续只训练 LoRA 注入的低秩增量），同时开启梯度检查点所需的输入梯度，并将 LayerNorm 等层保持为 float32 精度，避免 4-bit 精度损失影响训练稳定性。**`use_gradient_checkpointing=True` 必须和后面 TrainingArguments 里的 `gradient_checkpointing=True` 对应**，两者要同时开。

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

只训练低秩增量参数（LoRA），原始模型权重冻结。`target_modules` 指定注入哪些模块——这里选了 `q_proj` 和 `v_proj`（注意力层的 Q 和 V 投影），实际使用时按你的模型模块名调整，或者直接用 `"all-linear"` 省事。`print_trainable_parameters()` 会打印可训练参数量，帮你确认 LoRA 注入成功。

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

Causal LM 的 `labels` 直接复制 `input_ids`（因为语言模型是自回归的，输入即标签）。真实场景替换成你自己的数据，这里只是跑通流程。

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
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

try:
    trainer.train()
except Exception as e:
    raise RuntimeError(f"训练中断，请检查显存或数据格式：{e}")
```

两个关键点：`optim="paged_adamw_8bit"` 是分页优化器（依赖 bitsandbytes，名称随版本可能略有不同）；`gradient_checkpointing=True` 与第 3 步呼应，用计算换显存——batch_size 设为 1 是因为 4-bit + 梯度检查点已经大幅降低显存占用（约需 8GB 显存），但小 batch 更稳。`tokenizer` 参数在 Transformers 新版本中也可写作 `processing_class`，旧版本请使用 `tokenizer`。

**整体逻辑回顾**：量化配置 → 加载模型 → 准备 k-bit 训练 → 注入 LoRA → 准备数据 → 配置训练参数 → 启动。每一步都是前一步的基础，顺序不能乱。

**模型许可证说明**：使用模型前请确认其许可证类型。例如，若使用 Llama 2，需遵守 Llama 2 Community License（商用月活>7亿需Meta授权）。实际部署时请以所选模型的具体许可证条款为准。

## 块 53

## 九、把 QLoRA 的“五步走”串起来（可跑通骨架）

你之前已经分别了解了 4-bit 量化（NF4）、双重量化和分页优化器。现在最关键的一步来了：**怎么把这些零件组装成一段能跑的代码骨架**。

下面这 5 步，每一步都缺一不可，顺序也不能乱：

**第 1 步：`BitsAndBytesConfig`——决定模型怎么“瘦身”加载**
这一步是给基础模型设定 4-bit 加载方式。核心三要素：用 NF4 量化、开启双重量化，并可指定计算精度（compute dtype）。说白了，就是告诉模型“你以 4-bit 的压缩形态进来，但计算时用更高精度”。

**第 2 步：`prepare_model_for_kbit_training()`——QLoRA 的“必要一步”**
这是最容易踩坑的地方。它的作用是冻结量化模型的参数并准备训练，可启用输入梯度以支持梯度检查点。**跳过它，训练常常不收敛或直接报错**。记住：这不是可选项，是必选项。

**第 3 步：`get_peft_model(..., LoraConfig)`——注入 LoRA 适配器**
这一步把可训练的 LoRA 参数“插”进冻结的 4-bit 主干模型里。主干模型保持冻结（不更新），只有 LoRA 参数在训练中更新。

**第 4 步：`gradient_checkpointing=True`——两处建议对应开启**
建议在 `TrainingArguments` 里设置 `gradient_checkpointing=True`，同时在 `prepare_model_for_kbit_training` 中设置 `use_gradient_checkpointing=True` 以保持一致，但并非强制。

**第 5 步：`optim="paged_adamw_8bit"`——启用分页优化器**
这一步用来缓解峰值显存。优化器状态可被分页到 CPU 内存或统一内存，以降低显存峰值。

> ⚠️ **提醒**：上面是“能说明关系”的最小骨架。真实训练时，你需要换成足够规模的数据、调好超参，并且**按你实际使用的库版本**核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数写法。

---

## 十、你的电脑能跑吗？

**如果你只学原理**：不需要 GPU，纯看代码和概念就行。

**如果你想实际跑 QLoRA**，硬件参考如下：

| 硬件 | 最低要求 |
|------|----------|
| CPU | 4核+ |
| 内存 | 16GB+ |
| GPU | NVIDIA CUDA GPU |

**显存参考**（以

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会误以为LoRA训练完，模型文件就变大了。其实不是这样。

训练过程中，**基础模型是被冻结的**，不参与参数更新。真正被训练的是一个额外的、小得多的参数集合，叫做 **LoRA Adapter**。

最终的结构是：

```text
基础模型（冻结） + LoRA Adapter（训练）
```

比如你用Llama-7B做LoRA训练，保存后通常只会得到两个文件：

```text
adapter_model.safetensors
adapter_config.json
```

而不是一个完整的Llama-7B模型文件。

### 1.2 Adapter有什么特点？

**优点：**

- 文件小——一个Adapter可能只有几十到几百MB，而完整模型动辄好几GB；
- 保存快——训练完保存权重非常迅速；
- 灵活——同一个基础模型可以搭配多个Adapter，各干各的活。

举个例子，同一个基础模型可以同时搭配：

```text
客服Adapter
医疗Adapter
法律Adapter
```

**缺点：**

部署推理时，你需要**基础模型 + Adapter**两个部分同时存在，缺一不可。

---

## 二、为什么需要合并LoRA权重？

既然Adapter这么方便，为什么还要合并？

核心原因在于**部署的便利性**。带着Adapter跑推理，每次都要加载两份文件、做两次前向计算，流程繁琐。而合并之后，Adapter的权重被直接写进基础模型，得到一个**完整的、独立的模型文件**，部署时只需要加载这一个文件即可。

```python
# 注意：Llama-2-7b FP16

## 块 55

**本块学习检查**：读完本块后，请合上资料，自己画出“未合并”和“合并后”两条推理流程图，并写出合并的数学公式。

---

**2.1 推理时的问题**

你刚接触LoRA，先看“未合并”的状态：模型推理时，输入要同时经过**基础模型**和**LoRA适配器**，所以你需要加载两个东西——基础模型和Adapter。这对部署来说很麻烦，你更希望输入直接进入一个“完整模型”就得到输出。因此，我们需要**合并权重**。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要

## 块 56

好的，我们来把这块内容变成你的专属学习指南。

你的目标是搞定 LoRA 微调，而且喜欢动手实践。咱们就直奔主题，把“合并”这一步彻底弄明白。

---

### 3.1 merge_and_unload 是干什么的？

简单说，`merge_and_unload()` 这个方法就干一件事：**把 LoRA 训练出来的“补丁”正式“焊死”到基础模型上，然后拆掉 LoRA 的“外挂包装”**。

你可以这样理解，训练完的模型是：

```text
基础模型 (Base Model) + LoRA 适配器 (LoRA Adapter)
```

执行 `merge_and_unload()` 之后，它就变成了一个**完整的、独立的模型**：

```text
完整模型 (包含 LoRA 更新后的权重)
```

### 3.2 合并前后，模型内部发生了什么变化？

- **合并前**：模型是一个 `PeftModel`，内部结构是“基础模型”和“LoRA适配器”两部分。
- **合并后**：模型变成了一个**普通的 Transformer 模型**，LoRA 的权重已经被整合进了基础模型的权重里，不再有单独的适配器。

---

### 四、LoRA 模型合并的完整流程

整个流程就像一条流水线，非常清晰：

```text
训练 LoRA → 保存 Adapter → 加载基础模型 → 加载 Adapter → 执行 merge_and_unload() → 保存完整模型
```

我们接下来就一步步走一遍。

### 五、第一步：加载基础模型和 LoRA Adapter

假设你已经有了训练好的 LoRA 适配器，现在要把它和基础模型“组装”起来。

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
base_model_path = "model_path"  # 你的基础模型路径
model = AutoModelForCausalLM.from_pretrained(base_model_path)
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存

# 2. 加载 LoRA 适配器
adapter_path = "adapter_path"  # 你的 LoRA 适配器路径
model = PeftModel.from_pretrained(model, adapter_path)
```

现在，你的 `model` 就是一个带着 LoRA 适配器的 `PeftModel` 了。

### 六、第二步：执行 merge_and_unload() 合并权重

这一步是核心，代码就一行：

```python
# 3. 合并权重并卸载 PEFT 包装
merged_model = model.merge_and_unload()
```

执行完这行代码后，`merged_model` 就不再是 `PeftModel`，而是一个**普通的 Transformer 模型**了。它的权重已经包含了 LoRA 的更新。

### 七、第三步：导出合并后的完整模型

最后，把合并好的模型和它的分词器（Tokenizer）保存到本地。

```python
# 4. 保存合并后的模型和分词器
merged_model.save_p

## 块 57

## 三、merge_and_unload 是什么？（重点）

一句话理解：`merge_and_unload` 就是把训练好的 LoRA 小适配器（Adapter）“焊死”回基础模型里，得到一个完整的、独立的模型文件。之后加载推理时，**不再需要 PEFT 库**，直接用 Transformers 就能跑。

### 合并后必须核查的 4 件事

合并导出不是点完按钮就结束，建议按下面清单逐项检查，避免“能加载但行为不对”：

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致。重点检查训练中**新增或修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里。如果训练时扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。
2. **generation_config.json**：如果你依赖特定的生成默认值（eos/pad token id、`max_new_tokens`、采样参数等），确认它一并导出，否则推理时可能回退到不合适的默认行为。
3. **dtype**：确认合并/保存时的精度符合预期（FP16/BF16 还是 FP32）。dtype 不一致会影响显存占用与数值表现。
4. **config.json 完整性**：确认模型结构配置（层数、hidden size、任务头等）正确，且已**移除 PEFT/adapter 相关配置**。合并后应是普通 Transformers 模型，不应再残留 `adapter_config.json`。

---

## 八、合并后如何加载推理？

合并后，加载流程变得非常简单：

**不需要**：`peft`

**只需要**：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

**推理流程**：

```text
用户输入 → 完整模型 → 生成结果
```

---

## 九、LoRA 合并与不合并的区别

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 P

## 块 58

**本块学习检查**：读完下面两个场景后，请暂停并回答——如果你只想部署一个固定模型，且不想引入额外依赖，该选合并还是不合并？为什么？

---

**场景1：上线部署（单一Adapter、追求简单）**

当生产环境只需要一个固定Adapter，且希望部署链路尽量简单——不额外依赖PEFT、不做运行时Adapter加载——合并通常更方便。

> 注意：如果推理框架本身支持直接加载LoRA（如vLLM多LoRA服务），或需要热切换多个Adapter，不合并反而更灵活。是否合并要看框架能力与业务需求。

**场景2：分享模型**

想让对方开箱即用、无需理解Adapter机制，可发布合并后的完整模型，用户无需额外下载Adapter。

> 权衡：合并后体积明显更大（等于一份完整基础模型），分发/存储成本更高；若对方已有同一基础模型，单独分享体积很小的Adapter更省带宽。按受众与成本选择。

## 块 59

## 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你的基础模型是一个“通用底座”，上面可以插不同的“技能卡”（Adapter A/B/C）。合并就像把卡片焊死在底座上——以后想换卡就得重来。**保留Adapter，随时切换，更灵活。**

**场景2：持续训练**

如果你打算继续微调LoRA，先合并反而多此一举。**保留Adapter**，下次训练直接接着调，省时省力。

---

## 十一、QLoRA合并注意事项（重点）

QLoRA有个“精度差”问题：

- 基础模型：**4-bit量化**存储（NF4）
- LoRA参数：**FP16/BF16**（高精度）

训练

## 块 60

### 直接合并可能遇到的情况（QLoRA 模型合并注意事项）

你训练完 QLoRA 模型后，最想做的事可能就是直接把 LoRA 权重合并回基础模型，然后保存成一个完整的模型文件。这个想法很直接，但**在 4-bit 量化状态下直接合并，可能会遇到两个问题**：

1. **精度损失**：4-bit 权重需要先反量化（转回高精度）再合并，这个过程和全程用高精度计算并不完全等价，可能会导致最终模型的质量略有下降。
2. **类型或版本不兼容**：可能会出现权重类型不匹配，或者你当前使用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接调用 `merge_and_unload`。

**注意**：上面说的只是“可能出现的情况”，并不是每次都会报错。能不能顺利合并、合并质量如何，取决于你用的库版本、具体的量化配置，以及是否按下面说的流程先反量化。**请以你所用版本的官方文档和实际运行结果为准**。

---

### 常见做法：先反量化到高精度再合并

为了避免上面那些坑，一个稳妥的做法是：**不要在 4-bit 状态下直接合并**。而是先用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一个 Adapter，最后合并。

整个流程可以概括为：

```text
以高精度（FP16/BF16）重新加载基础模型
        ↓
加载同一份 LoRA Adapter
        ↓
merge_and_unload()
        ↓
保存完整模型
```

这样，合并发生在高精度权重上，就避开了 4-bit 状态下的类型和精度问题。

下面是这个流程对应的完整可执行代码。**核心要点**：重新加载基础模型时，**不要传 `quantization_config`**，这样才能以高精度加载。

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

**模型许可证**：使用本流程合并模型时，请遵守基础模型的许可证条款。例如，若基础模型为 Llama 2，则适用 Llama 2 Community License（商用月活>7亿需Meta授权）。具体请以你所用基础模型的官方许可证为准。

## 块 61

你正在做LoRA微调，目标是最终导出合并后的模型。这一块讲的是QLoRA合并时最关键的“坑”，直接关系到你能否顺利拿到最终权重。

**核心动作**：合并时，**加载基础模型不要再传 `quantization_config` 或 `load_in_4bit`**。也就是说，要用高精度（FP16/BF16）重新加载完整模型，再执行 `merge_and_unload`。

**为什么**：不同PEFT版本对“量化模型能否直接合并”支持不同，有的要求先反量化。用高精度加载是稳妥做法。

**内存警告**：高精度加载会占用约等于一份完整FP16/BF16模型的内存/显存。显存不够时，把 `device_map` 设为 `{"": "cpu"

## 块 62

**合并时开启校验并做一致性验证（建议）**

`merge_and_unload` 支持 `safe_merge=True`，合并时会做数值检查（比如检测是否产生 NaN），相当于一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，**对比合并前后模型在同一批输入上的输出**，确认没引入明显偏差。流程：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits / 生成文本是否基本一致

最小验证代码：用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本是否一致。

**本块学习检查**：跑通

## 块 63

## 合并后别急着用——先做一次“前后对照”验证

你已经完成了 QLoRA 模型的合并（§11.2），现在 `merged_model` 就是合并后的底层模型。但这里有个**关键陷阱**：`merge_and_unload()` 返回一个新的合并后的模型对象，原 `PeftModel` 对象仍保留未合并状态，可用于对比验证。

所以，**不能拿 §11.2 里被合并过的对象当“合并前”的对比基准**。正确做法是：**重新加载一份全新的、未合并的 PeftModel**，专门用来做对比。

### 第一步：重新加载一份“未合并”的基准模型

```python
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

注意：这里用的是 `base_model_path` 和 `adapter_path`，和 §11.2 里的变量一致。`model_before` 就是你的“合并前”参照物。

### 第二步：准备一个固定输入

```python
prompt = "请用一句话介绍你自己。"        # 换成你任务里有代表性的固定输入
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
```

### 第三步：对比 logits（数值层面）

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果**：`max_abs_diff` 越接近 0，说明合并前后的模型输出越一致。QLoRA 场景下，因为量化精度的原因，通常是一个**较小的近似值**，而不是精确的 0——这属于正常现象。

### 第四步：对比生成文本（行为层面）

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

**关键点**：`do_sample=False` 是为了**去掉随机性**，让生成结果可复现、可对比。如果合并正确，两次生成文本应该**完全一致**。

### 为什么这一步很重要

审核意见（本块）：
- 位置[部署说明] 问题：部署文档引用模型但未标注许可证类型。；建议：补充许可证说明，例如：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
- 位置[部署说明] 问题：部署文档涉及模型加载但未标注显存要求。；建议：补充注释，例如：# 注意：Llama-2-7b FP16 需要至少14GB显存。
- 位置[] 问题：错误声称 merge_and_unload() 是原地操作，会直接修改并返回底层

## 块 64

## 十一、合并时开启校验并做一致性验证（建议）

**为什么不能直接拿合并后的模型当“合并前”的对照？**

`merge_and_unload()` 会**原地**修改模型——它把 LoRA Adapter 合进底层权重，同时移除 PEFT 包装。如果你拿合并后的模型去和“合并前”对比，两边其实是同一份已合并的权重，`max_abs_diff` 永远是 0，验证就完全失效了。

**正确做法**：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照基准，再去和合并后的模型比较。

**显存不够怎么办？**

如果无法同时放下两个模型，就**分两次跑**：
1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本存下来，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比结果。

**结果预期（重点）**

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff` 一般是个较小的非零值，生成文本也可能有个别 token 不同。重点不是逐位相同，而是确认**没有明显退化或 NaN**。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。
- 是否可接受，以你的任务指标和验证集结果为准。

---

## 十二、电脑配置批注

本章节需要**实际加载模型**，涉及：加载完整模型 → 合并权重 → 导出模型。

**模型许可证**：以 Llama 2 为例，适用 Llama 2 Community License，商用月活超 7 亿需 Meta 授权。

**最低配置**：
- CPU：4核+
- 内存

## 块 65

## 十一、QLoRA模型合并注意事项（重点）——合并时开启校验并做一致性验证（建议）

### 为什么需要校验？

QLoRA合并时需注意：**不应在4-bit量化状态下直接合并**，而应重新加载高精度基础模型进行合并。在量化状态下直接合并会导致类型不匹配、精度损失甚至直接报错。解决办法不是手动逐层还原权重，而是**重新加载一份不带量化配置的高精度基础模型**。

### 具体做法（三步走）

1. **重载高精度基础模型**：用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`，注意**不传** `quantization_config` 或 `load_in_4bit`。
2. **挂载同一份Adapter**：用 `PeftModel.from_pretrained` 加载你训练好的LoRA适配器。
3. **合并并开启校验**：调用 `merge_and_unload` 时设置 `safe_merge=True`，合并后做一致性验证（对比合并前后模型输出）。

这样合并发生在高精度权重上，避开4-bit状态下的类型/精度问题。

### 版本兼容性提醒

不同版本的 PEFT / bitsandbytes / Transformers 对"量化模型能否直接调用 `merge_and_unload`"支持不同——部分版本直接报错，部分较新版本有一定支持但仍可能有精度损失。**"先反量化（重载高精度）再合并"是跨版本较稳妥的做法**。是否可以在量化状态下直接合并，请以你所装版本的文档和实际运行结果为准。

### 模型许可证

本模型基于 Llama 2 Community License 发布（商用月活>7亿需Meta授权）。

### 常见问题速查

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（忘了挂载Adapter，或已合并成普通模型） | 确认加载方式，先得到 PeftModel 再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积Adapter | 无需处理；若在意体积可保留Adapter不合并 |
| 显存不足 | 高精度完整模型占用高；或用了FP32、未offload/分片 | 改用CPU合并或offload、用FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存tokenizer；或特殊token/generation_config未导出 | 重新 `save_pretrained` 并核查tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 4-bit状态直接合并、版本不支持、量化包装

## 块 66

## 14.1 路线A：合并后部署完整模型

先看最简单的路线。你训练完 LoRA 后，用 `merge_and_unload()` 把 LoRA 权重合并回基础模型，然后保存成一份**完整模型**。部署时，推理框架（比如 vLLM）直接加载这份合并后的模型，就像加载一个普通模型一样。

**这条路线的好处：**
- 部署链路简单——推理框架不需要知道 LoRA 是什么，按普通模型处理即可；
- 推理时没有 Adapter 切换成本，性能干净利落。

**代价：** 每个 Adapter 都要保存和分发一份完整模型。如果你有多个任务，每个任务对应一个 Adapter，那就要存多份完整模型，存储成本会很高。

---

## 14.2 路线B：保留 Adapter，用支持 LoRA 的框架动态服务

如果你有多个任务/多个租户，不想每份都存完整模型，可以走路线B。像 **vLLM** 这样的框架支持 Multi-LoRA serving：**基础模型只加载一份**，运行时挂载多个 LoRA Adapter（A/B/C），请求时按需指定用哪个 Adapter。

**这条路线的好处：**
- 只需一份基础模型 + 多个小体积 Adapter，显存和存储都省；
- 支持热切换，不用为每个任务单独存完整模型。

**代价：** 依赖框架对 LoRA 的支持和配置，链路比直接加载合并模型略复杂；不同框架/版本对 LoRA 的支持范围和限制不同。

---

### vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
    --max-loras 2 \
    --max-lora-rank 16
```

关键点：
- `--enable-lora`：开启多 LoRA 服务；
- `--lora-modules 名字=路径`：声明可用 Adapter。**注意：这里的路径是训练保存下来的 Adapter 目录**（含 `adapter_model.safetensors` / `adapter_config.json`），**不是**合并后的完整模型；
- `--max-loras`：限制同时驻留的 Adapter 数；
- `--max-lora-rank`：需 **≥ 训练时 LoRA 的 r**；
- 请求端把 `model` 字段设为对应 Adapter 名字来选择用哪个。

**限制与版本提示（重要）：**
- vLLM 的多 LoRA 是**运行时挂 Adapter**，走的是**路线B**——传入的是**未合并的 Adapter 目录**，不是合并后的完整模型（合并后的模型属于路线A，按普通模型直接加载）；
- 具体参数名、默认值、是否支持运行时动态增删 Adapter、是否

## 块 67

**14.3 怎么选？——先记住：合并不是唯一出路**

你正在学LoRA微调，记住一个关键认知：**合并不是唯一部署方式**。根据你的场景，参考下面两条经验（非硬性规则，以实际部署为准）：

- **只有一个Adapter，想部署最简单** → 选**合并**（路线A）。把Adapter融进基础模型，得到一个完整模型，直接部署，省事。
- **同一个基础模型要服务多个Adapter，且推理框架原生支持LoRA** → 选**保留Adapter动态服务**（路线B）。不合并，运行时动态加载不同Adapter，灵活切换。
- **QLoRA精度敏感** → 无论走哪条路线，都建议按§11.3做合并前后/上线前的一致性验证，确保精度没跑偏。

具体框架用法和限制，第15板块（vLLM部署）再展开。现在你只需建立这个认知：**合并不是唯一选择**。

---

**本块学习检查**（动手验证，不引入新知识）：

对照上面三条经验，写下你的判断：如果你训练了一个LoRA Adapter，想

## 测评题目

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。
