# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么跑的：

```text
输入数据
    ↓
整个模型参数更新
    ↓
保存完整模型
```

以 7B 参数模型为例（70亿参数），训练时你需要同时保存三样东西：

- **参数**：模型本身的权重；
- **梯度**：反向传播时算出的更新方向；
- **优化器状态**：如 Adam 的动量信息。

这三者叠加，显存需求会非常高。普通消费级 GPU 根本扛不住。

---

**为什么这对你很重要？**

你的目标是 LoRA 微调，而 LoRA 正是为了解决上面这个问题而生的——它不更新全部参数，只训练一小部分额外参数，从而大幅降低显存压力。先理解“全参数微调为什么贵”，你才能明白 LoRA 省在哪里。

---

**动手验证一下（不引入新知识）**

打开你的 GPU 监控（如 `nvidia-smi`），观察当前显存占用。然后回想：如果直接训练一个 7B 模型，参数、梯度、优化器状态三份数据同时驻留显存，你的卡还能装下吗？这个对比能帮你直观感受全参数微调的瓶颈。

## 块 2

**动手理解PEFT：只训练“小配件”**

你的目标是LoRA微调，核心就一句话：**不碰整个大模型，只训练新增的少量参数**。

想象原始大模型是台现成电脑，LoRA适配器是一个外接小硬盘。流程是：

1. 原始大模型 + LoRA适配器参数
2. 只训练LoRA参数（小硬盘）
3. 保存这个小规模Adapter（只存小硬盘，不存整台电脑）

**为什么省显存？** 对比一下：

- 全参数微调：训练全部参数，显存需求**非常高**
- LoRA：只训练少量参数，显存需求**较低**
- QLoRA：量化模型+LoRA，

## 块 3

**LoRA是什么？——用“小补丁”微调大模型**

想象原模型权重是`W`，它**不更新**。我们新增一个变化量`ΔW`，最终效果是`W + ΔW`。

关键来了：`ΔW = (α/r) × B × A`。这里`A`、`B`是两个**低秩小矩阵**——若`W`形状为`d_out × d_in`，则`A`为`d_in × r`、`B`为`r × d_out`，相乘后还原成`W`的形状，`α`为缩放系数。注意：不同资料可能写`A × B`，只是符号顺序不同，关键是形状能相乘并还原。

**只训练`A`和`B

## 块 4

## 3.1 LoRA 的问题：模型还是太大

你已经知道 LoRA 能大幅减少**需要训练**的参数。但有个关键点别忽略：**基础模型本身仍然要完整加载到显存里**。

举个例子，一个 7B 参数的模型，如果用 FP16（半精度）格式加载，模型权重本身大约要占 **14GB 显存**。注意，这 14GB 仅为模型权重占用，实际训练时还需额外显存用于优化器状态和激活值。也就是说，就算 LoRA 帮你省了训练时的参数开销，光是把模型“搬”进显存这一步，就已经很吃资源了。

## 3.2 QLoRA 的解决思路：先把模型“压缩”再训练

QLoRA 全称是 **Quantized LoRA**，也就是“量化版 LoRA”。它的核心思路很简单：**既然基础模型太大，那就先把它变小，再挂上 LoRA 训练。**

流程如下：

```text
FP16模型
   ↓
4bit量化（NF4格式，压缩）
   ↓
减少显存占用
   ↓
加载LoRA进行训练
```

训练时的结构是：

```text
4bit基础模型 + LoRA参数 → 一起训练
```

这里要明确：LoRA 训练时**冻结基础模型**，仅训练低秩矩阵 A 和 B 形成增量更新，而不是对原始权重 W 执行 SVD 分解。这样既保留了 LoRA 省训练参数的优势，又通过量化把基础模型的

## 块 5

## 电脑配置批注：跑 LoRA / QLoRA 需要什么？

你要做 LoRA 微调，先别急着装包，看看你的电脑够不够“扛得住”。下面按 LoRA 和 QLoRA 两档给你拆清楚。

### LoRA 最低配置（先过这一关）

- **CPU**：4 核以上
- **内存**：16GB 起步
- **磁盘**：至少留出 50GB 空间（模型和数据集都要占地方）
- **GPU**：推荐 NVIDIA 显卡
- **显存**：最低 8GB，推荐 12GB 以上

> 简单说：8GB 显存能跑，但很紧；12GB 以上才舒服。

### QLoRA 要求（显存更友好）

QLoRA 的核心思路是**用 4-bit 量化加载基础模型**，所以显存占用通常比 LoRA 全精度加载低不少。但注意，下面给的是**经验参考值**，不是硬性保证：

- **参考下限**：约 8~12GB（针对 7B 级别模型，短序列、小 batch）
- **推荐**：16GB 以上

**为什么不是固定值？** 实际显存占用受这些因素影响：

- **模型规模**：7B 是基准，13B/70B 显存需求会明显上升
- **序列长度**：短序列（≤512）占用低，越长激活值越占显存
- **batch 大小**：batch=1

## 块 6

## 5.1 Python版本检查

在开始LoRA微调之前，先确认你的Python环境。打开终端，运行：

```bash
python --version
```

**建议版本：Python 3.10+**

如果版本低于3.10，建议先升级Python，否则后续安装PEFT和bitsandbytes时可能遇到兼容性问题。

---

## 5.2 PyTorch检查（关键步骤）

LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的CUDA量化路径要求安装**与本机CUDA匹配的GPU版PyTorch**。

> ⚠️ **常见坑**：直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

**正确做法**：前往[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)，选择与你GPU匹配的CUDA版本。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证PyTorch是否能看到GPU**，运行以下Python代码：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)   # 查看torch编译使用的CUDA版本
```

**正常输出**：`torch.cuda.is_available()`返回`True`，表示PyTorch侧能使用GPU。

> 📌 **注意**：`True`只说明PyTorch能用GPU，**不足以证明bitsandbytes的4-bit量化路径可用**。后者还需bitsandbytes正确编译/匹配本机CUDA，具体检查在后续章节进行。

---

## 5.3 查看GPU信息

终端运行：

```bash
nvidia-smi
```

输出应包含：
- **GPU型号**（如RTX 4090）
- **CUDA版本**（如CUDA 12.x）
- **显存大小**（如Memory 24GB）

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是Hugging Face官方的**参数高效微调库**，支持：
- **LoRA**（你的学习目标）
- Prefix Tuning
- Prompt Tuning
- Adapter

### 6.2 安装PEFT

```bash
pip install peft
```

**验证安装**：

## 块 7

**bitsandbytes 是干嘛的？**

简单说，它是让大模型“瘦身”的工具，专门做两件事：

- **8bit/4bit量化**：把模型参数从高精度压缩到低精度，大幅减少显存占用。
- **QLoRA训练**：这是你学LoRA微调的关键——它让量化后的模型还能继续训练。

**为什么必须装？**

本教程采用 **bitsandbytes + 4-bit量化** 的方案来实现QLoRA，所以它是**前置依赖**。注意：QLoRA的核心思想（4-bit量化基础模型+LoRA）不强制绑定某个库，但本教程统一用它演示。如果你用其他量化库，请参考对应文档。

**安装命令：**

```bash
pip install bitsandbytes
```

---

**本块学习检查**：在终端执行上述

## 块 8

**动手前先查兼容性**  
你马上要跑LoRA微调，但bitsandbytes这个量化库依赖CUDA才能调用GPU。版本不匹配，轻则CUDA不可用，重则量化失败或加载模型直接报错。  

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。  
# 注意：Llama-2-7b FP16 需要至少14GB显存。  

**两步检查（跟着做）**  
1. 终端运行 `nvidia-smi`，看输出里的 `CUDA Version`。  
2. 再运行 `python -c "import bitsandbytes as bnb; print(bnb.cuda_setup())"`，

## 块 9

### 9.1 推荐使用WSL2

在Windows上跑LoRA微调，建议先装WSL2，再装Ubuntu，然后在里面配CUDA环境和bitsandbytes。这样做的原因是Linux环境兼容性更好，能少踩很多坑。

### 9.2 Windows原生可能问题

如果你坚持用Windows原生环境，可能会遇到一个问题：**bitsandbytes无法加载CUDA动态库**。

常见原因可能不止一种：
- 你装的`bitsandbytes`版本，和本机CUDA或GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本的支持不如Linux完善（不同版本差异较大）。

所以这不是单一的“Windows不支持”问题，而是受**PyTorch、bitsandbytes版本与本机CUDA/硬件**共同影响。推荐优先用WSL2获得更接近

## 块 10

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

#### 问题2：CUDA版本不匹配

**症状**：安装完成后，`import torch` 报错，或 QLoRA 4-bit 加载失败。

**排查步骤**：在终端执行以下命令，确认三项关键信息：

```bash
nvidia-smi
```

重点检查：
- **NVIDIA驱动**：是否已安装且正常运行；
- **CUDA版本**：驱动支持的CUDA版本号；
- **PyTorch版本**：是否与CUDA版本匹配。

---

## 十、安装完整LoRA环境

如果你已有基础环境（`transformers`、`datasets`、`accelerate`），直接跳到下一步。否则先安装：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

> **重要前置**：确保已按 5.2 安装**匹配本机 CUDA 的 GPU 版 PyTorch**（否则后续 `import torch` 或 QLoRA 4-bit 会失败）。CUDA 12.1 示例：
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

然后安装LoRA核心库：

```bash
pip install peft bitsandbytes
```

**一条命令装完所有依赖**：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

执行配置向导：

```bash
accelerate config
```

根据你的环境选择。单GPU示例：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

完成后会生成 `default_config.yaml` 配置文件。

---

## 十二、LoRA环境验证实验

创建文件 `test_lora_env.py`：

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

运行：

```bash
python test_lora_env.py
```

**成功标准**：输出显示 `CUDA: True`，且能打印出LoRA配置信息。这说明：
- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试

## 块 11

## 九、Windows用户特别注意事项：推荐使用WSL2

如果你在Windows上遇到报错，别急着怀疑自己——这通常是兼容性问题。最直接的解决办法是**使用WSL2**（Windows Subsystem for Linux 2）。

**为什么推荐WSL2？** 因为bitsandbytes等工具在Linux环境下运行更稳定，Windows原生环境容易出现兼容性报错。WSL2相当于在Windows里跑一个轻量Linux，能避开大部分坑。

**怎么判断自己是否需要？** 如果你在Windows上安装或运行bitsandbytes时遇到报错，优先检查CUDA版本是否匹配；如果确认CUDA没问题还报错，那就直接上WSL2。

---

## 常见问题速查表

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

## 你的学习路线图

**当前环境（已就绪）：**
Transformers → Datasets → Acceler

## 块 12

## 一、为什么需要LoRA？

想象一下，你手里有一个7B模型（70亿参数）。传统微调方式是这样的：

**预训练模型 → 更新全部参数 → 保存新模型**

这意味着你要同时保存三类东西：

1. **模型参数 W**：70亿个数值
2. **梯度 ∂W**：每个参数对应的梯度，又是70亿个
3. **优化器状态**（如Adam的动量项）：通常比模型参数还占空间

所以实际显存需求**远大于模型本身**。对7B模型来说，全参数微调动辄需要几百GB显存，普通设备根本跑不动。

---

## 二、LoRA的核心思想：不更新全部参数

LoRA的思路很直接：**微调时，冻结原始模型参数W不动，只训练一个很小的“补丁”**。

这个补丁用两个小矩阵的乘积来表示：

```
ΔW = B × A
```

其中：

- **ΔW**：对原始权重的更新量（补丁）
- **A**：一个很小的矩阵，负责把输入压缩到低维空间
- **B**：另一个很小的矩阵，负责把低维空间映射回原始维度

**关键点**：A和B的尺寸远小于W，所以训练参数量大幅减少。

---

## 三、rank（秩）为什么决定训练参数量？

假设原始权重W是 `d × d` 的矩阵

## 块 13

## 2.1 不直接修改原模型参数——先动手感受一下

**传统微调**：直接改原权重。原始权重是 `W`，训练后变成 `W'`，也就是直接修改 `W`。

**LoRA 的做法**：把 `W` 冻结（保持不动），另外新增两个低秩矩阵 `A` 和 `B`，用它们的乘积来形成增量更新 `ΔW`。最终模型用的是：

```
W' = W + ΔW = W + BA
```

- `W`：原模型参数（冻结，不训练）
- `A`、`B`：LoRA 新增的低秩矩阵（只训练这两个）

**动手验证**：你可以把 `W` 想象成一个已经训练好的模型权重，`ΔW` 是一个小

## 块 14

**本块学习检查**：看完后，请合上资料，在纸上画出“一个 4096×4096 的矩阵被拆成两个小矩阵”的示意图，并标出每个小矩阵的维度（提示：两个小矩阵相乘后要能还原出 4096×4096 的形状）。

---

### 3.1 矩阵基础概念

Transformer 里大量参数都是矩阵。比如注意力层中：

```text
Q = X · Wq
```

这里 `Wq` 就是一个权重矩阵。假设它的形状是 **4096 × 4096**，那它的参数数量就是：

```text
4096 × 4096 ≈ 1677万
```

### 3.2 直接训练矩阵的问题

如果我们要微调，就得修改这个矩阵的增量 `ΔW`。但 `ΔW` 同样需要

## 块 15

**本块学习检查**：动手算一遍——把 W 换成 1024×1024，r=4，LoRA 总参数是多少？对比完整矩阵，少了多少倍？

---

**4.1 矩阵尺寸变化**

原矩阵 W 是 4096×4096，直接训练要更新 **1677万** 个参数（4096×4096）。

LoRA 用 r=8 分解：A 是 8×4096（32768 参数），B 是 4096×8（32768 参数），总共 **65536**（约6.5万）。

对比：完整矩阵 1677万 vs LoRA 65536，参数量下降 **256倍**。这就是 LoRA 高效

## 块 16

### 5.1 神经网络参数变化具有低维特征

LoRA的核心思想源于一个关键观察：模型适应新任务时，参数变化 **ΔW** 并不需要覆盖所有方向。

**直觉理解**：想象一个4096维的空间，真正影响任务的“有效方向”可能远少于满维度，而是集中在一个**低维子空间**里——即远小于原维度的少数方向。

> **注意**：这是LoRA依据的**假设**——“权重更新往往具有较低的内在维度”。具体“有效维度”没有通用固定值，会随模型、任务、数据变化。这里只做**定性**理解，别记成“几十维”之类的确定数字。

**结论**：正因为变化集中在低维空间，我们可以用低秩矩阵 **BA** 来近似

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张高清图片，尺寸是 **4000×4000像素**。如果完整保存它，需要存储 **1600万个数据**。

但问题是——如果这张图片的主要变化其实很简单，只是：

- **亮度**调整；
- **对比度**调整；
- **少量结构变化**。

那么，你根本不需要保存1600万个像素值，只需要用**少量参数**就能描述这些变化。

**LoRA的核心思想正是如此：预训练模型权重W已固定，LoRA训练一个低秩增量ΔW，使得W+ΔW适应新任务。**

预训练模型本身已经很强大了，微调时我们不需要大改特改所有参数，只需要找到"变化的方向"，用低秩矩阵去捕捉它。具体来说，LoRA**冻结原权重W**，仅训练低秩矩阵A和B，形成增量 **ΔW=(α/r)BA**，最终模型权重为 **W+ΔW**，而非直接压缩W。

---

## 六、LoRA在Transformer中的位置

Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中 **Attention** 包含四个投影矩阵：

```text
Q（查询）、K（键）、V（值）、O（输出）
```

**LoRA早期实践中的常见做法**：只把LoRA加到 `Q投影` 和 `V投影` 上。

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示：只训练 `Wq` 的变化和 `Wv` 的变化。

> ⚠️ **重要提醒**：只挑 `q_proj`/`v_proj` 是LoRA原论文中的**常见选择**，**不是固定标准**。现代实践中，LoRA也经常被加到：
> - 注意力的全部投影（q/k/v/o）；
> - FFN的线性层；
> - 甚至用 `target_modules="all-linear"` 覆盖所有线性层。
>
> 选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

**一个覆盖更完整线性层的配置示例**（以Llama类模型常见模块名为例）：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0

## 块 18

## 8.1 rank（r）——用代码感受“低秩”的威力

你正在学 LoRA 微调，核心就是理解 `r` 这个参数。先看代码：

```python
r=8
```

**它代表什么？** 低秩矩阵的维度。简单说，`r` 决定了 LoRA 在微调时“用多大的空间”去学习新知识。

**调大 `r` 有什么影响？** 直接看利弊：

- **优点**：表达能力更强，模型能学得更复杂的东西。
- **缺点**：参数更多，显存占用增加，且 `r` 过大容易过拟合。

**经验起始范围**（注意：只是起点，不是铁律）：

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 16-32 |

**为什么不能直接查表定死？** 因为合适的 `r` 和模型规模、目标模块、数据量、任务难度都有关。上表只给你一个搜索起点，建议从 8 开始调参。

**

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，例如 `lora_alpha=16`。它和秩 `r` 一起，通过 `alpha/r` 这个比例来决定 LoRA 参数在微调时的更新强度。

**直观理解**：`alpha/r` 的值越大，LoRA 对模型权重的影响就越明显；值越小，影响越温和。你可以把它想象成一个“音量旋钮”——`r` 决定音质的细腻度，`alpha` 决定音量大小。

## 8.3 lora_dropout：防止过拟合

`lora_dropout=0.05` 表示在训练时随机丢弃 5% 的 LoRA 参数。它的作用是防止 LoRA 参数过拟合，让模型泛化能力更好。

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见MB级） |
| 部署 | 复制模型 | 加载Adapter |

**注意**：上表是**相对趋势**，不是绝对数值。LoRA 的实际表现会随 `target_modules` 选择、秩 `r`、模型层数与 `dtype` 变化——例如把 LoRA 加到很多模块并用较大 `r` 时，参数量和 Adapter 体积都会明显上升。

## 十、LoRA vs QLoRA

- **LoRA**：FP16模型 + LoRA参数
- **QLoRA**：4bit量化模型 + LoRA参数

| 项目 | LoRA | QLoRA |
|------|------|-------|
| 基础模型精度 | FP16/BF16 | 4bit |
| 显存

## 块 20

## 2.1 Rank表示什么？

Rank，中文叫**矩阵的秩**。但在LoRA里，你不用管数学上严格的定义，只需要记住一个更直观的说法：

> **Rank（r）决定了LoRA低秩矩阵的“中间维度”大小。**

我们直接看例子。假设原始权重矩阵是：

```text
W: 4096 × 4096
```

这个矩阵很大，直接训练它成本太高。LoRA的思路是：不直接动W，而是训练两个小矩阵A和B，让它们的乘积近似等于要更新的部分ΔW。

如果你设置：

```python
r=8
```

那么这两个小矩阵的形状就是：

```text
A: 8 × 4096
B: 4096 × 8
```

注意看，A的“行数”是8，B的“列数”也是8——这个8就是rank的值。它夹在两个4096中间，所以叫“中间维度”。

最后把A和B乘起来：

```text
ΔW = B × A
```

结果恢复成：

```text
4096 × 4096
```

也就是说，**不管r取多少，ΔW最终都能还原成和原始权重一样的形状**。r改变的只是中间那个“瓶颈”的宽度。

---

**一句话总结**：r越小，中间瓶颈越窄，需要训练的参数越少，速度越快，但能表达的信息也越有限；r越大，中间瓶颈越宽，参数更多，表达能力更强，但训练成本和显存占用也会上升。

---

**动手验证一下**：你可以打开Python，用PyTorch随机生成一个 `4096 × 4096` 的矩阵W，然后分别用 `r=4` 和 `r=16` 构造对应的A和B，计算 `B @ A` 的形状——你会发现两次结果都是 `4096 × 4096`，但A和B本身的参数量差了4倍。亲手跑一遍，这个“中间维度”的概念就彻底清楚了。

## 块 21

**Rank控制什么？**

Rank决定LoRA能学多少种变化方向。

**低rank（如r=4）**：变化模式少。优点是参数少、显存低、训练快；缺点是表达能力有限。

**高rank（如r=64）**：自由度更高。能学更复杂任务、适应力更强；但参数多、显存增加、训练成本高。

**动手验证**：在你的微调脚本里，把`r`从4改成64，对比训练时的显存占用和Loss下降速度，观察差异。

**本块学习检查**：用一句话回答——低rank和高rank各自的优缺点是什么？

## 块 22

**本块学习检查**：  
先别往下看，自己算一遍：如果输入和输出维度都是4096，LoRA的r=8时，参数量是多少？算完再对照下文。

---

**3.1 不用LoRA时**：  
完整训练要更新整个权重矩阵，参数量是：

`4096 × 4096 = 1677万`

---

**3.2 用LoRA时**：  
LoRA只训练两个小矩阵，参数量公式：

`参数量 = r × (输入维度 + 输出维度)`

- **r=8**：`8 × (4096+4096) = 6.5万`  
- **r=64**：`64 × 8192

## 块 23

**Rank太小：模型“学不动”了**

你正在用LoRA微调模型，如果把秩（Rank）设得太小，比如 `r=2`，会发生什么？

**核心问题：学习能力不足**

LoRA的本质是用低秩矩阵来近似权重更新。秩越小，能承载的信息就越少。当 `r=2` 时，这个“压缩通道”太窄了，模型没有足够的“空间”去学习和表达新任务需要的知识。

**直接后果：欠拟合**

模型无法充分捕捉新任务的特征，导致：

- **训练loss下降慢**：模型学得很吃力，损失值迟迟降不下来。
- **效果提升有限**：即使训练结束，模型在新任务上的表现也远达不到预期。

**一句话总结**：Rank太小，模型“脑子”不够用，新知识装不进去，最终欠拟合。

---

**本块学习检查**

请你在自己的LoRA训练脚本中，把 `r` 设为 `2`，跑一个极短的训练（比如几十步），观察训练loss的下降速度。然后对比你之前用较大 `r` 值（如 `8` 或 `16`）时的loss曲线，记录下你的观察结果。

## 块 24

**本块学习检查**：对照下表，为你的目标选一个起始 rank，并说明你打算用哪个验证指标来判断它是否合适。

**4.3 常见rank选择（经验起点，非自动标准）**

| 任务类型 | rank（起点参考） |
| ------ | ------ |
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**动手实践要点**：上表只是起点。实际调参需结合你的数据量、模型大小、序列长度、有效batch、目标模块数量、GPU显存，以及验证集表现来调整。同一个r在不同设置下，显存和效果差异可能很大，别直接照搬。

---

**五、Alpha（lora_alpha）是什么？**

## 块 25

## 6.1 控制LoRA影响强度

我们用一个具体例子来感受 `alpha` 如何控制 LoRA 的“劲儿”。

假设你设了 `r=8`、`alpha=16`，那么缩放系数就是：

```text
16/8 = 2
```

这意味着 LoRA 对模型的更新会被**放大 2 倍**，影响更强。

如果 `alpha=8`，缩放系数是：

```text
8/8 = 1
```

更新保持原始比例，不放大也不缩小。

如果 `alpha=4`，缩放系数变成：

```text
4/8 = 0.5
```

LoRA 的影响被**减弱**到一半。

---

## 七、Alpha与Rank的关系（重点）

总结一下：LoRA 加入模型的更新量会被 `α/r` 这个系数缩放，所以 `α/r` 是控制 LoRA 更新“强度”的**关键缩放系数**。

但注意：**不能说 LoRA 实际强度仅由 `α/r` 决定**。它只是一个缩放系数，实际落到权重上的更新量还受很多因素影响：

- **学习率**：直接决定每步更新步长。`α/r` 调大又不降学习率，容易一起放大更新导致震荡。实践中很多人通过调学习率来控制强度，而不是频繁改 `alpha`。
- **初始化**：常见实现里 `A` 用小随机值、`B` 初始化为 0，训练开始时 `ΔW = BA = 0`（不扰动原模型），之后再逐步学习。
- **训练状态**：`ΔW` 是训练

## 块 26

**动手验证：Rank与Alpha的“同步调”**

先看两组配置，感受一下规律：

- 配置1：`r=8`，`alpha=16` → 比例 = `16 / 8 = 2`
- 配置2：`r=16`，`alpha=32` → 比例 = `32 / 16 = 2`

发现了吗？虽然rank从8变成了16，但`alpha / r`的缩放比例始终是**2**，没变。

**核心结论**：调整rank时，常见做法是**同步调整alpha**，让比例大致稳定。但这只是**经验做法**，不是硬性规则。比如`alpha = 2 × r`（像上面的配置）只是常见的**

## 块 27

**方案1：小模型实验（动手首选）**

如果你是初学者且想快速跑通LoRA流程，直接复制这组配置：

```python
r=8
alpha=16
```

**为什么适合你？** 参数少、训练快，正好匹配你每天1小时的学习节奏。先用它完成一次完整微调，建立手感。

**核心逻辑（记住这两条）：**
- `r` 控制**LoRA容量**——值越小，可学习的参数越少。
- `alpha` 是**缩放系数**，用于调整低秩矩阵乘积的尺度，影响LoRA层的初始更新强度，但实际更新幅度还受学习率等因素影响。

**动手验证（基于原文）：** 用上面的配置跑一个最小训练脚本，观察训练速度。然后只把 `r` 改为16、`alpha` 改为32（

## 块 28

## 误区1：rank越大一定越好？错！

很多初学者有个直觉：rank越大，模型容量越大，效果肯定越好。但实际并非如此。

**为什么？** 如果任务本身比较简单，大rank会带来两个问题：
- **浪费显存**：更大的矩阵需要更多内存；
- **增加训练成本**：计算量上升，训练更慢。

**实操建议**：先从小rank（如8）开始，观察效果后再逐步增大。不要一上来就堆rank。

---

## 误区2：只调rank，不调alpha

这是最常见的配置错误。看这个例子：

```python
# 原来
r=8
alpha=16

# 改成
r=64
alpha=16
```

只把rank从8调到64，alpha保持16不变，那么缩放比例变成：

```text
16/64 = 0.25
```

**这意味着什么？** LoRA的更新强度被大幅削弱了。rank变大了（容量增加），但更新幅度反而变小了，最终效果可能还不如原来。

**正确做法**：调整rank时，alpha要跟着调整，保持 `α/r` 在一个合理范围（通常接近1或略小）。

---

## 误区3：alpha越大越好？错！

alpha过大（即 `α/r` 缩放过大）会导致：
- **更新过强**：每次参数更新幅度太大；
- **训练不稳定**：loss可能震荡甚至发散。

> **注意**：`α/r` 过大只是训练不稳定的**可能原因之一**。训练发散还常常来自学习率过高、数据质量差、batch size过小、精度问题（fp16/bf16）、缺少梯度裁剪等。排查时请综合检查，不要只盯着alpha。

---

## 本块学习检查

请回答：如果 `r=32`，想让 `α/r` 接近1，alpha应该设为多少？如果此时把alpha设为64，对LoRA更新强度有什么影响？

---

## 配置环境说明

本章节是**原理讲解**，不需要GPU。最低配置：
- CPU：任意现代CPU
-

## 块 29

好的，我们直接开始动手配置LoRA。既然你已经理解了LoRA的低秩分解原理，并且装好了PEFT环境，那我们就把注意力集中在最核心的配置对象——`LoraConfig`上。

你可以把`LoraConfig`想象成一份给PEFT库的“施工图纸”。它不直接参与训练，而是告诉PEFT库三件关键的事：**把LoRA模块插到模型的哪里**、**LoRA的规模（容量）要多大**、**训练时的强度和多变性如何控制**。

在PEFT（参数高效微调库）的LoRA训练流程中，它的位置非常清晰：

1.  加载基础模型
2.  **创建LoraConfig（也就是画这张图纸）**
3.  指定LoRA插入位置
4.  生成LoRA Adapter（根据图纸施工）
5.  训练LoRA参数

所以，这张“图纸”上需要明确四个核心参数，它们决定了微调的效果：

-   **`target_modules`**：决定“LoRA放在哪里”。你需要指定基础模型中的哪些模块（比如注意力层的`q_proj`、`v_proj`）要被LoRA改造。
-   **`r`**：决定“LoRA规模多大”。这是低秩分解的秩（rank），`r`值越大，LoRA引入的可训练参数越多，模型表达能力越强，但训练负担也越大。
-   **`lora_alpha`**：决定“LoRA训练强度如何”。它是一个缩放因子，控制着LoRA更新对原始模型权重的影响程度。
-   **`lora_dropout`**：决定“是否使用dropout”。在训练时随机“丢弃”一部分LoRA参数，防止模型过拟合。

**本块学习检查**

请打开你的Python环境，尝试导入`LoraConfig`，并创建一个包含`r=8`、`lora_alpha=32

## 块 30

## 一、先跑通一个最小配置

你是动手型学习者，咱们直接看代码。下面这段就是LoRA最核心的配置：

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

## 二、配置怎么用起来

配置本身不干活，它要配合`get_peft_model()`使用。流程是：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

代码长这样：

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

## 三、动手验证一下

**本块学习检查

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些模块里注入可训练参数”**。

Transformer 模型的 Attention 部分，通常由四个投影层组成：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA **不会默认修改所有层**，你需要手动指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思是：只给 **Query 投影层**和 **Value 投影层** 增加 LoRA。

---

## 四、为什么选 q_proj 和 v_proj？

先看 Attention 的计算公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

其中：

- **Q（Query）** 决定：模型**关注什么信息**
- **V（Value）** 决定：模型**传递什么信息**

所以，调整 Q 和 V，通常就能有效改变模型的行为——这就是为什么 `q_proj` 和 `v_proj` 是常见的首选目标。

> ⚠️ **重要提醒**：`q_proj`/`v_proj` 只是**常见起点**，不是所有模型通用的“标准配置”。不同模型的模块命名可能不同，**配置前务必先查看模型实际模块名**（用 `model.named_modules()`），再决定 `target_modules`。

---

## 偷懒技巧：target_modules="all-linear"

如果你不想手工枚举模块名，较新版本的 PEFT（一般 ≥ 0.7）支持一个便捷写法：

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
    target_modules="all-linear",   # 自动匹配所有 nn.Linear 层（包括输出层 lm_head）
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认 LoRA 已注入
```

**用它的代价**：覆盖的模块变多 → 可训练参数和显存占用都会增加。所以要在**参数量/显存**和**效果**之间权衡，最终以验证集结果为准。

如果版本太旧不支持这个字符串，就退回显式列模块名的写法，或者升级 PEFT。用 `print_trainable_parameters()` 可以快速验证注入是否符合预期。

---

## 本块学习检查

打开你的代码环境，加载任意一个 HuggingFace 模型，

## 块 32

你现在的目标是快速上手LoRA微调，而且喜欢动手实践。我们直接看最常用的起点：**方案1：只调整Q和V**。

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这个方案的特点很直接：**参数少、训练快**。它是常见的轻量级起点，适合先跑通流程，再根据效果调整。

**动手试一试**：先按方案1配置，跑一个小的训练循环，观察显存占用和训练速度。记住这个基线，后面调整方案时用来对比。

如果觉得效果不够，再考虑**方案2：调整全部Attention**，把 `k_proj` 和 `o_proj` 也加进去：

```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

代价是参数更多，但表达能力更强。

**动手试一试**：对比方案1和方案2在验证集上的损失

## 块 33

## LLaMA系列：target_modules怎么选？

对于LLaMA系列模型，最常见的做法是只对注意力层中的 **`q_proj`** 和 **`v_proj`** 这两个投影矩阵应用LoRA：

```python
target_modules=["q_proj", "v_proj"]
```

**为什么这样选？** 简单来说，`q_proj`（查询投影）和`v_proj`（值投影）是LoRA微调中经过验证的高性价比选择——在保持效果的同时，引入的额外参数量较少。你不需要纠结“为什么不是全部模块”，先记住这个默认配置，跑通流程后再实验其他组合。

---

## 对比：GPT-2的差异（帮你理解原理）

GPT-2的情况和LLaMA不同，它把Q/K/V三个投影**合并**在了一个叫 `c_attn` 的模块里。所以target_modules要写：

```python
target_modules=["c_attn"]
```

**关键坑点**：GPT-2用的是`Conv1D`层，它的权重形状是`(in_features, out_features)`，和PyTorch标准`nn.Linear`的`(out_features, in_features)`**正好转置**。因此在配置里必须显式设置：

```python
fan_in_fan_out=True
```

否则LoRA注入低秩增量矩阵ΔW=(α/r)BA（其中A和B为低秩矩阵，原权重被冻结）时会按错误的布局操作，导致模型出错。

> **版本提示**：PEFT库并不会自动处理GPT-2的Conv1D权重转置，遇到`Conv1D`时必须显式写上`fan_in_fan_out=True`。

---

## 动手验证（GPT-2示例）

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
    fan_in_fan_out=True,         # Conv1D 权重是 (in, out)
)

model = get_peft_model(model

## 块 34

**动手查一查：BERT该配哪些层？**

BERT的`target_modules`通常写成`["query", "value"]`，但别直接照搬LLaMA的`q_proj`/`v_proj`——不同模型命名不同，抄作业会报错。

**正确做法：先查再配**

用`named_modules()`打印模型结构，找到实际模块名：

```python
for name, _ in model.named_modules():
    print(name)
```

输出里会看到类似`bert.encoder.layer.0.attention.self.query`这样的名字（BERT系常见），再填进`target_modules`。

**匹配机制速记**
- 写`"query"`是**后缀匹配**：所有以`query`结尾的层都会命中。
- 想精确控制就写完整路径，或用正则。
- 注意：GPT-2的注意力模块叫`c_attn`（q/k/v合体），不是`query`，别套错。
- 版本不同命名可能变，`"all-linear"`这种快捷写法依赖新版本——一切以你实际打印结果为准。

**动手验证**
跑一下上面的打印代码，把你模型里BERT注意力层的实际模块名记下来，

## 块 35

## 6.1 r是什么？

r就是**rank**，它控制LoRA低秩矩阵的维度。

公式是：

```text
ΔW = BA
```

r决定的是A、B的中间维度。

**举个例子**：原矩阵是 `4096×4096`，设 `r=8`，那么：

- A：`8×4096`
- B：`4096×8`

## 6.2 r影响什么？

**r越大**：

- 优势：学习能力更强，能表达更复杂任务
- 缺点：参数增加，显存增加

**经验起始范围**（不是固定配方，以验证集/基准任务结果为准）：

| 场景 | r（起点参考） |
|------|--------------|
| 简单任务 | 约 4-8

## 块 36

### 7.1 alpha作用

LoRA最终的权重更新不是直接用 `BA`，而是乘以一个缩放系数：

```text
(alpha/r)BA
```

这里的 `alpha` 控制的就是 **LoRA更新的幅度**——alpha 越大，LoRA 对模型权重的影响越强。

**动手算一下：**
配置 `r=8`，`lora_alpha=16`，缩放系数就是 `16/8 = 2`。这意味着 LoRA 的更新被放大了 2 倍，影响增强。

---

### 7.2 alpha常见设置

一个常见的经验起点是：

```text
alpha = 2 × r
```

> 注意：这只是经验起点，不是通用规则。你也可以固定 alpha 只调 r，或用其他比例，最终以验证集表现定。

**试试看：

## 块 37

## 8.1 dropout在做什么？

训练时，LoRA会对低秩矩阵的权重或激活进行随机置零，比例由`lora_dropout`控制。比如：

```python
lora_dropout=0.05
```

意思是5%的LoRA权重或激活会被随机置零。这样做的目的是**减少过拟合**——防止模型死记训练数据，泛化到新数据时表现更好。

## 8.2 怎么选dropout？

经验起点（不是死规则，要结合验证集效果调整）：

| 数据规模 | dropout起点 |
|---------|------------|
| 数据少 | 0.05~0.1 |
| 数据多 | 0~0.05 |

如果训练数据

## 块 38

## 9.1 基础配置——动手搭一个LoRA

这段配置适合**7B模型**的快速测试，也是你上手LoRA微调的第一块拼图。直接看代码：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量：秩的大小
    lora_alpha=16,          # 更新缩放：控制微调强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：注意力层的q和v
    lora_dropout=0.05,      # 防过拟合
    bias="none",            # 不训练偏置
    task_type="CAUSAL_LM"   # 因果语言模型任务
)
```

### 每个参数在干嘛？

| 参数 | 值 | 作用 |
|---|---|---|
| r | 8 | LoRA容量，秩越大可学信息越多 |
| alpha | 16 | 更新缩放，alpha/r=2，微调强度适中 |
| target_modules | q/v | 只改注意力层的q_proj和v_proj |
| dropout | 0.05 | 随机丢弃5%防止过拟合 |
| task_type | CAUSAL_LM | 告诉PEFT这是语言模型任务 |

**动手验证**：把r改成4

## 块 39

## 9.1 基础配置：把 LoRA 装进 BERT 分类任务

你现在的目标是 **LoRA 微调**，而且喜欢动手实操。这一节我们直接看一个能跑通的完整案例——用 LoRA 微调 BERT 做二分类。代码不长，但每一行都有讲究，我们边看边拆。

### 第一步：加载基础模型

```python
# 模型许可证：BERT 使用 Apache 2.0 许可证，可商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练需根据 batch size 适当增加
base_name = "bert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里加载的是带分类头的 BERT。注意：**分类头是新初始化的**，它还没学过任何东西，所以后面必须让它参与训练，否则分类结果就是随机的。

### 第二步：配置 LoraConfig——核心参数逐个看

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

对照你的学习目标，重点盯住这几个：

- **`r=8`**：LoRA 的低秩维度。简单理解就是“用多小的矩阵去近似原来的大矩阵”，r 越小越省参数，但太小可能学不够。
- **`lora_alpha=16`**：缩放因子，控制 LoRA 更新的强度。它和 r 配合使用，实际缩放比例是 `alpha / r`，这里就是 2。
- **`lora_dropout=0.05`**：防止过拟合，训练时随机丢弃一部分 LoRA 层输出。
- **`target_modules=["query", "value"]`**：指定在 BERT 的哪些模块上挂 LoRA。这里选的是注意力机制里的 query 和 value 矩阵，这是 LoRA 最常用的挂载位置。
- **`modules_to_save=["classifier"]`**：**这一行是关键**。它告诉 PEFT：分类头也要参与训练，并且要跟着 LoRA 一起保存。如果不加这一行，分类头不会被训练，也不会被保存。

### 第三步：包装模型，查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

执行后你会看到：可训练参数 = **LoRA 参数 + classifier 参数**。这说明分类头确实被纳入训练了。

### 第四步：保存——不只是 Adapter

```python
model.save_pretrained("./lora_ckpt")
```

这里保存的是 **Adapter 权重 + `modules_to_save` 指定的分类头**，不是完整的基础模型。所以重载时必须先加载同一个基础模型，再把 Adapter 挂上去。

### 第五步：重载并验证分类头

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

这一步是在**验证**：重载后的分类头权重和保存前是否一致。如果打印结果是 `True`，说明分类头被正确保存和恢复了。

## 块 40

## 10.1 普通文本生成：LoRA 参数怎么配？

你正在学 LoRA 微调，目标是把大模型适配到具体任务上。对**普通文本生成**（比如让模型续写故事、生成对话），PEFT 库的 `LoraConfig` 推荐这样配置：

```python
r=8
alpha=16
dropout=0.05
```

**这三个参数怎么理解？**

- **`r`（秩）**：决定 LoRA 引入的低秩矩阵的维度。`r=8` 意味着每个被改造的层只增加 8 维的可训练参数，相比原模型动辄上千维的权重，参数量大幅减少。这是 LoRA 高效的关键。
- **`lora_alpha`（缩放因子）**：控制 LoRA 更新的强度。`alpha=16` 配合 `r=8`，缩放比例是 `alpha/r = 2`。这个比例影响微调时的更新幅度，比例越大，更新越激进。
- **`lora_dropout`（丢弃率）**：在 LoRA 层前随机丢弃一部分神经元，防止过拟合。`0.05` 是 5% 的丢弃率，属于比较保守的默认值。

**LoRA 的更新机制**：微调时冻结原始权重，仅训练低秩矩阵 A 和 B，通过 A×B 形成对权重的增量更新，而非对原始权重做分解。

**为什么普通文本生成用这套配置？**

普通文本生成任务（如续写、闲聊）对模型原有能力的依赖较强，不需要大幅改变模型行为。`r=8` 保持低参数量，`alpha=16` 提供温和的更新幅度，`dropout=0.05` 防止在小规模训练数据上过拟合。这是一套**稳妥的起步配置**。

**动手验证一下：**

在 Python 环境中运行以下代码，观察配置是否生效：

```python
from peft import LoraConfig

## 块 41

## 10.1 普通文本生成：LoRA核心参数速览

你正从零走向LoRA微调实战，这一节先看懂最常用的配置模板。

```python
LoraConfig(
    r=8,              # LoRA容量：秩越大，可学习参数越多
    lora_alpha=16,    # 缩放系数：实际缩放为 alpha/r = 2，用于缩放低秩增量ΔW
    target_modules=["q_proj", "v_proj"],  # 改哪里：只改注意力层的Q和V投影
    lora_dropout=0.05 # 随机丢弃5%的LoRA参数，缓解过拟合
)
```

**四个参数，一句话记忆：**
- `target

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经掌握了LoRA的核心思路：冻结原模型，只训练一小部分低秩矩阵。但这里有个关键点——**LoRA省的是“训练参数”，不是“显存占用”**。

看这个结构：

```text
原模型参数：冻结（但仍在显存里）
LoRA参数：训练（只占很小一部分）
```

问题就在第一行：**基础模型仍然需要完整加载到显存中**。

我们来算一笔账。假设你用的是7B模型，以FP16精度加载：

```text
7B × 2 bytes ≈ 14GB显存
```

这14GB只是“把模型放进去”的成本。实际训练时，显存里还要同时容纳：

- **激活值**：前向传播时每层的中间输出
- **梯度**：反向传播时计算出的更新方向
- **优化器状态**：如Adam的动量项和方差项

所以，**实际显存需求远高于14GB**。这就是为什么即便有了LoRA，普通消费级显卡依然跑不动大模型微调——瓶颈不在“训练多少参数”，而在“加载多少参数

## 块 43

**QLoRA核心思想：用4-bit省显存，用LoRA学新任务**

你已了解LoRA（冻结原模型，只训练小适配器）。QLoRA就是它的“省显存版”：

> **先把基础模型压缩成4-bit，再插入LoRA进行训练。**

流程如下：

```
FP16基础模型 → 4-bit量化 → 加载量化模型 → 插入LoRA → 训练LoRA参数
```

**和LoRA的区别一目了然：**

- LoRA：`FP16模型 + LoRA`
- QLoRA：`4-bit模型 + LoRA`

**关键点（务必记住）：**
- 量化的是**基础模型权重**，不是LoRA参数
- 训练时：4

## 块 44

**3.1 模型存储方式：为什么4-bit能省显存？**

模型参数按精度不同，每个参数占用空间也不同：

| 类型 | 每个参数占用 |
|------|------------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

**动手算一下：** 以7B（70亿参数）模型为例，用FP16存储需要 `7B × 2 ≈ 14GB`；换成4-bit只需 `7B × 0.5 ≈ 3.5GB`，显存直接减少约75%。

> ⚠️ 注意：这个3.5GB只是**理论下界**——仅统计量化后的权重本身。实际占用会更高，因为还要算上量化常数/缩放因子、分组元数据（NF4按组量化，每组都要存额外参数）、未量化的模块（如norm、embedding、lm_head、LoRA参数），以及训练时的计算缓存和激活值。所以把它当作“最低可能值”即可，真实占用以实际运行为准。

**3.2 为什么不用普通INT4？**

传统INT4整数范围是 `-8 ~ 7`

## 块 45

**4.1 NF4是什么？**

NF4（NormalFloat4，正态分布4位浮点）是QLoRA论文为神经网络权重设计的4-bit量化数据类型。

**关键理解**：别把它当成IEEE 754那种通用浮点格式。NF4本质是一套**离散码本**——预先定好16个（2⁴）取值点，按标准正态分布分位数排布，让0附近（权重密集区）划分更细。

**工作流程**：
1. 量化：每组权重归一化后，映射到最接近的码本取值，存4-bit索引
2. 反量化：按码本和缩放因子还原

**动手验证**：想象权重集中在0附近，NF4在0附近有

## 块 46

### 4.2 为什么适合模型权重？

大模型训练完的权重，**近似正态分布**——大量数值挤在0附近，越往两边越少。

普通INT4量化，是把范围**平均切分**（比如-8到7均匀分格），结果中间区域分得太粗，**表达不了0附近的密集细节**。

NF4不一样：它**按正态分布来设计量化档位**，0附近分得更细。效果就是：

- 0附近精度更高；
- 更贴合真实权重分布；
- 量化带来的精度损失更低。

**一句话记住**：NF4是“为权重分布量身定制”的4-bit量化。

---

### 4.3 NF4量化过程

原始权重是FP16（比如 `0.023`

## 块 47

**5.1 为什么需要双重量化？**

4-bit量化时，每组权重除了存4-bit数值，还得额外保存**scale**和**zero point**这两个量化参数。这些参数同样占显存。

**5.2 双重量化思想**

普通量化：FP16权重 → 4-bit权重 + scale参数。

双重量化：**再对scale参数做一次量化**。第一次量化模型权重，第二次量化量化参数本身，从而减少scale的存储开销。

**5.3 双重量化优势**

减少额外量化参数和显存占用。大模型参数多，即使少量优化也能节省可观显存。

---

**本块学习检查**：请用一句话说明双重量化“第二次量化”的对象是什么，并指出它节省了什么资源。

## 块 48

## 6.1 为什么需要分页优化器？

训练大模型时，显存不只被模型本身占用，还有两个“隐形大户”：

- **梯度**：反向传播时自动产生；
- **优化器状态**：比如 Adam 需要额外保存**一阶动量**和**二阶动量**。

QLoRA 已经把模型显存压得很低了，但训练过程中，**显存仍可能突然不足**——因为梯度、优化器状态这些临时开销，会在某些时刻集中爆发。

## 6.2 分页优化器思想

Paged Optimizer 的思路，借鉴了操作系统的**虚拟内存**。

核心机制：利用 CPU 内存作为交换空间，通过分页机制在 GPU 和 CPU 之间传输优化器状态。当显存出现峰值、即将 OOM（显存溢出）时，系统自动把**优化器状态**按“页”换出到 CPU 内存；等需要

## 块 49

## 6.3 类比理解：把显存想象成你的桌面

**普通优化器（非分页）**：所有参数状态都放在“桌面”（GPU显存）上。桌面一旦堆满，训练就报错（OOM），无法继续。

**分页优化器**：就像你处理文件一样——常用文件放桌面（显存），暂时不用的存进柜子（CPU内存），需要时再取出来。这样桌面永远不会被占满，训练就能继续跑下去。

---

## 6.4 实际启用：只需改一个参数

分页优化器不用你手写换页逻辑，`bitsandbytes` 库已经帮你实现了。你只需要在 `TrainingArguments` 里设置 `optim` 参数：

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
- `paged_adamw_8bit`：优化器状态用 8-bit 存储，更省显存
- `paged_adamw_32bit`：精度更高，但占用更多显存

**注意**：不同版本的 Transformers / bitsandbytes 支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报“未知 optim”错误，查一下你当前版本的文档，换用支持的名称即可。

**使用技巧**：分页优化器主要解决**峰值 OOM**，通常搭配 `gradient_checkpointing=True` + 小 `per_device_train_batch_size` + 梯度累积一起用，效果最好。

---

**本块学习检查**：打开你的训练脚本，找到 `TrainingArguments`，把 `optim` 改成 `"paged_adamw_8bit"`，运行一次训练（哪怕只

## 块 50

## 7.1 4-bit量化：给模型“瘦身”

**核心作用**：把模型权重从 FP16（16位）压缩到 NF4（4位），大幅减小模型体积。

**怎么理解**：想象一张高清照片（FP16）压缩成小尺寸缩略图（NF4）。文件变小了，但关键内容还在。QLoRA 就是用这种“缩略图”来加载基础模型，从而在显存有限的 GPU 上跑得动大模型。

---

## 7.2 双重量化：连“压缩参数”也压缩

**核心作用**：进一步减少量化过程中产生的额外参数占用。

**怎么理解**：4-bit 量化需要一个“缩放因子”（scale）来还原数值。双重量化就是把这个 scale 本身也做一次量化，省下这部分内存。相当于缩略图旁边的说明文字也用了更小的字体。

---

## 7.3 分页优化器：显存不够，内存来凑

**核心作用**：降低训练时的峰值显存。

**怎么理解**：训练时显存不够用？分页优化器允许 GPU 和 CPU 内存之间

## 块 51

## load_in_4bit：让模型“瘦身”加载

开启 `load_in_4bit=True`，模型权重就会以 **4-bit 精度加载**到显存中。相比默认的 16-bit，显存占用降至约 1/4，这是 QLoRA 能在消费级显卡上微调大模型的关键第一步。

## bnb_4bit_quant_type：量化方式选 NF4

设置 `"nf4"`，即使用 **NF4（4-bit NormalFloat）量化**。这是专为神经网络权重分布设计的量化格式，在 4-bit 精度下能保留更好的信息。

## bnb_4bit_use_double_quant：双重量化再省一笔

开启 `bnb_4bit_use_double_quant=True`，会对量化过程中产生的**量化常数再做一次量化**，进一步节省显存。建议开启，几乎无损。

## bnb_4bit_compute_dtype：计算精度与存储精度分离

这是最容易混淆的点，务必区分：

- **存储 dtype**：权重以 4-bit（NF4）格式存显存，省空间。
- **计算 dtype**：实际做矩阵运算时，4-bit 权重会被**反量化**成更高精度来算，这个精度由 `bnb_4bit_compute_dtype` 指定。

**不设置会怎样？** 默认通常是 `float32`，计算慢且显存开销大。所以 QLoRA 配置里**一定要显式指定**。

### 关键：别写死 bfloat16，按 GPU 能力选

`torch.bfloat16`（BF16）需要较新 GPU 支持（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系及更早）不支持，要用 `torch.float16`（FP16）。

**推荐做法——自动判断：**

```python
import torch
from transformers import BitsAndBytesConfig

# 支持 BF16 就用 bfloat16，否则退回 float16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

**BF16 vs FP16 怎么选？** 两者都是 16-bit，但数值分布不同：

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么用 4-bit 加载模型）、分页优化器（`optim` 参数）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们**拼成一个能跑的 QLoRA 训练骨架**——这是你从“看懂零件”到“组装整车”的关键一步。

### 整体流程一览（5 步）

1. **配置 4-bit 量化**：告诉模型“用 NF4 格式、双重量化、按 GPU 能力选计算精度”加载。
2. **加载模型**：用刚才的配置把基础模型以 4-bit 方式读进来。
3. **准备 k-bit 训练**：调用 `prepare_model_for_kbit_training`，处理归一化层/输出层的精度问题，并开启梯度检查点所需的输入梯度。
4. **注入 LoRA**：只训练低秩增量参数，冻结其余部分。
5. **设置训练参数并启动**：分页优化器 + 梯度检查点，跑起来。

### 逐步拆解代码

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

- `load_in_4bit=True`：启用 4-bit 加载。
- `bnb_4bit_quant_type="nf4"`：使用 NF4 量化格式。
- `bnb_4bit_use_double_quant=True`：开启双重量化，进一步省显存。
- `bnb_4bit_compute_dtype`：计算时用的精度——GPU 支持 bfloat16 就用它，否则退回 float16。

**第 2 步：以 4-bit 加载基础模型**

```python
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    raise
```

把 `bnb_config` 传给 `quantization_config`，模型就会按 4-bit 加载。`device_map="auto"` 让框架自动分配设备。用 `try-except` 捕获加载过程中的 IO/网络异常，便于定位问题。

**第 3 步：为 k-bit 训练做准备（关键步骤）**

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步**不能省**。它处理两件事：
- 让归一化层和输出层保持较高精度（否则 4-bit 下训练不稳定）。
- 开启梯度检查点所需的输入梯度（为后面省显存打基础）。

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

- `r=8`：低秩矩阵的秩，控制新增参数量。
- `lora_alpha=16`：缩放因子，影响 LoRA 更新的幅度。
- `target_modules`：指定注入 LoRA 的模块（按你的模型实际模块名调整，也可用 `"all-linear"`）。
- `print_trainable_parameters()`：打印可训练参数量，确认只有 LoRA 参数在训练。

**第 5 步：训练参数——分页优化器 + 梯度检查点**

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

- `optim="paged_adamw_8bit"`：分页优化器，显存不够时自动在 CPU 和 GPU 之间换页。
- `gradient_checkpointing=True`：与第 3 步的 `use_gradient_checkpointing=True` 呼应，共同省显存。

### 两个容易踩的坑

1. **第 3 步和第 5 步的梯度检查点必须同时开启**。只开一个会报错或达不到省显存效果。
2. **`optim` 参数名称可能随 bitsandbytes 版本变化**。如果报错，检查你安装版本的文档确认名称。

### 最小可运行骨架（完整代码）

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
from peft import LoraConfig, get_peft_model

# 模型许可证
```

## 块 53

## 九、把 QLoRA 的“五步”串起来（可跑通骨架）

你现在已经知道 4-bit 量化（NF4 + 双重量化）和分页优化器各自是干什么的了。但真正跑 QLoRA 时，**这五步必须按顺序配合好，缺一不可**。下面给你一个“能说明关系”的最小骨架，重点看每一步**为什么存在**。

### 第一步：`BitsAndBytesConfig` —— 决定模型怎么“瘦身”加载

这一步是**基础**。它告诉系统：基础模型要用 4-bit 方式加载，具体包括：

- 用 **NF4** 做 4-bit 量化（NF4 适合 LLM 权重分布）
- 开启**双重量化**（进一步节省量化常数占的额外空间）
- 设定**计算 dtype**（比如 BF16/FP16，用于实际运算时的精度）

> 一句话：**这一步决定模型以什么“形态”进入显存。**

### 第二步：`prepare_model_for_kbit_training()` —— 必做！别跳

这是 QLoRA 的**必要一步**。它会对量化后的模型做“训练适配”，比如：

- 把部分层转成计算精度
- 开启输入梯度（这是后面梯度检查点能用的前提）

> ⚠️ **常见坑**：跳过这一步，**常导致训练不收敛或直接报错**。别省。

### 第三步：`get_peft_model(..., LoraConfig)` —— 注入可训练的 LoRA

4-bit 主干是**冻结**的（不更新权重）。这一步把**可训练的 LoRA Adapter** 注入进去，让模型只更新 LoRA 那部分参数。LoRA 不修改原始权重，而是冻结原权重，仅训练低秩矩阵 A 和 B，形成对权重的增量更新。

> 一句话：**主干冻结，LoRA 负责学。**

### 第四步：梯度检查点要“两边对应”

在 `TrainingArguments` 里设 `gradient_checkpointing=True`，**必须**和第二步中 `use_gradient_checkpointing=True` **对应开启**。两边不一致会出问题。

### 第五步：`optim="paged_adamw_8bit"` —— 分页优化器

这一步启用**分页优化器**，目的是**缓解峰值显存**——当显存不够时，优化器状态可以临时“分页”到 CPU 内存，降低显存峰值。

---

### 骨架总结

```
BitsAndBytesConfig（4-bit加载）
    ↓
prepare_model_for_kbit_training()（必要适配）
    ↓
get_peft_model(..., LoraConfig)（注入LoRA）
    ↓
gradient_checkpointing=True（两边对应）
    ↓
optim="paged_adamw_8bit

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者会误以为LoRA训练完，基础模型就被“改好了”。其实不是这样。

训练过程中，**基础模型是被冻结的**，完全不参与更新；真正被训练的是额外附加的 **LoRA Adapter**。最终的结构是：

```text
基础模型 + LoRA Adapter
```

以Llama-7B为例，训练完成后保存的通常只有两个文件：

```text
adapter_model.safetensors
adapter_config.json
```

而不是整个完整模型。也就是说，你拿到的是一个“补丁”，不是“成品”。

### 1.2 Adapter有什么特点？

**优点：**

- 文件小，占用空间少；
- 保存速度快；
- 同一个基础模型可以搭配多个不同的Adapter，实现多任务复用。

比如同一个基础模型，可以分别搭配：

```text
客服Adapter
医疗Adapter
法律Adapter
```

**缺点：**

部署推理时，必须让**基础模型和Adapter同时存在**，缺一不可。这就带来了额外负担——每次使用都要加载两部分，流程更繁琐。

---

## 二、为什么需要合并LoRA权重？

既然Adapter有“必须搭配基础模型”的缺点，那有没有办法把两者合成一个完整模型？这就是**合并（merge）**要解决的问题。

合并的本质，就是把Adapter中训练得到的增量权重，**写回基础模型**，让基础模型本身就“记住”微调后的能力。合并完成后，你就不再需要单独加载Adapter了，直接使用合并后的完整模型即可。

这样做的直接好处是：**部署更简单、推理更高效**，不再依赖额外的Adapter文件。

---

**本块学习检查**

请用自己的话回答：LoRA训练后保存的文件有哪些？为什么说基础模型本身没有被直接修改？如果你要把同一个基础模型用于客服和医疗两个场景，

## 块 55

### 推理时，为什么要合并LoRA权重？

想象一下，你训练完LoRA后，模型其实是“两部分”组成的：一个基础模型（Base Model）和一个额外的LoRA适配器（Adapter）。推理时，数据要依次穿过这两部分才能得到输出。

但部署时，你往往希望模型是“一个整体”——输入直接进，输出直接出，就像下面这样：

```
输入 → 完整模型 → 输出
```

所以，你需要把LoRA适配器“融”进基础模型里，这个过程就叫**合并权重**。

---

### 合并的数学原理

训练LoRA时，基础模型的原始权重W被冻结，仅训练两个低秩矩阵A和B。模型权重的变化可以写成：

**W' = W +

## 块 56

## 3.1 merge_and_unload() 是什么？

你训练好LoRA后，模型其实还是“套着壳”的——它由**基础模型**和**LoRA适配器**两部分组成。`merge_and_unload()` 就是把这个壳拆掉，把LoRA学到的权重直接合并进基础模型里，得到一个**完整的普通模型**。

**一句话理解：**

```text
基础模型 + LoRA适配器 → 完整模型
```

## 3.2 合并前后对比

**合并前：**

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后：**

```text
普通Transformer模型（已包含LoRA更新后的权重）
```

---

## 四、完整流程一览

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

---

## 五、加载LoRA Adapter（动手前准备）

假设你已有基础模型路径和Adapter路径：

```python
base_model_path = "model_path"
adapter_path = "adapter_path"
```

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：**基础模型 + LoRA适配器**。

---

## 六、执行合并（核心一步）

```python
merged_model = model.merge_and_unload()
```

这一步完成后，`merged_model` **不再是** `PeftModel`，而是**普通Transformer模型**。合并时，将LoRA训练得到的低秩增量ΔW = (α/r)BA加到原始权重W上，得到最终权重W' = W + (α/r)BA。

---

## 七、导出合并后的模型

保存模型和分词器：

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

> 注意：以上是**示意结构**，实际文件会随模型/分词器类型变化。例如可能有 `generation_config.json`、`special_tokens_map.json`，大模型可能还有分片权重（`model.safetensors.index.json` + 多个分片），有些分词器是 `vocab.json`/`merges.txt` 或 `tokenizer.model`。

---

**动手验证一下：** 合并前后分别打印 `type(model)` 和 `type(merged_model)`，观察类型变化，确认合并后不再是 `PeftModel`。

> 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
> 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 57

## 三、merge_and_unload 是什么？（重点）

**一句话理解**：`merge_and_unload` 就是把“基础模型 + LoRA 适配器”这两层合并成一个完整的普通模型，合并后可以直接用 Transformers 加载，无需再通过 PEFT 加载适配器，但 PEFT 库仍可用于其他任务或后续微调。

**为什么需要它？** 训练时 LoRA 是“附加”在基础模型上的小模块，推理时如果每次都要先加载基础模型再挂上适配器，既麻烦又容易出错。合并之后，你得到的是一个“干干净净”的完整模型，直接用 Transformers 就能加载。

---

## 合并后如何加载推理？

合并完成后，**不需要**再写 `peft` 相关的加载代码，只需要：

```python
from transformers import AutoModelForCausalLM

# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程就变成了：**用户输入 → 完整模型 → 生成结果**，一步到位。

---

## 合并 vs 不合并，怎么选？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 基础模型+适配器（小） | 基础模型（大），合并后文件大小与基础模型相同 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要 PEFT | 直接 Transformers |

**核心权衡**：合并后部署简单、加载方便，但如果你想在多个任务间动态切换不同的 LoRA 适配器，不合并反而更灵活。另外，如果你的推理框架本身支持 LoRA，也不一定非要合并。

---

## 导出后必查清单（避免“能加载但行为不对”）

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中新增/修改过的特殊 token（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若训练时扩过词表，还要确认 embedding/lm_head 尺寸与词

## 块 58

**本块学习检查**：读完下面两个场景后，请合上资料，用自己的话说一说：什么情况下“合并”更划算？什么情况下“不合并”反而更好？

---

### 什么时候该合并？两个典型场景

#### 场景1：上线部署——只要一个固定模型，越简单越好

想象你的模型要上生产环境，流程是：服务器 → 直接加载模型 → 提供API。

如果你只需要**一个固定的Adapter**，而且希望部署链路最简单——不额外装PEFT、不做运行时加载——那**合并**通常更方便。

> ⚠️ 但注意：如果你的推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你需要**热切换多个Adapter**，那**不

## 块 59

## 10.2 不建议合并场景

你正在学LoRA微调，记住一个核心原则：**合并不是终点，灵活才是王道**。

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了Adapter A、B、C——每个Adapter对应一个不同任务。如果合并了，它们就“焊死”在模型里了；保留Adapter，你可以随时切换、组合、单独调整。**保留Adapter更灵活。**

**场景2：持续训练**

如果你打算继续调整LoRA参数，那就先别合并。合并后再训练等于把“可插拔”的模块变成“一体机”，后续改动成本高。**保留Adapter。**

---

## 十一、QLoRA模型合并注意事项（重点）

QLoRA有个关键差异：基础模型是

## 块 60

## 11.1 直接合并可能遇到的情况（重点）

你正在做 LoRA 微调，最终要把训练好的 Adapter 合并回基础模型。如果直接在 4-bit 量化状态下执行 `merge_and_unload()`，可能遇到两类问题：

- **精度损失**：4-bit 权重需要先反量化再合并，这个过程与全程高精度计算并不完全等价，结果会有细微偏差。
- **类型或版本不兼容**：可能出现权重类型不匹配，或者你使用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

> **注意**：以上只是**可能**出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你的**版本**、量化配置，以及是否先做了反量化。**以你所用版本文档和实际运行结果为准。**

---

## 11.2 常见做法：先反量化到高精度再合并

既然直接合并有风险，稳妥的做法是：**不在 4-bit 状态下合并**，而是先用 FP16/BF16 高精度重新加载基础模型，再挂载同一份 Adapter 后合并。

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

### 可执行完整代码

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
model = model.to(compute_dtype)  # 确保 Adapter 与基础模型精度一致

# 3) 合并；safe_merge=True 会在合并时做数值检查（如检测 NaN）
merged_model = model.merge_and_unload(safe_merge=True)

# 4) 保存完整模型与 tokenizer（tokenizer 从基础模型或训练目录加载均可，保持与训练一致）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)
```

### 代码要点拆解

- **第 1 步是核心**：重新加载基础模型时**不传** `quantization_config`，用 `torch_dtype` 指定 FP16/BF16。`torch.cuda.is_bf16_supported()` 会自动判断你的 GPU 是否支持 BF16。
- **第 2 步补充**：挂载 Adapter 后，用 `model.to(compute_dtype)` 将模型统一到

> **模型许可证**：本部署文档引用的基础模型需遵循其原始许可证。例如，若使用 Llama 2 系列模型，适用 Llama 2 Community License（商用月活>7亿需Meta授权）。请根据实际使用的模型确认

## 块 61

## 直接合并的三大关键点

QLoRA训练后直接合并，先记住三个“坑”：

**1. 版本决定能不能直接合并**
不同PEFT版本对“量化模型能否直接合并”支持不同，有的要求先反量化。**最稳的做法**：加载基础模型时**不传** `quantization_config` 或 `load_in_4bit`，用高精度权重来合并。动手前先查你所用版本的 `merge_and_unload` 文档。

**2. 内存会翻倍**
高精度重新加载完整模型，显存占用约等于一份FP16/BF16完整模型。显存不够？两个办法：把 `device_map` 改成 `{"": "cpu"}` 在

## 块 62

**本块学习检查**：运行下面的最小验证代码，把 `max_abs_diff` 的数值和两次生成文本是否一致记录下来。

合并模型时，给 `merge_and_unload` 加上 `safe_merge=True` 参数，它会在合并过程中做数值检查（比如检测是否产生 NaN），相当于一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，**强烈建议**对比合并前后模型在同一批输入上的输出，确认合并没有引入明显偏差。流程是：准备一小批固定输入（prompt）→ 分别用「合并前的 PeftModel」和「合并后的模型」推理 → 对比 logits / 生成文本是否基本一致。

最小验证代码如下（用固定 prompt，先比）：

```python

## 块 63

## 合并后别急着用——先做一次“一致性体检”

你刚在 §11.2 完成了 `merge_and_unload()` 合并，模型看起来能用了。但先别急着部署——**合并本质上是一次数学换算**，QLoRA 场景下由于量化精度损失，合并前后的模型输出会有微小差异。本块教你用一段代码给合并结果做“体检”，确认它没被“合坏”。

### 为什么必须单独加载一份“未合并”模型？

`merge_and_unload()` 是**原地操作**：它直接修改你手里的 `PeftModel` 对象，合并完成后，这个对象就不再是“未合并”状态了。所以你不能拿 §11.2 里那个已经被合并过的对象当“合并前”的对比基准——那等于拿同一个东西比两次。

正确做法：**重新从磁盘加载一份全新的 Base 模型，再挂上 adapter**，得到一份干净的“未合并”模型 `model_before`。这份模型保持 `Base(高精度) + Adapter` 的原始结构，专门用来和合并后的 `merged_model` 做对比。

### 体检项目一：logits 数值对比

```python
inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)

with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

- `prompt` 换成你任务里有代表性的固定输入（比如分类任务的一句话、生成任务的开头）。
- 计算两个模型输出 logits 的**最大绝对差**。理想值是 0（完全一致），QLoRA 场景下通常是一个**较小的近似值**——因为量化本身有精度损失，完全一致反而不正常。
- 如果这个值异常大（比如超过 1），说明合并过程可能出了问题，需要检查。

### 体检项目二：生成文本对比

数值接近还不够，还要看**实际生成效果**。用贪心解码（`do_sample=False`）去掉随机性，保证两次生成完全可复现：

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

- `max_new_tokens=32` 控制生成长度，够看出差异即可。
- 对比两段文本是否**完全一致**。如果一致，说明合并对实际输出没有影响；如果不一致，观察差异是否在可接受范围内。

### 部署前检查清单

- **模型许可证**：部署前确认模型许可证类型。例如：# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
- **显存要求**：加载模型前确认显存是否充足。例如：# 注意：Llama-2-7b FP16 需要至少

## 块 64

## 合并校验：别让验证变成“自欺欺人”

**为什么必须重新加载一份模型？**

`merge_and_unload()` 默认返回一个合并后的**新模型**，原模型保持不变；若需原地合并，需设置相应参数。这意味着合并后，原来的 `model` 仍是**合并前**的状态。

如果你拿它当对照，比较的是**未合并的原始模型**与**已合并的新模型**，`max_abs_diff` 不会恒等于 0，验证有效。

**正确做法**：在合并前保存原始模型引用，或重新加载基础模型并加载 adapter 作为对照，再和合并后的模型比较。

---

**显存不够怎么办？**

如果两个模型放不下，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍，把 logits 或生成文本**存下来**，然后**释放显存**；
2. 再加载合并后的模型，跑同样的输入，对比结果。

---

**结果预期：别追求“一模一样”**

- **QLoRA 场景**：合并后模型与原始模型在数值上存在差异（涉及反量化与权重相加），但生成结果通常一致。重点确认：**没有明显退化、没有 NaN**。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。

是否可接受，以你的**任务指标和验证集**为准。

---

**硬件要求（本章节需要实际加载模型）**

- **最低配置**：CPU 4核

## 块 65

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

**为什么建议做校验？**

QLoRA合并最容易踩的坑，就是“合并成功了，但结果不对”。开启校验并做一致性验证，是为了确保合并后的模型输出和预期一致，避免部署后才发现问题。

---

**核心操作：先反量化，再合并**

部分PEFT版本在量化模型上直接合并可能存在精度损失或报错，建议先反量化再合并以确保稳定性。正确做法是：

1. **重新加载一份不带量化配置的基础模型**：
   ```python
   base_model = AutoModelForCausalLM.from_pretrained(
       base_model_path,
       torch_dtype=torch.bfloat16,  # 或 torch.float16
       device_map='auto',
       trust_remote_code=True  # 某些模型（如LLaMA）需要
   )
   ```
   关键点：**不传** `quantization_config` 或 `load_in_4bit`。若内存不足，可使用CPU offload或分片加载。

2. **挂载同一份Adapter并合并**：
   ```python
   model = PeftModel.from_pretrained(base_model, adapter_path)
   model = model.merge_and_unload(safe_merge=True)
   ```

这样合并发生在高精度权重上，天然避开“在4-bit状态下合并”的类型/精度问题。注意：反量化后合并的权重类型与训练时（4-bit）不一致，可能引入微小差异，需通过一致性验证确认。

> **版本提示**：PEFT库自0.6.0版本起已支持在量化模型上直接调用`merge_and_unload`（需设置`safe_merge=True`），但部分版本直接报错，部分较新版本有一定支持但可能有精度损失。**“先反量化再合并”是跨版本最稳妥的做法**。是否可以在量化状态下直接合并，请以你所装版本的文档（如PEFT官方文档）和实际运行结果为准。

---

**一致性验证怎么做？**

合并前后各跑一次相同输入，对比输出是否一致。如果差异明显，说明合并过程有问题，需要检查反量化步骤或版本兼容性。建议在合并后使用验证集评估模型性能，确保与训练时一致。

---

### 常见问题排查速查表

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload`不存在 | 当前对象不是PeftModel（可能忘了用`PeftModel.from_pretrained`挂载Adapter，或已合并

## 块 66

## 路线A：合并后部署完整模型

你已经完成了 LoRA 微调，现在面临一个选择：**怎么把训练好的模型用起来？**

路线A的思路很直接——**把 LoRA 合并回基础模型，得到一个完整的模型文件**，之后部署时完全不用再管 LoRA 这回事。

### 操作流程

```
merge_and_unload()   ← 把 LoRA 权重合并回基础模型

↓

保存完整模型          ← 得到一个完整的、独立的模型

↓

推理框架按普通模型加载  ← 比如 vLLM 直接加载 merged_model
```

### 这条路线的特点

**优点：**

- **部署链路简单**——推理框架把合并后的模型当作**普通模型**加载，它根本不需要知道 LoRA 是什么；
- **推理时零额外开销**——没有 Adapter 切换成本，请求来了直接推理。

**代价：**

- 每个 Adapter 都要保存/分发一份**完整模型**。如果你有多个任务（比如 A/B/C 三个任务各训了一个 LoRA），就要存三份完整模型，**存储成本很高**。

### 什么时候选路线A？

- 你只需要部署**一个**微调后的模型；
- 你的推理环境**不支持 LoRA**（比如某些框架只认普通模型格式）；
- 你希望部署和推理链路**最简单**，不想引入额外配置。

> **模型许可证：** 部署前请确认基础模型的许可证类型。例如，Llama 2 Community License 规定商用月活用户超过 7 亿时需获得 Meta 授权。

---

## 对比：路线B（保留 Adapter 动态服务）

路线B的思路是：**基础模型只存一份，多个 LoRA Adapter 按需挂载**。

```
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（

## 块 67

**14.3 怎么选？记住两条经验**

先别纠结，这里不是硬性规则，按你的实际场景对号入座：

- **只训练了一个Adapter，想部署得最简单** → 选路线A（合并权重），一步到位。
- **同一个基础模型要跑多个Adapter，且推理框架原生支持LoRA** → 选路线B（保留Adapter动态服务），灵活切换。
- **用QLoRA训练，对精度敏感** → 不管走哪条路，上线前都建议做一次合并前后的一致性验证（参考§11.3）。

现在你只需要建立这个认知：**合并不是唯一的部署方式**。具体框架用法留到第15板块（vLLM部署）再展开。

> # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
> # 注意：Llama-2-7b FP16 需要至少14GB显存。

---

**动手小任务（30秒）**

用一句话回答：你的

## 测评题目

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1204-003
分页优化器的主要目标是？

### Q-KB12-ST1205-003
合并 LoRA 权重后最直接的用途是？
