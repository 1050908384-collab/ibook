# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么做的：

```
输入数据 → 整个模型参数更新 → 保存完整模型
```

以 7B 参数模型为例，就是 **70亿个参数**。训练时你需要同时保存三样东西：

- **参数**：模型本身的权重
- **梯度**：用于更新参数的中间结果
- **优化器状态**：优化算法（如 Adam）维护的额外信息

这三者加在一起，显存需求**非常高**。普通消费级 GPU 根本扛不住——这就是为什么全参数微调对大模型来说不现实。

---

**你的动手验证**：打开终端，运行 `nvidia-smi` 查看你的 GPU 显存大小。如果显存低于 16GB，全参数微调 7B 模型基本跑不动——这正是接下来要学 LoRA 的原因。

## 块 2

**PEFT：只训练少量新参数**

全参数微调要动整个大模型，显存需求非常高。PEFT（参数高效微调）的核心思想是：**不修改整个大模型，只训练少量新增参数**。

流程很简单：

```
原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter
```

对比一下三种方式：

| 方式 | 训练参数 | 显存需求 |
|------|---------|---------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**本块学习检查**：用自己的话解释PEFT的核心思想，并

## 块 3

**LoRA是什么？——用“小补丁”微调大模型**

想象原模型权重是`W`，我们**不更新它**。而是新增一个变化量`ΔW`，最终效果是`W + ΔW`。

关键来了：`ΔW = B × A`。这里`A`和`B`是两个**低秩小矩阵**——若`W`形状是`d_out × d_in`，则`A`约为`r × d_in`，`B`约为`d_out × r`，相乘后还原成`W`的形状。不同资料可能写`A × B`，只是符号顺序不同，关键是形状能相乘并还原。

**只训练`A`和`B`两个小矩阵**，这就是LoRA（低秩适配）。

**优点**：参数量少、训练快、显存低、Adapter文件小（通常几十到几百MB）。训练完保存`adapter_model`和`adapter_config`即可。

**动手验证**：检查`A`和`B`的矩阵

## 块 4

## 三、QLoRA是什么？（①③）

### 3.1 LoRA的问题

你已经知道LoRA能大幅减少训练参数——这是它的核心优势。但这里有个关键点：**LoRA只减少了“训练”的参数，基础模型本身还是要完整加载进显存的**。

举个例子，一个7B参数的基础模型，如果用FP16精度加载，大约需要**14GB显存**。这意味着，即使你用LoRA只训练少量参数，你的GPU依然要扛住这14GB的基础模型占用。显存不够，训练就跑不起来。

### 3.2 QLoRA解决方案

QLoRA（Quantized LoRA，量化版LoRA）就是来解决这个问题的。

核心思路一句话：**先把基础模型从FP16压缩成4bit量化格式，再在上面加LoRA参数进行训练**。

流程如下：

```
FP16模型 → 4bit量化 → 显存占用大幅降低 → 加载LoRA训练
```

结构上就是：

```
4bit基础模型 + LoRA参数 → 训练
```

这样，基础模型占用的显存大大减少，你就能在同样的GPU上跑更大的模型，或者用更小的GPU完成LoRA微调。

### 3.3 QLoRA依赖bitsandbytes

要实现QLoRA，你需要安装以下组件：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以NVIDIA CUDA GPU为准）

其中，**`bitsandbytes`** 是核心依赖，它负责：

- 低精度量化
- 4bit加载
- 8bit加载

> **重要说明**：本教程的实训与验证统一以 **NVIDIA CUDA GPU** 为准，这是项目支持范围，也是当前最常见、资料最全的路径。但“QLoRA/4-bit量化只能用NVIDIA CUDA”并不是技术必然——bitsandbytes近年已在扩展

## 块 5

## 电脑配置批注：跑 LoRA / QLoRA 需要什么？

你要做 LoRA 微调，先看电脑够不够“扛得住”。下面是两类配置参考，按你的动手习惯，直接对照自己机器看。

### LoRA 最低配置（先看这个）
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐 NVIDIA GPU
- **显存**：最低 8GB，推荐 12GB+

### QLoRA 要求（显存更省，但仍有门槛）
- **GPU**：本教程按 NVIDIA CUDA GPU 给出命令（其他后端请以 bitsandbytes 官方说明为准）
- **显存参考**（7B 级别模型的经验值，不是硬保证）：
  - 参考下限：12GB 以上（8GB 显存仅适用于极小模型或极端配置）
  - 推荐：16GB+

### 别把数字当死规矩
同样一个模型，显存占用会随以下条件变化：
- **模型规模**：7B 级别；13B/70B 会显著上升
- **序列长度**：短序列（≤512）占用低，越长激活值越占显存
- **batch 大小**：1~2 是小 batch，越大越吃显存
- **梯度检查点**：开启可降低峰值占用（用时间换显存）
- **量化配置**：4-bit（NF4）、是否双重量化、`bnb_4bit_compute_dtype` 等
- **LoRA 参数**：目标模块数量 / rank 越多

## 块 6

## 5.1 先确认你的Python版本

在终端里运行下面这条命令，看看你的Python版本是多少：

```bash
python --version
```

**建议版本：Python 3.10 或更高。** 如果低于这个版本，建议先升级Python再继续后面的步骤，否则后续安装PEFT和bitsandbytes时可能会遇到兼容性问题。

---

## 5.2 检查PyTorch是否装对了（关键一步）

> ⚠️ **为什么这一步很重要？** LoRA/QLoRA 依赖 PyTorch，而 `bitsandbytes` 的量化功能要求安装**和你本机CUDA版本匹配的GPU版PyTorch**。如果你直接 `pip install torch`，通常装到的是CPU版，这样后面QLoRA根本跑不起来。

**正确做法**：去 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 选择对应你CUDA版本的安装命令。例如，如果你的CUDA是12.1，就用：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**安装后验证**，运行下面这段Python代码：

```python
import torch

try:
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)   # 查看 torch 编译使用的 CUDA 版本
except Exception as e:
    print(f"验证过程中出现异常：{e}")
    print("请检查PyTorch是否正确安装，或确认CUDA环境是否正常。")
```

**看到什么算正常？** 第二行输出 `True`，说明PyTorch能识别到你的GPU。

> ⚠️ 注意：`True` 只代表PyTorch这边能用GPU，**不代表bitsandbytes的4-bit量化也能用**。后者还要看bitsandbytes是否和你的CUDA正确匹配，这个到后面章节再验证。

---

## 5.3 查看你的GPU信息

在终端运行：

```bash
nvidia-smi
```

你会看到：

- **GPU型号**（比如 RTX 4090）
- **CUDA版本**（比如 CUDA 12.x）
- **显存大小**（比如 24GB）

记下这些信息，后面配置环境时要用。

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT（

## 块 7

**bitsandbytes 是干嘛的？**

简单说，它是让模型“瘦身”的工具：支持 **8bit** 和 **4bit 量化**，也是 **QLoRA 训练**的必备依赖。

**为什么你必须装？**

因为本教程的 QLoRA 方案，就是基于 `bitsandbytes` 做 **4-bit 量化**的。所以它是前置条件，**必须安装**。

> 注意：QLoRA 的核心思想（4-bit 量化 + LoRA）并不绑定某个库，但本教程统一用 `bitsandbytes` + CUDA 演示。若你用其他量化后端，请以对应文档为准。

**动手装：**

```bash
pip install bitsandbytes
```

---

**本块学习检查**

装完后，在 Python 里运行 `import bitsandbytes`，确认无

## 块 8

**为什么先查CUDA？**  
bitsandbytes依赖CUDA驱动GPU计算。版本不匹配，会出现：CUDA不可用、量化失败、加载模型报错。

**两步检查（动手做）**  
1. 终端运行 `nvidia-smi`，看输出的 `CUDA Version`。  
2. 再运行 `python -m bitsandbytes`，正常会显示CUDA版本和GPU信息。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）  
# 注意：Llama-2-7b FP16 需要至少14GB显存

**本块学习检查**

## 块 9

**本块学习检查**：读完本节后，请用一句话回答：为什么在Windows上做LoRA微调时，推荐优先使用WSL2而不是Windows原生环境？

---

**9.1 推荐使用WSL2**

在Windows上准备LoRA微调环境时，推荐的安装路径是：

```text
Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes
```

简单说：**先装WSL2，再在WSL2的Ubuntu里配置CUDA和bitsandbytes**。这样做的核心原因是——**Linux环境的兼容性更好**。

---

**9.2 Windows原生可能遇到的问题**

如果你坚持在Windows原生环境安装，可能会遇到：

**问题1：bitsandbytes无法加载CUDA动态库**

这个报错可能由多种原因引起，常见的有：

- 你安装的`bitsandbytes`版本，和本机的CUDA或GPU版

## 块 10

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

#### 问题2：CUDA版本不匹配

在Windows上跑LoRA，最常遇到的坑就是CUDA版本对不上。别慌，先跑一条命令看看底细：

```bash
nvidia-smi
```

这条命令会告诉你三件事：

- **NVIDIA驱动**是否正常安装
- **CUDA版本**（驱动支持的版本）
- 显卡型号

记住：**驱动支持的CUDA版本 ≠ PyTorch实际用的CUDA版本**。两者不匹配，后面`import torch`或跑QLoRA 4-bit就会报错。

---

## 十、安装完整LoRA环境

假设你已经有基础环境了：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**关键一步**：确保已按5.2节安装了**匹配本机CUDA的GPU版PyTorch**。CUDA 12.1的安装示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

然后增加LoRA核心库：

```bash
pip install peft bitsandbytes
```

一条命令装全：

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
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

会生成`default_config.yaml`配置文件。

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

看到输出说明：

- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes：

```python
import torch
import bitsandbytes

print(torch.cuda.is_available())
print(bitsandbytes.__version__)
```

如果输出`True`，说明GPU环境正常，可以继续QLoRA微调。

---

**本块学习检查**：运行`nvidia-smi`，记录你的CUDA版本。

## 块 11

## 九、Windows用户特别注意事项：推荐使用WSL2

你在Windows上遇到报错时，**首选方案是使用WSL2**，这是官方推荐的兼容性解决方案。

### 常见问题速查表

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

> **实操提示**：遇到Windows报错时，直接切换到WSL2环境，而不是花时间在Windows上排查兼容问题。

### 你的学习路线

**当前环境栈**（从底层到上层）：
```
Transformers → Datasets → Accelerate → PEFT → bitsandbytes → LoRA → QLoRA
```

**下一步实战流程**：
```
加载基础模型 → 准备训练数据 → 配置LoraConfig → Trainer + PEFT

## 块 12

## 一、为什么需要LoRA？——先动手感受一下问题

想象你手上有一个7B模型（比如LLaMA、Qwen、Baichuan、Mistral这类），也就是**70亿参数**。如果按传统方式微调，流程是这样的：

```
预训练模型 → 更新全部参数 → 保存新模型
```

这个流程会带来三个“大麻烦”：

1. **显存爆炸**：训练时不仅要存模型参数W，还要存梯度∂W和优化器状态（比如Adam的moment）。假设模型有100亿参数，实际显存需求会**远大于模型本身的大小**。
2. **训练时间漫长**：每更新一个参数都要计算和回传梯度，全量更新意味着海量计算。
3. **存储成本高**：每个任务微调完都要保存一份完整的新模型，7B模型一份就是70亿参数，多个任务就是多份拷贝。

**动手验证**：打开你的PyTorch环境，创建一个7B规模的全连接层（`nn.Linear(7000000000, 7000000000)`），观察它占用的显存——你会发现根本加载不进去。这就是全参数微调的现实瓶颈。

---

## 二、LoRA的核心思想：不更新全部，只更新“关键”

LoRA的洞察是：**大模型微调时，不需要更新全部参数**。它冻结原始权重W，仅训练两个小矩阵A和B，用它们的乘积来模拟权重更新量。

### ΔW = BA 到底代表什么？

- **W**：预训练模型的原始权重（冻结不动）
- **ΔW**：

## 块 13

### 2.1 不直接修改原模型参数

**传统微调是怎么做的？**

想象你有一个训练好的模型，它的参数叫 **W**。传统微调直接改这些参数：

```text
W → W'
```

也就是把原来的 **W** 直接变成新的 **W'**，模型内部被“动刀”了。

**LoRA 的做法完全不同**

LoRA 的思路是：**原模型参数 W 保持不动，冻结起来**。然后额外增加一个新参数 **ΔW**（读作“delta W”）。

最终模型用的是：

```text
W' = W + ΔW
```

- **W**：原模型参数（冻结，不训练）
- **ΔW**：LoRA 新增的参数（只训练这个）

这样，你不需要动庞大的原始模型，只需要训练一个小得多的 **ΔW**。

---

### 2.2 为什么只训练 ΔW 就够了？

大模型已经通过海量数据学会了：

- 语言规律
- 世界知识
- 推理能力

微调的本质只是**对已有能力做方向

## 块 14

**3.1 矩阵基础概念（动手版）**

Transformer里到处是矩阵。比如注意力层算 `Q = X · Wq`：输入 `X` 乘上权重矩阵 `Wq`，就得到查询向量 `Q`。这里 `Wq` 就是一个大矩阵。

假设 `Wq` 是 `4096 × 4096`，那它有多少参数？算一下：`4096 × 4096 ≈ 1677万`。记住这个数，后面有用。

**3.2 直接训练矩阵的问题**

微调时，如果直接改 `Wq`，那 `ΔW`（权重的变化量）也得是 `4096 × 4096` 个参数——又是1677万。训练这么多参数，又慢又费

## 块 15

**4.1 矩阵尺寸变化——用数字感受LoRA的“瘦身”**

假设我们要微调一个大模型，其中有个权重矩阵 **W**，尺寸是 **4096 × 4096**。

如果直接训练这个完整矩阵，需要更新的参数量是：

- 4096 × 4096 = **16,777,216**（约1677万）

现在用LoRA，设置 **rank r = 8**。我们不直接动W，而是把它拆成两个小矩阵：

- **A**：尺寸 **4096 × 8**，参数量 = 4096 × 8 = **32,768**
- **B**：尺寸 **8 × 4096**，参数量 = 8 × 4096

## 块 16

**5.1 神经网络参数变化具有低维特征**

LoRA的核心依据是一个假设：模型适应新任务时，参数变化 ΔW 并不需要覆盖所有方向。有效变化往往集中在**低维子空间**里。

**动手理解**：想象一个4096维的空间（相当于模型参数的全部可能变化方向）。真正影响任务的“有效方向”可能远少于满维度，只集中在少数几个方向上。

> ⚠️ 注意：这里说的是“低维”这个**定性**概念，不是固定数字（比如“几十维”）。具体有效维度会随模型、任务、数据变化，别记成确定值。

**关键结论**：因为 ΔW 本质上是低维的，所以可以用一个低秩矩阵 **BA** 来近似它——这就是LoRA省

## 块 17

## 5.2 类比理解：为什么LoRA用少量参数就能微调大模型？

想象一张**4000×4000像素**的高清图片，完整保存需要**1600万个数据**。但如果这张图片的主要变化只是**亮度**、**对比度**和**少量结构变化**，你根本不需要保存全部1600万像素——用少量参数就能描述这些变化。

**LoRA的原理与此类似：在原有权重旁增加一个低秩增量，用少量参数实现模型行为的调整。**

也就是说，微调时我们不是去更新模型全部的原始参数（那太庞大了），而是学习一组很小的低秩矩阵A和B，通过它们的乘积ΔW=(α/r)BA来更新原始权重。这样既省显存，又省时间。

---

## 六、LoRA加在Transformer的哪里？

先回顾Transformer的主要结构：

```
Input → Embedding → Attention → FFN → Output
```

其中**Attention**部分包含四个投影矩阵：**Q、K、V、O**。LoRA通常加在这些投影矩阵的权重矩阵上。

LoRA早期实践中**常见**的做法是只加到`q_proj`和`v_proj`上，例如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示只训练`Wq`和`Wv`的变化，其中`q_proj`、`v_proj`即对应Q、V投影矩阵的权重矩阵。

> ⚠️ **注意**：只挑`q_proj`/`v_proj`是LoRA原论文中的一种**常见选择**，**不是固定标准**。现代实践中也经常把LoRA加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用`target_modules="all-linear"`覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

**一个覆盖更完整线性层的配置示例**（以Llama类模型常见模块名为例；不同模型模块名可能不同，请先用`print(model)`或`model.named_modules()`确认实际名称）：

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
        "q_proj", "k_proj", "v_proj",

## 块 18

## 8.1 rank（r）——低秩矩阵的“宽度”旋钮

在LoRA里，`r` 代表**低秩矩阵的维度**。你可以把它想象成“在原始权重旁边加一条多宽的小路”——`r` 越大，这条小路越宽，能“绕行”的信息就越多。

代码里这样设置：

```python
r=8
```

### r 越大，意味着什么？

- **优点**：表达能力更强，模型能学到更复杂的调整。
- **缺点**：参数更多，训练时**显存占用增加**。

### 从哪开始选？——经验起点参考

注意，下表只是**搜索起点**，不是“标准答案”：

| 任务类型 | rank（起点参考） |
|---------|----------------|
| 简单任务 | 约 4-8 |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 32-64 |

### 比“背配方”更可靠的做法：基于验证集选 r

`r` 的合适值受模型规模、目标模块、数据量、任务难度共同影响。与其猜，不如跑一组**受控对比**：

1. **划出

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制LoRA参数对原模型的影响程度。代码中常写作：

```python
lora_alpha=16
```

实际起作用的是 **alpha/r** 这个比值。比如 `r=8`、`alpha=16` 时，缩放比例为 2，LoRA 的更新会被放大；`alpha=8` 时比例是 1，更新保持原样。**调大 alpha，LoRA 对模型行为的改变更明显；调小则更保守。**

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

训练时随机丢弃 5% 的 LoRA 参数，防止模型过度拟合训练数据。数值越大，随机丢弃越多，模型越"保守"。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对） |
|------|-----------|-------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 常见MB级 |
| 部署 | 复制模型 | 加载Adapter |

> ⚠️ 注意：这是**相对趋势**，不是绝对保证。如果把 LoRA 加到很多模块、用较大 `r`，参数量和文件体积也会明显上升。

---

## 十、LoRA vs QLoRA

## 块 20

**本块学习检查**：读完本块后，请合上资料，用一句话回答：`r=8`时，A和B两个矩阵的维度分别是什么？最终ΔW的维度是多少？

---

### 2.1 Rank表示什么？

Rank，中文叫**矩阵的秩**。但在LoRA里，你可以把它直观理解为：**低秩矩阵中间那个“连接维度”的大小**。

我们用一个具体例子来看。假设原始权重矩阵W是：

```text
W: 4096 × 4096
```

这个矩阵很大，直接微调它成本太高。LoRA的做法是，不直接改W，而是训练两个小矩阵A和B，让它们的乘积近似等于要加的改动量ΔW。

如果你设置：

```python
r=8
```

那么这两个小矩阵的维度就由r决定：

- 矩阵A的维度是：`8 × 4096`
- 矩阵B的维度是：`4096 × 8`

你看，r=8正好是A的“行数”，也是B的“列数”，它就是中间那个“连接”的尺寸。

最后，把B和A相乘：

```text
ΔW = B × A
```

得到的结果是：

```text
4096 × 4096
```

正好和原始W的维度一样，可以叠加到原模型上。

**一句话总结**：r决定了中间通道的宽窄。r越大，A和B能携带的信息越多，但训练的参数和计算量也越大；r越小，越省资源，但能表达的改动范围也越有限。

## 块 21

**Rank控制什么？——动手理解**

Rank（r）决定LoRA能学多少种“变化方向”。想象成模型可用的“旋钮”数量。

**低rank（如 r=4）**：旋钮少，模型只能学较少变化模式。好处：参数少、显存低、训练快。代价：表达能力有限，复杂任务可能学不好。

**高rank（如 r=64）**：旋钮多，LoRA自由度更高。好处：能学更复杂任务、适应力强。代价：参数多、显存涨、训练成本高。

**实操验证**：在代码中分别设 r=4 和 r=64，对比训练后模型在验证集上的损失变化。你会直观看到：低

## 块 22

**动手算一算：不用LoRA要多少参数？**

假设模型某层权重是 **4096×4096** 的矩阵，直接训练它：

- 参数量 = 4096 × 4096 = **16,777,216**，约 **1677万**。

这就是“全参数训练”的代价——光这一层就要存1600多万个数字。

**对比一下用了LoRA之后：**

LoRA的参数量公式：`r × (输入维度 + 输出维度)`

- **r=8**：8 × (4096+4096) = **65,536**（约6.5万）
- **r=64**：64 × 8192 = **524,288**（约

## 块 23

### 4.1 Rank太小的问题

把 `r` 设得太小，比如 `r=2`，会带来两个直接后果：

- **学习能力不足**：模型能“记住”的新知识太少，导致**欠拟合**。
- **新任务知识无法充分表达**：你想让模型学会的新本领，它根本“装不下”。

**怎么判断自己是不是踩了这个坑？** 看训练时的表现：

```text
训练loss下降慢
效果提升有限
```

如果 loss 降得慢、效果上不去，先别急着调学习率，回头检查一下 `r` 是不是设太小了。

---

### 4.2 Rank太大的问题

反过来，`r` 设太大（比如 `r=256`）也有麻烦：

- 参数增加 → 显存增加 → 训练速度下降。
- **LoRA 的核心优势被削弱**。

**这里有个常见误解要澄清**：不是说 `r=256` 就“接近完整矩阵”了。以 `4096 × 4096` 的权重为例，矩阵的秩最大为 4096，`r=256` 还远小于它。

**用数字看清楚**：LoRA 增量参数量 = `r × (d_in + d_out)`，完整矩阵 = `d_out × d_in`。代入 `d_in = d_out

## 块 24

**4.3 常见rank选择（起点参考）**

| 任务类型 | rank（起点） |
|------|--------|
| 简单分类 | 约4~8 |
| 文本风格调整 | 约8~16 |
| 指令微调 | 约16~64 |
| 复杂领域任务 | 约64以上 |

**注意**：这只是经验起始范围，不是自动调参标准。实际项目需根据数据量、模型大小、序列长度、batch大小（含梯度累积）、目标模块选择、GPU显存及验证集表现调整。同一个`r`在不同模型规模、序列长度/batch、目标模块下的显存与效果差异很大，别直接照搬。

---

**五、Alpha（lora_alpha）是什么？**

LoRA不仅做`ΔW = BA`，还会缩放：

`W' = W + (α/r)BA`

- α：lora_alpha

## 块 25

### 6.1 控制LoRA影响强度（动手版）

我们直接看代码。假设你设了 `r=8`，同时设 `alpha=16`，那么缩放系数就是：

```text
16 / 8 = 2
```

意思是：LoRA 对权重的更新量 `ΔW` 会被**放大2倍**。

如果 `alpha=8`，则：

```text
8 / 8 = 1
```

更新保持原始比例，不放大也不缩小。

如果 `alpha=4`，则：

```text
4 / 8 = 0.5
```

LoRA 的更新量就**减弱一半**。

---

### 七、Alpha与Rank的关系（重点）

记住这个核心公式：

```text
alpha / r
```

这个比值就是 LoRA 更新量 `ΔW` 的**关键缩放系数**，即 `ΔW = (alpha / r) * BA`。

但注意：**不要以为实际强度只由 `alpha/r` 决定**。它只是一个缩放系数。真正落到权重上的更新量，还取决于训练得到的 `A`、`B` 参数值、初始化方式、学习率、优化器、训练步数与数据，以及不同 LoRA 变体的处理方式（比如 RS-LoRA

## 块 26

**动手试一下：先看比例，再调参数**

你正在学LoRA微调，记住一个关键点：**真正影响缩放效果的是 `alpha/r` 的比值，而不是rank单独的值**。

看两个配置：

- 配置1：`r=8, alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16, alpha=32` → 比例 = 32 ÷ 16 = **2**

虽然rank从8变成了16，但比例都是2，所以缩放效果一致。

**实操建议**：当你调整rank时，**同步调整alpha**，让 `alpha/r` 保持稳定。比如 `alpha = 2 × r` 就是个常见起点（

## 块 27

**本块学习检查**：按下面步骤动手验证，确认你理解了“小模型实验”方案。

**动手做**：打开你的Python环境，运行这段代码（直接复制）：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
print(config)
```

**观察**：代码中`r=8`、`lora_alpha=16`，这正是“小模型实验”方案。它参数少、训练快，适合测试。

**理解**：`r`控制LoRA容量（模型能学多少新知识），`alpha`控制更新幅度（每次学多猛），`target_modules`决定插入位置（改哪些层）。

**检查**：如果让你选一个方案做快速测试，你会选哪个？——答案：`r=8, alpha=16`，因为参数少、训练快。

## 块 28

## 误区1：rank越大一定越好？

**不对。** 简单任务下，大rank反而可能浪费显存、增加训练成本。rank决定LoRA容量，但容量≠效果，够用就好。

## 误区2：只调rank，不调alpha

看个例子：

```python
# 原配置
r=8
alpha=16

# 改成
r=64
alpha=16
```

此时 `α/r = 16/64 = 0.25`，缩放比例变小，LoRA对模型的影响反而**降低**了。所以改rank时，alpha要跟着调整，别只动一个。

## 误区3：alpha越大越好？

**不对。** alpha过大（即 `α/r` 缩放过大）可能导致更新过强、训练不稳定。

> 注意：`α/r` 过大只是训练不稳定的**可能原因之一**。训练震荡还可能来自学习率过高、数据质量差、batch size过小、精度问题、缺少梯度裁剪等。排查时综合看，别只盯着alpha。

---

## 核心公式与理解

```text
W' = W + (α/r)BA
```

- **r**：决定LoRA容量（能学多少）
- **α**：通过 `α/r` 缩放影响更新强度（改多猛）

理解这两个角色，LoRA参数配置逻辑就通了。

---

## 本块学习检查

**动手验证（不需要GPU）：** 用你熟悉的框架，分别计算 `r=8, alpha=16` 和 `r=64, alpha=16` 的 `α/r` 值，对比哪个配置下LoRA更新强度更高。再试一组 `r=64, alpha=64`，看看比例如何变化。

## 块 29

**LoraConfig是什么？——你的LoRA微调“控制面板”**

你已经理解了LoRA的低秩分解原理，现在进入实操阶段。在PEFT（Hugging Face的参数高效微调库）中，`LoraConfig`是你配置LoRA的核心对象。它不直接训练模型，而是告诉PEFT“怎么改”和“改哪里”。

**一句话理解**：`LoraConfig`就是一份配置单，PEFT根据它来决定LoRA的插入位置、规模、训练强度和防过拟合策略。

**LoRA训练流程回顾**：
1. 加载基础模型
2. **创建LoraConfig**（本块重点）
3. 指定LoRA插入位置
4. 生成LoRA Adapter
5. 训练LoRA参数

**LoraConfig具体管四件事**：
- **放在哪里**：通过`target_modules`指定要插入LoRA的模块（如注意力层的q、v矩阵）。
- **规模多大**：通过`r`（秩）控制LoRA矩阵的维度，`r`越小，可训练参数越少，模型容量越低。
- **训练强度**：通过`lora_alpha`控制更新权重，它决定LoRA对原模型输出的影响比例。
- **是否防过拟合**：通过`lora_dropout`设置dropout比例，随机丢弃部分LoRA输出，防止训练时过拟合。

**动手验证**：打开你的Python环境，导入`LoraConfig`，尝试创建一个最简单的配置（只设置`r=8`），打印出来看看它默认包含了哪些字段。这能帮你直观感受这个配置对象的结构。

## 块 30

## 1.2 一个简单示例——动手跑通你的第一个LoRA配置

你正在为LoRA微调打基础，这一节我们直接上手。下面这段代码就是LoRA最核心的配置，先把它复制到你的环境里跑一遍：

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

跑完没报错？恭喜，你已经成功创建了一个LoRA配置对象。现在我们来拆解这4个参数各自管什么：

| 参数 | 作用 |
| --- | --- |
| `target_modules` | 决定插入哪些层（本例是`q_proj`和`v_proj`） |
| `r` | 决定LoRA矩阵大小（本例为8） |
| `lora_alpha` | 控制缩放强度（本例为16） |
| `lora_dropout` | 控制随机失活比例（本例为0.05） |

**动手验证一下**：试着把`r`改成4或16，观察配置对象的变化——你会发现这只是数值不同，但含义完全一样。这就是LoRA配置的"手感"。

---

## 二、LoRA配置整体流程——配置怎么用起来

光有配置还不够，你需要把它接到模型上。整体流程是这样的

## 块 31

## 3.1 target_modules是什么？

一句话理解：**target_modules 就是告诉 LoRA“你要往模型的哪些部位动手”**。

Transformer 模型的 Attention 部分，内部通常长这样：

```text
Attention
├── q_proj   （Query 投影层）
├── k_proj   （Key 投影层）
├── v_proj   （Value 投影层）
└── o_proj   （Output 投影层）
```

LoRA **不会默认修改所有层**——你得手动指定。比如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的意思是：只给 **Query 投影层** 和 **Value 投影层** 加 LoRA，其他层不动。

---

## 四、为什么选 q_proj 和 v_proj？

看 Attention 的核心公式：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

拆开理解两个角色：

- **Q（Query）**：决定模型**关注什么信息**（找哪里）
- **V（Value）**：决定模型**传递什么信息**（拿什么）

所以，调整 Q 和 V，通常就能有效改变模型的行为——这是很多 LoRA 配置的**常见起点**。

> ⚠️ 注意：`q_proj`/`v_proj` 只是常见选择，**不是所有模型的通用标准配置**。不同模型的模块命名可能不同，配置前一定要先查看模型实际模块名（用 `model.named_modules()`），再决定 `target_modules`。

---

## 偷懒技巧：`target_modules="all-linear"`

如果你不想一个个枚举模块名，较新版本的 PEFT（一般 ≥ 0.7，具体以你装的版本文档为准）支持直接写：

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

如果你的 PEFT 版本太旧不支持这个写法，就退回**显式列出模块名**的方式，或者升级 PEFT。用 `print_trainable_parameters()` 可以快速验证 LoRA 是否按预期注入。

> 📄 模型许可证：使用 `AutoModelForCausalLM.from_pretrained("your/base

## 块 32

### 方案1：只调整Q和V（你的轻量起点）

先看代码——这是你第一个要动手试的配置：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**为什么先选它？** 三个字：**少、快、稳**。

- **参数少**：只训练注意力里的查询（Q）和值（V）两个投影矩阵，其他权重全部冻结。这意味着显存占用小，你的模型能跑得更轻松。
- **训练快**：可学习参数越少，反向传播计算量越小，每个epoch耗时更短。对于1.5小时/天的学习节奏，这能让你更快看到效果反馈。
- **常见起点**：这是LoRA微调里非常经典的“入门配置”，很多任务用它就能达到不错的效果。注意：它不限定模型规模，具体效果取决于模型结构和你的任务，但作为第一版实验，它足够帮你跑通流程。

**动手验证（5分钟

## 块 33

## LLaMA 系列：target_modules 怎么选？

对于 LLaMA 系列模型，最常见的做法是只对注意力层中的 **查询投影（q_proj）** 和 **值投影（v_proj）** 注入 LoRA：

```python
[
"q_proj",
"v_proj"
]
```

**为什么这样选？** 在 LoRA 微调实践中，注意力层的 Q 和 V 投影对任务适配最敏感，只微调这两个模块就能以较小参数量获得不错效果，同时显著降低显存占用和训练成本。

**动手验证（hands-on）**：如果你手边有 LLaMA 模型，可以打印 `model` 查看各层名称，确认 `q_proj` 和 `v_proj` 确实存在于每个 decoder layer 的注意力子模块中；再对比只选这两个模块与全选所有线性层时 `print_trainable_parameters()` 输出的可训练参数量差异。

---

## 对比：GPT-2 的 target_modules 写法

GPT-2 的写法与 LLaMA 不同，通常使用：

```python
[
"c_attn"
]
```

**关键区别（务必理解）**：GPT-2 的注意力层使用的是 Transformers 库中的 `Conv1D`（不是普通的 `nn.Linear`），它把 Q/K/V 三个投影合并放在一个叫 `c_attn` 的模块里。更关键的是，GPT-2 的 `Conv1D` 权重形状为 `(out_features, in_features)`，与 `nn.Linear` 相同，但前向传播方式不同。

因此，在 GPT-2 这类层上加 LoRA 时，**必须设置 `fan_in_fan_out=True`**，PEFT 才能按正确的布局注入低秩矩阵。

**可运行示例（GPT-2）**：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",

## 块 34

### BERT的target_modules配置：别照搬，先看模型结构

BERT的注意力层通常把`query`和`value`拆成独立模块，所以`target_modules`可以设为：

```python
["query", "value"]
```

**关键提醒**：不同模型的模块命名差异很大，不能直接复制其他模型的配置。你需要先查看实际加载的模型结构：

```python
for name, _ in model.named_modules():
    print(name)
```

运行后会看到类似`q_proj`、`v_proj`这样的名字，然后再把它们填进`target_modules`。

**匹配机制**：`target_modules`里的字符串按模块名做**后缀匹配**——写`"q_proj"`会命中所有以`q_proj`结尾的层（各层各头都算）。也可以写完整名字精确定位，或传正则表达式。

**动手验证**：加载一个BERT模型，运行上面的打印代码，把输出中所有包含`q`和`v`的模块名抄下来，然后配置`target_modules`。对比一下，你会发现和LLaMA系的`q_proj`/`v_proj`命名是否一致。

#

## 块 35

**6.1 r是什么？——用“压缩-还原”理解它**

想象你要把一张`4096×4096`的大表格（原矩阵）拆成两个小表格相乘：`ΔW = (α/r)BA`。这里的`r`（rank，秩）就是决定这两个小表格“中间维度”的旋钮。原始权重W被冻结，仅训练低秩矩阵A和B。

- 设`r=8`，则A是`8×4096`，B是`4096×8`。  
- LoRA参数量为65,536，原始参数量为16,777,216，占比约0.39%，远小于原来的`4096×4096`。

**6.2 r

## 块 36

### 7.1 alpha作用：控制LoRA的“音量”

想象一下，LoRA微调不是直接替换原模型权重，而是加了一个“小旋钮”来调整影响。LoRA冻结原始权重W，训练低秩矩阵A和B，最终更新公式为 **W' = W + (alpha/r)BA**，其中(alpha/r)是缩放系数，控制增量ΔW的幅度。

**动手验证一下：**
配置 `r=8, lora_alpha=16`，缩放系数为 `16/8=2`，即增量ΔW被放大2倍。你可以试着改小 `alpha`（比如 `lora_alpha=4`），感受更新幅度的变化。

### 7.2 alpha常见

## 块 37

## 8.1 dropout作用

训练时，LoRA会**随机关闭部分神经连接**，目的是**减少过拟合**。

```python
lora_dropout=0.05
```

含义：**5%的LoRA路径随机失活**。

---

## 8.2 如何选择dropout？

经验起点（非固定规则，需结合验证集是否过拟合来调）：

| 数据规模 | dropout起点参考 |
|---------|---------------|
| 数据少   | 约0.05~0.1    |
| 数据多   | 约0~0.05      |

若训练数据非常丰富，可直接设：

```python
lora_dropout=0
```

---

**动手验证**：

## 块 38

## 9.1 基础配置：动手搭一个LoRA

这段配置适合在**7B模型**上做测试，是LoRA微调最常用的起步模板。我们直接看代码，然后逐行拆解。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                    # LoRA容量：秩的大小，8是轻量级起点
    lora_alpha=16,          # 更新缩放：控制新权重对原模型的影响强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只改注意力层的q和v投影
    lora_dropout=0.05,      # 防止过拟合：训练时随机丢弃5%的LoRA参数
    bias="none",            # 不训练偏置项，保持轻量
    task_type="CAUSAL_LM"   # 任务类型：因果语言模型（如GPT类）
)
```

**四个核心参数速记：**

| 参数 | 值 | 作用 |
|------|-----|------|
| r | 8 | LoRA容量，越小越省显存 |
| alpha | 16 | 更新缩放，越大微调力度越强 |
| target_modules

## 块 39

## 9.1 基础配置：跑通一个完整的 LoRA 微调流程

你现在的目标是 **LoRA 微调**，而且喜欢动手实操。这一节我们就直接看一个能跑通的完整案例——用 BERT 做二分类，把 LoRA 配置、训练、保存、重载验证全流程走一遍。代码里每一步都有注释，跟着敲一遍就懂了。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 加载带分类头的基础模型（分类头是新初始化的，需要一起训练/保存）
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

注意：这里的分类头是**随机初始化**的，不是预训练好的，所以必须训练。通过 `modules_to_save` 指定后，分类头会与 LoRA 参数一起被训练和保存。

### 第二步：配置 LoraConfig

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

逐个参数看：

- **`r=8`**：LoRA 的低秩维度。秩越小，可训练参数越少，但表达能力也弱一些。8 是个常用起点。
- **`lora_alpha=16`**：缩放因子。实际生效的缩放比例是 `alpha / r`，这里就是 `16/8 = 2`。
- **`lora_dropout=0.05`**：防止过拟合的 dropout 比例。
- **`bias="none"`**：不训练偏置项。
- **`task_type="SEQ_CLS"`**：告诉 PEFT 这是序列分类任务。
- **`target_modules=["query", "value"]`**：只对 BERT 注意力层里的 query 和 value 矩阵加 LoRA。
- **`modules_to_save=["classifier"]`**：**最关键的一行**。它把分类头也纳入训练，并且保存时会一起存下来。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

这里会打印可训练参数的数量。你应该能看到两部分：LoRA 注入的低秩矩阵 + 分类头的全量参数。

### 第四步：保存

```python
# ...（此处进行训练，略）...

model.save_pretrained("./lora_ckpt")
```

保存的是 **LoRA 适配器权重和 `modules_to_save` 指定的分类头**，不是完整的基础模型。文件大小取决于分类头参数量，通常远小于完整模型。

### 第五步：重载并验证分类头

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
assert torch.allclose(w_before, w_after), "分类头权重不一致"
```

# 模型许可证：BERT 使用 Apache 2.0 许可证，商用无需额外授权。
# 注意：BERT-base 模型 FP32 推理约需 1.2GB 显存，训练时请根据 batch size 适当增加显存预留。

## 块 40

## 10.1 普通文本生成：从零开始的LoRA配置

你现在的目标是**LoRA微调**，而且喜欢动手实践——那咱们就直接切入正题。对于**普通文本生成**任务（比如让模型续写故事、生成对话），PEFT库的`LoraConfig`推荐这样配置：

```python
r=8
alpha=16
dropout=0.05
```

### 这三个参数分别是什么意思？

- **`r`（秩）**：决定LoRA引入的可训练参数规模。`r=8`意味着每个被适配的层会引入8维的低秩矩阵。**值越小，新增参数越少，训练越快，但模型能学到的“新东西”也越有限**。
- **`alpha`（缩放因子）**：控制LoRA更新的“步长”。`alpha=16`配合`r=8`，实际缩放比例是`alpha/r = 2`。这个比例决定了微调时对原模型权重的“扰动强度”。
- **`dropout`（丢弃率）**：在LoRA层中随机丢弃一部分神经元，防止过拟合。`0.05`是一个比较保守的默认值。

### 为什么普通文本生成用这套配置？

普通文本生成任务通常**不需要大幅度改变模型的行为**——模型本身已经会“写”，你只是想让它的风格或领域稍微偏向你的需求。所以用较小的`r`（8）就够了，配合`alpha=16`保持稳定更新，`dropout=0.05`防止在小数据上过拟合。

### 动手验证一下

如果你手边有Python环境，可以运行这段代码来确认配置能正常加载：

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

**观察点**：输出中应该能看到你设置的`r=8`、`lora_alpha=16`、`

## 块 41

**动手配置：普通文本生成的LoRA参数**

先看核心代码，这是你微调文本生成模型的起点：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**逐个参数理解，边看边想“改哪里”：**

- `target_modules`：决定改哪里。这里指定 `q_proj` 和 `v_proj`，即只修改注意力层的查询和值投影。
- `r`：决定LoRA容量。`r=8` 表示低秩矩阵的秩为8，数值越大，可学习参数越多，模型表达能力越强。
- `

## 块 42

好的，我们先来直面一个很现实的问题：你学LoRA是为了省显存，但为什么用起来还是卡？

**1.1 LoRA的“隐藏成本”：模型本体必须完整加载**

LoRA的精妙之处在于，它冻结了原来的大模型，只训练旁边新增的一小部分参数。这确实大幅减少了**需要更新**的参数数量。

但请注意，**冻结 ≠ 不需要加载**。你依然要把整个基础模型完整地放进显存里，才能进行前向传播计算。

我们来算一笔账。以常见的7B（70亿参数）模型为例，如果用FP16（半精度浮点数，每个数占2字节）格式加载，光是模型权重就需要：

`7 × 10⁹ 参数 × 2 字节 ≈ 14 GB 显存`

这14GB仅仅是模型“躺”在显存里的静态大小。一旦开始训练，还需要额外的空间来存放**激活值**（中间计算结果）、**梯度**（用于更新LoRA参数）以及**优化器状态**。这些加起来，实际显存需求会远超14GB。

所以，结论很直接：**LoRA解决了“训练哪些参数”的问题，但没解决“加载多少参数”的瓶颈。** 对于消费级显卡，这依然是个巨大的挑战。

---

**✋ 动手验证一下（动手做，印象更深）**

打开你的Python环境，用`transformers`库加载一个小的7B模型（比如`meta-llama/L

## 块 43

**QLoRA核心思想：用4-bit模型跑LoRA**

你的目标是LoRA微调，但直接微调大模型太吃显存。QLoRA的思路很直接：**先把基础模型压缩成4-bit，再在上面插LoRA训练**。

流程是：
1. 把FP16基础模型量化成4-bit
2. 加载这个量化后的模型
3. 插入LoRA适配器
4. 只训练LoRA参数

对比一下：
- LoRA：`FP16模型 + LoRA`
- QLoRA：`4-bit模型 + LoRA`

关键区别在于：**量化的是基础模型权重，不是LoRA参数**。训练时，4-bit基础模型被冻结，LoRA参数保持FP

## 块 44

**3.1 模型存储方式：为什么4-bit能省显存？**

先看一张表，记住每个参数占用的空间：

| 类型 | 每个参数占用 |
|---|---|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下：** 以7B（70亿参数）模型为例，用FP16存储需要 `7B × 2 ≈ 14GB`；换成4-bit后只需 `7B × 0.5 ≈ 3.5GB`。显存直接减少约75%！

**注意：** 这3.5GB只是**理论下界**（纯权重）。实际运行时，显存还会被量化常数、分组元数据（NF4按组量化要存额外参数）、未量化的模块（如norm、embedding、lm_head、LoRA参数）以及计算

## 块 45

**NF4是什么？——给想动手做LoRA的你**

NF4全称NormalFloat4，中文叫“正态分布4位浮点”。但别被名字骗了，它**不是**IEEE那种通用4位浮点格式，而是QLoRA论文为**神经网络权重近似正态分布**这一特点，专门设计的一套4-bit量化“码本”。

**怎么理解码本？** 想象一本只有16个（2⁴）取值的字典。这16个点不是均匀分布的，而是按标准正态分布的分位数排布——**0附近（权重最密集处）划分更细**，远离0的地方点更稀疏。

**量化过程通常包括：**
1. 把每组权重做归一化；
2. 在

## 块 46

**为什么NF4适合模型权重？**

大模型训练完的权重有个特点：**近似正态分布**，也就是大量参数都挤在0附近，越靠近0越密集。

普通INT4量化通常基于张量的实际数值范围进行线性映射（如min-max量化），但权重分布不均匀时精度损失较大。NF4采用分位数量化，根据权重分布的分位数构造量化级别，使得每个量化级别对应的概率质量大致相等，从而在0附近提供更高的精度。

**NF4量化过程（快速版）**

原始FP16权重（如0.023、-0.145、0.876）→ 归一化 → 映射到NF4 → 保存4-bit

## 块 47

**本块学习检查**：读完5.1后，请暂停并回答：4-bit量化除了保存量化后的权重，还需要额外保存哪两个参数？为什么这会增加显存负担？

---

**5.1 为什么需要双重量化？**

你已经知道4-bit量化能把FP16权重压缩成4-bit，但别忘了——量化不是白干的，它需要“说明书”。

这个说明书就是**量化参数**。具体到每组权重，你需要保存两个东西：

- **scale**（缩放因子）
- **zero point**（零点偏移）

问题来了：这些额外参数**也占显存**。模型越大，分组越多，这些“说明书”累积起来就是一笔不小的开销。

**5.2 双重量化思想**

普通量化只做一步：FP16权重 → 4-bit权重 + scale参数。

## 块 48

## 6.1 为什么需要分页优化器？

训练时显存峰值不只是模型本身，还有**梯度**和**优化器状态**。比如Adam优化器要额外保存**一阶动量**和**二阶动量**两份状态。

QLoRA虽然大幅降低了模型显存占用，但训练中仍可能遇到**显存突然不足（OOM）**——尤其是优化器状态在某个瞬间暴涨时。

## 6.2 分页优化器思想

Paged Optimizer借鉴了操作系统**虚拟内存**的思路，核心机制是：

> 借助CUDA的**虚拟内存管理（VMM）**和**统一内存（Unified Memory）**，当显存出现峰值、即将OOM时，自动把优化器状态按"页"换出到CPU内存，需要计算时再换回GPU。

流程如下：

```
GPU显存 → 出现峰值（接近/超出上限）
→ 优化器状态按页换

## 块 49

## 6.3 类比理解：分页优化器到底在干嘛？

把显存想象成你的**桌面**，把CPU内存想象成**柜子**。

**普通优化器**的做法是：把所有东西（模型参数、梯度、优化器状态）全堆在桌面上。桌面一旦堆满，程序直接崩溃（OOM），没法继续训练。

**分页优化器**的做法是：桌面上只放**当前正在用的文件**（比如当前层的梯度），暂时用不到的就存进柜子（CPU内存），等需要时再取回来。这样桌面永远有空间，训练就不会因为显存爆掉而中断。

---

## 6.4 代码里怎么启用？

不用手写换页逻辑，`bitsandbytes` 已经帮你实现了。只需要在 `TrainingArguments` 里设置 `optim` 参数：

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

**两个关键点：**

1. **`paged_adamw_8bit` vs `paged_adamw_32bit`**：前者优化器状态用8-bit存，更省显存；后者精度更高但占用更多。QLoRA论文里常用的是8-bit版本。
2. **参数名随版本变化**：不同Transformers/bitsandbytes版本支持的`optim`取值可能不同（还有`adamw_bnb_8bit`等）。如果报"未知optim"错误，查一下你当前版本的文档，换用支持的名称即可。

**使用场景**：分页优化器主要解决**峰值显存溢出**问题。实际训练时通常配合`gradient_checkpointing=True`、小`per_device_train_batch_size`+梯度累积一起用，把显存峰值降到最低

## 块 50

### 7.1 4-bit量化：把模型“瘦身”

咱们的目标是跑LoRA微调，但显存有限。QLoRA的第一招就是**4-bit量化**，把模型权重从FP16（16位）压缩到NF4（4位），体积直接缩小约4倍。就像把高清照片转成压缩格式，省空间但画质略降。

### 7.2 双重量化：连“尺子”也压缩

量化时需要“scale”（缩放因子）来还原数值。双重量化就是把这个scale也做一次量化，进一步减少额外占用。相当于连量衣服的尺子也折叠起来，省得背包太鼓。

### 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态很吃显存。分页优化器让GPU显存不够时，自动把部分数据换到CPU内存，用完了再换回来。就像手机内存不够时用云存储，虽然慢点但能跑起来。

---

## 八、QLoRA vs LoRA：一张表看懂

| 项目 | LoRA | QLoRA |
|

## 块 51

## 4-bit 加载：让大模型“瘦身”跑起来

你正在做 LoRA 微调，目标很明确：**用更少的显存，把大模型跑起来**。`load_in_4bit` 就是关键开关——开启后，模型权重会以 4-bit 精度加载，显存占用大幅下降。这是 QLoRA 的基石。

### 三个核心配置项

配置 QLoRA 时，你会看到四个参数，逐个拆解：

| 参数 | 值 | 作用 |
|------|-----|------|
| `load_in_4bit` | `True` | 开启 4-bit 加载 |
| `bnb_4bit_quant_type` | `"nf4"` | 使用 NF4 量化格式 |
| `bnb_4bit_use_double_quant` | `True` | 开启双重量化，进一步省显存 |
| `bnb_4bit_compute_dtype` | 见下文 | 指定计算精度 |

### 存储 vs 计算：两个精度要分清

这是最容易混淆的点。**存储 dtype** 和 **计算 dtype** 是两回事：

- **存储**：权重以 4-bit（NF4）格式存着，目的是省显存。
- **计算**：真正做矩阵运算时，权重会被**反量化**成更高精度来算。这个计算精度由 `bnb_4bit_compute_dtype` 决定。

**为什么不直接用 4-bit 算？** 因为精度太低，训练不稳定。反量化回高精度计算，是“省显存”和“保精度”的平衡。

### 计算精度怎么选？别写死！

常见选择是 `torch.bfloat16` 或 `torch.float16`。但**不要固定写死 bfloat16**——它需要较新的 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系）不支持，会报错。

**自动判断最稳妥**：

```python
import torch
from transformers import BitsAndBytesConfig

# 根据 GPU 能力选择：支持 BF16 就用 bfloat16，否则退回 float16
compute_d

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面几节我们分别看了 `BitsAndBytesConfig`（怎么用 4-bit 加载模型）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在最大的疑问是：**这些零件怎么拼成一台能跑的 QLoRA 训练机器？**

下面给你一个**最小可运行骨架**，把完整流程串起来。模型和数据集请替换成你自己的。

### 第一步：4-bit 量化配置

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

这里做了三件事：开启 4-bit 加载、用 NF4 量化类型、开启双重量化。`compute_dtype` 根据你的 GPU 能力自动选择 bfloat16 或 float16——这是计算时的精度，不是存储精度。

### 第二步：以 4-bit 加载基础模型

```python
# 模型许可证：请确认所选模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：4-bit量化加载7B模型约需6GB显存；FP16加载7B模型需至少14GB显存
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except (OSError, RuntimeError) as e:
    print(f"模型加载失败: {e}")
    raise
```

`quantization_config` 把第一步的配置传给模型加载。`device_map="auto"` 让框架自动分配设备。顺手把 pad_token 设为 eos_token，避免后续数据处理报错。

### 第三步：为 k-bit 训练做准备（关键一步）

```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
```

这一步**不能省**。它处理了量化模型的特殊问题：比如 norm 层和输出层的精度问题，以及开启梯度检查点所需的输入梯度设置。`use_gradient_checkpointing=True` 和后面 `TrainingArguments` 里的 `gradient_checkpointing=True` 是**对应关系**，两处都要开。

### 第四步：注入 LoRA

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

LoRA 只训练低秩增量参数，原始模型权重保持冻结。`target_modules` 按你的实际模型模块名调整，也可以用 `"all-linear"` 省事。`print_trainable_parameters()` 会打印可训练参数量，帮你确认注入是否成功。

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

Causal LM 的 `labels` 直接复制 `input_ids` 即可。真实场景请换成足够规模的数据。

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
    tokenizer=tokenizer,
)
try:
    trainer.train()
except (OSError, RuntimeError) as e:
    print(f"训练失败: {e}")
    raise
```

注意两个关键点：

- `optim="paged_adamw_8bit"` 是分页优化器，依赖 bitsandbytes，名称随版本可能不同。
- `gradient_checkpointing=True` 必须与第三步的 `use_gradient_checkpointing=True` 呼应。

### 完整流程回顾

```
量化配置 → 4-bit加载 → prepare（梯度检查点） → LoRA注入 → 数据准备 → 训练
```

每一步都在为下一步做准备：量化配置决定模型怎么加载，prepare 让量化模型能训练，LoRA 注入决定训练哪些参数，最后用分页优化器和梯度检查点跑起来。

### 本块学习检查

**操作验证任务**：把上面的骨架代码复制到你的环境中，替换 `model_name` 为你本地的任意 Causal LM（或 HuggingFace 上的小模型），运行并记录输出结果。

## 块 53

## 九、把QLoRA的“五步骨架”串起来（可跑通）

你已经知道QLoRA用NF4量化+双重量化把模型压到4-bit，也知道分页优化器能缓解显存峰值。但真正写代码时，这五步必须**按顺序配合**，缺一步就会出问题。下面就是“能说明关系”的最小骨架，跟着敲一遍，跑通它，你就掌握了QLoRA的调用逻辑。

### 第一步：`BitsAndBytesConfig` —— 决定模型怎么“瘦身”加载

这一步是**基础配置**，告诉系统：基础模型要用4-bit加载，具体用NF4量化、开启双重量化、并指定计算dtype（比如BF16/FP16）。

> 你可以把它理解为“给模型定做一套瘦身方案”——先定好规则，后面加载模型时才按这个规则执行。

### 第二步：加载模型并应用量化

使用`from_pretrained`加载模型时传入第一步的`BitsAndBytesConfig`，模型即以4-bit量化方式载入。

### 第三步：`prepare_model_for_kbit_training()` —— 建议调用

这一步是对量化模型做“训练前适配”，比如把部分层转成计算精度、开启输入梯度以支持梯度检查点等。

> **关键提醒**：建议调用`prepare_model_for_kbit_training`以启用梯度检查点等优化，但并非强制要求。跳过它不会直接导致训练不收敛或报错。

### 第四步：`get_peft_model(..., LoraConfig)` —— 注入可训练的LoRA

4-bit主干是**冻结**的（不更新权重），这一步把**可训练的LoRA适配器**注入进去。这样你只训练LoRA那一点点参数，而不是整个大模型。

### 第五步：设置`TrainingArguments` —— 包含梯度检查点与分页优化器

在`TrainingArguments`中设置`gradient_checkpointing=True`和`optim="paged_adamw_8bit"`。前者用于节省显存，后者启用分页优化器，当显存不够时优化器状态会被“分页”到CPU内存，需要时再换回GPU，从而降低单次显存峰值。

> **注意**：`gradient_checkpointing`可以在`prepare_model_for_kbit_training`中设置`use_gradient_checkpointing=True`，也可以在`TrainingArguments`中设置`gradient_checkpointing=True`，两者可独立使用，但通常建议同时开启以节省显存。

---

### 最小骨架（关系梳理）

```
BitsAndBytesConfig（4-bit加载：NF4 + 双重量化 + 计算dtype）
        ↓
from_pretrained(..., quantization_config=BitsAndBytesConfig)（加载模型并应用量化）
        ↓
prepare_model_for_kbit_training()（量化模型适配，建议调用）
        ↓
get_peft_model(..., LoraConfig)（注入可训练LoRA到冻结的4-bit主干）
        ↓
TrainingArguments(gradient_checkpointing=True, optim="paged_adamw_8bit")
```

> **注意**：这是“能说明关系”的最小骨架。真实训练时，你需要换成足够规模的数据、合理的超参，并按你所用版本核对`optim`名称、`prepare_model_for_kbit_training`的参数与`processing_class`/`tokenizer`写法。

---

## 十、电脑配置批注

> **模型许可证**：本部署文档引用的基础模型需遵守其许可证条款。

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 先搞懂：LoRA不是直接改基础模型

你训练LoRA时，**基础模型是被冻结的**（不参与训练），真正被训练的是一个很小的“补丁”——**LoRA Adapter**。

训练结束后，你的最终产物是：

```
基础模型（冻结，没变）
+
LoRA Adapter（训练后的增量）
```

**关键点**：你保存的通常**不是完整模型**，而是两个小文件：

```
adapter_model.safetensors   ← LoRA权重
adapter_config.json         ← 配置信息
```

> 举个例子：你用Llama-7B训练了一个LoRA，保存后得到的不是整个Llama-7B，而是那个小小的Adapter。

---

### 1.2 Adapter有什么特点？

**优点**：
- 文件小（可能只有几十MB，而不是几个GB）
- 保存快
- 一个基础模型可以搭配多个Adapter

比如同一个基础模型，可以分别搭配：
- 客服Adapter
- 医疗Adapter
- 法律Adapter

**缺点**：
部署推理时，**基础模型 + Adapter** 两部分必须同时存在，缺一不可。

---

## 二、为什么需要合并LoRA权重？

既然Adapter这么方便，为什么还要合并？

因为**部署时麻烦**——每次都要同时加载两个部分，而且推理框架不一定支持Adapter格式。

**合并（merge）**就是把Adapter的增量“写回”基础模型，得到一个**完整的、独立的模型**。合并后，你只需要一个文件，直接加载就能用，不再依赖Adapter。

---

## 三、`merge_and_unload()`怎么用？

PEFT库提供了现成方法：`merge_and_unload()`

核心逻辑：
1. 把Adapter权重加到基础模型上
2. **卸载**Adapter，只保留合并后的完整模型

```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("基础模型路径")

# 加载LoRA Adapter
model = PeftModel.from_pretrained(base_model, "你的adapter路径")

# 合并并卸载Adapter
model = model.merge_and_unload()
```

合并后，`model`就是一个**完整的独立模型**，可以直接保存和部署。

---

## 四、如何导出完整模型？

合并完成后，用常规方式保存：

```python
model.save_pretrained("导出路径")
```

> **部署提示**：加载合并后的模型进行推理时，请确保显

## 块 55

**本块学习检查**：读完本块后，请用一句话回答：为什么部署时希望把LoRA权重合并进基础模型？

---

**2.1 推理时的问题**

未合并时，推理时输入只经过基础模型一次前向传播，LoRA适配器通过合并权重生效，而非串联：

输入 → 基础模型（权重W）→ 输出

但对部署来说，你更希望得到一个“完整模型”，输入直接走一遍就出结果：

输入 → 完整模型 → 输出

所以，需要把LoRA权重合并进去。

---

**2.2 合并的数学原理**

LoRA训练时，最终权重是：

W' = W + (α/r)BA

其中，B和A是低秩分解

## 块 56

## 3.1 `merge_and_unload()` 是干什么的？

你已经训练好了LoRA，现在想把成果“固化”下来。PEFT库提供了一个关键方法：

```python
merge_and_unload()
```

它的作用一句话概括：**把LoRA的低秩增量合并进基础模型，然后移除PEFT的包装层**。

用公式理解就是：

```text
W' = W + ΔW，其中 ΔW = (α/r)BA
```

（`W`是原权重，`ΔW`是LoRA学到的低秩增量，`W'`是合并后的新权重）

---

## 3.2 合并前后，模型发生了什么变化？

**合并前**，你的模型是一个被PEFT包装过的结构：

```text
PeftModel
├── Base Model（基础模型）
└── LoRA Adapter（LoRA适配器）
```

**合并后**，包装被拆掉了，变成了一个**普通的Transformer模型**——LoRA学到的低秩增量已经写进了模型权重里，不再需要单独的适配器文件。

---

## 四、完整流程：从训练到导出

整体操作链路如下：

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步来。

---

## 五、第一步：加载基础模型和Adapter

假设你有两个路径：基础模型路径和Adapter路径。

```python
base_model_path = "model_path"   # 基础模型位置
adapter_path = "adapter_path"    # LoRA适配器位置
```

加载代码：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 把LoRA适配器挂载到基础模型上
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构为：`基础模型 + LoRA适配器`。

---

## 六、第二步：执行合并（重点）

一行代码完成合并：

```python
merged_model = model.merge_and_unload()
```

合并的数学本质：

```text
W' = W + (α/r)BA
```

（`W`是原权重，`A`、`B`是LoRA学到的低秩矩阵，`α/r`是缩放系数，`W'`是合并后的新权重）

**关键点**：执行完这一步，`merged_model` 已经**不再是** `PeftModel`，而是一个**普通的Transformer模型**。

---

## 七、第三步：导出合并后的模型

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

> **注意**：以上是示意结构。实际文件会因模型/分词器类型不同而变化，例如可能还有 `generation_config.json`、`special_tokens_map.json`，或者分片权重文件；有些模型的token

## 块 57

## 三、merge_and_unload是什么？（重点）

### 3.1 方法作用——一句话理解

**merge_and_unload** 就是把 LoRA 训练时加在底座模型上的“小补丁”（Adapter）**永久焊死**到模型里，得到一个完整的、独立的模型文件。

---

### 合并后，你得到什么？

合并后的模型是**普通 Transformers 模型**，不再需要 PEFT 库来加载。加载方式变得极其简单：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

用户输入 → 完整模型 → 生成结果，就这么直接。

---

### 合并 vs 不合并，怎么选？

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需要PEFT | 直接Transformers |

**核心权衡**：合并 = 牺牲灵活性换部署简单。如果你需要多个 Adapter 动态切换，就别合并；如果追求部署省事，就合并。

---

### 合并后必查清单（避免“能加载但行为不对”）

导出后务必核查以下四项：

1. **Tokenizer 完整且一致**：确认保存的 tokenizer 与训练时一致，特别是训练中**新增/修改过的特殊 token**（如自定义的 `<|im_start|>`、pad/eos 等）都在 `special_tokens_map.json` / `tokenizer_config.json` 里；若训练时扩过词表，还要确认 embedding/lm_head 尺寸与词表匹配。

2. **generation_config.json**：如果依赖特定的生成默认值（eos/pad token id、`max_new_tokens`、采样参数等），确认它一并导出，否则推理时可能回退到不合适的默认行为。

3. **dtype**：确认合并/保存时的精度符合预期（FP16/BF16 还是 FP32），dtype 不一致会影响显存占用与数值表现。

4. **config.json 完整性**：确认模型结构配置（层数、hidden size、任务头等）正确，且已**

## 块 58

## 10.1 推荐合并场景

你是要学LoRA微调的新手，先记住一个判断标准：**合并就是把LoRA“焊死”进基础模型**，得到一个普通模型。下面两个场景最该合并。

**场景1：上线部署（只要一个Adapter、追求简单）**

生产环境流程就是：

```text
服务器 → 直接加载模型 → 提供API
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```

如果你只需要**一个固定Adapter**，想让部署链路尽量简单——不额外装PEFT、不做运行时加载——合并更方便。

> ⚠️ 注意：如果推理框架本身支持直接加载LoRA（比如vLLM的多LoRA服务），或者你要在同一个基础模型上**热切换多个Adapter**，那**不合并**、以Adapter形式部署反而

## 块 59

## 10.2 不建议合并场景

**场景1：多个任务Adapter**

想象你有一个基础模型，上面挂了多个Adapter（比如Adapter A、B、C），每个负责不同任务。**保留Adapter更灵活**——想用哪个任务就加载哪个，随时切换，不用重新训练。

**场景2：持续训练**

如果你打算继续调整LoRA，也**保留Adapter**。合并后就变成固定权重，没法再微调了。

> **动手验证**：如果你有多个Adapter，试着不合并直接推理，感受切换的便利。

---

## 十一、QLoRA合并注意事项（重点）

QLoRA有个关键点：**基础模型以4-bit（NF4）存储，LoRA参数是高精度（FP16/BF

## 块 60

### 直接合并的“坑”与稳妥解法

你训练完 QLoRA 后，最想干的事肯定是把 LoRA 权重“焊死”回基础模型里，得到一个完整的模型文件。但如果你直接在 4-bit 量化状态下执行 `merge_and_unload`，可能会遇到两个问题：

1. **精度损失**：4-bit 权重需要先反量化再合并，这个过程和全程用高精度计算并不完全等价，结果可能有细微偏差。
2. **类型或版本不兼容**：可能出现权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

**注意**：上面只是“可能出现”的情况，不是每次都会报错。能不能顺利合并、合并质量如何，取决于你的版本、量化配置，以及是否先做了反量化。所以，**以你所用版本的文档和实际运行结果为准**。需要强调的是，QLoRA 的正确合并机制是：必须先反量化到 FP16/BF16 再合并，直接合并并非可行路径。

---

### 稳妥做法：先反量化到高精度再合并

既然直接合并有风险，那我们就绕开它。常见且稳妥的流程是：

1. **用高精度（FP16/BF16）重新加载基础模型**——注意，这里不传 `quantization_config`，让模型以高精度加载。
2. **加载同一份 LoRA Adapter**。
3. **执行 `merge_and_unload()`**。
4. **保存完整模型和 tokenizer**。

这样合并发生在高精度权重上，就避开了 4-bit 状态下的类型和精度问题。

下面是完整可运行的代码，你可以直接照着改路径用：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 1) 关键：以高精度（FP16/BF16）重新加载基础模型，而不是再走 4-bit 量化
#    合并要发生在高精度权重上，所以这里不传 quantization_config
try:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=compute_dtype,
        device_map="auto",        # 显存不足时可改为 device_map={"": "cpu"} 在 CPU 上合并
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

# 2) 挂载训练得到的同一份 Adapter
try:
    model = PeftModel.from_pretrained(base_model, adapter_path)
except Exception as e:
    print(f"Adapter 加载失败: {e}")
    raise

# 3) 合并；safe_merge=True 会在合并时做数值检查（如检测 NaN）
try:
    merged_model = model.merge_and_unload(safe_merge=True)
except Exception as e:
    print(f"合并失败: {e}")
    raise

# 4) 保存完整模型与 tokenizer（tokenizer 从基础模型或训练目录加载均可，保持与训练一致）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
merged_model.save_pretrained(merged_out)
tokenizer.save_pretrained(merged_out)
```

# 模型许可证：请确认你使用的基础模型许可证类型（如 Llama 2

## 块 61

## 直接合并QLoRA模型？先看这三点

你正在做LoRA微调，目标是拿到能用的模型。QLoRA训练完，直接`merge_and_unload`合并时，有三个坑必须先知道：

**1. 版本决定能不能直接合并**

不同PEFT版本对“量化模型能否直接合并”支持不一样。PEFT 0.6.0及以上版本支持直接合并QLoRA模型，更早版本需先反量化再合并。**关键操作**：加载基础模型时，**不要再传**`quantization_config`或`load_in_4bit`参数，这样拿到的是高精度权重，再合并就稳了。动手前先查你所用版本的`merge_and_unload`文档。

**

## 块 62

## 合并时开启校验并做一致性验证（建议）

`merge_and_unload` 支持 `safe_merge=True` 参数，合并时会自动做数值检查（比如检测是否产生 NaN），相当于给合并操作加一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，强烈建议你**对比合并前后模型在同一批输入上的输出**，确认合并没有引入明显偏差。流程如下：

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits 或生成文本是否基本一致

最小验证代码的思路：用固定 prompt，先比较 logits 的最大绝对差，再比较贪心生成的

## 块 63

**本块学习检查**：完成下方“动手验证”中的两个对比任务，并记录你得到的 `max_abs_diff` 数值与生成文本是否一致的结果。

---

### 为什么合并后还要“校验”？

你在 §11.2 已经用 `merge_and_unload()` 把 LoRA 适配器合并进了基础模型。但合并是个数学近似过程，尤其是 QLoRA 场景下，合并前后的输出**理论上会有微小差异**。本块教你一个“笨但可靠”的方法：**拿合并前的模型当对照，逐项对比**。

### 核心陷阱：不能拿“合并过的对象”当“合并前”

代码注释里强调了一个关键点：`merge_and_unload()` 返回合并后的模型，原 `PeftModel` 对象仍保留，但为了对比合并前后差异，仍需重新加载未合并的模型作为基准。

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

这段代码就是干这个的：从原始路径重新加载基础模型，再挂上 adapter，得到 `model_before`。

### 校验步骤 1：对比 logits（模型输出的原始分数）

用同一个固定输入（比如 `prompt = "请用一句话介绍你自己。"`），分别让 `model_before` 和 `merged_model` 前向传播，然后计算两者 logits 的**最大绝对差**：

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果**：这个数值越接近 0，说明合并前后模型内部计算越一致。QLoRA 场景下，由于量化误差，通常是一个**较小的近似值**，而不是精确的 0。

### 校验步骤 2：对比生成文本（更直观）

数值对比不够直观？那就直接**生成文本**看看。注意要设置 `do_sample=False`，去掉随机采样，保证两次生成是确定性的，才能公平对比：

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
print("合并前:", text_before)
print("合并后

## 块 64

## 合并时开启校验并做一致性验证（建议）

### 为什么必须单独再加载一份模型？

`merge_and_unload()` 会**原地**修改模型——它把 LoRA Adapter 直接合并进底层权重，并移除 PEFT 包装。这意味着合并完成后，原来的 `model` 变量已经变成了"合并后"的状态。

如果你拿它和合并前的自己对比，两边其实是**同一份已合并的权重**，`max_abs_diff` 会恒等于 0——验证就完全失去意义了。

**正确做法**：在合并之前，先用 `base_for_ref` + `model_before` 重新加载一份**未合并**的 PeftModel 作为对照基准。这样你才有真正的"合并前 vs 合并后"可比对。

### 校验代码示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base, "lora_path")
model_before = PeftModel.from_pretrained(base, "lora_path")  # 对照基准

merged = model.merge_and_unload()
inputs = tokenizer("测试输入", return_tensors="pt")
with torch.no_grad():
    logits_before = model_before(**inputs).logits
    logits_after = merged(**inputs).logits
diff = (logits_before - logits_after).abs().max().item()
print(f"max_abs_diff: {diff}")
```

### 显存不够怎么办？

如果同时放不下两个模型，就**分两次跑**：

1. 先用未合并的 PeftModel 跑一遍推理，把输出的 logits 或生成文本**保存到磁盘**，然后释放显存；

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

## 块 65

## 十一、QLoRA模型合并注意事项（重点）——合并时开启校验并做一致性验证（建议）

你在做 LoRA 微调，目标是把训练好的 Adapter 合并回基础模型。这一步最容易出问题，下面把常见坑和稳妥做法讲清楚。

> **模型许可证**：本部署文档引用的基础模型需遵守其原始许可证（如 Llama 2 Community License，商用月活>7亿用户需获得 Meta 授权）。部署前请确认所用模型的许可证类型及合规要求。

### 先记住一个最稳妥的原则

**"先反量化（重载高精度）再合并"是跨版本最稳妥的做法。**

这里的"反量化"**不是**手动逐层还原 4-bit 权重，而是——**重新加载一份不带量化配置的基础模型**。具体做法：

```python
# 不传 quantization_config / load_in_4bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16  # 或 float16
)
# 再挂载同一份 Adapter 后合并
peft_model = PeftModel.from_pretrained(model, adapter_path)
merged_model = peft_model.merge_and_unload()
```

这样合并发生在高精度权重上，天然避开了"在 4-bit 状态下合并"的类型/精度问题。

> **版本提示**：不同 PEFT / bitsandbytes / Transformers 版本对"量化模型能否直接调用 `merge_and_unload`"的支持不同——部分版本会直接报错或给出警告，部分较新版本对某些情形有一定支持但仍可能有精度损失。是否可以在量化状态下直接合并，请以你所装版本的 `merge_and_unload` 文档与实际运行结果为准。

### 常见问题排查表

| 问题 | 常见原因（可能不止一种） | 排查方向 |
| --- | --- | --- |
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到 PeftModel 再调用 |
| 合并后模型变大 | 正常现象：合并后是完整模型，不再是小体积 Adapter | 无需处理；若在意体积可保留 Adapter 不合并 |
| 显存不足 | 一次性加载高精度完整模型占用高；也可能是 dtype 用了 FP32、未做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只保存了模型未保存 tokenizer；或特殊 token / generation_config 未一并导出 | 重新 `save_pretrained` 并按 §七核查 tokenizer、generation_config、config |

## 块 66

## 路线A：合并后部署完整模型

假设你已经用 LoRA 微调好了一个模型，现在要把它部署上线。**路线A**的思路很直接：先把 LoRA 权重合并回基础模型，得到一个“完整模型”，然后让推理框架把它当成普通模型来加载。

**操作流程三步走：**

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

**这条路线的优点：**

- **部署链路简单**——推理框架完全不需要知道 LoRA 是什么，按普通模型加载就行，省心；
- **推理时没有 Adapter 切换成本**——因为 Adapter 已经“焊死”在模型里了，运行时不需要额外操作。

**代价（需要权衡的点）：**

- 每个 Adapter 都要保存和分发一份**完整模型**。如果你有多个任务、多个 Adapter，那就得存多份完整模型，**存储成本会比较高**。

---

## 路线B：保留 Adapter，用支持 LoRA 的推理框架动态服务

路线B的思路相反：**不合并**，保留 LoRA Adapter 的小体积文件，让推理框架在运行时动态挂载。

一些推理框架（如 **vLLM**）支持在**同一份基础模型**上运行时加载/切换**多个 LoRA Adapter**（Multi-LoRA serving）。

**工作方式：**

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

**这条路线的优点：**

- 多任务/多租户场景下，**只需一份基础模型 + 多个小体积 Adapter**，大幅节省显存与存储；
- 支持**热切换**——不用为每个任务各存一份完整模型，随时切换。

**代价：**

- 依赖框架对 LoRA 的支持与配置，链路比“直接加载合并模型”略复杂；
- 不同框架/版本对 LoRA 的支持范围和限制不同，需要额外学习配置。

---

## vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

以 vLLM 为例，启动 OpenAI 兼容服务时，通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，请求里再用 Adapter 名字指定用哪个。命令行大致长这样：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
    --max-loras 2 \
    --max-lora-rank 16
```

**参数含义：**

- `--enable-lora`：开启多 LoRA 服务；
- `--lora-modules 名字=路径`：声明可用 Adapter——注意，这里的**路径就是训练保存下来的 Adapter 目录**（含 `adapter_model.safetensors` /

## 块 67

## 十四、与部署阶段的衔接：合并部署 vs 保留 Adapter 动态服务

### 14.3 如何选择

你已经知道 LoRA 训练完可以合并权重，但**合并不是唯一出路**。怎么选？看你的实际场景：

**场景一：只训练了一个 Adapter，追求部署最简单**
→ 选**合并**（路线A）。一个文件搞定，部署链路最短。

**场景二：同一个基础模型要服务多个 Adapter，且推理框架原生支持 LoRA**
→ 选**保留 Adapter 动态服务**（路线B）。基础模型只加载一次，多个 Adapter 按需切换，省显存、更灵活。

**场景三：QLoRA 精度敏感**
→ 无论走哪条路线，**上线前务必做一致性验证**（按 §11.3 的方法，合并前后对比输出）。

> ⚠️ 注意：以上是经验参考，不是硬性规则，以实际部署环境为准。具体框架用法与限制，第 15 板块（vLLM 部署）会展开。

---

## 十五、总结

本章你

## 测评题目

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB12-ST1202-003
在其他条件相同时，增大 LoRA rank 最可能带来什么？

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。
