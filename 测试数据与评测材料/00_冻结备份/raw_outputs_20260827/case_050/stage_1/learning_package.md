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

以 7B 参数模型为例，就是 **70 亿个参数**。训练时，你需要同时保存三样东西：

- **参数**本身
- **梯度**（用于反向传播）
- **优化器状态**（如 Adam 的动量）

这三样加起来，显存需求非常高。结果就是：**普通消费级 GPU 根本跑不动**。

---

**动手验证一下**：打开你的 GPU 监控工具（如 `nvidia-smi`），看看当前显存占用。再想想：如果 7B 模型光参数就要占几十 GB，你的显卡能装下吗？这个直观感受，就是全参数微调最大的痛点。

---

**小结**：全参数微调的问题是“什么都更新、什么都保存”，导致显存爆表。接下来要学的 LoRA / QLoRA，正是为了解决这个问题而生——只更新一小部分参数，大幅降低显存需求。

## 块 2

**本块学习检查**：用自己的话解释“PEFT只训练少量新增参数”与全参数微调在显存需求上的区别。

---

**1.2 参数高效微调（PEFT）**

PEFT，即参数高效微调，核心思想是**不修改整个大模型，只训练少量新增参数**。

想象一下，原始大模型是固定的底座，我们只在上面加一个“LoRA适配器”。训练时，**只更新LoRA参数**，完成后保存这个规模很小的适配器即可。

对比三种方式的资源消耗：

| 方式 | 训练参数 | 显存需求 |
|------|---------|---------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数

## 块 3

## 2.1 LoRA是什么？

你现在的目标是学会用 LoRA 做微调，而且喜欢动手实践。咱们直接切入核心：**LoRA = Low-Rank Adaptation（低秩适配）**，它解决的是“微调大模型太贵”的问题。

**传统微调**：要更新整个原模型权重 `W`，参数量巨大，你的显存根本扛不住。

**LoRA 的做法**：原模型 `W` **冻结不动**，在旁边新增一个更新量 `ΔW`，最终效果等价于 `W + ΔW`。这个 `ΔW` 由两个小矩阵相乘并乘以缩放系数得到：

```
ΔW = (α/r) × B × A
```

- 原权重 `W`

## 块 4

### 3.1 LoRA的问题与QLoRA解决方案

你已经知道LoRA能大幅减少训练参数，但这里有个关键点：**基础模型本身仍然要完整加载到显存里**。

以7B模型为例，如果用FP16精度加载，大约需要 **14GB显存**。也就是说，即使LoRA只训练少量参数，你也得先有能装下整个大模型的显存，才能开始微调。这对个人电脑或小显卡来说，是个不小的门槛。

---

### 3.2 QLoRA解决方案

QLoRA（Quantized LoRA）就是来解决这个问题的——它是**量化版的LoRA**。

核心思路很简单：

```text
FP16模型 → 4bit量化 → 减少显存 → 加载LoRA训练
```

结构上，它把基础模型压缩成4bit精度，再在上面挂LoRA参数进行训练：

```text
4bit基础模型 + LoRA参数 → 训练
```

这样，显存占用大幅下降，你就有可能在更小的显卡上跑起大模型微调了。

---

### 3.3 QLoRA依赖bitsandbytes

要跑QLoRA，需要装这几个库：

- `transformers`
- `peft`
- `bitsandbytes`
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中，`bitsandbytes` 负责三件事：

- 低精度量化
- 4bit加载
- 8bit加载

> **注意**：本教程的实训和验证统一以 **NVIDIA CUDA GPU** 为准。这是项目支持范围，也是当前资料最全、验证最充分的路径。但“QLoRA只能用NVIDIA”并不是技术必然——bitsandbytes 近年已在扩展对 CPU、

## 块 5

## 电脑配置批注：你的 LoRA / QLoRA 实训环境

目标很明确：跑通 LoRA 微调。下面直接告诉你需要什么硬件，以及为什么。

### LoRA 最低配置（先看这个）

- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐 NVIDIA GPU
- **显存**：最低 8GB，推荐 12GB+（以7B模型、短序列、小batch为例）

> 动手提示：如果你手头只有 8GB 显存，也能跑，但建议用短序列、小 batch，别贪大。

### QLoRA 要求（显存更省，但别误解）

QLoRA 的核心优势是**在更小显存上微调大模型**——通过 4-bit 量化基础模型以降低显存占用，但冻结原始权重，仅训练 LoRA 低秩增量 ΔW=(α/r)BA。注意：**不是没有下限**。

**7B 级别模型的参考值（经验值，非硬保证）：**

- 参考下限：约 8~12GB（短序列、小 batch）
- 推荐：16GB+

**为什么是“参考”而不是“门槛”？** 实际显存占用取决于多个测试条件：

- **模型规模**：7B 级别；13B/70B 显存需求显著上升
- **序列长度**：短序列（≤512）占用低；越长激活值显存增长越明显
- **batch 大小**：小 batch（1~2）占用

## 块 6

## 5.1 Python版本检查

在动手安装任何库之前，先确认你的Python版本。打开终端，运行：

```bash
python --version
```

**建议版本：Python 3.10+**

如果版本低于3.10，建议先升级Python，否则后续的PEFT和bitsandbytes可能会遇到兼容性问题。

---

## 5.2 PyTorch检查（关键一步）

**为什么重要？** LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的CUDA量化路径要求安装**与本机CUDA匹配的GPU版PyTorch**。

**常见坑**：直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

**正确做法**：以[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)为准。例如NVIDIA CUDA 12.1，建议指定兼容版本：

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**验证GPU是否可用**，运行：

```python
try:
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)   # 查看torch编译使用的CUDA版本
except Exception as e:
    print('PyTorch或CUDA检查失败:', e)
```

正常输出：

```text
True
```

表示PyTorch能看到CUDA GPU。

> **注意**：`torch.cuda.is_available()`返回`True`只说明PyTorch侧能用GPU，**不足以证明bitsandbytes的4-bit量化路径可用**。后者还依赖bitsandbytes是否正确编译/匹配到本机CUDA。真正的4-bit验证留到后续章节。

---

## 5.3 查看GPU信息

终端运行：

```bash
nvidia-smi
```

输出包含：

- GPU型号
- CUDA版本
- 显存大小

例如：

```text
RTX 4090
CUDA 12.x
Memory 24GB
```

**显存影响**：显存大小直接影响能否加载大模型。QLoRA 4-bit量化可降低显存需求，但仍需至少8GB显存才能运行7B级模型。

---

## 六、安装PEFT库（重点）

### 6.1 PEFT是什么？

PEFT是Hugging Face官方参数高效微调库，支持：

- LoRA
- Prefix Tuning
- Prompt Tuning
- 

> **模型许可证**：本部署涉及的模型

## 块 7

## 7.1 bitsandbytes 是干什么的？

简单说，**bitsandbytes** 是一个帮你“压缩”模型的工具库。它主要干三件事：

- **8bit量化**：把模型权重从FP16/BF16量化到8位，显存占用直接缩水。
- **4bit量化**：进一步压到4位，显存占用更小。
- **QLoRA训练**：在4bit量化基础上跑LoRA微调，这就是你要做的事。

**为什么你必须要装？** 因为本教程的QLoRA方案就是基于bitsandbytes实现的。它的核心思路是：**先用4bit量化“瘦身”基础模型，再挂上LoRA做微调**。这样你的显

## 块 8

**动手前先搞懂：为什么必须查CUDA？**

bitsandbytes是QLoRA量化的核心工具，它依赖CUDA来调用GPU计算。如果CUDA版本不匹配，后果很直接：CUDA不可用、量化失败、加载模型报错——每一步都会卡住你的LoRA微调。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

**两步检查，快速排雷：**

1. **看CUDA版本**：终端运行`nvidia-smi`，在输出里找到`CUDA Version`一行，记下版本号。
2. **验证bits

## 块 9

你是动手实践型学习者，目标是用LoRA微调模型。在Windows上准备环境时，请优先走这条路：

**推荐路径：Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

为什么推荐WSL2？因为Linux环境兼容性更好，能少踩很多坑。

如果你坚持用Windows原生环境，可能会遇到**问题1：bitsandbytes无法加载CUDA动态库**。这通常不是单一原因，常见的有：

- 你装的`bitsandbytes`版本和本机CUDA或GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本支持不如Linux完善（不同版本差异较大）。

所以，这不是简单的“Windows不支持”，而是**PyTorch、bitsandbytes版本与本机CUDA/硬件**共同影响的结果。建议你优先用WSL2获得更接近Linux

## 块 10

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：CUDA版本不匹配**

这是Windows上最常见的坑。先运行下面命令检查：

```bash
nvidia-smi
```

确认三件事：

- **NVIDIA驱动**：是否已安装并能识别GPU；
- **CUDA版本**：驱动支持的CUDA版本号；
- **PyTorch版本**：是否与CUDA匹配。

如果不匹配，后续`import torch`或QLoRA 4-bit会直接失败。

---

## 十、安装完整LoRA环境

如果你已有基础环境：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**前置**：确保已安装**匹配本机CUDA的GPU版PyTorch**。CUDA 12.1示例：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

然后增加：

```bash
pip install peft bitsandbytes
```

一步到位完整安装：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

执行：

```bash
accelerate config
```

按提示选择，单GPU示例：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

生成`default_config.yaml`。

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

成功说明：✅ PyTorch正常 ✅ PEFT正常 ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes及量化配置：

```python
import torch
import bitsandbytes
from transformers import BitsAndBytesConfig

print(torch.cuda.is_available())
print(bitsandbytes.__version__)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
print(config)
```

输出`True`说明GPU环境正常；量化配置使用`float16`计算精度，确保与CUDA版本匹配。

---

**本块学习检查**：运行`nvidia-smi`，记录你的CUDA版本；再运行`python -c "import torch; print(torch.__version__)"`，确认两者匹配。不匹配则按第十节命令重装PyTorch。

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：Windows报错怎么办？**

在Windows上运行bitsandbytes时，你可能会遇到兼容性问题导致的报错。这是因为bitsandbytes对Windows的原生支持不够完善，容易因CUDA版本或系统环境不匹配而加载失败。

**解决办法：使用WSL2**

WSL2（Windows Subsystem for Linux 2）是Windows下的Linux子系统，能提供更接近Linux的环境，从而规避大部分兼容性问题。这是官方推荐的解决方案。

**常见问题速查表：**

| 问题 | 原因 | 解决 |
| --- | --- | --- |
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版 | 重装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch或用更小模型 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | 量化推理开销 | 使用vLLM等推理加速框架或关闭量化 |

---

**本块学习检查**：对照上表，检查

## 块 12

## 一、为什么需要LoRA？——先动手感受问题

你是个喜欢动手的人，那我们直接进入场景。假设你拿到一个7B模型（比如LLaMA、Qwen、Baichuan、Mistral），想让它学会某个新任务。传统做法是**全参数微调**：把预训练模型的所有参数都更新一遍，然后保存一个新模型。

问题来了——7B意味着**70亿个参数**。全参数微调时，你不仅要存模型参数本身（记作 W），还要额外存：

- **梯度**（∂W）：更新参数时算出来的变化量；
- **优化器状态**（比如Adam的动量项）。

这三样东西加起来，**实际显存需求远大于模型本身的大小**。换句话说，你的GPU可能连训练都跑不起来，更别提训练完还要存一份完整的新模型。这就是LoRA要解决的核心痛点。

---

## 二、LoRA的核心思路：不更新全部参数

LoRA的想法很直接：**微调时，我们不一定需要动所有参数**。

它假设：模型在预训练时已经学到了大量通用知识，微调只是在这个基础上做“小修小补”。这个“小修小补”的改动量，可以用一个**低秩矩阵**来表示。

具体来说，传统微调是直接更新权重矩阵 W，得到 W_new = W + ΔW。而LoRA的做法是：**冻结原始权重 W 不动**，只额外训练两个低秩矩阵 B 和 A，用它们的乘积来构造增量更新：

> **ΔW = (α/r) × B × A**

这里的 B

## 块 13

### 2.1 不直接修改原模型参数

**传统微调是怎么做的？**

想象你有一个训练好的模型，它的参数叫 `W`。传统微调直接把这些参数改掉：

```text
W → W'
```

也就是说，**直接修改**了原来的 `W`。

**LoRA 的做法完全不同**

LoRA 把原来的 `W` **冻结**住，一点不动。然后，在旁边**新增**一个参数 `ΔW`。最终模型用的参数是：

```text
W' = W + ΔW
```

- `W`：原模型参数（冻结，不训练）
- `ΔW`：LoRA 新增的参数（只训练这个）

**一句话记住**：LoRA 不碰原模型，只额外学一个“补丁”加上去。

---

### 2.2 为什么只训练 ΔW？

大模型已经通过海量数据学会了语言规律、世界知识、推理能力。微调通常只是**对已有能力做方向调整**，所以不必重新学全部参数，只学**任务相关的变化量**就够了。

## 块 14

**本块学习检查**：动手算一算——假设 `Wq` 是 `4096 × 4096`，它的参数数量是多少？再想想，如果直接学习 `ΔW`，需要学多少个参数？

---

### 3.1 矩阵基础概念

Transformer 里大量参数都是**矩阵**。比如注意力层中，`Q = X · Wq`，这里的 `Wq` 就是一个权重矩阵。

假设 `Wq` 的形状是 `4096 × 4096`，它的参数数量就是：

`4096 × 4096 ≈ 1678万`

也就是说，这一个矩阵就有约 1678 万个参数。

---

### 3.2 直接训练矩阵的问题

如果我们要微调模型，直接修改 `ΔW`，那 `ΔW` 也需要 `4096 × 4096` 个参数

## 块 15

**本块学习检查**：动手算一遍——把 W 从 4096×4096 换成 1024×1024，r=4，写出 A、B 的尺寸和总参数，对比完整矩阵参数。

---

**4.1 矩阵尺寸变化**

原矩阵 W 是 **1024×1024**。直接训练，参数就是 1024×1024 = **1048576**。

用 LoRA，设 **rank r=4**，把 W 拆成两个小矩阵：

- A：**4×1024**，参数 4096
- B：**1024×4**，参数 4096

总参数：**8192**。

对比一下：

## 块 16

**5.1 神经网络参数变化具有低维特征**

LoRA的核心洞察是：模型适应新任务时，参数变化（ΔW）**不需要覆盖所有方向**，有效变化集中在**低维子空间**。

想象一个4096维空间，真正影响任务的“有效方向”可能远少于满维度，只集中在少数方向上。这就是“权重更新具有较低的内在维度”这一假设。

**注意**：具体“有效维度”没有通用固定值，会随模型、任务、数据变化，这里只做定性理解，别记成具体数字。

因此，低秩矩阵BA可以近似ΔW，这就是LoRA高效的原因。

---

**本块学习检查**：用一句话解释“为什么ΔW可以被低秩矩阵近似？”（提示：关键在“低维子空间”

## 块 17

## 类比理解：为什么LoRA用少量参数就能微调大模型？

想象一张高清图片，尺寸是 **4000×4000像素**。如果要把这张图完整保存下来，需要存储 **1600万个数据**（4000×4000=1600万）。

但问题是——这张图的主要变化，其实只是**亮度**、**对比度**，加上**少量结构变化**。这些变化完全可以用**少量参数**来描述，不需要把1600万个像素全部重存一遍。

LoRA的核心思想正是如此：

> **用少量参数（低秩矩阵A和B）来近似模型权重的变化量ΔW。**

具体来说，LoRA冻结原始权重W，仅训练低秩矩阵A和B，通过增量ΔW=(α/r)BA来更新模型，而不是直接描述变化方向。

---

## LoRA在Transformer中的位置

先回顾Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中**Attention**部分包含四个投影矩阵：

```text
Q（查询）、K（键）、V（值）、O（输出）
```

在LoRA的早期实践中，**常见**的做法是把LoRA加到：

- **Q投影**
- **V投影**

代码示例：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这表示：只训练 `Wq` 的变化和 `Wv` 的变化。

> ⚠️ **注意**：只挑 `q_proj`/`v_proj` 是LoRA原论文里的**常见选择**，**不是固定标准**。现代实践中，也经常把LoRA加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定。

---

## 一个更完整的配置示例

下面给出一个覆盖更完整线性层的配置示例（以Llama类模型常见的模块名为例；不同模型模块名可能不同，请先用 `print(model)` 或 `model.named_modules()` 确认实际名称）：

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
        "q_proj", "k_proj", "v_proj", "o

## 块 18

## 8.1 rank（r）——LoRA的“内存旋钮”

先看代码里最直观的一行：

```python
r=8
```

这个 `r` 代表**低秩矩阵的维度**。你可以把它理解为“LoRA 内部用来模拟权重更新的小矩阵有多宽”。

### 调大 `r` 的代价与收益

**调大 `r`：**
- ✅ 表达能力更强（模型能学更复杂的模式）
- ❌ 参数更多、显存占用增加

**调小 `r`：**
- ✅ 更省显存、Adapter 更小
- ❌ 表达能力受限

### 经验起点（不是规则，是搜索起点）

| 任务类型 | rank 起点参考 |
|---------|-------------|
| 简单任务 | 约 4–8 |
| 普通微调 | 约 8–32 |
| 复杂任务 | 约 64+ |

注意：合适的 `r` 和模型规模、目标模块、数据量、任务难度都有关，**别背表，要实测**。

### 动手选 `r` 的 5 步流程（比查表可靠）

1. **划验证集**：从训练数据中留出 8:2 或 9:1 的验证集，后续所有比较都在它上面看指标。
2. **固定其它变量**：`target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数全部锁死，**只改 `r`**。
3. **小范围网格搜索**：从 `r ∈

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

`lora_alpha` 是一个缩放系数，直接控制 LoRA 参数的**更新强度**。代码中常这样设置：

```python
lora_alpha=16
```

它的核心用法是与秩 `r` 配合，形成缩放比例：

```text
alpha / r
```

这个比例决定了 LoRA 低秩矩阵对原始权重的影响幅度。比例越大，更新越激进；比例越小，更新越保守。

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

`lora_dropout` 在训练时随机丢弃一部分 LoRA 参数，防止模型对训练数据过度拟合。0.05 表示 5% 的丢弃率，是常见设置。

---

## 九、LoRA 与全参数微调的区别

| 项目 | 全参数微调 | LoRA（相对而言） |
|------|-----------|-----------------|
| 训练参数 | 全部 | 通常少很多 |
| 显存 | 高 | 通常更低 |
| 速度 | 慢 | 通常更快 |
| 保存文件 | GB级 | 通常较小（常见 MB 级） |
| 部署 | 复制模型 | 加载 Adapter |

> **注意**：上表是**相对趋势**，不是绝对数值。LoRA 的可训练参数量、显存、训练速度和 Adapter 文件大小会随 `target_modules` 选择、秩 `r`、模型层数与 `dtype` 变化——

## 块 20

### 2.1 Rank表示什么？

Rank，中文叫“矩阵的秩”。在LoRA里，它决定了一个关键数字：**低秩矩阵的中间维度大小**。

你可以把它理解为“压缩的力度”。我们看个具体例子：

假设原始权重矩阵 `W` 是 `4096 × 4096` 的。如果设置 `r=8`，那么LoRA会创建两个小矩阵：

- 矩阵 `A`：形状是 `4096 × 8`
- 矩阵 `B`：形状是 `8 × 4096`

这两个小矩阵相乘 `B × A`，结果正好恢复成 `4096 × 4096` 的形状，也就是 `ΔW`。实际计算中还需乘以缩放因子 `α/r`，即 `ΔW = (α/r) × B × A`，其中 `α` 为缩放系数。

**直观理解**：`r`

## 块 21

**Rank控制什么？**

Rank决定LoRA能学多少种“变化方向”。

**低rank（如r=4）**：变化模式少。好处是参数少、显存低、训练快；缺点是表达能力有限。

**高rank（如r=64）**：自由度更高。能学更复杂任务、适应力更强；但参数多、显存增、训练成本高。

**对参数量的影响**：假设某层是4096×4096，r越小，新增参数越少；r越大，新增参数越多。

**本块学习检查**：对比r=4和r=64，哪个更适合简单任务？为什么？

## 块 22

**动手算一算：不用LoRA要训练多少参数？**

假设模型权重矩阵是 **4096 × 4096**，直接全量训练：

- 参数量 = 4096 × 4096 = **16,777,216**（约 **1677万**）

**对比一下用LoRA：**

LoRA参数量公式：`r × (输入维度 + 输出维度)`

- **r=8**：8 × (4096 + 4096) = **65,536**（约6.5万）
- **r=64**：64 × 8192 = **524,288**（约52万）

**关键结论：**

| 方式 | 参数量 |
|------|--------|
|

## 块 23

## 4.1 Rank太小的问题：你的LoRA模型可能“学不动”

想象一下，你让一个实习生去负责一个超大项目，但只给了他一张便利贴大小的笔记本。`r=2` 就是这个便利贴——**表达能力有限**，能记录的信息太有限。

**具体表现是什么？**

- **欠拟合**：模型可能根本没“吃透”你的新任务，就像实习生只记了两条要点，完全没法干活。
- **新任务知识无法充分表达**：你想让模型学会的新风格、新知识，它“想说但说不出来”。
- **训练loss下降慢**：训练时损失值降得很慢，像蜗牛爬。
- **效果提升有限**：微调完，模型表现和原来差不多，白忙一场。

**怎么判断你遇到了这个问题？** 看训练曲线——如果loss半天降不下去，先别急着调学习率，检查一下你的`r`是不是设太小了。注意，低秩秩的选择需根据任务复杂度调整，`r=2`可能不足以捕捉复杂任务的知识，但并非绝对。

---

## 4.2 Rank太大的问题：你的LoRA“不省钱了”

反过来，如果你把`r`设成`16`，比如：

```

## 块 24

## 4.3 常见rank选择：从哪开始？

你不需要一开始就纠结“完美rank”，先记住一个经验起点表（注意：这只是**起点**，不是自动调参标准）：

| 任务类型 | rank（起点参考） |
|------|------|
| 简单分类 | 约 4~8 |
| 文本风格调整 | 约 8~16 |
| 指令微调 | 约 16~64 |
| 复杂领域任务 | 约 64以上 |

**怎么用这张表？** 比如你要做LoRA微调，先按任务类型选一个起点rank，跑通流程，再根据实际效果调整。

**但千万别照搬。** 同一个`r`值，在不同模型规模、序列长度、batch大小、目标模块下，显存占用和效果可能差别很大。实际项目中，你需要结合以下因素综合调整：

- 数据量
- 模型大小
- 序列长度
- batch大小（含梯度累积后的有效batch）
- 目标模块（`target_modules`）的选择与数量
- GPU显存
- 验证集/基准任务上的表现

**动手建议**：先选

## 块 25

**本块学习检查**：请用你手头的 LoRA 配置（或下面例子）算一下 `alpha/r` 的值，并判断这个值会让 LoRA 更新被放大、保持还是减弱。

---

### 6.1 控制LoRA影响强度

`alpha` 和 `rank` 的比值 `alpha/r` 是一个**缩放系数**，它决定 LoRA 更新被放大还是减弱。

**例子1：放大**

```python
r=8
alpha=16
```

缩放：`16/8 = 2`，表示 LoRA 更新被**放大2倍**。

**例子2：保持**

```python
r=8
alpha=8
```

缩放：`8/8 = 1`，表示保持**原始比例**。

**例子3：减弱**

```python
r=8
alpha=4
```

缩放：`4/8 = 0.5`，表示 LoRA 影响**减弱**。

---

### 七、Alpha与Rank的关系（重点）

LoRA 加入模型的更新量会被 `alpha/r` 缩放，所以 `alpha/r` 是影响 LoRA 更新“强度”的一个**关键缩放系数**。

**注意**：`alpha/r` 只是缩放系数，**不是**唯一决定实际强度的开关。实际落到权重上的更新量还取决于：

- **学习率**：直接决定每步对 `A`、`B` 的更新步长。学习率与 `alpha/r` 会叠加影响有效更新幅度——例如把 `alpha/r` 调大又不下调学习率，容易一起放大更新导致震荡；很多实践中甚至

## 块 26

**动手看：Rank变了，缩放没变**

先看两组配置，感受一下规律：

- 配置1：`r=8`，`alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16`，`alpha=32` → 比例 = 32 ÷ 16 = **2**

发现了吗？虽然rank从8变成16，但`alpha/r`始终是2。这就是“缩放比例相同”的含义。

**实操建议**

调整rank时，**同步调整alpha**，让`α/r`大致稳定。比如上面`alpha = 2 × r`就是常见起点。

**注意**：这只是经验做法，不是硬性规则。你可以保持alpha固定、只

## 块 27

## 方案1：小模型实验（适合你现在的起点）

既然你的目标是快速上手LoRA微调，而且偏好动手实践，那我们先从**方案1**开始：

```python
r=8
alpha=16
```

**为什么先选这个？** 三个字：**少、快、试**。

- **参数少**——训练时额外增加的参数不多，内存压力小；
- **训练快**——跑一轮实验不用等太久，适合你每天1小时的节奏；
- **适合测试**——先验证流程能不能跑通，再谈效果优化。

**动手建议**：先用这个配置跑通一个最小实验，确认代码和流程没问题，再考虑调大参数。

---

## 实际代码怎么配？

看下面这段PEFT配置，注意三个关键参数：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

## 块 28

**误区1：rank越大越好？**

很多初学者会直觉地认为：rank设得越大，模型能学的东西就越多，效果肯定越好。**这个想法是错的。**

原因很简单：如果你的任务本身不复杂，把rank设得很大，并不会带来额外的收益，反而会带来两个实际问题：

- **浪费显存**：更大的rank意味着更多的可训练参数，会占用更多的GPU内存。
- **增加训练成本**：参数多了，计算量自然就上去了，训练时间会变长。

所以，rank的选择要匹配任务的复杂度，不是越大越好。

---

**误区2：只调rank，不调alpha**

这是另一个常见操作。比如你原来用的是 `r=8, alpha=16`，现在想增强模型能力，直接把rank改成64：

```python
r=64
alpha=16
```

这样改完，`α/r` 的比例就变成了 `16/64 = 0.25`。这个缩放比例变小了，意味着LoRA对原模型权重的**实际影响反而被削弱了**。你虽然增大了容量，但更新强度没跟上，效果可能不升反降。

**记住**：调整rank时，记得同步检查alpha，保持合适的 `α/r` 比例。

---

**误区3：alpha越大越好**

这个想法同样是错的。alpha的作用是通过 `α/r` 这个缩放因子来控制LoRA的更新强度。如果alpha设得过大，导致 `α/r` 过大，可能会带来：

- **更新过强**：对原模型权重的改动幅度太大，可能破坏预训练学到的知识。
- **训练不稳定**：损失曲线可能会震荡甚至发散。

> **特别提醒**：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一原因。训练震荡或发散，更常见的原因还包括学习率过高、数据质量差、batch size过小、精度设置（fp16/bf16）问题、缺少梯度裁剪等。排查问题时，要综合考虑这些因素，不要只盯着alpha。

---

**动手验证一下**

为了帮你巩固这三点，可以做一个快速自查：

1.

## 块 29

## 一、LoraConfig是什么？

PEFT（Parameter-Efficient Fine-Tuning）是 Hugging Face 提供的参数高效微调库。在 LoRA 训练流程中，加载基础模型后，下一步就是创建 `LoraConfig`，它负责告诉 PEFT 四件事：

- **LoRA 放在哪里** → 由 `target_modules` 指定
- **LoRA 规模多大** → 由 `r` 控制
- **LoRA 训练强度如何** → 由 `lora_alpha` 调节
- **是否使用 dropout** → 由 `lora_dropout` 决定

你可以把 `LoraConfig` 理解成一张“配置单”，PEFT 拿到这张单子后，才知道该在模型的哪些位置插入 LoRA 适配器，以及用多大的容量去训练。

---

### 快速上手：先跑通一个最小配置

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,                # LoRA 的秩，控制容量
    lora_alpha=16,      # 更新强度缩放
    target_modules=["q_proj", "v_proj"],  # 插入位置
    lora_dropout=0.1    # 防过拟合
)
```

这段代码创建了一个最基础的 LoRA 配置。后续章节会逐一拆解每个参数的含义和选择思路，但你现在已经能读懂大多数 LoRA 微调代码里的配置部分了。

---

### 本块学习检查

**任务**：对照上面的最小配置，

## 块 30

## 1.2 一个简单示例——动手跑通第一个LoRA配置

你是初学者，目标是LoRA微调，喜欢动手实践。下面这段代码就是LoRA最核心的配置，直接复制运行即可：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**先别急着往下读，花1分钟观察这段代码**：它创建了一个`LoraConfig`对象，传入了4个参数。这4个参数就是LoRA的全部核心配置，记住它们的作用：

| 参数 | 作用 |
|---|---|
| `target_modules` | 决定插入哪些层 |
| `r` | 决定LoRA矩阵大小 |
| `lora_alpha` | 控制缩放强度 |
| `lora_dropout` | 控制随机失活 |

---

## 二、LoRA配置整体流程——配置怎么用起来？

光有配置还不够，你需要把它“接”到模型上。完整流程如下（**流程示意**，不是可运行代码）：

```
model → LoraConfig → get_peft_model() → LoRA模型
```

实际代码：

```python
from peft import LoraConfig, get_peft_model

config =

## 块 31

好的，我们来把这段关于`target_modules`的知识，用最适合你的方式重新梳理一遍。既然你喜欢动手实践，我们就直接切入正题，看看这个参数到底在干什么。

## 3.1 target_modules：告诉LoRA该改哪里

简单来说，`target_modules` 这个参数，就是用来**指定LoRA要注入到模型的哪些模块里**。

我们可以把大语言模型想象成一个复杂的机器，它内部有很多处理信息的“车间”。在Transformer架构中，最核心的“车间”就是**注意力机制（Attention）**。这个车间里有几个关键的“工位”，它们各有分工：

```text
Attention
├── q_proj   （Query投影层）
├── k_proj   （Key投影层）
├── v_proj   （Value投影层）
└── o_proj   （Output投影层）
```

LoRA微调不是要把所有“工位”都改造一遍，那样成本太高。相反，我们可以选择性地只对其中几个“工位”进行升级。比如，我们只选择改造`q_proj`和`v_proj`这两个工位：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的含义就是：**只给Query投影层和Value投影层增加LoRA**。

## 为什么常常选 q_proj 和 v_proj？

你可能会问，为什么大家经常选择这两个“工位”呢？这要从注意力机制的计算公式说起：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

在这个公式里，`Q`（Query）的作用是决定模型应该**“关注什么信息”**，而`V`（Value）的作用是决定模型最终**“传递什么信息”**。所以，通过调整`Q`和`V`，我们通常能非常有效地改变模型的行为和输出。

**不过，这里有一个重要的提醒：** `q_proj`和`v_proj`只是一个**常见的起点**，并不是所有模型都通用的“标准配置”。不同模型的模块命名可能不一样，所以在你动手配置之前，**务必先查看一下你所用模型的实际模块名**（可以用`model.named_modules()`来查看），然后再决定`target_modules`里写什么。

## 更省事的办法：`"all-linear"`

如果你用的是较新版本的PEFT（一般PEFT ≥ 0.7起支持），还有一个更省事的选择。你可以直接把`target_modules`设置为字符串`"all-linear"`，让PEFT自动帮你匹配模型中（几乎）所有的线性层，省去手工枚举的麻烦。

下面是一个可运行的示例：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained("your/base-model")

lora_config = LoraConfig(

## 块 32

**动手实验：从Q和V开始**

你正在学习LoRA微调，目标很明确。我们先从最轻量的方案入手。

**方案1：只调整Q和V**

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**特点：**
- **参数少**：只修改注意力机制中的查询（Q）和值（V）投影层。
- **训练快**：因为要更新的参数少，所以训练速度更快。
- **轻量起点**：这是常见的入门选择，适合快速验证想法。具体效果仍需结合你的模型结构和任务来观察。

**为什么先试这个？**
对于初学者，先用最少的参数跑通流程、观察效果，是最高效的学习路径。这能帮你快速建立对LoRA调参的直觉。

**本块学习检查**
请修改你现有的LoRA配置，将`target_modules`设置为只包含`"q_proj"`和`"v_proj"`，并运行

## 块 33

## LLaMA系列：最常用的LoRA注入目标

在LLaMA系列模型中，`target_modules`最常见的配置是：

```python
[
"q_proj",
"v_proj"
]
```

**为什么是这两个？** 注意力机制中有四个投影层：Q（查询）、K（键）、V（值）和O（输出）。实践中，只对Q和V做低秩适配是常见选择之一——既能有效微调模型行为，又比全量微调节省大量参数。不同选择的效果差异：适配Q和V是常见做法，而同时适配Q、K、V、O通常能获得更强的表达能力，但参数开销也相应增加。

**动手验证：** 你可以用`print(model)`查看LLaMA模型的层结构，找到`q_proj`和`v_proj`这两个`nn.Linear`层，确认它们的存在。

---

## 对比：GPT-2的`c_attn`与`fan_in_fan_out`

GPT-2与LLaMA不同，它的注意力层用的是`c_attn`——一个合并了Q/K/V三个投影的模块：

```python
[
"c_attn"
]
```

**关键陷阱——权重布局转置：** GPT-2的`c_attn`使用的是Transformers库中的`Conv1D`层，其权重布局与标准`nn.Linear`不同，需要设置`fan_in_fan_out=True`。具体来说，`Conv1D`在PEFT中的权重处理方式与`nn.Linear`存在差异，设置该参数后PEFT才能按正确布局注入低秩矩阵。

**可运行示例（GPT-2）：**

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
    fan_in_fan_out=True,         # Conv1D 权重布局与 nn.Linear 不同，需置 True

## 块 34

## 五、BERT 的 `target_modules` 配置

### 核心问题：不能照搬

BERT 的注意力结构与 LLaMA 不同，**不能直接复制**其他模型的 `target_modules` 配置。BERT 通常只对 **`query`** 和 **`value`** 两个投影矩阵做 LoRA 微调。

### 动手操作：找到你的模型模块名

**第一步**：打印模型所有模块名，确认实际命名：

```python
for name, _ in model.named_modules():
    print(name)
```

**第二步**：在输出中找到 `q_proj` 和 `v_proj`（BERT 的命名通常是这两个）。

**第三步**：配置 `target_modules`：

```python
target_modules=["query", "value"]
```

### 关键机制（务必理解）

- **后缀匹配**：`target_modules` 按模块名做**后缀匹配**。写 `"query"` 会命中所有以 `query` 结尾的层，各层各头都会覆盖到。
- **命名差异警告**：不同模型命名不同——GPT-2 用 `c_attn`（q/k/v 合并），LLaMA 用 `q_proj`/`v_proj`，**不能照搬**。
-

## 块 35

**6.1 r是什么？**

r就是rank，用来控制LoRA低秩矩阵的维度。核心公式是：

```text
ΔW = BA
```

r决定了A和B的中间维度大小。举个例子：原矩阵是4096×4096，如果设置r=8，那么A的维度是4096×8，B的维度是8×4096。

**6.2 r影响什么？**

r越大，学习能力越强，能表达更复杂的任务；但代价是参数和显存占用增加。常见起点参考（经验值，需按验证集调整）：

- 简单任务：r≈4-8
- 通用指令微调：r≈8-32
- 复杂领域任务：r≈32

## 块 36

**7.1 alpha作用：控制LoRA更新幅度**

LoRA最终更新不是`BA`，而是`(alpha/r)BA`。这里的`alpha`就是缩放因子，直接控制LoRA对模型的影响强度。

**动手算一算**：配置`r=8`、`lora_alpha=16`，缩放比例是`16/8=2`，表示LoRA影响被增强。

**7.2 alpha常见设置**

经验起点：`alpha = 2 × r`（注意：这只是常见起点，非通用规则，最终以验证集表现为准）。

示例：
- `r=8`，`alpha=16`
- `r=16`，`alpha=32`

**本块学习检查**：若`r=4`，`alpha=8`，缩放比例是多少？

## 块 37

**8.1 dropout作用**

训练时，LoRA会以一定概率随机将低秩矩阵A和B中的部分权重置零，目的是减少**过拟合**。

配置示例：
```python
lora_dropout=0.05
```
表示5%的LoRA路径随机失活。

---

**8.2 如何选择dropout？**

经验起点（非固定规则，需结合验证集是否过拟合来调整）：

| 数据规模 | dropout（起点参考） |
|---------|-------------------|
| 数据少   | 约 0.05~0.1       |
| 数据多   | 约 0~0.05         |

若训练数据非常丰富，可设：
```python
lora_d

## 块 38

## 9.1 基础配置：LoRA 微调的起步模板

这份配置是 LoRA 微调的经典起点，特别适合 **7B 规模模型**的测试场景。我们逐行拆解，让你明白每个参数在做什么。

### 核心代码

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,                # LoRA容量：低秩矩阵的秩，控制可训练参数量
    lora_alpha=16,      # 更新缩放：调节LoRA对原模型权重的影响强度
    target_modules=["q_proj", "v_proj"],  # 插入位置：只在这两个注意力层注入LoRA
    lora_dropout=0.05,  # 防止过拟合：训练时随机丢弃部分LoRA参数
    bias="none",        # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（如GPT类）
)
```

### 参数速查表

| 参数 | 值 | 作用 |
|------|-----|------|
| r | 8 | LoRA容量，秩越大可学习信息越多 |
| alpha | 16 | 更新缩放，控制LoRA对模型的影响程度 |
| target_modules |

## 块 39

## 九、完整LoraConfig案例（重点）—— 9.1 基础配置

你现在的目标是**LoRA微调**，而且喜欢动手实践，那我们就直接进入一个完整可运行的例子。这个例子会带你走通“配置 → 训练 → 保存 → 重载验证”的全流程，让你亲眼看到LoRA到底做了什么。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

base_name = "bert-base-uncased"

# 模型许可证：BERT 采用 Apache 2.0 许可证，可商用
# 注意：bert-base-uncased FP16 推理需要至少 2GB 显存，训练需根据 batch size 适当增加
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

这里用的是BERT做**序列分类**（比如情感二分类）。注意：分类头是随机初始化的，如果不训练它，模型输出就是乱的。

### 第二步：配置LoRA

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

逐项拆解：

- **`r=8`**：LoRA的低秩维度。越小参数越少，但表达能力也越弱。
- **`lora_alpha=16`**：缩放因子，控制LoRA更新的幅度。
- **`lora_dropout=0.05`**：防止过拟合的Dropout比例。
- **`target_modules=["query", "value"]`**：只对BERT注意力里的query和value矩阵做LoRA，这是最常用的选择。
- **`modules_to_save=["classifier"]`**：**这是本块的重点**。它让分类头也参与训练，并且随Adapter一起保存。

### 第三步：包装模型并查看可训练参数

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()       # 应看到 LoRA 参数 + classifier 一起可训练
```

这一步会打印出可训练参数的数量。你会看到只有LoRA参数和分类头是可训练的，BERT主体被冻结了——这就是“参数高效”的含义。

### 第四步：保存与重载验证

```python
# 保存：Adapter 权重 + modules_to_save 指定的分类头会一起被保存
model.save_pretrained("./lora_ckpt")

# 重新加载并验证分类头确实被带回（而不是随机初始化）
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

# 验证：重载后的分类头权重应与保存前一致
w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

### 为什么`modules_to_save`这么关键？

> 若不加它，重载时分类头会退回基础模型的**随机初始化**，导致分类结果错乱。

### LoRA的核心机制

LoRA并非对原始权重做分解，而是**冻结原权重W**，仅训练两个低秩矩阵A和B，通过`W + BA`形成增量更新。这样既保留了预训练知识，又大幅减少了可训练参数量。

## 块 40

## 10.1 普通文本生成：从默认值开始

你现在的目标是**LoRA微调**，而且喜欢动手实践——那我们就从最常用的配置入手。对于普通文本生成任务（比如续写、对话、故事生成），PEFT官方推荐的参数组合非常简洁：

```python
r=8
alpha=16
dropout=0.05
```

### 这三个参数怎么理解？

- **`r=8`**：这是LoRA的“秩”。可以把它理解为**低秩矩阵的维度**——`r`越大，能学习的参数越多，模型适应新任务的能力越强，但显存占用和过拟合风险也会上升。`r=8`是一个平衡的起点。
- **`alpha=16`**：这是缩放因子。实际生效的缩放比例是 **`alpha / r`**，也就是 `16 / 8 = 2`。这个比例控制LoRA更新对原始权重的影响强度。比例越大，微调时权重的变化越“激进”。
- **`dropout=0.05`**：这是LoRA层的随机失活率。训练时随机丢弃5%的LoRA神经元，防止过拟合。`0.05`是个保守但稳妥的默认值。

### 动手验证一下

如果你已经装了PEFT库，可以快速跑一下这段代码，观察配置对象长什么样：

```python
from peft import LoraConfig

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"  # 普通文本生成属于因果语言模型任务
)
print(config)
```

你会看到配置对象里包含了我们设置的参数值。**注意**：这里只创建配置，不加载模型，所以CPU和4GB内存就够用（见第十一章）。

### 一个容易踩的坑

如果你后面执行 `get_peft_model()` 加载真实大模型，**显存不够**时（比如只有8GB显存），优先检查是不是 `r` 设太大、目标模块选太多、序列太长或batch太大——`r` 只是

## 块 41

## 普通文本生成：LoRA配置实战

你是初学者，目标是用LoRA微调模型。咱们直接上手，看下面这个配置：

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

**四个参数，逐个击破：**

- **target_modules**：决定改哪里。这里只改`q_proj`和`v_proj`，即注意力层的查询和值投影。
- **r**：决定LoRA容量。`r=8`表示低秩矩阵的秩为8，容量适中。
- **lora_alpha**：通过`α/r`缩放

## 块 42

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

你已经知道，LoRA的核心思路是**冻结原模型、只训练一小部分额外参数**：

```text
原模型参数：冻结
LoRA参数：  训练
```

但这里有个关键点容易被忽略：**LoRA只是减少了“训练”的参数，并没有减少“加载”的模型**。

也就是说，无论LoRA多省，**基础模型本身仍然要完整放进显存里**。

我们来算一笔账。以7B模型为例：

- 7B = 70亿参数
- 如果用FP16精度存储，每个参数占2字节
- 显存需求 = 7B × 2 bytes ≈ **14GB**

这还只是**模型本身**的占用。实际训练时，还需要额外空间存放：

- **激活值**：前向传播的中间结果
- **梯度**：反向传播计算的更新方向
- **优化器状态**：如Adam的动量项

把这些都加起来，实际显存需求会**远高于14GB**。

所以结论很直接：**LoRA解决了“训练太多参数”的问题，但没解决“加载整个模型”的问题**。对于消费级GPU（通常8~16GB显

## 块 43

**QLoRA核心思想：让大模型微调更省显存**

你已经有LoRA基础，现在看QLoRA就很简单了——它就是在LoRA前面加了一步：**先把基础模型压缩成4-bit，再插入LoRA训练**。

**一句话对比：**
- LoRA：`FP16模型 + LoRA`
- QLoRA：`4-bit模型 + LoRA`

**训练流程（记住这个顺序）：**
```
FP16基础模型 → 4-bit量化 → 加载量化模型 → 插入LoRA → 训练LoRA参数
```

**关键区别（考试重点）：**
- **量化的是谁？** 基础模型的权重（4-bit）
- **谁保持高精度？** LoRA参数（FP16

## 块 44

## 3.1 模型存储方式：从“重量”看量化

先记住这张表，它是一切的基础：

| 类型 | 每个参数占用 |
|------|------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| INT4 | 0.5 Byte |

**动手算一下**：7B模型（70亿参数）用FP16存，需要 `7B × 2 = 14GB`；换成4-bit，只需 `7B × 0.5 = 3.5GB`。权重存储显存直接省约75%，实际总显存节省比例因模型和训练配置而异。

> ⚠️ 注意：3.5GB只是**理论下界**（理想权重估算）。实际占用更高，因为还要存量化常数/缩放因子、分组元数据（NF4按组量化，每组都要额外参数）、未量化的

## 块 45

### 4.1 NF4是什么？

NF4（NormalFloat4，正态分布4位浮点）是QLoRA论文为神经网络权重设计的4-bit量化数据类型。

**关键理解**：它不是IEEE 754那种通用浮点格式，而是一套**针对近似正态分布权重的离散码本**。

**工作原理**：
- 预先确定16个（2⁴）量化取值点，按标准正态分布分位数排布
- 权重密集的0附近划分更细，精度更高
- 量化时：将每组权重归一化，映射到最接近的码本取值，保存4-bit索引
- 反量化时：按码本与缩放因子还原

**动手验证**：想象一条正态分布曲线，中间

## 块 46

**本块学习检查**：读完下面内容后，请用一句话回答：NF4和普通INT4在量化范围设计上有什么关键区别？

---

**4.2 为什么适合模型权重？**

大模型训练后的权重通常近似**正态分布**，大量参数集中在0附近。

普通INT4是平均划分范围（如[-8,-7,...,7]），无法很好表示中心密集区域。

NF4则**根据正态分布设计量化范围**，特点是：0附近精度更高、更符合权重分布、精度损失更低。

---

**4.3 NF4量化过程**

原始FP16权重（如0.023、-0.145、0.876）经过：**归一化 → 映射到NF4 → 保存4-bit值**。推理时再反

## 块 47

## 5.1 为什么需要双重量化？

你已经知道4-bit量化能省显存，但别忘了：量化本身也要“记账”。

每组的权重，除了存4-bit数值本身，还得额外存两个参数：

- **scale**（缩放因子）
- **zero point**（零点偏移）

这些“账本”同样占显存。模型越大，组数越多，这笔开销越不可忽视。

## 5.2 双重量化思想

普通量化只做一步：

> FP16权重 → 4-bit权重 + 存scale

双重量化多做一步：

> 把scale参数**再量化一次**

也就是：第一次量化模型权重，第二次量化量化参数本身。

结构如下：

> FP16权重 → NF4量化 →

## 块 48

**为什么需要分页优化器？——用“内存不够”的场景来理解**

你在用 LoRA 微调时，显存不只是给模型用的。训练过程中，显存峰值来自三部分：模型本身、梯度、优化器状态。比如 Adam 优化器，需要额外保存一阶动量和二阶动量，这两项会占用不少显存。

QLoRA 通过 4-bit 量化把模型显存降下来了，但训练中仍可能突然显存不足（OOM），导致训练中断。这就是分页优化器要解决的问题。

**核心思想：像操作系统一样“换页”**

分页优化器借鉴了操作系统的虚拟内存机制，借助 NVIDIA 的统一内存（Unified Memory）。当显存出现峰值、即将 OOM 时，自动把优化器状态按“页”换出到 CPU 内存；需要参与计算时，再换回 GPU。

流程是：GPU 显存

## 块 49

## 6.3 先打个比方：分页优化器到底在干嘛？

想象你在一张很小的书桌上训练模型。普通优化器会把所有东西（模型参数、梯度、优化器状态）全堆在桌面上，桌面一满，程序就崩了（OOM）。

分页优化器的思路是：**桌面只放正在用的常用文件，暂时不用的挪进柜子，需要时再取出来**。这样桌面永远有空间，训练就不会因为“放不下”而中断。

## 6.4 代码里怎么开启？只需改一个参数

好消息是：你不需要自己写任何“换页”逻辑，`bitsandbytes` 库已经帮你实现了。你只需要在 `TrainingArguments` 里指定 `optim` 参数：

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

**两个选项怎么选？**
- `paged_adamw_8bit`：优化器状态用 8-bit 存储，更省显存。
- `paged_adamw_32bit`：精度更高，但占用更多显存。

**注意**：不同版本的 Transformers / bitsandbytes 支持的 `optim` 取值可能不同（还有 `adamw_bnb_8bit` 等）。如果报“未知 optim”错误，查一下你当前版本的文档，换用支持的名称即可。

**什么时候用？** 分页优化器主要解决**峰值 OOM**（显存突然爆掉）。建议和 `gradient_checkpointing=True`、小 `batch_size` + 梯度累积一起用，把显存峰值压到最低。

---

**本块学习检查**：打开你正在

## 块 50

## 7.1 4-bit量化：把模型“瘦身”

**核心作用**：减少模型权重大小，让模型占用更少的显存。

**直观理解**：把原本用FP16（16位精度）存储的权重，压缩成NF4格式（4位精度）。就像把高清照片转成压缩图，文件变小了，但画面依然可辨认。

```
FP16（16位）  →  NF4（4位）
```

**动手验证**：在代码中打印量化前后模型的内存占用，对比差异。

---

## 7.2 双重量化：连“尺子”也压缩

**核心作用**：减少量化参数的占用。

**直观理解**：量化时需要一组“缩放因子”（scale）来还原数值。双重量化就是把这组缩放因子**再量化一次**，进一步省空间。

```
scale（缩放因子）  →  再次量化
```

---

## 7.3 分页优化器：显存不够，内存来凑

**核心作用**：降低训练时的峰值显存。

**直观理解**：当GPU显存不够用时，自动把部分数据暂时挪到CPU内存，需要时再换回来，类似电脑的虚拟内存。

```
GPU  ⇄  CPU内存交换
```

---

## 八、QLoRA vs LoRA：一张表看懂

| 项目 | LoRA | QLoRA |
|------|------|-------|
| 基础模型 | FP16/BF16 |

## 块 51

## load_in_4bit：让大模型“瘦身”的关键开关

你正在学习 LoRA 微调，核心目标是用更少的显存跑更大的模型。QLoRA 的 `load_in_4bit=True` 就是第一步：**把模型权重以 4-bit 精度加载进显存**，而不是默认的 32-bit。这一下就能省下约 8 倍显存，是后续所有操作的基础。

## bnb_4bit_quant_type：选哪种 4-bit 量化方式？

设置 `"nf4"`，即使用 **NF4（4-bit NormalFloat）量化**。这是专门为神经网络权重设计的量化格式，相比普通 4-bit 整数，它在权重分布上更合理，精度损失更小。QLoRA 论文推荐的就是它，直接照用即可。

## bnb_4bit_use_double_quant：再省一笔显存

开启双重量化（`True`）。它的思路是：量化过程本身会产生一些额外的“缩放因子”参数，双重量化就是**把这些缩放因子再做一次量化**，从而进一步压缩显存占用。开启后几乎不损失精度，但能再省一点显存，建议保持开启。

## bnb_4bit_compute_dtype：存储精度 vs 计算精度

这是最容易混淆的点，务必分清两个概念：

- **存储 dtype**：权重以 4-bit（NF4）存着，目的是省显存。
- **计算 dtype**：权重参与矩阵运算时，会**先反量化回更高精度**再计算，这个精度由 `bnb_4bit_compute_dtype` 决定。

如果不显式设置，默认通常是 `float32`，计算慢且显存开销高。所以 QLoRA 配置里**必须显式指定**。

### 关键：按 GPU 能力选 BF16 还是 FP16，别写死

`torch.bfloat16`（BF16）需要较新的 GPU 支持（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100、H100）。老 GPU（如 RTX 20 系及更早）不支持，这时要用 `torch.float16`（FP16）。

**推荐做法：用代码自动判断**，而不是手动写死：

```python
import torch
from transformers import BitsAndBytesConfig

# 根据 GPU 能力选择计算 dtype：支持 BF16

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么用 4-bit 加载模型）、分页优化器（`optim`）和 `bnb_4bit_compute_dtype`（计算精度）。现在的问题是：**这些零件拼在一起，怎么组成一个真正能跑的 QLoRA 训练？**

下面给你一个**最小可运行骨架**。它把前面所有概念串成一条完整的流水线。你只需要把 `model_name` 换成你自己的模型路径，把数据集换成你自己的数据，就能跑起来。

### 整体流程（5 步）

1. **配置 4-bit 量化**（`BitsAndBytesConfig`）
2. **以 4-bit 加载基础模型**
3. **为 k-bit 训练做准备**（`prepare_model_for_kbit_training`）
4. **注入 LoRA**（`get_peft_model`）
5. **设置训练参数并启动训练**（`Trainer`）

### 完整代码骨架

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
# 模型许可证：请确认所用模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存；4-bit 量化可显著降低显存占用

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

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

# 3) 关键：为 k-bit 训练做准备（处理 norm/输出层精度、并开启梯度检查点所需的输入梯度等）
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # 与 TrainingArguments 的 gradient_checkpointing 对应
)

# 4) 注入 LoRA（只训练低秩增量参数）
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
    tokenizer=tokenizer,
)

try:
    trainer.train()
except Exception as e:
    print(f"训练失败: {e}")
    raise
```

### 关键点拆解（按执行顺序）

**第 1 步 — 量化配置**：`load_in_4bit=True` 开启 4-bit 加载；`bnb_4bit_quant_type="nf4"` 指定 NF4 量化类型；`bnb_4bit_use_double_quant=True` 开启双重量化；`bnb_4bit_compute_dtype` 根据 GPU 是否支持 bfloat16 自动选择计算精度。

**第 2 步 — 加载模型**：`quantization_config

## 块 53

## 九、把QLoRA的“五步骨架”串起来（可跑通版）

你之前已经分别了解了4-bit量化、NF4、双重量化和分页优化器，现在我们把它们**串成一条能跑通的流水线**。记住：下面这五步，**缺一不可**。

### 第1步：`BitsAndBytesConfig` —— 决定模型怎么“瘦身”加载

这一步是**配置源头**。它决定了基础模型以什么方式做4-bit加载，具体包含三件事：
- **NF4**：4-bit量化用的数据类型
- **双重量化**：进一步压缩量化常数占用的额外空间
- **计算dtype**：实际计算时用的精度（如BF16/FP16）

> 一句话：**先在这里定好“怎么量化”，后面才有得玩。**

### 第2步：`prepare_model_for_kbit_training()` —— 千万别跳过的“适配层”

这是QLoRA的**必要一步**。量化后的模型不能直接拿去训练，需要它来做适配：
- 把部分层转成计算精度
- 开启输入梯度（为梯度检查点做准备）

> ⚠️ **常见坑：跳过这一步，训练常常不收敛或直接报错。** 所以无论多急，这步必须做。

### 第3步：`get_peft_model(..., LoraConfig)` —— 注入LoRA“外挂”

这一步把**可训练的LoRA适配器**注入到**冻结的4-bit主干模型**上。主干模型保持4-bit不动，只有LoRA参数参与更新——这就是QLoRA省显存的核心逻辑。

### 第4步：两个开关要“对表”

- **`gradient_checkpointing=True`**（在`TrainingArguments`里）
- **第2步里的`gradient_checkpointing=True`**

这两个**必须对应开启**，否则梯度检查点不会生效，显存占用会飙升。

### 第5步：`optim="paged_adamw_8bit"` —— 分页优化器

最后，在`TrainingArguments`中设置`optim="paged_adamw_8bit"`，用分页优化器来**缓解峰值显存**。它把优化器状态放到CPU内存与GPU显存之间“分页”交换，避免显存瞬间爆掉。

---

### 最小骨架（能说明关系即可）

上面五步就是QLoRA的**最小可跑骨架**。真实训练时，你需要：
- 换成足够规模的数据
- 调合理的超参数
- 按你用的库版本核对`optim`名称、`prepare_model_for_kbit_training

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

先看一个关键认知：**LoRA训练过程中，基础模型是被冻结的**。

```
基础模型（Base Model）  →  冻结，不动
LoRA参数（Adapter）     →  训练，更新
```

训练结束后，你得到的结构是：

```
基础模型 + LoRA Adapter
```

举个例子：你用Llama-7B做LoRA微调，训练完成后保存的**通常不是完整模型**，而是两个文件：

```
adapter_model.safetensors   ← LoRA权重
adapter_config.json         ← 配置信息
```

### 1.2 Adapter有什么特点？

**优点：**
- 文件小，保存快
- 同一个基础模型可以搭配多个Adapter，实现多任务复用

比如同一个基础模型，可以分别搭配：

```
客服Adapter
医疗Adapter
法律Adapter
```

**缺点：**
- 部署推理时，需要**基础模型 + Adapter**两部分同时存在，缺一不可

---

## 二、为什么需要合并LoRA权重？

既然Adapter有"轻量、灵活"的优点，为什么还要合并？

核心原因在于**部署场景**。当你需要把模型投入实际推理时，带着"基础模型 + Adapter"两个部分一起跑，流程更复杂、管理更麻烦。通过合并，可以把Adapter的权重**融入基础模型**，得到一个完整的、独立的模型文件，部署时只需要加载这一个模型即可。

```
# 注意：Llama-2-7b FP16 需要至少14GB显存。
```

---

## 本块学习检查

**动手验证一下你的理解：**

1. 打开你训练好的LoRA输出目录，确认里面有哪些文件？是否包含`adapter_model.safetensors`和`adapter_config.json`？
2

## 块 55

## 二、为什么需要合并LoRA权重？（②③）

### 2.1 推理时的问题

你刚接触LoRA微调，先看一个关键场景：**推理（生成）时**。

如果不合并权重，模型运行时是这样的：

```
输入 → Base Model + LoRA Adapter → 输出
```

这意味着每次推理你都要：

1. 先加载基础模型（Base Model）；
2. 再单独加载LoRA适配器（Adapter）。

**对部署来说，这很麻烦**——你希望交付的是一个完整的模型文件，而不是“基础模型+插件”的组合。

所以，你需要把LoRA权重**合并**进基础模型，得到：

```
输入 → 完整模型 → 输出
```

# 模型许可证：Llama

## 块 56

## 3.1 merge_and_unload() 是干什么的？

你现在已经训练好了LoRA，但模型还是“基础模型 + 小补丁”的组合。`merge_and_unload()` 就是把这个补丁**焊死**进基础模型里，然后拆掉包装。

**执行前：**

```text
Base Model
+
LoRA Adapter
```

**执行后：**

```text
完整模型（权重已更新）
```

一句话：**合并权重 + 移除PEFT包装**。

---

## 3.2 合并前后对比

**合并前**，你的模型是一个 `PeftModel`，结构是：

```text
PeftModel
├── Base Model
└── LoRA Adapter
```

**合并后**，它变成了一个**普通Transformer模型**，LoRA的更新已经融进权重里，不再有单独的Adapter。

---

## 四、完整流程（记住这个顺序）

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

---

## 五、加载LoRA Adapter（动手写代码）

假设你有基础模型路径 `base_model_path` 和训练好的Adapter路径 `adapter_path`：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 把Adapter挂上去
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构 = `Base Model + LoRA Adapter`。

---

## 六、执行合并（核心一步）

```python
# 注意：Llama-2-7b FP16 需要至少14GB显存
merged_model = model.merge_and_unload()
```

这一行做了两件事：
- 把LoRA的权重变化（数学上就是 `W + BA`）写进基础模型，得到更新后的 `W'`
- 把 `PeftModel` 包装拆掉

所以 `merged_model` 现在就是**普通Transformer模型**，不再是 `PeftModel`。

---

## 七、导出合并后的模型

```python
# 保存模型权重
merged_model.save_pretrained("./merged_model")

# 别忘了Tokenizer
tokenizer.save_pretrained("./merged_model")
```

保存后目录大致长这样（**示意**，实际文件随模型类型变化）：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：不同模型可能多出 `generation_config.json`、`special_tokens_map.json`，或者分片权重（`model.safetensors.index.json` + 多个分片）；有些模型的tokenizer是 `vocab.json`/`merges.txt` 或 `tokenizer.model`。这些都属于正常情况。

---

## 本块学习检查

动手验证一下：按上面的代码，加载一个你训练好的Adapter，执行 `merge_and_unload()` 后，

## 块 57

## 3.1 方法作用：合并后，你拿到的是什么？

你可以把 LoRA 微调想象成“给原版模型戴了一副特制眼镜”——**不合并**时，模型 = 原版底座（Base）+ 眼镜（Adapter），每次推理都要先戴上眼镜（加载 PEFT 适配器）才能正常工作。

**合并（merge_and_unload）** 就是把“眼镜”的度数直接“焊”进底座里，得到一个**完整的、独立的模型**。合并后：

- **不再需要 PEFT 库**，直接用 `AutoModelForCausalLM.from_pretrained("./merged_model")` 就能加载，流程变成：用户输入 → 完整模型 → 生成结果。
- **部署更简单**：文件就是一个标准 Transformers 模型，任何支持 Transformers 的框架都能直接跑。

> **模型许可证**：请确认底座模型的许可证类型（如 Llama 2 Community License，商用月活>7亿需 Meta 授权），并在部署文档中标注。

> **显存要求**：加载合并模型前需评估显存，例如 Llama-2-7b FP16 至少需要 14GB 显存。

---

### 合并 vs 不合并，一张表看懂

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base + Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高（需 PEFT） | 简单（直接 Transformers） |
| 多个任务切换 | 方便（换 Adapter 即可） | 不方便（需重新合并） |
| 推理加载 | 需要 PEFT | 直接 Transformers |

---

### 什么时候该合并？（关键权衡）

合并不是“必须做”，而是**权衡**。问自己三个问题：

1. **需要多个任务动态切换吗？** 需要 → 不合并，保留多个 Adapter 更方便。
2. **部署框架原生支持 LoRA 吗？** 不支持 → 合并。
3. **在意存储/分发体积吗？** 在意 → 不合并（Adapter 文件小很多）。

> 注意：

## 块 58

**本块学习检查**：读完下面两个场景后，请用自己的话说：什么情况下“合并”是更好的选择？什么情况下“不合并”反而更灵活？

---

**场景1：上线部署（单一Adapter、追求简单）**

假设你训练好了一个LoRA模型，要放到服务器上提供API服务。如果只需要**一个固定Adapter**，且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载），那么**合并通常更方便**——直接加载模型就能提供服务。

> 但注意：如果你的推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或者你需要在同一基础模型上**热切换多个Adapter**，那么不合并、以Adapter形式部署反而更灵活。此时是否合并要看框架能力和业务需求。
>
> # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
> # 注意：Llama-2-7b FP16 需要至少14GB显存。

---

**场景2：分享模型**

如果你希望对方

## 块 59

### 10.2 不建议合并场景（动手理解版）

**场景1：多个任务Adapter**  
如果你有一个基础模型，上面挂了多个Adapter（比如Adapter A、B、C），**别合并**。  
保留它们，你就能随时切换不同任务，灵活调整。合并了反而绑死。

**场景2：持续训练**  
如果你还想继续调LoRA，**保留Adapter**。合并后就没法接着微调了。

**动手小验证**：  
想象你训练了“中文对话”和“代码生成”两个Adapter。  
问自己：合并成一个模型后，还能单独用其中一个吗？——不能，所以不合并。

---

### 十一、QLoRA合并注意事项（重点）

QLoRA里，基础模型是**4-bit量化**

## 块 60

## 动手实操：QLoRA 模型合并的正确姿势

### 先别急着合——直接合并可能踩的坑

你是要上手做 LoRA 微调的，那这一步很关键。很多人在 QLoRA 训练完，想直接把 Adapter 合并回基础模型，结果在 **4-bit 量化状态**下直接执行 `merge_and_unload`，可能会遇到两种情况：

1. **精度损失**——在4-bit量化状态下直接执行`merge_and_unload`，PEFT会自动处理反量化，但可能因版本或配置导致精度损失或报错。
2. **报错**——权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

注意：上面说的只是**可能**出现的问题，不是每次都会报错。能不能顺利合并、合并质量如何，取决于你的**版本**、量化配置，以及是否按下面的流程先反量化。**以你所用版本的文档和实际运行结果为准。**

### 稳妥做法：先反量化到高精度，再合并

既然直接合并有风险，那我们就绕开它。**核心思路：不在 4-bit 状态下合并，而是用高精度（FP16/BF16）重新加载一份基础模型，再挂载同一个 Adapter 去合并。**

流程如下：

```
用 FP16/BF16 重新加载基础模型（不传 quantization_config）
        ↓
加载同一份 LoRA Adapter
        ↓
执行 merge_and_unload()
        ↓
保存完整模型
```

这样合并发生在高精度权重上，就避开了 4-bit 状态下的类型和精度问题。

### 完整可运行代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
base_model_path = "your/base-causal-lm"   # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的 LoRA Adapter 目录
merged_out = "./merged_model"

# 设置随机种子以保证可复现性
torch.manual_seed(42)

try:
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
except Exception as e:
    print(f"模型加载或合并失败: {e}")
```

### 代码要点拆解

- **第

## 块 61

### QLoRA合并：先“还原”再合并

直接合并QLoRA模型时，关键一步是**加载基础模型时不再传`quantization_config`或`load_in_4bit`**。这样得到的是高精度权重（FP16/BF16），再执行`merge_and_unload`完成合并。

**两个注意点：**

- **内存**：高精度加载会占约一份完整模型的内存/显存。显存不够时，把`device_map`设为`{"": "cpu"}`在CPU上合并（较慢），或用offload。
- **一致性**：合并后的高精度模型与训练时“4-bit + Adapter”的推理结果**不会逐位相同**，只是近似。精度敏感场景需按§11.3验证。

## 块 62

**本块学习检查**：按下面三步动手验证，确认合并没问题。

1. **开启安全合并**：用 `safe_merge=True` 合并，代码就一行：
   ```python
   merged_model = model.merge_and_unload(safe_merge=True)
   ```
   这会在合并时自动检查数值（比如有没有 NaN），相当于加了一道保险。

2. **准备固定输入**：挑一小批固定的 prompt，别换，后面对比全靠它。

3. **对比前后输出**：分别用合并前的 PeftModel 和合并后的模型，在这同一批 prompt 上推理，然后做两件事：
   - 比 logits 的最大绝对差，看数值差多少；
   - 比贪心生成的文本，看内容是否一致。

如果两者

## 块 63

## 11.3 合并时开启校验并做一致性验证（建议）

这一步是给你的合并操作上“保险”。你刚学LoRA，最担心的就是合并完模型“变傻”了——输出跟原来不一样。这段代码就是帮你验证：合并前后，模型回答是否一致。

**核心思路**：拿一份“未合并”的模型当参照物，跟“已合并”的模型比一比。

### 为什么需要一份“未合并”的模型作参照？

代码注释里有个**关键点**：`merge_and_unload()` 返回一个合并后的新模型，而原来的 `PeftModel` 对象保持不变，仍保留 adapter 权重，可以继续用于推理或进一步操作。

因此，如果你需要对比合并前后的模型，可以直接使用合并前的 `PeftModel` 作为参照，无需重新加载。但为了代码结构清晰、避免误用已被合并的对象，下面仍演示如何显式准备一份独立的对比基准。

### 三步验证流程

**第一步：准备对比基准**

```python
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)
```

这里 `model_before` 的结构是：**Base(高精度) + Adapter**，也就是还没合并的原始状态。

> **# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）**
> **# 注意：Llama-2-7b FP16 需要至少14GB显存。**

**第二步：比较 logits（模型输出的原始分数）**

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)
```

`max_abs_diff` 越接近 0，说明合并前后模型内部计算越一致。QLoRA 场景下，由于量化精度损失，通常是一个**较小的近似值**——别指望完全等于 0，只要数值很小就说明合并没问题。

**第三步：比较实际生成的文本**

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

注意 `do_sample=False`——这是**去掉随机性**，保证两次生成用的是贪心策略，结果才能公平对比。如果这里输出 `True`，说明合并后模型行为没变，可以放心用。

### 动手验证

把 `prompt` 换成你任务里有代表性的固定输入（比如你微调时用的典型问题），跑一遍完整流程。如果 `max_abs_diff` 很小且生成

## 块 64

## 十一、QLoRA模型合并注意事项（重点）（③）

### 11.3 合并时开启校验并做一致性验证（建议）

**为什么要单独再加载一份？**

`merge_and_unload()` 返回合并后的新模型，原模型保持不变。若需验证，应保存合并前的模型引用或重新加载原始模型作为对照。否则若直接拿合并后的结果与自身比较，`max_abs_diff` 会恒为 0，验证就完全失去意义了。

**正确做法**：用 `base_for_ref` + `model_before` 重新挂一份**未合并**的 Adapter 作为对照，再和合并后的模型做比较。

**显存不够怎么办？**

如果无法同时放下两个模型，就**分两次跑**：
1. 先用未合并的 PeftModel 跑一遍，把 logits/文本存下来，释放显存；
2. 再加载合并后的模型，对比之前保存的结果。

**结果预期（重点）**

- **QLoRA 场景**：合并前后可能存在数值差异，`max_abs_diff` 的大小取决于量化配置和模型，应关注是否出现 NaN 或明显退化，而非预设"较小"。生成文本也可能存在差异。
- **LoRA（非量化）场景**：一般更接近一致，`max_abs_diff` 可以非常小。
- 是否可接受，以你的任务指标与验证集为准。

---

## 十二、电脑配置批注

本章节涉及加载完整模型、合并权重、导出模型，**需要实际加载模型**。

**模型许可证**：以 Llama 2 为例，适用 Llama 2 Community License（商用月活>7亿需Meta授权），具体以所用模型为准。

**最低配置**：
- CPU：4核+
- 内存：32GB+（若为量化模型可适当降低）
-

## 块 65

## 十一、合并时开启校验并做一致性验证（建议）

### 为什么建议做这一步？

你正在走 LoRA 微调这条路，目前处于**入门阶段**，而且偏好动手实操。QLoRA 合并是微调流程里最容易“悄悄出错”的环节——模型能跑，但结果不对。开启校验并做一致性验证，就是给合并结果上一道保险，确保你辛苦微调出来的权重没有在合并过程中被“搞坏”。

### 核心操作：QLoRA 合并前必须先反量化

**QLoRA 合并前，必须先反量化到 FP16/BF16，再执行 `merge_and_unload`。** 这是强制步骤，原因在于：QLoRA 使用 4-bit 量化存储基础模型权重，若直接在 4-bit 状态下合并，合并过程可能因精度不足或量化包装未正确反量化而失败，或产生异常结果。先反量化到 FP16/BF16，可确保合并基于完整精度进行，避免上述问题。

```python
# 先反量化到 FP16/BF16
model = model.to(torch.float16)  # 或 torch.bfloat16
# 再执行合并
model = model.merge_and_unload(safe_merge=True)
```

### 一致性验证：合并后跑一次推理

合并完成后，**不要急着保存**。先加载合并后的模型，用你微调时用过的**同一条测试输入**跑一次推理，对比合并前后的输出是否一致（或非常接近）。

具体做法：

1. 合并前，先用原始 PeftModel（未合并）对某条输入做一次推理，记录输出。
2. 合并后，用同一个模型对**同一条输入**再做一次推理。
3. 对比两次输出。如果差异明显，说明合并过程可能出了问题，需要排查。

### 常见问题速查（合并阶段）

| 问题 | 常见原因 | 排查方向 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 PeftModel（可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或已经合并/加载成普通模型） | 确认加载方式，先得到 PeftModel 再调用 |
| 合并后模型变大 | **正常现象**：合并后是完整模型，不再是小

## 块 66

## 14.1 路线A：合并后部署完整模型

你已经完成了 LoRA 微调，现在面临一个选择：**怎么把模型部署上线？** 这一节先讲最直接的一条路——**合并后部署**。

### 操作流程

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

三步走：先用 `merge_and_unload()` 把 LoRA 权重合并进基础模型，保存成一个完整的模型文件，然后部署时推理框架**根本不需要知道 LoRA 的存在**，把它当普通模型加载就行。

> # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
> # 注意：Llama-2-7b FP16 需要至少14GB显存

### 这条路的特点

**优点：**

- **部署链路简单**——推理框架只认"完整模型"这一种形态，不需要额外配置，不用理解 LoRA 是什么；
- **推理时零额外开销**——因为权重已经合进去了，运行时**不再有 Adapter 切换成本**，性能最直接。

**代价（你需要权衡的点）：**

- 每训练一个 Adapter，就要**保存/分发一份完整模型**。如果你只做一个任务，这完全没问题；但如果要做多个任务，每个任务都得存一份完整模型，**存储成本会线性增长**。

###

## 块 67

## 14.3 怎么选？——先记住“合并不是唯一出路”

你现在刚接触LoRA，最容易形成的惯性思维是：训练完必须合并权重才能用。这一节就是要打破这个惯性。

**记住三条经验参考**（非硬性规则，实际部署时灵活判断）：

1. **你只调了一个Adapter，想怎么简单怎么来** → 走**合并路线**（把LoRA权重融进基础模型，得到一个完整模型文件）。
2. **同一个基础模型要服务多个Adapter，且推理框架原生支持LoRA** → 走**保留Adapter动态服务**路线（不合并，部署时动态加载不同Adapter）。
3. **你是QLoRA微调，对精度敏感** → 不管选哪条路，**上线前都要做一致性验证**（合并前后/上线前对比，确保结果一致）。

> 具体框架怎么用、有什么限制，下一板块（vLLM部署）再展开。现在你只需要建立这个认知：**合并不是唯一部署方式**。

---

## 十五、总结：你已掌握

✅ LoRA Adapter保存方式  
✅ 为什么

## 测评题目

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB12-ST1205-003
合并 LoRA 权重后最直接的用途是？

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB12-ST1204-003
分页优化器的主要目标是？
