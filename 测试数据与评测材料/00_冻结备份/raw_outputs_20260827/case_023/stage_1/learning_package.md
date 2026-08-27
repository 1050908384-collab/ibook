# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

想象一下，你要让一个已经读过海量书籍的AI“转行”做客服。传统做法是：把AI大脑里**全部70亿个参数**（以Llama-2-7B为例，其参数量约等于层数×隐藏层维度等结构参数的总和）都拿出来重新训练一遍。

这个过程是这样的：

```text
输入数据
    ↓
整个模型参数更新
    ↓
保存完整模型
```

**问题出在哪？** 训练时，你的GPU不仅要存下这70亿个参数，还得额外腾出空间保存：

- **参数本身**：模型的基础“记忆”
- **梯度**：训练中计算出的调整方向
- **优化器状态**：控制调整步伐的“油门”

三者叠加，显存需求直接翻了好几倍（以Adam优化

## 块 2

**PEFT：参数高效微调**

核心思想：**不动整个大模型，只训练少量新增参数**。

流程：原始大模型 + LoRA适配器参数 → 只训练LoRA参数 → 保存小规模Adapter。

**三种方式对比**：

- 全参数微调：训练全部参数，显存需求非常高
- LoRA：训练少量参数，显存需求较低
- QLoRA：量化模型+LoRA，显存需求更低

**你的目标**是QLoRA微调，它比LoRA更省显存，适合在有限GPU环境下运行。动手时记住：你只需关注新增的LoRA参数，原始模型保持冻结。

## 块 3

**本块学习检查**：读完本节后，请用一句话向自己解释：LoRA 为什么只训练两个小矩阵就能微调大模型？

---

**2.1 LoRA 是什么？**

LoRA 全称 **Low-Rank Adaptation（低秩适配）**，核心思路是：**冻结原模型权重，只训练新增的小矩阵**。

原模型权重记为 `W`，训练时**不更新**。我们新增一个增量 `ΔW`，最终效果相当于 `W + ΔW`。

关键来了：`ΔW` 不是随便一个大矩阵，而是由两个**低秩小矩阵**相乘得到：

```
ΔW = B × A
```

- 若原权重 `W` 形状为 `d_out × d_in`，秩为 `r`
- 则 `A` 形状约为 `r × d_in`，`B` 形状约为 `d_out × r`
- 相乘后 `B × A` 得到与 `W` 同形状的 `ΔW`

> 注意：不同资料可能

## 块 4

## 3.1 LoRA的问题：模型还是太大

你已经知道LoRA能大幅减少**训练参数**——这是它的核心优势。但这里有个关键点容易被忽略：**基础模型本身仍然需要完整加载到显存里**。

以7B模型为例，如果用FP16精度加载，大约需要 **14GB显存**。这意味着即使LoRA只训练少量参数，你的GPU也得先装得下整个基础模型，否则根本跑不起来。

## 3.2 QLoRA的解决思路：给模型“瘦身”

QLoRA（Quantized LoRA，量化版LoRA）的思路很直接：**先把模型变小，再加载LoRA训练**。

流程如下：

```
FP16模型 → 4bit量化 → 减少显存占用 → 加载LoRA训练
```

训练时的结构变成：

```
4bit基础模型 + LoRA参数 → 训练
```

简单说：**基础模型用4bit精度存储，LoRA参数照常训练**。这样既保留了LoRA省训练参数的优势，又解决了基础模型占显存的问题。

## 3.3 关键依赖：bitsandbytes

QLoRA的实现依赖以下组件：

- `transformers`：模型加载与训练框架
- `peft`：提供LoRA等参数高效微调方法
- `bitsandbytes`：负责低精度量化、4bit加载、8bit加载
- 支持的量化后端（本教程以 **NVIDIA CUDA GPU** 为准）

其中，`bitsandbytes` 是量化的核心工具，它负责把模型压缩到4bit或8bit精度。

> **模型许可证**：本教程涉及的模型（如 Llama 2）需遵守其许可证，例如 Llama 2 Community License（商用月活>7亿需

## 块 5

## 电脑配置批注：为你的QLoRA微调备好“行囊”

你的目标是QLoRA微调，咱们直接看硬件门槛。别被“大模型”吓到，QLoRA的设计初衷就是**在更小的显存上跑起来**——它通过4-bit量化基础模型，显存占用通常比同规模的LoRA全精度加载更低。

**LoRA最低配置（先打个底）：**
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA GPU，显存**最低8GB**，**推荐12GB+**

**QLoRA要求（你的主战场）：**
- **GPU**：本教程实训范围限定为**NVIDIA CUDA GPU**（项目支持范围）。注意：QLoRA并非技术上只能用NVIDIA，但其它后端请以bitsandbytes官方说明为准，本文命令按NVIDIA CUDA给出。
- **显存参考**（以常见7B级别模型为例）：
  - **参考下限**：约8~12GB（短序列、小batch）
  - **推荐**：16GB+

**⚠️ 别把上面的数值当“硬门槛”**。实际显存占用受多项测试条件影响，条件不同结论就不同：
- **模型规模**：7B级别；若上13B/70B，显存需求显著上升
- **序列长度**：短序列（如≤512）；越长，激活值显存增长越明显
- **batch大小**：小batch（如1~2）；越大占用越高
- **梯度检查点（gradient checkpointing）**：开启可用时间换显存，降低峰值占用
- **量化配置**：4-bit（NF4）、是否双重量化、`bnb_

## 块 6

## 5.1 Python版本检查

在开始QLoRA微调之前，先确认你的Python环境。打开终端，运行：

```bash
python --version
```

建议使用 **Python 3.10或更高版本**。如果版本过低，请先升级Python环境，否则后续安装PEFT和bitsandbytes时可能遇到兼容性问题。

---

## 5.2 PyTorch检查（关键步骤）

**为什么重要？** LoRA/QLoRA依赖PyTorch，而`bitsandbytes`的CUDA量化路径要求安装**与本机CUDA匹配的GPU版PyTorch**。

**常见坑**：直接执行`pip install torch`通常装到CPU版，QLoRA会无法使用。

**正确做法**：前往[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)，选择与你GPU驱动匹配的CUDA版本。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证PyTorch能否看到GPU**，运行以下Python代码：

```python
import torch

print(torch.__version__)          # 查看PyTorch版本
print(torch.cuda.is_available())  # 是否可用CUDA
print(torch.version.cuda)         # 查看torch编译使用的CUDA版本
```

如果输出`True`，说明PyTorch侧能识别GPU。

> **注意**：`torch.cuda.is_available()`返回`True`只说明PyTorch能用GPU，**不足以证明bitsandbytes的4-bit量化路径可用**。后者还依赖bitsandbytes是否正确编译/匹配到本机CUDA。真正的4-bit验证将在后续章节进行。

---

## 5.3 查看GPU信息

在终端运行：

```bash
nvidia-smi
```

输出应包含：
- **GPU型号**（如RTX 4090）
- **CUDA版本**（如CUDA 12.x）
- **显存大小**（如24GB）

确认这些信息后，你就能确定后续安装bitsandbytes时应该匹配哪个CUDA版本。

## 块 7

**bitsandbytes 是什么？**

简单说，它是让大模型“瘦身”的工具，主要干三件事：

- **8bit量化**：把模型精度从32位降到8位，省显存。
- **4bit量化**：进一步降到4位，更省。
- **QLoRA训练**：你的目标——用4bit量化跑LoRA微调，就靠它。

**为什么必须装？**

因为本教程的QLoRA方案，就是用`bitsandbytes`做4bit量化。它是前置依赖，不装跑不了。注意：QLoRA本身不绑定某个量化库，但本教程统一用它+NVIDIA CUDA演示。

**安装命令**

```bash
pip install bitsandbytes
```

**本块学习检查

## 块 8

**动手前先查兼容性**

你的目标是跑通QLoRA微调，而bitsandbytes是量化依赖。它依赖CUDA驱动GPU计算，版本不匹配会直接导致CUDA不可用、量化失败或加载模型报错。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存

**两步检查，30秒搞定：**

1. **查CUDA版本**  
   终端运行`nvidia-smi`，看输出里的`CUDA Version`。

2. **查bitsandbytes**  
   运行`python -m bitsandbytes`，正常会显示CUDA版本和

## 块 9

### 9.1 推荐使用WSL2

**推荐路径**：Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes

**为什么推荐？** Linux环境兼容性更好，能减少你在QLoRA微调路上的环境折腾。

---

### 9.2 Windows原生可能问题

**问题1：bitsandbytes无法加载CUDA动态库**

这不是单一原因，常见情况包括：

- 所装`bitsandbytes`版本与本机CUDA / GPU版PyTorch不匹配；
- 缺少或找不到对应的CUDA运行库；
- Windows原生对某些`bitsandbytes`版本支持不如Linux完善（不同版本差异较大）。

**关键理解**：问题受**PyTorch、bitsandbytes版本与本机CUDA/硬件**共同影响，不是简单的"Windows不支持"。

**行动建议**：优先用WSL2获得更接近Linux的兼容性

## 块 10

## 九、Windows用户特别注意事项：CUDA版本不匹配怎么办？

你在Windows上准备QLoRA环境时，最常遇到的坑就是**CUDA版本不匹配**。别慌，先运行下面这条命令，看看你的机器到底什么状态：

```bash
nvidia-smi
```

这条命令会告诉你三件关键信息：

- **NVIDIA驱动**是否正常安装；
- 当前支持的**CUDA版本**；
- 之后装PyTorch时要匹配的版本依据。

注意：`nvidia-smi`输出的CUDA Version表示驱动支持的最高CUDA版本，需确保PyTorch的CUDA版本不高于此值。可用以下命令查看PyTorch的CUDA版本：

```bash
python -c "import torch; print(torch.version.cuda)"
```

记住：**驱动、CUDA、PyTorch三者必须对齐**，缺一个或版本不一致，后面`import torch`或QLoRA 4-bit量化就会直接报错。

---

## 十、安装完整LoRA环境

假设你已经有了一个Python环境，先装基础库：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

**关键前置**：确保你已经按5.2节安装了**匹配本机CUDA的GPU版PyTorch**。比如你的CUDA是12.1，就用这条命令：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

然后安装LoRA和QLoRA的核心库：

```bash
pip install peft bitsandbytes
```

想一步到位，直接跑这条完整命令：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate

执行：

```bash
accelerate config
```

按提示选择你的环境。比如你是单GPU，就这样选：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

完成后会生成一个`default_config.yaml`配置文件，Accelerate会用它来管理训练时的精度和分布式设置。

---

## 十二、LoRA环境验证实验

创建一个测试文件`test_lora_env.py`，写入以下代码：

```python
import torch

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM


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


model = AutoModelForCausalLM.from_pretrained('bert-base-uncased')
peft_model = get_peft_model(model, config)
print(peft_model)
```

运行：

```bash
python test_lora_env.py
```

如果顺利跑通，说明：

✅ PyTorch正常

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

# 注意：Llama-2-7b FP16 需要至少14GB显存。

## 块 11

## Windows用户特别注意事项：用WSL2解决兼容性问题

你在Windows上准备QLoRA环境时，最可能遇到的坑就是**兼容性问题**——比如bitsandbytes加载失败、CUDA报错等。这些问题的根源往往是Windows原生环境与深度学习工具链的适配不佳。

**最直接的解决方案：使用WSL2**（Windows Subsystem for Linux 2）。它让你在Windows上运行一个完整的Linux环境，而绝大多数深度学习工具（包括bitsandbytes、PEFT）都是优先为Linux设计的。

### 常见问题速查表

| 问题 | 原因 | 解决 |
|------|------|------|
| bitsandbytes加载失败 | CUDA不匹配 | 检查CUDA版本 |
| CUDA unavailable | PyTorch装了CPU版本 | 重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | **使用QLoRA的4bit量化配置（load_in_4bit=True）**，这是QLoRA降低显存占用的核心手段；也可降低batch或换更小模型 |
| Windows报错 | 兼容性问题 | **使用WSL2** |
| QLoRA速度慢 | GPU性能不足 | 优先优化batch size、使用梯度累积、检查数据加载瓶颈；升级GPU作为最后选项 |

### 你的

## 块 12

## 一、为什么需要LoRA？——先动手感受一个痛点

假设你现在拿到一个7B模型（比如LLaMA、Qwen这类），想让它学会某个新任务。传统做法是“全参数微调”：把模型所有参数都更新一遍，然后保存一个新模型。

**你猜会发生什么？**

- **显存爆炸**：7B参数不仅模型本身占显存，训练时还要额外保存梯度和优化器状态（比如Adam的动量）。实际显存需求远大于模型文件本身。
- **时间漫长**：更新全部参数意味着每次反向传播都要计算海量梯度。
- **存储昂贵**：每个任务都要保存一份完整的新模型副本。

**动手验证一下**：如果你有GPU，可以试着用`torch.cuda.get_device_properties(0)`查看显存总量，再对比7B模型仅参数就需要约14GB（FP16）——你马上会意识到全量微调有多“奢侈”。

---

## 二、LoRA的核心思想：不更新全部，只学“变化”

LoRA的思路很直接：**预训练模型的权重W已经学得很好，微调时不需要大改，只需要学一个“小补丁”ΔW。**

关键问题来了：这个ΔW怎么表示才够“省”？

**LoRA的答案是：低秩分解。**

把ΔW拆成两个小矩阵的乘积，并引入缩放因子α：

```
ΔW = (α/r) × B × A
```

- 假设W是 d×

## 块 13

**本块学习检查**：对照下面的图示，在纸上画出传统微调与LoRA的权重更新流程，并标出哪个参数被冻结、哪个参数是新增的。

---

**2.1 不直接修改原模型参数**

传统微调是直接改原权重：

```text
W → W'
```

LoRA的思路是：**冻结**原权重 `W`，不动它；额外增加一个新参数 `ΔW`。最终模型用的是：

```text
W' = W + ΔW
```

- `W`：原模型参数（冻结，不训练）
- `ΔW`：LoRA新增参数（只训练这个）

---

**2.2 为什么只训练ΔW？**

大模型已经通过海量数据学会了语言规律、世界知识、推理能力。微调通常只是**对已有能力做方向调整**，不必重新学全部参数，只学**任务相关的变化量**往往就够了。

> 注意：这是帮助理解的直觉，不是严格结论。任务与预训练能力差距大或数据充足时，全参数微调仍可能更好

## 块 14

## 3.1 矩阵基础概念

Transformer里大量参数都是矩阵。比如注意力层中：

```text
Q = X · Wq
```

这里 `X` 是输入（形状示意：`(序列长度, d_model)`），`Wq` 是权重矩阵（形状示意：`(d_model, d_head)`），两者相乘得到 `Q`。实际实现还涉及 batch 维、多头拆分，每个头独立计算，这里只为说明“Q 由输入乘以权重矩阵得到”。

假设 `Wq` 是 `4096 × 4096`，参数数量就是 `4096 × 4096 ≈ 1677万`。

## 3.2 直接训练矩阵的问题

如果直接训练更新量 `ΔW`，它仍然需要 `4096 × 4096` 个参数——成本太高。

**LoRA

## 块 15

**4.1 矩阵尺寸变化：从1677万到6.5万**

假设你要微调一个大模型，原始权重矩阵是 **4096 × 4096**。如果直接训练它，需要更新的参数数量是：

```
4096 × 4096 = 16,777,216（约1677万）
```

这非常庞大。LoRA的思路是：**不直接动这个大矩阵**，而是把它拆成两个小矩阵。

设置 **rank r = 8**，将原矩阵分解为：

- **A**：形状 `4096 × 8`，参数 = 4096 × 8 = **32,768**
- **B**：形状 `8 × 4096`，参数 = 8 × 409

## 块 16

**5.1 神经网络参数变化具有低维特征**

咱们要微调QLoRA，核心就是理解这个“低维”直觉。别被“4096维”吓到，想象一个房间，灯光的有效变化其实只集中在几个开关上。

LoRA论文的洞察是：模型适应新任务时，参数变化（ΔW）**不需要覆盖所有方向**。真正影响任务的“有效方向”可能远少于满维度，集中在一个**低维子空间**里——也就是远小于原维度的少数方向。

**关键提醒**：这是“权重更新具有较低内在维度”的**假设**，具体有效维度没有固定值，随模型、任务、数据变化。咱们定性理解即可，别记成“几十维”这种死数字。

所以，低秩矩阵

## 块 17

## 5.2 类比理解：为什么低秩分解有效？

想象一张高清图片，尺寸是 **4000×4000像素**。如果要把这张图片的每一个像素都完整保存下来，你需要存储 **1600万个数据**——这就像大模型里完整的权重矩阵，非常庞大。

但关键问题来了：这张图片真正的主要变化，其实可能只有几件事——比如**亮度**调整、**对比度**变化，以及**少量结构上的微调**。这些变化用**少量参数**就能描述清楚，完全不需要保存全部1600万个数据。

**LoRA的核心思想正是如此**：它不直接更新模型里巨大的权重矩阵，而是**用少量参数来描述模型权重的“变化方向”**。就像你不需要重新保存整张图片，只需要记录“亮度+10、对比度+5”这几个关键调整就够了。

> ⚠️ **需要澄清**：上述类比仅用于直观理解“少量参数描述主要变化”的思路。实际上，LoRA **冻结原始权重矩阵 W**，并不直接修改或调整它，而是训练两个低秩矩阵 **A** 和 **B**，通过增量 **ΔW = (α/r)·BA** 来更新，最终将 **W + ΔW** 作为新权重使用。

---

## 六、LoRA在Transformer中的位置

先回顾Transformer的主要结构：

```text
Input → Embedding → Attention → FFN → Output
```

其中**Attention模块**内部包含四个投影矩阵：

```text
Q（查询）、K（键）、V（值）、O（输出）
```

在LoRA的**早期实践**中，一个**常见做法**是只把LoRA加到其中两个投影上：

```python
target_modules=[
    "q_proj",   # Q投影
    "v_proj"    # V投影
]
```

这意味着：只训练 `Wq` 的变化和 `Wv` 的变化，其他参数全部冻结。

> ⚠️ **重要说明**：只选 `q_proj`/`v_proj` 是LoRA原论文中的一种**常见选择**，**并不是固定标准**。现代实践中，LoRA也经常被加到注意力的全部投影（q/k/v/o）、FFN的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存占用和最终效果，需要结合模型结构（先确认实际模块名）和验证集表现来权衡。

下面给出一个**覆盖更完整线性层**的配置示例（以Llama类模型常见的模块名为例；不同模型模块名可能不同，请

## 块 18

## 8.1 rank（r）——先动手，再理解

你现在的目标是 **QLoRA 微调**，而 `r`（rank）是 LoRA 里第一个要亲手调的核心参数。别急着背表格，我们先用“手感”建立直觉。

**一句话理解**：`r` 就是低秩矩阵的“维度”。代码里写成 `r=8`，意思是把权重更新压缩到一个 8 维的中间空间里。

**直观类比**：想象你要记住一本书的内容。`r` 小，就像只做 8 条摘要——省空间但可能丢细节；`r` 大，就像做 64 条详细笔记——记得全但占地方。LoRA 的 `r` 就是控制这个“摘要粗细”的旋钮。

**核心权衡（记住这一对矛盾就够了）**：

- `r` 越大 → 表达能力越强（能学更复杂的模式）
- `r` 越大 → 参数更多、显存占用更高（训练更贵）

**起点参考（不是规则，是搜索起点）**：

| 任务类型 | rank 起点 |
|---------|----------

## 块 19

## 8.2 lora_alpha：控制LoRA更新强度

在代码里你常会看到这样的参数设置：

```python
lora_alpha=16
```

这个参数的作用是**控制LoRA更新强度**。简单说，它决定了训练时对原模型权重的“改动幅度”有多大。

实际使用中，LoRA 的增量更新公式为 **ΔW = (α/r)BA**，其中 `alpha` 是缩放系数，控制增量矩阵的缩放比例，而非直接控制更新强度。比如 `alpha=16`、`r=8` 时，缩放比例就是 2。这个比例越大，LoRA 对模型的影响就越强；越小则越保守。

**动手理解**：你可以把 `alpha` 想象成“音量旋钮”——`r` 决定了有多少个“调节通道”，而 `alpha` 决定每个通道的缩放比例。两者配合使用，而不是单独调一个。

---

## 8.3 lora_dropout：防止过拟合

```python
lora_dropout=0.05
```

这个参数的作用是**防止LoRA参数过拟合**。它会在训练过程中，在LoRA层的前向传播中随机丢弃一部分激活值（这里是5%），迫使模型不依赖某几个特定激活，从而提升泛化能力。

**动手理解**：如果你发现训练集表现很好但验证集表现差（过拟合迹象），可以适当调大 `lora_dropout`；反之如果欠拟合，可以调小。

---

## 九、LoRA vs 全参数微调

| 项目 | 全参数微调 | LoRA（相对

## 块 20

## 2.1 Rank表示什么？

Rank，中文叫**矩阵的秩**。在LoRA里，它决定的是**低秩矩阵中间维度的大小**——也就是你打算用多“窄”的通道去近似原来的权重变化。

**用数字感受一下：**

假设原始权重矩阵是：

```text
W: 4096 × 4096
```

如果设置 `r=8`，那么LoRA会拆出两个小矩阵：

```text
A: 8 × 4096
B: 4096 × 8
```

两者相乘：

```text
ΔW = (α/r) × B × A
```

结果仍然是：

```text
4096 × 4096
```

**关键点来了：**

- 你**没有改变最终输出形状**——ΔW和W一样大；
- 但你**只训练了8×4096 + 4096×8 = 65,536个参数**，而不是4096×4096 ≈ 1678万个；
- 这就是LoRA“省”的地方：用低秩矩阵去**近似**完整的权重更新。

**直观理解：**

rank=8，意味着你用一个“8维的中间通道”去压缩和重建权重变化。rank越大，这个通道越宽，能表达的信息越多，效果可能越好——但训练和存储成本也越高。rank越小，越省资源，但可能“装不下”足够的信息，效果打折。

**对你的QLoRA目标来说：**

选rank就是在**效果**和**成本**之间做权衡。新手可以先从 `r=8` 或 `r=16` 起步，跑通流程后再调大

## 块 21

**Rank控制什么？——用“低秩矩阵维度”来理解**

把LoRA想象成给模型装“旋钮”。**Rank（r）是低秩矩阵的维度**，决定了可训练参数的数量（r×(d_in+d_out)）和模型的表达能力。

- **低rank（如 r=4）**：只有4个维度。模型能调的变化少，但好处是**参数少、显存低、训练快**；代价是**表达能力有限**，学不了太复杂的任务。
- **高rank（如 r=64）**：有64个维度。模型自由度大，**能学更复杂任务、适应力更强**；但**参数多、显存涨、训练成本高**。

**一句话总结

## 块 22

**3.1 不使用LoRA：全量训练的代价**

假设模型某层权重是 `4096 × 4096` 的矩阵，直接训练它需要更新：

`4096 × 4096 = 16,777,216` 个参数，约 **1677万**。

**3.2 使用LoRA：只学一个小补丁**

LoRA不碰原矩阵，只训练两个小矩阵，参数量公式：

`参数量 = r × (输入维度 + 输出维度)`

- **r=8**：`8 × (4096+4096) = 65,536`（约6.5万）
- **r=64**：`64 × 8192 = 524,288`

## 块 23

### 4.2 Rank太大的问题

那反过来，把 `r` 设成 `256` 呢？这次是“用力过猛”：

- **参数增加**：要训练的参数量暴涨。
- **显存增加**：显卡内存吃紧，可能直接OOM。
- **训练速度下降**：每一步都变慢。

更重要的是，**LoRA 的核心优势——参数高效——被削弱了**。

**为什么？用数字说话**：

对一个 `4096 × 4096` 的权重矩阵（完整参数量 16,777,216），LoRA 的参数量是 `r × (4096 + 4096)`。算一下占比：

| rank r | LoRA 参数量 | 占完整矩阵比例 |
| ---: | ---: | ---: |
| 8 | 65,536 | ≈ 0.39% |
| 64 | 524,288 | ≈ 3.13% |
| 256 | 2,097,152 | ≈ 12.5% |

**关键点**：就算 `r=256`，LoRA 参数量也仅占完整矩阵参数的 **12.5%**，远低于完整参数量的 **100%**。说它“参数高效优势下降

## 块 24

**4.3 常见rank选择（实操起点）**

你是要上手QLoRA微调，所以这里直接给你一套**经验起始值**，注意是“起点”，不是“标准答案”：

| 任务类型 | rank起点参考 |
|---------|------------|
| 简单分类 | 4~8 |
| 文本风格调整 | 8~16 |
| 指令微调 | 16~64 |
| 复杂领域任务 | 64以上 |

**但别直接照抄。** 同一个`r`，在不同模型大小、序列长度、batch大小、目标模块下，显存占用和效果可能差很多。实际调参要看：数据量、模型大小、序列长度、有效batch（含梯度累积）、`target_modules`选择、GPU显存、验证集表现。

**实操建议**：先按上表选个起点，跑通流程后，用验证集对比效果，再决定增大还是减小rank。

---

**五

## 块 25

## 6.1 控制LoRA影响强度——用`α/r`调音量

你已经知道LoRA通过低秩矩阵`A`、`B`来更新模型权重。那怎么控制这个更新的“音量”呢？关键就是**`α`（alpha）和`r`（rank）的比值**。

**先看一个具体例子：**

假设你设了`r=8`，`alpha=16`，那么缩放系数就是：

```text
16/8 = 2
```

这意味着LoRA的更新会被**放大2倍**。

如果`alpha=8`：

```text
8/8 = 1
```

更新保持**原始比例**，不放大也不缩小。

如果`alpha=4`：

```text
4/8 = 0.5
```

LoRA的影响就**减弱一半**。

---

### 核心公式

LoRA加入模型的更新量会被`α/r`这个系数缩放：

```text
alpha/r
```

所以`α/r`就是控制LoRA影响强度的**关键缩放系数**——你可以把它想象成一个“音量旋钮”。

---

### ⚠️ 重要提醒：`α/r`不是唯一决定因素

别误以为“实际强度只由`α/r`决定”。它只是一个**缩放系数**，实际落到权重上的更新量还取决于：

- **学习率**：直接决定每步更新步长。`α/r`调大又不降学习率，容易一起放大导致震荡。很多实践里甚至直接调学习率来控制强度，而不是频繁改`α`。
- **初始化**：常见实现中`A`用小随机值、`B`初始化为0，训练开始时

## 块 26

**配置1 vs 配置2：看比例，不看数字**

直接看代码：

- 配置1：`r=8, alpha=16` → 比例 = 16 ÷ 8 = **2**
- 配置2：`r=16, alpha=32` → 比例 = 32 ÷ 16 = **2**

**关键点**：虽然rank从8变成了16，但缩放比例都是2，没变。

**为什么这很重要？**

调整rank时，常见做法是**同步调整alpha**，让 `alpha / r` 这个比例大致稳定。这只是**经验做法**，不是硬性规则。上面的 `alpha = 2 × r` 只是常见起点，不是最优解。

**动手验证**：把配置

## 块 27

**方案1：小模型实验（你的起点）**

既然你的目标是QLoRA微调，且喜欢动手实践，建议从小模型实验开始。配置很简单：

```python
r=8
alpha=16
```

**为什么适合你？** 参数少、训练快，能快速跑通流程。对初学者来说，先看到结果、建立信心比追求效果更重要。

**关键理解：** `r`控制LoRA容量（模型能学多少新知识），`alpha`控制更新幅度（每次学多猛）。两者搭配，`alpha`通常是`r`的2倍。

**动手验证：** 用这个配置微调一个小模型，观察训练速度。如果跑得动且不报错，说明环境没问题，再考虑加大参数。

**注意：** 后续方案2（`r=16, alpha=32`）平衡效果和成本，方案3（`r=64, alpha=128`）适合复杂任务但参数更多。现阶段不用急着上大参数，先掌握流程。

## 块 28

好的，我们直接进入正题。既然你的目标是QLoRA微调，并且喜欢动手实践，那我们就把这三个“坑”讲透，让你在写代码时能避开它们。

### 误区1：rank越大一定越好？

**结论：错。**

很多初学者觉得rank（秩）越大，模型能学的东西就越多，效果肯定越好。但事实并非如此。

**为什么？** 想象一下，你要记住一个简单的规则，比如“红灯停”。你用一个巨大的笔记本（大rank）去记录，不仅浪费纸（**浪费显存**），而且翻找起来也慢（**增加训练成本**）。对于简单任务，大rank带来的额外表达能力是冗余的，得不偿失。

### 误区2：只调整rank，不调整alpha

这是一个非常常见的操作失误。我们来看一个例子：

你原来设置的是 `r=8, alpha=16`，这时缩放比例 `alpha/r = 16/8 = 2`。

现在，你想把rank调大来增强模型能力，于是只改了rank，变成了 `r=64, alpha=16`。

但这时，缩放比例变成了 `16/64 = 0.25`。

**这意味着什么？** LoRA的最终更新公式是 `W' = W + (alpha/r)BA`。你的rank变大了（B和A矩阵的参数量变多），但 `alpha/r` 这个缩放系数却从2骤降到0.25。结果就是，虽然模型容量变大了，但实际对原始权重 `W` 的“影响力度”反而被大幅削弱了。你本意是增强，结果却可能适得其反。

**记住：** 调整rank时，一定要同步考虑alpha，保持 `alpha/r` 的比例在一个合理的范围（比如1到2之间），否则参数配置就失衡了。

### 误区3：alpha越大越好？

**结论：错。**

alpha的作用是通过 `alpha/r` 这个缩放比来控制LoRA的更新强度。如果alpha过大，意味着 `alpha/r` 的值会非常大，导致对原始权重的更新过强。

**后果就是

## 块 29

**LoraConfig是什么？——给要跑QLoRA的你**

你现在的目标是QLoRA微调，而QLoRA的配置起点就是`LoraConfig`。PEFT（Parameter-Efficient Fine-Tuning）是Hugging Face的参数高效微调库，`LoraConfig`就是告诉PEFT“LoRA怎么装”的配置对象。

**LoRA训练流程**：

```
加载基础模型 → 创建LoraConfig → 指定LoRA插入位置 → 生成LoRA Adapter → 训练LoRA参数
```

`LoraConfig`负责四件事：

1. **LoRA放在哪里**——决定哪些层被改造
2. **LoRA规模多大**——决定参数量
3. **LoRA训练强度如何**——决定更新幅度
4. **是否使用dropout**——决定防过拟合策略

**动手验证**：打开Python环境，输入`from peft import LoraConfig`，确认无报错。然后打印`LoraConfig()`的默认参数，观察每个字段的默认值——这能帮你直观感受它“管了哪些事”。

**本块学习检查**：用自己的话说出`LoraConfig`在LoRA训练流程中处于哪一步、负责哪

## 块 30

## 动手跑通你的第一个LoRA配置

你正在为QLoRA微调打基础，咱们直接上手。下面这段代码就是LoRA最核心的配置，先看它长什么样：

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
|------|------|
| `target_modules` | 决定插入哪些层（这里选了q和v投影层） |
| `r` | 决定LoRA矩阵大小（8表示低秩维度） |
| `lora_alpha` | 控制缩放强度（16） |
| `lora_dropout` | 控制随机失活（0.05，防止过拟合） |

**先记一个直觉**：`r`越小，新增参数越少，训练越快；`lora_alpha`和`r`的比值影响最终效果。

---

## 配置怎么用起来

光有配置还不够，它要配合模型一起用。完整流程是：

```text
model → LoraConfig → get_peft_model() → LoRA模型
```

实际代码：

```

## 块 31

## 3.1 target_modules是什么？

`target_modules` 的作用就一句话：**指定LoRA要注入模型的哪些模块**。

Transformer模型里，Attention结构通常长这样：

```text
Attention
├── q_proj
├── k_proj
├── v_proj
└── o_proj
```

**关键点**：LoRA不会默认修改所有层，你必须手动告诉它改哪里。

比如你想只改Query和Value投影层：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

这段代码的含义就是：给`q_proj`（Query投影层）和`v_proj`（Value投影层）分别加上LoRA。

---

## 四、为什么选q_proj和v_proj？

看Attention计算公式就明白了：

```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
```

- **Q（Query）**：决定模型**关注什么信息**
- **V（Value）**：决定模型**传递什么信息**

所以调整Q和V，通常就能有效改变模型行为——这就是它俩成为常见选择的原因。

> ⚠️ **重要提醒**：`q_proj`/`v_proj`只是**常见起点**，不是所有模型的"标准配置"。不同模型模块命名可能不同，**配置前务必先用`model.named_modules()`查看实际模块名**，再决定`target_modules`。

---

**省事技巧**：较新版本的PEFT（一般≥0.7）支持`target_modules="all-linear"`，自动匹配（几乎）所有线性层，不用手工枚举。可运行示例：

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
    target_modules="all-linear",   # 自动匹配（几乎）所有nn.Linear；一般不含lm_head
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 打印可训练参数量/占比，确认LoRA已注入
```

**原理说明**：LoRA不修改原始权重W，而是冻结原权重，仅训练低秩矩阵A和B，形成增量更新ΔW = BA。**权衡点**：`"all-linear"`覆盖模块多，可训练参数和显存占用会增加。如果PEFT版本太旧不支持，就退回显式列出模块名。用`print_trainable_parameters()`快速验证注入是否符合预期。

---

**本块学习检查**：动手跑一下上面的示例代码，观察`print_trainable_parameters()`输出的可训练参数量。然后对比：把`target_modules`改回`["q_proj", "v_proj"]`，再跑一次，看参数量差多少。这个对比能帮你直观感受

## 块 32

## 方案1：只调整Q和V——轻量起步

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**这个方案的特点：**
- **参数少**——只训练注意力中的查询和值投影，改动范围小；
- **训练快**——因为要更新的参数少，整体训练速度更快；
- **常见轻量起点**——很多实践会从这里开始尝试，具体效果取决于模型结构。

---

### 动手验证一下

在你自己的微调脚本里，先用这个方案跑一个短训练（比如几十步），观察两个指标：
1. 显存占用是否在可接受范围；
2. loss 是否在下降。

如果训练顺利且效果尚可，就保持这个配置；如果觉得表达能力不够，再考虑切换到方案2或方案3。

---

### 三种方案速览（供你后续选择）

| 方案 | target_modules | 特点 |
|------|---------------|------|
| 1 | q_proj

## 块 33

### LLaMA系列：`target_modules` 怎么选？

在LLaMA这类模型中，`target_modules` 最常见的配置是只针对注意力层里的 `q_proj` 和 `v_proj` 这两个投影矩阵：

```python
[
"q_proj",
"v_proj"
]
```

**为什么这样选？**  
简单来说，`q_proj`（Query投影）和 `v_proj`（Value投影）是LoRA最常“动手脚”的地方。只微调这两个模块，就能在保持效果的同时大幅减少可训练参数量。你不需要改 `k_proj` 或 `o_proj`，这是经过实践验证的省力配置。

**动手验证一下（hands-on）**  
如果你手头有LLaMA模型（或同结构的模型），可以这样快速检查：

1. 用 `LoraConfig` 设置 `target_modules=["q_proj", "v_proj"]`，`r=8`，`lora_alpha=16`。
2. 调用 `get_peft_model` 包装模型。
3. 打印 `model.print_trainable_parameters()`，观察可训练参数占总参数的比例——你会看到它非常小，这正是LoRA“高效”的体现。

---

### 对比：GPT-2 的 `c_attn` 与 `fan_in_fan_out`

作为对照，GPT-2 的写法不同，因为它的注意力层把Q/K/V三个投影合并成了一个模块 `c_attn`：

```python
[
"c_attn"
]
```

**关键坑点：`fan_in_fan_out=True`**  
GPT-2 用的是 `Conv1D` 层（不是普通的 `nn.Linear`），其权重形状为 `(out_features, in_features)`，与 `nn.Linear` 相同，但因其实现方式不同，需设置 `fan_in_fan_out=True` 以正确注入 LoRA。

**完整示例（GPT-2）**：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 注意：Llama-2-7b FP16 需要至少14GB显存。
model = AutoModelForCausalLM.from_pretrained("gpt2")

lora

## 块 34

**动手查一下，别照抄**

你的目标是QLoRA微调，但BERT和LLaMA的层命名不一样。直接复制别人的`target_modules`配置大概率会报错或没效果。

**核心操作：先看模型里到底有什么**

加载模型后，运行这段代码，把模块名打印出来：

```python
for name, _ in model.named_modules():
    print(name)
```

在输出里找类似`query`、`value`这样的名字（BERT的LoRA通常应用于query和value，具体取决于任务）。

**然后这样配置：**

```python
target_modules = ["query", "value"]
```

**关键机制：**

- `target_modules`是按**模块名后缀**匹配的。
- 不同模型命名差异很大（比如GPT-2用的是`c_attn`），**不能照搬**。
- 版本不同，命名也可能变。一切以你实际打印出的`named_modules()`为准。
- 注意：Llama-2-7b FP16 需要至少14GB显存。

**本块学习检查**

运行`named_modules()`，从输出中找出BERT的query和value投影层的完整模块名，并写出对应的`target_modules`配置。

## 块 35

**r是什么？——用QLoRA微调前必懂的一个数**

想象你要把一个大矩阵（比如4096×4096）压缩成两个小矩阵相乘：ΔW = B × A。这里的**r**，就是这两个小矩阵的“中间维度”，直接决定压缩程度。

- 设r=8时：A是4096×8（d×r），B是8×4096（r×d），总参数量从约1678万降到约6.5万，实际更新时还需乘以缩放因子α/r。
- **r越大**：学习能力越强，能表达复杂任务，但参数和显存占用也增加。
- **r越小**：更省资源，但可能学不够。

**经验

## 块 36

**7.1 alpha作用（动手理解）**

LoRA最终更新公式不是 `BA`，而是 `(alpha/r)BA`。

**核心作用**：alpha 控制 LoRA 更新幅度。

**动手验证**：配置 `r=8`、`lora_alpha=16`，缩放系数 = `16/8 = 2`，表示 LoRA 影响增强。

**7.2 alpha常见设置**

经验起点：`alpha = 2 × r`（非通用规则，最终以验证集表现为准）。

示例：
- `r=8, alpha=16`
- `r=16, alpha=32`

**本块学习检查**：若 `r=4`，按经验起点设置 alpha，计算缩放系数，并说明其含义。

## 块 37

## 8.1 dropout作用

训练时，LoRA会对低秩矩阵（如A或B）的元素进行随机置零，目的是**减少过拟合**。

配置示例：

```python
lora_dropout=0.05
```

含义：**5%的LoRA参数被随机置零**。

---

## 8.2 如何选择dropout？

经验起点（非固定规则，需结合验证集是否过拟合来调）：

| 数据规模 | dropout（起点参考） |
|---------|------------------|
| 数据少   | 约 0.05~0.1      |
| 数据多   | 约 0~0.05        |

如果训练数据非常丰富，可以直接：

```python
l

## 块 38

## 9.1 基础配置：动手搭一个LoRA

你正在为7B模型做QLoRA微调，先掌握这个最常用的基础配置。下面这段代码就是LoRA的“启动器”：

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

**逐参数拆解（对照表格理解）：**

| 参数 | 值 | 作用 | 你的理解角度 |
|------|-----|------|-------------|
| `r` | 8 | LoRA容量 | 秩的大小，决定新增参数量，8是7B模型的稳妥起点 |
| `lora_alpha` | 16 | 更新缩放 | 控制微调时权重更新的力度，16是常用搭配 |
| `target_modules` | q/v | 插入位置 | 只改注意力层的q和v投影，减少改动面 |
| `lora_dropout` | 0.05 | 防止过拟合 | 训练时随机丢弃部分更新，5%是轻量保护 |
| `task_type` | CAUSAL_LM | 语言模型任务 | 告诉P

## 块 39

## 9.1 基础配置：用 LoRA 微调 BERT 分类器（动手实操版）

你的目标是 QLoRA 微调，但先别急——QLoRA 是在 LoRA 基础上加量化，核心配置逻辑完全一样。这一块我们直接上手跑通一个完整案例，把 `target_modules`、`r`、`lora_alpha`、`lora_dropout` 这几个参数全部用起来。

### 第一步：加载基础模型

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification

# 模型许可证：BERT 采用 Apache 2.0 许可证，可商用
# 注意：bert-base-uncased FP16 推理约需 0.5GB 显存，训练时需额外预留优化器与梯度显存
base_name = "bert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

注意：这里加载的是**带分类头**的模型，分类头是随机初始化的，必须参与训练，否则结果就是乱猜。

### 第二步：配置 LoraConfig

```python
lora_config = LoraConfig(
    r=8,                # 低秩矩阵的秩，越小参数越少
    lora_alpha=16,      # 缩放系数，控制 LoRA 的影响强度
    lora_dropout=0.05,  # 防止过拟合
    bias="none",        # 不训练偏置项
    task_type="SEQ_CLS", # 序列分类任务
    target_modules=["query", "value"],  # 只微调 BERT 的注意力 query/value
    modules_to_save=["classifier"],     # 关键！把分类头也纳入训练并保存
)
```

**重点理解 `modules_to_save`**：如果不加这一行，训练时分类头不会被更新，保存时也不会被保存。重载模型时分类头会退回随机初始化状态，分类结果直接错乱。

### 第三步：包装模型并训练

```python
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 你会看到 LoRA 参数 + classifier 一起可训练
```

训练过程略过，直接看保存和验证。

### 第四步：保存

```python
model.save_pretrained("./lora_ckpt")
```

这里保存的是 **Adapter 权重 + `modules_to_save` 指定的分类头**，不是完整的基础模型。

### 第五步：重载并验证

```python
reload_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
reloaded = PeftModel.from_pretrained(reload_base, "./lora_ckpt")

w_before = model.base_model.model.classifier.weight.detach().cpu()
w_after = reloaded.base_model.model.classifier.weight.detach().cpu()
print("classifier 权重是否一致:", torch.allclose(w_before, w_after))
```

如果输出 `True`，说明分类头被正确保存和恢复了。

### 两个易踩的坑

1. **重载时必须先加载同一个基础模型**（`bert-base-uncased`），再用 `PeftModel.from_pretrained` 挂载 Adapter。
2. **不同模型的分类头属性名可能不同**（比如 `classifier` 或 `score`），访问路径也可能随版本变化，以实际结构为准。

### 动手验证任务

跑完上面的代码后，把 `modules_to_save=["classifier"]` 这行删掉，重新训练并保存、

## 块 40

## 十、普通文本生成：LoRA参数怎么配？

你的目标是QLoRA微调，而QLoRA是在LoRA基础上做的优化，所以先掌握LoRA的经典参数配置，是必经之路。这一节我们直接看“普通文本生成”任务下，官方推荐的参数组合。

### 推荐配置

```python
r=8
alpha=16
dropout=0.05
```

这三个参数分别是什么意思？

- **`r`（秩）**：决定LoRA低秩矩阵的维度。`r=8`意味着用8维的矩阵去近似完整的权重更新。`r`越小，参数量越少，训练越快，但表达能力也越弱。
- **`alpha`（缩放因子）**：控制LoRA更新的缩放比例。`alpha=16`配合`r=8`，缩放比是`alpha/r = 2`。这个比值影响训练稳定性。
- **`dropout`（丢弃率）**：在LoRA层中随机丢弃一部分神经元，防止过拟合。`0.05`是一个比较保守的默认值。

> **动手验证一下**：如果你在代码里配置了`r=8`和`alpha=16`，那么实际生效的缩放系数就是`16/8=2`。你可以试着把`alpha`改成`32`，感受一下缩放比变化对训练loss的影响。

### 对比：其他任务的推荐值

为了帮你建立直觉，这里列出原文中其他任务的推荐配置，方便对比：

| 任务类型 | r | alpha | dropout |
|---------|---|-------|---------|
| 普通文本生成 | 8 | 16 | 0.05 |
| 指令微调 | 16 | 32 | 0.05 |
| 领域知识增强 | 32 | 64 | （未指定） |

规律很明显：**任务越复杂、越需要模型学习新知识，`r`和`alpha`就越大**。普通文本生成是最轻量的场景，所以用最小的配置。

### 扩展边界：LoRA不止这些参数

`LoraConfig`还支持一些扩展能力，你不需要现在深入，但要知道它们存在：

- `use_rslora=True`：启用RS-LoRA，缩放改用`α/√r`
- `use_dora=True`：启用DoRA（权重分解式LoRA变体）
- `rank_pattern` / `alpha_pattern`：为不同层单独设置`r`/`alpha`
- `init_lora_weights`：LoRA权重初始化策略

这些扩展的行为随PEFT版本变化，使用前务必查阅对应版本的官方文档。

### 硬件要求：你的电脑能跑吗？

- **只运行配置代码**：CPU即可，内存4GB+。
- **执行`get_peft_model()`并加载真实大模型**：需要NVIDIA CUDA GPU，显存最低8GB+，推荐

## 块 41

**本块学习检查**：对照下方配置，口头说出每个参数“管什么”，再在代码中改 `r=4` 并观察输出变化。

```python
LoraConfig(
    r=8,               # LoRA容量
    lora_alpha=16,     # 通过 α/r 缩放影响更新强度
    target_modules=["q_proj", "v_proj"],  # 决定改哪里
    lora_dropout=0.05  # 缓解过拟合
)
```

**核心理解**：`target_modules` 选要改的层，`r` 定容量，`alpha` 调更新强度，`dropout` 防过拟合。下一步将用 PEFT + Trainer

## 块 42

好的，我们直接进入正题。你已经掌握了 LoRA 的原理，现在我们要解决它的一个“痛点”。

**1.1 LoRA 的“隐形负担”：模型本体依然庞大**

LoRA 很聪明，它冻结了原始模型，只训练一小部分新增参数，这大大减少了**需要训练**的量。但请注意，它并没有减少**需要加载**的量。

想象一下：你要在一个房间里工作，LoRA 只是让你只负责改造房间里的一张小桌子（新增参数），但整个房间（基础模型）依然得完整地放在那里，你不能把它拆了。

对于 7B 模型，如果以 FP16（半精度）格式加载，光模型本身就需要：

`7B × 2 bytes ≈ 14GB 显存`

这仅仅是模型权重。一旦开始训练，还需要额外的空间来存放**激活值**（前向传播的中间结果）、**梯度**（用于更新参数）和**优化器状态**（如 Adam 的动量项）。这些加起来，实际显存需求会远高于 14GB。

**核心结论**

## 块 43

**QLoRA核心思想：用更小的显存跑大模型微调**

你的目标是QLoRA微调，先抓住核心：QLoRA = **Quantized LoRA**，即“量化版LoRA”。

**一句话理解**：把基础模型从FP16（16位精度）压到**4-bit**（4位精度），再在上面挂LoRA训练。模型变小了，显存占用大幅下降，但训练效果依然接近原版。

**流程对比**：
- LoRA：FP16模型 + LoRA
- QLoRA：**4-bit模型** + LoRA

**训练时的分工**：
- 基础模型：**4-bit权重**，**冻结**（不更新）
- LoRA：**FP16/BF16参数**，**训练**（只更新这部分）

**关键点**

## 块 44

## 3.1 模型存储方式：从“重量”看量化

先记住这张表——它决定了模型能塞进多大的显存：

| 类型 | 每个参数占用 |
|------|------------|
| FP32 | 4 Bytes |
| FP16/BF16 | 2 Bytes |
| INT8 | 1 Byte |
| **INT4** | **0.5 Byte** |

**动手算一下（7B模型）：**
- FP16：7B × 2 ≈ **14GB**
- 4-bit：7B × 0.5 ≈ **3.5GB**

显存直接减少约75%——这就是4-bit量化的核心吸引力。

> ⚠️ **注意**：3.5GB只是**理论下界**（仅权重本身）。实际占用更高，因为还要算上量化常数、分组元数据（NF4按组量化）、未量化的模块（norm、embedding、lm_head、LoRA参数）以及训练时的激活值缓存。**真实占用以实际运行为准**。

---

## 3.2 为什么不用普通INT4？

普通INT4范围是 **-8 ~ 7**，但神经网络权重通常：
- 接近**正态分布**，中心在0附近
- 小数

## 块 45

**NF4是什么？——用“查表”理解它**

别被“正态分布4位浮点”这个名字吓到。它**不是**IEEE那种通用浮点格式，而是一套**为神经网络权重“量身定制”的4-bit码本（codebook）**。

核心就一句话：**预先定好16个取值点（2⁴=16），基于标准正态分布的分位数初始化，并经过优化调整以最大化信息熵，使得量化值更接近正态分布的信息论最优**。因为权重大多集中在0附近，所以0附近的取值点更密，量化更精细。

**操作流程（三步）**：
1. **归一化**：将每组权重按块除以绝对最大值，缩放到[-1,1]范围

## 块 46

**为什么NF4适合模型权重？**

大模型训练完的权重有个特点：**近似正态分布**，也就是大量参数都挤在0附近。如果用普通INT4（比如把范围平均分成`[-8,-7,...,7]`），它没法精细表示中心密集区域，精度就浪费了。

NF4的思路是：**根据正态分布来设计量化范围**。这样0附近精度更高，更贴合真实权重分布，损失自然更低。

**NF4量化过程（快速过一遍）**

原始FP16权重（如`0.023, -0.145, 0.876`）经过absmax归一化后，根据正态分布的分位数映射到NF4的16个量化级别，保存4-bit索引；推理时通过

## 块 47

**为什么需要双重量化？**

你已经知道，4-bit量化会把FP16权重压成4-bit，但这个过程**不是免费的**——每组权重还得额外保存两个“量化参数”：`scale`（缩放因子）和`zero point`（零点）。这些参数本身也占显存，模型越大，累积起来越可观。

**双重量化的核心思路：**

普通量化只做一步：`FP16权重 → 4-bit权重 + scale参数`。

双重量化多做一步：**把scale参数本身也量化一次**。也就是：

1. 第一次量化：量化模型权重（NF4），得到4-bit权重和scale。
2. 第二次量化：量化scale参数，减少它的存储开销。

结构上就是：

```
FP16权重

## 块 48

### 6.1 为什么需要分页优化器？

训练时显存峰值不只来自模型本身，还有**梯度**和**优化器状态**。比如Adam优化器需要保存**一阶动量**和**二阶动量**，这些都会占用显存。

QLoRA虽然大幅降低了模型显存，但训练过程中仍可能遇到**显存突然不足**（OOM）的情况。

---

### 6.2 分页优化器思想

Paged Optimizer借鉴了操作系统**虚拟内存**的思路，核心是：

> 借助NVIDIA的**统一内存（Unified Memory）**，在显存出现峰值、即将OOM时，把优化器状态自动“分页”换出到CPU内存，需要时再换回GPU。

流程如下：

```text
GPU显存
   ↓
出现峰值（接近/超出显存上限）
   ↓
优化器状态按页换出到CPU内存（Unified Memory）
   ↓
需要参与计算时再换回GPU
```

**注意**：不要把它理解成“平时就把优化器状态常驻CPU”。分页优化器的重点是**应对显存峰值**——在显存不足的瞬间临时

## 块 49

## 6.3 类比理解：把显存想象成你的桌面

**普通方式**：所有东西都堆在桌面上（显存里）。桌面满了，就再也放不下任何东西，训练直接报错（OOM）。

**分页优化器**：像真实办公一样——常用文件放桌面（活跃的优化器状态），暂不用的文件放柜子（CPU内存），需要时再取出来。这样桌面永远有空间，训练不会因显存爆掉而中断。

## 6.4 实际启用：一行代码搞定

分页优化器不需要你手写换页逻辑，bitsandbytes 库已经实现了。只需在 `TrainingArguments` 里设置 `optim` 参数：

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

**关键点**：
- `paged_adamw_8bit`：优化器状态用8-bit存，更省显存；`paged_adamw_32bit`：精度高但占更多。
- **参数名随版本变**：不同 Transformers/bitsandbytes 版本支持的取值可能不同（如 `adamw_bnb_8bit`）。报"未知optim"错误时，查你版本的文档换名称即可。
- 分页优化器主要解决**峰值OOM**，通常和 `gradient_checkpointing=True`、小batch + 梯度累积搭配用。

---

**本块学习检查**：打开你环境里的 `TrainingArguments` 文档（或运行 `python -c "from transformers import TrainingArguments; print(TrainingArguments.__doc__)"`），找到 `optim` 参数当前支持的取值列表，确认你的版本里有没有 `paged_adamw_8bit`。如果没有，

## 块 50

## 7.1 4-bit量化：给模型“瘦身”

QLoRA 的第一步，是把模型权重从 FP16（16位浮点数）压缩成 NF4 4-bit 格式。你可以把它想象成把一张高清照片压缩成更小的文件——显存占用大幅下降，但画面依然可辨认。**作用：减少模型权重大小。**

```
FP16 → NF4 4-bit
```

## 7.2 双重量化：连“尺子”也压缩

量化过程中需要 scale（缩放因子）来还原数值。QLoRA 对量化过程中产生的 scale 参数进行二次量化，进一步减少内存占用。**作用：减少量化参数占用。**

```
scale → 再次量化
```

## 7.3 分页优化器：显存不够，内存来凑

训练时优化器状态很占显存。分页优化器利用统一内存技术，在 GPU 显存不足时将优化器状态分页到 CPU 内存，避免 OOM。**作用：降低训练峰值显存。**

```
GPU + CPU内存交换
```

---

## 八、QLoRA 与 LoRA

## 块 51

## 4-bit 加载与 QLoRA 核心配置

你正在做 QLoRA 微调，核心目标是在有限显存里跑大模型。这一块就是教你如何用 `BitsAndBytesConfig` 把模型“压扁”再加载。

### 第一步：开启 4-bit 加载

```python
load_in_4bit=True
```

这行代码让模型权重以 4-bit 精度加载，而不是默认的 32-bit。与32-bit相比，显存占用降至约1/8；与16-bit相比，降至约1/4。这是 QLoRA 能跑大模型的关键。

### 第二步：选择量化类型

```python
bnb_4bit_quant_type="nf4"
```

`nf4` 是 NF4 量化（NormalFloat4），专门为神经网络权重分布设计，比普通 4-bit 量化精度损失更小。

### 第三步：开启双重量化

```python
bnb_4bit_use_double_quant=True
```

双重量化会对量化常数再做一次量化，进一步压缩显存开销。建议开启，几乎不损失精度。

### 第四步：设置计算精度（最容易踩坑）

这里要区分两个概念：

- **存储 dtype**：权重以 4-bit 存，省显存
- **计算 dtype**：实际运算时，权重会**反量化**成更高精度来计算

`bnb_4bit_compute_dtype` 控制的就是计算精度。不设置时默认使用float32，可能导致计算速度较慢和显存开销较大，建议显式指定计算精度。

**关键：不要写死 `bfloat16`！** 它需要较新的 GPU（NVIDIA Ampere 架构及以后，如 RTX 30/40 系、A100）。老 GPU（如 RTX 20 系）支持BF16但性能较差，可能影响训练速度。正确做法是自动判断：

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

**BF16 vs FP16 怎么选？** 两者都是 16

## 块 52

## 9.1 把量化配置、prepare、梯度检查点、LoRA 注入串起来（可跑通骨架）

前面我们分别看了 `BitsAndBytesConfig`（怎么 4-bit 加载）、分页优化器（`optim`）、`bnb_4bit_compute_dtype`（计算精度）。现在把它们拼成一个**能直接跑的最小骨架**，让你看到 QLoRA 训练的全貌。

### 整体流程（6步）

1. **配置 4-bit 量化**：用 `BitsAndBytesConfig` 指定 NF4 量化、双重量化、计算 dtype
2. **加载模型**：通过 `quantization_config` 参数以 4-bit 加载基础模型
3. **准备 k-bit 训练**：调用 `prepare_model_for_kbit_training`，处理 norm/输出层精度，并开启梯度检查点所需的输入梯度
4. **注入 LoRA**：用 `LoraConfig` 配置低秩增量参数，冻结原权重，仅训练这些低秩矩阵
5. **准备数据**：极简示例数据，Causal LM 的 labels 直接复制 input_ids
6. **训练**：设置分页优化器 + 梯度检查点，用 `Trainer` 启动

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
# 模型许可证：请查阅所选基础模型的许可证（如 Llama 2 Community License，商用月活>7亿需Meta授权）
# 注意：以 Llama-2-7b 为例，FP16 加载需要至少14GB显存；4-bit 量化后显存需求显著降低

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

# 3) 关键：为 k-bit 训练做准备
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # 与 TrainingArguments 的 gradient_checkpointing 对应
)

# 4) 注入 LoRA（冻结原权重，仅训练低秩增量矩阵 A 和 B）
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

# 5) 极简数据（真实场景替换为足够规模的数据）
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

### 几个关键呼应点

- **第 3 步和第 6 步的梯度检查点**：`prepare_model_for_kbit_training` 里的 `use_gradient_checkpointing=True` 和 `TrainingArguments` 里的 `gradient_checkpointing=True` 是一对，要同时开启
- **分页优化器**：在 `TrainingArguments` 中通过 `optim="paged_adamw_8bit"` 指定，依赖 bitsandbytes
- **LoRA 参数量**：LoRA 不改变原权重，而是冻结原权重并训练低秩矩阵 A 和 B 形成增量更新。以单个线性层为例，原参数为 d_in×d_out，LoRA 参数为 r(d_in+d_out)，远小于原参数量
- **计算 dtype**：优先

## 块 53

## 九、把 QLoRA 的四个关键步骤串起来

你之前已经分别了解了 4-bit 量化（NF4）、双重量化和分页优化器，现在我们把它们组装成一个**能跑通的最小骨架**。这四步的顺序是固定的，缺一不可：

**第 1 步：`BitsAndBytesConfig` —— 决定模型怎么被加载**

这一步配置基础模型以 4-bit 方式加载，包含三个核心设定：NF4 量化类型、双重量化开关、计算 dtype（BF16/FP16）。它决定了模型在内存里的"存储格式"。

**第 2 步：`prepare_model_for_kbit_training()` —— QLoRA 的"必要一步"**

这一步对量化后的模型做训练适配：把部分层转成计算精度、开启输入梯度以支持梯度检查点等。**跳过它，常见后果是训练不收敛或直接报错**。记住：这不是可选项，是必选项。

**第 3 步：`get_peft_model(..., LoraConfig)` —— 注入 LoRA**

把可训练的 LoRA 适配器"挂"到冻结的 4-bit 主干模型上。主干参数不动，只训练 LoRA 的小参数。

**第 4 步：开启梯度检查点**

在 `TrainingArguments` 中设置 `gradient_checkpointing=True` 即可启用梯度检查点，无需在 `prepare_model_for_kbit_training` 中额外设置。

**第 5 步：`optim="paged_adamw_8bit"`**

启用分页优化器，将优化器状态分页存储在 CPU 内存中，按需传输到 GPU，以缓解峰值显存压力。

> 注意：上面是"能说明关系"的最小骨架。真实训练需要换成足够规模的数据、合理的超参，并且按你实际使用的库版本核对 `optim` 名称、`prepare_model_for_kbit_training` 的参数写法。

---

## 十、你的电脑能跑吗？

**只学原理**：不需要 GPU，纯看代码和概念就行。

**实际运行 QLoRA**，最低配置参考：

| 配置项 | 最低要求 |
|--------|----------|
| CPU | 4核+ |
| 内存 | 16GB+ |
| GPU | NVIDIA CUDA GPU |

**显存需求（7B 级别模型的经验参考）**：

| 显存 | 适合场景 |
|------|----------|
| 8GB | 小模型 LoRA，或短序列/小 batch 的 7B

## 块 54

## 一、LoRA训练后保存的是什么？

### 1.1 LoRA不是直接修改基础模型

很多初学者以为LoRA训练完会得到一个“新模型”，其实不是。训练过程中，**基础模型是被冻结的**，真正被训练的是额外附加的LoRA Adapter（适配器）。

训练完保存时，通常只保存两个文件：

- `adapter_model.safetensors`（LoRA权重）
- `adapter_config.json`（配置信息）

**而不是完整模型**。这一点很关键——你训练完得到的只是“补丁”，不是“完整程序”。

### 1.2 Adapter有什么特点？

**优点：**
- 文件小，保存快
- 一个基础模型可以搭配多个Adapter，比如同一个Llama-7B，可以分别搭配客服、医疗、法律等不同任务的Adapter，互不干扰

**缺点：**
- 部署推理时，**基础模型和Adapter必须同时存在**，缺一不可

---

## 二、为什么需要合并LoRA权重？

既然Adapter这么方便，为什么还要合并？

因为部署时如果每次都要同时加载基础模型+Adapter，流程更复杂，还要额外维护Adapter文件。合并的目的，就是把“基础模型+Adapter”**融合成一个完整的模型文件**，之后部署时只需要加载这一个模型即可，不再依赖Adapter。

> **注意：** 部署合并后的模型时，需考虑显存占用。例如，Llama-2-7B FP16 需要至少14GB显存。

---

## 动手验证一下你的理解

在继续学习`merge_and_unload()`之前，先确认你掌握了本节内容：

**检查任务：** 用自己的话回答下面两个问题（写下来或说出来）：

1. LoRA训练完成后

## 块 55

**本块学习检查**：请观察下方“未合并”流程图，说出部署时你希望模型变成什么样，再对照“合并后”的公式确认你的想法。

---

**推理时的问题**：未合并时，模型是“基础模型 + LoRA适配器”两件套。部署时我们希望它变成一个完整的模型，所以需要**合并权重**。

**合并原理**：训练时，新权重 `W' = W + ΔW`，其中 `ΔW = BA`。合并就是把 `BA` 加进 `W`，得到 `W'`。合并后，原来的“W + Adapter”就变成了一个整体 `W'`。

---

**操作验证**：想象你有一个训练好的LoRA模型，部署前

## 块 56

## 3.1 merge_and_unload() 是什么？

想象你刚用LoRA微调完模型——训练时，我们并没有直接修改基础模型的全部参数，而是冻结了原始权重，在旁边挂了一个轻量的“补丁”（LoRA Adapter），只训练低秩矩阵A和B形成增量更新。现在模型用起来没问题，但每次推理都要带着这个补丁，结构上还是“基础模型 + 适配器”的组合。

`merge_and_unload()` 就是把这个补丁**真正焊死**进基础模型里，然后拆掉外面的包装。执行后，你手里拿到的就是一个**完整的、包含LoRA更新后权重的普通Transformer模型**，不再是PeftModel。

用公式看就是：

```text
基础模型 + LoRA适配器 → 完整模型
```

---

## 四、合并完整流程（先看全局）

```text
训练LoRA → 保存Adapter → 加载基础模型 → 加载Adapter → merge_and_unload() → 保存完整模型
```

下面我们一步步走。

---

## 五、加载LoRA Adapter

假设你已经有训练好的Adapter，路径存在 `adapter_path`，基础模型路径存在 `base_model_path`：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 加载基础模型
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 把LoRA适配器挂上去
model = PeftModel.from_pretrained(model, adapter_path)
```

此时模型结构是：

```text
Base Model + LoRA Adapter
```

---

## 六、执行合并（关键一步）

一行代码搞定：

```python
merged_model = model.merge_and_unload()
```

这一步做的事：

```text
W + BA → W'
```

（W是基础权重，BA是LoRA学到的增量，合并后得到更新后的权重W'）

**合并后**，`merged_model` 不再是PeftModel，而是**普通Transformer模型**——LoRA的痕迹已经融进权重里了。

---

## 七、导出合并后的模型

合并完要保存，同时**别忘了Tokenizer**：

```python
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

保存后目录大致长这样（示意）：

```text
merged_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

> 注意：实际文件会随模型/分词器类型变化，比如可能还有 `generation_config.json`、`special_tokens_map.json`，大模型可能是分片权重（`model.safetensors.index.json` + 多个分片），有些分词器是 `vocab.json`/`merges.txt` 或 `tokenizer.model`。以你实际保存出来的为准。

> 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

---

## 本块学习检查

**操作验证**：按上面流程，用你手头任意一个微调过的LoRA模型（或临时用一个小模型+随机Adapter）走一遍完整合并流程，然后回答

## 块 57

## 三、merge_and_unload是什么？（重点）

### 3.1 方法作用：合并后，你要检查这四件事

你正在做QLoRA微调，目标是把训练好的LoRA适配器合并回基础模型，得到一个“完整模型”。合并完成后，别急着欢呼——先做下面四项核查，避免“能加载但行为不对”的坑。

**① Tokenizer 完整且一致**

保存的tokenizer必须和训练时完全一致。特别注意：如果你在训练中**新增或修改过特殊token**（比如自定义的`<|im_start|>`、pad/eos等），确认它们都写进了`special_tokens_map.json`或`tokenizer_config.json`里。另外，如果训练时扩过词表，还要确认embedding和lm_head的尺寸与词表匹配——不匹配会导致加载报错或生成乱码。

**② generation_config.json**

如果你依赖特定的生成默认值（比如eos/pad token id、`max_new_tokens`、采样参数），确认这个文件一并导出了。否则推理时模型可能回退到不合适的默认行为，生成结果和你训练时的预期不一致。

**③ dtype（精度）**

确认合并/保存时的精度符合预期：FP16、BF16还是FP32？dtype不一致会影响显存占用和数值表现。QLoRA场景下尤其注意，因为量化训练后合并的精度选择会影响最终效果。

**④ config.json 完整性**

确认模型结构配置（层数、hidden size、任务头等）正确，并且**已经移除了PEFT/adapter相关配置**。合并后应该是普通Transformers模型，不应该再残留`adapter_config.json`——否则加载时可能还会走PEFT流程，而不是你想要的完整模型。

---

### 合并后如何加载推理？

合并完成后，**不再需要PEFT库**。直接使用Transformers加载：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

流程很简单：用户输入 → 完整模型 → 生成结果。

---

### LoRA合并与不合并的区别

| 项目 | 不合并 | 合并 |
|------|--------|------|
| 模型结构 | Base+Adapter | 完整模型 |
| 文件大小 | 小 | 大 |
| 部署复杂度 | 较高 | 简单 |
| 多个任务切换 | 方便 | 不方便 |
| 推理加载 | 需PEFT |

---

###

## 块 58

### 10.1 推荐合并场景

你正在做QLoRA微调，最终目标是把模型用起来。什么时候该把LoRA权重合并回基础模型？记住两个典型场景：

**场景1：上线部署（单一Adapter、追求简单）**

生产环境链路：服务器 → 直接加载模型 → 提供API。

# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
# 注意：Llama-2-7b FP16 需要至少14GB显存。

当你只需要**一个固定Adapter**、且希望部署链路尽量简单（不额外依赖PEFT、不做运行时Adapter加载）时，合并通常更方便。

> 注意：如果推理框架本身支持直接加载LoRA（例如vLLM的多LoRA服务），或你需要在同一基础模型上**热切换多个Adapter**，那么不合并、以Adapter形式部署反而更灵活；此时是否合并要看框架能力与业务需求。

**场景2：分享模型**

如果希望对方**开

## 块 59

**本块学习检查**：读完下面两种“不建议合并”的场景后，请用自己的话说：为什么保留 Adapter 比合并更灵活？然后对照你正在做的 QLoRA 微调，判断你当前属于哪种场景。

---

**10.2 不建议合并场景**

**场景1：多个任务Adapter**
如果你在一个基础模型上训练了多个 Adapter（比如 A、B、C 分别对应不同任务），**不要合并**。保留它们，你才能随时切换或组合使用，合并后就“焊死”了，失去灵活性。

**场景2：持续训练**
如果你打算继续调整 LoRA（比如还想接着训），也**保留 Adapter**。合并后就没法继续用 LoRA 方式微调了。

---

**十一、QLoRA

## 块 60

## 11.1 直接合并可能遇到的情况

你现在做 QLoRA 微调，最后一步是把 LoRA 权重合并回基础模型。如果直接在 4-bit 量化状态下执行 `merge_and_unload`，可能会碰到两类问题：

- **精度损失**：4-bit 权重需要先反量化再合并，这个过程和全程用高精度计算并不完全等价，结果会有细微偏差。
- **类型或版本不兼容**：可能出现权重类型不匹配，或者你用的 PEFT / bitsandbytes / Transformers 版本不支持在量化状态下直接合并。

**注意**：上面只是“可能”出现的情况，不是每次都会报错。能否顺利合并、合并质量如何，取决于你的版本、量化配置，以及是否先做了反量化。最终以你所用版本文档和实际运行结果为准。

## 11.2 常见做法：先反量化到高精度再合并

最稳妥的思路是：**别在 4-bit 状态下直接合并**。先以高精度（FP16/BF16）重新加载一份基础模型，再挂载同一份 Adapter，然后合并。流程如下：

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

下面是完整可执行代码，核心步骤已标注：

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

**动手验证**：跑完上面代码后，检查 `merged_out` 目录下是否生成了 `config.json`、`model.safetensors` 和 `tokenizer` 相关文件。然后用 `AutoModelForCausalLM.from_pretrained(merged_out)` 加载合并后的模型，随便输入

**模型许可证**：部署引用模型时需注意许可证要求。例如，Llama 2 Community License 规定，若商用且月活用户超过 7 亿，需获得 Meta 授权。请根据你使用的基础模型确认对应许可证条款。

## 块 61

### 直接合并QLoRA模型：先看这三个“坑”

你正在做QLoRA微调，最后一步合并模型时，别急着点运行。直接合并有3个关键点，先记住：

**1. 版本决定“能不能直接合并”**
不同PEFT版本对“量化模型能否直接合并”支持不同。有的版本要求先反量化。**关键操作**：加载基础模型时，**不要传** `quantization_config` 或 `load_in_4bit`，这样得到的是高精度权重，再合并就稳了。

**2. 内存可能不够**
用高精度重新加载完整模型，会占**约一份FP16/BF16完整模型**的内存/显存。显存不足时

## 块 62

## 合并时开启校验并做一致性验证

`merge_and_unload` 支持 `safe_merge=True` 参数，合并时自动做数值检查（比如检测 NaN），相当于给合并过程加一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

**为什么建议做一致性验证？** 合并可能引入微小偏差，所以合并后要对比合并前后模型在**同一批固定输入**上的输出，确认没出明显问题。

**验证流程（三步走）：**

1. 准备一小批固定输入（prompt）
2. 分别用「合并前的 PeftModel」和「合并后的模型」推理
3. 对比 logits / 生成文本是否基本一致

**

## 块 63

## 合并后别急着用——先做一次“一致性体检”

你已经完成了 QLoRA 模型的合并（§11.2），现在模型看起来一切正常。但这里有个关键问题：**合并后的模型真的和合并前表现一样吗？** 尤其是 QLoRA 场景下，合并过程涉及精度转换，输出可能会有微小偏差。这一节就是教你如何用代码给合并结果做一次“体检”。

### 为什么需要单独加载一份“合并前”的模型？

先看代码里的一个重要提示：

> `merge_and_unload()` 是“原地”操作，会就地修改并返回底层模型，合并后原来的 `PeftModel` 已经不再是“未合并”状态。

这句话的意思是：你在 §11.2 里执行合并后，那个 `merged_model` 对象已经被改写了。如果你拿它去和“合并前”对比，实际上是在拿“合并后”和“合并后”比——没有意义。

所以代码里专门重新加载了一份**未合并的 PeftModel** 作为对比基准：

```python
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
base_for_ref = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=compute_dtype, device_map="auto",
)
# 注意：Llama-2-7b FP16 需要至少14GB显存
model_before = PeftModel.from_pretrained(base_for_ref, adapter_path)   # 未合并：Base(高精度)+Adapter
```

这里的 `model_before` 就是“合并前”的状态：原始底座模型加上 LoRA 适配器，两者是分离的。

### 体检项目一：比较 logits（模型输出的原始分数）

模型在生成文字之前，会先计算每个词的概率分数，这个分数就叫 logits。如果合并前后模型的“想法”一致，logits 应该非常接近。

```python
with torch.no_grad():
    logits_before = model_before(**inputs.to(model_before.device)).logits
    logits_after = merged_model(**inputs).logits

max_abs_diff = (logits_before.float().cpu() - logits_after.float().cpu()).abs().max().item()
print("logits 最大绝对差:", max_abs_diff)   # 越接近 0 越好；QLoRA 场景通常是较小的近似值
```

**怎么看结果？** 看 `max_abs_diff` 这个数值——它表示两份 logits 之间最大的差异。**越接近 0 越好**。在 QLoRA 场景下，由于量化精度损失，这个值通常是一个较小的近似值，而不是精确的 0。只要差异不大，就说明合并没有破坏模型的核心能力。

### 体检项目二：比较实际生成的文本

logits 是“想法”，但最终用户看到的是“说出来的话”。所以还要对比实际生成效果。

```python
with torch.no_grad():
    gen_before = model_before.generate(**inputs.to(model_before.device), max_new_tokens=32, do_sample=False)
    gen_after = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)

text_before = tokenizer.decode(gen_before[0], skip_special_tokens=True)
text_after = tokenizer.decode(gen_after[0], skip_special_tokens=True)
```

## 块 64

## 十一、合并后的一致性验证方法（建议）

**为什么必须单独再加载一份模型？**

`merge_and_unload()` 会**原地**修改模型——它把 LoRA Adapter 直接合并进底层权重，并移除 PEFT 包装。这意味着合并后，原来的 `model` 变量已经变成了"已合并"状态。如果你拿它和合并前的自己对比，两边其实是同一份权重，`max_abs_diff` 恒为 0，验证完全失效。

**正确做法**：用 `base_for_ref` + `model_before` 重新挂载一份**未合并**的 Adapter 作为对照基准，再与合并后的模型逐层比较。具体代码示例如下：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(...)
model_before = PeftModel.from_pretrained(base_model, adapter_path)
model_after = model_before.merge_and_unload()
# 逐层比较 model_before 与 model_after 的权重差异
```

**显存不够怎么办？**

如果无法同时放下两个模型，就**分两次跑**：
1. 先用未合并的 PeftModel 跑一遍推理，把 logits 或生成文本**保存到磁盘**，然后释放显存；
2. 再加载合并后的模型，跑同样的输入，对比两次输出。

两次推理前请设置相同的随机种子（如 `torch.manual_seed(42)`），确保输入完全一致。在 QLoRA 场景下，由于量化误差，logits 的微小差异是正常的，应关注生成文本的语义一致性而非数值完全一致。

**结果预期（QLoRA 重点）**

- **QLoRA 场景**：合并前后**通常只是近似一致**。`max_abs_diff`

## 块 65

## QLoRA合并常见问题与部署选择（实操版）

你现在已经掌握了 `merge_and_unload` 的基本用法，但实际跑起来总会遇到几个“坑”。这一块我们直接对着问题表来，边看边记，遇到哪个查哪个。

### 一、最常碰到的5个问题

| 问题 | 常见原因 | 怎么排查 |
|------|----------|----------|
| `merge_and_unload` 不存在 | 当前对象不是 `PeftModel`——可能忘了用 `PeftModel.from_pretrained` 挂载 Adapter，或者已经合并完变成普通模型了 | 确认加载方式，先拿到 `PeftModel` 再调用 |
| 合并后模型变大 | **正常现象**。合并后是完整模型，不再是那个小体积的 Adapter | 不用处理；嫌大就保留 Adapter 不合并 |
| 显存不足 | 一次性加载高精度完整模型太占显存；也可能是用了 FP32、没做 offload/分片 | 改用 CPU 合并或 offload、用 FP16/BF16、分片加载 |
| Tokenizer/配置缺失 | 只存了模型没存 tokenizer；或特殊 token、`generation_config` 没一起导出 | 重新 `save_pretrained`，按 §七 核查 tokenizer、generation_config、config |
| QLoRA合并失败或结果异常 | 在 4-bit 状态下直接合并、PEFT/bitsandbytes 版本不支持、量化包装没正确反量化、内存/显存不足 | 先反量化到 FP16/BF16 再合并；核对版本；必要时用 CPU 合并；开 `safe_merge=True` 并做一致性验证 |

### 二、重点理解：“先反量化”到底是什么意思？

这里最容易误解。**“反量化”不是让你手动逐层去还原 4-bit 权重**——那太麻烦了，也不现实。

真正的做法是：**重新加载一份不带量化配置的基础模型**。具体两步：

1. 用 `AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)` 加载，**注意：不传** `quantization_config`，也**不传** `load_in_4bit`。这样拿到的就是高精度权重。
2. 再用 `PeftModel.from_pretrained` 挂载**同一份 Adapter**，然后合并。

这样合并发生在高精度权重上，天然绕开了“在 4-bit 状态下合并”带来的类型/精度问题。

## 块 66

## 路线A：合并后部署完整模型

这一路线的核心思路是：**把 LoRA 权重合并进基础模型，得到一个完整的、独立的模型文件**，之后部署时不再需要任何 LoRA 相关的概念。

### 操作流程

```text
merge_and_unload()

↓

保存完整模型

↓

推理框架按普通模型加载（如 vLLM 直接加载 merged_model）
```

第一步调用 `merge_and_unload()` 将 LoRA 权重合并进基础模型并卸载 Adapter 结构；第二步把合并后的结果保存为完整模型；第三步部署时，推理框架（比如 vLLM）直接把这个完整模型当作普通模型加载——**框架完全不需要知道 LoRA 是什么**。

### 这条路线的好处

- **部署链路简单**：推理框架按普通模型处理，无需任何 LoRA 相关的配置或理解。
- **推理时无 Adapter 切换成本**：因为 Adapter 已经被合并，推理过程中**不再有 Adapter 切换成本**。

### 代价是什么

- **存储成本高**：每个 Adapter 都要保存并分发一份**完整模型**。如果你有多个任务（多个 Adapter），就要存多份完整模型，存储开销随任务数线性增长。

---

## 对比：路线B——保留 Adapter，动态服务

与路线A相对，路线B的思路是：**基础模型只加载一份，多个 LoRA Adapter 在运行时按需挂载和切换**。一些推理框架（如 vLLM）原生支持这种 Multi-LoRA 服务模式。

```text
基础模型（只加载一份）

+

运行时挂载多个 LoRA Adapter（A / B / C）

↓

按请求指定使用哪个 Adapter
```

### 路线B的特点

- **多任务/多租户时省显存、省存储**：只需一份基础模型 + 多个小体积 Adapter。
- **支持热切换**：无需为每个任务各存一份完整模型，请求来了再指定用哪个 Adapter。
- **代价**：依赖框架对 LoRA 的支持与配置，链路比路线A略复杂；不同框架/版本对 LoRA 的支持范围和限制不同。

### vLLM 启动 LoRA 服务的大致形态（版本敏感，仅示意）

以 vLLM 为例，启动 OpenAI 兼容服务时通常需要**显式开启 LoRA 支持**并**声明每个 Adapter 的名字与路径**，命令行大致形如：

```text
vllm serve <base_model_path> \
    --enable-lora \
    --lora-modules adapterA=/path/to/adapterA adapterB=/path/to/adapterB \
    --max-loras 2 \
    --max-lora-rank 16
```

参数含义：

- `--enable-l

## 块 67

## 14.3 怎么选？两条路线，一个判断标准

你已经知道 LoRA 训练完可以合并权重，但**合并不是唯一出路**。这一节帮你建立判断框架，不涉及具体代码，先搞懂思路。

**两条路线：**

- **路线A（合并）**：把 LoRA 权重融进基础模型，导出成一个完整模型文件。部署最简单，一个模型文件搞定一切。
- **路线B（保留Adapter）**：基础模型不动，单独保存 LoRA 适配器，推理时动态加载。适合一个基础模型挂多个不同任务的 LoRA。

**选择参考（非硬性规则）：**

| 你的情况 | 倾向路线 |
|---------|---------|
| 只有一个 LoRA，追求最简部署 | A（合并） |
| 同一基础模型要服务多个 LoRA，且推理框架原生支持 LoRA | B（保留Adapter） |
| QLoRA

## 测评题目

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。
