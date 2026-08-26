# 学习包

## 块 1

## 一、为什么需要参数高效微调？

### 1.1 全参数微调的问题

先看传统大模型训练是怎么工作的：

```text
输入数据
    ↓
整个模型参数更新
    ↓
保存完整模型
```

以 7B 参数模型为例，就是 **70 亿个参数**。训练时，你不仅要保存这些参数本身，还要额外保存**梯度**和**优化器状态**。

这意味着什么？显存需求会非常高——普通消费级 GPU 根本扛不住。

**动手验证一下**：打开你的 GPU 监控工具（比如 `nvidia-smi`），看看当前显存占用。再想想：如果 70 亿参数 + 梯度 + 优化器状态全塞进显存，你的卡够用吗？

---

**为什么这对你很重要**：你的目标是 LoRA 微调，而 LoRA 正是为了解决"全参数微调太吃显存"这个问题而生的。理解了这个痛点，你才能明白后面为什么要装 PEFT 和 bitsandbytes。

**下一步**：接下来我们会讲 LoRA 与 QLoRA 的区别，以及如何搭建环境。

## 块 2

**1.2 参数高效微调（PEFT）——动手理解**

PEFT，即参数高效微调。核心思想一句话：**不动整个大模型，只训练少量新增参数**。

流程上，原始大模型加上LoRA适配器参数，然后只训练LoRA参数，最后保存小规模Adapter。

对比三种方式：

| 方式 | 训练参数 | 显存需求 |
|------|----------|----------|
| 全参数微调 | 全部参数 | 非常高 |
| LoRA | 少量参数 | 较低 |
| QLoRA | 量化模型+LoRA | 更低 |

**本块学习检查**：对照表格，说出LoRA相比全参数微调在训练参数和显存需求上的差异

## 块 3

**本块学习检查**：读完本块后，请合上资料，用一句话向自己解释“LoRA为什么能省显存？”（提示：只训练两个小矩阵，不碰原模型）。

---

### 2.1 LoRA是什么？（①）

LoRA全称**Low-Rank Adaptation**（低秩适配）。核心思路：**原模型权重 `W` 冻结不动**，在旁边新增一个更新量 `ΔW`，最终效果相当于 `W + ΔW`。

这个 `ΔW` 不是随便加的，它被拆成两个小矩阵的乘积：

```
ΔW = B × A
```

- 若原权重 `W` 形状是 `d_out × d_in`，秩为 `r`
- 则 `A` 形状约 `r × d_in`，`B` 形状约 `d_out × r`
- 相乘后 `B × A` 还原成与 `W` 同形状的 `ΔW`

> 注意：不同资料可能写 `A × B`，只是符号

## 块 4

## 3.1 LoRA 的问题：模型还是要“全量”装进显存

你已经知道 LoRA 能大幅减少**训练参数**，但这里有个关键点：**基础模型本身仍然需要完整加载到显存里**。

举个例子，一个 7B 参数的模型，如果用 FP16 精度加载，大约需要 **14GB 显存**。也就是说，不管 LoRA 让训练参数变得多小，这个“底座”模型占用的显存是省不掉的。

> **动手感受一下**：如果你手头有 GPU，可以试着用 `nvidia-smi` 看一下当前显存占用，再想想 14GB 这个数字——这就是 LoRA 在显存上的“底线”。

---

## 3.2 QLoRA 的解法：把底座模型“压缩”后再训练

QLoRA（Quantized LoRA，量化版 LoRA）的思路很直接：既然底座模型省不掉，那就把它**变小**。

核心流程：

```text
FP16模型 → 4bit量化 → 显存占用大幅下降 → 加载LoRA进行训练
```

训练时的结构：

```text
4bit基础模型 + LoRA参数 → 训练
```

简单说：**底座模型用 4bit 精度存，LoRA 参数照常训练**。这样既保留了 LoRA 的高效训练，又把底座模型的显存开销砍了下来。

---

## 3.3 关键依赖：bitsandbytes 负责“量化”这件事

QLoRA 的实现需要几个库配合：

- `transformers`：模型加载与训练框架
- `peft`：LoRA 等参数高效微调工具
- `bitsandbytes`：**核心量化库**，负责低精度量化、4bit 加载和 8bit 加载

其中 `bitsandbytes` 是让“4bit 加载”成为可能的关键组件。

> **注意（重要）**：本教程的示例

## 块 5

**本块学习检查**：读完本块后，请对照你的电脑配置，写下“我的显存是___GB，按经验参考值，我能跑___（LoRA最低/QLoRA参考下限/QLoRA推荐）”，并说明理由。

---

### 电脑配置批注（针对LoRA微调目标）

你目标是跑通LoRA微调，且偏好动手实操。这里直接给你**能落地的配置门槛**，别被术语吓到。

**LoRA最低配置（先看这个，满足它你就能开始）**
- **CPU**：4核以上
- **内存**：16GB+
- **磁盘**：50GB以上
- **GPU**：推荐NVIDIA
- **显存**：最低8GB，推荐12GB+

**QLoRA要求（显存不够时的省钱方案）**
QLoRA的核心价值：**用更小显存跑大模型**（通过4-bit量化基础模型，显存占用通常低于LoRA全精度加载）。但注意，本教程实训范围限定为**NVIDIA CUDA GPU**，命令也按此给出；其它后端请以bitsandbytes官方说明为准。

**显存经验参考（7B级别模型）**
- 参考下限：约8~12GB（短序列、小batch）
- 推荐：16GB+

**关键提醒：别把上面的数值当死门槛**
实际显存占用受这些**测试条件**影响，条件变了结论就变：
- **模型规模**：7B级别；13B/70B显存需求显著上升
- **序列长度**：短序列（≤512）省显存；越长激活值增长越明显
- **batch大小**：小batch（1~2）省显

## 块 6

## 5.1 Python版本检查

先确认你的Python版本，打开终端运行：

```bash
python --version
```

建议使用 **Python 3.10+**。如果版本过低，后续安装PEFT和bitsandbytes时可能会遇到兼容性问题。

---

## 5.2 PyTorch检查（关键一步）

**为什么重要？** LoRA/QLoRA依赖PyTorch，而且`bitsandbytes`的4-bit量化功能要求安装**与你本机CUDA匹配的GPU版PyTorch**。直接`pip install torch`通常装到CPU版，QLoRA会无法使用。

请到 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 选择对应命令。例如NVIDIA CUDA 12.1：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**验证PyTorch能否看到GPU**，运行：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())   # 应为 True
print(torch.version.cuda)          # 查看torch编译用的CUDA版本
```

`torch.cuda.is_available()`返回`True`表示PyTorch侧能用GPU。**注意**：这并不保证bitsandbytes的4-bit量化路径可用，后者还需要bitsandbytes正确匹配本机CUDA，后续章节会专门检查。

---

## 5.3 查看GPU信息

终端运行：

```bash
nvidia-smi
```

输出包含：GPU型号、CUDA版本、显存大小。例如：

```text
RTX 4090
CUDA 12.x
Memory 24GB
```

---

## 六、安装PEFT库

### 6.1 PEFT是什么？

PEFT是Hugging Face官方的**参数高效微调库**，支持LoRA、Prefix Tuning、Prompt Tuning、Adapter等方法。

### 6.2 安装PEFT

```bash
pip install peft
```

验证安装：

```python
import peft
print(peft.__version__)
```

能打印版本号即安装成功。

---

## 七、安装bitsandbytes

> 本块原始资料未给出bitsandbytes的具体安装命令和验证方法，相关内容将在后续章节（第八章、第十三章）展开。当前你只需知道：**bitsandbytes是QLoRA实现4-bit量化的关键依赖库**。

安装bitsandbytes：

```bash
pip install bitsandbytes
```

验证安装：

```python
import bitsandbytes
```

## 块 7

**7.1 bitsandbytes 在 QLoRA 里的角色**

你要做 LoRA 微调，而 QLoRA 是它的进阶版——先把大模型压成 4-bit 再训练。`bitsandbytes` 就是干这个压缩的库：负责 8-bit 和 4-bit 量化，是 QLoRA 训练的前置依赖。

**注意**：本教程统一用 `bitsandbytes` + NVIDIA CUDA 方案，所以“必须安装”是针对这套流程。QLoRA 本身不绑定某个量化库，你以后用别的后端，以对应文档为准。

**7.2 安装**

```bash
pip install bitsandbytes
```

---

**本块学习检查**：请口头回答

## 块 8

**为什么必须检查CUDA？**

你的目标是跑通LoRA微调，而bitsandbytes是量化的核心工具。它依赖CUDA来调用GPU计算。如果CUDA版本不匹配，后果很直接：CUDA不可用、量化失败、加载模型直接报错。所以，动手前先花1分钟确认环境，能省下后面排查的大把时间。

**两步检查法（跟着做）**

1. **查CUDA版本**：打开终端，运行：
   ```bash
   nvidia-smi
   ```
   在输出末尾找到`CUDA Version`，记下这个数字。

2. **查bitsandbytes是否匹配**：接着运行：
   ```bash
   python -m bitsandbytes
   ```

## 块 9

**本块学习检查**：在终端运行 `python -c "import bitsandbytes"`，若报错提示 CUDA 动态库加载失败，请记录错误信息，并对照下方“可能问题”逐项排查。

---

### 9.1 推荐使用WSL2

你的目标是快速跑通LoRA微调，建议直接走这条路径：

**Windows → WSL2 Ubuntu → CUDA环境 → bitsandbytes**

原因很简单：Linux环境兼容性更好，能少踩很多坑。

---

### 9.2 Windows原生可能问题

如果你坚持在Windows原生环境跑，最常遇到的是：

**问题1：bitsandbytes无法加载CUDA动态库**

这不是单一原因，常见的有三种：

- 你装的 `bitsandbytes` 版本和本机 CUDA / GPU 版

## 块 10

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：CUDA版本不匹配**

在Windows上跑LoRA，最常遇到的坑就是CUDA版本对不上。别慌，按下面三步排查：

**第一步：查看驱动和CUDA版本**

打开终端，运行：

```bash
nvidia-smi
```

这个命令会告诉你三件事：
- **NVIDIA驱动**是否正常安装
- 驱动支持的**CUDA版本**（注意：这是驱动支持的最高版本）
- GPU型号和显存使用情况

**第二步：确认PyTorch版本**

记住一个关键点：**PyTorch的CUDA版本必须和驱动兼容**。驱动支持的CUDA版本可以高于PyTorch要求的版本，但不能低于。

---

## 十、安装完整LoRA环境

如果你已经有基础环境（transformers、datasets、accelerate），直接加装LoRA核心库：

```bash
pip install peft bitsandbytes
```

**完整安装命令**（一步到位）：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

> **⚠️ 前置提醒**：确保已安装**匹配本机CUDA的GPU版PyTorch**，否则后面`import torch`或QLoRA 4-bit会失败。CUDA 12.1示例：
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 十一、配置Accelerate

运行配置向导：

```bash
accelerate config
```

单GPU环境推荐选择：
- Compute environment：`This machine`
- Distributed：`No`
- Mixed precision：`fp16`

配置完成后会生成`default_config.yaml`文件。

---

## 十二、LoRA环境验证实验

创建`test_lora_env.py`文件，写入：

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

**成功标准**：看到`CUDA: True`且能打印出LoRA配置，说明：
- ✅ PyTorch正常
- ✅ PEFT正常
- ✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验

测试bitsandbytes是否正常工作：

```python
import torch
import bitsandbytes as bnb

# 检查bitsandbytes是否可用
print("bitsandbytes:", bnb.__version__)

# 测试4-bit量化加载
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config
```

> **模型许可证**：本部署示例默认使用Llama 2模型，其许可证为**Llama 2 Community License**。若商用且月活跃用户超过7亿，需获得Meta的额外授权。请根据实际使用的模型确认对应许可证条款。

## 块 11

## 九、Windows用户特别注意事项（新增）

### 9.1 推荐使用WSL2

**问题2：Windows报错**

如果你在Windows上遇到兼容性问题，推荐使用WSL2来解决。

---

## 十四、常见问题排查

| 问题 | 原因 | 解决 |
| ---- | ---- | ---- |
| bitsandbytes加载失败 | CUDA不匹配 | 运行`nvidia-smi`检查CUDA版本，安装匹配的CUDA版bitsandbytes |
| CUDA unavailable | PyTorch安装CPU版本 | 运行`pip install torch --index-url https://download.pytorch.org/whl/cu118`重新安装CUDA版PyTorch |
| 显存不足 | 模型过大 | 降低batch size或使用梯度累积 |
| Windows报错 | 兼容性问题 | 使用WSL2 |
| QLoRA速度慢 | GPU性能不足 | 升级GPU |

---

## 十五、完整LoRA/QLoRA学习路线

**当前环境：**

Transformers → Datasets → Accelerate → PEFT → LoRA → bitsandbytes → QLoRA

**后续学习：**

加载基础模型 → 准备训练数据 →

## 测评题目

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1202-001
LoRA 中 rank r 决定低秩更新矩阵的________。

### Q-KB12-ST1200-002
QLoRA 的 4-bit 量化常依赖的库是________。

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。
