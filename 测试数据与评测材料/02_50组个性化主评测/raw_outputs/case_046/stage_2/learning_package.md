# 学习包

## 块 1

## 本块学习检查

在开始之前，请先快速回顾前置知识：确认你已安装 Python 3.10+，配置好 conda 虚拟环境，安装了 IDE（如 PyCharm 或 VS Code），并配置了 pip 国内镜像源。若这些尚未完成，请先回到对应章节补齐，再继续本块。

---

## 环境准备：根据 CUDA 版本安装 PyTorch

**适合人群**：已完成"数据处理基础"板块，在 Ubuntu 24.04 上成功配置过 conda 虚拟环境并使用过 `nvidia-smi` 的初学者。本节目标是成功安装 PyTorch（GPU 版或 CPU 版）并验证 `torch.cuda.is_available()` 返回预期结果，为后续张量操作、自动求导和神经网络训练做好准备。

**前置说明**：如果你还没有 NVIDIA 显卡，可以安装 CPU 版 PyTorch，后续大部分代码仍可运行（只是训练速度较慢）。本文会同时提供 GPU 版和 CPU 版的安装命令。

---

## 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

### 1.1 确认 conda 可用（conda/pip 虚拟环境管理 章节已学过）

```bash
conda --version
# 预期输出：conda 24.x.x
```

> **你的实训条件**：RTX

## 块 2

## 1.2 确认 nvidia-smi 有输出

你的目标是 LoRA 微调，而微调的前提是 GPU 能被 PyTorch 正确调用。第一步，先确认驱动是否就绪。

在终端执行：

```bash
nvidia-smi
```

**判断标准：**

- **有正常输出**：能看到显卡型号（你的情况是 RTX 3060）、驱动版本和右上角的 `CUDA Version`，说明驱动没问题，直接进入第二节。
- **报错 `command not found`**：说明系统没识别到 NVIDIA 驱动。此时有两个选择：先安装驱动（见下方 1.3 节），或者改装 CPU 版 PyTorch（但你的 RTX 3060 有

## 块 3

## 一、先看懂 nvidia-smi 的输出

运行 `nvidia-smi` 后，你会看到类似下面的表格。**先别被密密麻麻的数字吓到**，我们只需要盯住两个关键位置：

```
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              15W / 170W|    256MiB / 12288MiB |      0%      Default |
```

**重点①：右上角的 `CUDA Version: 12.2`**

这一行表示你的**显卡驱动当前支持的 CUDA 版本为 12.2**。安装 PyTorch 时，建议选择与该版本匹配的 CUDA 版本，以避免兼容性问题。如果选了更高版本（比如 12.4），PyTorch 可能无法调用 GPU，甚至直接报错。

**重点②：显存信息**

看 `Memory-Usage` 一列：`256MiB / 12288MiB`。你的 RTX 3060 有 **12GB 显存**，当前只用了 256MiB，说明显存充足。这对你后续做 LoRA 微调是个好消息——12GB 显存可以运行 7B 量级模型的 LoRA 训练，但需使用 4-bit 量化（如 QLoRA）或选择更小的模型（如 3B/1B）。

> **如果 `nvidia-smi` 报错**

## 块 4

## 5.1 有 NVIDIA GPU：安装 GPU 版 PyTorch

你的目标是 LoRA 微调，而 LoRA 微调的前提是 PyTorch 能调用你的 GPU。好消息是，你的 RTX 3060（12GB 显存，Ampere 架构）完全支持 GPU 版 PyTorch，而且安装过程并不复杂。

**核心原则（务必记住）**：PyTorch 的 CUDA 版本必须 **≤** 你显卡驱动支持的 CUDA 版本。这个“驱动支持的 CUDA 版本”可以通过在终端运行 `nvidia-smi` 命令查看，输出中会显示 `CUDA Version: xx.x`。

### 推荐做法：去官网复制命令

这是最不容易出错的方式，因为官网会根据你的选择自动生成正确的安装命令。

**操作步骤**：

1. 打开 PyTorch 官方安装页面：`https://pytorch.org/get-started/locally/`
2. 按你的实际情况选择：
   - **PyTorch Build**：Stable（稳定版，推荐）
   - **Your OS**：Linux
   - **Package**：Pip（推荐，版本更灵活）
   - **Language**：Python
   - **Compute Platform**：选择你驱动支持的 CUDA 版本（即 ≤ `nvidia-smi` 显示的 CUDA Version）
3. 页面会自动生成安装命令，**直接复制粘贴到终端执行**即可。

### 如何选择 CUDA 版本？

假设 `nvidia-smi` 显示 `CUDA Version: 12.2`，那么你可以选择 CUDA 12.1 版本的 PyTorch（因为 12.1 ≤ 12.2）：

```bash
# CUDA 12.1 版本（适合驱动支持 CUDA 12.x 的用户）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

如果你的驱动只支持 CUDA 11.8（较旧显卡），则使用：

```bash
# CUDA 11.8 版本（适合旧显卡或驱动较旧的用户）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

**判断标准**：
- `n

## 块 5

你的电脑有 RTX 3060（12GB 显存），属于 NVIDIA GPU，因此**不需要**安装本块的 CPU 版 PyTorch。本块内容仅适用于无 NVIDIA 显卡的环境（如虚拟机、WSL2 无 GPU 穿透、Mac）。

CPU 版安装命令为：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

**关键认知**：CPU 版与 GPU 版的 PyTorch API（张量操作、自动求导、神经网络搭建）几乎完全一致。CPU 版能正常学习语法和调试代码，但大模型微调

## 块 6

## 5.3 安装过程说明

PyTorch 安装包体积较大（约 2~3GB），下载需要几分钟到十几分钟，取决于网络速度。安装过程中如果看到进度条在走，耐心等待即可。

**针对你的情况（RTX 3060 12GB，Ampere 架构）：** 你后续要做 LoRA 微调，GPU 驱动和 CUDA 版本匹配是前提，所以安装时务必确认下载的是 CUDA 12.1 对应的 wheel 包，而不是 CPU 版。

**安装太慢？** 如果你在国内网络环境下速度很慢，建议先按本章节第三节的方法配置好全局 pip 国内镜像源，然后仅指定 PyTorch 官方 wheel 仓库即可：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**注意：** PyTorch 的 CUDA 版本包必须从官方 `download.pytorch.org` 仓库

## 块 7

### 6.1 快速验证安装：用一段脚本确认 PyTorch 就绪

既然你的目标是 LoRA 微调，而微调的第一步是确保 PyTorch 能用上你的 GPU——RTX 3060。下面这段脚本就是“体检报告”，逐行复制到终端执行即可。

```bash
python -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('CUDA 是否可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('PyTorch 内置 CUDA 版本:', torch.version.cuda)
    print('GPU 数量:', torch.cuda.device_count())
    print('GPU 型号:', torch.cuda.get_device_name(0))
else:
    print('当前使用 CPU 模式')
"
```

**怎么看结果？** 如果你看到类似下面的输出，说明安装成功，GPU 已被 PyTorch 识别：

```
PyTorch 版本: 2.5.1+cu121
CUDA 是否可用: True
PyTorch 内置 CUDA 版本: 12.1
GPU 数量: 1
GPU 型号: NVIDIA GeForce RTX 3060

## 块 8

### 6.2 运行一个简单的张量运算（①）

既然你的目标是 **LoRA 微调**，那 PyTorch 的 GPU 能力就是你的“发动机”。这一步我们用一个极简的测试，确认这台发动机真的能点火。

在终端里运行下面这段代码：

```bash
python -c "
import torch

# 创建一个 2x3 的随机张量
x = torch.rand(2, 3)
print('CPU 张量:')
print(x)

# 如果有 GPU，把张量移到 GPU 上
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print('\nGPU 张量:')
    print(x_gpu)
    print('\n张量运算结果（GPU）:')
    print(x_gpu * 2)
else:
    print('\n张量运算结果（CPU）:')
    print(x * 2)
"
```

**这段代码在做什么？**

1. **`torch.rand(2, 3)`**：生成一个 2 行 3 列的随机数张量，这是 PyTorch 最基础的数据结构。
2. **`x.cuda()`**：把张量从 CPU 内存搬到 GPU 显存。你的 RTX 3060 有 12GB 显存，LoRA 微调时模型和优化器状态都会放在这里。
3. **`x_gpu * 2`**：在 GPU 上做一次简单的乘法运算。如果输出正常，说明 GPU 计算链路是通的。

**你预期看到什么？**

- 如果一切正常，你会看到 **CPU 张量**、**GPU 张量** 和 **GPU 运算结果** 三部分输出，且 GPU 张量内容与 CPU 张量一致（只是数值随机）。
- 如果 `torch.cuda.is_available()` 返回 `False`，程序会走 `else` 分支，只输出 CPU 结果——这说明 GPU 没被 PyTorch 识别，需要回到上一节排查。

**为什么这对 LoRA 微调很重要？**

LoRA 微调的核心是在 GPU 上反复执行矩阵运算（前向传播、反向传播、参数更新）。如果这一步的 `x_gpu * 2` 能跑通，说明 PyTorch 已经能调用你的 RTX 3060 进行张量运算——这正是后续微调的基础。如果这一步失败，后面所有训练代码都会

## 测评题目

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB0401-005
给定张量 `x` 形状为 `(2, 3, 4)`，请写出将其转置为 `(4, 3, 2)` 的代码。

### Q-KB0401-001
以下哪个命令可以创建一个全为0的3x3张量？

### Q-KB0403-003
请定义一个简单的神经网络类 `SimpleNet`，包含：
- 一个输入层到隐藏层的线性层（10 -> 20）
- ReLU激活
- 隐藏层到输出层的线性层（20 -> 1）

### Q-KB0401-003
请将以下两个张量按**列方向**（dim=1）拼接：
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5], [6]])
```

### Q-KB0405-005
请写代码：实现带梯度裁剪的训练步骤，使用 `AdamW`，学习率 `5e-5`，梯度裁剪阈值为1.0。

### Q-KB0406-002
典型的训练循环顺序是：前向传播 → 计算损失 → ________ → 更新参数。
