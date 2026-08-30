# 学习包

## 块 1

## 环境准备：根据 CUDA 版本安装 PyTorch

**本块学习检查**（先完成再继续）：

在终端依次执行以下命令，快速确认前置环境就绪：

```bash
conda --version
# 预期输出：conda 24.x.x
```

若输出正常，说明 conda 可用。若报错，请回到“conda/pip 虚拟环境管理”章节排查。同时确认你已安装 Python 3.10+、配置好 IDE（如 PyCharm 或 VS Code）以及 pip 国内镜像源。

---

### 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

#### 1.1 确认 conda 可用

```bash
conda --version
# 预期输出：conda 24.x.x
```

#### 1.2 确认 NVIDIA 驱动与 CUDA 版本

```bash
nvidia-smi
```

查看右上角 **CUDA Version**，记下该数字（例如 12.x）。你的 RTX 3060 12GB 支持 GPU 版 PyTorch，安装时需根据此 CUDA 版本选择对应的 PyTorch 版本。

---

### 二、安装 PyTorch（GPU 版）

## 块 2

## 一、先跑一条命令，确认你的显卡“在线”

你已经学过 `nvidia-smi` 显存监控，这里直接用它做环境体检。打开终端，执行：

```bash
nvidia-smi
```

**看到什么算通过？**
- 正常输出：显示显卡型号（比如你的 RTX 3060）、驱动版本、右上角有 `CUDA Version`。看到这些，直接跳到**第二节**。
- 报错 `command not found`：说明系统没识别到 NVIDIA 驱动。先别慌，看下面的处理方案。

**你的情况（RTX 3060 12GB 本地物理机）**：如果 `nvidia-smi` 报错，需要手动装驱动。推荐用 Ubuntu 自带的“附加驱动”工具，比去官网下载 `.run` 文件安全得多：

```bash
ubuntu-drivers devices          # 查看推荐驱动
sudo ubuntu-drivers autoinstall # 自动安装推荐驱动
```

执行前请备份重要数据，并确认推荐的驱动版本与你的 CUDA 需求兼容；安装后如遇问题，可通过 `sudo ubuntu-drivers list` 查看可用版本并手动选择。装完重启，再跑 `nvidia-smi` 验证。注意：**不要**直接复制执行官网的 `.run` 安装命令，涉及禁用 nouveau、切换图形界面等操作，搞不好系统进不去图形界面。

> 小提示：云服务器 GPU 实例通常已预装驱动；WSL2 会自动穿透 Windows 的驱动，一般不用在 WSL2 里单独装。

---

## 二、显卡、驱动、CUDA、PyTorch 的关系（重点，别跳）

这是装 PyTorch 最容易翻车的地方。很多人装完跑 `torch.cuda.is_available()` 返回 `False`，就是没搞懂这四者的关系。看这张依赖链：

```
显卡硬件（RTX 3060）
    ↓ 决定驱动版本上限
NVIDIA 驱动
    ↓ nvidia-smi 右上角的 CUDA Version = 驱动支持的最高 CUDA 版本
CUDA Toolkit（可选，一般不用装）
    ↓ PyTorch 自带 CUDA Runtime
PyTorch（内置 CUDA Runtime）+ cuDNN
```

**记住两句话：**

1. **`nvidia-smi` 右上角的 `CUDA Version` 是驱动支持的“最高”CUDA 版本，不是你已经装的 CUDA 版本。** 你选的 PyTorch CUDA 版本必须 **≤** 这个值。
2. **PyTorch

## 块 3

## 一、先看你的显卡“体检报告”

打开终端，输入 `nvidia-smi`，你会看到类似下面的输出：

```
NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              15W / 170W|    256MiB / 12288MiB |      0%      Default |
```

**重点看右上角 `CUDA Version: 12.2`**——这表示你的驱动支持的最高 CUDA 版本为 12.2。你安装的 PyTorch 的 CUDA 版本应小于等于该版本（即不能超过 12.2）。

> 如果 `nvidia-smi` 报错或你没有 NVIDIA 显卡，请直接跳到 **第五节（CPU 版安装）**。

## 二、创建 PyTorch 专用 conda 虚拟环境

> 此步骤在"conda/pip 虚拟环境管理"章节已详细学过，此处快速执行。

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.10
conda create -n pytorch python=3.10 -y

# 激活环境
conda activate pytorch
```

激活成功后，命令行提示符前面会从 `(base)` 变成

## 块 4

### 安装 GPU 版 PyTorch：为你的 RTX 3060 选对命令

你已经会用 Pandas 和 NumPy 处理数据了，现在要装 PyTorch，这是迈向 LoRA 微调的第一步。别担心，安装本身不复杂，核心就一句话：**确保你的 NVIDIA 驱动版本满足 PyTorch 所选 CUDA 版本的最低要求**（例如，CUDA 12.x 需要驱动版本 ≥ 525.60.13）。通常，较新的驱动支持所有较旧的 CUDA 版本，因此选择较新的 CUDA 版本（如 cu121）通常没问题，但需检查驱动版本。

怎么知道驱动支持多少？在终端输入 `nvidia-smi`，看右上角的 “CUDA Version”。比如显示 `12.2`，那你就装 CUDA 12.1 或 11.8 的 PyTorch（只要 ≤ 12.2 就行）。

**最稳的方法：去官网复制命令**

1. 打开 https://pytorch.org/get-started/locally/
2. 按下图选：
   - **PyTorch Build**: Stable
   - **Your OS**: Linux
   - **Package**: Pip（推荐，灵活）
   - **Language**: Python
   - **Compute Platform**: 选你驱动支持的 CUDA 版本（就是 `nvidia-smi` 里看到的）
3. 页面自动生成命令，复制到终端运行即可。

**你的 RTX 3060 该选哪个？**

RTX 3060 是 Ampere 架构，驱动通常都支持 CUDA 12.x。所以大概率选 `cu121`。判断标准：

- `nvidia-smi` 显示 CUDA Version ≥ 12.1 → 用 `cu121`
- 显示在 11.8~12.0 之间 → 用 `cu118`
- 不确定 → 先检查驱动版本：若驱动支持 CUDA 12.x，则选择 cu121；若驱动较旧（支持 CUDA 11.x），则选择 cu118。两者均为官方支持，选择哪个取决于你的驱动和生态兼容性。

**示例命令（CUDA 12.1）**：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## 块 5

你的电脑有 RTX 3060（12GB 显存），属于 NVIDIA GPU，因此**不需要**安装本块的 CPU 版 PyTorch。本块内容对你而言是“了解即可”的备选方案。

CPU 版适用于无 NVIDIA 显卡的环境（如虚拟机、Mac）。它用 `pip` 安装，命令如下：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

关键点：CPU 版能正常学习 PyTorch 语法（张量、自动求导、搭建网络），API 与 GPU 版几乎一致；但大模型微调（如 LoRA）时会很慢甚至跑不动。

**本块学习检查**：完成本块后，

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyTorch 安装包约 2~3GB，下载需几分钟到十几分钟，看网速。看到进度条在走，耐心等即可。

**国内网络慢？** 先按前面章节配置好全局 pip 国内镜像源，然后只指定 PyTorch 官方 wheel 仓库：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**关键提醒**：CUDA 版 PyTorch 必须从官方 `download.pytorch.org` 下载，**不要**叠加 `-i` 或 `--extra-index-url` 指定其他 PyPI 镜像源，否则 pip 会优先从镜像源拉到 CPU 版或版本不匹配的包，导致 `torch.cuda.is_available()` 返回 `False`

## 块 7

你已会用 Pandas/NumPy 处理数据，现在要进入 PyTorch 环境验证环节。这一步是 LoRA 微调的地基——只有确认 GPU 能被 PyTorch 调用，后续训练才能真正跑在显卡上。

在终端逐行执行以下脚本：

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

**你的 RTX 3060 预期输出**：

```
PyTorch 版本: 2.5.1+cu121
CUDA 是否可用: True
PyTorch 内置 CUDA 版本: 12.1
GPU 数量: 1
GPU 型号: NVIDIA GeForce RTX 3060
```

若显示 `CUDA 是否可用: False`，则说明 PyTorch

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 8

### 6.2 运行一个简单的张量运算（①）

你已经会用 NumPy 处理数据了，PyTorch 的张量（Tensor）和它非常像——你可以把 Tensor 理解为“能跑在 GPU 上的 NumPy 数组”。这一步我们用一段小代码，验证 PyTorch 的核心功能是否真的装好了。

在终端里运行下面这段命令（直接复制整段，回车执行）：

```bash
python -c "
import torch

# 创建一个 2x3 的随机张量（和 NumPy 的 random 类似）
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

**你预期会看到什么？**

- 先打印出一个 2 行 3 列的随机数矩阵（CPU 张量）。
- 如果你的 PyTorch 是 GPU 版且驱动正常，会继续打印 GPU 张量，以及 `x_gpu * 2` 的结果——每个数字都翻倍。
- 如果没装 GPU 版，会打印 CPU 上的 `x * 2` 结果。

**怎么看结果对不对？**

- 只要能看到张量内容（一串数字矩阵），就说明 PyTorch 环境完全就绪。
- 如果你用的是 RTX 3060，且之前安装步骤正确，这里应该会走 GPU 分支，看到 `GPU 张量` 和 `GPU 运算结果` 的字样。

**小提示**：这段代码里的 `if torch.cuda.is_available()` 就是环境准备阶段最关键的检查——它返回 `True` 说明 PyTorch 能调用你的显卡。你现在跑通了它，后面做 LoRA

## 测评题目

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB0404-002
`DataLoader` 中，`batch_size=32` 表示________；`shuffle=True` 表示________。

### Q-KB0403-001
以下哪个是PyTorch中定义**线性层**（全连接层）的类？

### Q-KB0406-005
请写一个完整的训练循环，包含：
- 模型训练模式
- 损失计算
- 反向传播
- 每100个batch打印一次损失

### Q-KB0404-005
请写代码：使用 `DataLoader` 加载 `MyDataset`，批次大小为16，使用4个子进程，并在每个epoch开始时打乱数据。

### Q-KB0403-003
请定义一个简单的神经网络类 `SimpleNet`，包含：
- 一个输入层到隐藏层的线性层（10 -> 20）
- ReLU激活
- 隐藏层到输出层的线性层（20 -> 1）

### Q-KB0402-001
PyTorch中，用于**反向传播**计算梯度的方法是？
