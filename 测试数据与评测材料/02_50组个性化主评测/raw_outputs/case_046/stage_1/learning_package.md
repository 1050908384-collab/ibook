# 学习包

## 块 1

## 环境准备：根据 CUDA 版本安装 PyTorch

> **适合人群**：已完成“数据处理基础”板块，在 Ubuntu 24.04 上成功配置过 conda 虚拟环境并使用过 `nvidia-smi` 的初学者。本节目标是成功安装 PyTorch（GPU 版或 CPU 版）并验证 `torch.cuda.is_available()` 返回预期结果，为后续张量操作、自动求导和神经网络训练做好准备。
>
> **前置说明**：如果你还没有 NVIDIA 显卡，可以安装 CPU 版 PyTorch，后续大部分代码仍可运行（只是训练速度较慢）。本文会同时提供 GPU 版和 CPU 版的安装命令。

---

## 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

### 1.1 确认 conda 可用（conda/pip 虚拟环境管理 章节已学过）

```bash
conda --version
# 预期输出：conda 24.x.x
```

**本块学习检查**

在继续之前，请用最小示例回顾前置知识：

1. 确认 Python 3.10+ 已安装：`python --version`
2. 创建一个新的 conda 虚拟环境（如 `lora-env`），并激活它
3. 确认 IDE（PyCharm 或 VS Code）能识别该虚拟环境的解释器

## 块 2

好的，我们开始吧。既然你的目标是进行 LoRA 微调，那么一块能用的 NVIDIA 显卡是绝对的前提。这一步，我们就是来确认你的电脑已经“看见”了这块显卡。

在之前的“nvidia-smi 显存监控”章节，你已经学过如何使用 `nvidia-smi` 命令。现在，请打开你的终端（Terminal），输入以下命令并回车：

```bash
nvidia-smi
```

这个命令会直接告诉我们，你的系统是否成功识别了 NVIDIA 显卡及其驱动。

**接下来，请根据命令的输出结果，对号入座：**

**情况一：命令有正常输出**
如果你看到屏幕上显示了类似下面的信息：
- 一个表格，列出了你的显卡型号（例如，NVIDIA GeForce RTX 3060）
- 驱动版本（Driver Version）
- 输出中会显示 CUDA Version 信息，通常位于表格的右上角

那么恭喜你，你的环境已经准备好了！这说明 NVIDIA 驱动工作正常，我们可以直接进入下一节，开始安装 PyTorch。

**情况二：命令报错 `command not found`**
如果终端提示 `command not found`，这说明你的系统**没有安装或无法识别 NVIDIA 驱动**。这通常意味着你的电脑目前无法使用 GPU 进行加速。

遇到这种情况，你有两个选择：
1.  **安装 NVIDIA 驱动**：请参考下方“1.3 （如需要）安装 NVIDIA 显卡驱动”章节的内容进行安装。
2.  **安装 CPU 版 PyTorch**：如果你的电脑暂时无法使用 GPU，可以先安装 CPU 版本的 PyTorch 来学习 LoRA 的原理和代码逻辑，但实际的微调训练将无法进行。

---

### 1.3 （如需要）安装 NVIDIA 显卡驱动

**什么情况下需要手动安装驱动？**
- **云服务器 GPU 实例**：通常云服务商已经预装好了驱动，你不需要做任何操作。
- **本地物理机（双系统 / 独立 Linux）**：如果上一步 `nvidia-smi` 报错，那么你需要手动安装。
- **WSL2**：如果你使用的是 WSL2，Windows 主机的驱动会自动“穿透”到 WSL2 内部，一般不需要在 WSL2 里单独安装。

**针对 Linux 本地物理机的安装方法：**

最稳妥的方式是使用 Ubuntu 自带的“附加驱动”工具，这比手动下载 `.run` 文件要安全得多，可以避免因操作不当导致系统无法进入图形界面的风险。

```bash
# 1. 查看系统推荐的驱动版本
ubuntu-drivers devices

# 2. 自动安装推荐版本的

## 块 3

## 确认你的显卡驱动支持的 CUDA 版本

在开始安装 PyTorch 之前，我们需要先确认一个关键信息：你的显卡驱动支持的 CUDA 版本。这一步决定了你后续安装 PyTorch 时应该选择哪个 CUDA 版本，选错了可能会导致 PyTorch 无法正常调用 GPU。

打开终端（命令行），输入以下命令并回车：

```bash
nvidia-smi
```

你会看到类似下面的输出（这是 RTX 3060 的示例）：

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
+=========================================+======================+======================+
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              15W / 170W|    256MiB / 12288MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

**请重点关注右上角的 `CUDA Version: 12.2`**。这个数字表示你的显卡驱动当前支持的 CUDA 版本。换句话说，你安装的 PyTorch 所带的 CUDA 版本**不能超过 12.2**，否则 PyTorch 可能无法正常工作。

> **注意**：如果 `nvidia-smi` 命令报错，或者你没有 NVIDIA 显卡，请直接跳到 **第五节（CPU 版安装）**，那里会讲解如何安装不依赖 GPU 的 PyTorch 版本。

---

## 创建 PyTorch 专用 conda 虚拟环境

在安装 PyTorch 之前，我们先创建一个独立的虚拟环境，避免与其他项目的依赖冲突。这一步在之前的"虚拟环境管理"章节已经详细学过，这里快速执行即可。

打开终端，依次运行以下两条命令：

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.10
conda create -n pytorch python=3.10 --yes

# 激活环境
conda activate pytorch
```

激活成功后，你会注意到命令行提示符前面从 `(base)` 变成了 `(pytorch)`，说明你已经进入了这个虚拟环境。

## 块 4

好的，我们开始学习如何为你的 RTX 3060 安装 GPU 版 PyTorch。这是进行 LoRA 微调的第一步，也是关键一步。

你的显卡是 RTX 3060，拥有 12GB 显存，属于 Ampere 架构，性能足以运行大多数微调任务。我们的目标就是让 PyTorch 能充分利用这块显卡的算力。

**核心原则：** 安装的 PyTorch 版本所要求的 CUDA 版本，必须 **小于或等于** 你电脑驱动所支持的 CUDA 版本。

**如何查看驱动支持的 CUDA 版本？**
在终端里运行这个命令：
```bash
nvidia-smi
```
在输出信息的右上角，你会看到 `CUDA Version: xx.x`。这个数字就是你的驱动能支持的最高 CUDA 版本。请记下这个数字，我们接下来要用到它。

---

**推荐做法：从官网复制安装命令**

这是最不容易出错的方法，因为官网会自动为你匹配正确的命令。

1.  打开 PyTorch 官方安装页面：`https://pytorch.org/get-started/locally/`
2.  根据你的情况，在页面上选择：
    *   **PyTorch Build**: Stable (稳定版)
    *   **Your OS**: Linux
    *   **Package**: Pip (推荐，版本管理更灵活)
    *   **Language**: Python
    *   **Compute Platform**: 选择你刚才在 `nvidia-smi` 里看到的 CUDA 版本（或者一个**小于等于**它的版本）。
3.  页面下方会自动生成安装命令，直接复制并粘贴到你的终端里执行即可。

---

**具体选择哪个版本？**

假设你的 `nvidia-smi` 显示 `CUDA Version: 12.2`，那么你可以选择 CUDA 12.1 版本的 PyTorch，因为 12.1 ≤ 12.2。安装命令如下：

```bash
# CUDA 12.1 版本（适合驱动支持 CUDA 12.x 的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

如果你的驱动较旧，`nvidia-smi` 显示 `CUDA Version: 11.8`，那么请使用：

```bash
# CUDA 11.8 版本（适合旧显卡或驱动较旧的用户）
pip install torch==

## 块 5

你的电脑有 RTX 3060（12GB 显存），属于 NVIDIA GPU，因此**不需要**安装本节的 CPU 版 PyTorch。本节内容供你了解备用。

**CPU 版适用场景**：虚拟机、WSL2 无 GPU 穿透、Mac 等无 NVIDIA 显卡的环境。安装命令为：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

**关键认知**：CPU 版能正常学习 PyTorch 语法（张量操作、自动求导、神经网络搭建），API 与 GPU 版几乎一致，只是训练速度较

## 块 6

### 5.3 安装过程说明

PyTorch 安装包较大（约 2~3GB），下载耗时几分钟到十几分钟，取决于网速。看到进度条在走，耐心等待即可。

**针对你的情况（RTX 3060 12GB）**：RTX 3060 支持 CUDA 12.1，但需确保驱动版本不低于 530，所以直接按下面命令安装即可。

**国内网络加速提示**：若下载很慢，请先按第三节配置好全局 pip 国内镜像源，然后执行：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

**关键警告**：PyTorch 的 CUDA 版本包**必须**从官方 `download.pytorch.org` 下载，**切勿**叠加 `-i` 或 `--extra-index-url` 指定其他 PyPI 镜像源。否则 pip 会优先从镜像源拉到 CPU 版或版本

## 块 7

**快速验证：确认 PyTorch 与 CUDA 已就绪**

安装完成后，我们需要用一段简短的 Python 脚本来确认一切正常。这一步很关键，它能直接告诉你 PyTorch 是否识别到了你的 RTX 3060 显卡。

在终端中逐行输入以下命令：

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

**预期输出（GPU 版）**：

```
PyTorch 版本: 2.5.1+cu121
CUDA 是否可用: True
PyTorch 内置 CUDA 版本: 12.1
GPU 数量: 1
GPU 型号: NVIDIA GeForce RTX 3060
```

**预期输出（CPU 版）**：

```
PyTorch 版本: 2.5.1
CU

## 块 8

### 6.2 运行一个简单的张量运算（①）

环境装好了，我们来跑一段真正的 PyTorch 代码，验证核心功能是否正常。

打开终端，输入以下命令：

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

这段代码做了三件事：

1. **创建张量**：用 `torch.rand(2, 3)` 生成一个 2 行 3 列的随机张量，并打印在 CPU 上的内容。
2. **检测 GPU**：通过 `torch.cuda.is_available()` 判断当前环境是否支持 GPU。
3. **执行运算**：如果支持 GPU，把张量移到 GPU 上并乘以 2；否则在 CPU 上完成同样的运算。

**你该看到什么？**

- 如果一切正常，你会先看到 CPU 张量的内容（一组随机数）。
- 如果你的 RTX 3060 驱动和 CUDA 版本对齐正确，还会看到 GPU 张量及其乘以 2 的结果。
- 如果 GPU 不可用，程序会自动降级到 CPU 运算，并打印 CPU 结果。

**为什么这一步重要？**

对于你接下来的 LoRA 微调任务，GPU 是核心算力来源。这一步能确认 PyTorch 是否真正调用了你的 RTX 3060，而不是在 CPU 上“假装运行”。只有 GPU 路径打通，后续微调大模型时才能

## 测评题目

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB0406-001
训练循环中，`optimizer.zero_grad()` 的作用是？

### Q-KB0406-005
请写一个完整的训练循环，包含：
- 模型训练模式
- 损失计算
- 反向传播
- 每100个batch打印一次损失

### Q-KB0404-005
请写代码：使用 `DataLoader` 加载 `MyDataset`，批次大小为16，使用4个子进程，并在每个epoch开始时打乱数据。

### Q-KB0404-004
`DataLoader` 的 `num_workers` 参数作用是？

### Q-KB0407-005
请写代码：从 `checkpoint.pth` 加载模型和优化器状态，并恢复训练。

### Q-KB0403-003
请定义一个简单的神经网络类 `SimpleNet`，包含：
- 一个输入层到隐藏层的线性层（10 -> 20）
- ReLU激活
- 隐藏层到输出层的线性层（20 -> 1）
