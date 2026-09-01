# 学习包

## 块 1

## 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

### 1.1 确认 conda 可用（conda/pip 虚拟环境管理 章节已学过）

```bash
conda --version
# 预期输出：conda 24.x.x
```

> **给“老 Python 2 手”的提醒**：你熟悉的 `python` 命令可能还指向系统自带的旧版本。请务必在 conda 虚拟环境中操作，用 `conda activate 你的环境名` 进入环境后，再执行 `python --version` 确认是 3.10+。这一步不确认好，后面装 PyTorch 容易装错环境。

**本块学习检查**（先做这个，再继续）：

1. 打开终端，执行 `conda --version`，确认输出为 `conda 24.x.x` 格式。
2. 执行 `conda env list`，确认你之前创建的虚拟环境还在列表中。
3. 激活你的虚拟环境（`conda activate 环境名`），然后执行 `python --version`，确认显示的是 Python 3.10 或更高版本。
4. 如果你还没配置 pip 国内镜像源，先按前面章节的方法配置好——这能避免后面下载 PyTorch 时卡在网速

## 块 2

好的，学习者。既然我们要用 LoRA 微调模型，你的 RTX 3060 显卡就是核心工具。在开始之前，我们必须先确认它能被系统正常识别。

**第一步：检查显卡驱动是否就绪**

打开你的终端，输入以下命令并回车：

```bash
nvidia-smi
```

**请仔细观察命令的输出结果，会出现两种情况：**

1.  **正常情况**：屏幕上会显示一个表格，包含你的显卡型号（RTX 3060）、驱动版本，以及右上角有一个 `CUDA Version` 的提示。如果看到这些，恭喜你，环境准备的第一步就完成了，可以直接跳到下一节内容。
2.  **报错情况**：如果终端提示 `command not found`，这表示系统没有识别到 NVIDIA 驱动。这时你有两个选择：
    *   参考下方 **1.3 节** 的指引安装驱动。
    *   或者，暂时选择安装 **CPU 版 PyTorch** 来学习，但这样将无法利用你的 12GB 显存进行 LoRA 微调。

**（如需要）安装 NVIDIA 显卡驱动**

*   **云服务器**：通常已预装驱动，无需操作。
*   **本地物理机**：如果上一步报错，需要手动安装。最简单的方法是使用 Ubuntu 自带的“附加驱动”工具：
    ```bash
    ubuntu-drivers devices          # 查看推荐驱动
    sudo ubuntu-drivers autoinstall # 自动安装推荐驱动
    ```
    安装完成后重启系统，再执行 `nvidia-smi` 验证。
*   **WSL2**：Windows 主机的驱动会自动穿透，一般不需要在 WSL2 内部单独安装。

---

**第二步：理清四个关键概念的关系**

这是安装 PyTorch 最容易出错的地方。很多人装完运行 `torch.cuda.is_available()` 返回 `False`，就是因为没搞清下面这四者的关系。

```
显卡硬件 (RTX 3060)
    ↓ 决定驱动版本上限
NVIDIA 显卡驱动 (Driver)
    ↓ nvidia-smi 右上角显示的 CUDA Version = 驱动支持的最高 CUDA 版本
CUDA Toolkit (可选)
    ↓ PyTorch 自带 CUDA Runtime，一般不需要单独装
PyTorch (内置 CUDA Runtime) + cuDNN
```

**请务必记住这两句最关键的话：**

1.  **`nvidia-smi` 右上角的 `CUDA Version`，是驱动支持的“最高”CUDA 版本，不是你已经安装的版本。** 你安装的 PyTorch 的 CUDA 版本必须 **≤** 这个值。
2.  **如果你只是跑

## 块 3

## 一、先看懂这张表：你的显卡“说了算”

还记得你在“nvidia-smi 显存监控”章节学过的命令吗？现在打开终端，输入 `nvidia-smi`，你会看到一张表。**先别管其他数字，直接看右上角**，那里写着 `CUDA Version: 12.2`。

这个数字的意思是：**nvidia-smi 右上角的 CUDA Version 表示当前驱动兼容的 CUDA 运行时版本**。换句话说，你后面安装 PyTorch 时，选的 CUDA 版本**不能高于这个版本**——选 12.1、12.0 都可以，但选高于驱动支持的 CUDA 版本（如未来发布的更高版本）可能无法正常工作，建议选择等于或低于驱动支持的 CUDA 版本。

你可能会问：为什么非要看这个？因为 PyTorch 是靠 CUDA 来调用显卡算力的，版本对不上，轻则装不上，重则装上了也用不了 GPU。所以这一步是“地基”，地基歪了，后面 LoRA 微调的大楼就盖不起来。

另外注意：如果 `nvidia-smi` 报错，或者你根本没有 NVIDIA 显卡，那就别折腾了，直接跳到第五节，走 CPU 版安装路线。你的 RTX 3060 是 12GB 显存，跑 LoRA 微调完全够用，所以这一步对你来说应该很顺利。

## 二、创建 PyTorch 专用环境（快速过）

这一步在“conda/pip 虚拟环境管理”章节已经详细学过，这里只做快速执行。打开终端，依次输入：

```bash
conda create -n pytorch python=3.10 -y
conda activate pyt

## 块 4

### 安装 GPU 版 PyTorch：从“只会 Python2”到跑通现代生态

你之前主要写 Python2 和老式脚本，对现代 Python 生态（比如 `pip`、虚拟环境）可能有点陌生。别担心，这一节只讲一件事：**怎么把 PyTorch 装对**。装对了，后面做 LoRA 微调才有地基。

#### 核心原则（先记住这一条）

**PyTorch 的 CUDA 版本 ≤ `nvidia-smi` 显示的 CUDA Version。**

这句话的意思是：你的显卡驱动支持某个 CUDA 版本（用 `nvidia-smi` 命令查看），而 PyTorch 是依赖 CUDA 来调用显卡的。只要 PyTorch 要求的 CUDA 版本不高于你驱动支持的版本，就能正常工作。**版本号小一点没关系，大了不行。**

#### 最不容易出错的做法：去官网复制命令

自己记命令容易错，推荐直接去 PyTorch 官网生成安装命令：

1. 打开 https://pytorch.org/get-started/locally/
2. 按实际情况选：
   - **PyTorch Build**：Stable（稳定版，推荐）
   - **Your OS**：Linux
   - **Package**：Pip（推荐，版本更灵活）
   - **Language**：Python
   - **Compute Platform**：选你驱动支持的 CUDA 版本（≤ `nvidia-smi` 显示的版本）
3. 页面会自动生成命令，**直接复制粘贴到终端执行**。

#### 具体怎么选版本？

先打开终端，运行 `nvidia-smi`，看右上角的 `CUDA Version`。假设显示 `CUDA Version: 12.2`，那你可以选 CUDA 12.1 的 PyTorch（因为 12.1 ≤ 12.2）。以下命令中的版本为示例，请以官网当前最新稳定版为准：

```bash
# CUDA 12.1 版本（适合驱动支持 CUDA 12.x 的用户）
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
```

如果你的驱动只支持 CUDA 11.8（比如旧显卡），就用：

```bash
# CUDA 11.8 版本（适合旧显卡或驱动较旧的用户）
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

## 块 5

你的电脑若配有NVIDIA GPU（如RTX 3060），则**不需要**安装本块的CPU版PyTorch，请参考上一节《有NVIDIA GPU：安装GPU版PyTorch》操作。

不过，为了让你理解这条命令的含义（未来在无GPU机器上调试时可能用到），简要说明：

- 它通过`pip`安装CPU版的`torch`、`torchvision`、`torchaudio`三个包，版本号分别为2.5.1、0.20.1、2.5.1。
- `--index-url`指定了PyTorch官方CPU版下载源。

**学习检查**：请确认你的安装命令中`--index-url`指向的是GPU版（如`

## 块 6

## 5.3 安装过程说明

PyTorch 安装包体积较大（约 2~3GB），下载需要几分钟到十几分钟，取决于网络速度。安装过程中如果看到进度条在走，耐心等待即可。

**安装太慢？** 如果你在国内网络环境下速度很慢，建议先按本章节第三节的方法配置好全局 pip 国内镜像源，然后仅指定 PyTorch 官方 wheel 仓库即可：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

**关键提醒**：PyTorch 的 CUDA 版本包必须从官方 `download.pytorch.org` 仓库下载，请勿叠加 `-i` / `--extra-index-url` 指定其他 PyPI 镜像源，否则 pip 会优先从镜像源拉到 CPU 版或版本不匹配的包，导致 `torch.cuda.is_available()` 返回 `False`。通用依赖（如 numpy）会自动走你已配置的全局镜像

## 块 7

## 6.1 快速验证脚本：确认 PyTorch 能“看见”你的显卡

你已经装好了 PyTorch，但怎么知道它真的能用上你的 RTX 3060？下面这段脚本就是“体检报告”。请在终端里逐行执行（注意是**逐行**，不是一次性粘贴）：

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

**这段脚本在做什么？** 它先问 PyTorch“你是什么版本”，再问“你能用 CUDA 吗”。如果能，就进一步报告 CUDA 版本、GPU 数量和型号；如果不能，就告诉你现在跑在 CPU 上。

**你（GPU 版）应该看到：**

```
PyTorch 版本: 2.5.1+cu121

## 块 8

### 6.2 运行一个简单的张量运算（①）

这一步是**验证 PyTorch 核心功能是否正常**的关键动作。你只需要在终端里执行下面这段 Python 代码，它会完成三件事：创建张量、尝试调用 GPU、做一次乘法运算。

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

**这段代码在做什么？** 用你熟悉的 Python 2 思维来理解：

- `torch.rand(2, 3)` 生成一个 2 行 3 列的随机数矩阵，这就是“张量”。
- `torch.cuda.is_available()` 是一个**判断开关**，类似你以前写的 `if` 条件判断。
- 如果返回 `True`，就把张量搬到 GPU 上（`.cuda()`），并在 GPU 上做乘以 2 的运算；否则就在 CPU 上做。

**你该看到什么？** 如果环境正常，会依次输出三部分内容：CPU 张量的数字矩阵、GPU 张量的数字矩阵（前提是 GPU 可用）、以及每个元素乘以 2 后的结果。只要看到这些数字正常打印出来，就说明 PyTorch 环境**完全就绪**。

> **特别提醒**：你的机器是 RTX 3060（12GB 显存），属于 Ampere 架构，完全支持 CUDA。所以正常情况下，`torch.cuda.is_available()` 应该返回 `True`，代码会走 GPU 分支。如果它返回 `False`，请对照下一节的“常见问题与解决”表格逐一排查——最常见的原因是装了 CPU 版 PyTorch，或者 CUDA 版本超出驱动支持上限。

---

**本块学习检查**

## 测评题目

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB04-ST0400-001
安装带 CUDA 支持的 PyTorch 时，应使 PyTorch 所支持的 CUDA 版本与本机________版本兼容。

### Q-KB0404-005
请写代码：使用 `DataLoader` 加载 `MyDataset`，批次大小为16，使用4个子进程，并在每个epoch开始时打乱数据。

### Q-KB0406-003
请补全以下训练循环代码：
```python
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        # 前向传播
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        # 反向传播
        ________
        # 更新参数
        ________
```

### Q-KB0405-003
请写代码：为模型 `model` 创建 `AdamW` 优化器，学习率 `1e-4`，权重衰减 `1e-2`。

### Q-KB0403-001
以下哪个是PyTorch中定义**线性层**（全连接层）的类？

### Q-KB0407-003
请写代码：保存模型 `model` 的参数到 `checkpoint.pth`，并包含优化器状态和当前epoch。

### Q-KB0401-001
以下哪个命令可以创建一个全为0的3x3张量？
