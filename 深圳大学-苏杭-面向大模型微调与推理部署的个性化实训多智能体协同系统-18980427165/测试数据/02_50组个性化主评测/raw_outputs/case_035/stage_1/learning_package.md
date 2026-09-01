# 学习包

## 块 1

## 环境准备：根据 CUDA 版本安装 PyTorch

**本块学习检查**（先做这一步，再继续）

在开始之前，请先用一个最小示例回顾前置知识：打开终端，执行 `conda --version`，确认 conda 可用；再执行 `python --version`，确认 Python 是 3.10 或更高版本。如果你能解释“为什么需要 conda 虚拟环境”以及“pip 国内镜像源的作用”，就可以继续本块了。如果某一步报错，请回到对应章节排查。

---

### 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

#### 1.1 确认 conda 可用（conda/pip 虚拟环境管理 章节已学过）

```bash
conda --version
# 预期输出：conda 24.x.x
```

#### 1.2 确认已创建并激活虚拟环境

```bash
conda activate your_env_name
# 预期输出：命令行前缀变为 (your_env_name)
```

#### 1.3 确认 NVIDIA 驱动与 CUDA 版本（仅 GPU 版需要）

```bash
nvidia-smi
# 预期输出：显示显卡型号、驱动版本和 CUDA 版本号
```

>

## 块 2

好的，我们开始检查你的环境。这一步非常关键，因为你的目标是**LoRA微调**，而LoRA微调通常是在GPU上进行的。如果GPU环境没准备好，后面所有的代码都无法运行。

### 1.2 确认 `nvidia-smi` 有输出

请打开你的终端（Terminal），输入以下命令并回车：

```bash
nvidia-smi
```

这个命令会显示你的显卡状态。根据输出结果，你会遇到两种情况：

- **情况一：有正常输出**。你会看到类似下面的信息：顶部有显卡型号（比如 NVIDIA GeForce RTX 4060）、驱动版本（Driver Version），以及右上角的 **CUDA Version**。如果看到了这些，恭喜你，你的GPU驱动已经就绪，可以直接跳到**第二节**，开始了解显卡、驱动和CUDA的关系。
- **情况二：报错 `command not found`**。这说明系统没有找到NVIDIA驱动。你需要先解决驱动问题，或者选择安装CPU版的PyTorch（但这样LoRA微调速度会非常慢，甚至无法进行）。解决驱动问题请看下面的 **1.3 节**。

### 1.3 （如需要）安装 NVIDIA 显卡驱动

**什么情况需要这一步？**

- **云服务器GPU实例**：通常已经预装驱动，**不需要**操作。
- **本地物理机**：如果 `nvidia-smi` 报错，需要手动安装。
- **WSL2**：Windows主机的驱动会自动穿透到WSL2，一般**不需要**在WSL2内部单独安装。

**Linux本地物理机安装驱动（简单方法）**：

如果你用的是Ubuntu系统，最简单的方法是使用系统自带的“附加驱动”工具：

```bash
ubuntu-drivers devices          # 查看推荐驱动
sudo ubuntu-drivers autoinstall # 自动安装推荐驱动
```

安装完成后**重启系统**，再执行 `nvidia-smi` 验证是否成功。

> **注意**：如果你选择从NVIDIA官网下载 `.run` 文件手动安装，请务必按照官方说明操作。不要直接复制执行安装命令，因为禁用nouveau、切换图形界面等步骤因系统版本而异，操作不当可能导致系统无法进入图形界面。

---

## 二、先搞清楚：显卡、驱动、CUDA、PyTorch 四者是什么关系？

这是安装PyTorch最容易踩坑的地方。很多人装完后运行 `torch.cuda.is_available()`，发现返回 `False`，然后对着一堆版本号不知道哪里错了。我们先理清四者的依赖关系：

```
显卡硬件（如 RTX 4060）
    ↓ 决定支持的驱动版本上限
NVIDIA 显卡驱动（Driver）
    ↓ nvidia-smi 右上角显示的 CUDA Version = 驱动支持的最高 CUDA 版本
CUDA Toolkit（nvcc --version，可选）
    ↓ PyTorch 自带 CUDA Runtime，一般不需要单独装
PyTorch（内置 CUDA Runtime）+ cuDNN
```

**最关键的两句话**：

1. **`nvidia-smi` 右上角显示的 `CUDA Version`，是驱动支持的“最高”CUDA版本，不是你已经安装

## 块 3

## 本块学习检查

完成本块后，请用自己的话说明：`nvidia-smi` 输出的 `CUDA Version` 与安装 PyTorch 时选择的 CUDA 版本之间是什么关系？这对你后续做 LoRA 微调有什么影响？遇到无法解释的术语（如“驱动”“CUDA”），先记录下来，稍后提问。

---

### 一、先看懂 `nvidia-smi` 的输出

你之前学过用 `nvidia-smi` 监控显存。现在请把注意力放在**右上角**：

```
NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2
```

这里 `CUDA Version: 12.2` 表示**当前驱动兼容的 CUDA 版本**，安装 PyTorch 时选择的 CUDA 版本应不高于此版本，以确保兼容性。注意：它不是你当前已安装的 CUDA 工具包版本，而是驱动能力上限。

**关键规则**：你安装的 PyTorch 的 CUDA 版本应不高于驱动支持的 CUDA 版本，例如 cu121（CUDA 12.1）或 cu122（CUDA 12.2）均可，但需确认驱动兼容性。

> 如果 `nvidia-smi` 报错，或你根本没有 NVIDIA 显卡，直接跳到第五节（CPU 版安装）——你的实训条件（无独立 GPU 或 8GB 显存）也适用这条路径。

---

### 二、创建 PyTorch 专用虚拟环境

这一步在“conda/pip 虚拟环境管理”章节已学过，这里快速执行：

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.

## 块 4

### 安装 GPU 版 PyTorch：跟着官网走，最稳

你的目标是 LoRA 微调，而微调大模型离不开 PyTorch。这一块就是帮你把 PyTorch 装对，确保它能用上你的 NVIDIA 显卡。

**核心原则（一句话记住）**：你安装的 PyTorch CUDA 版本，必须 **小于或等于** 你电脑驱动支持的 CUDA 版本。

怎么查驱动版本？在终端输入 `nvidia-smi`，看右上角 `CUDA Version` 那一行。比如显示 `12.2`，那你装 PyTorch 时选 `12.1` 或 `12.0` 都可以（只要 ≤ 12.2）。

**推荐做法：去官网复制命令**

这是最不容易出错的方法，不用记命令：

1. 打开 PyTorch 官网安装页：`https://pytorch.org/get-started/locally/`
2. 按你的情况选：
   - **PyTorch Build**：Stable（稳定版）
   - **Your OS**：Linux
   - **Package**：Pip（推荐，版本更灵活）
   - **Language**：Python
   - **Compute Platform**：选你驱动支持的 CUDA 版本（≤ `nvidia-smi` 的结果）
3. 页面会自动生成安装命令，**直接复制粘贴到终端运行**。

**两个常见选择（以 `nvidia-smi` 显示 12.2 为例）**

- 驱动支持 CUDA 12.x，选 CUDA 12.1：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

- 旧显卡或驱动较旧（支持 11.8），选 CUDA 11.8：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

**怎么选？记住这个判断**：
- `nvidia-smi` 显示 ≥ 12.1 → 用 `cu121`
- 显示在 11.8~12.0 之间 → 用 `cu118`
- 不确定？优先 `cu121`，PyTorch 2.x 对 CUDA 12.x 支持更完善。

**

## 块 5

你的电脑没有NVIDIA显卡？没关系，CPU版PyTorch足够你学完本课程。

安装命令如下：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

CPU版能学完全部课程吗？可以。PyTorch的API（张量操作、自动求导、神经网络搭建）在CPU和GPU上几乎完全一致。后续大模型微调时CPU版会跑得慢甚至跑不动，但学习概念和调试代码完全没问题。

**本块学习检查**：完成本块后，用自己的话说明CPU版PyTorch与你的LoRA微调目标的关系；遇到无法解释的术语（如

## 块 6

### 5.3 安装过程说明

PyTorch 安装包较大（约 2~3GB），下载耗时取决于网速，看到进度条走动就耐心等待。

**国内加速技巧**：若下载慢，先按第三节配置全局 pip 国内镜像源，再执行：
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```
**关键提醒**：CUDA 版必须从官方仓库下载，**不要**叠加 `-i` 或 `--extra-index-url` 指定其他镜像，否则可能拉到 CPU 版，导致后续 `torch.cuda.is_available()` 返回 `False`。通用依赖（如 numpy）会自动走全局镜像。

---

## 六、验证安装

安装后，验证 PyTorch 能否导入及 GPU 是否可用。

**本块学习检查**：完成本块后，用自己的话说明本块内容与“LoRA 微调”目标的关系（提示：LoRA 微调需在 PyTorch 环境中运行，且 GPU 可用性影响后续实操方式

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

运行下面的脚本，确认你的PyTorch环境是否就绪：

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

**怎么看结果**：

- **GPU版**：看到`CUDA 是否可用: True`，并显示GPU型号（如RTX 3060），说明PyTorch能调用你的显卡，后续LoRA微调可加速。
- **CPU版**：看到`CUDA 是否可用: False`和`当前使用 CPU 模式`，说明PyTorch在CPU上运行。你的实训条件是无独立GPU或8GB显存，

## 块 8

好的，我们来完成这个验证步骤。既然你的目标是LoRA微调，那PyTorch环境就是一切的基础。这一步我们只做一件事：**跑一个最简单的张量运算，确认PyTorch的核心功能正常**。

请在你的终端中，确保已经激活了`pytorch`环境，然后运行下面这段代码：

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

**这段代码做了什么？**

1.  它创建了一个 2行3列的随机数张量`x`。
2.  它检查你的电脑是否能用GPU（`torch.cuda.is_available()`）。
3.  **如果有GPU**，就把张量搬到GPU上，并计算`x * 2`。
4.  **如果没有GPU**，就直接在CPU上计算`x * 2`。

**怎么看结果？**

如果你看到类似下面的输出（数字是随机的），就说明你的PyTorch环境**完全就绪**了：

```
CPU 张量:
tensor([[0.1234, 0.5678, 0.9012],
        [0.3456, 0.7890, 0.2345]])
张量运算结果（CPU）:
tensor([[0.2468, 1.1356, 1.8024],
        [0.6912, 1.5780, 0.4690]])
```

如果输出的是`GPU 张量:`，那说明你的GPU环境也配置成功了，这对你后续做LoRA微调会很有帮助

## 测评题目

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB0405-004
`CrossEntropyLoss` 的输入 `logits` 是否需要先经过 `Softmax`？

### Q-KB0401-004
`x.view(-1, 3)` 中 `-1` 的含义是？

### Q-KB0403-004
`nn.Module` 的子类中，`forward` 方法的作用是？

### Q-KB0404-003
请实现一个简单的自定义数据集类 `MyDataset`，接收数据列表 `data` 和标签列表 `labels`，实现 `__len__` 和 `__getitem__` 方法。

### Q-KB0403-005
请写代码：使用 `nn.Sequential` 定义一个包含 `Linear(10, 5)`、`ReLU`、`Linear(5, 1)` 的网络。

### Q-KB0401-002
张量 `x = torch.tensor([[1, 2], [3, 4]])`，其形状 `x.shape` 为________；数据类型 `x.dtype` 默认为________。
