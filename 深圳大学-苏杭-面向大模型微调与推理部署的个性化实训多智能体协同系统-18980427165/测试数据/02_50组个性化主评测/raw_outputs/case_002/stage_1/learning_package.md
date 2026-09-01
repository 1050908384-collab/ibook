# 学习包

## 块 1

**本块学习检查**：在继续之前，请先用一个最小示例快速回顾前置知识：在终端运行 `conda --version`，确认能输出 conda 版本号；再执行 `conda create -n test python=3.10 -y` 创建一个临时虚拟环境，并激活它（`conda activate test`）。如果你能解释这几条命令各自的作用（创建环境、指定 Python 版本、激活环境），说明前置配置已就绪，可以继续本块；若报错，请回到“conda/pip 虚拟环境管理”章节排查。

---

## 一、前置检查：确认前面章节的配置（不重复教学）

以下步骤在前面板块已经详细学过，本节只做**快速确认**。如果某一步报错，请回到对应章节排查。

### 1.1 确认 conda 可用（conda/pip 虚拟环境管理 章节已学过）

```bash
conda --version
# 预期输出：conda 24.x.x
```

**说明**：你已掌握基础 Python 并写过脚本，这里只需确认 conda 命令能被系统识别。若输出版本号（如 `conda 24.1.2`），说明环境管理工具正常；若提示“command not found”，请检查 conda 是否已加入 PATH。

**小提示**：由于你的实训环境无独立 GPU，后续安装时请选择 **CPU 版 PyTorch**。

## 块 2

好的，我们直接进入正题。你已经掌握了 Python 基础，也写过爬虫脚本，现在要往深度学习方向走，尤其是要做 LoRA 微调，那环境这关必须过。

这一节我们不重复讲 `nvidia-smi` 怎么用，而是用它来做一个关键判断：**你的电脑到底能不能用 GPU 来跑 PyTorch**。

---

### 1.2 动手检查：你的电脑有 NVIDIA 驱动吗？

打开你的终端（Terminal），输入下面这个命令，然后回车：

```bash
nvidia-smi
```

**你会看到两种结果，对应两条完全不同的路：**

- **✅ 有正常输出**：屏幕上会显示一个表格，里面有你的显卡型号（比如 RTX 4060）、驱动版本，以及右上角有个 `CUDA Version`。  
  **看到这个，说明你的 NVIDIA 驱动已经装好了，可以直接跳到第二节**，去搞清楚驱动、CUDA 和 PyTorch 的关系。

- **❌ 报错 `command not found`**：这说明你的系统**没找到 NVIDIA 驱动**。  
  这时候你有两个选择：
  1. 去 **1.3 节** 看看怎么安装驱动（但要注意，这通常只针对本地物理机）。
  2. **更省事的方案**：直接安装 **CPU 版 PyTorch**。  
     > **特别提醒**：根据你的实训条件（无独立GPU），你大概率会走这条路。CPU 版 PyTorch 完全够你用来学习和跑通 LoRA 微调的代码逻辑，只是训练速度会慢一些。这完全不影响你掌握核心原理。

---

### 1.3 （备用）如果确实需要装驱动，怎么搞？

先判断一下你的情况，别瞎折腾：

- **云服务器 GPU 实例**：一般**已经预装好了**，不用管。
- **WSL2**：Windows 主机的驱动会自动穿透进去，**一般也不用在 WSL2 里单独装**。
- **本地物理机（双系统/独立Linux）**：如果 `nvidia-smi` 报错，才需要手动装。

**如果你属于最后一种，最简单的办法是用 Ubuntu 自带的工具：**

```bash
ubuntu-drivers devices          # 先看看系统推荐哪个驱动
sudo ubuntu-drivers autoinstall # 然后自动安装推荐的那个
```

装完重启系统，再跑一次 `nvidia-smi` 验证就行。

> **⚠️ 高风险警告**：千万别自己去 NVIDIA 官网下载 `.run` 文件然后直接复制命令安装。这里面涉及禁用 nouveau、切换图形界面等操作，不同发行版步骤不一样，搞不好系统会进不去图形界面。用系统自

## 块 3

## 本块学习检查

完成本块后，请用自己的话说明：为什么 `nvidia-smi` 输出的 `CUDA Version` 决定了你能安装哪个版本的 PyTorch？这个检查和你的 LoRA 微调目标有什么关系？遇到无法解释的术语（比如“驱动”“CUDA”）时，先记录下来，稍后提问。

---

## 一、先看你的显卡驱动支持到什么程度

运行 `nvidia-smi` 后，你会看到类似下面的输出：

```
NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2
```

**重点关注右上角 `CUDA Version: 12.2`** ——这表示你的驱动最高支持 CUDA 12.2。安装 PyTorch 时，需确保所选 CUDA 版本不超过该值，并参考 PyTorch 官方安装命令。

> 如果 `nvidia-smi` 报错或你没有 NVIDIA 显卡，请直接跳到 **第五节（CPU 版安装）**。

---

## 二、创建 PyTorch 专用 conda 虚拟环境

> 此步骤在"conda/pip 虚拟环境管理"章节已详细学过，此处快速执行。

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.10
conda create -n pytorch python=3.10 -y

# 激活环境
conda activate pytorch
```

激活成功后，命令行提示符前面会从 `(base)` 变成 `(pytorch)`。

> **为什么选 Python 3.10？** PyTorch 2.x 对 Python 3.8~3.12 都有良好支持。3.10 是目前生态兼容性最好的版本，绝大多数深度学习库都已适配。

---

## 三、安装 PyTorch

**你的实训条件说明：当前无独立 GPU。** 这意味着你不需要纠结 CUDA 版本匹配问题——直接

## 块 4

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 5.1 有 NVIDIA GPU：安装 GPU 版 PyTorch（②+③）

**核心原则**：PyTorch 的 CUDA 版本 ≤ `nvidia-smi` 显示的 CUDA Version。（②）

**推荐做法：去 PyTorch 官网复制安装命令**（最不容易出错）：

1. 打开 PyTorch 官方安装页面：https://pytorch.org/get-started/locally/
2. 按你的实际情况选择：
   - **PyTorch Build**：Stable（推荐）
   - **Your OS**：Linux
   - **Package**：Pip（推荐，版本更灵活）
   - **Language**：Python
   - **Compute Platform**：选择你驱动支持的 CUDA 版本（≤ `nvidia-smi` 的 CUDA Version）
3. 页面会自动生成安装命令，**直接复制粘贴到终端执行**。

以 `nvidia-smi` 显示 `CUDA Version: 12.2` 为例，你可以选择 CUDA 12.1 版本的 PyTorch（因为 12.1 ≤ 12.2）：（②）

```bash
# CUDA 12.1 版本（适合驱动支持 CUDA 12.x 的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

如果你的驱动支持 CUDA 11.8（较旧显卡），使用：（②）

```bash
# CUDA 11.8 版本（适合旧显卡或驱动较旧的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

> **如何确定用 cu121 还是 cu118？**
> - `nvidia-smi` 显示 CUDA Version ≥ 12.1 → 用 `cu121`
> - `nvidia-smi` 显示 CUDA Version 在 11.8~12.0 之间 → 用 `cu118`
> - 不确定时，优先选 `cu121`，因为 PyTorch 2.x 对 CUDA 12.x 的支持更完善。（②）

> **需要安装历史版本？** 访问 https://pytorch.org/get-started/previous-versions/ 查看完整对照

## 块 5

你的电脑没有 NVIDIA 显卡，这完全不影响学习 PyTorch 基础。安装 CPU 版即可，语法和 GPU 版几乎一样，只是训练速度慢一些。

打开终端，执行下面这条命令：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

装好后，你就能正常写张量操作、自动求导和神经网络代码了。后续做 LoRA 微调时，CPU 跑大模型会非常慢甚至跑不动，但用来理解概念、调试代码完全够用。

**本块学习检查**：完成安装后，用自己的话说说：为什么 CPU 版能支撑你完成 LoRA 微

## 块 6

### 5.3 安装过程说明

PyTorch 安装包约 2~3GB，下载耗时取决于网速，看到进度条走动就耐心等。（①）

**国内网络提速技巧**：若下载慢，先按第三节配置全局 pip 国内镜像源，再执行以下命令（仅指定 PyTorch 官方仓库）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**关键警告**：CUDA 版必须从官方 `download.pytorch.org` 下载，**不要**叠加 `-i` 或 `--extra-index-url` 指向其他镜像，否则 pip 可能拉到 CPU 版或版本不匹配的包，导致后续 `torch.cuda.is_available()` 返回 `False`。通用依赖（如 numpy）会自动走你配置的全局镜像。（②）

---

## 六、验证安装

装完必须验证 PyTorch 能否导入、GPU 是否可用。

**

## 块 7

你的电脑没有独立GPU，这意味着安装时选的是CPU版PyTorch。本块就是验证安装是否成功——用一段小脚本，让PyTorch自报家门。

在终端逐行执行下面命令（复制整段即可）：

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

**你的预期输出（CPU版）**：

```
PyTorch 版本: 2.5.1+cpu
CUDA 是否可用: False
当前使用 CPU 模式
```

看到“CUDA 是否可用: False”和“当前使用 CPU 模式”，说明安装成功。`False`不是错误，只是确认你的环境没有GPU，后续LoRA微调会用CPU跑，速度慢但能学。

**本块学习检查**：完成本

## 块 8

好的，我们来把这段环境验证的内容，改造成更适合你的学习材料。

你的目标是掌握 LoRA 微调，而且你已经有 Python 基础，喜欢动手实践。我们目前没有独立 GPU，所以整个验证过程会以 CPU 运行为主，但代码里会保留 GPU 的判断逻辑，让你清楚未来在有 GPU 的机器上会发生什么。

---

### 6.2 运行一个简单的张量运算（①）

环境装没装好，跑个小程序试试就知道。这个测试会创建一个最简单的“张量”（你可以先把它理解成多维数组），然后对它做一次乘法运算。

**请打开你的终端，确保 `pytorch` 环境已激活，然后运行以下命令：**

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

**你应该会看到类似下面的输出：**

第一行是 `CPU 张量:`，下面是一个 2 行 3 列的随机数字矩阵。紧接着是 `张量运算结果（CPU）:`，下面的矩阵每个数字都是上面那个矩阵对应位置数字的 2 倍。

因为你的电脑没有独立 GPU，所以程序会走 `else` 分支，只打印 CPU 的结果。这完全正常，说明 PyTorch 的核心功能已经就绪，可以开始后续的学习了。

> **本块学习检查**
>
> 完成这个测试后，请用自己的话回答：这个“

## 测评题目

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB0405-005
请写代码：实现带梯度裁剪的训练步骤，使用 `AdamW`，学习率 `5e-5`，梯度裁剪阈值为1.0。

### Q-KB0401-001
以下哪个命令可以创建一个全为0的3x3张量？

### Q-KB0401-002
张量 `x = torch.tensor([[1, 2], [3, 4]])`，其形状 `x.shape` 为________；数据类型 `x.dtype` 默认为________。

### Q-KB0404-002
`DataLoader` 中，`batch_size=32` 表示________；`shuffle=True` 表示________。

### Q-KB0401-005
给定张量 `x` 形状为 `(2, 3, 4)`，请写出将其转置为 `(4, 3, 2)` 的代码。

### Q-KB0403-005
请写代码：使用 `nn.Sequential` 定义一个包含 `Linear(10, 5)`、`ReLU`、`Linear(5, 1)` 的网络。
