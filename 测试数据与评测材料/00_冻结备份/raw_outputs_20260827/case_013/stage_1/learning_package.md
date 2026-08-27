# 学习包

## 块 1

## 环境准备：根据 CUDA 版本安装 PyTorch

> **本块定位**：你的目标是 LoRA 微调，而 PyTorch 是它的地基。本块只做一件事——装好 PyTorch 并验证 GPU 可用。你已学过 conda 管理，我们直接快进。

### 一、前置检查：30 秒确认环境

你之前已经配好 conda，这里只做快速确认，不重复教学：

```bash
conda --version
# 预期输出：conda 24.x.x
```

如果报错，回到“conda/pip 虚拟环境管理”章节排查；如果正常，继续往下。

### 二、核心操作：装 PyTorch（GPU 版 / CPU 版二选一）

**先判断你的情况**：

- **有 NVIDIA 显卡** → 走 GPU 版（推荐，LoRA 微调需要）
- **没有显卡** → 走 CPU 版（代码能跑，只是慢）

**GPU 版安装命令**（以 CUDA 12.1 为例）：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

> 注意：CUDA 版本必须和你的驱动匹配。用 `nvidia-smi` 查看驱动支持的 CUDA 版本，再选择对应的 PyTorch 安装命令（将 `pytorch-cuda=12.1` 中的版本号改为你驱动对应的版本）。版本不匹配是常见坑。

**CPU 版安装命令**：

```bash

## 块 2

## 1.2 确认 nvidia-smi 有输出

你的目标是 **LoRA 微调**，这意味着你马上要跑 GPU 训练。在装 PyTorch 之前，先花 30 秒确认你的显卡驱动是通的。

打开终端，执行：

```bash
nvidia-smi
```

**看结果，分两种情况：**

- **有正常输出**：能看到显卡型号、驱动版本、右上角有 `CUDA Version` 字样——说明驱动没问题，直接跳到**第二节**，搞清楚版本关系后装 PyTorch。
- **报错 `command not found`**：当前环境没识别到 NVIDIA 驱动。先看下面 **1.3 节**决定要不要装驱动，或者干脆装 **CPU 版 PyTorch**（但 LoRA 微调强烈建议用 GPU，否则训练慢到怀疑人生）。

---

## 1.3 （如需要）安装 NVIDIA 显卡驱动

**先判断你的场景：**

- **云服务器 GPU 实例**：通常**已预装驱动**，直接跳过本节。
- **本地物理机（双系统 / 独立 Linux）**：`nvidia-smi` 报错才需要手动装。
- **WSL2**：Windows 主机的驱动会自动穿透到 WSL2，**一般不需要在 WSL2 内部单独装**。

**Linux 本地物理机装驱动（两条路）：**

**路线 A：NVIDIA 官方驱动（不推荐新手）**

1. 访问 https://www.nvidia.com/Download/index.aspx
2. 选你的显卡型号 + Linux 64-bit，下载 `.run` 文件
3. 按官方说明安装。**注意**：不要直接复制执行 `.run` 命令；禁用 nouveau、切换图形界面等步骤因发行版而异，操作不当可能进不了图形界面。

**路线 B：Ubuntu 附加驱动工具（推荐，省心）**

```bash
ubuntu-drivers devices          # 查看推荐驱动
sudo ubuntu-drivers autoinstall # 自动安装推荐驱动
```

装完**重启系统**，再跑 `nvidia-smi` 验证。

---

## 二、显卡、驱动、CUDA、PyTorch 四者关系

这是装 PyTorch 最容易踩坑的地方。很多人装完跑 `torch.cuda.is_available()` 返回 `False`，就是没搞清这四者的关系。看这张依赖链：

```
显卡硬件（如 RTX 4060）
    ↓ 决定支持的驱动版本上限
NVIDIA 显卡驱动（Driver）
    ↓ nvidia-smi 右上角显示的 CUDA Version =

## 块 3

## 一、读图：你的显卡在说什么？

刚才你运行 `nvidia-smi` 后，应该看到了一张类似下面的表。别被一堆数字吓到，我们只抓重点：

```
NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2
+-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
+=========================================+======================+======================+
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              15W / 170W|    256MiB / 12288MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

**右上角 `CUDA Version: 12.2` 是今天的主角。** 它表示当前驱动兼容的 CUDA 运行时版本。记住这个数字，下一步装 PyTorch 时，你选的 CUDA 版本**不能超过 12.2**，以确保驱动兼容。

> 如果 `nvidia-smi` 报错，或者你根本没有 NVIDIA 显卡，直接跳到**第五节（CPU 版安装）**，别在这里卡住。

## 二、创建 PyTorch 专用环境

这一步在"conda/pip 虚拟环境管理"章节已经练过，我们快速过一遍：

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.10
conda create -n pytorch python=3.10 -y

# 激活环境
conda activate pytorch
```

激活成功后，命令行提示符前面会从 `(base)` 变成 `(pytorch)`。

**为什么选 Python 3.10？** PyTorch 2.x 对 Python 3.8~3.12 都支持，而 3.10 是目前生态兼容性最好的版本，绝大多数深度学习库都已适配。选它最省心。

## 三、安装 PyTorch（核心步骤）

现在到了最关键的一步。打开 PyTorch 官网的安装命令生成器，选择你的系统、包管理器（conda 或 pip）、CUDA 版本——**注意，这里要选 12.1**，因为你的驱动支持 12.2，而 PyTorch 官方目前提供的最高 CU

## 块 4

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，而这一切的第一步，就是让 PyTorch 能在你的 NVIDIA 显卡上跑起来。别担心，这个过程比你想的简单，我们直接上手操作。

核心就一句话：**你安装的 PyTorch CUDA 版本，不能高于你显卡驱动支持的版本**。

怎么知道驱动支持多少？打开终端，输入 `nvidia-smi`，看右上角的 “CUDA Version”。比如显示 `12.2`，那你就装 CUDA 12.1 或 11.8 的 PyTorch，只要数字小于等于 12.2 就行。

最稳妥、最不容易出错的方法，是去 PyTorch 官网自动生成安装命令。打开这个页面：`https://pytorch.org/get-started/locally/`，然后按下面的选：

- **PyTorch Build**: 选 `Stable`（稳定版）
- **Your OS**: 选你的系统，比如 `Linux`
- **Package**: 选 `Pip`（推荐，版本更灵活）
- **Language**: 选 `Python`
- **Compute Platform**: 选你驱动支持的 CUDA 版本（就是刚才 `nvidia-smi` 看到的那个，或者比它低一点的）

选完，页面就会自动生成一条安装命令，直接复制到终端运行就行。

举个例子，假设 `nvidia-smi` 显示 `CUDA Version: 12.2`，你可以选择 CUDA 12.1 的版本，命令长这样：

```bash
# CUDA 12.1 版本（适合驱动支持 CUDA 12.x 的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

如果你的显卡比较老，驱动只支持 CUDA 11.8，那就用这个：

```bash
# CUDA 11.8 版本（适合旧显卡或驱动较旧的用户）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

**怎么选 cu121 还是 cu118？** 简单记一下：

- `nvidia-smi` 显示 CUDA Version **≥ 12.1** → 用 `cu121`
- `nvidia-smi` 显示 CUDA Version 在 **11.8 ~ 12.0** 之间 → 用 `cu118`
- 不确定？优先选 `cu121`，因为

## 块 5

你的电脑没有 NVIDIA 显卡（比如用虚拟机、WSL2 或 Mac），装 CPU 版 PyTorch 就够了。它能正常跑通张量操作、自动求导和神经网络搭建这些 API，和 GPU 版几乎没差别，只是训练速度慢。

安装命令如下（直接复制执行）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

**本块学习检查**：运行 `python -c "import torch; print(torch.__version__)"`，看到 `2.5.1` 即安装成功。后续学 LoRA 微调时，CPU 版跑大模型会非常慢，

## 块 6

### 安装过程说明（针对 LoRA 微调目标）

你的目标是 LoRA 微调，这意味着 PyTorch 必须能调用 GPU，否则后续训练会非常慢。安装包约 2~3GB，下载需几分钟到十几分钟，看到进度条在走就耐心等待。

**关键提醒（国内网络）**：若下载慢，先按第三节配置好全局 pip 国内镜像源，然后执行：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**千万别踩的坑**：PyTorch 的 CUDA 版本包**必须**从官方 `download.pytorch.org` 下载，**不要**叠加 `-i` 或 `--extra-index-url` 指定其他镜像源。命令中的 `--index-url` 会覆盖全局镜像源，pip 只会从官方源拉取包，不会从镜像源拉到 CPU 版或版本不匹配的包，因此无需担心 `torch.cuda

## 块 7

**快速验证：你的 PyTorch 装对了吗？**

装完 PyTorch 后，别急着开始 LoRA 微调，先花 1 分钟确认环境没问题。打开终端，逐行复制下面这段代码并回车：

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

**怎么看结果？**

如果你有 NVIDIA GPU 且驱动正常，会看到类似：

```
PyTorch 版本: 2.5.1+cu121
CUDA 是否可用: True
PyTorch 内置 CUDA 版本: 12.1
GPU 数量: 1
GPU 型号: NVIDIA GeForce RTX 3060
```

如果没 GPU 或没装好，会看到：

```
PyTorch 版本:

## 块 8

### 6.2 运行一个简单的张量运算（①）

你已经装好了 PyTorch，现在我们来亲手“摸”一下它，确认核心功能真的能跑起来。这一步很关键，因为后面做 LoRA 微调时，所有模型和数据的计算都建立在张量运算之上。

打开终端，输入下面这段 Python 代码（直接复制整段，回车执行）：

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

**这段代码在做什么？** 简单说，它创建了一个 2 行 3 列的随机数表格（张量），然后把它乘以 2。如果 PyTorch 检测到你的电脑有可用的 GPU，它会把计算放到 GPU 上执行；如果没有，就在 CPU 上执行。

**你该看到什么？** 屏幕上会打印出两到三组数字：先是 CPU 上的原始张量，然后（如果有 GPU）是 GPU 上的张量，最后是乘以 2 之后的结果。只要这些数字能正常打印出来，就说明 PyTorch 环境完全就绪，可以开始后续的学习了。

**小提示**：如果你看到 `torch.cuda.is_available()` 返回 `False`，别慌，这不一定是装错了。可能是你装的是 CPU 版 PyTorch，或者电脑没有 NVIDIA 显卡。后面的“常见问题”表格里会帮你排查。对你来说，即使只有 CPU 也能先跑通 LoRA 的基础流程，只是速度慢一些。

**动手验证一下**：执行完

## 块 9

## 二、张量创建与基本属性（②）

### 2.1 创建张量（②）

咱们的目标是LoRA微调，而LoRA微调的第一步，就是得先学会“造”出模型参数张量。别担心，这一步和NumPy几乎一模一样，你很快就能上手。

先确保环境已激活：

```bash
conda activate pytorch
```

然后，咱们直接开干。下面这几种创建方式，是后续微调中最常用的“地基”：

**1. 从列表直接创建**

这是最直观的方式，把Python列表“翻译”成张量：

```python
import torch

# 从列表创建
a = torch.tensor([[1, 2], [3, 4]])
```

`a` 就是一个2行2列的张量，里面的值就是列表里的数字。

**2. 创建全0 / 全1张量**

在LoRA中，我们经常需要初始化权重矩阵，全0和全1就是最常用的“起点”：

```python
zeros = torch.zeros((2, 3))      # 2行3列，值全为0
ones = torch.ones((3, 2))        # 3行2列，值全为1
```

注意看，`zeros` 和 `ones` 接收的是一个**元组** `(行, 列)`，用来指定形状。

**3. 随机张量**

模型初始化时，随机权重是标配。这里有两种最常见的随机分布：

```python
rand = torch.rand((2, 2))        # [0, 1) 均匀分布
normal = torch.randn(2, 3)        # 标准正态分布
```

- `rand`：生成0到1之间（不含1）的均匀分布随机数。
- `randn`：生成均值为0、标准差为1的标准正态分布随机数。

**4. 整数随机张量**

如果你想生成指定范围内的随机

## 块 10

## 2.2 查看张量属性（②）

在动手写 LoRA 代码前，先学会“体检”你的张量。创建完张量后，用下面四个属性快速确认它的状态：

```python
x = torch.randn(3, 4, dtype=torch.float32)

print(x.shape)           # torch.Size([3, 4])  ——形状
print(x.dtype)           # torch.float32        ——数据类型
print(x.ndim)            # 2                    ——维度数
print(x.device)          # cpu                  ——所在设备
```

| 属性 | 含义 | 示例值 |
|---|---|---|
| `.shape` / `.size()` | 各维度大小 | `torch.Size([3, 4])` |
| `.ndim` | 维度数 | `2` |
| `.dtype` | 数据类型 | `torch.float32` |
| `.device` | 所在设备 | `cpu` 或 `cuda:0` |

> **注意**：PyTorch 默认数据类型是 `torch.float32`。神经网络训练时通常用这个类型，但某些场景需要 `torch.long`（如分类标签）。

---

## 三、索引与切片（①+②）

张量的索引规则与 NumPy 几乎完全一致：**先行后列**，支持整数索引、范围切片、布尔索引和列表索引。

```python
data = torch.randint(0, 10, (4,

## 块 11

`reshape()` 只改变“看待数据的方式”，不碰数据本身。你正为LoRA微调打基础，先掌握这个最安全的形状变换。

```python
t1 = torch.arange(12)      # 1维，12个元素
t2 = t1.reshape(3, 4)      # 变为3行4列
print(t2.shape)            # torch.Size([3, 4])

t3 = t1.reshape(2, -1)     # 2行6列，-1自动算出为6
print(t3.shape)            # torch.Size([2, 6])
```

**关键规则**：元素总数必须一致——3×4=12，2×6=12。`-

## 块 12

**目标**：LoRA微调前，先搞懂`view()`和`reshape()`的区别——这决定了你改模型权重时会不会踩内存坑。

**核心区别**（记住这一条就够）：
- `reshape()`：**不挑食**，张量内存连续不连续都能用。
- `view()`：**有洁癖**，只认连续内存的张量，不连续就报错。

**动手验证**（跟着敲）：
```python
t1 = torch.arange(6).reshape(2, 3)
print(t1.is_contiguous())     # True

# 连续张量：view 和 reshape 都能用
t2 = t1.view(3, 2)            # ✅ 正常

## 块 13

## 4.3 为什么 transpose/permute 后 view 会报错？

你在做 LoRA 微调时，经常要操作权重矩阵的维度，比如把 `(out_features, in_features)` 转成 `(in_features, out_features)`。这时如果直接用 `view()`，很可能就报错了。

**根本原因**：`transpose()` 和 `permute()` **不会拷贝数据**，只是改变维度的“查看方式”，这会导致张量在内存中不再连续。而 `view()` 要求张量必须是连续的。

动手验证一下：

```python
t1 = torch.arange(6).reshape(2, 3)
print(t1.is_contiguous())     # True

# 转置后
t2 = t1.transpose(0, 1)
print(t2.is_contiguous())     # False  ← 不连续了！

# t2.view(6)                  # ❌ 报错！因为 t2 非连续，view 要求连续
t3 = t2.contiguous().view(6)  # ✅ 先调用 contiguous() 恢复连续，再 view
t4 = t2.reshape(6)            # ✅ reshape 会自动处理连续性（内部先 contiguous 再 view）
```

**实操建议**：不确定时用 `reshape()`，它更宽容；追求极致性能且确认连续时用 `view()`。

---

## 五、升维与降维：squeeze 与 unsqueeze

在 LoRA 微调中，你经常需要给张量加一个 batch 维度或删掉多余的维度。比如把 `(3, 224, 224)` 的图像升为 `(1, 3, 224, 224)` 来模拟 batch 维度。

```python
t1 = torch.randn(3, 1, 2, 1)
print(t1.shape)               # torch.Size([3, 1, 2, 1])

# squeeze：删除所有大小为 1 的维度
t2 = torch.squeeze(t1)
print(t2.shape)               # torch.Size([3,

## 块 14

## 6.1 transpose：交换两个维度

你的目标是 LoRA 微调，在操作 HuggingFace 模型时经常要调整张量维度。`transpose(dim0, dim1)` 就是**交换两个指定维度**，其他维度保持不变。

```python
t1 = torch.randn(2, 3, 4)
print(t1.shape)               # torch.Size([2, 3, 4])

# 交换第 0 维和第 1 维
t2 = t1.transpose(0, 1)
print(t2.shape)               # torch.Size([3, 2, 4])
```

**动手试一下**：把上面的代码跑一遍，观察形状变化。再试试 `t1.transpose(1, 2)`，看看结果是什么形状？

## 6.2 permute：任意重排所有维度

`permute(dims)` 更灵活，一次性指定**所有维度**的新顺序：

```python
t1 = torch.randn(2, 3, 4)
print(t1.shape)               # torch.Size([2, 3, 4])

# 新顺序：原第2维 → 新第0维，原第0维 → 新第1维，原第1维 → 新第2维
t2 = t1.permute(2, 0, 1)
print(t2.shape)               # torch.Size([4, 2, 3])
```

**动手试一下**：把 `permute(2, 0, 1)`

## 块 15

### 元素级运算与矩阵乘法：为LoRA微调打基础

你正为LoRA微调做准备，核心是理解张量如何运算。这里聚焦两种最常用的操作。

**元素级运算**：两个张量形状相同，对应位置直接计算。

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

print(a + b)          # tensor([5., 7., 9.])
print(a * b)          # tensor([ 4., 10., 18.])
print(torch.sqrt(a))  # tensor([1., 1.4142, 1.7321])
```

`*` 是逐元素相乘，不是矩阵乘法。

**矩阵乘法**：线性代数中的点积

## 块 16

## 7.3 广播机制（②）

你的LoRA微调中，权重矩阵形状常不一致，广播能自动对齐形状完成运算，省去手动扩展。

看例子：`a`形状是(3,1)，`b`形状是(3,)，直接相加得到(3,3)：

```python
a = torch.tensor([[1], [2], [3]])   # shape (3, 1)
b = torch.tensor([10, 20, 30])      # shape (3,)

print(a + b)
# tensor([[11, 21, 31],
#         [12, 22, 32],
#         [13, 23, 33]])
```

规则：从尾部维度对齐，尺寸为1的维度自动扩展，缺失维度补1。

## 7

## 块 17

## 8.1 布尔掩码：用条件筛选数据

在LoRA微调中，你经常需要从大量参数中筛选出符合条件的部分。PyTorch的布尔掩码就是为此设计的——用一个条件表达式生成“真假面具”，再据此挑选数据。

```python
x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 3          # 生成布尔掩码：tensor([False, False, False, True, True])
print(x[mask])        # tensor([4, 5])  ← 只保留满足条件的元素
```

**核心理解**：`x > 3` 返回一个与 `x` 形状相同的布尔张量，`True` 表示该位置满足条件。用这个掩码索引原张量，就能筛出符合条件的元素。

---

## 8.2 clone：创建独立副本

微调LoRA时，你可能需要保留一份原始权重作为基准，再在副本上做修改。这时必须用 `clone()`：

```python
a = torch.tensor([1, 2, 3])
b = a.clone()         # 深拷贝，b 和 a 不共享内存
b[0] = 100
print(a)              # tensor([1, 2, 3])  ← a 没变，安全！
```

> **关键区别**：直接写 `b = a` 只是让 `b` 指向 `a` 的同一块内存，修改 `b` 会连带修改 `a`。需要独立副本时，务必用 `.clone()`。

---

## 九、张量与NumPy互转：默认共享内存

PyTorch张量和NumPy数组可以无缝转换，但**默认共享内存**——修改一个，另一个也会变。

```python
import numpy as np

# Tensor → NumPy
x = torch.tensor([1, 2

## 块 18

## 综合实战：图像预处理流程（张量操作串联）

你的目标是LoRA微调，而LoRA微调的第一步，就是**把数据喂给模型之前，先把数据整理成模型认识的形状**。图像预处理就是最典型的张量操作实战场景——你会在里面看到`unsqueeze`、`permute`、`reshape`、`to(device)`这些操作如何配合使用。

### 场景：模拟一张图像从原始数据到可训练输入

假设你拿到一张3通道、224×224的图像，PyTorch里它就是一个形状为`(3, 224, 224)`的张量：

```python
import torch

# 1. 模拟一张 3通道 224x224 的图像数据
image = torch.randn(3, 224, 224)
print(f"原始图像形状: {image.shape}")   # torch.Size([3, 224, 224])
```

**第一步：升维加batch。** 模型训练时一次处理一批数据，所以需要加一个batch维度，放在最前面。`unsqueeze(0)`就是在第0维插入一个大小为1的维度：

```python
# 2. 升维：添加 batch 维度 → (1, 3, 224, 224)
image_batch = image.unsqueeze(0)
print(f"添加batch后: {image_batch.shape}")  # torch.Size([1, 3, 224, 224])
```

**第二步：维度重排。** 有些框架（比如TensorFlow）要求数据格式是`(B, H, W, C)`——通道在最后。PyTorch默认是`(B, C, H, W)`，所以用`permute`把维度顺序换一下。`permute(0, 2, 3, 1)`的意思是：第0维不动，第1维挪到最后：

```python
# 3. 维度重排：从 (B, C, H, W) 转为 (B, H, W, C)
image_nhwc = image_batch.permute(0, 2, 3, 1)
print(f"维度重排后: {image_nhwc.shape}")    # torch.Size([1, 224, 224, 3])
```

**第三步：标准化。** 图像预处理通常要按通道减均值、除标准差。注意这里用的是`torch.randn`生成的数据，它本身不在`[0, 1]`范围，所以这个

## 块 19

## 十三、总结与后续学习路线（①+②+④）

恭喜你完成了 PyTorch 张量操作章节的学习！你现在应该掌握了：

- ✅ 张量创建：`torch.tensor`、`torch.zeros`、`torch.ones`、`torch.rand`、`torch.arange`
- ✅ 张量属性：`shape`、`dtype`、`ndim`、`device`
- ✅ 索引切片：基础索引、范围切片、列表索引、布尔索引
- ✅ 形状变换：`reshape`（安全）、`view`（需连续）、`squeeze`/`unsqueeze`（升降维）
- ✅ 维度交换：`transpose`（交换两个维度）、`permute`（任意重排）、`contiguous`（恢复连续）
- ✅ 张量运算：元素级运算、矩阵乘法（`@`）、广播机制、聚合统计
- ✅ 条件筛选：布尔掩码、独立拷贝（`clone`）
- ✅ NumPy 互转：`numpy()`、`from_numpy()`（共享内存）
- ✅ 设备切换：`.to(device)`、`.cuda()`、`.cpu()`

> **下一步预告**：在下一节 **"自动求导机制"** 中，我们将学习 PyTorch 最核心的特性之一 —— **Autograd**。你将了解 `requires_grad`、`backward()`、梯度计算图

## 测评题目

### Q-KB04-ST0400-002
在 Python 中可通过 `torch.cuda.is_available()` 检查 PyTorch 是否识别到________。

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB0401-001
以下哪个命令可以创建一个全为0的3x3张量？

### Q-KB0401-003
请将以下两个张量按**列方向**（dim=1）拼接：
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5], [6]])
```

### Q-KB0407-002
加载模型参数时，需要先________模型结构，然后调用________方法加载参数字典。

### Q-KB0403-001
以下哪个是PyTorch中定义**线性层**（全连接层）的类？

### Q-KB0405-003
请写代码：为模型 `model` 创建 `AdamW` 优化器，学习率 `1e-4`，权重衰减 `1e-2`。

### Q-KB0407-004
加载checkpoint时，`map_location` 参数的作用是？
