# 学习包

## 块 1

## 环境准备：根据 CUDA 版本安装 PyTorch（前置检查）

你的目标明确：**LoRA 微调**。在动手之前，环境必须一次到位。你已有 conda 虚拟环境管理基础，且硬件是 **RTX 4060 8GB**（Ada Lovelace 架构，驱动支持 CUDA 12.x 没问题）。本节只做快速确认，不重复教学。

### 1.1 确认 conda 可用

前面章节已学过虚拟环境管理，这里只需一条命令验证：

```bash
conda --version
# 预期输出：conda 24.x.x
```

如果输出版本号，说明 conda 就绪，继续下一步。若报错，回到"conda/pip 虚拟环境管理"章节排查。

---

**本块学习检查**

先用一个最小示例回顾本节核心：确认 conda 可用后，你能否解释——为什么在安装 PyTor

## 块 2

## 1.2 确认 nvidia-smi 有输出

你的目标是 LoRA 微调，而 LoRA 微调（尤其是 QLoRA）对 GPU 环境有硬性要求。在动手之前，先花 30 秒确认你的环境是否就绪。

打开终端，执行：

```bash
nvidia-smi
```

**判断标准：**

- **有正常输出**（显示显卡型号、驱动版本、CUDA Version）→ 环境就绪，直接跳到第二节。
- **报错 `command not found`** → 当前环境没有识别到 NVIDIA 驱动。你需要安装驱动（见下方 1.3 节），或者选择安装 CPU 版 PyTorch。

> ⚠️ 注意：如果你在云服务器 GPU 实例上，通常已预装驱动，不需要操作。如果你用 WSL2，Windows 主机的驱动会自动穿透到 WSL2，一般也不需要单独安装。只有本地物理机（双系统/独立 Linux）才需要手动装驱动。

---

## 二、先搞清楚：显卡、驱动、CUDA、PyTorch 四者是什么关系？

这是安装 PyTorch 最容易踩坑的地方。很多人装完后运行 `torch.cuda.is_available()` 返回 `False`，然后对着一堆版本号不知道哪里错了。先把四者的依赖关系搞清楚，后面每一步都有据可依。

```
显卡硬件（如 RTX 4060）
    ↓ 决定支持的驱动版本上限
NVIDIA 显卡驱动（Driver）
    ↓ nvidia-smi 右上角显示的 CUDA Version = 驱动支持的最高 CUDA 版本
CUDA Toolkit（nvcc --version，可选）
    ↓ PyTorch 自带 CUDA Runtime，一般不需要单独装
PyTorch（内置 CUDA Runtime）+ cuDNN
```

**最关键的两句话：**

1. **`nvidia-smi` 右上角显示的 `CUDA Version`，是驱动支持的"最高"CUDA 版本，不是你已经安装的 CUDA 版本。** 你安装的 PyTorch 的 CUDA 版本必须 **≤** 这个值。
2. **如果你只是跑 PyTorch 训练或推理，不需要单独安装 CUDA Toolkit。** PyTorch 的 wheel/conda 包里已经自带了 CUDA Runtime，装上就能用 GPU。

> 什么时候才需要单独装 CUDA Toolkit？只有你需要编译 CUDA 扩展（如 `flash-attn`、`bitsandbytes`）或使用 `nvcc` 编译自定义 CUDA 代码时。后续"参数高效微调"章节涉及这些库时，我们会再补充安装步骤。

---

## 三、查看驱动支持的 CUDA 版本上限

再次执行 `nvidia-smi`，这次重点关注右上角的 `CUDA Version`

## 块 3

## 确认你的显卡驱动上限：`nvidia-smi` 输出解读

在动手安装 PyTorch 之前，先花 30 秒确认你的硬件驱动能支持到什么 CUDA 版本。打开终端，输入：

```bash
nvidia-smi
```

你会看到类似下面的输出（以下示例仅为示意，实际请以你运行 `nvidia-smi` 的输出为准）：

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|  Fan  Temp   Perf          Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
+=========================================+======================+======================+
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   45C    P8              15W / 170W|    256MiB / 12288MiB |      0%      Default |
+-----------------------------------------+----------------------+----------------------+
```

**关键信息在右上角：`CUDA Version: 12.2`**。这不是你当前正在使用的 CUDA 版本，而是**当前驱动支持的 CUDA 版本**。这意味着你后续安装的 PyTorch，其 CUDA 版本**不应高于 12.2**——否则驱动无法承载，PyTorch 将无法调用 GPU。

> 如果你运行 `nvidia-smi` 报错，或者根本没有 NVIDIA 显卡，请直接跳到第五节（CPU 版安装）。

---

## 创建 PyTorch 专用 conda 虚拟环境

虚拟环境管理你在前面章节已学过，这里快速执行即可：

```bash
# 创建名为 pytorch 的虚拟环境，Python 版本 3.10
conda create -n pytorch python=3.10 -y

# 激活环境
conda activate pytorch
```

激活成功后，命令行提示符前的 `(base)` 会变成 `(pytorch)`。

**为什么选 Python 3.10？** PyTorch 2.x 对 Python 3.8~3.12 都有良好支持，而 3.10 是目前生态兼容性最好的版本——你后续做 LoRA 微调时会用到的 HuggingFace

## 块 4

### 5.1 有 NVIDIA GPU：安装 GPU 版 PyTorch（②+③）

你的目标是 LoRA 微调，而 LoRA 微调的前提是模型能在 GPU 上跑起来。你的 RTX 4060（8GB 显存）完全够用，关键是把 PyTorch 装对——装错了，后面所有微调代码都会报“CUDA 不可用”的错。

**核心原则就一条**：PyTorch 的 CUDA 版本必须 ≤ 你驱动支持的 CUDA 版本。

怎么查驱动支持的版本？终端执行 `nvidia-smi`，看右上角 `CUDA Version`。比如显示 `CUDA Version: 12.2`，那你就装 CUDA 12.1 的 PyTorch（12.1 ≤ 12.2，满足条件）。

**最不容易出错的做法**：去 PyTorch 官网复制命令，别自己拼。

1. 打开 https://pytorch.org/get-started/locally/
2. 按你的情况选：
   - **PyTorch Build**：Stable
   - **Your OS**：Linux
   - **Package**：Pip（版本更灵活）
   - **Language**：Python
   - **Compute Platform**：选你驱动支持的 CUDA 版本（≤ `nvidia-smi` 的结果）
3. 页面自动生成命令，直接复制粘贴到终端执行。

**怎么选 cu121 还是 cu118？** 看 `nvidia-smi` 的结果：

- CUDA Version ≥ 12.1 → 用 `cu121`
- CUDA Version 在 11.8 ~ 12.0 之间 → 用 `cu118`
- 不确定时优先 `cu121`，因为 PyTorch 2.x 对 CUDA 12.x 支持更完善

以 `nvidia-smi` 显示 `CUDA Version: 12.2` 为例，装 CUDA 12.1 版本：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

如果你的驱动只支持 CUDA 11.8（旧显卡），用：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==

## 块 5

你的目标是用 LoRA 做微调，且手头有 RTX 4060（8GB 显存），因此本块 CPU 版安装属于“了解即可”的备选路径，不必实操。

**适用场景**：仅当你的环境没有 NVIDIA GPU（如虚拟机、WSL2 无 GPU 穿透、Mac）时，才用 CPU 版。命令如下：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

**关键认知**：CPU 版与 GPU 版的 PyTorch API（张量操作、自动求导、神经网络搭建）几乎完全一致，所以学习概念和调试代码没问题

## 块 6

### 安装过程说明：耐心等待，但别踩坑

PyTorch 安装包体积较大（约 2~3GB 或更多），下载耗时取决于网络速度，通常几分钟到十几分钟。看到进度条在走就耐心等待，这是正常现象。

**国内网络加速提示**：若下载速度过慢，可先配置全局 pip 国内镜像源，然后仅指定 PyTorch 官方 wheel 仓库执行安装：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

**关键警告**：PyTorch 的 CUDA 版本包必须从官方 `download.pytorch.org` 仓库下载，**不要**叠加 `-i` 或 `--extra-index-url` 指定其他 PyPI 镜像源。否则 pip 会优先从镜像源拉到 CPU 版或版本不匹配的包，导致后续 `torch.cuda.is_available()` 返回 `False`。通用依赖（如 numpy）会自动走已

## 块 7

### 6.1 快速验证脚本

你的目标是 LoRA 微调，而微调的前提是 PyTorch 能真正调用你的 RTX 4060。下面这个脚本就是“体检报告”——它只做一件事：确认 PyTorch 是否“看见”了你的 GPU。

在终端逐行执行：

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
GPU 型号: NVIDIA GeForce RTX 4060
```

**预期输出（CPU 版）**：

```

## 块 8

### 6.2 运行一个简单的张量运算（①）

环境装好了，光看 `torch.cuda.is_available()` 返回 `True` 还不够，我们跑一段真实代码，确认 PyTorch 的核心功能——张量创建与运算——在 CPU 和 GPU 上都能正常工作。这一步对你后续做 LoRA 微调至关重要，因为微调过程中所有操作都是对张量进行的。

在终端执行以下命令：

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

1. `torch.rand(2, 3)` 创建一个 2 行 3 列、元素值在 0 到 1 之间随机分布的张量。
2. 先打印 CPU 上的张量。
3. 检查 GPU 是否可用。你的机器是 RTX 4060 8GB，如果驱动和 CUDA 版本对齐正确，这里会进入 GPU 分支。
4. `x.cuda()` 把张量从 CPU 显式移动到 GPU 显存。
5. 在 GPU 上执行 `x_gpu * 2`——每个元素乘以 2——并打印结果。

**你预期看到什么？**

- 如果一切正常，你会看到三部分输出：CPU 张量、GPU 张量、GPU 张量运算结果。GPU 张量和 CPU 张量的数值可能不同（因为是随机生成的），但运算结果（乘以 2）应该与 GPU 张量一一对应。
- 如果 `torch.cuda.is_available()` 返回 `False`，代码会走 else 分支，只打印 CPU 结果。这时需要回到上一节排查：`nvidia-smi` 是否有输出、PyTorch 版本是否含 `+cuXXX` 而非 `+cpu`、`cuXXX` 是否 ≤ 驱动支持的 CUDA Version。

**为什么这一步值得认真做？**

你最终目标是 LoRA 微调。微调时，模型参数、梯度、优化器状态全部以张量形式存在，且默认在 GPU 上计算。如果

## 块 9

## 2.1 创建张量：为LoRA微调打地基

你正在为LoRA微调做准备，而LoRA的核心操作——冻结原模型权重W、仅训练低秩矩阵A和B——本质上就是**在已有张量旁新建小矩阵并训练增量**。所以，先花15分钟把张量创建这关过了，后面看LoRA代码才不会卡壳。

### 为什么先学创建张量？

LoRA微调的第一步是加载预训练模型，模型里的权重（如`q_proj.weight`）就是一个巨大的张量。你后续要做的所有操作——冻结它、在旁边新建一个小矩阵、把两个矩阵相加——都建立在对张量基本属性的理解上。现在创建张量的手感越熟，后面读LoRA源码就越顺。

### 五种创建方式，一次到位

```python
import torch

# 1. 从列表创建——最直观，适合小规模测试
a = torch.tensor([[1, 2], [3, 4]])
# 这就是一个2x2的矩阵，你可以把它想象成LoRA中某个权重矩阵的迷你版

# 2. 全0 / 全1——初始化"空白"张量
zeros = torch.zeros((2, 3))      # 2行3列，值全为0
ones = torch.ones((3, 2))        # 3行2列，值全为1
# 在LoRA中，低秩矩阵B初始化为0，保证训练开始时增量ΔW=(α/r)BA为0，不影响原模型

# 3. 随机张量——模拟"噪声"或初始化
rand = torch.rand((2,

## 块 10

**个人化教学正文**

你的目标是LoRA微调，而LoRA的核心操作之一就是**精准地“切”出模型参数矩阵的某一部分**（比如只对`q_proj`和`v_proj`的权重做低秩分解）。所以，张量的索引与切片对你来说不是基础练习，而是**微调时的日常操作**。

---

### 2.2 查看张量属性（②）

在动手改模型之前，你得先知道手里这个张量长什么样。用`torch.randn(3, 4)`创建一个3行4列的张量，然后依次查看它的四个关键属性：

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

> **注意**：PyTorch 默认数据类型是 `torch.float32`。神经网络训练时通常用这个类型，但某些场景需要 `torch.long`（如分类标签）。（②）

---

### 三、索引与切片（①+②）

索引规则与NumPy一致：**先行后列**。这对你尤其重要——因为LoRA要修改的权重矩阵通常是`(out_features, in_features)`形状，你经常需要按行（神经元）或按列（输入特征）来定位。

```python
data = torch.randint(0, 10, (

## 块 11

**4.1 reshape：最安全的形状变换**

`reshape()` 只改变张量的“看待方式”，不触碰数据本身。这对你后续做 LoRA 微调很关键——你只需重排权重矩阵的维度，无需复制数据。

```python
t1 = torch.arange(12)          # 1维，12个元素
t2 = t1.reshape(3, 4)        # 变为 3行4列
print(t2.shape)               # torch.Size([3, 4])

# -1 表示自动计算该维度的大小
t3 = t1.reshape(2, -1)        # 2行6列（-1 被自动算出为 6）
print(t3.shape)

## 块 12

**view 与 reshape 的核心区别**

你的目标是 LoRA 微调，这需要频繁操作预训练模型的权重张量。理解 `view()` 和 `reshape()` 的区别，能帮你避免微调脚本中常见的“内存不连续”报错。

两者功能相似，但关键区别在于：

- **`reshape()`**：无论张量在内存中是否连续，都能正常工作。
- **`view()`**：**要求张量内存连续**，否则直接报错。

```python
t1 = torch.arange(6).reshape(2, 3)
print(t1.is_contiguous())     # True

# 连续张量：view 和 reshape 都能用
t2 = t1.view(3, 2)

## 块 13

## 4.3 为什么 transpose/permute 后 view 会报错？

你在追 LoRA 微调时，迟早会遇到 `view` 报错——尤其是在处理注意力矩阵或权重维度时。这个报错不是 bug，而是 PyTorch 在保护你。

**核心原因**：`transpose()` 和 `permute()` **不会拷贝数据**，只是改变维度的"查看方式"。这意味着张量在内存中可能不再连续，而 `view` 要求内存连续。

```python
t1 = torch.arange(6).reshape(2, 3)
print(t1.is_contiguous())     # True

# 转置后
t2 = t1.transpose(0, 1)
print(t2.is_contiguous())     # False  ← 不连续了！

# t2.view(6)                  # ❌ 报错！view 要求连续
t3 = t2.contiguous().view(6)  # ✅ 先调用 contiguous() 恢复连续，再 view
t4 = t2.reshape(6)            # ✅ reshape 直接就能用
```

**给你的实操建议**：不确定时用 `reshape()`，它更宽容；追求极致性能且确认连续时用 `view()`。

---

## 五、升维与降维：squeeze 与 unsqueeze

微调时你常需要调整维度数量。例如，把 `(3, 224, 224)` 的图像升为 `(1, 3, 224, 224)` 来模拟 batch 维度。

```python
t1 = torch.randn(3, 1, 2, 1)
print(t1.shape)               # torch.Size([3, 1, 2, 1])

# squeeze：删除所有大小为 1 的维度
t2 = torch.squeeze(t1)
print(t2.shape)               # torch.Size([3, 2])

# squeeze(dim=1)：只删除指定维度（如果该维度大小为1）
t3 = torch

## 块 14

```markdown
`transpose(dim0, dim1)` 交换两个指定维度，其他维度不变。对 LoRA 微调，你常需调整注意力头的维度顺序，比如把 `[batch, seq_len, heads, head_dim]` 换成 `[batch, heads, seq_len, head_dim]`，这正是 `transpose(1, 2)` 的典型场景。

```python
t1 = torch.randn(2, 3, 4)
print(t1.shape)               # torch.Size([2, 3, 4])

# 交换第 0 维和第 1 维
t2 = t1.transpose(0, 1)
print(t2.shape)               # torch.Size([3, 2, 4])
```

`permute(dims)` 更灵活，一次性指定所有维度的新顺序。当你需要同时调整多个轴（如从 `[batch, seq, heads, head_dim]` 到 `[batch, heads, seq, head_dim]` 再配合 reshape 合并维度），用 `permute` 一步到位：

```python
t1 = torch.randn(2, 3, 4)
print(t1.shape)               # torch.Size([2, 3, 4])

# 新顺序：原第2维 → 新

## 块 15

**本块核心：分清 `*` 与 `@`，这是你后续实现 LoRA 时矩阵分解的基础。**

你目标是用 LoRA 微调大模型，而 LoRA 的核心操作就是“把权重矩阵拆成两个小矩阵相乘”。所以，这一块必须刻进肌肉记忆。

---

**1. 元素级运算（逐元素）**

`*` 表示对应位置相乘，要求两个张量形状一致：

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

print(a * b)  # tensor([ 4., 10., 18.])
```

同理，`+` 也是逐元素相加。`torch.sqrt(a)` 对每个

## 块 16

### 广播机制（②）——直击LoRA中的形状对齐

在LoRA微调中，你常需将低秩矩阵（如`(3,1)`）与原始权重（如`(3,)`）相加。PyTorch的广播机制让不同形状张量自动对齐：

```python
a = torch.tensor([[1], [2], [3]])   # shape (3, 1)
b = torch.tensor([10, 20, 30])      # shape (3,)

# 广播后相加 => shape (3, 3)
print(a + b)
# tensor([[11, 21, 31],
#         [12, 22, 32],
#         [13, 23, 33]])
```

## 块 17

## 8.1 布尔掩码：用条件“筛选”数据

你在 LoRA 微调中经常需要按条件挑选数据（比如只保留损失低于阈值的样本）。布尔掩码就是干这个的：

```python
x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 3          # 生成布尔张量 [False, False, False, True, True]
print(x[mask])        # tensor([4, 5])  ← 只取出满足条件的元素
```

`mask` 和 `x` 形状相同，每个位置是 `True` 或 `False`。用 `x[mask]` 就能把 `True` 对应位置的元素取出来。

## 8.2 clone：切断共享，独立拷贝

微调时你常需要保留一份“原始权重”做对比。直接赋值是共享内存的：

```python
a = torch.tensor([1, 2, 3])
b = a.clone()         # 深拷贝，b 和 a 不共享内存
b[0] = 100
print(a)              # tensor([1, 2, 3])  ← a 没变
```

> 默认 `b = a` 只是让 `b` 指向同一块内存，改 `b` 会连带改 `a`。需要独立副本时用 `.clone()`。

## 九、张量与 NumPy 互转：默认共享内存

PyTorch 和 NumPy 互转很方便，但**默认共享内存**——改一个，另一个跟着变：

```python
import numpy as np

# Tensor → NumPy
x = torch.tensor([1, 2, 3])
y = x.numpy()
x.add_(1)             # in-place 修改 Tensor
print

## 块 18

## 模拟图像预处理：从张量操作到LoRA实战的桥梁

你正在学习LoRA微调，核心是高效修改预训练模型。但无论模型多复杂，数据进出都依赖张量操作。本场景用一张模拟图像走完预处理全流程，帮你把张量操作与真实任务衔接起来。

### 一、构建模拟图像

```python
import torch

# 模拟一张 3通道 224x224 的图像（0-1范围，模拟真实图像归一化后的数据）
image = torch.rand(3, 224, 224)
print(f"原始图像形状: {image.shape}")   # torch.Size([3, 224, 224])
```

`rand`生成均匀分布数据，形状为`(C, H, W)`——通道在前，这是PyTorch默认格式。此处仅为演示张量操作，使用模拟数据。

### 二、添加batch维度

```python
image_batch = image.unsqueeze(0)
print(f"添加batch后: {image_batch.shape}")  # torch.Size([1, 3, 224, 224])
```

模型一次处理多张图，所以需要batch维度。`unsqueeze(0)`在位置0插入大小为1的维度。

### 三、维度重排：CHW转NHWC

```python
image_nhwc = image_batch.permute(0, 2, 3, 1)
print(f"维度重排后: {image_nhwc.shape}")    # torch.Size([1, 224, 224, 3])
```

`permute`按指定顺序重排维度。`(0, 2, 3, 1)`表示：保持batch位置，把通道维度移到最后。某些框架或算子要求这种`(B, H, W, C)`布局。

### 四、按通道标准化

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
image_norm = (image_nhwc - mean) / std
print(f"标准化后范围: [{image_norm.min():.2f}, {image_norm.max():.2f}]")
```

`view(1, 1, 1, 3)`把均值/标准差形状对齐到`(B, H, W, C)`，触发广播机制逐通道运算。此操作使数据分布接近ImageNet数据集的分布，标准化后数据范围可能超出`[0,1]`。

### 五、移回CHW并展平

```python
image_back = image_norm.permute(0

## 块 19

## 十三、总结与后续学习路线（①+②+④）

恭喜你完成张量操作章节！针对你的LoRA微调目标，这些基础是必须打牢的。你已掌握：

- ✅ **创建与属性**：`torch.tensor`、`zeros`、`ones`、`rand`、`arange`；`shape`、`dtype`、`ndim`、`device`
- ✅ **索引与形状**：基础/范围/列表/布尔索引；`reshape`（安全）、`view`（需连续）、`squeeze`/`unsqueeze`、`transpose`/`permute`/`contiguous`
- ✅ **运算与筛选**：元素级运算、`@`矩阵乘法、广播、聚合统计；布尔掩码、`clone`
- ✅ **互转与设备**：`numpy()`/`from_numpy()`（共享内存）；`.to(device)`、`.cuda()`、`.cpu()`

**为什么这对LoRA重要**：LoRA微调中，你需要频繁操作权重矩阵（形状变换、维度交换、设备切换），并冻结原参数、注入低秩矩阵——这些全部依赖上述张量技能。

> **下一步预告**：下一节 **"自动求导机制"** 将学习PyTorch核心——**Autograd**。你会掌握`requires_grad`、`backward()`、梯度计算图，并实现简单梯度下降

## 测评题目

### Q-KB04-ST0400-003
下列哪项最适合用于确认当前 PyTorch 安装包是否支持 CUDA？

### Q-KB04-ST0400-001
安装带 CUDA 支持的 PyTorch 时，应使 PyTorch 所支持的 CUDA 版本与本机________版本兼容。

### Q-KB0401-004
`x.view(-1, 3)` 中 `-1` 的含义是？

### Q-KB0401-003
请将以下两个张量按**列方向**（dim=1）拼接：
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5], [6]])
```

### Q-KB0404-005
请写代码：使用 `DataLoader` 加载 `MyDataset`，批次大小为16，使用4个子进程，并在每个epoch开始时打乱数据。

### Q-KB0401-005
给定张量 `x` 形状为 `(2, 3, 4)`，请写出将其转置为 `(4, 3, 2)` 的代码。

### Q-KB0403-002
`nn.Linear(in_features=10, out_features=5)` 表示输入维度为________，输出维度为________。

### Q-KB0403-003
请定义一个简单的神经网络类 `SimpleNet`，包含：
- 一个输入层到隐藏层的线性层（10 -> 20）
- ReLU激活
- 隐藏层到输出层的线性层（20 -> 1）
