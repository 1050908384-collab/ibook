# Dataset/DataLoader 数据装载

① 底稿：《详解PyTorch中加载数据的方法--Dataset、Dataloader、Sampler、collate_fn等》\
　　网址：https://blog.csdn.net/qq_41971355/article/details/125616331

② 参考文章：《【youcans的深度学习 11】PyTorch 入门教程：加载数据集》\
　　网址：https://blog.csdn.net/youcans/article/details/130217675

③ 参考文章：《PyTorch自定义数据类的__getitem__ 方法》\
　　网址：https://blog.csdn.net/pai_da_xing8/article/details/148083118

④ 参考文章：《【torch.utils.data.sampler】采样器的解析和使用》\
　　网址：https://blog.csdn.net/zyw2002/article/details/128176507

⑤ 参考文章：《深入解析PyTorch中的collate_fn：从基础到实战应用》\
　　网址：https://blog.csdn.net/weixin_29054245/article/details/157591567

⑥ 参考文章：《torch里dataloader中的pin_memory和num_workers详解》\
　　网址：https://blog.csdn.net/weixin_41002311/article/details/141577076

> **适合人群**：已完成"torch.nn 模块"章节，能够使用 `nn.Module` 搭建简单神经网络的初学者。本节时长约 1 小时，目标是掌握 PyTorch 数据加载的核心机制，能够使用内置数据集、自定义 `Dataset` 和 `DataLoader` 将数据批量喂给模型，并理解 `Sampler`、`collate_fn` 等关键组件的作用。（①+②）
>
> **前置说明**：上节我们学会了搭建神经网络模型，但模型需要"吃"数据才能训练。本节解决的就是**数据怎么来、怎么组织、怎么批量喂给模型**的问题。所有代码请在已安装 PyTorch 的 `pytorch` conda 环境中运行。

---

## 一、为什么需要 Dataset 和 DataLoader？（①+②）

在机器学习项目中，数据通常以文件形式散落在磁盘上（图片、文本、CSV 等）。训练时，我们需要：（①）

1. **按需读取**：数据集可能很大（几十万张图片），无法一次性全部载入内存
2. **批量处理**：模型一次处理一个 batch（如 32 张图片），而不是单张
3. **随机打乱**：每个 epoch 重新打乱数据顺序，防止模型记住固定顺序
4. **多进程加速**：用多个子进程并行加载数据，减少 GPU 等待时间

PyTorch 提供了 `torch.utils.data` 模块来优雅地解决这些问题，核心组件有三者：（①）

| 组件 | 职责 | 类比 |
|---|---|---|
| **Dataset** | 定义数据集：告诉 PyTorch"我的数据在哪里、怎么读取一条样本" | 菜谱（定义食材和做法）|
| **Sampler** | 定义采样策略：告诉 PyTorch"按什么顺序取数据" | 出餐顺序（先上哪道菜）|
| **DataLoader** | 执行加载：整合 Dataset 和 Sampler，批量输出数据 | 厨师（真正做菜并端上桌）|

> **官方文档**：
> - PyTorch Data 加载工具 https://pytorch.org/docs/stable/data.html
> - torchvision 内置数据集 https://pytorch.org/vision/stable/datasets.html

---

## 二、Dataset：定义你的数据集（①+②+③）

`torch.utils.data.Dataset` 是一个抽象类，你需要继承它并实现两个方法：（①+③）

- `__len__(self)`：返回数据集的大小（样本总数）
- `__getitem__(self, index)`：根据索引返回一条样本（数据和标签）

### 2.1 自定义 Dataset 模板（①+③）

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化：加载数据路径、标签等元数据
        self.data = ...          # 例如：图片路径列表
        self.labels = ...        # 例如：对应的标签列表
        self.transform = transform

    def __len__(self):
        return len(self.data)    # 返回数据集大小

    def __getitem__(self, index):
        # 根据索引读取一条样本
        sample = self.data[index]
        label = self.labels[index]

        # 数据预处理/转换
        if self.transform:
            sample = self.transform(sample)

        return sample, label
```

> **关键设计原则**：`__init__` 只加载**元数据**（如路径列表），`__getitem__` 才**真正读取数据**（如打开图片文件）。这样实现"按需加载"，节省内存。（③）

### 2.2 实战：自定义一个数值数据集（①）

```python
import torch
from torch.utils.data import Dataset

class NumberDataset(Dataset):
    def __init__(self, start, end):
        # 生成从 start 到 end-1 的整数作为数据
        self.numbers = list(range(start, end))
        # 标签：奇数为 1，偶数为 0
        self.labels = [n % 2 for n in self.numbers]

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, index):
        # 返回 (数据, 标签)
        x = torch.tensor(self.numbers[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y

# 使用
dataset = NumberDataset(0, 10)
print(len(dataset))          # 10
print(dataset[3])            # (tensor(3.), tensor(1))  ← 奇数标签为1
print(dataset[4])            # (tensor(4.), tensor(0))  ← 偶数标签为0
```

---

## 三、DataLoader：批量加载数据（①+②+⑥）

`DataLoader` 将 `Dataset` 包装成可迭代对象，每次迭代返回一个 batch 的数据。（①）

### 3.1 核心参数速查（①+⑥）

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,            # Dataset 实例
    batch_size=32,      # 每个 batch 的样本数
    shuffle=True,        # 是否每个 epoch 打乱顺序
    num_workers=0,      # 加载数据的子进程数（0=主进程）
    pin_memory=False,   # 是否将数据拷贝到 CUDA 固定内存
    drop_last=False,    # 是否丢弃最后不足 batch_size 的样本
    collate_fn=None     # 如何组装 batch（默认自动堆叠）
)
```

| 参数 | 作用 | 建议 |
|---|---|---|
| `batch_size` | 每批样本数 | 根据显存调整，常见 16/32/64 |
| `shuffle` | 是否打乱 | 训练集 `True`，测试集 `False` |
| `num_workers` | 数据加载子进程数 | 从 0 开始逐步增加，通常设为 CPU 核心数的一半（⑥）|
| `pin_memory` | 固定内存加速 GPU 传输 | GPU 训练时设为 `True`（⑥）|
| `drop_last` | 丢弃不足 batch | 需要固定 batch 大小时设为 `True` |
| `collate_fn` | 自定义 batch 组装逻辑 | 变长数据时需要自定义（见第五节）|

### 3.2 使用示例（①+②）

```python
from torch.utils.data import DataLoader

dataset = NumberDataset(0, 100)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 遍历一个 epoch
for batch_idx, (data, labels) in enumerate(loader):
    print(f"Batch {batch_idx}: data shape={data.shape}, labels shape={labels.shape}")
    if batch_idx == 2:
        break
# 输出：
# Batch 0: data shape=torch.Size([8]), labels shape=torch.Size([8])
# Batch 1: data shape=torch.Size([8]), labels shape=torch.Size([8])
# Batch 2: data shape=torch.Size([8]), labels shape=torch.Size([8])
```

> `DataLoader` 返回的 `data` 自动在第一个维度增加了 batch 维度。原本 `dataset[i]` 返回标量，batch 后变成 `(batch_size,)`。（①）

---

## 四、Sampler：控制采样策略（①+④）

Sampler 决定了从 Dataset 中**按什么顺序取索引**。如果不指定，DataLoader 会根据 `shuffle` 参数自动选择 `SequentialSampler`（`shuffle=False`）或 `RandomSampler`（`shuffle=True`）。（①+④）

### 4.1 常用 Sampler 子类（①+④）

| Sampler | 作用 | 适用场景 |
|---|---|---|
| `SequentialSampler` | 按顺序采样 | 测试集、需要固定顺序时 |
| `RandomSampler` | 随机采样 | 训练集打乱 |
| `SubsetRandomSampler` | 对子集随机采样 | 划分训练/验证集 |
| `WeightedRandomSampler` | 按权重采样 | 类别不平衡时加权采样 |
| `BatchSampler` | 将上述 Sampler 包装成 batch | 通常由 DataLoader 内部使用 |

### 4.2 实战：划分训练集和验证集（②+④）

```python
from torch.utils.data import DataLoader, SubsetRandomSampler
import random

dataset = NumberDataset(0, 100)

# 随机打乱索引后划分
indices = list(range(len(dataset)))
random.shuffle(indices)

train_indices = indices[:80]    # 前80%训练
val_indices = indices[80:]       # 后20%验证

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=8, sampler=val_sampler)

print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
```

### 4.3 实战：类别不平衡时的加权采样（①+④）

```python
from torch.utils.data import WeightedRandomSampler

# 假设 100 个样本中，类别 0 有 90 个，类别 1 有 10 个（严重不平衡）
labels = [0]*90 + [1]*10
# 为每个样本计算权重：少数类权重高
weights = [1.0/90 if label == 0 else 1.0/10 for label in labels]

sampler = WeightedRandomSampler(weights, num_samples=100, replacement=True)
loader = DataLoader(dataset, batch_size=8, sampler=sampler)
```

> `replacement=True` 表示有放回采样，少数类样本可能被多次采到，从而平衡每个 batch 中的类别分布。（①）

---

## 五、collate_fn：批量数据组装（①+⑤）

`collate_fn` 的作用是将一个 batch 的**单个样本**组装成**批量张量**。默认情况下，PyTorch 会自动将相同形状的张量堆叠起来。（⑤）

### 5.1 默认行为（⑤）

```python
# 默认 collate_fn 的行为示意：
# 输入：[(tensor(x1), tensor(y1)), (tensor(x2), tensor(y2)), ...]
# 输出：(tensor([x1, x2, ...]), tensor([y1, y2, ...]))
```

### 5.2 什么时候需要自定义？（⑤）

当样本**形状不一致**时（如 NLP 中句子长度不同），默认 collate_fn 会报错，需要自定义。

### 5.3 实战：变长序列的 padding（⑤）

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self):
        # 模拟不同长度的文本序列（已转为 token 索引）
        self.texts = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9]),
            torch.tensor([10])
        ]
        self.labels = [0, 1, 0, 1]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

# 自定义 collate_fn：对变长序列进行 padding
def collate_fn_pad(batch):
    # batch 是列表，每个元素是 (text, label)
    texts = [item[0] for item in batch]      # 提取所有文本
    labels = torch.tensor([item[1] for item in batch])  # 提取所有标签

    # pad_sequence 将序列填充到相同长度（以 0 填充）
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, labels

dataset = TextDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_pad)

for texts, labels in loader:
    print(f"Texts shape: {texts.shape}, Labels: {labels}")
# 输出：
# Texts shape: torch.Size([2, 3]), Labels: tensor([0, 1])
# Texts shape: torch.Size([2, 4]), Labels: tensor([0, 1])
```

> `pad_sequence` 会自动将较短的序列用 `padding_value`（默认 0）填充到 batch 内最大长度。（⑤）

---

## 六、内置数据集与本地数据加载（②）

PyTorch 生态提供了大量内置数据集，开箱即用。

### 6.1 torchvision 内置图像数据集（②）

```python
from torchvision import datasets, transforms

# 定义图像预处理流程
transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL Image → Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化
])

# 加载 MNIST 数据集（自动下载到 ../dataset 目录）
train_set = datasets.MNIST(root='../dataset', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print(f"训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")
```

> 其他常用内置数据集：`FashionMNIST`、`CIFAR10`、`CIFAR100`、`ImageFolder` 等。完整列表见 https://pytorch.org/vision/stable/datasets.html（②）

### 6.2 用 ImageFolder 加载本地分类数据集（②）

如果你的本地数据按文件夹组织（每个类别一个文件夹），可以直接用 `ImageFolder`：

```
dataset/
├── cat/
│   ├── cat_01.jpg
│   └── cat_02.jpg
├── dog/
│   ├── dog_01.jpg
│   └── dog_02.jpg
```

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root='./dataset', transform=transform)
print(dataset.classes)        # ['cat', 'dog']
print(dataset.class_to_idx)   # {'cat': 0, 'dog': 1}

loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 6.3 加载本地文件的自定义 Dataset（②+③）

如果数据结构不符合 `ImageFolder` 要求，需要自定义 Dataset：

```python
import os
from PIL import Image

class LocalImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        # 从标签文件读取 (图片名, 标签)
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.samples.append((img_name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, label = self.samples[index]
        img_path = os.path.join(self.image_dir, img_name)

        # 读取图片（按需加载，节省内存）
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
```

---

## 七、综合实战演练（①+②+⑤）

请在你的 Python 环境中完成以下操作：

### 场景：构建一个完整的图像分类数据管道

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

# ========== 1. 自定义 Dataset（模拟 CIFAR-10 风格的数据） ==========
class DummyImageDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, transform=None):
        self.transform = transform
        # 模拟数据：随机生成 32x32 的"图片"和标签
        self.images = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

# ========== 2. 定义数据增强变换 ==========
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到 [-1, 1]
])

test_transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ========== 3. 创建数据集并划分训练/验证集 ==========
full_dataset = DummyImageDataset(num_samples=1000, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# ========== 4. 创建 DataLoader ==========
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,      # 先设为0确保无报错，熟悉后可逐步增加
    pin_memory=False    # CPU训练时False，GPU训练时True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# ========== 5. 模拟一个训练循环 ==========
for epoch in range(2):
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Epoch {epoch}, Batch {batch_idx}: {images.shape}, {labels.shape}")
        if batch_idx == 2:
            break
```

---

## 八、常见问题与解决（①+⑤+⑥）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `RuntimeError: stack expects each tensor to be equal size` | 数据形状不一致，默认 collate_fn 无法堆叠 | 自定义 `collate_fn`，对变长数据做 padding |
| 数据加载很慢，GPU 利用率低 | `num_workers=0`，数据加载成为瓶颈 | 逐步增加 `num_workers`（建议从 2 或 4 开始）|
| `pin_memory=True` 后报错 | 在 CPU 上训练时不需要 pin_memory | GPU 训练时开启，CPU 训练时关闭 |
| `ImageFolder` 报错 `Found 0 files` | 文件夹结构不符合要求 | 确保是 `root/class_name/image.jpg` 的层级结构 |
| 多进程加载报错 `RuntimeError: DataLoader worker exited unexpectedly` | `num_workers > 0` 时主进程数据不可序列化 | 将 Dataset 的数据改为可序列化类型，或在 `__main__` 中运行 |
| 每个 epoch 数据顺序相同 | `shuffle=False` | 训练集的 DataLoader 必须设置 `shuffle=True` |
| 验证集结果波动大 | 验证时 `shuffle=True` | 验证和测试必须设置 `shuffle=False` |

---

## 九、总结与后续学习路线（①+②）

恭喜你完成了 Dataset/DataLoader 数据装载章节的学习！你现在应该掌握了：

- ✅ Dataset 的核心作用：继承 `Dataset`，实现 `__len__` 和 `__getitem__`
- ✅ DataLoader 的核心参数：`batch_size`、`shuffle`、`num_workers`、`pin_memory`、`drop_last`
- ✅ Sampler 的作用：`SequentialSampler`、`RandomSampler`、`SubsetRandomSampler`、`WeightedRandomSampler`
- ✅ collate_fn 的作用：默认堆叠行为、自定义 padding 处理变长序列
- ✅ 内置数据集：`torchvision.datasets.MNIST`、`ImageFolder` 等
- ✅ 自定义本地数据集：按需加载、数据增强、错误处理

> **下一步预告**：在下一节 **"优化器与损失函数"** 中，我们将学习 PyTorch 提供的优化工具。你会了解 SGD、Adam、AdamW 等优化器的工作原理，学习如何配置学习率、权重衰减等超参数，并把它们与 `nn.Module` 的 `parameters()` 结合起来。随后进入 **"训练循环编写"**，我们将把本节的数据加载、上节的模型定义、以及下节的优化器串成一个完整的训练流程。现在，试着为你手头的数据写一个自定义 Dataset 吧！