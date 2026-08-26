# 学习包

## 块 1

## 二、Dataset：定义你的数据集（①+②+③）

### 2.1 自定义 Dataset 模板

上节我们搭建了模型，但模型需要"吃"数据才能训练。本节解决的是**数据怎么来、怎么组织、怎么批量喂给模型**的问题。所有代码请在已安装 PyTorch 的 `pytorch` conda 环境中运行。

#### 为什么需要 Dataset 和 DataLoader？

训练时，我们需要：（①）

1. **按需读取**：数据集可能很大（几十万张图片），无法一次性全部载入内存
2. **批量处理**：模型一次处理一个 batch（如 32 张图片），而不是单张
3. **随机打乱**：每个 epoch 重新打乱数据顺序，防止模型记住固定顺序
4. **多进程加速**：用多个子进程并行加载数据，减少 GPU 等待时间

PyTorch 提供了 `torch.utils.data` 模块来解决这些问题，核心组件有三者：（①）

| 组件 | 职责 | 类比 |
|---|---|---|
| **Dataset** | 定义数据集：告诉 PyTorch"我的数据在哪里、怎么读取一条样本" | 菜谱（定义食材和做法）|
| **Sampler** | 定义采样策略：告诉 PyTorch"按什么顺序取数据" | 出餐顺序（先上哪道菜）|
| **DataLoader** | 执行加载：整合 Dataset 和 Sampler，批量输出数据 | 厨师（真正做菜并端上桌）|

#### 自定义 Dataset 的三个核心方法

自定义 Dataset 需要继承 `torch.utils.data.Dataset` 类，并实现三个方法：（①+③）

**`__init__`**：初始化数据集。在这里读取文件列表、标签等元信息，但**不加载全部数据到内存**。例如，保存图片路径和对应标签的列表。

**`__len__`**：返回数据集的总样本数。DataLoader 通过它知道一共有多少条数据，从而计算每个 epoch 的步数。

**`__getitem__`**：根据索引返回一条样本。这是最核心的方法——当 DataLoader 需要第 `i` 条数据时，会调用 `dataset[i]`，你需要在里面完成"读取文件 → 预处理 → 返回 (数据, 标签)"的完整流程。（③）

#### 一个最小模板

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, file_list, label_list):
        self.file_list = file_list   # 文件路径列表
        self.label_list = label_list # 对应标签列表
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 读取第 idx 个文件，做预处理
        # data = load_file(self.file_list[idx])
        # label = self.label_list[idx]
        # return data, label
        pass
```

**关键理解**：`__getitem__` 是"按需读取"的入口。DataLoader 每次只调用它获取一个 batch 所需的数据，而不是一次性加载全部。这正是大数据集无法全部载入内存时的解决方案。（①+③）

#### 本块学习检查

先用一个最小示例回顾 Dataset/DataLoader 数据装载：请创建一个包含 10 个样本的简单 Dataset（数据可用随机数代替），用 `DataLoader` 以 `batch_size=4` 加载，并打印第一个 batch 的形状。

## 块 2

## 2.1 自定义 Dataset 模板：为 LoRA 微调准备数据

你的目标是 **LoRA 微调 + 权重合并 + vLLM 部署**，整个流程的第一步，就是让 PyTorch 知道“你的数据长什么样、怎么取”。`Dataset` 就是干这个的。

### 核心：继承 + 实现两个方法

`torch.utils.data.Dataset` 是一个抽象类，你不能直接用它，必须**继承**它并实现两个方法：

- **`__len__(self)`**：返回数据集大小（样本总数）
- **`__getitem__(self, index)`**：根据索引返回一条样本（数据和标签）

### 模板拆解

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

### 关键设计原则（务必记住）

**`__

## 块 3

## 二、Dataset：定义你的数据集（①+②+③）

### 2.2 实战：自定义一个数值数据集（①）

你的目标是 **LoRA 微调 + 权重合并 + vLLM 部署**，而这一切的起点，是先把数据喂给模型。这里我们直接动手，用 PyTorch 自定义一个数值数据集，让你快速掌握 `Dataset` 的核心套路。

**代码先行，边看边想：**

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
print(dataset[4])            # (tensor(4.), tensor(0))
```

## 块 4

### 3.1 核心参数速查

`DataLoader` 是 PyTorch 中批量喂数据的核心工具。你只需传入一个 `Dataset` 实例，再配好下面几个参数即可。

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,            # Dataset 实例
    batch_size=32,      # 每个 batch 的样本数
    shuffle=True,       # 是否每个 epoch 打乱顺序
    num_workers=0,      # 加载数据的子进程数（0=主进程）
    pin_memory=False,   # 是否将数据拷贝到 CUDA 固定内存
    drop_last=False,    # 是否丢弃最后不足 batch_size 的样本
    collate_fn=None     # 如何组装 batch（默认自动堆叠）
)
```

**关键参数速查：**

- **`batch_size`**：每批样本数。显存不够就调小，常见 16/32/64。
- **`shuffle`**：训练集设 `True` 打乱顺序，测试集设 `False`。
- **`num_workers`**：数据加载子进程数。从 0

## 块 5

## 3.2 使用示例：让 DataLoader 帮你批量取数

你已经知道 `Dataset` 负责“按索引取一个样本”，但训练时我们不会一个个喂数据，那样太慢。`DataLoader` 就是那个帮你**自动打包**的工具。

看下面这段代码——它把 `NumberDataset(0, 100)`（100个样本）装进 `DataLoader`，设置 `batch_size=8`（每批8个）和 `shuffle=True`（打乱顺序）：

```python
from torch.utils.data import DataLoader

dataset = NumberDataset(0, 100)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 遍历一个 epoch
for batch_idx, (data, labels) in enumerate(loader):
    print(f"Batch {batch_idx}: data shape={data.shape}, labels shape={labels.shape}")
    if batch_idx == 2:
        break
```

输出结果：

```
Batch 0: data shape=torch.Size([8]), labels shape=torch.Size([8])
Batch 1: data shape=torch.Size([8]), labels shape=torch.Size([8])
Batch 2: data shape=torch.Size([8]), labels shape=torch.Size([8])
```

**关键点**：`DataLoader` 返回的 `data` 自动在**第一个维度**增加了 batch 维度。原本 `dataset[i]` 返回的是标量，经过 batch 后变成 `(batch_size,)` 的形状——也就是 `(8,)`。这意味着你拿到的不再是单个样本，而是一整批样本，可以直接喂给模型做矩阵运算。

**对你后续 LoRA 微调的意义**：微调

## 块 6

**Sampler 速览：控制数据顺序的“调度员”**

你的目标是 LoRA 微调 + vLLM 部署，核心是数据流控制。Sampler 决定数据喂给模型的顺序，直接影响训练效果。

- **SequentialSampler**：按顺序取数据，适合测试集或需要固定顺序时。
- **RandomSampler**：随机打乱，训练集必备，避免模型记住顺序。
- **SubsetRandomSampler**：先划分子集再随机，常用于手动切分训练/验证集。
- **WeightedRandomSampler**：按权重采样，类别不平衡时让少数类被抽中更多。
- **BatchSampler**：接受一个 Sampler 对象（如 SequentialSampler

## 块 7

你正在准备用LoRA微调模型，之后还要合并权重并用vLLM部署。这个流程里，**训练集和验证集的划分质量**会直接影响微调效果和部署后的泛化能力。下面这段代码就是划分数据集的实战模板，直接对着敲一遍，边敲边想每一步在干什么。

```python
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import logging

dataset = NumberDataset(0, 100)

try:
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
except Exception as e:
    logging.error(f"数据加载或划分失败: {e}")
    raise
```

**拆解一下关键动作：**

1. **`indices = list(range(len(dataset)))`

## 块 8

## 4.3 实战：类别不平衡时的加权采样

你的目标是 **LoRA微调 + 权重合并 + vLLM部署**，而微调阶段最常见的一个坑就是**类别不平衡**——比如你处理的数据里90%是“正常”样本，只有10%是“异常”样本。如果直接训练，模型会偷懒，全猜“正常”就能拿到90%准确率，但你的任务就废了。

**核心思路**：让少数类样本在采样时被“多抽几次”，强行让每个batch里两类样本数量均衡。

直接看代码，跟着敲一遍：

```python
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 构造严重不平衡数据集：类别0有90个，类别1有10个
class ImbalancedDataset(Dataset):
    def __init__(self):
        self.labels = [0] * 90 + [1] * 10
        self.features = torch.arange(len(self.labels), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], torch.tensor(self.labels[index], dtype=torch.long)

dataset = ImbalancedDataset()
labels = dataset.labels
# 为每个样本计算权重：少数类权重高
weights = [1.0/90 if label == 0 else 1.0/10 for label in labels]

sampler = WeightedRandomSampler(weights, num_samples=100, replacement=True)
loader = DataLoader(dataset, batch_size=8, sampler=sampler)
```

**关键点拆解**：

- **权重计算**：`1.0/90`（多数类）和 `1.0/10`（少数类）。权重与样本数量成反比——样本越少，权重越高。这样少数类被抽中的概率更高，从而平衡每个batch中的类别分布。

## 块 9

## 5.1 默认行为：DataLoader 怎么自动打包数据？

先看默认情况。当你用 `DataLoader` 加载数据时，它内部有一个默认的 `collate_fn`，负责把一批样本“叠”成一个批次。它的工作方式很简单：

- **输入**：一个列表，里面每个元素是 `(tensor(x_i), tensor(y_i))`，即一个样本的特征和标签。
- **输出**：两个张量——`tensor([x1, x2, ...])` 和 `tensor([y1, y2, ...])`，即把所有样本的特征堆叠成一个更高维的张量，标签同理。

**关键限制**：默认 `collate_fn` 要求每个样本的 `x_i` 形状完全一致，才能堆叠成规则的矩形张量。一旦形状不一致，它就会直接报错。

---

## 5.2 什么时候要自定义？

最常见的场景是 **NLP 中的变长序列**。比如一句话有 3 个 token，另一句有 7 个 token，它们的张量长度不同，默认 `collate_fn` 无法把它们堆叠成同一形状的张量，就会报错。

这时就需要你**自己写一个 `collate_fn`**，告诉 DataLoader 如何处理这种形状不一致的样本。

---

## 5.3 实战：用 padding 处理变长序列

下面是一个完整的例子，模拟了 4 个长度不同的文本序列（已转为 token 索引），并用 `pad_sequence` 把它们填充到相同长度。

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
```

## 块 10

## 6.1 torchvision 内置图像数据集（②）——为LoRA微调做准备

你接下来的目标是 **LoRA微调 + 权重合并 + vLLM部署**，但一切微调都始于“把数据正确装进模型”。这一块，我们聚焦 **torchvision 内置图像数据集** 的加载方式——这是你后续做图像LoRA微调时的数据入口。

### 核心代码拆解

```python
from torchvision import datasets, transforms

# 定义图像预处理流程
transform = transforms.Compose([
    transforms.ToTensor(),                    # PIL Image → Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化
])
```

**这段在做什么？** 每个图像数据集在喂给模型前，都要先“格式化”。`Compose` 把多个预处理步骤按顺序打包：
- `ToTensor()`：把PIL图像转成PyTorch张量（Tensor），这是模型能读的格式；
- `Normalize((0.1307,), (0.3081,))`：标准化，让像素值分布更规整，训练更稳定。

```python
# 加载 MNIST 数据集（自动下载到 ../dataset 目录）
train_set = datasets.MNIST(root='../dataset', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)
```

**关键参数逐个看：**
- `root='../dataset'`：数据下载/存放位置；
- `train=True/False`：True拿训练集，False拿测试集。

## 块 11

## 6.2 用 ImageFolder 加载本地分类数据集（②）

你的目标是把本地图片喂给模型做微调，第一步就是让 PyTorch 认识你的数据。`ImageFolder` 就是为此设计的：只要你的数据按“每个类别一个文件夹”组织，它就能自动读取。

**数据组织方式**（这是前提，务必照做）：

```
dataset/
├── cat/      # 文件夹名即类别名
│   ├── cat_01.jpg
│   └── cat_02.jpg
├── dog/
│   ├── dog_01.jpg
│   └── dog_02.jpg
```

**三步加载代码**：

```python
from torchvision import datasets, transforms

# 1. 定义预处理：统一尺寸 + 转张量
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 2. 加载数据集：指定根目录和预处理
dataset = datasets.ImageFolder(root='./dataset', transform=transform)

# 3. 查看自动生成的标签映射
print(dataset.classes)        # ['cat', 'dog']

## 块 12

## 6.3 自定义 Dataset 加载本地文件

当图片数据是“一张图片对应一个标签”格式（如图片名列表加标签文件），而非按类别分文件夹存放时，`ImageFolder` 无法使用。此时需自定义 Dataset 类，核心三件事：**读标签、数长度、按索引取数据**。

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class LocalImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        # 从标签文件读取 (图片名, 标签)
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    img_name, label = line.strip().split()
                    self.samples.append((img_name, int(label)))
        except FileNotFoundError:
            raise FileNotFoundError(f"标签文件未找到: {label_file}")
        except OSError as e:
            raise OSError(f"读取标签文件失败: {label_file}, 错误: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, label = self.samples[index]
        img_path = os.path.join(self.image_dir, img_name)

        # 读取图片（按需加载，节省内存）
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"图片文件未找到: {img_path}")
        except OSError as e:
            raise OSError(f"图片加载失败: {img_path}, 错误: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label
```

**拆开来看：**

- `__

## 块 13

```markdown
## 实战演练：构建图像分类数据管道

你的目标是完成 **LoRA微调 → 权重合并 → vLLM部署** 这条链路，而这一切的起点，是先把数据喂给模型。本块我们聚焦一个完整的数据管道：从自定义 `Dataset` 到 `DataLoader`，再到训练循环的雏形。

### 1. 自定义 Dataset：数据从哪里来

PyTorch 中，`Dataset` 负责定义“如何取一条数据”。看下面的类：

```python
class DummyImageDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, transform=None):
        self.transform = transform
        self.images = torch.randn(num_samples, 3, 32, 32)  # 模拟 32x32 彩色图
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label
```

**核心机制**：`__len__` 告诉 DataLoader 总共有多少样本；`__getitem__` 按索引返回 `(图片, 标签)` 对。这里用随机张量模拟 CIFAR-10 风格的 1000 张 32×32 彩色图，10 个类别。

### 2. 数据增强：训练集和测试集的区别

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

**关键差异**：训练集多了 `RandomHorizontalFlip`（随机水平翻转），这是一种数据增强，让模型见过更多样化的输入；测试集只做标准化，保证评估的稳定性。`Normalize` 将像素值标准化到 `[-1, 1]`。

### 3. 划分训练/验证集：一个易踩的坑

```python
base_dataset = DummyImageDataset(num_samples=1000)
train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size
train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

train_dataset = DummyImageDataset(num_samples=0, transform=train_transform)
train_dataset.images = base_dataset.images[train_subset.indices]
train_dataset.labels = base_dataset.labels[train_subset.indices]

val_dataset = DummyImageDataset(num_samples=0, transform=test_transform)
val_dataset.images = base_dataset.images[val_subset.indices]
val_dataset.labels = base_dataset.labels[val_subset.indices]
```

**为什么这么绕？** `random_split` 返回的 `Subset` 会共享同一个 `Dataset`，无法为训练/验证集分别设置不同的变换。所以这里先固定索引，再创建两个使用相同原始数据、不同变换的新 `Dataset`。注意 `num_samples=0` 创建空对象，然后手动填充 `images` 和 `labels`。

### 4. DataLoader：批量喂数据

```python
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
```

**参数要点**：
- `batch_size=32`：每次取 32 个样本
- `shuffle=True`（训练集）：每个 epoch 打乱顺序，避免模型学到样本顺序
- `shuffle=False`（验证集）：不打乱，保证评估一致性
- `num_workers=0`：先用单进程，确保无报错；`pin_memory=False` 适配 CPU 训练

### 5. 训练循环雏形

```python
for epoch in range(2):
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Epoch {epoch}, Batch {batch_idx}: {images.shape}")
```

## 块 14

## 八、常见问题与解决

跑通管道只是第一步，实战中你大概率会遇到下面这些坑。按“症状→原因→方案”对照排查：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `RuntimeError: stack expects each tensor to be equal size` | 数据形状不一致，默认 `collate_fn` 无法堆叠 | 自定义 `collate_fn`，对变长数据做 padding |
| 数据加载很慢，GPU 利用率低 | `num_workers=0`，数据加载成为瓶颈 | 逐步增加 `num_workers`（建议从 2 或 4 开始）|
| 开启 `pin_memory=True` 后没有加速 | CPU 训练或数据传输并非瓶颈 | 它通常不会报错；仅在 CUDA 训练且数据加载成为瓶颈时再评估是否开启 |
| `ImageFolder` 报错 `Found 0 files` | 文件夹结构不符合要求 | 确保是 `root/class_name/image.jpg` 的层级结构 |
| 多进程加载报错 `RuntimeError: DataLoader worker exited unexpectedly` | `num_workers > 0` 时主进程数据不可序列化 | 将 Dataset 的数据改为可序列化类型，或在 `__main__` 中运行 |
| 每个 epoch 数据顺序相同 | `shuffle=False` | 训练集的 DataLoader 应设置 `shuffle=True`，验证集和测试集应设置 `shuffle=False`，因为训练时需要打乱顺序以增强泛化，而验证/测试时需保持顺序以稳定评估 |
| 验证集结果波动大 | 验证时 `shuffle=True` | 验证和测试必须设置 `shuffle=False` |

**排查思路**：先看报错类型——形状问题找 `collate_fn`，性能问题调 `num_workers`，目录问题查 `ImageFolder` 结构，顺序问题查 `shuffle`。

## 九、总结与后续学习路线

恭喜你完成 Dataset/DataLoader 数据装载章节！你现在应该掌握了：

- ✅ Dataset 核心作用：继承 `Dataset`，实现 `__len__` 和 `__getitem__`
- ✅ DataLoader 核心参数：`batch_size`、`shuffle`、`num_workers`、`pin_memory`、`drop_last`
- ✅ Sampler 作用：`SequentialSampler`、`RandomSampler`、`SubsetRandomSampler`、`WeightedRandomSampler`
- ✅ `collate_fn` 作用：默认堆叠行为、自定义 padding 处理变长序列
- ✅ 内置数据集：`torchvision.datasets.MNIST`、`ImageFolder` 等
- ✅ 自定义本地数据集：按需加载

## 块 15

### 2.1 L1Loss：平均绝对误差（MAE）

**一句话理解**：L1Loss 就是“预测值和真实值差多少，取绝对值，再求平均”。它只关心“差了多少”，不关心“差的方向”。

**为什么你现在需要它？**  
你的目标是 LoRA 微调 + vLLM 部署。在微调阶段，损失函数决定模型“学什么”；在部署阶段，你不再关心损失，但微调时选错损失函数，部署效果会直接崩。L1Loss 是回归任务（预测连续数值）的入门选择，也是你后续理解更复杂损失函数（如 Smooth L1）的基础。

**代码与机制**  
```python
criterion = nn.L1Loss()
```
- **计算方式**：对每个样本，计算 `|预测值 - 真实值|`，然后对所有样本取平均。
- **关键特性**：L1Loss对异常值的梯度影响是恒定的（梯度为±1），不会像MSE那样因误差增大而梯度爆炸，因此对异常值更鲁棒。这适合数据中存在少量噪声或极端值的回归问题。

**动手验证（不引入新事实）**  
在 `pytorch` conda 环境中运行：
```python
import torch
import torch.nn as nn

criterion = nn.L1Loss()
pred = torch.tensor([2.0, 3.0])
target = torch.tensor([1.0, 5.0])
loss = criterion(pred, target)
print(loss)  # 手动算一下：|2-1| + |3-5| = 1 + 2 = 3，平均 = 1.5
```
观察输出是否为 `1.5`。这能帮你确认自己对“平均绝对误差”的理解是否正确。

**与你的学习路径的关联**  
- 你在 LoRA 微调时，如果任务是回归型（比如预测连续值），L1Loss 是一个可选损失函数。
- 但注意：LoRA 微调

## 块 16

## 2.2 MSELoss：均方误差——回归任务的默认选择

你的目标是 **LoRA微调 + 权重合并 + vLLM部署**，这条链路里回归类任务（比如预测分数、连续值）会频繁用到 MSELoss。我们直接上手。

### 核心用法

```python
criterion = nn.MSELoss(reduction='mean')
```

**它做了什么？** 计算预测值与真实值之间**平方差值的平均值**。平方操作意味着：**大误差会被放大惩罚**（比如误差 2 的惩罚是误差 1 的 4 倍），这让模型更倾向于避免大偏差。

### reduction 参数：控制输出形式

| 取值 | 返回内容 | 适用场景 |
|------|----------|----------|
| `'mean'`（默认） | 所有样本损失的**平均值** | 最常用，梯度尺度稳定 |
| `'sum'` | 所有样本损失的**总和** | 需要精确累计损失时 |
| `'none'` | **不聚合**，返回每个样本的损失，形状与输入相同（若输入有额外维度则保留） | 想逐样本分析误差分布时 |

### 实操建议（贴合你的 hands-on 风格）

在微调脚本里，你通常会这样组合：

```python
loss = criterion(outputs, targets)   # outputs 和 targets 形状需一致
loss.backward()                      # 反向传播
```

**一个常见坑**：`outputs` 和 `targets` 的**形状必须完全一致**，否则会报错。调试时先用 `.shape` 确认。

---

**本块学习检查**：  
用 `torch.randn(4, 1)` 生成预测值，用 `torch.tensor([[0.5],

## 块 17

**本块学习检查**：对照速查表，说出 `CrossEntropyLoss` 的输入形状要求，并解释为何它内部已含 softmax 仍适合多分类。

你的目标是 LoRA 微调后合并权重再部署，微调时分类头常用 `CrossEntropyLoss`，回归头则用 `L1Loss` 或 `MSELoss`。速查：

- **L1Loss**：回归，预测与目标形状相同，对异常值不敏感。
- **MSELoss**：回归，形状相同，大误差惩罚重。
- **CrossEntropyLoss**：多分类，输入 logits `(N, C)` + 索引 `(N,)`，内部含 softmax，数值稳定

## 块 18

### 3.1 SGD with Momentum：经典但有效

SGD（随机梯度下降）是最基础的优化器。加入 Momentum（动量）后，优化器会“记住”之前的更新方向，像滚雪球一样加速收敛并减少震荡。

**核心参数**（PyTorch 示例）：

```python
from torch.optim import SGD

optimizer = SGD(
    model.parameters(),
    lr=0.01,          # 学习率
    momentum=0.9,     # 动量系数（0.9 是常用值）
    weight_decay=1e-4 # 权重衰减（L2 正则化）
)
```

- `lr`：学习率，控制每次更新的步长。SGD 对学习率敏感，通常需要配合学习率调度器使用。
- `momentum`：动量系数。设为 0 表示纯 SGD，设为 0.9 表示带动量。
- `weight_decay`：L2 正则化系数，防止过拟合

## 块 19

**Adam：让每个参数自己决定步长**

你之前可能用过 SGD，它对所有参数用同一个学习率。Adam 的核心改进是：**每个参数都有自己独立的自适应学习率**，收敛更快，而且对学习率的选择不那么敏感。

```python
from torch.optim import Adam

optimizer = Adam(
    model.parameters(),
    lr=0.001,            # 学习率（通常比 SGD 小一个数量级）
    betas=(0.9, 0.999),  # 一阶矩和二阶矩衰减率
    eps=1e-8,            # 数值稳定性常数
    weight_decay=0       # 权重衰减（注意：Adam 中此参数有坑！）
)
```

- `betas=(0.9, 0.999)`：第一个值控制动量（类似 SGD 的惯性），第二个值控制自适应步长（

## 块 20

### 3.3 AdamW：解耦权重衰减（重点）

AdamW 是 Adam 的改进版，核心改动只有一点：**将权重衰减与梯度更新彻底解耦**。

#### 通俗理解

把训练模型比作"培养运动员"：

- **梯度更新** = 根据训练表现调整营养配比（让运动员跑得更快）
- **权重衰减** = 控制饮食防止过度肥胖（防止过拟合）

Adam 的问题是：它把"减肥计划"混进了"营养配比"里，由智能手环（自适应学习率）统一调配。结果手环看到今天训练量大（梯度大），就自作主张放松减肥计划，导致减肥力度不稳定。

AdamW 的解决方案：**吃饭归吃饭，减肥归减肥**。先按 Adam 的规则更新参数，再单独、固定地执行权重衰减，互不干扰。

#### 数学对比

- **原始 Adam**（带 weight_decay）：参数更新项中包含被自适应学习率缩放过的权重衰减
- **AdamW**：权重衰减独立于自适应更新，直接对参数进行衰减

#### PyTorch 用法

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-

## 块 21

**优化器速查：选对工具，少走弯路**

你接下来的目标是 LoRA 微调 + 权重合并 + vLLM 部署，核心战场是 Transformer 模型。这里直接给你一张速查表，帮你快速锁定该用谁：

| 优化器 | 核心特点 | 学习率建议 | 适用场景 |
|---|---|---|---|
| **SGD + Momentum** | 固定学习率 + 动量 | 0.01 ~ 0.1 | 图像分类，追求泛化 |
| **Adam** | 自适应学习率 | 0.001 | 快速原型验证（建议改用 AdamW） |
| **AdamW** | 自适应 + 解耦权重衰减 | 0.001 | **Transformer、大多数

## 块 22

**StepLR：阶梯式衰减**

你的目标是LoRA微调后合并权重并部署vLLM，训练时学习率不能一成不变。StepLR的做法是：每隔固定步数，学习率乘以一个衰减系数。

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# 每 30 个 epoch，学习率 × 0.1
```

**CosineAnnealingLR：余弦退火**

现代训练更常用余弦退火：学习率按余弦曲线从初始值平滑下降到接近0。

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
# T_max：半个余弦周期的 epoch 数
# eta_min

## 块 23

好的，我们直接进入正题。你现在的目标是完成 LoRA 微调，然后合并权重并用 vLLM 部署。这个流程里，学习率调度器是让模型训练得更稳、效果更好的关键一环。我们直接看它怎么用。

## 4.3 调度器怎么用：每个 epoch 结束后“动”一下

你不需要在每个 batch 之后都调整学习率，那样太频繁了。标准做法是**在每个 epoch（遍历完一遍所有训练数据）结束后**，调用一次 `scheduler.step()`。

看这个简化的循环结构，注意 `scheduler.step()` 的位置：

```python
for epoch in range(num_epochs):
    for data, target in train_loader:
        # ... 训练步骤 ...
        optimizer.step()  # 更新模型参数
    scheduler.step()   # ← 每个 epoch 结束后，更新学习率
```

## 五、完整训练循环：把损失函数、优化器、调度器串起来

下面是一个完整的、可以直接运行的训练代码骨架。它把前面学的 `CrossEntropyLoss`（损失函数）、`AdamW`（优化器）和 `CosineAnnealingLR`（学习率调度器）组合在了一起。你重点看代码的结构和注释，理解每个部分是怎么协作的。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ========== 1. 准备数据（以CIFAR10为例） ==========
# 注意：训练CIFAR10需至少4GB显存
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
try:
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
except Exception as e:
    print(f"数据集下载或加载失败: {e}")
    raise
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# ========== 2. 定义模型（简单CNN） ==========
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleCNN()

# ========== 3. 损失函数 ==========
criterion = nn.CrossEntropyLoss()

# ========== 4. 优化器（AdamW） ==========
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# ========== 5. 学习率调度器（余弦退火） ==========
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# ========== 6. 训练循环 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()   # 1. 清零旧梯度
        loss.backward()         # 2. 计算新梯度
        optimizer.step()        # 3. 更新参数

        running_loss += loss.item()

    # 每个 epoch 结束后更新学习率
    scheduler.step()

    avg_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
```

## 块 24

## 四、学习率调度器：让学习率“动”起来（⑥）——4.3 使用方式

你已经掌握了 AdamW 的核心原理，现在进入实战环节：**学习率调度器**。它的作用是让学习率在训练过程中**动态变化**，而不是固定不变。

### 为什么需要调度器？

训练后期，如果学习率一直不变，损失值容易**停滞**。这时就需要调度器出场——它能让学习率**逐步衰减**，帮助模型在后期更精细地收敛。

### 两种常用调度器

| 调度器 | 行为 | 适用场景 |
|---|---|---|
| `StepLR` | 每隔固定步数，学习率乘一个衰减因子 | 训练阶段清晰，需要阶梯式下降 |
| `CosineAnnealingLR` | 按余弦曲线平滑衰减 | 训练后期损失停滞时优先尝试 |

### 使用方式（核心口诀）

调度器与优化器**配合使用**，在训练循环中按固定顺序调用：

**训练循环四步口诀**：前向 → 反向 → 清零 → 更新（`zero_grad()` 应在 `backward()` 之前调用）。

调度器的调用时机在优化器更新之后，具体调用频率取决于调度器类型：基于 epoch 的调度器（如 `StepLR`）在每个 epoch 结束后调用 `scheduler.step()`；基于 step 的调度器（如 `CosineAnnealingLR`）在每个 `optimizer.step()` 后调用。

---

## 六、常见问题速查表（实战排错）

| 症状 | 可能原因 |

## 测评题目

### Q-KB0404-001
自定义数据集类需要继承的基类是？

### Q-KB0404-003
请实现一个简单的自定义数据集类 `MyDataset`，接收数据列表 `data` 和标签列表 `labels`，实现 `__len__` 和 `__getitem__` 方法。

### Q-KB0405-004
`CrossEntropyLoss` 的输入 `logits` 是否需要先经过 `Softmax`？

### Q-KB0405-001
以下哪个是**交叉熵损失**函数？

### Q-KB0407-005
请写代码：从 `checkpoint.pth` 加载模型和优化器状态，并恢复训练。

### Q-KB0407-003
请写代码：保存模型 `model` 的参数到 `checkpoint.pth`，并包含优化器状态和当前epoch。

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

### Q-KB0401-003
请将以下两个张量按**列方向**（dim=1）拼接：
```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5], [6]])
```
