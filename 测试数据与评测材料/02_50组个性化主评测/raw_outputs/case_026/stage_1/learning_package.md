# 学习包

## 块 1

# 参数高效微调（LoRA与QLoRA）环境准备：安装 PEFT、bitsandbytes 并配置GPU环境（最终版）

## 参考资料

① 权威：《Hugging Face PEFT Quicktour：参数高效微调快速入门》
　　网址：https://huggingface.co/docs/peft/quicktour

② 底稿：《LoRA微调大模型环境配置与实践指南》
　　网址：https://blog.csdn.net/l35633/article/details/161326236

③ 参考文章：《LoRA与QLoRA微调环境搭建详解》
　　网址：https://blog.csdn.net/u012856866/article/details/139502628

④ 官方工具说明：《bitsandbytes GitHub仓库与安装说明》
　　网址：https://github.com/TimDettmers/bitsandbytes

---

**适合人群**：已经完成 Hugging Face Transformers、Datasets、Accelerate 环境准备，希望进一步学习大语言模型微调的学习者。

本文将在前面的 Hugging Face 环境基础上，继续搭建 LoRA / QLoRA 参数高效微调环境。

你将学习：

* 为什么需要 PEFT；
* LoRA 与 QLoRA 的区别；
* 安装 PEFT 库；
* 安装 bitsandbytes；
* CUDA兼容性检查；
* GPU环境验证；
* QLoRA显存要求；
* 微调环境完整测试。

完成本章后，你将具备运行 LoRA / QLoRA 微调实验的基础环境。

---

## 一、为什么需要参数高效微调？（①②）
### 1.1 全参数微调的问题
传统大模型训练：

```text
输入数据

↓

整个模型参数更新

↓

保存完整模型
```

例如：

7B参数模型：

```text
70亿参数
```

训练时需要：

* 保存参数；
* 保存梯度；
* 保存优化器状态。

显存需求非常高。

普通消费级GPU难以完成。

---

（个性化提示：学习风格：hands-on）

## 块 2

### 1.2 参数高效微调（PEFT）
PEFT：

Parameter-Efficient Fine-Tuning

中文：

参数高效微调。

核心思想：

> 不修改整个大模型，只训练少量新增参数。

流程：

```text
原始大模型

        +
        
LoRA适配器参数

        ↓

只训练LoRA参数

        ↓

保存小规模Adapter
```

优势：

| 方式    | 训练参数      | 显存需求 |
| ----- | --------- | ---- |
| 全参数微调 | 全部参数      | 非常高  |
| LoRA  | 少量参数      | 较低   |
| QLoRA | 量化模型+LoRA | 更低   |

---

## 二、LoRA与QLoRA基础概念（①②）

（个性化提示：学习风格：hands-on）

## 块 3

### 2.1 LoRA是什么？（①）
LoRA：

Low-Rank Adaptation

低秩适配。

原模型：

```text
W
```

不更新。

新增：

```text
W + ΔW
```

其中：

```text
ΔW = B × A
```

> 说明：本课程统一采用 `ΔW = B × A` 的符号约定（与后续《理解 LoRA 的低秩分解原理》一章一致）。其中 `A`、`B` 是两个低秩小矩阵：若原权重 `W` 形状为 `d_out × d_in`、秩为 `r`，则 `A` 形状约为 `r × d_in`、`B` 形状约为 `d_out × r`，相乘后 `B × A` 得到与 `W` 同形状的 `ΔW`。不同资料可能写成 `A × B`，只是符号顺序不同、含义一致；关键是矩阵形状要能相乘并还原成 `W` 的形状。

只训练：

```text
A

B
```

两个小矩阵。

---

### 2.2 LoRA训练特点
优点：

* 参数量少；
* 训练速度快；
* 显存占用低；
* Adapter文件小。

训练完成：

保存：

```text
adapter_model

adapter_config
```

通常只有几十MB到几百MB。

---

## 三、QLoRA是什么？（①③）

（个性化提示：学习风格：hands-on）

## 块 4

```markdown
### 3.1 LoRA的问题
虽然LoRA减少训练参数。

但是：

基础模型仍然需要加载。

例如：

7B模型：

FP16：

约：

```text
14GB显存
```

---

### 3.2 QLoRA解决方案
QLoRA：

Quantized LoRA

量化版LoRA。

核心：

```text
FP16模型

↓

4bit量化

↓

减少显存

↓

加载LoRA训练
```

结构：

```text
4bit

## 块 5

### 电脑配置批注
本章节涉及 LoRA / QLoRA 本地训练环境。

#### LoRA最低配置：
CPU：

* 4核以上

内存：

* 16GB+

磁盘：

* 50GB以上

GPU：

推荐：

* NVIDIA GPU

显存：

* 最低：

  * 8GB

推荐：

* 12GB+

---

#### QLoRA要求：
GPU：

本教程实训范围：

* NVIDIA CUDA GPU（**项目支持范围**；如前所述，QLoRA 并非在技术上只能用 NVIDIA，但其它后端请以 bitsandbytes 官方说明为准，本文命令按 NVIDIA CUDA 给出）

显存：

QLoRA 的设计目标正是**在更小的显存上微调大模型**（通过 4-bit 量化基础模型，其显存占用通常低于同规模的 LoRA 全精度加载）。具体可运行的显存下限取决于模型规模、序列长度、batch 大小与量化配置，下面的数值仅为常见 7B 级别模型的

## 块 6

### 5.1 Python版本检查

为了顺利进行参数高效微调（LoRA与QLoRA）的环境准备，我们需要确认当前使用的Python版本是否符合要求。

#### 推荐操作

1. **运行Python版本检查命令**：
   ```bash

## 块 7

### 7.1 bitsandbytes作用
bitsandbytes：

用于：

* 8bit量化；
* 4bit量化；
* QLoRA训练。

QLoRA环境：

必须安装。

> 说明：这里的"必须安装"是指**本教程采用的是基于 `bitsandbytes` 的 QLoRA 4-bit 量化方案**，因此本方案下 `bitsandbytes` 是前置依赖。QLoRA 的核心思想（4-bit 量化基础模型 + LoRA）本身并不与某个特定量化库强绑定，业界也存在其它量化实现路径；只是本教程统一以 `bitsandbytes` + NVIDIA CUDA 演示。若你采用其它量化后端/库，请以对应文档为准。

---

### 7.2 安装bitsandbytes
基础安装：

```bash
pip install bitsandbytes
```

---

## 八、bitsandbytes CUDA兼容检查（重点）

（个性化提示：学习风格：hands-on）

## 块 8

### 8.1 为什么需要检查CUDA？
bitsandbytes依赖：

```text
CUDA

↓

GPU计算
```

如果版本不匹配：

可能出现：

* CUDA不可用；
* 量化失败；
* 加载模型报错。

---

### 8.2 查看CUDA版本
运行：

```bash
nvidia-smi
```

查看：

```text
CUDA Version
```

---

### 8.3 检查bitsandbytes
运行：

```bash
python -m bitsandbytes
```

正常：

显示：

* CUDA版本；
* GPU信息。

---

## 九、Windows用户特别注意事项（新增）
Windows环境运行QLoRA需要额外注意。

（个性化提示：学习风格：hands-on）

## 块 9

### 9.1 推荐使用WSL2
推荐：

```text
Windows

↓

WSL2 Ubuntu

↓

CUDA环境

↓

bitsandbytes
```

原因：

Linux环境兼容性更好。

---

### 9.2 Windows原生可能问题
可能遇到：

#### 问题1：
bitsandbytes无法加载CUDA动态库。

常见原因（可能不止一种）：

* 所装 `bitsandbytes` 版本与本机 CUDA / GPU 版 PyTorch 不匹配；
* 缺少或找不到对应的 CUDA 运行库；
* Windows 原生对某些 `bitsandbytes` 版本的支持不如 Linux 完善（不同版本差异较大）。

因此这不是单一的"Windows 不支持"问题，而是受 **PyTorch、bitsandbytes 版本与本机 CUDA/硬件**共同影响。推荐优先用 WSL2 获得更接近 Linux 的兼容性；若坚持 Windows 原生，请对照官方说明确认版本组合。

---

（个性化提示：学习风格：hands-on）

## 块 10

## 参数高效微调（LoRA与QLoRA）环境准备

### 1. 环境准备
在开始参数高效微调（LoRA与QLoRA）之前，请确保您的环境已正确配置。以下是操作步骤：

#### 1.1 安装必要的软件包
在终端中运行以下命令，安装所需的软件包：
```bash
pip install transformers datasets accelerate peft bitsandbytes
```

#### 1.2 检查 CUDA 环境
在终端中运行以下命令，检查 CUDA 是否

## 块 11

## 十四、常见问题排查
| 问题               | 原因             | 解决               |
| ---------------- | -------------- | ---------------- |
| bitsandbytes加载失败 | CUDA不匹配        | 检查CUDA版本         |
| CUDA unavailable | PyTorch安装CPU版本 | 重新安装CUDA版PyTorch |
| 显存不足             | 模型过大           | 降低batch或使用更小模型   |
| Windows报错        | 兼容性问题          | 使用WSL2           |
| QLoRA速度慢         | GPU性能不足        | 升级GPU            |

---

## 十五、完整LoRA/QLoRA学习路线
当前环境：

```text
Transformers

↓

Datasets

↓

Accelerate

↓

PEFT

↓

bitsandbytes

↓

LoRA

↓

QLoRA
```

后续学习：

```text
加载基础模型

↓

准备训练数据

↓

配置LoraConfig

↓

Trainer + PEFT

↓

训练Adapter

↓

合并模型

↓

部署推理
```

---

## 十六、总结
完成本章后，你已经完成：

✅ PEFT环境安装
✅ LoRA微调依赖准备
✅ bitsandbytes安装
✅ CUDA兼容检查
✅ QLoRA GPU环境确认
✅ Windows兼容方案了解
✅ Google/Hugging Face训练环境衔接完成

下一阶段：

**LoRA微调实战：使用PEFT + Trainer训练大语言模型**

将正式进入模型微调阶段。

（个性化提示：学习风格：hands-on）

## 块 12

```markdown
# LoRA微调原理详解：高效微调大模型的实践指南

## 一、为什么需要LoRA？（①

## 块 13

### 2.1 不直接修改原模型参数
传统微调：

原始权重：

```text
W
```

训练：

```text
W → W'
```

也就是：

直接修改：

```text
W
```

---

LoRA：

保持：

```text
W
```

冻结。

增加一个新的参数：

```text
ΔW
```

最终：

模型使用：

```text
W' = W + ΔW
```

其中：

* W：

  * 原模型参数；
* ΔW：

  * LoRA新增参数。

---

### 2.2 为什么只训练ΔW？
因为大模型已经通过海量数据学习了：

* 语言规律；
* 世界知识；
* 推理能力。

微调通常只是：

> 对已有能力进行方向调整。

因此在很多任务上，**不必**重新学习全部参数，只学习任务相关的变化量往往就能取得不错的效果。

> 说明：这是一种**教学直觉/假设**，帮助理解 LoRA 为何可行，并**不是普遍成立的严格结论**。是否能用 LoRA 替代全参数微调，取决于任务与预训练能力的差距、数据规模、目标模块与秩的设置等；某些差距较大的任务或数据充足时，全参数微调仍可能明显更好。

只需要学习：

```text
任务相关变化量
```

即可。

---

## 三、什么是低秩分解？（底稿③）

（个性化提示：学习风格：hands-on）

## 块 14

### 3.1 矩阵基础概念
Transformer中大量参数都是矩阵。

例如：

注意力层（下面是**数学表达/示意**，不是可运行代码）：

```text
Q = X · Wq
```

> 形状说明（示意）：设输入 `X` 形状为 `(序列长度, d_model)`、权重 `Wq` 形状为 `(d_model, d_model)`（此处按同维投影举例），则 `Q = X · Wq` 得到形状 `(序列长度, d_model)`。实际实现中还涉及 batch 维、多头拆分等，这里只为说明"Q 由输入乘以权重矩阵得到"。

其中：

```text
Wq
```

是一个权重矩阵。

假设：

```text
Wq:

4096 × 4096
```

参数数量：

```text
4096 × 4096

≈1677万
```

---

### 3.2 直接训练矩阵的问题
如果修改：

```text
ΔW
```

仍然需要：

```text
4096 × 4096
```

个参数。

LoRA提出：

不要直接学习完整矩阵。

而是：

拆成两个小矩阵。

---

## 四、LoRA低秩分解数学原理（重点）（①②③）
LoRA核心公式：

其中：

原始：

```text
W
```

保持冻结。

新增：

```text
ΔW
```

由：

```text
B × A
```

组成。

---

（个性化提示：学习风格：hands-on）

## 块 15

### 4.1 矩阵尺寸变化
假设：

原矩阵：

```text
W:

4096 × 4096
```

如果直接训练：

参数：

```text
4096×4096
=
16777216
```

---

LoRA：

设置：

```text
rank r=8
```

拆成：

A：

```text
8 × 4096
```

B：

```text
4096 × 8
```

参数：

A：

```text
32768
```

B：

```text
32768
```

总：

```text
65536
```

---

对比：

| 方式        |  参数数量 |
| --------- | ----: |
| 完整矩阵      | 1677万 |
| LoRA(r=8) |  6.5万 |

参数量大幅下降。

---

## 五、为什么低秩分解有效？（参考文章④⑤）

（个性化提示：学习风格：hands-on）

## 块 16

### 5.1 神经网络参数变化具有低维特征
LoRA论文提出：

模型适应新任务时：

参数变化：

```text
ΔW
```

并不需要覆盖所有方向。

实际上：

有效变化集中在：

```text
低维子空间
```

---

例如：

一个4096维空间。

真正影响任务的"有效方向"可能远少于满维度，集中在一个**低维子空间**里：

```text
远小于原维度的少数方向
```

> 说明：LoRA 的依据是"权重更新往往具有**较低的内在维度**（low intrinsic dimension）"这一**假设**。具体"有效维度"是多少并没有一个通用固定值，会随模型、任务、数据而变化；这里只做**定性**理解，不宜记成"几十维"之类的确定数字。

因此：

低秩矩阵：

```text
BA
```

可以近似：

```text
ΔW
```

---

（个性化提示：学习风格：hands-on）

## 块 17

# LoRA微调与部署指导

## 5.2 类比理解

假设：

一张高清图片：

```text
4000×4000像素
```

完整保存：

1600万个数据。

但是：

如果图片主要变化：

只是：

* 亮度；
* 对比度；
* 少量结构变化。

可以用少量参数描述。

LoRA类似：

> 用少量参数描述模型变化方向。

---

## 6. LoRA在Transformer中的位置

Transformer主要结构：

```text
Input

↓

Embedding

↓

Attention

↓

FFN

↓

Output
```

其中Attention：

包含：

```text
Q

K

V

O
```

四个投影矩阵。

---

LoRA常见加入位置：

```text
q_proj, v_proj
```

例如：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

表示：

只训练：

```text
Wq变化, Wv变化
```

> 说明：选择q_proj和v_proj是常见但非唯一选择。现代实践中也常将LoRA扩展至注意力的其他投影（k_proj、o_proj）或FFN的线性层。具体选择需结合模型结构与验证集表现。

---

## 7. LoRA训练过程

完整流程：

```text
加载预训练模型

↓

冻结所有参数

↓

插入

## 块 18

```markdown
# LoRA微调中的rank（r）

## 块 19

# LoRA微调与权重合并：基于 hands-on 实战的深入理解

## 8.2.1 LoRA_alpha 参数解析

### 参数定义与作用

在 LoRA 参数微调中，`lora_alpha` 是一个关键超参数，用于控制 LoRA 参数的更新强度。其作用机制如下：

#### 示例配置：
```python
lora_alpha=16
```

#### 参数范围与

## 块 20

```markdown
## LoRA核心参数详解：Rank与Alpha参数作用原理（最终版）

### 学习目标
* 理解Rank参数的作用及其对模型性能的影响
* 掌握Rank参数

## 块 21

### 2.2 Rank控制什么？
简单理解：

> Rank决定LoRA能够学习多少种变化方向。

低rank：

例如：

```python
r=4
```

表示：

模型只能学习较少的变化模式。

优点：

* 参数少；
* 显存低；
* 训练快。

缺点：

* 表达能力有限。

---

高rank：

例如：

```python
r=64
```

表示：

LoRA拥有更多自由度。

优点：

* 能学习更复杂任务；
* 适应能力更强。

缺点：

* 参数更多；
* 显存增加；
* 训练成本提高。

---

## 三、Rank对参数量的影响（重点）（②）
假设：

模型某层：

```text
4096 × 4096
```

---

（个性化提示：学习风格：hands-on）

## 块 22

### 3.1 不使用LoRA
完整参数：

```text
4096 × 4096

=16777216
```

约：

1677万参数。

---

### 3.2 使用LoRA
公式：

```text
参数量=r×(输入维度+输出维度)
```

---

#### r=8
参数：

```text
8×(4096+4096)
```

=

```text
65536
```

---

#### r=64
参数：

```text
64×8192
```

=

```text
524288
```

---

对比：

| 方式        |   参数量 |
| --------- | ----: |
| 完整训练      | 1677万 |
| LoRA r=8  |  6.5万 |
| LoRA r=64 |   52万 |

可以看到：

即使提高rank：

仍远低于全参数训练。

---

## 四、Rank如何影响模型效果？（②③）

（个性化提示：学习风格：hands-on）

## 块 23

```markdown
# LoRA微调与权重合并：Rank选择与参数效率优化

## 4.1 Rank太小的问题
在LoRA微调过程中，Rank参数的设置直接影响模型的

## 块 24

### 4.3 常见rank选择
经验（下面仅为**经验起始范围**，不是自动调参标准）：

| 任务类型   | rank（起点参考） |
| ------ | ------ |
| 简单分类   | 约 4~8    |
| 文本风格调整 | 约 8~16   |
| 指令微调   | 约 16~64  |
| 复杂领域任务 | 约 64以上   |

实际项目需要根据：

* 数据量；
* 模型大小；
* 序列长度；
* batch 大小（含梯度累积后的有效 batch）；
* 目标模块（`target_modules`）的选择与数量；
* GPU显存；
* 验证集/基准任务上的表现；

调整。不要把上表当作可以直接照搬或自动套用的固定配置——同一个 `r` 在不同模型规模、不同序列长度/batch、不同目标模块下的显存与效果都可能差别很大。

---

## 五、Alpha（lora_alpha）是什么？（①②）

### 5.1 Alpha作用
LoRA除了：

```text
ΔW = BA
```

还会进行缩放：

公式：

```text
W' = W + (α/r)BA
```

其中：

* α：

  * lora_alpha；
* r：

  * rank。

---

因此：

真正加入模型的变化量：

不是：

```text
BA
```

而是：

```text
(α/r)BA
```

---

## 六、为什么需要Alpha缩放？（②③）

（个性化提示：学习风格：hands-on）

## 块 25

```markdown
# LoRA微调与权重合并：控制LoRA影响强度

## 6.1 控制LoRA影响强度

### rank与alpha参数的作用
LoRA（Low-Rank Adaptation）是一种高效的微调方法，通过分解权重矩阵来实现模型的在线微调。rank和alpha是LoRA核心参数，它们共同决定了LoRA更新

## 块 26

### 配置1
```python
r=8

alpha=16
```

比例：

```text
2
```

---

### 配置2
```python
r=16

alpha=32
```

比例：

```text
2
```

虽然rank不同：

但是缩放比例相同。

---

因此：

调整rank时：

常见做法是**同步调整**alpha，以保持 `α/r` 缩放比例大致稳定（这只是**经验做法**，不是必须遵守的规则）。下文各"方案"里出现的 `alpha = 2 × r`（如 r=8/alpha=16、r=16/alpha=32）也只是常见的**经验起点**，并非通用最优规则；实际仍应结合验证集表现调整，也可保持 alpha 固定而单独调 r。

---

## 八、Rank和Alpha如何搭配？（②③）

（个性化提示：学习风格：hands-on）

## 块 27

### 方案1：小模型实验
配置：

```python
r=8

alpha=16
```

特点：

* 参数少；
* 训练快；
* 适合测试。

---

### 方案2：普通指令微调
配置：

```python
r=16

alpha=32
```

特点：

* 平衡效果和成本。

---

### 方案3：复杂任务
配置：

```python
r=64

alpha=128
```

特点：

* 更强表达能力；
* 参数更多。

---

## 九、PEFT中的参数配置示例（①）
实际代码：

```python
from peft import LoraConfig


config = LoraConfig(

    r=8,

    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05

)
```

其中：

### r
控制：

```text
LoRA容量
```

---

### lora_alpha
控制：

```text
LoRA更新幅度
```

---

### target_modules
控制：

```text
LoRA插入位置
```

---

## 十、Rank与Alpha选择误区（②③）

（个性化提示：学习风格：hands-on）

## 块 28

### 误区1：
认为：

> rank越大一定越好。

错误。

原因：

任务简单时：

大rank可能：

* 浪费显存；
* 增加训练成本。

---

### 误区2：
只调整rank，不调整alpha。

例如：

原：

```python
r=8
alpha=16
```

改：

```python
r=64
alpha=16
```

比例：

```text
16/64=0.25
```

LoRA影响降低。

---

### 误区3：
认为alpha越大越好。

错误。

alpha过大（即 `α/r` 缩放过大）：

可能：

* 更新过强；
* 训练不稳定。

> 说明：`α/r` 过大只是训练不稳定的**可能原因之一**，不是唯一或主要原因。训练震荡/发散还常常来自学习率过高、数据质量差、有效 batch size 过小、精度（fp16/bf16）与数值稳定性、缺少梯度裁剪等。排查不稳定时应综合这些因素，不要只盯着 alpha。

---

## 十一、电脑配置批注
本章节主要讲解：

* 数学原理；
* 参数含义；
* 配置策略。

不需要GPU。

最低：

* CPU：

  * 任意现代CPU；
* 内存：

  * 4GB+

---

如果进行实际LoRA训练：

推荐：

GPU：

* NVIDIA CUDA GPU

显存（下面为**带条件经验参考**，非固定门槛；实际取决于模型规模、序列长度、batch、梯度累积、精度与量化配置，请以实际运行为准）：

| 训练方式  | 经验参考显存  |
| ----- | ----- |
| LoRA  | 约 8GB+（随模型规模/序列长度/batch 变化）  |
| QLoRA | 因 4-bit 量化通常低于同规模 LoRA，7B 级别常见参考约 8GB 起，以实际运行为准 |

---

## 十二、总结
完成本章后，你已经掌握：

✅ Rank是什么
✅ Rank如何影响参数量
✅ Rank如何影响表达能力
✅ Alpha作用
✅ α/r缩放公式
✅ Rank和Alpha关系
✅ PEFT参数配置方式
✅ 常见LoRA参数选择策略

核心公式：

```text
W' = W + (α/r)BA
```

理解：

```text
r

决定LoRA容量

α

通过 α/r 缩放影响 LoRA 更新强度
```

即可理解LoRA的大部分参数配置逻辑。

下一阶段：

**LoRA实战：使用PEFT + Trainer完成模型微调**

将进入真正的大模型训练流程。

（个性化提示：学习风格：hands-on）

## 块 29

```markdown
## PEFT库LoraConfig配置详解：掌握target_modules、r、lora_alpha、lora_dropout参数（最终版）

### 适合人群
- 已经理解LoRA低秩分解原理，并完成PEFT环境安装
- 希望进入实际LoRA微调配置阶段的学习者

### 本次教学动作
- **解释LoraConfig的作用**
- **掌握如何创建LoRA配置**
- **理解target_modules的作用**
- **理解r参数的作用**
- **理解lora_alpha参数的作用**
- **理解lora_dropout参数的作用**
- **掌握如何组合完整LoRA配置**

### 学习目标
- 独立阅读和修改各种LoRA微调代码中的参数配置

### 学习证据引用
- 无

---

###

## 块 30

### 1.2 一个简单示例
```python
from peft import LoraConfig


config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05
)
```

这几个参数就是LoRA最核心配置：

| 参数             | 作用         |
| -------------- | ---------- |
| target_modules | 决定插入哪些层    |
| r              | 决定LoRA矩阵大小 |
| lora_alpha     | 控制缩放强度     |
| lora_dropout   | 控制随机失活     |

---

## 二、LoRA配置整体流程（①③）
完整代码关系（下面是**流程关系示意**，不是可运行代码）：

```text
model
    |
    |
LoraConfig
    |
    |
get_peft_model()
    |
    |
LoRA模型
```

例如：

```python
from peft import (
    LoraConfig,
    get_peft_model
)


config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05
)


model = get_peft_model(
    model,
    config
)
```

执行后：

原模型参数：

```text
冻结
```

新增LoRA参数：

```text
训练
```

---

## 三、target_modules参数详解（重点）（①②③）

（个性化提示：学习风格：hands-on）

## 块 31

```markdown
# PEFT库LoraConfig配置详解：掌握target_modules、r、lora_alpha、lora_dropout参数（最终版）

## 3.1 target_modules是什么？

**定义：**  
`target_modules` 是指定需要注入 LoRA（Low-Rank Adaptation）的模型模块。LoRA 通过在这些模块中增加低秩调整矩阵，实现对模型参数的高效微调。

在 Transformer 模块中，常见结构包括：

```text
Attention

├── q_proj

├── k_proj

├── v_proj

└── o_proj
```

**常见配置：**  
默认情况下，LoRA 会自动匹配并注入所有线性层（`lora_config.target_modules="all-linear"`）。但为了更精准地控制 LoRA 的作用范围，建议手动指定 `target_modules`。

**示例：**  
```python
lora_config = LoraConfig(

## 块 32

### 方案1：只调整Q和V
```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

特点：

* 参数少；
* 训练快；
* 是一种常见的轻量起点（不限于某个特定规模，具体仍需看模型结构与效果）。

---

### 方案2：调整全部Attention
```python
target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]
```

特点：

* 参数更多；
* 表达能力更强。

---

### 方案3：加入FFN层
部分任务：

还会加入：

```python
gate_proj

up_proj

down_proj
```

例如：

```python
target_modules=[
    "q_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

特点：

* 可学习能力更强；
* 显存消耗增加。

---

## 五、不同模型target_modules区别（参考文章⑤）
不同模型命名不同。

例如：

（个性化提示：学习风格：hands-on）

## 块 33

```markdown
### LLaMA系列
常见：

```python
[
"q_proj",
"v_proj"
]
```

---

### GPT-2
可能：

```python
[
"c_attn"
]
```

> **GPT-2 的 `Conv1D` 权重布局与 `fan_in_fan_out=True`（重要）**：
> GPT-2 的注意力用的是 Transformers 里的 `Conv1D`（不是普通 `nn.Linear`），它把 Q/K/V 三个投影合并在一个 `c_attn` 模块里。`Conv1D` 的权重形状是 `(in_features, out_features)`——**与 `nn.Linear` 的 `(out_features, in_features)` 正好转置**。因此在这类层上加 LoRA 时，需要设置 `fan_in_fan_out=True`，PEFT 才能按转置布局正确注入低秩矩阵。
>
> 可运行示例（GPT-2）：
>
> ```python
> from peft import LoraConfig, get_peft_model
>

## 块 34

### BERT
可能：

```python
[
"query",
"value"
]
```

---

因此：

不能直接复制所有模型配置。

需要查看：

```python
model.named_modules()
```

例如：

```python
for name,_ in model.named_modules():

    print(name)
```

找到：

```text
q_proj

v_proj
```

再配置。

> 补充：
> - **匹配机制**：`target_modules` 传入的字符串是按模块名做匹配的。PEFT 通常支持**后缀匹配**——例如写 `"q_proj"` 会匹配所有以 `q_proj` 结尾的层（各层各头都会命中）；也可以传更完整的名字做精确定位，或传正则。
> - **GPT-2（Conv1D）**：GPT-2 的注意力用的是 `Conv1D` 而非普通 `nn.Linear`，模块名是 `c_attn`（把 q/k/v 合在一起）。它同样能被 LoRA 支持，但命名与结构和 LLaMA 系不同，不能照搬 `q_proj`/`v_proj`。
> - **版本差异**：不同 Transformers/PEFT 版本、不同模型实现，模块命名可能变化；`"all-linear"` 等便捷写法也依赖较新版本。请以你实际加载模型 `named_modules()` 的输出与所用版本文档为准。

---

## 六、r参数详解（rank）（③④）

（个性化提示：学习风格：hands-on）

## 块 35

### 6.1 r是什么？
r：

即rank。

控制LoRA低秩矩阵维度。

公式：

```text
ΔW = BA
```

其中：

r决定：

A、B中间维度。

---

例如：

原矩阵：

```text
4096×4096
```

设置：

```python
r=8
```

则：

A：

```text
8×4096
```

B：

```text
4096×8
```

---

### 6.2 r影响什么？
r越大：

优势：

* 学习能力更强；
* 可以表达更复杂任务。

缺点：

* 参数增加；
* 显存增加。

---

常见（下面均为**经验起始范围**，不是固定配方，应以验证集/基准任务结果选择）：

| 场景     | r（起点参考）     |
| ------ | ----- |
| 简单任务   | 约 4-8   |
| 通用指令微调 | 约 8-32  |
| 复杂领域任务 | 约 32-64 |

---

## 七、lora_alpha参数详解（重点）（③④）

（个性化提示：学习风格：hands-on）

## 块 36

### 7.1 alpha作用
LoRA最终更新：

不是：

```text
BA
```

而是：

```text
(alpha/r)BA
```

因此：

alpha控制：

> LoRA更新幅度。

---

例如：

配置：

```python
r=8

lora_alpha=16
```

缩放：

```text
16/8=2
```

表示：

LoRA影响增强。

---

### 7.2 alpha常见设置
经验：

一种常见的**起点**是：

```text
alpha = 2 × r
```

> 说明：`alpha = 2 × r` 只是常见经验起点，不是通用规则。也有做法保持 alpha 固定而单独调 r，或用其它比例；最终应以验证集表现为准。

例如：

```python
r=8

alpha=16
```

或者：

```python
r=16

alpha=32
```

---

## 八、lora_dropout参数详解（③⑤）

（个性化提示：学习风格：hands-on）

## 块 37

### 8.1 dropout作用
训练过程中：

随机关闭部分LoRA神经连接。

目的：

减少：

```text
过拟合
```

---

配置：

```python
lora_dropout=0.05
```

表示：

5%的LoRA路径随机失活。

---

### 8.2 如何选择dropout？
常见（**经验起点**，非固定规则，应结合验证集是否过拟合来调）：

| 数据规模 | dropout（起点参考）  |
| ---- | -------- |
| 数据少  | 约 0.05~0.1 |
| 数据多  | 约 0~0.05   |

---

如果：

训练数据非常丰富：

可以：

```python
lora_dropout=0
```

---

## 九、完整LoraConfig案例（重点）（①③）

（个性化提示：学习风格：hands-on）

## 块 38

### 9.1 基础配置
适合：

7B模型测试。

```python
from peft import LoraConfig


lora_config = LoraConfig(

    r=8,

    lora

## 块 39

```markdown
# LoRA微调与权重合并完整配置指南

## 1. 前言
在大语言模型预训练模型的微调过程中，参数高效微调（LoRA）是一种高效的方法，能够显著减少模型的参数量，同时保持模型性能。LoRA通过微调部分关键参数即可实现良好的效果。在实际应用中，LoRA配置的正确性至关重要。本节将详细讲解如何配置LoRA参数，并完成完整的模型保存与加载流程。

---

## 2. LoRA参数配置

### 2.1 参数设置
在使用PEFT库时，可以通过`LoraConfig`对象来配置LoRA参数。以下是关键参数的解释：

- **`r`**：LoRA的秩，决定了LoRA矩阵的大小。通常取值较小（如8、16等），以控制LoRA参数的稀疏性。
- **`lora_alpha`**：LoRA矩阵的缩放因子，影响LoRA参数的初始值大小。
- **`lora_dropout`**：LoRA层的Dropout概率，用于防止过拟合。
- **`target_modules`**：需要微调的模型模块（如注意力层中的查询和值层）。
- **`modules_to_save`**：指定需要保存的关键模块，如分类头。

### 2.2 实战示例
以下代码展示了如何配置LoRA参数并加载模型：

```python
from peft import LoraConfig, get_peft_model

base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 配

## 块 40

```markdown
## 十、普通文本生成

### 推荐参数配置
在进行普通文本生成任务时，推荐使用以下参数配置：

```python
r=8
alpha=16
dropout=0.05
```

这些参数设置平衡了模型的生成能力与计算效率，适用于大多数文本生成场景。

---

### 参数意义
- **r**：表示LoRA模块的秩，控制分解的大小。较小的r值可以减少显存占用，但可能会影响生成质量。
- **alpha**：缩放LoRA权重的大小，直接影响模型的参数量。较大的alpha值会导致模型参数数量增加。
- **dropout**：防止过拟合的一种常用方法，有助于提高模型的泛化能力。

---

### 代码示例
以下是一个使用推荐参数配置进行普通文本生成的示例代码：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("

## 块 41

核心配置：

```python
LoraConfig(

    r=8,

    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05

)
```

理解：

```text
target_modules
决定改哪里

r
决定LoRA容量

alpha
通过 α/r 缩放影响更新强度

dropout
用于缓解过拟合（是影响泛化的因素之一，而非唯一决定项）
```

下一阶段：

**使用PEFT + Trainer完成LoRA微调实战**

将进入完整模型训练流程。

（个性化提示：学习风格：hands-on）

## 块 42

# QLoRA原理详解：4-bit量化（NF4、双重量化）与分页优化器机制（最终版）

## 一、为什么需要QLoRA？

### 1.1 LoRA仍然需要加载完整模型

LoRA（Low-Rank Decomposition）通过分解权重矩阵，成功减少了模型的

## 块 43

### 1.2 QLoRA核心思想
QLoRA：

Quantized LoRA

即：

> 对基础模型进行低比特量化，再结合LoRA训练。

流程：

```text
FP16基础模型

↓

4-bit量化

↓

加载量化模型

↓

插入LoRA

↓

训练LoRA参数
```

区别：

LoRA：

```text
FP16模型 + LoRA
```

QLoRA：

```text
4-bit模型 + LoRA
```

---

## 二、QLoRA整体结构（①④）
QLoRA训练时：

基础模型：

```text
4-bit权重

冻结
```

LoRA：

```text
FP16/BF16参数

训练
```

结构：

```text
输入

↓

4-bit量化模型

↓

LoRA Adapter

↓

输出
```

重点：

量化的是：

```text
基础模型权重
```

不是：

```text
LoRA参数
```

---

## 三、为什么需要4-bit量化？（①③④）

（个性化提示：学习风格：hands-on）

## 块 44

### 3.1 模型存储方式
模型参数通常：

| 类型        |   每个参数占用 |
| --------- | -------: |
| FP32      |  4 Bytes |
| FP16/BF16 |  2 Bytes |
| INT8      |   1 Byte |
| INT4      | 0.5 Byte |

---

例如：

7B模型：

FP16：

```text
7B × 2

≈14GB
```

4-bit：

```text
7B × 0.5

≈3.5GB
```

显存减少约75%。

> 说明：上面的 `3.5GB`、`减少约75%` 只是**理想权重位数的估算**（7B × 0.5 Byte），仅统计了量化后的权重本身。实际显存占用通常明显更高，因为还需要包含：量化常数/缩放因子、分组元数据（NF4 按组量化，每组都要存额外参数）、未量化的模块（如部分 norm、embedding、lm_head、LoRA 参数），以及推理/训练过程中的计算缓存、激活值与运行时开销。因此请把它当作**理论下界**，真实占用以实际运行为准。

---

### 3.2 为什么不用普通INT4？
传统整数：

INT4：

范围：

```text
-8 ~ 7
```

问题：

神经网络权重通常：

* 接近正态分布；
* 小数较多；
* 分布中心接近0。

直接整数映射：

容易损失精度。

因此QLoRA提出：

NF4。

---

## 四、NF4（NormalFloat4）量化原理（重点）（①④⑤）

（个性化提示：学习风格：hands-on）

## 块 45

### 4.1 NF4是什么？
NF4：

NormalFloat4

中文：

正态分布4位浮点。

QLoRA论文提出：

> 针对神经网络权重正态分布特点设计的4-bit数据类型。

> 说明：这里的"正态分布4位浮点"是一种**通俗叫法**，不要理解成 IEEE 754 那种"通用的 4 位浮点格式"。NF4（NormalFloat4）本质上是**一套针对"近似正态分布"权重设计的 4-bit 量化数据类型/码本（codebook）**：它预先确定了 16 个（2⁴）量化取值点，这些取值点按标准正态分布的分位数排布，使得 0 附近（权重密集区）划分更细。量化时把每组权重归一化后映射到最接近的码本取值，并保存对应的 4-bit 索引；反量化时再按码本与缩放因子还原。它不是可以任意表示实数的"浮点格式"，而是**为特定分布量身定制的离散码本**。

---

（个性化提示：学习风格：hands-on）

## 块 46

### 4.2 为什么适合模型权重？
大模型训练后的权重通常：

近似：

```text
正态分布
```

例如：

```text
大量参数集中在0附近
```

普通INT4：

平均划分：

```text
[-8,-7,...,7]
```

无法很好表示：

中心密集区域。

---

NF4：

根据正态分布：

设计量化范围。

特点：

* 0附近精度更高；
* 更符合权重分布；
* 精度损失更低。

---

### 4.3 NF4量化过程
原始：

```text
FP16权重
```

例如：

```text
0.023

-0.145

0.876
```

经过：

```text
归一化

↓

映射到NF4

↓

保存4-bit值
```

推理时：

再反量化：

```text
NF4

↓

近似FP16
```

---

## 五、双重量化（Double Quantization）（重点）（①④⑥）

（个性化提示：学习风格：hands-on）

## 块 47

### 5.1 为什么需要双重量化？
4-bit量化需要保存：

```text
量化参数
```

例如：

每组权重：

需要：

* scale；
* zero point。

这些额外参数也占显存。

---

### 5.2 双重量化思想
普通量化：

```text
FP16权重

↓

4-bit权重

+

scale参数
```

双重量化：

进一步：

```text
scale参数

↓

再次量化
```

即：

第一次：

量化模型权重。

第二次：

量化量化参数。

---

结构：

```text
FP16权重

↓

NF4量化

↓

保存4-bit权重


scale参数

↓

再次量化

↓

减少scale存储
```

---

### 5.3 双重量化优势
减少：

* 额外量化参数；
* 显存占用。

尤其：

大模型：

参数数量巨大。

少量优化：

也会节省大量显存。

---

## 六、分页优化器（Paged Optimizers）原理（重点）（①④⑤）

（个性化提示：学习风格：hands-on）

## 块 48

### 6.1 为什么需要分页优化器？
训练过程中：

显存峰值来源：

除了模型：

还有：

* 梯度；
* 优化器状态。

例如：

Adam：

需要保存：

```text
一阶动量

二阶动量
```

---

QLoRA虽然降低模型显存：

但是训练过程中：

仍可能：

```text
显存突然不足
```

---

### 6.2 分页优化器思想
Paged Optimizer借鉴：

操作系统虚拟内存。

核心：

> 借助 NVIDIA 的统一内存（Unified Memory），在显存出现峰值、即将 OOM 时，把优化器状态自动"分页"换出到 CPU 内存，需要时再换回 GPU。

流程：

```text
GPU显存

↓

出现峰值（接近/超出显存上限）

↓

优化器状态按页换出到CPU内存（Unified Memory）

↓

需要参与计算时再换回GPU
```

> 说明：不要把它简单理解成"平时就把一部分优化器状态常驻 CPU、需要时交换"。QLoRA 论文里分页优化器的重点是**应对显存峰值**：借助 Unified Memory 的自动分页机制，在显存不足的瞬间把优化器状态临时换出，避免因为偶发峰值直接 OOM 而中断训练；平稳阶段这些状态仍尽量留在 GPU。它解决的是"峰值 OOM"，而不是"常态化地把优化器放到 CPU 以省显存"。

---

（个性化提示：学习风格：hands-on）

## 块 49

```markdown
# QLoRA与分页优化器机制详解

## 6.3 分页优化器的类比理解

### 概念解释
分页优化器是一种通过物理化内存空间的方式，将模型参数划分为多个独立的内存块的技术。与普通内存管理方式不同，分页优化器允许模型在物理内存不足时，通过将不常用的参数移动到内存外的物理存储设备（如硬盘或显存分区）来缓解显存占用问题。

### 分页优化器的工作原理

## 块 50

### 7.1 4-bit量化
作用：

减少模型权重大小。

```text
FP16

↓

NF4 4-bit
```

---

### 7.2 双重量化
作用：

减少量化参数占用。

```text
scale

↓

再次量化
```

---

### 7.3 分页优化器
作用：

降低训练峰值显存。

```text
GPU

+

CPU内存交换
```

---

## 八、QLoRA与LoRA区别（④⑤）
| 项目   | LoRA      | QLoRA             |
| ---- | --------- | ----------------- |
| 基础模型 | FP16/BF16 | 4-bit（本教程用 NF4）    |
| 训练参数 | LoRA      | LoRA              |
| 量化   | 无         | 4-bit（NF4 为常见选择）   |
| 显存   | 较低        | 通常更低              |
| 依赖   | PEFT      | PEFT+bitsandbytes |
| 适合   | 中高显存GPU   | 显存更受限的场景（含消费级GPU） |

> 说明：上表是**面向本教程配置的对比**，不是绝对定义。(1) QLoRA 的量化类型并非只能是 NF4——bitsandbytes 也支持 `fp4` 等其它 4-bit 类型，NF4 只是 QLoRA 论文推荐、也是本教程采用的**常见选择**。(2) "适合消费级 GPU"是 QLoRA 的**典型收益场景**，但它本质是"用更低显存换取一定精度/速度代价"的方法，并不局限于消费级显卡，服务器级 GPU 上同样可用来在更大模型/更长序列上省显存。请把"NF4""消费级 GPU"理解为常见搭配，而非固定绑定。

---

## 九、bitsandbytes中的QLoRA配置（③）
实际使用：

```python
import torch
from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

参数：

（个性化提示：学习风格：hands-on）

## 块 51

# QLoRA微调+权重合并+vLLM部署教学

## 1. 4-bit量化配置

### 1.1 设置4-bit量化参数
在模型加载时，开启4-bit量化并指定参数类型和双重量化方式。

```python
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True
```

### 1.2 配置计算 dtype
为了优化计算效率，根据GPU类型选择合适的计算 dtype。

```python
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

验证计算 dtype 是否正确：

```python
print(f"计算 dtype: {compute_dtype}")
```

### 1.3 验证配置
检查量化配置是否正确：

```python
assert bnb_4bit_quant_type in ["nf4", "nf4v4"], "仅支持NF4量化类型"
```

## 2.

## 块 52

```markdown
## 9.1 QLoRA微调与权重合并实战指南

### 1. 量化配置与模型加载
为了实现4-bit量化，我们需要配置`BitsAndBytesConfig`，并确保模型在训练过程中能够正确加载。以下是实现步骤：

```python
# 配置4-bit量化参数
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)
```

### 2. 准备模型进行k-bit训练
在量化模型后，我们需要使用`prepare_model_for_kbit_training`来优化模型的训练过程：

```python
# 准备模型进行4-bit训练
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)
```

### 3. 注入LoRA结构
LoRA结构的注入是实现参数高效微调的关键步骤：

```python
# 定义LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# 注入LoRA结构
model = get_peft_model(model, lora_config)
print(f"可训练参数数量：{model.print_trainable_parameters()}")
```

### 4. 构建微调数据集
微调数据集需要满足模型的需求，以下是一个简单的示例：

```python
# 构建微调数据集
raw = Dataset.from_dict({"text": ["Hello QLoRA", "Fine-tune with 4-bit"]})
def tokenize(ex):
    enc = tokenizer(ex["text"], truncation=True, max_length=128)
    enc["labels"] = enc["input_ids"].copy()
    return enc
train_dataset = raw.map(tokenize, remove_columns=raw.column_names)
```

### 5. 设置微调参数
训练参数的设置直接影响微调效果，以下是推荐配置：

```python
# 设置微调参数
training_args = TrainingArguments(
    output_dir="./qlora_ckpt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit"
)
```

### 6. 执行微调训练
最后，可以执行微调训练并验证效果：

```python
# 初始化并执行微调训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer
)
trainer.train()
```

### 本块学习检查
为了验证学习效果，请完成以下任务：
1. 使用提供的配置参数，运行完整的微调流程。
2. 检查模型的参数数量，并确保与预期一致。
3. 验证微调后的模型在指定任务上的性能。
```

## 块 53

# QLoRA微调+权重合并+vLLM部署教学正文

## 1. 关系梳理

QLoRA微调与权重合并部署是实现低成本大模型微调的重要步骤。以下是各步骤之间的关系：

1. **`BitsAndBytesConfig`** 配置决定基础模型的4-bit加载方式（NF4 + 双重量化 + 计算 dtype）。这是实现低成本量化的基础。
2. **`prepare_model_for_kbit_training()`** 是QLoRA微调的必要一步。它对量化模型进行适配，包括开启输入梯度以支持梯度检查点，确保训练的稳定性。
3. **`get_peft_model(..., LoraConfig)`** 将可训练的LoRA注入到冻结的4-bit主干中。这是QLoRA微调的核心操作。
4. **

## 块 54

```markdown
# LoRA微调与模型导出详解：merge_and_unload使用方法

## 适合人群
已经完成LoRA/QLoRA微调训练的用户，希望将训练得到的Adapter转换为完整模型并用于部署推理。

## 本块学习目标
1. 理解LoRA训练后保存的文件结构与内容
2. 掌握为什么需要合并LoRA权重
3. 学习`merge_and_unload()`函数的工作原理
4. 掌

## 块 55

### 2.1 推理时的问题
未合并：

```text
输入

↓

Base Model

+

LoRA Adapter

↓

输出
```

需要：

* 加载基础模型；
* 加载Adapter。

---

对于部署：

可能希望：

```text
输入

↓

完整模型

↓

输出
```

因此需要：

合并权重。

---

### 2.2 LoRA合并数学原理（②）
LoRA训练时：

公式：

```text
W' = W + ΔW
```

其中：

```text
ΔW = BA
```

合并：

就是将：

```text
BA
```

加入：

```text
W
```

得到：

```text
W'
```

---

合并后：

原来的：

```text
W

+

Adapter
```

变成：

```text
W'
```

---

## 三、merge_and_unload是什么？（重点）（①②）

（个性化提示：学习风格：hands-on）

## 块 56

# LoRA微调与权重合并：merge_and_unload方法详解

## 一、基础知识回顾
- **LoRA**（Low-Rank Adaptation）：一种参数高效微调方法，通过在基础模型上增加少量可学习参数实现微调效果。
- **merge_and_unload**：PEFT库中用于合并LoRA权重并移除PEFT包装的方法。

## 二、merge_and_unload的作用
1. **合并LoRA权重**：将LoRA adapter（可学习参数）与基础模型权重结合，形成完整的模型。
2. **移除PEFT包装**：去除PEFT附加的包装层，使模型恢复为基础模型形式。

## 三、merge_and_unload的使用流程
1. **加载基础模型**：使用`from_pretrained`加载基础模型。
2. **加载LoRA adapter**：使用`PeftModel.from_pretrained`加载LoRA adapter。
3. **合并权重**：执行`model.merge_and_unload()`，完成权重合并。
4. **导出模型**：调用`save_pretrained`将合并后的模型保存为文件。

## 四、实践步骤
1. **加载基础模型**：
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("model_path")
   ```

2. **加载LoRA adapter**：
   ```python
   from peft import PeftModel
   model

## 块 57

```markdown
# 合并后的模型加载与使用指南

## 1. 合并后如何加载推理？

在完成模型合并后，推理加载的流程非常简单，无需额外的PEFT相关操作。以下是具体的加载步骤：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./merged_model"
)
```

### 2. 用户输入

用户向模型发送输入文本，例如：

```
输入文本：

## 块 58

### 10.1 推荐合并场景
#### 场景1：上线部署（单一 Adapter、追求简单）
例如：

生产环境：

```text
服务器

↓

直接加载模型

↓

提供API
```

当只需要**一个固定 Adapter**、且希望部署链路尽量简单（不额外依赖 PEFT、不做运行时 Adapter 加载）时，合并通常更方便。

> 注意：如果推理框架本身支持直接加载 LoRA（例如 vLLM 的多 LoRA 服务，见 §十四），或你需要在同一基础模型上**热切换多个 Adapter**，那么不合并、以 Adapter 形式部署反而更灵活；此时是否合并要看框架能力与业务需求。

---

#### 场景2：分享模型
如果希望对方**开箱即用、无需理解 Adapter 机制**，可以发布合并后的完整模型：

```text
merged_model
```

用户无需额外下载 Adapter。

> 权衡：合并后的完整模型体积**明显更大**（等于一份完整基础模型），分发/存储成本更高；如果对方已经有同一基础模型，单独分享体积很小的 Adapter 往往更省带宽与存储。两种方式各有取舍，可按受众与分发成本选择。

---

（个性化提示：学习风格：hands-on）

## 块 59

### 10.2 不建议合并场景
#### 场景1：多个任务Adapter
例如：

一个基础模型：

```text
Base Model
```

多个：

```text
Adapter A

Adapter B

Adapter C
```

保留Adapter更灵活。

---

#### 场景2：持续训练
如果继续调整LoRA：

保留Adapter。

---

## 十一、QLoRA模型合并注意事项（重点）（③）
QLoRA：

基础模型：

```text
4-bit量化
```

LoRA：

```text
FP16/BF16
```

QLoRA 训练时基础模型是以 4-bit（NF4）**存储**的，而 LoRA 参数是较高精度（FP16/BF16）。因此把 LoRA 更新量直接加回 4-bit 权重，涉及**精度与权重类型的处理**，需要留意以下几点。

（个性化提示：学习风格：hands-on）

## 块 60

```markdown
## QLoRA微调与权重合并详解

### 11.1 直接合并可能遇到的情况

在进行直接合并时，可能会遇到以下问题：

* **精度损失**：直接在4-bit状态下合并权重，可能导致精度下降，因为4-bit量化后再进行合并，与全程使用高精度权重不完全匹配。
* **权重类型不匹配**：如果所使用的模型或量化版本不支持直接使用`merge_and_unload`方法，可能会导致报错。

> **解决方法**：在进行直接合并之前，建议先反量化到高精度状态，确保权重类型一致后再进行合并。

### 11.2 常见做法：先反量化到高精度再合并

为了避免直接合并时出现的问题，推荐以下操作流程：

1. **加载高精度基础模型**：使用FP16或BF16加载基础模型，确保

## 块 61

> 提示（版本/内存）：
> - **API/版本**：不同 PEFT 版本对"量化模型能否直接合并"的支持不同，部分版本要求或推荐先反量化；请查阅所用版本的 `merge_and_unload` 文档。上例的关键是**加载基础模型时不再传 `quantization_config`/`load_in_4bit`**，从而得到高精度权重再合并。
> - **内存**：以高精度重新加载完整基础模型会占用**较多内存/显存**（约等于一份 FP16/BF16 完整模型），显存不足时可把 `device_map` 改为 `{"": "cpu"}` 在 CPU 上加载并合并（较慢），或使用 offload。
> - **一致性**：QLoRA 合并得到的高精度模型，与"训练时 4-bit + Adapter"的推理结果**不会逐位相同**，通常只是**近似**；对精度敏感的场景需要按 §11.3 做验证。

---

（个性化提示：学习风格：hands-on）

## 块 62

### 11.3 合并时开启校验并做一致性验证（建议）
`merge_and_unload` 支持 `safe_merge=True`，在合并时会做数值检查（例如检测是否产生 NaN），可作为一道保险：

```python
merged_model = model.merge_and_unload(safe_merge=True)
```

合并完成后，建议**对比合并前后模型在同一批输入上的输出**，确认合并没有引入明显偏差：

```text
准备一小批固定输入（prompt）

↓

分别用「合并前的 PeftModel」和「合并后的模型」推理

↓

对比 logits / 生成文本是否基本一致
```

对应的**最小验证代码**如下（用固定 prompt，先比 logits 的最大绝对差，再比贪心生成文本是否一致）：

（个性化提示：学习风格：hands-on）

## 块 63

## 个性化教学正文：LoRA权重合并与模型导出详解

### 学习目标
掌握 LoRA 模型合并时开启校验并做一致性验证的操作方法，理解其意义和作用。

### 11.3 合并时开启校验并做一致性验证
在 LoRA 模型合并过程中，开启校验并做一致性验证是一个重要的步骤，其意义如下：

#### 1. 校验的作用
校验能够确保合并前的模型参数和合并后的模型参数在数值上的一致性，从而保证模型合并的准确性。具体来说：
- 校验可以帮助发现模型参数在合并过程中可能存在的偏差或错误。
- 校验结果为 0 表示参数完全一致，非零值则表示存在差异，差异大小与校验结果相关。

#### 2. 校验的具体操作
在合并过程中，通过以下步骤进行校验：
- 使用 `generate()` 方法生成预测结果。
- 比较校验前和校验后的预测结果，确保输出一致。
- 检查校验前和校验后的模型参数，确保参数数值一致。

#### 3. 校验后的验证步骤
- 检查校验结果，确保其为 0。
- 比较校验前和校验后的预测结果，确保完全一致。
- 检查校验前和校验后的模型参数，

## 块 64

## 合并时开启校验并做一致性验证（建议

## 块 65

# LoRA微调+权重合并+vLLM部署教学正文

## 1. 确认加载方式
在进行merge_and_unload操作之前，请确保当前对象是`PeftModel`。如果遇到`merge_and_unload`不存在的问题，请检查以下几点：
1. 是否使用了`PeftModel.from_pretrained`挂载Adapter
2. 是否已经合并或加载了模型为普通模型

解决方法：确认加载方式，先得到`PeftModel`后再调用`merge_and_unload`。

---

## 2. 反量化到高精度权重
在直接合并量化模型时，可能会遇到精度问题。解决方法如下：
1. 使用`AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16/float16)`重新加载一份不带量化配置的基础模型
2. 使用`PeftModel.from_pretrained`挂载同一份Adapter
3. 调用`merge_and_unload`进行合并

解决方法：先反量化到FP16/BF16，再合并。

---

## 3. 启用一致性验证
在合并过程中，建议开启校验并做一致性验证。这有助于确保模型在量化和合并过程中保持一致性。操作步骤如下：
1. 在`merge_and_unload`函数中设置`consistency_check=True`
2. 调用`merge_and_unload(adapter_model, output_dir=output_dir, safe_merge=True)`

解决方法：开启校

## 块 66

# 14.1 合并后部署完整模型

在 LoRA 微调和权重合并的基础上，我们可以采用路线A：**合并后部署完整模型**。这一方法在部署阶段具有以下特点：

## 技术实现
1. **merge_and_unload() 函数**：在 LoRA 权重合并完成后，使用 `merge_and_unload()` 函数将模型参数合并并保存为完整模型文件。
2. **推理框架加载**：推理框架可以直接加载合并后的完整模型，无需额外的 LoRA 解析逻辑。这使得部署过程简单高效。

## 优势分析
- **部署简单**：无需额外的 Adapter 切换成本，推理框架可以直接加载合并后的完整模型。
- **存储成本高**：每个 Adapter 需要保存一份完整的模型文件，多任务场景下存储开销较大。

## 适用场景
适用于单任务或少任务场景，尤其是当无需同时支持多个 LoRA Adapter 的情况。

---

## 任务验证
为了验证这一部署方式的正确性，请完成以下操作：
1. 在部署完成后，使用推理框架直接加载合并后的完整模型。
2. 在 vLLM 平台上，尝试使用命令加载完整模型，并验证推理功能是否正常。

---

通过本块学习，您应能够理解路线A的部署逻辑及其适用场景，并能够通过实践验证其正确性。

## 块 67

### 14.3 如何选择
> 经验参考（非硬性规则，以实际部署为准）：
> - **单一 Adapter、追求最简部署** → 倾向路线A（合并）；
> - **同一基础模型要服务多个 Adapter、且推理框架原生支持 LoRA** → 倾向路线B（保留 Adapter 动态服务）；
> - **QLoRA 精度敏感** → 无论哪条路线，都建议按 §11.3 做合并前后/上线前的一致性验证。

具体框架用法与限制留待第 15 板块（vLLM 部署）展开，这里只需建立"合并不是唯一部署方式"的认知。

---

## 十五、总结
完成本章后，你已经掌握：

✅ LoRA Adapter保存方式
✅ 为什么需要权重合并
✅ merge_and_unload作用
✅ LoRA数学合并过程
✅ 完整模型导出流程
✅ 合并模型独立部署方式
✅ LoRA与QLoRA合并注意事项

核心流程：

```text
LoRA训练

↓

保存Adapter

↓

加载Base Model

↓

PeftModel加载Adapter

↓

merge_and_unload()

↓

保存完整模型
```

下一阶段：

**LoRA/QLoRA完整微调实战：PEFT + Trainer训练大语言模型**

将进入端到端训练流程。

（个性化提示：学习风格：hands-on）

## 测评题目

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB12-ST1202-002
LoRA 中 alpha 通常用于对低秩更新进行________。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB12-ST1205-002
使用 `save_pretrained()` 导出模型或适配器时，通常会保存权重和________文件。
