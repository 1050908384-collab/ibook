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
# LoRA微调与QLoRA实现指南

## 3.1 LoRA的问题

虽然LoRA可以有效减少模型的训练参数量，但在实际应用中仍面临一些挑战。

例如，对于一个7B模型：

- 使用FP16精度进行训练，其显存需求约为14GB。

这一显存需求可能会对模型的加载和训练效率带来一定限制。此外，即使在模型训练完成后，加载LoRA参数也需要一定的计算资源支持。

---

## 3.2 QLoRA解决方案

为了克服

## 块 5

```markdown
# LoRA微调与权重合并：从理论到实践

##

## 块 6

```markdown
## 5.1 Python版本

### 推荐

```bash
python --version
```

建议：

```text
Python 3.10+
```

---

### 验证

```python
import sys

print(f"Python版本：{sys.version}")
```

正常：

```text
Python 3.10.5
```

---

### 操作验证

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

```markdown
## 参数高效微调（LoRA与QLoRA）环境准备

### 9.1 推荐使用WSL2

对于Windows用户，我们强烈推荐在WSL2（Windows Subsystem for Linux）环境下进行参数高效微调（LoRA与QLoRA）的设置。WSL2提供了一个基于Linux的环境，使得在Windows系统上安装和配置深度学习工具链变得更加简便。以下是WSL2的安装步骤：

```bash
# 首先安装WSL2
# 在控制面板中找到并安装WSL2
# 或者通过命令提示符运行以下命令：
WSL2-Setup
```

安装完成后，

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

# LoRA微调原理详解：高效

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

```markdown
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

## 六、LoRA在Transformer中的位置

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

四个投影

## 块 18

### 8.1 rank（r）
参数：

```python
r=8
```

代表：

低秩矩阵维度。

越大：

优点：

* 表达能力更强；

缺点：

* 参数更多；
* 显存增加。

常见（下面仅为**经验起始范围**，非确定规则）：

| 任务   | rank（起点参考） |
| ---- | ---- |
| 简单任务 | 约 4-8  |
| 普通微调 | 约 8-32 |
| 复杂任务 | 约 64+  |

> 说明：合适的 `r` 与模型规模、目标模块、数据量、任务难度都相关，上表只作为**起点参考**，实际应结合验证集/基准任务的表现调整。`r` 越大表达能力越强但参数与显存也越多，两者需要权衡。
>
> **基于验证集选 `r` 的一个可操作流程（比背"配方"更可靠）**：
> 1. **划出验证集**：从数据里留出不参与训练的验证集（如 8:2 / 9:1），后续所有比较都在它上面看指标。
> 2. **固定其它变量**：先固定 `target_modules`、学习率、`lora_alpha`（或固定 `alpha/r` 比例）、训练步数等，只改 `r`，做**受控对比**。
> 3. **小范围网格搜索**：从较小值起步（如 `r ∈ {4, 8, 16, 32}`），逐个训练并在验证集上记录指标（loss/accuracy/F1 等）与显存、耗时。
> 4. **看边际收益**：若 `r` 增大带来的验证集提升已很小、甚至开始过拟合（训练指标升但验证指标降），说明再加大 `r` 不划算，取"性价比拐点"处的值。
> 5. **结合成本决策**：在效果接近时，优先选更小的 `r`（更省显存、Adapter 更小）；效果差距明显且资源允许时再上调。
>
> 换言之，`r` 不是查表就能定死的，上表只给**搜索起点**，最终值应由**验证集表现 + 资源成本**共同决定。

---

（个性化提示：学习风格：hands-on）

## 块 19

## 8.2 lora_alpha

用于控制LoRA更新强度。通常作为缩放比例使用。例如：

```python
lora_alpha=16
```

---

## 8.3 lora_dropout

用于防止LoRA参数过拟合。例如：

```python
lora_dropout=0.05
```

---

## 九、LoRA与全

## 块 20

# LoRA核心参数详解：Rank与Alpha参数作用原理（最终版）

## 参考资料

① 权威：《Hugging Face PEFT LoRA Based Methods》
　　网址：https://huggingface.co/docs/peft/main/en/task_guides/lora_based_methods

② 底稿：《LoRA微调参数详解：rank、alpha、dropout参数作用》
　　网址：https://blog.csdn.net/Lvyizhuo/article/details/160256948

③ 参考文章：《LoRA中的rank和alpha参数深入理解》
　　网址：https://blog.csdn.net/qq_48764574/article/details/147729485

---

**适合人群**：已经理解 LoRA 低秩分解原理，希望进一步掌握 LoRA 参数配置方法的学习者。

本文重点解释 LoRA 微调中最核心的两个参数：

* `r（rank）`
* `lora_alpha（alpha）`

理解这两个参数后，你将能够：

* 根据任务规模选择LoRA配置；
* 理解为什么rank越大效果可能越好但成本更高；
* 理解alpha为什么影响训练稳定性；
* 为后续PEFT实战配置LoraConfig打基础。

---

## 一、回顾LoRA基本公式（①②）
上一节已经学习：

LoRA不会直接更新原模型参数：

```text
W
```

而是在原权重基础上增加：

```text
ΔW
```

最终：

```text
W' = W + ΔW
```

其中：

```text
ΔW = B × A
```

两个低秩矩阵：

```text
A

B
```

共同组成LoRA参数。

---

## 二、Rank（r）是什么？（②③）
### 2.1 Rank表示什么？
Rank：

中文：

> 矩阵的秩。

在LoRA中：

rank表示低秩矩阵中间维度大小。

例如：

原始权重：

```text
W:

4096 × 4096
```

如果设置：

```python
r=8
```

那么：

A：

```text
8 × 4096
```

B：

```text
4096 × 8
```

最终：

```text
ΔW = B × A
```

恢复：

```text
4096 × 4096
```

---

（个性化提示：学习风格：hands-on）

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

### 4.1 Rank太小的问题
例如：

```python
r=2
```

可能导致：

* 学习能力不足；
* 欠拟合；
* 新任务知识无法充分表达。

表现：

```text
训练loss下降慢

效果提升有限
```

---

### 4.2 Rank太大的问题
例如：

```python
r=256
```

可能：

* 参数增加；
* 显存增加；
* 训练速度下降。

同时：

LoRA优势降低。

因为：

rank过大时：

参数高效优势进一步下降。

> 说明：这里不宜说成"接近学习完整矩阵"。以文中 `4096 × 4096` 的权重为例，其完整秩为 4096，而 `r=256` 仍**明显小于** 4096，远没有"接近完整矩阵"；只是相对小 rank，`r=256` 的参数量、显存和训练成本都更高，**参数高效的优势被削弱**。
>
> **用参数效率公式和真实比例看清楚（避免"接近完整矩阵"的直觉误解）**：对一个 `d_out × d_in` 的权重，
> - 完整矩阵参数量：`d_out × d_in`；
> - LoRA 增量参数量：`r × (d_in + d_out)`；
> - 两者比例（LoRA 占完整矩阵的比例）：`r × (d_in + d_out) / (d_out × d_in)`。
>
> 代入本文 `d_in = d_out = 4096`：
>
> | rank r | LoRA 参数量 `r×(4096+4096)` | 完整矩阵 `4096×4096` | 占比 |
> | ---: | ---: | ---: | ---: |
> | 8 | 65,536 | 16,777,216 | ≈ 0.39% |
> | 64 | 524,288 | 16,777,216 | ≈ 3.13% |
> | 256 | 2,097,152 | 16,777,216 | ≈ 12.5% |
>
> 可以看到：即便 `r=256`，LoRA 参数也**只占完整矩阵的约 1/8**，谈不上"接近完整矩阵"。之所以说它"参数高效优势下降"，是相对 `r=8`（0.39%）而言占比明显变大、显存与成本升高，而**不是**因为它逼近了满秩。

---

（个性化提示：学习风格：hands-on）

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

### 6.1 控制LoRA影响强度

假设：

rank：

```python
r=8
```

alpha：

```python
alpha=16
```

缩放：

```text
$$\alpha/r = 16

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
# PEFT库LoraConfig配置详解：掌握target_modules、r、lora_alpha、lora_dropout参数（最终版）

## 一、Lora

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

**定义**：
`target_modules` 是指指定需要注入 LoRA 的模型模块。LoRA 通过在这些模块中添加低秩矩阵（low-rank matrix），对模型进行微调和优化。在 Transformer 模型中，`target_modules` 通常指定了注意力层（Attention）中的某些子层，例如查询投影层 (`q_proj`) 和值投影层 (`v_proj`)。

**常见模块结构**：
在 Transformer 中，常见结构如下：

```text
Attention

├── q_proj

├── k_proj

├── v_proj

└── o_proj
```

**配置示例**：
```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**表示**：
通过上述配置，LoRA 会增加 Query 投影层 (`q_proj`) 和 Value 投影层 (`v_proj`) 的低秩矩阵，从而对模型行为进行调整。

---

## 3.2 为什么选择q_proj和v_proj？

**

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

# LoRA微调与权重合并：LLaMA系列模型的参数高效微调配置

## 课程目标
掌握LLaMA系列模型的`target_modules`配置，理解不同模型中`target_modules`的具体区别，并能够熟练进行LoRA微调与

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

    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)
```

---

参数说明：

| 参数             | 值         | 作用     |
| -------------- | --------- | ------ |
| r              | 8         | LoRA容量 |
| alpha          | 16        | 更新缩放   |
| target_modules | q/v       | 插入位置   |
| dropout        | 0.05      | 防止过拟合  |
| task_type      | CAUSAL_LM | 语言模型任务 |

> 补充：`modules_to_save`
>
> 有些**额外模块**不属于 LoRA 注入的目标层，但在你的任务里需要**完整训练并一起保存**，这时用 `modules_to_save` 指定。典型场景：
> - 序列分类任务里新初始化的**分类头**（如 `"classifier"`/`"score"`）；
> - 奖励模型（RLHF）里的**奖励头**；
> - 词表变化时需要一起训练的 embedding/lm_head 等。
>
> 例如：`LoraConfig(..., modules_to_save=["classifier"])`。不加它时，这些新头可能不会被训练/保存，导致重载后任务头缺失或为随机初始化。

带 `modules_to_save` 的**完整配置 + 保存 + 重新加载验证**示例（以序列分类的分类头为例）：

（个性化提示：学习风格：hands-on）

## 块 39

```markdown
# LoRA微调与权重合并：完整LoraConfig案例

## 1. 前言
在之前的课程中，我们学习了Lo

## 块 40

# 十、普通文本生成任务配置

## 学习目标
掌握如何配置LoRA微调参数，实现普通文本生成任务的高效训练。

## 学习方式
基于 hands-on 实践，通过代码示例理解参数配置逻辑，并完成实际任务。

## 学习证据
无具体外部技术事实引用，但所有代码示例均来自原始资料。

---

## 教学任务

### 1. 学习目标
通过实践掌握以下参数的配置与作用：
- `r`：LoRA的秩大小
- `lora_alpha`：LoRA的缩放因子
- `lora_dropout`：LoRA的Dropout概率

### 2. 学习任务
- **理解参数意义**：通过代码注释和运行结果，理解每个参数对模型行为的影响。
- **实践配置**：在实际项目中，根据任务需求动态调整参数值。
- **验证效果**：运行配置后的模型，观察生成文本的质量和效率。

### 3. 学习检查
```python
# 模型微调配置示例
from peft import LoraConfig, get_peft_model

model = get_peft_model(base_model, task_type="text generation")
model_config = LoraConfig(
    target_modules=["l4", "l5

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

```markdown

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

### 6.3 分页优化器（Paged Optimizers）类比理解

#### 类比理解
普通方式：
所有东西都放在桌面上。

桌面满：
无法继续。

分页优化器：
```text
桌面放常用文件
柜子保存暂不用文件
需要时

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

## 9.3.3.2 load_in_4bit配置

### 9.3.3.2.1 启用4-bit加载

开启4-bit量化加载，以减少模型权重的存储空间需求。这将使模型在内存中占用更少，但计算时会使用更高精度的类型来保持准确性。

```python
"enable": True
```

### 9.3.3.2.2 设置量化类型

选择NF4量化类型，这是一种高效的量化方式，能够以4-bit精度存储权重，同时保持较高的模型性能。

```python
"quantization_type": "nf4"
```

### 9.3.3.2.3 使用双重量化

启用双重量化，这是一种结合了NF4量化和双重精度计算的技术。它允许在计算过程中使用更高的精度（如16-bit或32-bit）来提高计算的稳定性和准确性，同时仍然以4-bit精度存储权重。

```python
"enable": True
```

### 9.3.3.2.4 设置计算dtype

指定计算dtype为torch.bfloat16，这是在支持BF16的GPU上进行计算的最佳选择。如果GPU不支持BF16，可以尝试使用torch.float16。

```python
"compute_dtype": "torch.bfloat16"
```

### 9.3.3.2.5 验证配置

在配置完成后，可以通过以下命令验证参数是否正确：

```python
print(f"当前计算dtype: {bnb_4bit_compute_dtype}")
```

### 9.3.3.2.6 注意事项

- 确保GPU支持BF16，否则可能会导致计算

## 块 52

---

### 9.1 QLoRA微调训练的完整实现步骤

## 1. 配置量化参数
- 配置计算精度类型：根据GPU支持选择计算精度类型，确保量化格式正确。
- 使用`BitsAndBytesConfig`参数：
  ```python
  compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=compute_dtype,
  )
  ```

## 2. 加载并量化模型
- 加载基础模型并应用量化配置：
  ```python
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      device_map="auto",
  )
  ```

## 3. 准备模型进行k-bit训练
- 调用`prepare_model_for_kbit_training`：
  ```python
  model =

## 块 53

# QLoRA配置实践指导

## 1. 前言

在进行参数高效微调（LoRA）时，量化配置是实现低成本大模型微调的核心技术之一。本节将详细讲解如何在PyTorch中配置QLoRA相关参数，包括`BitsAndBytesConfig`、`prepare_model_for_kbit_training()`、`get_peft_model()`以及优化器设置。通过实践操作，掌握如何在有限的显存条件下高效运行QLoRA微调任务。

---

## 2. QLoRA配置基础

### 2.1 BitsAndBytesConfig参数配置

`BitsAndBytesConfig`决定了量化相关的参数设置。以下是关键参数及其作用：

- **`quantization_bit`**：指定量化位数。设置为4时，表示进行4-bit量化。
- **`bf16`/`fp16`**：选择使用半精度（BF16/FP16）还是16位全精度。
- **`bnb_4bit_compute_dtype`**：定义计算类型的精度。设置为`bf16`或`fp16`时，会自动启用相应的量化配置。

**示例配置：**
```python
from bitsandbytes import QuantizationConfig

bq = QuantizationConfig(
    quantization_bit=4,
    bf16=True,
   bnb_4bit_compute_dtype="bf16",
)
```

### 2.2 prepare_model_for_kbit_training()

在进行量化微调前，必须调用`prepare_model_for_kbit_training()`函数。该函数会对模型进行适配，包括开启输入梯度、转置某些层等操作，以支持4-bit量化训练。

**示例使用：**
```python
from peft import prepare_model_for_kbit_training

model =

## 块 54

# LoRA权重合并与模型导出详解：merge_and_unload使用方法（最终版）

## 参考资料

① 权威：《Hugging Face PEFT LoraModel.merge_and_unload API》
　　网址：https://huggingface.co/docs/peft/package_reference/lora#peft.LoraModel.merge_and_unload

② 底稿：《LoRA训练后模型合并与导出详解》
　　网址：https://blog.csdn.net/l35633/article/details/161430812

③ 参考文章：《QLoRA/LoRA训练完成后的模型保存与合并》
　　网址：https://blog.csdn.net/weixin_44081096/article/details/163345345

④ 参考文章：《PEFT中LoRA权重合并实践》
　　网址：https://blog.csdn.net/weixin_33773084/article/details/157409170

---

**适合人群**：已经完成 LoRA / QLoRA 微调训练，希望将训练得到的 Adapter 转换为完整模型并用于部署推理的学习者。

本文重点学习：

* LoRA训练后保存的是什么；
* 为什么需要merge；
* `merge_and_unload()`如何工作；
* 如何导出完整模型；
* 合并后如何独立加载模型。

---

## 一、LoRA训练后保存的是什么？（①②）
### 1.1 LoRA不是直接修改基础模型
LoRA训练过程中：

基础模型：

```text
Base Model

冻结
```

LoRA参数：

```text
Adapter

训练
```

最终结构：

```text
基础模型

+

LoRA Adapter
```

---

例如：

训练：

```text
Llama-7B

+

LoRA Adapter
```

保存后：

通常只有：

```text
adapter_model.safetensors

adapter_config.json
```

而不是完整模型。

---

### 1.2 Adapter有什么特点？
优点：

* 文件小；
* 保存快；
* 可以多个任务共享一个基础模型。

例如：

同一个基础模型：

```text
Base Model
```

可以搭配：

```text
客服Adapter

医疗Adapter

法律Adapter
```

---

缺点：

部署时需要：

```text
基础模型

+

Adapter
```

两个部分同时存在。

---

## 二、为什么需要合并LoRA权重？（②③）

（个性化提示：学习风格：hands-on）

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

```markdown
## 三、merge_and_unload是什么？（重点）（①②）

### 3.1 方法作用
PEFT提供：

```python
merge_and_unload()
```

作用：

> 将LoRA权重合并到基础模型，并移除PEFT包装。

简单理解：

执行：

```text
Base Model

+

LoRA Adapter

↓

完整模型
```

---

### 3.2 合并后的变化
合并前：

```text
PeftModel

├── Base Model

└── LoRA Adapter
```

---

合并后：

```text
普通Transformer模型

包含LoRA更新后的权重
```

---

## 四、LoRA模型合并完整流程（重点）（②③）
整体流程：

```text
训练LoRA

↓

保存Adapter

↓

加载基础模型

↓

加载Adapter

↓

merge_and_unload()

↓

保存完整模型
```

---

## 五、加载LoRA Adapter（②）
假设：

基础模型：

```python
base_model_path = "model_path"
```

Adapter：

```python
adapter_path = "adapter_path"
```

加载：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel


model = AutoModelForCausalLM.from_pretrained(
    base_model_path
)

model = PeftModel.from_pretrained(

## 块 57

```markdown
## 合并后的模型加载与推理

### 合并后的模型加载流程

在完成模型合并后，加载模型的过程变得简单高效。以下是详细的加载流程：

1. **模型加载**：
   - 使用 `AutoModelForCausalLM` 从预训练模型路径加载合并后的模型。
   ```python
   from transformers import AutoModelForCausalLM

   model = AutoModelForCausal

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

# 直接合并可能遇到的情况

在进行 LoRA 权重合并时，直接在 4-bit 状态下进行合并可能会遇到以下问题：

* **精度损失**：4-bit 量化状态下的权重反量化与全程高精度状态下的权重存在不完全等价性，可能导致合并后的模型精度下降；
* **权重类型不匹配**：如果 PEFT 或 bitsandbytes 库不支持在量化状态下直接进行合并，可能会导致错误抛出；

> **说明**：以上情况并非必然发生，具体结果取决于 PEFT 或 bitsandbytes 的版本、量化配置以及是否遵循正确的流程。请确保在合并前先反量化到高精度状态。

---

## 常见做法：先反量化到高精度再合并

为了避免直接在 4-bit 状态下合并带来的风险，一种稳妥的做法是：

1. **先反量化到高精度**：在合并前，重新加载基础模型到高精度（如 FP16/BF16）状态；
2. **合并高精度权重**：在反量化后的高精度权重上进行合并，避免 4-bit 量化状态下的精度问题；

这种做法确保了合并过程的安全性和稳定性。

---

## 实战验证

为了验证这一做法的有效性，请完成以下任务：

```python
# 验证任务：验证直接合并可能导致的问题
任务描述：直接合并可能导致精度损失或类型不匹配的问题，请尝试以下步骤并观察结果：

1. 使用原始代码直接在 4-bit 状态下进行合并，并记录是否报错；
2. 改动代码，先反量化到高精度状态再合并，并观察是否成功

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

## 个性化教学正文

### 11.3 合并时开启校验并做一致性验证

在使用 `merge_and_unload()` 函数进行模型合并时，必须开启校验（`check=True`）并进行一致性验证。这是确保合并过程顺利进行的关键步骤。

#### 1. 合并时开启校验
在调用 `merge_and_unload()` 函数时，需要将 `check` 参数设置为 `True`，以启用校验功能。这将帮助我们验证模型的合并是否正确。

```python
merged_model = model_before.merge_and_unload(
    adapter_name=adapter_name,
    rank=rank,
    bias=bias,
    check=True  # 启用校验
)
```

#### 2. 校验的作用
校验的主要作用是确保合并后的模型与原始模型在参数上的一致性。具体包括以下内容：
- **参数值的校验**：检查合并后的参数值是否与原始模型一致。
- **模型结构的校验**：确保模型的结构没有发生改变。

#### 3. 校验的验证方式
校验可以通过以下方式验证：
- 比较合并前后的模型参数值是否一致。
- 检查模型的结构是否与预期一致。

#### 4. 校验的验证代码示例
```python
# 比较合并前后的模型参数值
for param_name, param_before in model_before.named_parameters():
    param_after = merged_model.named_parameters()[param_name]
    assert torch.allclose(param_before, param_after), f"参数 {param_name} 的值在合并过程中发生了变化"
```

#### 5. 校验的重要性

## 块 64

## 十二、电脑配置批注
本章节涉及：

* 加载完整模型；
* 合并权重；
* 导出模型。

需要实际加载模型。

最低：

CPU：

* 4核+

内存：

* 16GB+

磁盘：

* 预留模型大小2倍空间。

---

GPU推荐（下面为**带条件经验参考**，不是固定门槛）：

7B模型：

经验参考：

* 约 16GB 显存起（宽松些约 24GB+）

> 说明：合并本身是"权重相加"的一次性计算，**不一定必须在 GPU 上完成**——很多情况下直接在 **CPU + 内存**里加载 FP16/BF16 模型并合并即可，只是速度较慢。上面的 16GB /

## 块 65

# LoRA微调与权重合并：安全高效部署指南

## 课程背景

在深度学习模型优化中，`merge_and_unload` 是一个关键操作，用于将微调后的模型参数与原模型进行高效合并，确保推理部署的顺利进行。本课程将结合实际案例，深入讲解如何在量化模型中安全地使用 `merge_and_unload`，并解决实际应用中可能遇到的问题。

## 学习目标

1. 掌握 LoRA 模型微调与权重合并的基本流程
2. 熟悉如何在量化模型中使用 `merge_and_unload` 操作

## 块 66

```markdown
# 14.1 合并后部署完整模型

### 学习目标：
- 掌握如何通过合并LoRA权重后直接部署完整模型
- 理解部署链路的简单性及其潜在成本

### 方法步骤：
1. **合并LoRA权重**：
   - 使用`merge_and_unload()`命令将多个LoRA权重合并到基础模型中
   - 确保所有LoRA权重已正确加载并保存

2. **保存完整模型**：
   - 使用`save()`命令将合并后的模型保存为完整形式
   - 确保模型文件完整无误

3. **推理框架加载模型**：
   - 在推理框架中直接加载保存的完整模型
   - 注意：无需额外配置，推理框架将直接使用合并后的模型

4. **验证部署效果**：
   - 使用示例推理命令（如`vllm generate`）进行测试
   - 观察推理速度和资源占用情况

### 注意事项：
- 该方法仅适用于单任务场景
- 存储成本较高，建议在多任务场景中使用路线B

---

### 学习检查：
**任务：验证合并后完整模型的推理性能**
- **任务描述**：
  - 在实际部署环境中，使用合并后的完整模型进行推理测试
  - 记录推理速度、资源占用和模型加载时间
- **检查要点**：
  - 是否成功加载完整模型
  - 是否出现推理延迟或资源不足
  - 是否需要额外的配置或优化

---

### 总结：
路线A通过直接加载合并后的完整模型实现部署，操作简单但存储

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

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB12-ST1200-001
使用 Hugging Face PEFT 实现 LoRA 时，需要安装的库名称是________。

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB12-ST1204-001
QLoRA 常用的 4-bit 数据类型名称是________。

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。
