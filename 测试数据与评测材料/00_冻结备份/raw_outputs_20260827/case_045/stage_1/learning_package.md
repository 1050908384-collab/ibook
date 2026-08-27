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
4bit基础模型

+

LoRA参数

↓

训练
```

---

### 3.3 QLo

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

QL

## 块 6

## 七、安装bitsand

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

# 参数高效微调（LoRA与QLoRA）环境准备指南

## 一、简介
参数高效微调（LoRA与QLoRA）是一种高效的方法，用于在保持模型性能的同时减少模型的

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

```
# LoRA低秩分解原理详解：理解大模型参数高效微调核心思想（最终版）

## 适合人群
- 已经完成 Transformers、Datasets、Trainer、PEFT 环境准备，希望深入理解 LoRA 微调原理的学习者。

本文将重点解释以下核心问题：
1. 为什么大模型微调不需要更新全部参数；
2. LoRA为什么采用低秩矩阵；
3. ΔW = BA到底代表什么；
4.

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

### 5.2 类比理解

**假设：**

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

**LoRA类似：**

> 用少量参数描述模型变化方向。

---

**LoRA在Transformer中的位置**

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

**LoRA训练过程**

完整流程：

```text
加载预训练模型

↓

冻结所有参数

↓

插入LoRA矩阵

↓

训练LoRA参数

↓

保存Adapter
```

---

**训练时：**

原模型：

```text
W

冻结
```

LoRA：

```text
A

B

更新
```

---

**最终：**

模型：

```text
原模型

+

LoRA Adapter
```

---

**学习检查**

请完成以下任务以验证对LoRA的理解：

1. 请解释LoRA的核心思想，并与图片压缩类比对比。
2. 请描述LoRA在Transformer中的具体应用位置（如Q_proj、V_proj等）。
3. 请设计一个简单的LoRA配置示例，并说明其含义。

**答案示例：**

1. LoRA的核心思想是通过少量参数描述模型的变化方向，类似于用少量的调色板调整一张高清图片的整体色调和细节变化，从而实现参数

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

```markdown
## 8.2 lora_alpha

### 什么是lora_alpha？
`lora_alpha` 是 LoRA（Low-Rank Adaptation）中的一个核心超参数，用于控制 LoRA 参数的更新强度。它类似于全参数微调中的学习率，但具体作用有所不同。

### lora_alpha的作用
- **缩放比例**：`lora_alpha` 通常与 `lora_rank` 一起使用，作为低秩分解的缩放因子。例如：
  ```python
  lora_alpha=16
  ```
  这意味着 LoRA 参数的更新会以 16 倍的比例进行调整。

- **更新强度控制**：通过调整 `lora_alpha`，我们可以控制 LoRA 参数对模型的更新程度。

## 块 20

```markdown
# LoRA核心参数详解：Rank与Alpha参数作用原理（最终版）

## 适合人群
已经理解LoRA低秩分解原理，希望进一步掌握LoRA参数配置方法的学习者。

本文重点解释LoRA微调中最核心的两个参数：

* `rank（r）`
* `lora_alpha（alpha）`

理解这两个

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

```markdown
# LoRA微调：控制LoRA影响强度

在LoRA微调中，Rank和Alpha参数是调节模型更新强度的关键超参数。通过调整这两个参数，我们可以有效控制LoRA对模型的影响程度，从而实现更精细的微调效果。

## 6.1 控制LoRA影响强度

### Rank参数的作用
Rank参数决定了LoRA分解矩阵的大小，直接影响模型的参数规模。较大的Rank值意味着更复杂的LoRA结构，但同时也增加了模型的计算和内存需求。

### Alpha参数的作用
Alpha参数是一个缩放因子，用于调整LoRA更新的强度。通过Alpha参数，我们可以控制LoRA对模型权重的更新幅度。

### Alpha与Rank的组合
Alpha和Rank的组合决定了LoRA更新的最终强度。具体来说，LoRA的更新强度由`α/r`决定

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

## 十、Rank与Alpha选择误区（②③） > 误区1：

## 块 29

```markdown
# PEFT库LoraConfig配置详解：掌握target_modules、r、lora_alpha、lora_dropout参数（最终版）

## 适合人群
- 已经理解LoRA低秩分解原理，并完成PEFT环境安装
- 希望进入实际LoRA微调配置阶段的学习者

本文重点学习PEFT中最核心的配置对象：**LoraConfig**。通过本章学习，你可以掌握：
*

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

**目标模块**：指定LoRA要注入哪些模型模块。

在Transformer架构中，常见模块包括：

```text
Attention

├── q_proj

├── k_proj

├── v_proj

└── o_proj
```

**LoRA不会默认修改所有层**。需要明确指定要注入的模块，通过`target_modules`参数来实现。

**示例配置**：

```python
target_modules=[
    "q_proj",
    "v_proj"
]
```

**表示**：

LoRA将对以下模块进行微调：

* **q_proj**：查询投影层；
* **v_proj**：值投影层；

通过这些模块的调整，可以有效改变模型的行为。

---

## 3.2 为什么选择q_proj和v_proj？

在注意力计算中，查询投影层（q_proj）和值投影层（v_proj）的调整通常能够有效影响模型

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
# LoRA微调与参数高效微调：LLaMA系列模型的target_modules配置

## 课程目标
掌握LLaMA系列模型中不同模型的`target_modules`配置，重点理解并实践如何在

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
> - **版本差异**：不同 Transformers/PEFT 版本、不同模型实现，模块命名可能变化；`"all-linear"` 等便捷写法也依赖较新版本。请以你实际加载模型 `named_modules()` 的输出与所用版本文

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

带 `

## 块 39

```markdown
# LoRA微调与参数高效微调（LoRA与QLoRA）案例

## 9.1 基础配置

在进行LoRA微调时，配置`LoraConfig`是关键步骤之一。以下将通过一个完整案例来展示如何配置和使用`LoraConfig`。

### 1. 导入必要的库和工具

```python
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForSequenceClassification
```

### 2. 加载基础模型和配置

```python
base_name = "bert-base-uncased"

# 加载带有分类头的基础模型
base_model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
```

### 3. 配置LoRA参数

```python
# 配置LoRA参数
lora_config = LoraConfig(
    r=8,  # LoRA的秩（rank）
    lora_alpha=16,  # LoRA的缩放因子（alpha）
    lora_dropout=0.05,  # LoRA的Dropout率
    bias="none",  # LoRA的偏置项设置
    task_type="SEQ_CLS",  # 序列分类任务
    target_modules=["query", "value"],  # BERT中注意力模块的名称
    modules_to_save=["classifier"]  # 指定需要保存和恢复的模块
)
```

### 4. 应用LoRA配置到模型

```python
# 应用LoRA配置到基础模型
model = get_peft_model(base_model, lora_config)
```

### 5. 可视化LoRA参数

```python
# 显示可训练
```

## 块 40

```markdown
## 十、普通文本生成

### 10.1 学习目标
掌握如何配置 `LoraConfig` 参数以实现普通文本生成任务。重点理解以下参数的作用：
- `target_modules`：指定需要微调的模块
- `r`：模型压缩率
- `lora_alpha`：缩放因子
- `lora_dropout`：Dropout率

### 10.2 推荐配置
为实现普通文本生成任务，推荐以下配置：
```python
r=8
alpha=16
dropout=0.05
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
# 注意：Llama-2-7b FP16 需要至少14GB显存
```
这些参数结合了模型压缩率和适当的缩放因子，能够在保持性能的同时实现微调。

### 10.3 实践步骤
1. **初始化模型**：加载预训练模型并启用LoRA微调。
2. **配置LoRA**：使用推荐参数创建 `LoraConfig` 实例。
3. **微调训练**：设置训练参数并执行微调过程。
4. **评估性能**：检查模型生成文本的质量和多样性。

### 10.4 学习检查
完成本节

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

LoRA虽然减少了训练参数：

```text
原模型参数：
冻结

LoRA参数：
训练
```

但是：

基础模型仍然需要加载。

例如：

7B模型：

FP16：

```text
7B × 2 bytes
≈14GB显存
```

如果加入：

* 激活值；
* 梯度；
* 优化器；

实际需求更高。

---

### 1.2 Q

> 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。

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
基础

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

#### 学习目标
掌握分页优化器的基本原理，理解其在训练中的实际

## 块 50

# QLoRA显存优化技术详解

## 7.1 4-bit量化

### 作用
减少模型权重大小，降低显存占用。

### 实现方式
```text
FP16

↓

NF4 4-bit
```

### 实际操作步骤
1. 导入必要的库：
```python
from bitsandbytes import NhseForLoRA
```

2. 配置量化参数：
```python
bnb_config = BitsAndBytes
```

## 块 51

# QLoRA微调：4-bit量化与分页优化器机制

## 9.3.1 QLoRA配置：开启4-bit量化

### 9.3.1.1 启用4-bit加载
在QLoRA微调过程中，4-bit量化是一种高效且节省资源的量化方式。开启4-bit加载可以显著降低模型的显存占用，同时保持模型性能。以下是实现4-bit加载的命令配置：

```python
# 启用4-bit量化
from bitsandbytes import Quantization

# 使用默认设置即可
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）
```

### 9.3.1.2 设置量化类型
在4-bit量化中，可以选择不同的量化类型来适应不同的场景。以下是常见的两种量化类型：

#### 9

## 块 52

```markdown
# 9.1 QLoRA微调实战指南：构建可运行的QLoRA训练框架

## 学习目标
通过本节学习，你将能够：
1. 理解如何将量化配置、梯度检查点和LoRA结构串联起来
2. 掌握最小可运行的QLoRA微调训练框架
3. 熟悉从模型加载到微调的完整流程

## 前言
在学习完QLoRA相关理论后，本节将通过一步步的实践，帮助你构建一个完整的QLoRA微调训练框架。通过本节的学习，你可以将理论知识转化为实际的代码操作，并验证其正确性。

---

## 实战步骤：构建QLoRA微调框架

### 步骤1：加载模型并配置量化参数
在开始微调之前，我们需要为模型加载预训练权重并配置量化参数。以下是具体的代码实现：

```python
from bitsandbytes import QuantizationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

# 加载预训练模型
# 注意：Llama-2-7b FP16 需要至少14GB显存
model = LlamaForCausalLM.from_pretrained("your-model-name")
tokenizer = LlamaTokenizer.from_pretrained("your-model-name")

# 配置量化参数
quantization_config = QuantizationConfig(
    quantization_method="nf4",
    use_f16l16bit=True,
    use_diff quant=True,
    compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
```

### 步骤2：准备模型进行k-bit训练
为了进行低精度训练，我们需要对模型进行一些参数调整：

```python
from peft import PeftModel, LoraConfig, get_peft_model

# 启用梯度检查点
training_args = TrainingArguments(
    output_dir='./qlora_ckpt',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit"
)

# 准备模型进行k-bit训练
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"))
model.print_trainable_parameters()
```

### 步骤3：注入LoRA结构
接下来，我们需要将LoRA结构注入到模型中：

```python
from peft import PeftModel

# 注入LoRA结构
model = PeftModel(model, "qlora")
model.push_to_model("your-model-name")
```

### 步骤4：准备训练数据
为了训练模型，我们需要准备一个训练数据集：

```python
from datasets import Dataset
from transformers import AutoTokenizer

def preprocess_function(examples):
    inputs = examples["
```

## 块 53

# QLoRA微调详解：从原理到实践

## 1. 什么是QLoRA？
QLoRA（Quantized LoRA）是一种结合了4-bit量化和分页优化器的微调方法，旨在降低模型显存占用，同时保持微调效果。它通过优化权重矩阵的存储和计算方式，将模型参数规模大幅缩减。

## 2. QLoRA的工作原理
QLoRA分为三个主要步骤：
1. **量化配置**：使用`BitsAndBytesConfig`设置模型的量化参数，如`nf4`（NF4量化）、双重量化和数据类型设置。
2. **模型适配**：通过`prepare_model_for_kbit_training()`将模型适配为适合量化训练的形式，包括开启梯度检查点和调整计算精度。
3. **LoRA注入**：使用`get_peft_model()`将可训练的LoRA结构注入到量化后的模型中，完成微调准备。

## 3. QLoRA与分页优化器的结合
分页优化器通过将模型参数拆分为多个页框（paged），有效降低了显存占用。在微调过程中，合理配置分页参数可以显著缓解显存压力。

## 4. 实训指南
- **硬件要求**：最低需16GB显存，更高模型规模需更大显存。
- **参数配置**：根据模型规模和训练需求调整`optim`参数，如`"paged_adamw_8bit"`。
- **训练优化**：通过开启梯度检查点和优化批处理方式，进一步提升训练效率。

## 5. 实验验证
通过实验可以发现：
- QLoRA在微调效率和显存占用之间找到了平衡。
- 适当调整显存设置和训练参数，可实现更高效的模型微调。

## 本块学习检查
请完成以下任务以验证对QLoRA的理解：
1. 使用`BitsAndBytesConfig`配置一个7B级别模型的NF4量化设置。
2. 编写代码实现

## 块 54

# LoRA权重合并与模型导出详解：merge_and_unload使用方法

## 适合人群
- 已经完成 LoRA / QLoRA 微调训练，并希望将训练得到的 Adapter 转换为完整模型以便部署推理的学习者。

本文重点学习：
1. LoRA训练后保存的内容；

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

## 块 56

```markdown
# LoRA微调：merge_and_unload方法详解

## 什么是merge_and_unload？

`merge_and_unload()` 是 PEFT（参数高效微调）框架中的核心方法之一，主要用于将 LoRA（Low-Rank Adaptation）权重与基础模型合并，并移除 PEFT 包裹的结构。通过此方法，可以将微调后的 LoRA 权重直接加载到基础模型中，生成一个完整且可直接使用的训练模型。

### 简单理解

1. **加载 LoRA 权重**：将微调过程中生成的 LoRA 权重加载到基础模型中。
2. # 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权）。
3. # 注意：Llama-

## 块 57

```markdown
## 合并后的模型加载与推理流程

在完成LoRA权重合并后，模型的加载与推理流程如下：

1. **确认模型完整性**：确保导出

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

> 权衡：合并后的完整模型体积**明显更大**（等于一份完整基础模型），分发/存储成本更高；如果对方已经有同一基础模型，单独分享体积很小的 Adapter 往往更省带宽

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

# LoRA微调：直接合并的注意事项与常见问题

## 学习目标
掌握直接合并LoRA权重时可能遇到的问题，并学习如何通过先反量化到高精度再合并的方法解决这些问题。

## 学习方式
通过代码示例和实践操作，逐步掌握直接合并的潜在风险及应对策略。

## 学习检查
完成代码实践并验证模型合并后的权重状态。

---

### 11.1 直接合并可能遇到的问题
在4-bit量化状态下直接合并LoRA权重时，可能会遇到以下问题：
1. **精度损失**：4-bit量化状态下的权重反量化与全程高精度不完全匹配，可能导致合并后的模型精度下降。
2. **权重类型不匹配**：如果所选的量化版本不支持直接使用`merge_and_unload`方法，可能导致合并失败。

> **说明**：直接合并并非总是成功，具体情况取决于PEFT、bitsandbytes和Transformers的版本、量化配置以及是否遵循正确的流程（即先反量化到高精度再合并）。请以实际运行结果为准。

### 11.2 常见做法：先反量化到高精度再合并
为了避免上述问题，建议采用以下步骤进行模型合并：
1. **加载高精度基础模型**：在量化状态下直接加载基础模型可能会导致量化配置与LoRA权重的不匹配，因此需要先加载高精度的模型。
2. **加载LoRA Adapter**：挂载

## 块 61

> 提示（版本/内存）：
> - **API/版本**：不同 PEFT 版本对"量化模型能否直接合并"的支持不同，部分版本要求或推荐先反量化；请查阅所用版本的 `merge_and_unload` 文档。上例的关键是**加载基础模型时不再传 `quantization_config`/`load_in_4bit`**，从而得到高精度权重再合并。
> - **内存**：以高精度重新加载完整基础模型会占用**较多内存/显存**（约等于一份 FP16/BF16 完整模型），显存不足时可把 `device_map` 改为 `{"": "cpu"}` 在 CPU 上加载并合并（较慢），或使用 offload。
> - **一致性**：QLoRA 合并得到的高精度模型，与"训练时 4-bit + Adapter"的推理结果**不会逐位相同**，通常

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

# LoRA微调：合并时开启校验并做一致性验证

## 学习目标
通过本节学习，掌握在LoRA微调过程中如何开启校验功能并进行模型一致性验证，确保合并后的模型性能与原模型一致。

## 实训目标
1. 理解合并时开启校验的必要性
2. 掌握校验方法及其在模型微调中的应用
3. 学会通过校验验证模型性能是否一致

## 块 64

# LoRA微调与QLoRA模型合并注意事项（重点）

## 1. 合并时开启校验并做一致性验证

在使用`merge_and_unload()`函数进行模型合并时，必须在合并前进行校验，并确保合并后的权重与原权重在数值上具有高度一致性。以下是具体操作步骤：

1. **加载基础模型**：确保加载的是未合并的`PeftModel`，并保存其参数作为对照。
2. **加载合并模型**：使用`base_for_ref`和`model_before`变量分别加载基础模型和合并前的`LoRA`或`QLoRA`参数。
3. **开启校验**：在合并过程中，启用校验功能

# 模型许可证：Llama 2 Community License

## 块 65

## QLoRA模型合并与部署：从基础到实践

### 1. QLoRA基础概念
QLoRA（Quantized Low-Rank Adaptation）是一种高效的微调方法，结合了参数高效微调（LoRA）和量化技术。其核心思想是通过分解权重矩阵并引入低秩近似，显著减少模型参数量，同时保持模型性能。

## 块 66

# LoRA微调部署路线详解

## 路线A：合并后部署完整模型

### 技术特点
1. **部署简单**：推理框架直接加载合并后的完整模型，无需额外配置。
2. **切换成本低**：推理时无需切换Adapter，直接使用合并后的模型。
3. **代价高昂**：每个Adapter都需要存储完整模型，多任务场景下存储成本较高。

### 实战场景
- **单模型推理**：适合仅需一个模型的场景，部署效率高。
- **资源消耗高**：适用于单模型场景，但不适合多任务或多租户场景。

### 示例代码
```text
# 模型许可证：Llama 2 Community License（商用月活>7亿需Meta授权

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

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB12-ST1204-002
双重量化指的是进一步量化量化常数，以继续节省________。

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB1201-005
请使用 `peft` 库，为 `model` 配置 LoRA，要求 `r=16`，`lora_alpha=32`，目标模块为所有线性层（`all-linear`），`lora_dropout=0.05`。

### Q-KB12-ST1203-003
若希望将 LoRA 注入注意力层的 query 与 value 投影，应配置什么？

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。
