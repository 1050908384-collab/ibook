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

## 块 5

### 电脑配置批注

#### 环境准备步骤指南：LoRA/QLoRA参数高效微调

**1. 安装依赖库**
为了进行参数高效微调，我们需要安装以下工具包：

- **PEFT**：用于量化模型和LoRA/QLoRA微调。
- **bitsandbytes**：用于模型量化。
- **GPU

## 块 6

# 参数高效微调（LoRA与QLoRA）环境准备：安装PEFT、bitsandbytes并配置GPU环境（最终版）

## 5.1 Python版本

### 推荐
Python 3.10+ 是当前推荐版本，因为它支持最新的工具链和库。

### 建议
```bash
python --version
```

运行上述命令以确认已安装的Python版本。

### 检查Python版本
```text
Python 3.10+
```

### 验证
```bash
python3 --version
```

### 输出示例
```
Python 3.10.8
```

### 注意事项
确保Python版本与后续依赖项兼容，避免因版本不兼容导致的错误。

---

## 5.2 PyTorch检查

### 前置条件
LoRA/QLoRA依赖PyTorch，且`bits

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

#### 问题2：
CUDA版本不匹配。

检查：

```bash
nvidia-smi
```

确认：

* NVIDIA驱动；
* CUDA版本；
* PyTorch版本。

---

## 十、安装完整LoRA环境
在已有环境：

```bash
pip install transformers
pip install datasets
pip install accelerate
```

基础上：

> 前置：确保已按 5.2 安装**匹配本机 CUDA 的 GPU 版 PyTorch**（否则后续 `import torch` 或 QLoRA 4-bit 会失败）。CUDA 12.1 示例：
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

增加：

```bash
pip install peft bitsandbytes
```

完整：

```bash
pip install "transformers[sentencepiece]" datasets accelerate peft bitsandbytes
```

---

## 十一、配置Accelerate（参考文章③）
执行：

```bash
accelerate config
```

根据环境选择：

例如单GPU：

```text
Compute environment:
This machine

Distributed:
No

Mixed precision:
fp16
```

生成：

```text
default_config.yaml
```

---

## 十二、LoRA环境验证实验（参考文章③）
创建：

```text
test_lora_env.py
```

代码：

```python
import torch

from peft import LoraConfig


print(
    "CUDA:",
    torch.cuda.is_available()
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


print(config)
```

运行：

```bash
python test_lora_env.py
```

成功：

说明：

✅ PyTorch正常
✅ PEFT正常
✅ LoRA配置可用

---

## 十三、QLoRA环境验证实验（参考文章③）
测试bitsandbytes：

```python
import torch
import bitsandbytes


print(
    torch.cuda.is_available()
)

print(
    bitsandbytes.__version__
)
```

如果：

```text
True
```

说明：

GPU环境正常。

---

（个性化提示：学习风格：hands-on）

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
# LoRA低秩分解原理详解

## 为什么需要LoRA？

### 1.1 大模型微调的问题

现代大语言模型通常拥有数十亿甚至上百亿参数，采用传统微调方式需要更新所有参数，这会带来以下问题：

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

# 5.2 类比理解

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

四个投影矩阵。

---

LoRA在早期实践中**常见**加入：

```text
Q投影

V投影
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
Wq变化

Wv变化
```

> 说明：只挑 `q_proj`/`v_proj` 是 LoRA 原论文里的一种**常见选择**，并**不是固定标准**。现代实践中也经常把 LoRA 加到注意力的全部投影（q/k/v/o）、FFN 的线性层，甚至用 `target_modules="all-linear"` 覆盖所有线性层。选择哪些模块会影响可训练参数量、显存和效果，应结合模型结构（先确认实际模块名）与验证集表现来定，详见后续《LoraConfig 配置》一章。

为了让上面的说明不与实操脱节，这里给出**一个覆盖更完整线性层**的配置示例（以 Llama 类模型

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

用于控制LoRA更新强度的参数是`lora_alpha`。通常我们会设置为`lora_alpha=16`，这个值决定了LoRA参数更新的幅度。alpha/r是LoRA更新中的一个缩放比例，其中r表示LoRA的秩。例如，如果我们设置`lora_alpha=

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

```markdown
# 6.1 控制LoRA影响强度

## 6.1.1 rank与alpha的作用

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

# 误区1：Rank越大越好  
认为：  
- rank越大一定越好。  
错误：  
- 在任务简单时，较大的rank可能导致显存浪费和增加训练成本。  
原因：  
-

## 块 29

# PEFT库LoraConfig配置详解：掌握target_modules、r、lora_alpha、lora_dropout参数（最终版）

## 一、LoraConfig是什么？（①②③）

### 1.1 PEFT中的LoraConfig作用

PEFT（Parameter-Efficient Fine-Tuning）是Hugging Face提供的高效微调库，主要用于在不增加模型参数的情况下进行模型微调。LoRA（Low-Rank Adaptation）训练流程大致如下：

```text
加载基础模型

↓

创建Lora

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
# PEFT库LoraConfig配置详解

## 3.1 target_modules是什么？
**定义：**
`target_modules` 是指指定 LoRA 配置将注入到模型中的模块。简单来说，它告诉 PEFT 应该对哪些模型部分进行低秩调整。

**常见模块示例：**
```python
target_modules=[
    "q_proj",
    "v_proj"
]
```
这表示 LoRA 将注入到模型的 `q_proj`（查询投影层）和 `v_proj`（值投影层）模块中。

**作用：**
通过调整这些模块的参数（如权重矩阵），模型可以在不大幅增加参数量的情况下，实现对原模型行为的微调。

---

## 4. 为什么选择q_proj和v_proj？
### 4.1 q_proj和v_proj的作用
在注意力机制中，`q_proj` 和 `v_proj` 分别负责生成查询和值向量。调整这些投影层可以有效改变注意力计算的结果，从而影响模型的输出。

**公式回顾：**
```text
Attention(Q,K,V) = softmax(QKᵀ/√d)V
``

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

# LLaMA系列模型中PEFT库LoraConfig配置详解

## 课程目标：
掌握 LLaMA 模型中不同 target_modules 对 LoRA 配置的影响。

## 学习步骤：

### 1. 理解基础概念
- **LoRA（Low-Rank Adaptation）**：一种高效微调方法，通过在模型权重

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
# 九、完整LoraConfig案例（重点）

## 9.1 基础配置

在使用PEFT库进行参数高效微调（LoRA与QLoRA）时，LoraConfig是一个核心配置工具。它允许我们定义如何对基础模型进行低秩逼近（Low-Rank Approximation）优化，从而减少模型参数量并提高微调效率。以下是LoraConfig的主要参数及其作用：

1. **`r`**：表示LoRA分解的秩，决定了我们希望优化的参数数量。较大的`r`值会保留更多的参数，但也会增加微调的计算量和内存占用。

2. **`lora_alpha`**：控制LoRA权重的缩放比例。它决定了每个LoRA块的初始权重大小，较大的`alpha`值会增加权重的多样性，有助于模型更好地适应特定任务。

3. **`lora_dropout`**：表示LoRA块中被随机丢弃的参数比例。适当的`dropout`值可以防止模型过拟合，提高模型的泛化能力。

4. **`bias`**：指定LoRA块中偏置项（bias）的处理方式。支持的值包括：
   - `"none"`：不添加偏置项。
   - `"low"`：仅添加低秩偏置项。
   - `"full"`：同时添加低秩和全秩偏置项。

5. **`task_type`**：指定任务类型，影响模型的输出层设计。常见的值包括：
   - `"SEQ_CLS"`：适用于序列分类任务（如文本分类）。
   - `"CAUSAL"`：适用于带有因果掩码的下游任务（

## 块 40

# PEFT库LoraConfig配置详解：掌握target_modules、r、lora_alpha、lora_dropout参数（最终版）

## 10.1 普通文本生成

### 推荐配置参数
在进行普通文本生成任务时，推荐使用以下参数配置：

```python
r=8
alpha=16
dropout=0.05
```

### 参数解释
- **r=8**：表示LoRA分解的低

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

## QLoRA微调：高效量化微调指南

### 一、为什么需要QLoRA？
在微调大模型时，传统方法需要加载完整模型，这会带来巨大的显存需求。QLoRA通过优化LoRA方法，显著降低了显存占用，使其在消费级GPU上运行。

#### 1.

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

# QLoRA微调中的分页优化器实践

## 6.3 分页优化器类比理解

### 普通方式
所有东西都放在桌面上。

当桌面满时：
无法继续。

### 分页优化器
分页优化器就像整理文件到抽屉和橱柜中。

```text

## 块 50

### 7.1 4-bit量化

作用：减少模型权重大小。

```text
FP16

↓

NF4 4-bit
```

---

### 7

## 块 51

# QLoRA微调：4-bit量化与bitsandbytes配置

## 4-bit量化配置指南

### 基本概念
4-bit量化是一种通过压缩权重精度来降低模型大小并提升推理效率的技术。与传统的16-bit量化相比，4-bit量化可以显著减少显存占用，同时保持较高的推理性能。

### 配置步骤

#### 1. 启用4-bit量化
在模型初始化时，开启4-bit量化：

```python
load_in_4bit = True
```

#### 2. 设置量化类型
选择NF4量化方式：

```python
bnb_4bit_quant_type = "nf4"
```

#### 3. 启用双重量化
双重量化结合NF4量化，进一步提升性能：

```python

## 块 52

```markdown
## 9.1 QLoRA微调：量化配置、梯度检查点与LoRA注入的完整流程

### 学习目标
掌握如何将QLoRA微调的各个关键组件（量化配置、梯度检查点与LoRA注入）串联起来，构建一个可执行的微调框架。

### 学习流程

#### 1. 量化配置与模型加载
- **量化配置**：根据模型的计算能力选择合适的精度（如BF16或BF16/FP16混合精度）。
- **模型加载**：使用`BitsAndBytesConfig`参数化模型加载，确保模型在较低精度下运行。

#### 2. 梯度检查点与模型准备
- **梯度检查点**：启用梯度检查点，以优化微调过程中的内存使用。
- **模型准备**：通过`prepare_model_for_kbit_training`函数准备模型，确保其支持微调所需的中间层激活。

#### 3. LoRA注入
- **LoRA配置**：定义LoRA的参数（如秩r、alpha值等）。
- **模型注入**：通过`get_peft_model`函数将LoRA结构注入到模型中，仅训练低秩增量参数。

#### 4. 数据准备与训练设置
- **数据预处理**：构建适合微调的数据集，并定义数据加载器。
- **训练参数设置**：配置训练参数（如批次大小、学习率、训练轮数等），确保与模型准备相匹配。

#### 5. 实战验证与检查
- **模型检查**：在模型加载完成后，进行关键组件的检查，确保量化配置、梯度检查点与LoRA注入均正确配置。
- **训练验证**：运行微调训练过程，观察模型性能与训练稳定性。

### 实战任务
1. **模型检查任务**：
   ```python
   # 检查模型是否正确加载
   print("模型是否正确加载？", model)
   ```

2. **微调训练任务**：
   ```python
   # 启动微调训练
   trainer.train()
   ```

通过以上步骤的学习与实践，您可以构建一个完整的QLoRA微调框架，实现模型的高效训练与优化。
```

### 学习检查
**本块学习检查**

任务内容：
1. 检查模型是否正确加载并进行梯度检查点的设置。
2. 验证LoRA是否成功注入到模型中。
3. 确保微调参数设置与训练器配置一致。

通过以上任务的完成，您可以验证自己是否掌握了QLoRA微调的核心技术与实现细节。

## 块 53

```markdown
# QLoRA微调指南

## 1. 前言
QLoRA（Quantized LoRA）是一种高效微调大模型的方法，结合了4-bit量化、分页优化器等技术，显著降低了模型的显存占用。本章节将从原理到实践逐步讲解QLoRA微调的配置与运行。

## 2. QLoRA原理详解
### 2.1 4-bit量化
QLoRA的核心在于4-bit量化，通过将权重压缩到4个bit，显著降低了模型的显存占用。这需要配置合适的量化参数，如`BitsAndBytesConfig`，并选择合适的计算数据类型（如`bnb_4bit_compute_dtype`）。

### 2.2 分页优化器
分页优化器通过将模型权重划分为多个页框，减少了显存的使用上限。在训练过程中，需要启用`use_gradient_checkpointing=True`，以支持梯度检查点的使用。

### 2.3 LoRA注入
LoRA（Low-Rank Adaptation）是一种高效的微调方法，通过在主模型中注入可训练的低秩层，实现了参数的高效优化。在配置LoRA时，需要指定目标模块（如注意力层）以及相关的LORA参数。

### 2.4 训练参数设置
为了确保训练的稳定性与效率，需要合理设置训练参数。例如，`optim="paged_adamw_8bit"`是一种适合分页优化器的优化器配置，能够缓解显存压力

## 块 54

# LoRA权重合并与模型导出详解：merge_and_unload使用方法（最终版）

## 参考资料

① 权威：《Hugging Face PEFT LoraModel.merge_and_unload API》  
    网址：https://huggingface.co/docs/peft/package_reference/lora#peft.LoraModel.merge_and_un

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
# QLoRA微调与模型导出详解

## 3.1 方法作用

### PEFT提供：

- **合并LoRA权重**：将LoRA（Low-Rank Adaptation）权重整合到基础模型中。
- **移除PEFT包装**：去除多余的PEFT包装层，优化模型结构。

### 简单理解：

执行流程如下：
1. **基础模型**：加载原始模型。
2. **LoRA Adapter**：加载LoRA权重。
3. **完整模型**：合并权重后，生成带有LoRA更新的完整模型。

---

## 4. 合并后的变化

### 合并前：

模型结构如下：
```
PeftModel
├── Base Model
└── LoRA Adapter
```

### 合并后：

模型结构简化为：
```
普通Transformer模型
```
包含已更新的LoRA权重。

---

## 5. 加载LoRA Adapter

假设：
- 基础模型路径：`base_model_path = "model_path"`
- Adapter路径：`adapter_path = "adapter_path"`

加载操作如下：
```python
from transformers import AutoModelForCausal

## 块 57

```markdown
## 三、merge_and_unload是什么？（重点）

### 3.1 方法作用

`merge_and_unload` 是一个用于处理 LoRA（Low-Rank Adaptation）或 QLoRA（Quantized LoRA）模型的工具，主要用于模型的合并与导出。它通过将模型的参数以更

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

# 11.1 直接合并可能遇到的情况

在进行 4-bit 状态下的模型合并时，可能会遇到以下问题：

- **精度损失**：直接在 4-bit 状态下进行权重合并，可能导致精度下降，因为 4-bit 量化后的权重与全程高精度权重不完全匹配。
- **权重不兼容**：如果所使用的 PEFT 或其他库不支持在量化状态下直接进行 `merge_and_unload` 操作，可能会导致报错。

> **说明**：直接合并可能遇到的问题并非必然发生。如果合并过程不顺利，可能是因为 PEFT、bitsandbytes 或 Transformers 的版本不兼容，或者量化配置与实际操作不一致。

---

# 常见做法：先反量化到高精度再合并

为了避免直接在 4-bit 状态下合并带来的问题，建议采用以下步骤进行模型合并：

1. **加载高精度权重**：在合并过程中，确保基础模型和 Adapter 都是加载到高精度（如 FP16/BF16）状态。
2. **合并操作**：在加载高精度权重的基础上进行 `merge_and_unload` 操作。
3. **保存合并后的模型**：确保合并后的模型和 tokenizer 与训练时保持一致。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 示例代码：直接合并可能导致问题
base_model_path = "your/base-causal-lm"  # 训练 QLoRA 时用的同一个基础模型
adapter_path = "./qlora_adapter"          # 训练保存下来的

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

## 11.3 合并时开启校验并做一致性验证（建议）

在进行 LoRA 模型的合并操作时，校验（Validation）和一致性验证（Consistency Check）是非常重要的步骤。这些步骤可以帮助确保合并操作的正确性，避免潜在的错误对模型性能造成影响。

### 1. 校验（Validation）的重要性
校验操作主要是为了验证模型在合并过程中是否保持了数据的完整性。具体来说，校验可以帮助我们确认以下几点：
- **模型参数的一致性**：合并后的模型参数是否与原始模型参数保持一致。
- **计算图的正确性**：合并操作是否正确地构建了新的计算图，并且没有引入任何错误。
- **模型功能的完整性**：合并后的模型是否完整地保留了原始模型的功能和行为。

在 LoRA 模型的合并过程中，开启校验可以帮助我们及时发现问题，并在出现问题时及时回滚到合并前的状态。这样可以避免因为操作失误而导致模型数据的丢失或模型性能的下降。

### 2. 一致性验证的具体操作
为了验证合并操作的正确性，我们需要进行以下操作：
- **校验模型参数**：比较合并前和合并后的模型参数，确保它们的值在合理范围内。
- **校验模型行为**：通过输入一个测试样本，比较合并前和合并后的模型输出，确保它们的输出结果一致。

在代码实现中，我们可以使用以下方法进行一致性验证：
```python
# 比较模型参数
for name, parameter in merged_model.named_parameters():
    if name in base_model.named_parameters():
        # 比较参数值
        assert torch.allclose(parameter, base_model_dict[name], atol=1e-5), f"参数 {name} 不一致"
    else:
        #

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

> 说明：合并本身是"权重相加"的一次性计算，**不一定必须在 GPU 上完成**——

## 块 65

# QLoRA微调：合并时开启校验并做一致性验证

## 课程目标
掌握如何在QLoRA微调过程中，正确开启校验并进行一致性验证，以确保模型合并过程的顺利进行。

---

## 课程内容

### 1. 理解校验和一致性验证的作用
校验和一致性验证是确保模型合并过程中数据完整性的重要步骤。通过校验，我们可以验证模型参数在合并前后的数据一致性，避免因参数不一致导致的模型性能下降或训练失败。

### 2. 校验的必要性
在合并过程中，尤其是当模型参数数量较大时，参数的顺序和数值可能会发生变化。如果不进行校验，可能导致模型在加载后出现参数不匹配的问题，影响模型的正常运行。

### 3. 如何开启校验
在使用`merge_and_unload`函数时，可以通过以下步骤开启校验：
- 确保模型已加载并正确挂载
- 调用`modelu.merge

## 块 66

### 14.1 合并后部署完整模型

在使用merge_and_unload()方法部署模型时，模型的部署过程较为简单。具体步骤如下：

1. **保存完整模型**：在合并模型后，需要将完整的模型文件进行保存。
2. **推理框架加载模型**：推理框架可以直接加载保存的完整模型，无需额外处理Adapter。

这种方法的优势在于部署过程简单，推理框架可以当作**普通模型**直接加载，无需理解LoRA的相关知识。此外，**推理时不再有Adapter切换成本**，因为模型已经被完整加载。然而，这种方法的缺点是**每个Adapter都需要保存一份完整模型**，这在多任务场景下会导致存储成本较高。

### 14.2 保留Adapter，用支持LoRA的推理框架动态服务

为了优化存储成本，可以采用保留Adapter的方法。具体步骤如下：

1. **保留基础模型**：仅加载一份基础模型。
2. **运行时挂载多个Adapter**：在推理框架中动态挂载多个Adapter（如vLLM），根据请求选择使用哪个Adapter。

这种方法的优势在于：
- **存储成本低**：仅需一份基础模型和多个小体积的Adapter目录。
- **支持热切换**：可以根据请求动态选择使用哪个Adapter，无需为每个任务单独部署一份模型

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

### Q-KB12-ST1203-001
LoraConfig 中用于指定注入 LoRA 层模块名列表的参数是________。

### Q-KB1201-001
LoRA（Low-Rank Adaptation）的核心思想是？

### Q-KB12-ST1200-003
QLoRA 对显存要求相对较低，主要因为它将基座模型以何种精度加载？

### Q-KB12-ST1203-002
LoraConfig 中控制 LoRA 随机失活概率的参数是________。

### Q-KB1201-003
LoRA中，`r`（rank）参数的作用是？

### Q-KB12-ST1205-001
PEFT 模型调用 `merge_and_unload()` 后，会将 LoRA 权重________到基座模型中。

### Q-KB1201-004
使用LoRA微调后的模型，可以将LoRA权重合并（Merge）回基座模型中，合并后推理不需要额外加载LoRA适配器。（ ）

### Q-KB1201-002
QLoRA 相比 LoRA 的主要改进是引入了 **4-bit** 量化技术，进一步降低了微调时的________占用。
