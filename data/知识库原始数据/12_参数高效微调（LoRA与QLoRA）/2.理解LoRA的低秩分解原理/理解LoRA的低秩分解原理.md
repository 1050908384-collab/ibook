# LoRA低秩分解原理详解：理解大模型参数高效微调核心思想（最终版）

## 参考资料

① 权威：《Hugging Face PEFT LoRA Conceptual Guide》
　　网址：https://huggingface.co/docs/peft/main/en/conceptual_guides/lora

② 权威：《LoRA: Low-Rank Adaptation of Large Language Models》
　　网址：https://arxiv.org/abs/2106.09685

③ 底稿：《LoRA低秩适配原理详解》
　　网址：https://blog.csdn.net/2301_76243776/article/details/160592435

④ 参考文章：《LoRA原理深入理解：从矩阵分解到参数高效微调》
　　网址：https://blog.csdn.net/bestpasu/article/details/147881448

⑤ 参考文章：《大模型LoRA微调中的低秩分解详解》
　　网址：https://blog.csdn.net/2601_95388234/article/details/159550526

---

**适合人群**：已经完成 Transformers、Datasets、Trainer、PEFT 环境准备，希望深入理解 LoRA 微调原理的学习者。

本文不会直接进入代码训练，而是重点解释：

* 为什么大模型微调不需要更新全部参数；
* LoRA为什么采用低秩矩阵；
* ΔW = BA到底代表什么；
* rank为什么决定训练参数量；
* LoRA如何插入Transformer模型。

理解这些概念后，你才能真正理解后续：

* LoraConfig参数配置；
* target_modules选择；
* QLoRA量化训练；
* Adapter合并部署。

---

# 一、为什么需要LoRA？（①②③）

## 1.1 大模型微调的问题

现代大语言模型：

例如：

```text id="6rh3a4"
LLaMA

Qwen

Baichuan

Mistral
```

通常拥有：

* 数十亿参数；
* 上百亿参数。

例如：

7B模型：

```text id="b8m8qn"
70亿参数
```

如果采用传统方式：

```text
预训练模型

↓

更新全部参数

↓

保存新模型
```

需要：

* 巨大的GPU显存；
* 大量训练时间；
* 巨大的存储空间。

---

## 1.2 全参数微调的问题

假设模型有：

```text
100亿参数
```

训练时不仅保存：

模型参数：

```text
W
```

还需要保存：

梯度：

```text
∂W
```

优化器状态：

```text
Adam moment
```

实际显存需求远大于模型大小。

---

# 二、LoRA核心思想（①②）

## 2.1 不直接修改原模型参数

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

## 2.2 为什么只训练ΔW？

因为大模型已经通过海量数据学习了：

* 语言规律；
* 世界知识；
* 推理能力。

微调通常只是：

> 对已有能力进行方向调整。

因此不需要重新学习全部参数。

只需要学习：

```text
任务相关变化量
```

即可。

---

# 三、什么是低秩分解？（底稿③）

## 3.1 矩阵基础概念

Transformer中大量参数都是矩阵。

例如：

注意力层：

```python
id="k7s2dq"
Q = XWq
```

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

## 3.2 直接训练矩阵的问题

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

# 四、LoRA低秩分解数学原理（重点）（①②③）

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

## 4.1 矩阵尺寸变化

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

# 五、为什么低秩分解有效？（参考文章④⑤）

## 5.1 神经网络参数变化具有低维特征

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

真正影响任务的方向可能只有：

```text
几十维
```

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

# 六、LoRA在Transformer中的位置（①③）

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

LoRA通常加入：

```text
Q投影

V投影
```

例如：

```python
id="u8xq6h"
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

---

# 七、LoRA训练过程（底稿③）

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

训练时：

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

最终：

模型：

```text
原模型

+

LoRA Adapter
```

---

# 八、LoRA核心参数理解（参考文章④⑤）

## 8.1 rank（r）

参数：

```python
id="u0dt9w"
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

常见：

| 任务   | rank |
| ---- | ---- |
| 简单任务 | 4-8  |
| 普通微调 | 8-32 |
| 复杂任务 | 64+  |

---

## 8.2 lora_alpha

例如：

```python
id="8py9jk"
lora_alpha=16
```

用于控制：

LoRA更新强度。

通常：

```text
alpha/r
```

作为缩放比例。

---

## 8.3 lora_dropout

例如：

```python
id="3yq6cw"
lora_dropout=0.05
```

作用：

防止LoRA参数过拟合。

---

# 九、LoRA与全参数微调区别（①②）

| 项目   | 全参数微调 | LoRA      |
| ---- | ----- | --------- |
| 训练参数 | 全部    | 少量        |
| 显存   | 高     | 低         |
| 速度   | 慢     | 快         |
| 保存文件 | GB级   | MB级       |
| 部署   | 复制模型  | 加载Adapter |

---

# 十、LoRA与QLoRA关系

LoRA：

```text
FP16模型

+

LoRA参数
```

QLoRA：

```text
4bit量化模型

+

LoRA参数
```

区别：

| 项目     | LoRA      | QLoRA             |
| ------ | --------- | ----------------- |
| 基础模型精度 | FP16/BF16 | 4bit              |
| 显存     | 较低        | 更低                |
| 依赖     | PEFT      | PEFT+bitsandbytes |
| 适合     | GPU训练     | 消费级GPU训练          |

---

# 十一、电脑配置批注

本章节主要讲解数学原理。

不需要GPU。

最低：

* CPU：

  * 任意现代CPU
* 内存：

  * 4GB+

说明：

矩阵公式理解不涉及模型运行。

---

如果进行后续LoRA训练：

最低：

* GPU：

  * NVIDIA CUDA
* 显存：

  * 8GB+

推荐：

* 显存：

  * 16GB+

QLoRA：

* 显存：

  * 因 4-bit 量化基础模型，同规模下通常比 LoRA 需要的显存更低；7B 级别常见参考下限约 8GB 起，具体以实际运行为准

---

# 十二、总结

完成本章后，你已经理解：

✅ 为什么需要LoRA
✅ 全参数微调的问题
✅ W + ΔW思想
✅ 低秩矩阵分解
✅ ΔW = BA公式
✅ rank参数含义
✅ LoRA插入Transformer位置
✅ LoRA与QLoRA区别

完整LoRA学习路线：

```text
Transformer模型

↓

冻结原参数

↓

插入LoRA矩阵

↓

训练低秩参数

↓

保存Adapter

↓

结合基础模型推理
```

下一阶段：

**LoRA实战：使用PEFT + Trainer完成大模型微调**

将进入代码实践阶段。
