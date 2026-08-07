# QLoRA原理详解：4-bit量化（NF4、双重量化）与分页优化器机制（最终版）

## 参考资料

① 权威：《QLoRA: Efficient Finetuning of Quantized LLMs》
　　网址：https://arxiv.org/abs/2305.14314

② 权威：《Hugging Face GPU Performance Optimization》
　　网址：https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one

③ 权威：《Hugging Face bitsandbytes量化文档》
　　网址：https://huggingface.co/docs/transformers/quantization/bitsandbytes

④ 底稿：《QLoRA原理详解：NF4量化、双重量化与Paged Optimizer》
　　网址：https://blog.csdn.net/weixin_44081096/article/details/163221776

⑤ 参考文章：《QLoRA低显存微调技术详解》
　　网址：https://blog.csdn.net/l35633/article/details/161399952

⑥ 参考文章：《大模型QLoRA量化训练原理分析》
　　网址：https://blog.csdn.net/l35633/article/details/161148157

---

**适合人群**：已经学习 LoRA 原理，并完成 PEFT、bitsandbytes 环境准备，希望理解 QLoRA 为什么可以在消费级GPU上微调大模型的学习者。

本文重点解释：

* QLoRA相比LoRA做了什么优化；
* 为什么使用4-bit量化；
* NF4为什么适合大模型权重；
* 双重量化如何进一步降低显存；
* 分页优化器如何解决显存峰值问题。

---

# 一、为什么需要QLoRA？（①④）

## 1.1 LoRA仍然需要加载完整模型

LoRA虽然减少训练参数：

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

## 1.2 QLoRA核心思想

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

# 二、QLoRA整体结构（①④）

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

# 三、为什么需要4-bit量化？（①③④）

## 3.1 模型存储方式

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

---

## 3.2 为什么不用普通INT4？

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

# 四、NF4（NormalFloat4）量化原理（重点）（①④⑤）

## 4.1 NF4是什么？

NF4：

NormalFloat4

中文：

正态分布4位浮点。

QLoRA论文提出：

> 针对神经网络权重正态分布特点设计的4-bit数据类型。

---

## 4.2 为什么适合模型权重？

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

## 4.3 NF4量化过程

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

# 五、双重量化（Double Quantization）（重点）（①④⑥）

## 5.1 为什么需要双重量化？

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

## 5.2 双重量化思想

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

## 5.3 双重量化优势

减少：

* 额外量化参数；
* 显存占用。

尤其：

大模型：

参数数量巨大。

少量优化：

也会节省大量显存。

---

# 六、分页优化器（Paged Optimizers）原理（重点）（①④⑤）

## 6.1 为什么需要分页优化器？

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

## 6.2 分页优化器思想

Paged Optimizer借鉴：

操作系统虚拟内存。

核心：

> 将部分优化器状态放入CPU内存，需要时交换到GPU。

流程：

```text
GPU显存

↓

不足

↓

部分状态移动到CPU

↓

需要训练时加载
```

---

## 6.3 类比理解

普通方式：

```text
所有东西放桌面
```

桌面满：

无法继续。

分页优化器：

```text
桌面放常用文件

柜子保存暂不用文件

需要时取出
```

---

# 七、QLoRA三大显存优化技术总结（①）

QLoRA能够低显存训练，依靠三个核心：

## 7.1 4-bit量化

作用：

减少模型权重大小。

```text
FP16

↓

NF4 4-bit
```

---

## 7.2 双重量化

作用：

减少量化参数占用。

```text
scale

↓

再次量化
```

---

## 7.3 分页优化器

作用：

降低训练峰值显存。

```text
GPU

+

CPU内存交换
```

---

# 八、QLoRA与LoRA区别（④⑤）

| 项目   | LoRA      | QLoRA             |
| ---- | --------- | ----------------- |
| 基础模型 | FP16/BF16 | 4-bit             |
| 训练参数 | LoRA      | LoRA              |
| 量化   | 无         | NF4               |
| 显存   | 较低        | 更低                |
| 依赖   | PEFT      | PEFT+bitsandbytes |
| 适合   | 中高显存GPU   | 消费级GPU            |

---

# 九、bitsandbytes中的QLoRA配置（③）

实际使用：

```python
from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

参数：

## load_in_4bit

开启：

4-bit加载。

---

## bnb_4bit_quant_type

设置：

```python
"nf4"
```

使用：

NF4量化。

---

## bnb_4bit_use_double_quant

开启：

双重量化。

---

# 十、电脑配置批注

本章节包含QLoRA运行相关内容。

如果只学习原理：

无需GPU。

---

如果实际运行QLoRA：

最低：

CPU：

* 4核+

内存：

* 16GB+

GPU：

* NVIDIA CUDA GPU

显存：

QLoRA 的设计目标就是在更小的显存上微调大模型，实际可运行的下限取决于模型规模、序列长度与 batch 大小。下面为 7B 级别的**经验参考**（更短序列/更小 batch 时可更低），以实际运行为准：

参考下限：

* 约 8~12GB+

推荐：

* 16GB以上。

---

常见情况：

| 显存    | 适合           |
| ----- | ------------ |
| 8GB   | 小模型LoRA      |
| 12GB  | 7B QLoRA     |
| 16GB+ | 7B/13B QLoRA |
| 24GB+ | 更大模型         |

---

# 十一、总结

完成本章后，你已经理解：

✅ QLoRA为什么比LoRA更省显存
✅ 4-bit量化原理
✅ NF4为什么适合LLM权重
✅ 双重量化如何节省额外空间
✅ 分页优化器如何降低显存峰值
✅ bitsandbytes QLoRA配置方式

QLoRA核心公式：

```text
4-bit基础模型

+

LoRA Adapter

+

显存优化技术

=

低成本大模型微调
```

下一阶段：

**使用QLoRA + PEFT + Trainer完成低显存大模型微调实战**
