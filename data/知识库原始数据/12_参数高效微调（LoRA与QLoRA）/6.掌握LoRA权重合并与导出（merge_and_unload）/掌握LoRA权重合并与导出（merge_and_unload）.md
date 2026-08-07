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

# 一、LoRA训练后保存的是什么？（①②）

## 1.1 LoRA不是直接修改基础模型

LoRA训练过程中：

基础模型：

```text id="8f3m2a"
Base Model

冻结
```

LoRA参数：

```text id="p7k2cz"
Adapter

训练
```

最终结构：

```text id="j9x4qp"
基础模型

+

LoRA Adapter
```

---

例如：

训练：

```text id="5r2v8n"
Llama-7B

+

LoRA Adapter
```

保存后：

通常只有：

```text id="m8k1zs"
adapter_model.safetensors

adapter_config.json
```

而不是完整模型。

---

## 1.2 Adapter有什么特点？

优点：

* 文件小；
* 保存快；
* 可以多个任务共享一个基础模型。

例如：

同一个基础模型：

```text id="x7p4dq"
Base Model
```

可以搭配：

```text id="v2m8ls"
客服Adapter

医疗Adapter

法律Adapter
```

---

缺点：

部署时需要：

```text id="k6q2mw"
基础模型

+

Adapter
```

两个部分同时存在。

---

# 二、为什么需要合并LoRA权重？（②③）

## 2.1 推理时的问题

未合并：

```text id="b9x3ra"
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

```text id="t5w8cx"
输入

↓

完整模型

↓

输出
```

因此需要：

合并权重。

---

## 2.2 LoRA合并数学原理（②）

LoRA训练时：

公式：

```text id="e4s9kp"
W' = W + ΔW
```

其中：

```text id="w3p7mv"
ΔW = BA
```

合并：

就是将：

```text id="n5z8hj"
BA
```

加入：

```text id="q2x6ds"
W
```

得到：

```text id="a8m4yr"
W'
```

---

合并后：

原来的：

```text id="d7p3zs"
W

+

Adapter
```

变成：

```text id="u9k2hf"
W'
```

---

# 三、merge_and_unload是什么？（重点）（①②）

## 3.1 方法作用

PEFT提供：

```python id="w5n8qp"
merge_and_unload()
```

作用：

> 将LoRA权重合并到基础模型，并移除PEFT包装。

简单理解：

执行：

```text id="r4k8mx"
Base Model

+

LoRA Adapter

↓

完整模型
```

---

## 3.2 合并后的变化

合并前：

```text id="c8v5jx"
PeftModel

├── Base Model

└── LoRA Adapter
```

---

合并后：

```text id="q6m3hz"
普通Transformer模型

包含LoRA更新后的权重
```

---

# 四、LoRA模型合并完整流程（重点）（②③）

整体流程：

```text id="h4z8mq"
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

# 五、加载LoRA Adapter（②）

假设：

基础模型：

```python id="x5m9qb"
base_model_path = "model_path"
```

Adapter：

```python id="z8k3vc"
adapter_path = "adapter_path"
```

加载：

```python id="m4s7yn"
from transformers import AutoModelForCausalLM
from peft import PeftModel


model = AutoModelForCausalLM.from_pretrained(
    base_model_path
)


model = PeftModel.from_pretrained(
    model,
    adapter_path
)
```

此时：

模型结构：

```text id="v6p3kj"
Base Model

+

LoRA Adapter
```

---

# 六、使用merge_and_unload合并权重（重点）（①②）

执行：

```python id="z3h7mq"
merged_model = model.merge_and_unload()
```

完成：

```text id="m8v2sd"
W

+

BA

↓

W'
```

---

此时：

`merged_model`

已经不是：

```text id="1k9xpw"
PeftModel
```

而是：

```text id="r7n3mv"
普通Transformer模型
```

---

# 七、导出合并后的模型（②④）

合并后：

保存：

```python id="c5j8nx"
merged_model.save_pretrained(
    "./merged_model"
)
```

同时保存Tokenizer：

```python id="h8q2mv"
tokenizer.save_pretrained(
    "./merged_model"
)
```

目录：

```text id="y4p7ks"
merged_model/

├── config.json

├── model.safetensors

├── tokenizer.json

└── tokenizer_config.json
```

---

# 八、合并后如何加载推理？（③④）

合并后：

不需要：

```text id="f9k4zp"
peft
```

只需要：

```python id="b2m7qx"
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "./merged_model"
)
```

流程：

```text id="q5m8vx"
用户输入

↓

完整模型

↓

生成结果
```

---

# 九、LoRA合并与不合并区别（②③）

| 项目     | 不合并          | 合并             |
| ------ | ------------ | -------------- |
| 模型结构   | Base+Adapter | 完整模型           |
| 文件大小   | 小            | 大              |
| 部署复杂度  | 较高           | 简单             |
| 多个任务切换 | 方便           | 不方便            |
| 推理加载   | 需要PEFT       | 直接Transformers |

---

# 十、什么时候应该合并？（③④）

## 10.1 推荐合并场景

### 场景1：上线部署

例如：

生产环境：

```text id="m7x4ps"
服务器

↓

直接加载模型

↓

提供API
```

推荐合并。

---

### 场景2：分享模型

如果发布：

完整模型：

推荐：

```text id="x8q2mk"
merged_model
```

用户无需额外下载Adapter。

---

## 10.2 不建议合并场景

### 场景1：多个任务Adapter

例如：

一个基础模型：

```text id="a9v5mz"
Base Model
```

多个：

```text id="d2k8qx"
Adapter A

Adapter B

Adapter C
```

保留Adapter更灵活。

---

### 场景2：持续训练

如果继续调整LoRA：

保留Adapter。

---

# 十一、QLoRA模型合并注意事项（重点）（③）

QLoRA：

基础模型：

```text id="z5m7qx"
4-bit量化
```

LoRA：

```text id="v8n3kc"
FP16/BF16
```

直接合并可能出现：

* 精度问题；
* 权重类型不匹配。

---

通常流程：

```text id="s6p2mq"
4-bit模型

↓

转换到FP16/BF16

↓

加载LoRA

↓

merge

↓

保存
```

---

# 十二、电脑配置批注

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

GPU推荐：

7B模型：

最低：

* 16GB显存

推荐：

* 24GB显存+

原因：

合并阶段需要同时保存：

* 基础模型；
* LoRA参数；
* 合并后权重。

---

# 十三、常见问题排查

| 问题                  | 原因           | 解决                 |
| ------------------- | ------------ | ------------------ |
| merge_and_unload不存在 | 未使用PeftModel | 检查加载方式             |
| 合并后模型变大             | 正常现象         | 完整模型包含全部参数         |
| 显存不足                | 加载FP16模型     | 使用CPU offload或增加显存 |
| Tokenizer缺失         | 未保存Tokenizer | 重新save_pretrained  |
| QLoRA合并失败           | 4bit权重限制     | 先转换精度              |

---

# 十四、总结

完成本章后，你已经掌握：

✅ LoRA Adapter保存方式
✅ 为什么需要权重合并
✅ merge_and_unload作用
✅ LoRA数学合并过程
✅ 完整模型导出流程
✅ 合并模型独立部署方式
✅ LoRA与QLoRA合并注意事项

核心流程：

```text id="m3q8zs"
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
