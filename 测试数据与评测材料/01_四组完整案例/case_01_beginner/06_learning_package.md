# 学习包 - case_01_beginner（案例A：大模型方向初学者）

## 环境冻结
- 生成模型：deepseek-chat（DeepSeek 云端主模型）
- Embedding 模型：qwen3-embedding:0.6b
- 知识库版本：kb-2026-08-24
- Prompt 版本：v3.4
- Skill 版本：technical_review v3.5 / content_generation v3.4

## 阶段1：深度学习框架PyTorch基础（KB-04，初级）

> 学习者无深度学习经验，教程解释更基础；理论结合简单实操风格。

### 内容块 block_s1_01：张量基础与自动求导

张量（Tensor）是 PyTorch 的核心数据结构，可类比为带 GPU 加速的多维数组。

```python
import torch
# 创建张量
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = (x ** 2).sum()
y.backward()  # 自动求导
print(x.grad)  # 输出梯度 2x
```

### 内容块 block_s1_02：训练循环（初稿→审核发现问题→修订）

初稿曾给出错误的 `optimizer.zero_grad()` 位置（放在 `loss.backward()` 之后），经专业知识与教学质量审核 Agent 发现 `gradient_flow_error`，个性化学习资源生成 Agent 修订后复核通过。

修订后正确写法：

```python
for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()        # 必须在 backward 前
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

### 来源说明
- 知识库：04_深度学习框架PyTorch基础/7.训练循环编写.md
- 生成模型：deepseek-chat

## 阶段2：Transformer架构（KB-08，中级，阶段1低分降维后）

### 内容块 block_s2_01：Self-Attention 直观理解

阶段1测试正确率 0.52（低分），学情评估与动态决策 Agent 决策 `simplify`，阶段2拆分为更小颗粒并补充图示。

（阶段2学习包内容略，保存为可执行状态）

## 资源与视频
- 推荐视频：已通过 `learning_resource_tool.validate_recommended_videos` 校验，URL 与内容匹配
