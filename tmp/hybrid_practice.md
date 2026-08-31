## 立即动手 · 显卡监控命令 nvidia-smi 详解 最简操作
#### 本节关键命令 · 你自己的运行记录（先执行、再读正文）
| 序号 | 命令（原文给出） | 你预期的输出（动手前猜） | 实际终端输出 | 是否看懂每个输出字段？ |
|---|---|---|---|---|
| 1 | `nvidia-smi` |  |  | 待填 |
| 2 | `nvidia-smi dmon -s p` |  |  | 待填 |
| 3 | `nvidia-smi -l 1` |  |  | 待填 |
| 4 | `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits` |  |  | 待填 |
---
# 显卡监控命令 nvidia-smi 详解

深度学习训练中，GPU 状态监控是定位问题的第一步。**nvidia-smi** 是 NVIDIA 官方命令行工具，用于查看 GPU 实时状态。

## 基础用法

🛠【实操请暂停 · 第 1 段代码】先实际运行，把输出填回上表，再读后文。

```
nvidia-smi
```

⚙️【字段提示】重点关注下表字段，执行完回头核对你自己的输出值。

会输出一张包含下面 **核心字段** 的总览表：

| 字段 | 含义 |
|---|---|
| Driver Version | 当前已安装的 NVIDIA 驱动版本 |
| CUDA Version | 驱动支持的最高 CUDA 版本 |
| GPU Name | GPU 型号，例如 RTX 4060 / A100 |
| Memory-Usage | 显存占用（已用 / 总量） |
| GPU-Util | GPU 计算核心利用率（0% ~ 100%） |
| Temperature | GPU 温度（摄氏度） |

你的机器是 RTX 4060 8G 显存，重点看 Memory-Usage 的已用/总量是否接近 8G，以及 GPU-Util 是否跑满。

## 常见报错与排查

🐛【报错登记】你本机运行出现下面任一条吗？按文中排查方法先自查一遍再问。

- **`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`**
  - 现象：执行 nvidia-smi 时直接报错，没有任何表格输出。
  - 排查 1：确认你机器上有 NVIDIA 显卡并且驱动已安装。
  - 排查 2：重启系统后再试；很多时候驱动升级后未重启导致与内核模块失联。
  - 排查 3：在 Linux 上执行 `lsmod | grep nvidia`，如果没有输出，说明 nvidia 内核模块没加载。

- **显存 OOM（Out of Memory）**
  - 现象：训练脚本抛出 `RuntimeError: CUDA out of memory. Tried to allocate X GiB`。
  - 排查 1：用 nvidia-smi 看 Memory-Usage，找出占用显存最高的进程 PID（表格最右侧 Process Name 列）。
  - 排查 2：对于不再需要的进程执行 `kill <PID>` 或 `kill -9 <PID>` 释放显存。
  - 排查 3：训练代码层面降低 batch_size，或者启用 mixed precision / gradient checkpointing。

8G 显存跑深度学习很容易触发 OOM，这条对你很实用——先学会用 nvidia-smi 定位占用进程，再决定 kill 还是调参。

## 进阶参数

🛠【实操继续】逐条执行下面三条命令，观察输出差异。

- `nvidia-smi dmon -s p`：持续监控功率 / 温度 / SM 占用等多项指标每秒一行。
- `nvidia-smi -l 1`：每 1 秒刷新一次总览表（等同 `watch -n 1 nvidia-smi`）。
- `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits`：CSV 格式导出，便于写入日志。

第三条命令输出的是纯数字 CSV，适合你在 Python 训练脚本里用 subprocess 调用后写日志——配合你的 Python 技能正好。

### 动手验证（全部在你本机上独立完成，不会写的地方回正文找）
1. 执行与输出核对：独立执行 `nvidia-smi` → 粘贴输出 + 列 3 个实际出现字段 → 回正文找字段解释。
2. 报错对照：若本机运行时出现「NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver」，按文中给的 3 种排查方法先自查并记录动作；无法解决再问。
3. 字段自查：从你输出找 `Memory-Usage` / `GPU-Util` / `Temperature`，分别用 1 句写你这次实际运行里它们的含义。
4. 流程顺序自检：先跑 `nvidia-smi` 看总览，再跑 `nvidia-smi -l 1` 看动态刷新，最后跑 CSV 导出——顺序是否与正文一致？不一致会导致什么后果？

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（掌握 Linux 基础命令）的关系；遇到无法解释的术语时记录下来再提问。