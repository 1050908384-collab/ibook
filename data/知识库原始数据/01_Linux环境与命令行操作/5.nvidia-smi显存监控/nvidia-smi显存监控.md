# nvidia-smi 显存监控

① 底稿：《nvidia-smi 用法详解》  
　　网址：https://blog.csdn.net/qq_36803941/article/details/142182349

② 参考文章：《实时查看 GPU 显存占用、功耗、进程状态》  
　　网址：https://blog.csdn.net/qq_36693723/article/details/142974892

③ 参考文章：《【监控 GPU】监控 NVIDIA GPU 显卡占用状态的常用命令》  
　　网址：https://blog.csdn.net/Ever_____/article/details/136973595

④ 参考文章：《nvidia-smi 命令实战：如何像老司机一样监控 GPU 状态》  
　　网址：https://blog.csdn.net/weixin_29305313/article/details/159296452

> **适合人群**：已完成前三节，掌握 conda/pip 并能安装 Python 包的初学者。本节时长约 0.5 小时，目标是掌握 GPU 显存监控的核心命令，能在模型训练时快速排查 OOM、查看进程占用、释放显存资源。（①+④）

---

## 一、nvidia-smi 是什么？（①+④）

`nvidia-smi` = **NVIDIA System Management Interface**，是 NVIDIA 官方提供的命令行工具，用来监控和管理 NVIDIA GPU 的状态。（①）

在大模型训练和推理中，你几乎每天都会用到它，核心用途包括：（④）

| 用途 | 说明 |
|---|---|
| **查看显存占用** | 当前用了多少显存、还剩多少，预防 OOM（Out Of Memory） |
| **查看 GPU 利用率** | 判断 GPU 是否在干活，还是空转 |
| **查看温度与功耗** | 防止硬件过热，监控能耗 |
| **查看进程绑定** | 谁占了 GPU？哪个 PID？占了多少显存？ |
| **释放显存** | 找到僵尸进程并杀掉，回收显存 |

> **前提条件**：`nvidia-smi` 只有在安装了 NVIDIA GPU 驱动后才能使用。如果你的机器没有 NVIDIA 显卡（如纯 CPU 环境、Mac、部分云服务器），命令会提示找不到。**但 CPU 推理和不带 GPU 的基础学习仍可进行**，本 GPU 实训与高性能 vLLM 部署需要 NVIDIA GPU 环境。（①+④）

---

## 二、基础用法：单次查看 GPU 状态（①+④）

### 2.1 最简命令（①）

```bash
nvidia-smi
```

执行后会输出一个表格，类似这样：（④）

```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:1A:00.0 Off |                    0 |
| N/A   33C    P0    24W / 250W |    128MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 2.2 关键指标解读（①+④）

| 区域 | 字段 | 含义 |
|---|---|---|
| **顶部信息** | `NVIDIA-SMI 525.85.12` | nvidia-smi 工具版本 |
| | `Driver Version: 525.85.12` | NVIDIA 显卡驱动版本 |
| | `CUDA Version: 12.0` | 当前驱动支持的最高 CUDA 版本 |
| **GPU 列表** | `GPU 0` | GPU 编号（多卡时会有 0, 1, 2...）|
| | `Name` | GPU 型号（如 Tesla V100、RTX 4090）|
| | `Temp` | 温度（摄氏度）|
| | `Pwr:Usage/Cap` | 当前功耗 / 最大功耗（如 24W / 250W）|
| | `Memory-Usage` | **显存占用 / 总显存**（如 128MiB / 32510MiB）|
| | `GPU-Util` | **GPU 计算利用率**（0% 表示空闲，90%+ 表示满载）|
| **进程区** | `PID` | 进程 ID |
| | `Process name` | 进程名（如 python、python3）|
| | `GPU Memory Usage` | 该进程占用的显存量 |

> **对大模型开发最重要的两个数字**：`Memory-Usage`（显存占用）和 `GPU-Util`（利用率）。训练时如果 Memory-Usage 接近上限，就要调小 batch size；如果 GPU-Util 长期很低，说明数据加载太慢，GPU 在空等。（④）

---

## 三、实时动态监控（①+②+③）

单次执行只能看"快照"，训练时需要持续观察。以下是三种实时方案：（②+③）

### 3.1 方案 A：watch + nvidia-smi（系统自带，无需安装）（①+②）

Linux 自带的 `watch` 命令可以周期性执行任意命令，配合 `nvidia-smi` 实现实时刷新：（②+③）

```bash
# 每 1 秒刷新一次
watch -n 1 nvidia-smi

# 每 0.5 秒刷新一次（更实时）
watch -n 0.5 nvidia-smi
```

- 终端会全屏显示，每秒自动更新
- 按 `Ctrl + C` 退出

> **注意**：`watch` 命令在 Linux/macOS 下可用，Windows 命令提示符不支持。Windows 用户可用 `nvidia-smi -l 1` 代替（体验稍差，每次打印新表格）。（③）

### 3.2 方案 B：nvitop（推荐，可视化体验最佳）（②+③）

`nvitop` 是一个基于 Python 的 GPU 实时监控工具，界面类似 `top`，有颜色区分、进度条、历史曲线，体验远胜原生 `nvidia-smi`。（②+③）

```bash
# 安装（上节课学过的 pip！）
pip install nvitop

# 运行
nvitop
```

界面特点：（②）
- 彩色高亮：显存占用高时变红，空闲时变绿
- 进程列表：显示每个进程的 CPU%、MEM%、运行时间
- 历史曲线：右上角有 GPU 利用率和显存占用的趋势图
- 按 `h` 查看帮助，`q` 退出

> **强烈建议**：在服务器上训练模型时，开两个终端——一个跑训练脚本，一个跑 `nvitop`，实时监控。

### 3.3 方案 C：gpustat（轻量简洁）（②）

如果你只需要快速看一眼所有卡的显存占用，不需要太多细节：（②）

```bash
# 安装
pip install gpustat

# 运行
gpustat

# 实时刷新
watch -n 1 gpustat
```

输出示例：（②）

```bash
c01n04  Wed Oct 16 10:55:19 2024  455.23.05
[0] Tesla V100-PCIE-32GB | 33°C,  0 % |  0 / 32510 MB |
[1] Tesla V100-PCIE-32GB | 30°C,  0 % |  0 / 32510 MB |
```

> **三种方案对比**：

| 方案 | 安装成本 | 可视化 | 适用场景 |
|---|---|---|---|
| `watch -n 1 nvidia-smi` | 无需安装 | 一般 | 临时查看、无 pip 权限的服务器 |
| `nvitop` | `pip install nvitop` | 优秀 | **日常主力推荐** |
| `gpustat` | `pip install gpustat` | 简洁 | 快速扫一眼多卡状态 |

---

## 四、进阶查询：精准获取信息（①+④）

### 4.1 查看显存详情（①）

```bash
# 只查看显存相关信息
nvidia-smi -q -d MEMORY
```

输出包括：总显存、已用显存、空闲显存、显存类型（GDDR6/GDDR6X/HBM2 等）。（①）

### 4.2 精准查询特定字段（①+④）

使用 `--query-gpu` 可以只提取你关心的字段，配合 `--format=csv` 输出为表格，方便脚本处理：（①+④）

```bash
# 查询 GPU 名称、温度、显存占用、利用率
nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv

# 输出示例：
# name, temperature.gpu, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
# Tesla V100-PCIE-32GB, 33, 128 MiB, 32510 MiB, 0 %
```

常用查询字段：（①+④）

| 字段名 | 说明 |
|---|---|
| `name` | GPU 型号 |
| `temperature.gpu` | GPU 温度 |
| `memory.used` | 已用显存 |
| `memory.total` | 总显存 |
| `utilization.gpu` | GPU 利用率 |
| `power.draw` | 当前功耗 |
| `power.limit` | 功耗上限 |
| `pcie.link.gen.current` | 当前 PCIe 版本 |
| `ecc.mode.current` | ECC 内存模式 |

### 4.3 限制 GPU 功率（高级）（①）

在某些场景下（如实验室限电、散热不足），可以限制 GPU 的最大功耗：（①）

```bash
# 查看当前功率限制
nvidia-smi -q -d POWER

# 设置 GPU 0 的功率上限为 200W（需要 sudo）
sudo nvidia-smi -pl 200
```

> **注意**：调低功率会降低性能，仅在特殊场景使用。且部分云服务器不支持此操作。（①）

---

## 五、进程管理与显存释放（①+④）

### 5.1 查看 GPU 进程（①+④）

`nvidia-smi` 默认输出底部的 **Processes** 区域会列出占用 GPU 的进程：（④）

```bash
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      12345    C   python3                         10240MiB |
+-----------------------------------------------------------------------------+
```

字段说明：（④）
- `PID`：进程 ID，后续杀进程要用
- `Type`：`C` = Compute（计算进程，如 PyTorch 训练），`G` = Graphics（图形进程）
- `Process name`：进程名
- `GPU Memory Usage`：该进程占用的显存

### 5.2 杀死僵尸进程，释放显存（①+④）

训练脚本崩溃后，有时进程没有彻底退出，继续占用显存（俗称"僵尸进程"）。这时需要手动杀掉：（①+④）

```bash
# 1. 先用 nvidia-smi 找到 PID（比如 12345）
nvidia-smi

# 2. 杀死进程
kill 12345

# 3. 如果进程顽固不死，强制终止
kill -9 12345
```

> **安全提醒**：`kill -9` 是强制终止，进程会立刻消失，未保存的数据会丢失。优先用 `kill PID`，不行再用 `kill -9`。（④）

### 5.3 OOM 预防思路（④）

OOM（Out Of Memory，显存溢出）是大模型训练中最常见的报错之一。通过 `nvidia-smi` 可以提前发现风险：（④）

1. **训练前**：`nvidia-smi` 查看空闲显存，估算 batch size
2. **训练中**：用 `watch -n 1 nvidia-smi` 或 `nvitop` 观察显存增长趋势
3. **OOM 后**：检查是否有僵尸进程占用显存，有则 `kill` 释放

---

## 六、日志记录与自动化（①+④）

### 6.1 导出监控日志（①）

```bash
# 每 2 秒采样一次，输出到 gpu_log.txt，按 Ctrl+C 停止
nvidia-smi -l 2 -f gpu_log.txt

# 或者用 query 模式导出 CSV（更适合分析）
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 -f gpu_log.csv
```

### 6.2 后台持续监控（④）

```bash
# 放到后台运行，即使关闭终端也继续记录
nohup nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 -f ~/gpu_monitor.csv > /dev/null 2>&1 &
```

> **用途**：长时间训练时，后台记录 GPU 状态，事后分析训练期间的显存波动和利用率曲线。（④）

---

## 七、常见问题排查（④）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `nvidia-smi: command not found` | 未安装 NVIDIA 驱动 | 安装驱动，或确认机器是否有 NVIDIA 显卡（④） |
| 显示 `No running processes found` | GPU 空闲，没有程序在跑 | 正常状态，开始训练后会显示进程 |
| 显存占用高但 GPU-Util 为 0% | 可能是采样时机、同步等待、训练流程暂停，或 DataLoader 瓶颈 | 持续观察 `nvitop` 确认趋势；如长期为 0%，可尝试增加 DataLoader 的 `num_workers`，或用 SSD 加速数据读取（④） |
| 训练报错 CUDA out of memory | 显存不足 | `nvidia-smi` 查看剩余显存，减小 batch size 或模型规模 |
| 杀死进程后显存未释放 | 有多个关联进程 | `nvidia-smi` 再次确认 PID，全部杀掉 |
| `nvidia-smi` 显示驱动版本与 CUDA 版本不匹配 | 驱动太旧 | 升级 NVIDIA 驱动到与目标 CUDA 版本兼容的版本 |

---

## 八、总结与后续学习路线（①+②+③+④）

恭喜你完成了 nvidia-smi 显存监控章节的学习！你现在应该掌握了：

- ✅ `nvidia-smi` 基础输出解读（驱动版本、CUDA 版本、显存、利用率、温度、功耗）
- ✅ 三种实时动态监控方案（`watch + nvidia-smi`、`nvitop`、`gpustat`）
- ✅ 进阶精准查询（`--query-gpu`、`--format=csv`、`-q -d MEMORY`）
- ✅ 进程管理（查看 PID、杀死僵尸进程、释放显存）
- ✅ OOM 预防与排查思路
- ✅ 日志导出与后台自动化监控

> **下一步预告**：在下一节 **"实际操作与练习"** 中，我们将综合运用 Linux 环境准备、Shell 命令、环境变量、conda 虚拟环境和 nvidia-smi 监控的全部知识，完成一个综合实战任务。随后我们将正式进入 **"Python 编程基础"** 板块，开始编写代码！现在，如果你的机器有 NVIDIA GPU，打开终端输入 `nvidia-smi`，看看你的显卡状态吧！