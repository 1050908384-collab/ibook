# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

你已经掌握了 Transformer 架构和 LLM 三阶段训练流程，现在要进入 LoRA 微调实战。但在动手之前，先搞清楚一个底层问题：**Python 解释器到底是什么**？

**核心类比**：你写的 Python 代码是人类可读的文本，但你的 RTX 4060 只认 0 和 1。Python 解释器就是中间的"翻译官"——它把 `.py` 文件里的代码翻译成底层指令，并通过 PyTorch 等库调用 CUDA 驱动 GPU 执行矩阵运算。没有它，你写的 LoRA 训练脚本只是一堆文本文件。

**关键认知**：当你后续运行 `transformers` 库进行 LoRA 微调时，你写的 `trainer.train()` 这行代码通过 Python 解释器执行，并由 PyTorch 等框架在内部调用 CUDA 指令驱动 GPU 完成矩阵运算。理解这层"翻译"关系，你就能明白为什么环境配置是微调的第一步——没有解释器，整个工具链都无法启动。

**实操验证**：安装完成后，在终端输入 `python`，看到版本号提示符（如 `>>>`）即表示

## 块 2

**IDE 与虚拟环境：为 LoRA 微调搭好工作台**

你已掌握 Transformer 与三阶段训练，现在要动手 LoRA 微调。但先别急着写代码——工具没备好，后面每一步都会卡壳。

**IDE（集成开发环境）** 就是你的“工作台”。记事本也能写代码，但效率极低；IDE 提供语法高亮、自动补全、运行调试，让你写代码像用 Word 写文档一样舒服。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个顺手的即可。

**虚拟环境** 是更关键的一环。微调 LoRA 时，你会同时装 PyTorch、Transformers 等多个库，版本稍有不慎就冲突报错。虚拟环境为每个项目创建独立的 Python 空间，互不影响。本文用 **conda** 管理，这是目前最主流的方案。

**Python 版本**

## 块 3

**Windows 安装 Python：为 LoRA 微调铺路**

你的目标是 LoRA 微调，而微调依赖 PyTorch 等工具链。这些工具都通过 `pip` 安装，所以本步的核心是：**装好 Python 并确保 `pip` 可用**。

**第一步：下载安装包**

访问 https://www.python.org，鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。若不勾选，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令行操作（包括安装 PyTorch）都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户若改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。

## 块 4

你的目标是 LoRA 微调，但在此之前，得先把“厨房”搭好。macOS 上装 Python 是第一步，别让环境问题消耗你研究 RLHF 的精力。

**核心原则**：别碰系统自带的 Python。现代 macOS（Catalina 10.15 起）默认不预装 Python，但安装 Xcode Command Line Tools 后可能提供 `/usr/bin/python3` 供系统脚本使用。直接依赖它可能版本过旧，建议自行安装最新版 Python。

**方法一：官方安装包（推荐）**

1. 访问 python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 “Continue” → “Install” 完成。
3. 打开终端，验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：现代 macOS 上，命令行默认**没有** `python` 命令（早期指向 Python 2，已移除）。系统与官方包统一用 `python3`。请始终用 `python3` / `pip3`，避免“命令不存在”或误用系统版本。

**方法二：Homebrew（若已安装）**

终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本

## 块 5

**Linux 安装 Python 3.10+（Ubuntu/Debian）**

你的目标是LoRA微调，但微调前得先把“工作台”搭好。Linux通常自带Python，但版本可能偏老，直接装新版本最稳妥。

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**关键点**：`-y` 自动确认安装，免去手动输入；`python3-venv` 是虚拟环境模块，提前装好，避免后续补装打断思路。

**本块学习检查**：执行上述命令后

## 块 6

### 环境准备：搭建你的 Python 开发工作台 > 第二章：安装 Python 3.10+ > 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，且已掌握 Transformer 架构与 LLM 三阶段训练流程，因此环境选择应直接服务于后续的 GPU 实训。以下是按你的场景拆解的要点：

**1. 日常写 Python、练命令行（Windows 用户）**
启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它提供接近真实生产的 Linux 环境，命令、路径、包管理均与服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**2. 日常写 Python（macOS / Linux 用户）**
直接使用本机终端，本身就是类 Unix 环境，无需额外配置。

**3. LoRA / QLoRA 微调等 GPU 实训（你的核心场景）**
应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"

## 块 7

**镜像源：为你的 LoRA 实验加速**

你的目标是跑通 LoRA 微调，这意味着要频繁安装 `transformers`、`peft` 等库。默认的 `pip` 从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，应始终作为首选。

**但何时需要镜像？** 仅当你网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像只是“网络不佳时的加速手段”，**不必永久替换默认设置**。

**推荐做法：临时指定，不改全局**。若某次安装慢，只对本次安装加 `-i` 参数：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样不影响平时默认行为，网络恢复或镜像出问题时也无需清理配置。

**本块学习检查**：请用临时镜像方式安装 `peft` 库

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，而微调过程需要频繁下载预训练模型和依赖包。如果你的网络环境长期受限，每次都用 `-i` 参数指定镜像确实繁琐。本节教你将清华镜像写入 pip 配置，作为默认源——但请记住，**官方 PyPI 仍是最可靠的来源**，网络恢复后建议改回或删除该配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开 `C:\Users\你的用户名\AppData\Roaming` 文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**此配置。只有企业内网自建源、或确认来源可信且遇到证书报错时才临时加。

**macOS / Linux 用户：**

1. 终端执行：`mkdir -p ~/.config/pip`
2. 创建配置文件：`nano ~/.config/pip/pip.conf`
3. 写入同样的内容（见上方 ini 代码块）。
4. `Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即配置成功。

### 3.5 改回官方源

删除上面创建的 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：在终端执行 `pip config list`，确认输出中包含 `global.index-url` 且指向

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要频繁编辑 Python 脚本和配置文件，它都能胜任。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2.

## 块 10

**PyCharm Community：为LoRA微调准备的“专业工作台”**

你正朝着LoRA微调前进，而PyCharm Community正是支撑你后续编写、调试训练脚本的可靠基地。它免费、专业，且“开箱即用”——内置代码分析、版本控制等能力，能帮你减少环境层面的干扰，把精力集中在模型本身。

**安装要点（三步走）：**
1. 访问 JetBrains 官网下载页，选择 **Community Edition**（社区版）。
2. 运行安装程序时，务必勾选 `Add "Open Folder as Project"` 和 `Add to PATH`。前者让你能右键快速打开项目文件夹，后者便于在命令行直接启动。
3. 首次启动时，选择 `New Project`，在 `Location` 处指定项目路径——**切记路径不要包含中文**，避免后续工具链解析出错。

**关键一步：关联解释器**
在 `Interpreter` 处，选择 `Previously configured interpreter`，并指向你系统里已安装的 Python 3.10+。这一步相当于告诉Py

## 块 11

## 5.1 为什么用 Conda：为你的 LoRA 项目建一个“干净房间”

你现在的目标是 **LoRA 微调**。微调一个模型，意味着你要在自己的电脑上跑 PyTorch、加载 HuggingFace 的模型库，还可能涉及 CUDA 版本的匹配。这些依赖关系非常“脆弱”——版本不对，模型就加载失败，报错能让你排查一整天。

Conda 的价值就在这里：**它能为每个项目创建独立的 Python 空间**。你做 LoRA 微调时，可以单独建一个环境，里面只装这个项目需要的 PyTorch 版本、CUDA 版本、transformers 库。以后做别的项目，再开一个新环境，互不干扰。

更关键的是，**Conda 不仅能管 Python 包，还能管 Python 解释器本身**。深度学习项目经常需要特定 Python 版本（比如 3.10 或 3.11），Conda 可以一键切换。对于后续学习深度学习（需要 PyTorch、CUDA 等复杂依赖），conda 几乎是必备工具。

## 5.2 安装 Conda：选 Miniconda，保持环境干净

安装 Conda 有两个选择，对比如下：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐你选 Miniconda**。原因很直接：你的目标是 LoRA 微调，需要的包就是 PyTorch、transformers 那几个，不需要 Anaconda 预装的一大堆数据分析库。环境越干净，后面排查

## 块 12

### 5.3 创建你的第一个虚拟环境

你的目标是 LoRA 微调，而微调不同模型时，常需要不同版本的 PyTorch 和 CUDA。现在学会创建隔离环境，以后就能直接复用这套流程，避免版本冲突。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，并询问：

```bash
Proceed ([y]/

## 块 13

## 5.4 激活与退出环境：为 LoRA 项目隔离“工作台”

你正在为 LoRA 微调做准备，而 Conda 环境就是你的“独立工作台”。激活它，意味着后续所有 `python` 和 `pip` 命令都只作用于这个项目，不会污染系统或其他项目——这对后续安装 PyTorch、HuggingFace 等依赖至关重要。

**激活环境**：

```bash
conda activate llm-learn
```

激活成功后，命令行前面会出现 `(llm-learn)` 标志，这就是“当前在独立房间内”的信号。

**验证环境**：

```bash
python --version
```

应显示 `Python 3.10.x`。确认版本无误后，再继续安装 LoRA 相关依赖。

**退出环境**：

```bash
conda deactivate
```

退出后回到系统默认环境，`(llm-learn)` 标志消失。

### 5.5 查看与管理环境

## 块 14

**为什么这一章对你重要**：你的目标是 LoRA 微调，而微调脚本必须在正确的 Python 环境里运行。如果 VS Code 或 PyCharm 选错了解释器，即使 `llm-learn` 环境里装好了 PyTorch 和 Transformers，代码也会报 `ModuleNotFoundError`。所以，配置编辑器指向 Conda 环境，是微调前必须打通的一环。

**6.1 VS Code 配置 Conda 环境**

1. 先激活环境：`conda activate llm-learn`。
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入 `Python: Select Interpreter` 并回车。
4. 在列表中选择带有 `conda` 字样的 Python 路径（例如 `Python 3.10.x ('llm-learn': conda)`）。

**原理**：VS Code 会自动识别已存在的 Conda 环境。选中后，当前工作区所有代码都会使用 `llm-learn` 里的 Python 解释器和已安装的包。这一步确保你后续写的 LoRA 训练脚本能 `import torch` 成功。

**6.2 PyCharm 配置 Conda 环境**

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `

## 块 15

## 在命令行验证你的 Python 环境

既然你的目标是 LoRA 微调，那么环境验证就是一切的基础——如果连 Python 都跑不起来，后面加载模型、跑训练脚本都无从谈起。我们一步步来。

**第一步：激活环境**

在终端输入：

```bash
conda activate llm-learn
```

这行命令会切换到我们专门为 LLM 学习创建的环境。你之后跑 LoRA 脚本时，都要先执行这一步。

**第二步：进入 Python 交互模式**

```bash
python
```

输入后回车，你会看到 `>>>` 提示符。这表示你已经进入了 Python 的交互式命令行，可以逐行执行代码。

**第三步：运行你的第一行代码**

看到 `>>>` 后，

## 块 16

**7.3 在 PyCharm 中验证**

环境已就绪，现在用 PyCharm 跑通第一行代码。操作三步：

1. 新建项目，创建 `hello.py` 文件。
2. 右键点击文件 → 选择 `Run 'hello'`。
3. 下方 Run 窗口应输出结果。

跑通后，你的开发工作台就正式可用了。

---

**第八章：常见问题排查**

验证过程中若遇到问题，对照下表快速定位：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

---

**本章小结**

完成本章后，你的电脑上应具备：

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ | ✅ 已安装 | `python --version` |
| pip 国内镜像源 | ⭕ 可选（网络受限时） | `pip config list` |
| VS Code 或 PyCharm | ✅ 已安装 | 打开软件确认 |
| Miniconda / Anaconda |

## 测评题目

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0205-001
以下哪个是装饰器的正确语法？

### Q-KB0205-004
请写一个生成器函数 `fibonacci(n)`，生成前 `n` 个斐波那契数。

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。

### Q-KB0204-003
请写代码：读取 `data.txt` 文件，捕获文件不存在的异常，若不存在则打印 "文件未找到"。

### Q-KB0202-005
以下代码输出是？
```python
d = {}
d.setdefault("key", []).append(1)
d.setdefault("key", []).append(2)
print(d)
```
