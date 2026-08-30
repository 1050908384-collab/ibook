# 学习包

## 块 1

## 1.1 Python 解释器：程序的“翻译官”

你已经熟悉 PyTorch、Transformer 和 HuggingFace 工具链，但请先别跳过这一节——因为后面你要做的 LoRA 微调、权重合并和 vLLM 部署，每一步都依赖这个“翻译官”在后台工作。

**核心概念**：你写的 Python 代码是人类能读懂的文本，但你的 RTX 4090 只认 0 和 1。Python 解释器的作用，就是把你的代码翻译成 GPU 能执行的指令。没有它，你的 `.py` 文件只是一堆文本，vLLM 根本不会启动。

**对你的意义**：当你运行 `python train_lora.py` 或 `python -m vllm.entrypoints.openai.api_server` 时，最先被调用的就是解释器。它负责把你的 Python 代码逐行翻译成底层指令，再交给 PyTorch/CUDA 栈去驱动你的 4090。解释器版本不对或环境混乱，你会在最不该出错的地方——环境启动阶段——卡住。

**实践建议**：既然你偏好在实践中学习，请打开终端，运行 `python --version`，确认解释器可用。如果输出正常，说明你的“翻译官”已就位，可以继续搭建后续环境。

---

**本块学习检查**：完成本块后，用自己的话说明“Python 解释器”与你当前 LoRA 微调 + vLLM 部署目标的关系

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是跑通 LoRA 微调、权重合并和 vLLM 部署。在这条流水线上，你每天要面对大量 `.py` 脚本、配置文件、日志输出。记事本写代码效率极低，所以你需要一个 IDE——也就是你的“工作台”。

**IDE 是什么**：它提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：
- **VS Code**：轻量免费，适合快速改脚本、看日志。
- **PyCharm**：专业

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是用 PyTorch 跑 LoRA 微调、合并权重，再用 vLLM 部署。这些操作全部依赖命令行（cmd 或终端）执行 `python` 和 `pip` 命令。本块就是确保你的 Windows 环境能正确响应这两个命令。

**第一步：下载安装包**

1. 浏览器访问 https://www.python.org
2. 鼠标悬停顶部 **Downloads** 菜单，网站自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令操作（包括安装 PyTorch、运行训练脚本）都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Program

## 块 4

你的目标很明确：在 RTX 4090 上完成 LoRA 微调、权重合并和 vLLM 部署。在动手前，先把 Python 环境弄干净是第一步——macOS 上这一步尤其容易踩坑，因为系统自带的 Python 版本混乱。

**macOS 的 Python 现状**：较老的 macOS 曾预装 Python 2.7，macOS 12.3 起不再预装 Python 2，但系统可能仍保留旧版 `python3` 供系统脚本使用。无论哪种情况，都**强烈建议自行安装最新版 Python**，避免直接依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：在现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除），系统与官方安装包统一使用 `python3`。因此请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用

**本块学习检查**：完成本块后，

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

Ubuntu/Debian 通常自带 Python，但版本可能偏老。为后续跑 LoRA 微调和 vLLM 部署，先确保环境干净可用：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y` 自动确认安装；`python3-venv` 用于建虚拟环境，避免依赖冲突。验证版本号确认安装成功。

## 块 6

## 2.4 操作系统与实训环境该怎么选

你已经熟悉 PyTorch 和 HuggingFace 生态，接下来要跑 LoRA 微调和 vLLM 部署，环境选择直接决定你能否顺利用上 RTX 4090 的 24GB 显存。这里按你的实际场景精简说明：

**GPU 实训（你的主战场）**：直接选 **Ubuntu LTS**（如 22.04/24.04 LTS）的实体机或云服务器。你的 RTX 4090 需要 NVIDIA 驱动、CUDA 和显卡直通才能发挥算力，Ubuntu LTS 对深度学习生态支持最成熟、文档最全，这是你跑 LoRA 微调和 vLLM 部署的稳妥路线。

**日常写 Python、练命令行（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**。它在 Windows 里提供接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**macOS / Linux 用户**：直接用本机终端即可，本身就是类 Unix 环境。

> **⚠️ 不推荐**：别把 **CentOS + VirtualBox 虚拟机**当最佳方案。CentOS 8 已停止维护，建议使用 Ubuntu LTS 或 Rocky Linux 8

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认它从官方 PyPI（`pypi.org`）下载——这是最权威、最完整的源，**应始终保留为默认**。

只有当你网络受限、访问官方源明显变慢或经常超时，才临时切到国内镜像。**镜像只是“网络不好时的加速手段”，不是永久替换项**——官方源永远是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

某次安装慢？最干净的办法是**只对这一次安装临时指定镜像**，不动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方 PyPI 仍是默认；网络恢复或镜像出问题时，无需清理配置。

---

**本块

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你接下来的 LoRA 微调和 vLLM 部署会频繁安装依赖包。如果你长期处于网络受限环境，每次手动加 `-i` 镜像参数确实繁琐。本节的配置能让 pip 默认走清华镜像，省去重复输入。

**⚠️ 前提提醒**：官方 PyPI 始终是最可靠来源。网络恢复后，建议改回或删除该配置。

---

### Windows 配置步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车（打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹）。
2. 新建文件夹，命名为 `pip`。
3. 在 `pip` 文件夹内新建文本文件，命名为 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从该地址下载包。

---

### macOS / Linux 配置步骤

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入相同内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

---

### ⚠️ 关于 `trusted-host` 的警告

网上教程常顺手加 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用无需此项。仅在企业内网自建源或确认来源可信的证书报错场景下，才临时添加。

---

### 验证配置

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即配置成功。

---

### 恢复官方源

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：用自己的话说明——为什么长期网络受限时才建议设为默认？`trusted-host` 的风险是什么？这个配置对你后续 LoRA 微调（安装 peft、transformers 等）和 vLLM 部署的依赖安装环节有什么帮助？遇到无法解释的术语请

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

VS Code 是微软出品的免费编辑器，启动快、插件丰富。你的目标是完成 LoRA 微调和 vLLM 部署，后续写训练脚本、改配置、看日志都会用到它，所以先把它装好。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

安装 Python 插件后，VS Code 才能识别 `.py`

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具，适合大型项目开发——比如你后续的LoRA微调工程。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`。
2. `Location` 选项目路径（**路径中不要含中文**）。
3. `Interpreter` 选 `Previously configured interpreter`，再选系统已装的 Python 3.10+。

---

## 第五章：配置 Conda 虚拟环境（项目隔离）

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语

## 块 11

## 5.1 为什么用 Conda

你已经有 PyTorch、Transformer 和 HuggingFace 的基础，接下来要完成 LoRA 微调、权重合并和 vLLM 部署。这个流程涉及多个项目、多套依赖，环境隔离不是可选项，而是必需品。

Conda 是目前最主流的虚拟环境管理工具。它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对深度学习来说，PyTorch、CUDA 这类复杂依赖经常互相牵制，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你的目标是 LoRA 微调和 vLLM 部署，环境越干净越好，需要什么装什么，避免预装库干扰版本。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 后，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 conda

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你已熟悉Transformer/HF工具链，但微调和部署不同项目时，PyTorch和CUDA版本常互相冲突。虚拟环境就是给每个项目一个独立的Python“房间”，避免版本打架。本块教你搭建这间房，后续LoRA微调和vLLM部署都会复用此流程。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

拆解：`conda create`是创建命令；`-n llm-learn`给环境命名（`n

## 块 13

## 5.4 激活与退出环境

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

### 5.5 查看和管理环境

**查看所有已创建的环境**：

```bash
conda env list
```

带 `*` 的就是当前激活的环境。

**删除环境**（如果以后需要清理）：

```bash
conda remove -n llm-learn --all
```

---

## 第六章：在 VS Code / PyCharm 中使用 Conda 环境

---

**本块

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是完成 LoRA 微调、权重合并和 vLLM 部署，这三步都依赖同一个干净的 Python 环境。下面把 VS Code 指向你已激活的 `llm-learn` 环境：

1. 终端里先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表中选择带有 `conda` 字样的 Python 路径，例如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，你在当前工作区写的所有代码都会使用 `llm-learn` 里的 Python 和已安装的包——也就是你后续跑 LoRA 脚本和 vLLM 推理时用的那套环境。

## 6.2 PyCharm 配置 Conda 环境

如果你更习惯 PyCharm，按以下步骤操作：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 上是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 下拉框右侧点击浏览

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你已熟悉PyTorch与HuggingFace，但环境验证是后续微调与部署的地基。本块只做一件事：确认Python能跑通。

**命令行验证**（适合快速检查）：
1. 激活环境：`conda activate llm-learn`
2. 输入`python`进入交互模式，看到`>>>`提示符
3. 输入`print("Hello, Python! 我的环境准备好了！")`并回车
4. 正确打印后，用`exit()`退出

**VS Code验证**（

## 块 16

### 7.3 在 PyCharm 中验证

你的环境已经就绪，现在用 PyCharm 跑通第一行代码，确认整条工具链（解释器→项目→运行）是通的。这一步对后续 LoRA 微调和 vLLM 部署至关重要——如果连 `hello.py` 都跑不起来，后面加载模型、启动推理服务时会浪费大量时间在环境排查上。

**操作步骤**：

1. 打开 PyCharm，新建一个项目（路径不要包含中文或空格）。
2. 在项目中创建文件 `hello.py`，输入 `print("hello")`。
3. 右键点击文件 → 选择 `Run 'hello'`。
4. 观察下方 Run 窗口，应输出 `hello`。

如果输出正确，说明你的 Python 解释器、项目配置和运行机制全部正常。后续你在 PyCharm 中写 LoRA 训练脚本、调用 `transformers` 加载模型，都是基于这个运行通道。

---

### 第八章：常见问题排查（速查表）

你在微调和部署时大概率会遇到以下问题，先记住解决方案，遇到时直接对照：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

**重点提醒**：你后续要装 `peft`、`vllm` 等库，务必先 `conda activate llm-learn`，再用 `python -m pip install`，避免装错环境导致 `import` 失败。

---

### 本章小结

完成本章后，你的电脑应具备：

|

## 测评题目

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
```

### Q-KB0201-003
以下代码的输出是？
```python
x = 10
y = 3
print(x // y, x % y, x ** y)
```

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0205-008
写一个类 `Stack`（栈），实现 `push(item)`、`pop()`、`peek()`、`is_empty()` 方法。`pop()` 在栈空时抛出 `IndexError`。
