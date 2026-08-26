# 学习包

## 块 1

**本块学习检查**：读完这一节，请合上资料，用自己的话回答：为什么说“没有解释器，`.py` 文件只是一堆文本”？如果答不上来，回头重读一遍。

---

### 1.1 Python 解释器：程序的“翻译官”

你的目标很明确：**LoRA微调 + 权重合并 + vLLM部署**。这条路最终要跑在电脑上，而电脑只认0和1，不认任何人类语言。所以，你写的每一行Python代码，都需要一个“翻译官”把它变成电脑能执行的指令——这个翻译官就是**Python解释器**。

**为什么必须先装它？** 因为你现在写的 `.py` 文件，本质上只是纯文本。没有解释器，这些文本对电脑来说毫无意义，就像一份没人翻译的外文合同。只有解释器接手，你的代码才会被“读”懂并执行。

**对你这个动手派来说，记住一个画面就够了**：你写代码 → 解释器翻译 → 电脑执行。后续装环境、跑微调脚本，每一步都离不开这个翻译官在后台工作。

**现在动手验证一下**：打开你的命令行（终端），输入 `python --version` 并回车。如果看到版本号，说明解释器已就位；如果提示“找不到命令”，说明还没装好——这正是你接下来要解决的。

## 块 2

## 1.2 代码编辑器（IDE）：你的“工作台”

你马上要跑 LoRA 微调和 vLLM 部署，代码量不小。记事本能写，但改起来太痛苦。**IDE（集成开发环境）** 就是干这个的：语法高亮让你一眼看清结构，自动补全少敲一半键盘，运行调试帮你快速定位报错——就像用 Word 写文档，而不是用记事本。

推荐两款，按需选：

- **VS Code**：轻量、免费、启动快，装个 Python 插件就能用。适合你这种想快速上手、边写边跑的风格。
- **PyCharm**：功能更全，项目导航、调试器更强大，但更重。等你的微调脚本变复杂了再换也不迟。

**本块学习检查**：打开你选的 IDE，新建一个 `.py` 文件，输入 `print("hello")`，用 IDE 的“运行

## 块 3

### 2.1 Windows 用户：3 步搞定 Python 安装

你的目标是跑通 LoRA 微调和 vLLM 部署，Python 是这一切的地基。别担心，跟着下面 3 步走，10 分钟就能搭好。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads** 菜单，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，你打开命令行输入 `python`，系统会提示"不是内部或外部命令"，后续所有命令行操作（包括装 LoRA 相关库）都会卡住。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装

## 块 4

### 2.2 macOS 用户：装好 Python，为 LoRA 微调铺路

你的目标是跑通 LoRA 微调 + vLLM 部署，而这一切的第一步，就是在本机装一个干净、可用的 Python。macOS 这点比较特殊，我们直接说重点。

**先避坑**：macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 是给系统脚本用的，版本旧且不可控。**别碰系统版**，自己装最新的。

---

**方法一：官方安装包（推荐，最省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 即可。
3. 打开终端（Terminal），输入下面两条命令验证：

```bash
python3 --version
pip3 --version
```

看到版本号输出，就装好了。

**关键提醒**：在 macOS 上，命令是 `python3` 和 `pip3`，不是 `python` 和 `pip`。系统默认不提供 `python` 命令，直接用 `python3` 能避免“命令不存在”或误用系统

## 块 5

**动手装好 Python，为 LoRA 微调和 vLLM 部署打地基**

你的目标是跑通 LoRA 微调 + 权重合并 + vLLM 部署，而这一切都建立在 Python 环境上。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，逐行执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

- `sudo apt update`：刷新软件源，确保能装到最新版本。
- `-y`：自动确认安装，不用手动敲 `y`。
- `python3-venv`：创建虚拟环境的核心模块，现在

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 **LoRA 微调 + 权重合并 + vLLM 部署**，这三步都离不开 GPU。选对环境，直接决定你后面能不能顺利跑通。

**日常练命令（Windows 用户）**：别在 CMD 里耗时间，启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。管理员 PowerShell 执行 `wsl --install` 即可。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，练完直接无缝切换。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境，不用额外折腾。

**GPU 实训（LoRA/QLoRA 微调、vLLM 部署）**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于2021年

## 块 7

## 3.1 什么时候才需要镜像源

你正在为“LoRA微调+权重合并+vLLM部署”做准备，安装 `transformers`、`peft`、`vllm` 这些库时，`pip` 默认从官方 PyPI（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

**关键判断**：只有网络受限、访问官方源明显变慢或经常超时时，才临时切到国内镜像。**镜像是“网络不好时的临时加速手段”，不是永久替换项**——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源（推荐）

某次安装慢？只对**这一次**临时指定镜像，不动全局配置：

```bash
pip install numpy -

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为“LoRA微调+权重合并+vLLM部署”做准备，后面会频繁安装依赖包。如果网络长期不稳定，每次手动加镜像参数太麻烦，可以一次性写入配置，让 pip 默认走清华源。

**重要提醒**：官方 PyPI 永远是最可靠的。网络恢复后，记得改回或删除配置，避免长期依赖镜像。

---

**Windows 操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开配置目录（类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 新建文件夹，命名为 `pip`。
3. 进入 `pip` 文件夹，新建文本文件 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：“以后默认从这里下载包”。

**⚠️ 别乱加 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`，**除非镜像的 HTTPS 证书验证确实报错，否则不要加**。加了等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法证书，正常使用**不需要**。只有企业内网自建源、确认来源可信时才临时加。

---

**macOS / Linux 操作步骤：**

1. 终端执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建并编辑配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

4. `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### 3.4 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

---

### 3.5 改回官方源

删除刚才创建的 `pip.ini` / `pip.conf` 文件即可，或执行：

```bash
pip config unset global.index-url
```

---

**本块学习检查**：

## 块 9

**4.1 方案一：VS Code（轻量、免费、推荐）**

你的目标是完成LoRA微调和vLLM部署，这需要稳定、顺手的开发环境。VS Code是微软出品的免费编辑器，启动快、插件丰富，非常适合Python开发，后续写脚本、改配置也能用。

**安装步骤：**
1. 访问 https://code.visualstudio.com，下载对应系统安装包。
2. 运行安装程序，一路下一步即可。

**必装插件：**
打开VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python语言服务器，让代码提示更智能）

安装Python插件后，VS Code才能识别`.py`文件，提供代码补全、错误提示和一键运行功能。

**配置Python解释器：**
1. 按`Ctrl + Shift + P`（Mac是`Cmd + Shift + P`）。
2. 输入`Python: Select Interpreter`，选择你

## 块 10

## 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制、数据库工具，适合大型项目开发——你后续做 LoRA 微调和 vLLM 部署时，项目文件多、依赖杂，正好需要这种专业工具。

**安装步骤**（跟着做，3分钟搞定）：

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费那个）
3. 运行安装程序，务必勾选这两项：
   - `Add "Open Folder as Project"`（右键菜单直接打开项目文件夹）
   - `Add to PATH`（命令行也能启动 PyCharm）

**首次启动配置**（关键一步）：

1. 打开 PyCharm，选择 `New Project`
2. `Location` 处选项目存放路径——**路径中不要含中文**，否则后面跑模型会报编码错误

## 块 11

好的，我们直接进入正题。你的目标是搞定 LoRA 微调和 vLLM 部署，这两件事都离不开 PyTorch 和 CUDA 这些“重”依赖。如果直接在系统 Python 里装，很容易把环境搞乱，到时候哭都来不及。

Conda 就是来解决这个问题的。它不只是创建虚拟环境，还能帮你管理不同版本的 Python 解释器本身。这对深度学习项目来说几乎是必备的，因为不同项目可能依赖不同版本的 Python 和 CUDA，Conda 能帮你把它们干净地隔离开。

安装 Conda 有两个选择：Anaconda 和 Miniconda。Anaconda 体积大（500MB+），预装了很多数据科学库，适合做数据分析、不想手动装包的人。但你的目标是项目开发，环境越干净越好，所以推荐 **Miniconda**。它只有 50MB 左右，只包含 conda 和基础 Python，需要什么装什么，不拖泥带水。

安装步骤很简单，跟着做就行：

1.  去官网下载对应系统的安装包（Windows 选 `.exe`，macOS/Linux 选 `.sh`）。
2.  **Windows 用户注意**：安装时一路下一步，但**千万别勾选** “Add Miniconda3 to my PATH environment variable”。这个选项容易和系统已有的 Python 冲突。日常使用，请从开始菜单打开 **“Anaconda Prompt (Miniconda3)”** 来敲 conda 命令。
3.  **macOS/Linux 用户注意**：运行 `.sh` 文件，同意协议，最后一步选择 `yes` 让安装器执行 `conda init`。这样每次打开终端，conda 就能自动接管环境，不用手动改 PATH。
4.  **验证安装**：打开一个**新的**终端（Windows 用 Anaconda Prompt，macOS/Linux 重开终端），输入 `conda --version`。如果看到类似 `conda 24.x.x` 的版本号，就说明安装成功了。

装好 Conda，你的开发工作台就搭好了一半。下一步就是用它创建项目专属的虚拟环境，为 LoRA 微调做好准备。

## 块 12

**动手创建你的第一个虚拟环境**

你的目标是跑通 LoRA 微调和 vLLM 部署，而这两步都依赖 PyTorch 和 CUDA。不同项目往往需要不同版本的 PyTorch 和 CUDA，如果全装在一起很容易冲突。现在学会创建独立环境，后面每个项目都能直接复用这套流程。

打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在做什么？**
- `conda create`：创建环境的核心命令
- `-n llm-learn`：给环境起名为 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定该环境使用 Python 3.10 版本

## 块 13

### 5.4 激活与退出环境

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志，表示后续所有的 `python` 和 `pip` 命令都作用在这个环境里，不会影响系统或其他项目。

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

## 第六章：在 VS Code / PyCharm 中使用 Conda

## 块 14

### 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。VS Code 会自动识别 conda 环境，选好后，工作区代码就会用 `llm-learn` 里的 Python 和包运行。

### 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选 `Project: 你的项目名` → `Python Interpreter`。点齿轮图标 → `Add...`，左侧选 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径选 conda 环境的 Python 可执行文件：
- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

---

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：在 VS Code 中新建一个 Python 文件，输入 `import torch; print(torch.__version__)`，运行。若输出版本号，则说明环境配置成功。

## 块 15

**7.1 在命令行验证**

打开终端，激活环境：`conda activate llm-learn`。然后输入 `python` 回车，看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车后若正确打印文字，输入 `exit()` 退出。

**7.2 在 VS Code 中验证**

新建文件夹 `my-first-project`，用 VS Code 打开。新建文件 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角（或按 `Ctrl + F5`），下方终端应输出 `Hello from VS Code!`。

**本块学习检查**：按上述步骤在命令行和 VS Code 各运行一次，

## 块 16

### 7.3 在 PyCharm 中验证

你的目标很明确：**LoRA微调 + 权重合并 + vLLM部署**。在动手跑这些大模型任务之前，先确保你的“工作台”能正常运转。这一步就是**点火测试**——用最简代码确认环境没毛病。

**操作步骤（跟着做，别跳步）：**

1. **新建项目**：打开 PyCharm，`File → New Project`，位置选一个**纯英文、无空格**的路径（比如 `D:\llm-projects`），这能避免后续一堆莫名其妙的报错。
2. **创建文件**：在项目里新建一个 Python 文件，命名为 `hello.py`。
3. **运行**：在 `hello.py` 里输入 `print("Hello, LLM!")`，然后**右键点击文件 → `Run 'hello'`**。
4. **看结果**：下方弹出 **Run 窗口**，看到 `Hello, LLM!` 就说明 PyCharm 能正常执行 Python 代码，你的开发环境**验证通过**。

> **小提示**：如果 Run 窗口报错，先别慌，对照第八章的排查表找原因。最常见的坑是“解释器没选对”——按 `Ctrl + Shift + A`（macOS 是 `Cmd + Shift + A`）→ 输入 `Python: Select Interpreter`，手动选你创建的 `llm-learn` 环境。

---

### 第八章：常见问题排查（速查表）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） |
