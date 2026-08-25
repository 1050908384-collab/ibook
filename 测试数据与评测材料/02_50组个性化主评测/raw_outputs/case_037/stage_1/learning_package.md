# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写代码时用的是人类语言，但电脑只懂 0 和 1。**Python 解释器**就是那个“翻译官”——它把你写的 Python 代码翻译成电脑能执行的指令。没有它，你的 `.py` 文件就只是一堆普通文本。

**为什么这和你部署 vLLM 有关？**  
你后面要跑的 vLLM 推理服务，本质上是让电脑执行大量 Python 代码。解释器装不对，后面每一步都会卡壳。

**动手验证（2分钟）**  
打开终端（Mac/Linux）或命令提示符（Windows），输入：
```bash
python --version
```
如果看到类似 `Python 3.x.x` 的输出，说明解释器已就位。如果提示“找不到命令”，别急——这正是下一节要解决的问题。

> **记住**：装解释器不是“装个软件”那么简单，它是你电脑和代码之间的桥梁。桥没搭好，车（代码）就过不去。

## 块 2

### 你的“工作台”：选对工具，事半功倍

你的目标是部署 vLLM，这意味着要写、改、调试不少 Python 代码。记事本能写，但你会疯掉。**IDE（集成开发环境）**就是干这个的：语法高亮让你一眼看清结构，自动补全减少打字，内置调试帮你揪出错误。推荐两款：**VS Code**（轻量免费，上手快）和 **PyCharm**（专业强大，功能全）。先装 VS Code，够用且不吓人。

### 项目的“独立房间”：虚拟环境

部署 vLLM 时，你会装一堆依赖包。不同项目可能要求同一个包的不同版本，全装一起必冲突。**虚拟环境**就是给每个项目一个独立的 Python 空间，

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 Python 是这一切的基础。下面直接进入正题，按步骤操作，每一步都有明确目的。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

> 注意：网站推荐的 **Python 3.13.x 满足 vLLM 要求**，可直接使用。若你更倾向稳定版本，也可选择 **3.10 或 3.11**，两者同样满足要求。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是最关键的一步。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令行操作都会卡住，包括你之后要用的 `pip` 安装命令。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。如果你想改路径，务必确保

## 块 4

### macOS 安装 Python：为 vLLM 部署打好地基

你的目标是部署 vLLM，而 Python 环境是第一步。现代 macOS（Catalina 10.15 起）已不再预装 Python 2，即便有旧版 `python3` 也是给系统脚本用的。**强烈建议自己装最新版**，别依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 完成。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**关键坑**：命令行默认**没有** `python` 命令，只有 `python3`。始终用 `python3` / `pip3`，别敲 `python`，否则会报"命令不存在"或误用系统旧版。

**方法二：Homebrew（已有的话）**

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：打开终端，依次运行 `python3 --version` 和 `pip3 --version`，确认两条命令都输出版本号（如 Python 3.x.x）。如果报错，回到方法一重装。

## 块 5

**动手装：Ubuntu/Debian 两步到位**

你的目标是部署 vLLM，但先别急，Python 环境是地基。Linux 一般自带 Python，但版本可能偏老，我们直接装发行版仓库中的最新版本。

**第一步：更新软件源**
```bash
sudo apt update
```
这步让系统知道有哪些新软件包可用，避免装到旧版本。

**第二步：安装 Python 3 全家桶**
```bash
sudo apt install python3 python3-pip python3-venv -y
```
- `python3`：Python 本体
- `python3-pip`：装第三方库（比如 vLLM）的工具
- `python3-venv`：创建隔离环境，防止

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这是 GPU 密集型任务，环境选错会直接卡住进度。按你的动手习惯，直接给结论：

**日常练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。管理员 PowerShell 执行 `wsl --install` 即可。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境，不用额外配置。

**GPU 实训（你的核心场景）**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。vLLM 部署依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 别踩坑**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于 2021 年停止维护，CentOS Stream 8 也于 2024 年停止维护；VirtualBox 默认**无法

## 块 7

## 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、应保留的默认源**。

**什么时候才需要镜像源？** 只有当你处在网络受限环境、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。记住：**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 vLLM 部署准备环境，网络不稳定会反复打断下载。如果你**长期**处于这种环境，才把清华镜像设为默认源。注意：官方 PyPI 最可靠，网络恢复后记得改回。

**Windows 操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开配置文件夹（类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 新建文件夹，命名为 `pip`。
3. 进入 `pip` 文件夹，新建文本文件 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从清华源下载包。

> **⚠️ 别急着加 `trusted-host`**：网上教程常让你顺手加一行 `trusted-host = ...`，但除非镜像 HTTPS 证书验证失败，否则**不要加**。加了它等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常用**不需要**。只有企业内网自建源、或确认来源可信且确实报证书错误时，才临时加。

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

### 3.4 验证配置

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

### 3.5 改回官方源

删除 `pip.ini` / `pip.conf` 文件，或执行：

```bash
pip config unset global.index-url
```

**本块学习检查**：按你的系统

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。它不只适合 Python，后续你学 Linux 命令行、写脚本时都能复用，一次配置，长期受益。

**安装步骤**（三步搞定）：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

> 为什么必须装？不装 Python 插件，VS Code 无法识别 `.py` 文件，也就没有代码补全、错误提示和一键运行功能。

**配置 Python 解释器**（让编辑器知道用

## 块 10

**动手装 PyCharm：为 vLLM 项目准备专业工作台**

你目标是部署 vLLM，这属于大型项目开发，PyCharm Community 版正好合适——免费、轻量、适合学习。

**安装三步走：**
1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）
3. 运行安装程序，务必勾选这两项：
   - `Add "Open Folder as Project"`（以后右键文件夹就能直接打开）
   - `Add to PATH`（若需命令行启动 PyCharm 才勾选；不需要可取消，避免环境变量混乱）

**首次启动配置：**
1. 打开后选 `New Project`
2. `Location` 选项目存放路径——**路径别带中文**，否则后续 vLLM 相关工具可能出问题
3. `Interpreter` 若首次使用且无已配置解释器，可选 `New environment using Virtualenv` 或 `Existing interpreter`

## 块 11

### 5.1 为什么用 Conda

你之前已经知道，虚拟环境能为每个项目创建独立的 Python 空间。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对你后续学习深度学习（需要 PyTorch、CUDA 等复杂依赖）来说，conda 几乎是必备工具——尤其是你计划用 vLLM 做模型推理部署时，依赖管理会非常复杂，conda 能帮你把环境理得清清楚楚。

### 5.2 安装 Conda：Anaconda vs Miniconda

有两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB 空间），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包——这正好符合你“hands-on”的学习风格，也适合 vLLM 部署这种需要精确控制依赖的场景。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   - **macOS/Linux**：

## 块 12

**创建你的第一个虚拟环境：为 vLLM 部署打好地基**

你马上要部署 vLLM，不同项目常需要不同版本的 PyTorch 和 CUDA。现在花两分钟学会创建隔离环境，以后每个项目都能直接复用这套流程，避免依赖冲突。

打开终端，执行下面这条命令：

```bash
conda create -n llm-learn python=3.10
```

**逐段拆解，确保你完全看懂**：
- `conda create`：告诉 Conda 你要新建一个环境。
- `-n llm-learn`：给这个环境起名 `llm-learn`（`n` 是 `name` 的缩写，后面跟的就是环境名）。
- `python=3.10`：指定该环境内安装的 Python

## 块 13

**5.4 激活与退出环境（动手做）**

你的目标是部署 vLLM，所以现在就把这个环境当成你的“专属工作台”。先激活它：

```bash
conda activate llm-learn
```

激活后，注意命令行开头会出现 `(llm-learn)` 标志。这意味着你之后敲的 `python` 和 `pip` 命令，都只在这个环境里生效，不会弄乱系统或其他项目——这对后面部署 vLLM 时保持依赖干净很重要。

马上验证一下当前 Python 版本：

```bash
python --version
```

应该显示 `Python 3.10.x`。看到这个，说明环境已就绪。

想退出这个环境、回到系统默认环境时，执行：

```bash
conda deactivate
```

**快速自检**：激活环境后，命令行前出现什么标志？执行 `python --version` 后，你期望看到什么版本？

## 块 14

**本块学习检查**：完成配置后，在 VS Code 或 PyCharm 中新建一个 Python 文件，输入 `import sys; print(sys.executable)`，运行并确认输出的路径中包含 `llm-learn`。

---

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件。可先运行 `conda env list` 查看实际路径，例如 `<你的用户名

## 块 15

**目标**：验证你的 Python 环境已就绪，为 vLLM 部署打基础。

**在命令行验证**（推荐，更贴近部署场景）：
1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符。
3. 输入：`print("Hello, Python! 我的环境准备好了！")`，回车。
4. 正确打印后，输入 `exit()` 退出。

**在 VS Code 中验证**（可选）：
1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建 `hello.py`，输入：`print("Hello from VS Code!")`。
3. 点击右上角运行按钮或

## 块 16

### 7.3 在 PyCharm 中验证

既然你的目标是 vLLM 部署，那环境验证就是第一道关卡。跟着做，别跳步：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个文件，命名为 `hello.py`。
2. **运行脚本**：在 `hello.py` 文件里随便写一行代码（比如 `print("hello")`），然后**右键点击文件** → 选择 **`Run 'hello'`**。
3. **查看输出**：下方会弹出 **Run 窗口**，里面应该能看到你写的代码运行后的结果。

看到输出，说明你的开发工作台已经能跑代码了。

---

## 第八章：常见问题排查（重点看这几个）

部署 vLLM 时，环境问题最容易卡人。遇到下面这些情况，直接对照解决：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

> **特别提醒**：vLLM 依赖的包很多，**装包前一定先 `conda activate` 进入你的目标环境**，否则很容易出现"装了半天，import 还是报错"的情况。

---

## 本章小结

完成本章后，你的电脑上应该具备以下内容：
