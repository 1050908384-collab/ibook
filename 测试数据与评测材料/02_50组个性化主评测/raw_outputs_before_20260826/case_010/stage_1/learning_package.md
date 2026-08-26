# 学习包

## 块 1

**本块学习检查**：读完本节后，请合上资料，用一句话向自己解释“为什么没有解释器，你的 .py 文件就只是一堆文本？”能说清楚，再继续往下装环境。

---

### 1.1 Python 解释器：程序的“翻译官”

你的目标是部署 vLLM，但先别急着碰模型——所有 AI 工具链都跑在 Python 上，而**解释器**是 Python 的发动机。

**它解决什么问题？**
你写的代码是人类语言，电脑只认 0 和 1。解释器就是中间的“翻译官”：把你写的每一行 Python 指令，实时翻译成电脑能执行的机器指令。

**为什么你必须先懂这个？**
- 没有解释器，你写的 `.py` 文件在电脑眼里就是一堆普通文本，双击也不会运行。
- 后面装 vLLM、跑推理，本质上都是先让解释器运行你的 Python 代码，再由代码去调用 GPU 等硬件。

**动手验证（30秒）**
打开你的终端，输入 `python --version`。如果看到版本号，说明解释器已就位；如果报错，说明还没装——这正是下一节要解决的。

## 块 2

好的，我们直接进入正题。你的目标是部署 vLLM，这需要一套干净、可控的 Python 环境。别在“记事本写代码”上浪费时间，我们直接搭建专业工作台。

**IDE：选 VS Code 就对了**
IDE 就是你的代码工作台，提供语法高亮、自动补全和调试功能。虽然记事本也能写，但效率天差地别。针对 vLLM 这种大型项目，推荐 **VS Code**：轻量、免费，配合插件完全够用。

**虚拟环境：项目的“独立房间”**
这是避免环境冲突的关键。想象项目 A 需要 `numpy 1.0`，项目 B 需要 `numpy 2.0`，装一起必然报错。虚拟环境

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是部署 vLLM，而 Python 是这一切的地基。下面三步走，装好它。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`。
2. 鼠标移到顶部菜单 **Downloads**，网站会根据你的操作系统推荐版本，但请**手动选择 vLLM 兼容的 Python 版本（如 3.10 或 3.11）**，不要直接使用推荐的最新版（如 `Python 3.13.x`），因为 vLLM 通常要求 Python 3.8-3.11。
3. 在所选版本的下载页面中，点击 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这是最关键的一步。勾选后，Python 会被加入系统环境变量。否则，你之后在命令行输入 `python` 会提示"不是内部或外部命令"，后续所有操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径不含中文和特殊符号，能避免后续路径报错。如果你想改路径，务必确保路径中**没有

## 块 4

**macOS 上装 Python，别用系统自带的**

现代 macOS（Catalina 10.15 起）不再预装 Python 2，部分系统虽有旧版 `python3`，但那是给系统脚本用的。**强烈建议自己装最新版**，别依赖系统版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install"。
3. 打开终端，验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 命令行默认**没有** `python` 命令。系统与官方安装包统一用 `python3`，所以始终用 `python3` / `pip3`，避免“命令不存在”或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

已装 Homebrew 的话，终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：打开终端，依次运行 `python3 --version` 和 `pip3 --version`，确认两条命令都输出版本号（如 `Python 3.x.x`），且不是“command not found”。若

## 块 5

**目标导向**：你要用 vLLM 部署模型，Python 环境是地基。Ubuntu/Debian 通常自带 Python，但版本偏老，直接装 3.10+ 更稳。

**动手做**（终端逐行执行）：

```bash
sudo apt update          # 刷新软件源，确保能装到最新包
sudo apt install python3 python3-pip python3-venv -y
python3 --version        # 确认版本号，3.10+ 即可
```

**关键点**：`-y` 免去手动确认；`python3-venv` 是虚拟环境模块，现在装好，后面部署 vLLM 隔离依赖时不用补。

## 块 6

**本块学习检查**：读完本节后，请先别往下翻，用一句话回答：你的 vLLM 部署实训，应该选哪种操作系统环境？为什么？

---

### 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，而且喜欢动手练。选对环境，能直接决定你后面是“顺畅跑通”还是“卡在环境里”。

**先分清两种场景，别混着来：**

**1. 日常练 Python、敲命令（打基础）**
- **如果你是 Windows 用户**：别在原生 CMD 里练，推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它相当于在 Windows 里开了一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致。安装很简单：管理员 PowerShell 执行 `wsl --install` 即可。
- **如果你是 macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境，不用额外配置。

**2. 跑 GPU 实训（vLLM 部署就属于这类）**
- **必须使用装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。原因很直接：vLLM 部署依赖 NVIDIA 驱动、CUDA 和显卡直通，而 Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 一个要避开的坑：**

## 块 7

好的，我们直接进入正题。你的目标是部署 vLLM，这意味着你会频繁安装大型依赖包。如果网络不畅，每次等待都可能浪费你宝贵的练习时间。

**核心原则**：官方 PyPI 源（`pypi.org`）永远是最权威、最完整的默认选择。镜像源只是**网络不佳时的临时加速手段**，不要永久替换默认设置。

**什么时候需要**：当你访问官方源下载明显变慢或经常超时时，才考虑切换。

**推荐做法**：只对**当前这一次**安装临时指定镜像源，不改动任何全局配置。这样最干净，不会影响你平时的默认行为，也免去了日后清理配置的麻烦。

**操作示例**（安装 `numpy` 时使用清华镜像）：
```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**本块学习检查**：请你在终端中，用上述命令临时安装一个你

## 块 8

### 3.3 进阶：把镜像设为默认（长期网络受限才做）

你正在为 vLLM 部署准备环境，网络不稳会卡住下载。但记住：**只有长期受限、几乎每次都要用镜像时**才设默认。官方 PyPI 最可靠，网络恢复后建议改回。

**Windows 操作：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开 `C:\Users\你的用户名\AppData\Roaming`。
2. 新建文件夹，命名为 `pip`。
3. 进入 `pip` 文件夹，新建文本文件 `pip.ini`。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从清华镜像下载。

> **⚠️ 别乱加 `trusted-host`**：网上教程常让你加一行 `trusted-host = ...`，**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**。加上它等于让 pip 跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常用**不需要**。只有企业内网自建源、或确认来源可信且遇到证书报错时，才临时加。

**macOS / Linux 操作：**

1. 终端执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建并编辑配置：

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

**本块学习检查**：动手操作一遍——按你的系统完成配置，运行 `pip config list` 确认输出，然后执行 `pip config unset global.index-url` 改回官方源，

## 块 9

**目标**：装好 VS Code，为后续 vLLM 部署写 Python 脚本做准备。

**为什么选它**：VS Code 免费、启动快、插件多，既能写 Python，以后改配置文件、敲 Linux 命令也用得上。你每天只有 2.5 小时，它轻量不折腾，适合快速上手。

**安装（约 5 分钟）**：
1. 打开 https://code.visualstudio.com，下载你系统的安装包。
2. 运行安装程序，一路“下一步”就行。

**装完必做：装两个插件**：
1. 点左侧最下方的四个方块图标（扩展）。
2. 搜索并安装：
   - **Python**（微软官方）：提供语法高亮、调试、智能提示。
   - **Pylance**：让代码提示更聪明。

> 不装 Python 插件，VS Code 认不出 `.py` 文件，也没代码补全和错误提示。

**配置解释器（关键一步）**：
1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P`）。
2. 输入 `Python

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具等功能，适合大型项目开发——这对你后续跑 vLLM 这类复杂推理项目尤其有用。

**安装步骤（跟着做，5分钟搞定）：**

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费那个）
3. 运行安装程序，**务必勾选**这两个选项：
   - `Add "Open Folder as Project"`（以后右键文件夹就能直接打开）
   - `Add to PATH`（命令行里敲 `pycharm` 就能启动）

**首次启动配置（关键，别跳过）：**

1. 打开 PyCharm，选 `New Project`
2. `Location` 处选项目存放路径——**路径

## 块 11

## 5.1 为什么用 Conda

你的目标是部署 vLLM，这意味着你很快会接触 PyTorch、CUDA 这类依赖地狱。Conda 就是为你解决这个问题的：它不仅能创建独立的 Python 环境，还能直接管理 Python 解释器版本——比如某个项目要 Python 3.10，另一个要 3.11，Conda 都能各自隔离，互不干扰。

**一句话记住**：Conda = 虚拟环境 + Python 版本管理器，深度学习必备。

## 5.2 安装 Conda：Anaconda vs Miniconda

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 约 500MB+，预装 numpy、pandas、jupyter 等 | 数据分析、不想手动装包 |
| **Miniconda** | 约 50MB，只有 conda + 基础 Python | **你**——项目开发，按需装包 |

**选 Miniconda**，环境干净，后面部署 vLLM 时缺什么装什么，不背一堆用不上的库。

### 安装步骤（跟着做）

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS/Linux 选 `.sh`）
3. 运行安装：
   - **Windows**：一路下一步，**千万别勾** "Add Miniconda3 to my PATH environment variable"（安装器自己都标了"不推荐"）。勾了容易和系统已有的 Python 冲突，导致 `python`、`pip` 指向混乱。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 操作。
   - **macOS/Linux**：运行 `.sh`，输入 `yes` 同意协议、选安装路径，最后一步**必须选 `yes` 执行 `conda init`**——这样终端会自动接管 conda，而不是硬塞 PATH。
4. **验证**：打开**新的**终端（Windows 用 Anaconda Prompt；macOS/Linux 重开终端），输入：

```bash
conda --version
```

看到 `conda 24.x.x` 之类的输出，就成功了。

---

**本块学习检查**：打开终端输入 `conda --version`，把输出结果记下来。

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么先学这个**：你之后要部署 vLLM，不同项目对 PyTorch 和 CUDA 版本要求不同。现在把环境建好，后面直接复用这套流程，省去反复折腾。

**动手做**：打开终端，执行下面这行命令：

```bash
conda create -n llm-learn python=3.10
```

**这行命令在干嘛**，拆开看：
- `conda create`：让 Conda 帮你新建一个环境
- `-n llm-learn`：给这个环境起名 `llm-learn`（`n` 就是 `name`）
- `python=3.10`：指定环境里装 Python 3.10

**执行

## 块 13

## 5.4 激活与退出环境

**激活环境**（进入这个"独立房间"）：

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

**

## 块 14

**本块学习检查**：完成配置后，在 VS Code 中新建一个 `.py` 文件，输入 `print("env ok")`，确认右下角解释器显示 `llm-learn`，运行并看到输出。

---

### 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境，选好后，你在这个工作区写的代码就会用 `llm-learn` 里的 Python 和已装包。

### 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件（路径因用户

## 块 15

**在命令行验证（7.1）**

你的目标是部署 vLLM，但一切从环境跑通开始。按顺序做：

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符。
3. 输入：`print("Hello, Python! 我的环境准备好了！")`
4. 回车，正确打印后，输入 `exit()` 退出。

若提示 `command not found`，请检查conda环境是否激活或Python是否安装。

**在 VS Code 中验证（7.2）**

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`，输入：

## 块 16

**7.3 在 PyCharm 中验证**

环境装好了，现在用 PyCharm 跑通第一行代码。你的目标是 vLLM 部署，但任何部署都从“能跑通一个脚本”开始，这一步就是验证你的开发台是否真正可用。

操作三步走：
1. 打开 PyCharm，新建一个项目，在项目里创建文件 `hello.py`。
2. 在编辑区右键点击该文件，选择 `Run 'hello'`。
3. 看下方 **Run 窗口**，如果输出了预期结果，说明环境链路（Python → 解释器 → 运行）已打通。

---

**第八章：常见问题排查（重点看前两条）**

你学习节奏快，直接上高频坑：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install
