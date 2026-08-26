# 学习包

## 块 1

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，这需要你亲手操作代码。但在跑通那些酷炫的模型之前，我们得先把“地基”打牢。

**别急着下载安装包，先花 2 分钟搞懂一个核心概念：Python 解释器。**

你可以把它想象成一个“翻译官”。你写的 Python 代码是人类语言，但电脑只懂 0 和 1。这个“翻译官”的工作，就是实时把你的代码“翻译”成电脑能执行的指令。

**为什么这对你很重要？**
因为你做 LoRA 微调，要写、要跑大量的 `.py` 脚本。如果没有这个“翻译官”，你写的文件对电脑来说就是一堆没用的文本，根本跑不起来。

**动手验证一下（约1分钟）：**
打开你的终端（Mac/Linux）或命令提示符（Windows），输入 `python` 然后回车。如果你看到类似 `>>>` 的提示符，说明“翻译官”已经就位，随时准备为你服务。输入 `exit()` 回车即可退出。

搞懂了它，我们下一步就去请这位“翻译官”正式入驻你的电脑。

## 块 2

**本块学习检查**：打开 VS Code，新建一个 `test.py`，输入 `print("hello")`，观察语法高亮是否出现。

---

你的目标是 LoRA 微调，这需要跑模型训练脚本。记事本写代码能跑，但调试起来会让人崩溃。**IDE（集成开发环境）**就是你的“工作台”，提供语法高亮、自动补全、运行调试，写代码像用 Word 一样舒服。

推荐两款：**VS Code**（轻量免费，适合快速上手）和 **PyCharm**（专业强大）。作为新手，先用 VS Code 就够了。

接下来是**虚拟环境**——项目的“独立房间”。想象你同时做两个项目：A 需要 `numpy 1.0`，B 需要 `numpy 2.0`，装一起就冲突报错。虚拟环境为每个项目创建独立 Python 空间，互不干扰。我们用 **conda** 管理，这是目前

## 块 3

**目标：LoRA微调的第一步——先把Python装好**

你的目标是LoRA微调，而所有微调工具都跑在Python上。所以这一块，我们只干一件事：在Windows上把Python装好、验证能用。全程跟着做，5分钟搞定。

**第一步：下载安装包**

1. 打开浏览器，访问 `https://www.python.org`
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐Windows版本（比如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这一步不勾，后面你在命令行敲 `python` 会提示"不是内部或外部命令"，所有后续操作全卡住。勾上它，系统才知道去哪找Python。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续报错。如果你想改路径，记住：**路径中不能有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --

## 块 4

### 2.2 macOS 用户：装好 Python，为 LoRA 微调铺路

你的目标是 LoRA 微调，这需要 Python 环境。macOS 上有个坑：**macOS 12.3 起不再预装 Python 2**，且系统自带的 `python3` 可能为旧版，仅供系统脚本使用。**强烈建议自行安装最新版 Python**，别依赖系统版本。

**方法一：官方安装包（推荐新手，操作最直接）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入以下命令验证：

```bash
python3 --version
pip3 --version
```

**关键提醒**：在 macOS 上，命令行默认**不提供** `python` 命令（macOS 12.3 起不再存在）。请**始终用 `python3` / `pip3`**，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已装 Homebrew（macOS 包管理器），在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，第一步就是把 Python 环境弄扎实。Ubuntu/Debian 通常自带 Python，但版本可能偏老，直接装新的最省心。

打开终端，依次执行下面三行命令：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**命令拆解**：第一行更新软件源，让系统知道最新软件包；第二行安装 Python 3、pip（装库的工具）和 venv（虚拟环境模块），`-y` 表示自动确认，不用你手动输 `y`；第三行验证安装成功，会显示版本

## 块 6

**本块学习检查**：读完本节后，请你在终端里执行 `uname -a`，观察输出结果，并判断当前环境属于“日常写 Python”还是“GPU 实训”场景。

---

### 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选择直接决定后面能不能顺利用上 GPU。别纠结，按场景对号入座：

**场景一：日常写 Python、练命令行（Windows 用户）**
启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**场景二：日常写 Python（macOS / Linux 用户）**
直接用本机终端即可，本身就是类 Unix 环境，无需额外配置。

**场景三：LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**
使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu

## 块 7

**本块学习检查**：运行 `pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`，观察是否成功安装且速度更快；再运行 `pip install numpy`，确认仍走官方源。

---

你要做 LoRA 微调，装 `transformers`、`peft` 这类包时，`pip` 默认从官方 PyPI（`pypi.org`）下载——这是最权威、最完整的源，**永远首选**。

但如果你网络受限、下载明显变慢或经常超时，才临时换国内镜像。记住：**镜像是网络不好时的临时加速手段，不是永久设置**。

最干净的做法是**只对某一次安装临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

推荐这种方式，因为它不影响你平时的默认行为——官方源仍是默认。哪天网络恢复

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才用）

你正在为 LoRA 微调做准备，这意味着后面会频繁用 pip 安装 HuggingFace 生态的库（比如 transformers、peft）。如果网络长期不给力，每次手动指定镜像确实烦人——那就把它写进配置，一劳永逸。

**但先记住一句话**：官方 PyPI 永远是最可靠的。这个配置只在长期网络受限时用，网络恢复后建议改回或删除。

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车——会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在这个文件夹里**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹里**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：以后默认从这里下载包。

### macOS / Linux 操作步骤

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样的内容（同上），然后 `Ctrl + O` 保存，`Ctrl + X` 退出。

### ⚠️ 关于 `trusted-host`，别乱加

网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——加了它，pip 会跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法的 HTTPS 证书，正常用**不需要** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加。

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'

## 块 9

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，而这一切都始于一个趁手的代码编辑器。VS Code 就是那个“轻量、免费、推荐”的选项，它不会拖慢你快速学习的节奏。

**为什么选它**：VS Code 启动快、插件丰富，不仅写 Python 顺手，以后折腾 Linux 命令行、改配置文件都用得上。对你这种动手实操型的学习者来说，它足够灵活，不会在工具上浪费你宝贵的 2.5 小时。

**安装步骤（快进版）**：

1.  直接访问 `https://code.visualstudio.com`，下载对应你操作系统的安装包。
2.  运行安装程序，一路“下一步”到底，不用改任何设置。

**关键一步：装插件**：
打开 VS Code，点击左侧活动栏最下方的**四个方块图标**（扩展），搜索并安装这两个：
-   **Python**（微软官方出品）：提供语法高亮、

## 块 10

### 4.2 方案二：PyCharm Community（免费、轻量）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费，但功能较专业版精简，缺少数据库工具、科学工具等高级功能。它内置了代码分析、版本控制等基础功能，适合大型项目开发。

**安装步骤：**

1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add launchers dir to the PATH`（命令行可直接启动 PyCharm）

**首次启动配置：**

1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，然后选择系统安装的 Python 3.10

## 块 11

### 5.1 为什么用 Conda

你之前已经知道，虚拟环境可以为每个项目创建独立的 Python 空间。而 **Conda** 是目前最主流的虚拟环境管理工具——它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。这一点对你特别重要：因为你接下来的目标是 **LoRA 微调**，这需要用到 PyTorch、CUDA 等复杂依赖，而 conda 几乎是处理这类依赖的必备工具。简单说，Conda 就是帮你把每个项目的“工具箱”彻底隔离开，避免不同项目之间的包版本打架。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装 Conda 有两个选择，区别如下：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包。对你这种以项目开发（LoRA 微调）为目标的学习者来说，这更合适。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   -

## 块 12

**本块学习检查**：执行下面的命令后，在终端输入 `y` 并回车，确认环境创建成功。

---

你的目标是 LoRA 微调，而微调不同项目时，需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**（按你的操作习惯，先看结果再看细节）：

- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后

## 块 13

**本块学习检查**：打开终端，依次运行下面的命令，观察每一步输出，确认你已进入独立环境并能安全退出。

**1. 进入你的“独立房间”**
运行：
```bash
conda activate llm-learn
```
激活后，命令行前面会出现 `(llm-learn)` 标志。看到它，就说明你后续的 `python` 和 `pip` 命令都只作用在这个环境里，不会干扰系统或其他项目——这正是你之后做 LoRA 微调时需要的干净空间。

**2. 验证房间里的 Python 版本**
运行：
```bash
python --version
```
应该显示 `Python 3.10.x`。确认版本无误，再继续下一步。

**3. 退出房间，回到系统默认环境**
运行：
```bash
conda deactivate
```
退出后，`(llm-learn)` 标志会消失，命令行恢复原样。

**4. 查看你有哪些环境（确认刚才

## 块 14

## 6.1 VS Code 配置 Conda 环境

既然你的目标是 LoRA 微调，先确保环境是对的。跟着做：

1. 在终端里激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表里选带有 `conda` 字样的项（形如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这么做**：VS Code 会自动识别 conda 环境。选好后，你在当前工作区写的所有代码都会使用 `llm-learn` 里的 Python 和已装好的包——后续做 LoRA 微调时，依赖才不会乱。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程也很快：

1. 打开 PyCharm，点 `File` → `Settings`（Mac 上是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径里选 conda 环境的 Python 可执行文件：

## 块 15

**本块学习检查**：完成下面两步，确认你的 Python 工作台已就绪。

你的目标是 LoRA 微调，而微调离不开 Python 环境。现在就用最直接的方式验证它。

**第一步：命令行验证**
1. 打开终端，输入 `conda activate llm-learn` 激活环境。
2. 输入 `python` 回车，看到 `>>>` 提示符。
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 回车。
4. 看到文字打印出来，输入 `exit()` 退出。

**第二步：VS Code 验证**
1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建 `hello.py`，输入 `

## 块 16

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，而这一切都建立在你的开发环境能顺畅运行的基础上。既然你喜欢动手实践，我们就不空谈理论，直接来验证你的“工作台”是否已经搭好。

## 7.3 在 PyCharm 中验证：跑通第一行代码

这一步是给你信心的一步。环境搭建得再好，跑不通代码都是白搭。我们用一个最经典的方式确认一切正常：

1.  **新建项目**：打开 PyCharm，创建一个新的纯 Python 项目。
2.  **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3.  **编写代码**：在 `hello.py` 中输入 `print("Hello, World!")`。
4.  **运行**：在编辑区**右键点击**鼠标，选择 **`Run 'hello'`**。
5.  **验证**：看 PyCharm 底部的 **Run** 窗口，如果输出了 `Hello, World!`，恭喜你，你的 Python 开发工作台已经可以正常工作了！

---

## 7.4 常见问题排查（你的“避坑”指南）

在动手过程中，你几乎肯定会遇到下面某个问题。别慌，对照着解决就行。

| 问题现象 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| 输入 `python` 提示“不是内部或外部命令” | 安装 Python 时没勾选 “Add to PATH” | 重新运行 Python 安装程序，选 **“Modify”**，然后勾选 **“Add Python to environment variables”**。 |
| `pip install` 速度极慢或超时 | 网络问题，访问官方源太慢 | 临时用：在命令后加 `-i` 指定国内镜像（如清华源）。长期用：把镜像设为默认源。 |
| `conda` 命令找不到 | 没在正确的终端里操作 | **Windows**：用开始菜单的 **`Anaconda Prompt (Miniconda3)`
