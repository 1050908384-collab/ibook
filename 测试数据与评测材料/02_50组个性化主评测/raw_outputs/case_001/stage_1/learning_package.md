# 学习包

## 块 1

好的，我们直接开始。既然你的目标是LoRA微调，那第一步就是让电脑能“听懂”你写的指令。这一节，我们只搞懂一个东西：**Python解释器**。

想象一下，你是一位导演（写代码的人），Python解释器就是你的翻译官。你写的剧本（`.py`文件）是人类语言，但电脑演员（CPU）只懂0和1的机器语言。翻译官的工作，就是把你写的每一句“台词”实时翻译给电脑听，电脑才能照做。

**所以，没有这个翻译官，你写的代码文件就只是一堆普通的文本，电脑根本不会理它。**

你可能会问：“那我直接跟电脑说机器语言不行吗？”理论上可以，但那就像用二进制代码写小说，效率极低且容易出错。我们写Python，就是为了用接近人类思维的方式表达逻辑，然后把翻译的苦活累活交给解释器。

**一个关键认知**：你的电脑是核显笔记本，没有独立GPU。这没关系，因为**Python解释器本身不挑硬件**，它只是负责“翻译”。你之后做LoRA微调时，真正需要强大算力的是“训练”这个环节，而我们现在搭建的“翻译官”，是所有后续工作的地基。

**本块学习检查**：
请用自己的话，向一位完全不懂技术的朋友解释：为什么我们写Python代码，还需要额外安装一个“解释器”？把它说清楚，就说明你真正理解了。

## 块 2

好的，我们开始吧。既然你的目标是学习 LoRA 微调，那么一个得心应手的“工作台”就是你的第一块基石。

**1.2 代码编辑器（IDE）：你的“工作台”**

记事本也能写代码，但效率极低。**IDE（集成开发环境）** 提供了语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。对于零基础的你，推荐 **VS Code**，它轻量免费，上手快，足够支撑你后续的微调学习。

**1.3 虚拟环境：项目的“独立房间”**

想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径为 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python313`（版本号随安装版本变化）。路径中应避免中文和空格，但用户名本身可能包含空格，需注意。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python 3.

## 块 4

### macOS 用户：安装 Python 3.10+

你可能会听说“Mac 自带 Python”，但请注意：**现代 macOS（Catalina 10.15 至 Monterey 12.2）仍预装 Python 2.7，但自 macOS Monterey 12.3 起已移除 Python 2**。部分系统可能带有供系统脚本使用的旧版 `python3`，且系统自带的 Python 3 版本可能较旧，**强烈建议自行安装最新版 Python**，避免依赖系统自带版本——这对你后续做 LoRA 微调至关重要，因为旧版本可能缺少必要的库支持。

#### 方法一：官方安装包（推荐新手）

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入以下命令验证安装：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`？** 在现代 macOS 上，命令行默认**并不提供** `python` 命令。系统与官方安装包统一使用 `python3`，因此请始终用 `python3` / `pip3`，避免踩到“命令不存在”或误用系统版本的坑。

#### 方法二：Homebrew（适合已有 Homebrew 的开发者）

如果你已经

## 块 5

**Linux 用户（Ubuntu/Debian）安装 Python**

你的电脑是 Linux 系统（Ubuntu 或 Debian 系列）吗？别担心，虽然系统通常自带 Python，但版本可能偏老，不适合后续的模型微调工作。我们一步步来更新它。

打开终端（Terminal），依次输入以下三条命令，每输完一条按回车：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**这三条命令在做什么？**

- 第一条：更新软件源，让系统知道最新可用的软件版本。
- 第二条：安装 Python 3、pip（Python 的包管理工具）和 venv（虚拟环境模块

## 块 6

## 2.4 操作系统与实训环境该怎么选

作为零基础文科背景、目标是大模型微调的你，选对环境比选“最好”的环境更重要。这里按你的实际场景分两步走：

**第一步：日常练 Python（用你的 Windows 笔记本）**

先别急着装虚拟机，推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它是在 Windows 里运行的一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致。你现在练的命令行习惯，将来直接迁移到云端服务器，不用重学。安装只需管理员 PowerShell 执行 `wsl --install` 即可。

> 如果你用的是 macOS 或 Linux，直接使用本机终端即可，本身就是类 Unix 环境，无需额外配置。

**第二步：LoRA 微调实训（你目前无独立 GPU）**

你的笔记本只有核显，跑不了 GPU 训练。LoRA/QLoRA 微调这类任务，应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态的支持最成熟、文档最全。

**⚠️ 不推荐的做法**：不要把 **“CentOS + VirtualBox

## 块 7

### 3.1 镜像源：什么时候用，怎么用

**先理解背景**：`pip` 是 Python 的“软件安装管家”，帮你下载第三方库（比如 `numpy`）。默认它从官方源（`pypi.org`）下载——这是**最权威、最完整**的选择，**永远应该优先保留**。

**什么时候才需要镜像源**：只有当你**网络受限**、访问官方源**明显变慢**或**经常超时**时，才临时切换。记住：**镜像是“网络不好时的加速手段”，不是永久设置**。

**推荐做法（临时指定）**：如果某次安装慢，只对**这一次**临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响你平时的默认行为，官方源仍是首选；网络恢复或镜像出问题时，无需清理

## 块 8

## 3.3 进阶：把镜像设为默认（仅在长期网络受限时）

这一步是“一劳永逸”的配置，但请先记住一个前提：**只有当你长期处于网络受限环境、几乎每次下载都要用镜像时，才这样做**。官方 PyPI 始终是最可靠的来源，网络恢复后建议改回或删除配置。

### Windows 用户操作步骤

1. 按 `Win + R`，输入 `%APPDATA%` 并回车，会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

### macOS / Linux 用户操作步骤

1. 打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样的内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://

## 块 9

## 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，既要写 Python 代码，也可能要改配置文件、看训练日志，它都能胜任。

**安装步骤**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：

- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：

1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac

## 块 10

## 4.2 方案二：PyCharm Community（免费、轻量）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制等功能，适合大型项目开发——你之后做 LoRA 微调时，项目文件会比较多，这类工具能帮你理清结构。注意：社区版不包含数据库工具，如需该功能请考虑专业版。

**安装步骤**（文科背景不用怕，跟着点就行）：

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费那个）
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Update PATH variable (restart needed)`（命令行可直接启动 PyCharm）

**首次启动配置**：

1. 打开 PyCharm，选择 `New Project`
2. 在 `Location` 处选择项目存放路径（**路径中

## 块 11

好的，同学，欢迎来到环境准备的关键一步。我知道你完全没有编程经验，而且是从文科转行来做大模型微调，这很酷！别担心，我们一步步来。这一节，我们只解决一个问题：**为什么你的电脑需要一个叫 Conda 的“工具箱”**。

想象一下，你以后要做的“大模型微调”，就像是在一个非常精密的实验室里做实验。这个实验室里，不同的项目（比如用 LoRA 微调不同的大模型）需要不同的“试剂”（也就是 Python 版本和各种软件包）。如果所有实验都在同一个脏乱的桌子上做，试剂很容易混在一起，实验就全乱了。

**Conda** 就是帮你管理这个实验室的“智能储物柜”。它最厉害的地方在于，不仅能给每个项目创建独立的“实验台”（虚拟环境），还能直接管理“实验台”上用的“基础试剂”——也就是 Python 解释器本身的版本。对于你后续要学习的深度学习（需要 PyTorch 等复杂依赖），用 Conda 来管理几乎是必备技能，能帮你省去无数“环境冲突”的烦恼。

既然我们确定了要用 Conda，那具体选哪个版本呢？这里有两个选择：**Anaconda** 和 **Miniconda**。

- **Anaconda** 像一个“全家桶”，体积大（通常超过 800MB），预装了大量科学计算库。适合不想手动装包、做数据分析的人。
- **Miniconda** 像一个“精简包”，体积小（约50MB），只包含 Conda 本身和基础的 Python。你需要什么，就自己动手装什么。

**我推荐你选择 Miniconda**。因为你的目标是做项目开发，保持环境干净、按需安装，是更专业的习惯，也能让你更清楚自己每一步在做什么。

安装过程很简单，跟着做就行：

1.  打开安装包下载页面：`https://docs.conda.io/en/latest/miniconda.html`。
2.  下载对应你系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3.  运行安装程序，这里有个**特别重要的提醒**：
    - **如果你用 Windows**

## 块 12

**创建你的第一个虚拟环境**

这一步是为后续大模型微调打基础。不同项目可能需要不同版本的 PyTorch 和 CUDA，虚拟环境能让每个项目“各用各的”，互不干扰。现在学会，以后直接复用。

在终端中执行以下命令：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在做什么？** 拆开看：

- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名为 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境内安装 Python 3.10

执行后，终端会列出准备安装的包，并询问：

```bash
Proceed

## 块 13

## 5.4 激活与退出环境：进入你的“独立房间”

想象一下，你正在为“LoRA微调”这个项目搭建一个专属工作间。Conda环境就是这样一个独立房间——里面只放这个项目需要的工具，不会弄乱你电脑的其他地方。

**进入房间（激活环境）**

在终端输入：

```bash
conda activate llm-learn
```

输入后，注意看命令行最前面会出现 `(llm-learn)` 标志。这个标志就像房间门口的门牌，告诉你：现在你已经在“llm-learn”这个专属工作间里了。此后，你输入的所有 `python` 和 `pip` 命令，都只在这个房间里生效，不会影响系统或其他项目。

**确认房间配置（验证Python版本）**

进入房间后，可以检查一下工具是否齐全：

```bash
python --version
```

屏幕上应该显示 `Python 3

## 块 14

**6.1 VS Code 配置 Conda 环境**

打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境。选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

**6.2 PyCharm 配置 Conda 环境**

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选择 `Project: 你的项目名` → `Python Interpreter`。点击齿轮图标 → `Add...`。左侧选择 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

---

**第七章：验证——运行你的第一行代码**

环境搭好后，必须验证一遍，确保每个环节都通。

**本块学习检查**：在 VS Code 中新建一个 `.py` 文件，输入 `print("hello")`

## 块 15

## 7.1 在命令行验证

你不需要图形界面，命令行就是你的“对话窗口”。先激活环境：

```bash
conda activate llm-learn
```

输入 `python` 进入交互模式，看到 `>>>` 提示符后，逐字输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车，若正确打印出这句话，说明环境已通。输入 `exit()` 退出。

## 7.2 在 VS Code 中验证

新建文件夹 `my-first-project`，用 VS Code 打开。新建文件 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角（或按 `Ctrl + F5`），下方终端应输出 `Hello from VS

## 块 16

好的，学习者。我们已经把开发工作台搭建好了，现在到了最激动人心的时刻：运行你的第一行代码。这一步就像按下新车的启动键，确认一切正常，准备出发。

在 PyCharm 中验证非常简单，跟着做就行：

1.  **新建项目并创建文件**：打开 PyCharm，新建一个项目。在项目里，新建一个名为 `hello.py` 的 Python 文件。
2.  **运行文件**：在编辑区右键点击 `hello.py` 文件，在弹出的菜单中选择 `Run 'hello'`。
3.  **查看结果**：PyCharm 窗口底部会打开一个 `Run` 面板，你的代码输出结果就会显示在这里。

看到输出结果，就说明你的开发环境完全打通了，可以开始写真正的代码了。

---

**遇到问题？别慌，这是新手必经之路。**

环境配置偶尔会闹点小脾气，这里有一份“故障排除”指南，帮你快速定位问题：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时使用 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名` 指定清华镜像源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误

## 测评题目

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
```

### Q-KB0205-005
请写一个**带参数的装饰器** `retry(times=3)`，使被装饰的函数在抛出异常时自动重试指定次数。

### Q-KB0204-003
请写代码：读取 `data.txt` 文件，捕获文件不存在的异常，若不存在则打印 "文件未找到"。

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？
