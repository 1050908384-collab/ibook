# 学习包

## 块 1

**1.1 Python 解释器：程序的"翻译官"**

想象一下：你写的 Python 代码是给人类看的"中文"，而电脑只懂"0 和 1"的机器语言。**Python 解释器**就是那个随身翻译官——它把你的代码逐句翻译成电脑能执行的指令。

**为什么你必须先懂这个？**

因为你用手机或平板学 LoRA 微调时，写的每一行代码（比如加载模型、调整参数）都要靠解释器来"翻译"给设备执行。没有它，你辛苦写的 `.py` 文件只是一堆没用的文本。

**记住这个核心逻辑：**

- **你写代码** → 人类可读的文字
- **解释器翻译** → 变成电脑能跑的指令
- **电脑执行** → 完成你想要的微调任务

**一个帮你验证理解的检查：**

假设你在手机上写了一个 `hello.py` 文件，里面只有一行 `print("你好")`。请问：如果没有 Python 解释器，双击这个文件会发生什么？

（答案：什么都不会

## 块 2

**你的"工作台"：代码编辑器（IDE）**

用手机或平板学大模型微调，记事本写代码会非常痛苦。**IDE（集成开发环境）**就像你的专属工作台，提供语法高亮、自动补全、运行调试，让写代码像发消息一样顺手。推荐两款：**VS Code**（轻量免费，适合手机/iPad）和 **PyCharm**（专业强大，但更重）。

**项目的"独立房间"：虚拟环境**

做 LoRA 微调时，你可能同时跑多个项目：A 项目要 `numpy 1.0`，B 项目要 `numpy 2.0`，全装一起会冲突报错。**虚拟环境**就是给

## 块 3

**Windows 上安装 Python：为 LoRA 微调铺路**

你在手机/平板上学习，但微调模型最终需要电脑环境。本节先解决 Windows 上的安装，这是你后续跑通代码的基石。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。把鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（比如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)** 即可。

**第二步：运行安装程序（⚠️ 最关键的一步）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**。  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，之后在命令行输入 `python`，系统会提示"不是内部或外部命令"，所有后续操作都无法进行——包括你将来要用的 pip 安装命令。

2. 点击 **"Install Now"**（默认安装）。  
   默认安装路径通常为 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python313`，若用户名含中文或空格，建议自定义安装路径为纯英文无空格目录，如 `C:\Python313`。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车

## 块 4

**macOS 安装 Python 3.10+（手机/iPad 也能懂）**

你的目标是学 LoRA 微调，但设备是手机或平板，没有电脑终端。别急，这节内容先帮你搞懂 macOS 上 Python 的安装逻辑，为以后在云环境或远程服务器上操作打基础。

**核心认知：别用系统自带的 Python**

现代 macOS（Catalina 10.15 起）不再预装 Python 2，但可能带旧版 `python3` 供系统脚本用。**强烈建议自己装最新版**，别依赖系统版本——否则可能遇到版本不匹配的坑。

**安装方法（二选一）**

**方法一：官方安装包（新手推荐）**

1. 在电脑浏览器访问 python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 完成。
3. 打开终端（Terminal），验证：
```bash
python3 --version
pip3 --version
```

**关键提醒**：macOS 上命令行默认**没有** `python` 命令，只有 `python3`。所以以后所有操作都用 `python3` 和 `pip3`，避免"命令不存在"的报错

## 块 5

**Linux 用户（Ubuntu/Debian）安装 Python 3.10+**

Linux 通常自带 Python，但版本可能偏老。在 Ubuntu/Debian 上，用下面三条命令搞定：

```bash
sudo apt update          # 更新软件源
sudo apt install python3 python3-pip python3-venv -y
python3 --version        # 验证安装
```

**关键点**：`-y` 让系统自动确认，不用手动输 `y`；`python3-venv` 是建虚拟环境要用的模块，一次装齐，省得后面补。

**注意**：上述命令安装的是系统默认 Python 版本（如 Ubuntu 20.04 为 3.8，

## 块 6

### 2.4 操作系统与实训环境该怎么选

你只有手机或平板，没有电脑，但目标是学大模型微调（LoRA）。这完全可行，关键是选对环境，别在第一步就卡住。

**先明确你的核心场景**：你的目标是 LoRA 微调，这属于 GPU 实训。根据资料，这类任务应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。因为微调依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**你的实操路径**：用手机/平板远程连接云服务器。你不需要在本地跑训练，只需通过 SSH 客户端（如 Termius）连上云服务器操作。日常练命令、写 Python 脚本，也在云服务器上完成——它就是你的“远程电脑”。

**为什么不推荐其他方案**：资料明确警告，不要用“CentOS + VirtualBox 虚拟机”。CentOS 8 已停止维护，其替代版本 CentOS Stream 8 也已于 2024 年 5 月停止维护，建议使用 Ubuntu LTS；VirtualBox 的 GPU 直通配置复杂且依赖硬件支持，通常不适用于深度学习训练，建议使用云 GPU

## 块 7

**3.1 什么时候才需要镜像源？**

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是最权威、最完整、最应保留的默认源。

**关键认知**：镜像源不是必需品。只有当你网络受限、访问官方源明显变慢或经常超时，才临时切换。镜像是“网络不好时的临时加速手段”，不是永久设置。

**首选做法：临时指定镜像源（推荐）**

如果某次安装慢，最干净的做法是只对这一次安装临时指定镜像，不改动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方源仍是首选；网络恢复或镜像出问题时，无需清理配置。

**本块学习检查**：请

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才做）

你现在用手机或平板学习，网络环境可能不太稳定。如果你**长期**下载包都很慢、几乎每次都要手动加镜像地址，才需要把它写进配置文件，让 pip 默认就走镜像。**但请记住**：官方 PyPI 才是最可靠的来源，网络恢复正常后，建议改回或删掉这个配置。

### 手机/平板上怎么做？

虽然你用的是 iPad 或手机，但原理和电脑一样——只是没有图形界面时，你需要用终端类 App（比如 Termius 或 iSH）来执行命令。下面以通用命令行为准：

**第一步：创建配置文件夹**

打开终端，执行：

```bash
mkdir -p ~/.config/pip
```

这条命令会创建一个名为 `pip` 的隐藏文件夹，用来存放 pip 的配置文件。

**第二步：创建并编辑配置文件**

```bash
nano ~/.config/pip/pip.conf
```

这会打开一个文本编辑器。如果提示 `nano` 不存在，可以先安装它（iSH 中执行 `apk add nano`），或者改用 `vi` 编辑器，也可以直接用 `echo` 命令写入配置：

```bash
echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.config/pip/pip.conf
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。这里用的是清华镜像源。

如果用 `nano` 或 `vi` 编辑，保存方法：`nano` 按 `Ctrl + O` 确认保存，再按 `Ctrl + X` 退出；`vi` 按 `Esc` 后输入 `:wq` 保存退出。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

### 验证配置

## 块 9

**本块学习检查**（读完后，请用手机完成以下验证）

打开 VS Code 的扩展面板（四个方块图标），搜索并安装 **Python** 和 **Pylance** 两个插件。安装完成后，新建一个 `.py` 文件，观察是否出现彩色语法高亮——如果出现，说明插件生效。

---

**为什么选 VS Code**：它是微软出品的免费编辑器，启动快、插件丰富。你只有手机/iPad，无法本地跑大模型，但写微调脚本、改配置文件、看训练日志都靠它。后续学 Linux 命令、写脚本时也能复用。

**安装步骤（电脑版）**：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包（Windows/macOS/Linux 选对应版本）。
2. 运行安装程序，一路“下一步”即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（

## 块 10

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具，适合大型项目开发——这正是你未来做 LoRA 微调时需要的。

**安装步骤**（手机/iPad 上先下载到电脑，再操作）：

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（免费版）
3. 运行安装程序，勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：

1. 打开 PyCharm，选 `New Project`
2. `Location` 选项目路径（**不要含中文**）
3. `Interpreter` 选 `Previously configured interpreter`，再选系统已装的 Python 3.10+

**本块学习检查**：在 PyCharm 中新建一个项目，确认路径无中文，并截图记录 Interpreter 选择的 Python

## 块 11

### 5.1 为什么用 Conda

你在手机或平板上学习，目标是大模型微调（LoRA）。虽然当前设备没有 CUDA，但后续你会在云端或租用 GPU 服务器上跑代码，那时环境管理就是第一道坎。

虚拟环境能为每个项目创建独立的 Python 空间。**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于深度学习（需要 PyTorch、CUDA 等复杂依赖），conda 几乎是必备工具——它能帮你把不同项目的依赖彻底隔开，避免“装一个包搞坏另一个项目”的灾难。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要什么装什么包。你之后做 LoRA 微调，项目依赖很明确，Miniconda 更合适。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Min

## 块 12

你正在手机或平板上学习，目标是大模型微调（LoRA）。记住：**不同项目需要不同版本的 PyTorch 和 CUDA**，而虚拟环境就是为每个项目准备独立“工作台”的方法。现在学会，以后直接复用。

在终端输入以下命令并回车：

```bash
conda create -n llm-learn python=3.10
```

**这条命令在做什么？**
- `conda create`：创建环境的指令
- `-n llm-learn`：给环境命名为 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定该环境使用 Python 3.10 版本

执行后，终端会显示将要安装的包列表，并询问：

```bash

## 块 13

### 5.4 激活与退出环境

你正在为“LoRA微调”搭建隔离的Python工作台。想象Conda环境是独立的“房间”，互不干扰。

**进入房间（激活环境）**：
```bash
conda activate llm-learn
```
激活后，命令行前出现 `(llm-learn)` 标志，表示后续 `python` 和 `pip` 命令都作用于这个环境，不会影响系统或其他项目。

**验证房间版本**：
```bash
python --version
```
应显示 `Python 3.10.x`。

**离开房间（退出环境）**：
```bash
conda deactivate
```

### 5.5 查看与管理环境

**查看所有环境**：
```bash
conda env list
```
带 `*` 的是当前激活的环境。

**删除环境**（如需清理）：
```bash
conda remove -n llm-learn --all
```

---

**本块学习检查

## 块 14

## 6.1 在 VS Code 中连接你的 Conda 环境

你已经创建好了 `llm-learn` 环境（里面装着 Python 和之后微调要用的库），现在要让 VS Code 用上它。

**操作步骤：**

1. **先激活环境**：在终端里输入 `conda activate llm-learn`。
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`（手机/平板外接键盘同样适用）。
3. **选择解释器**：输入 `Python: Select Interpreter` 并回车。
4. **挑选 conda 环境**：在列表中选择带有 `conda` 字样的路径，例如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这么做**：VS Code 会自动识别你电脑上的 conda 环境。选中后，你在当前工作区写的所有代码，都会默认使用 `llm-learn` 里的 Python 和已安装的包——这是后续跑 LoRA 微调脚本的基础。

---

## 6.2 在 PyCharm 中配置 Conda 环境

如果你用的是 PyCharm，流程稍有不同：

1. 打开 `File` → `Settings`（Mac 上是 `PyCharm` → `Preferences`）。
2. 左侧找到 `Project: 你的项目名`

## 块 15

### 在命令行验证

你只有手机或平板，没有电脑，这完全没问题——验证环境是否装好，命令行是最轻量的方式，而且你以后做LoRA微调时，也常会用到它。

**操作步骤：**

1. **激活环境**：输入 `conda activate llm-learn` 并回车。这相当于“进入”你为学习大模型专门准备的工作间。
2. **进入Python**：输入 `python` 并回车。你会看到 `>>>` 提示符，说明Python已就绪。
3. **运行第一行代码**：在 `>>>` 后输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
   然后回车。
4. **

## 块 16

**本块学习检查**：在 PyCharm 中完成 `hello.py` 的运行验证，并观察 Run 窗口的输出结果。

---

## 7.3 在 PyCharm 中验证

你的目标是 LoRA 微调，但任何大模型训练都从“跑通第一行代码”开始。现在，我们用 PyCharm 验证环境是否真正可用。

**操作步骤（共三步）**：

1. **新建项目**：打开 PyCharm，创建一个新项目（项目路径不要包含中文或空格）。
2. **创建 `hello.py` 文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3. **运行文件**：在编辑区右键点击 `hello.py` → 选择 `Run 'hello'`。

**预期结果**：PyCharm 下方的 **Run 窗口**会显示程序输出。如果看到输出内容，说明你的 Python 解释器、项目配置和运行链路全部正常，可以进入后续学习。

---

## 常见问题排查（速查表）

如果你在验证时遇到问题，对照下表定位：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选 “Modify”，勾选 “Add Python to environment variables” |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda

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

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0205-001
以下哪个是装饰器的正确语法？

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0205-003
以下代码输出什么？
```python
def my_decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

### Q-KB0205-006
以下关于生成器的说法，**错误**的是？
