# 学习包

## 块 1

**本块学习检查**：请用自己的话解释“为什么没有 Python 解释器，你写的 `.py` 文件就只是一堆文本？”（提示：结合“翻译官”的比喻，说清楚电脑和人类语言的差异即可。）

---

### 1.1 Python 解释器：程序的“翻译官”

你的目标是部署 vLLM，但别急着碰模型——先花2分钟搞懂这个地基。你写的Python代码是给人看的文字，电脑只认0和1。**Python解释器**就是那个“翻译官”，把你的代码逐句翻译成电脑能执行的指令。

**关键点**：没有解释器，你写的 `.py` 文件对电脑来说就是一堆无意义的文本，它根本不知道你想让它干什么。

**动手验证（30秒）**：随便新建一个 `test.py` 文件，里面写 `print("hello")`，然后双击运行——你会发现它要么闪退，要么报错。这就是因为系统还没装“翻译官”，电脑读不懂这堆文字。

**记住这个逻辑**：你写代码 → 解释器翻译 → 电脑执行。后面安装Python，装的就是这个“翻译官”。装好了，你的 `.py` 文件才能被“听懂”。

## 块 2

**IDE 是你的“工作台”**  
记事本也能写代码，但效率极低。IDE（集成开发环境）提供语法高亮、自动补全、运行调试，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

**虚拟环境：项目的“独立房间”**  
假设你同时做两个项目：A 需要 `numpy 1.0`，B 需要 `numpy 2.0`，都装进同一个 Python 会冲突报错。虚拟环境为每个项目创建独立 Python 空间，互不影响。可以用 **conda** 或 Python 自带的 **venv** 来管理，conda

## 块 3

**Windows 上装好 Python，为 vLLM 部署铺路**

你的目标是部署 vLLM，这需要扎实的 Python 环境。别担心，跟着下面三步走，10分钟搞定。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。**这是新手最容易踩的坑，一定要勾上。**

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。如果你想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

## 块 4

你的目标是快速把 vLLM 部署跑起来，而这一步的关键是先把 Python 环境准备好。macOS 上有个常见坑：**系统自带的 Python 别直接用**。现代 macOS（Catalina 10.15 起）不再预装 Python 2，但系统自带的 `python3` 是 Apple 提供的，版本通常较旧，直接依赖它，后面装 vLLM 依赖时容易出问题。应使用官方安装包或 Homebrew 安装最新版 Python。

**方法一：官方安装包（推荐你，省事）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 搞定。
3. 打开终端（Terminal），验证：

```bash
python3 --version
pip3 --version
```

**注意**：macOS 命令行默认**没有** `python` 命令，统一用 `python3`。以后所有命令都写 `python3` / `pip3`，别省那个 `3`，否则会报"命令不存在"或误用系统版本。如果 `pip3` 不存在，可运行 `python3 -m ensurepip` 或 `python3 -m pip install --upgrade pip`。

**方法

## 块 5

**目标**：在 Ubuntu/Debian 上装好 Python 3，为 vLLM 部署铺路。

**三步走**（打开终端，逐条执行）：

1. **更新软件源**  
   ```bash
   sudo apt update
   ```
   让系统知道有哪些新软件包可用。

2. **安装 Python 3 全家桶**  
   ```bash
   sudo apt install python3 python3-pip python3-venv -y
   ```
   - `python3`：解释器本体  
   - `python3-pip`：装第三方库（vLLM 靠它装）  
   - `python3-venv`：建虚拟环境，隔离项目依赖  
   - `-y`

## 块 6

**本块学习检查**：学完后，请用一句话回答：你的 GPU 实训环境应该选哪种操作系统？为什么？

---

### 2.4 操作系统与实训环境该怎么选

你的目标是 vLLM 部署，这是 GPU 实训，环境选择直接决定你能否顺利跑通。记住一条主线：**命令练习和 GPU 实训，都往 Linux 靠拢**。

**如果你是 Windows 用户**，日常写 Python、练命令行时，别用 CMD，启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。它让你在 Windows 里获得接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`。

**如果你是 macOS / Linux 用户**，直接使用本机终端即可，本身就是类 Unix 环境。

**关键一步——GPU 实训（你的 vLLM 部署）**：必须使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

**⚠️ 避坑提醒**：别把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是最权威、最完整、应保留的默认源。

**你现在的目标（vLLM 部署）会装不少大体积依赖，如果网络卡顿，下载会非常折磨。** 但记住：只有网络受限、官方源明显变慢或经常超时时，才临时切到国内镜像。镜像只是“网络不好时的临时加速手段”，不是永久替换——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果某次安装慢，最干净

## 块 8

**本节目标**：把清华镜像设为 pip 默认源，并学会验证与还原。仅当你长期网络受限、每次都要手动加 `-i` 时才做这一步。

**为什么值得做**：你的目标是 vLLM 部署，后续要装大量依赖包。网络不稳时，默认源能省下反复敲参数的时间。但请记住：官方 PyPI 最可靠，网络恢复后建议改回。

---

**Windows 操作（三步）**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开配置目录（类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 新建文件夹 `pip`，在里面新建文本文件 `pip.ini`。
3. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从哪下载。

**macOS / Linux 操作（三步）**

1. 终端执行 `mkdir -p ~/.config/pip`。
2. 执行 `nano ~/.config/pip/pip.conf`。
3. 写入同样的两行内容，按 `Ctrl + O` 保存，`Ctrl + X` 退出。

---

**⚠️ 别乱加 `trusted-host`**：网上教程常让你加一行 `trusted-host = ...`，但除非镜像 HTTPS 证书验证失败，否则不要加——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法证书，正常用不上。只有企业内网自建源或确认来源可信时才临时加。

---

**验证配置**：命令行输入 `pip config list`，看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

**改回官方源**：删除 `pip.ini` / `pip.conf` 文件，或执行 `pip config unset global.index-url`。

---

**本块学习检查**：动手做一遍——按你的系统完成配置，运行 `pip config list` 确认输出，再执行 `pip config unset global.index-url` 还原，最后再次运行 `pip config list` 确认配置已清除。全程约 3 分钟，能帮你把命令记牢。

## 块 9

**目标对齐**：你正为 vLLM 部署做准备，而 VS Code 是你写推理脚本、调试代码的“主战场”。本块只讲一件事：装好它，并让它认得出 Python。

**为什么选 VS Code**：免费、启动快、插件多。你之后写 Python 脚本、改配置文件、敲 Linux 命令，都用得上它。

**安装步骤（3 步）**：
1. 打开 https://code.visualstudio.com，下载对应你系统的安装包。
2. 运行安装程序，一路“下一步”完成。
3. **装插件（关键）**：打开 VS Code，点左侧最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（微软官方，提供语法高亮、调试、智能提示）
   - **Pylance**（让代码提示更聪明）

装完这两个，VS Code 才能识别 `.py` 文件，给你代码补全、报错提示和“一键运行”。

**配置 Python 解释器（2 步）**：
1. 按 `Ctrl + Shift + P`（Mac 用 `Cmd + Shift + P

## 块 10

personalization_action: explain
personalization_reason: 学习目标为vLLM部署，需快速搭建专业IDE；hands-on风格要求步骤精简、可操作。
fact_source_chunk_ids: ["KB-15-04-02"]

---

## 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。内置代码分析、版本控制、数据库工具，适合大型项目开发——vLLM 部署涉及多文件工程，正合适。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开 PyCharm，选 `New Project`。
2. `Location` 选项目保存路径。
3. 在 `

## 块 11

### 5.1 为什么用 Conda

你的目标是部署 vLLM，这意味着后面要面对 PyTorch、CUDA 这些“重依赖”。Conda 的价值在于：它不只是管 Python 包，还能直接管 Python 解释器版本。每个项目一个独立环境，互不干扰，这是深度学习项目的基本盘。

### 5.2 安装 Conda：选 Miniconda

两个发行版二选一：

- **Anaconda**：500MB+，预装 numpy、pandas 等一堆库。适合数据分析、不想手动装包的人。
- **Miniconda**：约 50MB，只带 conda 和基础 Python，要什么装什么。**推荐你选这个**——环境干净，部署 vLLM 时依赖关系更清晰。

### 安装步骤（跟着做）

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包：Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`
3. 运行安装程序，注意系统差异：
   - **Windows**：一路下一步，**勾选** "Add Miniconda3 to my PATH environment variable"，以便在命令行中直接使用 conda。若担心与系统已有 Python 冲突，可仅使用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来敲 conda 命令。
   - **macOS/Linux**：运行 `.sh` 文件，输入 `yes` 同意协议、选安装路径，最后一步**必须选 `yes`** 让安装器执行 `conda init`。这样终端每次打开都会自动被 conda 接管，不用手动改 PATH。
4. **验证安装**：打开一个**新的**终端（Windows 用 Anaconda Prompt 或普通命令行；macOS/Linux 关掉终端重开），输入：

```bash
conda --version
```

看到类似 `conda 24.x.x` 的输出，就成了。

---

**本块学习检查**：打开新终端，运行 `conda --version`，把输出结果记下来。如果提示 `conda:

## 块 12

**动手创建你的第一个虚拟环境**

你的目标是部署 vLLM，而大模型项目常需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

打开终端，执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y` 并回车，等待安装完成

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

你的目标是部署 vLLM，所以环境必须干净、可控。跟着做，别跳步：

1. **激活环境**：在终端运行 `conda activate llm-learn`。这是前提，不激活后面全白搭。
2. **打开命令面板**：在 VS Code 里按 `Ctrl + Shift + P`。
3. **选择解释器**：输入 `Python: Select Interpreter` 并回车。
4. **挑带 conda 的**：在列表里选带有 `conda` 字样的路径，比如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这么做**：VS Code 会自动识别 conda 环境。选完后，你在这个工作区写的所有代码，都会用 `llm-learn` 里的 Python 和已装好的包——包括你后面要用的 vLLM 依赖。选错环境，代码跑不起来，别在这上面浪费时间。

### 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程类似：

1. 打开 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `

## 块 15

### 7.1 在命令行验证

你的目标是部署 vLLM，但任何部署都从“环境能跑通”开始。先激活环境：

```bash
conda activate llm-learn
```

输入 `python` 进入交互模式，看到 `>>>` 提示符后，敲入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车，若正确打印文字，说明 Python 解释器工作正常。按 `exit()` 退出。

### 7.2 在 VS Code 中验证

命令行验证通过后，再用编辑器走一遍流程：新建文件夹 `my-first-project`，用 VS Code 打开，新建 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角小三角

## 块 16

### 7.3 在 PyCharm 中验证

环境装好了，现在用最直接的方式确认它真的能用——在 PyCharm 里跑通第一行代码。三步走：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目里新建一个文件，命名为 `hello.py`。
2. **运行文件**：在左侧文件列表中，**右键点击** `hello.py`，在弹出的菜单中选择 `Run 'hello'`。
3. **查看结果**：看 PyCharm 底部的 **Run 窗口**，如果代码正确，这里会输出你写的内容。

> **动手试试**：在 `hello.py` 里输入 `print("Hello, vLLM!")`，然后运行。看到 Run 窗口输出这行字，就说明你的开发工作台已经打通了。

---

### 第八章：常见问题排查

跑代码时如果卡住，先对照这张表自查：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import`
