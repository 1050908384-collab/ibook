# 学习包

## 块 1

**本块学习检查**：读完本节后，请合上资料，用自己的话说清楚：为什么没有解释器，你的 `.py` 文件就“只是一堆文本”？说不上来就回头再看一遍。

---

### 1.1 Python 解释器：程序的"翻译官"

你的目标是用 LoRA 微调模型，但先别急着碰模型——所有 AI 工具都得靠 Python 跑起来，而 Python 代码本身电脑看不懂。

**一句话理解**：你写的是人话，电脑只认 0 和 1。**Python 解释器**就是夹在中间的"翻译官"，把你的代码逐句翻译成电脑能执行的指令。

**关键点**：没有解释器，你写的 `.py` 文件就只是一堆普通文本，电脑不会执行它。

**给你的实操提示**：后面装环境时，你会看到"安装 Python"这个步骤——装的就是这个"翻译官"。装好它，你的代码才有机会被电脑听懂。现在不用深究细节，先记住这个角色定位，下一步我们看它怎么工作。

## 块 2

## 你的“工作台”与“独立房间”：为 LoRA 微调铺路

你马上要动手做 LoRA 微调，这需要写代码、跑实验。用记事本写代码？效率太低。**IDE（集成开发环境）**就是你的“工作台”，提供语法高亮、自动补全、运行调试，让你写代码像用 Word 一样顺手。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个装上即可。

接下来是虚拟环境——你的“独立房间”。微调 LoRA 时，你可能会同时折腾多个项目：一个要 `numpy 1.0`，另一个要 `numpy 2.0`，全装一起就冲突报

## 块 3

### 2.1 Windows 用户：装好 Python，为 LoRA 微调铺路

你学 LoRA 微调，第一步就是把 Python 这个“地基”打牢。跟着做，10 分钟搞定。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（比如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这一步不勾，后面你在命令行敲 `python` 会提示“不是内部或外部命令”，所有命令都跑不了。勾上，一劳永逸。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径通常为 `C:\Users\<用户名>\AppData\Local\Programs\Python\Python313`，若用户名含中文或路径含空格，建议自定义安装路径为纯英文无空格目录。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入

## 块 4

### 你的目标：为 LoRA 微调铺好路

既然你的目标是 LoRA 微调，且喜欢动手操作，那环境搭建这一步必须稳。macOS 上装 Python 有个关键点：**别用系统自带的旧版 `python3`**，自己装一个干净的。

**方法一：官方安装包（推荐你，省心）**

1. 打开 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点 "Continue" → "Install" 装完。
3. 打开终端（Terminal），输入下面两行验证：

```bash
python3 --version
pip3 --version
```

看到版本号就成功了。

**为什么用 `python3` 而不是 `python`？** 现代 macOS 默认**没有** `python` 命令，只有 `python3`。以后你跑 LoRA 脚本、装 PyTorch，全用 `python3` 和 `pip3`，别混用。

**方法二：Homebrew（如果你已经装了）**

终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：在终端运行 `python3 --version`，把输出结果记下来。如果提示 "command not found"，说明没装好，回头重看方法一。

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是跑 LoRA 微调，但 Linux 自带的 Python 可能太老，装不上新版依赖。别急，三条命令搞定：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**拆开看**：第一条刷新软件源，让系统知道最新版本；第二条一次装齐 Python、pip 和虚拟环境模块（`-y` 免去手动确认）；第三条验证版本，看到输出就成功了。

**本块学习检查**：运行完三条命令后，把 `python3 --version` 的输出记下来——如果版本号低于 3.10，

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是跑通 LoRA 微调，那环境选型就一句话：**跟着任务走，别贪多**。

**日常练命令、写 Python——Windows 用户**  
直接启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里拥有一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`，一条命令搞定，别在 CMD 里浪费时间。

**macOS / Linux 用户**  
直接用本机终端，本身就是类 Unix 环境，无需额外配置。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**  
用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 避坑**：别把 **"CentOS + VirtualBox 虚拟机"** 当最佳方案。CentOS 8 已停止维护（2021 年 EOL），CentOS Stream 8

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

**什么时候才用镜像？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是“网络不好时的临时加速手段”，不是永久替换**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它

## 块 8

## 3.3 进阶：把镜像设为默认（仅限长期网络受限）

你正在为 LoRA 微调做准备，后面会频繁用 pip 安装 PyTorch、Transformers 等大包。如果网络长期不给力，每次手动加 `-i` 太烦，那就把它写进配置文件，一劳永逸。

**但先记住**：官方 PyPI 永远是最可靠的。网络恢复后，建议改回或删掉这个配置。

---

### Windows 操作（跟着做，3 分钟搞定）

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开配置文件夹（路径类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 在里面**新建文件夹**，命名为 `pip`。
3. 进入 `pip` 文件夹，**新建文本文件**，命名为 `pip.ini`（注意扩展名是 `.ini`，不是 `.txt`）。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**这行什么意思**：`index-url` 就是告诉 pip：“以后默认从清华这个地址下载包。”

---

### macOS / Linux 操作

打开终端，依次执行：

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

在打开的编辑器里写入同样的两行：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

按 `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### ⚠️ 别乱加 `trusted-host`

网上很多教程会让你顺手加一行 `trusted-host = ...`。**除非镜像真的报 HTTPS 证书错误，否则别加**——加了等于让 pip 跳过 TLS 安全校验，传输不再加密。清华、阿里这些主流镜像都有合法证书，正常用**不需要**。只有企业内网自建源、或确认来源可信且确实报证书错时，才临时加。若确需添加，正确写法是在命令行使用 `--trusted-host pypi.tuna.tsinghua.edu.cn` 临时指定，或写入配置文件对应小节

## 块 9

**本块学习检查**：安装完 VS Code 和 Python 插件后，新建一个 `test.py` 文件，输入 `print("hello")`，按右上角运行按钮，看能否在下方终端输出 `hello`。

---

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调要写 Python 脚本、改配置文件，它都能胜任。

**安装步骤**（跟着做，5分钟搞定）：

1. 打开 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路“下一步”即可。

**必装插件**（关键一步）：

打开 VS Code，点击左侧活动栏最下方的**四个方块图标**（扩展），搜索并安装：

- **Python**（微软官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（让代码提示更智能）

装完这两个，VS Code 才能识别 `.py` 文件，给你代码补全、错误提示和一键运行功能

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具等功能，适合大型项目开发——对 LoRA 微调这类需要频繁调试代码的任务，能帮你更快定位问题。

**安装步骤**（跟着做，5分钟搞定）：
1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，务必勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**，否则后续可能报错）。
3. `Interpreter`

## 块 11

## 5.1 为什么用 Conda

你马上要开始 LoRA 微调了，这意味着要装 PyTorch、CUDA 这些"重依赖"。如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 就是来解决这个问题的——它能为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对于深度学习，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB），预装了大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（安装包约 50-100MB，安装后占用数百 MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境干净，后面做 LoRA 需要什么包再装什么，不拖泥带水。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python 冲突。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**。这样每次打开终端 conda 会自动接管，而不是靠硬加 PATH。
4. **验证安装**：打开一个**新的**终端

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：你要做 LoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后，终端会列出将要安装的包，并

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

## 块 14

**你的目标：** 用 VS Code 跑通 LoRA 微调，第一步是让编辑器用上正确的 Python 环境。跟着做，3 步搞定。

**操作步骤（VS Code）**

1. 打开终端，先激活环境：`conda activate llm-learn`
2. 在 VS Code 里按 `Ctrl + Shift + P`，弹出命令面板。
3. 输入并选择 `Python: Select Interpreter`。
4. 在列表里选带 `conda` 字样的项，例如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这么做**：VS Code 会自动认出 conda 环境。选好后，你写代码时用的就是 `llm-learn` 里的 Python 和已装好的包（比如 PyTorch），不会跑错环境。

**验证一下（动手做）**

在 VS Code 新建一个 `.py` 文件，输入以下代码并运行：

```python
import sys
print(sys.executable)
import torch
```

如果打印出的路径中包含 `llm-learn`，且 `

## 块 15

**7.1 在命令行验证**

你的目标是LoRA微调，动手跑通环境是关键。先打开终端，激活环境：

```bash
conda activate llm-learn
```

然后输入 `python` 回车，进入交互模式。看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车，如果正确打印出这句话，说明Python环境没问题。最后输入 `exit()` 退出。

**7.2 在 VS Code 中验证**

命令行验证通过后，我们换到VS Code里再确认一次。新建文件夹 `my-first-project`，用VS Code打开。新建文件 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

点击右上角的小三角运行

## 块 16

好的，我们直接进入正题。你的目标是搞定 LoRA 微调，但在此之前，得先确保你的“工作台”能跑起来。这一节，我们就用最直接的方式，在 PyCharm 里跑通你的第一行代码，为后续的模型训练热热身。

**在 PyCharm 中验证**

1.  打开 PyCharm，新建一个项目。
2.  在项目里新建一个 Python 文件，命名为 `hello.py`。
3.  在文件中输入 `print("Hello, LoRA!")`。
4.  在文件编辑区**右键点击**，选择 **`Run 'hello'`**。
5.  看下方弹出的 **Run** 窗口，如果输出了 `Hello, LoRA!`，说明你的开发环境已经一切就绪，可以开始干活了。

---

**常见问题排查（遇到再看）**

如果刚才的步骤没成功，别急，大概率是下面几个原因：

*   **提示“不是内部或外部命令”**：这是安装 Python 时忘了勾选“Add to PATH”。解决方法是重新运行 Python 安装程序，选择“Modify”，然后勾选“Add Python to environment variables”。
*   **`pip install` 速度极慢**：网络问题。可以在命令后临时加 `-i` 参数指定国内镜像源（比如清华源）来提速。
*   **`conda` 命令找不到**：Windows 用户请从开始菜单打开 `Anaconda Prompt (Miniconda3)`；macOS/Linux 用户则需先运行 `conda init` 并重开终端。
*   **VS Code 提示“未选择 Python 解释器”**：按 `Ctrl + Shift + P`，输入并选择 `Python: Select Interpreter`，手动指定你的 conda 环境。
*   **激活环境后 `python` 版本不对**：系统里可能有多个 Python。用 `where python` (Windows) 或 `which python` (Mac/Linux) 查看当前使用的是哪个路径，确保它指向你的 conda 环境。
*   **`pip` 安装后 `import` 报错**：很可能装错了环境。请先激活目标 conda 环境，再用 `python -m pip install
