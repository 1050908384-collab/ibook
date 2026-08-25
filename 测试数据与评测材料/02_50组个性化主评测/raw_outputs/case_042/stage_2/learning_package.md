# 学习包

## 块 1

```markdown
#

## 块 2

### 1.2 代码编辑器（IDE）：你的"工作台"

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。本文推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。

### 1.3 虚拟环境：项目的"独立房间"

想象你同时在做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果把它们都装到同一个 Python 里，就会冲突报错。**虚拟环境**就是为每个项目创建一个独立的 Python 空间，互不影响。本文使用 **conda** 来管理虚拟环境，它是目前最主流的方案。

---

## 第二章：安装 Python 3.10+

**为什么选 3.10+**：Python 3.10 引入了更友好的错误提示、模式匹配等新特性，且目前主流第三方库均已兼容。不建议使用 Python 2（已于 2020 年停止维护）。

根据你的操作系统，选择对应步骤。

（个性化提示：学习风格：hands-on）

## 块 3

### 2.1 Windows 用户

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   **为什么这样写**：这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   **为什么这样写**：默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python 3.13.x`（或你下载的版本号），说明安装成功！
4. 再输入：

```bash
pip --version
```

如果也显示版本号，说明 Python 的包管理工具 `pip` 也一并安装成功了。

---

（个性化提示：学习风格：hands-on）

## 块 4

### 2.2 macOS 用户

较老的 macOS 曾预装 Python 2.7，但**现代 macOS（Catalina 10.15 起）已不再预装 Python 2**；部分系统可能带有供系统脚本使用的旧版 `python3`。无论哪种情况，都**强烈建议自行安装最新版 Python**，避免直接依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：在现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向系统自带的 Python 2，现已移除），系统与官方安装包统一使用 `python3`。因此请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已经安装了 Homebrew（macOS 的包管理器），在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

（个性化提示：学习风格：hands-on）

## 块 5

### 2.3 Linux 用户（Ubuntu/Debian）

Linux 通常预装 Python，但版本可能较老。以下是 Ubuntu/Debian 系列的安装命令：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**为什么这样写**：`-y` 参数表示自动确认安装，不需要你手动输入 `y`。`python3-venv` 是创建虚拟环境所需的模块，一起安装避免后续补装。

（个性化提示：学习风格：hands-on）

## 块 6

### 2.4 操作系统与实训环境该怎么选

不同学习阶段对环境的要求不同，选对环境能少走很多弯路：

- **日常写 Python、练命令行（Windows 用户）**：推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续的服务器一致，比在 Windows 原生 CMD 里练命令更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install` 即可。
- **日常写 Python（macOS / Linux 用户）**：直接使用本机终端即可，本身就是类 Unix 环境。
- **LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**：应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态的支持最成熟、文档最全。

> **⚠️ 不推荐的做法**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护、生态逐渐落后；而 VirtualBox 这类桌面虚拟机**默认无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练，只适合做纯 CPU 的隔离小实验。命令练习用 WSL2/Ubuntu LTS，GPU 实训用云上或实体的 Ubuntu LTS，才是稳妥路线。

---

## 第三章：pip 下载太慢？——按需使用国内镜像源

（个性化提示：学习风格：hands-on）

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用于安装第三方库（如 `numpy`、`pandas`）。默认情况下，`pip` 从官方的 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

只有当你处在网络受限环境、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终应作为首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不会影响你平时的默认行为，官方 PyPI 仍是默认源；哪天网络恢复正常或镜像出问题，你无需再去清理配置。

（个性化提示：学习风格：hands-on）

## 块 8

```markdown
# 进阶：优化 Python 环境：将国内镜像设为默认下载源

## 3.3 进阶：将镜像设为默认（

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富，不仅适合 Python，也适合前端、配置文件编辑等。后续学习 Linux 命令行、写脚本时都能用到。

**安装步骤：**

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**为什么这样写**：安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器：**

1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚安装的 Python 版本（如 `Python 3.13.x`）。

---

（个性化提示：学习风格：hands-on）

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置了代码分析、版本控制、数据库工具等功能，适合大型项目开发。

**安装步骤：**

1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置：**

1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，然后选择系统安装的 Python 3.10+。

---

## 第五章：配置 Conda 虚拟环境（项目隔离）

（个性化提示：学习风格：hands-on）

## 块 11

```markdown
## 为什么使用 Conda？

在 Python 开发中，虚拟环境的使用可以帮助我们隔离项目依赖，避免环境冲突。而 **Conda** 是一个功能强大的工具，能够帮助我们轻松创建和管理多个独立的 Python 环境。它不仅支持虚拟环境的构建，还能够管理不同版本的 Python 解释器，这对于需要依赖特定版本库（如 PyTorch、CUDA 等）的深度学习项目尤为重要。

### 安装 Conda 的选择

在安装 Conda 时，有两个主要选项：Anaconda 和 Miniconda。两者的主要区别在于体积大小和功能实用性。

| 选项   | 特点                        | 推荐人群                          |
|-------|----------------------------|

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：后续学习大模型微调时，不同项目需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后直接复用这套流程。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
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

输入 `y` 并回车，等待安装完成。

（个性化提示：学习风格：hands-on）

## 块 13

### 5.4 激活与退出环境

**激活环境**（进入这个"独立房间"）：

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

## 第六章：在 VS Code / PyCharm 中使用 Conda 环境

（个性化提示：学习风格：hands-on）

## 块 14

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
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

---

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节都通。

（个性化提示：学习风格：hands-on）

## 块 15

### 7.1 在命令行验证

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式：

```bash
python
```

3. 看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

4. 回车，如果正确打印文字，按 `exit()` 退出。

### 7.2 在 VS Code 中验证

1. 新建一个文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`，输入：

```python
print("Hello from VS Code!")
```

3. 点击右上角的小三角（运行按钮），或按 `Ctrl + F5`。
4. 下方终端应输出 `Hello from VS Code!`。

（个性化提示：学习风格：hands-on）

## 块 16

# 在 PyCharm 中验证你的 Python 代码

## 教学目标
通过本节学习，你将能够：
1. 创建并运行一个简单的 Python 脚本。
2. 验证你的 Python 环境配置是否正确。

## 教学步骤

### 第一步：创建并保存 Python 脚本
1. 打开 PyCharm，选择一个已有的或新建的项目。
2. 点击菜单栏中的 **File**，然后选择 **New File**。
3. 在弹出的窗口中，输入文件名（例如：`hello.py`），点击 **Create**。
4. 编写以下代码：
   ```python
   # 指定输出文件路径
   print("Hello, World!")
   ```
   代码解释：
   - 使用 `print` 函数输出“Hello, World!”字符串。
   - 这是 Python 中最基础的代码，用于验证你的环境是否配置正确。

### 第二步：运行 Python 脚本
1. 选择文件后，右键点击文件，选择 **Run 'hello'**。
2. PyCharm 会自动运行文件，并在下方的 Run 窗口中显示输出结果。

### 第三步：验证运行结果
