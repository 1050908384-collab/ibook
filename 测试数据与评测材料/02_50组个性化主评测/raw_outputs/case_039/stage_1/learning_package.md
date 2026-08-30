# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写下的 Python 代码是人类语言，但电脑只懂 0 和 1。**Python 解释器**就是那个“翻译官”——它把你写的代码逐行翻译成电脑能执行的指令。没有它，你写的 `.py` 文件就只是一堆普通文本，电脑根本看不懂。

**为什么这和你学 LoRA 有关？**  
你要做的 LoRA 微调，本质是运行 Python 代码来训练和调整模型。如果解释器没装好，后面所有操作都无从谈起。所以这一步不是“随便装个软件”，而是给你的深度学习工作台打地基。

**你现在的情况**：你有 PyTorch 和 Transformer 基础，电脑是 RTX 4060 Ti 16GB，完全够跑 LoRA 微调。但前提是——解释器得先正常工作。别跳过这一步，否则后面报错时你会分不清是代码问题还是环境问题。

**本块学习检查**  
完成本块后，用自己的话说说：为什么 LoRA 微调需要 Python 解释器？如果遇到“解释器”“翻译”这类术语说不清的地方，先记下来，课后提问。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

IDE是你的代码"工作台"，记事本也能写但效率极低。**IDE**提供语法高亮、自动补全、运行调试，让你写代码像用Word一样舒服。推荐**VS Code**（轻量免费）和**PyCharm**（专业强大）。

**虚拟环境**是项目的"独立房间"：比如A项目要`numpy 1.0`，B项目要`numpy 2.0`，装一起会冲突。虚拟环境为每个项目创建独立Python空间，互不影响。用**conda**管理，这是主流方案。

**Python 3.10+**

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.1 Windows 用户：装好 Python，才能跑 LoRA

你的目标是做 LoRA 微调，这需要运行 PyTorch 等深度学习框架，而它们都依赖 Python 环境。本块就是帮你把地基打牢。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，之后在命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径不含中文和特殊符号，避免后续路径报错。进阶用户若改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并

## 块 4

## 2.2 macOS 用户：为 LoRA 微调铺好第一步

你的目标是 LoRA 微调，而微调的第一步是让 Python 环境干净、可控。macOS 上最常踩的坑，就是误用系统自带的旧版 Python——这会导致后续安装 PyTorch 等依赖时版本错乱。所以，请务必自己装一个全新的 Python。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认不提供 `python` 命令，系统与官方安装包统一使用 `python3`。始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明：为什么安装 Python 时要避开系统自带版本？这与

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

Linux 通常预装 Python，但版本可能较老。为后续 LoRA 微调做准备，需确保环境干净可用。执行以下命令：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

`-y` 参数自动确认安装，免手动输入 `y`。`python3-venv` 用于创建虚拟环境，一并装好避免后续补装。验证命令确认版本，为

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，所以环境选择要分两步走：

**第一步：日常练 Python 和命令行**

- **Windows 用户**：启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。它让你在 Windows 里获得接近真实服务器的 Linux 环境，命令、路径、包管理都和后续云端一致。安装只需管理员 PowerShell 执行 `wsl --install`。
- **macOS / Linux 用户**：直接用本机终端，本身就是类 Unix 环境。

**第二步：LoRA / QLoRA 微调等 GPU 实训**

必须使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已于 2024 年 5 月 31 日停止维护；VirtualBox 默认**无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

**你的选择**：你的 RTX 4060 Ti

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认它从官方 PyPI（`pypi.org`）下载——这是最权威、最完整的源，**应始终保留为默认**。

只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像只是“网络不好时的加速手段”，不是永久替换**——官方源永远是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：它不影响你平时的默认行为，官方 PyPI 仍是默认源；网络恢复或镜像出问题时，你无需清理配置

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正在为 LoRA 微调准备环境，下载依赖包是绕不开的一步。如果只是偶尔下载慢，用 `-i` 参数临时指定镜像就够了；但如果你长期处于网络受限环境，每次都要敲镜像地址很麻烦，这时才值得把它写进配置当默认源。

**一个重要提醒**：官方 PyPI 始终是最可靠的来源。配置默认镜像只是权宜之计，等网络恢复后，建议改回官方源或直接删除配置。

---

**Windows 用户操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在此文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 的作用就是告诉 pip：默认从这个地址下载包。

> **⚠️ 关于 `trusted-host` 的警告**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则绝对不要加**——加上它等于让 pip 跳过对该域名的 TLS 校验，直接放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**完全不需要**。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加。

---

**macOS / Linux 用户操作步骤：**

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

---

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list

## 块 9

**VS Code：你的 LoRA 微调工作台（轻量、免费、推荐）**

你已有 PyTorch 和 Transformer 基础，目标是用 LoRA 微调模型。VS Code 是微软出品的免费编辑器，启动快、插件丰富，后续写脚本、调配置都能用，适合你周末集中攻坚。

**安装步骤：**

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能——这对调试 LoRA 训练脚本很关键。

**配置 Python 解释器：**

1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community是免费的Python专用IDE，内置代码分析、版本控制、数据库工具，适合大型项目。安装时勾选`Add "Open Folder as Project"`和`Add to PATH`。首次启动选`New Project`，项目路径避免中文，解释器选系统Python 3.10+。

**关联你的目标**：LoRA微调涉及大量脚本调试与文件管理，PyCharm的代码分析和项目视图能帮你高效组织训练代码与配置。你的RTX 4060 Ti 16GB足够本地运行微调，用此IDE管理代码即可。若时间紧，可先装好并新建项目验证Python解释器可用，后续章节再深入配置Conda环境。

## 块 11

### 5.1 为什么用 Conda

你之前已经接触过虚拟环境的概念——它为每个项目创建独立的 Python 空间。现在要学的 **Conda**，是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对你接下来的 LoRA 微调学习来说，这几乎是必备工具，因为深度学习项目（比如 PyTorch、CUDA 这些复杂依赖）对环境隔离的要求非常高。

### 5.2 安装 Conda：Anaconda vs Miniconda

安装 Conda 有两个选择，区别主要在于体积和预装内容：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐你选 Miniconda**。理由很简单：环境更干净，后续需要什么包装什么包，不会有一堆用不上的库占地方。

**安装步骤（跟着做）：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器会标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda 即可。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes` 同意协议、选择安装路径，并在最后一步选择 `yes` 让安装器执行 **`conda init`**（完成 shell 初始化）。这样每次打开终端时 cond

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你周末只有5小时，目标直指LoRA微调。微调大模型时，不同项目常需不同版本的PyTorch和CUDA，混在一起会互相干扰。现在花几分钟学会建环境，就是为后续省时间。

在终端执行：

```bash
conda create -n llm-learn python=3.10 -y
```

拆开看：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名`llm-learn`（`n`是`name`的缩写

## 块 13

## 5.4 激活与退出环境：进入你的“独立房间”

在 LoRA 微调实战中，你需要在干净、隔离的环境里安装 PyTorch、Transformers 等依赖，避免和系统其他项目互相干扰。Conda 虚拟环境就是这样一个“独立房间”。

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

**本块学习检查**：完成本块后，用自己的话说明本块内容

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
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件（将 `你的用户名` 替换为实际用户名）：
   - Windows：`%USERPROFILE%\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`$HOME/miniconda3/envs/llm-learn/bin/python`

---

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，确保每个环节

**本块学习检查**：完成本块后，用自己的话说明

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是LoRA微调，但任何微调都要在能运行Python的环境里进行。本块就是验证你的“工作台”是否能用——用命令行和VS Code各跑一次`print`，确认环境没问题。

**在命令行验证**（适合快速检查）：
1. 激活环境：`conda activate llm-learn`
2. 输入`python`回车，看到`>>>`提示符
3. 输入`print("Hello, Python! 我的环境准备好了！")`回车
4. 正确打印后，输入`exit()`退出

**在VS Code

## 块 16

## 7.3 在 PyCharm 中验证

你的周末时间宝贵，我们直接聚焦核心动作。在 PyCharm 中验证环境是否就绪，只需三步：

1. **新建项目**，在项目中创建 `hello.py` 文件。
2. **右键点击** `hello.py` 文件 → 选择 `Run 'hello'`。
3. 观察下方 **Run 窗口**，应输出你写在 `print()` 里的内容。

这一步的意义在于：确认你前面安装的 Python 解释器、PyCharm 和项目配置是**连通**的。如果 Run 窗口正常输出，说明你的开发工作台已经可以开始写代码了。

---

## 第八章：常见问题排查（快速对照）

你可能会遇到以下问题，按表格快速定位：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像，如：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名`；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

**重点提醒**

## 测评题目

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0204-003
请写代码：读取 `data.txt` 文件，捕获文件不存在的异常，若不存在则打印 "文件未找到"。

### Q-KB0202-005
以下代码输出是？
```python
d = {}
d.setdefault("key", []).append(1)
d.setdefault("key", []).append(2)
print(d)
```
