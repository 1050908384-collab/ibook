# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你写 Python 代码，电脑只认 0 和 1，中间需要**Python 解释器**当“翻译官”——把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本。

**为什么这对你（数据分析师，目标LoRA微调）很重要？** 你之后要跑 LoRA 微调脚本，本质就是让解释器去执行那些训练代码。先装好这个“翻译官”，后面所有操作才有地基。

**你现在只需要记住一句话**：装 Python 解释器 = 给你的电脑装一个能“读懂并执行”Python 代码的翻译工具。至于它内部怎么翻译，现阶段不用深究。

**动手验证（可选）**：打开终端（Mac/Linux）或命令提示符（Windows），输入 `python --version` 并回车。如果看到版本号，说明解释器已就位；如果提示“未找到命令”，说明还没装好，这正是下一步要解决的。

## 块 2

## 你的“工作台”：从记事本到 IDE

你写 SQL 用惯了工具，写 Python 也别委屈自己。记事本能写代码，但效率极低。**IDE（集成开发环境）** 提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。作为数据分析师，VS Code 上手快，够用。

## 虚拟环境：项目的“独立房间”

你熟悉 SQL 和 Excel，但 Python 项目有个坑：不同项目可能依赖同一个库的不同版本。比如 A 项目要 `numpy 1.0`，B 项目要 `numpy 2.0`，全装一起就冲突报错。**虚拟环境**就是为每个项目建一个独立的 Python 空间，互不干扰。

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.1 Windows 用户：安装 Python

你已熟练SQL和Excel，现在迈向LLM微调。Python是运行微调脚本的基础，本块确保你的Windows环境能运行Python和pip。

**第一步：下载安装包**

1. 浏览器访问 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站自动推荐Windows版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这会将Python加入系统"环境变量"。不勾选的话，命令行输入`python`会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免路径报错。进阶用户想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按

## 块 4

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是LoRA微调，这需要先搭好Python环境。macOS上别用系统自带的旧版Python，自己装最新的最稳妥。

**方法一：官方安装包（推荐新手）**

1. 去 python.org 下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install" 装完。
3. 打开终端，验证：

```bash
python3 --version
pip3 --version
```

**注意**：现代macOS默认没有 `python` 命令，统一用 `python3` 和 `pip3`。这样能避免“命令不存在”或误用系统旧版本的坑。

**方法二：Homebrew（适合已装Homebrew的开发者）**

终端执行：

```bash
brew install python
```

装完同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

Linux 常预装 Python 但版本偏老。执行以下命令安装：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y` 自动确认安装；`python3-venv` 用于后续创建虚拟环境，一并装好。验证版本即完成。

## 块 6

## 2.4 操作系统与实训环境该怎么选

作为数据分析师，你熟悉 SQL 和 Excel，现在要迈向 LoRA 微调。环境选择直接决定你后续能否顺利跑通微调流程，所以这一步值得认真对待。

**你的日常练习环境（无 GPU 场景）**

- **Windows 用户**：强烈建议启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得一个接近真实生产的 Linux 环境，命令、路径、包管理方式都和后续服务器一致，比在 Windows 原生 CMD 里练命令更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。
- **macOS / Linux 用户**：直接使用本机终端即可，本身就是类 Unix 环境，无需额外配置。

**你的 LoRA 微调实训环境（目标场景）**

LoRA 微调依赖 NVIDIA 驱动、CUDA 与显卡直通，应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护（CentOS 8 Stream 除外）；VirtualBox 默认无法直通 NVIDIA GPU，跑不了真正的 GPU 训练，只适合

## 块 7

## 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如你熟悉的 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、应保留的默认源**。

**什么时候才需要镜像？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像是“网络不好时的临时加速手段”，不是永久替换项——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源

如果某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响你平时的默认行为，官方 PyPI 仍是

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你目前没有独立 GPU，后续跑 LoRA 微调大概率会用到云服务器或远程环境。如果你长期处于网络受限环境、几乎每次都要用镜像，才考虑把它写入配置作为默认源。配置后请记得：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车。这会打开一个类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入以下内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃了传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

**macOS / Linux 用户：**

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

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 3

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你已熟悉SQL和Excel，后续写Python脚本、调LoRA代码时，它都能胜任。

**安装步骤：**

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python语言服务器，让代码提示更智能）

**为什么这样写**：安装Python插件后，VS Code才能识别`.py`文件，提供代码补全、错误提示和一键运行功能。

**配置Python解释器：**

1. 打开VS Code，

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community是免费的Python专用IDE，内置代码分析、版本控制、数据库工具，适合大型项目。安装时从官网下载社区版，勾选“Add Open Folder as Project”和“Add to PATH”。首次启动选New Project，Location路径避免中文，Interpreter选系统Python 3.10+。

结合你的目标：LoRA微调需要写Python代码（如PyTorch脚本），PyCharm提供代码分析和版本控制，帮你管理微调项目。无GPU时，你会在本地写代码、调试逻辑，再考虑云GPU运行——PyCharm正好适合本地开发。路径无中文可避免编码问题，选Python 3.10+确保兼容深度学习库。

## 块 11

### 5.1 为什么用 Conda

你已经熟悉 SQL 和 Excel，接下来要学 LoRA 微调，这意味着你会接触 PyTorch、HuggingFace 等深度学习工具链。这些工具依赖复杂，不同项目可能需要不同版本的 Python 或库。Conda 就是帮你管理这些"项目隔离空间"的工具——它不仅能创建独立环境，还能管理 Python 解释器版本。对深度学习来说，Conda 几乎是必备的。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（安装包约 500MB-1GB，安装后占用数 GB），预装 numpy、pandas、jupyter 等 | 做数据分析、不想手动装包 |
| **Miniconda** | 体积小（安装包约 50-100MB，安装后占用约 400MB），只含 conda 和基础 Python | 希望环境干净、按需装包 |

**推荐 Miniconda**，因为后续 LoRA 微调需要什么装什么，环境更干净。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装：
   - **Windows**：一路下一步，**通常不勾选 "Add Miniconda3 to my PATH environment variable"**，以避免与系统已有 Python 冲突；若你熟悉 PATH 配置，也可勾选。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：推荐使用 `.sh` 安装，运行后按提示输入 `yes`，最后一步选 `yes` 执行 **`conda init`**，让终端正确接管 conda。若使用 `.pkg` 安装，安装后需手动运行 `conda init`。
4. **验证安装**：打开**新的**终端（Windows 用 Anac

## 块 12

你已经是SQL和Excel的老手，对“环境隔离”应该不陌生——就像给不同项目准备独立的数据库。现在学LoRA微调，不同项目会需要不同版本的PyTorch，所以我们要用Conda给每个项目一个独立空间。

打开终端，执行：

```bash
conda create -n llm-learn python=3.10
```

**拆解一下**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名（`n` = name）
- `python=3.10`：指定Python版本

执行后终端会问 `Proceed ([y]/n)?`，输入 `y` 回车，等它装完。

**本块学习检查**：完成本块后，用自己的话说明

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

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

**

## 块 14

### 6.1 VS Code 配置 Conda 环境

你已用 `conda activate llm-learn` 激活环境，现在让 VS Code 也用上它：

1. 打开 VS Code，按 `Ctrl + Shift + P` 打开命令面板。
2. 输入 `Python: Select Interpreter` 并回车。
3. 从列表中选择带 `conda` 字样的项，如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这样做**：VS Code 会自动识别 conda 环境。选中后，当前工作区运行的代码就会使用 `llm-learn` 里的 Python 和已装好的包——这正是你后续做 LoRA 微调时依赖的 PyTorch 等库所在的环境。

### 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程类似：

1. 打开 `File` → `Settings`（Mac 为 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`/Users/你的用户名/miniconda3/envs/

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已熟悉SQL和Excel，现在开始接触Python环境。本块目标很简单：**确认Python能跑起来**——这是后续所有LoRA微调操作的地基。

**在命令行验证**（适合快速确认环境）：

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符
3. 输入 `print("Hello, Python! 我的环境准备好了！")` 并回车
4. 正确打印后，输入 `exit()` 退出

**在VS Code中验证**

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

## 7.3 在 PyCharm 中验证

你已经熟悉 SQL 和 Excel，现在要进入 Python 环境。PyCharm 是你写代码的“工作台”，验证步骤很简单：

1. **新建项目**：打开 PyCharm，创建一个新项目（项目路径不要包含中文或空格）。
2. **创建文件**：在项目中新建一个 Python 文件，命名为 `hello.py`。
3. **运行**：右键点击 `hello.py` → 选择 `Run 'hello'`。
4. **看结果**：下方 Run 窗口会输出运行结果。

> **为什么这对 LoRA 微调重要？**  
> LoRA 微调需要写 Python 脚本（如加载模型、训练、评估）。如果连“运行第一行代码”都不顺畅，后续调试会非常痛苦。这一步是确保你的“工作台”能正常运转。

---

## 第八章：常见问题排查（重点看前两行）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |

> **你的关注点**：你无独立 GPU，后续 LoRA 微调大概率用云端或 CPU 环境，但本地 Python 环境仍需干净可用。**pip 慢**是常见痛点，记住 `-i` 加镜像源这个临时方案即可。

---

## 本章小结（你的环境应具备）

| 组件 | 状态 | 验证命令 |
|------|------|----------|
|

## 测评题目

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0202-003
以下代码输出是？
```python
s = {1, 2, 3, 3, 2, 1}
print(len(s))
```

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？

### Q-KB0201-003
以下代码的输出是？
```python
x = 10
y = 3
print(x // y, x % y, x ** y)
```

### Q-KB0203-001
以下函数定义中，哪个参数是**默认参数**？
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

### Q-KB0203-003
请定义一个类 `Dog`，包含：
- 类属性 `species = "Canis familiaris"`
- 实例属性 `name` 和 `age`（通过 `__init__` 初始化）
- 方法 `bark()` 返回 `"{name} says: Woof!"`

### Q-KB0205-005
请写一个**带参数的装饰器** `retry(times=3)`，使被装饰的函数在抛出异常时自动重试指定次数。
