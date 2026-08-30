# 学习包

## 块 1

## 1.1 Python 解释器：程序的"翻译官"

你学习 LoRA 微调，最终要运行的是 Python 代码——但电脑本身只懂 0 和 1，看不懂你写的代码。**Python 解释器**就是两者之间的"翻译官"：它把你写的代码逐句翻译成电脑能执行的指令。

**为什么这个环节对 LoRA 微调很重要？**

你的 RTX 3060 显卡有 12GB 显存，微调模型时需要调用 GPU 加速。而 Python 代码本身只是文本，必须通过解释器翻译后，才能进一步调用显卡的计算能力。没有解释器，你写的 `.py` 文件只是一堆无法运行的文字。

**一个关键认知**：解释器是"翻译"，不是"编译"。它是一边翻译一边执行，所以代码写错一行，运行到那一行才会报错。这对你后续调试微调脚本很有帮助——错误信息会直接指向具体行。

**学习检查**：用自己的话说说，为什么没有 Python 解释器，你写的微调代码就无法运行？

## 块 2

**IDE 与虚拟环境：你的 AI 工作台**

既然你的目标是 LoRA 微调，那代码编辑器和环境管理就是你的“工作台”和“独立房间”。记事本写代码效率极低，而 **IDE** 提供语法高亮、自动补全、运行调试，让你写代码像用 Word 一样舒服。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个顺手的即可。

装好 IDE 后，还要解决“环境冲突”问题。想象两个项目：A 要 `numpy 1.0`，B 要 `numpy 2.0`，全装一起会报错。**虚拟环境**就是为每个项目建独立 Python 空间

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

你的目标是 LoRA 微调，而微调模型的第一步，就是让电脑能运行 Python。下面三步帮你搞定。本教程以 Python 3.13 为例，3.10 及以上版本均可，操作步骤完全一致。

**第一步：下载安装包**

打开浏览器，访问 https://www.python.org。鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项会把 Python 添加到系统的"环境变量"中。如果不勾选，你打开命令行输入 `python` 时，系统会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。进阶用户如果想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

现代 macOS（Catalina 10.15 起）已不再预装 Python 2，部分系统可能带有供系统脚本使用的旧版 `python3`。**强烈建议自行安装最新版 Python**，避免依赖系统自带版本。

#### 方法一：官方安装包（推荐新手）

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**并不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

#### 方法二：Homebrew（适合已有 Homebrew 的开发者）

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：在终端执行 `python3 --version`，确认输出为 Python 3.10 或更高版本。若输出为旧版本，请

## 块 5

**目标**：为后续 LoRA 微调铺路，先确保 Linux 环境能跑 Python 3.10+。

Ubuntu 默认 python3 版本可能低于 3.10，需用 deadsnakes PPA 安装指定版本：

```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip -y
python3.10 --version        # 验证安装
```

**要点**：`-y` 自动确认；`python3.10-venv` 用于建虚拟环境，避免后续

## 块 6

### 2.4 操作系统与实训环境该怎么选

你现在的目标是 **LoRA 微调**，并且手头有 **RTX 3060 12GB** 显卡。这意味着，你的环境选择必须围绕“能否跑 GPU 训练”来思考，而不是随便装个 Python 就完事。

**核心原则：日常练习和 GPU 实训，环境要求不同。**

- **如果你用 Windows 写 Python、练命令**：强烈建议启用 **WSL2 + Ubuntu LTS**（如 22.04/24.04）。它让你在 Windows 里获得一个接近真实服务器的 Linux 环境，命令、路径、包管理方式都和后续云端一致。安装只需管理员 PowerShell 执行 `wsl --install`（需 Windows 10 版本 2004 及以上或 Windows 11，可能需要重启系统）。别在原生 CMD 里练，那是浪费时间。
- **如果你用 macOS 或 Linux**：直接用本机终端即可，本身就是类 Unix 环境，无需额外配置。

- **关键点——LoRA 微调这类 GPU 实训**：必须使用 **装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。为什么？因为这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，而 Ubuntu LTS 对

## 块 7

**镜像源：网络不畅时的临时加速器**

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是最权威、最完整的源，**应始终保留为默认**。

**什么时候才需要镜像源？** 只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像只是“加速手段”，不是永久替代品。

**首选做法：临时指定，一次有效**

如果某次安装慢，最干净的方式是只对这一次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样不会影响平时的默认行为；网络恢复或镜像出问题时，也无需清理配置。

**本块学习检查**：观察上述命令，指出它相比“永久修改默认源”的两个

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，而网络是绕不开的一关。如果你**长期处于网络受限环境**、几乎每次下载包都要靠镜像，才值得把镜像写进配置设为默认。否则，用临时参数 `-i` 就够了，别动全局配置。

**为什么谨慎？** 官方 PyPI 始终是最可靠的来源。配置默认镜像只是权宜之计，网络恢复后建议改回或删除该配置。

---

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需**它。只有企业内网自建源、或确认来源可信且遇到证书报错时，才临时加。

---

**macOS / Linux 用户：**

1. 终端执行：

```bash
mkdir -p ~/.config/pip
```

2. 创建配置文件：

```bash
nano ~/.config/pip/pip.conf
```

3. 写入同样内容：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

4. `Ctrl + O` 保存，`Ctrl + X` 退出。

---

### 3.4 验证配置是否生效

命令行输入：

```bash
pip config list
```

看到 `global.index-url='https://p

## 块 9

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要写 Python 脚本、改配置文件，它都能胜任。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

安装后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：
1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚安装的 Python 版本（如 `

## 块 10

**PyCharm Community：你的AI学习工作台**

你已确定要学LoRA微调，而微调需要写代码、看结果，一个好用的编辑器能让你专注在“学AI”而不是“折腾工具”上。PyCharm Community版正是为此准备的——它是JetBrains出品的Python专用IDE，社区版完全免费，内置代码分析、版本控制等能力，适合你后续管理微调项目这类多文件工程。

**安装要点**（约10分钟）：
1. 打开官网下载页，选**Community Edition**（免费版）。
2. 安装时勾选两项：`Add "Open Folder as Project"`（方便右键打开项目文件夹）和`Add to PATH`（之后可在命令行直接启动）。

**首次启动**：
1. 点`New Project`，在`Location`处选项目路径——**务必不含中文**，避免后续环境解析出错。
2. `Interpreter`选`Previously configured interpreter`，指向你系统已装的Python 3.10+。

**本块学习检查**：打开PyCharm新建一个空项目

## 块 11

## 5.1 为什么用 Conda

你现在的目标很明确：**LoRA 微调**。这意味着你要接触 PyTorch、HuggingFace 生态等一堆复杂依赖。如果直接在系统 Python 里装，很容易出现版本冲突——这个项目要 A 版本，那个项目要 B 版本，最后全乱套。

**虚拟环境**就是给每个项目一个独立的 Python 空间，互不干扰。而 **Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于深度学习这种依赖复杂的场景，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 1GB 以上），预装了数百个数据科学库（numpy、pandas、jupyter、scipy、matplotlib 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50-100MB），仅包含 conda、Python 及少量必需依赖，按需安装其他包 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你要做 LoRA 微调，环境越干净越好，需要什么装什么，避免一堆用不到的库占地方、添乱。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"**

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：你的目标是LoRA微调，而微调不同模型时，往往需要不同版本的PyTorch和CUDA。现在学会创建隔离环境，以后每个项目各用一套配置，互不干扰。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行

## 块 13

## 5.4 激活与退出环境——为 LoRA 微调准备的“独立房间”

你学习 LoRA 微调，会用到 PyTorch 和 HuggingFace 工具链。如果所有项目共用一套 Python 环境，版本冲突会让人头疼。Conda 虚拟环境就是给每个项目一个“独立房间”。

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

**查看所有已创建的环境**

## 块 14

**本块学习检查**：完成下面 4 步后，在 VS Code 终端运行 `python -c "import torch; print(torch.__version__)"`，看到版本号即配置成功。

---

### 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，所以代码必须跑在 `llm-learn` 环境里。按顺序做：

1. **先激活环境**：打开终端，运行 `conda activate llm-learn`。这一步是前提，不激活后面选了也没用。
2. **打开命令面板**：在 VS Code 中按 `Ctrl + Shift + P`。
3. **选择解释器**：输入 `Python: Select Interpreter` 并回车。
4. **挑带 conda 的**：在列表里选带有 `conda` 字样的路径，比如 `Python 3.10.x ('llm-learn': conda)`。

**为什么这样写**：VS Code 会自动识别 conda 环境。选完后，你在这个工作区写的所有代码，都会用 `llm-learn` 里的 Python 和已装好的包（比如 PyTorch），不会误用系统默认环境。

---

### 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程类似：

1. 打开 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名

## 块 15

**本块学习检查**：完成下方两个验证步骤后，在终端输入 `exit()` 退出，并确认你已看到两段不同的欢迎文字。

---

**第一步：命令行验证**

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符后，输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```
3. 回车，若正确打印文字，输入 `exit()` 退出。

**第二步：VS Code 验证**

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`，输入：
   ```python
   print("Hello from VS Code!")
   ```
3

## 块 16

## 7.3 在 PyCharm 中验证——跑通你的第一行代码

环境搭好了，现在我们来确认它真的能用。这一步的目标很简单：让 PyCharm 帮你运行一个 Python 文件，并在窗口里看到输出结果。这是你后续所有代码工作的“开机自检”。

**操作步骤（三步完成）**

1. **新建项目，创建 `hello.py`**  
   打开 PyCharm，新建一个项目，然后在项目里新建一个 Python 文件，命名为 `hello.py`。文件名不需要复杂，`hello` 就好。

2. **右键点击文件 → `Run 'hello'`**  
   在左侧文件列表中，找到 `hello.py`，点击鼠标右键，在弹出的菜单中选择 `Run 'hello'`。PyCharm 会启动 Python 解释器来执行这个文件。

3. **下方 Run 窗口应输出结果**  
   执行后，PyCharm 窗口底部会打开一个名为 `Run` 的面板，你的代码输出会显示在这里。如果看到了输出内容，说明你的开发工作台已经完整跑通了。

> **学习提示**：这一步的本质是验证“编辑器 → 解释器 → 输出”这条链路是否通畅。对于你后续要做的 LoRA 微调，这个链路是基础——你写的每一行训练代码，最终都要通过它来运行和查看结果。

---

## 第八章：常见问题排查（遇到再看）

环境搭建中，下面几个问题最常出现。如果没遇到，跳过即可；遇到了，对照表格处理。

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的

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

### Q-KB0205-001
以下哪个是装饰器的正确语法？

### Q-KB0201-003
以下代码的输出是？
```python
x = 10
y = 3
print(x // y, x % y, x ** y)
```

### Q-KB0203-004
以下代码输出是？
```python
class A:
    def __init__(self):
        self.x = 1
class B(A):
    def __init__(self):
        super().__init__()
        self.y = 2
b = B()
print(b.x, b.y)
```

### Q-KB0203-006
关于Python的MRO（方法解析顺序），以下说法**错误**的是？

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
```

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
