# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你正在为 **LoRA 微调** 做准备，而微调的第一步，是确保你的电脑能“听懂”你写的 Python 代码。本块只讲一个核心概念：**Python 解释器**。

**它是什么？**

你写的代码是人类语言，电脑只认 0 和 1。**Python 解释器**就是那个“翻译官”——把你写的 `.py` 文件翻译成电脑能执行的指令。

**为什么和 LoRA 有关？**

LoRA 微调需要运行 PyTorch 等深度学习框架，这些框架的代码最终都要靠解释器来驱动。没有它，你写的任何训练脚本都只是一堆无法运行的文本。

**一句话记住**：解释器 = 代码与电脑之间的翻译官。

**动手验证**：先不用安装，只需在脑中复述——如果我把 `print("hello")` 存成 `.py` 文件，电脑能直接运行吗？答案是不能，必须经过解释器翻译。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是LoRA微调，这需要写代码、跑脚本。记事本也能写，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用Word写文档一样舒服。推荐两款：**VS Code**（轻量免费）和**PyCharm**（专业强大）。

微调涉及多个实验，依赖版本容易冲突。**虚拟环境**就是为每个项目创建一个独立的Python空间，互不影响。本文用**conda**管理虚拟环境，这是常用的虚拟环境管理方案

## 块 3

**Windows 安装 Python：三步搞定**

你的目标是LoRA微调，而PyTorch是它的基础。装好Python，就是给PyTorch铺好第一块砖。下面三步，跟着做就行。

**第一步：下载安装包**

打开浏览器，访问 `https://www.python.org`。鼠标移到顶部菜单 **Downloads**，网站会自动推荐Windows版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续报错。若改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入 `python --version` 并回车。显示 `Python 3.13.x` 即成功。
3. 再输入 `pip --version`，显示版本号说明包管理工具 `pip` 也装好了。

---

**

## 块 4

## 2.2 macOS 用户：安装 Python 3.10+

macOS 自 Catalina 起仍预装 Python 2.7，但已弃用；系统自带的 `python3` 版本较旧，建议安装最新版 Python 3.10+。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

安装后运行 `which python3` 确认路径，或使用 `python3 --version` 检查版本是否为 3.10+。

**为什么用 `python3` 而不是 `python`**：macOS 上默认的 `python` 命令指向 Python 2.7，为避免混淆，建议使用 `python3` 命令。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

已安装 Homebrew 的话，在终端执行：

```bash
brew install python
```

安装完成后，运行 `which python3` 确认路径为 Homebrew 路径，必要时重启终端或添加 PATH 配置，再用 `python3 --version` 验证。

---

**本块学习检查**：完成本块

## 块 5

**本块学习检查**  
完成本块后，用自己的话说明：为什么安装 Python 时要一并安装 `python3-venv`？这与你的 LoRA 微调目标有何关联？遇到无法解释的术语（如“虚拟环境”）时记录下来再提问。

---

**动手实践（Ubuntu/Debian）**  
在终端依次执行以下命令，注意观察每一步的输出：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

**关键点**：`-y` 让系统自动确认安装，无需手动输入 `y`。`python3-venv` 是创建隔离环境的核心模块，提前装好可避免后续为 LoRA

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是LoRA微调，且手头是RTX 4060 8GB显卡。选对环境，直接决定你能否顺利跑通训练。

**日常写Python、练命令（Windows用户）**：启用 **WSL2 + Ubuntu LTS**（如Ubuntu 22.04/24.04 LTS）。它让你在Windows里获得接近真实生产的Linux环境，命令、路径、包管理都和后续服务器一致，比在CMD里练更贴近实战。安装只需管理员PowerShell执行 `wsl --install`。

**日常写Python（macOS/Linux用户）**：直接用本机终端，本身就是类Unix环境。

**LoRA/QLoRA微调等GPU实训**：使用**装有Ubuntu LTS的NVIDIA GPU云服务器，或本地实体Linux（Ubuntu LTS）机器**。这类任务依赖NVIDIA驱动、CUDA与显卡直通，Ubuntu LTS对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把**“CentOS + VirtualBox虚拟机”当作最佳方案**。CentOS 8已停止维护；VirtualBox默认无法直通NVIDIA GPU，跑不了真正的GPU训练，只适合纯CPU隔离小实验。命令练习用WSL2/Ubuntu LTS，GPU实训用云上或实体Ubuntu LTS，才是稳妥路线。

---

**本块学习检查**：完成本块后，

## 块 7

## 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载——这是**最权威、最完整、最应保留的默认源**。

**只有**当你网络受限、访问官方源明显变慢或经常超时时，才考虑临时切换国内镜像。**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

## 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不影响你平时的默认行为，官方源仍是首选。

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，而 PyTorch 等依赖包的下载速度直接影响你的进度。如果你长期处于网络受限环境，每次安装包都慢得让人着急，那就可以把清华镜像设为 pip 的默认源，一劳永逸。

**但请注意**：官方 PyPI 仍是最可靠的来源。网络恢复后，建议改回或删除该配置。

---

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

---

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

---

### 3.4 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

VS Code 是微软出品的免费编辑器，启动快、插件丰富。它不只适合 Python，也适合前端、配置文件编辑，后续写脚本时都能用到。

**安装步骤：**
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器：**
1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community 是你的 Python 开发工作台，免费且专业。它内置代码分析、版本控制等功能，适合你后续开展 LoRA 微调这类大型项目。

**安装要点**：
1. 访问官网下载 **Community Edition**（社区版）。
2. 安装时勾选 `Add "Open Folder as Project"` 和 `Add to PATH`。

**首次启动**：
1. 新建项目，`Location` 路径**不要含中文**。
2. `Interpreter` 选择 `Previously configured interpreter`，指向系统 Python 3.10+。

**动手验证**：按上述步骤完成安装与项目创建，确认 PyCharm 能正常打开并识别 Python 解释器。

## 块 11

### 5.1 为什么用 Conda

你的目标是 **LoRA 微调**，这意味着你马上要动手训练模型。训练离不开 PyTorch、CUDA 这类依赖，而它们对版本极其敏感——装错一个版本，模型可能直接跑不起来。**Conda** 就是帮你管住这些版本的工具：它能为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备的。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 安装包约 500MB-1GB，安装后占用空间通常超过 3GB，预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 安装包约 50-100MB，安装后占用空间约 400MB 以上，只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境干净，后续微调需要什么包装什么包，不浪费你 8GB 显存的机器资源。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`；macOS 需根据 CPU 架构（Intel 或 Apple Silicon）选择对应的 `.pkg` 或 `.sh`；Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。若不勾选，日常需通过开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda，或后续手动添加 PATH。
   - **macOS/Linux**：运行 `.sh` 文件后，按提示输入 `yes`

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你正在为LoRA微调做准备，而微调需要独立的PyTorch和CUDA环境。本块教你创建第一个虚拟环境，这是后续所有项目隔离的基础。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

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

**实践

## 块 14

## 6.1 VS Code 配置 Conda 环境

先激活环境：`conda activate llm-learn`。建议激活后立即验证是否成功，例如运行 `conda activate llm-learn && python -c "import sys; print(sys.executable)"`，确认输出路径指向 `llm-learn` 环境。打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 自动识别 conda 环境后，你在此工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。左侧选择 `Project: 你的项目名` → `Python Interpreter`。点击齿轮图标 → `Add...`，左侧选择 `Conda Environment` → `Existing environment`。在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：

- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍，

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

**7.1 在命令行验证**

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式，看到 `>>>` 提示符后，输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

3. 正确打印文字后，按 `exit()` 退出。

**7.2 在 VS Code 中验证**

1. 新建文件夹 `my-first-project`，用 VS Code 打开。
2. 新建文件 `hello.py`，输入：

```

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

## 7.3 在 PyCharm 中验证

你的目标是 LoRA 微调，这意味着后面要反复运行、调试 Python 脚本。现在先确认工作台能跑通最基础的代码。

**操作步骤**：
1. 打开 PyCharm，新建一个项目。
2. 在项目中新建文件，命名为 `hello.py`。
3. 在文件中输入一行代码：`print("Hello")`（这是最简单的输出语句）。
4. 在文件编辑区**右键单击**，选择 `Run 'hello'`。
5. 观察窗口**下方**的 Run 面板，如果看到输出了 `Hello`，说明环境验证成功。

> 你的显卡是 RTX 4060 8GB，后续跑 LoRA 微调时会在 PyCharm 里频繁执行类似操作——只不过运行的脚本会从 `hello.py` 变成训练脚本。现在把这一步跑通，等于确认了“编辑代码 → 运行 → 看输出”这条链路是通的。

---

## 第八章：常见问题排查

运行 `hello.py` 时如果报错，对照下表处理：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在

## 测评题目

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0201-004
以下代码输出是？
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```

### Q-KB0203-005
以下代码输出什么？
```python
class Counter:
    count = 0
    def __init__(self):
        Counter.count += 1
    @classmethod
    def get_count(cls):
        return cls.count

a = Counter()
b = Counter()
c = Counter()
print(Counter.get_count())
```

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0205-008
写一个类 `Stack`（栈），实现 `push(item)`、`pop()`、`peek()`、`is_empty()` 方法。`pop()` 在栈空时抛出 `IndexError`。
