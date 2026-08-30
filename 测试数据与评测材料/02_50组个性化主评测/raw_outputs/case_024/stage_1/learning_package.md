# 学习包

## 块 1

### 1.1 Python 解释器：程序的“翻译官”

你写代码时用的是人类语言，但你的双卡 RTX 4060 只认 0 和 1。**Python 解释器**就是那个“翻译官”——它把你写的 `.py` 文件里的文字，逐行翻译成字节码，再由虚拟机执行，最终驱动 CPU 和 GPU 完成计算。没有它，你的 LoRA 微调脚本只是一堆无法运行的文本。

**为什么这对你很重要？** 你要跑的 LoRA 微调，本质上是让 PyTorch 调用 GPU 执行大量矩阵运算。PyTorch 的 Python 接口让你能用 Python 编写训练代码，其底层由 C++ 和 CUDA 实现，高效驱动 GPU 执行矩阵运算。你写的每一行训练代码，最终都要靠解释器翻译成底层指令，才能驱动你的 4060 算力。装好解释器，是打通“你的想法 → 代码 → GPU 算力”这条链路的第一步。

**动手验证**：打开终端，输入 `python

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

**IDE：你的"工作台"**

记事本也能写代码，但效率极低。IDE提供语法高亮、自动补全、运行调试等功能，让你写代码像用Word写文档一样舒服。推荐两款：**VS Code**（轻量免费）和**PyCharm**（专业强大）。你已有PyTorch基础，选VS Code即可——轻量、启动快，配合双卡RTX 4060跑LoRA微调时，资源留给训练而非编辑器。

**虚拟环境：项目的"独立房间"**

你正要做LoRA微调，可能同时跑多个实验：一个项目需要`numpy 1.0`，另一个需要`numpy 2.0`。全装进同一个Python会冲突报错。**虚拟环境**

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是双卡RTX 4060上做LoRA微调，而Python是这一切的地基。本块只做一件事：**在Windows上装好Python和pip**。

**第一步：下载安装包**

打开浏览器访问 `https://www.python.org`，鼠标移到顶部 **Downloads**，网站会自动推荐Windows版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，你后面装PyTorch、跑微调脚本全都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。如果你想改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入 `python --version` 并回车，显示 `Python 3.13.x`（或你下载的版本号）即成功。
3. 再输入

## 块 4

### 2.2 macOS 用户：为 LoRA 微调铺路

现代 macOS（Catalina 10.15 起）不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。为了后续顺利跑 LoRA 微调，**务必自行安装最新版 Python**，别依赖系统版本。

**方法一：官方安装包（推荐）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路 "Continue" → "Install"。
3. 终端验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而非 `python`**：macOS 默认不提供 `python` 命令，系统与官方包统一用 `python3`。始终用 `python3` / `pip3`，避免命令不存在或误用系统版本。

**方法二：Homebrew（已装则用）**

```bash
brew install python
```

同样用 `python3 --version` 验证。

---

**本块学习检查**：用自己的话说明，为什么安装 Python 时要避开系统自带版本？这对你后续配置 PyTorch 和 LoRA 微调环境有何影响？遇到不懂的术语（如 universal2）先记下，稍后提问。

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

Ubuntu/Debian 常预装 Python 但版本偏老。执行以下命令安装 Python 3.10：

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip -y
python3.10 --version
```

`-y` 自动确认，免手动输入。`python3.10-venv` 用于创建虚拟环境，避免后续补装。若系统默认 `python3` 仍指向旧版本，后续微调命令中请显式使用 `python

## 块 6

## 环境准备：操作系统与实训环境该怎么选

你已经掌握 PyTorch 基础和 Transformer 架构，目标是用双卡 RTX 4060 做 LoRA 微调。环境选对，后面少踩坑。

**日常写代码（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。管理员 PowerShell 执行 `wsl --install` 即可。

**日常写代码（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**GPU 实训（你的场景）**：LoRA 微调依赖 NVIDIA 驱动、CUDA 与显卡直通，应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已于 2021 年停止维护，CentOS Stream 8 也已于 2024 年停止维护；VirtualBox 默认**无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

**你的路线**：双卡 RTX 4060 是本地实体 GPU

## 块 7

你的目标是LoRA微调，而微调离不开 `pip` 安装各种依赖库。本块解决一个实际问题：**什么时候需要换镜像源**。

`pip` 默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，**应始终作为首选**。只有当网络受限、访问官方源明显变慢或经常超时，才临时切换国内镜像。记住：镜像只是“网络不好时的临时加速手段”，不是永久默认设置。

**首选做法：临时指定镜像源**。如果某次安装慢，只对这一次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这种方式干净且不影响平时默认行为——官方源仍是默认，网络恢复或镜像出问题时无需清理配置。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正打算用双卡 RTX 4060 做 LoRA 微调，这意味着要频繁从 PyPI 拉取 PyTorch 生态的依赖包。如果网络不稳，每次下载都卡在等待上，会直接拖慢你的实验节奏。本节的“设为默认”操作，就是为这种长期受限场景准备的——配置一次，后续 `pip install` 自动走镜像，省去每次手动指定源的麻烦。

**但请先记住一个前提**：官方 PyPI 始终是最可靠的来源。这个配置只是权宜之计，网络恢复后建议改回或删除，避免长期依赖镜像带来的潜在风险。

---

**Windows 用户操作步骤：**

1. 按 `Win + R`，输入 `%APPDATA%` 回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 新建一个名为 `pip` 的文件夹。
3. 在 `pip` 文件夹内新建一个名为 `pip.ini` 的文本文件。
4. 用记事本打开，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 的作用就是告诉 pip：默认从这个地址下载包。

**macOS / Linux 用户操作步骤：**

1. 终端执行 `mkdir -p ~/.config/pip`
2. 执行 `nano ~/.config/pip/pip.conf`
3. 写入同样的 `[global]` 配置内容
4. 按 `Ctrl + O` 保存，`Ctrl + X` 退出

---

**⚠️ 关于 `trusted-host` 的提醒**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——加了它，pip 会跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需**此配置。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加。

---

**验证配置是否生效：**

```bash

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续写 LoRA 微调脚本、改配置文件时都用得上。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

**配置 Python 解释器**：
1. 打开 VS Code，按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚安装的 Python 版本（如 `Python 3.13.x`）。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community 是 Python 专用 IDE 的免费开源版本，内置代码分析、版本控制等基础功能，适合中小型项目。你正为 LoRA 微调搭建工作台，一个 IDE 能帮你管理微调脚本、调试代码，是后续开发的基础设施。

**安装步骤**：访问官网下载社区版，运行安装程序时勾选 `Add launchers dir to the PATH`（命令行直接启动）等选项。

**首次启动**：新建项目，`Location` 选路径（**不要含中文**），`Interpreter` 选 `Previously configured interpreter`，指定系统 Python 3.10+。

**实践任务**：按上述步骤完成安装并新建一个项目，确认 IDE 能识别你的 Python 解释器

## 块 11

## 5.1 为什么用 Conda

你已经有 PyTorch 和 Transformer 的基础，接下来要搞 LoRA 微调，环境隔离这一步绕不开。原因很简单：LoRA 微调要装 `peft`、`transformers`、`datasets` 这些包，不同项目对 PyTorch 版本、CUDA 版本的要求可能不一样。如果全装在一个环境里，版本冲突会让你怀疑人生。

**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于深度学习这种依赖复杂的场景（PyTorch、CUDA 等），conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**——环境干净，后面 LoRA 微调需要什么包装什么包，不拖泥带水。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 执行 **`conda init`

## 块 12

你已经有 PyTorch 和 Transformer 基础，现在要进入 LoRA 微调实战。但微调不同模型时，PyTorch 和 CUDA 版本要求可能不同，直接装在一个环境里会互相冲突。

所以，我们先用 Conda 建一个独立环境，把项目隔离开。以后你跑 LoRA 微调，直接复用这套流程，换项目就换环境，互不干扰。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

拆开看：
- `conda create`：创建环境的命令
- `-n llm-learn`：环境名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定装 Python

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

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

## 第六章：在 VS Code / PyCharm 中使用 Conda

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

你的目标是 LoRA 微调，而微调脚本必须跑在正确的环境里。先激活环境：

```bash
conda activate llm-learn
```

打开 VS Code，按 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 会自动识别 conda 环境。选中后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包——包括后续微调要用的 PyTorch 等依赖。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm：`File` → `Settings`（Mac 是 `PyCharm` → `Preferences`），左侧选 `Project: 你的项目名` → `Python Interpreter`，点齿轮图标 → `Add...`，左侧选 `Conda Environment` → `Existing environment`，在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：

- Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
- macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后，必须验证一遍

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是LoRA微调，但任何微调都建立在“能跑通代码”的基础上。本块就是验证开发环境是否就绪的“点火测试”——用最简代码确认Python能执行，后续才能放心加载模型、跑训练脚本。

**动手验证（双卡RTX 4060）**：
1. 终端激活环境：`conda activate llm-learn`
2. 输入`python`进入交互模式，看到`>>>`后输入：
   ```python
   print("Hello, Python! 我的环境准备好了！")
   ```

## 块 16

## 7.3 在 PyCharm 中验证——你的第一行代码

环境已经搭好，现在来确认它真的能跑。既然你的目标是双卡 RTX 4060 上做 LoRA 微调，那第一步就是确保 Python 环境能正常执行代码——这是后续所有训练脚本的基础。

**操作步骤（跟着做一遍）：**

1. 打开 PyCharm，新建一个项目（Project），项目路径**不要包含中文或空格**。
2. 在项目中新建文件，命名为 `hello.py`。
3. 在文件中输入一行代码：`print("Hello, LoRA!")`（这是验证环境是否通畅的最简方式）。
4. 右键点击 `hello.py` 文件 → 选择 `Run 'hello'`。
5. 查看 PyCharm 下方的 **Run 窗口**——如果能看到输出结果，说明你的 Python 解释器配置正确，开发环境可以正常工作了。

**常见问题速查（遇到再看）：**

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install

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

### Q-KB0203-003
请定义一个类 `Dog`，包含：
- 类属性 `species = "Canis familiaris"`
- 实例属性 `name` 和 `age`（通过 `__init__` 初始化）
- 方法 `bark()` 返回 `"{name} says: Woof!"`

### Q-KB0204-004
以下代码中，`finally` 块在什么情况下执行？
```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("除零错误")
finally:
    print("清理完成")
```

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。

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
