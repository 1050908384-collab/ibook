# 学习包

## 块 1

你已经有 PyTorch 和 Transformer 基础，也清楚最终目标是 LoRA 微调 + 权重合并 + vLLM 部署。但环境这关，很多人会跳过“解释器是什么”直接装包，结果后面报错时一头雾水。

**1.1 Python 解释器：程序的“翻译官”**

你写的 `.py` 文件是人类可读的代码，但 GPU 和 CPU 只执行 0 和 1。**Python 解释器**就是那个把代码翻译成机器指令的“翻译官”。没有它，你的 `.py` 文件只是一堆文本，跑不起来。

对你而言，这个“翻译官”还有一层特殊含义：你后续用 PyTorch 写 LoRA 训练脚本、调用 vLLM 部署接口，所有代码都要先经过解释器翻译，才能被 RTX 4090 执行。所以，解释器装对了，后面每一步才顺。

**本块学习检查**

完成本块后，用自己的话说明：Python 解释器在你的“LoRA 微调 + 权重合并 + vLLM 部署”流程中扮演什么角色？如果遇到无法解释的术语（比如“解释器”和“编译器”的区别），先记录下来，稍后提问。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你的 RTX 4090 已就位，但写代码不能靠记事本。**IDE（集成开发环境）**就是你的“工作台”，提供语法高亮、自动补全、运行调试，让你像用 Word 一样写 Python。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），二选一即可。

接下来是**虚拟环境**——每个项目的“独立房间”。你马上要跑 LoRA 微调和 vLLM 部署，不同项目依赖版本可能冲突（比如 A

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是用 RTX 4090 做 LoRA 微调并合并权重，再部署 vLLM。这一切都建立在 Python 能正常运行的基础上。本块就是帮你把 Python 环境装好，让后续的 `pip install torch`、`transformers` 等命令有地方执行。

**第一步：下载安装包**

打开浏览器访问 `https://www.python.org`，鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，后续在命令行输入 `python` 会提示"不是内部或外部命令"，所有命令操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径不含中文和特殊符号，避免后续路径报错。若改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`

## 块 4

## 2.2 macOS 用户：装好 Python，为 LoRA 微调铺路

macOS 12.3 起不再预装 Python 2，系统自带的 `python3` 仅供系统脚本使用。**强烈建议自行安装最新版 Python**，避免依赖系统版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 命令行默认**不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；

## 块 5

**本块学习检查**：完成安装后，用自己的话说明“Python 环境”与你后续“LoRA微调+权重合并+vLLM部署”目标的关系；遇到不懂的术语（如venv）先记下再提问。

在Ubuntu/Debian上，先更新软件源，再安装Python 3、pip和venv（`-y`自动确认）。验证版本：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

你的RTX 4090跑LoRA微调，依赖Python环境承载PyTorch和HuggingFace工具链。venv隔离项目依赖，避免污染系统Python——这是后续微调和部署前的

## 块 6

## 2.4 操作系统与实训环境该怎么选

你已有 PyTorch 和 Transformer 基础，接下来要跑 LoRA 微调、合并权重，再用 vLLM 部署——这三步都绕不开 GPU 环境。选对环境，能让你少踩很多坑。

**日常写 Python、练命令行（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比在 Windows 原生 CMD 里练更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**LoRA / QLoRA 微调、vLLM 部署等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。你的 RTX 4090 在 Ubuntu LTS 下驱动和 CUDA 支持最稳，后续跑 LoRA 训练和 vLLM 推理都顺畅。

> **⚠️ 不推荐**：别把 **"CentOS +

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`）。默认情况下，它从官方 PyPI 服务器（`pypi.org`）下载——这是最权威、最完整的源，**应该始终作为首选**。

只有当你网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。记住：**镜像是网络不好时的临时加速手段，不是永久替换**。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响你平时的默认行为，官方源仍是默认；网络恢复或镜像出问题时，无需清理配置。

---

**本块

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 8

### 3.3 进阶：把镜像设为默认（仅限长期网络受限时）

你正盯着 RTX 4090 准备跑 LoRA 微调，但每次 `pip install` 都卡在下载上——这才是你真正要解决的问题。**只有当你长期处于网络受限环境、几乎每次都要用镜像时**，才把镜像写入配置作为默认源。配置后记住：官方 PyPI 仍是最可靠的来源，网络恢复后建议改回或删除该配置。

**Windows 用户：**

1. 按 `Win + R`，输入 `%APPDATA%`，回车。这会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么这样写**：`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都提供合法的 HTTPS 证书，正常使用**无需** `trusted-host`。只有在企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

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

如果看到 `global.index-url='https://pypi.tuna.ts

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的目标是用 RTX 4090 做 LoRA 微调并合并权重，最终用 vLLM 部署。这需要你写大量 Python 脚本、改配置、跑训练命令。VS Code 就是你的“主战场”——它轻量、免费、启动快，插件丰富，后续学 Linux 命令行、写脚本时都能复用。

**安装要点**（按步骤操作）：

1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。
3. **必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
   - **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
   - **Pylance**（Python 语言服务器，让代码提示更智能）

装完插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能——这对你

## 块 10

**本块学习检查**：完成安装后，请用自己的话说明：为什么选择 PyCharm Community 作为你的 LoRA 微调项目开发环境？它与你后续的权重合并、vLLM 部署目标有何关联？遇到无法解释的术语（如“解释器”）时记录下来再提问。

---

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具，适合大型项目开发——你的 LoRA 微调脚本、权重合并逻辑都将在这里编写和调试。

**安装步骤**：

1. 访问 https://www.jetbrains.com/pycharm/download/
2. 下载 **PyCharm Community Edition**（社区版，免费）
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开项目文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置

## 块 11

### 5.1 为什么用 Conda

你之前已经掌握了 PyTorch 基础和 Transformer 架构，接下来要做的 LoRA 微调、权重合并和 vLLM 部署，每一步都依赖复杂的 Python 包组合（PyTorch、CUDA 版本、transformers、peft、vllm 等）。如果直接在系统 Python 里装，版本冲突会很快让你寸步难行。

**Conda** 是目前最主流的虚拟环境管理工具，它不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身。对于深度学习这种依赖复杂的场景，conda 几乎是必备工具——你可以为 LoRA 微调建一个环境，为 vLLM 部署建另一个环境，互不干扰。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 1GB 或更大，安装后占用空间可能达数 GB），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 80MB，安装后基础环境占用约 400MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**本文推荐 Miniconda**，因为环境更干净，后续需要用什么包装什么包——你的 RTX 4090 环境只需要 PyTorch、CUDA 相关包，不需要一堆用不到的数据科学库。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html。
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）。
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**（该选项被安装器标为"不推荐"）。把 Miniconda 直接塞进系统 PATH 容易和系统已有的 Python / 其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你已有 PyTorch 和 Transformer 基础，但 LoRA 微调与 vLLM 部署需要独立的环境来隔离不同版本的 PyTorch 和 CUDA。现在创建第一个虚拟环境，就是为后续项目铺路。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

拆解一下：
- `conda create`：创建环境的命令
- `-n llm-learn`：环境名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定装 Python 3.10

## 块 13

## 5.4 激活与退出环境：为 LoRA 微调项目隔离“工作台”

你的目标是 LoRA 微调 + 权重合并 + vLLM 部署，这意味着你会在同一台 RTX 4090 上反复切换不同项目。Conda 环境就是为每个项目准备的“独立房间”——激活它，你的 `python` 和 `pip` 命令就只作用于当前项目，不会污染系统或其他项目。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。看到它，就确认你后续所有命令都运行在这个环境里了。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`——这是你为微调项目配置好的版本。

**退出环境**（回到系统默认环境）：

```bash
conda deactivate
```

**本块学习检查**：完成

## 块 14

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

## 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

> **实操提示**：你的 RTX 4090 上跑 LoRA 微调，依赖 PyTorch 等包。确保 VS Code 选中 `llm-learn` 环境，代码才能调用该环境里装好的 PyTorch 和 HuggingFace 工具链。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调+权重合并+vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是微调并部署模型，但一切从“能跑通代码”开始。本块就是验证你的工作台是否可用——用命令行和VS Code各跑一次`print`，确认环境没白装。

**命令行验证**（适合快速确认环境）：
1. 激活环境：`conda activate llm-learn`
2. 输入`python`回车，看到`>>>`提示符。
3. 输入`print("Hello, Python! 我的环境准备好了！")`，回车。
4. 正确打印后，输入`

## 块 16

### 7.3 在 PyCharm 中验证

你的目标是用 RTX 4090 做 LoRA 微调并合并权重，再通过 vLLM 部署。在进入这些高级流程前，先确认开发环境能跑通最简单的代码，这能帮你提前暴露环境配置问题，避免后续微调时被琐碎错误打断。

操作步骤：

1. 打开 PyCharm，新建一个项目（路径不要包含中文或空格）。
2. 在项目中新建文件 `hello.py`，输入一行代码：`print("Hello, LLM!")`。
3. 右键点击文件 → 选择 `Run 'hello'`。
4. 查看下方 Run 窗口，确认输出了 `Hello, LLM!`。

如果输出正确，说明 PyCharm 已正确关联 Python 解释器，你的开发工作台就绪了。

---

### 第八章：常见问题排查

你在配置环境时可能遇到以下问题，对照处理：

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import

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

### Q-KB0203-003
请定义一个类 `Dog`，包含：
- 类属性 `species = "Canis familiaris"`
- 实例属性 `name` 和 `age`（通过 `__init__` 初始化）
- 方法 `bark()` 返回 `"{name} says: Woof!"`

### Q-KB0202-002
字典 `d = {"a": 1, "b": 2}`，获取键 `"a"` 对应值的方法是________；安全获取不存在的键 `"c"`（返回默认值0）的方法是________。

### Q-KB0203-002
Python中，`*args` 接收的是________类型；`**kwargs` 接收的是________类型。

### Q-KB0205-006
以下关于生成器的说法，**错误**的是？

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0201-004
以下代码输出是？
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```
