# 学习包

## 块 1

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已经有预训练和SFT的底子，微调时真正要操作的对象是**代码**，而代码要跑起来，靠的就是解释器。它不关心你是做LoRA还是全参微调——它只负责把你写的Python指令翻译成GPU能执行的运算。

打个比方：你写`model.load_adapter("lora_weights")`，这行字GPU看不懂，解释器把它翻译成底层指令，RTX 4070才能干活。没有解释器，你的`.py`文件就是一堆文本，LoRA权重加载不了，训练也启动不了。

所以这一步不用深究，但要知道：**你之后所有微调脚本，最终都是通过解释器来驱动GPU的**。装好它，你的工作台才算通电。

## 块 2

你的目标是跑通 LoRA 微调，而微调脚本的调试、修改和运行都离不开一个顺手的编辑器。本块帮你把“工作台”搭好。

IDE 就像 Word 之于文档：提供语法高亮、自动补全和调试功能，让你改训练脚本时不再对着纯文本“抓瞎”。推荐 **VS Code**（轻量免费）或 **PyCharm**（专业强大），选一个装上即可。

微调时你可能同时维护多个实验项目，不同项目依赖的库版本可能冲突。**虚拟环境**就是给每个项目一个“独立房间”，互不干扰。用 **conda** 管理虚拟环境，这是目前最主流的方案。

Python 版本选 **3.10+**：错误提示更友好，且主流库已兼容。别用 Python 2（已停止维护）。

**本块学习检查**：完成安装后，用自己的话说说——IDE 和虚拟环境分别在你接下来的 LoRA 微

## 块 3

### 2.1 Windows 用户：为 LoRA 微调铺路

你的目标是跑通 LoRA 微调，而 Python 是启动 PyTorch 和 HuggingFace 工具链的“地基”。下面三步帮你把地基打牢。

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统“环境变量”。不勾选的话，后续在命令行输入 `python` 会提示“不是内部或外部命令”，你后续安装 PyTorch、运行微调脚本都会卡在这一步。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python 3.13.x

## 块 4

**macOS 用户：安装 Python 3.10+**

现代 macOS（Catalina 10.15 起）不再预装 Python 2，但系统可能带有旧版 `python3`。**强烈建议自行安装最新版**，避免依赖系统版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install"。
3. 打开终端，验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认不提供 `python` 命令，系统与官方包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

```bash
brew install python
```

安装后同样用 `python3 --version` 验证。

---

**本块学习检查**

完成本块后，用自己的话说明：本块内容（安装 Python 环境）与你的学习目标（LoRA 微调）有何关系？遇到无法解释的术语（如 `universal2`、`Homebrew`）时记录下来再提问。

## 块 5

在动手做 LoRA 微调前，你得先确保 Linux 环境里有可用的 Python 3 和 pip。Ubuntu/Debian 系统通常自带 Python，但版本可能偏老，所以用下面三条命令完成安装：

```bash
# 更新软件源
sudo apt update

# 安装 Python 3 和 pip
sudo apt install python3 python3-pip python3-venv -y

# 验证
python3 --version
```

**为什么这样写**：`-y` 参数让系统自动确认安装，省去手动输入 `y` 的步骤；`python3-venv` 是创建虚拟环境的模块，提前装好，后续做 LoRA 实验时隔离依赖会方便很多

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 6

### 2.4 操作系统与实训环境该怎么选

你已掌握预训练和SFT流程，下一步做LoRA微调，核心是让GPU真正跑起来。环境选择直接决定你能否顺利进入实操。

**日常写Python、练命令行（Windows用户）**：启用 **WSL2 + Ubuntu LTS**（如22.04/24.04）。它在Windows里提供接近真实生产的Linux环境，命令、路径、包管理与后续服务器一致，比原生CMD更贴近实战。管理员PowerShell执行 `wsl --install` 即可。

**macOS/Linux用户**：直接用本机终端，本身就是类Unix环境。

**LoRA/QLoRA微调、vLLM部署等GPU实训**：使用**装有Ubuntu LTS的NVIDIA GPU云服务器，或本地实体Linux（Ubuntu LTS）机器**。这类任务依赖NVIDIA驱动、CUDA与显卡直通，Ubuntu LTS对深度学习生态支持最成熟、文档最全。

> **⚠️ 不推荐**：不要把"CentOS + VirtualBox虚拟机"当最佳方案。CentOS 8已于2021年停止维护，CentOS Stream 8也已于2024年停止维护、生态落后；VirtualBox默认无法直通NVIDIA GPU，跑不了真正的GPU训练，只适合纯CPU隔离小实验。命令练习用WSL2/Ubuntu LTS

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你在准备 LoRA 微调时，需要安装 `transformers`、`peft` 等库。`pip` 默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，应始终保留为默认。

**只有**当你访问官方源明显变慢或经常超时时，才临时切换国内镜像——镜像是“网络不好时的临时加速手段”，不是永久替换。

**首选做法**：只对某次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

这样官方源仍是默认，网络恢复或

## 块 8

## 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，而微调过程需要反复下载预训练权重和依赖包。如果你长期处于网络受限环境，每次手动加 `-i` 参数指定镜像确实繁琐——这时才值得把镜像写进配置设为默认。但请记住：**官方 PyPI 仍是最可靠的来源**，网络恢复后建议改回或删除该配置。

### Windows 用户操作

1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 告诉 pip 默认从这个地址下载包。

> **⚠️ 关于 `trusted-host`**：网上很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**无需** `trusted-host`。只有企业内网自建源、或确实遇到证书报错且已确认来源可信时，才临时加它。

### macOS / Linux 用户操作

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

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`，说明配置成功。

### 改回官方源

删除上面创建的 `pip.ini`

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 9

**本块学习检查**：完成安装后，请用自己的话说明：VS Code 的插件与解释器配置，如何服务于你后续的 LoRA 微调代码编写与运行？遇到无法解释的术语（如“语言服务器”）时记录下来再提问。

---

**4.1 方案一：VS Code（轻量、免费、推荐）**

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。它不只适合 Python，也适合后续编写配置文件或脚本——这些在微调实验的调试过程中都会用到。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（微软官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（让代码提示更智能）

安装 Python 插件后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能——这对你

## 块 10

### 4.2 方案二：PyCharm Community（专业、开箱即用）

**为什么选它**：PyCharm 是 JetBrains 出品的 Python 专用 IDE，社区版完全免费。它内置代码分析、版本控制、数据库工具，适合大型项目开发——比如你后续做 LoRA 微调时，会同时管理数据脚本、训练脚本和配置文件，这类多文件项目正是它的强项。

**安装步骤**：
1. 访问 https://www.jetbrains.com/pycharm/download/。
2. 下载 **PyCharm Community Edition**（社区版，免费）。
3. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键菜单快速打开文件夹）
   - `Add to PATH`（命令行可直接启动 PyCharm）

**首次启动配置**：
1. 打开 PyCharm，选择 `New Project`。
2. 在 `Location` 处选择项目存放路径（**路径中不要含中文**）。
3. `Interpreter` 选择 `Previously configured interpreter`，然后点击 `Create` 创建项目。

**本块学习检查**：完成本块后，用自己的话说明本块内容

## 块 11

## 5.1 为什么用 Conda

你已经掌握了预训练和 SFT 流程，接下来要做 LoRA 微调。微调意味着你要在本地跑起模型训练，而训练代码依赖 PyTorch、CUDA 等一堆复杂库。如果直接在系统 Python 里装，很容易把环境搞乱——这个项目要 PyTorch 2.x，那个项目要 1.x，冲突起来很头疼。

**Conda** 就是来解决这个问题的。它能为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身。对深度学习来说，conda 几乎是必备工具。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你要做 LoRA 微调，环境越干净越好，需要什么装什么，避免无关包干扰训练环境。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**。直接塞进系统 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 来使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**。这样终端会自动被 conda 接管，而不是硬加 PATH。
4. **验证安装**：打开**新的**终端（

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

你已掌握预训练和SFT流程，接下来要做LoRA微调。微调时，不同项目常需不同版本的PyTorch和CUDA，混在一起会冲突。本块教你用一条命令创建独立环境，为后续微调项目隔离依赖。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `

## 块 13

### 5.4 激活与退出环境：为 LoRA 微调锁定“工作台”

你已经掌握了预训练和SFT流程，现在要进入LoRA微调。微调实验最怕“环境打架”——不同项目依赖的包版本冲突，会浪费大量调试时间。Conda环境就是你的“独立房间”，把LoRA实验所需的一切隔离起来。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

激活后，命令行前面会出现 `(llm-learn)` 标志。这意味着后续所有 `python` 和 `pip` 命令都只作用于这个环境，不会影响你系统里其他项目（比如之前SFT用的环境）。

**验证当前环境的 Python 版本**：

```bash
python --version
```

应该显示 `Python 3.10.x`。确认版本正确，避免后续安装PyTorch或HuggingFace库时出现兼容性意外。

**退出环境**

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

## 6.1 VS Code 配置 Conda 环境

你已经掌握了预训练和SFT流程，现在要进入LoRA微调实战。第一步，就是让VS Code用上你为微调准备的`llm-learn`环境。

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

## 6.2 PyCharm 配置 Conda 环境

如果你用 PyCharm，流程同样直接：

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/env

## 块 15

**本块学习检查**  
完成本块后，用自己的话说明：在命令行和VS Code中验证Python环境，与你做LoRA微调有什么关系？遇到无法解释的术语（如`conda`、交互模式）时记录下来再提问。

---

### 7.1 在命令行验证

1. 激活环境：`conda activate llm-learn`  
2. 输入`python`进入交互模式，看到`>>>`提示符。  
3. 输入`print("Hello, Python! 我的环境准备好了！")`并回车。  
4. 正确打印后，用`exit()`退出。

### 7.2 在VS Code中验证

1. 新建文件夹`my-first-project`，用VS Code

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

## 7.3 在 PyCharm 中验证

你的目标是用 LoRA 微调大模型，而微调代码要在 IDE 里跑起来。所以先确认 PyCharm 能运行 Python 文件：

1. 新建项目，创建 `hello.py`。
2. 右键点击文件 → `Run 'hello'`。
3. 下方 Run 窗口应输出结果。

看到输出，说明你的开发工作台能执行代码了——这是后续写 LoRA 微调脚本的基础。

---

## 第八章：常见问题排查

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 使用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 | pip 装到了错误的环境 | 激活目标 conda 环境后，使用 `python -m pip install 包名` 安装 |

---

## 本章小结

完成本章后，你的电脑上应该具备以下内容：

| 组件 | 状态 | 验证命令 |
|------|------|----------|
| Python 3.10+ |

## 测评题目

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

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

### Q-KB0200-002
执行以下命令进入Python交互环境，将输出粘贴作答：
```bash
python
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。
