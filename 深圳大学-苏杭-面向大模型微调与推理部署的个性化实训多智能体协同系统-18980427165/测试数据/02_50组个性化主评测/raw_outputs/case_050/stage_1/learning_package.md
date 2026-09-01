# 学习包

## 块 1

**本块学习检查**：读完本节后，请合上资料，用自己的话说说：为什么没有 Python 解释器，你写的 `.py` 文件就“只是一堆文本”？说清楚即可，不用写代码。

---

## 1.1 Python 解释器：程序的“翻译官”

你学 AI、想做 LoRA 微调，最终要写的是 Python 代码。但先别急着敲键盘——你得知道代码是怎么“跑”起来的。

**问题在于**：你写的代码是人类能看懂的文字，而电脑只认 0 和 1。这两者之间，需要一个“翻译官”。

**Python 解释器**干的就是这件事：它把你写的代码逐句翻译成电脑能执行的指令。没有它，你的 `.py` 文件就只是一堆普通文本，电脑根本不知道你想干什么。

打个比方：你给一个只懂机器语言的外国人写了一封中文信，没人翻译，信就是废纸。解释器就是那个随叫随到的翻译。

**对你意味着什么**：你后面装 PyTorch、跑微调脚本，所有操作都建立在“解释器能读懂你的代码”这个基础上。所以这一步不是走形式，是整个开发工作台的“地基”。

**记住一句话**：先有解释器，你的代码才有机会变成 AI 模型真正跑起来。

## 块 2

**IDE 与虚拟环境：为 LoRA 微调搭好工作台**

你学 LoRA 微调，会写大量 Python 代码，还要反复调试。记事本写代码效率太低，所以需要 IDE——它提供语法高亮、自动补全、运行调试，就像用 Word 写文档一样顺手。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大），选一个即可。

接着是虚拟环境。微调 LoRA 时，不同项目可能依赖不同版本的库，比如 A 项目要 `numpy 1.0`，B 项目要 `numpy 2.0`，都装进同一个 Python 会冲突报错。虚拟环境就是给每个项目一个独立的 Python

## 块 3

### 2.1 Windows 用户：为 LoRA 微调铺平第一步

你想学 AI，目标是用 LoRA 微调模型，而你的电脑有 RTX 3060 12GB 显卡——这完全够用。但在跑任何微调代码之前，得先让 Python 在 Windows 上“安家”。下面三步，照着做就行。

**第一步：下载安装包**

打开浏览器，访问 https://www.python.org。鼠标移到顶部菜单 **Downloads**，网站会自动推荐适合你系统的 Windows 版本（比如 `Python 3.13.x`）。点击下载 **Windows installer (64-bit)** 即可。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这一步至关重要。它会把 Python 加入系统的“环境变量”。如果不勾选，之后在命令行输入 `python`，系统会提示“不是内部或外部命令”，后续所有操作都会卡住。

2. 点击 **"Install Now"**（默认安装）  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，能避免后续路径报错。想改路径的话，务必确保新路径**没有中文或空格**。

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

现代 macOS（12.3 起）不再预装 Python 2，但系统可能仍带有供脚本使用的旧版 `python3`。**强烈建议自行安装最新版 Python**，避免依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**不提供** `python` 命令，系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：打开终端，运行 `python3 --version` 和 `pip3 --version`，确认两行命令都输出版本号（如 `Python 3.10

## 块 5

**本块学习检查**：运行 `python3 --version`，确认输出为 Python 3 开头的版本号，即可进入下一步。

Linux 常自带 Python，但版本可能偏旧。在 Ubuntu/Debian 终端依次执行：

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

第一条命令刷新软件源；第二条安装 Python 3、pip 及虚拟环境模块（`-y` 自动确认，免手动输入）；第三条验证版本。`python3-venv` 是后续建虚拟环境的必备件，一次装齐避免麻烦。

## 块 6

### 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，所以环境选择要围绕“能跑 GPU 训练”来思考。不同阶段需求不同，选对能少走弯路：

**日常写 Python、练命令行（Windows 用户）**
推荐启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它让你在 Windows 里获得接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致，比 Windows 原生 CMD 更贴近实战。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**
直接用本机终端即可，本身就是类 Unix 环境。

**LoRA / QLoRA 微调等 GPU 实训（你的核心场景）**
应使用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。你的 RTX 3060 12GB 完全够入门 LoRA 微调，但前提是系统选对。

> **⚠️ 不推荐**：不要把 **“CentOS + VirtualBox 虚拟机”当作最佳

## 块 7

**pip 镜像源：什么时候用，怎么用**

你正在为 LoRA 微调做准备，安装 PyTorch 等库是第一步。`pip` 默认从官方 PyPI（`pypi.org`）下载，这是最权威、最完整的源，**应始终作为首选**。

**什么时候才需要镜像源？** 仅当你网络受限、访问官方源明显变慢或经常超时时，才临时切换。镜像是"网络不好时的临时加速手段"，**不是必须永久替换的默认设置**。

**推荐做法：临时指定镜像源**。如果某次安装慢，只对这一次安装临时指定镜像，不改动全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式？** 它不影响你平时的默认行为，官方 PyPI 仍是默认源；网络恢复或镜像出问题时，无需清理配置。

**本块学习检查

## 块 8

**个性化教学正文（3.3 进阶：将镜像设为默认）**

你正在为 LoRA 微调搭建环境，而 LoRA 微调依赖 PyTorch 和 HuggingFace 工具链，这些库体积大、依赖多，下载慢会浪费你宝贵的每日 1 小时学习时间。本节能帮你彻底解决这个问题——但请注意，**仅在长期网络受限时使用**。官方 PyPI 始终是最可靠来源，网络恢复后建议改回。

**为什么需要配置文件？**
之前你每次用 `-i` 指定镜像，现在把镜像“写死”为默认值，pip 每次自动从清华源下载，省去重复输入。

**Windows 操作步骤：**
1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开配置目录（类似 `C:\Users\你的用户名\AppData\Roaming`）。
2. 新建文件夹，命名为 `pip`。
3. 在 `pip` 文件夹内新建文本文件，命名为 `pip.ini`。
4. 用记事本打开，写入：
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```
`index-url` 告诉 pip 默认从该地址下载。

**macOS / Linux 操作步骤：**
1. 终端执行：`mkdir -p ~/.config/pip`
2. 创建并编辑配置：`nano ~/.config/pip/pip.conf`
3. 写入同样内容，保存退出（`Ctrl+O` 保存，`Ctrl+X` 退出）。

**⚠️ 关键警告：不要加 `trusted-host`**
网上教程常让你加 `trusted-host = ...`，**除非镜像无法通过 HTTPS 证书验证，否则千万别加**。它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用无需此参数。只有企业内网自建源或确认来源可信时才临时加。

**验证配置是否生效：**
命令行输入 `pip config list`，看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

**改

## 块 9

好的，我们开始搭建你的 Python 开发工作台。既然你的目标是学习 LoRA 微调，那么一个顺手的代码编辑器就是你的第一件装备。这里推荐 VS Code，它轻量、免费，而且插件生态丰富，能很好地支持 Python 开发。

**安装步骤**

1.  打开浏览器，访问 `https://code.visualstudio.com`，下载对应你操作系统的安装包。
2.  运行下载的安装程序，一路点击“下一步”完成安装即可。

**配置 Python 环境**

安装完成后，我们需要让 VS Code 认识 Python 代码。

1.  打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），在搜索框中搜索并安装以下两个插件：
    *   **Python**（微软官方出品，提供语法高亮、调试和智能提示）。
    *   **Pylance**（Python 语言服务器，让代码提示更智能）。
2.  安装插件后，VS Code 才能正确识别 `.py` 文件，并提供代码补全、错误提示和一键

## 块 10

**PyCharm Community：你的 LoRA 微调工作台**

既然你的目标是 LoRA 微调，代码会涉及模型加载、训练循环等复杂结构，一个专业的 IDE 能帮你事半功倍。PyCharm 社区版免费且开箱即用，内置代码分析，非常适合这类项目。

**安装要点**：从官网下载 Community 版。安装时务必勾选 `Add to PATH`，这样后续在终端启动 PyCharm 更方便。

**首次启动关键配置**：
1. 新建项目时，`Location` 路径**不要含中文**，避免编码问题。
2. `Interpreter` 选择 `Previously configured interpreter`，指向你已装好的 Python 3.10+。

**本块学习检查**：请口述安装 PyCharm 时需勾选的两个选项，以及新建项目时路径的注意事项。

## 块 11

### 5.1 为什么用 Conda

你想做 LoRA 微调，这意味着你马上要接触 PyTorch、CUDA 这类依赖关系复杂的深度学习工具。如果直接在系统 Python 里装，很容易出现版本冲突——这个项目要 PyTorch 2.x，那个项目要 1.x，装来装去系统就乱了。

**Conda** 就是来解决这个问题的。它不只是虚拟环境工具，还能直接管理 Python 解释器版本。每个项目可以拥有独立的 Python 和独立的包，互不干扰。对于深度学习，这几乎是标配。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等大量数据科学库 | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你的目标是 LoRA 微调，环境越干净越好，需要什么装什么，避免无关包干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为"不推荐"）。直接塞进系统 PATH 容易和已有 Python 冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 文件，按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`cond

## 块 12

### 5.3 创建你的第一个虚拟环境

**为什么这样做**：你计划用 LoRA 微调大模型，而不同项目常需要不同版本的 PyTorch 和 CUDA。现在学会创建环境，以后就能直接复用这套流程，避免版本冲突。

在终端中执行：

```bash
# 创建一个名为 llm-learn 的环境，指定 Python 3.10
conda create -n llm-learn python=3.10
```

**命令拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境起名叫 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定这个环境里安装 Python 3.10

执行后

## 块 13

## 5.4 激活与退出环境：进入你的“项目专属房间”

既然你的目标是 LoRA 微调，那这个“房间”就是你未来训练模型时，所有依赖和代码的专属工作区。我们一步步来。

**激活环境**（走进这个“独立房间”），在终端输入：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志。这意味着，你接下来运行的 `python` 和 `pip` 命令，都只作用于这个环境里，不会干扰系统里其他项目。这对 LoRA 微调尤其重要——不同项目依赖的 PyTorch 版本可能冲突，隔离环境能避免“改一个坏一片”。

**验证当前环境的 Python 版本**，输入：

```bash
python --version
```

应该显示 `Python 3.10

## 块 14

**为什么先配环境？**  
你正在为 LoRA 微调做准备，而微调要跑 PyTorch 等库。VS Code 只是个编辑器，它本身不运行 Python——真正干活的是你之前用 Conda 创建的 `llm-learn` 环境。配置这一步，就是让 VS Code 知道“该用哪个 Python 来跑你的代码”。配好后，你写的每一行代码都会自动使用这个环境里的库，不用每次手动切换。

**VS Code 配置步骤（跟着做，2 分钟搞定）**  
1. 打开终端（VS Code 里按 `` Ctrl+` ``），先激活环境：  
   `conda activate llm-learn`  
   （这步确保你当前终端已经在这个环境里）  
2. 在 VS Code 中按 `Ctrl + Shift + P` 打开命令面板。  
3. 输入并选择 `Python: Select Interpreter`。  
4. 在列表里找带有 `conda` 字样的选项，例如 `Python 3.10.x ('llm-learn': conda)`，点击它。  

**原理**：VS Code 能自动扫描你机器上的 conda 环境。选中后，VS Code 右下角会显示当前解释器路径，你在这个工作区里运行代码时，用的就是 `llm-learn` 里的 Python 和已装好的包。

**验证一下（必做）**  
在 VS Code 里新建一个文件 `test.py`，

## 块 15

## 7.1 在命令行验证

既然你的目标是LoRA微调，那第一步就是确保Python环境能跑起来。请按顺序操作：

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式
3. 看到 `>>>` 提示符后，输入：`print("Hello, Python! 我的环境准备好了！")`
4. 回车，若正确打印文字，输入 `exit()` 退出

**本块学习检查**：请观察并确认——你输入的代码是否原样打印出了那行文字？如果打印了，说明Python解释器工作正常，这是后续所有微调操作的基础。如果报错，请记录错误信息，这通常是环境激活或安装环节出了问题。

## 块 16

## 7.3 在 PyCharm 中验证

你的目标很明确——**LoRA 微调**。在真正开始微调之前，我们需要先确认你的“工作台”能正常运转。这一步就像厨师开火前先试灶：如果连 `hello.py` 都跑不起来，后面所有微调代码都无从谈起。

**操作步骤（三步走）**：

1. **新建项目**：打开 PyCharm，创建一个新项目，然后在项目中新建一个文件，命名为 `hello.py`。
2. **运行文件**：在 `hello.py` 文件上**右键点击**，从菜单中选择 `Run 'hello'`。
3. **查看结果**：PyCharm 窗口**下方**会弹出 Run 面板，里面应该显示你写的输出内容。

> **理论视角**：这一步的本质是验证“代码 → 解释器 → 输出”这条链路是否畅通。你写的每一行 Python 代码，都会经过解释器执行，结果呈现在 Run 面板中。LoRA 微调虽然复杂，但底层也是同样的执行机制——只是代码量更大、依赖更多而已。

---

## 常见问题排查（提前避坑）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选择 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 按第三章临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户使用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 cond

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

### Q-KB0203-001
以下函数定义中，哪个参数是**默认参数**？
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
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

### Q-KB0202-001
以下哪个是Python中**不可变**的数据结构？

### Q-KB0203-003
请定义一个类 `Dog`，包含：
- 类属性 `species = "Canis familiaris"`
- 实例属性 `name` 和 `age`（通过 `__init__` 初始化）
- 方法 `bark()` 返回 `"{name} says: Woof!"`

### Q-KB0205-009
写一个装饰器 `timer`，测量并打印被装饰函数的执行时间（秒），同时不改变原函数的返回值。然后用它装饰一个计算前100万个数平方和的函数。

### Q-KB0205-006
以下关于生成器的说法，**错误**的是？
