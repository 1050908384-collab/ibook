# 学习包

## 块 1

**本块目标**：先搞懂“Python 解释器”是什么，再动手验证它。

---

### 1.1 Python 解释器：程序的“翻译官”

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。**Python 解释器**的作用，就是把你的代码“翻译”成电脑能执行的指令。

**关键点**：没有解释器，你写的 `.py` 文件只是一堆文本，电脑根本看不懂。

**动手验证（约2分钟）**：
1. 打开终端（Windows 用 CMD 或 PowerShell，Mac 用“终端”App）。
2. 输入 `python` 并回车。
3. 如果看到 `>>>` 提示符，说明解释器已就绪——你现在就在和它对话。
4. 输入 `print("hello")` 回车，看到输出 `hello`，就证明“翻译官”工作正常。

**小提醒**：如果提示“不是内部或外部命令”，别急，这是环境变量没配好，下一节会带你解决。

---

**本块学习检查**：用自己的话说，为什么没有解释器，`.py` 文件就“只是一堆文本”？

## 块 2

**你的目标：LoRA微调。** 在动手调模型前，得先把“工作台”搭好。本块只讲两件事：选编辑器、建独立环境。

**1. 编辑器：选 VS Code 就对了**
记事本也能写代码，但你会疯掉。**IDE** 提供语法高亮、自动补全、运行调试，写代码像用 Word 一样舒服。**VS Code** 轻量免费，适合你快速上手、边看教程边敲。

**2. 虚拟环境：项目的“独立房间”**
做 LoRA 微调时，你会装很多库。假设项目 A 要 `numpy 1.0`，项目 B 要 `numpy 2.0`，全装一起就冲突报错。**虚拟环境**为每个项目创建独立 Python 空间，互不干扰。用 **conda** 管理，这是主流方案。

**动手验证**：打开 VS Code，

## 块 3

**Windows 上装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，这需要 Python 环境来运行训练脚本。下面三步，跟着做就行。

**第一步：下载安装包**

1. 浏览器打开 https://www.python.org
2. 鼠标移到顶部 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 加入系统"环境变量"。不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认安装路径通常在 `%LOCALAPPDATA%\Programs\Python\Python3xx`（此路径仅为示例，实际路径因用户而异），不含中文和特殊符号，避免后续路径报错。想改路径的话，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash

## 块 4

### 2.2 macOS 用户：装好 Python，为 LoRA 微调铺路

你的目标是 LoRA 微调，这需要一套干净的 Python 环境。macOS 可能自带旧版 `python3`，但**强烈建议自行安装最新版**，避免依赖系统版本——系统自带的通常老旧，且可能被系统脚本占用，不适合做开发。

**方法一：官方安装包（推荐新手，最省心）**

1. 打开浏览器，访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击下载的文件，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入以下命令验证安装：

```bash
python3 --version
pip3 --version
```

看到版本号输出，就说明装好了。

**为什么用 `python3` 而不是 `python`？** 这是新手最容易踩的坑。现代 macOS 命令行默认**不提供** `python` 命令（早期版本曾指向 Python 2，现已移除）。系统与官方安装包统一使用 `python3`。所以请**始终用 `python3` / `pip3`**，避免遇到"命令不存在"或误用系统旧版本的麻烦。

**方法二：Homebrew（适合已有 Homebrew 的

## 块 5

**动手装好 Python，为 LoRA 微调铺路**

你的目标是 LoRA 微调，但先别急着碰模型——Linux 上 Python 版本太老会直接卡住你。跟着做，三步搞定：

```bash
# 1. 更新软件源，确保能装到最新版
sudo apt update

# 2. 一次装齐 Python 3、pip 和虚拟环境模块
sudo apt install python3 python3-pip python3-venv -y

# 3. 验证安装结果
python3 --version
```

**关键点**：`-y` 是自动确认，省得你手动输 `y`；`python3-venv` 是建虚拟环境用的

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，这意味着最终要跑 GPU 训练。环境选对了，后面能少踩很多坑。

**日常写 Python、练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它给你一个接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`，一步到位。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境，不用额外折腾。

**LoRA / QLoRA 微调等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟、文档最全。

> **⚠️ 别走弯路**：**不要**把 **"CentOS + VirtualBox 虚拟机"** 当最佳方案。CentOS 8 已停止维护，建议使用其替代品如 Rocky Linux 8 或 AlmaLinux 8，或直接使用 Ubuntu LTS；VirtualBox 这类桌面虚拟机**

## 块 7

### 3.1 什么时候才需要镜像源

`pip` 是 Python 的包管理工具，用来安装第三方库（比如 `numpy`、`pandas`）。默认情况下，`pip` 从官方 PyPI 服务器（`pypi.org`）下载，这是**最权威、最完整、最应保留的默认源**。

只有当你网络受限、访问官方源明显变慢或经常超时时，才考虑临时切换到国内镜像。**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

### 3.2 首选做法：临时指定镜像源（推荐）

如果只是某次安装慢，最干净的做法是**只对这一次安装临时指定镜像**，不改动任何全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐这种方式**：它不会影响你平时的

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你正在为 LoRA 微调准备环境，如果长期网络受限，可以把清华镜像设为 pip 默认源。**注意**：官方 PyPI 最可靠，网络恢复后建议改回。

**Windows 用户：**
1. 按 `Win + R`，输入 `%APPDATA%`，回车，打开配置文件夹。
2. 新建 `pip` 文件夹，在里面新建 `pip.ini` 文件。
3. 用记事本写入：
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```
`index-url` 告诉 pip 默认从这里下载包。

> **⚠️ 别加 `trusted-host`**：网上教程常让你加，但除非镜像 HTTPS 证书验证失败，否则别加——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法证书，正常使用**无需**它。若确需使用，应仅在确认镜像源可信且网络环境安全时临时添加，并尽快恢复官方源。替代方案：若证书验证失败，可用 `--cert` 参数指定可信证书，而非跳过验证。

**macOS / Linux 用户：**
1. 终端执行：
```bash
mkdir -p ~/.config/pip
```
2. 创建并编辑配置：
```bash
nano ~/.config/pip/pip.conf
```
3. 写入同样内容，`Ctrl + O` 保存，`Ctrl + X` 退出。

### 3.4 验证配置
命令行输入：
```bash
pip config list
```
看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

### 3.5 改回

## 块 9

**本块目标**：装好 VS Code，为后续 LoRA 微调写 Python 代码做准备。你只需跟着做，不用记概念。

**为什么选 VS Code**：免费、启动快，装插件后能写 Python，以后写脚本也能用。适合你快速上手。

**安装（约5分钟）**：
1. 打开 `https://code.visualstudio.com`，下载你系统的安装包。
2. 运行安装程序，一路“下一步”完成。

**装两个必装插件**：
打开 VS Code，点左侧最下方的四个方块图标（扩展），搜索并安装：
- **Python**（微软官方，提供语法高亮、调试、智能提示）
- **Pylance**（让代码提示更智能）

> 不装 Python 插件，VS Code 认不出 `.py` 文件，就没代码补全和运行按钮。

**选 Python 解释器**（关键一步）：
1. 按 `Ctrl + Shift + P`（Mac

## 块 10

**动手装好 PyCharm，为 LoRA 微调铺路**

你目标是跑通 LoRA 微调，那代码编辑器就是你的“主战场”。PyCharm 社区版免费且功能完整，适合管理稍大的项目，正好匹配你的学习节奏。

**安装要点（跟着点就行）**

1. 打开官网下载页：`https://www.jetbrains.com/pycharm/download/`
2. 选 **Community Edition**（社区版，免费那个）
3. 安装时勾选这两项，后面省事：
   - `Add "Open Folder as Project"`（右键文件夹就能直接打开）
   - `Add to PATH`（命令行也能启动 PyCharm；Windows/Linux 上对应选项为 `Add launchers dir to the PATH` 或 `Create Desktop Entry`，macOS 无此选项）

**首次启动，两步搞定**

1. 点 `New Project` 新建项目
2. `Location` 选存放路径——**记住：路径别带

## 块 11

### 5.1 为什么用 Conda

你的目标是 LoRA 微调，这属于深度学习实战。这类项目依赖复杂（比如 PyTorch、CUDA 等），如果直接在系统 Python 里装，很容易把环境搞乱。**Conda** 能帮你为每个项目创建独立的 Python 空间，还能管理不同版本的 Python 解释器本身——这正是深度学习的刚需。

### 5.2 安装 Conda：选 Miniconda

| 选项 | 特点 | 适合谁 |
|------|------|--------|
| **Anaconda** | 大（500MB+），预装 numpy、pandas 等 | 做数据分析、不想手动装包 |
| **Miniconda** | 小（约 70-100MB），只含 conda 和基础 Python | 希望环境干净、按需装包 |

**推荐 Miniconda**——你后面做 LoRA 微调，需要什么装什么，环境干净不臃肿。

**安装步骤：**

1. 打开 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS/Linux 选 `.sh`）
3. 运行安装：
   - **Windows**：一路下一步，**别勾选** "Add Miniconda3 to my PATH environment variable"（安装器也标了"不推荐"）。避免和系统已有 Python 冲突。日常用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 操作 conda。
   - **macOS/Linux**：运行 `.sh`，输入 `yes` 同意协议、选路径，最后一步**选 `yes` 执行 `conda init`**。这样终端会自动接管 conda，不用手动改 PATH。
4. **验证**：打开**新**终端（Windows 用 Anaconda Prompt；macOS/Linux 重开终端），输入：

```bash
conda --version
```

看到 conda 版本号（如 conda 24

## 块 12

## 5.3 创建你的第一个虚拟环境

**为什么现在做**：你马上要学 LoRA 微调，不同项目需要不同版本的 PyTorch 和 CUDA。学会创建环境，以后直接复用这套流程。

打开终端，执行：

```bash
conda create -n llm-learn python=3.10
```

**逐段拆解**：
- `conda create`：创建环境的命令
- `-n llm-learn`：给环境命名为 `llm-learn`（`n` 是 `name` 的缩写）
- `python=3.10`：指定环境内安装 Python 3.10

执行后终端会列出待安装包，并询问：

```bash
Proceed ([y]/n)?
```

输入 `y`

## 块 13

### 5.4 激活与退出环境：进入你的“独立房间”

你的目标是 LoRA 微调，这意味着你马上要开始动手跑代码了。为了让实验不“污染”系统，我们先把环境激活，就像走进自己的专属工作间。

**激活环境**（进入这个“独立房间”）：

```bash
conda activate llm-learn
```

**为什么这样写**：激活后，命令行前面会出现 `(llm-learn)` 标志。记住，只要看到这个标志，就表示你后续敲的所有 `python` 和 `pip` 命令，都只作用在这个环境里，不会影响系统或其他项目。这是项目隔离的核心。

**验证当前环境的 Python 版本**（确认你走对了房间）：

```bash
python --version
```

应该显示 `Python 3.10.x`。看到这个版本号，就说明环境激活成功了。

**退出环境**（

## 块 14

**本块学习检查**：完成下面两步，确认你的 `llm-learn` 环境已被 VS Code 正确选中。

**1. 在 VS Code 中切换 Conda 环境（约2分钟）**

- 先打开终端（Terminal），输入并运行：`conda activate llm-learn`
- 回到 VS Code 编辑器，按 `Ctrl + Shift + P` 打开命令面板。
- 输入 `Python: Select Interpreter` 并回车。
- 在弹出的列表中，选择带有 `conda` 字样的 Python 路径（例如 `Python 3.10.x ('llm-learn': conda)`）。

> **为什么这样做**：VS Code 会自动识别你电脑上的 conda 环境。选中后，你在当前工作区写的所有代码，都会使用 `llm-learn` 环境里的 Python 和已经装好的包（比如后面 LoRA 微调要用的库）。

**2. 快速验证（约1分钟）**

- 在 VS Code 里新建一个 Python 文件（比如 `test.py`）。
- 输入一行代码：`import sys; print(sys.executable)`
- 点击右上角的“运行”按钮。
- **检查输出**：如果打印出的路径中包含 `llm-learn`，说明配置成功，你的代码正在使用正确的环境。如果路径不对，回到第1步重新选择解释器。

## 块 15

### 在命令行验证你的 Python 环境

既然你的目标是 LoRA 微调，动手跑通环境是第一步。跟着做，很快就能看到反馈。

1. **激活环境**：打开终端，输入 `conda activate llm-learn` 并回车。
2. **进入 Python**：输入 `python` 回车，看到 `>>>` 提示符就对了。
3. **运行第一行代码**：在 `>>>` 后输入 `print("Hello, Python! 我的环境准备好了！")` 并回车。
4. **确认输出**：屏幕打印出这句话，说明环境正常。输入 `exit()` 退出。

**本块学习检查**：在命令行重复步骤 2-3，确认能再次看到输出。如果失败，检查是否已

## 块 16

**本块学习检查**：完成下面三步，确认你的 PyCharm 工作台能跑通第一行代码。

---

### 7.3 在 PyCharm 中验证

你的目标是 LoRA 微调，但任何微调实验都要从“能运行代码”开始。现在用 PyCharm 验证环境是否打通。

**操作步骤**：

1. 打开 PyCharm，新建一个项目（项目路径不要包含中文或空格）。
2. 在项目中新建文件，命名为 `hello.py`。
3. 在文件中输入：`print("Hello, LoRA!")`
4. 右键点击文件 → 选择 `Run 'hello'`。
5. 看下方 **Run 窗口**，如果输出了 `Hello, LoRA!`，说明你的开发环境已就绪。

---

### 常见问题排查（遇到再看）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示"不是内部或外部命令" | 安装时未勾选 "Add to PATH" | 重新运行 Python 安装程序，选 "Modify"，勾选 "Add Python to environment variables" |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示"未选择 Python 解释器" | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报错 |

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

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
```

### Q-KB0201-002
Python中，`type(3.14)` 的返回结果是________；`type("3.14")` 的返回结果是________。

### Q-KB0201-003
以下代码的输出是？
```python
x = 10
y = 3
print(x // y, x % y, x ** y)
```

### Q-KB0205-007
写一个函数 `count_words(text)`，接收一段英文字符串，返回一个字典，统计每个单词出现的次数（忽略大小写）。

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
