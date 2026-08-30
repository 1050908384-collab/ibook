# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你写下的 Python 代码是给人读的文字，而电脑只认 0 和 1。**Python 解释器**就是那个"翻译官"——它把你写的代码逐行翻译成电脑能执行的指令。没有它，你保存的 `.py` 文件只是一堆普通文本，电脑根本不知道你想让它做什么。

**用你熟悉的 C++/Java 来类比**：就像编译器负责把 C++/Java 源码变成机器码一样，Python 解释器承担了类似职责。区别在于，Python 解释器是边翻译边执行（解释执行），不需要单独的编译步骤。你写好代码，直接交给解释器运行即可。

**对你的 LoRA

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已熟悉C++/Java的工程化工具链，这里用类比快速对齐Python生态。

**IDE**：记事本也能写代码，但效率极低。IDE提供语法高亮、自动补全、运行调试，让你写代码像用Word写文档。推荐**VS Code**（轻量免费）或**PyCharm**（专业强大）——选一个装好即可，后续微调脚本都用它打开。

**虚拟环境**：想象两个项目分别需要`numpy 1.0`和`numpy 2.0`，装进同一个Python会冲突。虚拟环境为每个项目建独立Python空间，互不影响。我们用**conda**管理，这是最主流方案。你之后做LoRA微调，会为它单独建一个

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

作为资深C++/Java工程师，你熟悉环境变量和命令行，本块的关键就一个：**勾选“Add python.exe to PATH”**，否则后续所有命令都跑不起来。

**第一步：下载安装包**

1. 浏览器访问 https://www.python.org
2. 顶部菜单 **Downloads**，网站自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击 `.exe`，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这会把 Python 加入系统环境变量。不勾选的话，命令行输入 `python` 会提示“不是内部或外部命令”，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径在 `C:\Users\你的用户名\AppData\Local\Programs\Python\Python3xx`，不含中文和特殊符号，避免后续路径报错。想改路径的话，确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `

## 块 4

### macOS 用户：安装 Python 3.10+

作为资深 C++/Java 工程师，你熟悉环境配置的重要性。macOS 上有个关键点：**macOS 12.3 起不再预装 Python 2**，系统自带的 `python3` 可能是旧版本，仅供系统脚本使用。因此，**强烈建议自行安装最新版 Python**，避免依赖系统版本。

**方法一：官方安装包（推荐）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install"。
3. 打开终端，验证安装：

```bash
python3 --version
pip3 --version
```

**安装注意事项**：若系统提示“无法打开，因为来自身份不明的开发者”，请右键点击安装包选择“打开”，或前往“系统偏好设置-安全性与隐私”允许安装。

**为什么用 `python3` 而不是 `python`**：现代 macOS 默认**不提供** `python` 命令（早期版本曾指向 Python 2，现已移除）。请始终用 `python3` / `pip3`，避免"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，终端执行：

```bash
brew install python
```

安装

## 块 5

作为资深工程师，你熟悉包管理，这里直接给你最精简的路径。

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv -y
python3 --version
```

`-y` 跳过确认；`python3-venv` 是虚拟环境模块，为后续隔离微调依赖做准备，一次装齐。

**本块学习检查**：用自己的话说明，安装 Python 环境与你 LoRA 微调目标的关系；遇到无法解释的术语（如 venv）记录下来再提问。

## 块 6

## 2.4 操作系统与实训环境该怎么选

你是资深 C++/Java 工程师，对命令行和系统环境不陌生，但 Python 生态的部署细节可能不熟。这里直接给你结论，按你的目标（LoRA 微调）来选环境。

**日常写 Python、练命令行（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**LoRA 微调等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当最佳方案**。CentOS 8 已停止维护；VirtualBox 默认**无法直通 NVIDIA GPU**，跑不了真正的 GPU 训练，只适合纯 CPU 隔离小实验。

**你的选择**：你有 RTX 4070 12GB，本地实体机装 Ubuntu LTS 即可直接跑 LoRA 实训；

## 块 7

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你作为资深C++/Java工程师，对“依赖管理”和“网络源”的概念不会陌生。Python的`pip`类似Maven或NuGet，默认从官方PyPI（`pypi.org`）拉包——这是最权威、最完整的源，应始终作为首选默认。

**什么时候才需要镜像源？** 仅当网络受限、官方源明显变慢或频繁超时。镜像只是“临时加速手段”，不是永久替换项。

**首选做法（推荐）**：只对某次安装临时指定镜像，不改全局配置：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**为什么推荐**：不影响平时默认行为，官方源仍是默认；网络恢复或镜像

## 块 8

### 3.3 进阶：将镜像设为默认（仅在长期网络受限时）

你作为资深工程师，肯定理解“默认配置”的利弊。这一节只针对**长期网络受限**的场景——如果你只是偶尔下载慢，用 `-i` 参数临时指定镜像即可，别动全局配置。官方 PyPI 始终是最可靠来源，网络恢复后记得改回。

**Windows 配置步骤：**

1. `Win + R` 输入 `%APPDATA%` 回车，打开 `C:\Users\你的用户名\AppData\Roaming` 目录。
2. 新建文件夹 `pip`，在其中新建文本文件 `pip.ini`。
3. 用记事本写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip 默认从清华镜像下载包。

> **⚠️ 别加 `trusted-host`**：网上教程常让你加一行 `trusted-host = ...`，但除非镜像 HTTPS 证书验证失败，否则**不要加**——它会跳过 TLS 校验，放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用无需此项。只有企业内网自建源或确认来源可信时才临时加。

**macOS / Linux 配置步骤：**

```bash
mkdir -p ~/.config/pip
nano ~/.config/pip/pip.conf
```

写入同样的 `[global]` 配置，`Ctrl + O` 保存，`Ctrl + X` 退出。

**验证配置：**

```bash
pip config list
```

看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'` 即成功。

**改回官方源：** 删除 `pip.ini` / `pip.conf`，或执行 `pip config unset global.index-url`。

---

**本块学习检查**

完成本块后，用自己的话说明：这个镜像默认配置与你当前“LoRA 微调”目标的关系是什么？你会在什么条件下启用它，什么条件下不用？遇到无法解释的术语（如 `index-url`、`trusted-host`）先记录下来，稍后提问。

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

## 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你作为资深C++/Java工程师，对IDE不陌生——VS Code 的定位类似轻量级工具，后续写Python脚本、调配置文件都用得上。

**安装步骤**（对你简化为三步）：

1. 访问 https://code.visualstudio.com，下载对应系统安装包。
2. 运行安装程序，一路下一步。
3. **必装插件**：点击左侧活动栏最下方的四个方块图标（扩展），搜索安装：
   - **Python**（Microsoft官方，提供语法高亮、调试、智能提示）
   - **Pylance**（Python语言服务器，让代码提示更智能）

> 安装Python插件

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

PyCharm Community 是 JetBrains 出品的免费 Python 专用 IDE，内置代码分析、版本控制、数据库工具，适合大型项目。作为资深 C++/Java 工程师，你熟悉 IDE 的价值——它帮你管理项目结构、调试和运行代码，这正是后续做 LoRA 微调实验时需要的“工作台”。

**安装要点**：
1. 访问官网下载页，选 **Community Edition**（社区版，免费）。
2. 安装时勾选 `Add "Open Folder as Project"`（右键快速打开项目）和 `Add to PATH`（命令行直接启动）。

**首次启动**：
1. 打开后选 `New Project`。
2. `Location` 选项目路径（**路径中不要含中文**，避免编码问题）。
3. `Interpreter` 选 `Previously configured interpreter`，指向系统已装的 Python 3.10+。

**

## 块 11

### 5.1 为什么用 Conda

你作为资深 C++/Java 工程师，对“依赖冲突”和“环境隔离”应该不陌生——就像 Maven 管理 Java 依赖、vcpkg 管理 C++ 库一样，Python 项目也需要一个“隔离舱”。虚拟环境就是为每个项目创建独立的 Python 空间，而 **Conda** 是目前最主流的虚拟环境管理工具。

它的独特之处在于：**不仅能创建虚拟环境，还能管理不同版本的 Python 解释器本身**。这对你后续学习大模型微调至关重要——因为 PyTorch、CUDA 这些深度学习依赖极其复杂，版本错一个就可能跑不起来。Conda 能帮你把整套环境“锁”住，避免“在我机器上能跑”的尴尬。

### 5.2 安装 Conda：Anaconda vs Miniconda

两个选择，本质是“全家桶” vs “纯净版”：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装 numpy、pandas、jupyter 等 | 数据分析、不想手动装包 |
| **Miniconda** | 体积小（约 50MB），只含 conda 和基础 Python，按需安装 | 项目开发、想要干净环境 |

**推荐 Miniconda**——你习惯了精确控制依赖，Miniconda 的“按需安装”哲学更贴合你的习惯。后续做 LoRA 微调时，需要什么装什么，环境清爽不臃肿。

**安装步骤（以 Miniconda 为例）：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统安装包（Windows 选 `.exe`，macOS 选 `.pkg`/`.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认，不要勾选 "Add Miniconda3 to my PATH environment variable"**（安装器标为“不推荐”）。直接塞进 PATH 容易和系统已有 Python 冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda

## 块 12

作为资深C++/Java工程师，你熟悉依赖管理和环境隔离（如Maven、Gradle、CMake）。Conda的虚拟环境本质类似，只是服务于Python生态。

直接执行：

```bash
conda create -n llm-learn python=3.10
```

- `conda create`：创建环境
- `-n llm-learn`：环境名（`n` = `name`）
- `python=3.10`：指定Python版本

终端会列出待装包并询问`Proceed ([y]/n)?`，输入`y`回车等待完成。

**为什么现在做这个**：后续LoRA微调需要特定版本的PyTorch和CUDA，这个环境就是你的隔离工作台。RTX 4070需要CUDA 11.8或更高版本，具体版本需根据驱动和PyTorch版本选择。

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你作为资深C++/Java工程师，理解“环境隔离”应该很自然——就像不同项目用不同版本的JDK或Node。Conda环境就是Python的“独立房间”。

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

---

**查看所有已创建的环境**：

```bash
conda env list

## 块 14

## 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 支持自动识别 conda 环境，选择后，你在这个工作区写的代码就会使用 `llm-learn` 环境中的 Python 和已安装的包。

> **工程师视角**：这和你为 C++ 项目切换工具链、为 Java 项目切换 JDK 版本是同一件事——确保编译器/解释器指向正确的依赖集合。选错环境，代码跑不起来时，第一反应就查这里。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点击 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选择 `Project: 你的项目名` → `Python Interpreter`。
3. 点击齿轮图标 → `Add...`。
4. 左侧选择 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径中选择 conda 环境的 Python 可执行文件：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

你已熟悉C++/Java的编译运行，Python的交互模式（`>>>`）相当于即时执行，无需编译。先激活环境：`conda activate llm-learn`，输入`python`进入交互模式，看到`>>>`后输入：

```python
print("Hello, Python! 我的环境准备好了！")
```

回车打印文字即环境可用，用`exit()`退出。

VS Code验证：新建文件夹`my-first-project`并打开，新建`hello.py`，输入`print("Hello from VS Code!")`，点击右上角小三角或按`Ctrl+F5

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

## 7.3 在 PyCharm 中验证

你已熟悉 IDE 的工程结构，这里只提炼关键动作：

1. 新建项目，创建 `hello.py`。
2. 右键点击文件 → `Run 'hello'`。
3. 下方 Run 窗口应输出结果。

> 对资深工程师的提示：这等价于你在 Java/C++ 里的“编译运行”入口，只是 Python 无需显式编译，直接解释执行。

---

## 第八章：常见问题排查（重点看与你相关的）

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
|------|----------|----------|
| Python 3.10+ | ✅ 已安装 | `python --version

## 测评题目

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0200-003
在Python交互环境中执行以下代码，验证pip是否可用：
```python
import pip
print(pip.__version__)
```

### Q-KB0205-002
生成器函数使用________关键字返回值（而非 `return`），调用生成器函数返回的是一个________对象。

### Q-KB0201-004
以下代码输出是？
```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)
```

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0204-001
以下哪种文件打开模式表示**追加写入**（不覆盖原内容）？

### Q-KB0202-005
以下代码输出是？
```python
d = {}
d.setdefault("key", []).append(1)
d.setdefault("key", []).append(2)
print(d)
```

### Q-KB0201-005
以下代码输出什么？
```python
for i in range(5):
    if i == 3:
        continue
    print(i, end=" ")
```
