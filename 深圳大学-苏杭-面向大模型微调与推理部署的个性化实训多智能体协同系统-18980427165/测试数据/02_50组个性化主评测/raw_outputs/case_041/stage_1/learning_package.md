# 学习包

## 块 1

### 1.1 Python 解释器：程序的"翻译官"

你写的 Python 代码是人类能看懂的文字，但电脑只认 0 和 1。**Python 解释器**的作用就是把你的代码翻译成电脑能执行的指令。没有它，你写的 `.py` 文件只是一堆文本。

**为什么这对你的 LoRA 微调目标重要？**
你后续要运行 PyTorch 和 Transformer 相关的训练脚本，这些脚本本质上是 `.py` 文件。解释器就是让这些脚本真正"跑起来"的引擎。现在把这一步装对，后面才能顺利执行微调代码。

**动手验证（约2分钟）**
打开终端（Windows 用 CMD 或 PowerShell，macOS/Linux 用 Terminal），输入：
```
python --version
```
如果看到类似 `Python 3.x.x` 的输出，说明解释器已装好；如果提示"不是内部或外部命令"，说明还没装或没配置好，回到上文检查安装步骤。

**本块学习检查**
完成本块后，用自己的话说明：Python 解释器在你的 LoRA 微调学习目标中扮演什么角色？如果遇到无法解释的术语（如"环境变量"），先记录下来，在后续学习中提问。

## 块 2

## 1.2 代码编辑器（IDE）：你的“工作台”

记事本也能写代码，但效率极低。**IDE（集成开发环境）**提供语法高亮、自动补全、运行调试等功能，让你写代码像用 Word 写文档一样舒服。推荐两款：**VS Code**（轻量免费）和 **PyCharm**（专业强大）。对于你的 LoRA 微调任务，选 VS Code 足够，轻量且启动快。

## 1.3 虚拟环境：项目的“独立房间”

想象你同时做两个项目：A 项目需要 `numpy 1.0`，B 项目需要 `numpy 2.0`。如果都装到同一个 Python 里，就会冲突报错。**虚拟环境**就是为每个项目创建独立 Python 空间，互不影响。本文用 **conda** 管理虚拟环境，这是最主流方案。

## 第二章：安装 Python 3.10+

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 3

### 2.1 Windows 用户：安装 Python 3.10+

**第一步：下载安装包**

1. 打开浏览器，访问 https://www.python.org。
2. 鼠标移到顶部菜单 **Downloads**，网站会自动推荐 Windows 版本（如 `Python 3.13.x`）。
3. 点击下载 **Windows installer (64-bit)**。

**第二步：运行安装程序（⚠️ 关键步骤）**

双击下载的 `.exe` 文件，弹出安装界面后，**务必做两件事**：

1. **勾选最下方的 "Add python.exe to PATH"**  
   这个选项把 Python 添加到系统"环境变量"。不勾选的话，命令行输入 `python` 会提示"不是内部或外部命令"，后续所有命令行操作都无法进行。

2. 点击 **"Install Now"**（默认安装）即可。  
   默认路径不含中文和特殊符号，避免后续路径报错。进阶用户若改路径，务必确保路径中**没有中文或空格**。

**第三步：验证安装**

1. 按 `Win + R`，输入 `cmd`，回车打开命令提示符。
2. 输入以下命令并回车：

```bash
python --version
```

3. 如果显示 `Python 3.13.x`（或你下载的版本号），说明安装成功！
4. 再输入：

```bash
pip --version
```

如果也显示版本号，说明 `pip` 也一并安装成功了。

---

**本块学习检查**：完成本块后，用自己的话说明本块

## 块 4

### 2.2 macOS 用户：安装 Python 3.10+

现代 macOS（Catalina 10.15 起）不再预装 Python 2，部分系统可能带有供系统脚本使用的旧版 `python3`。**强烈建议自行安装最新版 Python**，避免依赖系统自带版本。

**方法一：官方安装包（推荐新手）**

1. 访问 https://www.python.org，下载 **macOS 64-bit universal2 installer**（`.pkg` 文件）。
2. 双击运行，一路点击 "Continue" → "Install" 完成安装。
3. 打开终端（Terminal），输入验证：

```bash
python3 --version
pip3 --version
```

**为什么用 `python3` 而不是 `python`**：现代 macOS 上，命令行默认**并不提供** `python` 命令（早期版本曾指向 Python 2，现已移除），系统与官方安装包统一使用 `python3`。请始终用 `python3` / `pip3`，避免踩到"命令不存在"或误用系统版本的坑。

**方法二：Homebrew（适合已有 Homebrew 的开发者）**

如果你已安装 Homebrew，在终端执行：

```bash
brew install python
```

安装完成后，同样用 `python3 --version` 验证。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA 微

## 块 5

**本块学习检查**  
完成安装后，请用自己的话说明：为什么安装 Python 时需要同时安装 `python3-venv`？这与后续进行 LoRA 微调时创建独立环境有什么关系？若遇到不理解的术语（如“虚拟环境”），先记录下来再提问。

---

**安装步骤（Ubuntu/Debian）**  

1. 更新软件源：  
   ```bash
   sudo apt update
   ```

2. 安装 Python 3、pip 和虚拟环境模块：  
   ```bash
   sudo apt install python3 python3-pip python3-venv -y
   ```
   `-y` 表示自动确认，无需手动输入 `y`。`python3-venv` 用于

## 块 6

## 2.4 操作系统与实训环境该怎么选

你的目标是 LoRA 微调，这属于 GPU 实训，环境选择直接决定你能不能跑起来。

**日常写 Python、练命令（Windows 用户）**：启用 **WSL2 + Ubuntu LTS**（如 Ubuntu 22.04/24.04 LTS）。它在 Windows 里提供接近真实生产的 Linux 环境，命令、路径、包管理都和后续服务器一致。安装只需管理员 PowerShell 执行 `wsl --install`。

**日常写 Python（macOS / Linux 用户）**：直接用本机终端，本身就是类 Unix 环境。

**LoRA / QLoRA 微调等 GPU 实训**：用**装有 Ubuntu LTS 的 NVIDIA GPU 云服务器，或本地实体 Linux（Ubuntu LTS）机器**。这类任务依赖 NVIDIA 驱动、CUDA 与显卡直通，Ubuntu LTS 对深度学习生态支持最成熟。

> **⚠️ 不推荐**：不要把 **"CentOS + VirtualBox 虚拟机"当作最佳方案**。CentOS 8 已停止维护（2021年EOL），CentOS 7 也已于2024年6月停止维护，建议使用 Ubuntu LTS 等活跃维护的发行版；VirtualBox 对 NVIDIA GPU 直通支持有限且配置复杂，通常无法用于 GPU 训练，建议使用原生 Linux 或云 GPU 实例。

**结合你的 RTX 4060 8GB**：本地

## 块 7

**本块学习检查**：完成本块后，用自己的话说明镜像源与LoRA微调环境准备的关系；遇到无法解释的术语时记录下来再提问。

---

**3.1 什么时候才需要镜像源**

`pip` 是 Python 的包管理工具，用来安装第三方库（如 `numpy`）。默认从官方 PyPI（`pypi.org`）下载——这是最权威、最完整、应保留的默认源。

**只有**网络受限、访问官方源明显变慢或经常超时时，才临时切换到国内镜像。**镜像是"网络不好时的临时加速手段"，不是必须永久替换的默认设置**——官方源始终是首选。

**3.2 首选做法：临时指定镜像源（推荐）**

某次安装慢时，最干净的做法是**只对这一次

## 块 8

## 3.3 进阶：把镜像设为默认（长期网络受限才做）

你正在准备 LoRA 微调，后面会频繁用 pip 安装 PyTorch 生态的包。如果网络环境长期受限，每次手动加 `-i` 镜像地址确实麻烦，这时可以把它写进 pip 的配置文件，让 pip 默认就走镜像。

**但先记住**：官方 PyPI 始终是最可靠的来源。网络恢复后，建议把配置改回去或直接删掉。

---

### Windows 操作步骤

1. 按 `Win + R`，输入 `%APPDATA%`，回车，会打开类似 `C:\Users\你的用户名\AppData\Roaming` 的文件夹。
2. 在该文件夹下**新建一个名为 `pip` 的文件夹**。
3. 在 `pip` 文件夹内**新建一个名为 `pip.ini` 的文本文件**。
4. 用记事本打开 `pip.ini`，写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

`index-url` 就是告诉 pip：默认从这个地址下载包。

---

### macOS / Linux 操作步骤

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

### ⚠️ 关于 `trusted-host`，别乱加

很多教程会顺手加一行 `trusted-host = ...`。**除非镜像确实无法通过 HTTPS 证书验证，否则不要加**——`trusted-host` 会让 pip 跳过对该域名的 TLS 校验，等于放弃传输安全。清华、阿里等主流镜像都有合法 HTTPS 证书，正常使用**不需要**它。只有企业内网自建源、或确实遇到证书报错且确认来源可信时，才临时加。

---

### 验证配置是否生效

在命令行输入：

```bash
pip config list
```

如果看到 `global.index-url='https://pypi.tuna.tsinghua.edu.cn

## 块 9

### 4.1 方案一：VS Code（轻量、免费、推荐）

**为什么选它**：VS Code 是微软出品的免费编辑器，启动快、插件丰富。你后续做 LoRA 微调时，要频繁改 Python 脚本、看配置文件，它都能胜任。

**安装步骤**：
1. 访问 https://code.visualstudio.com，下载对应系统的安装包。
2. 运行安装程序，一路下一步即可。

**必装插件**：打开 VS Code，点击左侧活动栏最下方的四个方块图标（扩展），搜索并安装：
- **Python**（Microsoft 官方出品，提供语法高亮、调试、智能提示）
- **Pylance**（Python 语言服务器，让代码提示更智能）

装完后，VS Code 才能识别 `.py` 文件，提供代码补全、错误提示和一键运行功能。

**配置 Python 解释器**：
1. 按 `Ctrl + Shift + P`（Mac 是 `Cmd + Shift + P`）。
2. 输入 `Python: Select Interpreter`，选择你刚安装的 Python 版本（如 `Python 3.11.

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

PyCharm Community是JetBrains出品的免费Python专用IDE，内置代码分析、版本控制等功能，适合大型项目开发。

**安装步骤**：
1. 访问官网下载页，选择Community Edition（社区版，免费）
2. 运行安装程序，建议勾选：
   - `Add "Open Folder as Project"`（右键快速打开文件夹）
   - `Add to PATH`（命令行直接启动）

**首次启动配置**：
1. 打开PyCharm，选择`New Project`
2. `Location`处选择项目路径（**路径中不要含中文**）
3. `Interpreter`选择`Previously configured interpreter`，再选系统已装的Python 3.10+

**与LoRA微调的关系**：LoRA微调需要编写和调试Python代码（如加载模型、配置训练参数），PyCharm提供代码分析和版本控制，帮你更高效地管理微

## 块 11

## 5.1 为什么用 Conda

你已经掌握了 Python 基础，现在要进入 LoRA 微调实战。微调需要 PyTorch、CUDA 等复杂依赖，不同项目之间很容易互相干扰——这就是虚拟环境要解决的问题。

**Conda** 是目前最主流的虚拟环境管理工具。它的独特之处在于：不仅能创建独立的 Python 空间，还能直接管理不同版本的 Python 解释器本身。这意味着你可以为 LoRA 微调项目单独创建一个环境，指定需要的 Python 版本，再安装 PyTorch 等依赖，完全不影响系统其他项目。

## 5.2 安装 Conda：Anaconda vs Miniconda

两个选择：

| 选项 | 特点 | 推荐人群 |
|------|------|----------|
| **Anaconda** | 体积大（约 500MB+），预装了大量数据科学库（numpy、pandas、jupyter 等） | 做数据分析、不想手动装包的用户 |
| **Miniconda** | 体积小（约 50MB），只包含 conda 和基础 Python，按需安装 | 希望环境干净、做项目开发的用户 |

**推荐 Miniconda**。你的目标是 LoRA 微调，环境越干净越好，需要什么装什么，避免无关包干扰。

**安装步骤：**

1. 访问 https://docs.conda.io/en/latest/miniconda.html
2. 下载对应系统的安装包（Windows 选 `.exe`，macOS 选 `.pkg` 或 `.sh`，Linux 选 `.sh`）
3. 运行安装程序：
   - **Windows**：一路下一步，**保持默认、不要勾选 "Add Miniconda3 to my PATH environment variable"**。直接塞进系统 PATH 容易和已有 Python/其他工具冲突，导致 `python`、`pip` 指向不明。日常改用开始菜单里的 **"Anaconda Prompt (Miniconda3)"** 使用 conda。
   - **macOS/Linux**：运行 `.sh` 后按提示输入 `yes` 同意协议、选择安装路径，最后一步选 `yes` 让安装器执行 **`conda init`**。这样终端会自动通过初始化脚本接管 conda，而不是硬加 PATH。
4. **验证安装**：打开一个**新的**终端（Windows 用 "Anaconda

## 块 12

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

在终端执行：

```bash
conda create -n llm-learn python=3.10
```

- `conda create`：创建环境
- `-n llm-learn`：环境名（`n` 即 `name`）
- `python=3.10`：指定 Python 版本

回车后，看到 `Proceed ([y]/n)?`，输入 `y` 等待完成。

**为什么这样做**：后续 LoRA 微调需不同 PyTorch/CUDA 版本，现在学会建环境，以后直接复用。

## 块 13

## 5.4 激活与退出环境——为 LoRA 微调准备“独立房间”

你的目标是 LoRA 微调，这意味着后续要安装 PyTorch、Transformers 等库。如果直接装在系统环境里，很容易和别的项目冲突。Conda 环境就是给每个项目一个“独立房间”，互不干扰。

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

**本块学习检查**：完成本块后，用自己的话说明本

## 块 14

## 6.1 VS Code 配置 Conda 环境

1. 先激活环境：`conda activate llm-learn`
2. 打开 VS Code，按 `Ctrl + Shift + P`。
3. 输入 `Python: Select Interpreter`。
4. 选择带有 `conda` 字样的 Python 路径（如 `Python 3.10.x ('llm-learn': conda)`）。

**为什么这样写**：VS Code 能自动识别 conda 环境，选好后，工作区代码就会用 `llm-learn` 里的 Python 和已装包。

## 6.2 PyCharm 配置 Conda 环境

1. 打开 PyCharm，点 `File` → `Settings`（Mac 是 `PyCharm` → `Preferences`）。
2. 左侧选 `Project: 你的项目名` → `Python Interpreter`。
3. 点齿轮图标 → `Add...`。
4. 左侧选 `Conda Environment` → `Existing environment`。
5. 在 `Interpreter` 路径选 conda 环境的 Python（请根据实际安装路径调整）：
   - Windows：`C:\Users\你的用户名\miniconda3\envs\llm-learn\python.exe`
   - macOS/Linux：`~/miniconda3/envs/llm-learn/bin/python`

## 第七章：验证——运行你的第一行代码

环境搭好后，必须

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（LoRA微调）的关系；遇到无法解释的术语时记录下来再提问。

---

### 7.1 在命令行验证

1. 激活环境：`conda activate llm-learn`
2. 输入 `python` 进入交互模式
3. 看到 `>>>` 后，输入并回车：

```python
print("Hello, Python! 我的环境准备好了！")
```

4. 正确打印后，用 `exit()` 退出。

### 7.2 在 VS Code 中验证

1. 新建文件夹 `my-first-project`，用 VS Code 打开
2. 新建 `hello.py`，

## 块 16

### 在 PyCharm 中验证：跑通你的第一行代码

你的目标是 LoRA 微调，而微调的前提是能顺畅地运行 Python 代码。PyCharm 是你写代码的“工作台”，本块的任务就是确认这个工作台能正常“点火”。

**操作步骤（跟着做一遍）**

1. 打开 PyCharm，新建一个项目（Project）。
2. 在项目中新建一个 Python 文件，命名为 `hello.py`。
3. 在文件中输入一行代码：`print("Hello, LoRA!")`。
4. 在文件编辑区**右键点击**，选择 **`Run 'hello'`**。
5. 看屏幕**下方**的 **Run 窗口**，如果出现了 `Hello, LoRA!`，说明你的环境已经能跑代码了。

> 小提示：如果 Run 窗口没出现，检查一下是不是选错了文件，或者项目路径里带了中文/空格——这会影响运行。

---

### 常见问题排查（遇到再看）

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| 输入 `python` 提示“不是内部或外部命令” | 安装时未勾选 “Add to PATH” | 重新运行 Python 安装程序，选 “Modify”，勾选 “Add Python to environment variables” |
| `pip install` 速度极慢或超时 | 网络受限，访问官方 PyPI 慢 | 临时加 `-i` 指定国内镜像（如清华源）；长期受限再考虑设为默认源 |
| `conda` 命令找不到 | 未使用 Anaconda/Miniconda Prompt（Windows）/ 未执行 `conda init`（macOS/Linux） | Windows 用户用开始菜单的 `Anaconda Prompt (Miniconda3)`；macOS/Linux 运行 `conda init` 后重开终端 |
| VS Code 提示“未选择 Python 解释器” | 未配置解释器路径 | 按 `Ctrl + Shift + P` → `Python: Select Interpreter` 手动选择 |
| 激活 conda 环境后，`python` 版本不对 | 系统里存在多个 Python | 用 `where python`（Windows）或 `which python`（Mac/Linux）查看路径，确保指向 conda 环境 |
| pip 安装后，Python 里 `import` 报

## 测评题目

### Q-KB0200-004
验证conda环境是否配置正确：
```bash
conda --version && conda info --envs
```

### Q-KB0200-001
请在CMD/终端中执行以下命令，验证Python是否安装成功，将完整输出粘贴作答：
```bash
python --version
```

### Q-KB0205-007
写一个函数 `count_words(text)`，接收一段英文字符串，返回一个字典，统计每个单词出现的次数（忽略大小写）。

### Q-KB0202-004
请用**列表推导式**生成1到20中所有能被3整除的数的平方。

### Q-KB0202-006
请写代码：给定嵌套字典 `data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}`，提取所有年龄大于26的用户姓名。

### Q-KB0205-005
请写一个**带参数的装饰器** `retry(times=3)`，使被装饰的函数在抛出异常时自动重试指定次数。

### Q-KB0204-002
使用 `with open("file.txt", "r") as f:` 的好处是________。

### Q-KB0202-005
以下代码输出是？
```python
d = {}
d.setdefault("key", []).append(1)
d.setdefault("key", []).append(2)
print(d)
```
