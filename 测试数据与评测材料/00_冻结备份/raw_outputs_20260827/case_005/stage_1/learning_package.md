# 学习包

## 块 1

## 环境变量配置：为LoRA微调铺路

> 你正在准备LoRA微调，但发现前置知识——环境变量配置——还不牢靠。别急，这一节我们用最快的方式补上这块短板。你已经有 `cd`、`ls`、`mkdir`、`cp`、`mv` 和文件编辑的基础，本节约1小时，目标是理解环境变量原理，掌握临时与永久配置方法。

---

### 一、环境变量是什么？

环境变量是操作系统提供给进程的键值对，进程可以读取和修改，子进程会继承父进程的环境变量。把它想成操作系统的"备忘录"——系统启动时加载，程序据此调整行为。

**常见例子**：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪找可执行程序（你输入 `python3`，系统靠它找到位置） |
| `HOME` | 当前用户家目录（如 `/home/zhangsan`） |
| `USER` | 当前登录用户名 |
| `SHELL` | 当前Shell路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码 |
| `PWD` | 当前工作目录，由 shell 维护，`cd` 命令会更新它 |
| `EDITOR` | 默认文本编辑器（影响 `git commit` 等） |

**为什么需要？** ①控制程序行为（如 `JAVA_HOME` 指定JDK路径）；②简化配置，免去每次输长路径；③子进程自动继承父

## 块 2

**临时变量：当前终端会话有效**

临时变量就像一次性的便利贴——只在当前终端窗口或 SSH 连接里有效。一旦关闭窗口或断开连接，它就消失了。适合快速测试，不用长期保存。

**动手试试（跟着敲一遍）：**

```bash
# 设置一个临时变量
export MY_VAR="hello_linux"

# 查看它的值
echo $MY_VAR
# 输出：hello_linux

# 删除这个变量
unset MY_VAR

# 再查看
echo $MY_VAR
# 输出：（空，因为已经删除了）
```

**关键点：`export` 的作用**

不加 `export`，变量只对当前 Shell 有效，子进程（比如你启动的另一个

## 块 3

**永久变量：写入配置文件，持久生效**

临时变量只在当前终端有效，关掉就没了。想让变量每次打开终端都在，就得写进**配置文件**——这相当于给系统一份“开机自动加载”的清单。

| 级别 | 配置文件 | 影响范围 | 是否需要 sudo |
|---|---|---|---|
| **用户级（推荐）** | `~/.bashrc` | 仅当前用户 | 否 |
| **用户级** | `~/.bash_profile` 或 `~/.profile` | 仅当前用户（登录时生效） | 否 |
| **系统级** | `/etc/profile` 等 | 所有用户 | 是 |

**选哪个？** 日常开发优先用 `~/.bashrc`（用户级）。它只影响你自己，改错了不会拖累系统

## 块 4

## 3.1 用户级配置：修改 `~/.bashrc`（推荐）

你要做 LoRA 微调，后面跑训练脚本、设环境变量是家常便饭。这个配置方法就是给你打地基的——改一次，终身有效。

**为什么推荐它？** `~/.bashrc` 是 Bash 的配置文件，**每次打开新终端自动加载**，日常开发最常用。

**动手操作（跟着敲）：**

```bash
# 1. 用 nano 打开配置文件（上节课学过的命令）
nano ~/.bashrc

# 2. 滚动到文件末尾，添加一行：
export MY_VAR="hello_permanent"

# 3. 按 Ctrl + O 保存，Ctrl + X 退出

# 4. 让修改立即生效（不重启终端）
source ~/.bashrc

# 5. 验证
echo $MY_VAR
# 输出：hello_permanent
```

**两个关键点：**

- **`source` 是什么？** 相当于"重新执行一遍这个配置文件"，让刚才的修改在当前终端立刻生效，不用关掉重开。
- **加载限制**：`~/.bashrc` 通常在打开新的**交互式终端**时加载，但并非所有 SSH 会话

## 块 5

## 3.2 用户级配置：~/.bash_profile 或 ~/.profile

这两个文件在**登录 Shell** 时生效，典型场景是 SSH 远程登录或系统开机登录。注意：如果两个文件同时存在，系统**优先读取 `~/.bash_profile`**。

日常开发中，`~/.bashrc` 更常用——因为每开一个终端它都会自动加载。只有当你明确需要“登录时才生效”的配置时，才考虑这两个文件。

## 四、环境变量的优先级

同一个变量在多处定义时，生效优先级从高到低：

1. **临时变量**（当前终端里 `export` 的）
2. **系统级配置**（`/etc/profile` 等）
3. **用户级配置**（`~/.bashrc` 等）

**核心规则：临时设置会覆盖配置文件。** 加载顺序上

## 块 6

### 5.1 PATH：命令的"搜索路径"

`PATH` 是最重要的环境变量，没有之一。它定义了系统**查找可执行文件**的目录列表。

#### 5.1.1 查看 PATH

```bash
echo $PATH
```

输出类似这样（多个路径用冒号 `:` 分隔）：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

当你在终端输入 `python3` 时，系统会**从左到右**依次在这些目录里找名为 `python3` 的可执行文件，找到了就执行，找不到就提示 `command not found`。

#### 5.1.2 查看命令具体位置

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 命令能告诉你某个命令到底是从 PATH 的哪个目录里找到的。

#### 5.1.3 修改 PATH

**追加路径**（添加到末尾，优先级较低）：

```bash
export PATH=$PATH:/new/path
```

**前置路径**（添加到开头，优先级更高，系统会先搜这里）：

```bash
export PATH=/new/path:$PATH
```

> **安全提示**：不要把 `.`（当前目录）加入 PATH。如果你在某个目录里下载了一个名为 `ls` 的恶意脚本，而当前目录又在 PATH 里，那么你执行 `ls` 时可能运行的是恶意脚本而非真正的系统命令。
>
> **避免重复添加**：多次 `source ~/.bashrc` 或在 `~/.bashrc` 中重复写入同一行，可能导致 PATH 里出现重复路径。建议定期检查 `~/.bashrc`，确保每条 PATH 修改只出现一次。

---

**本块

## 块 7

## 5.2 LD_LIBRARY_PATH：运行时动态库路径

这个变量解决的是“**程序跑起来时去哪找库**”的问题。你编译的程序依赖某个 `.so` 文件，但系统默认只在 `/usr/lib`、`/lib` 里找——如果库在其他目录，程序就会报错。`LD_LIBRARY_PATH` 就是告诉系统“额外去这些目录找”。

**先看当前值：**
```bash
echo $LD_LIBRARY_PATH
```

**临时添加 CUDA 的库路径：**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

> 注意：`export` 只对当前终端

## 块 8

**LIBRARY_PATH：编译时告诉编译器去哪儿找库**

你正在学 LoRA 微调，后面会用到 PyTorch、CUDA 这些库。当你在终端里敲命令编译或安装东西时，系统需要知道“库文件”放在哪儿。`LIBRARY_PATH` 就是干这个的——它告诉编译器（比如 `gcc`）在**编译阶段**去哪些路径找静态库（`.a`）和动态库（`.so`）。

设置方法很简单，把库路径追加进去：

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**三个变量别搞混**，记住这张表就够了：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令在任何位置都能运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态链接库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

**给你的重点提示**：做 LoRA 微调时，

## 块 9

**本块学习检查**：运行 `echo $PATH | grep cuda`，若看到 `/usr/local/cuda/bin` 即已配置；否则继续。

你的目标是 LoRA 微调，而 CUDA 环境变量是 PyTorch 调用 GPU 的前提。先快速确认系统是否已认识 CUDA：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

有输出（如 `/usr/local/cuda/bin`）→ 已配置，跳到下方验证。无输出 → 手动添加：

```bash
nano ~/.bashrc
```

在文件末尾追加两行（若 CUDA 非默认路径，替换 `/usr/local/cuda` 为实际路径）：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

保存退出后，加载配置并验证：

```bash
source ~/.bashrc
echo $PATH | grep cuda
nvcc --version
```

`nvcc --version` 输出版本号即成功。这步通过后，后续 PyTorch 才能识别 GPU，LoRA 微调才有硬件基础。

## 块 10

**7.1 查看所有环境变量**

先来快速浏览系统里有哪些环境变量。打开终端，输入：

```bash
printenv      # 简洁输出所有变量
env           # 更详细的输出
```

两个命令效果类似，`env` 输出更详细。先跑一下，感受下变量列表的规模。

**7.2 查看特定变量**

列表太长，我们只关心某个变量，比如 `PATH`（命令查找路径）。三种方式：

```bash
echo $PATH
printenv PATH
declare -p PATH    # 显示定义细节（包括是否 export）
```

`echo $PATH` 最常用，`declare -p` 能告诉你这个变量是否被导出（export）到子进程。

**7.3 测试变量是否生效**

假设你在 `~/.bash

## 块 11

## 8.1 变量未生效？先别急，按这个顺序查

你在配置环境变量后，发现新开的终端里变量还是没生效？这通常不是玄学，而是下面几个原因之一。既然你在为 LoRA 微调做准备，环境变量配不对，后面跑训练脚本时找不到 `CUDA_HOME` 或 `PYTHONPATH` 会非常头疼，所以这一步值得花两分钟排查干净。

**常见原因就四类：**

1. 改了 `~/.bashrc` 但没重新加载——配置文件只在终端启动时读取一次，改完必须手动刷新。
2. 配置文件写错了——最常见的是漏写了 `export` 关键字，导致变量根本没被导出。
3. 变量被后面某处覆盖了——比如你在 `~/.bashrc` 里设了 `PATH`，但 `~/.profile` 或某个脚本又把它重写了。
4. 你用的终端或 IDE 压根没加载 `~/.bashrc`——比如某些图形界面终端默认只读 `~/.profile`。

**动手排查（三步走）：**

```bash
#

## 块 12

## 9.1 脚本中临时修改变量（①）

在 LoRA 微调项目中，你经常需要在不同脚本间切换环境变量。最直接的方式是在脚本内部临时设置：

```bash
#!/bin/bash
export TEMP_VAR=123
# 仅在脚本执行期间生效，脚本结束后变量消失
```

`export` 的作用是让变量在当前 Shell 及它启动的子进程中可见。脚本运行完，这个变量就自动消失，不会污染你的终端环境。

---

## 9.2 使用 env 命令指定单次运行的变量（①）

如果你只想让某个变量在**一次命令**中生效，不需要写脚本，用 `env` 更轻量：

```bash
env MY_VAR=hello python3 my_script.py
# MY_VAR 仅在这次 python3 运行时有效
```

这对 LoRA 训练很有用——比如临时指定 `CUDA_VISIBLE_DEVICES=0` 或修改 batch size 相关的环境变量，跑完就恢复原状。

---

## 9.3 使用 .env 文件分组管理变量（④）

LoRA 项目通常涉及多个配置项（数据路径、模型名、输出目录等）。把它们集中到一个 `.env` 文件里，管理更清晰：

```bash
# 创建 ~/.env 文件
nano ~/.env
```

内容示例：

```bash
export PROJECT_HOME="$HOME/llm-learning"
export EDITOR="nano"
export TZ="Asia

## 块 13

### 场景 1：创建并验证自定义变量

既然你的目标是 LoRA 微调，配置环境变量是第一步——很多训练脚本依赖 `PATH` 和自定义变量来定位模型或数据。我们直接动手，快速过一遍。

**第一步：设置临时变量**

打开终端，输入：

```bash
export MY_NAME="zhangsan"
```

`export` 让这个变量对当前终端会话生效。注意：关掉终端就没了，这是“临时”的。

**第二步：验证**

```bash
echo $MY_NAME
```

`$MY_NAME` 会取出变量的值，终端应输出 `zhangsan`。如果没输出，检查引号是否配对。

**第三步：永久生效**

临时变量重启就丢，所以写入配置文件：

```bash
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc
```

`>>` 是追加，不会覆盖原有内容。`~/.bashrc` 是 Bash 启动时读取的配置。

**第四步：重新加载并验证**

```bash
source ~/.bashrc
echo $MY_NAME
```

`source` 让配置立即生效，不用重启终端。再次 `echo` 确认输出 `zhangsan`，说明永久配置成功。

---

**本块学习检查**：关掉终端

## 块 14

**本块学习检查**：完成下面三步，确认你已掌握 CUDA 环境变量配置的流程。

1. 打开终端，执行 `nano ~/.bashrc`，在文件末尾添加以下两行（假设 CUDA 装在 `/usr/local/cuda`）：
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
2. 保存退出后，运行 `source ~/.bashrc` 使配置生效。
3. 执行 `echo $PATH | grep cuda`，如果输出中包含 `/usr/local/cuda/bin`，说明配置成功。

**提示**：即使你还没装 CUDA，也可以先按此方法练习

## 块 15

## 系统级配置文件（①）——公共服务器环境配置速览

你在做 LoRA 微调时，如果用的是自己的机器，通常不需要碰系统级配置。但如果你在实验室或公司的**公共服务器**上训练，这些文件就很重要——因为系统级配置会影响**所有用户**。

### A.1 `/etc/environment`（①）

**特点**：不需要 `export` 关键字，直接写 `KEY=value`。但只支持简单键值对，**不支持** Shell 脚本语法（比如 `$PATH` 引用）。

```bash
# 需要 sudo 权限
sudo nano /etc/environment

# 添加：
JAVA_HOME=/usr/lib/jvm/java-11-openjdk

# 保存后，需要重启系统或重新登录才能生效
```

### A.2 `/etc/profile`（②）

**特点**：全局 Bash 配置，**需要** `export` 关键字。

```bash
sudo nano /etc/profile

# 添加：
export PATH=$PATH:/usr/local/custom/bin

## 块 16

**本块学习检查**：先运行 `echo $SHELL`，确认输出是 `/bin/bash` 还是其他路径，再对照下表找到你该改的配置文件。

| Shell 类型 | 配置文件 | 作用范围 |
|---|---|---|
| **Bash** | `~/.bashrc` | 非登录 Shell（日常终端） |
| **Bash** | `~/.bash_profile` | 登录 Shell（SSH 登录） |
| **Zsh** | `~/.zshrc` | 交互式非登录 Shell |
| **Zsh** | `~/.zprofile` | 登录 Shell |
| **Fish** | `~/.config/fish/config.fish` | 所有会话 |

**怎么用

## 块 17

## C. 清理 PATH 中的重复项（④）

多次修改 PATH 后，里面可能堆积了不少重复路径。虽然不影响功能，但会让排查问题变得混乱。用下面这一行命令，就能一键去重：

```bash
export PATH=$(echo $PATH | awk -F: '!a[$0]++' | paste -sd: -)
```

**拆解一下它在做什么**：`echo $PATH` 把当前 PATH 内容输出，`awk` 按冒号分割并只保留第一次出现的路径，最后 `paste` 再把它们用冒号重新拼回去。该命令仅影响当前 shell 会话，不会改动配置文件；若需永久生效，需将命令写入 `~/.bashrc` 等配置文件。

**动手试一下**：先运行 `echo $PATH` 看看原始内容，再执行上面的命令，然后再次 `echo $PATH` 对比——你会发现重复项消失了。

## D. 根据环境加载不同配置（④）

如果你在不同场景（比如开发、生产）需要不同的环境变量，可以在 `~/.bashrc` 里加一段判断逻辑：

```bash
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

这样，当 `ENVIRONMENT` 变量值为 `production` 时加载生产配置，否则加载开发配置。**注意**：`ENVIRONMENT` 需要提前设置好，例如运行 `export ENVIRONMENT=production` 或将其写入配置文件，否则走 `else` 分支。

## E. 脚本调试模式（④）

调试脚本时，想看到每一行执行前的变量值，有两种方式：

```bash
# 方式一：运行脚本时加 -x 参数
bash -x your_script.sh

# 方式二：在当前 Shell 中临时开启追踪
set -o xtrace
source ~/.bashrc

## 块 18

## 2.1 安装 Miniconda（②）

你的目标是 LoRA 微调，这意味着你很快要管理 PyTorch、Transformers 等一堆依赖。Conda 虚拟环境就是你的“项目隔离舱”——每个项目一个独立空间，版本冲突、环境搞坏都不怕。本教程推荐 **Miniconda**，只包含 conda + Python，体积约 100MB，按需安装，不浪费空间。

> **对比一下**：Anaconda 预装 1500+ 数据科学包，体积约 3GB，适合本地个人电脑想开箱即用；Miniconda 更适合服务器和普通开发环境，也是本教程推荐的选择。

**安装前你需要确认**：已完成前三节内容，掌握 Shell 命令和环境变量配置。如果对 `cd`、`export` 这些命令还不熟，先回去复习一下。

**动手验证（约 5 分钟）**：
1. 打开终端，输入 `python --version`，记下当前 Python 版本——这是你系统自带的“原始环境”。
2. 根据参考文章②的指引，下载并安装 Miniconda。
3. 安装完成后，重新打开终端，输入 `conda --version`，如果能看到版本号，说明安装成功。

> **小提示**：安装过程中如果遇到“是否将 conda 加入 PATH”的询问，建议选择“是”，这样后续使用更方便。如果选“否”，之后每次使用 conda 都需要手动激活。

**为什么这一步对你重要**：LoRA 微调通常需要特定版本的 Py

## 块 19

## 安装 Miniconda：为 LoRA 微调铺路

你正在为 LoRA 微调做准备，而 Conda 是你管理 Python 环境和依赖的“工具箱”。这一步安装 Miniconda，就是把这个工具箱搬回家。别担心，跟着下面的步骤走，很快就能搞定。

### 第一步：看看你的电脑是什么“架构”

打开终端，输入：

```bash
uname -m
```

输出通常是 `x86_64`（大多数 Intel/AMD 电脑）或 `aarch64`（部分 ARM 架构设备）。记住这个结果，下一步要用。

### 第二步：下载对应的安装脚本

根据上一步的结果，运行下面的命令自动下载：

```bash
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
elif [ "$ARCH" = "aarch64" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
else
    echo "不支持的架构: $ARCH"
    exit 1
fi
```

如果官方源下载慢，把命令里的域名换成清华镜像即可：

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-${ARCH}.sh
```

### 第三步：校验文件完整性（推荐）

下载完先别急着装，花 10 秒确认文件没被篡改：

```bash
sha256sum Miniconda3-latest-Linux-${ARCH}.sh
```

把输出的哈希值和官方公布的值对比（查询地址：`https://docs.conda.io/en/latest/miniconda.html`）。一致就继续，不一致就重新下载。

### 第四步：运行安装脚本

```bash
bash Miniconda3-latest-Linux-${ARCH}.sh
```

安装过程中会有几个交互提示，按下面操作：

1. **阅读许可协议**：按 `Enter` 滚动，或按 `q` 跳到末尾
2. **接受条款**：输入 `yes` 回车
3. **选择安装路径**：直接回车使用默认路径（`~/miniconda3`）
4. **初始化 Conda**：当询问 `Do you wish the installer to initialize Miniconda3?` 时，**务必输入 `yes`** —— 这会自动修改你的 `~/.bashrc` 文件，把 conda 加入 PATH

### 第五步：让修改生效并验证

```bash
source ~/.bashrc
conda --version
```

预期输出类似 `conda 25.x.x`。看到版本号，说明安装成功。

---

**两个你可能关心的问题：**

**为什么输入 yes 后还要 `source ~/.bashrc`？** 因为安装脚本在 `~/.bashrc` 末尾追加了一段初始化代码（你可以用

## 块 20

### 2.2 配置国内镜像源

默认官方源在国外，下载包时可能很慢甚至超时。配置国内镜像源能显著提升速度，尤其适合你即将进行的 LoRA 微调——那需要安装大量依赖包。

**操作步骤（跟着做一遍）：**

1. **添加清华镜像源**（推荐）：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```

2. **显示下载源地址**（方便排查问题）：
```bash
conda config --set show_channel_urls yes
```

3. **清理索引缓存**（使配置立即生效）：
```bash
conda clean -i
```

**两个关键提醒：**
- 旧频道 `pkgs/free` 已被官方停止维护，**不要添加**。保留默认源 `defaults` 作为备用，国内源优先即可。
- 镜像地址可能随时间失效。如果某天报错，去清华镜像站（https://mirrors.tuna.ts

## 块 21

### 环境管理核心命令：为 LoRA 微调铺路

你正在为 LoRA 微调准备环境，下面这些命令就是你的“工具箱”。我们直接上手，边看边敲。

#### 1. 查看环境：先看看你有哪些“房间”

```bash
conda env list
```

运行后，列表里带 `*` 号的就是你当前所在的环境。也可以用 `conda info --envs`，效果一样。

#### 2. 创建环境：为 LoRA 项目建一个“独立房间”

```bash
# 基础创建：指定环境名和 Python 版本
conda create -n lora_env python=3.10
```

建议环境名用 `lora_env`，Python 版本按需选。想一步到位，创建时直接装包：

```bash
conda create -n lora_env python=3.10 numpy pandas
```

另外两个实用技巧：
- **克隆环境**：想备份当前环境，用 `conda create --name prod_env --clone dev_env`，把名字换成你的即可。
- **指定路径创建**：不想放默认位置，用 `conda create --prefix ./custom_env python=3.10`，会在当前目录下建环境。

#### 3. 激活与退出：进入和离开“房间”

```bash
# 激活环境
conda activate lora_env

# 退出当前环境（回到 base）
conda deactivate
```

## 块 22

### 2.4 包管理核心命令（①）

安装包是LoRA微调准备工作的第一步。用 `conda install numpy` 装最新版，或 `conda install numpy=1.24.0` 指定版本。装 HuggingFace 的 transformers 时，推荐用 `conda install -c conda-forge transformers`，conda-forge 频道包更全。批量安装依赖用 `pip install -r requirements.txt`。

查看环境用 `conda list` 看当前环境所有包，`conda list -n llm_env` 看指定环境。搜索可用版本用 `conda search pytorch`。

更新单个包用 `conda update numpy`，卸载用 `conda remove numpy`。遇到诡异依赖问题，先 `conda remove numpy` 再 `conda install numpy=1.24.0` 重装。

**新手避坑**：别用 `conda update --all`。它会一次性升级所有包，极易引发依赖版本漂移，破坏环境可复现性。只更新需要的单个包，或新建环境重装指定版本。

**本块学习检查**：在终端依次执行 `conda list` 查看当前环境，再用 `conda search numpy` 确认你需要的版本存在，最后用

## 块 23

### 环境导出与复现：让队友一键复刻你的环境

你正在准备LoRA微调，环境配置是团队协作的第一步。这一节教你如何把当前环境“打包”成文件，让别人（或未来的你）一键重建。

**核心操作：**

1. **导出环境配置**（推荐加 `--no-builds`，跨平台更兼容）：
   ```bash
   conda env export --no-builds > environment.yml
   ```
   这会生成一个 `environment.yml` 文件——它记录了环境名、Python版本和所有包及版本。把它放进Git仓库，队友就能一键重建一模一样的环境。

2. **根据文件重建环境**：
   ```bash
   conda env create -f environment.yml

## 块 24

### 2.6 Conda 与 Pip 的协作（①）

你的目标是 LoRA 微调，很可能需要安装特定版本的 PyTorch。Conda 仓库虽全，但有些细分版本只有 pip 有。记住原则：**优先 conda，找不到再用 pip**。

```bash
# 在 conda 环境中先安装 pip
conda install pip

# 然后用 pip 安装 conda 仓库没有的包
pip install some-package-only-on-pypi
```

> **⚠️ 注意**：混用可能导致依赖解析混乱。如果出现问题，建议新建一个干净环境重新安装。

**动手验证**：在你的 conda 环境中执行 `conda install pip`，然后 `pip --version` 确认 pip 可用。这为后续安装 PyTorch 做准备。

### 2.7 常用维护命令（①+②）

```bash
# 查看 conda 系统信息
conda info

# 查看当前配置
conda config --show

# 清理所有缓存（释放磁盘空间）
conda clean --all --yes

# 设置频道优先级策略（strict 模式减少冲突）
conda config --set channel_priority strict
```

**

## 块 25

**创建你的LoRA微调环境（动手版）**

先确认Python版本，运行：

```bash
python3 --version
```

然后，在当前目录创建名为`myenv`的虚拟环境：

```bash
python3 -m venv myenv
```

这会生成一个`myenv`文件夹，内含独立的Python解释器和pip——你的LoRA微调实验将在这里隔离进行，避免污染系统环境。

**激活与退出**（Linux/macOS）：

```bash
source myenv/bin/activate
```

激活后，命令行提示符前会出现`(myenv)`，表示你已进入虚拟环境。用完记得退出：

```bash
deactivate
```

**动手验证**：创建并激活环境后，运行`pip list`，观察

## 块 26

### 3.3 包管理（④）

你的LoRA微调项目会用到很多Python库（比如transformers、torch）。用`pip`管理它们很简单：

```bash
# 安装包（默认最新版）
pip install numpy

# 安装指定版本（LoRA项目常需固定版本）
pip install numpy==1.24.0

# 查看当前环境已装了什么
pip list

# 升级某个包
pip install --upgrade numpy

# 卸载某个包
pip uninstall numpy
```

**动手练**：先`pip install numpy`，再`pip list`确认它出现了。

### 3.4 依赖导出与复现（④）

做LoRA微调时，环境一致性很重要。把当前环境所有包导出到文件，换机器时一键恢复：

```bash
# 导出所有依赖到 requirements.txt
pip freeze > requirements.txt

# 新环境里按文件安装
pip install -r requirements.txt

# 看看哪些包有新版可升
pip list --outdated
```

**动手练**：

## 块 27

**临时使用（单次生效）**

在终端里执行：

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`-i` 后面跟的是镜像地址，这次安装会从清华源下载，但只对当前命令生效。

**永久配置（推荐）**

先创建并编辑配置文件：

```bash
mkdir -p ~/.pip
nano ~/.pip/pip.conf
```

在打开的 `pip.conf` 里写入：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

保存退出后，以后所有 `pip install` 都会自动走清华镜像源。

---

**本块学习检查**

在终端

## 块 28

**本块学习检查**：执行下方全部命令，确认输出中能看到 `llm-dev` 环境及 `numpy`、`pandas` 包。

---

你的目标是 LoRA 微调，先从搭建专属环境开始。跟着做，边敲边看结果：

1. **创建环境**：`conda create -n llm-dev python=3.10 -y`  
   名字叫 `llm-dev`，Python 用 3.10，`-y` 表示自动确认。

2. **激活**：`conda activate llm-dev`  
   进入该环境，后续命令都在里面执行。

3. **装基础包**：`conda install numpy pandas -y`  
   先装数据处理常用的

## 块 29

**场景 2：用 pip + venv 创建轻量环境（面向 LoRA 微调准备）**

你的目标是 LoRA 微调，动手前先搭一个干净的环境，避免包冲突。跟着做：

```bash
# 1. 建项目目录并进入
mkdir -p ~/projects/test_project
cd ~/projects/test_project

# 2. 创建虚拟环境（名为 venv）
python3 -m venv venv

# 3. 激活环境（激活后命令提示符前会出现 (venv)）
source venv/bin/activate

# 4. 安装包（示例：requests）
pip install requests

# 5. 导出当前环境所有包到文件
pip freeze > requirements

## 块 30

好的，让我们直接进入实战验证环节。既然你的目标是 LoRA 微调，那么一个干净、可复现的 Python 环境就是一切的基石。这个场景就是帮你确认这块基石是否稳固。

**场景 3：验证 conda 对 ~/.bashrc 的修改**

刚才我们安装了 Miniconda，现在来验证它是否成功“接管”了你的终端。这直接关系到你之后能不能顺利地在终端里敲出 `conda` 命令。

打开你的终端，执行下面的命令，看看 `~/.bashrc` 文件最后 20 行的内容：

```bash
tail -n 20 ~/.bashrc
```

**你应该会看到类似这样的内容：**

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
...
# <<< conda initialize <<<
```

这段代码就是关键。它正是你之前学过的“环境变量配置”的实际应用。`conda` 通过修改这个文件，添加了初始化代码块，该代码块在 shell 启动时配置 conda 命令的可用性。这样，系统才能在任何地方找到 `conda` 命令。

**动手验证一下：**

现在，请执行 `source ~/.bashrc` 让配置立即生效，然后输入 `conda --version`。如果能看到版本号，说明 conda 已经成功安装并配置好了，你的虚拟环境管理之旅可以正式开始了。

---

**小贴士：遇到问题别慌**

如果在配置过程中遇到麻烦，可以参考这个快速排查表：

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `conda: command not found` | conda 未加入 PATH | 执行 `~/miniconda3/bin/conda init bash`，然后 `source ~/.bashrc` |
| 环境激活失败 | Shell 未初始化 conda | 运行 `conda init bash` 并重启终端 |
| 包安装缓慢或超时 | 未配置国内镜像源 | 配置清华镜像源 |
| 依赖冲突 | 多个包版本要求矛盾 | 新建干净环境，逐步安装排查 |
| pip 安装提示权限不足 | 在系统 Python 中安装 | 先激活虚拟环境，再用 pip 安装 |
| 每次打开终端自动进 (base) | conda 默认设置 | `conda config --set auto_activate_base false` |
| 环境名含中文或特殊字符 | conda 不支持 | 使用纯英文小写 + 下划线命名 |

---

**总结与

## 测评题目

### Q-KB0102-002
请写出将 `/usr/local/cuda/bin` 永久添加到 `PATH` 环境变量的完整步骤（以bash为例）。

### Q-KB0102-003
以下哪个文件是bash shell用户登录时**最先**读取的全局配置文件？

### Q-KB0103-003
请写出以下操作的完整命令序列：
1. 创建名为 `llm_train` 的conda环境（Python 3.10）
2. 激活该环境
3. 用pip安装 `torch` 包
4. 查看已安装包列表
5. 导出环境配置到 `environment.yml`

### Q-KB0103-005
在conda环境中使用 `pip install` 安装的包，可以用 `conda list` 查看到吗？反过来呢？

### Q-KB0103-001
创建一个名为 `myenv`、Python版本为3.10的conda虚拟环境，正确命令是？

### Q-KB0101-001
以下哪个命令用于**切换当前工作目录**？

### Q-KB0104-004
`nvidia-smi` 输出中 `GPU-Util` 为 0% 表示________；`Pwr:Usage/Cap` 为 `20W / 200W` 表示________。

### Q-KB0100-003
执行以下命令确认系统发行版信息：
```bash
cat /etc/os-release
```
