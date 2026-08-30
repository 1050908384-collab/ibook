# 学习包

## 块 1

### 1.1 什么是 Linux？我为什么要装它？

你已经熟悉 PyTorch 和分布式训练，但 vLLM 部署的目标环境几乎都是 Linux。Windows/macOS 是日常操作系统，而 Linux 统治着服务器和 AI 集群——你最终要在 Linux 上跑多卡推理。

本教程使用 **Ubuntu 24.04 LTS**（长期支持版），稳定且适合入门。若你已有 22.04，步骤同样适用。

**你的重点**：不是学“什么是操作系统”，而是快速获得一个能执行 Shell 命令的 Ubuntu 环境。你有 4×RTX 4090，建议直接用云服务器（带 GPU 的 Ubuntu 实例），省去本地虚拟机配置驱动的麻烦。

---

**本块学习检查**

先用最小示例回顾前置知识，确认能解释关键写法后再继续：

1. 打开终端，依次执行：
   ```bash
   cd ~
   mkdir test_dir
   ls
   cp -r test_dir test_dir_backup
   mv test_dir_backup /tmp/
   ```
2. 解释每条命令的作用（`cd` 切换目录、`ls` 列出文件、`mkdir` 创建目录、`cp -r` 递归复制、`mv` 移动）。
3. 若 `ls` 能看到 `test_dir`，且 `/tmp/test_dir_backup` 存在，说明终端操作就绪。

确认无误后，再进入下一节。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

你已有PyTorch和分布式训练基础，且目标是用4×RTX 4090跑vLLM部署。环境选择直接影响后续能否顺利执行分布式推理命令，所以先定方案。

**三个选项速览**：

- **虚拟机**：在你电脑里模拟一台新电脑，隔离最彻底，但吃内存（建议至少4GB），配置低会卡。适合内存≥8GB、想体验完整Linux桌面的情况。
- **WSL2**：Windows自带的Linux子系统，像装App一样轻量、启动快，与Windows无缝互通。仅限Win10/11，部分高级功能受限。
-

## 块 3

### 1.3 三个方案最终都要达到什么效果？

无论你选虚拟机、云服务器还是 WSL2，三个方案最终都能得到一台运行 Ubuntu 24.04 的环境，但操作方式不同：虚拟机/云服务器通过SSH远程连接，WSL2与Windows直接集成。

- **虚拟机 / 云服务器**：通过 SSH 从 Windows 远程连过去操作，就像远程控制另一台电脑。
- **WSL2**：与 Windows 直接互通，用 VS Code 本地开发即可，无需手动配 SSH。

> **SSH 通俗理解**：远程桌面的命令行版。你不需要跑到服务器跟前，在自己电脑上敲命令就能控制它。

---

## 二、方案一：虚拟机（VirtualBox + Ubuntu 24.04）

**开始前检查

## 块 4

**VirtualBox 是什么？** 它是一款开源虚拟机软件，能在你的 Windows 里造出一台“假电脑”，用来安装任何操作系统——包括后续要用的 Ubuntu。

**下载与安装步骤：**
1. 浏览器打开：`https://www.virtualbox.org/wiki/Downloads`
2. 找到 **"VirtualBox 7.x.x platform packages"** 下的 **Windows hosts** 版本，点击下载。
3. 双击安装包，一路“下一步”保持默认，点“安装”等进度条跑完，最后“完成”。

**常见坑预警**：弹出“是否允许更改设备”选“是”；杀毒软件拦截就选“允许”或“信任”。

> 你的目标是多卡 vLLM 部署，虚拟机只是环境准备的第一步。装好后能打开 VirtualBox 主界面，即表示安装成功。

**本块学习检查**：完成本块后，用自己的话

## 块 5

**下载 Ubuntu 24.04 镜像文件（.iso）**

镜像文件就是“操作系统安装光盘的电子版”，VirtualBox 靠它来装系统。你的目标是 vLLM 部署，这一步只是搭好环境地基。

操作步骤：
1. 浏览器打开：https://releases.ubuntu.com/24.04/
2. 找文件名类似 `ubuntu-24.04.x-desktop-amd64.iso`（x 是小版本号），点击下载
3. 文件约 5GB，下载需几分钟到十几分钟（看网速）
4. 下载完**别双击打开**，放下载文件夹即可

**常见坑**：确认是 `desktop-amd64.iso`，不是 `server` 版——server 版无图形

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 6

## 2.3 在 VirtualBox 里创建虚拟机

你的目标是vLLM部署，但先别急着碰GPU——多卡环境需要一个稳定的Linux底座。这一步就是搭好这个底座。

**操作步骤：**

1. 打开 **Oracle VM VirtualBox**，点 **"新建"**
2. 填写：名称 `Ubuntu-24.04`，文件夹选 **D盘或E盘**（别放C盘），类型 `Linux`，版本 `Ubuntu (64-bit)`，下一步
3. 内存拖到 **4096 MB**，下一步
4. 虚拟硬盘选 **"现在创建虚拟硬盘"**，大小 **60 GB**，类型 **VDI**，存储 **动态分配**，完成
5. 选中虚拟机 → **"设置"** → **"系统"→"处理器"**：处理器数量设为 `2`（或根据主机CPU情况调整）
6. **"存储"** → 点光盘图标 → **"选择虚拟盘"** → 找到 `ubuntu-24.04.x-desktop-amd64.iso

## 块 7

你的目标是多卡环境下用 vLLM 做分布式部署，所以虚拟机这一步只是“练兵场”——用 60G 虚拟磁盘模拟真实服务器，把安装流程跑通，后面在 4×RTX 4090 上装系统时就不会慌。

**操作步骤（跟着点就行）**

1. 在 VirtualBox 窗口选中你的虚拟机，点 **“启动”**。
2. 等几十秒进入安装界面，按顺序操作：
   - 语言选 **“中文(简体)”**，点“安装 Ubuntu”。
   - 键盘布局默认，点“继续”。
   - 更新和软件：选 **“正常安装”**，勾选“安装时下载更新”和“安装第三方软件”，点“继续”。
   - 安装类型：选 **“清除整个磁盘并安装 Ubuntu”**——这里清的是虚拟机的 60G 盘，碰不到你真实硬盘，放心点“现在安装”，弹窗再点“继续”。
   - 时区：输入“Shanghai”或

## 块 8

### 2.5 安装 SSH 远程连接工具（①）

**为什么这一步对你的 vLLM 部署至关重要？**

你的目标是 4×RTX 4090 上做分布式 vLLM 部署。这意味着你最终要在多台机器上跑推理服务，而 SSH 就是你从 Windows 远程操控这些 Linux 机器的“遥控器”。现在在虚拟机里练熟它，后面管理多卡集群时才能游刃有余。

**动手操作（跟着做一遍）**

1. **打开 Ubuntu 终端**：在桌面空白处点右键 → 选“在终端中打开”；如果右键没反应，按 `Ctrl + Alt + T`。

2. **安装 SSH 服务**：依次输入以下命令（每行输完按回车）：

```bash
# 第1步：更新软件列表（类似刷新应用商店货架）
sudo apt update

# 第2步：安装 SSH 服务端和网络工具
sudo apt install -y openssh-server net-tools

# 第3步：设置 SSH 开机自启
sudo systemctl enable ssh

# 第4步：立刻启动 SSH
sudo systemctl start ssh
```

> **sudo 是什么？** 相当于“以管理员身份执行”。Linux 普通用户不能随便装软件，加 sudo 就是临时提权。输入后会让你输密码，**屏幕不显示任何字符**（连 `***` 都没有），这是正常的安全设计，盲打完按回车即可。

3. **查看虚拟机 IP**：

```bash
ifconfig
```

4. 输出内容较多，找到类似 `inet 192.168.xxx.xxx` 的那一行，那一串数字就是 IP 地址。**记下来！** 例如 `192.168.126.128`。

5. **从 Windows 验证连接**：按 `Win + R`，输入 `cmd` 回车，在命令提示符里输入：

```bash
ssh 你的用户名@192.168.xxx.xxx
```

6. 提示输入密码时，输入 Ubuntu 的密码并回车。看到 `Welcome to Ubuntu` 之类的欢迎信息，说明连接成功。

**常见坑预警**：
- 提示 `connection refused`：回到 Ubuntu 终端，输入 `sudo systemctl status ssh` 检查 SSH 是否运行，绿色 `active (running)` 才对。
- 连不上可能是 VirtualBox 网络

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在4×RTX 4090多卡环境下，你最终要部署vLLM。但先别急——本块解决的是“虚拟机与Windows的协作问题”。

**为什么这步重要？** 你后续要写配置、复制模型路径、粘贴命令，如果虚拟机不能和Windows互通剪贴板，每次都得手敲，效率极低。增强功能就是打通这条“复制粘贴”通道。

**操作步骤**（若新版VirtualBox未自动生效）：

```bash
# 安装 VirtualBox 增强工具包
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11

# 重启虚拟机
sudo reboot
```

重启后，在Windows里`Ctrl+C`复制文字，到Ubuntu里`Ctrl+V`即可粘贴。

**完成标志**

## 块 10

### 3.1 确认虚拟化已开启（①）

你的目标是vLLM部署，但先别急——WSL2是Windows下运行Linux的“套娃”方案，虚拟化没开，后面一切免谈。这一步就是给环境“通电”。

**检查方法（30秒）**：
1. 按 `Ctrl + Shift + Esc` 打开**任务管理器**
2. 点顶部**“性能”**标签页
3. 右下角找**“虚拟化”**，看是“已启用”还是“已禁用”

**如果“已启用”** → 直接跳到下一步。

**如果“已禁用”** → 必须进BIOS开启：
- 重启电脑，开机时狂按 `F2` 或 `Del`（联想F2，惠普F10，戴尔F2）
- 进入BIOS后，用方向键找**“Configuration”**或**“Advanced”**菜单
- 找到**“Intel Virtualization Technology”**或**“AMD SVM”**，改成**

## 块 11

你的目标是在4×RTX 4090 96GB的多卡环境上部署vLLM，这意味着你最终需要在Linux环境下运行命令。WSL2正是Windows下最轻量的Linux运行方案——它提供完整Linux内核，且安装只需几分钟。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

按以下步骤操作：

1. 按 `Win` 键，输入 `powershell`，在搜索结果里**右键** "Windows PowerShell" → **"以管理员身份运行"**，弹出提示时点"是"。

2. 在PowerShell窗口中，**逐行**输入以下两条命令（复制一整行，右键粘贴，回车）：

```bash
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

每条命令执行后应显示"操作成功完成"。

3. **重启电脑**。

4. 重启后，再次以管理员身份打开PowerShell，输入：

```bash
wsl --set-default-version 2
```

若提示需要更新内核，请访问 https://aka.ms/wsl2kernel 下载并安装最新的 WSL2

## 块 12

### 3.3 安装 Ubuntu 24.04（①）

你的目标是vLLM部署，而WSL2是Windows上跑Linux环境的最快路径。装好Ubuntu后，你才能执行后续的Shell命令和模型推理脚本。下面直接进入安装。

**方法一（推荐，简单）**：
1. 按 `Win` 键，输入 `Microsoft Store`，打开微软应用商店。
2. 搜索框输入 `Ubuntu 24.04`。
3. 找到 **"Ubuntu 24.04.x LTS"**（发布者是 Canonical Group Limited），点"安装"。
4. 下载约几百MB（具体大小视网络和系统而定），装完后点"打开"。

**方法二（命令行）**：
```bash
# 一条命令自动下载并安装 Ubuntu 24.04
wsl --install -d Ubuntu-24.04
```

5. 首次启动会显示 "Installing, this may take a few minutes..."，等2~5分钟。
6. 安装完会提示设置用户名和密码：
   - 输入用户名（全英文小写，如 `

## 块 13

### 本块学习检查

完成本块后，请用自己的话说明：为什么在 WSL2 中通常不需要像虚拟机那样配置 SSH，以及这对你后续在 4×RTX 4090 上部署 vLLM 的开发流程意味着什么？遇到无法解释的术语（如 `apt`、`Remote-WSL`）时记录下来再提问。

---

在 Ubuntu 终端（绿色提示符窗口）中，逐行执行以下命令，让系统保持最新：

```bash
# 刷新软件包列表（相当于更新“应用商店的货架”）
sudo apt update

# 升级所有已安装软件到最新版（-y 表示自动确认所有询问）
sudo apt upgrade -y
```

> **关键点**：WSL2 与 Windows 紧密集成，通常无需额外配置 SSH 即可进行本地开发，但若需远程访问或使用 SSH 协议，仍可安装和配置 SSH 服务。你直接用 VS Code 的 Remote-WSL 插件即可开发，省去了虚拟机那套额外配置。

**验证环境**：

1. 检查系统与用户：
```bash
lsb_re

## 块 14

### 4.1 购买服务器（①）

你的目标是vLLM部署，但注意：**最低配2核2G仅够基础练习，不适用于大模型训练或vLLM部署**。你的4×RTX 4090 96GB环境才是主力，云服务器只作备用或调试。

以阿里云为例（腾讯云、华为云流程类似）：
1. 打开 https://www.aliyun.com/ ，右上角用支付宝/淘宝账号注册/登录。
2. 顶部搜索"轻量应用服务器"进入；学生先搜"学生优惠"完成认证，价格更低。
3. 点"立即购买"：
   - **地域**：选最近的（如华东1（杭州）、华南1（深圳

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的4卡4090集群需要vLLM做分布式推理，但部署前先要筑牢安全地基。云服务器默认用`root`（超级管理员）登录，权限过高，误操作可能删系统文件。所以第一步是创建普通用户日常使用。

通过云厂商控制台的“远程连接”或VNC（网页版终端）登录后，执行：

```bash
adduser yourname   # 创建用户，yourname用英文小写；设密码两次，其余信息回车跳过
usermod -aG sudo yourname   # 赋予sudo（管理员资格）
```

密码

## 块 16

## 4.3 配置 SSH 密钥登录（安全第二步）

**为什么先做这一步？** 你要在4张4090上跑vLLM分布式部署，服务器会长期开放端口。密码登录等于把门虚掩着，暴力破解随时可能闯进来。密钥登录相当于换了把只有你有的钥匙，安全且省去每次输密码的麻烦。

**操作步骤（Windows → 云服务器）：**

1. `Win + R` 输入 `cmd` 打开命令提示符
2. 生成密钥对：
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```
3. 提示保存路径 → 直接回车（默认路径）
4. 提示输入 passphrase → 设置一个强密码短语并牢记；若选择不设，务必妥善保管私钥文件并限制其权限
5. 看到 randomart image 即生成成功

6. 通过云厂商 VNC/远程连接登录服务器（通常以 root 或云厂商默认用户登录），创建授权文件：
```bash
sudo -i
mkdir -p ~/.ssh
nano ~/.ssh/authorized_keys
```

7. 回到 Windows 查看公钥：
```bash
type %USERPROFILE%\.ssh\id_rsa.pub
```
全选复制输出内容，粘贴到服务器 nano 编辑器里，`Ctrl

## 块 17

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

**目标**：为你的4×RTX 4090多卡环境，安全开放SSH端口。

**操作**：
1. 云厂商控制台 → **轻量应用服务器** → **防火墙**标签页
2. 添加规则：端口`22`，协议`TCP`
3. 来源：填你本机公网IP（浏览器搜"IP"查看），格式`123.45.67.89/32`；IP不固定可暂填`0.0.0.0/0`（允许所有，安全性低，

## 块 18

## 从 Windows 连接你的云服务器

你已经有了云服务器，现在从 Windows 连上去操作它。这是你后续部署 vLLM 的第一步——所有命令都要在这台远程 Ubuntu 上执行。

**操作步骤：**

1. 在阿里云控制台找到 **公网 IP**（形如 `47.96.xxx.xxx`），复制它
2. 按 `Win + R`，输入 `cmd` 打开命令提示符
3. 执行连接命令：

```bash
ssh yourname@47.96.xxx.xxx
```

4. 首次连接会询问 `Are you sure you want to continue connecting (yes/no)?`，输入 `yes` 回车
5. 若密钥配置正确，将直接登录到服务器，出现欢迎

## 块 19

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

在命令提示符执行 `ssh 用户名@你的IP地址`，免密即成功。若提示 `ssh-copy-id 不是内部或外部命令`，装 Git for Windows 用 Git Bash，或手动复制（见 4.3 节）。

## 块 20

## 5.2 WSL2：无需 SSH，直接集成

你用的是 WSL2，所以**不需要配置 SSH**。VS Code 通过 Remote-WSL 插件直接与 WSL2 通信，体验更流畅——这意味着你可以跳过 SSH 密钥、端口转发等繁琐步骤，直接在 Windows 的 VS Code 里操作 Linux 环境，对后续在 WSL2 里跑 vLLM 分布式部署非常顺手。

## 5.3 安装 VS Code 及远程开发插件

**VS Code**（Visual Studio Code）是微软出的免费代码编辑器，轻量、插件多、跨平台。你之后要写 Python 脚本、调试 vLLM 推理代码，都靠它。

### 第一步：安装 VS Code

1. 浏览器打开：https://code.visualstudio.com/
2. 点 **"Download for Windows"** 按钮
3. 下载完成后双击安装包
4. 安装过程：
   - 同意协议 → 下一步
   - 安装路径默认 → 下一步
   - **重点！** 这页勾选：
     - ☑ **"将 Code 添加到"PATH"中"**（重要！这样可以在命令行直接输入 `code` 打开 VS Code）
     - ☑ **"将 Code 注册为受支持的编辑器"**
     - ☑ **"添加到文件资源管理器上下文菜单"**
   - 一路"下一步" → "安装" → 完成后勾上"启动 VS Code" → "完成"

### 第二步：安装必备插件

**插件（Extension）** 是 VS Code 的"外挂功能包"。VS Code 本身只是编辑器壳子，装了插件才能高亮代码、自动补全、调试。

1. 打开 VS Code
2. 点左侧竖排图标栏里**第五个图标**（四个小方块），或按 `Ctrl + Shift + X`
3. 在**扩展商店**搜索框里依次搜索并安装：

| 搜索关键词 | 插件全名 | 发布者 | 作用 |
|---|---|---|---|
| Python | Python | Microsoft | Python 代码高亮、智能提示、

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 21

## 5.4 连接你的 Linux 环境

你的目标是 vLLM 分布式部署，4×RTX 4090 意味着你需要把 VS Code 连到那台多卡 Linux 机器上。下面按你的场景走一遍。

**虚拟机或云服务器：**

1. 打开 VS Code，看左下角绿色图标（`><` 形状）
2. 点击它 → 顶部菜单选 **"Connect to Host…"**
3. 输入 `用户名@IP地址`（如 `root@192.168.1.100`）→ 回车
4. 选平台 **"Linux"**
5. 配过密钥直接连上；没配则输密码
6. 左下角变成 `SSH: IP地址` 即成功
7. 点 **"File"** → **"Open Folder…"**，选服务器目录（如 `/home/你的用户名`），点 OK

连上后，你就能像编辑本地文件一样改远程代码——这对后续调 vLLM 配置、改启动脚本是基础操作。

**WSL2 用户更快的路径：**

在 Ubuntu 终端 `cd` 进项目目录，直接输 `code .`，VS Code 自动连接 WSL 并打开。

---

## 六、终端入门

终端就是那个黑框框。若提示 `command not found`（如 `vim`、`nano` 未装），先执行：

```bash
sudo apt install -y vim nano
```

---

**本块学习检查**：用自己的话说明——为什么"连接 Linux 环境 + 熟悉终端"是你做 vLLM 分布式部署的第一步？遇到无法解释的术语（如 SSH、WSL）记录下来再提问。

## 块 22

## 6.1 文件与目录操作：最基础也最常用

> 你的目标是用 vLLM 在多卡环境部署模型。部署前，模型权重、日志、配置文件都散落在服务器目录里——这一节就是让你能高效地“找到、移动、清理”这些文件。

### 核心命令速览

**定位与导航**

- `pwd`：显示当前所在目录的绝对路径。部署时先确认自己在哪，避免在错误目录执行命令。
- `ls`：列出目录内容。`ls -l`（或 `ll`）查看详细权限和大小，`ls -a` 显示隐藏文件（如 `.cache`、`.env`）。
- `cd`：切换目录。`cd ..` 上级、`cd /` 根目录、`cd ~` 用户主目录、`cd /path/to/dir` 指定路径。

**创建与查看**

- `mkdir`：创建目录。`mkdir -p test/aaa/bbb` 可递归创建多级目录。
- `touch`：创建空文件。例如 `touch config.yaml`。
- `cat`：查看小文件内容，`cat -n` 显示行号。
- `more/less`：分页查看大文件（如日志）。`less app.log`，空格翻页、`q` 退出。

**复制、移动与删除**

- `cp`：复制。文件直接复制，目录需加 `-r` 递归：`cp -r model_dir/ /data/backup/`。
- `mv`：移动或重命名。`mv old_name new_name` 改名，`mv file.txt target_dir/` 移动。
- `rm`：删除。**

## 块 23

## 6.2 权限管理：Linux 系统的安全核心

你已经熟悉 PyTorch 分布式训练，但多卡环境（4×RTX 4090）下，**权限错误**是 vLLM 部署最常见的拦路虎。比如模型权重文件不可读、启动脚本无执行权限，都会直接导致推理服务起不来。

Linux 中每个文件/目录都有三类权限归属：**所有者、所属组、其他用户**。权限值用数字表示：读=4、写=2、执行=1，可叠加。

**核心命令一：`chmod`（改权限）**

数字法（推荐）：`chmod 744 file1.txt` 表示所有者可读+写+执行（7=4+2+1），所属组只读（4），其他用户只读（4）。部署 vLLM 时，模型文件通常设为 `644`（所有者读写，其他人只读），脚本设为 `755`（所有者全权，其他人可读+执行）。

符号法：`chmod g+w file1.txt` 给所属组加写权限

## 块 24

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在4×RTX 4090 96GB多卡环境部署vLLM前，先用系统查询命令确认机器状态。`uname -a`查看内核与架构，确保驱动兼容；`free -h`确认96GB显存与内存充足；`df -h`检查磁盘空间，避免模型下载中断；`top`或`ps -ef | grep python`监控进程，排查残留任务占用资源。这些命令让你在部署前摸清环境底细，避免资源冲突。

**操作验证**：依次执行`uname -a`、`free -h`、`df -h`，再用`top`观察CPU/内存占用，按`q`退出；最后用`ps -ef | grep python`确认无异常Python进程。

## 块 25

你的目标是vLLM部署，但先别急着碰模型。在4×RTX 4090 96GB的多卡环境里，你迟早要改配置文件（比如`vllm serve`的启动脚本或模型路径）。本块教你两种快速改文件的方法，够用且不啰嗦。

**1. 用`vim`改文件（适合手动微调）**
- 打开文件：`vim file1.txt`
- 按`i`进入编辑模式（左下角出现`--INSERT--`），此时可输入或修改内容
- 改完按`Esc`退出编辑，输入`:wq`回车保存退出；若想放弃修改，输入`:q!`回车
- 查找内容：按`Esc`后输入`/关键词`回车，用`n`跳下一个、`N`跳上一个

## 块 26

## 6.5 网络操作：基础网络查询

多卡分布式部署前，先确认节点间网络连通与网卡状态，这是 vLLM 多机通信的基础。

**ping**：测试网络连通性。语法 `ping [选项] 目标IP/域名`。示例：`ping www.baidu.com`，按 `Ctrl+C` 停止。若目标可达，会持续返回响应时间；若超时，说明网络不通。

**ifconfig**：查看网卡信息（IP、MAC 地址等）。语法 `ifconfig` 或 `ip addr`（部分系统默认 `ip` 命令）。示例：直接运行 `ifconfig`，输出中 `eth0`/`wlan0` 的 `inet` 字段即 IP

## 块 27

## 6.6 命令实战场景演练（面向 vLLM 部署）

你已掌握 PyTorch 与分布式训练，但 vLLM 部署依赖 Linux 终端操作。下面两个场景直接对应你即将面对的任务：**整理模型文件**与**排查 GPU 进程**。

### 场景 1：创建并整理文件（对应模型权重目录管理）

```bash
# 1. 进入主目录
cd ~

# 2. 创建 work 及子文件夹（-p 可一次性创建多级目录）
mkdir -p work/doc work/code

# 3. 创建并写入文件（> 为重定向符）
echo "Linux命令学习笔记" > note.txt

# 4. 移动文件（mv 也可用于重命名）
mv note.txt work/doc/

# 5. 查看结果

## 块 28

## 6.7 新手避坑指南（针对多卡 vLLM 部署）

你的目标是多卡 vLLM 部署，终端操作失误会直接浪费调试时间。以下四个坑最常绊倒新手：

**1. 大小写敏感**：Linux 命令严格区分大小写。`ls` 正确，`LS` 直接报错。部署脚本里写错一个大小写，可能让整个服务起不来。

**2. 路径写错**：切换目录或加载模型权重时，路径错会提示 "No such file or directory"。先用 `ls` 确认路径存在再操作，别凭记忆敲路径。

**3. `rm -rf` 极度危险**：这是"删库跑路"级命令。清理旧权重或日志前，务必 `ls` 确认目标正确，防止误删根目录 `/`——一旦发生，系统直接报废。

**4. 权限不足**：访问系统文件或他人文件时提示 "Permission denied"。在命令前加 `sudo` 获取临时管理员权限（需输当前用户密码

## 块 29

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的4×RTX 4090环境已就绪，但vLLM部署前必须先确认Python工具链可用。Ubuntu 24.04默认带Python 3.12，直接验证：

```bash
python3 --version   # 预期输出：Python 3.12.x
pip3 --version      # 若提示未找到，执行下方安装
```

若`pip3`缺失，运行：

```bash
sudo apt update
sudo apt install -y python3-pip
```

**关键理解**：pip是Python的包管理器（相当于

## 块 30

## 7.2 （可选但推荐）安装 Miniconda（①）

你的目标是 vLLM 部署，后续会涉及多卡并行与虚拟环境隔离。系统自带 Python 虽够用，但 Miniconda 能让你更干净地管理 PyTorch、vLLM 等依赖，避免版本冲突。这里先做最简安装，详细虚拟环境配置后续深入。

**安装步骤：**

1. **下载安装脚本**（若提示 `wget` 未找到，先执行 `sudo apt install wget`）：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. **运行脚本**：
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

3. **交互选项**：按 `Enter` 阅读许可 → 输入 `yes` 同意 → 确认路径（默认回车）→ 提示更新 shell profile 时输入 `yes`。

4. **激活配置**：关闭重开终端，或执行 `source ~/.bashrc`。

5. **验证**：
```bash
conda --version
# 预期输出类似：conda 24.9.2

## 块 31

### 7.3 运行你的第一个 Python 程序（①）

你的目标是 vLLM 部署，而 vLLM 的推理引擎运行在 Python 之上。所以，让 Python 在你的 Linux 环境里跑起来，是验证环境可用性的第一步——这一步不通过，后面所有分布式推理都无从谈起。下面直接操作。

**1. 创建并编辑文件**

在终端执行：

```bash
cd ~
nano hello.py
```

`cd ~` 回到主目录，`nano hello.py` 用 nano 编辑器打开（或创建）一个叫 `hello.py` 的文件。

**2. 输入代码**

在 nano 里输入：

```python
print("Hello, Python on Ubuntu 24.04!")
```

**3. 保存并退出**

按 `Ctrl + O` 保存，回车确认文件名；再按 `Ctrl + X` 退出 nano。

**4. 运行程序**

```bash
python3 hello.py
```

屏幕输出 `Hello, Python on Ubuntu 24.04!`，说明 Python 环境就绪，且你已经用上了 `cd`、`nano`、`python3` 这几个终端命令。

---

#### 常见问题速查（与本块直接相关）

| 症状 | 可能原因 | 解决办法 |
|---|---|---|
| `python3` 提示 `command not found` | Python 未安装或 PATH 未配置 | 回到上一节确认 Python 3.12 与 pip 已就绪 |
| 保存时文件名输错 | 手误 | 重新 `nano hello.py` 再存一次 |
| 运行后无输出 | 代码没保存成功 | 重新执行 `nano hello.py`，确认 `print` 那行代码完整，再保存退出 |

---

#### 本块学习检查

完成本块后，用自己的话说明：**本块内容（运行第一个 Python 程序）与你当前学习目标（vLLM 部署）的关系是什么？** 如果遇到无法解释的术语（比如 `nano` 或 `python3` 的具体机制），记录下来，之后提问。

## 块 32

## 二、你在哪？pwd 与路径的概念

### 2.1 pwd：打印当前工作目录

在分布式训练中，你经常需要确认模型权重、日志文件到底保存在哪台机器的哪个目录下。`pwd` 就是干这个的——它是 **P**rint **W**orking **D**irectory 的缩写，作用是告诉你“我现在在哪个文件夹里”。

```bash
pwd
```

示例输出：

```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。在操作文件之前，养成先敲 `pwd` 的习惯，能避免很多“文件找不到了”的困惑。

**和你的 vLLM 部署有什么关系？** 当你用 4×RTX 4090 启动多卡推理时，每个 GPU 进程的工作目录可能不同。如果脚本里用了相对路径加载模型，某个进程找不到文件就会报错。先 `pwd` 确认位置，再用绝对路径，能省下大量

## 块 33

### 2.2 绝对路径 vs 相对路径

你在多卡机上部署 vLLM 时，经常要指定模型路径或日志目录。搞不清这两种路径，命令就会“找不到文件”。

| 路径类型 | 定义 | 示例 |
|---|---|---|
| **绝对路径** | 从根目录 `/` 开始，不依赖当前位置 | `/home/zhangsan/Documents` |
| **相对路径** | 相对于你当前所在的位置 | `./file.txt`、`../backup` |

相对路径中两个重要符号：

- `./`：表示**当前目录**（可省略）
- `../`：表示**上一级目录**
- `~`：表示**当前用户的家目录**（如 `/home/zhangsan`）

> **举例**：假设你在 `/home/zhangsan/Documents` 目录下：
> - `cd /home/zhangsan` 是绝对路径，去哪都管用
> - `cd ..` 是相对路径，

## 块 34

## 3.1 ls：列出目录内容（基础用法）

你的目标是 vLLM 部署，而 vLLM 的模型文件、配置和日志都存放在目录中。`ls` 是你查看这些文件的第一工具，尤其在多卡环境下，你需要快速确认模型权重、日志文件是否就位。

**基础命令**：

```bash
# 列出当前目录下的文件/文件夹（不包括隐藏文件）
ls

# 列出指定路径下的内容
ls /home/zhangsan

# 列出当前目录下所有文件（包括隐藏文件，以 . 开头）
ls -a

# 以详细列表形式显示（权限、所有者、大小、修改时间）
ls -l

# 组合使用：详细列出所有文件，并以人类可读格式显示大小
ls -ahl
```

**关于隐藏文件**：Linux 中以 `.` 开头的文件/文件夹默认是隐藏的，比如 `.bashrc`、`.ssh`。这些文件往往很重要，后续配置环境变量时会频繁遇到。

## 3.2 ls -l 的输出解读

执行 `ls -l` 后

## 块 35

### ls 常用选项：直击 vLLM 部署中的文件管理

在管理 4×RTX 4090 的模型权重、日志和数据集时，`ls` 的这些选项能帮你快速定位关键文件。

**核心组合：`ls -lh`**  
以详细列表显示，文件大小自动转为 KB/MB/GB。注意：`-h` 必须搭配 `-l` 才有效，单独使用无效果。部署时用 `ls -lh` 检查 `.safetensors` 权重文件是否完整下载。

**实用排序与筛选**  
- `ls -t`：按修改时间排序，最新在前——快速找到刚生成的日志。  
- `ls -S`：按大小从

## 块 36

## 4.1 cd 基础用法：切换目录

你在多卡环境下做 vLLM 分布式部署，经常需要在模型权重目录、日志目录、配置文件目录之间来回穿梭。`cd` 就是你移动的“传送门”。

```bash
# 进入指定目录（绝对路径，从根目录 / 开始写起）
cd /home/zhangsan/Documents

# 进入当前目录下的子文件夹（相对路径，./ 可省略）
cd ./Downloads
cd Downloads

# 返回上一级目录
cd ..

# 返回上两级目录
cd ../..

# 回到家目录
cd ~
cd        # 直接 cd 不加参数也是回家目录

# 返回刚才所在的目录（快速来回切换）
cd -
```

**实操建议**：部署 vLLM 时，你经常要在 `/models`（权重）和 `/logs`（日志）之间切换，`cd -` 能帮你省下大量重复输入路径的时间。

## 4.2 路径切换示例

假设当前在 `/usr/local` 下，想切换到 `/home/zhangsan`：

```bash
# 方法1：绝对路径（去哪都管用，推荐）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

> **建议**：新手优先使用绝对路径，不容易迷路；熟悉后多用相对路径，效率更高。

## 五、mkdir：创建目录

`mkdir` = **M**a**k**e **Dir**ectory，用来创建文件夹。部署 vLLM 时，你需要为不同实验创建独立目录来隔离配置和日志。

```bash
# 在当前目录下创建一个文件夹
mkdir myproject

# 在指定路径下创建文件夹
mkdir /home/zhangsan/myproject

# 递归创建多级目录（父目录不存在时自动创建）
mkdir -p myproject/src/models

# 一次性创建多个同级文件夹
mkdir dir1 dir2 dir3

# 使用花括号批量创建同级文件夹
mkdir {dir1,dir2,dir3}

# 批量创建 100 个文件夹（dir1 ~ dir100）
mkdir dir{1..100}
```

> **注意**：`mkdir` 默认只能创建单级目录，若父目录不存在会报错；使用 `-p` 参数可递归创建多级目录

## 块 37

## 7.1 复制文件：为 vLLM 部署做准备

在分布式 vLLM 部署中，你经常需要把模型配置文件、启动脚本复制到多台机器的相同路径下。`cp` 就是干这个的。

```bash
# 把 hello.py 复制到 backup 目录（文件名不变）
cp hello.py ./backup/

# 复制并改名，比如把不同版本的配置区分开
cp hello.py ./backup/hello_backup.py

# 一次复制多个文件
cp file1.txt file2.txt ./backup/

# -p：保留修改时间等属性，排查问题时很有用
cp -p hello.py ./backup/

# -i：目标已存在时先询问，防止覆盖重要配置
cp -i hello.py ./backup/
```

## 7.2 复制目录：整个项目迁移

复制整个项目文件夹时，**必须加 `-r`**（递归），否则系统直接拒绝。

```bash
cp -r myproject ./backup/
```

注意行为差异：`backup` 不存在时会新建并把内容放进去；`backup` 已存在时，会生成 `backup/myproject/` 嵌套目录。部署时路径别搞错。

## 7.3 复制软链接

```bash

## 块 38

## 8.1 移动文件/文件夹（①）

```bash
# 把文件移动到另一个文件夹
mv hello.py ./backup/

# 把文件夹移动到另一个位置
mv myproject ./archive/

# 使用通配符移动多个文件（把所有以 data_ 开头的文件移到 data 文件夹）
mv data_* ./data/
```

## 8.2 重命名（①）

```bash
# 把 a.txt 重命名为 b.txt（本质是在同一目录下移动）
mv a.txt b.txt

# 移动的同时改名
mv hello.py ./backup/hello_v2.py
```

## 8.3 安全选项（①）

```bash
# 覆盖前询问（推荐日常使用）
mv -i old.txt new.txt

# 强制覆盖，不提示（慎用）
mv -f old.txt new.txt
```

---

## 九、rm：删除文件与目录（①）

`rm` = **R**e**m**ove，用来删除文件或文件夹。**此命令不可逆，请谨慎使用！**（①）

```bash
# 删除单个文件
rm hello.py

# 删除多个文件
rm file1.txt file2.txt

# 删除文件名以 - 开头的文件（必须用 -- 告诉 rm 后面是文件名而非选项）
rm -- -file.txt

# 删除文件夹（必须加 -r，递归删除其中的所有内容）
rm -r myproject/

# 只删除空文件夹
rm -d empty_dir/

# 强制删除，不提示确认
rm -f hello.py

# 强制递归删除文件夹及其中所有内容（⚠️ 极度危险，务必确认路径正确！）
rm -rf my

## 块 39

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是vLLM部署，但先别急着碰推理。分布式部署前，项目结构必须清晰。本块用最基础的Shell命令搭建`llm-learning`目录，这正是你后续存放数据、模型和训练脚本的骨架。

**动手做**（在4×RTX 4090环境上）：

1. `cd ~` 回到家目录
2. `mkdir llm-learning` 建根目录
3. `cd llm-learning` 进入
4. `mkdir data models scripts` 一次建三个子目录
5. `touch scripts/train.py`

## 块 40

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

在4×RTX 4090 96GB多卡环境中准备vLLM部署，你需要先整理好实验文件。这里练习用`cp`命令把`~/Downloads/`下的三个CSV数据集复制到`~/llm-learning/data/`。

**操作步骤**：

1. **确认源文件存在**：`ls ~/Downloads/*.csv`，用通配符`*.csv`列出所有CSV文件。
2. **复制文件**：`cp ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/val

## 块 41

## 场景 3：清理临时文件（结合 vLLM 部署实战）

在 vLLM 部署过程中，你经常需要清理实验产生的临时文件——比如下载失败的模型缓存、测试脚本的中间输出。这个场景帮你养成安全清理的习惯。

**需求**：你在 `~/llm-learning/` 下误建了一个名为 `temp` 的空文件夹和一个 `old.txt` 文件，请将它们删除。

```bash
# 1. 先确认要删的内容（好习惯！）
ls ~/llm-learning/temp ~/llm-learning/old.txt

# 2. 删除文件
rm ~/llm-learning/old.txt

# 3. 删除空文件夹
rmdir ~/llm-learning/temp

# 4. 验证
ls ~/llm-learning/
```

注意第 3 步：`rmdir` 只能删除**空**文件夹。如果 `temp` 里有文件，会报错——这时需要用 `rm -r`（递归删除），但务必谨慎。

---

## 十一、新手避坑指南（重点看与 vLLM 部署相关的）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。部署 vLLM 时复制模型权重目录，漏掉 `-r` 会直接报错。

**2. 路径写错**：切换目录或复制文件时，路径写错会提示 "No such file or directory"，可先用 `pwd` 和 `ls` 确认当前位置和目录内容。多卡环境下，你经常要在多个路径间切换（模型权重、日志、虚拟环境），这个习惯尤其重要。

**3. 路径中包含空格必须用引号包裹**：如果文件夹或文件名包含空格，命令行会把它当成多个参数。务必用双引号包裹：

```bash
# 错误：mkdir my folder 会创建两个文件夹 my 和 folder
# 正确：
mkdir "my folder"
cd "my folder"
touch "my file.txt"
cp "my file.txt" ./backup/
rm "my file.txt"
```

**4. 文件名以 `-` 开头时使用 `--`**：Linux 命令中 `-` 开头通常表示选项。

## 块 42

## 环境变量是什么？——先搞懂这张“备忘录”

你已经掌握 `cd`、`ls`、`mkdir` 这些命令了，但有没有想过：为什么你在终端敲 `python3`，系统就知道去哪找它？答案就是环境变量。

**一句话理解**：环境变量是操作系统或应用程序运行时使用的**动态键值对**，用来传递配置信息。你可以把它想象成操作系统的“备忘录”——系统启动时加载这些备忘录，所有程序都能读到里面的内容，并据此调整自己的行为。

**几个最常见的“备忘录条目”**：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪里找可执行程序（比如你输入 `python3`，系统就是靠 PATH 找到它的位置） |
| `HOME` | 当前用户的家目录路径（如 `/home/zhangsan`） |
| `USER` | 当前登录的用户名 |
| `SHELL` | 当前使用的 Shell 程序路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码（如 `en_US.UTF-8`） |
| `PWD` | 当前工作目录路径（随 `cd` 命令自动更新） |
| `EDITOR` | 默认文本编辑器（影响 `git commit` 等操作） |

**为什么需要环境变量？** 三个核心原因：

1. **控制程序行为**：例如通过 `JAVA_HOME` 指定 JDK 路径，让依赖 Java 的程序知道去哪找它。
2. **简化配置**：避免在每次运行程序时都手动输入长长的路径。
3. **跨进程通信**：子进程会自动继承父进程的环境变量，实现配置的无缝传递。

**环境变量的分类**：按生命周期分为两类——**临时**和**永久**。这个区别是后续操作的基础，下一节我们会

## 块 43

你的目标是**vLLM部署**，而vLLM启动时往往需要临时指定一些环境变量（比如模型路径、显存分配策略）来做快速验证。这一块讲的正是这类“用完即走”的变量。

**临时变量**的生命周期很短：只在**当前终端会话**内有效。一旦你关闭终端窗口或断开SSH连接，它就自动消失。这非常适合在调试时快速测试某个配置，而不污染系统全局设置。

看一个最直接的操作流程：

```bash
# 设置一个临时变量
export MY_VAR="hello_linux"

# 查看它的值
echo $MY_VAR
# 输出：hello_linux

# 删除这个变量
unset MY_VAR

# 再查看

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 44

**永久变量：写入配置文件，持久生效**

临时变量关掉终端就没了。要让变量每次打开终端都在，就得写进**配置文件**。

| 级别 | 配置文件 | 影响范围 | 是否需要 sudo |
|---|---|---|---|
| **用户级（推荐）** | `~/.bashrc` | 仅当前用户 | 否 |
| **用户级** | `~/.bash_profile` 或 `~/.profile` | 仅当前用户（登录时生效） | 否 |
| **系统级** | `/etc/profile` 等 | 所有用户 | 是 |

**选哪个？** 日常开发优先用 `~/.bashrc`，只影响你自己，改错不伤系统。系统级配置要 `sudo` 且影响所有用户，不建议日常动。

**本块学习

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 45

你的目标是多卡环境下用 vLLM 部署模型，而 vLLM 启动时会读取环境变量（如模型路径、并行度设置）。如果这些变量每次都要手动敲，既慢又容易错。本块教你用 `~/.bashrc` 把变量永久固化，让新终端自动加载，省去重复劳动。

**操作步骤**（跟着做一遍）：

```bash
# 1. 打开配置文件（用上节课的 nano）
nano ~/.bashrc

# 2. 滚到文件末尾，添加一行：
export MY_VAR="hello_permanent"

# 3. Ctrl+O 保存，Ctrl+X 退出

# 4. 立即生效，不用重启终端：
source ~/.bashrc

# 5. 验证：
echo $MY_VAR
# 应输出：hello_permanent
```

**关键点**：
- `~/.bashrc` 在每次打开**新的交互式终端**时自动加载，是日常开发最常用的配置位置。
- `source` 相当于“重新执行一遍这个配置文件”，让修改在当前终端立刻生效。
- **注意限制**：并非所有 SSH 会话、IDE 集成终端或服务进程都会自动加载它。比如某些 IDE 终端只加载 `~/.profile`，

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 46

## 用户级配置与优先级：为vLLM部署铺路

你的目标是多卡vLLM部署，这通常通过SSH登录远程服务器完成。此时生效的正是**登录Shell**——也就是本节主角`~/.bash_profile`或`~/.profile`的用武之地。

**关键规则**：登录Shell按顺序读取`~/.bash_profile`、`~/.bash_login`、`~/.profile`，且仅当`~/.bash_profile`不存在时才读取`~/.profile`。日常开终端更常用`~/.bashrc`（每次打开都加载），但SSH登录场景下，配置写在`~/.bash_profile`里才确保生效。

**优先级**（从高到低）：
1. **临时变量**（当前终端`export`的）
2. **用户级配置**（`~/.bash_profile`、`~/.profile`、`~/.bashrc`等，不同Shell类型加载不同文件）
3. **系统级配置**（`/etc/profile`等）

临时设置会覆盖配置文件。比如`

## 块 47

## 5.1 PATH：命令的“搜索路径”

PATH 是最重要的环境变量，没有之一。它定义了系统**查找可执行文件**的目录列表。对你即将进行的 vLLM 多卡部署来说，`python3`、`vllm`、`torchrun` 这些命令能否被正确找到，全靠 PATH 说了算。

### 查看 PATH

```bash
echo $PATH
```

输出类似这样（多个路径用冒号 `:` 分隔）：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

当你在终端输入 `python3` 时，系统会**从左到右**依次在这些目录里找名为 `python3` 的可执行文件，找到了就执行，找不到就提示 `command not found`。多卡环境下，你大概率会用 conda 创建虚拟环境，如果 `vllm` 装好了却提示找不到，多半就是 PATH 没包含 conda 环境的 `bin` 目录。

### 查看命令具体位置

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 命令能告诉你某个命令到底是从 PATH 的哪个目录里找到的。部署 vLLM 时，如果 `which python3` 指向的不是你 conda 环境里的解释器，后续 `torchrun` 拉起多卡进程时就会版本错乱。

### 修改 PATH

**追加路径**（添加到末尾，优先级较低）：

```bash
export PATH=$PATH:/new/path
```

**前置路径**（添加到开头，优先级更高，系统会先搜这里）：

```bash
export PATH=/new/path:$PATH
```

> **安全提示**：不要把 `.`（当前目录）添加到 PATH 中，以避免安全风险

## 块 48

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

`LD_LIBRARY_PATH` 告诉系统在**运行时**去哪里找额外的动态库（`.so` 文件）。你的 4×RTX 4090 多卡环境要跑 vLLM 分布式部署，CUDA 和 PyTorch 的库文件常不在默认路径，配置它就是让程序能找到这些库。

先看当前值：
```bash
echo $LD_LIBRARY_PATH
```
临时添加 CUDA 库路径：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

**典型场景**：你编译的程序依赖某个 `.so`

## 块 49

### 5.3 LIBRARY_PATH：编译时库文件路径（②）

`LIBRARY_PATH` 是给编译器用的，告诉 `gcc` 在**编译链接阶段**去哪里找库文件（静态库 `.a` 或动态库 `.so`）。它和运行时无关，只影响“能不能链接上”。

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**三个变量一表分清**（②）：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令在任何位置都能运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态链接库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

> **对你的提示**：做 vLLM 部署时，`PATH` 和 `LD_LIBRARY_PATH` 是你最常碰的。装 Miniconda、CUDA、PyTorch 都要配好它们。（②+③）

---

## 六

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 50

## 6.1 先确认 CUDA 是否已在环境变量中

在动手配置前，先检查当前环境是否已经包含 CUDA 路径：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

有输出（如 `/usr/local/cuda/bin`）说明已配置，可直接跳到 6.3 验证；无输出则继续 6.2。

## 6.2 编辑 ~/.bashrc 添加 CUDA 路径

```bash
nano ~/.bashrc
```

在文件末尾添加：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

> **注意**：`/usr/local/cuda` 是默认安装路径。若你的 CUDA 装在别处，请替换为实际路径。

## 6.3 使配置生效并验证

```bash
source ~/.bashrc
echo $PATH | grep cuda
nvcc --version
```

`nvcc --version` 能输出版本号即配置成功。

---

**本块学习检查**：完成本块后，用自己的话说明本块

## 块 51

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你在4×RTX 4090 96GB多卡环境上做vLLM分布式部署，配置环境变量是启动前的“体检”。本块教你三件事：

**1. 看全局**：`printenv` 或 `env` 列出所有变量，快速确认CUDA_VISIBLE_DEVICES、NCCL相关设置是否就位。

**2. 查单个**：`echo $PATH` 看路径，`printenv PATH` 同样效果，`declare -p PATH` 显示是否export（决定子进程能否继承）。部署vLLM时，确认python和CUDA路径正确是关键。

**3. 验生效**：改完 `~/.bashrc` 后必须 `source ~/.bash

## 块 52

### 8.1 变量未生效（①）

**可能原因**：
- 改了配置文件但没 `source` 或重启终端
- 配置文件语法错误（比如漏写 `export`）
- 变量被其他位置覆盖
- 终端或 IDE 没加载 `~/.bashrc`

**排查步骤**：

```bash
# 1. 确认配置正确
cat ~/.bashrc | grep "你的变量名"

# 2. 重新加载
source ~/.bashrc

# 3. 检查重复定义
grep -r "PATH=" ~/.bashrc
```

---

**本块学习检查**：完成本块后，用自己的话说明“变量未生效”的排查思路与你 vLLM 分布式部署目标的关系；遇到无法解释的术语（如 `source` 机制）记录下来再提问。

## 块 53

## 9.1 脚本中临时修改变量（①）

在分布式 vLLM 部署中，你经常需要为不同 GPU 卡设置不同的环境变量（如显存分配、并行策略参数）。直接在终端 `export` 的变量会污染当前会话，而脚本内临时修改变量则更干净：

```bash
#!/bin/bash
export TEMP_VAR=123
# 仅在脚本执行期间生效，脚本结束后变量消失
```

**要点**：这个 `export` 写在脚本里，变量只在脚本运行时存在。脚本退出后，终端里 `echo $TEMP_VAR` 不会有输出。这对你管理 4×RTX 4090 的多卡启动脚本非常实用——每个脚本可以独立设置自己的变量，互不干扰。

## 9.2 使用 env 命令指定单次运行的变量（①）

当你只想为某一次命令临时设置变量，而不想写脚本时：

```bash
env MY_VAR=hello python3 my_script.py
# MY_VAR 仅在这次 python3 运行时有效
```

**要点**：`env` 命令把变量"注入"到紧随其后的命令环境中，命令结束后变量即消失。在 vLLM 部署场景，你可以这样为单次推理测试临时指定参数，无需修改任何配置文件。

## 9.3 使用 .env 文件分组管理变量（④）

多卡部署往往需要一组变量（模型路径、端口、并行度等

## 块 54

## 场景 1：创建并验证自定义变量

在 vLLM 分布式部署中，你经常需要设置模型路径、并行度等环境变量。先掌握最基础的变量创建与持久化。

**临时变量**（仅当前终端有效）：

```bash
export MY_NAME="zhangsan"
echo $MY_NAME   # 输出：zhangsan
```

**永久生效**：将 export 命令追加到 `~/.bashrc`，每次打开终端自动加载：

```bash
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc
source ~/.bashrc   # 重新加载配置
echo $MY_NAME      # 验证仍可输出
```

## 场景 2：为自定义脚本目录配置 PATH

假设你把 vLLM 启动脚本放在 `~/scripts`，希望任何目录下都能直接调用。

```bash
# 1. 创建目录与测试脚本
mkdir -p ~/scripts
echo '#!/bin/bash' > ~/scripts/hello_tool.sh
echo 'echo "Hello from my tool!"' >> ~/scripts/hello_tool.sh
chmod +x ~/scripts/hello_tool.sh

# 2. 将目录加入 PATH（注意用 $PATH 保留原有路径）
echo 'export PATH=$PATH:~/scripts' >> ~/.bashrc
source ~/.bashrc
```

**本块学习检查**：

## 块 55

**场景 3：模拟 CUDA 环境变量配置**

你还没装 CUDA 也能先练配置。目标是让系统找到 CUDA，为 vLLM 部署铺路。

1. 编辑 `~/.bashrc`：`nano ~/.bashrc`
2. 末尾加两行（假设 CUDA 在 `/usr/local/cuda`）：
   `export PATH=/usr/local/cuda/bin:$PATH`
   `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
3. 保存并生效：`source ~/.bashrc`
4. 验证：`echo $PATH | grep cuda`

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（

## 块 56

你已掌握分布式训练，现在要迈向 vLLM 部署。在配置多卡推理服务时，系统级环境变量是公共服务器上必须理解的底层机制。本块聚焦两个关键文件，它们决定了所有用户的默认环境。

**A.1 /etc/environment**
- 特点：**无需 `export`**，直接写 `KEY=value`。
- 限制：仅支持简单键值对，**不支持 Shell 语法**（如 `$PATH` 引用）。
- 操作：`sudo nano /etc/environment` 添加 `JAVA_HOME=/usr/lib/jvm/java-11-openjdk`，保存后需**重启或重新登录**生效。

**A.2 /etc/profile**
- 适用：全局 Bash 配置，**需要 `export`**。
- 操作：`sudo nano /etc/profile` 添加 `export PATH=$PATH:/usr/local/custom/bin`，保存后 `source /etc/profile` 可当前会话生效，或重启系统。

**本块学习检查**：用自己的话说明

## 块 57

你的目标是vLLM多卡部署，但环境变量配置是第一步——不同Shell的配置文件位置不同，直接影响你能否正确加载CUDA、vLLM等环境变量。

**先确认你的Shell**：执行 `echo $SHELL`，若输出 `/bin/bash` 即Bash。

**对照表（重点看Zsh）**：
- **Bash**：日常终端用 `~/.bashrc`；SSH登录用 `~/.bash_profile`
- **Zsh**：所有交互式Shell都用 `~/.zshrc`（macOS默认或你手动切换时）
- **Fish**：`~/.config/fish/config.fish`

**实践建议**：完成Shell配置后，在启动vLLM前先验证环境变量是否生效（如执行 `echo $CUDA

## 块 58

## 清理 PATH 重复项与调试技巧（面向 vLLM 分布式部署）

你在多卡环境下部署 vLLM，会反复修改 `PATH`、`LD_LIBRARY_PATH` 等变量来切换不同版本的 CUDA、cuDNN 或 Python 环境。改得多了，`PATH` 里难免出现重复路径——这不会报错，但会让 `which`、`type -a` 的排查结果变得混乱，也可能让系统加载到旧版本的可执行文件。

**清理重复路径**，一条命令即可：

```bash
export PATH=$(echo $PATH | awk -v RS=: -v FS=: '!a[$0]++' | paste -sd: -)
```

这条命令把 `PATH` 按冒号拆开，去重后再用冒号拼回去。注意它只对当前 Shell 生效，确认无误后可以把它追加到 `~/.bashrc` 里。

**按环境加载不同配置**：如果你在开发机和生产机上用同一份 `~/.bashrc`，可以这样区分：

```bash
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

这样 `ENVIRONMENT` 变量不同，加载的环境变量文件就不同，适合区分开发/生产环境的 CUDA 或 vLLM 配置。

**调试脚本**：部署 vLLM 时如果启动脚本行为异常，用详细模式运行：

```bash
bash -x your_script.sh
```

这会打印每一行执行前的变量值，方便定位是哪个环境变量没生效。也可以在当前 Shell 中临时

## 测评题目

### Q-KB0100-002
执行以下命令验证当前用户和主机名：
```bash
whoami && hostname
```

### Q-KB0100-003
执行以下命令确认系统发行版信息：
```bash
cat /etc/os-release
```

### Q-KB0101-003
请依次执行以下命令并描述每一步的结果：
```bash
mkdir -p /tmp/test_dir/sub_dir
cd /tmp/test_dir
touch hello.txt
cp hello.txt sub_dir/
mv hello.txt renamed.txt
ls -R
```

### Q-KB0101-007
请写出一条管道命令，统计当前目录下 `.py` 文件的数量。

### Q-KB0102-001
临时设置环境变量 `MY_VAR=hello`，命令为________；若要使其在当前shell及子进程中生效，需使用________命令。

### Q-KB0102-003
以下哪个文件是bash shell用户登录时**最先**读取的全局配置文件？

### Q-KB0103-004
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch` 中 `-i` 参数的作用是？

### Q-KB0104-004
`nvidia-smi` 输出中 `GPU-Util` 为 0% 表示________；`Pwr:Usage/Cap` 为 `20W / 200W` 表示________。
