# 学习包

## 块 1

**本块学习检查**  
先用一个最小示例回顾：在终端执行 `ls` 查看当前目录文件，`cd` 切换目录，`mkdir` 创建新文件夹，`cp` 复制文件，`mv` 移动或重命名。确认你能解释每条命令的作用后，再继续本块。

---

你的目标是 vLLM 推理服务，而 vLLM 跑在 Linux 上。你已有 PyTorch 和 Transformer 基础，但缺 Linux 环境这一环。本块只解决一个问题：**Linux 是什么，为什么你的 A100 服务器必须用它**。

Linux 是操作系统，和 Windows/macOS 同类，但全球绝大多数服务器（包括 AI 训练集群）都跑 Linux。你的 A100 云服务器几乎必然是 Linux 系统，所以你必须能熟练操作它。

本教程用 **Ubuntu 24.04 LTS**（长期支持版），厂家持续提供多年安全更新，稳定且适合入门。若你已有 Ubuntu 22.04，步骤基本一致，可继续用。

**为什么你绕不开**：vLLM 部署涉及 GPU 驱动、CUDA、Python 环境管理，这些在 Linux 终端下操作最直接。Windows 的 WSL2 或云服务器都能提供 Linux 环境，但既然你有 A100 云服务器，直接在上面操作即可。

**核心动作**：打开终端（Terminal），确认你能执行 Shell 命令。后续所有 vLLM 操作都在这个终端里完成。

## 块 2

你的目标是 A100 上跑 vLLM 高并发推理，所以环境选择的核心不是“哪个最像 Linux”，而是“哪个能让你最快、最稳地开始敲命令”。对照下表，直接看结论：

| 方案 | 一句话 | 对你的关键点 |
|---|---|---|
| 虚拟机 | 电脑里模拟一台新电脑 | 最隔离，但吃内存（≥4GB），配置低会卡 |
| **WSL2** | Windows 自带的 Linux 子系统 | **轻量、启动快**，几分钟装完，适合不想折腾 |
| 云服务器 | 租远程电脑 | 有公网 IP、不占本地资源，但需付费 |

**建议**：优先选 **WSL2**。你的实训机是云上的 A100，本地环境只是用来写代码和发命令的入口，WSL2 足够

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

三个方案最终体验一致：你得到一台运行 Ubuntu 24.04 的“电脑”。虚拟机/云服务器通过 SSH 从 Windows 远程操作（命令行版远程桌面）；WSL2 与 Windows 互通，用 VS Code 本地开发。装好插件后，写代码有高亮、补全和调试。

**练习**：确认你的 A100 云服务器能通过 SSH 连上，并执行 `lsb_release -a` 验证系统版本。

## 块 4

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是A100云服务器上跑vLLM，但本块是**在本地Windows里装虚拟机**——这是为后续在虚拟Linux环境中练习Shell命令做铺垫。既然你最终用云服务器，**本块可跳过**，直接进入下一节。若仍想本地练习，按下面操作：

1. 浏览器打开 `https://www.virtualbox.org/wiki/Downloads`
2. 找到 **"VirtualBox 7.x.x platform packages"** 下的 **Windows hosts**，点击下载
3. 双击安装包 → 一路“下一步”保持默认 → 点“安装”等

## 块 5

**本块学习检查**：完成下载后，用自己的话说明“镜像文件”和“desktop版”与你最终在A100上跑vLLM推理服务的关系；遇到不懂的术语（如ISO、amd64）先记下再提问。

---

**下载 Ubuntu 24.04 镜像（.iso）**

镜像文件就是“操作系统安装光盘的电子版”，VirtualBox靠它装系统。

1. 浏览器打开：`https://releases.ubuntu.com/24.04/`
2. 找文件名类似 `ubuntu-24.04.x-desktop-amd64.iso`（x是小版本号），点击下载
3. 文件约5GB，下载需几分钟到十几分钟
4. 下载完**别双击打开**，放下载文件夹

## 块 6

## 2.3 在 VirtualBox 里创建虚拟机

你的目标是跑 vLLM 推理，但先别急着上 A100——本地虚拟机是练手环境，用来熟悉 Linux 操作和 Shell 命令，这些是后续在云服务器上部署 vLLM 的基础。

**操作步骤：**

1. 打开 VirtualBox，点 **"新建"**。
2. 填写：
   - **名称**：`Ubuntu-24.04`
   - **文件夹**：选 D 盘或 E 盘（别放 C 盘）
   - **类型**：`Linux`，**版本**：`Ubuntu (64-bit)`
3. 内存拉到 **4096 MB**（4GB）。
4. 虚拟硬盘：选 **"现在创建虚拟硬盘"**，大小 **60 GB**，类型 **VDI**，存储选 **动态分配**。
5. 创建后点 **"设置"**：
   - **系统 → 处理器**：数量 `1`，

## 块 7

**本块学习检查**：完成安装后，请用自己的话说明：为什么在A100云服务器上做vLLM推理，还需要先在本地虚拟机里装Ubuntu？如果遇到不懂的术语（如“虚拟盘”），记下来再提问。

---

### 2.4 安装 Ubuntu 系统

1. 在 VirtualBox 窗口选中虚拟机，点 **“启动”**。
2. 等待出现安装界面，按顺序操作：
   - **语言**：选“中文(简体)”或“English” → 点“安装 Ubuntu”。
   - **键盘布局**：默认即可，点“继续”。
   - **更新和软件**：选“正常安装”，勾选“安装 Ubuntu 时下载更新”和“安装第三方软件” → 点“继续”。
   - **安装类型**：选“清除整个磁盘并安装 Ubuntu”——注意这里的“磁盘”是虚拟机的60G，不是你的真实硬盘 → 点“现在安装” → 弹窗点“继续”。
   - **时区**：地图点中国或输入“Shanghai” → 点“继续”。
   - **用户信息**：填写名字、计算机名、用户名、密码 → 点“继续”。
3

## 块 8

### 2.5 安装 SSH 远程连接工具（①）

你的目标是 vLLM 部署，但模型推理通常不在你眼前这台 Windows 上跑，而是在云端的 A100 服务器上。SSH 就是你从本地“远程操控”那台服务器的命令行通道——装好它，你就能在本地终端里敲命令，直接指挥远端 Ubuntu，不用来回切换窗口。

**操作步骤（在 Ubuntu 虚拟机里完成）：**

1. 在 Ubuntu 桌面空白处点**鼠标右键** → 选**“在终端中打开”**。如果右键没有，按 `Ctrl + Alt + T` 也能打开终端。
2. 依次输入以下命令（输完一行按一下回车）：

```bash
# 第1步：更新软件列表（类似“刷新一下应用商店的货架”）
sudo apt update

# 第2步：安装 SSH 服务（让别的电脑能远程连过来）
#     openssh-server 是 SSH 服务端
#     net-tools 提供 ifconfig 命令，用来查看 IP 地址
sudo apt install -y openssh-server net-tools

# 第3步：设置 SSH 开机自动启动
sudo systemctl enable ssh

# 第4步：立刻启动 SSH 服务
sudo systemctl start ssh
```

> **sudo 是什么？** `sudo` = “以管理员身份执行”。Linux 里普通用户不能随便装软件，加了 sudo 就相当于“给我最高权限，让我干这件事”。输入 sudo 命令后会让你输密码，输的时候**屏幕不会有任何显示**（不会出现 `***`），这是正常的安全设计，盲打完按回车就行。

3. 查看虚拟机的 IP 地址：

```bash
# 显示网络信息，找到你的 IP 地址
ifconfig
```

4. 输出内容很多，找到类似这一行：`inet 192.168.xxx.xxx`，那一串数字就是 IP 地址。**记下来！** 例如：`192.168.126.128`
5. 验证 SSH 是否可用：回到 Windows，按 `Win + R`，输入 `cmd` 回车，打开命令提示符，输入：

```bash
# 用 SSH 连接你的虚拟机。把 ubuntu 换成你自己的用户名，IP 换成你自己的

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.6 安装增强功能（让虚拟机和 Windows 更好配合）

这一步让你能在 Windows 和虚拟机之间**直接复制粘贴文字和文件**，省去来回传文件的麻烦。

新版 VirtualBox 通常会自动装好。若没生效，手动执行：

```bash
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11
sudo reboot
```

重启后，Windows 里 `Ctrl+C` 复制，Ubuntu 里 `Ctrl+V` 粘贴即可。

**方案一完成！** 你已有一台完整的 Ubuntu 虚拟机。

---

**练习验证**：重启后，在 Windows 复制一行文字，粘贴到 Ubuntu 终端里确认成功。若失败，记录报错信息。

## 块 10

### 3.1 确认虚拟化已开启（①）

**为什么先查这个？** 你要在 Windows 里跑 WSL2，虚拟化是它的地基。没开，后面全白搭。

**1分钟自查：**
1. 按 `Ctrl + Shift + Esc` 打开**任务管理器**
2. 点顶部**“性能”**标签页
3. 右下角找**“虚拟化”**，看状态

- **已启用** → 直接跳到下一步
- **已禁用** → 进 BIOS 开启（见下）

**BIOS 开启步骤（仅禁用时需做）：**
1. 重启电脑，开机时狂按 `F2` 或 `Del`（联想 F2，惠普 F10，戴尔 F2）
2. 进入 BIOS 界面，用方向键找**“Configuration”**或**“Advanced”**菜单
3. 找到**“Intel Virtualization Technology”**或**“AMD SVM”**，改成**“Enabled

## 块 11

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的目标是A100云服务器上跑vLLM推理，但本块是Windows本机装WSL2——这是“环境准备”的备选方案。若你后续想在本机调试或测试，WSL2能提供Linux终端环境；但你的实训条件是云A100，实际推理部署应在云上Linux完成。本块价值在于：让你理解WSL2是Windows下的轻量Linux环境，装好后可执行Shell命令，为熟悉Linux操作打基础。

**操作步骤**（按顺序执行）：

1. 按`Win`键，输入`powershell`，右键“Windows PowerShell” → “以管理员身份运行”，弹窗点“是”。
2. 在PowerShell中逐行输入以下命令（复制整行，右键粘贴，回车）：
   ```bash
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```
3. 两条命令均显示“操作成功完成”后，重启电脑

## 块 12

**本块学习检查**：完成安装后，请用自己的话说明：为什么在A100云服务器上做vLLM高并发推理，仍需先在本地Windows装好Ubuntu 24.04？遇到无法解释的术语（如WSL2）先记下，稍后提问。

---

**安装Ubuntu 24.04（二选一）**

**方法一（推荐）**：按`Win`键搜索并打开`Microsoft Store`，搜索`Ubuntu 24.04`，选择发布者为**Canonical Group Limited**的版本，点“安装”（约500MB），完成后点“打开”。

**方法二（命令行）**：直接运行：
```bash
wsl --install -d Ubuntu-24.04
```

**首次启动**：显示“Installing...”等待2~5分钟，然后设置用户名（全英文小写）和密码（输入时屏幕无显示，属正常），确认密码后出现`用户名@计算机名:~$`即成功。

> **坑**：商店搜不到？先更新Windows到最新版。

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在Ubuntu终端逐行执行：

```bash
sudo apt update
sudo apt upgrade -y
```

第一条刷新软件包列表，第二条把所有已装软件升级到最新版（`-y`自动确认）。WSL2无需SSH，直接通过VS Code的Remote-WSL插件开发即可。

验证系统：

```bash
lsb_release -a
whoami
```

应显示Ubuntu 24.04.x和你设置的用户名。再测与Windows互通：

```bash
explorer.exe .
```

弹出资源管理器窗口即无缝衔接。方案二完成。

---

**方案三：云服务器（阿里云轻量应用服务器）**

需付费，学生通常有优惠价。适合想要24小时在线、任何地方都能连的服务器场景。

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

## 块 14

**本块学习检查**：完成购买后，用自己的话说明“云服务器”与“vLLM部署”的关系；遇到不懂的术语（如“镜像”）先记下再提问。

你的目标是用A100做高并发vLLM推理，但本块是“买服务器”的入门步骤。阿里云轻量服务器的**最低配2核2G仅够基础练习，不适用于大模型训练或vLLM部署**——所以购买时重点不是选最低配，而是熟悉流程。

操作步骤（以阿里云为例，腾讯云、华为云类似）：

1. 打开 https://www.aliyun.com/ ，用支付宝/淘宝账号注册登录。
2. 顶部搜索“**轻量应用服务器**”进入购买

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的A100服务器默认是root登录，但root权限过高，误操作可能删系统文件。日常跑vLLM推理，应先用普通用户操作。

通过云厂商控制台的“远程连接”或VNC（网页终端）登录后，执行：

```bash
adduser yourname   # yourname换成全英文小写用户名，设密码两次，其余回车跳过
usermod -aG sudo yourname   # 给该用户sudo权限（管理员资格）
```

**操作验证**：创建用户后，用`exit`退出root，再用

## 块 16

你的目标是在 A100 服务器上跑高并发 vLLM 推理，SSH 密钥登录是安全上服务器的第一步。密码登录容易被暴力破解，密钥登录更安全，也省去每次输密码的麻烦。

**操作步骤（Windows 本地 → 云服务器）：**

1. **生成密钥对**：在 Windows 上按 `Win + R`，输入 `cmd` 打开命令提示符，执行：
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```
2. **一路回车**：出现 `Enter file...` 直接回车（默认路径）；出现 `Enter passphrase` 直接回车（不设密码）；再出现 `Enter same passphrase again` 再回车。看到 randomart image 即生成成功。
3. **查看公钥**：执行 `type %USERPROFILE%\.ssh\id_rsa.pub`，复制全部输出。
4. **服务器上配置**：通过云厂商 VNC/远程连接登录服务器，确保目标用户 yourname 已存在，然后切换到该用户（若当前以 root 登录，执行 `su - yourname` 并输入该用户密码）：
   ```bash
   su - yourname
   mkdir -p ~/.ssh
   nano ~/.ssh/authorized_keys
   ```
   将复制的公钥粘贴进 nano，`Ctrl+O

## 块 17

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的A100服务器要跑vLLM，必须先能安全登录。本块只做一件事：在云控制台**防火墙**页，添加规则开放22端口（TCP协议）。来源建议填你本机公网IP，格式如`123.45.67.89/32`；IP不固定就暂填`0.0.0.0/0`，但这会让SSH暴露给所有公网IP，易遭暴力破解，建议改用云服务商的安全组限制来源IP，或使用VPN/跳板机。记住：别长期用root+密码登录，

## 块 18

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的 A100 服务器已就绪，现在只需从 Windows 远程登录它。打开命令提示符（`Win + R` 输入 `cmd`），用控制台复制的公网 IP 执行：

```bash
ssh yourname@<公网IP>
```

首次连接提示 `yes/no` 时输入 `yes` 回车。若密钥配置正确，将直接登录成功——无需密码。至此你已拥有一台 24 小时在线的 Ubuntu 服务器，这正是后续部署 vLLM 推理服务的基础环境。

## 块 19

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

在A100云服务器上部署vLLM前，你需要能顺畅地远程操作服务器。SSH免密登录就是为此服务的：它让你每次连接时不用反复输密码，节省时间，也避免中断。

**验证步骤**（假设4.3节已完成密钥配置，包括生成密钥对并将公钥添加到服务器）：

在Windows命令提示符或PowerShell中执行：

```bash
ssh 用户名@你的IP地址
```

能直接连上且不提示密码，即配置成功。

**常见坑**：若提示`ssh-copy-id

## 块 20

### 5.2 WSL2：无需 SSH，直接集成（①）

你用的是云服务器（A100），所以本小节只做了解：WSL2 是 Windows 下的 Linux 子系统，VS Code 通过 Remote-WSL 插件直接与它通信，**不需要配置 SSH**。你的场景走 5.3 里的 Remote-SSH 路线。

### 5.3 安装 VS Code 及远程开发插件（①）

**VS Code**（Visual Studio Code）是微软的免费代码编辑器，轻量、插件多、跨平台。你后续写 vLLM 推理脚本、调试 Python 代码都会在它里面进行。

#### 第一步：安装 VS Code（①）

1. 浏览器打开 https://code.visualstudio.com/
2. 点 **"Download for Windows"** 下载安装包
3. 双击运行，安装过程中**重点勾选**：
   - ☑ **"将 Code 添加到"PATH"中"** —— 这样可在命令行直接输 `code` 打开编辑器
   - ☑ **"将 Code 注册为受支持的编辑器"**
   - ☑ **"添加到文件资源管理器上下文菜单"**
4. 其余一路"下一步"→"安装"→完成

#### 第二步：安装必备插件（①）

插件是 VS Code 的功能扩展包，装了才有代码高亮、智能提示、远程连接等能力。

1. 打开 VS Code
2. 点左侧竖排图标栏**第五个图标**（四个小方块），或按 `Ctrl + Shift + X` 打开扩展商店
3. 搜索并安装：

| 搜索关键词 | 插件全名 | 发布者 | 作用 |
|---|---|---|---|
| Python | Python | Microsoft | Python 代码高亮、智能提示、调试 |
| Remote - SSH | Remote - SSH | Microsoft | **云服务器用户装**，通过 SSH 远程连接 A100 服务器 |
| Remote - WSL | Remote - WSL | Microsoft |

## 块 21

### 连接你的 Linux 环境（①）

你的目标是 A100 上跑 vLLM 推理，所以远程连上服务器是第一步。下面是针对云服务器的操作：

1. 打开 VS Code，看左下角有个**绿色小图标**（像 `><`）。
2. 点击它 → 顶部弹出菜单，选 **"Connect to Host…"（连接到主机）**。
3. 输入 `用户名@你的IP地址`（如 `zhangsan@192.168.126.128`）→ 回车。
4. 如果弹出选平台，选 **"Linux"**。
5. 之前配了密钥就直接连上；没配就输密码。
6. 连上后左下角图标变成 `SSH: IP地址`，说明成功。
7. 点左上角 **"File（文件）"** → **"Open Folder…（打开文件夹）"**，选服务器上的目录（如 `/home/你的用户名`），点 OK。
8. 现在就能像编辑本地文件一样编辑远程文件了。

> **WSL2 用户**：点绿色图标 → 选 **"Connect to WSL"**，自动连上 Ubuntu。更省事的方法：在 Ubuntu 终端里 `cd` 进项目目录，输入 `code .`，一步到位。

### 熟悉终端：Linux 基本命令入门（②）

终端就是那个黑框框，等待你输入命令。下面按场景介绍最常用的命令：

> **前置提示**：如果提示 `command not found`（如 `vim`、`nano` 未预装），先执行：
> ```bash
> sudo apt install -y vim nano
> ```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM 部署）的关系；遇到无法解释的术语时记录下来再提问

## 块 22

## 6.1 文件与目录操作：最基础也最常用

你在 A100 上部署 vLLM 推理服务时，模型权重文件、日志、配置脚本都离不开文件管理。这一节掌握好，后面操作才顺手。

### 核心命令速览

**导航三件套**

- `pwd`：显示当前绝对路径，比如在 `/home/user` 下执行就输出 `/home/user`。迷路时先敲它。
- `ls`：列出当前目录内容。`ls -l`（Ubuntu 可简写 `ll`）显示权限、大小等详细信息；`ls -a` 显示隐藏文件。
- `cd`：切换目录。`cd ..` 上级、`cd /` 根目录、`cd ~` 回主目录、`cd /home/user/Documents` 进指定目录。

**创建与查看**

- `mkdir test` 创建单目录；`mkdir -p test/aaa/bbb` 递归创建多级目录，部署时建项目层级很实用。
- `touch file1.txt file2.log` 一次创建多个空文件。
- `cat -n file1.txt` 查看小文件并显示行号；大日志文件用 `less app.log`，空格翻页、回车换行、`q` 退出。

**复制、移动与删除**

- `cp file1.txt /home/user/Desktop` 复制文件；复制目录必须加 `-r`：`cp -r test/ /home/user/Desktop`。
- `mv file1.txt test/` 移动；`mv file1.txt newfile.txt` 重命名；`mv test/ /home/user/newtest` 移动并改名。
- `rm file1.txt` 删文件；`rm -f` 强制不提示；`rm -rf test/` 递归强制删目录。**⚠️ 不可恢复，部署时千万别在模型目录上乱敲。**

### 本块学习检查

完成本块后，用自己的话说明：这些文件操作命令与你当前“在 A100 上部署 vLLM 推理服务”的目标有什么关系？比如模型权重文件、日志文件、配置文件分别可能用到哪些命令？遇到无法解释的术语（如权限、递归）先记录下来，稍后提问。

## 块 23

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

## 6.2 权限管理：Linux 系统的安全核心

在A100云服务器上部署vLLM时，你运行`vllm serve`、修改配置、执行脚本，每一步都涉及文件权限。权限错了，命令会报"Permission denied"。

**三类权限**：每个文件/目录有「所有者、所属组、其他用户」三类角色，权限分「读(r=4)、写(w=2)、执行(x=1)」。

**chmod**：修改权限，两种写法：
- **数字法**（推荐）：`chmod 744 file1.txt` → 7=4+2+1（所有者读+写+执行），4=4（其他只读）
- **符号法**：`chmod g+w file1.txt`（给所属组加写权限）；`chmod a+x test.sh`（给所有用户加执行权限）

**chown**：改所有者，需root权限：`sudo chown root file1.txt`

**sudo回顾**：`sudo`

## 块 24

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

部署vLLM前，先学会“看体检报告”。A100云服务器上，用`uname -a`确认内核与架构；`top`实时盯CPU/内存，按`P`或`M`排序，`q`退出；`ps -ef | grep python`查推理进程是否存活；`df -h`看磁盘余量；`free -h`看内存余量。这些命令帮你判断服务器是否“健康”，为后续跑大模型推理扫清环境障碍。

## 块 25

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你已掌握PyTorch和HuggingFace，部署vLLM时需频繁修改配置文件（如模型路径、并发参数）。本块教你两种快速改文件的方法，直接服务于你的A100推理服务调试。

**1. vim编辑器**（适合精确修改）
- 打开：`vim config.py`
- 按 `i` 进入编辑模式（左下角显示 `--INSERT--`）
- 修改后：按 `Esc` → 输入 `:wq` 回车（保存退出）；若改错，用 `:q!` 不

## 块 26

## 网络操作：基础网络查询

在部署 vLLM 推理服务前，你需要能快速确认服务器的网络状态。本块只学两个命令：`ping` 和 `ifconfig`。

**1. `ping` — 测试网络连通性**

语法：`ping [选项] 目标IP/域名`

实战：`ping www.baidu.com`，按 `Ctrl+C` 停止。能收到回复说明网络通畅，这对后续从 HuggingFace 拉取模型至关重要。

**2. `ifconfig` — 查看网卡信息**

直接输入 `ifconfig`，在输出中找到 `eth0`（或 `ens` 开头）网卡的 `inet` 字段，那就是你的 IP 地址。部分系统默认用 `ip addr`。

> 若提示找不到命令，

## 块 27

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是部署 vLLM 推理服务，这离不开对 Linux 文件系统和进程管理的基本操作。下面两个场景直接对应你日常会遇到的真实需求。

**场景 1：创建并整理文件**

部署服务时，你需要组织模型文件、代码和日志。这个场景帮你熟悉目录创建、文件写入和移动。

```bash
# 1. 进入主目录
cd ~

# 2. 创建 work 及子文件夹（-p 可一次创建多级目录）
mkdir -p work/doc work/code

# 3. 创建并写入文件
echo "Linux命令学习笔记" > note.txt

# 4. 移动文件到 doc 目录
mv note.txt work/doc/

# 5. 查看结果
ls work/doc/
# 预期输出：note.txt
```

**场景 2：查询系统

## 块 28

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**避坑指南（针对A100云服务器实操）**

1. **大小写敏感**：Linux命令区分大小写。`ls`正确，`LS`报错。在vLLM部署中，文件名、路径、参数同样区分大小写，请严格照抄。

2. **路径核对**：切换目录或复制文件报错“No such file or directory”时，先用`ls`确认路径存在。部署vLLM时，模型权重路径写错是常见问题，务必先`ls`验证。

3. **rm慎用**：`rm -rf`危险，删除前用`ls`确认目标，避免误删根目录`/`。清理旧模型或日志时，先列出内容再删。

4. **权限不足**：操作系统文件提示“Permission denied”时

## 块 29

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

你的A100服务器上，Ubuntu 24.04已自带Python 3.12。先验证环境：

```bash
python3 --version   # 预期输出：Python 3.12.x
pip3 --version      # 确认pip是否可用
```

若`pip3`未找到，执行安装：

```bash
sudo apt update
sudo apt install -y python3-pip
```

**pip是什么？** pip是Python的包管理器，相当于“应用商店”。后续安装PyTorch、Transformers等库都靠它——这些正是v

## 块 30

### 7.2 安装 Miniconda（推荐）

你的目标是vLLM部署，后续会频繁用到`conda/pip`管理不同项目的依赖。系统自带Python虽够用，但Miniconda能让你为vLLM单独建一个干净环境，避免依赖冲突。这里先做最简安装。

**操作步骤：**

1. **下载安装脚本**（若提示`wget`未找到，先执行`sudo apt install wget`）：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. **运行安装**：
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

3. **安装过程**：按`Enter`阅读协议 → 输入`yes`同意 → 确认路径（直接回车）→ 提示更新shell profile时输入`yes`。

4. **激活配置**：关闭并重开终端，或执行`source ~/.bashrc`。

5. **验证**：
```bash
conda --version
# 预期输出类似：conda 24.x.x
```

**本块学习检查**：完成安装

## 块 31

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

### 7.3 运行你的第一个 Python 程序（①）

你的目标是部署 vLLM 推理服务，而 vLLM 是一个 Python 程序。所以，验证 Python 环境能跑通，是后续所有工作的地基。这一步跑通了，说明你的 A100 云服务器已经具备运行 vLLM 的基本条件。

**动手做：**

1. 在终端里执行：

```bash
# 回到主目录
cd ~

# 创建一个叫 hello.py 的文件，用 nano 编辑器打开
nano hello.py
```

2. 在 nano 里输入以下代码：

```python
print("Hello, Python on Ubuntu 24.04!")
```

3. 按 `Ctrl + O`（保存），然后按回车确认文件名
4. 按 `Ctrl + X`（退出 nano）
5. 运行程序：

```bash
python3 hello.py
```

6. 如果屏幕输出了 `Hello, Python on Ubuntu 24.04!`，恭喜！你的 Linux 环境已经全部就绪，而且你已经成功使用了 `cd`、`nano`、`python3` 等终端命令，为后续学习 PyTorch 与大模型微调迈出了坚实的第一步！

7. 验证 Python 版本和 pip：

```bash
python3 --version
pip3 --version
```

---

### 本块常见问题速查

| 症状 | 可能原因 | 解决办法 |
|---|---|---|
| `sudo apt update` 很慢 | 默认软件源在国外 | 搜"Ubuntu 24.04 换阿里云源"，按教程把下载源换成国内的 |
| 提示 `command not found: ifconfig` | net-tools 未安装 | 执行 `sudo apt install net-tools` |
| 误用 `rm -rf` 删错文件 | 命令太危险且无回收站 | 删除前先用 `ls` 确认；重要数据定期备份 |

---

### 本块小结与下一步

你现在应该拥有了：

- ✅ 一台运行 Ubuntu 24.04 的云服务器（A100 80GB）
- ✅ SSH 远程连接配置完毕，能从本地安全地操控服务器
- ✅ 掌握了

## 块 32

## 二、你在哪？pwd 与路径的概念

### 2.1 pwd：打印当前工作目录

`pwd` 是 **P**rint **W**orking **D**irectory 的缩写，作用就是告诉你"我现在在哪个文件夹里"。

```bash
pwd
```

示例输出：

```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。

**为什么这对你很重要？**

你接下来的目标是部署 vLLM 推理服务。在 A100 云服务器上，你会频繁地下载模型权重、编辑配置文件、启动推理进程。这些操作都发生在"某个目录"里——如果你不清楚自己当前在哪个目录，就很容易出现"文件下载了却找不到""配置文件改了但没生效"这类问题。

**实操建议**：在 A100 上每次登录终端后，第一件事就是敲 `pwd`，确认自己站在哪里，再开始后续操作。这能避免大量"文件找不到了"的困惑。

---

**本块学习检查**

完成本块后，请用自己的话回答：`pwd` 命令和你部署 vLLM 的目标有什么关系？在 A100 服务器上操作时，为什么先执行 `pwd` 是个好习惯？如果遇到无法解释的术语（比如"目录"或"路径"），记录下来，在提问时一并提出。

## 块 33

### 2.2 绝对路径 vs 相对路径

搞清这两个概念，对你在A100上部署vLLM时定位模型文件至关重要。

- **绝对路径**：从根目录`/`写起，不依赖你当前在哪。例如：`/home/zhangsan/Documents`。
- **相对路径**：从你当前所在位置出发。例如：`./file.txt`（当前目录）、`../backup`（上一级目录）。

记住三个符号：
- `./`：当前目录（可省略）
- `../`：上一级目录
- `~`：当前用户家目录（如`/home/zhangsan`）

**举例**：假设你在`/home/zhangsan/Documents`下：
- `cd /home/zhangsan`（绝对路径，任何位置都能用）
- `cd ..`（相对路径，回到上一级`/home/zhangsan`）
- `cd ~`（直接回家目录）

###

## 块 34

### 3.1 基础用法

在 vLLM 部署中，你经常需要确认模型文件是否就位、查看目录结构。`ls` 就是干这个的。

```bash
# 列出当前目录下的文件/文件夹（不包括隐藏文件）
ls

# 列出指定路径下的内容，比如你的模型目录
ls /home/zhangsan

# 列出所有文件（包括隐藏文件，以 . 开头）
ls -a

# 详细列表显示（权限、所有者、大小、修改时间）
ls -l

# 组合使用：详细列出所有文件，大小以人类可读格式显示
ls -ahl
```

> **关于隐藏文件**：Linux 中以 `.` 开头的文件/文件夹默认隐藏，比如 `.bashrc`、`.ssh`。这些文件往往很重要，后续配置环境变量时会频繁遇到。

### 3.2 ls -l 的输出解读

执行 `ls -l` 后，输出类似：

```bash
drwxr-xr-x 2 zhangsan zhangsan 4096 Aug  3 10:00 Documents
-rw-r--r-- 1 zhangsan zhangsan  220 Aug  3 09:30 hello

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 35

**ls 常用选项（针对 vLLM 部署场景）**

在管理模型权重或日志文件时，`ls -lh` 能直观显示文件大小（如 13GB 的模型），注意 `-h` 通常与 `-l` 搭配使用以显示人类可读的文件大小，单独使用不报错但无实际效果。`ls -t` 按修改时间排序，方便找最新日志；`ls -S` 按大小排序，快速定位大文件。组合 `ls -lhtr` 可逆序查看，便于检查最旧文件。`ls -F` 区分目录（`/`）与软链接（`@`），`ls -d` 查看目录本身属性。

## 块 36

### 本块学习检查
完成本块后，请用自己的话说明：`cd` 和 `mkdir` 如何帮助你搭建 vLLM 推理服务的项目目录结构？遇到无法解释的术语（如“相对路径”）时记录下来，课后提问。

---

## 四、cd：切换目录（基础用法）

`cd` 是你在 A100 服务器上“移动”到不同文件夹的命令。部署 vLLM 时，你需要频繁在模型权重目录、日志目录和代码目录间切换。

**核心用法：**

```bash
# 进入指定目录（绝对路径，从根目录 / 开始写）
cd /home/zhangsan/Documents

# 进入当前目录下的子文件夹（相对路径，./ 可省略）
cd ./Downloads
cd Downloads

# 返回上一级目录
cd ..

# 返回上两级目录
cd ../..

# 回到家目录（~ 是你的个人主目录）
cd ~
cd        # 直接 cd 不加参数也是回家目录

# 返回刚才所在的目录（快速来回切换）
cd -
```

**路径切换示例：**

假设当前在 `/usr/local` 下，想切换到 `/home/zhangsan`：

```bash
# 方法1：绝对路径（去哪都管用，推荐新手使用）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

> **建议**：新手优先使用绝对路径，不容易迷路；熟悉后多用相对路径，效率更高。

---

## 五、mkdir：创建目录

`mkdir` = **M**a**k**e **Dir**ectory，用来创建文件夹。部署 vLLM 时，你需要为模型、日志、配置分别建目录。

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

# 批量创建 100 个文件夹（dir1

## 块 37

## 7.1 复制文件

在部署 vLLM 推理服务时，你经常需要把写好的模型脚本或配置文件备份到其他目录，`cp` 就是干这个的。

```bash
# 复制 hello.py 到 backup 文件夹（文件名不变）
cp hello.py ./backup/

# 复制并改名
cp hello.py ./backup/hello_backup.py

# 一次复制多个文件
cp file1.txt file2.txt ./backup/

# -p：保留修改时间等属性
cp -p hello.py ./backup/

# -i：目标已存在时先询问，避免误覆盖
cp -i hello.py ./backup/
```

## 7.2 复制目录

复制文件夹**必须加 `-r`**（递归），否则系统拒绝。

```bash
cp -r myproject ./backup/
```

注意：若 `backup` 不存在，会新建它并把内容放进去；若已存在，则生成 `backup/myproject/`。

## 7.3 复制软链接

```bash
# -d 保持链接属性，不加则复制源文件本身

## 块 38

## mv 与 rm：模型文件整理与安全删除

你的 vLLM 部署会涉及大量模型权重文件（如 `.safetensors`、`.bin`）和日志文件，掌握 `mv` 和 `rm` 能让你高效管理这些文件，同时避免误删宝贵数据。

### 8.1 移动文件/文件夹

```bash
# 把模型文件移动到 backup 文件夹
mv model.safetensors ./backup/

# 把整个项目文件夹移动到 archive 位置
mv myproject ./archive/

# 用通配符移动多个文件（把所有以 data_ 开头的文件移到 data 文件夹）
mv data_* ./data/
```

### 8.2 重命名

```bash
# 把 a.txt 重命名为 b.txt（本质是在同一目录下移动）
mv a.txt b.txt

# 移动的同时改名（例如给模型文件加版本号）
mv model.safetensors ./backup/model_v2.safetensors
```

### 8.3 安全选项

```bash
# 覆盖前询问（推荐日常使用）
mv -i old.txt new.txt

# 强制覆盖，不提示（慎用）
mv -f old.txt new.txt
```

---

## 九、rm：删除文件与目录

`rm` = **R**e**m**ove，用来删除文件或文件夹。**此命令不可逆，请谨慎使用！**

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

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 39

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

在A100云服务器上，先`cd ~`回到家目录，用`mkdir llm-learning`建项目根目录，再`cd llm-learning`进入，接着`mkdir data models scripts`一次建三个子目录，用`touch scripts/train.py`创建空脚本，最后`ls -R`验证结构。这套命令为后续vLLM部署准备好`models`（存权重）和`scripts`（放推理脚本）的目录骨架。

## 块 40

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

在A100 80GB云服务器上准备vLLM推理服务前，先学会用`cp`命令整理实验数据。假设`~/Downloads/`下已有`train.csv`、`test.csv`、`dev.csv`三个数据集文件。

**第一步：确认文件存在**  
`ls ~/Downloads/*.csv`——用通配符`*.csv`列出所有CSV文件，确保源文件就位。

**第二步：复制到目标目录**  
`cp ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/dev.csv

## 块 41

## 场景 3：清理临时文件（实战演练）

你在 `~/llm-learning/` 下误建了一个空文件夹 `temp` 和一个 `old.txt` 文件，现在把它们删掉。这个操作在你后续管理 vLLM 部署目录时很常用——模型文件、日志、临时缓存堆积后，你需要快速清理。

**操作顺序**（先确认，再删除，最后验证）：

```bash
# 1. 先确认要删的内容（好习惯！）
ls ~/llm-learning/temp ~/llm-learning/old.txt

# 2. 删除文件
rm ~/llm-learning/old.txt

# 3. 删除空文件夹
rm -d ~/llm-learning/temp

# 4. 验证
ls ~/llm-learning/
```

注意：`rm -d` 只能删空文件夹，`temp` 是空的所以可以。如果文件夹里有内容，需要 `rm -r`（递归删除）——这个区别在清理模型缓存目录时尤其重要。

---

## 新手避坑指南（重点看与你相关的）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。你在部署 vLLM 时经常要复制整个模型目录，漏了 `-r` 会直接报错。

**2. 路径写错**：切换目录或复制文件时，路径写错会提示 "No such file or directory"，可先用 `pwd` 和 `ls` 确认当前位置和目录内容。

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

**4. 文件名以 `-` 开头时使用 `--`**：Linux 命令中 `-` 开头通常表示选项。如果文件名本身以 `-` 开头，命令会误把文件名当选项解析。用 `--` 告诉命令"后面都是文件名，不是选项"：

```bash
# 错误：rm -file.txt 会把 -file.txt 当成选项，报错
# 正确：
rm -- -file.txt
touch -- -data.csv
mv -- -old.txt new.txt
```

**5. rm 命令慎用**：`rm -rf` 没有回收站，删除即永久消失。删除前务必用 `ls` 确认目标。你在清理 vLLM 的旧

## 块 42

## 环境变量是什么？——从 vLLM 部署说起

你在 A100 上跑 vLLM 推理服务时，一定会遇到这类问题：输入 `vllm serve` 系统提示“命令未找到”，或者程序报错找不到 CUDA 库。这些问题的根源，往往就是**环境变量**没配好。

### 1.1 通俗理解

环境变量是操作系统或应用程序运行时使用的**动态键值对**，用于传递配置信息。你可以把它理解为操作系统的“备忘录”——系统启动时加载这些备忘录，所有程序都能读到里面的内容，并据此调整自己的行为。

对你而言，最直接相关的场景是：当你执行 `vllm serve` 时，系统靠 `PATH` 变量找到 `vllm` 可执行文件的位置；当 vLLM 调用 GPU 加速时，系统靠 `LD_LIBRARY_PATH` 或 `CUDA_HOME` 找到 CUDA 运行库。配置错了，A100 再强也跑不起来。

**常见例子**：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪里找可执行程序（比如你输入 `python3`，系统就是靠 PATH 找到它的位置） |
| `HOME` | 当前用户的家目录路径（如 `/home/zhangsan`） |
| `USER` | 当前登录的用户名 |
| `SHELL` | 当前使用的 Shell 程序路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码（如 `en_US.UTF-8`） |
| `PWD` | 当前工作目录路径（随 `cd` 命令自动更新） |
| `EDITOR` | 默认文本编辑器（影响 `git commit`

## 块 43

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

临时变量只活在当前终端会话里——关掉终端或断开SSH就没了，适合快速测试。你的A100云服务器上，跑vLLM前想临时试个参数，用这个就够了。

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

**

## 块 44

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

---

**永久变量：写入配置文件，持久生效**

想让变量每次打开终端都存在，就写入**配置文件**。

| 级别 | 配置文件 | 影响范围 | 是否需要 sudo |
|---|---|---|---|
| **用户级（推荐）** | `~/.bashrc` | 仅当前用户 | 否 |
| **用户级** | `~/.bash_profile` 或 `~/.profile` | 仅当前用户（登录时生效） | 否 |
| **系统级** | `/etc/profile` 等 | 所有用户 | 是 |

**该选哪个？** 日常开发中，建议优先使用 `~

## 块 45

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是在A100上跑vLLM推理服务，而vLLM的启动参数（如模型路径、并发数）往往通过环境变量传递。本块教你用`~/.bashrc`永久保存这些变量，避免每次开终端都重新设置。

**操作步骤**（跟着做）：

```bash
# 1. 打开配置文件
nano ~/.bashrc

# 2. 滚到文件末尾，加一行（把变量名和值换成你vLLM要用的）
export MY_VAR="hello_permanent"

# 3. Ctrl+O保存，Ctrl+X退出

# 4. 立即生效
source ~/.bashrc

# 5. 验证
echo $MY_VAR
# 输出：hello_permanent
```

**关键理解**：

- `~/.bashrc`在**每次打开新交互式终端**时自动加载——适合你日常SSH登录后手动启动vLLM的场景。
- `source`相当于"重新执行配置文件"，让修改**立刻生效**，不用重开终端。
- **注意限制**：并非所有终端都加载`

## 块 46

## 用户级配置与优先级：为vLLM部署铺路

你在A100上跑vLLM时，经常需要设置`CUDA_VISIBLE_DEVICES`、`VLLM_USE_V1`这类变量。理解配置位置和优先级，能让你少踩坑。

**用户级配置（登录时生效）**

对于**bash登录Shell**（如SSH远程登录A100服务器时），`~/.bash_profile`和`~/.profile`在登录时加载。若两者同时存在，bash优先读`~/.bash_profile`；仅当`~/.bash_profile`不存在时，才读取`~/.profile`。其他Shell（如zsh）或非登录Shell的加载规则不同。

> 实操建议：你日常开终端调试vLLM，用`~/.bashrc`更顺手（每个新交互式终端都加载）。只有需要“登录时才生效”的场景（如设置SSH登录后的初始环境），才用`~/.bash_profile`。

**优先级（

## 块 47

## 5.1 PATH：命令的"搜索路径"

PATH 是最重要的环境变量，没有之一。它定义了系统**查找可执行文件**的目录列表。

### 查看 PATH

```bash
echo $PATH
```

输出类似这样（多个路径用冒号 `:` 分隔）：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

当你在终端输入 `python3` 时，系统会**从左到右**依次在这些目录里找名为 `python3` 的可执行文件，找到了就执行，找不到就提示 `command not found`。

### 查看命令具体位置

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 命令能告诉你某个命令到底是从 PATH 的哪个目录里找到的。

### 修改 PATH

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

**本块学习检查**：完成本块后，用自己的话说明 PATH 的作用、查看与修改方法

## 块 48

## LD_LIBRARY_PATH：让程序找到“零件库”

你马上要部署 vLLM 推理服务，这背后是 CUDA、PyTorch 等一堆依赖动态库（`.so` 文件）的组件。系统默认只在 `/usr/lib`、`/lib` 里找这些库，但 CUDA 装在 `/usr/local/cuda/lib64`，不在默认路径里。**`LD_LIBRARY_PATH` 就是告诉系统“额外去这些目录找”的环境变量**。

先看当前值：
```bash
echo $LD_LIBRARY_PATH
```
大概率是空的，没关系。

临时把 CUDA 库路径加进去：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 49

**LIBRARY_PATH：编译时库文件路径（②）**

`LIBRARY_PATH` 告诉编译器（如 `gcc`）在**编译链接阶段**去哪里找库文件——包括静态库（`.a`）和动态库（`.so`）。它和运行时无关，只在编译时生效。

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**三个变量的分工**：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令在任何位置都能运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态链接库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

> **对你的建议**：做 vLLM 部署时，`PATH` 和 `LD_LIBRARY_PATH` 是你最常打交道的两个变量。安装 Miniconda、CUDA、PyTorch 时，都需要正确配置它们。

---

## 实战：配置 CUDA 环境变量

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

## 块 50

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的目标是让 A100 跑起 vLLM 推理服务，而 vLLM 依赖 CUDA 才能调用 GPU。本块就是确认系统能找到 CUDA，这是部署前的必要一步。

**第一步：检查现状**

在终端执行：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

有输出（如 `/usr/local/cuda/bin`）说明已配置，直接跳到第三步验证；无输出则继续第二步。

**第二步：配置环境变量**

```bash
nano ~/.bashrc
```

在文件末尾追加：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

> 注意：`/usr/local/cuda` 是默认路径，若你的 CUDA 装在其他位置，请替换为实际路径。

**第三步：生效并验证**

```bash
source ~/.bashrc
echo $PATH | grep cuda
nvcc --version
```

`nvcc --version`

## 块 51

## 7.1 查看所有环境变量

在部署vLLM前，先确认当前环境里有哪些变量：

```bash
printenv      # 简洁输出所有变量
env           # 更详细的输出
```

两者都能列出全部环境变量，`env` 输出格式更完整。

## 7.2 查看特定变量

只关心某个变量时，用以下任一方式：

```bash
echo $PATH
printenv PATH
declare -p PATH    # 显示定义细节（含是否 export）
```

`declare -p` 能额外告诉你该变量是否已导出到子进程——这对确认 vLLM 相关配置是否真正生效很关键。

## 7.3 测试变量是否生效

假设你在 `~/.bashrc` 里设置了 `export MY_VAR="hello"`，验证流程：

```bash
source ~/.bashrc   # 重新加载配置
echo $

## 块 52

你的目标是让 vLLM 在 A100 上跑高并发推理，而环境变量没生效会直接导致启动命令找不到或版本错乱。本块帮你快速定位这类问题。

**排查步骤（按顺序执行）**

```bash
# 1. 确认变量确实写进了配置文件
cat ~/.bashrc | grep "你的变量名"

# 2. 重新加载配置（修改后必须执行）
source ~/.bashrc

# 3. 检查是否有重复定义，导致变量被覆盖
grep -r "PATH=" ~/.bashrc
```

**常见原因对应解决**
- 改完没 `source` 或重启终端 → 执行第 2 步
- 配置文件语法错误（如漏写 `export`）→ 检查第 1 步输出
- 变量被其他位置覆盖 → 看第 3 步结果，调整定义顺序
- 终端/IDE 没加载 `~/.bashrc` → 确认启动方式（见 3.1 节）

**本块学习检查**：用自己的话说说，环境变量未生效时你会按什么顺序排查？并说明这一步与你最终启动 vLLM 服务的关系。遇到不理解的术语（如 `source` 的作用）先记下来，之后提问。

## 块 53

## 9.1 脚本中临时修改变量（①）

在写 vLLM 部署脚本时，你常常需要临时设置一些参数（比如模型路径、并发数）。用 `export` 就能在**当前脚本运行期间**生效：

```bash
#!/bin/bash
export TEMP_VAR=123
# 仅在脚本执行期间生效，脚本结束后变量消失
```

这个特性对你很有用：把 vLLM 启动参数写进脚本，脚本跑完变量自动清理，不会污染后续操作。

## 9.2 使用 env 命令指定单次运行的变量（①）

如果你只想**某一次命令**带特定变量，用 `env` 更干净：

```bash
env MY_VAR=hello python3 my_script.py
# MY_VAR 仅在这次 python3 运行时有效
```

比如临时指定 vLLM 的 GPU 显存比例，只影响这一次启动，不用改全局配置。

## 9.3 使用 .env 文件分组管理变量（④）

vLLM 项目往往需要多个变量（模型路径、端口、日志级别等）。把它们集中到一个 `.env` 文件里：

```bash
# 创建 ~/.env 文件
nano ~/.env
```

内容示例：

```bash
export PROJECT_HOME="$HOME/llm-learning"
export EDITOR="nano"
export TZ="Asia/Shanghai"
```

加载方法：

```bash
set -o allexport
source ~/.env
set +o allexport
```

> **⚠️ 安全警告**：

## 块 54

## 场景 1：创建并验证自定义变量

你正在为 vLLM 推理服务做准备，环境变量是部署时的基础操作——比如设置模型路径、缓存目录等。下面用最直接的方式掌握它。

**第一步：设置临时变量**

```bash
export MY_NAME="zhangsan"
```

`export` 把变量注入当前终端会话，`MY_NAME` 是变量名，`"zhangsan"` 是值。这个变量只在当前终端有效，关掉就没了。

**第二步：验证**

```bash
echo $MY_NAME
```

`$` 加变量名取出值，`echo` 打印。看到 `zhangsan` 就对了。

**第三步：永久生效**

```bash
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc
```

`>>` 把命令追加到 `~/.bashrc` 文件末尾。每次打开新终端，系统会自动加载这个文件，变量就永久存在了。

**第四步：重新加载并验证**

```bash
source ~/.bashrc
echo $MY_NAME
```

`source` 立即执行 `~/.bashrc`，不用重启终端。再次 `echo` 确认值还在。

**本块学习检查**：完成本块后，用自己的话说明本

## 块 55

**本块学习检查**  
完成本块后，用自己的话说明：CUDA环境变量配置与你的vLLM部署目标有何关系？遇到无法解释的术语（如PATH、LD_LIBRARY_PATH）时记录下来再提问。

---

**场景 3：模拟 CUDA 环境变量配置**  
即使你还没装CUDA，也可先练习配置方法，为后续vLLM推理服务铺路：

1. 编辑 `~/.bashrc`：`nano ~/.bashrc`  
2. 末尾添加（假设CUDA装在`/usr/local/cuda`）：  
   `export PATH=/usr/local/cuda/bin:$PATH`  
   `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

## 块 56

### 系统级配置文件：为你的A100公共推理服务器做准备

你在云上部署vLLM时，如果这台A100服务器是多人共用的，就需要了解系统级配置——它影响**所有用户**，修改需要`sudo`权限。日常开发不建议动，但配置公共服务器时是必知项。

#### A.1 /etc/environment：最简写法

这个文件**不需要`export`**，直接写`KEY=value`。它支持变量展开（如`PATH=$PATH:/foo`），但不支持命令替换和通配符。

```bash
sudo nano /etc/environment
# 添加：
JAVA_HOME=/usr/lib/jvm/java-11-openjdk
# 保存后，需重启系统或重新登录才生效
```

#### A.2 /etc/profile：全局Bash配置

这个文件**需要`export`**，适合设置全局PATH。

```bash
sudo nano /etc/profile
# 添加：
export PATH=$PATH:/usr/local/custom/bin
# 保存后，source使当前会话生效
source /etc/profile  # 使当前会话生效
```

**本块学习检查**：完成本块后，用自己的话

## 块 57

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM部署）的关系；遇到无法解释的术语时记录下来再提问。

先确认你的Shell：执行 `echo $SHELL`，若输出 `/bin/bash` 即Bash。

- **Bash**：日常终端用 `~/.bashrc`；SSH登录用 `~/.bash_profile`
- **Zsh**：用 `~/.zshrc`（所有交互式Shell）
- **Fish**：用 `~/.config/fish/config.fish`（所有会话）

**实践**：在A100云实例上执行 `echo $SHELL`，确认Shell类型后，检查对应配置文件是否存在（

## 块 58

## 清理 PATH 中的重复项，为 vLLM 部署扫清障碍

你在 A100 上部署 vLLM 时，会频繁安装 CUDA、conda、vLLM 等工具，每次安装都可能往 `PATH` 里追加路径。多次操作后，`PATH` 里可能出现重复项——这不会直接报错，但会让 `which` 和 `type -a` 的排查结果变得混乱，浪费你排查问题的时间。

**清理重复项**只需一行命令：

```bash
export PATH=$(echo $PATH | awk -v RS=: '!a[$0]++' | paste -sd: -)
```

这条命令把 `PATH` 按冒号拆开，用 `awk` 去重，再用 `paste` 重新拼回冒号分隔的格式。你可以把它加到 `~/.bashrc` 里，每次打开终端自动清理。

**按环境加载不同配置**：如果你在本地测试、在 A100 上跑生产推理，可以用环境变量区分配置：

```bash
# 在 ~/.bashrc 中添加
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

这样 `~/.env.prod` 和 `~/.env.dev` 里可以分别放不同的 `CUDA_VISIBLE_DEVICES`、`VLLM_*` 等变量。

**调试脚本**：如果你写的启动脚本出了问题，用详细模式运行，会打印每一行执行前的变量值：

```bash
bash -x your_script.sh

# 或者在当前 Shell 中开启追踪
set -o xtrace
source ~/.bashrc
set +o xtrace
```

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（vLLM 部署）的关系；遇到无法解释的术语时

## 测评题目

### Q-KB0100-001
请在终端中执行以下命令，验证Linux/WSL是否安装成功，将完整输出粘贴作答：
```bash
uname -a
```

### Q-KB0100-003
执行以下命令确认系统发行版信息：
```bash
cat /etc/os-release
```

### Q-KB0101-001
以下哪个命令用于**切换当前工作目录**？

### Q-KB0101-004
`cp -r source_dir/ dest_dir/` 中 `-r` 参数的含义是？

### Q-KB0102-001
临时设置环境变量 `MY_VAR=hello`，命令为________；若要使其在当前shell及子进程中生效，需使用________命令。

### Q-KB0102-002
请写出将 `/usr/local/cuda/bin` 永久添加到 `PATH` 环境变量的完整步骤（以bash为例）。

### Q-KB0103-003
请写出以下操作的完整命令序列：
1. 创建名为 `llm_train` 的conda环境（Python 3.10）
2. 激活该环境
3. 用pip安装 `torch` 包
4. 查看已安装包列表
5. 导出环境配置到 `environment.yml`

### Q-KB0104-002
`nvidia-smi` 输出中，`Memory-Usage: 2048MiB / 8192MiB` 表示什么？
