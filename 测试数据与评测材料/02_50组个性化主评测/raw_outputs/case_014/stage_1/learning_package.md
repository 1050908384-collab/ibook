# 学习包

## 块 1

**本块学习检查**  
先用最小示例回顾：在终端里依次执行 `cd`（切换目录）、`ls`（查看文件）、`mkdir`（新建文件夹）、`cp`（复制）、`mv`（移动/重命名），确认能解释每个命令的作用后再继续本块。

---

### 1.1 什么是 Linux？我为什么要装它？

你平时用的 Windows 或 macOS 叫“操作系统”，Linux 是另一个操作系统。全世界绝大多数服务器（网站后台、云服务、AI 训练集群）跑的都是 Linux。学编程到后面，你大概率要和它打交道。

本教程用的是 **Ubuntu 24.04 LTS**（长期支持版），厂家会持续提供多年安全更新，稳定且适合入门。如果你已安装 Ubuntu 22.04，后续步骤同样适用。

**为什么和你有关？** 你的目标是 DeepSpeed 分布式训练，双卡 RTX 4090 的实训环境几乎必然跑在 Linux 上。装好 Linux、熟悉终端，是后续执行 Shell 命令、启动训练脚本的第一步。

## 块 2

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

你已有PyTorch和Transformer基础，目标是用DeepSpeed跑双卡RTX 4090的分布式训练。动手前，先解决“代码跑在哪”的问题——三种环境选错，后面会踩坑。

- **虚拟机**：在你电脑里“模拟”出一台新电脑。最像真服务器、完全隔离，但占内存（建议至少4GB），配置低会卡。适合内存≥8GB的用户。
- **WSL2**：Windows自带的Linux子系统，像装App。轻量、启动快、和Windows无缝互通，仅限Win10/11，部分高级功能受限。适合不想折腾的用户。
- **云服务器

## 块 3

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

三个方案最终都给你一台跑着 Ubuntu 24.04 的“电脑”，开发体验一致：

1. **虚拟机/云服务器**：通过 SSH 从 Windows 远程操作（相当于“远程桌面命令行版”）；**WSL2** 则与 Windows 直接互通，用 VS Code 本地开发。
2. VS Code 装好插件后，写代码有高亮、自动补全、能调试。

**动手验证**：确认你的电脑内存 ≥ 8GB（右键“此电脑”→“属性”查看）。若不足，直接跳到方案二或三。

## 块 4

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

**VirtualBox 是什么？** 一款开源虚拟机软件，能在你的 Windows 里凭空造出一台假电脑，这台假电脑可以装任何操作系统。

**下载步骤：**
1. 浏览器打开：https://www.virtualbox.org/wiki/Downloads
2. 找到 **"VirtualBox 7.x.x platform packages"** 下的 **Windows hosts** 版本，点击下载
3. 双击安装包
4. 一路“下一步”保持默认 → 点“安装”等进度条跑完 → 点“完成”

> **坑预警**：弹出“是否允许

## 块 5

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

下载 Ubuntu 24.04 镜像，就是为虚拟机准备一张“系统安装盘”。你已熟悉 PyTorch 和 Transformer，但 DeepSpeed 分布式训练需要 Linux 环境，而虚拟机是零风险起步方式。

**操作步骤：**

1. 浏览器打开 `https://releases.ubuntu.com/24.04/`
2. 找文件名类似 `ubuntu-24.04.x-desktop-amd64.iso` 的文件（x 是小版本号），点击下载
3. 文件约 5GB，下载需几分钟到十几分钟（看网速）
4

## 块 6

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.3 在 VirtualBox 里创建虚拟机

你的目标是跑 DeepSpeed 分布式训练，但先得有 Linux 环境。这一步就是搭好虚拟机，作为后续安装 Ubuntu 和训练环境的基础。

**操作步骤（跟着做）：**

1. 双击桌面 **Oracle VM VirtualBox** 图标，打开软件。
2. 点窗口上方 **“新建”** 按钮。
3. 弹出窗口，按下表填写：
   - **名称**：`Ubuntu-24.04`
   - **文件夹**：选 D 盘或 E 盘（**别放 C 盘**）
   - **类型**：`Linux`
   - **版本**：`Ubuntu (64-bit)`
   - 点“下一步”
4. 内存：

## 块 7

**本块学习检查**：完成安装后，请用自己的话说明：为什么在双卡 RTX 4090 上做 DeepSpeed 分布式训练，需要先有一个能跑 Shell 命令的 Linux 环境？遇到不懂的术语（如“虚拟盘”）先记下来，课后提问。

---

### 2.4 安装 Ubuntu 系统（动手实操）

你的目标是跑 DeepSpeed 分布式训练，所以现在先搭好 Linux 环境。跟着做：

1. 在 VirtualBox 窗口里，选中你的虚拟机，点 **“启动”（Start）**。
2. 等几十秒，出现 Ubuntu 安装界面，按顺序操作：
   - **语言**：选“中文(简体)”或“English” → 点“安装 Ubuntu”。
   - **键盘布局**：默认即可 → 点“继续”。
   - **更新和软件**：选“正常安装”，并**勾选**“安装 Ubuntu 时下载更新”和“安装第三方软件” → 点“继续”。
   - **安装类型**：选“清除整个磁盘并安装 Ubuntu”。注意：这里的“磁盘”只是你虚拟机的 60G 空间，**不会碰你真实的 C 盘/D 盘**

## 块 8

## 2.5 安装 SSH 远程连接工具

**SSH 是什么？** 相当于“远程桌面的命令行版”——装好后，你在 Windows 终端里敲命令，就能直接操控虚拟机里的 Ubuntu，不用来回切换窗口。这对你后续在双卡 RTX 4090 上跑 DeepSpeed 分布式训练至关重要：因为真正的训练任务会在 Linux 环境（虚拟机或服务器）上执行，而 SSH 就是你从 Windows 进入这个环境的“大门”。

### 操作步骤

**第一步：打开 Ubuntu 终端**

在 Ubuntu 桌面空白处点**鼠标右键** → 选**“在终端中打开”**。如果右键没有，按 `Ctrl + Alt + T` 也能打开。

**第二步：安装 SSH 服务**

依次输入以下命令（每行输完按回车）：

```bash
# 第1步：更新软件列表（类似“刷新应用商店的货架”）
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

> **sudo 是什么？** `sudo` = “以管理员身份执行”。Linux 里普通用户不能随便装软件，加了 sudo 就相当于“给我最高权限”。输入 sudo 命令后会让你输密码，输的时候**屏幕不会有任何显示**（不会出现 `***`），这是正常的安全设计，盲打完按回车就行。

**第三步：查看虚拟机的 IP 地址**

```bash
# 显示网络信息，找到你的 IP 地址
ifconfig
```

输出内容很多，找到类似这一行：`inet 192.168.xxx.xxx`，那一串数字就是 IP 地址。**记下来！** 例如：`192.168.126.128`

**第四步：验证 SSH 是否可用**

回到 Windows，按 `Win + R`，输入 `cmd` 回车，打开命令提示符，输入：

```bash
# 用 SSH 连接你的虚拟机。把 username 和 IP 换成你自己的
ssh username@192.168.xxx

## 块 9

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

### 2.6 安装增强功能（让虚拟机和 Windows 更好配合）

这一步让你可以在 Windows 和虚拟机之间**直接复制粘贴文字和文件**，非常实用。

新版 VirtualBox 通常会自动安装增强功能。如果未能自动生效，可手动安装：

```bash
# 安装 VirtualBox 增强工具包
sudo apt install -y virtualbox-guest-utils virtualbox-guest-x11

# 重启虚拟机
sudo reboot
```

重启后，你就可以在 Windows 里 `Ctrl+C` 复制文字，到 Ubuntu 里 `Ctrl+V` 粘贴了！

**方案一完成！** 你现在有了一台完整的 Ubuntu 虚拟机。

---

**练习提示**：既然你目标是双卡 RTX 4090 上的 DeepSpeed 分布式训练，请先确认

## 块 10

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

**3.1 确认虚拟化已开启**

你要跑DeepSpeed双卡训练，WSL2是Windows下最顺手的Linux环境。但WSL2依赖CPU的虚拟化功能，这步没开，后面全卡住。

**查状态**：按`Ctrl + Shift + Esc`打开任务管理器 → 点“性能”标签 → 右下角看“虚拟化”是“已启用”还是“已禁用”。

**已启用** → 直接跳到下一步。

**已禁用** → 进BIOS开启：
1. 重启，开机狂按`F2`或`Del`（联想F2，惠普F10，戴尔F2）
2. 进BIOS后，用方向键找“Configuration”或“Advanced”菜单
3. 找“Intel Virtualization Technology”或“AMD SVM”，改成“Enabled”
4. 按`F10`保存退出
5. 回Windows再查

## 块 11

## 安装 WSL2：为 DeepSpeed 分布式训练铺路

你已经有 PyTorch 和 Transformer 基础，接下来要跑 DeepSpeed 分布式训练。双卡 RTX 4090 很给力，但分布式训练需要 Linux 环境——WSL2 就是 Windows 上最轻量的方案，几分钟搞定。

**操作步骤：**

1. 按 `Win` 键，输入 `powershell`，在搜索结果中**右键** "Windows PowerShell" → **"以管理员身份运行"**
2. 弹出"是否允许此应用更改你的设备" → 点"是"
3. 在 PowerShell 窗口中，**逐行**输入以下命令（复制一整行，右键粘贴，回车）：

```bash
# 第1条：启用"适用于 Linux 的 Windows 子系统"功能
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 第2条：启用"虚拟机平台"功能
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

4. 两条命令都显示"操作成功完成"后，**重启电脑**
5. 重启后，再以管理员身份打开 PowerShell，输入：

```bash
# 将 WSL 默认版本设置为 2
wsl --set-default-version 2
```

**本块学习检查**：完成本块后，用自己的

## 块 12

**本块学习检查**：完成安装后，请用自己的话说明：为什么在 Windows 上装 Ubuntu 24.04 是通往 DeepSpeed 分布式训练的第一步？遇到不懂的术语（如 LTS、WSL）先记下来，课后提问。

---

**安装 Ubuntu 24.04（推荐方法一）**

1. 按 `Win` 键，输入 `Microsoft Store`，打开微软应用商店。
2. 搜索框输入 `Ubuntu 24.04`。
3. 找到 **"Ubuntu 24.04.x LTS"**（发布者：Canonical Group Limited），点“安装”。
4. 下载约 500MB，完成后点“打开”。

**方法二（命令行，更快）**

```bash
wsl --install -d Ubuntu-24.04
```

5. 首次启动显示 "Installing, this may take a few minutes..."，等 2~5 分钟。
6. 设置用户名和密码：
   - 用户名：全英文小写（如 `zhangsan`）→ 回车
   - 密码：输入时不显示任何字符，**这是正常的** → 回车
   - 再

## 块 13

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

在 Ubuntu 终端（绿色提示符窗口）逐行执行：

```bash
sudo apt update
sudo apt upgrade -y
```

`update` 刷新软件包列表，`upgrade -y` 升级所有已装软件（`-y` 自动确认）。WSL2 无需 SSH，用 VS Code 的 Remote-WSL 插件即可开发。

验证系统：

```bash
lsb_release -a   # 应显示 Ubuntu 24.04.x
whoami           # 显示你设的用户名
```

测试与 Windows 互通：

```bash
explorer.exe .   # 弹出资源管理器即无缝衔接
```

**方案二完成！** 系统就绪后，下一步即可安装 PyTorch 与 DeepSpeed，为双卡 RTX 4090 的分布式训练做准备。

## 块 14

**本块学习检查**：完成购买后，请用自己的话说明：为什么云服务器是搭建 DeepSpeed 分布式训练环境的一种可行方案？遇到不懂的术语（如“镜像”）先记下来，稍后提问。

---

以阿里云为例（腾讯云、华为云流程类似）：

1. 打开 https://www.aliyun.com/，用支付宝或淘宝账号注册/登录。
2. 顶部搜索“轻量应用服务器”进入产品页；学生可先搜“学生优惠”完成认证，价格更低。
3. 点“立即购买”，按以下配置选择：
   - **地域**：选离你最近的（如华东1（杭州）、华南1（深圳））
   - **镜像类型**：选“系统镜像

## 块 15

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

你的目标是 DeepSpeed 分布式训练，双卡 4090 已就绪。但先别急着跑训练——安全第一步，是给服务器建一个“普通用户”，别用 root 直接干活。

root 权限太高，误操作可能删系统文件。所以，通过云厂商控制台的“远程连接”或 VNC（网页终端）登录后，执行：

```bash
# 创建新用户（yourname 换成你想要的，全英文小写）
adduser yourname
# 提示设密码，输两次；其他信息直接回车跳过

# 给该用户 sudo

## 块 16

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

### 4.3 配置 SSH 密钥登录（安全第二步）

你接下来要在双卡 RTX 4090 上跑 DeepSpeed 分布式训练，这通常需要远程连接服务器。密码登录容易被暴力破解，SSH 密钥既安全又免密，是训练前的必备安全配置。

**在 Windows 上生成密钥对：**

1. 按 `Win + R`，输入 `cmd`，打开命令提示符。
2. 执行命令生成密钥对（把邮箱换成你自己的）：

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

3. 提示 `Enter file in which to save the key` → 直接按**回车**（用默认路径）。
4. 提示 `Enter passphrase` → 直接按**回车**（不设密码短语）。
5. 提示 `Enter same passphrase again` → 再按一次**回车**。
6. 看到 randomart image 即生成成功。

**把公钥放到服务器上：**

7. 通过云厂商的 VNC/远程连接登录服务器，创建 `authorized_keys` 文件：

```bash
su - yourname          # 切换到你的用户
mkdir -p ~/.ssh        #

## 块 17

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

**配置防火墙（安全第三步）**

现在才开放22端口，并建议限制访问来源：

1. 进入云厂商的**轻量应用服务器控制台**
2. 点**“防火墙”**标签页
3. 点“添加规则”：
   - 端口：22
   - 协议：TCP
   - 来源：**建议填写你自己的公网IP**（浏览器搜“IP”查看），格式如`123.45.67.89/32`。IP不固定可暂填`0.0.0.0/

## 块 18

## 从 Windows 连接你的云服务器

你的目标是跑 DeepSpeed 分布式训练，双卡 RTX 4090 的算力需要一台真正能跑起来的机器。现在你已经有了阿里云轻量服务器，接下来就是打通从 Windows 到它的通道。

**操作步骤：**

1. 在阿里云控制台找到你的 **公网 IP**（形如 `47.96.xxx.xxx`），复制它。
2. 按 `Win + R`，输入 `cmd` 回车，打开命令提示符。
3. 输入命令连接（把 `yourname` 换成你的用户名，IP 换成你的）：

```bash
ssh yourname@47.96.xxx.xxx
```

4. 首次连接会询问

## 块 19

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

你在虚拟机或云服务器上，若4.3节已配好密钥，现在验证：在Windows命令提示符执行 `ssh 用户名@你的IP地址`，能直接连上不提示密码即成功。

**常见坑**：若提示 `ssh-copy-id 不是内部或外部命令`，说明Windows版本较老。方法一（推荐）：安装Git for Windows（https://git-scm.com/download/win），装完Git Bash自带ssh-copy-id；方法二：手动复制（见4.3节）。

**练习**：立即执行

## 块 20

## 5.2 WSL2：无需 SSH，直接集成（①）

你走 WSL2 路线，**不用配 SSH**。VS Code 通过 Remote-WSL 插件直接和 WSL2 通信，体验更流畅，省掉一层网络配置。

## 5.3 安装 VS Code 及远程开发插件（①）

**VS Code**（Visual Studio Code）是微软的免费代码编辑器，轻量、插件多、跨平台。你后面写 PyTorch 训练脚本、调 DeepSpeed 配置，都在这个编辑器里干。

### 第一步：安装 VS Code（①）

1. 浏览器打开：https://code.visualstudio.com/
2. 点 **"Download for Windows"** 按钮
3. 下载完双击安装包
4. 安装过程：
   - 同意协议 → 下一步
   - 安装路径默认 → 下一步
   - **重点！** 这页勾选：
     - ☑ **"将 Code 添加到"PATH"中"**（重要！这样命令行直接输 `code` 就能打开 VS Code）
     - ☑ **"将 Code 注册为受支持的编辑器"**
     - ☑ **"添加到文件资源管理器上下文菜单"**
   - 一路"下一步" → "安装" → 完成后勾上"启动 VS Code" → "完成"

### 第二步：安装必备插件（①）

**插件（Extension）** 是 VS Code 的"外挂功能包"。编辑器本身是个壳子，装了插件才能高亮代码、自动补全、调试。

1. 打开 VS Code
2. 点左侧竖排图标栏**第五个图标**（四个小方块），或按 `Ctrl + Shift + X`
3. 在**扩展商店**搜索框里依次安装：

| 搜索关键词 | 插件全名 | 发布者 | 作用 |
|---|---

## 块 21

## 5.4 连接你的 Linux 环境——为 DeepSpeed 双卡训练铺路

你的目标是跑 DeepSpeed 分布式训练，而训练代码要跑在 Linux 上。VS Code 远程连接就是你操作 Linux 的“遥控器”——本地写代码，远程执行训练。

**虚拟机或云服务器：**

1. 打开 VS Code，看左下角绿色 `><` 图标
2. 点击 → 选 **"Connect to Host…"**
3. 输入 `用户名@IP地址`（如 `zhangsan@192.168.126.128`）→ 回车
4. 选平台 **"Linux"**；有密钥直接连，没有就输密码
5. 左下角变成 `SSH: IP地址` 即成功
6. **"File"** → **"Open Folder…"**，选服务器目录（如 `/home/你的用户名`）

**WSL2：**

1. 点绿色图标 → **"Connect to WSL"**，自动连上 Ubuntu
2. 左下角变 `WSL: Ubuntu-24.04`，打开文件夹即可
3. 更快捷：Ubuntu 终端里 `cd` 到项目目录，输 `code .` 一步到位

> 连上后，你就能像编辑本地文件一样改远程代码——这是后续配置 DeepSpeed 训练脚本的基础操作。

---

## 六、终端基本命令

终端是黑框框，等你输命令。若提示 `command not found`（如 `vim`、

## 块 22

## 6.1 文件与目录操作：最基础也最常用

你马上要开始 DeepSpeed 分布式训练，而训练脚本、数据集、日志文件都存放在 Linux 文件系统中。这一节掌握的命令，是你后续组织训练目录、管理模型文件的“地基”。

### 核心命令速览

**导航三件套**

- `pwd`：显示当前所在目录的绝对路径。比如在 Documents 目录下执行，输出 `/home/user/Documents`。
- `ls`：列出目录内容。`ls` 列出当前目录；`ls -l`（简写 `ll`）详细列出含权限和大小；`ls -a` 显示隐藏文件。
- `cd`：切换目录。`cd ..` 进入上级目录；`cd /` 进入根目录；`cd ~` 回到当前用户主目录；`cd /home/user/Documents` 进入指定目录。

**创建与删除**

- `mkdir`：创建目录。`mkdir test` 创建单个目录；`mkdir -p test/aaa/bbb` 递归创建多级目录。
- `touch`：创建空文件。`touch file1.txt file2.log` 一次创建两个空文件。
- `rm`：删除文件或目录（⚠️ 谨慎！）。`rm file1.txt` 删除文件；`rm -f file1.txt` 强制删除不提示；`rm -rf test/` 强制递归删除目录——**不可恢复，避免误删系统文件**。

**复制与移动**

- `cp`：复制。复制文件用 `cp file1.txt /home/user/Desktop`；复制目录必须加 `-r` 递归：`cp -r test/ /home/user/Desktop`。
- `mv`：移动或重命名。`mv file1.txt test/` 移动文件；`mv file1.txt newfile.txt` 重命名；`mv test/ /home/user/newtest` 移动并改名。

**查看文件**

- `cat`：查看小文件内容。`cat -n file1.txt` 显示行号。
- `more/less`：分页查看大文件。`less app.log` 打开后空格翻页、回车换行、`q` 退出。

### 实操建议

## 块 23

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

**权限管理：Linux 安全核心**

Linux 每个文件/目录都有**三类角色**：所有者、所属组、其他用户；每类角色有**三种权限**：读（r=4）、写（w=2）、执行（x=1）。

**chmod：修改权限**

- **数字法（推荐）**：`chmod 744 file1.txt` → 所有者读+写+执行（7=4+2+1），其他只读（4）
- **符号法**：`chmod g+w file1.txt` 给所属组加写；`chmod a+x test.sh` 给所有用户加执行

**chown：修改所有者**（需 root 权限）

- `sudo chown root file1.txt` → 将所有者改为 root

**sudo 回顾**：`sudo` 表示"以管理员身份执行"。输密码时屏幕无显示，是正常安全设计。

---

**实践任务**（结合双卡 RTX 4090 实训）：

1. 在

## 块 24

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

你的目标是双卡RTX 4090跑DeepSpeed，训练前先学会“看状态”。本块5个命令，都是查系统“体检报告”的：

- **`uname -a`**：看内核和架构。确认系统是x86_64，驱动和CUDA才装得对。
- **`top`**：实时看CPU/内存占用。训练时按`P`看哪个进程吃CPU，按`M`看内存，`q`退出。
- **`ps -ef | grep python`**：查进程。启动DeepSpeed后，用它确认两个GPU各自起了训练进程。
- **`df -h`

## 块 25

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

你马上要跑DeepSpeed双卡训练，改配置文件（如`deepspeed_config.json`）是家常便饭。本块只讲两个最实用的文件修改工具。

**1. vim——终端里的编辑器**

改配置前先打开它：`vim deepspeed_config.json`

- 按`i`进入编辑模式（左下角出现`--INSERT--`），此时可正常打字。
- 改完按`Esc`退出编辑，输入`:wq`回车——保存并退出。
- 改乱了想放弃？按`Esc`，输入`:q!`回车——不保存直接退出。
- 想快速定位某个参数（比如`"train_b

## 块 26

## 网络操作：基础网络查询

在进入 DeepSpeed 分布式训练之前，你需要先掌握两个最基础的网络命令——它们是后续多卡通信、节点互联排障的起点。

**1. `ping` — 测试网络连通性**

```bash
ping www.baidu.com
```

执行后持续显示响应，按 `Ctrl+C` 停止。如果能看到回复，说明当前机器能正常访问外网。分布式训练中，常用它快速确认各节点之间是否互通。

**2. `ifconfig` — 查看网卡信息**

```bash
ifconfig
```

输出中 `eth0`、`wlan0` 等网卡下的 `inet` 字段就是该机器的 IP 地址。训练时你需要知道本机 IP，以便配置 DeepSpeed 的通信地址。若提示 `command not found`，说明

## 块 27

## 命令实战场景演练

结合你的 DeepSpeed 分布式训练目标，终端命令是操作双卡 RTX 4090 的基础。下面两个场景请动手操作一遍。

### 场景 1：创建并整理文件

训练脚本、日志、配置文件的整理是日常操作。按顺序执行：

```bash
# 1. 进入主目录
cd ~

# 2. 创建 work 及子文件夹
mkdir -p work/doc work/code

# 3. 创建并写入文件
echo "Linux命令学习笔记" > note.txt

# 4. 移动文件
mv note.txt work/doc/

# 5. 查看结果
ls work/doc/
# 预期输出：note.txt
```

### 场景 2：查询系统资源并结束进程

训练时需监控内存和进程状态。执行：

```bash
# 1. 查看内存
free -h

# 2. 查找 python 进程
ps -ef | grep python
# 获取进程 ID，即输出中第二列的 PID 字段

# 3.

## 块 28

### 6.7 新手避坑指南（针对你的双卡RTX 4090实训）

搞分布式训练前，先把这些坑填平，否则后面会卡到你怀疑人生：

1. **大小写敏感**：Linux 命令区分大小写。`ls` 能列出文件，`LS` 直接报错。敲命令时别开大写锁定。

2. **路径写错**：切换目录或复制文件时，报 `No such file or directory` 就是路径不对。先 `ls` 看看目标路径在不在，再操作。你的数据、代码、模型权重，路径规划好，省得后面 DeepSpeed 加载时找不到文件。

3. **rm 慎用**：`rm -rf` 是删库跑路级命令。删任何东西前，先 `ls` 确认目录/文件名，尤其别手滑删到 `/`（根目录）。你的 48GB 显存机器上如果存了预训练权重，误删就哭吧。

4. **权限不足**：操作系统文件或别人文件

## 块 29

## 7.1 检查并安装 Python 与 pip（①）

你的目标是 DeepSpeed 分布式训练，而 DeepSpeed 运行在 Python 生态之上。所以第一步，先确认这台双卡 RTX 4090 机器上的 Python 环境是否就绪。

Ubuntu 24.04 默认自带 Python 3.12，直接验证：

```bash
# 查看 Python 版本
python3 --version
# 预期输出：Python 3.12.x

# 查看 pip 是否已安装
pip3 --version
```

如果 `pip3` 提示未找到，则安装它：

```bash
sudo apt update
sudo apt install -y python3-pip
```

> **pip 是什么？** pip 是 Python 的包管理工具，用于安装和管理 Python 软件包。

**本块

## 块 30

## 7.2 （可选但推荐）安装 Miniconda（①）

你的目标是 DeepSpeed 分布式训练，后续会频繁创建独立环境来隔离不同项目的依赖。Miniconda 就是为此准备的轻量工具——系统自带 Python 虽够用，但 conda 虚拟环境管理会让多项目切换更干净。这里先做最简安装，详细配置后续再讲。

**安装步骤：**

1. 下载安装脚本：
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
（若提示 wget 未找到，先执行 `sudo apt install wget`）

2. 校验下载文件的完整性（对比官方提供的哈希值）：
```bash
sha256sum Miniconda3-latest-Linux-x86_64.sh
```
将输出结果与官网公布的 SHA-256 校验和进行比对，确认一致后再继续。

3. 运行安装：
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

4. 安装过程交互：
   - `Enter` 阅读许可 → 输入 `yes` 表示接受 Miniconda 的许可协议
   - 安装路径直接回车（

## 块 31

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

### 7.3 运行你的第一个 Python 程序

你的目标是 DeepSpeed 分布式训练，而 DeepSpeed 跑在 Python 里。所以，先确认这台 Linux 机器能运行 Python，是后续一切的基础。

**第一步：创建文件**

在终端里执行：

```bash
cd ~
nano hello.py
```

`cd ~` 回到主目录，`nano hello.py` 用 nano 编辑器打开（或创建）一个叫 `hello.py` 的文件。

**第二步：输入代码**

在 nano 里输入：

```python
print("Hello, Python on Ubuntu 24.04!")
```

**第三步：保存并退出**

1. 按 `Ctrl + O` 保存，回车确认文件名。
2. 按 `Ctrl + X` 退出 nano。

**第四步：运行程序**

```bash
python3 hello.py
```

屏幕输出 `Hello, Python on Ubuntu 24.04!` 即成功。

---

**这一步的意义**：你已成功使用 `cd`、`nano`、`python3` 三个终端命令，验证了 Python 环境可用。后续装 PyTorch、跑 DeepSpeed 脚本，都依赖这个基础链路。现在环境已就绪，可以放心进入下一节。

---

**常见问题速查**（遇到再查）：

| 症状 | 可能原因 | 解决办法 |
|---|---|---|
| `sudo apt update` 很慢 | 默认软件源在国外 | 搜"Ubuntu 24.04 换阿里云源"，按教程换源 |
| 终端中文乱码 | 字符编码问题 | 输入 `export LANG=en_US.UTF-8` 回车 |
| `command not found: ifconfig` | net-tools 未安装 | 执行 `sudo apt install net-tools` |

---

**本块学习检查**：

## 块 32

## 二、你在哪？pwd 与路径的概念（新增②）

### 2.1 pwd：打印当前工作目录（新增②）

在开始分布式训练之前，你需要先搞清楚一件事：**你现在站在哪个目录下**。DeepSpeed 的配置文件和训练脚本都依赖正确的路径定位，路径错了，模型权重和日志就会存到意想不到的地方。

`pwd` = **P**rint **W**orking **D**irectory，作用是告诉你"我现在在哪个文件夹里"。（②）

```bash
pwd
```

示例输出：
```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。在操作文件之前，养成先敲 `pwd` 的习惯，能避免很多"文件找不到了"的困惑。（②）

**为什么这对你很重要？** 当你用 DeepSpeed 启动双卡训练时，通常需要指定 `--deepspeed_config` 的路径。如果当前目录不对，你写的相对路径就会指向错误的位置，导致训练直接报错。所以，**每次打开终端，第一件事就是 `pwd`**，确认自己在哪里。

---

**本块学习检查**

完成本块后，请用自己的话回答：`pwd` 命令和你的 DeepSpeed 分布式训练目标有什么关系？比如，在启动训练前，为什么需要先确认当前目录？如果你在解释过程中遇到无法理解的术语（比如"工作目录"），请记录下来，在后续提问时一并提出。

## 块 33

## 2.2 绝对路径 vs 相对路径

搞分布式训练，你经常要在多个目录间切换配置文件、日志和 checkpoint。路径搞错，训练脚本直接报错。所以这块必须吃透。

**绝对路径**：从根目录 `/` 开始写，比如 `/home/zhangsan/Documents`。不管你在哪，它都指向同一个地方，最稳。

**相对路径**：相对于你“当前所在位置”来写。比如 `./file.txt` 表示当前目录下的文件，`../backup` 表示上一级目录里的 backup。

记住三个符号：
- `./` = 当前目录（通常可省略）
- `../` = 上一级目录
- `~` = 当前用户家目录（如 `/home/zhangsan`）

**举例**：假设你在 `/home/zhangsan/Documents`：
- `cd /home/zhangsan` → 绝对路径，从哪执行都行
- `cd ..` → 相对

## 块 34

## 3.1 ls：列出目录内容（基础用法）

在 DeepSpeed 分布式训练中，你经常需要确认模型权重、日志文件、配置文件是否就位。`ls` 就是干这个的。

**最常用组合：**

```bash
# 列出当前目录内容（不含隐藏文件）
ls

# 列出指定路径，比如你的训练脚本目录
ls /home/zhangsan

# 查看所有文件，包括隐藏文件（以 . 开头）
ls -a

# 详细列表：权限、所有者、大小、修改时间
ls -l

# 组合：详细列出所有文件，大小显示为人类可读格式（K/M/G）
ls -ahl
```

**隐藏文件为什么重要？** 以 `.` 开头的文件默认隐藏，如 `.bashrc`、`.ssh`。配置分布式训练环境（如 SSH 免密登录多机节点）时，你会频繁和它们打交道。

## 3.2 读懂 ls -l 的输出

执行 `ls -l` 后，输出类似：

```bash
drwxr

## 块 35

### 3.3 其他常用选项（①）

`ls` 的选项能帮你快速定位训练脚本或日志文件。重点掌握这几个：

- **`ls -lh`**：详细列表 + 人类可读大小（KB/MB/GB）。注意 `-h` 必须搭配 `-l` 才有效。
- **`ls -t`**：按修改时间排序（最新在前）；**`ls -S`**：按大小排序（大到小）。
- **`ls -rt`**：逆转排序，常配合 `-t` 看最旧文件。
- **`ls -i`**：显示 inode 编号；**`ls -F`**：在文件夹后加 `/`，软链接

## 块 36

## 4.1 cd 基础用法——为 DeepSpeed 训练铺路

你在双卡 RTX 4090 上跑分布式训练时，**每一步都离不开路径切换**。DeepSpeed 启动脚本、数据集挂载、日志输出，全都依赖你准确到达目标目录。本块掌握 `cd`，是后续操作的地基。

```bash
# 进入指定目录（绝对路径，从根目录 / 开始写全）
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

### 4.2 路径切换示例

假设当前在 `/usr/local` 下，想切换到 `/home/zhangsan`：

```bash
# 方法1：绝对路径（去哪都管用，推荐新手）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

> **建议**：新手优先使用绝对路径，不容易迷路；熟悉后多用相对路径，效率更高。

---

## 五、mkdir：创建目录

`mkdir` = **M**a**k**e **Dir**ectory，用来创建文件夹。训练项目需要清晰的目录结构来存放代码、模型和日志。

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

> **注意**：`mkdir` 默认**不能**直接创建多级嵌套目录，必须加 `-p` 参数。这是新手最常踩的坑之一。

---

## 六、touch：创建空文件

`touch` 原本用来"更新文件的时间戳"，但最常用的功能是**快速创建一个空文件**。比如创建 DeepSpeed 配置文件或 Python

## 块 37

## 7.1 复制文件（①+②）

在分布式训练中，你经常需要把训练脚本、配置文件复制到不同目录（比如备份或整理实验版本）。`cp` 就是干这个的。

```bash
# 复制 hello.py 到 backup 文件夹（文件名不变）
cp hello.py ./backup/

# 复制并改名
cp hello.py ./backup/hello_backup.py

# 一次复制多个文件
cp file1.txt file2.txt ./backup/

# 保留文件属性（如修改时间）——对实验版本管理很有用
cp -p hello.py ./backup/

# 覆盖前提示，防止误覆盖已有文件
cp -i hello.py ./backup/
```

## 7.2 复制目录（①+②）

复制文件夹时**必须加 `-r`**（递归），否则系统拒绝。这就像复制整个项目目录，需要把里面所有子文件一起带上。

```bash
cp -r myproject ./backup/
```

注意区别：如果 `backup` 不存在，会新建它并把 `myproject` 内容放进去；如果 `backup` 已存在，会生成 `backup/myproject/` 整个文件夹。

## 7.3 复制软链接（①）

```bash
# 保持链接

## 块 38

## 8.1 移动文件/文件夹（①）

在分布式训练项目中，你经常需要整理模型脚本、数据集和日志。`mv` 就是你的“文件搬运工”：

```bash
# 把文件移动到另一个文件夹
mv hello.py ./backup/

# 把文件夹移动到另一个位置
mv myproject ./archive/

# 使用通配符移动多个文件（把所有以 data_ 开头的文件移到 data 文件夹）
mv data_* ./data/
```

> 💡 **训练场景联想**：当你用 DeepSpeed 跑完一轮实验，可以用 `mv` 把 `data_` 开头的多个 checkpoint 文件统一归档到 `./data/`，保持工作目录干净。

## 8.2 重命名（①）

```bash
# 把 a.txt 重命名为 b.txt（本质是在同一目录下移动）
mv a.txt b.txt

# 移动的同时改名
mv hello.py ./backup/hello_v2.py
```

> 💡 **训练场景联想**：保存不同版本的训练脚本时，`mv train.py train_v2.py` 比复制再删除更高效。

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

# 删除

## 块 39

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

你在双卡RTX 4090上跑分布式训练，第一步不是写代码，而是把项目目录搭好。本块就用最少的命令，建出一个`llm-learning`骨架。

**操作步骤**（跟着敲）：

1. `cd ~` 回到家目录
2. `mkdir llm-learning` 建根目录
3. `cd llm-learning` 进入
4. `mkdir data models scripts` 一次建三个子目录
5. `touch scripts/train.py` 在`scripts`下建空训练脚本
6. `ls -R`

## 块 40

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

**场景 2：复制与整理实验文件**

你的 `~/Downloads/` 里有 `train.csv`、`test.csv`、`dev.csv`，把它们复制到 `~/llm-learning/data/` 并验证。

```bash
# 1. 确认文件存在
ls ~/Downloads/*.csv

# 2. 复制到 data 目录
cp ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/dev.csv ~/llm-learning/data/

# 3. 验证
ls ~/llm-learning/data/

## 块 41

## 场景 3：清理临时文件——为分布式训练腾出干净环境

在准备 DeepSpeed 分布式训练时，你的 `~/llm-learning/` 目录下误建了一个空文件夹 `temp` 和一个 `old.txt` 文件。这些临时文件会干扰后续训练脚本的路径查找，现在将它们清理掉。

**操作步骤：**

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

**关键点**：删除前用 `ls` 确认目标，删除后再次 `ls` 验证结果——这在管理训练脚本和模型权重文件时尤为重要，避免误删。

---

## 新手避坑指南（重点标注与你相关的部分）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。

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

**5. rm 命令慎用**：`rm -rf` 没有回收站，删除即永久消失。删除前务必用 `ls` 确认目标。

**6. 复制目录忘加 `-r`**：新手常犯 `cp mydir ./backup/` 然后报错。记住：**文件夹复制必须加 `-r`**。

**7. 命令记不住？**：输入命令前按 `Tab` 键

## 块 42

## 环境变量是什么？——从“备忘录”到分布式训练的桥梁

你已经掌握了 PyTorch、Transformer 和 HuggingFace 工具链，现在要挑战 DeepSpeed 分布式训练。在动手之前，先花几分钟理解一个基础但关键的概念：**环境变量**。

### 1.1 通俗理解：操作系统的“备忘录”

环境变量是操作系统或应用程序运行时使用的**动态键值对**，用于传递配置信息。系统启动时会加载一部分环境变量，应用程序运行时也可以动态设置和读取。你可以把它理解为操作系统的“备忘录”——系统启动时加载这些备忘录，所有程序都能读到里面的内容，并据此调整自己的行为。

**常见例子**：

| 变量名 | 作用 |
|---|---|
| `PATH` | 告诉系统去哪里找可执行程序（比如你输入 `python3`，系统就是靠 PATH 找到它的位置） |
| `HOME` | 当前用户的家目录路径（如 `/home/zhangsan`） |
| `USER` | 当前登录的用户名 |
| `SHELL` | 当前使用的 Shell 程序路径（如 `/bin/bash`） |
| `LANG` | 系统语言和字符编码（如 `en_US.UTF-8`） |
| `PWD` | 当前工作目录路径（随 `cd` 命令自动更新） |
| `EDITOR` | 默认文本编辑器（影响 `git commit`、`crontab -e` 等需要编辑器的操作） |

### 1.2 为什么需要环境变量？

1. **控制程序行为**：例如通过 `JAVA_HOME` 指定 JDK 路径，让依赖 Java 的程序知道去哪找它。
2. **简化配置**：避免在每次运行程序时都手动输入长长的路径。
3. **跨进程通信**：子进程会自动继承父进程的环境变量，实现配置的无缝传递。

### 与你的目标有什么关系？

当你运行 `deepspeed` 命令启动分布式训练时，系统需要通过 `PATH

## 块 43

**临时变量：当前终端会话有效**

临时变量只活在当前终端里——你关掉窗口或断开 SSH，它就没了。适合快速测试，不用改配置文件。

```bash
export MY_VAR="hello_linux"   # 设置
echo $MY_VAR                  # 查看，输出 hello_linux
unset MY_VAR                  # 删除
echo $MY_VAR                  # 输出空
```

**`export` 关键点**：不加 `export`，变量只对当前 Shell 可见，子进程读不到；加了，子进程能继承。设环境变量时习惯加 `export`。

**本块学习检查**：完成本块后，用自己的话说明临时变量与 DeepSpeed 分布式训练的关系（比如

## 块 44

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

永久变量要写入**配置文件**，每次开终端都生效。选哪个？日常开发优先 `~/.bashrc`（用户级），只影响你，改错不伤系统；系统级 `/etc/profile` 要 `sudo` 且影响所有用户，别动。

**动手做**（双卡4090实训）：打开终端，执行 `echo $PATH` 看当前值，再 `nano ~/.bashrc` 找到 `export PATH=...` 行，观察格式。这为后续配置DeepSpeed的CUDA路径打基础——写错文件只影响你，可放心试。

## 块 45

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

### 3.1 用户级配置：修改 ~/.bashrc（推荐）

`~/.bashrc` 是 Bash 的配置文件，**每次打开新交互式终端时自动加载**，是日常开发最常用的配置位置。

**操作步骤**：

```bash
# 1. 用 nano 打开配置文件
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

> **`source` 是什么？** 相当于"重新执行一遍这个配置文件"，让修改立刻生效，不用关掉重开。

> **`~/.bashrc` 的加载限制**：它通常在打开新的**交互式终端**时加载，但并非所有 SSH 会话、IDE 集成终端或服务进程都会自动加载。某些 IDE

## 块 46

## 用户级配置与优先级：为DeepSpeed铺路

你正在为双卡RTX 4090配置DeepSpeed分布式训练环境。用户级配置（`~/.bash_profile` 或 `~/.profile`）在**登录Shell**时生效——比如SSH远程登录服务器时。注意：若 `~/.bash_profile` 和 `~/.profile` 同时存在，bash 登录Shell会优先读取 `~/.bash_profile`，而忽略 `~/.profile`。

日常开发中 `~/.bashrc` 更常用，因为每个新终端都会加载它。只有需要"登录时才生效"的场景才用这两个文件——例如你通过SSH连接训练服务器时，登录即加载的环境变量可在此配置。

**优先级**：在登录Shell中，系统先加载 `/etc/profile`，再加载 `~/.bash_profile`；在交互式非登录Shell中加载 `~/.bashrc`。`export` 命令设置的变量会覆盖之前加载的配置文件中的同名变量。

**本块学习检查**：完成本块后

## 块 47

**PATH：命令的“搜索路径”**

你马上要跑 DeepSpeed 分布式训练，终端里敲的每个命令（`deepspeed`、`torchrun`、`python`）能不能被找到，全靠 PATH。它就是一个目录列表，系统从左到右依次在这些目录里找可执行文件，找到就执行，找不到就报 `command not found`。

**查看 PATH**

```bash
echo $PATH
```

输出类似（冒号 `:` 分隔多个路径）：

```bash
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

**查看命令具体位置**

```bash
which python3
# 输出：/usr/bin/python3

which conda
# 输出：/home/zhangsan/miniconda3/bin/conda
```

`which` 告诉你某个命令到底来自 PATH 的哪个目录。排查“明明装了却找不到”时，先用它。

**修改 PATH**

追加到末尾（优先级低）：

```bash
export PATH=$PATH:/new/path
```

前置到开头（优先级高，系统先搜这里）：

```bash
export PATH=/new/path:$PATH
```

> **安全提示**：不要把 `.`（当前目录）加入 PATH。若某目录里有恶意脚本叫 `ls`，而当前目录又在 PATH 里，执行 `ls` 可能运行恶意脚本。
>
> **避免重复添加**：多次 `source ~/.bashrc` 或重复写入同一行，会导致 PATH 里出现重复路径。定期检查 `~/.bashrc`，确保每条修改只出现一次。

**本块学习检查**

完成本块后，用自己的话说明：PATH 的作用是什么？它和你即将配置的 DeepSpeed 分布式训练环境有什么关系（比如 `deepspeed` 命令为什么能直接敲）？遇到无法解释的术语（如 `export

## 块 48

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

`LD_LIBRARY_PATH` 告诉系统在**运行时**去哪里找额外的动态库（`.so` 文件）。你的目标是 DeepSpeed 分布式训练，而 CUDA、PyTorch 的库文件常不在默认路径里，配置它能让程序顺利找到这些库。

先看当前值：
```bash
echo $LD_LIBRARY_PATH
```

临时添加 CUDA 库路径（仅当前终端有效）：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

**典型场景**：你编译的程序依赖某个 `.so`

## 块 49

## 5.3 LIBRARY_PATH：编译时库文件路径

`LIBRARY_PATH` 告诉编译器（如 `gcc`）在**编译链接阶段**去哪里找库文件，包括静态库（`.a`）和动态库（`.so`）。它和另外两个变量分工不同：

| 变量名 | 作用阶段 | 查找目标 | 典型用途 |
|---|---|---|---|
| `PATH` | 执行命令时 | 可执行文件 | 让命令在任何位置都能运行 |
| `LD_LIBRARY_PATH` | 程序运行时 | 动态链接库（`.so`） | 运行依赖自定义动态库的程序 |
| `LIBRARY_PATH` | 编译链接时 | 静态库/动态库 | 编译时链接第三方库 |

配置方式：

```bash
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/your/libraries
```

**对你的意义**：你后续安装 Miniconda、CUDA、PyTorch 时，`PATH` 和 `LD_LIBRARY_PATH` 是打交道最多的两个变量，需要正确配置。`LIBRARY_PATH` 则在你需要从源码编译依赖库（比如某些自定义算子）时发挥作用。

## 块 50

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语时记录下来再提问。

---

先确认CUDA是否已配置好。在终端执行：

```bash
echo $PATH | grep cuda
echo $LD_LIBRARY_PATH | grep cuda
```

有输出（如`/usr/local/cuda/bin`）说明已配置，直接跳到验证步骤；无输出则继续。

编辑配置文件：

```bash
nano ~/.bashrc
```

文件末尾添加：

```bash
# CUDA 环境变量配置
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

> **注意**：`/usr/local/cuda`是默认路径，若CUDA装在其他位置请替换。

使配置生效并验证：

```bash
source ~/.bashrc
echo $PATH | grep cuda
nvcc --version
```

`nvcc --version`能输出版本号即配置成功。

---

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标（DeepSpeed分布式训练）的关系；遇到无法解释的术语

## 块 51

**本块学习检查**：完成本块后，用自己的话说明本块内容与当前学习目标的关系；遇到无法解释的术语时记录下来再提问。

在配置DeepSpeed分布式训练前，先确认环境变量已正确加载。用`printenv`或`env`查看全部变量，用`echo $PATH`或`printenv PATH`检查特定项，`declare -p PATH`可看是否已导出。若在`~/.bashrc`中设置了变量，执行`source ~/.bashrc`使其生效，再用`echo $MY_VAR`验证输出。遇到异常值，用`grep -r "MY_VAR" ~/ /etc/`追踪来源，或用`type -a python3`定位命令路径。这些操作确保后续DeepSpeed的路径和配置变量正确，避免训练时找不到依赖。

## 块 52

## 8.1 变量未生效（①）——排查与解决

你在配置 DeepSpeed 环境时，如果改了 `~/.bashrc` 但新变量没生效，通常逃不出下面四个原因：

1. **改完没重新加载**——配置文件不会自动生效，必须手动 `source` 或重启终端。
2. **语法写错了**——比如漏了 `export` 关键字，变量根本没被导出。
3. **变量被覆盖**——后面某处又给同一个变量赋了新值，把你设的冲掉了。
4. **终端/IDE 没加载 `~/.bashrc`**——有些图形界面工具（如某些 IDE 的终端）默认不读这个文件。

**按顺序排查三步：**

```bash
# 1. 确认配置文件里确实有你的变量，且语法正确
cat ~/.bashrc | grep "你的变量名"

# 2. 重新加载配置
source ~/.bashrc

# 3. 检查是否有重复定义（重点看 PATH 是否被多次赋值）
grep -n "PATH=" ~/.bashrc
```

第

## 块 53

## 9.1 脚本中临时修改变量（①）

在跑 DeepSpeed 训练脚本时，你经常需要临时调整环境变量（比如设置 `CUDA_VISIBLE_DEVICES` 来控制用哪张卡）。最直接的方式是在脚本内部用 `export`：

```bash
#!/bin/bash
export TEMP_VAR=123
# 仅在脚本执行期间生效，脚本结束后变量消失
```

关键点：这个变量**只在当前脚本运行期间存在**，脚本结束就自动消失，不会污染你后续的终端会话。这对双卡调试很实用——你可以在脚本开头临时指定变量，跑完即弃。

## 9.2 使用 env 命令指定单次运行的变量（①）

如果你只想**某一次命令**用特定变量，不需要写脚本，用 `env` 前缀即可：

```bash
env MY_VAR=hello python3 my_script.py
# MY_VAR 仅在这次 python3 运行时有效
```

这在分布式训练中很常用：比如你想快速测试 DeepSpeed 在不同配置下的表现，可以直接 `env CUDA_VISIBLE_DEVICES=0 python3 train.py` 指定只用第一张卡，而不用改任何文件。

## 9.3 使用 .env 文件分组管理变量（④）

当项目需要**多个**环境变量时（比如数据路径、编辑器、时区），逐个 `export` 太乱。创建一个 `.env` 文件统一管理：

```bash
# 创建 ~/.env 文件
nano ~/.env
```

内容示例：

```bash
export PROJECT_HOME

## 块 54

## 场景 1：创建并验证自定义变量

在进入 DeepSpeed 分布式训练之前，先掌握环境变量的基本操作——这是配置训练脚本、指定 GPU 卡、传递超参数的基础能力。

**操作步骤：**

```bash
# 1. 设置一个临时变量（仅当前终端有效）
export MY_NAME="zhangsan"

# 2. 验证
echo $MY_NAME
```

临时变量在终端关闭后消失。要让变量永久生效，需要写入 shell 配置文件：

```bash
# 3. 写入 ~/.bashrc 使其永久生效
echo 'export MY_NAME="zhangsan"' >> ~/.bashrc

# 4. 重新加载并验证
source ~/.bashrc
echo $MY_NAME
```

**要点：** `export` 将变量传递给子进程；`source` 重新加载配置文件使改动立即生效。

---

**本块学习检查**

完成本块后，用自己的话说明：环境变量配置与 DeepSpeed 分布式训练目标之间有什么关系？遇到无法解释的术语时记录下来再提问。

## 块 55

**场景 3：模拟 CUDA 环境变量配置**

你的目标是跑通 DeepSpeed 分布式训练，而 CUDA 环境变量是让 PyTorch 找到 GPU 计算库的第一步。即使现在没装 CUDA，也可以先练熟这套配置流程，后面配服务器时直接复用。

操作步骤：

1. 编辑配置文件：`nano ~/.bashrc`
2. 在文件末尾追加两行（假设 CUDA 装在 `/usr/local/cuda`）：
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. 保存并生效：`source ~/.bashrc`
4. 验证是否配置成功：`nvcc --version`，预期输出 CUDA 版本信息；或运行 `echo $PATH

## 块 56

## 系统级配置文件（选读）

你正在为 DeepSpeed 分布式训练配置多卡环境，系统级配置虽然日常开发不常用，但当你需要为团队公共服务器统一设置环境时，会用到。

**A.1 `/etc/environment`**：影响所有用户，修改需 `sudo`。特点是**不需要 `export`**，直接写 `KEY=value`，但不支持命令替换和通配符，支持变量引用（如 `$PATH`）。

```bash
sudo nano /etc/environment
# 添加：
JAVA_HOME=/usr/lib/jvm/java-11-openjdk
# 保存后需重启或重新登录生效
```

**A.2 `/etc/profile`**：全局 Bash 配置，**需要 `export`**。

```bash
sudo nano /etc/profile
# 添加：
export PATH=$PATH:/usr/local/custom/bin
# 保存后 source 或重启生效
source /etc/profile
```

**本块学习检查**：用自己的话说明本块内容与 DeepSpeed 分布式训练目标的关系；遇到无法解释的术语

## 块 57

**B. 不同 Shell 的配置文件对照表（①）**

你平时敲命令的终端，用的“解释器”可能不是 Bash。不同 Shell 的配置文件位置不同，改错了地方不生效。

| Shell 类型 | 配置文件 | 作用范围 |
|---|---|---|
| **Bash** | `~/.bashrc` | 非登录 Shell（日常终端） |
| **Bash** | `~/.bash_profile` | 登录 Shell（SSH 登录） |
| **Zsh** | `~/.zshrc` | 所有交互式 Shell |
| **Fish** | `~/.config/fish/config.fish` | 所有会话 |

**怎么知道自己用的是什么 Shell？** 执行 `echo $SHELL`

**本块学习检查**：完成本块后，用自己的话

## 块 58

## C. 清理 PATH 中的重复项（④）

多次修改 PATH 后，重复路径会累积，影响命令查找效率。用一条命令去重：

```bash
export PATH=$(echo $PATH | awk -v RS=: '!a[$0]++' | paste -sd: -)
```

这条命令把 PATH 按冒号拆开、去重、再拼回冒号分隔的字符串，结果写回 PATH。你的双卡 RTX 4090 环境里，CUDA、DeepSpeed 相关路径可能被多次添加，运行这条命令能保持 PATH 干净。

## D. 根据环境加载不同配置（④）

在 `~/.bashrc` 中按环境变量切换配置：

```bash
if [ "$ENVIRONMENT" = "production" ]; then
    source ~/.env.prod
else
    source ~/.env.dev
fi
```

训练时用 dev 配置，部署时切 production，避免手动改配置。

## E. 脚本调试模式（④）

```bash
# 打印脚本每一行执行前的变量值
bash -x your_script.sh

# 或在当前 Shell 中开启追踪
set -o xtrace
source ~/.bashrc
set +o xtrace
```

调试 DeepSpeed 启动脚本时，`bash -x` 能帮你定位环境变量是否传递正确。

---

## 总结与后续学习路线

你已掌握：环境变量概念、临时与永久变量、`source` 生效、PATH/LD_LIBRARY_PATH/LIBRARY_PATH 区别、PATH 修改方法、CUDA 环境变量配置、调试方法、`.env` 安全规范。

**下一步**：安装 Miniconda 时，`conda init` 会修改你的 `~/.bashrc`，在 PATH 中加入 conda 路径——届时打开文件看看，就能理解本节内容如何在实际工具安装中生效。之后学习 `nvidia-smi` 显存监控时，你会用到 PATH 确保命令可用。现在打开 `~/.bashrc`，试着添加

## 测评题目

### Q-KB0100-002
执行以下命令验证当前用户和主机名：
```bash
whoami && hostname
```

### Q-KB0100-004
执行以下命令检查系统架构和内核版本：
```bash
arch && uname -r
```

### Q-KB0101-004
`cp -r source_dir/ dest_dir/` 中 `-r` 参数的含义是？

### Q-KB0101-002
命令 `ls -la` 中，`-l` 的作用是________，`-a` 的作用是________。

### Q-KB0102-001
临时设置环境变量 `MY_VAR=hello`，命令为________；若要使其在当前shell及子进程中生效，需使用________命令。

### Q-KB0102-002
请写出将 `/usr/local/cuda/bin` 永久添加到 `PATH` 环境变量的完整步骤（以bash为例）。

### Q-KB0101-001
以下哪个命令用于**切换当前工作目录**？

### Q-KB0104-002
`nvidia-smi` 输出中，`Memory-Usage: 2048MiB / 8192MiB` 表示什么？
