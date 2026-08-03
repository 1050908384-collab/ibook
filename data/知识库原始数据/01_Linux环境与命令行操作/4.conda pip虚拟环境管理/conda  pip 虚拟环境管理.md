# conda / pip 虚拟环境管理

① 底稿：《Conda命令详解：从入门到实战的完全指南》  
　　网址：https://blog.csdn.net/superfreeman/article/details/148233913

② 参考文章：《Linux 下极简安装 Conda（Miniconda / Anaconda），5 分钟搞定环境配置》  
　　网址：https://blog.csdn.net/qq_41797451/article/details/160309627

③ 参考文章：《Conda虚拟环境管理（创建、删除、进入、退出）详细教程》  
　　网址：https://blog.csdn.net/m0_74738450/article/details/135827553

④ 参考文章：《没有conda，怎么用pip进行虚拟环境创建和包管理》  
　　网址：https://blog.csdn.net/weixin_53505093/article/details/145258454

> **适合人群**：已完成前三节，掌握 Shell 命令和环境变量配置的初学者。本节时长约 1 小时，目标是掌握 conda 和 pip 两种虚拟环境管理方案，能够独立创建、切换、删除环境，并配置国内镜像源加速下载。（①+②）

---

## 一、为什么需要虚拟环境？（①+④）

在正式安装工具之前，先理解"虚拟环境"到底解决什么问题。（①+④）

想象你同时在做两个项目：
- **项目 A** 需要 PyTorch 1.13 + Python 3.9
- **项目 B** 需要 PyTorch 2.0 + Python 3.10

如果所有包都装在同一个系统环境里，版本冲突会让你崩溃。虚拟环境就是为每个项目创建**独立的、隔离的 Python 运行空间**，互不影响。（①+④）

**核心好处**：（①+④）
1. **隔离依赖**：不同项目用不同版本的库，互不干扰
2. **环境可复现**：导出配置文件，换台机器能一键重建相同环境
3. **玩坏了不心疼**：环境出问题直接删掉重建，不影响系统

> **conda vs pip + venv 怎么选？** 本教程两种都教。conda 功能更强大（能管理非 Python 依赖，如 CUDA），推荐作为主力；pip + venv 更轻量，适合简单项目或服务器上没有 conda 权限时应急使用。（①+④）

---

## 二、方案一：Conda 环境管理（推荐）（①+②+③）

### 2.1 安装 Miniconda（②）

Conda 有两个发行版：（②）

| 版本 | 特点 | 推荐场景 |
|---|---|---|
| **Miniconda** | 只包含 conda + Python，体积小（约 100MB） | 服务器、普通开发环境（**推荐**） |
| **Anaconda** | 预装 1500+ 数据科学包，体积大（约 3GB） | 本地个人电脑，希望开箱即用 |

> 本教程推荐 **Miniconda**，按需安装，不浪费空间。（②）

#### 安装步骤（②）

```bash
# 1. 查看系统架构（通常是 x86_64）
uname -m

# 2. 下载 Miniconda 安装脚本（官方源）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 如果官方源下载慢，可用北大镜像：
# wget -c https://mirrors.pku.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
# 3. 运行安装脚本
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中的交互提示：（②）
1. **阅读许可协议**：按 `Enter` 滚动，或按 `q` 跳到末尾
2. **接受条款**：输入 `yes` 回车
3. **选择安装路径**：直接回车使用默认路径（`~/miniconda3`）
4. **初始化 Conda**：当询问 `Do you wish the installer to initialize Miniconda3?` 时，务必输入 **`yes`** —— 这会自动修改你的 `~/.bashrc` 文件，把 conda 加入 PATH

```bash
# 5. 让修改生效（上节课学过的 source 命令！）
source ~/.bashrc

# 6. 验证安装
conda --version
# 预期输出：conda 25.x.x 或类似版本号
```

> **为什么输入 yes 后还要 source ~/.bashrc？** 因为安装脚本在 `~/.bashrc` 末尾追加了一段初始化代码（你可以用 `nano ~/.bashrc` 翻到最后看看）。`source` 让这段新代码在当前终端立即生效，不用重开窗口。（②+上节知识）

> **安装后每次打开终端都自动进入 (base) 环境？** 这是 conda 的默认行为。如果你不喜欢，可以关闭：（③）
> ```bash
> conda config --set auto_activate_base false
> ```
> 然后 `source ~/.bashrc` 生效。之后打开终端就不会自动进 base 了，需要手动 `conda activate`。

### 2.2 配置国内镜像源（①+②）

默认官方源在国外，下载包时可能很慢甚至超时。配置国内镜像源能显著提升速度。（①+②）

```bash
# 添加清华镜像源（推荐）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# 设置显示下载源地址（方便排查问题）
conda config --set show_channel_urls yes

# 清理索引缓存，使配置立即生效
conda clean -i
```

> **注意**：部分教程会让你删除默认源 `conda config --remove channels defaults`。建议保留默认源作为 fallback，国内源优先即可。（②）

### 2.3 环境管理核心命令（①+③）

#### 查看环境（①+③）

```bash
# 列出所有环境（带 * 的是当前激活的环境）
conda env list

# 或
conda info --envs
```

#### 创建环境（①+③）

```bash
# 基础创建：指定环境名和 Python 版本
conda create -n llm_env python=3.10

# 创建时同时安装包
conda create -n ml_project python=3.11 numpy pandas

# 克隆现有环境（备份/迁移必备）
conda create --name prod_env --clone dev_env

# 在指定路径创建环境（非默认位置）
conda create --prefix ./custom_env python=3.10
```

#### 激活与退出环境（①+③）

```bash
# 激活环境
conda activate llm_env

# 退出当前环境（回到 base）
conda deactivate

# 直接退出到 base（无论嵌套多深）
conda deactivate
```

> **激活后提示符变化**：激活 `llm_env` 后，终端提示符会变成 `(llm_env) zhangsan@ubuntu:~$`，前面的 `(llm_env)` 就是当前环境标识。（③）

#### 删除环境（①+③）

```bash
# 删除整个环境（包括其中所有包）
conda remove -n llm_env --all

# 或
conda env remove -n llm_env
```

### 2.4 包管理核心命令（①）

#### 安装包（①）

```bash
# 安装最新版
conda install numpy

# 安装指定版本
conda install numpy=1.24.0

# 从指定频道安装（conda-forge 是社区维护的频道，包很全）
conda install -c conda-forge transformers

# 从 requirements.txt 批量安装
conda install --yes --file requirements.txt
```

#### 查看与搜索包（①）

```bash
# 查看当前环境已安装的所有包
conda list

# 查看指定环境的包
conda list -n llm_env

# 搜索包（查看可用版本）
conda search pytorch
```

#### 更新与卸载包（①）

```bash
# 更新单个包
conda update numpy

# 更新所有包
conda update --all

# 卸载包
conda remove numpy

# 强制重装（解决诡异依赖问题）
conda install --force-reinstall numpy=1.24.0
```

### 2.5 环境导出与复现（①）

这是团队协作和项目部署的核心技能。（①）

```bash
# 导出当前环境配置到 YAML 文件（推荐加 --no-builds 增强跨平台兼容性）
conda env export --no-builds > environment.yml

# 根据 YAML 文件重建环境
conda env create -f environment.yml

# 生成精确版本清单（含构建号，同一平台复现最准确）
conda list --explicit > locked_deps.txt
```

> **environment.yml 是什么？** 它是一个文本文件，记录了环境名、Python 版本、所有包及其版本。把它放进 Git 仓库，队友就能一键重建和你一模一样的环境。（①）

### 2.6 Conda 与 Pip 的协作（①）

conda 的包仓库虽然丰富，但某些包（如最新版 PyTorch 的某些细分版本）可能只有 pip 上有。两者可以混用，但**建议优先用 conda，conda 找不到再用 pip**。（①）

```bash
# 在 conda 环境中先安装 pip
conda install pip

# 然后用 pip 安装 conda 仓库没有的包
pip install some-package-only-on-pypi
```

> **⚠️ 注意**：混用可能导致依赖解析混乱。如果出现问题，建议新建一个干净环境重新安装。（①）

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

---

## 三、方案二：pip + venv 虚拟环境管理（④）

如果你不想装 Miniconda，或者服务器上没有权限安装，Python 自带的 `venv` 模块 + `pip` 也能完成虚拟环境管理。（④）

### 3.1 创建虚拟环境（④）

```bash
# 确认 Python 版本
python3 --version

# 在当前目录下创建名为 myenv 的虚拟环境
python3 -m venv myenv
```

> `python3 -m venv myenv` 会在当前目录创建一个 `myenv` 文件夹，里面包含独立的 Python 解释器和 pip。（④）

### 3.2 激活与退出环境（④）

```bash
# 激活环境（Linux/macOS）
source myenv/bin/activate

# 激活后提示符会显示 (myenv)
# 退出环境
deactivate
```

### 3.3 包管理（④）

```bash
# 安装包
pip install numpy

# 安装指定版本
pip install numpy==1.24.0

# 查看已安装包
pip list

# 升级包
pip install --upgrade numpy

# 卸载包
pip uninstall numpy
```

### 3.4 依赖导出与复现（④）

```bash
# 导出当前环境的所有依赖到 requirements.txt
pip freeze > requirements.txt

# 在新环境中根据 requirements.txt 安装所有依赖
pip install -r requirements.txt

# 查看有哪些包可以升级
pip list --outdated
```

### 3.5 删除环境（④）

```bash
# 直接删除环境文件夹即可
rm -rf myenv
```

> **pip + venv 的局限**：只能管理 Python 包，无法管理非 Python 依赖（如 CUDA 运行时）。做大模型开发时，conda 仍是更推荐的选择。（④）

---

## 四、配置 pip 国内镜像源（④）

即使使用 pip，也应该配置国内镜像源加速下载。（④）

#### 临时使用（单次生效）（④）

```bash
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 永久配置（推荐）（④）

```bash
# 创建/编辑 pip 配置文件
mkdir -p ~/.pip
nano ~/.pip/pip.conf
```

写入以下内容：（④）

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

保存后，以后所有 `pip install` 都会自动走清华镜像源。（④）

---

## 五、综合实战演练（①+②+④）

请在你的 Ubuntu 终端中完成以下操作，巩固本节所学：

### 场景 1：用 Conda 创建大模型开发环境

```bash
# 1. 创建一个名为 llm-dev 的环境，Python 3.10
conda create -n llm-dev python=3.10 -y

# 2. 激活环境
conda activate llm-dev

# 3. 安装常用包
conda install numpy pandas -y

# 4. 查看已安装包
conda list

# 5. 导出环境配置
conda env export --no-builds > ~/llm-dev-environment.yml

# 6. 退出环境
conda deactivate

# 7. 查看所有环境
conda env list
```

### 场景 2：用 pip + venv 创建轻量环境

```bash
# 1. 创建项目目录
mkdir -p ~/projects/test_project
cd ~/projects/test_project

# 2. 创建虚拟环境
python3 -m venv venv

# 3. 激活环境
source venv/bin/activate

# 4. 安装包
pip install requests

# 5. 导出依赖
pip freeze > requirements.txt

# 6. 退出环境
deactivate

# 7. 验证 requirements.txt 内容
cat requirements.txt
```

### 场景 3：验证 conda 对 ~/.bashrc 的修改

```bash
# 1. 查看 ~/.bashrc 末尾的 conda 初始化代码
tail -n 20 ~/.bashrc

# 2. 你应该能看到类似这样的内容：
# >>> conda initialize >>>
# ...
# <<< conda initialize <<<

# 3. 这就是上节课学的"环境变量配置"的实际应用！
#    conda 通过修改 ~/.bashrc，把 ~/miniconda3/bin 加入了你的 PATH
```

---

## 六、常见问题与解决（①+②+③+④）

| 症状 | 可能原因 | 解决方案 |
|---|---|---|
| `conda: command not found` | conda 未加入 PATH | 执行 `~/miniconda3/bin/conda init bash`，然后 `source ~/.bashrc`（②） |
| 环境激活失败 | Shell 未初始化 conda | 运行 `conda init bash` 并重启终端（②） |
| 包安装缓慢或超时 | 未配置国内镜像源 | 按 2.2 节配置清华/北大镜像源（①+②） |
| 依赖冲突 | 多个包版本要求矛盾 | 新建干净环境，逐步安装排查（①） |
| pip 安装提示权限不足 | 在系统 Python 中安装 | 先激活虚拟环境，再用 pip 安装（④） |
| 每次打开终端自动进 (base) | conda 默认设置 | `conda config --set auto_activate_base false`（③） |
| 环境名含中文或特殊字符 | conda 不支持 | 使用纯英文小写 + 下划线命名（①） |

---

## 七、总结与后续学习路线（①+②+③+④）

恭喜你完成了 conda / pip 虚拟环境管理章节的学习！你现在应该掌握了：

- ✅ 虚拟环境的核心价值：隔离依赖、避免冲突、环境可复现
- ✅ Miniconda 的下载、安装与验证（`conda --version`）
- ✅ conda 环境管理（create、activate、deactivate、remove、clone、export）
- ✅ conda 包管理（install、list、search、update、remove）
- ✅ conda 国内镜像源配置（清华源）
- ✅ pip + venv 的轻量替代方案
- ✅ pip 国内镜像源配置
- ✅ 环境导出与复现（`environment.yml`、`requirements.txt`）
- ✅ 理解 `conda init` 如何修改 `~/.bashrc`（上节课知识的实际应用）

> **下一步预告**：在下一节 **"nvidia-smi 显存监控"** 中，我们将学习如何查看 GPU 的显存占用、温度、利用率等信息。这些命令需要在已配置好 CUDA 环境变量的系统中运行——如果你安装了 NVIDIA GPU 驱动，nvidia-smi 通常已在 PATH 中。随后我们将进入 **"实际操作与练习"**，综合运用 Linux 环境准备、Shell 命令、环境变量和虚拟环境管理的全部知识。现在，创建你的第一个 conda 环境，试着安装几个包吧！