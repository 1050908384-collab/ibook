# 常用 Shell 命令（cd / ls / mkdir / cp / mv / pwd / touch / rm）

① 底稿：《cd, ls, mkdir, touch, cp, mv, rm_cd文件夹》  
　　网址：https://blog.csdn.net/Aaron_neil/article/details/125324594

② 参考文章：《【Linux学习02】基础指令（ls，pwd，cd，mkdir，touch，cp）》  
　　网址：https://blog.csdn.net/qq_42764468/article/details/103444957

> **适合人群**：已完成上一节"环境准备"，已拥有 Ubuntu 终端并能执行基础命令的初学者。本节时长约 1 小时，目标是让你能熟练地在 Linux 中"看目录、切目录、建目录、复制和移动文件"。（①+②）

---

## 一、Linux 命令的基本格式（新增②）

在正式学习命令之前，先记住 Linux 命令的**标准格式**：（②）

```bash
# 指令主体 [选项] [操作对象]
```

- **指令主体**：你要做什么，比如 `ls`、`cd`、`mkdir`
- **选项**（可选）：以 `-` 或 `--` 开头，用来修饰指令的行为，比如 `-l` 表示"详细列表"
- **操作对象**（可选）：指令作用的目标，比如一个文件夹路径、一个文件名

一个指令可以包含**多个选项**和**多个操作对象**。选项的顺序通常不影响结果，例如 `ls -ahl` 和 `ls -lha` 效果相同。（①+②）

---

## 二、你在哪？pwd 与路径的概念（新增②）

### 2.1 pwd：打印当前工作目录（新增②）

`pwd` = **P**rint **W**orking **D**irectory，用来告诉你"我现在在哪个文件夹里"。（②）

```bash
pwd
```

示例输出：
```bash
/home/zhangsan
```

这表示你当前位于 `/home/zhangsan` 目录下。在操作文件之前，养成先敲 `pwd` 的习惯，能避免很多"文件找不到了"的困惑。（②）

### 2.2 绝对路径 vs 相对路径（新增②）

这是新手最容易混淆的概念，务必理解：（②）

| 路径类型 | 定义 | 示例 |
|---|---|---|
| **绝对路径** | 从根目录 `/` 开始，不依赖当前位置 | `/home/zhangsan/Documents` |
| **相对路径** | 相对于你当前所在的位置 | `./file.txt`、`../backup` |

相对路径中两个重要符号：（②）

- `./`：表示**当前目录**（可省略）
- `../`：表示**上一级目录**
- `~`：表示**当前用户的家目录**（如 `/home/zhangsan`）

> **举例**：假设你在 `/home/zhangsan/Documents` 目录下：
> - `cd /home/zhangsan` 是绝对路径，去哪都管用
> - `cd ..` 是相对路径，表示回到上一级（即 `/home/zhangsan`）
> - `cd ~` 直接回到家目录

---

## 三、ls：列出目录内容（①+②）

`ls` = list，用来查看当前目录（或指定目录）下有哪些文件和文件夹。（①+②）

### 3.1 基础用法（①+②）

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

> **关于隐藏文件**：Linux 中以 `.` 开头的文件/文件夹默认是隐藏的，比如 `.bashrc`、`.ssh`。这些文件往往很重要，后续配置环境变量时会频繁遇到。（①+②）

### 3.2 ls -l 的输出解读（新增②）

执行 `ls -l` 后，你会看到类似这样的输出：（②）

```bash
drwxr-xr-x 2 zhangsan zhangsan 4096 Aug  3 10:00 Documents
-rw-r--r-- 1 zhangsan zhangsan  220 Aug  3 09:30 hello.py
```

**第一列**的**第一个字符**表示文件类型：（②）

- `d`：directory，文件夹
- `-`：普通文件
- `l`：link，软链接（快捷方式）

其余列分别表示：权限、硬链接数、所有者、所属组、文件大小、最后修改时间、文件名。（②）

### 3.3 其他常用选项（①）

```bash
# 以详细列表+人类可读格式显示文件大小（KB, MB, GB）
# 注意：-h 必须与 -l 配合使用才能看到大小，单独 ls -h 无效果
ls -lh

# 以完整时间格式输出
ls --full-time

# 按最后修改时间排序（最新的在最前面）
ls -t

# 按文件大小排序（从大到小）
ls -S

# 逆转排序（配合 -t 或 -S 使用）
ls -rt

# 显示文件 inode 信息（文件的元信息编号）
ls -i

# 在不同文件结尾输出不同符号（文件夹后有 /，软链接后有 @）
ls -F

# 显示文件夹本身信息，不输出其中内容
ls -d

# 组合示例：详细、人类可读、按时间逆序排列
ls -lhtr
```

---

## 四、cd：切换目录（①+②）

`cd` = **C**hange **D**irectory，用来在不同文件夹之间跳转。（①+②）

### 4.1 基础用法（①+②）

```bash
# 进入指定目录（绝对路径）
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

### 4.2 路径切换示例（新增②）

假设当前在 `/usr/local` 下，想切换到 `/home/zhangsan`：（②）

```bash
# 方法1：绝对路径（去哪都管用）
cd /home/zhangsan

# 方法2：相对路径（先回根目录，再去 home）
cd ../../home/zhangsan
```

> **建议**：新手优先使用绝对路径，不容易迷路；熟悉后多用相对路径，效率更高。（②）

---

## 五、mkdir：创建目录（①+②）

`mkdir` = **M**a**k**e **Dir**ectory，用来创建文件夹。（①+②）

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

> **注意**：`mkdir` 默认**不能**直接创建多级嵌套目录，必须加 `-p` 参数。这是新手最常踩的坑之一。（①+②）

---

## 六、touch：创建空文件（①）

`touch` 原本用来"更新文件的时间戳"，但最常用的功能是**快速创建一个空文件**。（①）

```bash
# 创建一个空文件
touch hello.py

# 一次性创建多个文件
touch file1.txt file2.txt file3.txt

# 使用花括号批量创建（file1.txt ~ file10.txt）
touch file{1..10}.txt

# 在指定路径下创建文件
touch /home/zhangsan/myproject/README.md
```

> **与 mkdir 的区别**：`mkdir` 创建的是**文件夹**（目录），`touch` 创建的是**文件**。如果你 `touch` 了一个名字，后面发现它是个文件而不是文件夹，不要惊讶。（①）

---

## 七、cp：复制文件与目录（①+②）

`cp` = **C**o**p**y，用来复制文件或文件夹。（①+②）

### 7.1 复制文件（①+②）

```bash
# 复制文件到另一个文件夹（保留原文件名）
cp hello.py ./backup/

# 复制文件到另一个文件夹，并改名
cp hello.py ./backup/hello_backup.py

# 一次性复制多个文件到目标文件夹
cp file1.txt file2.txt ./backup/

# 复制时保持文件属性不变（如修改时间）
cp -p hello.py ./backup/

# 覆盖前进行提示（如果目标文件已存在）
cp -i hello.py ./backup/
```

### 7.2 复制目录（①+②）

复制文件夹时，**必须加 `-r` 参数**（recursive，递归），否则系统会拒绝。（①+②）

```bash
# 复制整个目录到另一个位置
cp -r myproject ./backup/

# 注意：
# 如果 backup 文件夹不存在，会新建 backup 并把 myproject 的内容复制进去
# 如果 backup 文件夹已存在，会新建 backup/myproject/，把整个文件夹复制进去
```

### 7.3 复制软链接（①）

```bash
# 拷贝软链接时保持链接属性（不加 -d 会复制源文件本身）
cp -d link1 link2
```

---

## 八、mv：移动与重命名（①）

`mv` = **M**o**v**e，功能有两个：**移动文件/文件夹** 和 **重命名**。（①）

### 8.1 移动文件/文件夹（①）

```bash
# 把文件移动到另一个文件夹
mv hello.py ./backup/

# 把文件夹移动到另一个位置
mv myproject ./archive/

# 使用通配符移动多个文件（把所有以 data_ 开头的文件移到 data 文件夹）
mv data_* ./data/
```

### 8.2 重命名（①）

```bash
# 把 a.txt 重命名为 b.txt（本质是在同一目录下移动）
mv a.txt b.txt

# 移动的同时改名
mv hello.py ./backup/hello_v2.py
```

### 8.3 安全选项（①）

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
rm -rf myproject/
```

> **⚠️ 生命警告**：`rm -rf /` 会删除你的整个系统！`rm -rf ~` 会删除你的家目录！执行 `rm -rf` 前，务必先用 `ls` 确认目标路径，确保没有多打空格、没有写错路径。（①）

---

## 十、综合实战演练（①+②）

光记命令不够，结合场景练习才记得牢！以下是 3 个常见场景，请在你的 Ubuntu 终端里手动操作一遍。（①+②）

### 场景 1：搭建一个项目目录结构

**需求**：在家目录下创建一个名为 `llm-learning` 的文件夹，内部包含 `data`、`models`、`scripts` 三个子文件夹，并在 `scripts` 下创建一个空的 `train.py` 文件。

```bash
# 1. 回到家目录
cd ~

# 2. 创建项目根目录
mkdir llm-learning

# 3. 进入项目目录
cd llm-learning

# 4. 创建三个子目录
mkdir data models scripts

# 5. 创建空文件
touch scripts/train.py

# 6. 验证结构
ls -R
```

预期输出结构：
```bash
llm-learning/
├── data
├── models
├── scripts
│   └── train.py
```

### 场景 2：复制与整理实验文件

**需求**：假设你下载了一些数据集文件在 `~/Downloads/` 中，文件名分别是 `train.csv`、`test.csv`、`dev.csv`。请把它们复制到 `~/llm-learning/data/` 中，并验证。

```bash
# 1. 确认文件存在
ls ~/Downloads/*.csv

# 2. 复制到 data 目录
cp ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/dev.csv ~/llm-learning/data/

# 3. 验证
ls ~/llm-learning/data/
```

### 场景 3：清理临时文件

**需求**：你在 `~/llm-learning/` 下误建了一个名为 `temp` 的空文件夹和一个 `old.txt` 文件，请将它们删除。

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

---

## 十一、新手避坑指南（①+②）

**1. 区分文件和文件夹**：`ls -l` 第一列是 `d` 表示文件夹，是 `-` 表示文件。对文件夹操作别忘了 `-r`（如 `cp -r`、`rm -r`）。（①+②）

**2. 路径写错**：切换目录或复制文件时，路径写错会提示 "No such file or directory"，可先用 `pwd` 和 `ls` 确认当前位置和目录内容。（②）

**3. 路径中包含空格必须用引号包裹**：如果文件夹或文件名包含空格，命令行会把它当成多个参数。务必用双引号包裹：（②）

```bash
# 错误：mkdir my folder 会创建两个文件夹 my 和 folder
# 正确：
mkdir "my folder"
cd "my folder"
touch "my file.txt"
cp "my file.txt" ./backup/
rm "my file.txt"
```

**4. 文件名以 `-` 开头时使用 `--`**：Linux 命令中 `-` 开头通常表示选项。如果文件名本身以 `-` 开头，命令会误把文件名当选项解析。用 `--` 告诉命令"后面都是文件名，不是选项"：（②）

```bash
# 错误：rm -file.txt 会把 -file.txt 当成选项，报错
# 正确：
rm -- -file.txt
touch -- -data.csv
mv -- -old.txt new.txt
```

**5. rm 命令慎用**：`rm -rf` 没有回收站，删除即永久消失。删除前务必用 `ls` 确认目标。（①）

**6. 复制目录忘加 `-r`**：新手常犯 `cp mydir ./backup/` 然后报错。记住：**文件夹复制必须加 `-r`**。（①+②）

**7. 命令记不住？**：输入命令前按 `Tab` 键自动补全（文件/目录名）；忘记命令语法时，用 `命令 --help` 查看帮助，或 `man 命令` 查看详细手册。（②）

---

## 十二、总结与后续学习路线（①+②）

恭喜你完成了常用 Shell 命令的学习！你现在应该掌握了：

- ✅ Linux 命令的基本格式（`指令 [选项] [对象]`）
- ✅ 绝对路径与相对路径的区别，以及 `pwd`、`cd` 的灵活运用
- ✅ `ls` 的各种选项（`-a`、`-l`、`-h`、`-t` 等）及其组合使用
- ✅ `mkdir` 创建单级/多级/批量目录
- ✅ `touch` 快速创建空文件
- ✅ `cp` 复制文件与目录（`-r`、`-p`、`-i`）
- ✅ `mv` 移动与重命名文件（`-i`、`-f`）
- ✅ `rm` 删除文件与目录（`-r`、`-f`、`-d`、`--`）
- ✅ 带空格路径的引号包裹、`-` 开头文件名的 `--` 处理

> **下一步预告**：在下一节 **"环境变量配置"** 中，我们将学习如何用 `echo`、`export` 等命令查看和修改环境变量，并编辑 `~/.bashrc` 文件让配置永久生效——这些操作都会用到本节学到的 `cd`、`ls`、`touch` 等命令。现在，打开你的终端，把本章的命令都敲一遍吧！