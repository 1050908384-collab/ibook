# -*- coding: utf-8 -*-
# 清理个人数据 - PowerShell 入口（_clean_personal.ps1）
# 所有中文输出在这里，避免 .bat 在 GBK 终端里乱码。

param(
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'

# 切到项目根目录（快捷方式/.. ）
Set-Location -Path (Join-Path $PSScriptRoot '..')

# 找 Python
$pythonExe = $null
if (Test-Path '.venv\Scripts\python.exe') {
    $pythonExe = '.venv\Scripts\python.exe'
} elseif (Test-Path 'venv\Scripts\python.exe') {
    $pythonExe = 'venv\Scripts\python.exe'
} else {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { $pythonExe = $found.Source }
}
if (-not $pythonExe) {
    Write-Host ''
    Write-Host '[错误] 找不到 Python，请先安装 Python 或在项目目录创建 .venv 虚拟环境。'
    Read-Host '按 Enter 退出'
    exit 1
}

Write-Host ''
Write-Host '========================================'
Write-Host '  清理个人数据（发布前专用）'
Write-Host '========================================'
Write-Host ''
Write-Host '此脚本会：'
Write-Host '  1. 自动备份当前 project.db（保存为 before_clean 时间戳备份）'
Write-Host '  2. 删除所有个人学习记录 / 画像 / 会话 / 对话 / 答题 / 审核记录'
Write-Host '  3. 保留所有公共知识库（知识点、本地文章、视频、题库）'
Write-Host ''
Write-Host '注意：'
Write-Host '  * 如果之后你自己还要继续用本目录的数据，'
Write-Host '    请先用备份文件替换 project.db 再启动！（详见 发布流程说明.txt）'
Write-Host ''

$args = @('scripts\clean_personal_data.py', '--db', 'data\sqlite_data\project.db')
if ($Yes) { $args += '--yes' } else { $args += '--yes' }
# 总是用 --yes 跳过 Python 脚本里的 input()，避免卡住

& $pythonExe @args
$rc = $LASTEXITCODE

Write-Host ''
if ($rc -eq 0) {
    Write-Host '========================================'
    Write-Host '  清理完成！现在可以把项目文件夹发给别人了。'
    Write-Host '  如果你自己还要继续用，记得把备份文件恢复回来：'
    Write-Host '    copy data\sqlite_data\project.before_clean_*.bak data\sqlite_data\project.db'
    Write-Host '========================================'
} else {
    Write-Host "[异常] 退出码 $rc，请检查上方日志。"
}

Write-Host ''
Read-Host '按 Enter 退出'
exit $rc
