$ErrorActionPreference = "Stop"
$VENV_DIR = ".venv"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SDE 项目 - 虚拟环境创建脚本 (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 是否可用
try {
    $pyVersion = python --version 2>&1
    Write-Host "[信息] 检测到 $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[错误] 未找到 Python，请先安装 Python 3.8+ 并加入 PATH" -ForegroundColor Red
    exit 1
}

# 如果虚拟环境已存在，询问是否重建
if (Test-Path $VENV_DIR) {
    Write-Host "[提示] 虚拟环境 $VENV_DIR 已存在" -ForegroundColor Yellow
    $rebuild = Read-Host "是否删除并重建？(y/N)"
    if ($rebuild -eq "y") {
        Write-Host "正在删除旧环境..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VENV_DIR
    } else {
        Write-Host "跳过创建，直接安装依赖..." -ForegroundColor Yellow
        & "$VENV_DIR\Scripts\Activate.ps1"
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        Write-Host ""
        Write-Host "依赖安装完成！" -ForegroundColor Green
        exit 0
    }
}

# 创建虚拟环境
Write-Host ""
Write-Host "[1/3] 正在创建虚拟环境 $VENV_DIR ..." -ForegroundColor Cyan
python -m venv $VENV_DIR
Write-Host "      虚拟环境创建成功" -ForegroundColor Green

# 激活虚拟环境
Write-Host ""
Write-Host "[2/3] 正在激活虚拟环境..." -ForegroundColor Cyan
& "$VENV_DIR\Scripts\Activate.ps1"

# 安装依赖
Write-Host ""
Write-Host "[3/3] 正在安装依赖包..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  环境搭建完成！" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  激活环境:  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  运行训练:  python train_mnist.py" -ForegroundColor White
Write-Host "  退出环境:  deactivate" -ForegroundColor White
Write-Host ""
