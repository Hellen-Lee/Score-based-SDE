@echo off
chcp 65001 >nul
setlocal

set VENV_DIR=.venv

echo ============================================
echo   SDE 项目 - 虚拟环境创建脚本
echo ============================================
echo.

:: 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+ 并加入 PATH
    pause
    exit /b 1
)

:: 如果虚拟环境已存在，询问是否重建
if exist "%VENV_DIR%" (
    echo [提示] 虚拟环境 %VENV_DIR% 已存在
    set /p REBUILD="是否删除并重建？(y/N): "
    if /i "%REBUILD%"=="y" (
        echo 正在删除旧环境...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo 跳过创建，直接安装依赖...
        goto install
    )
)

:: 创建虚拟环境
echo.
echo [1/3] 正在创建虚拟环境 %VENV_DIR% ...
python -m venv %VENV_DIR%
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败
    pause
    exit /b 1
)
echo       虚拟环境创建成功

:install
:: 激活虚拟环境
echo.
echo [2/3] 正在激活虚拟环境...
call %VENV_DIR%\Scripts\activate.bat

:: 升级 pip
echo.
echo [3/3] 正在安装依赖包...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败，请检查网络连接或 requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================
echo   环境搭建完成！
echo ============================================
echo.
echo   激活环境:  %VENV_DIR%\Scripts\activate.bat
echo   运行训练:  python train_mnist.py
echo   退出环境:  deactivate
echo.
pause
