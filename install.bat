@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === SemiReal Studio: установка ===

set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Не найден Python. Установите Python 3.12 с python.org ^(галочка "Add python.exe to PATH"^).
  pause
  exit /b 1
)

if not exist venv (
  echo Создаю виртуальное окружение...
  %PY% -m venv venv || goto :fail
)
call venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail

echo Устанавливаю PyTorch с поддержкой CUDA ^(~3 ГБ^)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail

echo Устанавливаю остальные библиотеки...
pip install -r requirements.txt || goto :fail

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo Готово. Положите SDXL-модель в models\checkpoints и запустите run.bat
pause
exit /b 0

:fail
echo.
echo Установка прервалась с ошибкой. Пролистайте вывод выше.
pause
exit /b 1
