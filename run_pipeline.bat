@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Harvested Robotics - Weed Segmentation Pipeline
echo ========================================================
echo.
echo Select Run Mode:
echo [1] Docker Mode (Zero Setup)
echo [2] Local Python/VENV Mode (RecommendedRequires Python installed)
echo.
set /p mode="Enter choice (1 or 2): "

if "%mode%"=="1" goto DOCKER_MODE
if "%mode%"=="2" goto LOCAL_MODE
goto END

:DOCKER_MODE
echo.
echo --- Starting Docker Pipeline ---
echo Checking for GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [GPU FOUND]: Using GPU acceleration!
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
) else (
    echo [GPU NOT FOUND]: Defaulting to CPU mode.
    docker-compose up --build
)
goto END

:LOCAL_MODE
echo.
echo --- Starting Local Python Pipeline ---
echo Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR]: Python is not installed or not in PATH!
    echo Please install Python 3.8+ to use this mode.
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Installing dependencies...
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Activating existing venv...
    call venv\Scripts\activate
)

echo.
echo Running Inference Script...
python src\inference.py --input_dir data\raw --output_dir results --weights models\weights\best.pt --weed_weights models\weights\yolov8l-seg.pt
echo.
echo Done! Check 'results/' folder.
pause
goto END

:END
endlocal
