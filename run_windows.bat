@echo off
echo Starting Song Genre Classifier...

REM Check if dependencies are installed
python -c "import torch" 2>NUL
IF %ERRORLEVEL% NEQ 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

python gui_predict.py
pause