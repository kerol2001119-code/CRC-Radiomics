@echo off
REM CRC Radiomics - Virtual Environment Setup Script

echo ========================================
echo CRC Radiomics - Setting up environment
echo ========================================

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the classification script:
echo   python classification_for_delong_test.py
echo.
pause
