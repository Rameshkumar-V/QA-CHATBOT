@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m pip install -r requirements-desktop.txt
if errorlevel 1 goto :fail
py -3.11 -m PyInstaller --noconfirm --clean --onedir --name PersonalDataAgent ^
 --collect-all streamlit --collect-all langchain --collect-all langchain_core ^
 --collect-all langchain_openai ^
 --collect-all keyring --collect-all pandas --collect-all pyarrow --collect-all altair ^
 --hidden-import openpyxl --hidden-import xlrd ^
 --add-data "app.py;." --add-data "agent;agent" --add-data "services;services" ^
 desktop\launcher.py
if errorlevel 1 goto :fail
echo Build completed: dist\PersonalDataAgent\PersonalDataAgent.exe
pause
exit /b 0
:fail
echo Build failed. Review errors above.
pause
exit /b 1
