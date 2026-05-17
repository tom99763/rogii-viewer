@echo off
REM Build a single-file Windows .exe of the viewer.
REM Run this from a Windows shell (not WSL) with a Windows Python install.

setlocal
cd /d %~dp0\..

python -m pip install --upgrade pip
python -m pip install -r viewer\requirements.txt
python -m pip install pyinstaller

REM PyInstaller emits dist\ROGIIViewer.exe
python -m PyInstaller ^
  --noconfirm ^
  --onefile ^
  --windowed ^
  --name ROGIIViewer ^
  --collect-submodules pyqtgraph ^
  --collect-submodules PySide6 ^
  viewer\__main__.py

echo.
echo Built: dist\ROGIIViewer.exe
endlocal
