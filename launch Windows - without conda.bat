@echo off
setlocal EnableDelayedExpansion

echo Voulez-vous telecharger les dependances? (o/n)
set /p download_dependencies=

if /i "!download_dependencies!"=="o" (
    pip install -r "./requirements/requirements.txt"
)

cd src
python interface.py
pause
