@echo off
setlocal EnableDelayedExpansion

echo Voulez-vous creer un nouvel environnement conda? (o/n)
set /p create_env=

if /i "!create_env!"=="o" (
    conda env create -n MIR --file "./requirements/requirements.yml"
    call conda activate MIR
    pip install -r "./requirements/requirements.txt"
) else (
    call conda activate MIR
)

cd src
python interface.py
pause