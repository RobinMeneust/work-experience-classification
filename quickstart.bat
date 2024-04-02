@echo off

call python -m pip install virtualenv
echo "installed virtualenv"
call python -m venv .venv
echo "venv created"
call ".venv/Scripts/activate.bat"
echo "venv enabled"

call python -m pip install -r pytorch_requirements.txt
echo "pytorch requirements installed"
call python -m pip install -r requirements.txt
echo "project requirements installed"
call pip install -e .
echo "project installed"