#!/bin/bash

python -m pip install virtualenv
echo "installed virtualenv"
python -m venv .venv
echo "venv created"
source .venv/bin/activate
echo "venv enabled"

python -m pip install -r pytorch_requirements.txt
echo "pytorch requirements installed"
python -m pip install -r requirements.txt
echo "project requirements installed"
pip install -e .
echo "project installed"