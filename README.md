# linkedin-work-experience-classification
Classify work experience from Linkedin using a few-shot learning AI model

## Install

1. Install Python (tested on Python 3.10.6)
2. Install virtualenv if you don't already have it: `python -m pip install virtualenv`
3. Create a virtual environment: `python -m venv .venv`
4. Activate this environment: `source .venv/bin/activate` (or `".venv/Scripts/activate.bat"` on Windows)
5. Install the dependencies for PyTorch: `python -m pip install -r pytorch_requirements.txt` **(Warning: please read the section "How to use the GPU" before)**
6. Install the other dependencies: `python -m pip install -r requirements.txt`

## How to use the GPU

### Nvidia

For both Windows and Linux:

1. Install CUDA 12.1.1 at https://developer.nvidia.com/cuda-toolkit-archive
2. Go to the step 5 of the section "Install" just above

### AMD

For Linux only (not available on Windows):

1. Install CUDA 5.7.0 Follow the instructions here: https://rocm.docs.amd.com/en/docs-5.7.0/deploy/linux/os-native/install.html
2. Change the first line of pytorch_requirements.txt to `--index-url https://download.pytorch.org/whl/rocm5.7`
3. Go to the step 5 of the section "Install" just above

## How to use the CPU (not recommended)

1. Change the first line of pytorch_requirements.txt to `--index-url https://download.pytorch.org/whl/cpu`
2. Go to the step 5 of the section "Install" just above

## Use

1. Run `jupyter notebook` at the root of this project
2. Go to the window that has been opened or open your web browser and go to http://localhost:8888 (or the URL that is given in the terminal)
3. Open on this web page the folder notebooks/ and open the notebook that you want to read
