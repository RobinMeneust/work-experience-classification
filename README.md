# linkedin-work-experience-classification

Classify work experience from Linkedin using a few-shot learning AI model

## How to use the GPU

### Nvidia

For both Windows and Linux:

Install CUDA 12.1.1 at https://developer.nvidia.com/cuda-toolkit-archive

### AMD

For Linux only (not available on Windows).

**Note**: This isn't garanteed to work since we couldn't test it.


1. Install ROCm 5.7.0. Follow the instructions here: https://rocm.docs.amd.com/en/docs-5.7.0/deploy/linux/os-native/install.html
2. Change the first line of pytorch_requirements.txt to `--index-url https://download.pytorch.org/whl/rocm5.7`

## How to use the CPU (not recommended)

Change the first line of pytorch_requirements.txt to `--index-url https://download.pytorch.org/whl/cpu`

## Install

First you need to install Python (tested on Python 3.10.6)

### Using the quickstart script

Run `quickstart.bat` on Windows and `quickstart.sh` on Linux

### Manually

1. Install virtualenv if you don't already have it: `python -m pip install virtualenv`
2. Create a virtual environment: `python -m venv .venv`
3. Activate this environment: `source .venv/bin/activate` (or `".venv/Scripts/activate.bat"` on Windows)
4. Install the dependencies for PyTorch: `python -m pip install -r pytorch_requirements.txt` **(Warning: please read the section "How to use the GPU" before)**
5. Install the other dependencies: `python -m pip install -r requirements.txt`
6. Setup the project packages using `python -m pip install -e .`

## Usage

1. Run `jupyter notebook` at the root of this project
2. Go to the window that has been opened or open your web browser and go to http://localhost:8888 (or the URL that is given in the terminal)
3. Open on this web page the folder notebooks/ and open the notebook that you want to read

## Notes

The results generated by the benchmark notebook are saved in the `results/` folder in JSON format

## Authors

- Estéban DARTUS
- Nino HAMEL
- Robin MENEUST
- Jérémy SAELEN
- Mathis TEMPO