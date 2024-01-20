# linkedin-work-experience-classification
Classify work experience from Linkedin using a few-shot learning AI model

## Install

1. Install Python (tested on Python 3.10.6)
2. Install virtualenv if you don't already have it: `python -m pip install virtualenv`
3. Create a virtual environment: `python -m venv .venv`
4. Activate this environment: `source .venv/bin/activate` (or `".venv/Scripts/activate.bat"` on Windows)
5. Install the dependencies: `python -m pip install -r requirements.txt`


## How to use GPU

1. Install CUDA 11.2.2 at https://developer.nvidia.com/cuda-11.2.2-download-archive
2. Install cuDNN 8.1.1 for CUDA 11.2 at https://developer.nvidia.com/rdp/cudnn-archive (move the archive content into your CUDA installation folder)
3. If it still doesn't work:
	- Add to your PATH env variable:
		- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib
		- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include
	- Create a new env variable named CUDNN, and set its value to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2

## Use

1. Run `jupyter notebook` at the root of this project
2. Go to the window that has been opened or open your web browser and go to http://localhost:8888 (or the URL that is given in the terminal)
3. Open on this web page the folder notebooks/ and open the notebook that you want to read
