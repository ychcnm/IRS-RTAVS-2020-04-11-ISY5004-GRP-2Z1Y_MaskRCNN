# IRS-RTAVS-2020-04-11-ISY5004-GRP-2Z1Y_MaskRCNN

## Requirements
Python 3.6, Tensorflow-gpu 1.9.0, Keras 2.2.0, cuda 9.0 and other common packages listed in `requirements.txt`

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)


## Installation Guideline
1. Open `Anaconda Prompt`
2. Append the channel conda-forge into your conda configuration.

	```bash
	conda config --append channels conda-forge
```
3. Create a new virtual environment `Maskrcnn` or install additional packages in your own environment
	
	```bash
	conda create -n Maskrcnn python=3.6 pip
```
4. Activate the environment `Maskrcnn`

	```bash
	conda activate Maskrcnn
```
5. Clone the CA repository

	```bash
	git clone https://github.com/ychcnm/IRS-RTAVS-2020-04-11-ISY5004-GRP-2Z1Y_MaskRCNN.git
	cd IRS-RTAVS-2020-04-11-ISY5004-GRP-2Z1Y_MaskRCNN
``` 

6. Clone the Mask-Rcnn repository

	```bash
	git clone https://github.com/matterport/Mask_RCNN.git
	cd Mask_RCNN
``` 

7. Install Mask_RCNN into env library

	```bash
	python setup.py install
	cd ../
``` 

8. install the library in `requirements.txt` and COCO API

	```bash
	pip install -r requirements.txt
	pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
``` 