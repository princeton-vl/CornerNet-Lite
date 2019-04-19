# CornerNet-Lite: Training, Evaluation and Testing Code
Code for reproducing results in the following paper:

[**CornerNet-Lite: Efficient Keypoint Based Object Detection**](https://arxiv.org/abs/1904.08900)  
Hei Law, Yun Teng, Olga Russakovsky, Jia Deng  
*arXiv:1904.08900* 

## Getting Started
### Software Requirement
- Python 3.7
- PyTorch 1.0.0
- CUDA 10
- GCC 4.9.2 or above

### Installing Dependencies
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list `conda_packagelist.txt`.
```
conda create --name CornerNet_Lite --file conda_packagelist.txt --channel pytorch
```

After you create the environment, please activate it.
```
source activate CornerNet_Lite
```

### Compiling Corner Pooling Layers
Compile the C++ implementation of the corner pooling layers. (GCC4.9.2 or above is required.)
```
cd <CornerNet-Lite dir>/core/models/py_utils/_cpools/
python setup.py install --user
```

### Compiling NMS
Compile the NMS code which are originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx).
```
cd <CornerNet-Lite dir>/core/external
make
```

### Downloading Models
In this repo, we provide models for the following detectors:
- [CornerNet-Saccade](https://drive.google.com/file/d/1MQDyPRI0HgDHxHToudHqQ-2m8TVBciaa/view?usp=sharing)
- [CornerNet-Squeeze](https://drive.google.com/file/d/1qM8BBYCLUBcZx_UmLT0qMXNTh-Yshp4X/view?usp=sharing)
- [CornerNet](https://drive.google.com/file/d/1e8At_iZWyXQgLlMwHkB83kN-AN85Uff1/view?usp=sharing)

Put the CornerNet-Saccade model under `<CornerNet-Lite dir>/cache/nnet/CornerNet_Saccade/`, CornerNet-Squeeze model under `<CornerNet-Lite dir>/cache/nnet/CornerNet_Squeeze/` and CornerNet model under `<CornerNet-Lite dir>/cache/nnet/CornerNet/`. (\* Note we use underscore instead of dash in both the directory names for CornerNet-Saccade and CornerNet-Squeeze.)

Note: The CornerNet model is the same as the one in the original [CornerNet repo](https://github.com/princeton-vl/CornerNet). We just ported it to this new repo.

### Running the Demo Script
After downloading the models, you should be able to use the detectors on your own images. We provide a demo script `demo.py` to test if the repo is installed correctly.
```
python demo.py
```
This script applies CornerNet-Saccade to `demo.jpg` and writes the results to `demo_out.jpg`.

In the demo script, the default detector is CornerNet-Saccade. You can modify the demo script to test different detectors. For example, if you want to test CornerNet-Squeeze:
```python
#!/usr/bin/env python

import cv2
from core.detectors import CornerNet_Squeeze
from core.vis_utils import draw_bboxes

detector = CornerNet_Squeeze()
image    = cv2.imread("demo.jpg")

bboxes = detector(image)
image  = draw_bboxes(image, bboxes)
cv2.imwrite("demo_out.jpg", image)
```

### Using CornerNet-Lite in Your Project
It is also easy to use CornerNet-Lite in your project. You will need to change the directory name from `CornerNet-Lite` to `CornerNet_Lite`. Otherwise, you won't be able to import CornerNet-Lite.
```
Your project
│   README.md
│   ...
│   foo.py
│
└───CornerNet_Lite
│
└───directory1
│   
└───...
```

In `foo.py`, you can easily import CornerNet-Saccade by adding:
```python
from CornerNet_Lite import CornerNet_Saccade

def foo():
    cornernet = CornerNet_Saccade()
    # CornerNet_Saccade is ready to use

    image  = cv2.imread('/path/to/your/image')
    bboxes = cornernet(image)
```

If you want to train or evaluate the detectors on COCO, please move on to the following steps.

## Training and Evaluation

### Installing MS COCO APIs
```
mkdir -p <CornerNet-Lite dir>/data
cd <CornerNet-Lite dir>/data
git clone git@github.com:cocodataset/cocoapi.git coco
cd <CornerNet-Lite dir>/data/coco/PythonAPI
make install
```

### Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CornerNet-Lite dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CornerNet-Lite dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

To train and evaluate a network, you will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `<CornerNet-Lite dir>/configs/`. Each configuration file should have a corresponding model file in `<CornerNet-Lite dir>/core/models/`. i.e. If there is a `<model>.json` in `<CornerNet-Lite dir>/configs/`, there should be a `<model>.py` in `<CornerNet-Lite dir>/core/models/`. There is only one exception which we will mention later.

### Training and Evaluating a Model
To train a model:
```
python train.py <model>
```

We provide the configuration files and the model files for CornerNet-Saccade, CornerNet-Squeeze and CornerNet in this repo. Please check the configuration files in `<CornerNet-Lite dir>/configs/`.

To train CornerNet-Saccade:
```
python train.py CornerNet_Saccade
```
Please adjust the batch size in `CornerNet_Saccade.json` to accommodate the number of GPUs that are available to you.

To evaluate the trained model:
```
python evaluate.py CornerNet_Saccade --testiter 500000 --split <split>
```

If you want to test different hyperparameters during evaluation and do not want to overwrite the original configuration file, you can do so by creating a configuration file with a suffix (`<model>-<suffix>.json`). There is no need to create `<model>-<suffix>.py` in `<CornerNet-Lite dir>/core/models/`.

To use the new configuration file:
```
python evaluate.py <model> --testiter <iter> --split <split> --suffix <suffix>
```

We also include a configuration file for CornerNet under multi-scale setting, which is `CornerNet-multi_scale.json`, in this repo. 

To use the multi-scale configuration file:
```
python evaluate.py CornerNet --testiter <iter> --split <split> --suffix multi_scale
