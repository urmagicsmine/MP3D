# Revisiting 3D Context Modeling with Supervised Pre-training for Universal Lesion Detection on CT Slices
This is an implementation of MICCAI 2020 paper [Revisiting 3D Context Modeling with Supervised Pre-training for Universal Lesion Detection on CT Slices](https://arxiv.org/pdf/2012.08770.pdf).

## Installation
This code is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please see it for installation.


## Data preparation
Download DeepLesion dataset [here](https://nihcc.app.box.com/v/deeplesion).

We provide coco-style json annotation files converted from DeepLesion. Please download json files [here](https://github.com/urmagicsmine/MVP-Net/tree/master/data/DeepLesion/annotation), unzip Images_png.zip and make sure to put files as following sturcture:

```
data
  ├──DeepLesion
        ├── annotations
        │   ├── deeplesion_train.json
        │   ├── deeplesion_test.json
        │   ├── deeplesion_val.json
        └── Images_png
              └── Images_png
               │    ├── 000001_01_01
               │    ├── 000001_03_01
               │    ├── ...
```

## Pre-trained Model
We provide models pre-trained on COCO dataset which can be used for different 3D medical image detection.

The pre-trained MP3D63 model can be downloaded from [BaiduYun](https://pan.baidu.com/s/1zMyw2tcPY1q0SPRpZKSKbQ)(verification code: bbrc). 

## Training
To train MP3D & P3d model on deeplesion dataset, run:

```
bash tools/dist_train.sh configs/deeplesion/mp3d_groupconv.py 8
bash tools/dist_train.sh configs/deeplesion/p3d.py 8
```

## Contact
If you have questions or suggestions, please open an issue here.
