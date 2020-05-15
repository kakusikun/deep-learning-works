# deep-learning-works
Work in deep learning via Pytorch.

## Installation
* [CUDA 9.2](https://developer.nvidia.com/cuda-toolkit-archive)
* [cudnn 7.3.1](https://developer.nvidia.com/rdp/cudnn-archive)
* [inplace-abn](https://github.com/mapillary/inplace_abn)
* Pytorch 1.4.0, torchvision 0.5.0
    ```
    pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    ```
* Cython
    ```
    pip install Cython
    ``` 
* python packages in requirement.txt

## Overview
Fork from following task
### Person Re-identification
* ~~[(SSG) Self-similarity Grouping: A Simple Unsupervised Cross Domain Adaptation Approach for Person Re-identification](https://arxiv.org/abs/1811.10144)~~
* ~~[(MAR) Unsupervised Person Re-identification by Soft Multilabel Learning](https://arxiv.org/abs/1903.06325)~~
* [(BNNeck) Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://arxiv.org/abs/1903.07071)
* [(OSNet) Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/abs/1905.00953)
* [(OSNet-IAP) Building Computationally Efficient and Well-Generalizing Person Re-Identification Models with Metric Learning](https://arxiv.org/abs/2003.07618)
* [(OSNet-AIN) Learning Generalisable Omni-Scale Representations for Person Re-Identification](https://github.com/KaiyangZhou/deep-person-reid/blob/6081989d7eb7577f56a4da523da4fc47ea9fcd32/torchreid/models/osnet_ain.py)

### Pedestrian Attribute Recognition
* Multitask Learning using fully connected layer

### Object Detection
* [(CenterNet) Objects as Points](https://github.com/xingyizhou/CenterNet) + [(CornerNet-Lite) CornerNet-Lite: Efficient Keypoint Based Object Detection](https://github.com/princeton-vl/CornerNet-Lite/tree/master)

### Clothing Detection, Keypoints
* [(DeepFashion2) A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images](https://github.com/switchablenorms/DeepFashion2)

## Result
*working on it*