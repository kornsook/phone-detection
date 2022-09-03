# Phone Detection

This work is to train a machine learning to detect a phone in an image by given a dataset of the images. The experiment in this work has been conducted in Ubuntu version 20.04.4 LTS with Intel® Xeon(R) CPU E5-1630 v4 @ 3.70GHz × 8 processor and NVIDIA Corporation GP104 [GeForce GTX 1080] GPU. 

This repository mainly includes 4 files:

1) train.ipynb: This is a notebook for training a machine learning model.
2) train_phone_finder.py: This is a script training a machine learning model.
3) find_phone.py: This is a script to use the machine learning model trained by one of the previous files to locate a phone in a given image.
4) requirement.txt: This is to install required Python modules for the notebook and scripts.

## Data

We have 129 images as our dataset, each of which contains one phone. We splited this dataset into 104 training samples and 25 validation samples. Then, we scaled the images into 70% of its size so that its batches can fit our GPU, and the parameters of the model is significantly decreased. If you have a higher-performance GPU, you may not need to scale the images. 

## Model

efdsgds
![image](https://user-images.githubusercontent.com/23422272/188261241-12e52a7f-966e-4279-86bd-02de91110f6a.png)

## Training

## Result
