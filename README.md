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

We use VGG-16 (shown below) as our architecture because it has a very good result on Image-net classification and is not very difficult to train. Therefore, we believe that it can extract features from an image very well and only have three dense layers including the output layer whose number of neurons is two (i.e., x and y coordinates).

![image](https://user-images.githubusercontent.com/23422272/188261241-12e52a7f-966e-4279-86bd-02de91110f6a.png)


## Training

We use the Adam optimizer with the learning rate of 0.0001 and the batch size of 10. We also set up the number of epochs to 1000. However, it will never reach 1000. Furthermore, we apply the ReduceOnPlateau callback to reduce the learning rate when the optimizer cannot improve the validation loss and also the EarlyStopping callback to stop the training when it is converge with respect to the validation loss so that the model will not be overfitting the training samples. In addition, we apply other three regularization techniques as follows:

1) Batchnormalization: Since VGG-16 is a deep network, we need to limit the values after some layers. We decide to place this batchnormalization layer after the last convolutional layer.
2) Dropout: This technique is very popular to use for regulazation because it can avoid the model to strongly rely only on a few features by dropping some parameters out during the training. We placee this dropout layer after the second dense layer.
3) L2 regularization: This is another popular technique to regularize the model because it will force the training to reduce the values of the parameters such that the model will not strongly rely on a few features. We also found that with the balancers of 0.01 and 0.001, the model is underfitting since it only focuses on minimizing the values of the parameters. Nonetheless, when we set the balancer to 0.00001, the model performs very well on both the training and validation samples. Therefore, we use this value for the training.

## Result

According to the results during training below, the model start beinig overfitting the training samples after epoch 130 since the training loss is going down while the validation loss is already converge. Interestingly, despite the fact that the model is converge around epoch 130, according to the figure on the right, MAE starts being stable around epoch 90. 

![image](https://user-images.githubusercontent.com/23422272/188262494-89698811-b5f2-4241-9c73-7179834334d7.png)
![image](https://user-images.githubusercontent.com/23422272/188262796-2ea17dd5-6c47-40e4-9369-7196d05f0031.png)

Additionally, we also evaluate the trained model by using the accuracy. Note that we consider the model can correctly predict the phone's location when the distance between the predicted location and the ground-truth location is less than 0.05. The accuracy on the training samples is 100%. This result is very good. However, we do not know if it is overfitting the training samples. Hence, we also measure the accuracy on the validation samples. The least accuracy on the validation samples that the model has achieved is around 61%, and the highest one is around 88%. That is, our model should pass the evaluation  critiria which only requires the model to achieve the accuracy of 50% on the unseen data and 70% on the training data. 

## References

1) https://idiotdeveloper.com/vgg16-unet-implementation-in-tensorflow/
