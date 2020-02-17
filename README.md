# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

O# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The following project harnesses the power of deep neural networks to clone the behaviour of a car, such to be deemed "Autonomous". By collecting data through driving the car manually in a simulator, images as data and steering angle measurements as classifications were used to construct the CNN.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
| File Name | Description |
| ------------- | ------------- |
| CarND-BehaviouralCloning-P3.ipynb | ipython notebook for data preprocessing and building the neural newtork |
| model.h5  | saved model by keras  |
| drive.py  | communicate with simulator and use saved model to predict steering angle  |
| drivePOV.mp4  | saved video of simulation run  |
| TrackOne.mov | Screen captur of simulation run |


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The CarND-BehaviouralCloning-P3.ipynb file contains the code for training and saving the convolution neural network, since Google Colab offers a free GPU. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model was derived from NVIDIA's neural network as seen below and referenced from [Nvidia's Published Documentary](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf):

![alt text](https://github.com/navoshta/behavioral-cloning/raw/master/images/model.png)

Now here the CNN that was used in this project.

![alt text](nn.svg)
![alt text](model.png)



The two architectures are relatively similar, with an addition to changes in filter sizes, an additional dropout layer and modifications to the dense layers.


The model includes ELU layers to introduce nonlinearity and the data is normalized in the model using a Keras by dividing each pixel value by 255. 

#### 2. Reducing Overfitting / Data Collection / Data Augmentation

Overfitting was targeted through:
1. Dropout Layers
2. Data Augmentaion
3. Training / Validation Data Split

The model contained dropout layers in order to reduce overfitting, as mentioned with model's architecture.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training and validation data was split (80/20) in addition to data augmentation practices.

Data Augmention techniques included:
1. Altering Brightness - to adapt to different lighting
2. Mirror the images on the Y-Axis - to prevent driving bias
3. Shifting Image via Vertical/Horizontal Translation
4. Scale Image

Images:

![alt text](augmentaion.png)

These technique helped the model generalize to new data.

Lastly, the images were preprocessed by converting the color channel from RGB to YUV (which highlighted details more effectively) and were normalized.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

The model was trained using a image generator which produced training and validation images parallel to the model being trained. Of the 20 epochs that the model trained on, the generator produced 15000 training images per epoch. The training loss diverged to around a value around 0.05.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. It was import to view the distribution of the steering measurements collected. Here is the total data after training.

![alt text](datadistribution.png)

I noticed the model would favor center-lane driving too much. As a result, around 80% of the straight driving data was ignored. This was necessary or else the model would second guess turning after only training on major of straight driving data. 

Thus the final dataset: 

![alt text](datadistributionafter.png)

