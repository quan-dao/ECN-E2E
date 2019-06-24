[//]: # (Image References)

[confusion_matrix]: ./images/norm_confusion_matrix_2.png
[layers_activation]: ./images/layers_activation.png
[conceptual_model_arch]: ./images/conceptual_model_arch.png

# Intro
This project is an attemp to train a neural network to drive a car in an end-to-end fashion. This approach to autonomous driving is not knew and had been proven by NVIDIA's DAVE-2. The difference we try to make here is to help our model learn `a sequence of steering angles` rather than just one steering angle which have been done is a number of publications. 

The major benifit of learning a sequence of steering angles is that this can help deliver the two pillars of driving task, namely vehicle's steering angle and velocity. While the former can be learnt given front-facing images, there are no means to infer velocity from static images which make it impossible for a network to learnt. It can be argued that the velocity can also be learnt end-to-end given video inputs and recurrent layers which can capture input's temporal relation. However, the velicty in its essence is not meaned to learn end-by-end. The velocity is a dynamic quantity which depends on the vehicles' surronding environment (mainly moving obstacles). As a result, the velocity should be calculated based on sensory input. Motion planning literature suggest that a velocity profile can be generated by timestamping a geometric path which in case of car-like motion just a sequence of steering angles. With this in mind, if a network can learn such a sequenece, a veloicty can be produce, hence the completion of driving task. 

# Getting started
Clone the repository and install the dependencies

`pip install -r requirement.txt`

## Dataset
The dataset used to train our model is [Udaicty dataset CH2](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2). This raw form of dataset is a collection of 6 bag files which need to be preprocessed to extract camera images and associated steering angles (as well as other helpful information such as GPS coordinate). The preprocessing is done thanks to [udacity-dataset-reader](https://github.com/rwightman/udacity-driving-reader). After preprocessing, a number of folders and files are created; however, only the folder `center` and `interpolated.csv` which respectively contains the images captured by front-facing camera and detail information (e.g. timestamp, filename, steering angle, GPS coordiante).

The repository structure should be organized as following
`
./data/
      |
      |-> training_data/
                        |
                        |->center/
                                  |
                                  |->timestamp.jpeg
                        |
                        |->interpolated.csv`

## Dataset preparation
To enable learning a geometrical path, the model is shown an image of the environment in front of the car and it is meant to output a sequence of steering angles in which the first angle is directly applied to the image and each subsequence angle is 2 meters away from its predecessor. To serve this purpose, the data needs to encorporate the distance between each label. 

Furthermore, to increase the reliable of model's prediction, the training problem is formulated as a classification problem. In details, the recorded steering angle spectrum is discretized into intervals of length 2 degree. Then instead of directly predicting a steering angle, the model predicts the bin that the steering angle belong to. The advantage of this approach is that model's performance during training can also be measured by the ***accuracy*** metric, in addition to the ***cross-entroy loss***. 

These preparation phases are done in ***./data/dataset_preparation.ipynb***.

The viability of this training formulation is shown in our implementation of ``learning a steering angle` where we just replace the regressor at top of the ***DroNet*** with a classifier. This classifier reaches the accuracy of 63% on the validation set.

# Model architect
The architect used in this project (Fig.??) is adapted from [DroNet: Learning to fly by driving](https://github.com/uzh-rpg/rpg_public_dronet). The body is made of 3 ResNet layers. The original output layer, with one neurons for performing regression on steering angle and another for predicting the collision probability, is replaced by an array of classififers each comprised of one Dense layer activated by ReLU and another Dense layer acitvated by Softmax to output the one-hot representation of steering angle class.

![alt text][conceptual_model_arch]
Fig. Model architect

# Model performance
After training for 35 epochs, our model's performance is compared to the ***DroNet*** and some other model on Root Mean Square Error (RMSE) and Explained Variance (EVA) metric. Since our model employs 5 classifiers (call ***Head***) to predict 5 steering angles, those two metrics are calculated for each classifier. 

Model | RMSE | EVA
---- | ---- | ----
Constant baseline | 0.2129 | 0
DroNet | 0.1090 | 0.7370
Head_0 | 0.1101 | 0.8245
Head_1 | 0.1159 | 0.8042
Head_2 | 0.1012 | 0.8491
Head_3 | 0.1159 | 0.8020
Head_4 | 0.1050 | 0.8409

The quality of classification is shown in confusition matrix below. This matrix features a strong main diagonal which means that the majority of predicted angle classes is actually the true class.

![alt text][confusion_matrix]
Fig. Confusion matrix

In addition, in an attemp to understand what the model has learned, the activation of each ResNet block is shown below. It can be seen that the first block recognizes lane mark and car-like objects. The second block seems to segment the drivable area; and the last block learns an identical mapping. 

![alt text][layers_activation]
Fig. Activation of each ResNet block