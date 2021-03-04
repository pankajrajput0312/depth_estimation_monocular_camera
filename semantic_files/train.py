#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:10:31 2021

@author: anunay
"""

from libraries import *
from resnet_model import ResNet50
from semantic_model import semantic_model
from losses import combined_semantic_losses
from data_utils import DataGenerator
#Custom created Resnet Model.
res_model = ResNet50(input_shape = (240, 320, 3))
#Resnet model having 'imagenet' weights 
res_model_for_weights = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(240,320,3), pooling=None, 
)
#Setting weights of Resnet Model('imagenet') to the custom created resnet model
for i in range(1,len(model.layers)):
    try:
        weights = model2.layers[i].get_weights()
        model.layers[i].set_weights(weights)
        print(i)
    except Exception as e:
        print("pass",i)
        pass

semantic_model = semantic_model(res_model.output)

#Import Input and Output data from our dataset
#Selecting Semantic Segmented Labels corresponding to our training images.
img_path = input("Enter training image's path : ")
label_path = input("Enter Segmented image's path : ")
x_data = []
y_data = []
for folder in os.listdir(path_img):
    folder_path = os.path.join(path_img,folder)
    for img in os.listdir(folder_path):
        img_name = folder_path + '/' + img
        label_name = path_label + '/' + folder + '/' + img
        x_data.append(img_name)
        y_data.append(label_name)
x_data = np.array(x_data)
y_data = np.array(y_data)

#We have taken our dataset in batch size of 4 i.e. training 4 images at a time
#Used DataGenerator for the above task
train_data = DataGenerator(x_data, y_data, batch_size=4, dim=(320,240))

loss = combined_semantic_losses(y_true, y_pred)

#ADAM optimizer is used which is quite computationally efficient.
optimizer=Adam(learning_rate=0.0001)

#Callbacks are used for saving Weights and Logs of epochs that ran during training only.
weight_path = input("Enter path where weights are to be stored : ")
logs_path = input("Enter path where logs are to be stored : ")
callbacks = [tf.keras.callbacks.ModelCheckpoint(weight_path + '/' + "train_{epoch}.tf", verbose = 1,
                                                save_weights_only=True), 
             TensorBoard(logs_path)]

#Compiling model
semantic_model.compile(optimizer = optimizer, loss = combined_semantic_losses)

#Training starts from here
number_of_epoch = int(input("Enter number of epochs : "))
semantic_model.fit(train_data, number_of_epoch, callbacks = callbacks, verbose=1)