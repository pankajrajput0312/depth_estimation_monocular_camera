#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:31:09 2021

@author: anunay, pankaj
"""

class DataGenerator(Sequence):
    def __init__(self, x_data, y_data,
                 batch_size=32, dim=(320,240),
                 shuffle=True):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.dim = dim


    def __len__(self):
        return int(np.floor(len(self.x_data) / self.batch_size))


    def __getitem__(self, index):
        start_index = index * self.batch_size
        x_train = []
        y_train = []
        i = start_index - 1
        while len(x_train) < self.batch_size:
            try:
                
                img = cv2.imread(self.x_data[i % len(self.x_data)])
                img = cv2.resize(img,(320,240))
                img = np.array(img, dtype = np.float32)
                img = img / 255.0
                y_img=cv2.imread(self.y_data[i%len(self.y_data)])
                y_img=cv2.resize(y_img, (320,240))
                y_img=np.array(y_img, dtype = np.float32)
                y_img = y_img / 255.0
#                 y_img = cv2.cvtColor(y_img,cv2.COLOR_BGR2GRAY)
                x_train.append(img)
                y_train.append(y_img)
                i += 1

            except Exception as err:
                print(err)
                continue
            
        x_train = np.array(x_train)
        y_train = np.array(y_train)
  
        return x_train, y_train
