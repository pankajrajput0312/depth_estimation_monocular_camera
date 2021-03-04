#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:54:35 2021

@author: anunay
"""

def depth_model(model_output):

    '''
      make depth model after resnet50 model( feature extractor ) 
        1. increasing size(double) of previous layer output using bilinear upsampling 2d layer 
        2. adding convolution layer after upsamling layer
        3. repeat step 2nd and step 3rd 4 times 
    ''' 
    f1_depth=tf.keras.layers.Conv2DTranspose(512,(2,2),name='conv1_d',activation='relu', strides = 2)(model.output)
    
    
    f2_depth=tf.keras.layers.Conv2DTranspose(128,(2,2), strides = 2, name='conv2_d',activation='relu')(f1_depth)

    f3_depth=tf.keras.layers.Conv2DTranspose(64,(2,2),strides = 2,name='conv3_d',activation='relu')(f2_depth)

    f4_depth=tf.keras.layers.Conv2DTranspose(1,(2,2),strides = 2, name="conv4_d",activation='relu')(f3_depth)



    '''
      Apply semi dense up skip connection between f_in, f1, f2, f3 layer
      f_out=h(f_in)+sumation_of_(h(f(i))) where i belong to (1,n)
    '''
    #add_shape  of f4 output
    f4_depth_shape_x_y = (240,320)
    f4_depth_filters=f4_depth.shape[3]

    # BASE INPUT LAYER UPSAMPLING
    bilinear_upsampling_in = tf.keras.layers.experimental.preprocessing.Resizing(f4_depth_shape_x_y[0], f4_depth_shape_x_y[1], interpolation='bilinear')(model_output)
    f_depth_out_in=tf.keras.layers.Conv2D(f4_depth_filters,(1,1),name='resize_filter_conv_in',activation='relu')(bilinear_upsampling_in)

    # FIRST LAYER BILINEAR UPsAMPLING
    bilinear_upsampling_1 = tf.keras.layers.experimental.preprocessing.Resizing(f4_depth_shape_x_y[0], f4_depth_shape_x_y[1], interpolation='bilinear')(f1_depth)
    f_depth_out1 = tf.keras.layers.Conv2D(f4_depth_filters,(1,1),name='resize_filter_conv1',activation='relu')(bilinear_upsampling_1)

    #SECOND LAYER BILINEAR UPSAMPLING
    bilinear_upsampling_2 = tf.keras.layers.experimental.preprocessing.Resizing(f4_depth_shape_x_y[0], f4_depth_shape_x_y[1], interpolation='bilinear')(f2_depth)
    f_depth_out2 = tf.keras.layers.Conv2D(f4_depth_filters,(1,1),name='resize_filter_conv2',activation='relu')(bilinear_upsampling_2)

    # 3rd LAYER UPSAMPLING
    bilinear_upsampling_3 = tf.keras.layers.experimental.preprocessing.Resizing(f4_depth_shape_x_y[0], f4_depth_shape_x_y[1], interpolation='bilinear')(f3_depth)
    f_depth_out3 = tf.keras.layers.Conv2D(f4_depth_filters,(1,1),name='resize_filter_conv3',activation='relu')(bilinear_upsampling_3)
    # bilinear_upsampling_3 = tf.keras.layers.experimental.preprocessing.Resizing(f4.shape[1], f4.shape[2], interpolation='bilinear')(f3)
    # f_out3 = tf.keras.layers.Conv2D(f4_filters,(1,1),name='resize_filter_conv3')(bilinear_upsampling_3)

    #F_out LAYER UPSAMPLING
    bilinear_upsampling_4 = tf.keras.layers.experimental.preprocessing.Resizing(f4_depth_shape_x_y[0], f4_depth_shape_x_y[1], interpolation='bilinear')(f4_depth)
    f_depth_out4 = tf.keras.layers.Conv2D(f4_depth_filters,(1,1),name='resize_filter_conv4',activation='relu')(bilinear_upsampling_4)

    
    ''' Applying formulae for semi dense up skip connection '''
    f_depth_out=tf.keras.layers.Add()([f_depth_out_in,f_depth_out1,f_depth_out2,f_depth_out3,f_depth_out4])

    # put the code below out of this function
    ''' Building depth model from Vgg16 to f_out'''
    from tensorflow.keras import Model    
    depth_model=Model(inputs=model.input,outputs=f_depth_out)
    depth_model.summary()
    return depth_model


