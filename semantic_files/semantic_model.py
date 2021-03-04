#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:56:54 2021

@author: anunay
"""

def semantic_model(model_output):

    '''
      make semantic_model after resnet50 model( feature extractor ) 
        1. increasing size(double) of previous layer output using bilinear upsampling 2d layer 
        2. adding convolution layer after upsamling layer
        3. repeat step 2nd and step 3rd 4 times 
    ''' 
#     x1_depth=tf.keras.layers.UpSampling2D(size=(2,2) , data_format=None, interpolation='bilinear',name='upsampling1_d')(model_output)
    f1_semantic=tf.keras.layers.Conv2DTranspose(512,(2,2),name='conv1_s',activation='relu', strides = 2)(model_output)
    
    
#     x2_depth=tf.keras.layers.UpSampling2D(size= (2,2),data_format=None,interpolation='bilinear',name='upsampling2_d')(f1_depth)
    f2_semantic=tf.keras.layers.Conv2DTranspose(128,(2,2), strides = 2, name='conv2_s',activation='relu')(f1_semantic)

#     x3_depth=tf.keras.layers.UpSampling2D(size=(2,2),data_format=None,interpolation='bilinear',name='upsampling3_d')(f2_depth)
    f3_semantic=tf.keras.layers.Conv2DTranspose(64,(2,2),strides = 2,name='conv3_s',activation='relu')(f2_semantic)

#     x4_depth=tf.keras.layers.UpSampling2D(size= (2,2), data_format=None, interpolation='bilinear',name='upsampling4_d')(f3_depth)
    f4_semantic=tf.keras.layers.Conv2DTranspose(3,(2,2),strides = 2, name="conv4_s",activation='relu')(f3_semantic)



    '''
      Apply semi dense up skip connection between f_in, f1, f2, f3 layer
      f_out=h(f_in)+sumation_of_(h(f(i))) where i belong to (1,n)
    '''
    #add_shape  of f4 output
    f4_semantic_shape_x_y = (240,320)
    f4_semantic_filters=f4_semantic.shape[3]

    # BASE INPUT LAYER UPSAMPLING
    bilinear_upsampling_in = tf.keras.layers.experimental.preprocessing.Resizing(f4_semantic_shape_x_y[0], f4_semantic_shape_x_y[1], interpolation='bilinear')(model_output)
    f_semantic_out_in=tf.keras.layers.Conv2D(f4_semantic_filters,(1,1),name='resize_filter_conv_in',activation='relu')(bilinear_upsampling_in)

    # FIRST LAYER BILINEAR UPsAMPLING
    bilinear_upsampling_1 = tf.keras.layers.experimental.preprocessing.Resizing(f4_semantic_shape_x_y[0], f4_semantic_shape_x_y[1], interpolation='bilinear')(f1_semantic)
    f_semantic_out1 = tf.keras.layers.Conv2D(f4_semantic_filters,(1,1),name='resize_filter_conv1',activation='relu')(bilinear_upsampling_1)

    #SECOND LAYER BILINEAR UPSAMPLING
    bilinear_upsampling_2 = tf.keras.layers.experimental.preprocessing.Resizing(f4_semantic_shape_x_y[0], f4_semantic_shape_x_y[1], interpolation='bilinear')(f2_semantic)
    f_semantic_out2 = tf.keras.layers.Conv2D(f4_semantic_filters,(1,1),name='resize_filter_conv2',activation='relu')(bilinear_upsampling_2)

    # 3rd LAYER UPSAMPLING
    bilinear_upsampling_3 = tf.keras.layers.experimental.preprocessing.Resizing(f4_semantic_shape_x_y[0], f4_semantic_shape_x_y[1], interpolation='bilinear')(f3_semantic)
    f_semantic_out3 = tf.keras.layers.Conv2D(f4_semantic_filters,(1,1),name='resize_filter_conv3',activation='relu')(bilinear_upsampling_3)
    # bilinear_upsampling_3 = tf.keras.layers.experimental.preprocessing.Resizing(f4.shape[1], f4.shape[2], interpolation='bilinear')(f3)
    # f_out3 = tf.keras.layers.Conv2D(f4_filters,(1,1),name='resize_filter_conv3')(bilinear_upsampling_3)

    #F_out LAYER UPSAMPLING
    bilinear_upsampling_4 = tf.keras.layers.experimental.preprocessing.Resizing(f4_semantic_shape_x_y[0], f4_semantic_shape_x_y[1], interpolation='bilinear')(f4_semantic)
    f_semantic_out4 = tf.keras.layers.Conv2D(f4_semantic_filters,(1,1),name='resize_filter_conv4', activation='relu')(bilinear_upsampling_4)

    
    ''' Applying formulae for semi dense up skip connection '''
    f_semantic_out_final1=tf.keras.layers.Add()([f_semantic_out_in,f_semantic_out1,f_semantic_out2,f_semantic_out3,f_semantic_out4])
    f_semantic_out_final = tf.keras.layers.Activation('sigmoid')(f_semantic_out_final1)
    # put the code below out of this function
    ''' Building semantic model from Vgg16 to f_out'''
    from tensorflow.keras import Model    
    semantic_model=Model(inputs=model.input,outputs=f_semantic_out_final)
    semantic_model.summary()
    return semantic_model


