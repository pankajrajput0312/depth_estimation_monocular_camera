#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:50:44 2021

@author: anunay
"""

def depth_aware_loss(y_true, y_pred):
    y_true = tf.conver_to_tensor(y_true,dtype=tf.float32)
    y_pred = tf.conver_to_tensor(y_pred, dtype= tf.float32)
    i = tf.image.convert_image_dtype(y_true, dtypoe = tf.float32)
    #lamba d :
    lambda_d = 1 - ((tf.math.minimum(tf.math.log(y_pred), tf.math.log(y_true))) / (tf.math.maximum(tf.math.log(y_pred), tf.math.log(y_true))))
    loss1 = tf.reduce_mean((i + lambda_d) * tf.math.abs(y_true - y_pred))
    
    return loss1

def depth_gradient_loss(y_true, y_pred):

    y_true = tf.image.convert_image_dtype(y_true, dtype=tf.float32)
    y_true = tf.expand_dims(y_true, axis = 3)
    y_pred = tf.image.convert_image_dtype(y_pred, dtype = tf.float32)
    y_pred = tf.expand_dims(y_pred, axis = 3)
    #y_true edges

    sobel_true = tf.image.sobel_edges(y_true)
    sobel_true_h = sobel_true[0, :, :, :, 0]
    sobel_true_w = sobel_true[0, :, :, :, 1]
    #y_pred edges
    sobel_pred = tf.image.sobel_edges(y_pred)
    sobel_pred_h = sobel_pred[0, :, :, :, 0]
    sobel_pred_w = sobel_pred[0, :, :, :, 1]
    loss_depth = tf.reduce_mean(tf.math.abs(sobel_pred_h - sobel_true_h) + tf.math.abs(sobel_pred_w - sobel_true_w))
    
    return loss_depth

def combined_depth_loss(y_true,y_pred):
    
    loss1=depth_aware_loss(y_true,y_pred)
    loss2 = depth_gradient_loss(y_true,y_pred)
    loss=loss1+loss2
    return loss
