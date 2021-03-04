#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:05:03 2021

@author: anunay
"""


def loss_focal(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype = tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0039215686274, 1.0)
    y_true = tf.clip_by_value(y_true, 0.0039215686274, 1.0)
    y_pred = tf.nn.softmax(y_pred)
    alpha = tf.ones_like(y_true, dtype = tf.float32) * 0.25
    gamma = 2
    loss = tf.reduce_mean(alpha * tf.math.pow((1 - y_pred), 2) * (-tf.reduce_sum(y_true * tf.math.log(y_pred))))
    return loss

def semantic_gradient_loss(y_true, y_pred):
    y_true = tf.image.convert_image_dtype(y_true, dtype=tf.float32)
#     y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.image.convert_image_dtype(y_pred, dtype = tf.float32)
#     y_pred = tf.expand_dims(y_pred, axis = -1)

    sobel_true = tf.image.sobel_edges(y_true)
    sobel_true_h = sobel_true[0, :, :, :, 0]
    sobel_true_w = sobel_true[0, :, :, :, 1]
    #y_pred edges
    sobel_pred = tf.image.sobel_edges(y_pred)
    sobel_pred_h = sobel_pred[0, :, :, :, 0]
    sobel_pred_w = sobel_pred[0, :, :, :, 1]
    #loss

    loss_semantic = tf.reduce_mean(tf.math.abs(sobel_pred_h - sobel_true_h) + tf.math.abs(sobel_pred_w - sobel_true_w))
    return loss_semantic


def combined_semantic_losses(y_true, y_pred):
    loss1 = loss_focal(y_true, y_pred)
    loss2 = semantic_gradient_loss(y_true, y_pred)
    loss = loss1 + loss2
    return loss