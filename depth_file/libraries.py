#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:37:09 2021

@author: anunay
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
import keras.backend as K
import json
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Add, Dense, Activation, UpSampling2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import plot_model, Sequence
from keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard