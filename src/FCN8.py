#! /usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *

def FCN8(input_height, input_width):
	input_img = Input(shape=(input_height, input_width, 1))
	
	#Block 1
	x = Conv2D(64, (3,3), activation='relu', padding='same', \
		name='block1_conv1', data_format='channels_last')\
		(input_img)
	x = Conv2D(64, (3,3), activation='relu', padding='same', \
		name='block1_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling((2, 2) strides=(2,2), name='block1_pool' \
		data_format='channels_last'(x)

	return Model(input_img, x)

img_name = '4bad52d5ef5f68e87523ba40aa870494a63c318da7ec7609e486e62f7f7a25e8'
seg_img = cv2.imread('/home/marcus/Desktop/' + img_name + '.png')
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(seg_img)
#plt.show()
model = FCN8(256, 256)
model.summary()
