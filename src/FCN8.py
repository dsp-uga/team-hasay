#! /usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras import optimizers

def FCN8(input_height, input_width, n_classes):
	input_img = Input(shape=(input_height, input_width, 1))
	
	#Block 1
	x = Conv2D(64, (3,3), activation='relu', padding='same', \
		name='block1_conv1', data_format='channels_last')\
		(input_img)
	x = Conv2D(64, (3,3), activation='relu', padding='same', \
		name='block1_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='block1_pool', \
		data_format='channels_last')(x)

	#Block 2 
	x = Conv2D(128, (3,3), activation='relu', padding='same', \
		name='block2_conv1', data_format='channels_last')\
		(x)
	x = Conv2D(128, (3,3), activation='relu', padding='same', \
		name='block2_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='block2_pool', \
		data_format='channels_last')(x)

	#Block 3
	x = Conv2D(256, (3,3), activation='relu', padding='same', \
		name='block3_conv1', data_format='channels_last')\
		(x)
	x = Conv2D(256, (3,3), activation='relu', padding='same', \
		name='block3_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(256, (3,3), activation='relu', padding='same', \
		name='block3_conv3', data_format='channels_last')\
		(x)
	pool3 = MaxPooling2D((2, 2), strides=(2,2), name='block3_pool', \
		data_format='channels_last')(x)

	#Block 4
	x = Conv2D(512, (3,3), activation='relu', padding='same', \
		name='block4_conv1', data_format='channels_last')\
		(pool3)
	x = Conv2D(512, (3,3), activation='relu', padding='same', \
		name='block4_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(512, (3,3), activation='relu', padding='same', \
		name='block4_conv3', data_format='channels_last')\
		(x)
	pool4 = MaxPooling2D((2, 2), strides=(2,2), name='block4_pool', \
		data_format='channels_last')(x)

	#Block 5
	x = Conv2D(512, (3,3), activation='relu', padding='same', \
		name='block5_conv1', data_format='channels_last')\
		(pool4)
	x = Conv2D(512, (3,3), activation='relu', padding='same', \
		name='block5_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(512, (3,3), activation='relu', padding='same', \
		name='block5_conv3', data_format='channels_last')\
		(x)
	pool5 = MaxPooling2D((2, 2), strides=(2,2), name='block5_pool', \
		data_format='channels_last')(x)

	#Deconvolution pool5
	x = (Conv2D(4096, (8,8), activation='relu', padding='same', \
		name='conv6', data_format='channels_last'))(pool5)

	x = (Conv2D(4096, (1,1), activation='relu', padding='same', \
		name='conv7', data_format='channels_last'))(x)

	pool5_deconv = (Conv2DTranspose(n_classes, kernel_size=(4, 4),\
		strides=(4, 4), use_bias=False, \
		data_format='channels_last'))(x)

	#Deconvolution pool4
	x = (Conv2D(n_classes, (1,1), activation='relu', padding='same', \
		name='pool4_filetered', data_format='channels_last'))(pool4)
	pool4_deconv = (Conv2DTranspose(n_classes, kernel_size=(2, 2),\
		strides=(2, 2), use_bias=False, \
		data_format='channels_last'))(x)

	#Layer Fusion
	pool3_filtered = (Conv2D(n_classes, (1,1), activation='relu', padding='same', \
		name='pool3_filetered', data_format='channels_last'))(pool3)
	x = Add(name='layer_fusion')([pool5_deconv, pool4_deconv, pool3_filtered])

	#8 Times Deconvolution and Softmax
	x = (Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), \
		use_bias=False, data_format='channels_last'))(x)
	x = (Activation('softmax'))(x)

	return Model(input_img, x)

#Main
train_names = open('../../bucket/train.txt').read().split()
x_train = []
y_train = []
img_shape = []
for file_name in train_names:
	#Train Images
	img = cv2.imread('../../bucket/data/' + file_name + '/frame0000.png')
	img_shape.append(img.shape)
	if img.shape[0] != 256 or img.shape[1] != 256:
		img  = cv2.resize(img, (256, 256), \
			interpolation = cv2.INTER_AREA) #CUBIC for upsample
	img = img[:, :, :1]
	x_train.append(img)
	#Masks
	img = cv2.imread('../../bucket/masks/' + file_name + '.png')
	if img.shape[0] != 256 or img.shape[1] != 256:
		img  = cv2.resize(img, (256, 256), \
			interpolation = cv2.INTER_AREA)
	y_train.append(img)

x_train = np.array(x_train)
y_train = np.array(y_train)

model = FCN8(256, 256, 3)
model.summary()

sgd = optimizers.SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, \
		metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=20)
model.save('../models/FCN8_Full_First.h5')

pred = model.predict(x_train)
pred_img = np.argmax(pred, axis=3)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121)
ax.imshow(pred_img[0])
ax = fig.add_subplot(122)
ax.imshow(y_train[0][:, :, 0])
#plt.show()
