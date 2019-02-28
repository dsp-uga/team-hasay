#! /usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.callbacks import *
from scipy.ndimage.filters import *

def FCN8(input_height, input_width, n_classes):
	input_img = Input(shape=(input_height, input_width, 1))
	activation = 'relu'
	
	#Block 1
	x = Conv2D(64, (3,3), activation=activation, padding='same', \
		name='block1_conv1', data_format='channels_last')\
		(input_img)
	x = Conv2D(64, (3,3), activation=activation, padding='same', \
		name='block1_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='block1_pool', \
		data_format='channels_last')(x)

	#Block 2 
	x = Conv2D(128, (3,3), activation=activation, padding='same', \
		name='block2_conv1', data_format='channels_last')\
		(x)
	x = Conv2D(128, (3,3), activation=activation, padding='same', \
		name='block2_conv2', data_format='channels_last')\
		(x)
	x = MaxPooling2D((2, 2), strides=(2,2), name='block2_pool', \
		data_format='channels_last')(x)

	#Block 3
	x = Conv2D(256, (3,3), activation=activation, padding='same', \
		name='block3_conv1', data_format='channels_last')\
		(x)
	x = Conv2D(256, (3,3), activation=activation, padding='same', \
		name='block3_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(256, (3,3), activation=activation, padding='same', \
		name='block3_conv3', data_format='channels_last')\
		(x)
	pool3 = MaxPooling2D((2, 2), strides=(2,2), name='block3_pool', \
		data_format='channels_last')(x)

	#Block 4
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block4_conv1', data_format='channels_last')\
		(pool3)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block4_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block4_conv3', data_format='channels_last')\
		(x)
	pool4 = MaxPooling2D((2, 2), strides=(2,2), name='block4_pool', \
		data_format='channels_last')(x)

	#Block 5
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block5_conv1', data_format='channels_last')\
		(pool4)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block5_conv2', data_format='channels_last')\
		(x)
	x = Conv2D(512, (3,3), activation=activation, padding='same', \
		name='block5_conv3', data_format='channels_last')\
		(x)
	pool5 = MaxPooling2D((2, 2), strides=(2,2), name='block5_pool', \
		data_format='channels_last')(x)

	#Deconvolution pool5
	x = (Conv2D(4096, (8,8), activation=activation, padding='same', \
		name='conv6', data_format='channels_last'))(pool5)

	x = (Conv2D(4096, (1,1), activation=activation, padding='same', \
		name='conv7', data_format='channels_last'))(x)

	pool5_deconv = (Conv2DTranspose(n_classes, kernel_size=(4, 4),\
		strides=(4, 4), use_bias=False, \
		data_format='channels_last'))(x)

	#Deconvolution pool4
	x = (Conv2D(n_classes, (1,1), activation=activation, padding='same', \
		name='pool4_filetered', data_format='channels_last'))(pool4)
	pool4_deconv = (Conv2DTranspose(n_classes, kernel_size=(2, 2),\
		strides=(2, 2), use_bias=False, \
		data_format='channels_last'))(x)

	#Layer Fusion
	pool3_filtered = (Conv2D(n_classes, (1,1), activation=activation, padding='same', \
		name='pool3_filetered', data_format='channels_last'))(pool3)
	x = Add(name='layer_fusion')([pool5_deconv, pool4_deconv, pool3_filtered])

	#8 Times Deconvolution and Softmax
	x = (Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), \
		use_bias=False, data_format='channels_last'))(x)
	x = (Activation('softmax'))(x)

	return Model(input_img, x)

def fourier_transform(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	img_magnitude_spectrum = 20 * np.log(np.abs(fshift))
	return img_magnitude

def laplacian(img):
	laplacian = cv2.Laplacian(img, cv2.CV_64F)
	#laplacian = laplacian[:, :, :1]
	return laplacian

def canny(img):
	canny = cv2.Canny(img, 100, 200)
	return np.expand_dims(canny, axis=2)

def normalize(img):
	mean = np.mean(img)
	std = np.std(img)
	row = img.shape[0]
	col = img.shape[1]
	for i in range(row):
		for j in range(col):
			img[i][j][0] = (img[i][j][0] - mean) / std
	return img[:, :, :1]

def load_data():
	train_names = open('../../bucket/train.txt').read().split()
	x_train = []
	y_train = []
	img_shapes = []
	n_classes = 3
	for file_name in train_names:
		img = cv2.imread('../frames/' + file_name + '.png')
		img_shapes.append(img.shape)
		if img.shape[0] != 256 or img.shape[1] != 256:
			img  = cv2.resize(img, (256, 256), \
				interpolation = cv2.INTER_AREA) #CUBIC for upsample
		img = median_filter(img, size=3)
		img = normalize(img)
		#img = fourier_transform(img)
		#img = canny(img)
		#img = laplacian(img)
		x_train.append(img)

		img = cv2.imread('../../bucket/masks/' + file_name + '.png')
		if img.shape[0] != 256 or img.shape[1] != 256:
			img  = cv2.resize(img, (256, 256), \
				interpolation = cv2.INTER_AREA)
		img = img[:,:,0]
		seg_mask = np.zeros((img.shape[0], img.shape[1], n_classes))
		for label in range(n_classes):
			seg_mask[:,:,label] = (img == label).astype(int)
		y_train.append(seg_mask)

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	return x_train, y_train, img_shapes
	
#Main
x_train, y_train, img_shapes = load_data()

model = FCN8(256, 256, 3)
model.summary()

rmsprop = optimizers.RMSprop()
sgd = optimizers.SGD(lr=0.01, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, \
		metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)
model.save('../models/FCN8_Thresholded_200.h5')

pred = model.predict(x_train)
pred_img = np.argmax(pred, axis=3)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121)
ax.imshow(pred_img[0])
ax = fig.add_subplot(122)
ax.imshow(y_train[0][:, :, 0])
#plt.show()
