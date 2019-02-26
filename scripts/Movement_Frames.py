#! /usr/bin/python3

import numpy as np
import os
import cv2

variances_path = '../variances/'
bucket_path = '../../bucket/data'
file_names = os.listdir(bucket_path)
for file_name in file_names:
	img = cv2.imread(bucket_path + '/' \
		+ file_name + '/frame0000.png')
	row = img.shape[0]
	col = img.shape[1]
	variance = np.load(variances_path + file_name + '.npy')
	mean_variance = np.mean(variance)
	for i in range(row):
		for j in range(col):
			if variance[i][j] < mean_variance:
				img[i][j][0] = 0
	img = img[:,:,0]
	cv2.imwrite('../frames/' + file_name + '.png', img)
