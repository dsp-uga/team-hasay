#! /usr/bin/python3

import cv2
import os
import numpy as np

bucket_path = '../../bucket/data'
dir_names = os.listdir(bucket_path)
for dir_name in dir_names:
	img = cv2.imread(bucket_path + '/' \
		+ dir_name + '/frame0000.png')
	row = img.shape[0]
	col = img.shape[1]
	mean = np.zeros((row, col))
	variance = np.zeros((row, col))
	for i in range(row):
		for j in range(col):
			mean[i][j] += img[i][j][0] / 100
	for i in range(1, 100):
		if i < 10:
			frame = 'frame000'
		else:
			frame = 'frame00'
		img = cv2.imread(bucket_path + '/' \
			+ dir_name + '/' + frame \
			+ str(i) + '.png')	
		row = img.shape[0]
		col = img.shape[1]
		for j in range(row):
			for k in range(col):
				mean[j][k] += img[j][k][0] / 100

	for i in range(100):
		if i < 10:
			frame = 'frame000'
		else:
			frame = 'frame00'
		img = cv2.imread(bucket_path + '/' \
			+ dir_name + '/' + frame \
			+ str(i) + '.png')	
		row = img.shape[0]
		col = img.shape[1]
		for j in range(row):
			for k in range(col):
				variance[j][k] += (img[j][k][0] - mean[j][k])**2 / 100
	np.save('../variances/' + dir_name, variance)
