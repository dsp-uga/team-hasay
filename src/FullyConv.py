#! /usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt

img_name = '4bad52d5ef5f68e87523ba40aa870494a63c318da7ec7609e486e62f7f7a25e8'
seg_img = cv2.imread('/home/marcus/Desktop/' + img_name + '.png')
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(seg_img)
plt.show()
