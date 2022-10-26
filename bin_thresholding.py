# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 18:56:50 2022

@author: USER
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
im = cv2.imread('image(35).jpg', 0)
ret,th=cv2.threshold(im, 170 ,255,cv2.THRESH_BINARY)
plt.hist(im.flat, bins=100,range=(0,255))

kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(th, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN , kernel)
figure_size=7
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,3,1),plt.imshow(im)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(th)
plt.title('Binary image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(opening)
plt.title('Eroded+Dilated img'), plt.xticks([]), plt.yticks([])
plt.show()
