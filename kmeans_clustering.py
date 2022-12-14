# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:19:56 2022

@author: USER
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("image(35).jpg")
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
ret,th=cv2.threshold(img, 170 ,255,cv2.THRESH_BINARY)
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(th, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN , kernel)
vectorized = opening.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((opening.shape))
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()