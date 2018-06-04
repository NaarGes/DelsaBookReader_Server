import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import time


print(time())
img1 = cv2.imread('file.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/narges/PycharmProjects/sift/missBrush/2.jpg', cv2.IMREAD_GRAYSCALE)
print(time())
# Initiate SIFT detector
# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
print(time())
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print(time())
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
print(time())

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.65*n.distance:
        good.append([m])
print(time())
print(des1, des2, sep='\n')
print(list(des1))
print('first image keypoints: ', len(kp1))
print('second image keypoints: ', len(kp2))
print('matched keypoints: ', len(matches))
print('good keypoints: ', len(good))
print(time())


# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.imshow(img3), plt.show()


