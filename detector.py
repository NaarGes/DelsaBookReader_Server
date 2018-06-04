import cv2
import numpy as np


img = cv2.imread('home.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray, None)  # finds the keypoint in the images
img = cv2.drawKeypoints(gray, kp, img)  # draws the small circles on the locations of keypoints
cv2.imwrite('sift_keypoints.jpg', img)