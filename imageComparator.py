# TODO do matching in C++
# TODO scan train images
# TODO more than one book
# TODO IsMan boolian
# TODO android send pic 2 times
# TODO android toast
# TODO logs in android


import numpy as np
import cv2
import KDStorage

# IsSameImage = False
# Qimg = cv2.imread('file.png', 0)  # query image
#
# sift = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
# Qkp, Qdes = sift.detectAndCompute(Qimg, None)  # find the keypoints and descriptors with SIFT
# bf = cv2.BFMatcher()  # BFMatcher with default params
#
# for i in range(1, 17):
#     tkd = KDStorage.load('MissBrush{}.kd'.format(i))
#     matches = bf.knnMatch(Qdes, tkd[1], k=2)
#     # Apply ratio test
#     good = []
#     for m, n in matches:
#         if m.distance < 0.65 * n.distance:
#             good.append([m])
#     if len(good)/len(tkd[0]) > 0.25:
#         IsSameImage = True
#     else:
#         IsSameImage = False
#
#     if IsSameImage is True:
#         print('Image is equal to {}.jpg'.format(i))
#         exit()
#
# compare image
sameimage = False
pagenumber = 0
Qimg = cv2.imread('/home/narges/PycharmProjects/sift/missBrushTest/20170821_195530.jpg', 0)  # query image

sift = cv2.xfeatures2d.SIFT_create()  # Initiate SIFT detector
Qkp, Qdes = sift.detectAndCompute(Qimg, None)  # find the keypoints and descriptors with SIFT
bf = cv2.BFMatcher()  # BFMatcher with default params

for i in range(1, 17):
    print('loading keypoints and descriptors of image {}...'.format(i))
    tkd = KDStorage.load('MissBrush{}.kd'.format(i))
    print('start matching with image {}...'.format(i))
    matches = bf.knnMatch(Qdes, tkd[1], k=2)
    # Apply ratio test
    good = []
    print('starting ratio test...')
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good.append([m])
    if len(good) / len(tkd[0]) > 0.1:
        pagenumber = i
        sameimage = True

    if sameimage is True:
        print('Image is equal to {}.jpg'.format(i))
        break
    elif i == 17:
        print('no match found')
    else:
        print('not equal to image {}'.format(i))

print('returning response: http://{ip}:{port}/statics/miss-brush/woman/{page_number}.ogg'.format(ip='0.0.0.0', port='5000',
                                                                                                 page_number=pagenumber))
