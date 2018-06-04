import cv2
import pickle


# save given image's keypoints and descriptors in a file with given name using pickle module
def save(image_path, file_path, sift):
    img = cv2.imread(image_path, 0)
    k, d = sift.detectAndCompute(img, None)
    with open(file_path, 'wb') as file:
        kd = ([(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in k], d)
        pickle.dump(kd, file)


# take a filename and returns keypoints and descriptors using pickle module
def load(file_path):
    with open(file_path, 'rb') as file:
        points, descriptors = pickle.load(file)

        keypoints = []
        for p in points:
            pt, size, angle, response, octave, class_id = p
            kp = cv2.KeyPoint(x=pt[0], y=pt[1], _size=size, _angle=angle,
                              _response=response, _octave=octave, _class_id=class_id)
            keypoints.append(kp)
    return keypoints, descriptors


# ###############################################################################
# # Loading SIFT
# sift = cv2.xfeatures2d.SIFT_create()
# ###############################################################################
# # Saving keypoints and descriptors in *.kd files.
# for i in range(1, 17):  # MissBrush book has 16 pages
#     print('Write {}'.format(i))
#     img = cv2.imread('/home/narges/PycharmProjects/sift/missBrush/{}.jpg'.format(i), 0)
#     k, d = sift.detectAndCompute(img, None)
#     with open('MissBrush{}.kd'.format(i), 'wb') as missBrush_kd:
#         kd = ([(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in k], d)
#         pickle.dump(kd, missBrush_kd)
#
# ###############################################################################
# # Load keypoints and descriptors from *.kd files.
# for i in range(1, 17):
#     print('Read {}'.format(i))
#     with open('MissBrush{}.kd'.format(i), 'rb') as missBrush_kd:
#         points, descriptors = pickle.load(missBrush_kd)
#
#         keypoints = []
#         for p in points:
#             pt, size, angle, response, octave, class_id = p
#             kp = cv2.KeyPoint(x=pt[0], y=pt[1], _size=size, _angle=angle,
#                               _response=response, _octave=octave, _class_id=class_id)
#             keypoints.append(kp)
#
#     print(keypoints)
#     print(descriptors)
#

