import numpy as np

import cv2
from matplotlib import pyplot as plt


__author__ = 'atamas'


def transform_image(input_image, angle, scale):
    center = (input_image.shape[0] / 2, input_image.shape[1] / 2)
    rotatation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(input_image, rotatation_matrix, input_image.shape)
    return rotated_image, rotatation_matrix


def transform_points(input_points, rotation_matrix):
    transformed_points = []
    for point in input_points:
        transformed_points.append(np.dot(rotation_matrix, (point[0], point[1], 1)))
    return np.array(transformed_points)


EPSILON = 50

img = cv2.imread("mandril.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)
transformed_img, rotation_matrix = transform_image(img, 45, 0.5)
# cv2.imwrite("transformed.bmp", transformed_img)

orb = cv2.ORB()

kp1, des1 = orb.detectAndCompute(img, None)
# keypoint_image = cv2.drawKeypoints(img, kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite("keypoints.bmp", keypoint_image)
kp2, des2 = orb.detectAndCompute(transformed_img, None)
# keypoint_image = cv2.drawKeypoints(transformed_img, kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite("keypoints_transformed.bmp", keypoint_image)

old_points = np.array(map(lambda x: np.array(x.pt, np.float32), kp1))
# for point in old_points:
#     cv2.circle(keypoint_image, (point[0], point[1]), 3, (255, 255, 255), thickness=-1)
# cv2.imwrite("keypoints_dots.bmp", keypoint_image)

transformed_old_points = np.int32(transform_points(old_points, rotation_matrix))
# keypoint_image = transformed_img
# for point in transformed_old_points:
#     cv2.circle(keypoint_image, (point[0], point[1]), 3, (255, 255, 255), thickness=-1)
# cv2.imwrite("keypoints_dots_transformed.bmp", keypoint_image)

des1 = np.float32(des1)
des2 = np.float32(des2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

no_of_real_matches = 0
for i, (m, n) in enumerate(matches):
    distance = np.linalg.norm(transformed_old_points[i] - kp2[m.trainIdx].pt)
    if distance < EPSILON:
        no_of_real_matches += 1

matched = float(no_of_real_matches) * 100 / len(kp1)
print('{0}% of keypoints were matched with epsilon = {1}'.format(matched, EPSILON))