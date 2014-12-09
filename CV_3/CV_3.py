import numpy as np

import cv2


__author__ = 'atamas'

CUTOFF_FREQUENCY = 30

img = cv2.imread("mandril.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)
fft = np.fft.fft2(img)

center = (fft.shape[0] / 2, fft.shape[1] / 2)
mask = np.zeros(fft.shape)
mask[CUTOFF_FREQUENCY:fft.shape[0] - CUTOFF_FREQUENCY,
     CUTOFF_FREQUENCY:fft.shape[1] - CUTOFF_FREQUENCY] = 1
filtered_fft = fft * mask

processedImage = np.fft.ifft2(filtered_fft)
cv2.imwrite('filtered.bmp', np.real(processedImage))

cv2.imwrite('laplassian.bmp', cv2.Laplacian(img, cv2.CV_32F))

