import cv2

__author__ = 'atamas'


def process_image(input_imagae):
    with_blur = cv2.GaussianBlur(input_imagae, (3, 3), 0.5)
    laplacian = cv2.Laplacian(with_blur, cv2.CV_16S, ksize=3)
    (thresh, mask) = cv2.threshold(laplacian, 127, 255, cv2.THRESH_BINARY)
    return mask

cv2.imwrite('out.bmp', process_image(cv2.imread("text.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)))



