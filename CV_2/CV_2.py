import cv2


__author__ = 'atamas'


def text_mask(input_image):
    with_blur = cv2.GaussianBlur(input_image, (3, 3), 0.5)
    laplacian = cv2.Laplacian(with_blur, cv2.CV_8U, ksize=3)
    (thresh, mask) = cv2.threshold(laplacian, 127, 255, cv2.THRESH_BINARY)
    return mask


text = cv2.imread("text.bmp", cv2.CV_LOAD_IMAGE_COLOR)
mask = text_mask(cv2.cvtColor(text, cv2.COLOR_RGB2GRAY))
# processedImage = cv2.erode(text, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
processedImage = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4)), iterations=1)
processedImage = cv2.dilate(processedImage, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
# processedImage = cv2.erode(processedImage, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
cv2.imwrite("processed.bmp", processedImage)
contours, hierarchy = cv2.findContours(processedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    poly = cv2.approxPolyDP(cnt, 3, True)
    rectangle = cv2.boundingRect(poly)
    if rectangle[2] > 5 and rectangle[3] > 5:
        cv2.rectangle(text, (rectangle[0], rectangle[1] - 2), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3] - 2)
                      , (0, 255, 33))

cv2.imwrite('output.bmp', text)



