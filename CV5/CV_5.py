__author__ = 'atamas'

import numpy as np

import cv2


MAX_POINTS = 200


def harris_detector(image):
    return cv2.goodFeaturesToTrack(image,
                                   mask=None,
                                   useHarrisDetector=True,
                                   maxCorners=MAX_POINTS,
                                   qualityLevel=0.15,
                                   minDistance=7,
                                   blockSize=7)


def fast_detector(image):
    features = list(sorted(cv2.FastFeatureDetector().detect(image, None), key=lambda f: f.response, reverse=True))[:MAX_POINTS]
    return np.array([[kp.pt] for kp in features], np.float32)


def process(output_video, detector):
    video = cv2.VideoCapture('sequence.mpg')
    previous_frame = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
    previous_frame_features = detector(previous_frame)
    video_with_keypoits = cv2.VideoWriter(output_video,
                                          int(video.get(cv2.cv.CV_CAP_PROP_FOURCC)),
                                          int(video.get(cv2.cv.CV_CAP_PROP_FPS)),
                                          (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                                           int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))))
    while True:
        ret, frame = video.read()
        if not ret:
            break
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_features, st, err = cv2.calcOpticalFlowPyrLK(previous_frame,
                                                         new_gray,
                                                         previous_frame_features,
                                                         None,
                                                         winSize=(15, 15),
                                                         maxLevel=2,
                                                         criteria=(
                                                             cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        good_new = new_features[st == 1]
        good_old = previous_frame_features[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = map(int, [a, b, c, d])
            cv2.circle(frame, (a, b), 3, (255, 0, 0), -1)
        video_with_keypoits.write(frame)

        previous_frame = new_gray
        previous_frame_features = good_new.reshape(-1, 1, 2)

    video.release()
    video_with_keypoits.release()


process('harris.avi', harris_detector)
process('fast.avi', fast_detector)