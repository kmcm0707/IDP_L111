#!../env/bin/python
# -*- coding: utf-8 -*-
# python 3.9.16
"""Comaints code for detection for cube, line or ar tag


"""

import cv2
import numpy as np
import calibration_clean as cal
import time
import numba

try:
    import apriltag
except ImportError:
    print("Apriltag not installed")
    print("not required so ignore this error")

"""Main code for computer vision

there as 3 different detect red function because open cv requires cv2.waitKey()
to work. if that is outside of the function it wont work.
"""


def detect_line(frame):
    "detects / highlights line in a an image"

    # print(repr(frame))
    "colour_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)"

    # setting the colour threshold
    #   if in range shows up white and if not black
    lower_lim = np.array([0, 0, 0])
    upper_lim = np.array([1, 1, 1]) * 200

    # black & white processed img
    mask = cv2.inRange(frame, lower_lim, upper_lim)
    mask = cv2.bitwise_not(mask)
    cv2.imshow("mask", mask)

    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    seg_output = cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 3)
    output = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Detect Line", output)
    cv2.imshow("seg img", seg_output)
    cv2.waitKey(0)  # press w key to close

    cv2.destroyAllWindows()


def detect_red(frame):
    "for detecting red cube in an image"

    # TODO: do the processing in particular section of the image

    font = cv2.FONT_HERSHEY_COMPLEX

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV

    lower_red = np.array([120, 70, 80])

    # lower_red = cv2.cvtColor([192, 107, 117])

    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only blue colours
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # segmented_img = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # seg_output = cv2.drawContours(segmented_img, contours, -1, (0, 255, 0),3)
    # output = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    centres = []
    for cont in contours:
        """if cv2.contourArea(cont) <= 20:
        continue"""
        x, y, w, h = cv2.boundingRect(cont)

        centres.append((x, y, w, h))

    dim = frame.shape
    half = dim[1] * 0.5
    for x, y, w, h in centres:
        if True:  # (x < half) and (x > 180):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            cv2.putText(
                frame,
                f"{x}, {y}, {cv2.contourArea(cont)}",
                (x, y),
                font,
                0.5,
                (255, 0, 0),
            )

            # print(f"cubes at : {x, y}")

    # print()

    # cv2.imshow('output', output)

    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    return frame
    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    # cv2.waitKey(1)

    # cv2.destroyAllWindows()


def detect_red_video(video):
    "detecting red cube in video and draws rectangle around it"

    while True:
        check, frame = video.read()

        # conveting from RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # red threshold values
        lower_red = np.array([120, 70, 80])
        upper_red = np.array([180, 255, 255])

        # one is ranges shows up white and else in black
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # noise reduction
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # finding contours
        contours, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # drawing rectangles
        centres = []
        for cont in contours:
            if cv2.contourArea(cont) <= 50:
                continue
            x, y, w, h = cv2.boundingRect(cont)

            centres.append((x, y, w, h))

        dim = frame.shape
        half = dim[1] * 0.5
        for x, y, w, h in centres:
            if (x < half) and (x > 180):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))

        # showing the feed with rectangles
        cv2.imshow("frame", frame)

        # for stopping the window wont run without it
        # see file docstring for detail
        key = cv2.waitKey(2)
        if key == ord("q"):
            return


def detect_red_stream(stream):
    "detects red cubes in stream without fixing distortion (for now)"

    while stream.isOpened():
        check, frame = stream.read()

        if not check:
            continue
        # frame = cal.undistorted_live_feed(frame)
        # converting from RGB to HVS
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # threshold red values
        lower_red = np.array([120, 70, 80])
        upper_red = np.array([180, 255, 255])

        # one is ranges shows up white and else in black
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # noise reduction
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # finding contours
        contours, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # drawing rectangle where the object was located
        centres = []
        for cont in contours:
            if cv2.contourArea(cont) <= 50:
                continue
            x, y, w, h = cv2.boundingRect(cont)

            centres.append((x, y, w, h))

        dim = frame.shape
        half = dim[1] * 0.5
        for x, y, w, h in centres:
            # if (x < half) and (x > 180):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))

        # showing the feed with the rectangle
        cv2.imshow("frame", frame)

        # for stopping the window (wont run without it)
        # see file docstring for detail
        key = cv2.waitKey(2)
        if key == ord("q"):
            return


def detect_apriltag(img):
    """detects apriltag in an image using OpenCV implementation of apriltag
    detection algorimths"""
    corners, ids, rejectedImgPoints = cv2.aruco.ArucoDetector.detectMarkers(img)
    print(corners, ids, rejectedImgPoints)


def detect_apriltag_stream_opencv(frame):
    """detects apriltag in a stream using OpenCV implementation of apriltag
    detection algorimth"""
    frame = cal.undistorted_live_feed(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = Detector.detectMarkers(frame)
    print(corners)
    if len(corners) > 0:
        print(corners[0].shape)
        x = corners[0][0, 0].astype("int32")
        y = corners[0][0, 2].astype("int32")

        cv2.rectangle(frame, x, y, (255, 0, 0))

    return frame


def detect_apriltag_stream_apriltag(video):
    """detects apriltag in a stream using apriltag implementation of apriltag
    detection algorimth"""
    option = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(option)
    while True:
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("video not ret")
            return

        result = detector.detect(frame)
        print(result)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            return


def perspective_transoformation(img, dim):
    """function for manual perspective transformation of an image
    returns the transformation matrix"""
    points = []

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))

        if len(points) >= 4:
            cv2.destroyAllWindows()
            print(points)

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    key = cv2.waitKey()
    points = np.float32(points)
    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points, new_points)
    return M


if __name__ == "__main__":

    # For lines
    """for i in range(1, 9):
    img = cv2.imread(f"test_imgs/{i}.png")
    detect_line(img)"""

    # For blocks
    """for i in range(1, 9):
        img = cv2.imread(f"test_imgs/{i}.png")
        img = cal.undistort_frame(img)
        dim = (810, 810)
        M = perspective_transoformation(img, dim)
        img = cv2.warpPerspective(img, M, dim)
        img = detect_red(img)
        cv2.imshow("img", img)
        cv2.waitKey(0)"""

    # this tries to apply this object detection with camera
    """video = cv2.VideoCapture(0)

    detect_red_video(video)

    video.release()"""

    # this code works for the mjpeg stream
    """stream = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

    detect_red_stream(stream)
    stream.release()"""

    # this does apriltag detection on stream
    # video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    """video = cv2.VideoCapture(0)
    Detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )

    while True:
        ret, frame = video.read()
        # frame = cal.undistorted_live_feed(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = Detector.detectMarkers(frame)
        print(corners)
        if len(corners) > 0:
            print(corners[0].shape)
            x = corners[0][0, 0].astype("int32")
            y = corners[0][0, 2].astype("int32")

            cv2.rectangle(frame, x, y, (255, 0, 0))
        cv2.imshow("feed", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break"""

    # detecting apriltag using apriltag liberary
    """video = cv2.VideoCapture(0)
    detect_apriltag_2(video)
    video.release()"""

    # perspective transformation on stream
    video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    video.set(cv2.CAP_PROP_FPS, 10)
    time.sleep(2)
    ret, frame = video.read()

    h, w = frame.shape[:2]
    mtx, dist, newcameramtx = cal.load_vals(2)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    dim = (810, 810)
    print("click on the corners of the table")
    M = perspective_transoformation(frame.copy(), dim)
    video.release()

    video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

    option = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(option)
    count = 0
    while True:
        if count % 3 != 0:
            ret, frame = video.read()
            count += 1
            continue

        ret, frame = video.read()
        # frame = detect_red(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            continue
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        frame = cv2.warpPerspective(frame, M, dim)
        result = detector.detect(frame)
        print(result)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
