import cv2
import numpy as np
import calibration_clean as cal
import time

"""Main code for computer vision

there as 3 different detect red function because open cv requires cv2.waitKey()
to work. if that is outside of the function it wont work.
"""


def detect_line(frame):
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
    # detecting red cube in video

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
    # detecting red cube in video

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
    # detects red cubes in stream (*untested)

    while stream.isOpened():
        check, frame = stream.read()

        if not check:
            continue
        frame = cal.undistorted_live_feed(frame)
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
    corners, ids, rejectedImgPoints = cv2.aruco.ArucoDetector.detectMarkers(img)
    print(corners, ids, rejectedImgPoints)


def detect_apriltag_stream(frame):
    frame = cal.undistorted_live_feed(frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = Detector.detectMarkers(frame)
    print(corners)
    if len(corners) > 0:
        print(corners[0].shape)
        x = corners[0][0, 0].astype("int32")
        y = corners[0][0, 2].astype("int32")

        cv2.rectangle(frame, x, y, (255, 0, 0))

    return frame


def perspective_transoformation(img, dim):
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
        detect_red(img)"""

    # this tries to apply this object detection with camera
    """video = cv2.VideoCapture(0)

    detect_red_video(video)

    video.release()"""

    # this code works for the mjpeg stream
    """stream = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

    detect_red_stream(stream)
    stream.release()"""

    # this does apriltag detection on stream
    """video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    Detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )

    while True:
        ret, frame = video.read()
        frame = cal.undistorted_live_feed(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    # perspective transformation on stream
    video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    # time.sleep(1)
    ret, frame = video.read()
    frame = cal.undistorted_live_feed(frame)
    dim = (810, 810)
    M = perspective_transoformation(frame.copy(), dim)

    while True:
        ret, frame = video.read()
        if not ret:
            continue
        frame = cal.undistorted_live_feed(frame)
        frame = cv2.warpPerspective(frame, M, dim)
        frame = detect_red(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
