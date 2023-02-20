#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""This was the original code that was used to test computer vision and the apriltag detection. 
It is not used in the final product, but is kept for reference purposes and used to show how the code evolved over time."""

import sys
import time
from threading import Thread
import calibration_clean as cal
import glob
from sys import platform
import math

try:
    import cv2
    import numpy as np
except ImportError:
    print("dependencies not installed")
    print("run 'pip install -r requirements.txt' to install dependencies")
    exit()

try:
    import apriltag
except ImportError:
    print("Apriltag not installed")
    print("not required so ignore this error")

try:
    import pupil_apriltags
except ImportError:
    print("Pupil apriltag not installed")
    print("not required so ignore this error")

if "pupil_apriltags" not in sys.modules and "apriltag" not in sys.modules:

    raise ModuleNotFoundError("neither apriltag detection module installed")


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True


class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):

        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


def detect_line(frame):
    "detects / highlights line in a an image"

    # setting the colour threshold
    #   if in range shows up white and if not black
    lower_lim = np.array([0, 0, 0])
    upper_lim = np.array([1, 1, 1]) * 200

    # black & white processed img
    mask = cv2.inRange(frame, lower_lim, upper_lim)

    # inverteing black and white for convenience
    mask = cv2.bitwise_not(mask)

    cv2.imshow("mask", mask)

    # cuttingout a segment where the line is
    segmented_img = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # drawing the outline for the line in green
    seg_output = cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 3)
    output = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Detect Line", output)
    cv2.imshow("seg img", seg_output)
    cv2.waitKey(0)  # press w key to close

    cv2.destroyAllWindows()


def detect_red(frame):
    "for detecting red cube in an image"

    font = cv2.FONT_HERSHEY_COMPLEX

    # Converting from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # defining range of red color
    lower_red = np.array([120, 70, 80])
    upper_red = np.array([180, 255, 255])

    # finding pixels in the range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # finding groups of pixls
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centres = []
    for cont in contours:
        # finding position and size of the contour
        x, y, w, h = cv2.boundingRect(cont)

        centres.append((x, y, w, h))

    # drawing the rectangle around the cube only if in the correct area
    for x, y, w, h in centres:
        if x < 243 or x > 573 or y < 64 or y > 211:
            pass
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            cv2.putText(
                frame,
                f"{x}, {y}, {cv2.contourArea(cont)}",
                (x, y),
                font,
                0.5,
                (255, 0, 0),
            )

    return frame


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

        # drawing the rectangle around the cube only if in the correct area
        dim = frame.shape
        half = dim[1] * 0.5
        for x, y, w, h in centres:
            if (x < half) and (x > 180):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))

        # showing the feed with rectangles
        cv2.imshow("frame", frame)

        key = cv2.waitKey(2)
        if key == ord("q"):
            return


def detect_red_stream(stream):
    """detects red cubes in stream without fixing distortion

    Parameters
    ----------
    stream: cv2.VideoCapture object
        stream to be processed
    """

    while stream.isOpened():
        check, frame = stream.read()

        if not check:
            continue

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
        for x, y, w, h in centres:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))

        # showing the feed with the rectangle
        cv2.imshow("frame", frame)

        key = cv2.waitKey(2)
        if key == ord("q"):
            return


def detect_apriltag_stream_opencv(frame):
    """detects apriltag in a stream using OpenCV implementation of apriltag
    detection algorimth

    Parameters
    ----------
    frame : numpy.ndarray / cv2 image
        image to detect apriltag in
    """
    # undistorting & converting to grayscale
    frame = cal.undistort_frame(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting apriltag
    corners, ids, rejected = Detector.detectMarkers(frame)
    print(corners)

    # when detected
    if len(corners) > 0:
        print(corners[0].shape)
        x = corners[0][0, 0].astype("int32")
        y = corners[0][0, 2].astype("int32")

        # drwas rectangle around the apriltag
        cv2.rectangle(frame, x, y, (255, 0, 0))

    return frame


def detect_apriltag_stream_apriltag(video, undistort=False):
    """Detects apriltag in a stream using apriltag implementation of apriltag
    detection algorimth

    Parameters
    ----------
    video : cv2.VideoCapture
        video stream
    undistort : bool, optional
        whether to undistort the frame, by default False
    """
    # parameters for detector
    option = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(option)

    while True:
        ret, frame = video.read()
        if not ret:
            continue
        # converting to grayscale and undistorting if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if undistort:
            gray = cal.undistort_frame(gray)
        result = detector.detect(gray)

        # when detected
        if len(result) > 0:
            print(result[0].center)
            # lines around the apriltag
            cv2.polylines(frame, [np.int32(result[0].corners)], True, (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            return


def perspective_transoformation(img, dim):
    """function for manual perspective transformation of an image
    returns the transformation matrix"

    Parameters
    ----------
    img : numpy.ndarray / cv2 image
        image to be transformed
    dim : tuple
        dimensions of the image to be transformed to
    """
    points = []

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))

        if len(points) >= 4:
            cv2.destroyAllWindows()
            print(points)

    # prompts for corners of the table
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    cv2.waitKey()

    # points inputed
    points = np.float32(points)
    # points to map to
    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points, new_points)
    return M


def apriltag_detector_procedure(
    src, module, fix_distortion=True, fix_perspective=True, alpha=1
) -> None:
    """Continueously detects for april tag in a stream

    Parameters
    ----------
    src: int, or string
        source of the stream e.g. 0 or url

    module: module, optional
        module to use for detection

    fix_distortion: bool, optional
        if true undistorts the image

    fix_perspective: bool, optional
        if true corrects the perspective of the image

    alpha: int, optional
        if 0 then undistorted images shows no void
    """

    if fix_distortion or fix_perspective:
        # calibration values
        mtx, dist, newcameramtx = cal.load_vals(6)

    if fix_perspective:
        # for perspective correction
        video = cv2.VideoCapture(src)
        time.sleep(2)
        ret, frame = video.read()
        if not ret:
            print("error with video feed")
            return -1

        # when alpha is 0 then undistorted image shows no void
        if alpha == 0:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 0, (w, h)
            )
            x, y, w, h = roi
            frame = frame[y : y + h, x : x + w]

        # undistorting the image
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        dim = (810, 810)
        print("click on the corners of the table")
        M = perspective_transoformation(frame.copy(), dim)
        video.release()

    # starting the video stream
    video_getter = VideoGet(src).start()

    # apriltag
    try:
        if module is apriltag:
            option = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(option)
            detect = detector.detect

            def angle(result):
                v = result.corners[1] - result.corners[0]
                v = v / np.linalg.norm(v)
                vertical = np.array([1, 0])
                angle = np.arccos(np.dot(v, vertical))
                return angle

    # pupil_apriltags
    except:
        # used because apriltag module dose not work on windows
        try:
            if module is pupil_apriltags:
                detector = pupil_apriltags.Detector(families="tag36h11")
                detect = detector.detect
                angle = lambda result: result.pose_R[0][0]
        except:
            try:
                # opencv aruco module
                # the last resort
                if module is cv2.aruco:
                    detector = cv2.aruco.ArucoDetector()
                    detect = detector.detectMarkers

                    def angle(result):
                        raise NotImplementedError

            except:
                raise ValueError("module not supported")

    # for calculating the velocity
    prev_point = np.array([0, 0])
    current_position = None
    interval = 0
    prev_time = time.time()
    positions = []
    first_time = True
    while True:
        # getting the frame & processing the frame
        frame = video_getter.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if fix_distortion:
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        if fix_perspective:
            frame = cv2.warpPerspective(frame, M, dim)

        # detecting the apriltag
        result = detect(frame)
        if len(result) > 0:

            x, y = result[0].center

            # detecting when the tag speares as a romboid
            # when at an incline on the ramp
            v = result[0].corners[1] - result[0].corners[0]
            v = v / np.linalg.norm(v)
            u = result[0].corners[2] - result[0].corners[1]
            u = u / np.linalg.norm(u)

            # doting the perpenducular sides of the apriltag
            if abs(np.dot(u, v)) > 0.1:
                print("ranmp")

            current_position = np.array([x, y])
            positions.append(current_position)

            # calculating speed
            interval = time.time() - prev_time
            speed = (result[0].center - prev_point) / interval
            current_position += speed * interval

            prev_time = time.time()

            prev_point = result[0].center

            # drawing cicele on the center and the predicted location if the tag
            frame = cv2.circle(
                frame, (int(x), int(y)), radius=10, color=(0, 0, 255), thickness=-1
            )
            frame = cv2.circle(
                frame,
                current_position.astype("int32"),
                radius=12,
                color=(255, 255, 255),
                thickness=-1,
            )
            frame = cv2.polylines(
                frame, [np.int32(result[0].corners)], True, (255, 255, 255), 2
            )

            if first_time:
                frame = cv2.polylines(
                    frame, [np.int32(positions)], False, (0, 0, 255), 2
                )
                first_time = False

        if not first_time:
            frame = cv2.polylines(frame, [np.int32(positions)], False, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            video_getter.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # apriltag_detector_procedure(0, fix_distortion=False, fix_perspective=False)
    if platform == "darwin":
        # os.system("python " + "../server/app.py &")
        # mac
        apriltag_detector_procedure(
            # "http://localhost:8081/stream/video.mjpeg",
            1,
            module=apriltag,
            fix_distortion=False,
            fix_perspective=False,
        )
    elif platform == "win32":
        # Windows
        # os.system("start /b python ../server/app.py")
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=pupil_apriltags,
        )

    # For lines detection
    """for i in range(1, 9):
    img = cv2.imread(f"calib_imgs/test_imgs/{i}.png")
    detect_line(img)"""

    # For blocks detection
    """video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")"""
    """images = glob.glob("calib_imgs/test_imgs/*.png")
    for fname in images:
        img = cv2.imread(fname)
        img = cal.undistort_frame(img)
        dim = (810, 810)
        M = perspective_transoformation(img, dim)
        img = cv2.warpPerspective(img, M, dim)
        img = detect_red(img)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break"""

    # this tries to apply this object detection with camera
    """video = cv2.VideoCapture(1)

    detect_red_video(video)

    video.release()"""

    # this code works for the mjpeg stream
    """stream = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")

    detect_red_stream(stream)
    stream.release()"""

    # this does apriltag detection on stream
    # video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    # video = cv2.VideoCapture(0)
    """Detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    )

    while True:
        ret, frame = video.read()
        # frame = cal.undistort_frame(frame)
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
    # video = cv2.VideoCapture(0)
    """apriltag_detector_procedure("http://localhost:8081/stream/video.mjpeg", apriltag)
    video.release()
    cv2.destroyAllWindows()"""
