#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""Main code for computer vision comaints code for detection for cube, line or ar tag
"""

import sys
import time
from threading import Thread
import calibration_clean as cal
import os
import json
import requests
from sys import platform
import math
from flask import Flask, jsonify
import json
import requests 
# import numba

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

rightspeed = 0
leftspeed = 0
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


def detect_apriltag(img):
    """detects apriltag in an image using OpenCV implementation of apriltag
    detection algorimths"""
    corners, ids, rejectedImgPoints = cv2.aruco.ArucoDetector.detectMarkers(img)
    print(corners, ids, rejectedImgPoints)


def detect_apriltag_stream_opencv(frame):
    """detects apriltag in a stream using OpenCV implementation of apriltag
    detection algorimth"""
    frame = cal.undistort_frame(frame)
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
    """Detects apriltag in a stream using apriltag implementation of apriltag
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


def apriltag_detector_procedure(
    src, module, fix_distortion=True, fix_perspective=True, alpha=1, controller=None
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
        mtx, dist, newcameramtx = cal.load_vals(6)

    if fix_perspective:
        video = cv2.VideoCapture(src)
        time.sleep(2)
        ret, frame = video.read()
        if not ret:
            print("error with video feed")
            return -1

        if alpha == 0:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 0, (w, h)
            )
            x, y, w, h = roi
            frame = frame[y : y + h, x : x + w]

        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        dim = (810, 810)
        print("click on the corners of the table")
        M = perspective_transoformation(frame.copy(), dim)
        
        video.release()

    
    video_getter = VideoGet(src).start()
    # video_shower = VideoShow(video_getter.frame).start()
    try:
        if module is apriltag:
            option = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(option)
            detect = detector.detect
    except:
        try:
            if module is pupil_apriltags:
                detector = pupil_apriltags.Detector(families="tag36h11")
                detect = detector.detect
        except:
            try:
                if module is cv2.aruco:
                    detector = cv2.aruco.ArucoDetector()
                    detect = detector.detectMarkers
            except:
                raise ValueError("module not supported")
    prev_point = np.array([0, 0])
    current_position = None
    interval = 0
    prev_time = time.time()
    positions = []
    first_time = True
    while True:

        frame = video_getter.frame
        # frame = detect_red(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        """if not ret:
            print("no frame retereived; trying apain")
            continue"""

        if fix_distortion:
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        if fix_perspective:
            frame = cv2.warpPerspective(frame, M, dim)

        result = detect(frame)
        if len(result) > 0:
            if first_time:
                def click_envent(event, x, y, flags, params):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        print(x, y)
                        controller.set_target((x, y))
                        cv2.destroyAllWindows()

                cv2.imshow("img", frame.copy())
                cv2.setMouseCallback("img", click_envent)
                first_time = False
                continue
            
            x, y = result[0].center
            theta = result[0].pose_R[0][0]
            # if first_time:
            current_position = np.array([x, y])
            controller.set_current_position(current_position)

            # first_time = False

            positions.append(current_position)

            interval = time.time() - prev_time
            speed = (result[0].center - prev_point) / interval
            current_position += speed * interval
            controller.set_predicted_position(current_position)
            controller.PID_controller_update()
            prev_time = time.time()

            prev_point = result[0].center

            a, b, c, d = result[0].corners
            frame = cv2.circle(
                frame, (int(x), int(y)), radius=10, color=(0, 0, 255), thickness=-1
            )
            print(current_position)
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
            print(speed)
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
        # This code dosen't work for me (dev) - Yh no idea why (kyle)
        """
        video_shower.frame = frame
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break"""

    cv2.destroyAllWindows()

app = Flask(__name__)

@app.route("/data", methods=['POST', 'GET'])
def data():
    dictToReturn = {'llf':leftspeed,
                    'rlf': rightspeed}
    return jsonify(dictToReturn)

class PID:
    def __init__(self):
        self.kp = 30
        self.ki = 0.001
        self.kd = 10
        self.prev_error = 0
        self.integral = 0
        self.error = 0
        self.left_speed = 200
        self.right_speed = 200
        self.current_position = np.array([0, 0])
        self.target_position = np.array([0, 0])
        self.predicted_position = np.array([0, 0])

    def get_right_speed(self):
        return self.right_speed
    def get_left_speed(self):
        return self.left_speed

    def set_current_position(self, current_position):
        self.current_position = current_position
    
    def set_target_position(self, target_position):
        self.target_position = target_position

    def set_predicted_position(self, predicted_position):
        self.predicted_position = predicted_position
    
    def PID_controller_update(self):
        basespeed = 200
        """This function will return the error for the PID controller"""
        deltaX = self.current_position[0] - self.predicted_position[0]
        deltaY = self.current_position[1] - self.predicted_position[1]
        targetX = self.current_position[0] - self.target_position[0]
        targetY = self.current_position[1] - self.target_position[1]

        velocityAngle = (math.atan2(deltaY.toDouble(), deltaX.toDouble())) + math.pi
        targetAngle = (math.atan2(targetY.toDouble(), targetX.toDouble())) + math.pi
        
        if(velocityAngle == 2*math.pi or targetAngle == 2*math.pi):
            velocityAngle = 0
            targetAngle = 0

        temp_error = targetAngle - velocityAngle
        if(self.error > math.pi or (self.error < 0 and self.error > -math.pi) ):
            #turn right - left faster
            temp_error = -abs(temp_error)
        else:
            #turn left - right faster
            temp_error = abs(temp_error)
        self.error = temp_error

        self.integral = self.integral + self.error
        D = self.error - self.prev_error
        self.prev_error = self.error

        motorspeed = int(self.k_p*self.error + self.k_d * D + self.k_i * self.integral)
        self.left_speed = basespeed - motorspeed
        self.right_speed = basespeed + motorspeed
        leftspeed = self.left_speed
        rightspeed = self.right_speed

def start_flaskapp():
    app.run(host='0.0.0.0', port=5000, debug=False)

def start_everythingelse():
    controller = PID()
    if platform == "darwin":
        # mac
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=apriltag,
            controller = controller,
        )
    elif platform == "win32":
        # Windows
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=pupil_apriltags,
            controller = controller,
        )
if __name__ == "__main__":
    # apriltag_detector_procedure(0, fix_distortion=False, fix_perspective=False)
    t2 = Thread(target=start_flaskapp)
    t2.start()
    t1 = Thread(target=start_everythingelse)
    t1.start()

    # For lines
    """for i in range(1, 9):
    img = cv2.imread(f"calib_imgs/test_imgs/{i}.png")
    detect_line(img)"""

    # For blocks
    """for i in range(1, 9):
        img = cv2.imread(f"calib_imgs/test_imgs/{i}.png")
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
    """video = cv2.VideoCapture(0)
    detect_apriltag_2(video)
    video.release()"""

