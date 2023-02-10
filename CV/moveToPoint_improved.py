#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""Main code for computer vision comaints code for detection for cube, line or ar tag
"""

import sys
import time
import math
from threading import Thread
import calibration_clean as cal
import paho.mqtt.client as mqtt
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import requests
from sys import platform
import math

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


# mqttBroker = "broker.hivemq.com"
# Alternative message brokers:
# mqttBroker = "public.mqtthq.com"
mqttBroker = "test.mosquitto.org"
# mqttBroker =  "public.mqtthq.com"
client = mqtt.Client("Python")
client.connect(mqttBroker)

global target
global targets


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


class Controller:
    def __init__(self, video, mtx, dist, newcameramtx, M, dim, detector):
        self.M = M
        self.dim = dim
        self.video_getter = video
        self.mtx = mtx
        self.dist = dist
        self.newcameramtx = newcameramtx
        self.frame = None
        self.stopped = False
        self.detector = detector
        self.result = None

        self.angle_diff = 10000
        self.current_position = np.zeros(2)

        # Thread(target=self.update, args=()).start()

    def update(self):
        while not self.stopped:
            time.sleep(0.8)
            frame = self.video_getter.frame
            if frame is None:
                continue

            # camera distortion & perspective distortion fix
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.undistort(frame, self.mtx, self.dist, None, self.newcameramtx)
            self.frame = cv2.warpPerspective(frame, self.M, self.dim)

            # detect the marker
            result = self.detector.detect(self.frame)
            if len(result) > 0:
                self.result = result[0]
                return
            else:
                self.result = None
                continue

    def rotate(self, target, threshold=0.2):
        print("rotating")
        angle_diff = 10000
        current_position = np.zeros(2)
        clockwise = False
        anticlockwise = False
        last_time = time.time()
        # alternating_couter = 0

        while True:
            self.update()
            cv2.imshow("frame", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stop()
            print(".", end="")
            if self.result is None:
                continue
            try:
                current_position[:] = self.result.center

                heading = np.float64(target - self.current_position)
                heading /= np.linalg.norm(heading)
                v = np.float64(self.result.corners[1] - self.result.corners[0])
                v /= np.linalg.norm(v)

                angle_diff = np.arccos(np.dot(v, heading))
                # print(angle_diff)
            except AttributeError:
                continue

            if angle_diff < threshold or angle_diff > (2 * math.pi) - threshold:
                return

            if alternating_couter > 4:
                client.publish("IDP_2023_Follower_left_speed", -250)
                client.publish("IDP_2023_Follower_right_speed", -250)
                time.sleep(0.5)
                # alternating_couter = 0
                client.publish("IDP_2023_Follower_left_speed", 0)
                client.publish("IDP_2023_Follower_right_speed", 0)
                anticlockwise = False
                clockwise = False

            try:
                direction = (
                    ((self.result.corners[1] + self.result.corners[2]))
                    - (self.result.corners[0] + self.result.corners[3])
                ) / 2

                if (
                    (np.float64(direction[0]) * (target[1] - self.result.corners[1][1]))
                    - (
                        np.float64(direction[1])
                        * (target[0] - self.result.corners[0][0])
                    )
                ) > 0:
                    # print("clockwise")
                    if not clockwise:
                        # alternating_couter += 1
                        client.publish("IDP_2023_Follower_left_speed", 125)
                        client.publish("IDP_2023_Follower_right_speed", -125)
                        clockwise = True
                        anticlockwise = False
                        # print("clockwise")
                        last_time = time.time()
                else:
                    # print("anticlockwise")
                    if not anticlockwise:
                        # alternating_couter += 1
                        client.publish("IDP_2023_Follower_left_speed", -125)
                        client.publish("IDP_2023_Follower_right_speed", 125)
                        # print("anticlockwise")
                        clockwise = False
                        anticlockwise = True
                        last_time = time.time()
            except AttributeError:
                continue

    def move(
        self,
        target,
        catchment_area=35,
        angle_threshold=0.2,
        angle_fix=True,
        speed=250,
    ):
        print("forward")
        client.publish("IDP_2023_Follower_left_speed", speed)
        client.publish("IDP_2023_Follower_right_speed", speed)
        while True:
            self.update()
            if self.result is None:
                continue

            heading = np.float64(target - self.result.center)
            heading /= np.linalg.norm(heading)
            v = np.float64(self.result.corners[1] - self.result.corners[0])
            v /= np.linalg.norm(v)
            angle_diff = np.arccos(np.dot(v, heading))

            if angle_fix:
                if (
                    angle_diff < angle_threshold
                    or angle_diff > (2 * math.pi) - angle_threshold
                ):
                    pass
                else:
                    print("course correction")
                    self.rotate(target, threshold=0.2)

            if np.linalg.norm(target - self.result.center) < catchment_area:
                print("checkpoint reached")
                client.publish("IDP_2023_Follower_left_speed", 0)
                client.publish("IDP_2023_Follower_right_speed", 0)
                return

    def show(self):
        while True:
            if self.frame is None:
                continue
            cv2.imshow("frame", self.frame)
            key = cv2.waitKey(2)
            if key == ord("q"):
                self.stop()
                self.video_getter.stop()
                cv2.destroyAllWindows()
                break

    def stop(self):
        self.stopped = True


def perspective_transoformation(img, dim):
    """function for manual perspective transformation of an image
    returns the transformation matrix"""
    """points = []

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            points.append((x, y))

        if len(points) >= 4:
            cv2.destroyAllWindows()
            print(points)

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    key = cv2.waitKey()"""
    points = np.float32([[261, 682], [818, 630], [236, 151], [738, 92]])
    # points = np.float32(points)
    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points, new_points)
    return M


def forward(time=2):
    client.publish("IDP_2023_Follower_left_speed", 200)
    client.publish("IDP_2023_Follower_right_speed", 200)
    time.sleep(time)


def get_points(img, destroy_window=True):
    """function for manual perspective transformation of an image
    returns the transformation matrix"""
    points = []

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            points.append((x, y))

    cv2.imshow("img", img)
    cv2.setMouseCallback("img", click_envent)
    cv2.waitKey()
    if destroy_window:
        cv2.destroyAllWindows()
    points = np.int32(points)
    return points


"""def controll():
    src = "http://localhost:8081/stream/video.mjpeg"
    video_getter = VideoGet(src).start()

    detector = apriltag.Detector()
    mtx, dist, newcameramtx, dim = cal.load_vals(6)
    M = perspective_transoformation(video_getter.frame, dim)
    mtx, dist, newcameramtx, dim = cal.load_vals(6)
    current_head = np.float64([1, 0])

    # points = get_points(video_getter.frame)
    # M = cv2.getPerspectiveTransform(points, new_points)   
        current_position = np.float64(result[0].center)
        rotate(video_getter, M, point, detector, mtx, dist, newcameramtx, dim, current_head)
        forward(2)"""


def main():
    if platform == "win32":
        # Windows
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=pupil_apriltags,
        )
    else:
        # mac
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=apriltag,
        )


def detect_red_video(frame):
    "detecting red cube in video and draws rectangle around it"

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
    key = cv2.waitKey(0)
    return [x + w / 2, y + h / 2]


if __name__ == "__main__":
    mtx, dist, newcameramtx = cal.load_vals(6)
    dim = (810, 810)

    video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    start = time.time()
    while time.time() - start < 2:
        ret, img = video.read()
        if not ret:
            continue
    ret, img = video.read()
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    M = perspective_transoformation(img, dim)
    img = cv2.warpPerspective(img, M, dim)
    # path = [] supposed to be predefined
    path = get_points(img, destroy_window=False)
    video.release()
    cv2.destroyAllWindows()
    time.sleep(1)
    print(path)

    video_getter = VideoGet("http://localhost:8081/stream/video.mjpeg").start()

    option = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(option)
    robot = Controller(video_getter, mtx, dist, newcameramtx, M, dim, detector)

    for each_point in path:
        robot.rotate(each_point, threshold=1)
        robot.move(each_point)
