#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""Main code for computer vision comaints code for detection for cube, line or ar tag
"""

import sys
import time
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


def rotate(
    video_getter,
    M,
    target,
    current_position,
    detector,
    mtx,
    dist,
    newcameramtx,
    dim,
    current_head=None,
    fix_distortion=True,
    fix_perspective=True,
):
    v = target - current_position
    v /= np.linalg.norm(v)
    current_head /= np.linalg.norm(current_head)
    angle_diff = np.artann2(np.dot(v, current_head))
    if angle_diff < 0:
        client.publish("IDP_2023_Follower_left_speed", -200)
        client.publish("IDP_2023_Follower_right_speed", 200)
    else:
        client.publish("IDP_2023_Follower_left_speed", 200)
        client.publish("IDP_2023_Follower_right_speed", -200)

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

        result = detector(frame)
        if len(result) > 0:
            if first_time:
                first_time = False
                continue

            current_position[:] = result[0].center
            print("target", target)
            print("current", current_position)
            heading = np.float64(target - current_position)
            heading /= np.linalg.norm(heading)
            v = np.float64(result[0].corners[1] - result[0].corners[0])
            v /= np.linalg.norm(v)
            angle_diff = np.arccos(np.dot(v, heading))

            print("angle: ", angle_diff)

            prev_time = time.time()

            prev_angle = angle_diff

            if angle_diff < 0.1:
                return

            a, b, c, d = result[0].corners
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

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            return


def forward(time=2):
    client.publish("IDP_2023_Follower_left_speed", 200)
    client.publish("IDP_2023_Follower_right_speed", 200)
    time.sleep(time)


def get_points(img):
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
        mtx, dist, newcameramtx = cal.load_vals(6)  # best one is 6

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
        M = perspective_transoformation(frame, dim)

        video.release()

    video_getter = VideoGet(src).start()
    # video_shower = VideoShow(video_getter.frame).start()
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

    except:
        try:
            if module is pupil_apriltags:
                detector = pupil_apriltags.Detector(families="tag36h11")
                detect = detector.detect
        except:
            raise ValueError("need a valid module")

    current_position = np.array([0, 0])
    first_time = True
    frame = video_getter.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if fix_distortion:
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    if fix_perspective:
        frame = cv2.warpPerspective(frame, M, dim)
    global target
    target = None

    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            global target
            print(x, y)
            target = np.array([x, y])
            cv2.destroyAllWindows()

    cv2.imshow("img", frame.copy())
    cv2.setMouseCallback("img", click_envent)
    key = cv2.waitKey(0)
    print("hello")

    frame_counter = 0
    client.publish("IDP_2023_Follower_left_speed", -100)
    client.publish("IDP_2023_Follower_right_speed", 100)

    # current_position = np.array([0, 0])
    moving_forward = False
    clockwise = True
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
                first_time = False
                continue

            current_position[:] = result[0].center
            print("target", target)
            print("current", current_position)
            heading = np.float64(target - current_position)
            heading /= np.linalg.norm(heading)
            v = np.float64(result[0].corners[1] - result[0].corners[0])
            v /= np.linalg.norm(v)
            angle_diff = np.arccos(np.dot(v, heading))

            print("angle diference: ", angle_diff)

            if np.linalg.norm(current_position - target) < 10:
                client.publish("IDP_2023_Follower_left_speed", 0)
                client.publish("IDP_2023_Follower_right_speed", 0)
                print("done")
                break

            if abs(angle_diff) < 0.2:
                if not moving_forward:
                    client.publish("IDP_2023_Follower_left_speed", 250)
                    client.publish("IDP_2023_Follower_right_speed", 250)
                    moving_forward = True
                    print("forward")

            else:
                if (heading[0] - np.float64(current_position[0])) * (
                    v[1] - np.float64(current_position[1])
                ) - (heading[1] - np.float64(current_position[1])) * (
                    v[0] - np.float64(current_position[0])
                ) > 0:
                    if not clockwise:
                        client.publish("IDP_2023_Follower_left_speed", -100)
                        client.publish("IDP_2023_Follower_right_speed", 100)
                        clockwise = True
                        print("clockwise")
                else:
                    if clockwise:
                        client.publish("IDP_2023_Follower_left_speed", 100)
                        client.publish("IDP_2023_Follower_right_speed", -100)
                        print("anticlockwise")
                        clockwise = False

                """if angle_diff > 0:
                    client.publish("IDP_2023_Follower_left_speed", -100)
                    client.publish("IDP_2023_Follower_right_speed", 100)
                    print("clockwise")
                else:
                    client.publish("IDP_2023_Follower_left_speed", 100)
                    client.publish("IDP_2023_Follower_right_speed", -100)
                    print("anticlockwise")"""

            """a, b, c, d = result[0].corners
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
            if first_time:
                frame = cv2.polylines(
                    frame, [np.int32(positions)], False, (0, 0, 255), 2
                )
                first_time = False

        if not first_time:
            frame = cv2.polylines(frame, [np.int32(positions)], False, (0, 0, 255), 2)"""

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


def main():
    if platform == "win32":
        # Windows
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg", module=pupil_apriltags
        )
    else:
        # mac
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg", module=apriltag
        )


if __name__ == "__main__":
    main()
