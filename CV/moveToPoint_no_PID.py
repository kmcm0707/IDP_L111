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
global color


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
    # points = np.float32(points)
    points = np.float32([(255, 694), (833, 641), (217, 137), (753, 75)])
    new_points = np.float32([(0, 0), (0, dim[1]), (dim[0], 0), dim])

    M = cv2.getPerspectiveTransform(points, new_points)
    return M


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


def on_message(client, userdata, msg):
    global color
    color = np.int32(msg.payload.decode())


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
    client.subscribe("IDP_2023_Color")
    client.on_message = on_message
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
    frame = video_getter.frame
    try:
        if module is apriltag:
            option = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(option)
            detect = detector.detect

            def angle(result):
                v = (result.corners[1] + result.corners[0]) - (
                    result.corners[3] + result.corners[2]
                )
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
    time.sleep(5)

    if fix_distortion:
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    if fix_perspective:
        frame = cv2.warpPerspective(frame, M, dim)

    # position_red = detect_red(frame)
    # frame, position_red = detect_red(frame)
    # print(position_red)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    global target
    global targets
    global color
    target = None
    targets = []
    color = None

    def click_envent(event, x, y, flags, params):
        global targets
        if event == cv2.EVENT_LBUTTONDOWN:
            global target
            # global targets
            print(x, y)
            target = np.array([x, y])
            targets.append(target)
        if len(targets) >= 7:
            cv2.destroyAllWindows()

    cv2.imshow("img", frame.copy())
    cv2.setMouseCallback("img", click_envent)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(targets)
    """targets = np.array(
        [
            [734, 560],
            [739, 198],
            [410, 128],
            [93, 223],
            [91, 531],
            [196, 698],
            [403, 712],
        ]
    )
    targets[2] = position_red"""
    print("hello")

    frame_counter = 0
    # client.publish("IDP_2023_Follower_left_speed", -150)
    # client.publish("IDP_2023_Follower_right_speed", 150)

    # current_position = np.array([0, 0])
    moving_forward = False
    clockwise = False
    anticlockwise = False
    target = targets[0]
    current_target = 0
    last_time = time.time()
    client.publish("IDP_2023_Servo_Horizontal", 1)
    client.publish("IDP_2023_Servo_Vertical", 1)
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
            # print("target", target)
            # print("current", current_position)
            heading = np.float64(target - current_position)
            heading /= np.linalg.norm(heading)
            v = np.float64(result[0].corners[1] - result[0].corners[0])
            v /= np.linalg.norm(v)
            u = np.float64(result[0].corners[2] - result[0].corners[1])
            u /= np.linalg.norm(u)
            cv2.circle(frame, np.int32(result[0].corners[3]), 10, (0, 0, 255), -1)
            cv2.circle(frame, np.int32(result[0].corners[0]), 10, (0, 255, 0))
            for each in targets:
                cv2.circle(frame, np.int32(each), 35, (0, 255, 0))
            cv2.polylines(frame, [np.int32(targets)], False, (0, 255, 0), 2)
            """ cv2.imshow("img", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break"""
            angle_diff = np.arccos(np.dot(v, heading))

            if np.linalg.norm(current_position - target) < 35:
                client.publish("IDP_2023_Follower_left_speed", 0)
                client.publish("IDP_2023_Follower_right_speed", 0)
                print("done")
                if current_target == 1:
                    client.publish("IDP_2023_Servo_Vertical", 0)
                    client.publish("IDP_2023_Servo_Horizontal", 0)
                    time.sleep(5)
                elif current_target == 2:
                    client.publish("IDP_2023_Servo_Horizontal", 1)
                    time.sleep(2)
                    client.publish("IDP_2023_Servo_Vertical", 1)
                    client.publish("IDP_2023_Set_Block", 1)
                    client.on_message = on_message
                    client.loop_start()
                    time.sleep(3)
                    while color is None:
                        time.sleep(0.1)
                    if color == 0:
                        # red
                        print("red")
                    elif color == 1:
                        # blue
                        print("blue")
                    else:
                        # didn't detect
                        print("error")
                    client.loop_stop()
                    color = None
                elif current_target == 3:
                    # client.publish("IDP_2023_Set_Ultrasound", 1)
                    client.publish("IDP_2023_Set_Ultrasound", 0)
                elif current_target == 4:
                    client.publish("IDP_2023_Set_Ultrasound", 0)
                elif current_target == 5:
                    client.publish("IDP_2023_Servo_Horizontal", 0)
                    client.publish("IDP_2023_Follower_left_speed", -255)
                    client.publish("IDP_2023_Follower_right_speed", -255)
                    time.sleep(5)
                    client.publish("IDP_2023_Follower_left_speed", 0)
                    client.publish("IDP_2023_Follower_right_speed", 0)
                elif current_target == 6:
                    video_getter.stop()
                    break
                current_target += 1
                target = targets[current_target]
                clockwise = False
                anticlockwise = False
                moving_forward = False

            if angle_diff < 0.2 or angle_diff > (2 * math.pi) - 0.2:
                if not moving_forward or last_time - time.time() > 0.5:
                    client.publish("IDP_2023_Follower_left_speed", 255)
                    client.publish("IDP_2023_Follower_right_speed", 255)
                    moving_forward = True
                    last_time = time.time()
                    clockwise = False
                    anticlockwise = False

            else:
                if abs(np.dot(u, v)) > 0.1:
                    continue
                """print(
                    "left or right :",
                    (current_position[0] - v[0]) * (target[1] - v[1])
                    - ((current_position[1] - v[1]) * (target[0] - v[0])),
                )"""
                direction = (
                    ((result[0].corners[1] + result[0].corners[2]))
                    - (result[0].corners[0] + result[0].corners[3])
                ) / 2
                if (
                    (np.float64(direction[0])) * (target[1] - result[0].corners[1][1])
                    - (
                        (np.float64(direction[1]))
                        * (target[0] - result[0].corners[0][0])
                    )
                ) > 0:
                    if not clockwise or time.time() - last_time > 0.5:
                        client.publish("IDP_2023_Follower_left_speed", 125)
                        client.publish("IDP_2023_Follower_right_speed", -125)
                        clockwise = True
                        moving_forward = False
                        anticlockwise = False
                        last_time = time.time()
                else:
                    if not anticlockwise or time.time() - last_time > 0.5:
                        client.publish("IDP_2023_Follower_left_speed", -125)
                        client.publish("IDP_2023_Follower_right_speed", 125)
                        clockwise = False
                        moving_forward = False
                        anticlockwise = True
                        last_time = time.time()

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
            "http://localhost:8081/stream/video.mjpeg",
            module=pupil_apriltags,
        )
    else:
        # mac
        # while True:
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=apriltag,
        )


def detect_red(frame):
    "for detecting red cube in an image"

    # TODO: do the processing in particular section of the image

    font = cv2.FONT_HERSHEY_COMPLEX

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV

    lower_red = np.array([130, 100, 100])

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
            cv2.imshow("red", frame)
            cv2.waitKey()
            return (x, y)

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

    cv2.imshow("red", frame)
    cv2.waitKey()

    return frame, centres


def detect_red_video(frame):
    "detecting red cube in video and draws rectangle around it"

    # conveting from RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red threshold values
    lower_red = np.array([120, 70, 80])
    upper_red = np.array([180, 255, 255])

    # one is ranges shows up white and else in black
    mask = cv2.inRange(hsv, lower_red, upper_red)
    cv2.imshow("mask", mask)
    cv2.waitKey()
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
        if cv2.contourArea(cont) <= 30:
            continue
        x, y, w, h = cv2.boundingRect(cont)
        # if (x < 243  or x > 573 or y < 64 or y > 211):
        #   continue
        centres.append((x, y, w, h))
    font = cv2.FONT_HERSHEY_COMPLEX
    block_red = None
    dim = frame.shape
    half = dim[1] * 0.5
    for x, y, w, h in centres:
        if x < 243 or x > 573 or y < 64 or y > 211:
            continue
        if (x < half) and (x > 180):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
            cv2.putText(
                frame,
                f"{x}, {y}, {cv2.contourArea(cont)}",
                (x, y),
                font,
                0.5,
                (255, 0, 0),
            )
            block_red = (x, y, w, h)
    print(block_red)
    # showing the feed with rectangles
    cv2.imshow("frame", frame)
    # for stopping the window wont run without it
    # see file docstring for detail
    key = cv2.waitKey(0)
    return [block_red[0] + block_red[2] / 2, block_red[1] + block_red[3] / 2]


if __name__ == "__main__":
    main()
