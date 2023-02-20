#!../env/bin/python
# coding: utf-8
# python 3.9.16

# Move to point using PID was an attempt to use PID to move to a point
# This was not used in the final code
# The PID controller ended up being unreliable it would sometimes work however many times would overshoot the point and have to turn around
# This was likely due to an error in detection of angle of apriltag
# The PID controller here used a predicted velocity point to calculate the error however this was very unstable
# This was later fixed by using the angle of the apriltag to calculate the error (which was much more stable) and what we should have used here
# Because using the angle of the apriltag was accurate enough we did not need to use PID to move to a point
# So this code was not used in the final code

import sys
import time
from threading import Thread
import calibration_clean as cal
import paho.mqtt.client as mqtt
import os
import json

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

    except:
        try:
            if module is pupil_apriltags:
                detector = pupil_apriltags.Detector(families="tag36h11")
                detect = detector.detect
        except:
            raise ValueError("need a valid module")

    prev_point = np.array([0, 0])
    current_position = np.array([0, 0])
    interval = 0
    prev_time = time.time()
    positions = []
    first_time = True
    frame = video_getter.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if fix_distortion:
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    if fix_perspective:
        frame = cv2.warpPerspective(frame, M, dim)

    """Function for clicking on the image to set the target position"""
    def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            controller.set_target_position(np.array([x, y]))
            cv2.destroyAllWindows()
            positions.append(np.array([x, y]))

    cv2.imshow("img", frame.copy())
    cv2.setMouseCallback("img", click_envent)
    key = cv2.waitKey(0)
    print("hello")

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

            x, y = result[0].center

            current_position[:] = [x, y]
            controller.set_current_position(current_position)

            positions.append(current_position)

            """PID controller update"""
            interval = time.time() - prev_time
            speed = (result[0].center - prev_point) / interval
            current_position += np.int64(speed * interval * 100)
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


class PID:
    def __init__(self):
        """Initialise the PID controller"""
        self.kp = 70
        self.ki = 0.001
        self.kd = 10
        self.prev_error = 0
        self.integral = 0
        self.error = 0
        self.left_speed = 170
        self.right_speed = 170
        self.current_position = np.array([0, 0])
        self.target_position = np.array([0, 0])
        self.predicted_position = np.array([0, 0])
        self.elapsed_time = time.time()

    # Getters and setters
    def get_right_speed(self):
        return self.right_speed

    def get_left_speed(self):
        return self.left_speed

    def set_current_position(self, current_position):
        self.current_position[:] = current_position

    def set_target_position(self, target_position):
        self.target_position[:] = target_position

    def set_predicted_position(self, predicted_position):
        self.predicted_position[:] = predicted_position

    def PID_controller_update(self):
        """This function will return the error for the PID controller"""
        basespeed = 170
        deltaX = self.current_position[0] - self.predicted_position[0]
        deltaY = self.current_position[1] - self.predicted_position[1]
        targetX = self.current_position[0] - self.target_position[0]
        targetY = self.current_position[1] - self.target_position[1]

        velocityAngle = math.atan2(deltaY, deltaX) + math.pi
        targetAngle = math.atan2(targetY, targetX) + math.pi

        if velocityAngle == 2 * math.pi or targetAngle == 2 * math.pi:
            velocityAngle = 0
            targetAngle = 0

        temp_error = targetAngle - velocityAngle
        if temp_error > math.pi or (temp_error < 0 and temp_error > -math.pi):
            # turn right - left faster
            temp_error = -abs(temp_error)
        else:
            # turn left - right faster
            temp_error = abs(temp_error)

        # Calculate the error
        self.error = temp_error
        self.integral = self.integral + self.error
        D = self.error - self.prev_error
        self.prev_error = self.error

        # Calculate the speed of the motors
        motorspeed = int(self.kp * self.error + self.kd * D + self.ki * self.integral)
        self.left_speed = basespeed - motorspeed
        self.right_speed = basespeed + motorspeed
        if time.time() - self.elapsed_time > 0.2:
            """This is to prevent the code from publishing too many times"""
            self.elapsed_time = time.time()
            self.publish()

    def publish(self):
        """This function will publish the speed of the motors"""
        client.publish("IDP_2023_Follower_left_speed", str(self.left_speed))
        client.publish("IDP_2023_Follower_right_speed", str(self.right_speed))
        print(f"right : {self.right_speed}")
        print(f"left : {self.left_speed}")


def start_everything():
    controller = PID()
    if platform == "win32":
        # Windows
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=pupil_apriltags,
            controller=controller,
        )
    else:
        # mac
        apriltag_detector_procedure(
            "http://localhost:8081/stream/video.mjpeg",
            module=apriltag,
            controller=controller,
        )
    
if __name__ == "__main__":
    start_everything()
