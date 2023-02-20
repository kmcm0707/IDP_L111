#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""Main code for computer vision code for detection for cube, line or ar tag"""
"""This is the code that was used for the competition"""
"""It is a modular version of moveToPoint_no_PID.py"""
"""Separate control and move were used to make the code more testable and determine where errors were occurring"""
"""All code in moveToPoint offshoots folders were earlier or different versions of this code that were not used in the competition"""

import sys
import time
import math
from threading import Thread
import calibration_clean as cal
import paho.mqtt.client as mqtt


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

"""The following try except statements are used to check if the apriltag detection modules are installed"""
"""This was done so both windows and mac users could use the code"""
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


"""The following code is used to connect to the MQTT broker"""
# mqttBroker = "broker.hivemq.com"
# Alternative message brokers:
# mqttBroker = "public.mqtthq.com"
mqttBroker = "test.mosquitto.org"
# mqttBroker =  "public.mqtthq.com"
client = mqtt.Client("Python")
client.connect(mqttBroker)
client.subscribe("IDP_2023_Color")

global color

"""Callback function for when a message is received from the MQTT broker"""
def on_message(client, userdata, msg):
    global color
    color = np.int32(msg.payload.decode())


client.on_message = on_message

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


def move_to(
    target,
    video_getter,
    mtx,
    dist,
    newcameramtx,
    dim,
    M,
    detect,
    only_rotate=False,
    angle_threshold=0.2,
    encatchment_radius=35,
    speed=255,
    rotate_speed=125,
    ultra=False,
) -> None:
    """Function for moving the robot to a target point"""
    """First the robot will rotate to face the target point"""
    """Then the robot will move forward until it is within a certain distance of the target point"""
    print("hello")
    ultra_time = time.time()
    moving_forward = False
    clockwise = False
    anticlockwise = False
    current_position = np.array([0, 0])
    last_time = time.time()

    # client.publish("IDP_2023_Servo_Horizontal", 1)
    # client.publish("IDP_2023_Servo_Vertical", 1)
    while True:
        if time.time() - ultra_time > 2 and ultra:
            client.publish("IDP_2023_Set_Ultrasound", 0)
            ultra = False
        frame = video_getter.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        frame = cv2.warpPerspective(frame, M, dim)

        result = detect(frame)
        if len(result) > 0:

            current_position[:] = result[0].center
            # print("target", target)
            # print("current", current_position)

            heading = np.float64(target - current_position)
            heading /= np.linalg.norm(heading)

            """Calculate the angle between the robot rotation and the vector from the robot to the target"""
            v = np.float64(result[0].corners[1] - result[0].corners[0])
            v /= np.linalg.norm(v)
            u = np.float64(result[0].corners[2] - result[0].corners[1])
            u /= np.linalg.norm(u)

            angle_diff = np.arccos(np.dot(v, heading))

            if np.linalg.norm(current_position - target) < encatchment_radius:
                client.publish("IDP_2023_Follower_left_speed", 0)
                client.publish("IDP_2023_Follower_right_speed", 0)
                print("done")
                break

            """if the robot is facing the target point"""
            if (
                angle_diff < angle_threshold
                or angle_diff > (2 * math.pi) - angle_threshold
            ):
                if only_rotate:
                    print("done rotate_only")
                    client.publish("IDP_2023_Follower_left_speed", 0)
                    client.publish("IDP_2023_Follower_right_speed", 0)
                    break

                if not moving_forward or last_time - time.time() > 0.5:
                    client.publish("IDP_2023_Follower_left_speed", speed)
                    client.publish("IDP_2023_Follower_right_speed", speed)
                    moving_forward = True
                    last_time = time.time()
                    clockwise = False
                    anticlockwise = False

            else:
                """checks if robot is on ramp"""
                if abs(np.dot(u, v)) > 0.1:
                    """if last_time - time.time() > 0.5:
                    client.publish("IDP_2023_Follower_left_speed", speed)
                    client.publish("IDP_2023_Follower_right_speed", speed)
                    last_time = time.time()"""
                    continue

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
                        """rotate clockwise"""
                        client.publish("IDP_2023_Follower_left_speed", rotate_speed)
                        client.publish("IDP_2023_Follower_right_speed", -rotate_speed)
                        clockwise = True
                        moving_forward = False
                        anticlockwise = False
                        last_time = time.time()
                else:
                    if not anticlockwise or time.time() - last_time > 0.5:
                        """rotate anticlockwise"""
                        client.publish("IDP_2023_Follower_left_speed", -rotate_speed)
                        client.publish("IDP_2023_Follower_right_speed", rotate_speed)
                        clockwise = False
                        moving_forward = False
                        anticlockwise = True
                        last_time = time.time()


def main():
    """Main function for the robot"""
    """This function will be called when the program is run"""
    """It moves the robot to the target points and picks up the cube"""
    global color
    color = None
    targets = []

    """def click_envent(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            target = np.array([x, y])
            targets.append(target)
        if len(targets) >= 7:
            cv2.destroyAllWindows()

        if event == cv2.EVENT_RBUTTONDOWN:
            print("done")
            cv2.destroyAllWindows()"""

    path = []
    src = "http://localhost:8081/stream/video.mjpeg"
    mtx, dist, newcameramtx = cal.load_vals(6)

    video = cv2.VideoCapture(src)
    time.sleep(2)
    ret, frame = video.read()
    if not ret:
        print("error with video feed")
        return -1

    video.release()

    video_getter = VideoGet(src).start()
    time.sleep(2)
    option = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(option)
    detect = detector.detect

    targets = np.array(
        [
            [740, 560],  # ramp
            [740, 185],
            [410, 128],  # red
            [76, 206],  # tunnel
            [73, 531],
            [100, 571],  # tunnel
            [196, 698],  # not needed [191, 719]
            [408, 564],
            [413, 748],  # end point
            [413, 801],  # end point
        ]
    )
    blocks_collected = 0
    frame = video_getter.frame
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    dim = (810, 810)
    M = perspective_transoformation(frame, dim)
    while blocks_collected < 2:
        frame = video_getter.frame
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        dim = (810, 810)
        M = perspective_transoformation(frame, dim)
        frame = cv2.warpPerspective(frame, M, dim)
        try:
            targets[2] = detect_red(frame)
        except:
            print("no red")

        client.publish("IDP_2023_Servo_Horizontal", 1)
        client.publish("IDP_2023_Servo_Vertical", 2)

        move_to(targets[0], video_getter, mtx, dist, newcameramtx, dim, M, detect)
        move_to(targets[1], video_getter, mtx, dist, newcameramtx, dim, M, detect)

        client.publish("IDP_2023_Servo_Vertical", 0)
        client.publish("IDP_2023_Servo_Horizontal", 0)
        time.sleep(5)
        # over the ramp claw down
        frame = video_getter.frame
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        frame = cv2.warpPerspective(frame, M, dim)
        try:
            targets[2] = detect_red(frame)
        except:
            print("no red")
        # now moves to red block
        move_to(
            targets[2],
            video_getter,
            mtx,
            dist,
            newcameramtx,
            dim,
            M,
            detect,
            only_rotate=True,
            angle_threshold=0.1,
            rotate_speed=140,
        )
        move_to(
            targets[2],
            video_getter,
            mtx,
            dist,
            newcameramtx,
            dim,
            M,
            detect,
            angle_threshold=0.1,
            encatchment_radius=40,
        )
        client.publish("IDP_2023_Servo_Horizontal", 1)
        time.sleep(2)
        client.publish("IDP_2023_Servo_Vertical", 2)
        client.publish("IDP_2023_Set_Block", 1)
        client.loop_start()
        time.sleep(3)
        red = False
        blue = False
        print(color)
        while color is None:
            time.sleep(0.1)
        if color == 0:
            # red
            print("red")
            red = True
        elif color == 1:
            # blue
            print("blue")
            blue = True
        else:
            # didn't detect
            print("error")
        client.loop_stop()
        color = None
        ## tunnel
        move_to(targets[3], video_getter, mtx, dist, newcameramtx, dim, M, detect)
        # client.publish("IDP_2023_Set_Ultrasound", 1)
        move_to(
            targets[4],
            video_getter,
            mtx,
            dist,
            newcameramtx,
            dim,
            M,
            detect,
            rotate_speed=140,
            # ultra=True,
        )
        move_to(
            targets[5],
            video_getter,
            mtx,
            dist,
            newcameramtx,
            dim,
            M,
            detect,
            # ultra=True,
        )
        # client.publish("IDP_2023_Set_Ultrasound", 0)
        client.publish("IDP_2023_Servo_Vertical", 3)
        ## move to put down areas
        if red:
            targets_red = np.array(
                [[622, 719], [635, 804]],
            )
            move_to(
                targets_red[0],
                video_getter,
                mtx,
                dist,
                newcameramtx,
                dim,
                M,
                detect,
                only_rotate=True,
            )
            move_to(
                targets_red[0],
                video_getter,
                mtx,
                dist,
                newcameramtx,
                dim,
                M,
                detect,
                angle_threshold=0.2,
            )
            move_to(
                targets_red[1],
                video_getter,
                mtx,
                dist,
                newcameramtx,
                dim,
                M,
                detect,
                only_rotate=True,
                angle_threshold=0.2,
            )
            client.publish("IDP_2023_Servo_Horizontal", 0)
            time.sleep(1)
            client.publish("IDP_2023_Follower_left_speed", -255)
            client.publish("IDP_2023_Follower_right_speed", -255)
            time.sleep(5)
            client.publish("IDP_2023_Follower_left_speed", 0)
            client.publish("IDP_2023_Follower_right_speed", 0)
            red = False
        elif blue:
            targets_blue = np.array(
                [[191, 719], [183, 804]],
            )
            move_to(
                targets_blue[0],
                video_getter,
                mtx,
                dist,
                newcameramtx,
                dim,
                M,
                detect,
                only_rotate=True,
            )
            move_to(
                targets_blue[0],
                video_getter,
                mtx,
                dist,
                newcameramtx,
                dim,
                M,
                detect,
                angle_threshold=0.2,
            )
            move_to(
                targets_blue[1],
                video_getter,
                mtx,
                dist,
                newcameramtx,
                dim,
                M,
                detect,
                only_rotate=True,
                angle_threshold=0.2,
            )
            client.publish("IDP_2023_Servo_Horizontal", 0)
            time.sleep(1)
            client.publish("IDP_2023_Follower_left_speed", -255)
            client.publish("IDP_2023_Follower_right_speed", -255)
            time.sleep(5)
            client.publish("IDP_2023_Follower_left_speed", 0)
            client.publish("IDP_2023_Follower_right_speed", 0)
            blue = False

        blocks_collected += 1
    move_to(targets[7], video_getter, mtx, dist, newcameramtx, dim, M, detect)
    move_to(targets[8], video_getter, mtx, dist, newcameramtx, dim, M, detect)
    move_to(
        targets[9],
        video_getter,
        mtx,
        dist,
        newcameramtx,
        dim,
        M,
        detect,
        only_rotate=True,
    )
    video_getter.stop()


def detect_red(frame):
    "for detecting red cube in an image"

    font = cv2.FONT_HERSHEY_COMPLEX

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV

    lower_red = np.array([130, 100, 100])

    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colours
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
            """cv2.imshow("red", frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()"""
            return (x, y - 9)

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

    return frame, centres


def detect_red_video(frame):
    "detecting red cube in video and draws rectangle around it"
    "not used in final code"

    # conveting from RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red threshold values
    lower_red = np.array([130, 100, 100])
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
        # if (x < 243  or x > 573 or y < 64 or y > 211):
        #   continue
        centres.append((x, y, w, h))
    font = cv2.FONT_HERSHEY_COMPLEX
    block_red = None
    dim = frame.shape
    half = dim[1] * 0.5
    for x, y, w, h in centres:
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
    return [block_red[0] + block_red[2] / 2, block_red[1] + block_red[3] / 2 - 3]


if __name__ == "__main__":
    main()
    """video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    detect_red_video(video.read()[1].copy())
    video.release()
    cv2.destroyAllWindows()"""
