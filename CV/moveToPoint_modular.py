#!../env/bin/python
# coding: utf-8
# python 3.9.16

"""This file is the main code responsible for navigation of the robot.

This is the code that was used for the competition
It is a modular version of moveToPoint_no_PID.py
Separate control and move were used to make the code more testable and determine where errors were occuring
All code in moveToPoint offshoots folders were earlier or different versions of this code that were not used in the competition
"""

import sys
import time
import math
from threading import Thread
import calibration_clean as cal
import paho.mqtt.client as mqtt

try:
    import cv2
    import numpy as np
except ImportError:
    print("dependencies not installed")
    print("run 'pip install -r requirements.txt' to install dependencies")
    exit()

# The following try except statements are used to check if the apriltag detection modules are installed
# This was done so both windows and mac users could use the code
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


def on_message(client, userdata, msg):
    """Callback function for when a message is received from the MQTT broker"""
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


def perspective_transoformation(img, dim, manual=False):
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

    if manual:
        cv2.imshow("img", img)
        cv2.setMouseCallback("img", click_envent)
        cv2.waitKey()
        points = np.float32(points)
    else:
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


def detect_red(frame, show=False):
    """for detecting red cube in an image

    Parameters
    ----------
    frame : numpy.ndarray
        image to detect red cube in
    show : bool, optional
        whether to show the image with the bounding box, by default False
    """

    font = cv2.FONT_HERSHEY_COMPLEX

    # converting BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # defining range of red color in HSV
    lower_red = np.array([130, 100, 100])
    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colours
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # removing noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # finding contours / group of pixels
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)

        # only considering ones in the rigon of interest
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

            if show:
                cv2.imshow("red", frame)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

            return (x, y - 9)


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
    """Function for moving the robot to a target point
    First the robot will rotate to face the target point
    Then the robot will move forward until it is within a certain distance of the target point

    Parameters
    ----------
    target : np.array
        The target point to move to
    video_getter : VideoGet
        The VideoGet object used to get frames from the camera
    mtx : np.array
        The camera matrix
    dist : np.array
        The distortion coefficients
    newcameramtx : np.array
        The new camera matrix
    dim : tuple
        The dimensions of the image
    M : np.array
        The transformation matrix
    detect : function
        The function used to detect the april tag

    only_rotate : bool, optional
        If True the robot will only rotate to face the target point, by default False
    angle_threshold : float, optional
        The threshold for the angle between the robot and the target point, by default 0.2
    encatchment_radius : int, optional
        The radius of the circle around the target point, by default 35
    speed : int, optional
        The speed of the robot, by default 255
    rotate_speed : int, optional
        The speed of the robot when rotating, by default 125
    ultra : bool, optional
        If True the robot will use the ultrasound sensor to detect obstacles, by default False
    """

    """Initialising variables"""
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

            heading = np.float64(target - current_position)
            heading /= np.linalg.norm(heading)

            # Calculate the angle between the robot rotation and the vector from the robot to the target
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
                # checks if robot is on ramp
                if abs(np.dot(u, v)) > 0.1:
                    continue

                direction = (
                    ((result[0].corners[1] + result[0].corners[2]))
                    - (result[0].corners[0] + result[0].corners[3])
                ) / 2

                # fmt: off  // this is to stop black from formatting the code
                if (
                    (np.float64(direction[0])) * (target[1] - result[0].corners[1][1])
                    - (
                        (
                            np.float64(direction[1])
                            * (target[0] - result[0].corners[0][0])
                        )
                    )
                ) > 0:
                    # fmt: on // ignore

                    if not clockwise or time.time() - last_time > 0.5:
                        # rotate clockwise
                        client.publish("IDP_2023_Follower_left_speed", rotate_speed)
                        client.publish("IDP_2023_Follower_right_speed", -rotate_speed)
                        clockwise = True
                        moving_forward = False
                        anticlockwise = False
                        last_time = time.time()
                else:
                    if not anticlockwise or time.time() - last_time > 0.5:
                        # rotate anticlockwise
                        client.publish("IDP_2023_Follower_left_speed", -rotate_speed)
                        client.publish("IDP_2023_Follower_right_speed", rotate_speed)
                        clockwise = False
                        moving_forward = False
                        anticlockwise = True
                        last_time = time.time()


def main():
    """Main function for the robot
    This function will be called when the program is run
    It moves the robot to the target points and picks up the cube"""
    global color
    color = None
    targets = []

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

    """Set of target points for the robot to move to"""
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
    """While the robot has not collected 2 blocks"""
    while blocks_collected < 2:
        """Get the frame from the video feed"""
        frame = video_getter.frame
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        dim = (810, 810)
        M = perspective_transoformation(frame, dim)
        frame = cv2.warpPerspective(frame, M, dim)
        try:
            targets[2] = detect_red(frame)
        except:
            print("no red")

        """Lifts and closes the claw"""
        client.publish("IDP_2023_Servo_Horizontal", 1)
        client.publish("IDP_2023_Servo_Vertical", 2)

        """Moves to the ramp and goes over it"""
        move_to(targets[0], video_getter, mtx, dist, newcameramtx, dim, M, detect)
        move_to(targets[1], video_getter, mtx, dist, newcameramtx, dim, M, detect)

        """lowers the claw and opens it"""
        client.publish("IDP_2023_Servo_Vertical", 0)
        client.publish("IDP_2023_Servo_Horizontal", 0)
        time.sleep(5)
        # detects the red block
        frame = video_getter.frame
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        frame = cv2.warpPerspective(frame, M, dim)
        try:
            targets[2] = detect_red(frame)
        except:
            print("no red")
        # now moves to red block
        # a rotate only is added to make sure the robot is facing the block and make it more accurate
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
            angle_threshold=0.1, ## more precise
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
            angle_threshold=0.1, ## more precise
            encatchment_radius=40, ## larger radius so can actually catch the block
        )

        ## close claw
        client.publish("IDP_2023_Servo_Horizontal", 1)
        time.sleep(2)

        ## lifts claw to middle position (so can fit through tunnel) and tells arduino to detect color
        client.publish("IDP_2023_Servo_Vertical", 2)
        client.publish("IDP_2023_Set_Block", 1)

        ## wait for color to be detected and message to be sent from arduino
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
        """Lifts claw fully"""
        client.publish("IDP_2023_Servo_Vertical", 3)
        ## move to put down areas and rotate to face the correct direction
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
            """Opens claw to drop block"""
            client.publish("IDP_2023_Servo_Horizontal", 0)
            time.sleep(1)
            """reverse to get out of the way"""
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
            """Opens claw to drop block"""
            client.publish("IDP_2023_Servo_Horizontal", 0)
            time.sleep(1)
            """reverse to get out of the way"""
            client.publish("IDP_2023_Follower_left_speed", -255)
            client.publish("IDP_2023_Follower_right_speed", -255)
            time.sleep(5)
            client.publish("IDP_2023_Follower_left_speed", 0)
            client.publish("IDP_2023_Follower_right_speed", 0)
            blue = False

        blocks_collected += 1
    """Finished so moves to the end zone"""
    move_to(targets[7], video_getter, mtx, dist, newcameramtx, dim, M, detect)
    move_to(targets[8], video_getter, mtx, dist, newcameramtx, dim, M, detect)
    """Rotates so fully in the end zone"""
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

    ## All done


if __name__ == "__main__":
    main()
    """video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
    detect_red_video(video.read()[1].copy())
    video.release()
    cv2.destroyAllWindows()"""
