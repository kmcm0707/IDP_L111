#!../env/bin/python
import paho.mqtt.client as mqtt
import time
from threading import Thread
import calibration_clean as cal

import cv2

# import keyboard  # pip install keyboard
import numpy as np

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


def forward():
    client.publish("IDP_2023_Follower_left_speed", "211")
    client.publish("IDP_2023_Follower_right_speed", "211")


def stop():
    client.publish("IDP_2023_Follower_left_speed", "1")
    client.publish("IDP_2023_Follower_right_speed", "1  ")


def left():
    client.publish("IDP_2023_Follower_left_speed", "1")
    client.publish("IDP_2023_Follower_right_speed", "211")


def right():
    client.publish("IDP_2023_Follower_left_speed", "211")
    client.publish("IDP_2023_Follower_right_speed", "1")


# insert keyboard code here with thread of video get

# use these for the speed
client.publish("IDP_2023_Follower_left_speed", "211")
client.publish("IDP_2023_Follower_right_speed", "211")

"""img = cv2.imread("calib_imgs/img_dump_manual_table3_2/0.jpg")
cv2.imshow("img", img)"""

video = VideoGet(0).start()

while True:
    img = video.frame
    cv2.imshow("img", img)

    key = cv2.waitKey(2)
    if key == ord("q"):
        break

    elif key == ord("w"):
        forward()

    elif key == ord("s"):
        stop()

    elif key == ord("a"):
        left()

    elif key == ord("d"):
        right()

video.stop()
cv2.destroyAllWindows()

"""
keyboard.on_press_key("w", forward)
keyboard.on_press_key("s", stop)
keyboard.on_press_key("a", left)
keyboard.on_press_key("d", right)"""
