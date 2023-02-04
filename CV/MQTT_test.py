import paho.mqtt.client as mqtt
import time
from threading import Thread
import calibration_clean as cal
import cv2
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

# insert keyboard code here with thread of video get

# use these for the speed
client.publish("IDP_2023_Follower_left_speed", "211")
client.publish("IDP_2023_Follower_right_speed", "211")

video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if ret == False:
            continue
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        if key == ord("w"):
            left = 155
            right = 155
            print("forward")

        elif key == ord("s"):
            left = 1
            right = 1
            print("back")

        elif key == ord("a"):
            left = 1
            right = 155
            print("right")

        elif key == ord("d"):
            left = 155
            right = 1
            print("left")

        client.publish("IDP_2023_Follower_left_speed", str(left))
        client.publish("IDP_2023_Follower_right_speed", str(right))

    cv2.destroyAllWindows()