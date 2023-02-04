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

# insert keyboard code here

# use these for the speed
client.publish("IDP_2023_Follower_left_speed", "211")
client.publish("IDP_2023_Follower_right_speed", "211")