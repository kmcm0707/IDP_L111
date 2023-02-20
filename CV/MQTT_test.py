#!../env/bin/python
# coding: utf-8

"""This is the code for the manual control of the robot using the keyboard. 

It is not used in the final product, but is kept for reference purposes 
and used to test mechanical components as well as the MQTT push requests."""

from threading import Thread
import paho.mqtt.client as mqtt
import cv2

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
    client.publish("IDP_2023_Follower_left_speed", 211)
    client.publish("IDP_2023_Follower_right_speed", 211)


def stop():
    client.publish("IDP_2023_Follower_left_speed", 1)
    client.publish("IDP_2023_Follower_right_speed", 1)


def back():
    client.publish("IDP_2023_Follower_left_speed", -211)
    client.publish("IDP_2023_Follower_right_speed", -211)


def left():
    client.publish("IDP_2023_Follower_left_speed", "1")
    client.publish("IDP_2023_Follower_right_speed", "211")


def back_left():
    client.publish("IDP_2023_Follower_left_speed", "1")
    client.publish("IDP_2023_Follower_right_speed", "-211")


def right():
    client.publish("IDP_2023_Follower_left_speed", "211")
    client.publish("IDP_2023_Follower_right_speed", "1")


def back_right():
    client.publish("IDP_2023_Follower_left_speed", "-211")
    client.publish("IDP_2023_Follower_right_speed", "1")


def clockwise_rotate():
    client.publish("IDP_2023_Follower_left_speed", "211")
    client.publish("IDP_2023_Follower_right_speed", "-211")


def anticlockwise_rotate():
    client.publish("IDP_2023_Follower_left_speed", "-211")
    client.publish("IDP_2023_Follower_right_speed", "211")


def down():
    client.publish("IDP_2023_Servo_Vertical", 0)


def up():
    client.publish("IDP_2023_Servo_Vertical", 1)


def close():
    client.publish("IDP_2023_Servo_Horizontal", 1)


def open():
    client.publish("IDP_2023_Servo_Horizontal", 0)


# insert keyboard code here with thread of video get

# use these for the speed
client.publish("IDP_2023_Follower_left_speed", "211")
client.publish("IDP_2023_Follower_right_speed", "211")


video = VideoGet(
    # "http://localhost:8081/stream/video.mjpeg"
    0
).start()

while True:
    # runs coresponding function when key is pressed
    img = video.frame
    cv2.imshow("img", img)

    key = cv2.waitKey(2)
    if key == ord("z"):
        break

    elif key == ord("w"):
        forward()

    elif key == ord(" "):
        back()

    elif key == ord("q"):
        left()

    elif key == ord("a"):
        back_left()

    elif key == ord("e"):
        right()

    elif key == ord("d"):
        back_right()

    elif key == ord("p"):
        clockwise_rotate()

    elif key == ord("o"):
        anticlockwise_rotate()

    elif key == ord("s"):
        stop()

    elif key == ord("1"):
        up()

    elif key == ord("2"):
        down()

    elif key == ord("3"):
        close()

    elif key == ord("4"):
        open()

    elif key == ord("5"):
        client.publish("IDP_2023_Servo_Vertical", 2)

video.stop()
cv2.destroyAllWindows()
