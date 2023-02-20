from flask import Flask, jsonify
import json
import time
from threading import Thread
"""An attempt to use flask to create a webserver to control the robot.
This was not used as it was too slow and not reliable enough to be used in the final product.
It was probably due to the fact that the server was running on the same code as the openCV code, which was not ideal."""
# import requests
import cv2

app = Flask(__name__)

left = 0
right = 0


def update():
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

    cv2.destroyAllWindows()


def start_flask():
    app.run(host="0.0.0.0", port=5000)


@app.route("/data", methods=["POST", "GET"])
def data():
    dictToReturn = {"llf": left, "rlf": right}
    return jsonify(dictToReturn)


if __name__ == "__main__":
    print("working")
    th1 = Thread(target=start_flask)
    th1.start()
    """
    count = 0
    while True:
        time.sleep(0.5)
        if count % 20 == 0:
            right = 155
            left = 1
            print("left: ", left, "right: ", right)

        elif count % 20 == 10:
            right = 1
            left = 155
            print("left: ", left, "right: ", right)
        count += 1"""

    # video = cv2.VideoCapture("http://localhost:8081/stream/video.mjpeg")
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

    cv2.destroyAllWindows()

# $ flask run --host=0.0.0.0
# flask --app main run --host=0.0.0.0
# POST http://localhost:5000/data
# GET http://localhost:5000/data
# POST http://10.248.155.126:5000/data
